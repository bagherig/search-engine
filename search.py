import pyspark.sql.functions as func
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row


def main(sc):
    """
    Reads the dataset, creates the inverted index, and search for the test queries given.

    :param sc: The SparkContext.
    """
    # Read the dataset.
    print("Reading the dataset...")
    sqlContext = SQLContext(sc)
    dataset = sqlContext.read.json('/user/root/shakespeare_full.json')
    dataset.cache()  # Cache the dataset for faster processing when querying.

    # Create the inverted index.
    print("Creating the inverted index...")
    tokensWithTfIdf = create_inverted_index(dataset)
    # Cache the inverted index for faster processing when querying.
    tokensWithTfIdf.cache()

    # Test queries:
    query1 = "to be or not"
    query2 = "so far so"
    query3 = "if you said so"
    queries = [query1, query2, query3]
    limits = [1, 3, 5]

    # Perform a search for each query and limit.
    for query in queries:
        for lim in limits:
            search_words(dataset, tokensWithTfIdf, query, lim)
            print('\n')


def clean_text(df, col):
    """
    Removes the punctuations from a certain column of a dataframe and lowers all
    characters.\n
    The list of punctuations removed is: !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~

    :param df: The dataframe.
    :param col: The name of the column to clean.
    :return: The original df with punctuations removed from column col and all characters
        lowered.
    """
    # Reference: https://stackoverflow.com/questions/53218312/pyspark-how-to-remove
    # -punctuation-marks-and-make-lowercase-letters-in-rdd

    # A regex that selects all punctuations in a string.
    regex = "[!\"#$%&\'()\*\+,\-.\/:;<=>\?@\[\\\\\\]\^_`{|}~]"
    return df.withColumn(col, func.lower(func.regexp_replace(df[col], regex, '')))


def create_inverted_index(data):
    """
    Creates the inverted index of the given dataset.

    :param data: The dataset for creating the inverted index.
    :return: The inverted index obtained from data.
    """
    num_docs = data.count()  # Total number of objects used for calculating idf.

    # Remove punctuations from each text_entry.
    cleaned_data = clean_text(data, 'text_entry')

    # Split each text entry into words
    tokens = cleaned_data.select('_id',
                                 func.explode(func.split(cleaned_data['text_entry'], ' '))
                                 .alias('token'))

    # Calculate the term frequencies (tf) for each document.
    # Set an alias for tf table to avoid confusion for a later join operation.
    tf = tokens.groupby('_id', 'token').count()\
        .withColumnRenamed('count', 'tf').alias('tf')

    # Get the number of documents each word apears in.
    df = tf.groupby(tf['token']).count()

    # Calculate the document frequency (df) and idf for each word.
    # Set an alias for df_idf table to avoid confusion for a later join operation.
    df_idf = df.select('token', (df['count']).alias('df'),
                       (func.log10(num_docs / df['count'])).alias('idf')).alias('df_idf')

    # Merge the tf, df, and idf for each pair of word-document.
    tf_df_idf = tf.join(df_idf, 'token', how='left')

    # Construct the column tf_idf using columns tf and idf.
    tokensWithTfIdf = tf_df_idf.select('token', '_id', 'tf', 'df', 'idf',
                                       (tf_df_idf['tf'] * tf_df_idf['idf'])
                                       .alias('tf_idf'))

    return tokensWithTfIdf


def search_words(data, tokensWithTfIdf, query, N):
    """
    Searches for a string in the dataset using the inverted index.

    :param data: The dataset to query from.
    :param tokensWithTfIdf: The inverted index associated with data.
    :param query: The query string.
    :param N: The number of results to return.

    :return: The top N results of the search for query from data using the inverted index
        tokensWithTfIdf.
    """
    sql_context = SQLContext(sc)
    # Convert the query into a dataframe with a single column named text_entry.
    q = sql_context.createDataFrame(sc.parallelize([Row(token=query)]))

    # Remove punctuations from the query.
    q = clean_text(q, 'token')

    # Split the query into words.
    tokens = q.withColumn('token', func.explode(func.split(q['token'], ' ')))

    # Get the total number of words in the query for calculating the document scores.
    num_tokens = tokens.count()

    # Count the term frequencies of each word in the query.
    tokens = tokens.groupby('token').count().withColumnRenamed('count', 'qtf')

    # Find and count the number of common words between the query and each document.
    # If the query contains, duplicated words, we try to hit as many of them as possible.
    # For example, for word X:
    # - if the query contains 2 Xs and the document contains 6 Xs, we can
    #   find both query's Xs in the document. So we have hit 2/2 words. However, when
    #   calculating the score, we will only add X's tf_idf to the score once, since the
    #   frequency of X is already reflected in its tf_idf value.
    # - if the query contains 2 Xs and the document contains only 1 X, we can only find
    #   one of query's Xs in the document. So we have hit 1/2 words. When
    #   calculating the score, we still add X's tf_idf to the score once.
    # Reference: https://stackoverflow.com/questions/40161879/pyspark-withcolumn-with-two
    # -conditions-and-three-outcomes
    com_tokens = tokensWithTfIdf.join(tokens, 'token')\
        .withColumn('count', func.when(func.col('qtf') < func.col('tf'), func.col('qtf'))
                    .otherwise(func.col('tf')))

    # Sum up the tf_idf values and the number of words hit, for each document.
    scores_parts = com_tokens.groupby('_id')\
        .agg(func.sum('tf_idf').alias('tf_idf'), func.sum('count').alias('count'))

    # Calculate the score of each document.
    scores = scores_parts.select('_id', (func.round(scores_parts['tf_idf'] *
                                                    scores_parts['count'] /
                                                    num_tokens, 3)).alias('score'))

    # Retrieve the top N documents with highest scores in descending order.
    results = scores.join(data, '_id')\
        .select('_id', 'score', 'text_entry')\
        .sort(func.desc('score'))\
        .limit(N)

    # Print the results.
    # Implementing a function that prints each row. This function had to be implemented
    # since lambda only takes a single expression and print is a statement in python 2.
    # So it had to be turned into a function.
    # Reference:
    # https://stackoverflow.com/questions/2970858/why-doesnt-print-work-in-a-lambda
    def print_row(r):
        """
        Print a Row object from the dataframe of the results of the query.

        :param r: The Row object.
        """
        print((r[0], r[1], str(r[2])))
    print("Showing top {} results for the query \"{}\"".format(N, query))
    results.foreach(print_row)

    return results


if __name__ == "__main__":
    conf = SparkConf().setAppName("Search Engine")
    sc = SparkContext(conf=conf)
    # Set pyspark's log level to WARN to avoid INFO log spamming when printing results.
    # Reference: https://stackoverflow.com/questions/32512684/how-to-turn-off-info-from
    # -logs-in-pyspark-with-no-changes-to-log4j-properties
    sc.setLogLevel("WARN")
    main(sc)
    sc.stop()
