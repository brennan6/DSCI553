from pyspark import SparkContext
import platform
import sys
import json
print(platform.python_version())

def get_count_total():
    """The total number of reviews"""
    count_total = rdd_review_data\
                .count()

    results["A"] = count_total

def get_count_year(year):
    """The number of reviews in a given year, y"""
    count_year = rdd_review_data\
                .filter(lambda x: year in x["date"])\
                .count()

    results["B"] = count_year

def get_count_distinct_user():
    """The number of distinct users who have written the reviews"""
    count_distinct_users = rdd_review_data\
                        .map(lambda x: x["user_id"])\
                        .distinct()\
                        .count()

    results["C"] = count_distinct_users

def get_top_m_user(m):
    """Top m users who have the largest number of reviews and its count"""
    top_m_users = rdd_review_data\
                .map(lambda x: (x["user_id"], 1))\
                .reduceByKey(lambda a, b: a+b)\
                .sortBy(lambda x: -x[1])\
                .take(m)

    results["D"] = top_m_users

def trim(word):
    """Helper Function to handle Punctuation."""
    if word not in stop_words:
        str1 = ""
        for c in word:
            if c in stop_punctuation:
                continue
            else:
                str1 += c
        return str1

def get_top_n_words(n):
    """Top n frequent words in the review text"""
    top_n_words = rdd_review_data\
                    .map(lambda x: x["text"])\
                    .flatMap(lambda line: line.lower().split(' ')) \
                    .filter(lambda x: x not in stop_words)\
                    .map(lambda x: (trim(x), 1))\
                    .reduceByKey(lambda a, b: a + b)\
                    .sortBy(lambda x: -x[1])\
                    .keys()\
                    .take(n)

    results["E"] = top_n_words

if __name__ == '__main__':
    input_fp = sys.argv[1]
    output_fp = sys.argv[2]
    stopwords_fp = sys.argv[3]
    year = sys.argv[4]
    m = sys.argv[5]
    n = sys.argv[6]

    sc = SparkContext('local[*]', 'task1')
    rdd_review = sc.textFile(input_fp)

    rdd_review_data = rdd_review\
                        .map(lambda x: json.loads(x))\
                        .cache()

    stop_words = [word.strip() for word in open(stopwords_fp)]
    stop_punctuation = ["(", "[", ",", ".", "!", "?", ":", ";", "]", ")", ""]
    stop_words = stop_words + stop_punctuation
    results = {}

    get_count_total()
    get_count_year(year)
    get_count_distinct_user()
    get_top_m_user(int(m))
    get_top_n_words(int(n))

    with open(output_fp, 'w') as w:
        w.write(json.dumps(results))