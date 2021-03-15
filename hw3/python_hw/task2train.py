from pyspark import SparkContext, SparkConf
from collections import defaultdict
import time
import json
import re
import math

def remove_stopwords(text):
    text_cleaned = []
    for w in text:
        w = re.sub(r'[^a-z\s]', '', w)
        if "\n" in w:
            w = re.sub(r'[\n]', ' ', w)
        if w in stop_words:
            continue
        else:
            w = w.strip()
            if " " in w:
                w_list = w.split(" ")
                for word in w_list:
                    if word in stop_words:
                        continue
                    else:
                        text_cleaned.append(word)
            else:
                text_cleaned.append(w)
    return text_cleaned
    #cleaned_word_list = [word for word in word_list if word not in stopwords]

def calculate_tf(document):
    tf_d = {}
    for review in document:
        for word in review:
            if word in tf_d:
                tf_d[word] += 1
            else:
                tf_d[word] = 1

    max_tf = tf_d[max(tf_d, key=lambda k: tf_d[k])]
    tf_d = {k: v/max_tf for k,v in tf_d.items()}

    return list(tf_d.items())


if __name__ == "__main__":
    start = time.time()
    input_fp = "./data/train_review.json"
    output_model = "./data/task3_model"
    stopwords_fp = "./data/stopwords"
    conf = SparkConf()
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.driver.memory", "4g")
    sc = SparkContext.getOrCreate(conf)

    stop_words = [word.strip() for word in open(stopwords_fp)]
    stop_words = stop_words + [""]  #In case empty string returns.

    train_rdd = sc.textFile(input_fp)
    business_review_rdd = train_rdd \
        .map(lambda x: json.loads(x)) \
        .map(lambda b_r: (b_r["business_id"], b_r["text"]))

    num_documents = len(business_review_rdd.map(lambda x: x[0]).distinct().collect())

    clean_reviews_rdd = business_review_rdd \
        .map(lambda b_r: (b_r[0], b_r[1].lower().split(' '))) \
        .map(lambda b_r: (b_r[0], remove_stopwords(b_r[1])))

    tf_rdd = clean_reviews_rdd \
        .groupByKey() \
        .map(lambda b_rLL: (b_rLL[0], list(b_rLL[1]))) \
        .map(lambda b_rLL_tfidf: (b_rLL_tfidf[0], calculate_tf(b_rLL_tfidf[1])))

    idf_rdd = tf_rdd \
                .flatMap(lambda b_wtfL: [(w_tf[0], w_tf[1]) for w_tf in b_wtfL[1]]) \
                .groupByKey() \
                .map(lambda w_tfL: (w_tfL[0], list(w_tfL[1]))) \
                .map(lambda w_tfL: (w_tfL[0], math.log2(num_documents/len(w_tfL[1])))) \
                .collectAsMap()

    tf_idf_200_rdd = tf_rdd \
                .flatMap(lambda b_wtfL: [(b_wtfL[0], (w_tf[0], w_tf[1]*idf_rdd[w_tf[0]])) for w_tf in b_wtfL[1]]) \
                .groupByKey() \
                .map(lambda b_wtfidfL: (b_wtfidfL[0], list(b_wtfidfL[1]))) \
                .map(lambda b_wtfidfL: (b_wtfidfL[0], sorted(b_wtfidfL[1], key = lambda x: -x[1])[:200])) \
                .take(5)
        #.filter(lambda b_wL: b_wL[1] not in stop_words) \
        #.take(5)
        #.groupByKey() \
        #.map(lambda b_rL: (b_rL[0], list(b_rL[1]))) \
        #.map(lambda b_rL: [])

    for val in tf_idf_200_rdd:
        print(val)



