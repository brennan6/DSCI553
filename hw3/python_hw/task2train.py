from pyspark import SparkContext, SparkConf
import time
import json
import re
import math
import sys

def remove_stopwords(text):
    text_cleaned = []
    for w in text:
        w = re.sub(r'[^a-z\s]', ' ', w)
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
                if w in stop_words:
                    continue
                else:
                    text_cleaned.append(w)
    return text_cleaned

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
    # input_fp = sys.argv[1]
    # output_model = sys.argv[2]
    # stopwords_fp = sys.argv[3]
    input_fp = "./data/train_review.json"
    output_model = "./data/task2.model"
    stopwords_fp = "./data/stopwords"

    conf = SparkConf()
    conf.set("spark.executor.memory", "8g")
    conf.set("spark.driver.memory", "8g")
    sc = SparkContext(conf=conf)

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

    # Create Business Profiles:
    business_profiles_w_score = tf_rdd \
                .flatMap(lambda b_wtfL: [(b_wtfL[0], (w_tf[0], w_tf[1]*idf_rdd[w_tf[0]])) for w_tf in b_wtfL[1]]) \
                .groupByKey() \
                .map(lambda b_wtfidfL: (b_wtfidfL[0], list(set(b_wtfidfL[1])))) \
                .map(lambda b_wtfidfL: (b_wtfidfL[0], sorted(b_wtfidfL[1], key = lambda x: -x[1])[:200])) \
                .collectAsMap()

    # Format business_profile in dictionary:
    business_profiles = {b: [w[0] for w in wtfidfL] for b, wtfidfL in business_profiles_w_score.items()}

    # Create user_profile:
    user_profiles = train_rdd \
        .map(lambda x: json.loads(x)) \
        .map(lambda u_b: (u_b["user_id"], u_b["business_id"])) \
        .groupByKey() \
        .map(lambda u_bL: (u_bL[0], list(set(u_bL[1])))) \
        .map(lambda u_bL: (u_bL[0], [business_profiles_w_score[b] for b in u_bL[1]])) \
        .map(lambda u_wtfidfLL: (u_wtfidfLL[0], list(set([item for sublist in u_wtfidfLL[1] for item in sublist])))) \
        .map(lambda u_wtfidfL: (u_wtfidfL[0], sorted(u_wtfidfL[1], key=lambda x: -x[1])[:600])) \
        .map(lambda u_wtfidfL: (u_wtfidfL[0], [w_tfidl[0] for w_tfidl in u_wtfidfL[1]])) \
        .collectAsMap()

    with open(output_model, "w") as w:
        business_arr = []
        user_arr = []
        output_d = {}
        for b, wL in business_profiles.items():
            business_arr.append({'business_id': b, 'feature_vector': wL})
        output_d["business"] = business_arr

        for u, wL in user_profiles.items():
            user_arr.append({'user_id': u, 'feature_vector': wL})
        output_d["user"] = user_arr
        w.write(json.dumps(output_d))

    end = time.time()
    print("Duration:", round((end-start),2))