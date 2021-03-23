from pyspark import SparkContext, SparkConf
import time
import json
import math
import sys

def cos_similarity(user_k, bus_k, user_features_d, bus_features_d):
    if (user_k not in user_features_d) or (bus_k not in bus_features_d):
        return .00001
    user_features = set(user_features_d[user_k])
    bus_features = set(bus_features_d[bus_k])
    num_ = len(bus_features.intersection(user_features))
    denom_ = math.sqrt(len(bus_features)) * math.sqrt(len(user_features))
    return num_/denom_

if __name__ == "__main__":
    start = time.time()
    #input_fp = sys.argv[1]
    #model_fp = sys.argv[2]
    #output_fp = sys.argv[3]

    input_fp = "./data/test_review.json"
    model_fp = "./data/task2.model"
    output_fp = "./data/task2.predict"

    conf = SparkConf()
    conf.set("spark.executor.memory", "8g")
    conf.set("spark.driver.memory", "8g")
    sc = SparkContext(conf=conf)

    # Read in test data:
    test_rdd = sc.textFile(input_fp)
    user_business_rdd = test_rdd \
        .map(lambda x: json.loads(x)) \
        .map(lambda u_b: (u_b["user_id"], u_b["business_id"]))

    # Read in model components:
    model = sc.textFile(model_fp)
    business_profiles = model \
        .map(lambda x: json.loads(x)) \
        .map(lambda x: x["business"]) \
        .flatMap(lambda x: [(k["business_id"], k["feature_vector"]) for k in x]) \
        .collectAsMap()

    user_profiles = model \
        .map(lambda x: json.loads(x)) \
        .map(lambda x: x["user"]) \
        .flatMap(lambda x: [(k["user_id"], k["feature_vector"]) for k in x]) \
        .collectAsMap()

    # Find User-Business recommendations:
    rec_user_businesses = user_business_rdd \
                        .map(lambda u_b: (u_b[0], u_b[1], cos_similarity(u_b[0], u_b[1], user_profiles, business_profiles))) \
                        .filter(lambda u_b_s: u_b_s[2] >= .01) \
                        .collect()

    with open(output_fp, "w") as w:
        for u_b_sim in rec_user_businesses:
            w.write(json.dumps({"user_id": u_b_sim[0], "business_id": u_b_sim[1], "sim": u_b_sim[2]}) + '\n')

    w.close()
    end = time.time()
    print("Duration:", round((end-start),2))
