from pyspark import SparkContext, SparkConf
import time
import json
import re
import math
import sys

def merge_if_possible(u1, user_business_ratings):
    if u1 in user_business_ratings:
        return user_business_ratings[u1]
    else:
        return []


def item_predict(ub_bsL, b1_b2_sim_model_rdd):
    """Use the Pearson Similarity Scoring for Predictions."""
    u1, b1 = ub_bsL[0][0], ub_bsL[0][1]
    possible_neighbors = []
    for bus_sim in ub_bsL[1]:
        bus_, rating = bus_sim[0], bus_sim[1]
        combo = (b1, bus_)
        if combo in b1_b2_sim_model_rdd:
            sim_ = b1_b2_sim_model_rdd[combo]
            possible_neighbors.append((rating, sim_))
            continue

        else:
            combo = (bus_, b1)
            if combo in b1_b2_sim_model_rdd:
                sim_ = b1_b2_sim_model_rdd[combo]
                possible_neighbors.append((rating, sim_))
                continue

    k_neighbors = sorted(possible_neighbors, key=lambda x: -x[1])[:5]
    num_ = sum([rating_sim[0] * rating_sim[1] for rating_sim in k_neighbors])
    if num_ == 0:
        return -1

    denom_	= sum([abs(rating_sim[1]) for rating_sim in k_neighbors])
    if denom_ == 0:
        return -1

    rating = num_/denom_
    return (u1, b1, rating)

if __name__ == "__main__":
    start = time.time()

    input_fp_train = sys.argv[1]
    input_fp_test = sys.argv[2]
    input_model = sys.argv[3]
    output_fp = sys.argv[4]
    cf_type = sys.argv[5]

    # input_fp_train = "./data/train_review.json"
    # input_fp_test = "./data/test_review.json"
    # input_model = "./data/task3item.model"
    # output_fp = "./data/task3item.predict"
    # cf_type = "item_based"

    conf = SparkConf()
    conf.set("spark.executor.memory", "8g")
    conf.set("spark.driver.memory", "8g")
    sc = SparkContext(conf=conf)

    if cf_type == "item_based":
        # Read in the test data: [(u1, b1), (u2, b2), ...]
        test_rdd = sc.textFile(input_fp_test)
        user_business_test_rdd = test_rdd \
            .map(lambda x: json.loads(x)) \
            .map(lambda u_b: (u_b["user_id"], u_b["business_id"]))

        # Read in the train data:
        train_rdd = sc.textFile(input_fp_train)
        user_bus_stars_train_rdd = train_rdd \
            .map(lambda x: json.loads(x)) \
            .map(lambda u_b_s: (u_b_s["user_id"], u_b_s["business_id"], u_b_s["stars"]))

        # Read in the model data: {(b1,b2): sim}
        model_rdd = sc.textFile(input_model)
        b1_b2_sim_model_rdd = model_rdd \
                    .map(lambda x: json.loads(x)) \
                    .map(lambda b1_b2_s: ((b1_b2_s["b1"], b1_b2_s["b2"]), b1_b2_s["sim"]) )\
                    .collectAsMap()

        # Business with all of it's user ratings: {u1: [(b1, r1), (b2, r2), ...)]
        user_business_ratings = user_bus_stars_train_rdd \
                        .map(lambda u_b_s: (u_b_s[0], (u_b_s[1], u_b_s[2]))) \
                        .groupByKey() \
                        .map(lambda u_bsL: (u_bsL[0], list(u_bsL[1]))) \
                        .collectAsMap()

        user_business_rating = user_business_test_rdd \
                            .map(lambda u_b: ((u_b[0], u_b[1]), merge_if_possible(u_b[0], user_business_ratings))) \
                            .groupByKey() \
                            .map(lambda ub_brLL: (ub_brLL[0], list(ub_brLL[1]))) \
                            .map(lambda ub_brLL: (ub_brLL[0], [item for sublist in ub_brLL[1] for item in sublist])) \
                            .map(lambda ub_brL: item_predict(ub_brL, b1_b2_sim_model_rdd)) \
                            .filter(lambda u_b_r: u_b_r != -1) \
                            .collect()

    else:
        str = 1


    with open(output_fp, "w") as w:
        for u1_b1_sim in user_business_rating:
            w.write(json.dumps({"user_id": u1_b1_sim[0], "business_id": u1_b1_sim[1], "stars": u1_b1_sim[2]}) + '\n')

    w.close()
    end = time.time()
    print("Duration:", round((end-start), 2))



