from pyspark import SparkContext, SparkConf
import time
import json
import re
import math
import sys

OVERALL_AVG = 3.5

def merge_if_possible_item(u1, user_business_ratings):
    if u1 in user_business_ratings:
        return user_business_ratings[u1]
    else:
        return []

def merge_if_possible_user(b1, bus_user_ratings):
    if b1 in bus_user_ratings:
        return bus_user_ratings[b1]
    else:
        return []


def get_predictions(ub_score_list, model, avg_d):
    """Use the Pearson Similarity Scoring for Predictions."""
    u1, b1 = ub_score_list[0][0], ub_score_list[0][1]
    possible_neighbors = []
    if cf_type == "item_based":
        for bus_sim in ub_score_list[1]:
            bus_, rating = bus_sim[0], bus_sim[1]
            combo = (b1, bus_)
            if combo in model:
                sim_ = model[combo]
                possible_neighbors.append((rating, sim_))
                continue

            else:
                combo = (bus_, b1)
                if combo in model:
                    sim_ = model[combo]
                    possible_neighbors.append((rating, sim_))
                    continue

        k_neighbors = sorted(possible_neighbors, key=lambda x: -x[1])[:5]
        num_ = sum([rating_sim[0] * rating_sim[1] for rating_sim in k_neighbors])
        if num_ == 0:
            if b1 in avg_d:
                return (u1, b1, avg_d[b1])
            else:
                return (u1, b1, OVERALL_AVG)

        denom_	= sum([abs(rating_sim[1]) for rating_sim in k_neighbors])
        if denom_ == 0:
            if b1 in avg_d:
                return (u1, b1, avg_d[b1])
            else:
                return (u1, b1, OVERALL_AVG)

        rating = num_/denom_
        return (u1, b1, rating)

    else:
        u1, b1 = ub_score_list[0][1], ub_score_list[0][0]
        for user_sim in ub_score_list[1]:
            user_, rating = user_sim[0], user_sim[1]
            combo = (u1, user_)
            if combo in model:
                sim_ = model[combo]
                possible_neighbors.append((rating, avg_d[user_], sim_))
                continue

            else:
                combo = (user_, u1)
                if combo in model:
                    sim_ = model[combo]
                    possible_neighbors.append((rating, avg_d[user_], sim_))
                    continue

        k_neighbors = sorted(possible_neighbors, key=lambda x: -x[2])[:5]
        num_ = sum([(rating_sim[0] - rating_sim[1]) * rating_sim[2] for rating_sim in k_neighbors])
        if num_ == 0:
            if u1 in avg_d:
                return (u1, b1, avg_d[u1])
            else:
                return (u1, b1, OVERALL_AVG)

        denom_ = sum([abs(rating_sim[2]) for rating_sim in k_neighbors])
        if denom_ == 0:
            if u1 in avg_d:
                return (u1, b1, avg_d[u1])
            else:
                return (u1, b1, OVERALL_AVG)

        rating = avg_d[u1] + num_ / denom_
        return (u1, b1, rating)

if __name__ == "__main__":
    start = time.time()

    # input_fp_train = sys.argv[1]
    # input_fp_test = sys.argv[2]
    # input_model = sys.argv[3]
    # output_fp = sys.argv[4]
    # cf_type = sys.argv[5]
    # user_avg_fp = "../resource/asnlib/publicdata/user_avg.json"
    # bus_avg_fp = "../resource/asnlib/publicdata/business_avg.json"

    input_fp_train = "./data/train_review.json"
    input_fp_test = "./data/test_review.json"
    input_model = "./data/task3user.model"
    output_fp = "./data/task3user.predict"
    cf_type = "user_based"
    user_avg_fp = "./data/user_avg.json"
    bus_avg_fp = "./data/business_avg.json"

    conf = SparkConf()
    conf.set("spark.executor.memory", "8g")
    conf.set("spark.driver.memory", "8g")
    sc = SparkContext(conf=conf)

    if cf_type == "item_based":
        # Read in the test data: [(u1, b1), (u2, b2), ....]
        test_rdd = sc.textFile(input_fp_test)
        user_business_test_rdd = test_rdd \
            .map(lambda x: json.loads(x)) \
            .map(lambda u_b: (u_b["user_id"], u_b["business_id"])).persist()

        # Read in the train data: [(u1, b1, s1), (u2, b2, s2), ...]
        train_rdd = sc.textFile(input_fp_train)
        user_bus_stars_train_rdd = sc.textFile(input_fp_train) \
            .map(lambda x: json.loads(x)) \
            .map(lambda u_b_s: (u_b_s["user_id"], u_b_s["business_id"], u_b_s["stars"])).persist()

        # Read in the model data: {(b1,b2): sim}
        model_rdd = sc.textFile(input_model)
        b1_b2_sim_model_rdd = model_rdd \
                    .map(lambda x: json.loads(x)) \
                    .map(lambda b1_b2_s: ((b1_b2_s["b1"], b1_b2_s["b2"]), b1_b2_s["sim"]) )\
                    .collectAsMap()

        # Business Avg file: {b1: avg}:
        avg_rdd = sc.textFile(bus_avg_fp)
        bus_avg_d = avg_rdd \
            .map(lambda x: json.loads(x)) \
            .map(lambda x: dict(x)) \
            .flatMap(lambda x: [(key, val) for key, val in x.items()]) \
            .collectAsMap()

        # User with all of it's business ratings: {u1: [(b1, r1), (b2, r2), ...)] train_rdd
        user_business_ratings = user_bus_stars_train_rdd \
                        .map(lambda u_b_s: (u_b_s[0], (u_b_s[1], u_b_s[2]))) \
                        .groupByKey() \
                        .map(lambda u_bsL: (u_bsL[0], list(u_bsL[1])))

        user_business_rating = user_business_test_rdd.leftOuterJoin(user_business_ratings) \
            .map(lambda x: ((x[0], x[1][0]), x[1][1])) \
            .filter(lambda ub_brL: ub_brL[1] != None) \
            .groupByKey() \
            .map(lambda ub_brLL: (ub_brLL[0], [item for sublist in ub_brLL[1] for item in sublist])) \
            .map(lambda ub_brL: get_predictions(ub_brL, b1_b2_sim_model_rdd, bus_avg_d)) \
            .collect()

    else:
        # Read in the test data: [(u1, b1), (u2, b2), ...]
        test_rdd = sc.textFile(input_fp_test)
        bus_user_test_rdd = test_rdd \
            .map(lambda x: json.loads(x)) \
            .map(lambda u_b: (u_b["business_id"], u_b["user_id"])).persist()

        # Read in the train data: [(u1, b1, s1), (u2, b2, s2), ...]
        train_rdd = sc.textFile(input_fp_train)
        user_bus_stars_train_rdd = train_rdd \
            .map(lambda x: json.loads(x)) \
            .map(lambda u_b_s: (u_b_s["user_id"], u_b_s["business_id"], u_b_s["stars"])).persist()

        # Read in the model data: {(u1,u2): sim}
        model_rdd = sc.textFile(input_model)
        u1_u2_sim_model_rdd = model_rdd \
            .map(lambda x: json.loads(x)) \
            .map(lambda u1_u2_s: ((u1_u2_s["u1"], u1_u2_s["u2"]), u1_u2_s["sim"])) \
            .collectAsMap()

        # User Avg file: {u1: avg}:
        avg_rdd = sc.textFile(user_avg_fp)
        user_avg_d = avg_rdd \
            .map(lambda x: json.loads(x)) \
            .map(lambda x: dict(x)) \
            .flatMap(lambda x: [(key, val) for key, val in x.items()]) \
            .collectAsMap()

        # business with all of it's user ratings: {b1: [(u1, r1), (b2, r2), ...)]
        bus_user_ratings = user_bus_stars_train_rdd \
                        .map(lambda u_b_s: (u_b_s[1], (u_b_s[0], u_b_s[2]))) \
                        .groupByKey() \
                        .map(lambda b_usL: (b_usL[0], list(b_usL[1])))

        user_business_rating = bus_user_test_rdd.leftOuterJoin(bus_user_ratings) \
            .map(lambda x: ((x[0], x[1][0]), x[1][1])) \
            .filter(lambda bu_urL: bu_urL[1] != None) \
            .groupByKey() \
            .map(lambda bu_urLL: (bu_urLL[0], [item for sublist in bu_urLL[1] for item in sublist])) \
            .map(lambda bu_urL: get_predictions(bu_urL, u1_u2_sim_model_rdd, user_avg_d)) \
            .collect()

    with open(output_fp, "w") as w:
        for u1_b1_sim in user_business_rating:
            w.write(json.dumps({"user_id": u1_b1_sim[0], "business_id": u1_b1_sim[1], "stars": u1_b1_sim[2]}) + '\n')

    w.close()
    end = time.time()
    print("Duration:", round((end-start), 2))



