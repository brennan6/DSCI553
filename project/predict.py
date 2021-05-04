from pyspark import SparkContext, SparkConf
import time
import json
import os
import re
import math
import sys

os.environ["PYSPARK_PYTHON"] = '/usr/local/bin/python3.6'
os.environ["PYSPARK_DRIVER_PYTHON"] = '/usr/local/bin/python3.6'
OVERALL_AVG = 3.8

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
    score_list = ub_score_list[1]

    possible_neighbors = []
    for bus_sim in score_list:
        bus_, rating = idx_bus_d[bus_sim[0]], bus_sim[1]
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

def get_friends_recs(user_business, user_friends_d):
    u1 = user_business[0]
    b1 = user_business[1]
    hasBusinessAvg = False
    hasUserAvg = False
    hasFriendsAvg = False

    if u1 in user_friends_d:
        friends_lst = user_friends_d[u1]
    else:
        friends_lst = None
    if b1 in bus_avg_d:
        business_avg = bus_avg_d[b1]
        hasBusinessAvg = True
    if u1 in bus_avg_d:
        user_avg = user_avg_d[u1]
        hasUserAvg = True
    if friends_lst is not None:
        friends_score = 0
        deduct = 0
        for friend in friends_lst:
            if friend in user_avg_d:
                friends_score += user_avg_d[friend]
            else:
                deduct += 1
        friends_avg = friends_score/(len(friends_lst)-deduct)
        if friends_avg != 0:
            hasFriendsAvg = True

    if hasBusinessAvg is False and hasUserAvg is True:
        final_score = .8*user_avg + .2*OVERALL_AVG
        return (u1, b1, final_score)
    elif hasBusinessAvg is True and hasUserAvg is False:
        if hasFriendsAvg:
            final_score = .8*friends_avg + .2*business_avg
            return (u1, b1, final_score)
        else:
            final_score = .8*business_avg + .2*OVERALL_AVG
            return (u1, b1, final_score)
    else:
        return (u1, b1, OVERALL_AVG)


def adjust_prediction(user, bus, cf_score):
    if user in user_avg_d:
        avg_user = user_avg_d[user]
    else:
        avg_user = OVERALL_AVG

    if bus in bus_avg_d:
        avg_bus = bus_avg_d[bus]
    else:
        avg_bus = OVERALL_AVG

    avg_user_bus = .6*avg_bus + .4*avg_user

    if avg_user_bus > 1.5:
        if avg_user_bus > 4.4:
            return cf_score*.1 + avg_user_bus*.9
        else:
            return cf_score*.8 + avg_user_bus*.2
    else:
        output_score = .2*cf_score + .8*avg_user_bus
        return output_score


if __name__ == "__main__":
    start = time.time()

    input_fp_train = "../resource/asnlib/publicdata/train_review.json"
    input_fp_test = sys.argv[1]
    cf_model = "./item_based_cf.model"
    content_model = "./content.model"
    output_fp = sys.argv[2]
    user_avg_fp = "../resource/asnlib/publicdata/user_avg.json"
    bus_avg_fp = "../resource/asnlib/publicdata/business_avg.json"

    # input_fp_train = "./data/train_review.json"
    # input_fp_test = "./data/test_review.json"
    # cf_model = "./output/item_based_cf.model"
    # content_model = "./output/content.model"
    # output_fp = "./output/project.predict"
    # user_avg_fp = "./data/user_avg.json"
    # bus_avg_fp = "./data/business_avg.json"

    # conf = SparkConf().set('spark.driver.host', '127.0.0.1')
    conf = SparkConf()
    conf.set("spark.executor.memory", "2g")
    conf.set("spark.driver.memory", "2g")
    sc = SparkContext(conf=conf)

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

    user_idx_d = user_bus_stars_train_rdd.map(lambda u_b_s: u_b_s[0]) \
                .distinct() \
                .sortBy(lambda u: u) \
                .zipWithIndex() \
                .map(lambda u_i: (u_i[0], u_i[1])) \
                .collectAsMap()

    idx_user_d = {idx: user for user, idx in user_idx_d.items()}

    bus_idx_d = user_bus_stars_train_rdd.map(lambda u_b_s: u_b_s[1]) \
                .distinct() \
                .sortBy(lambda b: b) \
                .zipWithIndex() \
                .map(lambda b_i: (b_i[0], b_i[1])) \
                .collectAsMap()

    idx_bus_d = {idx: bus for bus, idx in bus_idx_d.items()}

    # Read in the model data: {(b1,b2): sim}
    model_rdd = sc.textFile(cf_model)
    b1_b2_sim_model_rdd = model_rdd \
                .map(lambda x: json.loads(x)) \
                .map(lambda b1_b2_s: ((b1_b2_s["b1"], b1_b2_s["b2"]), b1_b2_s["sim"]))\
                .collectAsMap()

    # Business Avg file: {b1: avg}:
    bus_avg_rdd = sc.textFile(bus_avg_fp)
    bus_avg_d = bus_avg_rdd \
        .map(lambda x: json.loads(x)) \
        .map(lambda x: dict(x)) \
        .flatMap(lambda x: [(key, val) for key, val in x.items()]) \
        .collectAsMap()

    # User Avg file: {u1: avg}:
    user_avg_rdd = sc.textFile(user_avg_fp)
    user_avg_d = user_avg_rdd \
            .map(lambda x: json.loads(x)) \
            .map(lambda x: dict(x)) \
            .flatMap(lambda x: [(key, val) for key, val in x.items()]) \
            .collectAsMap()

    # User with all of it's business ratings: {u1idx: [(b1idx, r1), (b2idx, r2), ...)] train_rdd
    user_business_ratings = user_bus_stars_train_rdd \
                    .map(lambda u_b_s: (user_idx_d[u_b_s[0]], (bus_idx_d[u_b_s[1]], u_b_s[2]))) \
                    .groupByKey() \
                    .map(lambda u_bsL: (u_bsL[0], list(u_bsL[1])))

    print(user_business_test_rdd.count())

    user_business_test_mapped = user_business_test_rdd \
                    .map(lambda u_b: (user_idx_d[u_b[0]] if u_b[0] in user_idx_d
                                                        else u_b[0],
                                      bus_idx_d[u_b[1]] if u_b[1] in bus_idx_d
                                                        else u_b[1]))

    user_business_rating = user_business_test_mapped \
        .filter(lambda u_b: isinstance(u_b[0], int) and isinstance(u_b[1], int)) \
        .leftOuterJoin(user_business_ratings) \
        .map(lambda x: ((idx_user_d[x[0]], idx_bus_d[x[1][0]]), x[1][1])) \
        .filter(lambda ub_brL: ub_brL[1] != None) \
        .groupByKey() \
        .map(lambda ub_brLL: (ub_brLL[0], [item for sublist in ub_brLL[1] for item in sublist])) \
        .map(lambda ub_brL: get_predictions(ub_brL, b1_b2_sim_model_rdd, bus_avg_d)) \
        .map(lambda u_b_r: (u_b_r[0], u_b_r[1], adjust_prediction(u_b_r[0], u_b_r[1], u_b_r[2]))) \
        .collect()

    print(len(user_business_rating))

    #Utilize friends for additional content:
    # Read in the model data: {(b1,b2): sim}
    user_friends_model_rdd = sc.textFile(content_model)
    user_friends_model_d = user_friends_model_rdd \
                .map(lambda x: json.loads(x)) \
                .map(lambda u_fL: (u_fL["u1"], u_fL["friends"]))\
                .collectAsMap()

    cold_start_ratings = user_business_test_mapped \
        .filter(lambda u_b: isinstance(u_b[0], str) or isinstance(u_b[1], str)) \
        .map(lambda u_b: (idx_user_d[u_b[0]] if u_b[0] in idx_user_d
                                             else u_b[0],
                          idx_bus_d[u_b[1]] if u_b[1] in idx_bus_d
                                             else u_b[1])) \
        .map(lambda u_b: get_friends_recs(u_b, user_friends_model_d)) \
        .collect()

    user_business_ratings_final = list(set(user_business_rating).union(set(cold_start_ratings)))

    with open(output_fp, "w") as w:
        for u1_b1_sim in user_business_ratings_final:
            w.write(json.dumps({"user_id": u1_b1_sim[0], "business_id": u1_b1_sim[1], "stars": u1_b1_sim[2]}) + '\n')

    w.close()
    end = time.time()
    print("Duration:", round((end-start), 2))



