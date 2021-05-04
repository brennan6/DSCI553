from pyspark import SparkContext, SparkConf
from itertools import combinations
from collections import defaultdict
import time
import random
import json
import math
import sys
import os

os.environ["PYSPARK_PYTHON"] = '/usr/local/bin/python3.6'
os.environ["PYSPARK_DRIVER_PYTHON"] = '/usr/local/bin/python3.6'

SIGN_COUNT = 60
BANDS_COUNT = 30

def minhash(business_len):
    index_hash_dict = defaultdict(list)
    for sig_i in range(SIGN_COUNT):
        a = random.randint(1, 100000)
        b = random.randint(1, 100000)
        p = 77369
        m = 50000
        for idx in range(business_len):
            index_hash_dict[idx].append(((a * idx + b) % p) % m)

    return index_hash_dict

def reduce_signature(signature_lsts):
    return [min(col) for col in zip(*signature_lsts)]

def get_lsh_combinations(sign_matrix):
    """Check similarity between hashed signatures across bands."""
    print("LSH Combinations.")
    idx_ = 0
    stop_idx = math.floor((SIGN_COUNT/BANDS_COUNT)-1)
    candidates = []
    for band_num in range(BANDS_COUNT):
        d_ = defaultdict(list)
        for sign_ in sign_matrix:
            band_sign = sign_[1][idx_:stop_idx]
            d_[hash(tuple(band_sign))].append(sign_[0])

        for hashed_sign, b_id_list in d_.items():
            if len(b_id_list) > 1:
                for poss_cand in combinations(b_id_list, 2):
                    candidates.append(poss_cand)

        idx_+=int(SIGN_COUNT/BANDS_COUNT)
        stop_idx+=int(SIGN_COUNT/BANDS_COUNT)
    print("END LSH")
    return candidates

def get_jaccard_similarity(u1_bus_stars, u2_bus_stars):
    u1_businesses = u1_bus_stars.keys()
    u2_businesses = u2_bus_stars.keys()

    j_sim = (len(set(u1_businesses).intersection(set(u2_businesses)))) / (len(set(u1_businesses).union(set(u2_businesses))))
    return j_sim


def check_pearson_similarity(b1_user_stars, b2_user_stars):
    b1_users = b1_user_stars.keys()
    b2_users = b2_user_stars.keys()
    corated_items = set(b1_users).intersection(set(b2_users))
    if len(corated_items) < 3:
        return -1

    corated_1_stars = [b1_user_stars[u] for u in corated_items]
    corated_2_stars = [b2_user_stars[u] for u in corated_items]

    avg_1 = sum(corated_1_stars)/len(corated_1_stars)
    avg_2 = sum(corated_2_stars)/len(corated_2_stars )

    adjusted_1_stars = [score - avg_1 for score in corated_1_stars]
    adjusted_2_stars = [score - avg_2 for score in corated_2_stars]

    num_, denom_ = 0, 0
    for i in range(len(corated_items)):
        num_ += adjusted_1_stars[i]*adjusted_2_stars[i]

    if num_ == 0:
        return -1

    denom_1 = math.sqrt(sum([score**2 for score in adjusted_1_stars]))
    denom_2 = math.sqrt(sum([score**2 for score in adjusted_2_stars]))
    denom_ = denom_1*denom_2

    if denom_ == 0:
        return -1

    return num_ / denom_

if __name__ == "__main__":
    start = time.time()

    # input_fp = "../resource/asnlib/publicdata/train_review.json"
    # user_friends_fp = "../resource/asnlib/publicdata/user.json"
    # cf_model = "./item_based_cf.model"
    # content_model = "./content.model"

    input_fp = "./data/train_review.json"
    user_friends_fp = "./data/user.json"
    cf_model = "./output/item_based_cf.model"
    content_model = "./output/content.model"

    # conf = SparkConf().set('spark.driver.host', '127.0.0.1')
    conf = SparkConf()
    conf.set("spark.executor.memory", "2g")
    conf.set("spark.driver.memory", "2g")
    sc = SparkContext(conf=conf)

    train_rdd = sc.textFile(input_fp)
    user_bus_stars_rdd = train_rdd \
        .map(lambda x: json.loads(x)) \
        .map(lambda u_b_s: (u_b_s["user_id"], u_b_s["business_id"], u_b_s["stars"]))

    # Group business (item) with user, stars as value:
    bus_user_ratings = user_bus_stars_rdd \
                    .map(lambda u_b_s: (u_b_s[1], (u_b_s[0], u_b_s[2]))) \
                    .groupByKey() \
                    .filter(lambda b_usL: len(b_usL[1]) >= 3) \
                    .map(lambda b_usL: (b_usL[0], list(b_usL[1]))) \
                    .collectAsMap()

    #ITEM-BASED:
    # Convert user_ids to numeric mapping for hashing:
    user_idx_rdd = user_bus_stars_rdd.map(lambda u_b_s: u_b_s[0]) \
        .distinct() \
        .sortBy(lambda u: u) \
        .zipWithIndex() \
        .map(lambda u_i: (u_i[0], u_i[1]))

    user_idx_dict = {user_idx[0]: user_idx[1] for user_idx in user_idx_rdd.collect()}

    # Create Min_hash Signatures:
    index_hash_dict = minhash(len(user_idx_dict))

    idx_bus = user_bus_stars_rdd \
        .map(lambda u_b_s: (user_idx_dict[u_b_s[0]], u_b_s[1]))

    # Create Signature Matrix:
    signature_matrix = idx_bus \
        .groupByKey() \
        .map(lambda i_bL: (i_bL[0], list(set(i_bL[1])))) \
        .map(lambda i_bL: (i_bL[1], index_hash_dict[i_bL[0]])) \
        .map(lambda bL_hL: [(b_id, bL_hL[1]) for b_id in bL_hL[0]]) \
        .flatMap(lambda b_hL: b_hL) \
        .groupByKey() \
        .map(lambda b_hLL: (b_hLL[0], reduce_signature(b_hLL[1]))) \
        .collect()

    # Implement LSH:
    cands_set = set(list(get_lsh_combinations(signature_matrix)))

    #Group Business Ratings for Users we need to compare:
    #print("Cand Set size:", len(cands_set))
    output_pairs = []
    for pair in cands_set:
        b1 = pair[0]
        b2 = pair[1]
        if b1 not in bus_user_ratings or b2 not in bus_user_ratings:
            continue

        b1_user_stars = dict(bus_user_ratings[b1])
        b2_user_stars = dict(bus_user_ratings[b2])
        j_sim = get_jaccard_similarity(b1_user_stars, b2_user_stars)

        if j_sim < .01:
            continue

        pearson_sim = check_pearson_similarity(b1_user_stars, b2_user_stars)
        if pearson_sim > 0:
            output_pairs.append((b1, b2, pearson_sim))

        # business_lst = bus_user_ratings.keys()
        #
        # output_pairs = []
        # for pair in combinations(business_lst, 2):
        #     b1_user_stars = dict(bus_user_ratings[pair[0]])
        #     b2_user_stars = dict(bus_user_ratings[pair[1]])
        #
        #     pearson_sim = check_pearson_similarity(b1_user_stars, b2_user_stars)
        #
        #     if pearson_sim > 0:
        #         output_pairs.append((pair[0], pair[1], pearson_sim))

    with open(cf_model, "w") as w:
        for b1_b2_sim in output_pairs:
            w.write(json.dumps({"b1": b1_b2_sim[0], "b2": b1_b2_sim[1], "sim": b1_b2_sim[2]}) + '\n')
    w.close()

    # Read in User Data for Content Based:
    user_idx_d = user_bus_stars_rdd.map(lambda u_b_s: u_b_s[0]) \
                .distinct() \
                .sortBy(lambda u: u) \
                .zipWithIndex() \
                .map(lambda u_i: (u_i[0], u_i[1])) \
                .collectAsMap()

    idx_user_d = {idx: user for user, idx in user_idx_d.items()}

    user_friends_rdd = sc.textFile(user_friends_fp)
    user_friends_d = user_friends_rdd \
                    .map(lambda x: json.loads(x)) \
                    .map(lambda x: ((x["user_id"], x["useful"]), x["friends"].split(","))) \
                    .flatMap(lambda uv_f: [(friend.strip(), (uv_f[0][0], uv_f[0][1])) for friend in uv_f[1]]) \
                    .map(lambda f_uv: (f_uv[0], (user_idx_d[f_uv[1][0]] if f_uv[1][0] in user_idx_d
                                                                        else "unknown_user",
                                       f_uv[1][1]))) \
                    .groupByKey() \
                    .map(lambda f_uvL: (f_uvL[0], list(f_uvL[1]))) \
                    .filter(lambda f_uvL: len(f_uvL[1]) > 0) \
                    .map(lambda f_uvL: (f_uvL[0], sorted(f_uvL[1], key=lambda item: -item[1])[:5])) \
                    .map(lambda f_uvL: (f_uvL[0], [idx_user_d[u_v[0]] for u_v in f_uvL[1]])) \
                    .collect()

    with open(content_model, "w") as w:
        for user_friends in output_pairs:
            w.write(json.dumps({"u1": user_friends[0], "friends": user_friends[1]}) + '\n')
    w.close()

    end = time.time()
    print("Duration:", round((end-start), 2))
