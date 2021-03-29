from pyspark import SparkContext, SparkConf
from itertools import combinations
from collections import defaultdict
import time
import random
import json
import math
import sys

SIGN_COUNT = 50
BANDS_COUNT = 25

def minhash(business_len):
    index_hash_dict = defaultdict(list)
    for sig_i in range(SIGN_COUNT):
        a = random.randint(1, 50000)
        b = random.randint(1, 50000)
        p = 15121
        m = 15000
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

    # input_fp = sys.argv[1]
    # output_model = sys.argv[2]
    # cf_type = sys.argv[3]

    input_fp = "./data/train_review.json"
    output_model = "./data/task3item.model"
    cf_type = "item_based"

    conf = SparkConf()
    conf.set("spark.executor.memory", "8g")
    conf.set("spark.driver.memory", "8g")
    sc = SparkContext(conf=conf)

    train_rdd = sc.textFile(input_fp)
    user_bus_stars_rdd = train_rdd \
        .map(lambda x: json.loads(x)) \
        .map(lambda u_b_s: (u_b_s["user_id"], u_b_s["business_id"], u_b_s["stars"]))

    if cf_type == "item_based":
        # Group business (item) with user, stars as value:
        bus_user_ratings = user_bus_stars_rdd \
                        .map(lambda u_b_s: (u_b_s[1], (u_b_s[0], u_b_s[2]))) \
                        .groupByKey() \
                        .filter(lambda b_usL: len(b_usL[1]) >= 3) \
                        .map(lambda b_usL: (b_usL[0], list(b_usL[1]))) \
                        .collectAsMap()

        business_lst = bus_user_ratings.keys()

        output_pairs = []
        for pair in combinations(business_lst, 2):
            b1_user_stars = dict(bus_user_ratings[pair[0]])
            b2_user_stars = dict(bus_user_ratings[pair[1]])

            pearson_sim = check_pearson_similarity(b1_user_stars, b2_user_stars)

            if pearson_sim > 0:
                output_pairs.append((pair[0], pair[1], pearson_sim))

    else:
        user_bus_ratings = user_bus_stars_rdd \
            .map(lambda u_b_s: (u_b_s[0], (u_b_s[1], u_b_s[2]))) \
            .groupByKey() \
            .filter(lambda u_bsL: len(u_bsL[1]) >= 3) \
            .map(lambda u_bsL: (u_bsL[0], list(u_bsL[1]))) \
            .collectAsMap()

        # Convert business_ids to numeric mapping for hashing:
        bus_idx_rdd = user_bus_stars_rdd.map(lambda u_b_s: u_b_s[1]) \
            .distinct() \
            .sortBy(lambda b: b) \
            .zipWithIndex() \
            .map(lambda b_i: (b_i[0], b_i[1]))

        bus_idx_dict = {bus_idx[0]: bus_idx[1] for bus_idx in bus_idx_rdd.collect()}

        # Create Min_hash Signatures:
        index_hash_dict = minhash(len(bus_idx_dict))

        idx_user = user_bus_stars_rdd \
            .map(lambda u_b_s: (bus_idx_dict[u_b_s[1]], u_b_s[0]))

        # Create Signature Matrix:
        signature_matrix = idx_user \
            .groupByKey() \
            .map(lambda i_uL: (i_uL[0], list(set(i_uL[1])))) \
            .map(lambda i_uL: (i_uL[1], index_hash_dict[i_uL[0]])) \
            .map(lambda uL_hL: [(u_id, uL_hL[1]) for u_id in uL_hL[0]]) \
            .flatMap(lambda u_hL: u_hL) \
            .groupByKey() \
            .map(lambda u_hLL: (u_hLL[0], reduce_signature(u_hLL[1]))) \
            .collect()

        # Implement LSH:
        cands_set = set(list(get_lsh_combinations(signature_matrix)))

        #Group Business Ratings for Users we need to compare:
        #print("Cand Set size:", len(cands_set))
        output_pairs = []
        for pair in cands_set:
            u1 = pair[0]
            u2 = pair[1]
            if u1 not in user_bus_ratings or u2 not in user_bus_ratings:
                continue

            u1_bus_stars = dict(user_bus_ratings[u1])
            u2_bus_stars = dict(user_bus_ratings[u2])
            j_sim = get_jaccard_similarity(u1_bus_stars, u2_bus_stars)

            if j_sim < .01:
                continue

            pearson_sim = check_pearson_similarity(u1_bus_stars, u2_bus_stars)
            if pearson_sim > 0:
                output_pairs.append((u1, u2, pearson_sim))

    with open(output_model, "w") as w:
        if cf_type == "item_based":
            for b1_b2_sim in output_pairs:
                w.write(json.dumps({"b1": b1_b2_sim[0], "b2": b1_b2_sim[1], "sim": b1_b2_sim[2]}) + '\n')
        else:
            for u1_u2_sim in output_pairs:
                w.write(json.dumps({"u1": u1_u2_sim[0], "u2": u1_u2_sim[1], "sim": u1_u2_sim[2]}) + '\n')

    w.close()
    end = time.time()
    print("Duration:", round((end-start), 2))
