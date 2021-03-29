from pyspark import SparkContext, SparkConf
from collections import defaultdict
import random
from itertools import combinations
import json
import math
import time
import sys

SIGN_COUNT = 100
BANDS_COUNT = 50

def minhash(user_idx_len):
    index_hash_dict = defaultdict(list)
    for sig_i in range(SIGN_COUNT):
        a = random.randint(1, 50000)
        b = random.randint(1, 50000)
        p = 22777
        m = 20000
        for idx in range(user_idx_len):
            index_hash_dict[idx].append(((a * idx + b) % p) % m)

    return index_hash_dict

def get_lsh_combinations(sign_matrix):
    """Check similarity between hashed signatures across bands."""
    print("LSH Combinations.")
    idx_ = 0
    stop_idx = math.floor((SIGN_COUNT/BANDS_COUNT)-1)
    candidates = []
    for band_num in range(BANDS_COUNT):
        print("band num:", band_num)
        print("idx:", idx_)
        print("stop idx:", stop_idx)
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

def get_jaccard_similar_set(cands_set, b_i_grouped, t):
    """Calculate the Jaccard similarity between each pair and output similarity if >= .05 (t)."""
    print("Jaccard.")
    confirmed_set = []
    for pair in cands_set:
        pair_0_users = set(list(b_i_grouped[pair[0]]))
        pair_1_users = set(list(b_i_grouped[pair[1]]))
        j_sim = (len(pair_0_users.intersection(pair_1_users)))/(len(pair_0_users.union(pair_1_users)))

        if j_sim >= t:
            confirmed_set.append((pair[0], pair[1], j_sim))

    return set(confirmed_set)

def reduce_signature(signature_lsts):
    return [min(col) for col in zip(*signature_lsts)]

def score_against_ground_truth(confirmed_set, b_i_grouped, t):
    """If turned on, confirms Precision and Recall Scores against ground truth."""
    print("Precision/Recall.")
    correct_lst = []
    for pair in combinations(b_i_grouped.keys(), 2):
        pair_0_users = set(list(b_i_grouped[pair[0]]))
        pair_1_users = set(list(b_i_grouped[pair[1]]))
        j_sim = (len(pair_0_users.intersection(pair_1_users))) / (len(pair_0_users.union(pair_1_users)))

        if j_sim >= t:
            correct_lst.append((pair[0], pair[1], j_sim))

    correct_set = set(correct_lst)
    precision = len(confirmed_set.intersection(correct_set)) / len(confirmed_set)
    recall = len(confirmed_set.intersection(correct_set)) / len(correct_set)

    print("Precision:", precision)
    print("Recall:", recall)

if __name__ == "__main__":
    start = time.time()
    # input_fp = sys.argv[1]
    # output_fp = sys.argv[2]

    input_fp = "./data/train_review.json"
    output_fp = "./data/task1.res"
    conf = SparkConf()
    conf.set("spark.executor.memory", "8g")
    conf.set("spark.driver.memory", "8g")
    sc = SparkContext(conf=conf)

    train_rdd = sc.textFile(input_fp)
    user_business_rdd = train_rdd \
        .map(lambda x: json.loads(x)) \
        .map(lambda u_b: (u_b["user_id"], u_b["business_id"]))

    # Convert user_ids to numeric mapping for hashing:
    user_idx_rdd = user_business_rdd.map(lambda u_b: u_b[0])\
        .distinct() \
        .sortBy(lambda u: u) \
        .zipWithIndex() \
        .map(lambda u_i: (u_i[0], u_i[1]))

    user_idx_dict = {user_id[0]: user_id[1] for user_id in user_idx_rdd.collect()}

    # Create Min_hash Signatures:
    index_hash_dict = minhash(len(user_idx_dict))

    idx_business = user_business_rdd \
                .map(lambda u_b: (user_idx_dict[u_b[0]], u_b[1]))

    # Create Signature Matrix:
    signature_matrix = idx_business \
                    .groupByKey() \
                    .map(lambda x: (x[0], list(set(x[1])))) \
                    .map(lambda i_bL: (i_bL[1], index_hash_dict[i_bL[0]])) \
                    .map(lambda bL_hL: [(b_id, bL_hL[1]) for b_id in bL_hL[0]]) \
                    .flatMap(lambda business_hL: business_hL) \
                    .groupByKey() \
                    .map(lambda business_hLL: (business_hLL[0], reduce_signature(business_hLL[1]))) \
                    .collect()

    #Implement LSH:
    cands_set = set(list(get_lsh_combinations(signature_matrix)))

    b_i_grouped = user_business_rdd \
                        .map(lambda u_b: (u_b[1], user_idx_dict[u_b[0]])) \
                        .groupByKey() \
                        .map(lambda b_uL: (b_uL[0], set(b_uL[1]))) \
                        .collectAsMap()

    #Calulate Jaccard Across Candidates
    jaccard_sim_set = get_jaccard_similar_set(cands_set, b_i_grouped, .05)

    with open(output_fp, "w") as w:
        for sim_set in jaccard_sim_set:
            w.write(json.dumps({"b1": sim_set[0], "b2": sim_set[1], "sim": sim_set[2]}) + '\n')

    #Score the results against ground truth:
    #score_against_ground_truth(jaccard_sim_set, b_i_grouped, .05)

    w.close()
    end = time.time()
    print("Duration:", round((end-start),2))



