from pyspark import SparkContext
from itertools import combinations
import sys
import math
import time

def get_candidates_apriori(baskets, s, total_basket_count):
    #Get Count of all candidate itemsets within partition
    #A prior: Count singletons, eliminate not plausible combos, try all for 2, then 3, etc.
    #RUN WHOLE ALGORITHM HERE, b/c still "candidates" since SOM

    basket_lst = list(baskets)
    cnt = {}
    cand_lst = []
    threshold = math.ceil(s*(len(basket_lst)/total_basket_count))
    for basket in basket_lst:
        for elem in basket:
            if elem in cnt:
                cnt[elem] += 1
                if cnt[elem] >= threshold:
                    cand_lst.append(elem)
                    yield (elem, 1)
            else:
                cnt[elem] = 1
                if cnt[elem] >= threshold:
                    cand_lst.append(elem)
                    yield (elem, 1)

    cand_set_singletons = sorted(list(set(cand_lst))) #Removes duplicates from list of singletons
    print("cand_set_singletons:", cand_set_singletons)
    pairs_possible = sorted(combinations(cand_set_singletons, 2))

    size = 2
    while pairs_possible:
        cnt = {}
        cand_lst = []

        for basket in basket_lst:
            pruned_basket = sorted(list(set(basket).intersection(set(cand_set_singletons))))
            #print("pruned_basket:", pruned_basket)
            combos = list(combinations(pruned_basket, size))
            #print("combos:", combos)
            #print("pairs possible:", pairs_possible)
            approved_combos = list(set(combos).intersection(set(pairs_possible)))
            #print("approved_combos:", approved_combos)
            if len(approved_combos) == 0:
                continue
            for pair in approved_combos:
                pair = tuple(pair)
                if pair in cnt:
                    cnt[pair] += 1
                    if cnt[pair] >= threshold:
                        cand_lst.append(pair)
                        yield (pair, 1)  #Try removing parentheses
                else:
                    cnt[pair] = 1
                    if cnt[pair] >= threshold:
                        cand_lst.append(pair)
                        yield (pair, 1)

        cand_set_non_singletons = list(set(cand_lst))

        pairs_possible = []
        for i, outer_tup in enumerate(cand_set_non_singletons[:-1]):
            for inner_tup in cand_set_non_singletons[i+1:]:
                isMergable = False
                union_tup = sorted(set(outer_tup).union(set(inner_tup)))
                if len(union_tup) == size+1:
                    isMergable = True
                if isMergable:
                    combination = tuple(union_tup)
                    union_pair_lst = []
                    for pair in combinations(combination, size):
                        union_pair_lst.append(pair)
                    if set(union_pair_lst).issubset(set(cand_set_non_singletons)):
                        pairs_possible.append(combination)
                else:
                    continue
        size += 1
        #print(pairs_possible)
        pairs_possible = sorted(list(set(pairs_possible)))
        cand_set_singletons = sorted(list(set([item for tup in pairs_possible for item in tup])))

def get_frequent_itemsets(baskets, candidates):
    basket_lst = list(baskets)
    cnt = {}
    for basket in basket_lst:
        for cand_ in candidates:
            if type(cand_) is not tuple:
                if cand_ in basket:
                    if cand_ in cnt:
                        cnt[cand_] += 1
                    else:
                        cnt[cand_] = 1
            elif set(cand_).issubset(set(basket)):
                if cand_ in cnt:
                    cnt[cand_] += 1
                else:
                    cnt[cand_] = 1

    for pair, total in cnt.items():
        yield(pair, total)

def format_output(w, starter_str, singletons, non_singletons):
    if non_singletons:
        max_len = len(sorted(non_singletons, key=len)[len(non_singletons) - 1])
    else:
        max_len = 1
    w.write(starter_str)
    singletons_cnt = len(singletons)
    cnt = 0
    for fi in singletons:
        if cnt < (singletons_cnt - 1):
            str_ = "('" + fi + "'),"
            w.write(str_)
            cnt += 1
        else:
            str_ = "('" + fi + "')\n"
            w.write(str_)
    size = 2
    while size <= max_len:
        fi_size = []
        for fi in non_singletons:
            if len(fi) == size:
                fi_size.append(fi)
            else:
                continue
        pair_cnt = len(fi_size)
        fi_size = sorted(fi_size)
        cnt = 0
        for fi_pair in fi_size:
            if cnt < (pair_cnt - 1):
                str_ = "("
                for i in range(len(fi_pair)):
                    if i < len(fi_pair) - 1:
                        str_ = str_ + "'" + str(fi_pair[i]) + "', "
                    else:
                        str_ = str_ + "'" + str(fi_pair[i]) + "'),"
                w.write(str_)
                cnt += 1
            else:
                str_ = "("
                for i in range(len(fi_pair)):
                    if i < len(fi_pair) - 1:
                        str_ = str_ + "'" + str(fi_pair[i]) + "', "
                    else:
                        str_ = str_ + "'" + str(fi_pair[i]) + "')\n"
                w.write(str_)
        size += 1

if __name__ == "__main__":
    start = time.time()
    case_number = "1"
    support = int("5")
    input_fp = "./data/small2.csv"
    output_fp = "./output1.txt"
    sc = SparkContext('local[*]', 'task1')

    input_rdd = sc.textFile(input_fp)
    header = input_rdd.first()
    reviews_rdd = input_rdd\
                .filter(lambda row: row != header)

    if case_number == "1":
        user_baskets = reviews_rdd\
            .map(lambda line: line.split(","))\
            .groupByKey()\
            .map(lambda user_businesses: (user_businesses[0], sorted(list(user_businesses[1]))))\
            .map(lambda usr_basket: usr_basket[1])
        basket = user_baskets
    else:
        business_basket = reviews_rdd\
            .map(lambda line: line.split(","))\
            .map(lambda user_business: (user_business[1], user_business[0]))\
            .groupByKey()\
            .map(lambda businesses_user: (businesses_user[0], sorted(list(businesses_user[1]))))\
            .map(lambda business_basket: (business_basket[1]))
        basket = business_basket

    #num_partitions = basket.getNumPartitions()
    total_basket_count = basket.count()

    candidate_itemset = basket.mapPartitions(lambda partition:
                get_candidates_apriori(baskets=partition, s=support, total_basket_count=total_basket_count))\
                .reduceByKey(lambda x, y: x+y)\
                .distinct()\
                .keys()\
                .collect()

    frequent_itemsets = basket.mapPartitions(lambda partition:
                            get_frequent_itemsets(baskets=partition, candidates=candidate_itemset))\
                            .reduceByKey(lambda x, y: x+y)\
                            .filter(lambda key_cnt: key_cnt[1] >= support)\
                            .keys()\
                            .collect()

    singletons_cand = []
    non_singletons_cand = []
    for cand in candidate_itemset:
        if type(cand) is not tuple:
            singletons_cand.append(cand)
        else:
            non_singletons_cand.append(cand)

    singletons_cand = sorted(singletons_cand)

    singletons_fi = []
    non_singletons_fi = []
    for freq_item in frequent_itemsets:
        if type(freq_item) is not tuple:
            singletons_fi.append(freq_item)
        else:
            non_singletons_fi.append(freq_item)

    singletons_fi = sorted(singletons_fi)

    with open(output_fp, "w") as w:
        format_output(w, "Candidates:\n", singletons_cand, non_singletons_cand)
        w.write("\n")
        format_output(w, "Frequent Itemsets:\n", singletons_fi, non_singletons_fi)

    end = time.time()
    print("Duration:", round((end-start),2))
