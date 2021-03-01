from pyspark import SparkContext, SparkConf, StorageLevel
from itertools import combinations
import math
import sys
import time

def get_candidates_apriori(baskets, s, total_basket_count):
    #Get Count of all candidate itemsets within partition
    #A prior: Count singletons, eliminate not plausible combos, try all for 2, then 3, etc.
    #RUN WHOLE ALGORITHM HERE, b/c still "candidates" since SOM
    #print("Started Candidates")
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

    #print("support threshold partition:", threshold)
    #print("1 len candidates: ", len(cand_lst))

    cand_set_singletons = sorted(list(set(cand_lst))) #Removes duplicates from list of singletons
    pairs_possible = sorted(list(combinations(cand_set_singletons, 2)))
    #print("Finished Singletons.")

    size = 2
    while pairs_possible:
        #print("start:", size)
        cnt = {}
        cand_lst = []
        for basket in basket_lst:
            pruned_basket = sorted(list(set(basket).intersection(set(cand_set_singletons))))
            combos = list(combinations(pruned_basket, size))
            approved_combos = list(set(pairs_possible).intersection(set(combos)))
            if len(combos) == 0:
                continue
            for pair in approved_combos:
                if pair in cnt:
                    cnt[pair] += 1
                    if cnt[pair] >= threshold:
                        cand_lst.append(pair)
                        yield (pair, 1)
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

        #print(str(size) + " len candidates: ", len(cand_set_non_singletons))
        size += 1
        pairs_possible = sorted(list(set(pairs_possible)))
        cand_set_singletons = sorted(list(set([item for tup in pairs_possible for item in tup])))
        #Try flattening this set into a bucket. as new singletons
        #print("Finished size:", size)
    #print("Finished Candidates")

def get_frequent_itemsets(baskets, candidates):
    #print("StartedFrequents")
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
    #print("Finished Frequents")

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
            str_ = "('" + fi + "')\n\n"
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
                        str_ = str_ + "'" + str(fi_pair[i]) + "')\n\n"
                w.write(str_)
        size += 1

if __name__ == "__main__":
    start = time.time()
    filter_threshold = int(sys.argv[1])
    support = int(sys.argv[2])
    input_fp = sys.argv[3]
    output_fp = sys.argv[4]

    #filter_threshold = int("70")
    #support = int("50")
    #input_fp = "./data/user_business.csv"
    #output_fp = "./output2.txt"

    configuration = SparkConf()
    configuration.set("spark.driver.memory", "10g")
    configuration.set("spark.executor.memory", "10g")
    sc = SparkContext.getOrCreate(configuration)

    input_rdd = sc.textFile(input_fp)
    header = input_rdd.first()
    reviews_rdd = input_rdd\
                .filter(lambda row: row != header)

    basket = reviews_rdd\
        .map(lambda line: line.split(","))\
        .groupByKey()\
        .map(lambda user_businesses: (user_businesses[0], sorted(list(set(user_businesses[1])))))\
        .filter(lambda user_businessLst: len(user_businessLst[1]) > filter_threshold)\
        .map(lambda usr_basket: usr_basket[1]).persist(StorageLevel.DISK_ONLY)

    #num_partitions = basket.getNumPartitions()
    total_basket_count = basket.count()
    #print("num_partitions", num_partitions)

    candidate_itemset = basket.mapPartitions(lambda partition:
                get_candidates_apriori(baskets=partition, s=support, total_basket_count=total_basket_count)) \
                .reduceByKey(lambda x, y: x + y) \
                .distinct()\
                .keys()\
                .collect()
    #print(candidate_itemset)

    frequent_itemsets = basket.mapPartitions(lambda partition:
                            get_frequent_itemsets(baskets=partition, candidates=candidate_itemset))\
                            .reduceByKey(lambda x, y: x+y)\
                            .filter(lambda key_cnt: key_cnt[1] >= support)\
                            .keys()\
                            .collect()
    #print(frequent_itemsets)

    singletons_cand = []
    non_singletons_cand = []
    for cand in candidate_itemset:
        if type(cand) is not tuple:
            singletons_cand.append(cand)
        else:
            non_singletons_cand.append(cand)

    singletons_cand = sorted(singletons_cand)

    #print(singletons_cand)
    #print(non_singletons_cand)

    singletons_fi = []
    non_singletons_fi = []
    for freq_item in frequent_itemsets:
        if type(freq_item) is not tuple:
            singletons_fi.append(freq_item)
        else:
            non_singletons_fi.append(freq_item)

    singletons_fi = sorted(singletons_fi)

    #print(singletons_fi)
    #print(non_singletons_fi)

    with open(output_fp, "w") as w:
        format_output(w, "Candidates:\n", singletons_cand, non_singletons_cand)
        format_output(w, "Frequent Itemsets:\n", singletons_fi, non_singletons_fi)

    with open("saved_task2.txt", "w") as f:
        for fi in frequent_itemsets:
            f.write(str(fi) + "\n")

    end = time.time()
    print("Duration:", round((end-start),2))