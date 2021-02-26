from pyspark import SparkContext, SparkConf, StorageLevel
from itertools import combinations
import math
import sys
import numpy as np
from pyspark.mllib.fpm import FPGrowth
import time


if __name__ == "__main__":
    start = time.time()
    filter_threshold = int("70")
    support = int("50")
    input_fp = "./data/user_business.csv"
    output_fp = "./output3.txt"

    sc = SparkContext('local[*]', 'task3')

    input_rdd = sc.textFile(input_fp)
    header = input_rdd.first()
    reviews_rdd = input_rdd\
                .filter(lambda row: row != header)

    basket = reviews_rdd\
        .map(lambda line: line.split(","))\
        .groupByKey()\
        .map(lambda user_businesses: (user_businesses[0], sorted(list(user_businesses[1]))))\
        .filter(lambda user_businessLst: len(user_businessLst[1]) > filter_threshold)\
        .map(lambda usr_basket: usr_basket[1]).cache()

    min_support = support/(basket.count())
    num_partitions = basket.getNumPartitions()

    model = FPGrowth.train(basket, min_support, num_partitions)
    result = model.freqItemsets().collect()

    fi_items_task3 = []
    #with open("output3_results.txt", "w") as w:
        #for fi in result:
            #fi_items_task3.append(fi)
            #w.write(str(fi) + "\n")
    #print(fi_items_task3)

    fi_items_task2 = []
    with open("saved_task2.txt", "r") as f:
        for line in f:
            fi_items_task2.append(line.strip())
    print(fi_items_task2)

    with open(output_fp, "w") as w:
        w.write("Task 2: " + str(len(fi_items_task2)) + "\n")
        w.write("Task 3: " + str(len(result)))




