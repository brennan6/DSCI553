from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from collections import defaultdict
import time
import json
import math
import sys
import binascii
import datetime
import random

NUM_HASH_FUNCTIONS = 10
def kmeans(data, n, max_iter):
    """From the given data, creates n clusters, where all points in data are assigned a cluser \
       using Euclidian Distance. Returns info on the clusters formed."""

    iter_cnt = 0
    centroids_ = initialize_centroids(data, n) #[Estimation lst]
    cluster_pts_d = defaultdict(list)  #{c#0: [p1, p5, ...], c#1: [...]}
    outlier_lst = []

    while True:
        iter_cnt += 1
        for est_ in data:
            print("value:", est_)
            if est_ in outlier_lst:
                print("Outlier", est_)
                continue
            dist_d = {}
            for c, v in centroids_.items():
                c_data = v["MEAN"]
                i_data = est_

                dist = euclidian_distance(c_data, i_data)
                dist_d[c] = dist

            dist_d = sorted(dist_d.items(), key=lambda item: item[1])
            if dist_d[0][1] > 200:
                outlier_lst.append(est_)
                continue
            cluster_pts_d[dist_d[0][0]].append(est_)

        # for c_, v_ in cluster_pts_d.items():
        #     print(str(c_) + ": " + str(len(v_)))

        # for i_, pts in cluster_pts_d.items():
        #     print(i_)
        #     print(pts)

        centroids_ = change_centroids(data, cluster_pts_d)

        # for c_, v_ in centroids_.items():
        #     print(str(c_) + ": ")
        #     print(v_["MEAN"])

        #Check if too many iterations:
        if iter_cnt == max_iter:
            print("Max Iters.")
            break

        cluster_pts_d.clear()

    return centroids_

def euclidian_distance(c1, p2):
    """Calculate the Euclidian Distance b/w centroid 1 and point 2"""
    return math.sqrt((c1 - p2) ** 2)

def change_centroids(data, cluster_d):
    """Change the centroids based on the new points that are associated with each cluster."""
    centroids_ = defaultdict(dict)
    for c_, pts_list in cluster_d.items():
        c_dict = {}
        c_dict["MEAN"] = sum(pts_list) / len(pts_list)  #SUM/N

        centroids_[c_] = c_dict
    return centroids_

def initialize_centroids(data, n):
    """Initialize centroids with form: {centroid #: row #}"""
    random.seed(42)
    init_centroids = defaultdict(dict)

    if len(data) < n:  #Deals with the case when RS only has a few data points
        random_pts = random.sample(data, len(data))
    else:
        random_pts = random.sample(data, n)
    for i_, val in enumerate(random_pts):
        init_centroids[i_]["MEAN"] = val

    return init_centroids

def flajolet_martin(rdd):
    random.seed(42)
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

    actual_distinct = list(set(rdd.collect()))

    m = 2000
    p = 22777
    estimations = []
    print("Restart.")
    for hash_num in range(NUM_HASH_FUNCTIONS):
        R = 0
        a = random.randint(1, m)
        b = random.randint(1, m)
        for city_ in actual_distinct:
            str_city = int(binascii.hexlify(city_.encode('utf8')), 16)
            hashed_city = '{0:012b}'.format(int(((a * str_city + b) % p)))

            if int(hashed_city) == 0:
                continue
            trailing_zeros = len(hashed_city) - len(hashed_city.rstrip('0'))
            R = max(R, trailing_zeros)
        estimations.append(2**R)

    n = 2
    clusters = kmeans(estimations, n, 10) #num_clusters, #max_iters
    mean_clusters = []
    for c_, v_ in clusters.items():
        mean_clusters.append(v_["MEAN"])

    mean_clusters_sorted = sorted(mean_clusters)
    if len(mean_clusters) == n:
        median_of_means_of_clusters = math.floor((mean_clusters_sorted[len(mean_clusters_sorted)-1] + mean_clusters_sorted[0])/2)
    elif len(mean_clusters) == 1:
        median_of_means_of_clusters = math.floor(mean_clusters_sorted[0])
    else:
        median_of_means_of_clusters = math.floor((mean_clusters_sorted[len(mean_clusters_sorted)-1] + mean_clusters_sorted[0])/2)
    with open(output_fp, "a") as w:
        w.write(timestamp + "," + str(len(actual_distinct)) + "," + str(median_of_means_of_clusters))
        w.write("\n")

if __name__ == "__main__":
    start = time.time()
    port_num = int(sys.argv[1])
    output_fp = sys.argv[2]

    # port_num = 9999
    # output_fp = "./output/output2.csv"

    conf = SparkConf().set('spark.driver.host', '127.0.0.1')
    # conf = SparkConf()
    conf.set("spark.executor.memory", "2g")
    conf.set("spark.driver.memory", "2g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    ssc = StreamingContext(sc, 5)
    stream_data = ssc.socketTextStream("localhost", port_num)

    with open(output_fp, "w") as w:
        w.write("Time,Ground Truth,Estimation")
        w.write("\n")

    stream_data_rdd = stream_data.window(30, 10) \
                    .map(lambda x: json.loads(x)) \
                    .map(lambda x: x["city"]) \
                    .filter(lambda c: c != "") \
                    .foreachRDD(lambda rdd: flajolet_martin(rdd))

    ssc.start()
    ssc.awaitTermination()
    end = time.time()
    print("Duration:", round((end - start), 2))