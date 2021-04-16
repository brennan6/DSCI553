from pyspark import SparkContext, SparkConf
from collections import defaultdict
import time
import json
import math
import sys
import csv
import random
import os

def kmeans(data, n, max_iter):
    """From the given data, creates n clusters, where all points in data are assigned a cluser \
       using Euclidian Distance. Returns info on the clusters formed."""

    iter_cnt = 0
    centroids_ = initialize_centroids(data, n) #{c#0: {"MEAN": [m1, ... md], {"SUMSQ_N": [s1, ... sd]}, c#1: ...}
    cluster_pts_d = defaultdict(list)  #{c#0: [p1, p5, ...], c#1: [...]}
    has_stabilized = False
    prev_cluster_pts = None

    while True:
        iter_cnt += 1
        for data_idx in data.keys():
            dist_d = {}
            for c, v in centroids_.items():
                c_data = v["MEAN"]
                i_data = data[data_idx]

                dist = euclidian_distance(c_data, i_data)
                dist_d[c] = dist

            dist_d = sorted(dist_d.items(), key=lambda item: item[1])
            cluster_pts_d[dist_d[0][0]].append(data_idx)

        #Change Centroids to deal with new points associated:
        # for idx, v in cluster_pts_d.items():
        #     print(idx)
        #     print(v)
        centroids_ = change_centroids(data, cluster_pts_d)

        #Check if too many iterations:
        if iter_cnt == max_iter:
            print("Max Iters.")
            break

        #Check if clusters aren't changing
        has_stabilized = check_stability(prev_cluster_pts, cluster_pts_d)
        if has_stabilized:
            print("Stabilized.")
            break

        prev_cluster_pts = cluster_pts_d
        cluster_pts_d.clear()


    return centroids_, cluster_pts_d

def change_centroids(data, cluster_d):
    """Change the centroids based on the new points that are associated with each cluster."""
    centroids_ = defaultdict(dict)
    for c, pts_list in cluster_d.items():
        print("centroid #", c)
        centroid_data = []
        for pt_idx in pts_list:
            pt_data = data[pt_idx]
            centroid_data.append(pt_data)

        # for val in centroid_data:
        #     print(val)

        centroids_[c]["MEAN"] = [sum(i) / len(i) for i in zip(*centroid_data)]   #SUM/N
        centroids_[c]["SUMSQ_N"] = [sum([v ** 2 for v in i]) / len(i) for i in zip(*centroid_data)] #SUMSQ/N

    print("New Centroids:")
    print(centroids_)
    return centroids_

def check_stability(old_cluster, new_cluster):
    """Compare the old_cluster grouping with the new_cluster grouping."""
    if old_cluster == None:
        return False
    else:
        for idx in old_cluster.keys():
            old_pts = set(old_cluster[idx])
            new_pts = set(new_cluster[idx])
            if len(old_pts.difference(new_pts)) == 0:
                continue
            else:
                return False
        return True

def euclidian_distance(p1, p2):
    """Calculate the Euclidian Distance b/w points 1 and 2"""
    return math.sqrt(sum([(v1 - v2) ** 2 for (v1, v2) in zip(p1, p2)]))

def initialize_centroids(data, n):
    """Initialize centroids with form: {centroid #: row #}"""
    random_pts = random.sample(data.keys(), n)
    init_centroids = defaultdict(dict)
    for i, key in enumerate(random_pts):
        init_centroids[i]["MEAN"] = data[key]
        init_centroids[i]["SUMSQ_N"] = 0

    print("Intital Centroid:")
    print(init_centroids)
    return init_centroids

if __name__ == "__main__":
    start = time.time()
    # input_fp = sys.argv[1]
    # n_clusters = int(sys.argv[2])
    # output_fp_cluster = sys.argv[3]
    # output_fp_inter = sys.argv[4]

    input_fp = "./data/test1"
    n_clusters = 15
    output_fp_cluster = "./output/cluster1.json"
    output_fp_inter = "./output/intermediate.csv"

    conf = SparkConf()
    conf.set("spark.executor.memory", "8g")
    conf.set("spark.driver.memory", "8g")
    sc = SparkContext(conf=conf)

    alpha = 2
    ds_stats_pts = defaultdict(dict)
    cs_stats_pts = defaultdict(dict)
    rs = set()

    headers = ["round_id", "nof_cluster_discard", "nof_point_discard",
               "nof_cluster_compression", "nof_point_compression", "nof_point_retained"]

    for i, file in enumerate(os.listdir(input_fp)):
        curr_fp = input_fp + "/" + file

        data_rdd = sc.textFile(curr_fp)
        index_data_rdd = data_rdd \
                    .map(lambda x: x.split(",")) \
                    .map(lambda i_d: (int(i_d[0]), list(map(float, i_d[1:]))))

        #Need to initialize with K-Means:
        if i == 0:
            data_points_cnt = index_data_rdd.count()
            print("Total Data Points:", data_points_cnt)

            init_cutoff = math.floor(data_points_cnt/5)
            print("Sample Data Points:", init_cutoff)
            init_subset = index_data_rdd \
                        .filter(lambda i_d: i_d[0] <= init_cutoff) \
                        .collectAsMap()

            #Create the initial K-Means clusters using the subset of data:
            ds_cluster_info, ds_cluster_pts = kmeans(init_subset, n_clusters, 5)

            ds_stats_pts = ds_cluster_info
            tot_pts_ds = 0
            for c in ds_cluster_pts.keys():
                ds_stats_pts[c]["PTS"] = ds_cluster_pts[c]
                tot_pts_ds += len(ds_cluster_pts[c])

            remaining_data = index_data_rdd \
                        .filter(lambda i_d: i_d[0] > init_cutoff) \
                        .collectAsMap()

            #Create CS Sets using K*3 clusters:
            cs_cluster_info, cs_cluster_pts = kmeans(remaining_data, n_clusters*3, 5)

            for c, pts_lst in cs_cluster_pts.items():
                if len(pts_lst) == 0:
                    del cs_cluster_info[c]
                    del cs_cluster_pts[c]
                elif len(pts_lst) == 1:
                    data_pt_idx = pts_lst[0]
                    rs.add(data_pt_idx)
                    del cs_cluster_info[c]
                    del cs_cluster_pts[c]

            cs_stats_pts = cs_cluster_info
            tot_pts_cs = 0
            for c in cs_cluster_pts.keys():
                cs_stats_pts[c]["PTS"] = cs_cluster_pts[c]
                tot_pts_cs += len(cs_cluster_pts[c])

        else:
            new_data = index_data_rdd.collectAsMap()
            

        inter_results = {}
        inter_results[i] = [i+1, len(ds_stats_pts), tot_pts_ds, len(cs_stats_pts), tot_pts_cs, len(rs)]

        with open(output_fp_inter, "w+") as w:
            writer = csv.writer(w)
            if i == 0:
                writer.writerow(headers)
            for k, v in inter_results.items():
                writer.writerow(v)






