from pyspark import SparkContext, SparkConf
from collections import defaultdict
from itertools import combinations
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

def euclidian_distance(c1, p2):
    """Calculate the Euclidian Distance b/w centroid 1 and point 2"""
    return math.sqrt(sum([(v1 - v2) ** 2 for (v1, v2) in zip(c1, p2)]))

def mahalanobis_distance(c1, p2, std):
    """Calculate the Mahalonibis Distance b/w centroid 1 and  point 2"""
    return math.sqrt(sum([((v1 - v2) / sd) ** 2 for (v1, v2, sd) in zip(c1, p2, std)]))

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

def generate_std_key(ds_stats):
    """Generate the STD Key for the ds clusters using SUMSQ_N vector and MEAN vector."""
    for c, v in ds_stats.items():
        std_vector = [math.sqrt(sumsq_n - sum_n**2) for (sumsq_n, sum_n) in zip(v["SUMSQ_N"], v["MEAN"])]
        ds_stats[c]["STD"] = std_vector
    return ds_stats

def change_centroids_non_kmeans(new_data, cluster_d, new_cluster_pts):
    """Change the centroids based on the new points that are associated with each cluster."""
    #centroids_ = defaultdict(dict)
    for c, v in cluster_d.items():
        print("centroid #", c)
        centroid_data = []
        for pt_idx in new_cluster_pts[c]:
            pt_data = new_data[pt_idx]
            centroid_data.append(pt_data)

        # No Changes made to centroid:
        if len(centroid_data) == 0:
            continue

        new_mean_vector = [sum(i) for i in zip(*centroid_data)]   #SUM/N

        # print("Old Mean Vector:")
        # print(v["MEAN"])
        new_N = len(centroid_data)
        old_N = v["N"]
        cluster_d[c]["MEAN"] = [(new_sum + (old_mean*old_N)) / (new_N + old_N) for (new_sum, old_mean) in zip(new_mean_vector, v["MEAN"])]

        # print("New Mean Vector:")
        # print(centroids_[c]["MEAN"])
        #
        # print("Old SUMSQ_N Vector:")
        # print(v["SUMSQ_N"])
        new_sumsq_vector = [sum([v ** 2 for v in i]) for i in zip(*centroid_data)] #SUMSQ/N
        cluster_d[c]["SUMSQ_N"] = [(new_sumsq + (old_sumsq * old_N)) / (new_N + old_N) for (new_sumsq, old_sumsq) in zip(new_sumsq_vector, v["SUMSQ_N"])]

        # print("New SUMSQ_N Vector:")
        # print(centroids_[c]["SUMSQ_N"])
        #
        # print("Old N:")
        # print(v["N"])
        cluster_d[c]["N"] = old_N + new_N

        # print("New N:")
        # print(centroids_[c]["N"])

    #print("New Centroids:")
    #print(centroids_)
    return cluster_d

def merge_cs_clusters(cs_clusters):
    clusters_unmerged = set(cs_clusters.keys())
    for cs_pair in combinations(cs_clusters.keys(), 2):
        cs1 = cs_pair[0]
        cs2 = cs_pair[1]
        if cs1 in clusters_unmerged and cs2 in clusters_unmerged:
            dist_ = mahalanobis_distance(cs_clusters[cs1]["MEAN"], cs_clusters[cs2]["MEAN"], cs_clusters[cs1]["STD"])

            dim_data = len(cs_clusters[cs1]["MEAN"])
            if dist_ < alpha*math.sqrt(dim_data):
                new_cluster_info = {}
                new_cluster_info["MEAN"] = change_mean_on_cs_merge(cs_clusters[cs1], cs_clusters[cs2])
                new_cluster_info["SUMSQ_N"] = change_sumsq_on_cs_merge(cs_clusters[cs1], cs_clusters[cs2])
                new_cluster_info["PTS"] = cs_clusters[cs1]["PTS"] + cs_clusters[cs2]["PTS"]
                new_cluster_info["N"] = cs_clusters[cs1]["N"] + cs_clusters[cs2]["N"]

                del cs_clusters[cs1]
                del cs_clusters[cs2]
                clusters_unmerged.discard(cs1)
                clusters_unmerged.discard(cs2)

                cs_clusters[cs1] = new_cluster_info

        else:
            continue



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

    alpha = 3
    ds_stats_pts = defaultdict(dict) #{c#0: {"MEAN": [m1, ... md], {"SUMSQ_N": [s1, ... sd]},
                                        #{"STD": [std1, ... stdn]}, {"PTS": pt1, pt5, p10,...}, {"N": 150}, c#1: ...}
    cs_stats_pts = defaultdict(dict) #{c#0: {"MEAN": [m1, ... md], {"SUMSQ_N": [s1, ... sd]}, {"PTS": pt1, pt5, p10,...}, c#1: ...}
    rs = defaultdict(list)

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
                ds_stats_pts[c]["N"] = len(ds_cluster_pts[c])
                tot_pts_ds += ds_stats_pts[c]["N"]

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
                    rs[data_pt_idx] = remaining_data[data_pt_idx]
                    del cs_cluster_info[c]
                    del cs_cluster_pts[c]

            cs_stats_pts = cs_cluster_info
            tot_pts_cs = 0
            for c in cs_cluster_pts.keys():
                cs_stats_pts[c]["PTS"] = cs_cluster_pts[c]
                cs_stats_pts[c]["N"] = len(cs_cluster_pts[c])
                tot_pts_cs += cs_stats_pts[c]["N"]


            #Generate STD Key:
            ds_stats_pts = generate_std_key(ds_stats_pts)
            cs_stats_pts = generate_std_key(cs_stats_pts)

        else:
            new_data = index_data_rdd.collectAsMap()

            new_ds_stats_pts = defaultdict(list)
            new_cs_stats_pts = defaultdict(list)

            cnt_ds = 0
            for data_idx in new_data.keys():
                dist_d = {}
                i_data = new_data[data_idx]
                for c, v in ds_stats_pts.items():
                    c_mean = v["MEAN"]
                    c_std = v["STD"]

                    dist = mahalanobis_distance(c_mean, i_data, c_std)
                    dist_d[c] = dist

                dim_data = len(i_data)
                dist_d = sorted(dist_d.items(), key=lambda item: item[1])
                min_dist = dist_d[0][1]

                if min_dist < alpha*math.sqrt(dim_data):
                    new_ds_stats_pts[dist_d[0][0]].append(data_idx)
                    cnt_ds += 1
                    continue
                else:
                    #Compare to CS clusters
                    dist_cs = {}
                    for c, v in cs_stats_pts.items():
                        c_mean = v["MEAN"]
                        c_std = v["STD"]

                        dist = mahalanobis_distance(c_mean, i_data, c_std)
                        dist_cs[c] = dist

                    dist_cs = sorted(dist_cs.items(), key=lambda item: item[1])
                    min_dist = dist_cs[0][1]

                    if min_dist < alpha*math.sqrt(dim_data):
                        new_cs_stats_pts[dist_d[0][0]].append(data_idx)
                        continue
                    else:
                        rs[data_idx] = new_data[data_idx]
            # print("New Data:")
            # print(len(new_data.keys()))
            # print("New DS:")
            # print(cnt_ds)
            # print("New CS:")
            # print(new_cs_stats_pts)
            # print("New RS:")
            # print(new_rs)

            #Merge new DS points with old DS points:
            for c, v in ds_stats_pts.items():
                #print("Before len:", len(ds_stats_pts[c]["PTS"]))
                #print("Addition:", len(new_ds_stats_pts[c]))
                ds_stats_pts[c]["PTS"] = v["PTS"] + new_ds_stats_pts[c]
                #print("After len:", len(ds_stats_pts[c]["PTS"]))
            ds_stats_pts = change_centroids_non_kmeans(new_data, ds_stats_pts, new_ds_stats_pts)

            for c, v in cs_stats_pts.items():
                #print("Before len:", len(cs_stats_pts[c]["PTS"]))
                #print("Addition:", len(new_ds_stats_pts[c]))
                cs_stats_pts[c]["PTS"] = v["PTS"] + new_cs_stats_pts[c]
                #print("After len:", len(ds_stats_pts[c]["PTS"]))
            cs_stats_pts = change_centroids_non_kmeans(new_data, cs_stats_pts, new_cs_stats_pts)

            #Run Clustering Alg. on RS:
            if len(rs.keys()) != 0:
                rs_cluster_info, rs_cluster_pts = kmeans(rs, n_clusters * 3, 5)
            #Add RS Clusters to current cluster list if more than 1:
                new_rs = defaultdict(list)
                for c, pts_lst in rs_cluster_pts.items():
                    if len(pts_lst) == 0:
                        del rs_cluster_info[c]
                        del rs_cluster_pts[c]
                    elif len(pts_lst) == 1:
                        data_pt_idx = pts_lst[0]
                        new_rs[data_pt_idx] = rs[data_pt_idx]
                        del rs_cluster_info[c]
                        del rs_cluster_pts[c]
                    else:
                        new_c_key = len(cs_stats_pts.keys())
                        cs_stats_pts[new_c_key] = rs_cluster_info[c]
                        cs_stats_pts[new_c_key]["PTS"] = rs_cluster_pts[c]
                        cs_stats_pts[new_c_key]["N"] = len(rs_cluster_pts[c])

                generate_std_key(cs_stats_pts)
                rs.clear()
                rs = new_rs

            #Merge CS Clusters:
            cs_stats_pts = merge_cs_clusters(cs_stats_pts)
            cs_stats_pts = generate_std_key(cs_stats_pts)


            break











        inter_results = {}
        inter_results[i] = [i+1, len(ds_stats_pts), tot_pts_ds, len(cs_stats_pts), tot_pts_cs, len(rs)]

        with open(output_fp_inter, "w+") as w:
            writer = csv.writer(w)
            if i == 0:
                writer.writerow(headers)
            for k, v in inter_results.items():
                writer.writerow(v)






