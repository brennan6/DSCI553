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

os.environ["PYSPARK_PYTHON"] = '/usr/local/bin/python3.6'
os.environ["PYSPARK_DRIVER_PYTHON"] = '/usr/local/bin/python3.6'

def kmeans(data, n, max_iter):
    """From the given data, creates n clusters, where all points in data are assigned a cluser \
       using Euclidian Distance. Returns info on the clusters formed."""

    iter_cnt = 0
    centroids_ = initialize_centroids(data, n) #{c#0: {"MEAN": [m1, ... md], {"SUMSQ_N": [s1, ... sd]}, c#1: ...}
    cluster_pts_d = defaultdict(list)  #{c#0: [p1, p5, ...], c#1: [...]}
    prev_cluster_pts = None
    outlier_lst = defaultdict(list)

    while True:
        iter_cnt += 1
        for data_idx in data.keys():
            if data_idx in outlier_lst:
                continue
            dist_d = {}
            for c, v in centroids_.items():
                c_data = v["MEAN"]
                i_data = data[data_idx]

                dist = euclidian_distance(c_data, i_data)
                dist_d[c] = dist

            dist_d = sorted(dist_d.items(), key=lambda item: item[1])
            if dist_d[0][1] > 3000:
                outlier_lst[data_idx] = data[data_idx]
                continue
            cluster_pts_d[dist_d[0][0]].append(data_idx)

        # for c_, v_ in cluster_pts_d.items():
        #     print(str(c_) + ": " + str(len(v_)))

        centroids_ = change_centroids(data, cluster_pts_d)

        # for c_, v_ in centroids_.items():
        #     print(str(c_) + ": ")
        #     print(v_["MEAN"])

        #Check if too many iterations:
        if iter_cnt == max_iter:
            print("Max Iters.")
            break

        #Check if clusters aren't changing
        has_stabilized = check_stability(prev_cluster_pts, cluster_pts_d)
        if has_stabilized:
            print("Stabilized.")
            break

        prev_cluster_pts = defaultdict(list)
        for c_, v_ in cluster_pts_d.items():
            prev_cluster_pts[c_] = v_

        cluster_pts_d.clear()

    return centroids_, cluster_pts_d, outlier_lst

def change_centroids(data, cluster_d):
    """Change the centroids based on the new points that are associated with each cluster."""
    centroids_ = defaultdict(dict)
    for c_, pts_list in cluster_d.items():
        centroid_data = []
        c_dict = {}
        for pt_idx in pts_list:
            pt_data = data[pt_idx]
            centroid_data.append(pt_data)

        c_dict["MEAN"] = [sum(i_) / len(i_) for i_ in zip(*centroid_data)]   #SUM/N
        c_dict["SUMSQ_N"] = [sum([val ** 2 for val in i_]) / len(i_) for i_ in zip(*centroid_data)] #SUMSQ/N

        centroids_[c_] = c_dict
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
    # if len(data.keys()) < n:  #Deals with the case when RS only has a few data points
    #     random_pts = random.sample(data.keys(), len(data.keys()))
    # else:
    #     random_pts = random.sample(data.keys(), n)
    # init_centroids = defaultdict(dict)
    # num_iter = 0
    # pair_d = {}
    random.seed(666)

    init_centroids = defaultdict(dict)
    # if len(data.keys()) == 0:
    #     return init_centroids
    # random_pts = random.sample(data.keys(), 1)
    # if len(data.keys()) < n:  #Deals with the case when RS only has a few data points
    #     num_clusters = len(data.keys())
    # else:
    #     num_clusters = n
    # while True:
    #     #print("LEN RANDOM PTS:", len(random_pts))
    #     if len(random_pts) == num_clusters:
    #         break
    #     dist_euc = {}
    #     for pt_ in random_pts:
    #         for k_ in data.keys():
    #             dist_ = euclidian_distance(data[pt_], data[k_])
    #             if k_ in dist_euc:
    #                 dist_euc[k_] += dist_
    #             else:
    #                 dist_euc[k_] = dist_
    #
    #     dist_euc = sorted(dist_euc.items(), key=lambda item: -item[1])
    #     pt_to_pick = dist_euc[math.floor(.3*len(data.keys()))][0]
    #     random_pts.append(pt_to_pick)
    #
    #     pair_d = {}
    #     for combo in combinations(random_pts, 2):
    #         key0 = combo[0]
    #         key1 = combo[1]
    #         euc_dist = euclidian_distance(data[key0], data[key1])
    #         pair_d[combo] = euc_dist

    # while True:
    #     for combo in combinations(random_pts, 2):
    #         key0 = combo[0]
    #         key1 = combo[1]
    #         euc_dist = euclidian_distance(data[key0], data[key1])
    #         pair_d[combo] = euc_dist
    #
    #     pair_d = dict(sorted(pair_d.items(), key=lambda item: item[1]))
    #     pair_d = dict(filter(lambda elem: elem[1] < 25, pair_d.items()))
    #     if len(pair_d.keys()) == 0:
    #         print("Num Iters Init.: ", num_iter)
    #         print("Stabilized: Initialize")
    #         break
    #
    #     to_remove = set([item for t in pair_d.keys() for item in t])
    #     for k_ in to_remove:
    #         random_pts.remove(k_)
    #         random_pts += random.sample(data.keys(), 1)
    #
    #     if num_iter == 100:
    #         print("Max Iters: Initialize")
    #         break
    #
    #     num_iter += 1
    #     pair_d.clear()
    #

    if len(data.keys()) < n:  #Deals with the case when RS only has a few data points
        random_pts = random.sample(data.keys(), len(data.keys()))
    else:
        random_pts = random.sample(data.keys(), n)
    for i_, key in enumerate(random_pts):
        init_centroids[i_]["MEAN"] = data[key]
        init_centroids[i_]["SUMSQ_N"] = 0
    print("Selected Starting Pts.")
    return init_centroids

def generate_std_key(ds_stats):
    """Generate the STD Key for the ds clusters using SUMSQ_N vector and MEAN vector."""
    for c_, v_ in ds_stats.items():
        std_vector = [math.sqrt(sumsq_n - sum_n**2) for (sumsq_n, sum_n) in zip(v_["SUMSQ_N"], v_["MEAN"])]
        std_vector = [i_ if i_ != 0 else .001 for i_ in std_vector]
        ds_stats[c_]["STD"] = std_vector
    return ds_stats

def change_centroids_non_kmeans(new_data, cluster_d, new_cluster_pts):
    """Change the centroids based on the new points that are associated with each cluster."""
    for c_, v_ in cluster_d.items():
        centroid_data = []
        for pt_idx in new_cluster_pts[c_]:
            pt_data = new_data[pt_idx]
            centroid_data.append(pt_data)

        # No Changes made to centroid:
        if len(centroid_data) == 0:
            continue

        new_mean_vector = [sum(i_) for i_ in zip(*centroid_data)]   #SUM
        new_N = len(centroid_data)
        old_N = v_["N"]
        cluster_d[c_]["MEAN"] = [(new_sum + (old_mean*old_N)) / (new_N + old_N) for (new_sum, old_mean) in zip(new_mean_vector, v_["MEAN"])]

        new_sumsq_vector = [sum([val ** 2 for val in i_]) for i_ in zip(*centroid_data)] #SUMSQ/N
        cluster_d[c_]["SUMSQ_N"] = [(new_sumsq + (old_sumsq * old_N)) / (new_N + old_N) for (new_sumsq, old_sumsq) in zip(new_sumsq_vector, v_["SUMSQ_N"])]

        cluster_d[c_]["N"] = old_N + new_N

    return cluster_d

def change_centroids_midrun(new_data, cluster_d, pt_idx, c_):
    """Change the centroids based on the new points that are associated with each cluster."""
    pt_data = new_data[pt_idx]
    old_N = cluster_d[c_]["N"]

    cluster_d[c_]["MEAN"] = [(new_sum + (old_mean*old_N)) / (1 + old_N) for (new_sum, old_mean) in zip(pt_data, cluster_d[c_]["MEAN"])]

    #new_sumsq_vector = [sum([val ** 2 for val in i_]) for i_ in zip(*centroid_data)] #SUMSQ/N
    cluster_d[c_]["SUMSQ_N"] = [(new_sum**2 + (old_sumsq * old_N)) / (1 + old_N) for (new_sum, old_sumsq) in zip(pt_data, cluster_d[c_]["SUMSQ_N"])]

    cluster_d[c_]["N"] = old_N + 1

    return cluster_d

def change_mean_on_cs_merge(cs1_values, cs2_values):
    cs1_mean_vector = cs1_values["MEAN"]
    cs2_mean_vector = cs2_values["MEAN"]

    cs1_n = cs1_values["N"]
    cs2_n = cs2_values["N"]

    new_mean_vector = [(cs1_n*cs1_mean + cs2_n*cs2_mean) / (cs1_n + cs2_n) for (cs1_mean, cs2_mean) in zip(cs1_mean_vector, cs2_mean_vector)]
    return new_mean_vector

def change_sumsq_on_cs_merge(cs1_values, cs2_values):
    cs1_sumsq_vector = cs1_values["SUMSQ_N"]
    cs2_sumsq_vector = cs2_values["SUMSQ_N"]

    cs1_n = cs1_values["N"]
    cs2_n = cs2_values["N"]

    new_sumsq_vector = [(cs1_n*cs1_sumsq + cs2_n*cs2_sumsq) / (cs1_n + cs2_n) for (cs1_sumsq, cs2_sumsq) in zip(cs1_sumsq_vector, cs2_sumsq_vector)]
    return new_sumsq_vector

def merge_cs_clusters(cs_clusters, dim_):
    clusters_unmerged = set(cs_clusters.keys())
    dist_total = {}
    for cs_pair in combinations(cs_clusters.keys(), 2):
        cs1 = cs_pair[0]
        cs2 = cs_pair[1]
        dist_ = mahalanobis_distance(cs_clusters[cs1]["MEAN"], cs_clusters[cs2]["MEAN"], cs_clusters[cs1]["STD"])
        dist_total[(cs1, cs2)] = dist_

    dist_total_lst = sorted(dist_total.items(), key=lambda item: item[1])
    for pair_ in dist_total_lst:
        cs1 = pair_[0][0]
        cs2 = pair_[0][1]
        if cs1 in clusters_unmerged and cs2 in clusters_unmerged:
            dist_ = pair_[1]
            new_cluster_info = {}
            if dist_ < alpha * math.sqrt(dim_):
                new_cluster_info["MEAN"] = change_mean_on_cs_merge(cs_clusters[cs1], cs_clusters[cs2])
                new_cluster_info["SUMSQ_N"] = change_sumsq_on_cs_merge(cs_clusters[cs1], cs_clusters[cs2])
                new_cluster_info["PTS"] = cs_clusters[cs1]["PTS"] + cs_clusters[cs2]["PTS"]
                new_cluster_info["N"] = cs_clusters[cs1]["N"] + cs_clusters[cs2]["N"]

                del cs_clusters[cs1]
                del cs_clusters[cs2]
                clusters_unmerged.discard(cs1)
                clusters_unmerged.discard(cs2)
                print("Merge Occurs")

                cs_clusters[cs1] = new_cluster_info
            else:
                continue


    # clusters_unmerged = set(cs_clusters.keys())
    # for cs_pair in combinations(cs_clusters.keys(), 2):
    #     cs1 = cs_pair[0]
    #     cs2 = cs_pair[1]
    #     if cs1 in clusters_unmerged and cs2 in clusters_unmerged:
    #         dist_ = mahalanobis_distance(cs_clusters[cs1]["MEAN"], cs_clusters[cs2]["MEAN"], cs_clusters[cs1]["STD"])
    #
    #         new_cluster_info = {}
    #         if dist_ < (alpha*math.sqrt(dim_))/alpha:
    #             new_cluster_info["MEAN"] = change_mean_on_cs_merge(cs_clusters[cs1], cs_clusters[cs2])
    #             new_cluster_info["SUMSQ_N"] = change_sumsq_on_cs_merge(cs_clusters[cs1], cs_clusters[cs2])
    #             new_cluster_info["PTS"] = cs_clusters[cs1]["PTS"] + cs_clusters[cs2]["PTS"]
    #             new_cluster_info["N"] = cs_clusters[cs1]["N"] + cs_clusters[cs2]["N"]
    #
    #             del cs_clusters[cs1]
    #             del cs_clusters[cs2]
    #             clusters_unmerged.discard(cs1)
    #             clusters_unmerged.discard(cs2)
    #             print("Merge Occurs")
    #
    #             cs_clusters[cs1] = new_cluster_info
    #         else:
    #             continue
    #
    #     else:
    #         continue
    return cs_clusters

def merge_cs_to_ds_clusters(ds_clusters, cs_clusters, dim_):
    cs_remove = []
    for c_cs, v_cs in cs_clusters.items():
        dist_d = {}
        for c_ds, v_ds in ds_clusters.items():
            dist_ = mahalanobis_distance(v_ds["MEAN"], v_cs["MEAN"], v_ds["STD"])
            dist_d[c_ds] = dist_

        dist_d = sorted(dist_d.items(), key=lambda item: item[1])
        dist_winner = dist_d[0][1]
        ds_winner = dist_d[0][0]

        #Just add pts to new ds:
        if dist_winner < alpha*math.sqrt(dim_):
            ds_clusters[ds_winner]["PTS"] += v_cs["PTS"]
            cs_remove.append(c_cs)
            ds_clusters[ds_winner]["MEAN"] = change_mean_on_cs_merge(ds_clusters[ds_winner], cs_clusters[c_cs])
            ds_clusters[ds_winner]["SUMSQ_N"] = change_sumsq_on_cs_merge(ds_clusters[ds_winner], cs_clusters[c_cs])
            ds_clusters = generate_std_key(ds_clusters)
        else:
            continue

        # #SHould output all values:
        # ds_clusters[ds_winner]["PTS"] += v_cs["PTS"]
        # cs_remove.append(c_cs)
    return ds_clusters, cs_clusters, cs_remove

def merge_rs_to_ds_clusters(ds_clusters, rs, dim_):
    rs_copy = rs
    rs_remove = []
    for k_rs in rs_copy.keys():
        dist_d = {}
        for c_ds, v_ds in ds_clusters.items():
            dist_ = mahalanobis_distance(v_ds["MEAN"], rs[k_rs], v_ds["STD"])
            dist_d[c_ds] = dist_

        dist_d = sorted(dist_d.items(), key=lambda item: item[1])
        dist_winner = dist_d[0][1]
        ds_winner = dist_d[0][0]

        #Just add new pt to new ds:
        if dist_winner < alpha*math.sqrt(dim_):
            ds_clusters[ds_winner]["PTS"] += [k_rs]
            rs_remove.append(k_rs)
        else:
            continue
        # #Should output all values:
        # ds_clusters[ds_winner]["PTS"] += [k_rs]
        # rs_remove.append(k_rs)
    return ds_clusters, rs, rs_remove

def print_total_breakdown(ds_stats_pts):
    for c_, v_ in ds_stats_pts.items():
        print(str(c_) + ": " + str(len(v_["PTS"])))

if __name__ == "__main__":
    start = time.time()
    # input_fp = sys.argv[1]
    # n_clusters = int(sys.argv[2])
    # output_fp_cluster = sys.argv[3]
    # output_fp_inter = sys.argv[4]

    input_fp = "./data/test1"
    n_clusters = 10
    output_fp_cluster = "./output/cluster1.json"
    output_fp_inter = "./output/intermediate.csv"

    conf = SparkConf()
    conf.set("spark.executor.memory", "2g")
    conf.set("spark.driver.memory", "2g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    alpha = 3
    ds_stats_pts = defaultdict(dict) #{c#0: {"MEAN": [m1, ... md], {"SUMSQ_N": [s1, ... sd]},
                                        #{"STD": [std1, ... stdn]}, {"PTS": pt1, pt5, p10,...}, {"N": 150}, c#1: ...}
    cs_stats_pts = defaultdict(dict) #{c#0: {"MEAN": [m1, ... md], {"SUMSQ_N": [s1, ... sd]}, {"PTS": pt1, pt5, p10,...}, c#1: ...}
    rs = defaultdict(list)

    headers = ["round_id", "nof_cluster_discard", "nof_point_discard",
               "nof_cluster_compression", "nof_point_compression", "nof_point_retained"]
    inter_results = {}
    inter_results["headers"] = headers
    total_data = 0
    for i, file in enumerate(sorted(os.listdir(input_fp))):
        print("File #:", i)
        curr_fp = input_fp + "/" + file

        data_rdd = sc.textFile(curr_fp)
        index_data_rdd = data_rdd \
                    .map(lambda x: x.split(",")) \
                    .map(lambda i_d: (int(i_d[0]), list(map(float, i_d[1:]))))

        #Need to initialize with K-Means:
        if i == 0:
            data_points_cnt = index_data_rdd.count()
            total_data += data_points_cnt
            print("Round " + str(i) + ": " + str(total_data))
            init_cutoff = math.ceil(data_points_cnt * 0.75)
            # if data_points_cnt > 100000:
            #     init_cutoff = math.floor(data_points_cnt*.5)
            # else:
            #     init_cutoff = math.floor(data_points_cnt*.8)
            #print("Total Data Points:", data_points_cnt)

            #init_cutoff = math.floor(data_points_cnt/5)
            print("Sample Data Points:", init_cutoff)
            init_subset = index_data_rdd \
                        .filter(lambda i_d: i_d[0] < init_cutoff) \
                        .collectAsMap()

            dim_data = len(init_subset[0])

            #Create the initial K-Means clusters using the subset of data:
            ds_cluster_info, ds_cluster_pts, poss_outliers = kmeans(init_subset, n_clusters, 30)

            ds_stats_pts = ds_cluster_info
            tot_pts_ds = 0
            for c in ds_cluster_pts.keys():
                ds_stats_pts[c]["PTS"] = ds_cluster_pts[c]
                ds_stats_pts[c]["N"] = len(ds_cluster_pts[c])
                tot_pts_ds += ds_stats_pts[c]["N"]

            #print_total_breakdown(ds_stats_pts)

            remaining_data = index_data_rdd \
                        .filter(lambda i_d: i_d[0] >= init_cutoff) \
                        .collectAsMap()

            for idx_ in poss_outliers.keys():
                remaining_data[idx_] = poss_outliers[idx_]

            print("LENGTH DS_STATS:", len(ds_stats_pts))
            #Try to place the data into DS instead of CS first:
            new_ds_stats_pts = defaultdict(list)
            remove_from_data = []
            ds_stats_pts = generate_std_key(ds_stats_pts)
            for data_idx in remaining_data.keys():
                dist_d = {}
                i_data = remaining_data[data_idx]
                for c, v in ds_stats_pts.items():
                    c_mean = v["MEAN"]
                    c_std = v["STD"]

                    dist = mahalanobis_distance(c_mean, i_data, c_std)
                    dist_d[c] = dist

                dist_d = sorted(dist_d.items(), key=lambda item: item[1])
                min_dist = dist_d[0][1]

                if min_dist < (alpha*math.sqrt(dim_data)/(alpha*2)):
                    new_ds_stats_pts[dist_d[0][0]].append(data_idx)
                    remove_from_data.append(data_idx)
                    continue

            for c, v in new_ds_stats_pts.items():
                ds_stats_pts[c]["PTS"] += v
                tot_pts_ds += len(v)

            print("New DS:", len(remove_from_data))
            ds_stats_pts = change_centroids_non_kmeans(remaining_data, ds_stats_pts, new_ds_stats_pts)
            remaining_data = {key: remaining_data[key] for key in remaining_data if key not in remove_from_data}
            print("Remaining Data:", len(remaining_data))
            print("LENGTH DS_STATS:", len(ds_stats_pts))

            #Create CS Sets using K*3 clusters:
            cs_cluster_info, cs_cluster_pts, poss_outliers = kmeans(remaining_data, n_clusters*3, 10)
            for idx_ in poss_outliers.keys():
                rs[idx_] = poss_outliers[idx_]

            cs_to_remove = []
            for c, pts_lst in cs_cluster_pts.items():
                if len(pts_lst) == 0:
                    cs_to_remove.append(c)
                elif len(pts_lst) == 1:
                    data_pt_idx = pts_lst[0]
                    rs[data_pt_idx] = remaining_data[data_pt_idx]
                    cs_to_remove.append(c)
            cs_cluster_info = {key: cs_cluster_info[key] for key in cs_cluster_info if key not in cs_to_remove}
            cs_cluster_pts = {key: cs_cluster_pts[key] for key in cs_cluster_info if key not in cs_to_remove}

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
            #print("LENGTH DS_STATS:", len(ds_stats_pts))
            new_data = index_data_rdd.collectAsMap()
            total_data += len(new_data.keys())
            print("Round " + str(i) + ": " + str(total_data))
            print("New to Round:", len(new_data.keys()))

            new_ds_stats_pts = defaultdict(list)
            new_cs_stats_pts = defaultdict(list)

            cnt_ = 0
            rs_cnt = 0
            rs_len_before = len(rs.keys())
            for data_idx in new_data.keys():
                dist_ds = {}
                i_data = new_data[data_idx]
                for c, v in ds_stats_pts.items():
                    c_mean = v["MEAN"]
                    c_std = v["STD"]

                    dist = mahalanobis_distance(c_mean, i_data, c_std)
                    dist_ds[c] = dist

                dist_ds = sorted(dist_ds.items(), key=lambda item: item[1])
                min_dist = dist_ds[0][1]

                one_pt = {}
                if min_dist < alpha*math.sqrt(dim_data):
                    new_ds_stats_pts[dist_ds[0][0]].append(data_idx)
                    #ds_stats_pts = change_centroids_midrun(new_data, ds_stats_pts, data_idx, dist_ds[0][0])
                    #ds_stats_pts = generate_std_key(ds_stats_pts)
                    cnt_ += 1
                    continue
                else:
                    #Compare to CS clusters
                    if len(cs_stats_pts) != 0:
                        dist_cs = {}
                        for c, v in cs_stats_pts.items():
                            c_mean = v["MEAN"]
                            c_std = v["STD"]

                            dist = mahalanobis_distance(c_mean, i_data, c_std)
                            dist_cs[c] = dist

                        dist_cs = sorted(dist_cs.items(), key=lambda item: item[1])
                        min_dist = dist_cs[0][1]

                        if min_dist < alpha*math.sqrt(dim_data):
                            new_cs_stats_pts[dist_cs[0][0]].append(data_idx)
                            cnt_ += 1
                            continue
                        else:
                            rs[data_idx] = new_data[data_idx]
                            rs_cnt += 1
                            cnt_ += 1
                    else:
                        rs[data_idx] = new_data[data_idx]
                        rs_cnt += 1
                        cnt_ += 1

            print("Count:", cnt_)
            #Merge new DS points with old DS points:
            tot_pts_ds = 0
            for c, v in new_ds_stats_pts.items():
                ds_stats_pts[c]["PTS"] += v
                tot_pts_ds += len(ds_stats_pts[c]["PTS"])

            #print_total_breakdown(ds_stats_pts)
            ds_stats_pts = change_centroids_non_kmeans(new_data, ds_stats_pts, new_ds_stats_pts)
            ds_stats_pts = generate_std_key(ds_stats_pts)
            #print("LENGTH DS_STATS:", len(ds_stats_pts))

            tot_pts_cs = 0
            for c, v in new_cs_stats_pts.items():
                cs_stats_pts[c]["PTS"] += v
                tot_pts_cs += len(cs_stats_pts[c]["PTS"])
            cs_stats_pts = change_centroids_non_kmeans(new_data, cs_stats_pts, new_cs_stats_pts)

            #Run Clustering Alg. on RS:
            if len(rs.keys()) != 0:
                rs_cluster_info, rs_cluster_pts, poss_outliers = kmeans(rs, n_clusters * 3, 10)
                new_rs = defaultdict(list)
                for idx_ in poss_outliers.keys():
                    new_rs[idx_] = poss_outliers[idx_]

                rs_to_remove = []
                for c, pts_lst in rs_cluster_pts.items():
                    if len(pts_lst) == 0:
                        rs_to_remove.append(c)
                    elif len(pts_lst) == 1:
                        data_pt_idx = pts_lst[0]
                        new_rs[data_pt_idx] = rs[data_pt_idx]
                        rs_to_remove.append(c)
                    else:
                        new_c_key = len(cs_stats_pts.keys())
                        while True:
                            if new_c_key in cs_stats_pts:
                                new_c_key += 1
                            else:
                                break
                        cs_stats_pts[new_c_key] = {}
                        cs_stats_pts[new_c_key]["MEAN"] = rs_cluster_info[c]["MEAN"]
                        cs_stats_pts[new_c_key]["SUMSQ_N"] = rs_cluster_info[c]["SUMSQ_N"]
                        cs_stats_pts[new_c_key]["PTS"] = rs_cluster_pts[c]
                        cs_stats_pts[new_c_key]["N"] = len(rs_cluster_pts[c])

                cs_stats_pts = generate_std_key(cs_stats_pts)
                rs.clear()
                rs = new_rs

            #Merge CS Clusters:
            cs_stats_pts = merge_cs_clusters(cs_stats_pts, dim_data)
            cs_stats_pts = generate_std_key(cs_stats_pts)

            tot_pts_ds = 0
            for c, v in ds_stats_pts.items():
                tot_pts_ds += len(ds_stats_pts[c]["PTS"])

            tot_pts_cs = 0
            for c, v in cs_stats_pts.items():
                tot_pts_cs += len(cs_stats_pts[c]["PTS"])

            sum_ = tot_pts_ds + tot_pts_cs + len(rs)
            print("End Round" + str(i) + ": " +str(sum_))

            if i == len(os.listdir(input_fp))-1:
                #print_total_breakdown(ds_stats_pts)
                print("FINAL MERGE")
                #print(len(cs_stats_pts.keys()))
                #cs_stats_pts = merge_cs_clusters(cs_stats_pts, dim_data)
                #cs_stats_pts = generate_std_key(cs_stats_pts)
                #print(len(cs_stats_pts.keys()))
                if len(cs_stats_pts.keys()) != 0:
                    ds_stats_pts, cs_stats_pts, cs_to_remove = merge_cs_to_ds_clusters(ds_stats_pts, cs_stats_pts, dim_data)
                    cs_stats_pts = {key: cs_stats_pts[key] for key in cs_stats_pts if key not in cs_to_remove}
                    #print("LENGTH DS_STATS:", len(ds_stats_pts))

                if len(rs.keys()) != 0:
                    ds_stats_pts, rs, rs_to_remove = merge_rs_to_ds_clusters(ds_stats_pts, rs, dim_data)
                    rs = {key: rs[key] for key in rs if key not in rs_to_remove}
                    #print("LENGTH DS_STATS:", len(ds_stats_pts))

                #print_total_breakdown(ds_stats_pts)
                tot_pts_ds = 0
                for c, v in ds_stats_pts.items():
                    tot_pts_ds += len(ds_stats_pts[c]["PTS"])

                tot_pts_cs = 0
                for c, v in cs_stats_pts.items():
                    tot_pts_cs += len(cs_stats_pts[c]["PTS"])


        inter_results[i] = [i+1, len(ds_stats_pts), tot_pts_ds, len(cs_stats_pts), tot_pts_cs, len(rs)]

        with open(output_fp_inter, "w+") as w:
            writer = csv.writer(w)
            for k, v in inter_results.items():
                writer.writerow(v)

        #Organize DS_cluster results:
        results = {}
        for c, v in ds_stats_pts.items():
            c_pts = v["PTS"]
            for pt in c_pts:
                results[pt] = c

        for c, v in cs_stats_pts.items():
            c_pts = v["PTS"]
            for pt in c_pts:
                results[pt] = -1

        for k_ in rs:
            results[k_] = -1

        results = dict(sorted(results.items(), key=lambda item: item[0]))
        with open(output_fp_cluster, "w") as w:
            w.write(json.dumps(results))

    end = time.time()
    print("Duration:", round((end - start), 2))
