from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from graphframes import GraphFrame
import sys
import os
import time

os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")

def generate_graph_components(user_busList, t):
    edges, vertices = [], []
    for user1 in user_busList:
        for user2 in user_busList:
            if user1[0] is not user2[0]:
                set1, set2 = set(user1[1]), set(user2[1])
                if len(set1.intersection(set2)) >= t:
                    vertices.append((user1[0],))
                    vertices.append((user2[0],))
                    edges.append((user1[0], user2[0]))
    return list(set(edges)), list(set(vertices))

if __name__ == "__main__":
    start = time.time()
    filter_threshold = int(sys.argv[1])
    input_fp = sys.argv[2]
    output_fp = sys.argv[3]

    # filter_threshold = 7
    # input_fp = "./data/ub_sample_data.csv"
    # output_fp = "./data/output1.txt"

    conf = SparkConf()
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.driver.memory", "4g")
    conf.setMaster('local[3]')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    sqlc = SQLContext(sc)

    input_rdd = sc.textFile(input_fp)
    header = input_rdd.first()
    reviews_rdd = input_rdd\
                .filter(lambda row: row != header)

    user_busList = reviews_rdd \
                .map(lambda line: line.split(",")) \
                .map(lambda u_b: (u_b[0], u_b[1])) \
                .groupByKey() \
                .map(lambda u_bL: (u_bL[0], list(set(u_bL[1])))) \
                .collect()

    edges, vertices = generate_graph_components(user_busList, filter_threshold)

    vertices = sqlc.createDataFrame(vertices, ["id"])
    edges = sqlc.createDataFrame(edges, ["src", "dst"])
    g = GraphFrame(vertices, edges)
    result = g.labelPropagation(maxIter=5)

    # Convert to RDD, group by label (community), sort for output lexicographically:

    community_list = result.rdd \
                    .map(lambda id_c: (id_c[1], id_c[0])) \
                    .groupByKey() \
                    .map(lambda c_idL: (list(sorted(c_idL[1])))) \
                    .collect()

    community_list = sorted(community_list, key=lambda x: (len(x), x))

    with open(output_fp, "w") as w:
        for community in community_list:
            comm_len = len(community)
            counter = 0
            for id_ in community:
                if counter == (comm_len-1):
                    w.write("'" + id_ + "'\n")
                else:
                    w.write("'" + id_ + "', ")
                    counter += 1

    end = time.time()
    print("Duration:", round((end-start),2))