from pyspark import SparkContext, SparkConf
from collections import defaultdict
import sys
import time

def calculate_betweenness(node):
    """Create the graph given the starting node, calculate betweenness, return all edge scores."""


def generate_graph_components(user_busList, t):
    """Generate the edges, vertices, and adjacency edges of the graph."""
    edges = []
    vertices = []
    adjacent_edges = defaultdict(list)
    for user1 in user_busList:
        for user2 in user_busList:
            if user1[0] != user2[0]:
                set1, set2 = set(user1[1]), set(user2[1])
                if len(set1.intersection(set2)) >= t:
                    vertices.append((user1[0],))
                    vertices.append((user2[0],))
                    edges.append((user1[0], user2[0]))

    for edge in edges:
        adjacent_edges[edge[0]] += edge[1]

    return list(set(edges)), list(set(vertices)), adjacent_edges



if __name__ == "__main__":
    start = time.time()
    # filter_threshold = int(sys.argv[1])
    # input_fp = sys.argv[2]
    # comm_output_fp = sys.argv[3]
    # bwness_output_fp = sys.argv[4]

    filter_threshold = 7
    input_fp = "./data/ub_sample_data.csv"
    comm_output_fp = "./data/output2_comm.txt"
    bwness_output_fp = "./data/output2_bwness.txt"

    conf = SparkConf()
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.driver.memory", "4g")
    conf.setMaster('local[3]')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

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

    # Generate the graph edges, vertices, and adjacent_edges:
    edges, vertices, adjacent_edges = generate_graph_components(user_busList, filter_threshold)

    # Create Tree and calculate Betweenness using Girvan-Newman:
    betweeness_lst = sc.parallelize(vertices) \
                    .map(lambda starting_node: calculate_betweenness(starting_node)) \
                    .groupByKey() \
                    .map(lambda n_bL: (n_bL[0], sum(n_bL[1]) / 2)) \
                    .take(5)


    for val in betweeness_lst:
        print(val)


    end = time.time()
    print("Duration:", round((end-start),2))