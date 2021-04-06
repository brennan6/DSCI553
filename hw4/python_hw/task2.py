from pyspark import SparkContext, SparkConf
from collections import defaultdict
import sys
import math
import time
import random


def calculate_betweenness(node, adjacent_edges):
    """Create the graph given the starting node, calculate betweenness, return all edge scores."""
    tree_graph = defaultdict(list)
    get_parents = defaultdict(list)
    get_children, count_paths_d = {}, {}

    tree_graph[0] = node
    count_paths_d[node] = 1
    visited_ = [node]

    this_levels_nodes = adjacent_edges[node]
    get_children[node] = this_levels_nodes
    level = 1

    while len(this_levels_nodes) > 0:
        tree_graph[level] = set(this_levels_nodes)
        visited_ = set(visited_).union(set(this_levels_nodes))

        next_level_nodes = []
        for parent_ in this_levels_nodes:
            node_adjacents = set(adjacent_edges[parent_])
            children_ = node_adjacents.difference(set(visited_))

            get_children[parent_] = children_
            for parent, children in get_children.items():
                for child in children:
                    get_parents[child].append(parent)

            grand_parents_ = set(get_parents[parent_])
            if len(grand_parents_) == 0:
                count_paths_d[parent_] = 1
            else:
                count_paths_d[parent_] = sum([count_paths_d[node_] for node_ in grand_parents_])

            next_level_nodes.append(children_)

        level += 1
        next_level_nodes = set([node for sublst in next_level_nodes for node in sublst])
        this_levels_nodes = list(next_level_nodes.difference(visited_))

    node_value, edge_values = {}, {}
    traverse_level = level - 1
    while traverse_level > 0:
        this_levels_nodes = tree_graph[traverse_level]
        for node_ in this_levels_nodes:
            parents_ = set(get_parents[node_])
            for parent_ in parents_:
                w = count_paths_d[parent_] / count_paths_d[node_]
                if node_ not in node_value:
                    node_value[node_] = 1
                if parent_ not in node_value:
                    node_value[parent_] = 1
                edge_values[tuple(sorted((node_, parent_)))] = w * node_value[node_]
                node_value[parent_] += w * node_value[node_]

        traverse_level -= 1

    output_ = []
    for edge, e_v in edge_values.items():
        output_.append((edge, e_v))

    return output_


def generate_graph_components(user_busList, t):
    """Generate the edges, vertices, and adjacency edges of the graph."""
    edges, vertices = [], []
    adjacent_edges = defaultdict(list)
    for user1 in user_busList:
        for user2 in user_busList:
            if user1[0] is not user2[0]:
                set1, set2 = set(user1[1]), set(user2[1])
                if len(set1.intersection(set2)) >= t:
                    vertices.append((user1[0],))
                    vertices.append((user2[0],))
                    edges.append((user1[0], user2[0]))

    for edge in edges:
        adjacent_edges[edge[0]].append(edge[1])

    return list(set(edges)), list(set(vertices)), adjacent_edges


def create_communities(vertices, adj_edges):
    total_vertices = len(vertices)
    communities = []
    visited_global = set()
    print("Target:", total_vertices)
    while len(visited_global) < total_vertices:
        print("Visited:", len(visited_global))
        node_1 = random.choice(vertices)
        visited_local = set([node_1])
        adj_nodes = adj_edges[node_1]

        while True:
            visited_local = visited_local.union(adj_nodes)
            not_visited_local = set()
            for node in adj_nodes:
                inner_adj_nodes = adj_edges[node]
                not_visited_local = not_visited_local.union(inner_adj_nodes)

            visited_global = visited_global.union(visited_local)
            adj_nodes = not_visited_local.difference(visited_local)

            if len(adj_nodes) == 0:
                break

        communities.append(list(visited_local))
        vertices = list(set(vertices).difference(visited_global))

    return communities


def modularity(communities, m):
    mod_ = 0
    for comm_ in communities:
        s_mod = 0
        for i in comm_:
            for j in comm_:
                if j in A[i]:
                    a = 1
                else:
                    a = 0
                s_mod += (a - (K[i] * K[j])) / (2 * m)
        mod_ += s_mod

    return mod_ / (2 * m)


if __name__ == "__main__":
    start = time.time()
    # filter_threshold = int(sys.argv[1])
    # input_fp = sys.argv[2]
    # bwness_output_fp = sys.argv[3]
    # comm_output_fp = sys.argv[4]

    filter_threshold = 7
    input_fp = "./data/ub_sample_data.csv"
    bwness_output_fp = "./data/output2_bwness.txt"
    comm_output_fp = "./data/output2_comm.txt"

    conf = SparkConf()
    conf.set("spark.executor.memory", "4g")
    conf.set("spark.driver.memory", "4g")
    conf.setMaster('local[3]')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    input_rdd = sc.textFile(input_fp)
    header = input_rdd.first()
    reviews_rdd = input_rdd \
        .filter(lambda row: row != header)

    user_busList = reviews_rdd \
        .map(lambda line: line.split(",")) \
        .map(lambda u_b: (u_b[0], u_b[1])) \
        .groupByKey() \
        .map(lambda u_bL: (u_bL[0], list(set(u_bL[1])))) \
        .collect()

    # Generate the graph edges, vertices, and adjacent_edges:
    edges, vertices, adjacent_edges = generate_graph_components(user_busList, filter_threshold)
    vertices = [vert_[0] for vert_ in vertices]

    # Create Tree and calculate Betweenness using Girvan-Newman:
    betweeness_lst = sc.parallelize(vertices) \
        .map(lambda starting_node: calculate_betweenness(starting_node, adjacent_edges)) \
        .flatMap(lambda l: [(edge[0], edge[1]) for edge in l]) \
        .groupByKey() \
        .map(lambda e_bL: (e_bL[0], sum(e_bL[1]) / 2)) \
        .sortBy(lambda e_b: (-e_b[1], e_b[0])) \
        .collect()

    with open(bwness_output_fp, "w") as w:
        for edge in betweeness_lst:
            w.write(str(edge[0]) + ", " + str(edge[1]) + "\n")

    # Find Communities using Modularity:
    m = len(edges) / 2

    # Create K, where K[i] is the lookup for degree:
    K = {}
    for v in vertices:
        K[v] = len(adjacent_edges[v])

    # Adds value to A[i] only if Aij is 1, not if 0:
    A = defaultdict(list)
    for v1 in vertices:
        for v2 in vertices:
            if v2 in adjacent_edges[v1]:
                A[v1].append(v2)

    cuts = 0
    best_mod_ = -math.inf
    while cuts < m:
        # Make the cut based on highest betweenness, remove from adjacent_edge list:
        del_edge = betweeness_lst[0]
        p1, p2 = del_edge[0][0], del_edge[0][1]

        adjacent_edges[p1].remove(p2)
        adjacent_edges[p2].remove(p1)

        cuts += 1

        # Find Communities within new structure:
        communities = create_communities(vertices, adjacent_edges)

        # Calculate Modularity of the community:
        mod_ = modularity(communities, m)

        if best_mod_ < mod_:
            best_mod_ = mod_
            print("Best Mod:", mod_)
            best_communities = communities

        # Run through G-N to extract betweenness to remove next edge:
        betweeness_lst = sc.parallelize(vertices) \
            .map(lambda starting_node: calculate_betweenness(starting_node, adjacent_edges)) \
            .flatMap(lambda l: [(edge[0], edge[1]) for edge in l]) \
            .groupByKey() \
            .map(lambda e_bL: (e_bL[0], sum(e_bL[1]) / 2)) \
            .sortBy(lambda e_b: (-e_b[1], e_b[0])) \
            .collect()

    # Sort each list of list lexigraphically first:
    best_communities = [sorted(sublst) for sublst in best_communities]
    community_list = sorted(best_communities, key=lambda x: (len(x), x))

    with open(comm_output_fp, "w") as w:
        for community in community_list:
            comm_len = len(community)
            counter = 0
            for id_ in community:
                if counter == (comm_len - 1):
                    w.write("'" + id_ + "'\n")
                else:
                    w.write("'" + id_ + "', ")
                    counter += 1

    end = time.time()
    print("Duration:", round((end - start), 2))
