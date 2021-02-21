from pyspark import SparkContext
import sys
import json
import platform
print(platform.python_version())

def create_customized_partition(num_partitions):
    return reviews_rdd.partitionBy(num_partitions, lambda x: ord(x[0]))

if __name__ == "__main__":
    input_fp = sys.argv[1]
    output_fp = sys.argv[2]
    partition_type = sys.argv[3]
    n_partitions = int(sys.argv[4])
    n = int(sys.argv[5])

    #For local run only:
    #input_fp = "./data/review.json"
    #output_fp = "./output_3.txt"
    #partition_type = "customized"
    #n_partitions = 27
    #n = 300

    sc = SparkContext('local[*]', 'task3')

    results = {}

    input_rdd = sc.textFile(input_fp)
    reviews_rdd = input_rdd\
        .map(lambda x: json.loads(x))\
        .map(lambda x: (x["business_id"], 1))

    if partition_type == "customized":
        reviews_rdd = create_customized_partition(n_partitions)

    results["n_partitions"] = reviews_rdd.getNumPartitions()

    n_items = reviews_rdd\
                .glom()\
                .map(len)\
                .collect()
    results["n_items"] = n_items

    result_rdd = reviews_rdd\
                    .reduceByKey(lambda a,b: a+b)\
                    .filter(lambda kv: kv[1] > n)\
                    .collect()
    results["result"] = result_rdd

    with open(output_fp, 'w') as w:
        w.write(json.dumps(results))