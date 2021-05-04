from pyspark import SparkContext, SparkConf
import time
import json
import sys
import random
import binascii
import os

NUM_HASH_FUNCTIONS = 10

os.environ["PYSPARK_PYTHON"] = '/usr/local/bin/python3.6'
os.environ["PYSPARK_DRIVER_PYTHON"] = '/usr/local/bin/python3.6'

def create_hash_functions():
    hash_list = []
    m = 10000
    p = 22777

    for hash_num in range(NUM_HASH_FUNCTIONS):
        a = random.randint(1, p-1)
        b = random.randint(1, p-1)
        hash_list.append(hash_formula(a, b, p, m))

    return hash_list

def hash_formula(a, b, p, m):
    def formula(city):
        return ((a * city + b) % p) % m

    return formula

def predict(city, bits, fn_list):
    if city is not None and city != "":
        str_city = int(binascii.hexlify(city.encode('utf8')), 16)
        hashed_city = set([f(str_city) for f in fn_list])
        if hashed_city.issubset(bits):
            yield 1
        else:
            yield 0

    else:
        yield 0

if __name__ == "__main__":
    start = time.time()
    first_json_path = sys.argv[1]
    second_json_path = sys.argv[2]
    output_fp = sys.argv[3]

    # first_json_path = "./data/business_first.json"
    # second_json_path = "./data/business_second.json"
    # output_fp = "./output/output1.csv"

    conf = SparkConf().set('spark.driver.host', '127.0.0.1')
    # conf = SparkConf()
    conf.set("spark.executor.memory", "2g")
    conf.set("spark.driver.memory", "2g")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    data_rdd = sc.textFile(first_json_path)
    train_rdd = data_rdd \
                .map(lambda x: json.loads(x)) \
                .map(lambda x: x["city"]) \
                .distinct() \
                .filter(lambda c: c != "") \
                .map(lambda c: int(binascii.hexlify(c.encode('utf8')), 16))

    hash_lst = create_hash_functions()
    city_hashed = train_rdd \
                .flatMap(lambda c: [f(c) for f in hash_lst]) \
                .collect()

    active_bits = set(city_hashed)

    val_data = sc.textFile(second_json_path)
    val_rdd = val_data \
                .map(lambda x: json.loads(x)) \
                .map(lambda x: x["city"]) \
                .flatMap(lambda c: predict(c, active_bits, hash_lst)) \
                .collect()

    with open(output_fp, "w") as w:
        total_ = len(val_rdd)
        counter = 0
        for val in val_rdd:
            counter += 1
            if counter < total_:
                w.write(str(val) + " ")
            else:
                print(counter)
                w.write(str(val))

    end = time.time()
    print("Duration:", round((end - start), 2))
