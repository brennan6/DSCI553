from pyspark import SparkContext
import json
import csv
import time

if __name__ == "__main__":
    start = time.time()
    review_fp = "./data/review.json"
    business_fp = "./data/business.json"
    output_fp = "./data/user_business.csv"

    sc = SparkContext('local[*]', 'task2')

    rdd_reviews = sc.textFile(review_fp)
    rdd_business = sc.textFile(business_fp)

    rdd_business_nv = rdd_business \
        .map(lambda x: json.loads(x))\
        .map(lambda business_kv: (business_kv["business_id"], business_kv["state"]))\
        .filter(lambda id_state: id_state[1] == "NV")\
        .distinct()\
        .keys()\
        .collect()

    rdd_reviews_nv = rdd_reviews \
        .map(lambda x: json.loads(x))\
        .map(lambda reviews_kv: (reviews_kv["user_id"], reviews_kv["business_id"]))\
        .filter(lambda uid_bid: uid_bid[1] in rdd_business_nv)\
        .collect()

    with open(output_fp, "w") as user_business_file:
        file_writer = csv.writer(user_business_file, delimiter=',')
        file_writer.writerow(["user_id", "business_id"])

        for uid_bid in rdd_reviews_nv:
            file_writer.writerow(uid_bid)

    end = time.time()
    print("Duration:", round((end-start),2))



