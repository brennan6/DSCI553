from pyspark import SparkContext
import platform
import sys
import json
print(platform.python_version())

def get_average_stars_across_business_spark(n):
    """compute the average stars for each business category and output top n categories with the highest
    average stars with spark."""
    category_rdd = rdd_business\
                    .map(lambda business_kv: (business_kv["business_id"], business_kv["categories"]))\
                    .filter(lambda business_kv: business_kv[1] is not None)\
                    .map(lambda x: (x[0], x[1].split(",")))\
                    .mapValues(lambda category: [cat_.strip() for cat_ in category]).cache()

    score_rdd = rdd_reviews \
                .map(lambda score_kv: (score_kv["business_id"], score_kv["stars"]))\
                .groupByKey() \
                .mapValues(lambda stars: [float(star) for star in stars])\
                .map(lambda score_kvv: (score_kvv[0], (sum(score_kvv[1]), len(score_kvv[1])))).cache()

    combined_rdd = category_rdd.leftOuterJoin(score_rdd)

    result_rdd = combined_rdd\
                .map(lambda category_kvv: category_kvv[1])\
                .filter(lambda category_kv: category_kv[1] is not None)\
                .flatMap(lambda category_kv: [(cat, category_kv[1]) for cat in category_kv[0]])\
                .reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1]))\
                .mapValues(lambda values: round(values[0]/values[1], 1)) \
                .takeOrdered(n, key=lambda sort_order: (-sort_order[1], sort_order[0]))
    results["result"] = result_rdd

def get_average_stars_across_business_wo_spark(n):
    """compute the average stars for each business category and output top n categories with the highest
    average stars w/o spark."""
    combined_kv = {}
    for key in category_kv.keys():
        if key not in score_kv.keys():
            continue
        for cat_ in category_kv[key]:
            if cat_ not in combined_kv.keys():
                combined_kv[cat_] = score_kv[key]
            else:
                combined_kv[cat_] = [combined_kv[cat_][0] + score_kv[key][0], combined_kv[cat_][1] + score_kv[key][1]]

    result_kv = {}
    for key in combined_kv.keys():
        result_kv[key] = round((combined_kv[key][0]/combined_kv[key][1]), 1)
    result_kv = list(sorted(result_kv.items(), key=lambda x: (-x[1], x[0])))[:n]

    results["result"] = result_kv

if __name__ == "__main__":
    review_fp = sys.argv[1]
    business_fp = sys.argv[2]
    output_fp = sys.argv[3]
    if_spark = sys.argv[4]
    n = int(sys.argv[5])

    #For Local Run Only:
    #review_fp = "./data/review.json"
    #business_fp = "./data/business.json"
    #output_fp = "./output_wo.txt"
    #if_spark = "no_spark"
    #n = 20

    results = {}
    if if_spark == "spark":
        sc = SparkContext('local[*]', 'task2')
        rdd_reviews = sc.textFile(review_fp)
        rdd_business = sc.textFile(business_fp)

        rdd_reviews = rdd_reviews\
                        .map(lambda x: json.loads(x))

        rdd_business = rdd_business\
                        .map(lambda x: json.loads(x))

        get_average_stars_across_business_spark(n)
    else:
        reviews_file = open(review_fp, "r")
        score_kv = {}
        for record in reviews_file:
            review = json.loads(record)
            if review["business_id"] not in score_kv.keys():
                score_kv[review["business_id"]] = [float(review["stars"]), 1]
            else:
                score_kv[review["business_id"]] = [score_kv[review["business_id"]][0] + float(review["stars"]), score_kv[review["business_id"]][1] + 1]

        business_file = open(business_fp, "r")
        category_kv = {}
        for record in business_file:
            business = json.loads(record)
            if business["categories"] is None:
                continue

            business_lst = business["categories"].split(",")
            business_lst = [bus_.strip() for bus_ in business_lst]
            category_kv[business["business_id"]] = business_lst
        get_average_stars_across_business_wo_spark(n)

    with open(output_fp, 'w') as w:
        w.write(json.dumps(results))




