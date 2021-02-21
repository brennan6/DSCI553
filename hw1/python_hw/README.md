#Task 1 Sample Input
~/Documents/spark-2.3.0-bin-hadoop2.7/bin/spark-submit task1.py ./data/review.json ./output.txt ./data/stopwords 2016 10 22

#Task 2 Sample Input
w/ spark:
~/Documents/spark-2.3.0-bin-hadoop2.7/bin/spark-submit task2.py ./data/review.json ./data/business.json ./output_w.txt spark 20

w/o spark:
~/Documents/spark-2.3.0-bin-hadoop2.7/bin/spark-submit task2.py ./data/review.json ./data/business.json ./output_wo.txt no_spark 20

#Task 3 Sample Input
customized:
~/Documents/spark-2.3.0-bin-hadoop2.7/bin/spark-submit task3.py ./data/review.json ./output_3.txt customized 15 800

default:
~/Documents/spark-2.3.0-bin-hadoop2.7/bin/spark-submit task3.py ./data/review.json ./output_3.txt default 15 800

spark-submit task1.py $ASNLIB/publicdata/review.json task1_ans $ASNLIB/publicdata/stopwords 2018 10 10