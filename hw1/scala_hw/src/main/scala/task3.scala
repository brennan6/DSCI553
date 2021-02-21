import org.apache.spark.rdd.RDD
import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}
import org.json4s.jackson.JsonMethods.{parse, pretty, render}
import org.json4s.jackson.Serialization

import java.io.{BufferedWriter, File, FileWriter}
import scala.collection.mutable.ListBuffer

object task3 {
  implicit val formats = org.json4s.DefaultFormats

  def main(args: Array[String]): Unit = {
    val input_fp = args(0)
    val output_fp = args(1)
    val partition_type = args(2)
    val n_partitions = args(3).toInt
    val n = args(4).toInt

    val conf = new SparkConf().setAppName("task3").setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    var reviews_rdd = sc.textFile(input_fp)
      .map(line => parse(line))
      .map(x => (pretty(render(x \ "business_id")), 1))


    if (partition_type == "customized") {
      reviews_rdd = create_customized_partition(n_partitions, reviews_rdd)
    }

    val num_partitions = reviews_rdd.partitions.size

    val n_items = reviews_rdd
      .glom()
      .map(_.size).collect()

    val result_rdd = reviews_rdd
      .reduceByKey(_ + _)
      .filter(x => x._2 > n)
      .collect()

    val result_rdd_updated = new ListBuffer[Any]()
    for (x <- result_rdd) {
      var inner_lst = new ListBuffer[Any]()
      val k = x._1
      val v = x._2
      inner_lst += k
      inner_lst += v
      result_rdd_updated += inner_lst.toList
    }

    val results = Map[String, Any]("n_partitions" -> num_partitions, "n_items" -> n_items.toList,
      "result" -> result_rdd_updated.toList)

    val file = new File(output_fp)
    val bw = new BufferedWriter(new FileWriter(file))
    val output = Serialization.write(results)
    val output_new = output.replaceAll("\\\\", "").replaceAll("\"{2}", "\"")
    bw.write(output_new)
    bw.close()
  }

  def create_customized_partition(n_partitions: Int, rdd: RDD[(String, Int)]): RDD[(String, Int)] = {
    val partitioner = new HashPartitioner(n_partitions)
    return rdd.partitionBy(partitioner)
  }

}
