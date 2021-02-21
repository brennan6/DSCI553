import org.apache.spark.rdd.RDD

import scala.io.Source
import org.json4s._
import org.apache.spark.{SparkConf, SparkContext}
import org.json4s.jackson.JsonMethods.{compact, parse, pretty, render}
import org.json4s.jackson.{Json, Serialization}

import java.io.{BufferedWriter, File, FileWriter}
import scala.collection.immutable.ListMap
import scala.collection.mutable.ListBuffer

object task1 {
  def main(args: Array[String]): Unit = {

    implicit val formats = org.json4s.DefaultFormats
    val input_fp = args(0)
    val output_fp = args(1)
    val stopwords_fp = args(2)
    val year = args(3)
    val m = args(4).toInt
    val n = args(5).toInt

    val conf = new SparkConf().setAppName("task1").setMaster("local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val rdd_review = sc.textFile(input_fp).map(line => parse(line))
                    .cache()

    val stop_words = Source.fromFile(stopwords_fp).getLines.toList
    val stop_punctuation = List('(', '[', ',', '.', '!', '?', ':', ';', ']', ')', "", '"')
    val stop_words_full = stop_words ++ stop_punctuation

    val count_total = get_count_total(rdd_review)
    val count_year = get_count_year(year, rdd_review)
    val count_distinct_user = get_count_distinct_user(rdd_review)
    val top_m_users = get_top_m_users(m, rdd_review)
    val top_n_words = get_top_n_words(n, rdd_review, stop_words_full, stop_punctuation)
    val top_m_users_lst = new ListBuffer[Any]()
    for (x <- top_m_users) {
      var inner_lst = new ListBuffer[Any]()
      val k = x._1
      val v = x._2
      inner_lst += k
      inner_lst += v
      top_m_users_lst += inner_lst.toList
    }

    val results = Map[String, Any]("A" -> count_total, "B" -> count_year, "C" -> count_distinct_user, "D" -> top_m_users_lst.toList, "E" -> top_n_words)

    val file = new File(output_fp)
    val bw = new BufferedWriter(new FileWriter(file))
    val output = Serialization.write(ListMap(results.toSeq.sortBy(_._1): _*))
    val output_new = output.replaceAll("\\\\", "").replaceAll("\"{2}", "\"")
    bw.write(output_new)
    bw.close()
  }

  def get_count_total(rdd_review: RDD[JValue]): Long = {
    """The total number of reviews"""
    val count_total = rdd_review.count()
    return count_total
  }

  def get_count_year(year: String, rdd_review: RDD[JValue]): Long = {
    """The number of reviews in a given year, y"""
    val count_year = rdd_review
      .map(x => pretty(render(x\"date")))
      .filter(x => x.contains(year))
      .count()
    return count_year
  }

  def get_count_distinct_user(rdd_review: RDD[JValue]): Long = {
    """The number of distinct users who have written the reviews"""
    val count_distinct_user = rdd_review
      .map(x => pretty(render(x\"user_id")))
      .distinct()
      .count()
    return count_distinct_user
  }

  def get_top_m_users(m: Int, rdd_review: RDD[JValue]): Array[(String, Int)] = {
    val top_m_users = rdd_review.map(x => (pretty(render(x\"user_id")), 1))
      .reduceByKey(_ + _)
      .sortBy(_._2, false)
      .take(m)
    return top_m_users
  }

  def trim(word: String, stop_words_full: List[Any], stop_punctuation: List[Any]): Any = {
    if (!stop_words_full.contains(word)) {
      val str1 = word.filterNot(x => stop_punctuation.contains(x))
      return str1
    }
  }
  def get_top_n_words(n: Int, rdd_review: RDD[JValue], stop_words_full: List[Any], stop_punctuation: List[Any]): Array[Any] = {
    val top_n_words = rdd_review.map(x => pretty(render(x\"text")))
      .flatMap(line => line.toLowerCase().split(' '))
      .filter(x => !stop_words_full.contains(x))
      .map(x => (trim(x, stop_words_full, stop_punctuation), 1))
      .reduceByKey(_ + _)
      .sortBy(_._2, false)
      .keys
      .take(n)
    return top_n_words
  }
}



