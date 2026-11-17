package com.github.milenkovicm.spark.ai

import java.net.{InetSocketAddress, Socket}

import scala.util.Try

/** Runs against a local Ollama (`qwen2.5:1.5b-instruct`); cancelled when it isn't reachable. */
class OllamaIntegrationSuite extends SparkSessionSuite {
  private val model = "qwen2.5:1.5b-instruct"

  private lazy val available = Try {
    val s = new Socket()
    try s.connect(new InetSocketAddress("127.0.0.1", 11434), 1000)
    finally s.close()
  }.isSuccess

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    spark.conf.set(AiConfig.BaseUrl, "http://127.0.0.1:11434/v1")
    spark.conf.set(AiConfig.DefaultModel, model)
    spark.conf.set(AiConfig.Temperature, "0")
    spark.sql(
      """SELECT * FROM VALUES
        |  (1, 'I absolutely love this phone, the battery lasts for days!'),
        |  (2, 'Terrible service. The package arrived broken and nobody answered my emails.'),
        |  (3, 'The movie was a masterpiece, I would watch it again.'),
        |  (4, 'Worst purchase ever, it stopped working after one day.')
        |AS reviews(id, review)""".stripMargin).createOrReplaceTempView("reviews")
  }

  private def timed[T](label: String)(f: => T): T = {
    val start = System.nanoTime()
    val r = f
    info(s"$label: ${(System.nanoTime() - start) / 1000000} ms")
    r
  }

  test("ai_query returns an answer per row") {
    assume(available, "Ollama is not running on 127.0.0.1:11434")
    val rows = timed("ai_query x4") {
      spark.sql(
        s"""SELECT id, ai_query('$model', concat('Reply with one word describing this review: ', review),
           |                    named_struct('max_tokens', 10)) AS word
           |FROM reviews ORDER BY id""".stripMargin).collect()
    }
    rows.foreach(r => info(s"${r.getInt(0)} -> ${r.getString(1)}"))
    assert(rows.map(_.getInt(0)).toSeq == Seq(1, 2, 3, 4))
    assert(rows.forall(r => r.getString(1) != null && r.getString(1).trim.nonEmpty))
  }

  test("ai_classify picks one of the labels") {
    assume(available, "Ollama is not running on 127.0.0.1:11434")
    val rows = timed("ai_classify x4") {
      spark.sql(
        "SELECT id, ai_classify(review, array('positive', 'negative')) AS label FROM reviews ORDER BY id")
        .collect()
    }
    val labels = rows.map(r => r.getInt(0) -> r.getString(1).trim.toLowerCase.stripSuffix(".")).toMap
    info(labels.toString)
    assert(labels.values.forall(Set("positive", "negative")))
    assert(labels == Map(1 -> "positive", 2 -> "negative", 3 -> "positive", 4 -> "negative"))
  }

  test("ai_analyze_sentiment through the DataFrame API") {
    assume(available, "Ollama is not running on 127.0.0.1:11434")
    import org.apache.spark.sql.functions.{col, expr}
    val rows = spark.table("reviews").where(col("id") <= 2)
      .select(col("id"), expr("ai_analyze_sentiment(review)").as("s")).orderBy("id").collect()
    val s = rows.map(_.getString(1).trim.toLowerCase.stripSuffix("."))
    info(s.mkString(", "))
    assert(s.toSeq == Seq("positive", "negative"))
  }
}
