package com.github.milenkovicm.spark.ai

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.catalyst.plans.physical.{RangePartitioning, RoundRobinPartitioning, SinglePartition}
import org.apache.spark.sql.execution.CoalesceExec
import org.apache.spark.sql.execution.adaptive.AdaptiveSparkPlanHelper
import org.apache.spark.sql.execution.exchange.{REPARTITION_BY_NUM, ShuffleExchangeExec}

/** Planning and execution of `ai_query` against [[StubOpenAiServer]] (200 ms per request). */
class AiQuerySuite extends SparkSessionSuite with AdaptiveSparkPlanHelper {
  private val delayMs = 200L
  private lazy val server = new StubOpenAiServer(delayMs)

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    spark.conf.set(AiConfig.BaseUrl, server.baseUrl)
    spark.conf.set(AiConfig.DefaultModel, "stub")
    spark.conf.set(AiConfig.MaxRetries, "0")
    spark.range(0, 1000, 1, 1).selectExpr("id", "concat('prompt ', id) AS text").createOrReplaceTempView("t")
    spark.range(0, 40, 1, 4).selectExpr("id", "concat('prompt ', id) AS text").createOrReplaceTempView("t4")
  }

  override protected def afterAll(): Unit = {
    try server.close()
    finally super.afterAll()
  }

  private def aiExecs(df: DataFrame): Seq[AiQueryExec] =
    collect(df.queryExecution.executedPlan) { case e: AiQueryExec => e }

  /** Shuffles added for `spark.sql.ai.partitions` (REPARTITION_BY_NUM, not Spark's own). */
  private def aiShuffles(df: DataFrame): Seq[ShuffleExchangeExec] =
    collect(df.queryExecution.executedPlan) {
      case e: ShuffleExchangeExec if e.shuffleOrigin == REPARTITION_BY_NUM => e
    }

  private def withConf[T](pairs: (String, String)*)(f: => T): T = {
    pairs.foreach { case (k, v) => spark.conf.set(k, v) }
    try f finally pairs.foreach { case (k, _) => spark.conf.unset(k) }
  }

  private def run(sql: String): (DataFrame, Array[org.apache.spark.sql.Row]) = {
    server.reset()
    val df = spark.sql(sql)
    (df, df.collect())
  }

  test("requests in a partition run concurrently, results stay in input order") {
    spark.conf.set(AiConfig.Concurrency, "16")
    val start = System.nanoTime()
    val (df, rows) = run("SELECT id, ai_query('stub', text) AS answer FROM t WHERE id < 200")
    val elapsedMs = (System.nanoTime() - start) / 1000000

    assert(rows.map(_.getLong(0)).toSeq == (0L until 200L))
    assert(rows.forall(r => r.getString(1) == s"echo:prompt ${r.getLong(0)}"))
    assert(server.requests == 200)
    assert(server.peakInFlight == 16)
    // 200 sequential requests would take 40 s; 16 at a time is ~2.5 s.
    assert(elapsedMs < 10000, s"took $elapsedMs ms")
    info(s"200 requests x $delayMs ms in $elapsedMs ms, peak in flight ${server.peakInFlight}")

    val execs = aiExecs(df)
    assert(execs.size == 1)
    assert(execs.head.metrics("numRequests").value == 200)
  }

  test("concurrency setting bounds requests in flight") {
    spark.conf.set(AiConfig.Concurrency, "4")
    try {
      val (_, rows) = run("SELECT ai_query('stub', text) FROM t WHERE id < 20")
      assert(rows.length == 20)
      assert(server.peakInFlight == 4)
    } finally spark.conf.set(AiConfig.Concurrency, "16")
  }

  test("ai_query is planned as AiQueryExec and never left in a ProjectExec") {
    val df = spark.sql("SELECT id, upper(ai_query('stub', text)) AS a FROM t WHERE id < 3")
    df.collect()
    val plan = df.queryExecution.executedPlan
    assert(aiExecs(df).size == 1)
    val leaked = collect(plan) {
      case p if !p.isInstanceOf[AiQueryExec] && p.expressions.exists(_.exists(_.isInstanceOf[AiQuery])) => p
    }
    assert(leaked.isEmpty, plan.treeString)
    assert(df.collect().map(_.getString(1)).toSeq == Seq("ECHO:PROMPT 0", "ECHO:PROMPT 1", "ECHO:PROMPT 2"))
  }

  test("a call referenced twice is executed once per row") {
    val (_, rows) = run(
      """SELECT a, length(a) FROM (SELECT ai_query('stub', text) AS a FROM t WHERE id < 10)""")
    assert(rows.length == 10)
    assert(server.requests == 10)
  }

  test("multiple and nested calls") {
    val (df, rows) = run(
      """SELECT ai_query('stub', text) AS a,
        |       ai_query('stub', concat('x ', text)) AS b,
        |       ai_query('stub', ai_query('stub', text)) AS c
        |FROM t WHERE id = 7""".stripMargin)
    assert(rows.head.toSeq == Seq("echo:prompt 7", "echo:x prompt 7", "echo:echo:prompt 7"))
    // `ai_query('stub', text)` appears twice but is requested once; the nested level adds one more.
    assert(server.requests == 3)
    assert(aiExecs(df).size == 2)
  }

  test("ai_query in WHERE, plain predicates are applied before calling the model") {
    val (_, rows) = run("SELECT id FROM t WHERE ai_query('stub', text) = 'echo:prompt 42' AND id < 50")
    assert(rows.map(_.getLong(0)).toSeq == Seq(42L))
    assert(server.requests == 50)
  }

  test("ai_query in GROUP BY") {
    val (_, rows) = run(
      "SELECT ai_query('stub', CAST(id % 3 AS STRING)) AS g, count(*) FROM t WHERE id < 30 GROUP BY 1 ORDER BY 1")
    assert(rows.map(r => (r.getString(0), r.getLong(1))).toSeq ==
      Seq(("echo:0", 10L), ("echo:1", 10L), ("echo:2", 10L)))
    assert(server.requests == 30)
  }

  test("ai_query inside an aggregate function") {
    val (_, rows) = run("SELECT count(DISTINCT ai_query('stub', CAST(id % 4 AS STRING))) FROM t WHERE id < 40")
    assert(rows.head.getLong(0) == 4)
    assert(server.requests == 40)
  }

  test("ai_query over aggregation results") {
    val (_, rows) = run(
      """SELECT g, ai_query('stub', concat(g, ':', count(*))) AS a
        |FROM (SELECT CAST(id % 2 AS STRING) AS g FROM t WHERE id < 10) GROUP BY g ORDER BY g""".stripMargin)
    assert(rows.map(r => (r.getString(0), r.getString(1))).toSeq == Seq(("0", "echo:0:5"), ("1", "echo:1:5")))
    assert(server.requests == 2)
  }

  test("ORDER BY ... LIMIT only sends the top rows to the model") {
    val (_, rows) = run("SELECT id, ai_query('stub', text) FROM t ORDER BY id DESC LIMIT 3")
    assert(rows.map(_.getLong(0)).toSeq == Seq(999L, 998L, 997L))
    assert(rows.head.getString(1) == "echo:prompt 999")
    assert(server.requests == 3)
  }

  test("null requests return null without a request") {
    val (_, rows) = run(
      "SELECT ai_query('stub', CASE WHEN id % 2 = 0 THEN text END) FROM t WHERE id < 10")
    assert(rows.count(_.isNullAt(0)) == 5)
    assert(server.requests == 5)
  }

  test("failures fail the query by default") {
    val e = intercept[Exception](run("SELECT ai_query('stub', 'FAIL') FROM t WHERE id < 2"))
    assert(e.getMessage.contains("ai_query to model 'stub' failed"))
  }

  test("fail_on_error = false returns null") {
    val (_, rows) = run(
      """SELECT ai_query('stub', IF(id = 1, 'FAIL', text), named_struct('fail_on_error', false))
        |FROM t WHERE id < 3""".stripMargin)
    assert(rows.map(r => Option(r.getString(0))).toSeq ==
      Seq(Some("echo:prompt 0"), None, Some("echo:prompt 2")))
  }

  test("transient 5xx errors are retried by LangChain4j") {
    spark.conf.set(AiConfig.MaxRetries, "2")
    try {
      val (_, rows) = run("SELECT ai_query('stub', concat('FLAKY ', text)) FROM t WHERE id < 4")
      assert(rows.map(_.getString(0)).toSeq == (0 until 4).map(i => s"echo:FLAKY prompt $i"))
      assert(server.requests == 8)
    } finally spark.conf.set(AiConfig.MaxRetries, "0")
  }

  test("ai_query in a window PARTITION BY") {
    val (_, rows) = run(
      "SELECT id, count(*) OVER (PARTITION BY ai_query('stub', CAST(id % 2 AS STRING))) FROM t WHERE id < 20")
    assert(rows.forall(_.getLong(1) == 10))
    assert(server.requests == 20)
  }

  test("unsupported placement fails at analysis") {
    val e = intercept[Exception](run(
      "SELECT * FROM t a JOIN t b ON ai_query('stub', a.text) = b.text"))
    info(e.getMessage.linesIterator.next())
    assert(e.getMessage.contains("ai_query"))
  }

  test("endpoint must be a constant") {
    val e = intercept[Exception](spark.sql("SELECT ai_query(text, text) FROM t").collect())
    assert(e.getMessage.contains("endpoint must be a constant string"))
    val n = intercept[Exception](spark.sql("SELECT ai_query(CAST(NULL AS STRING), text) FROM t").collect())
    assert(n.getMessage.contains("endpoint must not be null"))
  }

  test("misspelled params and invalid settings fail with a clear error") {
    val p = intercept[Exception](spark.sql(
      "SELECT ai_query('stub', text, named_struct('temprature', 0.0)) FROM t").collect())
    assert(p.getMessage.contains("unknown params temprature"))
    val v = intercept[Exception](spark.sql(
      "SELECT ai_query('stub', text, named_struct('max_tokens', 'lots')) FROM t").collect())
    assert(v.getMessage.contains("ai_query param max_tokens must be an integer, got 'lots'"))
    withConf(AiConfig.Concurrency -> "many") {
      val c = intercept[Exception](spark.sql("SELECT ai_query('stub', text) FROM t").collect())
      assert(c.getMessage.contains(s"${AiConfig.Concurrency} must be an integer, got 'many'"))
    }
  }

  test("helper functions are prompt templates over ai_query") {
    val (_, rows) = run(
      """SELECT ai_classify('I love it', array('positive', 'negative')),
        |       ai_translate('hello', 'German'),
        |       ai_gen('hi')""".stripMargin)
    val Seq(classify, translate, gen) = rows.head.toSeq.map(_.toString)
    assert(classify.startsWith("echo:Classify the following text into exactly one of these labels: positive, negative."))
    assert(classify.endsWith("Text: I love it"))
    assert(translate.contains("to German") && translate.endsWith("hello"))
    assert(gen == "echo:hi")
  }

  test("helper functions are named after the call, not the generated prompt") {
    val df = spark.sql("SELECT ai_summarize(text), ai_translate(text, 'German'), ai_query('stub', text) FROM t")
    assert(df.columns.toSeq == Seq("ai_summarize(text)", "ai_translate(text, German)", "ai_query(stub, text, NULL)"))
  }

  test("DataFrame API via expr") {
    import org.apache.spark.sql.functions.{col, expr}
    server.reset()
    val rows = spark.table("t").where(col("id") < 3).select(expr("ai_query('stub', text)")).collect()
    assert(rows.map(_.getString(0)).toSeq == (0 until 3).map(i => s"echo:prompt $i"))
  }

  test("without spark.sql.ai.partitions every input partition makes calls") {
    withConf(AiConfig.Concurrency -> "4") {
      val (df, rows) = run("SELECT ai_query('stub', text) FROM t4")
      assert(rows.length == 40)
      assert(aiShuffles(df).isEmpty)
      // 4 partitions, 2 running at once on local[2]
      assert(server.peakInFlight > 4, s"peak ${server.peakInFlight}")
    }
  }

  test("spark.sql.ai.partitions = 1 shuffles to a single partition before calling the model") {
    withConf(AiConfig.Concurrency -> "4", AiConfig.Partitions -> "1") {
      val (df, rows) = run("SELECT id, ai_query('stub', text) FROM t4 WHERE id % 2 = 0")
      assert(rows.map(r => (r.getLong(0), r.getString(1))).sortBy(_._1).toSeq ==
        (0L until 40L by 2).map(i => (i, s"echo:prompt $i")))
      assert(server.requests == 20)
      assert(server.peakInFlight == 4)
      assert(aiShuffles(df).map(_.outputPartitioning) == Seq(SinglePartition))
      assert(aiExecs(df).map(_.execute().getNumPartitions) == Seq(1))
    }
  }

  test("spark.sql.ai.partitions = 2 uses round robin") {
    withConf(AiConfig.Concurrency -> "4", AiConfig.Partitions -> "2") {
      val (df, rows) = run("SELECT ai_query('stub', text) FROM t4")
      assert(rows.length == 40)
      assert(aiShuffles(df).map(_.outputPartitioning) == Seq(RoundRobinPartitioning(2)))
      assert(server.peakInFlight <= 8)
    }
  }

  test("partitionMode = coalesce coalesces without a shuffle") {
    withConf(AiConfig.Concurrency -> "4", AiConfig.Partitions -> "1", AiConfig.PartitionMode -> "coalesce") {
      val (df, rows) = run("SELECT ai_query('stub', text) FROM t4")
      assert(rows.length == 40)
      assert(aiShuffles(df).isEmpty)
      assert(collect(df.queryExecution.executedPlan) { case c: CoalesceExec => c.numPartitions } == Seq(1))
      assert(server.peakInFlight == 4)
    }
  }

  test("invalid partitionMode is rejected") {
    withConf(AiConfig.Partitions -> "1", AiConfig.PartitionMode -> "bogus") {
      val e = intercept[Exception](run("SELECT ai_query('stub', text) FROM t4"))
      assert(e.getMessage.contains("must be shuffle or coalesce"))
    }
  }

  test("ORDER BY with partitions = 1 sorts a single partition and keeps the order") {
    for (mode <- Seq("shuffle", "coalesce")) {
      withConf(AiConfig.Partitions -> "1", AiConfig.PartitionMode -> mode) {
        val (df, rows) = run("SELECT id, ai_query('stub', text) FROM t4 ORDER BY id DESC")
        assert(rows.map(_.getLong(0)).toSeq == (39L to 0L by -1), mode)
        assert(rows.forall(r => r.getString(1) == s"echo:prompt ${r.getLong(0)}"), mode)
        assert(server.requests == 40, mode)
        // no range-partitioning shuffle (whose sampling would call the model twice)
        val shuffles = collect(df.queryExecution.executedPlan) { case e: ShuffleExchangeExec => e }
        assert(shuffles.forall(_.outputPartitioning == SinglePartition), mode)
      }
    }
  }

  test("ORDER BY with partitions = 2 range-partitions before calling the model") {
    withConf(AiConfig.Partitions -> "2") {
      val (df, rows) = run("SELECT id, ai_query('stub', text) FROM t4 ORDER BY id")
      assert(rows.map(_.getLong(0)).toSeq == (0L until 40L))
      assert(server.requests == 40)
      assert(aiShuffles(df).map(_.outputPartitioning).forall(_.isInstanceOf[RangePartitioning]))
      assert(aiShuffles(df).size == 1)
    }
  }

  test("ORDER BY ... LIMIT with partitions set adds no shuffle") {
    withConf(AiConfig.Partitions -> "1") {
      val (df, rows) = run("SELECT id, ai_query('stub', text) FROM t4 ORDER BY id DESC LIMIT 3")
      assert(rows.map(_.getLong(0)).toSeq == Seq(39L, 38L, 37L))
      assert(server.requests == 3)
      assert(aiShuffles(df).isEmpty)
    }
  }

  test("global ORDER BY on the model output calls the model once per row") {
    val (_, rows) = run("SELECT id, ai_query('stub', text) AS a FROM t4 ORDER BY a")
    assert(rows.map(_.getString(1)).toSeq == (0L until 40L).map(i => s"echo:prompt $i").sorted)
    assert(server.requests == 40)

    // Already a single partition: sorted in place, no extra shuffle.
    withConf(AiConfig.Partitions -> "1") {
      val (df, rows1) = run("SELECT id, ai_query('stub', text) AS a FROM t4 ORDER BY a DESC")
      assert(rows1.map(_.getString(1)).toSeq == rows.map(_.getString(1)).toSeq.reverse)
      assert(server.requests == 40)
      val shuffles = collect(df.queryExecution.executedPlan) { case e: ShuffleExchangeExec => e }
      assert(shuffles.map(_.outputPartitioning) == Seq(SinglePartition))
    }
  }

  test("nested calls share one redistribution") {
    withConf(AiConfig.Partitions -> "1") {
      val (df, rows) = run("SELECT ai_query('stub', ai_query('stub', text)) FROM t4")
      assert(rows.length == 40)
      assert(aiExecs(df).size == 2)
      assert(aiShuffles(df).size == 1)
    }
  }
}
