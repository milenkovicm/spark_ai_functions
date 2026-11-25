package com.github.milenkovicm.spark.ai

import scala.jdk.CollectionConverters._

import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.execution.adaptive.AdaptiveSparkPlanHelper
import org.apache.spark.sql.types.{BooleanType, DoubleType, StringType, StructType}

/**
 * Decision functions against [[StubOpenAiServer]], in both wire formats (Ollama `systemone` and
 * OpenAI `decisions`). See the stub for how it answers.
 */
class AiDecisionSuite extends SparkSessionSuite with AdaptiveSparkPlanHelper {
  private lazy val server = new StubOpenAiServer(10)

  private val apis = Seq(AiConfig.SystemOne, AiConfig.Decisions)

  override protected def beforeAll(): Unit = {
    super.beforeAll()
    spark.conf.set(AiConfig.BaseUrl, server.baseUrl)
    spark.conf.set(AiConfig.DefaultModel, "stub")
    spark.conf.set(AiConfig.DecisionModel, "stub-decision")
    spark.conf.set(AiConfig.MaxRetries, "0")
    spark.sql(
      """SELECT * FROM VALUES
        |  (1, 'I want a refund, the billing was wrong and I am annoyed'),
        |  (2, 'The app crashes, a technical bug, I am furious'),
        |  (3, 'Just saying thanks, other than that all calm'),
        |  (4, CAST(NULL AS STRING))
        |AS tickets(id, body)""".stripMargin).createOrReplaceTempView("tickets")
  }

  override protected def afterAll(): Unit = {
    try server.close()
    finally super.afterAll()
  }

  private def withConf[T](pairs: (String, String)*)(f: => T): T = {
    val old = pairs.map { case (k, _) => k -> spark.conf.getOption(k) }
    pairs.foreach { case (k, v) => spark.conf.set(k, v) }
    try f
    finally old.foreach {
      case (k, Some(v)) => spark.conf.set(k, v)
      case (k, None) => spark.conf.unset(k)
    }
  }

  private def run(sql: String): (DataFrame, Array[Row]) = {
    server.reset()
    val df = spark.sql(sql)
    (df, df.collect())
  }

  private def aiExecs(df: DataFrame): Seq[AiQueryExec] =
    collect(df.queryExecution.executedPlan) { case e: AiQueryExec => e }

  private def failure(sql: String): Throwable = intercept[Throwable](spark.sql(sql).collect())

  /** The full chain of messages, since errors surface wrapped in Spark exceptions. */
  private def messages(t: Throwable): String =
    Iterator.iterate(t)(_.getCause).takeWhile(_ != null).map(String.valueOf(_)).mkString(" | ")

  private def apiTest(name: String)(f: String => Unit): Unit =
    apis.foreach(api => test(s"[$api] $name")(withConf(AiConfig.DecisionApi -> api)(f(api))))

  apiTest("single-question functions return typed values, one request per row") { _ =>
    val (df, rows) = run(
      """SELECT id,
        |       ai_if(body, 'refund requested')                                AS refund,
        |       ai_probability(body, 'refund requested')                       AS p,
        |       ai_choose(body, array('billing', 'technical', 'other'))        AS team,
        |       ai_score(body, 'How frustrated?', array('calm', 'annoyed', 'furious')) AS frustration
        |FROM tickets ORDER BY id""".stripMargin)
    assert(df.schema.map(_.dataType) == Seq(df.schema.head.dataType, BooleanType, DoubleType, StringType, DoubleType))
    assert(rows.map(r => (r.getInt(0), r.get(1), r.get(2), r.get(3), r.get(4))).toSeq == Seq(
      (1, true, 0.9, "billing", 1.0),
      (2, false, 0.1, "technical", 2.0),
      (3, false, 0.1, "other", 0.0),
      (4, null, null, null, null)))
    // all four calls share the input: one request per non-null row
    assert(server.requests == 3)
    val execs = aiExecs(df)
    assert(execs.size == 1 && execs.head.requests.size == 1)
  }

  apiTest("ai_decide returns a struct with one field per question") { _ =>
    val (df, rows) = run(
      """SELECT id, d['team']['choice'] AS team, d['urgent']['probability'] AS urgent,
        |       d.mood.score AS mood, d.team.probabilities['billing'] AS p_billing, d.mood.probabilities['furious'] AS p_furious
        |FROM (SELECT id, ai_decide(body, '{
        |  "team":   {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "technical": null}},
        |  "urgent": {"type": "noul",   "instructions": "crashes make it urgent"},
        |  "mood":   {"type": "score",  "instructions": "How frustrated?", "criteria": ["calm", "annoyed", "furious"]}
        |}') AS d FROM tickets) ORDER BY id""".stripMargin)
    assert(rows.map(r => (r.getInt(0), r.get(1), r.get(2), r.get(3), r.get(4), r.get(5))).toSeq == Seq(
      (1, "billing", 0.1, 1.0, 0.8, 0.0),
      (2, "technical", 0.9, 2.0, 0.2, 1.0),
      (3, "billing", 0.1, 0.0, 0.8, 0.0),
      (4, null, null, null, null, null)))
    assert(server.requests == 3)
    val inner = spark.sql(
      """SELECT ai_decide(body, '{"a": {"type": "predicate", "instructions": "x"}}') AS d FROM tickets""")
    assert(inner.schema.head.dataType == StructType.fromDDL("a struct<probability: double>"))
    assert(df.schema("team").dataType == StringType)
    // fields and choice labels keep the order of the JSON
    val ordered = spark.sql(
      """SELECT ai_decide(body, '{"z": {"type": "noul", "instructions": "x"}, "a": {"type": "noul", "instructions": "y"},
        |  "m": {"type": "noul", "instructions": "w"}}') AS d FROM tickets""".stripMargin)
    assert(ordered.schema.head.dataType.asInstanceOf[StructType].fieldNames.toSeq == Seq("z", "a", "m"))
  }

  apiTest("calls on different inputs are separate requests") { _ =>
    val (_, rows) = run(
      "SELECT ai_if(body, 'refund'), ai_if(concat(body, '!'), 'refund') FROM tickets WHERE id <= 2")
    assert(rows.map(r => (r.get(0), r.get(1))).toSeq == Seq((true, true), (false, false)))
    assert(server.requests == 4)
  }

  apiTest("plain WHERE predicates are applied before calling the model") { _ =>
    val (_, rows) = run("SELECT id FROM tickets WHERE id <= 2 AND ai_if(body, 'refund requested')")
    assert(rows.map(_.getInt(0)).toSeq == Seq(1))
    assert(server.requests == 2)
  }

  apiTest("GROUP BY and ORDER BY decision results") { _ =>
    val (_, grouped) = run(
      "SELECT ai_choose(body, array('billing', 'technical')) AS team, count(*) FROM tickets WHERE id <= 3 GROUP BY 1")
    assert(grouped.map(r => r.getString(0) -> r.getLong(1)).toMap == Map("billing" -> 2L, "technical" -> 1L))
    val (_, ordered) = run(
      "SELECT id FROM tickets WHERE id <= 3 ORDER BY ai_score(body, 'frustration', array('calm', 'annoyed', 'furious')) DESC LIMIT 2")
    assert(ordered.map(_.getInt(0)).toSeq == Seq(2, 1))
  }

  apiTest("model and on_error named arguments") { api =>
    run("SELECT ai_if(body, 'refund', model => 'other-model') FROM tickets WHERE id = 1")
    assert(server.lastDecisionRequest.get("model").asText() == "other-model")
    val (_, rows) = run(
      "SELECT ai_if(body, 'x', on_error => 'null') FROM VALUES ('please FAIL') AS t(body)")
    assert(rows.head.isNullAt(0))
    assert(messages(failure("SELECT ai_if(body, 'x') FROM VALUES ('please FAIL') AS t(body)"))
      .contains("decision request to model 'stub-decision' failed"))
    assert(messages(failure("SELECT ai_if(body, 'x', on_error => 'fail') FROM VALUES ('FAIL') AS t(body)"))
      .contains("failed"))
  }

  apiTest("transient 5xx errors are retried") { _ =>
    withConf(AiConfig.MaxRetries -> "1") {
      val (_, rows) = run("SELECT ai_if(body, 'flaky') FROM VALUES ('a FLAKY refund') AS t(body)")
      assert(rows.head.get(0) == true)
      assert(server.requests == 2)
    }
  }

  apiTest("struct input is sent as JSON, other types as text") { _ =>
    val (_, rows) = run(
      "SELECT ai_if(named_struct('subject', 'refund', 'n', 1), 'refund'), ai_if(42, '42') FROM VALUES (1) AS t(x)")
    assert(rows.head.get(0) == true && rows.head.get(1) == true)
  }

  apiTest("decision calls over chat output are nested execs") { _ =>
    val (df, rows) = run("SELECT ai_if(ai_gen(body), 'echo') FROM tickets WHERE id = 3")
    assert(rows.head.get(0) == true)
    assert(aiExecs(df).size == 2)
    assert(server.requests == 2)
  }

  apiTest("plan shows one merged decision request") { _ =>
    val df = spark.sql("SELECT ai_if(body, 'a'), ai_score(body, 'b', array('x', 'y')) FROM tickets")
    val execs = aiExecs(df)
    assert(execs.size == 1)
    val exec = execs.head
    assert(exec.requests.map(_.label) == Seq("decision:stub-decision"))
    assert(exec.simpleString(10).contains("ai_if_0"))
  }

  test("[systemone] wire format") {
    withConf(AiConfig.DecisionApi -> AiConfig.SystemOne) {
      run(
        """SELECT ai_if(body, 'refund'), ai_choose(body, '{"billing": "Payments", "technical": null}')
          |FROM tickets WHERE id = 1""".stripMargin)
      val body = server.lastDecisionRequest
      assert(body.get("state").asText().startsWith("I want a refund"))
      val qs = body.get("questions")
      assert(qs.isObject && qs.properties().asScala.map(_.getKey).toSet == Set("q0", "q1"))
      assert(qs.get("q0").get("type").asText() == "noul")
      assert(qs.get("q1").get("criteria").get("billing").asText() == "Payments")
    }
  }

  test("[decisions] wire format") {
    withConf(AiConfig.DecisionApi -> AiConfig.Decisions) {
      run("SELECT ai_if(body, 'refund'), ai_score(body, 'mood', array('calm', 'furious')) FROM tickets WHERE id = 1")
      val body = server.lastDecisionRequest
      assert(body.get("input").asText().startsWith("I want a refund"))
      val qs = body.get("questions").elements().asScala.toSeq
      assert(qs.map(_.get("type").asText()) == Seq("predicate", "score"))
      assert(qs.map(_.get("name").asText()) == Seq("q0", "q1"))
      assert(qs(1).get("levels").elements().asScala.map(_.get("label").asText()).toSeq == Seq("calm", "furious"))
    }
  }

  test("[decisions] refusals are null under on_error => 'null' and fail otherwise") {
    withConf(AiConfig.DecisionApi -> AiConfig.Decisions) {
      val (_, rows) = run(
        """SELECT ai_if(body, 'x', on_error => 'null'),
          |       ai_decide(body, '{"a": {"type": "noul", "instructions": "x"}}', on_error => 'null')
          |FROM VALUES ('REFUSE this') AS t(body)""".stripMargin)
      assert(rows.head.isNullAt(0))
      assert(rows.head.getStruct(1).isNullAt(0))
      assert(messages(failure("SELECT ai_if(body, 'x') FROM VALUES ('REFUSE') AS t(body)")).contains("refused"))
    }
  }

  test("[systemone] more questions than one request allows are split") {
    withConf(AiConfig.DecisionApi -> AiConfig.SystemOne) {
      val calls = (0 until 70).map(i => s"ai_probability(body, 'condition $i')").mkString(", ")
      val (_, rows) = run(s"SELECT $calls FROM tickets WHERE id <= 3")
      assert(rows.length == 3 && rows.head.length == 70)
      assert(server.requests == 6)
    }
  }

  test("arguments are validated at analysis") {
    def error(sql: String): String = messages(intercept[Throwable](spark.sql(sql)))
    assert(error("SELECT ai_choose(body, array('only')) FROM tickets").contains("2 to 255 labels"))
    assert(error("SELECT ai_choose(body, array('a', 'a')) FROM tickets").contains("distinct"))
    assert(error("SELECT ai_choose(body, array(body, 'a')) FROM tickets").contains("constant"))
    assert(error("SELECT ai_score(body, 'x', array('1','2','3','4','5','6','7','8','9','10','11')) FROM tickets")
      .contains("2 to 10 levels"))
    assert(error("SELECT ai_if(body, 'x', modle => 'm') FROM tickets").contains("unknown argument 'modle'"))
    assert(error("SELECT ai_if(body, 'x', on_error => 'skip') FROM tickets").contains("on_error must be"))
    assert(error("SELECT ai_if(body, body) FROM tickets").contains("constant string"))
    assert(error("SELECT ai_if(body) FROM tickets").contains("2 positional argument(s)"))
    assert(error("SELECT ai_decide(body, '{\"a\": {\"type\": \"maybe\"}}') FROM tickets").contains("type must be"))
    assert(error("SELECT ai_decide(body, 'not json') FROM tickets").contains("invalid JSON"))
    assert(error("SELECT ai_if(cast(body AS binary), 'x') FROM tickets").contains("binary"))
    withConf(AiConfig.DecisionModel -> "") {
      assert(error("SELECT ai_if(body, 'x') FROM tickets").contains(AiConfig.DecisionModel))
    }
    withConf(AiConfig.DecisionApi -> "grpc") {
      assert(messages(failure("SELECT ai_if(body, 'x') FROM tickets")).contains("must be systemone or decisions"))
    }
  }

  test("the existing chat-based ai_classify is unchanged") {
    val (_, rows) = run("SELECT ai_classify(body, array('a', 'b')) FROM tickets WHERE id = 1")
    assert(rows.head.getString(0).startsWith("echo:Classify"))
  }
}
