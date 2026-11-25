package com.github.milenkovicm.spark.ai

import java.time.Duration
import java.util.concurrent.{Callable, ConcurrentHashMap, ExecutorService, Executors, Future}
import java.util.concurrent.atomic.AtomicInteger

import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.decision.DecisionModel
import dev.langchain4j.model.decision.request.DecisionRequest
import dev.langchain4j.model.ollama.OllamaDecisionModel
import dev.langchain4j.model.openai.{OpenAiChatModel, OpenAiDecisionModel}

/**
 * Result of one model call, carried back to the task thread so metrics are updated there.
 * `value` is the reply text for chat calls and a `DecisionResponse` for decision calls.
 */
case class AiResponse(value: AnyRef, error: Throwable, nanos: Long)

/**
 * Executor-side model access through LangChain4j. `ChatModel.chat` and `DecisionModel.decide`
 * block, so every call runs on its own pooled daemon thread; the caller bounds how many are in
 * flight, so the pool stays small and idle threads expire after a minute.
 */
object AiClient {
  private val models = new ConcurrentHashMap[AiConfig, ChatModel]()
  private val decisionModels = new ConcurrentHashMap[DecisionConfig, DecisionModel]()

  private val threadIds = new AtomicInteger()

  private lazy val executor: ExecutorService = Executors.newCachedThreadPool { (r: Runnable) =>
    val t = new Thread(r, s"spark-ai-client-${threadIds.incrementAndGet()}")
    t.setDaemon(true)
    t
  }

  def model(config: AiConfig): ChatModel = models.computeIfAbsent(config, build)

  def decisionModel(config: DecisionConfig): DecisionModel = decisionModels.computeIfAbsent(config, buildDecision)

  private def build(c: AiConfig): ChatModel = {
    val b = OpenAiChatModel.builder()
      .baseUrl(c.baseUrl)
      .apiKey(c.apiKey)
      .modelName(c.model)
      .timeout(Duration.ofMillis(c.timeoutMs))
      .maxRetries(Int.box(c.maxRetries))
    c.temperature.foreach(t => b.temperature(Double.box(t)))
    c.maxTokens.foreach(t => b.maxTokens(Int.box(t)))
    b.build()
  }

  private def buildDecision(c: DecisionConfig): DecisionModel = c.api match {
    case AiConfig.SystemOne =>
      // OllamaDecisionModel appends /v1/systemone itself, so it takes the server root.
      OllamaDecisionModel.builder()
        .baseUrl(c.baseUrl.stripSuffix("/").stripSuffix("/v1"))
        .modelName(c.model)
        .timeout(Duration.ofMillis(c.timeoutMs))
        .maxRetries(Int.box(c.maxRetries))
        .customHeaders(headers(c, "Authorization", s"Bearer ${c.apiKey}"))
        .build()
    case _ =>
      OpenAiDecisionModel.builder()
        .baseUrl(c.baseUrl)
        .apiKey(c.apiKey)
        .customHeaders(headers(c, "X-API-Key", c.apiKey))
        .modelName(c.model)
        .timeout(Duration.ofMillis(c.timeoutMs))
        .maxRetries(Int.box(c.maxRetries))
        .build()
  }

  /** Sends the secret only when one is configured (the default placeholder is `none`). */
  private def headers(c: DecisionConfig, name: String, value: String): java.util.Map[String, String] =
    if (c.apiKey == null || c.apiKey == "none") java.util.Map.of() else java.util.Map.of(name, value)

  private def run(f: => AnyRef): Future[AiResponse] =
    executor.submit(new Callable[AiResponse] {
      override def call(): AiResponse = {
        val start = System.nanoTime()
        try AiResponse(f, null, System.nanoTime() - start)
        catch { case e: Throwable => AiResponse(null, e, System.nanoTime() - start) }
      }
    })

  /** Submits `prompt`; the returned future never fails, errors are captured in [[AiResponse]]. */
  def submit(config: AiConfig, prompt: String): Future[AiResponse] = run(model(config).chat(prompt))

  /** Asks `questions` (keyed by wire id) about `input` in one request; never fails, like [[submit]]. */
  def submitDecision(config: DecisionConfig, input: String, questions: Seq[(String, Question)]): Future[AiResponse] =
    run {
      val b = DecisionRequest.builder().input(input)
      questions.foreach { case (id, q) => b.question(id, q.toLangChain) }
      decisionModel(config).decide(b.build())
    }
}
