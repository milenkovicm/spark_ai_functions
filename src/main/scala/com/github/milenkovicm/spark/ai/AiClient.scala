package com.github.milenkovicm.spark.ai

import java.time.Duration
import java.util.concurrent.{Callable, ConcurrentHashMap, ExecutorService, Executors, Future}
import java.util.concurrent.atomic.AtomicInteger

import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.openai.OpenAiChatModel

/** Result of one model call, carried back to the task thread so metrics are updated there. */
case class AiResponse(text: String, error: Throwable, nanos: Long)

/**
 * Executor-side model access through LangChain4j. `ChatModel.chat` blocks, so every call runs on
 * its own pooled daemon thread; the caller bounds how many are in flight, so the pool stays small
 * and idle threads expire after a minute.
 */
object AiClient {
  private val models = new ConcurrentHashMap[AiConfig, ChatModel]()

  private val threadIds = new AtomicInteger()

  private lazy val executor: ExecutorService = Executors.newCachedThreadPool { (r: Runnable) =>
    val t = new Thread(r, s"spark-ai-client-${threadIds.incrementAndGet()}")
    t.setDaemon(true)
    t
  }

  def model(config: AiConfig): ChatModel = models.computeIfAbsent(config, build)

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

  /** Submits `prompt`; the returned future never fails, errors are captured in [[AiResponse]]. */
  def submit(config: AiConfig, prompt: String): Future[AiResponse] =
    executor.submit(new Callable[AiResponse] {
      override def call(): AiResponse = {
        val start = System.nanoTime()
        try AiResponse(model(config).chat(prompt), null, System.nanoTime() - start)
        catch { case e: Throwable => AiResponse(null, e, System.nanoTime() - start) }
      }
    })
}
