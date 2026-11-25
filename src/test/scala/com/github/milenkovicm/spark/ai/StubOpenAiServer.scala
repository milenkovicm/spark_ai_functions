package com.github.milenkovicm.spark.ai

import java.net.InetSocketAddress
import java.nio.charset.StandardCharsets
import java.util.concurrent.{ConcurrentHashMap, Executors}
import java.util.concurrent.atomic.AtomicInteger

import scala.jdk.CollectionConverters._

import com.fasterxml.jackson.databind.{JsonNode, ObjectMapper}
import com.fasterxml.jackson.databind.node.{JsonNodeFactory, ObjectNode}
import com.sun.net.httpserver.{HttpExchange, HttpServer}

/**
 * Minimal OpenAI-compatible `/v1/chat/completions` server. It replies `echo:<last user message>`
 * after `delayMs`, and records how many requests were in flight at once.
 *
 * Prompts containing `FAIL` always get HTTP 500; prompts containing `FLAKY` get HTTP 500 the
 * first time they are seen.
 *
 * It also serves decision models, in both wire formats: Ollama's `/v1/systemone` (`state`, keyed
 * questions) and OpenAI's `/v1/decisions` (`input`, question array). Answers are deterministic
 * over the lower-cased input:
 *  - yes/no: 0.9 if the input contains the first word of the instructions, else 0.1
 *  - choice: the first label the input contains, else the first label
 *  - score: the index of the first level the input contains, else 0
 * On `/v1/decisions`, an input containing `REFUSE` gets a refusal for every question.
 */
class StubOpenAiServer(delayMs: Long) extends AutoCloseable {
  private val mapper = new ObjectMapper()
  private val inFlight = new AtomicInteger()
  private val peak = new AtomicInteger()
  private val total = new AtomicInteger()
  private val seen = ConcurrentHashMap.newKeySet[String]()
  @volatile private var last: JsonNode = _

  private val server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0)
  server.setExecutor(Executors.newCachedThreadPool { (r: Runnable) =>
    val t = new Thread(r, "stub-openai-server")
    t.setDaemon(true)
    t
  })
  server.createContext("/v1/chat/completions", (ex: HttpExchange) => handle(ex))
  server.createContext("/v1/systemone", (ex: HttpExchange) => handleDecision(ex, systemOne))
  server.createContext("/v1/decisions", (ex: HttpExchange) => handleDecision(ex, decisions))
  server.start()

  def baseUrl: String = s"http://127.0.0.1:${server.getAddress.getPort}/v1"
  def peakInFlight: Int = peak.get()
  def requests: Int = total.get()
  /** Body of the most recent decision request. */
  def lastDecisionRequest: JsonNode = last

  def reset(): Unit = {
    peak.set(0)
    total.set(0)
    seen.clear()
  }

  private def handle(ex: HttpExchange): Unit = {
    val now = inFlight.incrementAndGet()
    peak.accumulateAndGet(now, Math.max)
    total.incrementAndGet()
    try {
      val body = mapper.readTree(ex.getRequestBody)
      val messages = body.get("messages")
      val prompt = messages.get(messages.size() - 1).get("content").asText()
      Thread.sleep(delayMs)
      if (prompt.contains("FAIL") || (prompt.contains("FLAKY") && seen.add(prompt))) {
        respond(ex, 500, """{"error":{"message":"stub failure","type":"server_error"}}""")
      } else {
        respond(ex, 200, completion(body.get("model").asText(), "echo:" + prompt))
      }
    } finally {
      inFlight.decrementAndGet()
    }
  }

  private def handleDecision(ex: HttpExchange, answer: JsonNode => String): Unit = {
    val now = inFlight.incrementAndGet()
    peak.accumulateAndGet(now, Math.max)
    total.incrementAndGet()
    try {
      val body = mapper.readTree(ex.getRequestBody)
      last = body
      val input = Option(body.get("state")).orElse(Option(body.get("input"))).map(_.asText()).getOrElse("")
      Thread.sleep(delayMs)
      if (input.contains("FAIL") || (input.contains("FLAKY") && seen.add(input))) {
        respond(ex, 500, """{"error":{"message":"stub failure","type":"server_error"}}""")
      } else {
        respond(ex, 200, answer(body))
      }
    } finally {
      inFlight.decrementAndGet()
    }
  }

  private def yes(input: String, instructions: String): Double =
    if (input.contains(instructions.toLowerCase.split("\\s+").head)) 0.9 else 0.1

  private def pick(input: String, labels: Seq[String]): Int =
    labels.indexWhere(l => input.contains(l.toLowerCase.split(":").head.trim)).max(0)

  private def systemOne(body: JsonNode): String = {
    val input = body.get("state").asText().toLowerCase
    val root = mapper.createObjectNode().put("model", body.get("model").asText())
    val answers = root.putObject("answers")
    body.get("questions").properties().asScala.foreach { e =>
      val q = e.getValue
      val a = answers.putObject(e.getKey).put("type", q.get("type").asText())
      q.get("type").asText() match {
        case "noul" => a.put("noul", yes(input, q.get("instructions").asText()))
        case "choice" =>
          val labels = q.get("criteria").properties().asScala.toSeq.map(_.getKey)
          val i = pick(input, labels)
          a.put("choice", labels(i)).put("confidence", 0.5)
          val p = a.putObject("probabilities")
          labels.zipWithIndex.foreach { case (l, j) => p.put(l, if (j == i) 0.8 else 0.2 / (labels.size - 1)) }
        case "score" =>
          val levels = q.get("criteria").elements().asScala.map(_.asText()).toSeq
          val i = pick(input, levels)
          a.put("score", i.toDouble).put("confidence", 1.0)
          val p = a.putObject("probabilities")
          levels.indices.foreach(j => p.put(j.toString, if (j == i) 1.0 else 0.0))
      }
    }
    root.putObject("usage").put("input_tokens", 1).put("output_tokens", 0)
    mapper.writeValueAsString(root)
  }

  private def decisions(body: JsonNode): String = {
    val raw = body.get("input").asText()
    val input = raw.toLowerCase
    val root = mapper.createObjectNode().put("model", body.get("model").asText())
    val answers = root.putArray("answers")
    body.get("questions").elements().asScala.foreach { q =>
      val a: ObjectNode = answers.addObject().put("name", q.get("name").asText())
      if (raw.contains("REFUSE")) {
        a.put("type", "refusal")
      } else {
        a.put("type", q.get("type").asText())
        q.get("type").asText() match {
          case "predicate" => a.put("probability", yes(input, q.get("instructions").asText()))
          case "choice" =>
            val labels = q.get("choices").elements().asScala.map(_.get("value").asText()).toSeq
            val i = pick(input, labels)
            a.put("choice", labels(i)).put("confidence", 0.5)
            val p = a.putArray("probabilities")
            labels.zipWithIndex.foreach { case (l, j) =>
              p.addObject().put("value", l).put("probability", if (j == i) 0.8 else 0.2 / (labels.size - 1))
            }
          case "score" =>
            val levels = q.get("levels").elements().asScala.map(_.get("label").asText()).toSeq
            val i = pick(input, levels)
            a.put("score", i.toDouble).put("confidence", 1.0)
            val p = a.putArray("probabilities")
            levels.zipWithIndex.foreach { case (l, j) =>
              p.addObject().put("value", j).put("label", l).put("probability", if (j == i) 1.0 else 0.0)
            }
        }
      }
    }
    mapper.writeValueAsString(root)
  }

  private def completion(model: String, text: String): String = {
    val f = JsonNodeFactory.instance
    val root = f.objectNode()
    root.put("id", "chatcmpl-stub").put("object", "chat.completion").put("created", 0L).put("model", model)
    val choice = root.putArray("choices").addObject()
    choice.put("index", 0).put("finish_reason", "stop")
    choice.putObject("message").put("role", "assistant").put("content", text)
    root.putObject("usage").put("prompt_tokens", 1).put("completion_tokens", 1).put("total_tokens", 2)
    mapper.writeValueAsString(root)
  }

  private def respond(ex: HttpExchange, status: Int, json: String): Unit = {
    val bytes = json.getBytes(StandardCharsets.UTF_8)
    ex.getResponseHeaders.add("Content-Type", "application/json")
    ex.sendResponseHeaders(status, bytes.length)
    ex.getResponseBody.write(bytes)
    ex.close()
  }

  override def close(): Unit = server.stop(0)
}
