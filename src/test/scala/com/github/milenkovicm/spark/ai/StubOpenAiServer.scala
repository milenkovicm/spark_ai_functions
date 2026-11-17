package com.github.milenkovicm.spark.ai

import java.net.InetSocketAddress
import java.nio.charset.StandardCharsets
import java.util.concurrent.{ConcurrentHashMap, Executors}
import java.util.concurrent.atomic.AtomicInteger

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.node.JsonNodeFactory
import com.sun.net.httpserver.{HttpExchange, HttpServer}

/**
 * Minimal OpenAI-compatible `/v1/chat/completions` server. It replies `echo:<last user message>`
 * after `delayMs`, and records how many requests were in flight at once.
 *
 * Prompts containing `FAIL` always get HTTP 500; prompts containing `FLAKY` get HTTP 500 the
 * first time they are seen.
 */
class StubOpenAiServer(delayMs: Long) extends AutoCloseable {
  private val mapper = new ObjectMapper()
  private val inFlight = new AtomicInteger()
  private val peak = new AtomicInteger()
  private val total = new AtomicInteger()
  private val seen = ConcurrentHashMap.newKeySet[String]()

  private val server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0)
  server.setExecutor(Executors.newVirtualThreadPerTaskExecutor())
  server.createContext("/v1/chat/completions", (ex: HttpExchange) => handle(ex))
  server.start()

  def baseUrl: String = s"http://127.0.0.1:${server.getAddress.getPort}/v1"
  def peakInFlight: Int = peak.get()
  def requests: Int = total.get()

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
