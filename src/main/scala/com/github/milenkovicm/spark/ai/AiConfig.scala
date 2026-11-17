package com.github.milenkovicm.spark.ai

import org.apache.spark.sql.internal.SQLConf

/**
 * Settings for one model endpoint. Resolved on the driver from the session conf (plus per-call
 * model parameters) and shipped to executors, where it is also the cache key for the client.
 */
case class AiConfig(
    baseUrl: String,
    apiKey: String,
    model: String,
    temperature: Option[Double],
    maxTokens: Option[Int],
    timeoutMs: Long,
    maxRetries: Int)

/**
 * Optional redistribution of the rows sent to the model: only `partitions` tasks make AI calls,
 * either after a shuffle or by coalescing (no shuffle, but upstream work runs in those tasks too).
 */
case class AiDistribution(partitions: Int, coalesce: Boolean)

object AiConfig {
  val Prefix = "spark.sql.ai."
  val BaseUrl = Prefix + "baseUrl"
  val ApiKey = Prefix + "apiKey"
  val DefaultModel = Prefix + "defaultModel"
  val TimeoutMs = Prefix + "timeoutMs"
  val MaxRetries = Prefix + "maxRetries"
  val Temperature = Prefix + "temperature"
  val MaxTokens = Prefix + "maxTokens"
  val Concurrency = Prefix + "concurrency"
  val FailOnError = Prefix + "failOnError"
  val Partitions = Prefix + "partitions"
  val PartitionMode = Prefix + "partitionMode"

  private def get(conf: SQLConf, key: String): Option[String] =
    Option(conf.getConfString(key, null)).map(_.trim).filter(_.nonEmpty)

  /** Builds the endpoint config for `model`, letting per-call `params` override session defaults. */
  def resolve(conf: SQLConf, model: String, params: Map[String, String]): AiConfig = AiConfig(
    baseUrl = get(conf, BaseUrl).getOrElse("http://127.0.0.1:11434/v1"),
    apiKey = get(conf, ApiKey).orElse(sys.env.get("OPENAI_API_KEY")).getOrElse("none"),
    model = model,
    temperature = params.get("temperature").orElse(get(conf, Temperature)).map(_.toDouble),
    maxTokens = params.get("max_tokens").orElse(get(conf, MaxTokens)).map(_.toInt),
    timeoutMs = get(conf, TimeoutMs).map(_.toLong).getOrElse(120000L),
    maxRetries = get(conf, MaxRetries).map(_.toInt).getOrElse(2))

  def concurrency(conf: SQLConf): Int = get(conf, Concurrency).map(_.toInt).getOrElse(16).max(1)

  def failOnError(conf: SQLConf, params: Map[String, String]): Boolean =
    params.get("fail_on_error").orElse(get(conf, FailOnError)).forall(_.toBoolean)

  def distribution(conf: SQLConf): Option[AiDistribution] = get(conf, Partitions).map { p =>
    val partitions = p.toInt
    require(partitions >= 1, s"$Partitions must be at least 1, got $partitions")
    val coalesce = get(conf, PartitionMode).getOrElse("shuffle").toLowerCase match {
      case "shuffle" => false
      case "coalesce" => true
      case other => throw new IllegalArgumentException(s"$PartitionMode must be shuffle or coalesce, got $other")
    }
    AiDistribution(partitions, coalesce)
  }

  def defaultModel(conf: SQLConf): Option[String] = get(conf, DefaultModel)
}
