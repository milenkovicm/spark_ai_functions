package com.github.milenkovicm.spark.ai

import java.util.concurrent.Future

import dev.langchain4j.model.decision.response.DecisionResponse

import org.apache.spark.sql.catalyst.expressions.{Expression, GenericInternalRow}
import org.apache.spark.unsafe.types.UTF8String

/** One `ai_query` call site, fully resolved on the driver. */
case class AiCall(config: AiConfig, failOnError: Boolean)

/**
 * One request per input row, resolved on the driver. It sends the string `input` and fills one
 * or more result columns of [[AiQueryExec]] (indices into its result attributes).
 */
sealed trait AiRequest extends Serializable {
  /** String expression over the exec's input, sent to the model; null means no request. */
  def input: Expression
  def columns: Seq[Int]
  def failOnError: Boolean
  /** Shown in the plan. */
  def label: String
  /** Used in error messages. */
  def describe: String
  def submit(input: String): Future[AiResponse]
  /** Writes the successful response's values into `out`. */
  def complete(value: AnyRef, out: GenericInternalRow): Unit
}

/** A chat call (`ai_query` and its prompt-template helpers): one column with the reply text. */
case class ChatRequest(input: Expression, call: AiCall, column: Int) extends AiRequest {
  override def columns: Seq[Int] = Seq(column)
  override def failOnError: Boolean = call.failOnError
  override def label: String = call.config.model
  override def describe: String = s"ai_query to model '${call.config.model}'"
  override def submit(input: String): Future[AiResponse] = AiClient.submit(call.config, input)
  override def complete(value: AnyRef, out: GenericInternalRow): Unit =
    out.update(column, UTF8String.fromString(value.asInstanceOf[String]))
}

/** One decision function call merged into a [[DecisionRequest]]. */
case class DecisionMember(
    column: Int,
    output: DecisionOutput,
    questions: Seq[(String, Question)],
    wireIds: Seq[String])

/**
 * Decision calls on the same input, model and error mode, answered by a single request whose
 * questions are the members' questions under unique wire ids.
 */
case class DecisionRequest(
    input: Expression,
    config: DecisionConfig,
    failOnError: Boolean,
    members: Seq[DecisionMember])
  extends AiRequest {

  override def columns: Seq[Int] = members.map(_.column)
  override def label: String = s"decision:${config.model}"
  override def describe: String = s"decision request to model '${config.model}'"

  override def submit(input: String): Future[AiResponse] =
    AiClient.submitDecision(config, input, members.flatMap(m => m.wireIds.zip(m.questions.map(_._2))))

  override def complete(value: AnyRef, out: GenericInternalRow): Unit = {
    val response = value.asInstanceOf[DecisionResponse]
    members.foreach { m =>
      out.update(m.column, Decision.value(m.output, m.questions, m.wireIds, response, failOnError))
    }
  }
}
