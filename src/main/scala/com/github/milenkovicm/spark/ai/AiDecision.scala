package com.github.milenkovicm.spark.ai

import org.apache.spark.sql.catalyst.analysis.TypeCheckResult
import org.apache.spark.sql.catalyst.analysis.TypeCheckResult.{TypeCheckFailure, TypeCheckSuccess}
import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.types.{DataType, StringType}

/**
 * A decision-model call (`ai_probability`, `ai_if`, `ai_choose`, `ai_score`, `ai_decide`): asks
 * `questions` about `input` and converts the answers according to `output`.
 *
 * Everything except `input` is a constant, parsed and validated when the function is built.
 * Calls in one [[AiQueryExec]] with the same input, model and error mode share one request per row.
 *
 * @param input   the text the questions are about (struct/map/array inputs are already JSON)
 * @param onError `Some(false)` for `on_error => 'null'`, `Some(true)` for `'fail'`, else session default
 * @param display the call as written, e.g. `ai_if(body, 'refund requested')`; used as its SQL text,
 *                and so as the default column name
 */
case class AiDecision(
    function: String,
    input: Expression,
    questions: Seq[(String, Question)],
    output: DecisionOutput,
    model: String,
    onError: Option[Boolean],
    display: String)
  extends AiExpression {

  override def children: Seq[Expression] = Seq(input)

  override def dataType: DataType = output.dataType(questions)

  override def prettyName: String = function

  override def name: String = prettyName

  override def modelName: String = model

  override def checkInputDataTypes(): TypeCheckResult =
    if (input.dataType.isInstanceOf[StringType]) TypeCheckSuccess
    else TypeCheckFailure(s"$function: input must be a string, got ${input.dataType.simpleString}")

  override def toString: String = s"$display [model=$model]"

  override def sql: String = display

  override protected def withNewChildrenInternal(newChildren: IndexedSeq[Expression]): AiDecision =
    copy(input = newChildren.head)
}
