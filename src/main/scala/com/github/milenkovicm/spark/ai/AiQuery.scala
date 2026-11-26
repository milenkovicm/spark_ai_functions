package com.github.milenkovicm.spark.ai

import org.apache.spark.sql.catalyst.analysis.TypeCheckResult
import org.apache.spark.sql.catalyst.analysis.TypeCheckResult.{TypeCheckFailure, TypeCheckSuccess}
import org.apache.spark.sql.catalyst.expressions.{Expression, Unevaluable, UserDefinedExpression}
import org.apache.spark.sql.types.{DataType, NullType, StringType, StructType}

/**
 * A model call that [[AiStrategy]] pulls out of the plan and evaluates in [[AiQueryExec]]:
 * [[AiQuery]] (chat) or [[AiDecision]] (decision model). See [[AiQuery]] for why it is
 * `Unevaluable`, non-deterministic and a `UserDefinedExpression`.
 */
object AiExpression {
  /** Whether `e` contains a model call anywhere. */
  def hasAi(e: Expression): Boolean = e.exists(_.isInstanceOf[AiExpression])
}

trait AiExpression extends Expression with Unevaluable with UserDefinedExpression {
  override lazy val deterministic: Boolean = false

  override def nullable: Boolean = true

  /** Model name shown in the plan. */
  def modelName: String
}

/**
 * `ai_query(endpoint, request[, params])`: a placeholder that is never evaluated row by row.
 * [[AiStrategy]] pulls it out of the plan and evaluates it in [[AiQueryExec]].
 *
 *  - `Unevaluable`: if an occurrence ever escapes the strategy the query fails loudly instead of
 *    silently making one synchronous request per row. It also stops `ConvertToLocalRelation`
 *    from evaluating it on the driver.
 *  - non-deterministic: keeps the optimizer from duplicating the call (e.g. `CollapseProject`
 *    inlining it into several references) or moving filters across it.
 *  - `UserDefinedExpression`: lets the analyzer's `PullOutNondeterministic` move it out of
 *    GROUP BY / ORDER BY into a Project below, where [[AiStrategy]] can plan it.
 *
 * `params` is a foldable struct (e.g. `named_struct('temperature', 0.0, 'max_tokens', 20)`) or null.
 * `display` is set for the prompt-template helpers: the call as written, e.g. `ai_summarize(text)`,
 * used as the SQL text (and so the default column name) instead of the generated prompt.
 */
case class AiQuery(endpoint: Expression, request: Expression, params: Expression, display: Option[String] = None)
  extends AiExpression {

  override def children: Seq[Expression] = Seq(endpoint, request, params)

  override def dataType: DataType = StringType

  override def prettyName: String = "ai_query"

  override def name: String = prettyName

  override def sql: String = display.getOrElse(super.sql)

  override def toString: String = display.map(d => s"$d [model=$modelName]").getOrElse(super.toString)

  override def checkInputDataTypes(): TypeCheckResult = {
    if (!endpoint.foldable || !endpoint.dataType.isInstanceOf[StringType]) {
      TypeCheckFailure("ai_query: endpoint must be a constant string")
    } else if (endpoint.eval() == null) {
      TypeCheckFailure("ai_query: endpoint must not be null")
    } else if (!request.dataType.isInstanceOf[StringType]) {
      TypeCheckFailure(s"ai_query: request must be a string, got ${request.dataType.simpleString}")
    } else if (!params.foldable ||
        !(params.dataType == NullType || params.dataType.isInstanceOf[StructType])) {
      TypeCheckFailure("ai_query: params must be a constant struct, e.g. named_struct('temperature', 0.0)")
    } else if (unknownParams.nonEmpty) {
      TypeCheckFailure(s"ai_query: unknown params ${unknownParams.mkString(", ")}, " +
        s"expected ${AiQuery.Params.mkString(", ")}")
    } else {
      TypeCheckSuccess
    }
  }

  private def unknownParams: Seq[String] = params.dataType match {
    case st: StructType => st.fieldNames.toSeq.filterNot(AiQuery.Params.contains)
    case _ => Nil
  }

  /** Model name; only valid once resolved (it is foldable). */
  override def modelName: String = Option(endpoint.eval()).map(_.toString).orNull

  /** Model parameters as strings, keyed by struct field name. */
  def modelParams: Map[String, String] = params.dataType match {
    case st: StructType =>
      val row = params.eval().asInstanceOf[org.apache.spark.sql.catalyst.InternalRow]
      if (row == null) Map.empty
      else st.fields.zipWithIndex.collect {
        case (f, i) if !row.isNullAt(i) => f.name -> row.get(i, f.dataType).toString
      }.toMap
    case _ => Map.empty
  }

  override protected def withNewChildrenInternal(newChildren: IndexedSeq[Expression]): AiQuery =
    copy(endpoint = newChildren(0), request = newChildren(1), params = newChildren(2))
}

object AiQuery {
  /** Fields `params` may set. */
  val Params: Seq[String] = Seq("temperature", "max_tokens", "fail_on_error")
}
