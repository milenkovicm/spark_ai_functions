package com.github.milenkovicm.spark.ai

import org.apache.spark.sql.catalyst.analysis.TypeCheckResult
import org.apache.spark.sql.catalyst.analysis.TypeCheckResult.{TypeCheckFailure, TypeCheckSuccess}
import org.apache.spark.sql.catalyst.expressions.{Expression, Unevaluable, UserDefinedExpression}
import org.apache.spark.sql.types.{DataType, NullType, StringType, StructType}

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
 */
case class AiQuery(endpoint: Expression, request: Expression, params: Expression)
  extends Expression with Unevaluable with UserDefinedExpression {

  override def children: Seq[Expression] = Seq(endpoint, request, params)

  override def dataType: DataType = StringType

  override def nullable: Boolean = true

  override lazy val deterministic: Boolean = false

  override def prettyName: String = "ai_query"

  override def name: String = prettyName

  override def checkInputDataTypes(): TypeCheckResult = {
    if (!endpoint.foldable || !endpoint.dataType.isInstanceOf[StringType]) {
      TypeCheckFailure("ai_query: endpoint must be a constant string")
    } else if (!request.dataType.isInstanceOf[StringType]) {
      TypeCheckFailure(s"ai_query: request must be a string, got ${request.dataType.simpleString}")
    } else if (!params.foldable ||
        !(params.dataType == NullType || params.dataType.isInstanceOf[StructType])) {
      TypeCheckFailure("ai_query: params must be a constant struct, e.g. named_struct('temperature', 0.0)")
    } else {
      TypeCheckSuccess
    }
  }

  /** Model name; only valid once resolved (it is foldable). */
  def modelName: String = Option(endpoint.eval()).map(_.toString).orNull

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
