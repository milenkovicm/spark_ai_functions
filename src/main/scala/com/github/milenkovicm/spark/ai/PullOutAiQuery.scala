package com.github.milenkovicm.spark.ai

import scala.collection.mutable

import org.apache.spark.sql.catalyst.expressions.{Alias, Expression, NamedExpression}
import org.apache.spark.sql.catalyst.expressions.aggregate.AggregateExpression
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, LogicalPlan, Project}
import org.apache.spark.sql.catalyst.rules.Rule

/**
 * Moves [[AiQuery]] out of an `Aggregate` so that it ends up in a `Project`, which [[AiStrategy]]
 * plans. (GROUP BY keys are already handled by Spark's `PullOutNondeterministic`.)
 *
 *  - inside aggregate functions, `count(ai_query(m, x))`: evaluated per input row in a Project
 *    below the Aggregate. Done at analysis, as an analyzer rule.
 *  - over the aggregation result, `ai_query(m, concat(g, count(*)))`: the Aggregate emits the
 *    grouping keys / aggregates it needs and a Project above evaluates the call. Done at
 *    planning time by [[AiStrategy]] via [[pullUp]], since `CollapseProject` would merge such a
 *    Project straight back into the Aggregate.
 */
object PullOutAiQuery extends Rule[LogicalPlan] {

  private def hasAi(e: Expression): Boolean = e.exists(_.isInstanceOf[AiQuery])

  /** Outermost calls only; nested ones travel with their parent. */
  private def outermost(e: Expression): Seq[AiQuery] = e match {
    case q: AiQuery => Seq(q)
    case other => other.children.flatMap(outermost)
  }

  override def apply(plan: LogicalPlan): LogicalPlan = plan.resolveOperatorsUp {
    case a: Aggregate if a.resolved && a.groupingExpressions.forall(_.deterministic) &&
        a.aggregateExpressions.exists(hasAi) =>
      pullDown(a)
  }

  private def pullDown(a: Aggregate): Aggregate = {
    val calls = a.aggregateExpressions.flatMap(_.collect {
      case ae: AggregateExpression => ae.children.flatMap(outermost)
    }.flatten).distinct
    if (calls.isEmpty) return a
    val aliases = calls.map(q => q -> Alias(q, "_ai_query")()).toMap[Expression, Alias]
    val newAggs = a.aggregateExpressions.map(_.transformDown {
      case ae: AggregateExpression => ae.transformDown {
        case q: AiQuery if aliases.contains(q) => aliases(q).toAttribute
      }
    }.asInstanceOf[NamedExpression])
    a.copy(
      aggregateExpressions = newAggs,
      child = Project(a.child.output ++ calls.map(aliases), a.child))
  }

  def pullUp(a: Aggregate): Project = {
    val emitted = mutable.LinkedHashMap[Expression, Alias]()
    def emit(e: Expression): Expression =
      emitted.getOrElseUpdate(e.canonicalized, Alias(e, "_agg")()).toAttribute
    val outer = a.aggregateExpressions.map {
      case ne if hasAi(ne) =>
        ne.transformDown {
          case ae: AggregateExpression => emit(ae)
          case g if !g.foldable && a.groupingExpressions.exists(_.semanticEquals(g)) => emit(g)
        }.asInstanceOf[NamedExpression]
      case ne => ne.toAttribute
    }
    val plain = a.aggregateExpressions.filterNot(hasAi)
    Project(outer, a.copy(aggregateExpressions = plain ++ emitted.values.toSeq))
  }
}
