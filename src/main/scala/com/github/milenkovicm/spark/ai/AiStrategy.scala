package com.github.milenkovicm.spark.ai

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.expressions.{And, AttributeReference, Expression, IntegerLiteral, NamedExpression, PredicateHelper, SortOrder}
import org.apache.spark.sql.catalyst.plans.logical.{Aggregate, Filter, Limit, LogicalPlan, Project, ReturnAnswer, Sort}
import org.apache.spark.sql.catalyst.plans.physical.{RangePartitioning, RoundRobinPartitioning, SinglePartition, UnknownPartitioning}
import org.apache.spark.sql.execution.{CoalesceExec, FilterExec, ProjectExec, SortExec, SparkPlan, SparkStrategy, TakeOrderedAndProjectExec}
import org.apache.spark.sql.execution.exchange.{REPARTITION_BY_NUM, ShuffleExchangeExec}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.StringType

/**
 * Plans `Project`/`Filter`/`Aggregate` nodes containing [[AiQuery]] as `AiQueryExec` + `ProjectExec`/`FilterExec`,
 * in the same spirit as `ExtractPythonUDFs` + `BatchEvalPythonExec`. Runs before Spark's own
 * strategies, so it sees these nodes before e.g. `FileSourceStrategy` folds them into a scan.
 */
case class AiStrategy(session: SparkSession) extends SparkStrategy with PredicateHelper {

  private def conf: SQLConf = session.sessionState.conf

  private def hasAi(e: Expression): Boolean = e.exists(_.isInstanceOf[AiQuery])

  /**
   * A projection containing ai_query: a `Project`, or an `Aggregate` whose output calls ai_query
   * over grouping keys / aggregates (split into `Project` over `Aggregate`, see [[PullOutAiQuery]]).
   */
  private object AiProject {
    def unapply(plan: LogicalPlan): Option[(Seq[NamedExpression], LogicalPlan)] = plan match {
      case Project(list, child) if list.exists(hasAi) => Some((list, child))
      case a: Aggregate if a.aggregateExpressions.exists(hasAi) =>
        val p = PullOutAiQuery.pullUp(a)
        Some((p.projectList, p.child))
      case _ => None
    }
  }

  /** Sort keys that can be evaluated before the projection, i.e. don't depend on model output. */
  private def sortableBelow(order: Seq[SortOrder], child: LogicalPlan): Boolean =
    order.forall(_.references.subsetOf(child.outputSet))

  /**
   * `ORDER BY ... LIMIT n` over a projection with ai_query, where the sort keys don't depend on
   * the model output. Spark would plan it as TakeOrderedAndProjectExec(projectList = ai_query(...))
   * or sort after calling the model for every row; we take the top n first, then call the model.
   */
  private object TopK {
    def unapply(plan: LogicalPlan): Option[(Int, Seq[SortOrder], Seq[NamedExpression], LogicalPlan)] =
      plan match {
        case Limit(IntegerLiteral(limit), Project(list, Sort(order, true, child, _)))
            if list.exists(hasAi) =>
          Some((limit, order, list, child))
        case Limit(IntegerLiteral(limit), Sort(order, true, AiProject(list, child), _))
            if sortableBelow(order, child) =>
          Some((limit, order, list, child))
        case _ => None
      }
  }

  override def apply(plan: LogicalPlan): Seq[SparkPlan] = plan match {
    case ReturnAnswer(TopK(limit, order, list, child)) =>
      project(list, TakeOrderedAndProjectExec(limit, order, child.output, planLater(child), 0)) :: Nil

    case TopK(limit, order, list, child) =>
      project(list, TakeOrderedAndProjectExec(limit, order, child.output, planLater(child), 0)) :: Nil

    // A global sort samples its input in a separate job, which would call the model twice per row.
    // Sort first when the keys allow it; the projection preserves the order.
    case Sort(order, global, AiProject(list, child), hint) if sortableBelow(order, child) =>
      val sorted = AiConfig.distribution(conf) match {
        case None => planLater(Sort(order, global, child, hint))
        // Redistributing after the sort would lose the order, so the sort's own partitioning does it.
        case Some(AiDistribution(1, true)) if global =>
          SortExec(order, global = true, CoalesceExec(1, planLater(child)))
        case Some(AiDistribution(n, _)) if global =>
          val partitioning = if (n == 1) SinglePartition else RangePartitioning(order, n)
          SortExec(order, global = true, ShuffleExchangeExec(partitioning, planLater(child), REPARTITION_BY_NUM))
        case Some(_) =>
          SortExec(order, global = false, distribute(planLater(child)))
      }
      project(list, sorted) :: Nil

    case AiProject(list, child) =>
      project(list, planLater(child)) :: Nil

    case Filter(condition, child) if hasAi(condition) =>
      // ai_query has no side effects, so cheap predicates can run first and save requests.
      val (withAi, plain) = splitConjunctivePredicates(condition).partition(hasAi)
      val input = plain.reduceOption(And).map(FilterExec(_, planLater(child))).getOrElse(planLater(child))
      val (exec, rewritten) = extract(withAi, input)
      ProjectExec(child.output, FilterExec(rewritten.reduce(And), exec)) :: Nil

    case p if p.expressions.exists(hasAi) =>
      throw new UnsupportedOperationException(
        s"ai_query is not supported in ${p.nodeName}: ${p.simpleString(5)}")

    case _ => Nil
  }

  private def project(list: Seq[NamedExpression], child: SparkPlan): SparkPlan = {
    val (exec, rewritten) = extract(list, child)
    ProjectExec(rewritten.map(_.asInstanceOf[NamedExpression]), exec)
  }

  /**
   * Applies `spark.sql.ai.partitions`: only that many tasks will make AI calls. Skipped when
   * `child` is already known to have no more partitions (e.g. a top-K or an already distributed
   * sort); a not-yet-planned child reports unknown partitioning and is always redistributed.
   */
  private def distribute(child: SparkPlan): SparkPlan = AiConfig.distribution(conf) match {
    case Some(AiDistribution(n, coalesce)) =>
      child.outputPartitioning match {
        case p if !p.isInstanceOf[UnknownPartitioning] && p.numPartitions <= n => child
        case _ if coalesce => CoalesceExec(n, child)
        case _ =>
          val partitioning = if (n == 1) SinglePartition else RoundRobinPartitioning(n)
          ShuffleExchangeExec(partitioning, child, REPARTITION_BY_NUM)
      }
    case None => child
  }

  /**
   * Replaces every [[AiQuery]] in `exprs` with a reference to a column produced by an
   * [[AiQueryExec]] stacked on `child`. Nested calls (`ai_query(m, ai_query(m, x))`) get one
   * exec per nesting level, innermost first. The input is redistributed once, below all of them.
   */
  private def extract(exprs: Seq[Expression], child: SparkPlan): (SparkPlan, Seq[Expression]) = {
    var current = exprs
    var plan = distribute(child)
    while (current.exists(hasAi)) {
      val innermost = current.flatMap(_.collect {
        case q: AiQuery if !q.children.exists(hasAi) => q
      }).distinct
      val attrs = innermost.indices.map(i => AttributeReference(s"ai_query_$i", StringType)())
      val replacements = innermost.zip(attrs).toMap[Expression, Expression]
      plan = AiQueryExec(innermost, innermost.map(resolve), attrs, AiConfig.concurrency(conf), plan)
      current = current.map(_.transformUp { case q: AiQuery if replacements.contains(q) => replacements(q) })
    }
    (plan, current)
  }

  private def resolve(q: AiQuery): AiCall = {
    val params = q.modelParams
    AiCall(AiConfig.resolve(conf, q.modelName, params), AiConfig.failOnError(conf, params))
  }
}
