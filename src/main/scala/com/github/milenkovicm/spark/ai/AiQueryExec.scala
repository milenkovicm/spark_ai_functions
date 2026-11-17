package com.github.milenkovicm.spark.ai

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeSet, SortOrder, UnsafeProjection}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.{SparkPlan, UnaryExecNode}
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}

/**
 * Evaluates `calls` for every input row, appending one string column per call (`resultAttrs`).
 * Requests are issued concurrently within each partition; see [[AsyncEvaluator]].
 */
case class AiQueryExec(
    calls: Seq[AiQuery],
    resolved: Seq[AiCall],
    resultAttrs: Seq[Attribute],
    concurrency: Int,
    child: SparkPlan)
  extends UnaryExecNode {

  override def output: Seq[Attribute] = child.output ++ resultAttrs

  override def producedAttributes: AttributeSet = AttributeSet(resultAttrs)

  override def outputPartitioning: Partitioning = child.outputPartitioning

  override def outputOrdering: Seq[SortOrder] = child.outputOrdering

  override lazy val metrics: Map[String, SQLMetric] = Map(
    "numRequests" -> SQLMetrics.createMetric(sparkContext, "number of AI requests"),
    "numFailures" -> SQLMetrics.createMetric(sparkContext, "number of failed AI requests"),
    "requestTime" -> SQLMetrics.createTimingMetric(sparkContext, "AI request time (sum)"))

  override protected def doExecute(): RDD[InternalRow] = {
    val requests = longMetric("numRequests")
    val failures = longMetric("numFailures")
    val requestTime = longMetric("requestTime")
    val prompts = calls.map(_.request)
    val inputAttrs = child.output
    val outputAttrs = output
    val calls0 = resolved
    // `concurrency` bounds requests in flight per partition; each buffered row carries one per call.
    val window = (concurrency / calls0.size).max(1)
    child.execute().mapPartitions { iter =>
      new AsyncEvaluator(
        iter,
        calls0,
        UnsafeProjection.create(prompts, inputAttrs),
        UnsafeProjection.create(outputAttrs, outputAttrs),
        window,
        requests,
        failures,
        requestTime)
    }
  }

  override def simpleString(maxFields: Int): String =
    s"AiQueryExec [${calls.map(_.modelName).mkString(", ")}] -> " +
      s"[${resultAttrs.mkString(", ")}], concurrency=$concurrency"

  override protected def withNewChildInternal(newChild: SparkPlan): AiQueryExec =
    copy(child = newChild)
}
