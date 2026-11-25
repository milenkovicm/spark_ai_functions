package com.github.milenkovicm.spark.ai

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeSet, SortOrder, UnsafeProjection}
import org.apache.spark.sql.catalyst.plans.physical.Partitioning
import org.apache.spark.sql.execution.{SparkPlan, UnaryExecNode}
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}

/**
 * Evaluates `calls` for every input row, appending one column per call (`resultAttrs`). The calls
 * are sent as `requests`: one per chat call, and one per group of decision calls that share an
 * input and model. Requests are issued concurrently within each partition; see [[AsyncEvaluator]].
 */
case class AiQueryExec(
    calls: Seq[AiExpression],
    requests: Seq[AiRequest],
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
    val numRequests = longMetric("numRequests")
    val failures = longMetric("numFailures")
    val requestTime = longMetric("requestTime")
    val inputs = requests.map(_.input)
    val inputAttrs = child.output
    val outputAttrs = output
    val requests0 = requests
    val width = resultAttrs.size
    // `concurrency` bounds requests in flight per partition; each buffered row carries one per request.
    val window = (concurrency / requests0.size).max(1)
    child.execute().mapPartitions { iter =>
      new AsyncEvaluator(
        iter,
        requests0,
        UnsafeProjection.create(inputs, inputAttrs),
        width,
        UnsafeProjection.create(outputAttrs, outputAttrs),
        window,
        numRequests,
        failures,
        requestTime)
    }
  }

  override def simpleString(maxFields: Int): String =
    s"AiQueryExec [${requests.map(_.label).mkString(", ")}] -> " +
      s"[${resultAttrs.mkString(", ")}], concurrency=$concurrency"

  override protected def withNewChildInternal(newChild: SparkPlan): AiQueryExec =
    copy(child = newChild)
}
