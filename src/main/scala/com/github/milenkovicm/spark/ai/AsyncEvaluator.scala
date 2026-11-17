package com.github.milenkovicm.spark.ai

import java.util.ArrayDeque
import java.util.concurrent.Future

import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{GenericInternalRow, JoinedRow, UnsafeProjection}
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.unsafe.types.UTF8String

/** One `ai_query` call site, fully resolved on the driver. */
case class AiCall(config: AiConfig, failOnError: Boolean)

/**
 * Wraps a partition iterator and keeps up to `window` input rows' requests in flight, emitting
 * rows in input order as soon as the head of the queue has all of its results.
 *
 * @param prompts projects an input row to one string column per call (reused, so read eagerly)
 * @param output  projects `input ++ results` to the operator's output
 */
class AsyncEvaluator(
    input: Iterator[InternalRow],
    calls: Seq[AiCall],
    prompts: UnsafeProjection,
    output: UnsafeProjection,
    window: Int,
    requests: SQLMetric,
    failures: SQLMetric,
    requestTime: SQLMetric)
  extends Iterator[InternalRow] with Logging {

  private val pending = new ArrayDeque[(InternalRow, Array[Future[AiResponse]])]()
  private val joined = new JoinedRow()

  Option(TaskContext.get()).foreach(_.addTaskCompletionListener[Unit](_ => cancelAll()))

  private def fill(): Unit = {
    while (pending.size < window && input.hasNext) {
      // Upstream operators reuse their row objects; we hold this one across next() calls.
      val row = input.next().copy()
      val p = prompts(row)
      val futures = calls.indices.map { i =>
        if (p.isNullAt(i)) null else AiClient.submit(calls(i).config, p.getUTF8String(i).toString)
      }.toArray
      pending.add((row, futures))
    }
  }

  override def hasNext: Boolean = {
    fill()
    !pending.isEmpty
  }

  override def next(): InternalRow = {
    if (!hasNext) throw new NoSuchElementException
    val (row, futures) = pending.poll()
    val results = new GenericInternalRow(futures.length)
    var i = 0
    while (i < futures.length) {
      if (futures(i) != null) {
        val r = futures(i).get()
        requests += 1
        requestTime += r.nanos / 1000000
        if (r.error == null) {
          results.update(i, UTF8String.fromString(r.text))
        } else {
          failures += 1
          if (calls(i).failOnError) {
            throw new RuntimeException(s"ai_query to model '${calls(i).config.model}' failed", r.error)
          }
          logWarning(s"ai_query to model '${calls(i).config.model}' failed, returning null", r.error)
        }
      }
      i += 1
    }
    // Keep the window full while the caller processes this row.
    fill()
    output(joined(row, results))
  }

  private def cancelAll(): Unit = {
    while (!pending.isEmpty) {
      pending.poll()._2.foreach(f => if (f != null) f.cancel(true))
    }
  }
}
