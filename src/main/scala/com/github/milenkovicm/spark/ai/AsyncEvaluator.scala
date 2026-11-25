package com.github.milenkovicm.spark.ai

import java.util.ArrayDeque
import java.util.concurrent.Future

import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{GenericInternalRow, JoinedRow, UnsafeProjection}
import org.apache.spark.sql.execution.metric.SQLMetric

/**
 * Wraps a partition iterator and keeps up to `window` input rows' requests in flight, emitting
 * rows in input order as soon as the head of the queue has all of its results.
 *
 * @param prompts projects an input row to one string column per request (reused, so read eagerly)
 * @param width   number of result columns (a request may fill several)
 * @param output  projects `input ++ results` to the operator's output
 */
class AsyncEvaluator(
    input: Iterator[InternalRow],
    requests: Seq[AiRequest],
    prompts: UnsafeProjection,
    width: Int,
    output: UnsafeProjection,
    window: Int,
    numRequests: SQLMetric,
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
      val futures = requests.indices.map { i =>
        if (p.isNullAt(i)) null else requests(i).submit(p.getUTF8String(i).toString)
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
    val results = new GenericInternalRow(width)
    var i = 0
    while (i < futures.length) {
      if (futures(i) != null) {
        val r = futures(i).get()
        val request = requests(i)
        numRequests += 1
        requestTime += r.nanos / 1000000
        if (r.error == null) {
          request.complete(r.value, results)
        } else {
          failures += 1
          if (request.failOnError) {
            throw new RuntimeException(s"${request.describe} failed", r.error)
          }
          logWarning(s"${request.describe} failed, returning null", r.error)
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
