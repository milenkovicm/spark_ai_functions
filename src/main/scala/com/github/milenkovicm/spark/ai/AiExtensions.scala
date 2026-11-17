package com.github.milenkovicm.spark.ai

import org.apache.spark.sql.SparkSessionExtensions

/**
 * Registers the AI SQL functions and their planner strategy. Enable with
 * `spark.sql.extensions=com.github.milenkovicm.spark.ai.AiExtensions`.
 */
class AiExtensions extends (SparkSessionExtensions => Unit) {
  override def apply(extensions: SparkSessionExtensions): Unit = {
    AiFunctions.all.foreach(extensions.injectFunction)
    extensions.injectResolutionRule(_ => PullOutAiQuery)
    extensions.injectPlannerStrategy(AiStrategy)
  }
}
