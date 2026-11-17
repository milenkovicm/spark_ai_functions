package com.github.milenkovicm.spark.ai

import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

trait SparkSessionSuite extends AnyFunSuite with BeforeAndAfterAll {
  lazy val spark: SparkSession = SparkSession.builder()
    .master("local[2]")
    .appName(getClass.getSimpleName)
    .config("spark.sql.extensions", classOf[AiExtensions].getName)
    .config("spark.ui.enabled", "false")
    .config("spark.sql.shuffle.partitions", "2")
    .getOrCreate()

  override protected def afterAll(): Unit = {
    try spark.stop()
    finally super.afterAll()
  }
}
