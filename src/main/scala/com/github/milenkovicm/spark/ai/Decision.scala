package com.github.milenkovicm.spark.ai

import scala.jdk.CollectionConverters._

import com.fasterxml.jackson.databind.{JsonNode, ObjectMapper}
import dev.langchain4j.model.decision.request.{ChoiceQuestion, ScaleQuestion, YesNoQuestion, Question => LcQuestion}
import dev.langchain4j.model.decision.response.DecisionResponse

import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.util.{ArrayBasedMapData, GenericArrayData}
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

/**
 * A closed question for a decision model, validated at analysis time. Options and levels carry
 * an optional description.
 */
sealed trait Question extends Serializable {
  def instructions: String
  def toLangChain: LcQuestion
}

/** Yes/no question; the answer is the probability of yes (`noul` / `predicate` on the wire). */
case class Noul(instructions: String) extends Question {
  override def toLangChain: LcQuestion = YesNoQuestion.from(instructions)
}

/** Pick one of `options` (2–255). */
case class Choice(instructions: String, options: Seq[(String, Option[String])]) extends Question {
  override def toLangChain: LcQuestion = {
    val b = ChoiceQuestion.builder().text(instructions)
    options.foreach {
      case (label, Some(d)) => b.option(label, d)
      case (label, None) => b.option(label)
    }
    b.build()
  }
}

/** Score on `levels` (2–10, lowest first); the answer is the probability-weighted level index. */
case class Score(instructions: String, levels: Seq[(String, Option[String])]) extends Question {
  override def toLangChain: LcQuestion = {
    val b = ScaleQuestion.builder().text(instructions)
    levels.foreach {
      case (label, Some(d)) => b.level(label, d)
      case (label, None) => b.level(label)
    }
    b.build()
  }
}

/** How a decision function turns its answer(s) into a column value. */
sealed trait DecisionOutput extends Serializable {
  def dataType(questions: Seq[(String, Question)]): DataType
}

object DecisionOutput {
  /** `ai_probability`: DOUBLE in 0..1. */
  case object Probability extends DecisionOutput {
    override def dataType(questions: Seq[(String, Question)]): DataType = DoubleType
  }
  /** `ai_if`: probability > 0.5. */
  case object If extends DecisionOutput {
    override def dataType(questions: Seq[(String, Question)]): DataType = BooleanType
  }
  /** `ai_choose`: the most probable label. */
  case object Choose extends DecisionOutput {
    override def dataType(questions: Seq[(String, Question)]): DataType = StringType
  }
  /** `ai_score`: probability-weighted level index, 0..n-1. */
  case object ScoreValue extends DecisionOutput {
    override def dataType(questions: Seq[(String, Question)]): DataType = DoubleType
  }
  /** `ai_decide`: a struct with one field per question. */
  case object Decide extends DecisionOutput {
    override def dataType(questions: Seq[(String, Question)]): DataType =
      StructType(questions.map { case (id, q) => StructField(id, Decision.answerType(q)) })
  }
}

object Decision {
  val MaxQuestions = 64

  private val probabilities = MapType(StringType, DoubleType, valueContainsNull = true)

  val NoulType: StructType = StructType(Seq(StructField("probability", DoubleType)))
  val ChoiceType: StructType = StructType(Seq(
    StructField("choice", StringType),
    StructField("probabilities", probabilities),
    StructField("confidence", DoubleType)))
  val ScoreType: StructType = StructType(Seq(
    StructField("score", DoubleType),
    StructField("probabilities", probabilities),
    StructField("confidence", DoubleType)))

  def answerType(q: Question): StructType = q match {
    case _: Noul => NoulType
    case _: Choice => ChoiceType
    case _: Score => ScoreType
  }

  /** Thrown when the model declines a question and the call fails on error. */
  class RefusedException(id: String) extends RuntimeException(s"decision model refused question '$id'")

  /**
   * Converts the answers to `questions` (sent under `wireIds`) into the value of one decision
   * column. A refused question is null, or fails when `failOnError`.
   */
  def value(
      output: DecisionOutput,
      questions: Seq[(String, Question)],
      wireIds: Seq[String],
      response: DecisionResponse,
      failOnError: Boolean): Any = {
    def answer(i: Int)(f: String => Any): Any = {
      val id = wireIds(i)
      if (response.isRefused(id)) {
        if (failOnError) throw new RefusedException(questions(i)._1)
        null
      } else {
        f(id)
      }
    }
    output match {
      case DecisionOutput.Probability => answer(0)(id => response.yesNo(id).probability())
      case DecisionOutput.If => answer(0)(id => response.yesNo(id).probability() > 0.5)
      case DecisionOutput.Choose => answer(0)(id => UTF8String.fromString(response.choice(id).value()))
      case DecisionOutput.ScoreValue => answer(0)(id => response.scale(id).mean())
      case DecisionOutput.Decide =>
        val fields = questions.indices.map { i =>
          answer(i)(id => struct(questions(i)._2, id, response))
        }
        InternalRow.fromSeq(fields)
    }
  }

  private def struct(q: Question, id: String, response: DecisionResponse): InternalRow = q match {
    case _: Noul =>
      InternalRow(response.yesNo(id).probability())
    case c: Choice =>
      val a = response.choice(id)
      val p = Option(a.probabilities()).map(_.asScala).getOrElse(Map.empty[String, java.lang.Double])
      InternalRow(
        UTF8String.fromString(a.value()),
        map(c.options.map(_._1), c.options.map(o => p.get(o._1).orNull)),
        a.confidence())
    case s: Score =>
      val a = response.scale(id)
      val p = Option(a.probabilities()).map(_.asScala.toIndexedSeq).getOrElse(IndexedSeq.empty)
      InternalRow(
        a.mean(),
        map(s.levels.map(_._1), s.levels.indices.map(i => if (i < p.size) p(i) else null)),
        a.confidence())
  }

  private def map(keys: Seq[String], values: Seq[java.lang.Double]): ArrayBasedMapData =
    new ArrayBasedMapData(
      new GenericArrayData(keys.map(UTF8String.fromString).toArray[Any]),
      new GenericArrayData(values.toArray[Any]))

  // ---- validation and parsing (analysis time) ----

  private val mapper = new ObjectMapper()

  def require(condition: Boolean, function: String, message: => String): Unit =
    if (!condition) throw new IllegalArgumentException(s"$function: $message")

  def choice(function: String, instructions: String, options: Seq[(String, Option[String])]): Choice = {
    require(options.size >= 2 && options.size <= 255, function, s"expects 2 to 255 labels, got ${options.size}")
    require(options.forall(o => o._1 != null && o._1.trim.nonEmpty), function, "labels must not be empty")
    require(options.map(_._1).distinct.size == options.size, function, "labels must be distinct")
    Choice(instructions, options)
  }

  def score(function: String, instructions: String, levels: Seq[(String, Option[String])]): Score = {
    require(levels.size >= 2 && levels.size <= 10, function, s"expects 2 to 10 levels, got ${levels.size}")
    require(levels.forall(l => l._1 != null && l._1.trim.nonEmpty), function, "levels must not be empty")
    require(levels.map(_._1).distinct.size == levels.size, function, "levels must be distinct")
    Score(instructions, levels)
  }

  def noul(function: String, instructions: String): Noul = {
    require(instructions != null && instructions.trim.nonEmpty, function, "the condition must not be empty")
    Noul(instructions)
  }

  /** `["a", "b"]` or `{"a": "description", "b": null}` (in order). */
  def options(function: String, node: JsonNode): Seq[(String, Option[String])] =
    if (node == null || node.isNull) {
      Seq.empty
    } else if (node.isArray) {
      node.elements().asScala.map(e => e.asText() -> None).toSeq
    } else if (node.isObject) {
      node.properties().asScala.toSeq.map { e =>
        e.getKey -> Option(e.getValue).filterNot(_.isNull).map(_.asText())
      }.toSeq
    } else {
      throw new IllegalArgumentException(s"$function: expected a JSON array or object, got ${node.getNodeType}")
    }

  def parseOptions(function: String, json: String): Seq[(String, Option[String])] =
    options(function, readJson(function, json))

  private def readJson(function: String, json: String): JsonNode =
    try mapper.readTree(json)
    catch { case e: Exception => throw new IllegalArgumentException(s"$function: invalid JSON: ${e.getMessage}") }

  /**
   * Parses `ai_decide` questions, in Spice's format:
   * `{"id": {"type": "noul|choice|score", "instructions": "...", "criteria": ...}}`.
   * `predicate` is accepted for `noul`; choice criteria are a label array or a label→description
   * object, score criteria the levels lowest first.
   */
  def parseQuestions(function: String, json: String): Seq[(String, Question)] = {
    val root = readJson(function, json)
    require(root.isObject, function, "questions must be a JSON object mapping question id to question")
    val questions = root.properties().asScala.toSeq.map { e =>
      val id = e.getKey
      val q = e.getValue
      val name = s"$function question '$id'"
      require(id.trim.nonEmpty, function, "question ids must not be empty")
      require(q.isObject, name, "must be a JSON object")
      val instructions = Option(q.get("instructions")).filterNot(_.isNull).map(_.asText()).getOrElse("")
      val criteria = q.get("criteria")
      Option(q.get("type")).map(_.asText().toLowerCase).getOrElse("") match {
        case "noul" | "predicate" => id -> noul(name, instructions)
        case "choice" => id -> choice(name, instructions, options(name, criteria))
        case "score" => id -> score(name, instructions, options(name, criteria))
        case other => throw new IllegalArgumentException(s"$name: type must be noul, choice or score, got '$other'")
      }
    }.toSeq
    require(questions.nonEmpty && questions.size <= MaxQuestions, function,
      s"expects 1 to $MaxQuestions questions, got ${questions.size}")
    questions
  }
}
