package com.github.milenkovicm.spark.ai

import org.apache.spark.sql.catalyst.FunctionIdentifier
import org.apache.spark.sql.catalyst.expressions.{ArrayJoin, Cast, Concat, Expression, ExpressionInfo, Literal, NamedArgumentExpression, StructsToJson}
import org.apache.spark.sql.catalyst.util.ArrayData
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.{ArrayType, BinaryType, MapType, StringType, StructType}

/** SQL functions: `ai_query` plus task-specific helpers that are prompt templates over it. */
object AiFunctions {
  type Builder = Seq[Expression] => Expression

  private def info(name: String, usage: String) = new ExpressionInfo(classOf[AiQuery].getName, null, name, usage, "")

  private def defaultModel: Expression = Literal(AiConfig.defaultModel(SQLConf.get).getOrElse(
    throw new IllegalArgumentException(s"${AiConfig.DefaultModel} must be set to use ai_* helper functions")))

  private def fail(name: String, expected: String, args: Seq[Expression]) =
    throw new IllegalArgumentException(s"$name expects $expected, got ${args.size} argument(s)")

  /** Builds `prompt prefix + text + suffix` without evaluating anything at analysis time. */
  private def template(parts: Expression*): Expression = Concat(parts)

  private def helper(name: String, usage: String, arity: Int)(prompt: Seq[Expression] => Expression) =
    (FunctionIdentifier(name), info(name, usage), { (args: Seq[Expression]) =>
      if (args.size != arity) fail(name, s"$arity argument(s)", args)
      AiQuery(defaultModel, prompt(args), Literal(null))
    }: Builder)

  private val aiQuery = (FunctionIdentifier("ai_query"),
    info("ai_query", "_FUNC_(endpoint, request[, params]) - Sends `request` to model `endpoint` and returns the reply."),
    {
      case Seq(endpoint, request) => AiQuery(endpoint, request, Literal(null))
      case Seq(endpoint, request, params) => AiQuery(endpoint, request, params)
      case args => fail("ai_query", "2 or 3 arguments", args)
    }: Builder)

  // ---- decision-model functions ----

  /** Positional arguments and `name => value` arguments, the latter constant and lower-cased by name. */
  private def splitArgs(name: String, args: Seq[Expression], allowed: Set[String]): (Seq[Expression], Map[String, String]) = {
    val (named, positional) = args.partition(_.isInstanceOf[NamedArgumentExpression])
    if (args.dropWhile(!_.isInstanceOf[NamedArgumentExpression]).exists(!_.isInstanceOf[NamedArgumentExpression])) {
      throw new IllegalArgumentException(s"$name: positional arguments must come before named arguments")
    }
    val pairs = named.map { case NamedArgumentExpression(key, value) =>
      val k = key.toLowerCase
      if (!allowed(k)) {
        throw new IllegalArgumentException(
          s"$name: unknown argument '$key', expected one of ${allowed.toSeq.sorted.mkString(", ")}")
      }
      k -> constantString(name, k, value)
    }
    val duplicates = pairs.groupBy(_._1).collect { case (k, v) if v.size > 1 => k }
    if (duplicates.nonEmpty) throw new IllegalArgumentException(s"$name: duplicate argument ${duplicates.mkString(", ")}")
    (positional, pairs.toMap)
  }

  private def constantString(name: String, arg: String, e: Expression): String = {
    if (!e.foldable || !e.dataType.isInstanceOf[StringType]) {
      throw new IllegalArgumentException(s"$name: $arg must be a constant string")
    }
    Option(e.eval()).map(_.toString).getOrElse(throw new IllegalArgumentException(s"$name: $arg must not be null"))
  }

  /** A constant string array as labels, or a JSON object string mapping label to description. */
  private def constantOptions(name: String, arg: String, e: Expression): Seq[(String, Option[String])] = {
    if (!e.foldable) throw new IllegalArgumentException(s"$name: $arg must be a constant")
    e.dataType match {
      case ArrayType(_: StringType, _) =>
        val a = e.eval().asInstanceOf[ArrayData]
        if (a == null) throw new IllegalArgumentException(s"$name: $arg must not be null")
        (0 until a.numElements()).map(i => (if (a.isNullAt(i)) null else a.getUTF8String(i).toString) -> None)
      case _: StringType => Decision.parseOptions(name, constantString(name, arg, e))
      case other =>
        throw new IllegalArgumentException(
          s"$name: $arg must be an array of strings or a JSON object string, got ${other.simpleString}")
    }
  }

  /** Text as is; structs, maps and arrays as JSON; binary rejected; anything else cast to string. */
  private def decisionInput(name: String, e: Expression): Expression = e.dataType match {
    case _: StringType => e
    case _: StructType | _: MapType | _: ArrayType => StructsToJson(Map.empty, e, None)
    case BinaryType => throw new IllegalArgumentException(s"$name: binary input must be cast to string")
    case _ => Cast(e, StringType)
  }

  private def onError(name: String, value: Option[String]): Option[Boolean] = value.map(_.toLowerCase).map {
    case "fail" => true
    case "null" => false
    case other => throw new IllegalArgumentException(s"$name: on_error must be 'fail' or 'null', got '$other'")
  }

  private def decision(name: String, usage: String, positional: Int, extra: Set[String] = Set.empty)(
      build: (Seq[Expression], Map[String, String]) => (Seq[(String, Question)], DecisionOutput)) =
    (FunctionIdentifier(name), info(name, usage), { (args: Seq[Expression]) =>
      val (pos, named) = splitArgs(name, args, Set("model", "on_error") ++ extra)
      if (pos.size != positional) fail(name, s"$positional positional argument(s)", pos)
      val model = named.get("model").orElse(AiConfig.decisionModel(SQLConf.get)).getOrElse(
        throw new IllegalArgumentException(s"$name: pass model => '...' or set ${AiConfig.DecisionModel}"))
      val (questions, output) = build(pos, named)
      AiDecision(name, decisionInput(name, pos.head), questions, output, model, onError(name, named.get("on_error")))
    }: Builder)

  private val decisionFunctions = Seq(
    decision("ai_probability",
      "_FUNC_(input, condition[, model => name][, on_error => 'fail'|'null']) - " +
        "Probability (0 to 1) that `condition` holds for `input`, from a decision model.", 2) { (args, _) =>
      Seq("answer" -> Decision.noul("ai_probability", constantString("ai_probability", "condition", args(1)))) ->
        DecisionOutput.Probability
    },
    decision("ai_if",
      "_FUNC_(input, condition[, model => name][, on_error => 'fail'|'null']) - " +
        "True when a decision model finds `condition` more likely than not for `input`.", 2) { (args, _) =>
      Seq("answer" -> Decision.noul("ai_if", constantString("ai_if", "condition", args(1)))) -> DecisionOutput.If
    },
    decision("ai_choose",
      "_FUNC_(input, labels[, instructions => text][, model => name][, on_error => 'fail'|'null']) - " +
        "The label that best fits `input`, from a decision model. `labels` is an array of strings or a " +
        "JSON object mapping label to description.", 2, Set("instructions")) { (args, named) =>
      val instructions = named.getOrElse("instructions", "Which label best fits the input?")
      Seq("answer" -> Decision.choice("ai_choose", instructions, constantOptions("ai_choose", "labels", args(1)))) ->
        DecisionOutput.Choose
    },
    decision("ai_score",
      "_FUNC_(input, instructions, levels[, model => name][, on_error => 'fail'|'null']) - " +
        "Probability-weighted level index (0 to n-1) of `input` on `levels`, lowest first.", 3) { (args, _) =>
      val instructions = constantString("ai_score", "instructions", args(1))
      Seq("answer" -> Decision.score("ai_score", instructions, constantOptions("ai_score", "levels", args(2)))) ->
        DecisionOutput.ScoreValue
    },
    decision("ai_decide",
      "_FUNC_(input, questions[, model => name][, on_error => 'fail'|'null']) - " +
        "Answers several questions about `input` in one request and returns a struct with one field per " +
        "question. `questions` is a JSON object: {\"id\": {\"type\": \"noul|choice|score\", " +
        "\"instructions\": \"...\", \"criteria\": ...}}.", 2) { (args, _) =>
      Decision.parseQuestions("ai_decide", constantString("ai_decide", "questions", args(1))) ->
        DecisionOutput.Decide
    })

  val all: Seq[(FunctionIdentifier, ExpressionInfo, Builder)] = decisionFunctions ++ Seq(
    aiQuery,
    helper("ai_gen", "_FUNC_(prompt) - Answers `prompt` using the default model.", 1)(args => args.head),
    helper("ai_summarize", "_FUNC_(text) - Summarizes `text`.", 1) { args =>
      template(
        Literal("Summarize the following text in at most 50 words. Reply with the summary only.\n\n"),
        args.head)
    },
    helper("ai_analyze_sentiment", "_FUNC_(text) - Returns positive, negative, neutral or mixed.", 1) { args =>
      template(
        Literal("Classify the sentiment of the following text as exactly one of: positive, negative, " +
          "neutral, mixed. Reply with that single lowercase word only.\n\nText: "),
        args.head)
    },
    helper("ai_classify", "_FUNC_(text, labels) - Returns the label from array `labels` that best fits `text`.", 2) {
      args =>
        template(
          Literal("Classify the following text into exactly one of these labels: "),
          ArrayJoin(args(1), Literal(", "), None),
          Literal(". Reply with the label only, exactly as written.\n\nText: "),
          args.head)
    },
    helper("ai_translate", "_FUNC_(text, language) - Translates `text` to `language`.", 2) { args =>
      template(
        Literal("Translate the following text to "), args(1),
        Literal(". Reply with the translation only.\n\n"), args.head)
    },
    helper("ai_extract", "_FUNC_(text, labels) - Extracts entities of types in array `labels` as JSON.", 2) {
      args =>
        template(
          Literal("Extract the following entity types from the text: "),
          ArrayJoin(args(1), Literal(", "), None),
          Literal(". Reply with a single JSON object mapping each type to the extracted value or null, " +
            "and nothing else.\n\nText: "),
          args.head)
    },
    helper("ai_fix_grammar", "_FUNC_(text) - Corrects grammatical errors in `text`.", 1) { args =>
      template(
        Literal("Correct the grammar and spelling of the following text. Reply with the corrected text only.\n\n"),
        args.head)
    },
    helper("ai_mask", "_FUNC_(text, labels) - Replaces entities of types in array `labels` with [MASKED].", 2) {
      args =>
        template(
          Literal("Rewrite the following text, replacing every occurrence of these entity types with " +
            "[MASKED]: "),
          ArrayJoin(args(1), Literal(", "), None),
          Literal(". Reply with the rewritten text only.\n\nText: "),
          args.head)
    })
}
