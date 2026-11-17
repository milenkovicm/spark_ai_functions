package com.github.milenkovicm.spark.ai

import org.apache.spark.sql.catalyst.FunctionIdentifier
import org.apache.spark.sql.catalyst.expressions.{ArrayJoin, Concat, Expression, ExpressionInfo, Literal}
import org.apache.spark.sql.internal.SQLConf

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

  val all: Seq[(FunctionIdentifier, ExpressionInfo, Builder)] = Seq(
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
