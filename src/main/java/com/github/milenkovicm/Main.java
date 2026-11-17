package com.github.milenkovicm;

import org.apache.spark.sql.SparkSession;

/**
 * Runs the AI SQL functions against a local Ollama (http://127.0.0.1:11434/v1).
 */
public class Main {
    public static void main(String[] args) {
        String model = args.length > 0 ? args[0] : "qwen2.5:1.5b-instruct";

        SparkSession spark = SparkSession.builder()
                .master("local[*]")
                .appName("spark-ai-functions")
                .config("spark.sql.extensions", "com.github.milenkovicm.spark.ai.AiExtensions")
                .config("spark.sql.ai.baseUrl", "http://127.0.0.1:11434/v1")
                .config("spark.sql.ai.defaultModel", model)
                .config("spark.sql.ai.temperature", "0")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
        spark.sparkContext().setLogLevel("WARN");

        spark.sql("""
                SELECT * FROM VALUES
                  (1, 'I absolutely love this phone, the battery lasts for days!'),
                  (2, 'Terrible service. The package arrived broken and nobody answered my emails.'),
                  (3, 'Das Essen war ausgezeichnet und der Kellner sehr freundlich.')
                AS reviews(id, review)
                """).createOrReplaceTempView("reviews");

        var df = spark.sql("""
                SELECT id,
                       ai_analyze_sentiment(review)                                   AS sentiment,
                       ai_classify(review, array('electronics', 'shipping', 'food'))  AS topic,
                       ai_query('%s', concat('Summarize in five words: ', review))    AS summary
                FROM reviews
                """.formatted(model));
        df.explain();
        df.show(false);

        spark.stop();
    }
}
