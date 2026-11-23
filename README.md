# Spark AI Functions

AI SQL functions for Apache Spark, similar to
[Databricks AI Functions](https://docs.databricks.com/aws/en/large-language-models/ai-functions).
They are backed by any OpenAI-compatible endpoint: OpenAI, Ollama, vLLM, LM Studio, and others.

```sql
SELECT id,
       ai_analyze_sentiment(review)                                  AS sentiment,
       ai_classify(review, array('electronics', 'shipping', 'food')) AS topic,
       ai_query('qwen2.5:1.5b-instruct', concat('Summarize in five words: ', review)) AS summary
FROM reviews
```

```
+---+---------+-----------+---------------------------------+
|id |sentiment|topic      |summary                          |
+---+---------+-----------+---------------------------------+
|1  |positive |electronics|Love phone, long-lasting battery.|
|2  |negative |shipping   |Package broken, no response.     |
|3  |positive |food       |Excellent food, friendly waiter. |
+---+---------+-----------+---------------------------------+
```

The functions are ordinary SQL functions. You can use them in `SELECT`, `WHERE`, `GROUP BY`, inside aggregates and in window `PARTITION BY`, from SQL or the DataFrame API. Within each Spark partition, the model requests are sent **concurrently**, not one row at a time.

- Spark 4.1.x, Scala 2.13, JDK 17+ (built and tested on JDK 21)
- Model access goes through [LangChain4j](https://github.com/langchain4j/langchain4j) (`OpenAiChatModel`)

## Getting started

### Build

```bash
mvn package          # builds target/spark_ai_functions-1.0-SNAPSHOT.jar
                     #    and target/spark_ai_functions-1.0-SNAPSHOT-uber.jar
```

Spark is a `provided` dependency. LangChain4j is not, so it has to be on the classpath too. You can do this in two ways:

- **Uber jar** (`-uber.jar`). It bundles LangChain4j and its dependencies. It leaves out Jackson and SLF4J, which Spark already ships.
- **Thin jar**. Pass it together with `--packages dev.langchain4j:langchain4j-open-ai:1.23.0`.

### Enable the extension

With the uber jar:

```bash
spark-submit \
  --jars target/spark_ai_functions-1.0-SNAPSHOT-uber.jar \
  --conf spark.sql.extensions=com.github.milenkovicm.spark.ai.AiExtensions \
  --conf spark.sql.ai.baseUrl=http://127.0.0.1:11434/v1 \
  --conf spark.sql.ai.defaultModel=qwen2.5:1.5b-instruct \
  ...
```

or with the thin jar:

```bash
spark-submit \
  --jars target/spark_ai_functions-1.0-SNAPSHOT.jar \
  --packages dev.langchain4j:langchain4j-open-ai:1.23.0 \
  --conf spark.sql.extensions=com.github.milenkovicm.spark.ai.AiExtensions \
  --conf spark.sql.ai.baseUrl=http://127.0.0.1:11434/v1 \
  --conf spark.sql.ai.defaultModel=qwen2.5:1.5b-instruct \
  ...
```

or programmatically:

```java
SparkSession spark = SparkSession.builder()
        .master("local[*]")
        .config("spark.sql.extensions", "com.github.milenkovicm.spark.ai.AiExtensions")
        .config("spark.sql.ai.baseUrl", "http://127.0.0.1:11434/v1")
        .config("spark.sql.ai.defaultModel", "qwen2.5:1.5b-instruct")
        .config("spark.sql.ai.temperature", "0")
        .getOrCreate();
```

Example settings for a local Ollama server:

```bash
ollama pull qwen2.5:1.5b-instruct
ollama serve                         # OpenAI-compatible API on http://127.0.0.1:11434/v1
```

Ollama processes only `OLLAMA_NUM_PARALLEL` requests at a time. Raise it to benefit from the concurrency described below.

## Functions

| Function | Description |
|---|---|
| `ai_query(model, request[, params])` | Sends `request` to `model` and returns the reply as a string. |
| `ai_gen(prompt)` | Answers `prompt` with the default model. |
| `ai_summarize(text)` | Summarizes `text` in at most 50 words. |
| `ai_analyze_sentiment(text)` | Returns `positive`, `negative`, `neutral` or `mixed`. |
| `ai_classify(text, labels)` | Returns the label from the array `labels` that best fits `text`. |
| `ai_translate(text, language)` | Translates `text` into `language`. |
| `ai_extract(text, labels)` | Extracts entities of the types in `labels` and returns them as a JSON string. |
| `ai_fix_grammar(text)` | Corrects grammar and spelling. |
| `ai_mask(text, labels)` | Replaces entities of the types in `labels` with `[MASKED]`. |

All functions return `STRING`. A `NULL` request returns `NULL` without contacting the model.

Every helper except `ai_query` is a prompt template over `ai_query` and uses `spark.sql.ai.defaultModel`.

### `ai_query` parameters

`model` must be a constant string. The optional third argument is a constant struct:

```sql
SELECT ai_query('qwen2.5:1.5b-instruct',
                concat('Reply with one word describing this review: ', review),
                named_struct('temperature', 0.0, 'max_tokens', 10, 'fail_on_error', false))
FROM reviews
```

| Field | Meaning |
|---|---|
| `temperature` | Sampling temperature (overrides `spark.sql.ai.temperature`). |
| `max_tokens` | Maximum number of tokens in the reply (overrides `spark.sql.ai.maxTokens`). |
| `fail_on_error` | `false`: return `NULL` for failed requests instead of failing the query. |

## Examples

**Basic query**

```sql
SELECT id, ai_query('qwen2.5:1.5b-instruct', concat('Translate to French: ', text)) AS fr
FROM docs
```

**Filtering**

```sql
SELECT * FROM tickets
WHERE created > '2026-01-01'
  AND ai_classify(body, array('bug', 'question', 'feature request')) = 'bug'
```

The plain predicate (`created > ...`) runs first, so only matching rows are sent to the model.

**Grouping and aggregation**

```sql
-- group by model output
SELECT ai_analyze_sentiment(review) AS sentiment, count(*) FROM reviews GROUP BY 1;

-- model call inside an aggregate
SELECT count(DISTINCT ai_classify(review, array('positive', 'negative'))) FROM reviews;

-- model call over aggregation results
SELECT product, ai_summarize(concat_ws('\n', collect_list(review))) AS summary
FROM reviews GROUP BY product;
```

**Top-K**

```sql
-- only the 10 newest rows are sent to the model
SELECT id, ai_summarize(body) FROM articles ORDER BY published DESC LIMIT 10
```

**Nested calls**

```sql
SELECT ai_translate(ai_summarize(body), 'German') FROM articles
```

**DataFrame API**

```scala
import org.apache.spark.sql.functions.{col, expr}

spark.table("reviews")
  .select(col("id"), expr("ai_analyze_sentiment(review)").as("sentiment"))
  .show()
```

## Configuration

| Key | Default | Description |
|---|---|---|
| `spark.sql.ai.baseUrl` | `http://127.0.0.1:11434/v1` | OpenAI-compatible base URL. |
| `spark.sql.ai.apiKey` | `$OPENAI_API_KEY` or `none` | API key. |
| `spark.sql.ai.defaultModel` | (none) | Model used by the helper functions (`ai_gen`, `ai_classify`, ...). |
| `spark.sql.ai.concurrency` | `16` | Maximum number of requests in flight **per partition** (per task). |
| `spark.sql.ai.timeoutMs` | `120000` | Per-request timeout. |
| `spark.sql.ai.maxRetries` | `2` | Retries for transient errors (HTTP 429 and 5xx), handled by LangChain4j. |
| `spark.sql.ai.temperature` | model default | Sampling temperature. |
| `spark.sql.ai.maxTokens` | model default | Maximum number of tokens in a reply. |
| `spark.sql.ai.failOnError` | `true` | `false`: failed requests return `NULL`. |
| `spark.sql.ai.partitions` | unset (off) | Number of partitions (tasks) that make AI calls. See [Limiting parallelism](#limiting-parallelism). |
| `spark.sql.ai.partitionMode` | `shuffle` | `shuffle` or `coalesce`: how rows are redistributed when `spark.sql.ai.partitions` is set. |

These are session settings. They are read when a query is planned, so `spark.conf.set(...)` takes effect for the next query.

### Limiting parallelism

Requests are sent concurrently at two levels:

- **Within a task:** each task keeps up to `spark.sql.ai.concurrency` requests in flight (default 16).
- **Across tasks:** every input partition is a task, and as many run at once as there are executor cores.

Without further settings, the total against the endpoint is roughly **running tasks × `spark.sql.ai.concurrency`**. On a cluster with 40 cores that is up to 640 requests in flight.

To cap the total, set `spark.sql.ai.partitions`. Rows are redistributed into that many partitions before any model call, so the total becomes **`spark.sql.ai.partitions` × `spark.sql.ai.concurrency`**:

```scala
// at most 1 × 8 requests in flight, e.g. for a local Ollama server
spark.conf.set("spark.sql.ai.partitions", "1")
spark.conf.set("spark.sql.ai.concurrency", "8")
```

```
Project [id, ai_query_0 AS answer]
+- AiQueryExec [qwen2.5:1.5b-instruct] -> [ai_query_0], concurrency=8
   +- Exchange SinglePartition, REPARTITION_BY_NUM      ← added by spark.sql.ai.partitions
      +- Filter (id % 2 = 0)                            ← plain predicates still run in parallel
         +- Scan t
```

| `partitionMode` | Plan | Trade-off |
|---|---|---|
| `shuffle` (default) | `Exchange SinglePartition` (n = 1) or `RoundRobinPartitioning(n)` | Work before the AI call (scan, filters, projections) stays fully parallel; only the rows going to the model are shuffled. |
| `coalesce` | `Coalesce n` | No shuffle, but the work before the AI call also runs in only n tasks. Suitable when that work is cheap. |

The redistribution is placed so it works with the other plan rewrites:

- **Nested calls:** they share a single redistribution below the bottom `AiQueryExec`.
- **`WHERE`:** plain predicates are applied before the shuffle, so fewer rows move.
- **`ORDER BY … LIMIT n`:** already a single partition, so nothing is added.
- **`ORDER BY` with keys that don't depend on model output:** a shuffle after sorting would lose the order, so the sort's own partitioning does the redistribution instead. With n = 1 the plan sorts a single partition (`Exchange SinglePartition` or `Coalesce 1`). With n > 1 it uses a range shuffle into n partitions; its sampling runs before the model call, so nothing is evaluated twice.
- **Already small enough:** the step is skipped when the input's partition count is known and is at most n.

```scala
spark.conf.set("spark.sql.ai.partitions", "1")         // only 1 task calls the model
spark.conf.set("spark.sql.ai.concurrency", "8")        // → at most 1 × 8 requests in flight
spark.conf.set("spark.sql.ai.partitionMode", "shuffle") // default; or "coalesce"
If spark.sql.ai.partitions is not set, behaviour is the same as before.
```

## How it works

The implementation follows the pattern Spark itself uses for Python UDFs (`ExtractPythonUDFs` and `BatchEvalPythonExec`). A placeholder expression represents the call in the query plan. The planner then replaces it with a dedicated physical operator, which evaluates whole partitions concurrently.

```
 SELECT id, upper(ai_query('m', text)) AS a FROM t WHERE id < 100
                         │
        ┌────────────────▼─────────────────┐
        │ Analysis                         │  ai_query resolves to the AiQuery expression
        │  (injectFunction,                │  (Unevaluable, non-deterministic).
        │   PullOutAiQuery,                │  Calls in GROUP BY / aggregate functions are moved
        │   PullOutNondeterministic)       │  into a Project below the Aggregate.
        └────────────────┬─────────────────┘
        ┌────────────────▼─────────────────┐
        │ Optimizer (unchanged)            │  Non-determinism stops the call from being
        │                                  │  duplicated or moved across filters.
        └────────────────┬─────────────────┘
        ┌────────────────▼─────────────────┐
        │ Planning (AiStrategy)            │  Each AiQuery is replaced with a column produced
        │                                  │  by an AiQueryExec operator:
        └────────────────┬─────────────────┘
                         ▼
   Project [id, upper(ai_query_0) AS a]                  ← ordinary, codegen-able
   +- AiQueryExec [m] -> [ai_query_0], concurrency=16    ← concurrent model calls
      +- Filter (id < 100)
         +- Scan t
```

### 1. `AiQuery`: the placeholder expression

`ai_query(...)` resolves to `AiQuery`, a Catalyst expression with three properties:

- **`Unevaluable`.** It has no row-by-row implementation. If a call ever ends up in a place the strategy doesn't handle, the query fails with an error instead of quietly sending one blocking request per row. It also stops Spark's `ConvertToLocalRelation` from evaluating it on the driver.
- **Non-deterministic.** This stops the optimizer from inlining one call into several places (`CollapseProject`), which would multiply requests. It also stops the optimizer from moving filters across the call. Spark allows non-deterministic expressions only in `Project`, `Filter`, `Aggregate`, `Window` and `Generate`, so an unsupported placement such as a join condition is rejected at analysis time.
- **`UserDefinedExpression`.** This marks the expression as user-defined. Spark's `PullOutNondeterministic` analyzer rule then moves calls out of `GROUP BY` and `ORDER BY` into a `Project` below. Without the marker, Spark rejects `GROUP BY ai_query(...)`.

The helper functions are only prompt templates. For example, `ai_classify(text, labels)` builds `AiQuery(defaultModel, concat('Classify ... ', array_join(labels, ', '), ...), null)`.

### 2. `PullOutAiQuery` and `AiStrategy`: getting calls into a projection

The analyzer rule `PullOutAiQuery` moves calls inside aggregate functions, as in `count(ai_query(...))`, into a `Project` below the `Aggregate`, where they are evaluated once per input row.

The planner strategy `AiStrategy` runs before Spark's built-in strategies and handles each place a call can appear:

| Logical shape | Physical plan |
|---|---|
| `Project` with calls | `ProjectExec(AiQueryExec(child))` |
| `Filter` with calls | Plain conjuncts go into a `FilterExec` **below** the `AiQueryExec`; conjuncts with calls go in a `FilterExec` above it. |
| `Aggregate` whose output calls `ai_query` over grouping keys or aggregates | The `Aggregate` computes the keys and aggregates, and `ProjectExec(AiQueryExec)` above it evaluates the calls. This happens at planning time because `CollapseProject` would merge such a `Project` back into the `Aggregate`. |
| `Sort` on columns that don't depend on model output | The sort runs **below** the `AiQueryExec`. A global sort samples its input in an extra job, which would otherwise call the model twice per row. |
| `ORDER BY … LIMIT n` | `TakeOrderedAndProjectExec` runs below the `AiQueryExec`, so only `n` rows are sent to the model. |

For each projection the strategy:

1. collects the innermost `AiQuery` calls (identical calls are deduplicated);
2. replaces each one with a new attribute `ai_query_<i>`;
3. stacks one `AiQueryExec` per nesting level, so `ai_query(m, ai_query(m, x))` becomes two `AiQueryExec`s.

It also resolves each call's settings on the driver (base URL, model, temperature, retries and so on) into a serializable `AiConfig` that is shipped to the executors.

### 3. `AiQueryExec` and `AsyncEvaluator`: concurrent evaluation

`AiQueryExec` is a unary physical operator. Its output is the child's output plus one string column per call, and it keeps the child's partitioning and ordering. It runs `child.execute().mapPartitions(...)` and wraps each partition's iterator in an `AsyncEvaluator`, which works as a sliding window:

```
 input iterator ──► copy row, compute prompts, submit requests ──► ArrayDeque [(row, futures)]
                                                                     │  up to `concurrency` in flight
 output iterator ◄── wait for the oldest row's futures, append results ◄┘  (refilled after each row)
```

- **Order is preserved.** Rows are emitted strictly in input order, and only the head of the queue is waited on. Later rows' requests are already in flight.
- **Bounded memory.** At most `concurrency / calls-per-row` rows are buffered per task.
- **Rows are copied.** Spark operators reuse row objects (`UnsafeRow`), so each buffered row is `copy()`'d before it is held across `next()` calls. Without the copy, results get attached to the wrong rows.
- **Cancellation.** A task-completion listener cancels outstanding requests when a task finishes or is killed.
- **Metrics.** The operator reports *number of AI requests*, *number of failed AI requests* and *AI request time*, visible in the Spark UI SQL tab.

### 4. `AiClient`: model access with LangChain4j

Each executor JVM keeps one LangChain4j `OpenAiChatModel` per distinct `AiConfig`. LangChain4j handles:

- the OpenAI wire format;
- timeouts;
- retries with backoff on HTTP 429 and 5xx errors.

`ChatModel.chat()` is blocking, so each request runs on a **JDK 21 virtual thread** (`Executors.newVirtualThreadPerTaskExecutor()`). Many requests can therefore wait on the network at once without tying up platform threads. The sliding window in `AsyncEvaluator`, not a thread pool, is what limits concurrency.

### 5. Optional redistribution

When `spark.sql.ai.partitions` is set, `AiStrategy.distribute` puts a `ShuffleExchangeExec`
(origin `REPARTITION_BY_NUM`, so adaptive query execution keeps the partition count) or a
`CoalesceExec` under the bottom `AiQueryExec`. Global sorts are handled separately (see
[Limiting parallelism](#limiting-parallelism)). `AiQueryExec` passes its input's partitioning and
ordering through, so later operators see the same distribution.

### Measured

With a stub endpoint that takes 200 ms per request, 200 rows in a single partition with `concurrency=16` finish in about 3.5 s. Sequential calls would take about 40 s. Peak in-flight requests measured exactly 16.

## Limitations

- **Sorting by model output:** `ORDER BY ai_query(...)` without a limit calls the model twice per row, because Spark samples the input for range partitioning. Cache the result first (`.cache()`) if that matters.
- **Join conditions:** `ai_query` in a join condition is rejected by Spark (`INVALID_NON_DETERMINISTIC_EXPRESSIONS`). Compute it in a subquery first.
- **Constant arguments:** `model` and `params` must be constants.
- **Retried tasks:** results are not cached, so a retried Spark task or a re-executed stage sends its requests again.
- **Jackson version:** LangChain4j requests Jackson 2.22, but Maven resolves Spark's Jackson 2.18.6. This works for everything the test suites cover. If a LangChain4j feature needs a newer Jackson, shade/relocate `com.fasterxml.jackson` into the jar.

## Development

```bash
mvn test                                                            # all suites
mvn test -Dsuites=com.github.milenkovicm.spark.ai.AiQuerySuite      # stub-server suite only
mvn test -Dsuites=com.github.milenkovicm.spark.ai.OllamaIntegrationSuite
```

- **`AiQuerySuite`** runs against an in-process OpenAI-compatible stub server (`StubOpenAiServer`). It checks:
  - the plan shape;
  - concurrency and ordering;
  - request counts for each SQL shape;
  - null handling, failures, `fail_on_error` and retries;
  - `spark.sql.ai.partitions` in both modes: peak in-flight requests, plan shape, ordering, and that nothing is called twice.
- **`OllamaIntegrationSuite`** runs against `qwen2.5:1.5b-instruct` on `127.0.0.1:11434`. It is cancelled when Ollama isn't reachable.

`com.github.milenkovicm.Main` is a runnable Java demo against a local Ollama server. Spark needs the `--add-opens` JVM flags listed in `pom.xml` (`spark.jvm.opens`).

### Source layout

```
src/main/scala/com/github/milenkovicm/spark/ai/
  AiExtensions.scala    registers functions, analyzer rule and planner strategy
  AiFunctions.scala     ai_query and the prompt-template helpers
  AiQuery.scala         the placeholder Catalyst expression
  PullOutAiQuery.scala  analyzer rule for calls inside aggregates
  AiStrategy.scala      planner strategy that introduces AiQueryExec
  AiQueryExec.scala     physical operator
  AsyncEvaluator.scala  ordered sliding-window iterator
  AiClient.scala        LangChain4j model cache and virtual-thread execution
  AiConfig.scala        configuration keys and resolution
```
