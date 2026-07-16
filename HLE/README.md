# HLE local benchmark harness

This workspace wraps the `cais/hle` dataset with `bench.py` for single-question timing runs.

The default output cap is `--max-tokens 100000`, mapped to OpenAI `max_output_tokens` and Anthropic `max_tokens`.
Model and judge requests both default to a one-hour timeout (`3600` seconds).
Reasoning effort defaults to the comparison-standard `xhigh`; pass `--thinking-effort none` only for a deliberately non-reasoning run.

## Environment

Create or refresh the project environment from the pinned root requirements file:

```bash
python3 -m venv --clear .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run the harness with `.venv/bin/python`. The environment pins OpenAI `2.44.0`, Anthropic `0.91.0`, and the dataset/runtime dependencies used by this workspace.

## Stable numeric question references

`bench.py` loads the HLE `test` split, keeps text-only rows, sorts them by the canonical HLE `id`, and assigns a zero-based `sorted_index`.

Use `--index N` to reference text-only question `N` predictably for the current dataset revision. Store both `sorted_index` and `question_id` in results; `question_id` is the durable identifier if the dataset changes.

## Fixed 100-question mini-benchmark

[`docs/hle-mini-bench-100.md`](docs/hle-mini-bench-100.md) defines the fixed `hle-mini-100-v1` subset: 100 text-only questions balanced across all eight HLE categories, with both `sorted_index` and `question_id` recorded. The manifest documents the deterministic sampling rule, dataset revision, category quotas, fingerprint, and a ready-to-run command.

[`docs/hle-verified-gold-100.md`](docs/hle-verified-gold-100.md) defines the separate `hle-verified-gold-100-v1` subset for cleaner recurring comparisons. It deterministically selects 100 category-balanced, text-only records from the pinned HLE-Verified Gold release after requiring explicit validity for the problem, answer, and rationale. Its canonical content is stored in [`hle-verified-gold-100-questions.csv`](hle-verified-gold-100-questions.csv). The original raw-HLE subset and historical results remain unchanged.

## OpenAI model runs

Model calls from `bench.py` use the OpenAI `/responses` API shape, not `/chat/completions`.

For models that only support the default temperature, including `model_fast` from `../.env`, pass `--omit-temperature`. The request should not include a `temperature` field.

Override the default OpenAI reasoning effort with, for example:

```bash
--thinking-effort high
```

For OpenAI `/responses`, this sends:

```python
reasoning={"effort": "high"}
```

Example:

```bash
.venv/bin/python bench.py \
  --endpoint-name std \
  --index 1 \
  --num-questions 1 \
  --runs 3 \
  --no-print-question \
  --max-tokens 100000 \
  --omit-temperature
```

Set `API_KEY`, `BASE_URL`, and `MODEL` in the environment, or pass the corresponding command-line flags. With `../.env`, `OpenAI-key` maps to `API_KEY` and `model_fast` maps to `MODEL`.

## Correctness judging

Normal benchmark runs automatically judge only unjudged completed responses after the model batch finishes. This includes pending responses recovered by resume. Use `--no-judge-after-run` to skip correctness scoring. Override the default `gpt-5.5` judge with `--judge-model` or `JUDGE_MODEL` when needed:

```bash
.venv/bin/python bench.py \
  --endpoint-name std \
  --index 1 \
  --num-questions 1 \
  --runs 3 \
  --no-print-question \
  --max-tokens 100000 \
  --omit-temperature \
  --judge-model gpt-5.5 \
  --judge-base-url https://api.openai.com/v1 \
  --judge-api-key "$OPENAI_KEY"
```

This checkpoints timing and raw output to `results.csv`, `responses.jsonl`, and `run_summaries.jsonl`, then batch-judges unjudged completed responses and writes them to `judgments.csv`. The resulting `yes`/`no` value is also written to the matching `correct` cell in `results.csv`.

Resume is enabled by default. A completed run is identified by `(endpoint_name, model, question_id, max_tokens, run_idx)` plus matching result/response timestamps. Re-running the same command skips completed requests, repairs partial result checkpoints, and finishes any pending judgments. Use `--no-resume` only when deliberately forcing new requests; use a new endpoint name for a distinct experiment.

Override the one-hour request timeouts with `--timeout-seconds` and `--judge-timeout-seconds`.

Every `results.csv` row explicitly records `total_wall_s`, `reasoning_tokens`, `output_tokens`, `correct`, `finish_reason`, and `refusal`. Explicit provider refusals and `content_filter` terminations set `refusal=yes`; ordinary completed answers set `refusal=no`. Providers that do not expose a separate reasoning-token count use `not_reported`; runs made without judging use `not_judged`. Historical finish reasons are backfilled from matching response checkpoints when the file is next opened by the harness.

The historical `tokens_per_s` CSV column is retained only for schema compatibility and mirrors `end_to_end_billed_tps` on eligible completed calls. New reports and terminal output use the explicit metric name.

## Counting and timing specification

The harness follows [`../benchmark-harness.md`](../benchmark-harness.md). The cross-provider headline throughput metric is:

```text
end_to_end_billed_tps =
  sum(final provider usage.output_tokens for strictly eligible calls)
  / sum(terminal event - request dispatch)
```

`billed_output_tokens` is populated only from terminal provider usage. Local tokenizer estimates remain in `output_tokens` with an explicit `token_count_method` and are never used for headline throughput. Reported reasoning/thinking tokens are an informational subset and are not added to billed output again. Refusals, max-token truncations, timeouts, API errors, and other incomplete outcomes are reported separately and excluded from headline TPS.

The monotonic request clock starts immediately before SDK dispatch and ends after the terminal event and final usage are available. A call is headline-eligible only when it completed normally, has positive request duration and final billed output usage, and its serialized authoritative usage reconciles exactly with `billed_output_tokens`. The harness records `matched`, `mismatched`, or `unavailable`; mismatches and unavailable usage are excluded rather than estimated. The SDKs are configured with zero automatic retries, and each attempted call is explicitly numbered.

Each new result also records the generation-start event, detail, confidence, hidden-reasoning observability, and active-generation time/TPS. OpenAI starts at `response.output_item.added` for a reasoning item; Anthropic starts at `content_block_start` for thinking/redacted-thinking, falling back to the first generated delta only when needed. Because the two providers do not always expose equivalent reasoning-aware boundaries, `active_generation_billed_tps` is provider-specific diagnostic data unless both cohorts prove equivalent coverage. `ttft_s` is request dispatch to first visible output and is diagnostic only; it is never subtracted from request time to manufacture a decode rate. This HLE harness has no tools, backoff, sandbox task, or economy cutoff, so those controls are recorded as zero or `not_applicable`. Calls run serially.

Timeouts and API failures are checkpointed with no invented billed-token count. They remain retryable under `--resume`; completed calls are skipped. Post-run correctness judging is outside the benchmark task timing buckets.

At the end of every normal benchmark invocation, the CLI prints a machine-readable aggregate line after judging finishes:

```text
run_report endpoint=std model=example attempted_runs=10 completed_inference_calls=8 billed_output_tokens=45000 end_to_end_billed_tps=123.456789 median_ttft_s=1.234000 usage_reconciliation={"matched":8,"unavailable":2} ... outcomes={"completed":8,"inference_timeout":1,"refusal":1} task_success=6/10 correctness=6/8_judged refusals=1/10
```

This report includes aggregate billed TPS (ratio of sums), token totals, every timing bucket, per-attempt averages, explicit outcome counts, attempted-task success, judged-answer correctness, and refusals. `task_success` keeps refusals and other failures in the denominator; `correctness` describes only responses actually sent to the judge. If a provider does not report a requested token breakdown, or judging is disabled, the corresponding value is an explicit status instead of a blank field.

`--judge-existing` is still available for backfills, but it matches prior records by `question_id`, `model`, and `max_tokens`, so it can include older runs.

## Anthropic Messages runs

Anthropic native runs use the Messages API, `POST /v1/messages`, through `client.messages.stream(...)`.

Use the comparison-standard effort with:

```bash
--provider anthropic --thinking-effort xhigh --omit-temperature
```

For Claude Opus 4.8, `bench.py` also sends adaptive thinking by default:

```python
thinking={"type": "adaptive", "display": "omitted"}
output_config={"effort": "xhigh"}
```

Example:

```bash
ANTHROPIC_API_KEY=... .venv/bin/python bench.py \
  --provider anthropic \
  --api-key "$ANTHROPIC_API_KEY" \
  --endpoint-name std \
  --model claude-opus-4-8 \
  --index 1 \
  --num-questions 1 \
  --runs 3 \
  --no-print-question \
  --max-tokens 100000 \
  --omit-temperature \
  --thinking-effort xhigh \
  --judge-model gpt-5.5 \
  --judge-base-url https://api.openai.com/v1 \
  --judge-api-key "$OPENAI_KEY"
```

This keeps the benchmark semantics close to OpenAI `/responses`: same HLE prompt, same streaming measurement points, and no explicit temperature. The token fields are not identical across providers. With Anthropic adaptive thinking and `display: omitted`, `output_tokens` is the provider-reported aggregate, while `reasoning_tokens` and `visible_output_tokens` are `not_reported`; the harness does not mislabel hidden thinking as visible output. OpenAI may expose separate reasoning-token details.
