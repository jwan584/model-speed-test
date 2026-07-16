# Local API benchmark harness

Use `bench_runner.py` for reproducible, streamed comparisons across OpenAI,
Anthropic, and Cerebras. Run commands from this repository root.

## Setup

```zsh
test -x .venv/bin/python || python3 -m venv .venv
.venv/bin/pip install 'openai>=2.0.0' 'anthropic>=0.42.0' \
  'datasets>=3.2.0,<4.0.0' 'packaging>=24.0' 'pebble>=5.1.0'
```

Keep credentials in an ignored `.env`. This workspace's shared file uses key
names that are not all valid shell identifiers, so parse it without sourcing:

```zsh
typeset -A cfg
while IFS='=' read -r k v || [[ -n "$k" ]]; do
  [[ -z "$k" || "$k" == \#* ]] && continue
  cfg[$k]="${v%$'\r'}"
done < ../../.env
```

## Example comparison

```zsh
HF_HOME=.cache/huggingface XDG_CACHE_HOME=.cache \
  .venv/bin/python bench_runner.py \
    --endpoint-name openai-fast --provider openai \
    --model "${cfg[model_fast]}" --api-key "${cfg[OpenAI-key]}" \
    --thinking-effort xhigh \
    --endpoint-name anthropic-fable --provider anthropic \
    --model claude-fable-5 --api-key "${cfg[ANTHROPIC_API_KEY]}" \
    --thinking-effort xhigh \
    --problem-ids hard:1-80 --release-version v6 \
    --max-tokens 32000 --runs 3 --checker-timeout 6 \
    --timeout-seconds 900 --csv results.csv --resume
```

Problem indexes are per difficulty: use `easy:1`, `medium:1`, or `hard:1`.
Inclusive ranges such as `hard:1-80` are supported. Endpoint order rotates by
problem and run to balance time-order effects. Multi-endpoint comparisons are
rejected unless every endpoint uses the standard `xhigh` reasoning effort.
Requests and tasks run serially.

## Cerebras GLM 4.7

```zsh
HF_HOME=.cache/huggingface XDG_CACHE_HOME=.cache \
  .venv/bin/python bench_runner.py \
    --endpoint-name cerebras-glm47 --provider cerebras \
    --model zai-glm-4.7 --api-key "${cfg[Cerebras_Key]}" \
    --problem-ids hard:1 --release-version v6 \
    --max-tokens 40000 --runs 1 --checker-timeout 6 \
    --csv cerebras_glm47.csv
```

Cerebras defaults to `https://api.cerebras.ai/v1`. GLM reasoning remains
enabled when `--thinking-effort` is omitted. The adapter requests parsed
reasoning so reasoning starts the observable generation clock but is not mixed
into code submitted to the checker. Cerebras documents 40,000 as GLM 4.7's
maximum completion-token limit; Q1 exhausted 32,000 reasoning tokens, while a
40,000-token run completed and passed.

## Reliability and resume behavior

- Every provider request runs in a killable child process by default.
- The parent prints a heartbeat every 60 seconds and enforces the complete
  `--timeout-seconds` wall deadline even when an SDK blocks in socket I/O.
- Each CSV row is flushed and `fsync`ed before continuing.
- `--resume` skips completed configuration/problem/run tuples.
- `--max-attempts-per-tuple` defaults to three durable attempts, preventing
  repeated resumes from retrying one failing tuple forever.
- Terminal responses with usage but no visible text retain usage and stop
  metadata; their `ttft_s` is blank and empty output normally fails checking.
- Inference timeouts are recorded as `inference_timeout`, with elapsed failed
  API time and no invented token count. Incomplete and failed calls never enter
  billed-token throughput.
- Generated deliverables are retained under `<csv-stem>_artifacts/`; each CSV
  row records its artifact path and provider request ID.
- Provider-neutral single-turn tasks outside LCB can use `--custom-tasks` with
  a JSON array of `id`, `prompt`, and optional regex `required_patterns` fields.
  They use the same provider, timing, token, isolation, interleaving, artifact,
  and resume paths as LCB. Custom HTML must be a bare, complete document;
  Markdown fences fail the structural gate. Run interactive browser QA
  separately because structural checks do not prove behavior.
- The final audit returns nonzero for missing, incomplete, errored, or duplicate
  completed tuples.
- A durable `<csv-stem>.summary.json` reports outcomes, averages per attempted
  run, timing-bucket totals, and aggregate ratio-of-sums throughput.

Use stable, configuration-specific endpoint labels. A fingerprint also records
the provider, model, effort, token cap, release, adaptive-thinking setting,
checker timeout, prompt style, request timeout, endpoint region, sandbox/CPU/
memory metadata, tool configuration, economy policy, and harness schema. Supply
the comparison-control metadata explicitly when the defaults do not describe
the environment.

## Counting and timing definitions

Schema 6 records two ratio-of-sums output rates. The cross-path lifecycle
metric used to compare API and native Codex runs is:

```text
provider_window_billed_tps =
  sum(provider usage.output_tokens) / sum(response.completed - response.created)
```

The active-generation metric remains available for comparisons that have a
scrutable reasoning/output boundary:

```text
active_generation_billed_tps =
  sum(provider usage.output_tokens) / sum(terminal event - generation start)
```

It is a ratio of sums, never an average of per-call rates. Provider output
usage includes billed reasoning/thinking tokens; `reasoning_tokens` is recorded
as a subset and is never added a second time. When the provider reports that
subset, `visible_output_tokens = billed_output_tokens - reasoning_tokens`.
Provider-native tokenizers differ, so this measures completed, provider-
reported output work rather than tokenizer-independent text.

- `ttft_s`: request dispatch to first visible output text.
- `first_stream_event_s`: request dispatch to the first stream event.
- `response_created_s`: request dispatch to the provider's explicit
  `response.created` event.
- `provider_window_inference_time_s`: `response.created` through the terminal
  response event; this matches the native Codex lifecycle boundary.
- `provider_window_coverage`: aggregate coverage is complete only when every
  completed request has both lifecycle boundaries.
- `first_observable_output_s` and `last_observable_output_s`: offsets from
  dispatch to the first and last visible output chunks.
- `observable_chunk_count`: number of visible output deltas.
- `gen_time_s`: first-to-last visible output text.
- `generation_start_s` and `generation_start_event_*`: the provider-specific
  reasoning/output boundary and its audit metadata.
- `active_generation_time_s`: generation boundary through terminal usage; this
  is the active-generation TPS denominator. `generation_wall_s` is its
  deprecated alias.
- `end_to_end_billed_tps`: the historical request-start comparison metric.
- `inference_time_s`: API dispatch through the terminal event, including
  queueing, TTFT, hidden reasoning, visible generation, network, and completion.
- `tool_time_s`: model-invoked tool execution (zero for this coding harness).
- `retry_api_time_s`: failed API-attempt duration.
- `backoff_time_s`: retry sleep only (zero because SDK retries are disabled).
- `harness_overhead_s`: visible residual including worker startup, artifact
  persistence, and correctness evaluation.
- `total_wall_s`: task dispatch through completion or recorded failure.

The accounting identity should hold within timestamp precision:

```text
total_wall_s = inference_time_s + tool_time_s + retry_api_time_s
             + backoff_time_s + harness_overhead_s
```

Schema 6 is intentionally incompatible with older CSV headers. Use a new CSV
filename rather than appending or resuming an older result file.

## Native Codex inference and tool accounting

Native Codex uses a pinned instrumented CLI because the stock CLI does not
reliably receive the server's private engine-duration event. Build it once:

```zsh
./build_instrumented_codex
```

The first release build is intentionally cached and can be large (about 5 GB
of temporary build artifacts in the current toolchain). The installed binary
is `.codex-instrumented/codex-v0.144.3`. Repeated benchmark runs do not invoke
Cargo while that validated binary exists.

Run the self-contained Q1 fixture with fine-grained heartbeats:

```zsh
./run_native_q1_inference
```

Choose a model or output directory without rebuilding:

```zsh
CODEX_NATIVE_MODEL=gpt-5.6-sol-ultrafast \
  ./run_native_q1_inference native-q1-client-timing
```

For a dataset-backed eval, use the same binary explicitly:

```zsh
./run_codex_native_benchmark \
  --timing-mode micro \
  --codex-bin .codex-instrumented/codex-v0.144.3 \
  --model gpt-5.6-sol-ultrafast --reasoning xhigh \
  --problems hard:1-80 --release release_v6 \
  --output-dir native-hard-1-80
```

Each completed response prints an `inference_micro_session` heartbeat. The
per-query summary reports:

- `total_inference_seconds`: summed target-model provider lifecycle windows.
- `total_all_models_inference_seconds`: unioned target and auxiliary model
  activity.
- `total_tool_seconds`: unioned tool time; nested `exec`/`exec_command` layers
  are not double-counted.
- `inference_tool_concurrency_seconds`: time where one conversation was doing
  model work while another conversation's tool was running.
- `total_unattributed_seconds`: startup, orchestration, export, and other
  residual wall time.
- `native_provider_window_output_tps`: target output tokens divided by summed
  target-model provider lifecycle windows.

`inference-calls.jsonl` contains every micro-session and
`tool-intervals.jsonl` retains every raw duration-bearing tool record for
audit. The inference clock starts at the first provider response-lifecycle
event and ends when `response.completed` is fully received; overlapping tool
intervals are reported as concurrency but are not subtracted. Tool-only gaps
are already excluded because every inference call has its own lifecycle
window. This remains a client-observed metric, not the unavailable server
`engine_service_total_ms`. The accounting
identity is:

```text
wall ≈ union(all model-active intervals) + union(tool intervals)
     - inference/tool concurrency + unattributed
```

The runner marks coverage partial when lifecycle records are malformed,
target calls are missing, output-token totals do not reconcile, or it must
fall back to request-sent time because no provider lifecycle boundary arrived.

## Tests

```zsh
.venv/bin/python -m unittest -v tests.test_bench_runner
```
