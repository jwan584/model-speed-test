# Native Codex benchmark

This harness runs LiveCodeBench problems through `codex exec`, preserving the
native Codex system prompt, agent loop, tools, and ChatGPT-managed CLI
authentication. It does not call a provider SDK directly.

## Setup

```zsh
cd "/Users/james/Library/Mobile Documents/com~apple~CloudDocs/Code/live-code-bench/LiveCodeBench"
.venv/bin/python -m pip install -r requirements-codex-native.txt
```

The local OTLP receiver binds only to `127.0.0.1`. User-prompt logging is
disabled. Each problem runs in an isolated output directory.

There are two inference-only timing modes:

- The stock-CLI server aggregate needs no Rust or compilation. It divides the
  target model's per-call output-token sum by that model's summed server
  `engine_service_total_ms` histogram. Every internal inference call is
  included, while wall-clock gaps for tools and agent orchestration are not.
- Exact per-call timing uses the pinned instrumented CLI. It records the full
  `response.created` to `response.completed` window and TPS value for every
  micro session and removes warmup calls.

The installed `codex-cli 0.144.3` requests server timing but exports it only as
an aggregate histogram. The builder applies a telemetry-only backport of the
upstream WebSocket timing trace hook; it does not change prompts, tools, model
requests, or agent behavior.

## Run Q1 with stock Codex (no build)

```zsh
./run_native_q1_server_tps
```

This is the simplest inference-only Q1 path. It uses the installed `codex`
binary and never calls Cargo. It retains the same activity messages and
30-second heartbeat output as the exact runner. Override the model with
`CODEX_NATIVE_MODEL` or pass an output directory as the first argument.

The Q1 launchers use the checked-in `abc387_f` problem fixture, so they do not
need a Hugging Face dataset cache or download the multi-gigabyte full release.
Its correctness check covers the three official public samples; the output
records `checker_scope=fixture_public_samples_only`. General `--problems` runs
continue to use the complete release dataset and its private tests.

The stock aggregate is a ratio of sums across all target-model inference calls
in the query. It excludes multi-turn tool time, but the CLI does not expose the
individual server timing observations, so it cannot report one TPS value per
micro session. It also includes any server warmup timing; warmup token counts
are zero. Use the instrumented path below when individual sessions or warmup
exclusion are required.

## Run with exact per-call timing

```zsh
./build_instrumented_codex
```

The first build requires Rust/Cargo and compiles the official
`rust-v0.144.3` tag. Subsequent runs reuse
`.codex-instrumented/codex-v0.144.3`.

```zsh
MODEL=gpt-5.6-sol-ultrafast

./run_codex_native_benchmark \
  --codex-bin .codex-instrumented/codex-v0.144.3 \
  --model "$MODEL" \
  --reasoning xhigh \
  --problems hard:1 hard:2 hard:3 \
  --release release_v6 \
  --output-dir native-codex-hard1-3
```

For the standard Q1 inference-only run, use:

```zsh
./run_native_q1_inference
```

The launcher builds the instrumented binary if it is missing, selects
`hard:1`, and writes to a timestamped output directory. Override the model with
`CODEX_NATIVE_MODEL` or pass a specific output directory as its first argument.

ChatGPT-managed native Codex cannot use a private API-only `model_fast` slug.
Use a model exposed to the signed-in Codex account. The Q1–Q3 convenience
launcher defaults to `gpt-5.6-sol-ultrafast`, which should be verified with
`codex debug models` because the available catalog depends on the account.
Using a private API-only slug requires API authentication and is a different
cohort.

Outputs include `results.csv`, an aggregate `summary.json`, the raw Codex JSONL
stream, decoded OTel events, the generated solution, and the official LCB
checker result. Aggregate mode records its reconciliation and coverage under
`server_aggregate_inference` in each summary. Exact mode additionally writes
per-call server timing to `inference-calls.jsonl`. The full
`codex-stderr.log` is retained in both modes.

## Metric interpretation

- `agent_end_to_end_output_tps` is always the provider-reported Codex turn
  output count divided by complete agent wall time. It includes tools.
- `codex_reported_inference_output_tps` uses Codex's exported Responses
  inference-duration metric when the installed client exports it. It is a
  compatibility alias for `server_aggregate_inference_output_tps`.
- `server_aggregate_inference_output_tps` is the stock-CLI inference-only
  metric. It filters both timing and token counts to the selected model, checks
  that metric observation counts match completed calls, and excludes tool-time
  gaps by construction.
- `native_inference_output_tps` is the primary inference-only metric. It pairs
  each lifecycle trace with that call's `response.completed` token counts,
  excludes warmups and auxiliary models, and divides total generated tokens by
  summed `response.created` to `response.completed` windows. Concurrent tool
  time is reported separately and is never subtracted from an in-flight call.
- `sse_event_window_output_tps_diagnostic` uses the first output SSE event to
  terminal completion. It remains diagnostic when OTel exposes the event kind
  but not `item.type=reasoning`.
- `active_generation_output_tps` is populated only when the captured boundary
  is scrutable under `../benchmark-harness.md`. The harness never silently
  substitutes visible TTFT or whole-request latency.

For multiturn tasks, aggregate token rates as a ratio of sums across internal
model calls, never as an average of per-call rates.

The harness enables Codex's `runtime_metrics` feature and opts into the timing
trace target where the binary provides it. Missing or malformed timing data is
reported as partial coverage; agent wall time is never substituted.

## Nested-Codex limitation

A Codex process launched from inside another managed Codex workspace may be
denied while initializing its in-process app-server client. Run the command in
a normal Terminal session in that environment. This is a launcher restriction,
not a benchmark or model failure.
