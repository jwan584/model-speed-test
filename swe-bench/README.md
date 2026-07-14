# Local SWE-bench timing harness

Use [`SWE_BENCH_RUNBOOK.md`](SWE_BENCH_RUNBOOK.md) for the full setup,
monitoring, and recovery procedure.

For ChatGPT-authenticated Codex experiments comparing
`koffing-updated` with medium reasoning under normal and ultrafast
service-tier requests, see
[`CODEX_SWEBENCH_MODEL_TIERS.md`](CODEX_SWEBENCH_MODEL_TIERS.md).

## Run any numbered Verified problem with exact inference timing

Use the one-based problem number after sorting Verified/test instance IDs:

```bash
bash ./run_codex_swebench_problem Q1 --skip-evaluation
bash ./run_codex_swebench_problem Q27 --skip-evaluation
bash ./run_codex_swebench_problem Q2 \
  --model gpt-5.6-sol \
  --reasoning high \
  --skip-evaluation
```

The launcher reuses the prebuilt instrumented Codex binary from the neighboring
LiveCodeBench repository; it never invokes Cargo. Defaults match the current
cross-harness cohort: `gpt-5.6-sol-ultrafast`, reasoning `high`, and no explicit
service tier. Override them per run with `--model`, `--reasoning`, and
`--service-tier`. The problem can be positional or passed with `--question`:

```bash
bash ./run_codex_swebench_problem \
  --question Q5 \
  --model gpt-5.6-sol-ultrafast \
  --skip-evaluation
```

For automation, the equivalent environment variables are
`CODEX_SWEBENCH_QUESTION`, `CODEX_SWEBENCH_MODEL`,
`CODEX_SWEBENCH_REASONING`, and `CODEX_SWEBENCH_SERVICE_TIER`. Explicit CLI
options take precedence over environment values.

Each Mac keeps its expanded runtimes, Hugging Face cache, Colima VM, Docker
images, pip cache, and temporary worktrees under the persistent local path
`~/.swe-bench-runtime`. Only the harness, pinned requirements, and compact run
artifacts live in iCloud. The requirements fingerprint automatically rebuilds
a local runtime when the shared lock changes.

Results are separated by sanitized hostname under `runs/<machine>/`, allowing
two Macs to run concurrently without writing the same run directory. A
performance-only `--skip-evaluation` run installs only the minimal pinned
runner and never installs or imports the evaluator. Temporary worktrees are
removed after the patch and timing artifacts are captured. First-time Docker
image pulls and worktree extraction emit 30-second heartbeats.

The optional official evaluator is separately pinned in
`swebench_eval_requirements.txt` and installed locally only when evaluation is
requested. To evaluate a completed solve without rerunning the model, pass its
machine-specific run directory:

```bash
bash ./run_codex_swebench_problem Q1 \
  --evaluate-existing runs/<machine>/codex_problem1_default_YYYYMMDD_HHMMSS
```

Every internal model response prints an `inference_micro_session` heartbeat,
and a passive `running` heartbeat is printed every 30 seconds during quiet
periods.
The run directory records `inference_calls.jsonl`, `tool_intervals.jsonl`, and
an `inference_timing` object in `run_metadata.json`. Inference TPS is the ratio
of target-model output tokens to summed `response.created` →
`response.completed` windows. Tool-only gaps are excluded by construction;
concurrent tool time is reported separately and never subtracted.

## Native Claude Code request and tool timing

`claude_swebench_problem1.py` runs the same sorted SWE-bench Verified problem
selection, worktree preparation, container bridge, patch capture, and optional
official evaluation through Claude Code. For example, after the persistent
runner environment has been prepared:

```bash
~/.swe-bench-runtime/runner-venv/bin/python \
  ./claude_swebench_problem1.py \
  --index 1 \
  --model claude-sonnet-5 \
  --effort high \
  --require-complete-inference-timing \
  --skip-evaluation
```

The adapter starts a loopback-only OTLP/HTTP receiver and enables Claude
Code's official enhanced telemetry for the non-interactive session. One
`claude_code.llm_request` span is captured per model request and one
`claude_code.tool` span per tool invocation. Raw telemetry is never persisted;
only a strict whitelist of model, duration, token, retry, query-source, and
tool-name fields is written. Prompt and tool-content telemetry flags are
explicitly removed from the child environment.

`inference_calls.jsonl` contains every target and auxiliary model request with
its client-observed start/end window. `tool_intervals.jsonl` contains the tool
spans, and `otel_trace_diagnostics.json` records collector coverage. The
canonical `run_metadata.json` reports:

- target-model request-duration sum and ratio-of-sums output TPS
- target and all-model request-window unions
- tool-window union and request/tool concurrency
- an additive wall partition: request-only, tool-only, overlap, and residual
- the terminal CLI `duration_api_ms` value as a diagnostic only

These spans measure client/API request-active time including latency and
retries, not server-engine GPU decode time. If request spans are unavailable,
the adapter falls back to the older stream/terminal-result accounting, marks
coverage partial, and fails `--require-complete-inference-timing`.

## One-command Codex ultrafast run: problem 1

From this directory, run:

```bash
./run_codex_problem1_ultrafast.sh
```

The launcher starts the isolated Colima VM when necessary, selects Verified/test
index 0 (`astropy__astropy-12907`), pre-pulls its image outside the solve timer,
runs ChatGPT-authenticated Codex with `koffing-updated`, medium reasoning, and
`service_tier="ultrafast"`, then runs the official evaluator. No Docker socket
is exposed to Codex.

Progress is printed in the terminal. Durable results are written under
`runs/codex_problem1_ultrafast_<timestamp>/`; the consolidated result is
`run_metadata.json`, the submitted patch is `model.patch`, and raw Codex events
are in `codex_events.jsonl`. The final terminal lines report `SOLVE_SECONDS`,
`EVALUATION_SECONDS`, `RESOLVED`, and the metadata path.

To validate setup without consuming a model run:

```bash
./run_codex_problem1_ultrafast.sh --prepare-only
```

For the matching normal-speed condition, which omits `service_tier` from the
Codex request, run:

```bash
./run_codex_problem1_normal.sh
```

## Apples-to-apples ultrafast versus normal speed benchmark

Run the balanced comparison with:

```bash
./run_codex_problem1_comparison.sh
```

This performs two exclusive, sequential solves in the order ultrafast →
normal. It pre-pulls the image once, refuses to start
when another Docker container is running, and does not run the official
SWE-bench correctness evaluator. The command makes two Codex model calls.
For clean latency measurements, close other CPU-, disk-, and network-intensive
applications before starting it.

The batch directory under `runs/codex_problem1_comparison_<timestamp>/`
contains `comparison_metadata.json`, `comparison.csv`, and `comparison.md`, as
well as every raw run artifact. Per-attempt and aggregate statistics include
solve, setup, tool, and container-test time; input, cached,
uncached, output, and reasoning tokens; commands and file changes; patch size
and hashes; and Codex exit statuses. Correctness is recorded as not evaluated.

Tool time is measured from host-timestamped Codex JSONL `item.started` and
`item.completed` events. Codex does not expose pure backend inference latency,
so the reported inference-and-orchestration estimate is solve wall time minus
observed tool intervals.

## Problems 1–10 tier batches

Run all ten Verified problems sequentially with ultrafast:

```bash
./run_codex_problems_1_10_ultrafast.sh
```

Then run the same ten problems at normal speed:

```bash
./run_codex_problems_1_10_normal.sh
```

Each command creates a new timestamped directory under `runs/`, continues past
individual solve failures when cleanup succeeds, and writes `batch_metadata.json`,
`batch.csv`, and `batch.md` alongside every raw run artifact. Official
correctness evaluation is disabled; Codex-chosen tests remain part of solve
wall time. Close other resource-intensive applications before each batch.

## Problem 1 interleaved 20-run benchmark

Run ten normal/ultrafast pairs—20 solves total—in strict alternating order:

```bash
./run_codex_problem1_interleaved_20.sh
```

The order is normal → ultrafast, repeated ten times. The runner pre-pulls the
image once, runs exclusively and sequentially, skips official correctness
evaluation, and writes `interleaved_results.csv`, `batch_metadata.json`, and
`batch.md` under a new `runs/codex_problem1_interleaved_20_<timestamp>/`
directory.

## Fast path when Colima is already running

Environment exports are per shell, so set them even when the VM and image are
already warm:

```bash
mkdir -p /tmp/swe-bench-docker-config /tmp/swe-bench-hf-cache
export DOCKER_HOST=unix:///tmp/swe-bench-home/.colima/default/docker.sock
export DOCKER_CONFIG=/tmp/swe-bench-docker-config
export HF_HOME=/tmp/swe-bench-hf-cache

HOME=/tmp/swe-bench-home colima status
docker version --format 'server={{.Server.Version}} arch={{.Server.Arch}}'
docker ps --filter name=minisweagent
./run_sequential_speed.sh --count 1 --prepull
```

If `colima status` says the isolated VM is not running, use the startup command
below. If the requested instance is already known, the separate `--dry-run` is
optional; the real invocation prints its selection before preflight and model
calls. Keep `--prepull` so image acquisition remains outside the recorded
problem time.

## Required managed-Codex setup

This Mac uses an isolated QEMU-backed Colima VM. Before running the harness:

```bash
mkdir -p \
  /tmp/swe-bench-home \
  /tmp/swe-bench-hf-cache \
  /tmp/swe-bench-docker-config

export DOCKER_HOST=unix:///tmp/swe-bench-home/.colima/default/docker.sock
export DOCKER_CONFIG=/tmp/swe-bench-docker-config
export HF_HOME=/tmp/swe-bench-hf-cache

HOME=/tmp/swe-bench-home colima start \
  --cpus 4 \
  --memory 8 \
  --runtime docker \
  --vm-type qemu \
  --mount-type 9p

docker version
```

`docker version` must not print a warning about `~/.docker/config.json`.
That warning corrupts mini-swe-agent's captured command output and prevents it
from recognizing the final submission sentinel.

Optionally confirm, then run the first SWE-bench Verified/test instance:

```bash
./run_sequential_speed.sh --count 1 --dry-run
./run_sequential_speed.sh --count 1 --prepull
```

Before model calls, the harness now starts a disposable container and verifies
that `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` is the first captured output line.
During the timed run it writes passive Docker/container heartbeats to the
per-instance `health.jsonl`. The monitor reports health but never terminates a
run.

After each completed instance, the harness derives and persists trajectory
statistics in `timings.jsonl`, `timings.csv`, `run_metadata.json`, and the
cross-run files under `runs/timed_run_history.*`. Recorded fields include:

- SWE-bench instance ID, title, and full problem statement
- inference provider and exact model
- run label, mode, and speed
- start/end timestamps and wall time
- agent exit status, API calls, tool calls, and test-oriented tool calls
- prompt, completion, total, cache-read, and cache-creation tokens
- approximate inference, tool-execution, and finalization time
- model cost, patch bytes, changed-line counts, commands, and return codes

Terminal failures are recorded too; `Submitted` is the successful agent status.

Do not infer a stall from a quiet `minisweagent.log`; agent steps are not
streamed there. Do not interrupt the foreground harness or remove its container
while a child process may still be running. See the runbook for diagnostics and
cleanup.

Do not wrap the harness in an arbitrary wall-clock timeout. SWE-bench tasks can
run substantially longer than prior examples, especially under QEMU. Once the
preflight checks pass, wait for the harness's own terminal result unless there
is specific evidence that the Docker daemon, container, or model endpoint has
failed.
