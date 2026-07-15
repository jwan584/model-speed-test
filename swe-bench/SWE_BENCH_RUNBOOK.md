# SWE-bench runbook for this repository

This is the known-working procedure for running the repository's timed
SWE-bench harness from the managed Codex environment on this Mac.

## Why the extra setup is required

- Docker Desktop is not installed.
- The installed Docker CLI defaults to `/var/run/docker.sock`, which does not
  exist on this Mac.
- The normal Colima profile lives under `~/.colima`, outside Codex's writable
  roots.
- macOS hardware virtualization is unavailable to the sandbox, so Colima's
  default `vz` VM fails. QEMU works.
- Hugging Face's default cache under `~/.cache` is outside Codex's writable
  roots.
- Docker's default config file at `~/.docker/config.json` is unreadable. This
  is not just cosmetic: Docker writes the warning to stderr, mini-swe-agent
  merges stderr into command output, and the warning appears before the
  `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` sentinel. Submission detection then
  fails and the agent loops until its step or cost limit.
- SWE-bench images are `linux/amd64`; the temporary Colima VM enables the
  required emulation.

Use isolated state under `/tmp` and explicitly set the Docker socket, Docker
config directory, and Hugging Face cache.

## One-time setup per `/tmp` lifetime

Create writable locations and set the required environment variables:

```bash
mkdir -p \
  /tmp/swe-bench-home \
  /tmp/swe-bench-hf-cache \
  /tmp/swe-bench-docker-config

export DOCKER_HOST=unix:///tmp/swe-bench-home/.colima/default/docker.sock
export DOCKER_CONFIG=/tmp/swe-bench-docker-config
export HF_HOME=/tmp/swe-bench-hf-cache
```

Keep these exports in the shell that launches the harness. The harness copies
them into mini-swe-agent's environment.

### Warm-runtime fast path

For subsequent runs within the same `/tmp` and Colima lifetime, do not recreate
the VM or repeat manual sentinel probes. Re-export the three variables above,
then check:

```bash
HOME=/tmp/swe-bench-home colima status
docker version --format 'server={{.Server.Version}} arch={{.Server.Arch}}'
docker ps --filter name=minisweagent
```

If Colima is running, Docker prints no warning, and there are no orphan
`minisweagent` containers, launch the harness directly. Its built-in preflight
rechecks the submission sentinel before model calls. A separate `--dry-run` is
only needed when selection is uncertain; skipping it avoids loading the dataset
twice and creating an extra selection-only run directory.

Start an isolated QEMU-backed Colima VM:

```bash
HOME=/tmp/swe-bench-home colima start \
  --cpus 4 \
  --memory 8 \
  --runtime docker \
  --vm-type qemu \
  --mount-type 9p
```

Do not use the default `colima start`: it tries to update `~/.colima` and is
blocked. Do not use `--vm-type vz`: it fails with "Virtualization is not
available on this hardware" in the managed environment.

Verify the isolated Docker daemon:

```bash
docker version
```

There must be no warning about `~/.docker/config.json`. If that warning appears,
`DOCKER_CONFIG` is missing and the benchmark run must not be started.

## Confirm the selected problem

From the repository root:

```bash
./run_sequential_speed.sh --count 1 --dry-run
```

The expected first Verified/test instance is:

```text
astropy__astropy-12907
```

The harness sorts instance IDs before applying `--start` and `--count`, so
"first" means lexicographically first, not necessarily upstream dataset row 0.

## Run and time the first problem

```bash
./run_sequential_speed.sh --count 1 --prepull
```

`--prepull` downloads the image before `run_problem()` starts its monotonic
timer. The resulting elapsed time therefore excludes image download but
includes container startup and the agent run.

The harness then performs a no-model preflight in a disposable container. It
refuses to start the paid run unless the submission sentinel is the first
captured Docker output line.

The known-good 2026-06-22 run completed this instance with `Submitted` status
in 234.991 seconds, using 26 API calls and $0.3457962. This is a reference point
for comparing runs, never a timeout or abort threshold.

The harness loads `ANTHROPIC_API_KEY` from the environment or, when present,
from `../newsletter-bot/.env.local`. Its default model is
`anthropic/claude-sonnet-4-5-20250929`. Override it with `--model` only when the
request requires a different model.

## Monitor without disrupting the run

The harness prints only start and end events to the terminal. Agent progress is
not streamed to `minisweagent.log`; that file normally contains only container
startup and the final trajectory-save event. A quiet log is therefore not
evidence of a stall.

For this instance, the prior native/default-runtime baseline was 372.608
seconds. QEMU is slower, so do not abort merely because the native baseline has
passed. With `DOCKER_CONFIG` set correctly, completion should occur shortly
after the agent emits its submission command instead of looping.

Never use a previous timing as a kill threshold or add an ad hoc wrapper
timeout. SWE-bench tasks legitimately vary from minutes to much longer, and
QEMU adds further variance. Use the harness's documented per-attempt bound and
heartbeat. A bound hit, dead Docker daemon, missing active container, terminal
API error, or explicit harness exception is persisted as a failed/excluded
attempt; log silence by itself is not failure evidence.

In a separate read-only command, inspect:

```bash
tail -100 runs/sequential_*/001_astropy__astropy-12907/minisweagent.log
```

The harness also records a passive heartbeat every 60 seconds:

```bash
tail -100 runs/sequential_*/001_astropy__astropy-12907/health.jsonl
```

Each JSON line records Docker responsiveness, matching benchmark containers,
and their current process list. This is observational only; the monitor never
terminates the run.

Also check the full captured output when diagnosing a failure:

```bash
tail -100 runs/sequential_*/001_astropy__astropy-12907/harness_stdout_stderr.log
```

Avoid interrupting the foreground harness. An interruption can terminate the
harness while leaving its Docker container running, and no `timings.jsonl` or
prediction files will be written for that incomplete run.

Do not remove a benchmark container while any harness or mini-swe-agent child
may still be running. A detached child can continue making paid model calls
against the deleted container until it reaches its 250-step limit.

### Detect the submission-warning failure

If a run reaches `LimitsExceeded`, returns an empty `model_patch`, or costs much
more than the expected run, inspect the trajectory's tool output. This pattern
means `DOCKER_CONFIG` was not set:

```text
WARNING: Error loading config file: open ~/.docker/config.json: operation not permitted
COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT
```

The sentinel must be the first output line. In the 2026-06-22 failed run, the
agent solved the task and first submitted after about 3.5 minutes, but the
warning prevented detection. It repeated commands until `step_limit: 250`,
cost $3.68, and produced an empty prediction after 821.154 seconds. Do not use
that result as a performance baseline.

On success, report and link these artifacts from the newly created
`runs/sequential_<timestamp>/` directory:

- `timings.jsonl` and `timings.csv`
- `timing_chart.md`
- `preds.json` and `preds.jsonl`
- the per-instance trajectory and logs

Before reporting success, verify that the exit-status YAML says `Submitted`
and that `preds.json` contains a nonempty `model_patch`. A harness return code
of zero alone is insufficient because terminal states such as `LimitsExceeded`
can still be serialized successfully.

`run_metadata.json` is the canonical detailed record for a run. It includes the
problem statement, provider/model, token and cache accounting, API/tool calls,
approximate inference/tool timing, cost, patch statistics, commands, and tool
return codes. `runs/timed_run_history.{jsonl,csv,md}` provides the cross-run
index and is updated after every completed instance, including terminal
failures.

## Native Codex exact lifecycle timing

To run one SWE-bench Verified problem by its one-based number, use:

```bash
bash ./run_codex_swebench_problem Q1 --skip-evaluation
bash ./run_codex_swebench_problem Q2 \
  --model gpt-5.6-sol \
  --reasoning high \
  --skip-evaluation
```

This launcher uses the prebuilt instrumented Codex executable in the sibling
LiveCodeBench checkout; it does not compile Rust. The default model is
`gpt-5.6-sol-ultrafast` with `high` reasoning and no explicit service tier.
Use `--question`, `--model`, `--reasoning`, and `--service-tier` to select a
cohort directly. The question can also remain the first positional argument.
Environment variables documented by
`bash ./run_codex_swebench_problem --help` provide equivalent automation
overrides; explicit CLI options take precedence.

Unlike the managed mini-swe-agent workflow above, the native launcher keeps
its state under the persistent machine-local `~/.swe-bench-runtime`. Each Mac
therefore retains its QEMU Colima VM, Docker layers, Hugging Face cache, pip
cache, and pinned Python runtimes across reboots. Do not place this expanded
state in iCloud. The shared repository contains the requirement locks and
compact results; outputs are separated under `runs/<machine>/`.

`--skip-evaluation` creates only the minimal runtime pinned by
`swebench_runner_requirements.txt`. The official evaluator pinned by
`swebench_eval_requirements.txt` is installed only for runs that request
evaluation. A changed lock fingerprint rebuilds the corresponding local
runtime automatically. Docker pulls, worktree extraction, solve, and optional
evaluation emit passive heartbeats. Per-run worktrees are removed after the
durable patch and timing files are written.

If evaluation is interrupted after a solve has been captured, resume only the
evaluation with:

```bash
bash ./run_codex_swebench_problem Q1 \
  --evaluate-existing runs/<machine>/<completed-run-directory>
```

Each target-model response is one micro-session. The comparable headline is
timed from request dispatch through the terminal response using the
instrumented monotonic duration. Aggregate `end_to_end_billed_tps` is total
provider-billed output tokens divided by the sum of eligible successful-call
durations, not the mean of per-call rates. Warmup, auxiliary-model, failed,
retried, truncated, incomplete, token-missing, and duration-missing calls are
excluded and reported separately. Summed per-call output tokens must reconcile
to final authoritative usage. `response.created` through `response.completed`
is retained only as a secondary diagnostic.
Tool intervals and inference/tool concurrency are reported separately; tool
overlap is never subtracted from a provider lifecycle window.

The terminal emits a passive heartbeat every 30 seconds plus an
`inference_micro_session` record after each completed response. Inspect
`run_metadata.json`, `inference_calls.jsonl`, `tool_intervals.jsonl`,
`codex_timeline.jsonl`, and `codex_stderr.log` in the run directory. A required
timing run exits nonzero if lifecycle coverage or output-token reconciliation
is incomplete. Codex solves default to a 3,600-second `--solve-timeout`; the
watchdog terminates a stuck child, records the timeout, and then executes normal
container/worktree cleanup. Timed-out tasks are excluded from strict cohort
TPS even if earlier internal calls had complete lifecycle records.

Before a native Codex batch, verify `codex login status` under the exact
`CODEX_HOME` the harness will use. Managed runners may be unable to read
`~/.codex`; in that case authenticate the Git-ignored
`LiveCodeBench/.codex-benchmark-home` once with `codex login --device-auth` and
export its absolute path. The batch fails before Q1 if the runner Python,
instrumented binary, or Codex authentication is unavailable, preventing ten
identical setup-only failures.

Run one task smoke before a paid cohort and confirm repository reads plus patch
capture. Under an outer managed macOS sandbox, nested `sandbox-exec` can fail
with `sandbox_apply: Operation not permitted`. In that specific environment,
use `--codex-sandbox danger-full-access` only inside the disposable worktree;
the prompt's Docker command bridge remains the execution boundary. Exclude any
cohort in which repository operations were sandbox-rejected, regardless of
whether timing traces reconciled.

## Recovery and cleanup

List potentially orphaned benchmark containers:

```bash
docker ps --filter name=minisweagent
```

If an interrupted run left one behind, remove the specific container shown by
that command before rerunning:

```bash
docker rm -f <container-id-or-name>
```

Stop the isolated VM when no more runs are needed:

```bash
HOME=/tmp/swe-bench-home colima stop
```

If the temporary Colima profile becomes corrupt, delete only that isolated
profile and recreate it using the setup above:

```bash
HOME=/tmp/swe-bench-home colima delete --force --data
```

Never delete or modify the user's normal `~/.colima` profile as part of this
workflow.
