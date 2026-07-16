# GDPval Timing Harness

This is a serial, provider-native agent harness for the 220-task public
[`openai/gdpval`](https://huggingface.co/datasets/openai/gdpval) gold set. Its primary artifact is a defensible timing trace; generated deliverables are retained under `runs/workspaces/` for later quality review.

## Architecture decision: clean-room, informed by Stirrup

The harness is clean-room rather than a Stirrup extension. Stirrup has good agent and tool boundaries and already records aggregate tool durations, but its provider path is OpenAI Chat Completions or LiteLLM, its client returns a completed response rather than exposing a uniform provider-native stream, and retry timing occurs below the agent boundary. Adding native TTFT, consistent token accounting, and a separately measured rate-limit wait would require replacing its client and retry path while retaining little of its loop. Direct Anthropic Messages, OpenAI Responses, and Gemini GenerateContent adapters provide cleaner measurement boundaries and preserve provider-specific tool state (including Gemini thought signatures). Stirrup's source was reviewed at the current public `main` branch in July 2026.

## Setup

Python 3.11+ is required.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
docker/build-native.sh
cp config.example.yaml config.yaml
```

Set only the keys for enabled providers:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
export BRAVE_API_KEY=...   # required only when the agent calls web_search
```

Keys are read from environment variables, never included in records, and explicitly removed from the container process environment. Docker is mandatory by default. `sandbox.backend: local` plus `allow_local: true` exists for development and is visibly recorded; the harness never silently falls back.

On Apple Silicon, `docker/build-native.sh` builds `linux/arm64` directly; it never uses QEMU/amd64 emulation. The build is serial, defaults to two CPUs and 8 GB, and retries failed builds with exponential backoff. Override those conservative limits with `GDPVAL_BUILD_CPUS`, `GDPVAL_BUILD_MEMORY`, or `GDPVAL_BUILD_ATTEMPTS`. Before spending API tokens, validate the daemon, image architecture, core imports, and system tools:

```bash
gdpval-time --config config.yaml --preflight
```

Preflight fails closed if Docker is absent, the daemon is stopped, the image is missing, the image architecture differs from the host, or a required capability is broken. It does not read API keys, load the dataset, or call a model.

## Run

One task by stable ID:

```bash
gdpval-time --config config.yaml --task-id 854f3814-681c-4950-91ac-55b0db0e3781
```

One task by dataset offset:

```bash
gdpval-time --config config.yaml --offset 0 --limit 1
```

For all 220 tasks set `limit: null`, or pass no CLI subset after doing so. Runs are intentionally serial. Each task record is atomically written to `runs/records/<provider>/<task-id>.json`; with `resume: true`, existing records are skipped. `runs/summary.json` is refreshed after every task, so interruption cannot erase completed work.

Set `run.repetitions` to collect independent serial samples of each provider/task pair. Repeated records use filenames such as `<task-id>.run-001.json`, and each repetition gets a clean workspace. The `gdp-10` mini-set is recorded in `minisets/gdp-10.yaml`; `config.gdp-10.yaml` runs the set and `config.gdp-10-task1-3x.yaml` runs its first task three times per provider.

### Economy guardrails

Benchmark configs should enable `run.economy`. The standard policy asks every run to validate and finish at turn 20, then allows three full grace turns before recording `deliverable_unfinished` (files exist) or `completion_unlikely` (none exist). It also stops after three consecutive inference turns without tools, five turns without artifact progress after turn 12, or five consecutive tool failures; the third repetition of one normalized tool error gets an explicit change-approach warning. For repeated samples, two consecutive structural failures with zero deliverables skip the remaining repetitions for that provider/task pair. These are explicit recorded outcomes; the harness never relabels them as successful completions.

Each provider stream has two independent safety ceilings. `inference_idle_timeout_seconds` defaults to 300 seconds and advances on every SSE line, including hidden-reasoning/metadata events, so a long active reasoning stream is not mistaken for a stall. `inference_absolute_timeout_seconds` defaults to 1,800 seconds per API call. An expiry is recorded as `inference_timeout` with the incomplete call duration and is never automatically retried. Historical tool results are compacted without breaking assistant/tool pairing: only the latest image payload is retained, tool text older than eight tool exchanges is capped at 4,000 characters, and original workspace files remain intact.

## Timing methodology

All durations use the monotonic `perf_counter`; UTC timestamps are labels only.

- `total_wall_seconds`: task dispatch through successful `finish`, max turns, error, or timeout.
- `inference_seconds`: sum of complete native streaming request latency, from request dispatch through the terminal stream event. A timeout during an active request is retained as an incomplete inference call.
- `first_stream_event_seconds`: dispatch to the first SSE event of any kind, including metadata or hidden-thinking events. `ttft_seconds` and `first_observable_output_seconds` are dispatch to the first non-empty assistant text or tool-argument delta. `last_observable_output_seconds` marks the final such delta. The record also includes exact observable chunk, Unicode-character, and UTF-8-byte counts.
- The cross-provider headline is `overall_request_active_billed_tps`: final provider-billed output tokens divided by monotonic request-dispatch-to-terminal time, as a ratio of sums. It is strict and null unless every call completed normally on its first attempt with final usage, a positive duration, and valid token accounting. Failed/retried, timed-out, truncated, refused, filtered, token-missing, and reconciliation-mismatched calls are explicit exclusions; partial coverage is diagnostic only.
- `active_generation_billed_tps` remains a provider-specific secondary diagnostic: terminal provider-billed output tokens divided by terminal-event time minus the provider-observed generation boundary. OpenAI uses reasoning-item start; Anthropic uses thinking-block start. Do not compare it across providers unless their reasoning-aware boundaries are proven equivalent.
- TTFT and fixed-tokenizer `post_ttft_tokens_per_second`, character/sec, and byte/sec are diagnostics for observable output only. Never subtract visible TTFT from request time to claim reasoning-inclusive decode speed.
- Both task-level billed TPS fields are ratios of summed provider `usage.output_tokens` to summed eligible durations, never averages of per-call rates. Strict fields are null if any call is ineligible. Partial values, exclusion reasons, eligible-call coverage, and eligible-token coverage remain available for diagnosis but must be labeled partial.
- Each adapter retains the provider's authoritative terminal usage. Per-call billed output is reconciled against it; a mismatch disqualifies strict TPS. Reasoning/thinking tokens are reconciled separately when available and remain a subset of billed output, never an added numerator.
- Anthropic adaptive thinking is requested with `display: omitted`; a `content_block_start` for thinking/redacted thinking remains the generation boundary even when no thinking text is exposed. `thinking_tokens` and OpenAI `reasoning_tokens` are informational subsets of inclusive billed `output_tokens` and are never added twice.
- `billed_output_tokens` preserves the provider-native count for billing and provider-internal analysis; it is not cross-provider comparable. `reasoning_tokens` preserves a separately reported hidden-thinking count. `non_reasoning_output_tokens` is populated only when the API separates reasoning from output (OpenAI and Anthropic when `output_tokens_details.thinking_tokens` is present) or specifies a non-overlapping candidate counter (Gemini). Never substitute it for the fixed-tokenizer comparison.
- `tool_seconds`: measured around each tool executor (`run_shell`, `web_search`, `web_fetch`, `view_image`, and `finish`). Calls execute serially even when a model emits several at once.
- `backoff_seconds`: actual exponential-backoff sleep after retryable statuses. It is excluded from inference and tools.
- `retry_api_seconds`: time consumed by failed retryable request attempts, also excluded from successful inference.
- `harness_overhead_seconds`: the visible residual: total minus inference, tools, backoff, and failed-attempt API time. It includes dataset staging, serialization, loop bookkeeping, sandbox startup/teardown, and unclassified failure paths.

Warmup runs once per enabled provider before recorded tasks and is reported as provider context, not charged to any task. Region/endpoint, model, sandbox backend/image, per-call request IDs, token counts, tool type, and retry counts are recorded. Summary latency uses median and p90 rather than means.

### Benchmark reasoning policy

Benchmark runs use `xhigh` reasoning effort consistently across comparable providers. Set `run.required_reasoning_effort: xhigh`; the harness then fails before warmup if any enabled provider does not explicitly declare `reasoning_effort: xhigh`. Anthropic Opus runs also set `adaptive_thinking: true`; Fable uses always-on adaptive thinking; OpenAI Responses requests send `reasoning: {effort: xhigh}`. Do not compare a run lacking these explicit settings with an xhigh run. Provider families that do not expose a semantically equivalent `xhigh` control require a separate labeled cohort rather than silently using their default.

## Sandbox and tools

Each task gets a persistent container with references staged under `/workspace/input` and output requested under `/workspace/output`. The image follows GDPval paper appendix A.6.4: the document/data/media Python stack, LibreOffice, Noto fonts, FFmpeg, OCR/PDF utilities, CadQuery, and the additional packages named by the paper. It also includes Node, Java, R, and Ruby; a task needing another unavailable runtime should end as an explicit tool error, not be silently counted as a model failure.

Runtime containers are CPU-, memory-, swap-, and PID-limited; capability-free, non-root, read-only outside `/workspace` and `/tmp`, and configured with `no-new-privileges`. API credentials stay in the host harness and are never passed through `docker run` or `docker exec`. Network access remains enabled because GDPval exposes web tools, but a no-network preflight verifies that image health does not depend on external state. Keep the same limits for every compared model. The image health check intentionally covers core document, spreadsheet, PDF, image, audio/video, scientific Python, and non-Python runtimes; a package outside that set can still fail at task time and should be reported as an environment capability failure.

The published package versions are old enough to contain internal conflicts: notably, appendix-era `librosa==0.8.1` cannot import with NumPy 1.24. The image pins NumPy 1.23.5 and otherwise prefers binary distributions to avoid uncontrolled source builds. `rdkit==2024.9.6` has no Linux arm64 wheel, so arm64 uses the nearest native release, 2024.3.2. `cadquery-ocp==7.7.0` and `aspose-words==25.8.0` do not publish Linux arm64 wheels; the native image omits them, reports both under `unavailable_optional_capabilities`, and CAD- or Aspose-dependent tasks must be tagged and skipped rather than counted as model failures. Architecture-specific failures should be fixed with a documented native-compatible version and rerun through preflight, never hidden by switching to an emulated amd64 image.

`web_fetch` accepts public HTTP(S) URLs. `web_search` uses Brave and returns a tool error if its key is absent. `view_image` returns both metadata and provider-native image content. Deliverables are indexed with absolute/relative paths, sizes, and SHA-256 hashes.

## Add a provider

Subclass `Provider` in `src/gdpval_timing/providers/`, implement one native streaming request that returns a normalized `Message`, first-content timestamp, provider usage fields, and request ID, then register it in `PROVIDERS`. Keep schema/history translation inside the adapter. Retry only genuinely transient statuses by raising `RetryableProviderError`; the base class will account failed request time and backoff separately.

## Verification

```bash
python -m compileall -q src tests
pytest -q
```

API behavior was checked against the official [Anthropic streaming docs](https://platform.claude.com/docs/en/build-with-claude/streaming), [OpenAI function-calling guide](https://developers.openai.com/api/docs/guides/function-calling), and [Gemini function-calling guide](https://ai.google.dev/gemini-api/docs/function-calling). The environment list comes from [GDPval appendix A.6.4](https://cdn.openai.com/pdf/d5eb7428-c4e9-4a33-bd86-86dd4bcf12ce/GDPval.pdf).
