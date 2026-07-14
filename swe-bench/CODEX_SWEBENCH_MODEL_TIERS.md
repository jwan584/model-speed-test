# Codex model and service-tier selection for SWE-bench

This guide defines how to compare normal and ultrafast Codex runs while holding
the model and reasoning effort fixed.

## Known configuration

The ChatGPT-authenticated Codex service on this machine accepted:

```text
model: koffing-updated
reasoning effort: medium
service tier: ultrafast
```

`koffing-updated` and `ultrafast` are not public OpenAI API identifiers. They
were accepted through `codex exec` using the saved ChatGPT/Codex login. Treat
them as requested internal configuration, not independently verified backend
metadata: Codex JSONL currently does not report the resolved model or service
tier.

The short word `koffing` may be used in experiment names, but the command must
request the exact tested model ID `koffing-updated`.

## Configuration dimensions

Keep these independent:

| Dimension | Normal condition | Ultrafast condition |
|---|---|---|
| Model | `koffing-updated` | `koffing-updated` |
| Reasoning | `medium` | `medium` |
| Service tier | omitted | `ultrafast` |
| Public Fast mode | disabled | disabled |
| Authentication | saved ChatGPT/Codex login | saved ChatGPT/Codex login |

Do not map ultrafast to public Fast mode for this comparison. Enabling
`features.fast_mode` while requesting `service_tier="ultrafast"` would change
two settings and make the result ambiguous.

## Probe both conditions

Normal/default service behavior:

```bash
./test_codex_model.py \
  --model koffing-updated \
  --reasoning medium \
  --speed-label normal \
  --service-tier default
```

Ultrafast request:

```bash
./test_codex_model.py \
  --model koffing-updated \
  --reasoning medium \
  --speed-label ultrafast \
  --service-tier ultrafast
```

`default` means the probe omits `service_tier` completely. The ultrafast probe
passes `service_tier="ultrafast"` exactly. A successful response proves the
request was accepted, but not which backend tier was ultimately resolved.

## Equivalent `codex exec` arguments

Arguments common to both conditions:

```bash
codex exec \
  --json \
  --ephemeral \
  --ignore-user-config \
  --model koffing-updated \
  --config 'model_reasoning_effort="medium"' \
  --config 'approval_policy="on-request"' \
  --sandbox workspace-write \
  --cd "$TASK_WORKTREE" \
  "$TASK_PROMPT"
```

For ultrafast only, add:

```bash
--config 'service_tier="ultrafast"'
```

`--ignore-user-config` prevents a personal default model, reasoning effort, or
service tier from contaminating the comparison. Authentication is still reused.
Repository `AGENTS.md` instructions remain part of the task environment.

## SWE-bench experiment design

`run_sequential_speed.sh` uses mini-swe-agent and does not invoke Codex. Its
existing `--mode` and `--speed` arguments are metadata only. For a one-command
Codex run of Verified problem 1 using ultrafast, use:

```bash
./run_codex_problem1_ultrafast.sh
```

The launcher calls `codex_swebench_problem1.py`, which invokes `codex exec` with
the arguments above and evaluates the resulting patch with the official
SWE-bench harness.

For every instance and condition:

1. Select the same SWE-bench dataset, split, instance, and base commit.
2. Pre-pull the same Docker image before timing.
3. Materialize an identical clean task worktree.
4. Expose only the problem statement; never expose the gold patch or hidden
   evaluation patch.
5. Start solve timing immediately before spawning `codex exec`.
6. Stop solve timing when Codex exits and the patch is captured.
7. Save `git diff` in standard SWE-bench prediction format.
8. Run the official SWE-bench evaluator after solving.
9. Record both speed and correctness. Never silently exclude failed or
   unresolved attempts from timing results.

Record setup, solve, requested-test, and official-evaluation time separately.
The primary model-speed metric should exclude image download and worktree setup
but include the complete Codex solve through final patch production.

## Pairing and repetition

Model output and infrastructure latency vary. Do not draw a conclusion from one
normal run and one ultrafast run.

- Use identical instances in both conditions.
- Interleave order (for example normal, ultrafast, ultrafast, normal) to reduce
  warm-cache and time-of-day bias.
- Run multiple paired repetitions when budget permits.
- Report median, mean, range, and every raw run.
- Compare pass rate as well as elapsed time. A faster unresolved patch is not a
  successful speedup.

## Required metadata

Persist at least:

```json
{
  "requested_model": "koffing-updated",
  "requested_reasoning_effort": "medium",
  "requested_service_tier": "default-or-ultrafast",
  "resolved_model": null,
  "resolved_service_tier": null,
  "codex_version": "...",
  "auth_mode": "chatgpt",
  "instance_id": "...",
  "base_commit": "...",
  "started_at": "...",
  "ended_at": "...",
  "solve_seconds": 0,
  "evaluation_seconds": 0,
  "input_tokens": 0,
  "cached_input_tokens": 0,
  "output_tokens": 0,
  "reasoning_output_tokens": 0,
  "exit_status": "...",
  "resolved": false
}
```

Use `requested_*` names intentionally. Until Codex emits resolved backend
metadata, do not claim that acceptance proves which internal tier served the
request.
