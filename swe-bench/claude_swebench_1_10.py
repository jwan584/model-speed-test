#!/usr/bin/env python3
"""Run Claude Code on sorted SWE-bench Verified Q1-Q10 sequentially."""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
ADAPTER = ROOT / "claude_swebench_problem1.py"
SUCCESS_STATUSES = {"completed_without_evaluation"}


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return {"status": "missing_metadata", "output_dir": str(run_dir)}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "invalid_metadata",
            "output_dir": str(run_dir),
            "metadata_error": str(error),
        }
    return value if isinstance(value, dict) else {"status": "invalid_metadata"}


def upgrade_terminal_fallback_metadata(run_dir: Path) -> dict[str, Any]:
    """Apply the all-model terminal fallback to a preserved pre-fallback run."""
    metadata = load_metadata(run_dir)
    timing = metadata.get("inference_timing") or {}
    if timing.get("timing_basis") == "claude_cli_terminal_duration_api_ms_all_models":
        return metadata
    events_path = run_dir / "claude_events.jsonl"
    terminal_events = [
        event
        for event in load_json_lines(events_path)
        if event.get("type") == "result"
    ]
    if not terminal_events:
        return metadata
    result = terminal_events[-1]
    model_usage = result.get("modelUsage") or {}
    duration_ms = result.get("duration_api_ms")
    model_tokens: dict[str, int] = {}
    for model, usage in model_usage.items():
        value = (usage or {}).get("outputTokens")
        if not isinstance(value, int) or value < 0:
            return metadata
        model_tokens[str(model)] = value
    if (
        result.get("is_error") is True
        or not model_tokens
        or not isinstance(duration_ms, (int, float))
        or duration_ms <= 0
    ):
        return metadata
    seconds = duration_ms / 1000
    output_tokens = sum(model_tokens.values())
    prior_reasons = list(timing.get("coverage_reasons") or [])
    has_target_otel_diagnostic = "primary_request_seconds_sum" in timing
    timing.update(
        {
            "timing_basis": "claude_cli_terminal_duration_api_ms_all_models",
            "all_model_terminal_billed_output_tokens": output_tokens,
            "all_model_terminal_output_tokens_by_model": model_tokens,
            "terminal_request_active_seconds": seconds,
            "end_to_end_inference_seconds": seconds,
            "end_to_end_billed_tps": output_tokens / seconds,
            "coverage": "complete",
            "coverage_reasons": [],
        }
    )
    if has_target_otel_diagnostic:
        timing["target_otel_diagnostic_coverage"] = (
            "complete" if not prior_reasons else "partial"
        )
        timing["target_otel_diagnostic_coverage_reasons"] = prior_reasons
    else:
        timing["target_otel_diagnostic_coverage"] = "unavailable"
        timing["target_otel_diagnostic_coverage_reasons"] = [
            "post_terminal_teardown_before_otel_normalization"
        ]
    metadata["inference_timing"] = timing
    metadata["inference_timing_complete"] = True
    patch_path = run_dir / "model.patch"
    if patch_path.exists() and "patch" not in metadata:
        patch = patch_path.read_text()
        metadata["patch"] = {
            "bytes": len(patch.encode()),
            "added_lines": sum(
                line.startswith("+") and not line.startswith("+++")
                for line in patch.splitlines()
            ),
            "removed_lines": sum(
                line.startswith("-") and not line.startswith("---")
                for line in patch.splitlines()
            ),
            "nonempty": bool(patch.strip()),
        }
    if metadata.get("status") in {
        "completed_with_incomplete_inference_timing",
        "failed",
    } and (metadata.get("patch") or {}).get("nonempty"):
        metadata["status"] = "completed_without_evaluation"
        metadata["recovered_terminal_success_after_teardown_hang"] = True
    write_json(run_dir / "run_metadata.json", metadata)
    return metadata


def load_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    values = []
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def refresh_resumed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refreshed = []
    for row in rows:
        paths = [
            Path(value)
            for value in str(row.get("attempt_run_metadata_paths") or "").split(";")
            if value
        ]
        attempts = []
        for path in paths:
            metadata = upgrade_terminal_fallback_metadata(path.parent)
            timing = metadata.get("inference_timing") or {}
            attempts.append(
                {
                    "run_metadata_path": str(path),
                    "status": metadata.get("status"),
                    "request_active_seconds": timing.get(
                        "terminal_request_active_seconds"
                    ),
                    "billed_output_tokens": timing.get(
                        "all_model_terminal_billed_output_tokens"
                    ),
                    "end_to_end_billed_tps": timing.get(
                        "end_to_end_billed_tps"
                    ),
                    "target_otel_reconciliation": timing.get(
                        "output_token_reconciliation"
                    ),
                }
            )
        if paths:
            canonical_path = paths[-1]
            metadata = load_metadata(canonical_path.parent)
            current = result_row(
                int(row["question_number"]),
                canonical_path.parent,
                metadata,
                0 if metadata.get("status") in SUCCESS_STATUSES else 3,
            )
            current["attempt_count"] = len(paths)
            current["attempt_return_codes"] = row.get("attempt_return_codes")
            current["attempt_run_metadata_paths"] = row.get(
                "attempt_run_metadata_paths"
            )
            current["valid_terminal_attempts_json"] = json.dumps(
                attempts, sort_keys=True
            )
            refreshed.append(current)
        else:
            refreshed.append(row)
    return refreshed


def result_row(
    question_number: int, run_dir: Path, metadata: dict[str, Any], returncode: int
) -> dict[str, Any]:
    timing = metadata.get("inference_timing") or {}
    partition = timing.get("wall_partition") or {}
    patch = metadata.get("patch") or {}
    problem = metadata.get("problem") or {}
    container = metadata.get("container_commands") or {}
    request_seconds = timing.get("terminal_request_active_seconds")
    output_tokens = timing.get("all_model_terminal_billed_output_tokens")
    return {
        "question_number": question_number,
        "dataset_index": question_number - 1,
        "instance_id": problem.get("instance_id"),
        "model_requested": metadata.get("requested_model"),
        "model_resolved": metadata.get("resolved_model"),
        "effort": metadata.get("requested_effort"),
        "status": metadata.get("status", "missing_metadata"),
        "return_code": returncode,
        "solve_wall_seconds": metadata.get("solve_seconds"),
        "request_active_seconds": request_seconds,
        "target_otel_request_duration_seconds_diagnostic": timing.get(
            "primary_request_seconds_sum"
        ),
        "end_to_end_billed_tps": timing.get("end_to_end_billed_tps"),
        "target_eligible_call_count": timing.get("primary_call_count"),
        "target_observed_call_count": timing.get("observed_target_call_count"),
        "target_excluded_call_count": timing.get("excluded_target_call_count"),
        "auxiliary_call_count": timing.get("auxiliary_call_count"),
        "billed_output_tokens": output_tokens,
        "target_otel_output_tokens_diagnostic": timing.get(
            "primary_output_tokens"
        ),
        "target_terminal_output_tokens": timing.get(
            "model_usage_primary_output_tokens"
        ),
        "total_tool_seconds": timing.get("total_tool_seconds"),
        "request_only_wall_seconds": partition.get("llm_request_only_seconds"),
        "tool_only_wall_seconds": partition.get("tool_only_seconds"),
        "request_tool_overlap_seconds": partition.get(
            "llm_request_tool_overlap_seconds"
        ),
        "orchestration_residual_seconds": partition.get(
            "orchestration_residual_seconds"
        ),
        "provider_reported_ttft_ms_median": timing.get(
            "provider_reported_ttft_ms_median"
        ),
        "timing_coverage": timing.get("coverage"),
        "timing_coverage_reasons": ";".join(timing.get("coverage_reasons") or []),
        "output_token_reconciliation": timing.get("output_token_reconciliation"),
        "target_otel_diagnostic_coverage": timing.get(
            "target_otel_diagnostic_coverage"
        ),
        "target_otel_diagnostic_coverage_reasons": ";".join(
            timing.get("target_otel_diagnostic_coverage_reasons") or []
        ),
        "patch_bytes": patch.get("bytes"),
        "patch_added_lines": patch.get("added_lines"),
        "patch_removed_lines": patch.get("removed_lines"),
        "container_command_count": container.get("count"),
        "container_test_count": container.get("test_count"),
        "container_test_seconds": container.get("test_seconds"),
        "run_metadata_path": str(run_dir / "run_metadata.json"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row.get("status") in SUCCESS_STATUSES
        and row.get("return_code") == 0
        and row.get("timing_coverage") == "complete"
        and isinstance(row.get("request_active_seconds"), (int, float))
        and float(row["request_active_seconds"]) > 0
        and isinstance(row.get("billed_output_tokens"), (int, float))
    ]
    request_seconds = sum(
        float(row["request_active_seconds"]) for row in eligible
    )
    output_tokens = sum(int(row["billed_output_tokens"]) for row in eligible)
    return {
        "attempted_problems": len(rows),
        "successful_problems": sum(
            row.get("status") in SUCCESS_STATUSES and row.get("return_code") == 0
            for row in rows
        ),
        "complete_timing_problems": len(eligible),
        "request_active_seconds_sum": round(request_seconds, 6),
        "billed_output_tokens_sum": output_tokens,
        "end_to_end_billed_tps_ratio_of_sums": (
            output_tokens / request_seconds if request_seconds else None
        ),
        "timing_basis": "claude_cli_terminal_duration_api_ms_all_models",
        "server_engine_equivalent": False,
    }


def render_markdown(batch: dict[str, Any]) -> str:
    lines = [
        "# Claude Code SWE-bench Verified Q1-Q10",
        "",
        f"Model: `{batch['model']}`  ",
        f"Effort: `{batch['effort']}`  ",
        "Official correctness evaluation: skipped",
        "",
        "| Q | Instance | Status | Solve s | Request s | Output | End-to-end billed TPS | Coverage | Reconciliation |",
        "|---:|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in batch["runs"]:
        def show(key: str) -> str:
            value = row.get(key)
            return "" if value is None else str(value)

        lines.append(
            f"| {row['question_number']} | {show('instance_id')} | {show('status')} | "
            f"{show('solve_wall_seconds')} | {show('request_active_seconds')} | "
            f"{show('billed_output_tokens')} | {show('end_to_end_billed_tps')} | "
            f"{show('timing_coverage')} | {show('output_token_reconciliation')} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            "```json",
            json.dumps(batch["aggregate"], indent=2, sort_keys=True),
            "```",
            "",
            "*Claude headline TPS is authoritative terminal billed output across all "
            "models divided by terminal `duration_api_ms` (ratio of sums). It is "
            "client/API request-active throughput, not server-engine decode throughput. "
            "Target-Fable OTel TPS is diagnostic only.*",
            "",
        ]
    )
    return "\n".join(lines)


def checkpoint_batch(
    output: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    """Persist a resumable manifest and human/machine-readable snapshots."""
    manifest["runs"] = rows
    manifest["aggregate"] = summarize(rows)
    write_json(output / "batch_metadata.json", manifest)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output / "batch.csv").open("w", newline="") as csv_file:
        if fieldnames:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    (output / "batch.md").write_text(render_markdown(manifest))


def run_streaming(
    command: list[str],
    log_path: Path,
    env: dict[str, str],
    timeout_seconds: float | None = None,
) -> int:
    """Stream output, adding a passive heartbeat during quiet periods."""
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert process.stdout is not None
        messages: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            for line in process.stdout:
                messages.put(line)
            messages.put(None)

        threading.Thread(target=read_output, daemon=True).start()
        started = time.monotonic()
        try:
            while True:
                elapsed = time.monotonic() - started
                if timeout_seconds is not None and elapsed >= timeout_seconds:
                    message = json.dumps(
                        {
                            "status": "batch_child_timeout",
                            "elapsed_s": round(elapsed, 1),
                            "timeout_s": timeout_seconds,
                        }
                    ) + "\n"
                    sys.stdout.write(message)
                    sys.stdout.flush()
                    log.write(message)
                    log.flush()
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    return 124
                try:
                    wait_seconds = 30.0
                    if timeout_seconds is not None:
                        wait_seconds = min(
                            wait_seconds, max(0.1, timeout_seconds - elapsed)
                        )
                    line = messages.get(timeout=wait_seconds)
                except queue.Empty:
                    line = json.dumps(
                        {
                            "status": "batch_child_running",
                            "elapsed_s": round(time.monotonic() - started, 1),
                        }
                    ) + "\n"
                if line is None:
                    break
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        return process.wait()


def pull_image(image: str, env: dict[str, str], log_path: Path) -> int:
    print(f"[prepull] {image}", flush=True)
    return run_streaming(["docker", "pull", image], log_path, env)


def run_batch(
    *,
    output: Path,
    python: Path,
    model: str,
    effort: str,
    env: dict[str, str],
    dry_run: bool = False,
    adapter: Path = ADAPTER,
    problem_loader: Callable[[int], dict[str, Any]] | None = None,
    command_runner: Callable[
        [list[str], Path, dict[str, str], float | None], int
    ] = run_streaming,
    image_puller: Callable[[str, dict[str, str], Path], int] = pull_image,
    problem_timeout_seconds: float = 3600,
    max_attempts: int = 2,
    resume: bool = False,
    start_question: int = 1,
) -> dict[str, Any]:
    if problem_loader is None:
        from claude_swebench_problem1 import load_problem

        problem_loader = load_problem
    if resume:
        manifest_path = output / "batch_metadata.json"
        if not manifest_path.exists():
            raise SystemExit(f"BLOCKER: resume manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        rows = refresh_resumed_rows(list(manifest.get("runs") or []))
        manifest["status"] = "running"
        manifest["resumed_at"] = iso_now()
        checkpoint_batch(output, manifest, rows)
    else:
        output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "batch_metadata.json"
    if not resume:
        manifest = {
            "schema_version": 2,
            "batch_id": output.name,
            "machine_id": env.get("SWE_BENCH_MACHINE_ID"),
            "model": model,
            "effort": effort,
            "problem_numbers": list(range(1, 11)),
            "exclusive_sequential_solves": True,
            "official_evaluation_enabled": False,
            "complete_inference_timing_required": True,
            "problem_timeout_seconds": problem_timeout_seconds,
            "max_attempts": max_attempts,
            "created_at": iso_now(),
            "status": "dry_run" if dry_run else "running",
            "runs": [],
        }
        rows = []
        checkpoint_batch(output, manifest, rows)
    try:
      for question_number in range(start_question, 11):
        problem = problem_loader(question_number - 1)
        run_dir = output / f"q{question_number:02d}_{problem['instance_id']}_attempt01"
        def build_command(attempt_run_dir: Path) -> list[str]:
            return [
                str(python),
                str(adapter),
                "--index",
                str(question_number - 1),
                "--model",
                model,
                "--effort",
                effort,
                "--require-complete-inference-timing",
                "--skip-evaluation",
                "--skip-pull",
                "--worktree-root",
                str(Path(env["SWE_BENCH_STATE_ROOT"]) / "worktrees"),
                "--output",
                str(attempt_run_dir),
            ]
        command = build_command(run_dir)
        if dry_run:
            print("DRY_RUN " + " ".join(command), flush=True)
            row = {
                "question_number": question_number,
                "dataset_index": question_number - 1,
                "instance_id": problem["instance_id"],
                "status": "dry_run",
                "return_code": None,
                "run_metadata_path": str(run_dir / "run_metadata.json"),
            }
        else:
            print(
                f"\n=== Q{question_number}/Q10 {problem['instance_id']} ===",
                flush=True,
            )
            pull_code = image_puller(
                problem["image"], env, output / f"q{question_number:02d}_docker_pull.log"
            )
            if pull_code != 0:
                returncode = pull_code
                metadata = {"status": "docker_pull_failed", "problem": problem}
                attempt_return_codes = [returncode]
                attempt_paths: list[str] = []
            else:
                attempt_return_codes = []
                attempt_paths = []
                metadata = {}
                for attempt in range(1, max_attempts + 1):
                    run_dir = output / (
                        f"q{question_number:02d}_{problem['instance_id']}"
                        f"_attempt{attempt:02d}"
                    )
                    command = build_command(run_dir)
                    returncode = command_runner(
                        command,
                        output / f"q{question_number:02d}_attempt{attempt:02d}.log",
                        env,
                        problem_timeout_seconds,
                    )
                    metadata = load_metadata(run_dir)
                    attempt_return_codes.append(returncode)
                    attempt_paths.append(str(run_dir / "run_metadata.json"))
                    retryable = returncode in {2, 3, 124} or metadata.get(
                        "status"
                    ) in {"missing_metadata", "invalid_metadata"}
                    if returncode == 0 or not retryable or attempt == max_attempts:
                        break
                    print(
                        f"[retry] Q{question_number} attempt {attempt + 1}/"
                        f"{max_attempts} after return code {returncode}",
                        flush=True,
                    )
            row = result_row(question_number, run_dir, metadata, returncode)
            row["attempt_count"] = len(attempt_return_codes)
            row["attempt_return_codes"] = ";".join(
                str(code) for code in attempt_return_codes
            )
            row["attempt_run_metadata_paths"] = ";".join(attempt_paths)
        rows.append(row)
        checkpoint_batch(output, manifest, rows)

    except KeyboardInterrupt:
        manifest["status"] = "paused"
        manifest["paused_at"] = iso_now()
        checkpoint_batch(output, manifest, rows)
        raise

    aggregate = summarize(rows)
    manifest.update(
        {
            "status": (
                "dry_run"
                if dry_run
                else "completed"
                if aggregate["successful_problems"] == 10
                else "completed_with_failures"
            ),
            "completed_at": iso_now(),
            "aggregate": aggregate,
        }
    )
    checkpoint_batch(output, manifest, rows)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="claude-fable-5")
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--problem-timeout-seconds", type=float, default=3600)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--resume-output", type=Path)
    parser.add_argument("--start-question", type=int, default=1)
    args = parser.parse_args()
    env = os.environ.copy()
    state_root = Path(env.get("SWE_BENCH_STATE_ROOT", Path.home() / ".swe-bench-runtime"))
    env["SWE_BENCH_STATE_ROOT"] = str(state_root)
    machine = env.get("SWE_BENCH_MACHINE_ID") or socket.gethostname().split(".")[0]
    safe_machine = "".join(c if c.isalnum() or c in "_.-" else "-" for c in machine).strip("-").lower()
    env["SWE_BENCH_MACHINE_ID"] = safe_machine
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (
        args.resume_output
        or args.output
        or ROOT / "runs" / safe_machine / f"claude_verified_q01_q10_fable5_xhigh_{stamp}"
    ).resolve()
    batch = run_batch(
        output=output,
        # Keep the virtualenv launcher path intact. Resolving its symlink to the
        # framework binary would discard virtualenv package discovery.
        python=args.python.absolute(),
        model=args.model,
        effort=args.effort,
        env=env,
        dry_run=args.dry_run,
        problem_timeout_seconds=args.problem_timeout_seconds,
        max_attempts=args.max_attempts,
        resume=args.resume_output is not None,
        start_question=args.start_question,
    )
    print(f"BATCH_METADATA={output / 'batch_metadata.json'}")
    print(f"BATCH_CSV={output / 'batch.csv'}")
    print(f"BATCH_MARKDOWN={output / 'batch.md'}")
    return 0 if args.dry_run or batch["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
