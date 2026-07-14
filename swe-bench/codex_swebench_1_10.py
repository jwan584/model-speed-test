#!/usr/bin/env python3
"""Run SWE-bench Verified problems 1 through 10 sequentially at one Codex tier."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "mini-swe-agent" / ".venv" / "bin" / "python"
ADAPTER = ROOT / "codex_swebench_problem1.py"
MODEL = "koffing-updated"
REASONING = "medium"
INSTRUMENTED_CODEX = (
    ROOT.parent
    / "live-code-bench"
    / "LiveCodeBench"
    / ".codex-instrumented"
    / "codex-v0.144.3"
)


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_streaming(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return process.wait()


def command_output(command: list[str], env: dict[str, str]) -> str:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout.strip()


def running_containers(env: dict[str, str]) -> list[str]:
    output = command_output(["docker", "ps", "--format", "{{.ID}} {{.Names}}"], env)
    return [line for line in output.splitlines() if line.strip()]


def load_metadata(run_dir: Path) -> dict:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return {"status": "missing_metadata", "output_dir": str(run_dir)}
    return json.loads(path.read_text())


def result_row(problem_number: int, tier: str, run_dir: Path, metadata: dict) -> dict:
    usage = metadata.get("codex", {}).get("usage") or {}
    timing = metadata.get("timing_breakdown") or {}
    inference = metadata.get("inference_timing") or {}
    container = metadata.get("container_commands") or {}
    patch = metadata.get("patch") or {}
    problem = metadata.get("problem") or {}
    input_tokens = usage.get("input_tokens")
    cached_tokens = usage.get("cached_input_tokens")
    return {
        "problem_number": problem_number,
        "dataset_index": problem_number - 1,
        "instance_id": problem.get("instance_id"),
        "tier": tier,
        "status": metadata.get("status"),
        "correctness": "not_evaluated",
        "setup_seconds": metadata.get("setup_seconds"),
        "solve_seconds": metadata.get("solve_seconds"),
        "tool_seconds": inference.get(
            "total_tool_seconds", timing.get("tool_seconds")
        ),
        "inference_and_orchestration_seconds_estimate": timing.get(
            "inference_and_orchestration_seconds_estimate"
        ),
        "provider_window_inference_seconds": inference.get(
            "provider_window_inference_seconds"
        ),
        "provider_window_output_tps": inference.get(
            "provider_window_output_tps"
        ),
        "provider_window_output_tokens": inference.get(
            "primary_output_tokens"
        ),
        "inference_timing_coverage": inference.get("coverage"),
        "inference_micro_sessions": inference.get("primary_eligible_call_count"),
        "inference_tool_concurrency_seconds": inference.get(
            "target_inference_tool_concurrency_seconds"
        ),
        "container_test_seconds": container.get("test_seconds"),
        "container_test_count": container.get("test_count"),
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "uncached_input_tokens": (
            input_tokens - cached_tokens
            if isinstance(input_tokens, int) and isinstance(cached_tokens, int)
            else None
        ),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
        "command_executions": metadata.get("codex", {}).get("command_executions"),
        "file_changes": metadata.get("codex", {}).get("file_changes"),
        "patch_bytes": patch.get("bytes"),
        "patch_added_lines": patch.get("added_lines"),
        "patch_removed_lines": patch.get("removed_lines"),
        "codex_returncode": metadata.get("codex_returncode"),
        "run_dir": str(run_dir),
    }


def summarize(rows: list[dict]) -> dict:
    fields = [
        "setup_seconds",
        "solve_seconds",
        "tool_seconds",
        "inference_and_orchestration_seconds_estimate",
        "provider_window_inference_seconds",
        "provider_window_output_tokens",
        "inference_micro_sessions",
        "inference_tool_concurrency_seconds",
        "container_test_seconds",
        "input_tokens",
        "cached_input_tokens",
        "uncached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
    ]
    result = {
        "attempts": len(rows),
        "completed_solves": sum(
            row.get("status") in {"completed", "completed_without_evaluation"}
            for row in rows
        ),
        "correctness": "not_evaluated",
    }
    for field in fields:
        values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
        result[field] = (
            {
                "count": len(values),
                "total": round(sum(values), 3),
                "mean": round(statistics.mean(values), 3),
                "median": round(statistics.median(values), 3),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
            }
            if values
            else {"count": 0, "total": None, "mean": None, "median": None, "min": None, "max": None}
        )
    complete_timing_rows = [
        row
        for row in rows
        if row.get("inference_timing_coverage") == "complete"
        and isinstance(row.get("provider_window_inference_seconds"), (int, float))
        and isinstance(row.get("provider_window_output_tokens"), (int, float))
    ]
    inference_seconds = sum(
        float(row["provider_window_inference_seconds"])
        for row in complete_timing_rows
    )
    output_tokens = sum(
        int(row["provider_window_output_tokens"])
        for row in complete_timing_rows
    )
    result.update(
        {
            "inference_timing_complete_runs": len(complete_timing_rows),
            "inference_timing_coverage": (
                "complete"
                if rows and len(complete_timing_rows) == len(rows)
                else "partial"
                if rows
                else "unavailable"
            ),
            "provider_window_inference_seconds_total": round(
                inference_seconds, 6
            ),
            "provider_window_output_tokens_total": output_tokens,
            "provider_window_output_tps_ratio_of_sums": (
                output_tokens / inference_seconds if inference_seconds else None
            ),
        }
    )
    return result


def render_markdown(batch: dict) -> str:
    lines = [
        f"# Codex SWE-bench problems 1–10: {batch['tier']}",
        "",
        f"Model: `{batch['model']}`  ",
        f"Reasoning: `{batch['reasoning']}`  ",
        "Correctness: not evaluated",
        "",
        "| # | Instance | Status | Solve s | Tool s | Inference s | TPS | Coverage | Calls | Test s | Inference output |",
        "|---:|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in batch["runs"]:
        def value(key: str) -> str:
            item = row.get(key)
            return "" if item is None else str(item)

        lines.append(
            f"| {row['problem_number']} | {value('instance_id')} | {value('status')} | "
            f"{value('solve_seconds')} | {value('tool_seconds')} | "
            f"{value('provider_window_inference_seconds')} | "
            f"{value('provider_window_output_tps')} | "
            f"{value('inference_timing_coverage')} | "
            f"{value('inference_micro_sessions')} | "
            f"{value('container_test_seconds')} | "
            f"{value('provider_window_output_tokens')} |"
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
            "*Inference uses client-observed response.created → response.completed windows; it is not server-engine timing.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=["ultrafast", "default"], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--reasoning", default=REASONING)
    parser.add_argument("--codex-bin", type=Path, default=INSTRUMENTED_CODEX)
    parser.add_argument(
        "--allow-running-containers",
        action="store_true",
        help="Permit unrelated running Docker containers (reduces benchmark integrity)",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    containers = running_containers(env)
    if containers and not args.allow_running_containers:
        raise SystemExit(
            "BLOCKER: running Docker containers would contaminate timing:\n"
            + "\n".join(containers)
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_id = f"codex_verified_001_010_{args.tier}_{stamp}"
    output = (args.output or ROOT / "runs" / batch_id).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "batch_metadata.json"
    manifest = {
        "schema_version": 1,
        "batch_id": batch_id,
        "tier": args.tier,
        "model": args.model,
        "reasoning": args.reasoning,
        "codex_binary": str(args.codex_bin.resolve()),
        "problem_numbers": list(range(1, 11)),
        "exclusive_sequential_solves": True,
        "official_evaluation_enabled": False,
        "created_at": iso_now(),
        "status": "running",
        "run_dirs": [],
        "returncodes": [],
    }
    write_json(manifest_path, manifest)

    run_dirs = []
    for problem_number in range(1, 11):
        active = running_containers(env)
        if active and not args.allow_running_containers:
            manifest["status"] = "blocked"
            manifest["blocker"] = active
            write_json(manifest_path, manifest)
            break

        run_dir = output / f"{batch_id}_{problem_number:03d}"
        run_dirs.append(run_dir)
        manifest["run_dirs"].append(str(run_dir))
        write_json(manifest_path, manifest)
        print(
            f"\n=== PROBLEM {problem_number}/10: {args.tier} ===",
            flush=True,
        )
        command = [
            str(PYTHON),
            str(ADAPTER),
            "--index",
            str(problem_number - 1),
            "--model",
            args.model,
            "--reasoning",
            args.reasoning,
            "--codex-bin",
            str(args.codex_bin.resolve()),
            "--require-inference-timing",
            "--service-tier",
            args.tier,
            "--output",
            str(run_dir),
            "--skip-evaluation",
        ]
        returncode = run_streaming(
            command,
            output / f"problem_{problem_number:03d}.log",
            env,
        )
        manifest["returncodes"].append(returncode)
        write_json(manifest_path, manifest)

    rows = [
        result_row(problem_number, args.tier, run_dir, load_metadata(run_dir))
        for problem_number, run_dir in enumerate(run_dirs, 1)
    ]
    batch = {
        **manifest,
        "status": "completed" if len(rows) == 10 else manifest["status"],
        "completed_at": iso_now(),
        "runs": rows,
        "aggregate": summarize(rows),
    }
    write_json(manifest_path, batch)

    csv_path = output / "batch.csv"
    if rows:
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    markdown_path = output / "batch.md"
    markdown_path.write_text(render_markdown(batch))

    print(f"\nBATCH_JSON={manifest_path}")
    print(f"BATCH_CSV={csv_path}")
    print(f"BATCH_MARKDOWN={markdown_path}")
    successful = {"completed", "completed_without_evaluation"}
    return 0 if len(rows) == 10 and all(row["status"] in successful for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
