#!/usr/bin/env python3
"""Run ten interleaved normal/ultrafast pairs for SWE-bench problem 1."""

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
PROBLEM_IMAGE = "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
SEQUENCE = [tier for _ in range(10) for tier in ("default", "ultrafast")]


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


def result_row(run_number: int, tier: str, run_dir: Path, metadata: dict) -> dict:
    usage = metadata.get("codex", {}).get("usage") or {}
    timing = metadata.get("timing_breakdown") or {}
    container = metadata.get("container_commands") or {}
    patch = metadata.get("patch") or {}
    input_tokens = usage.get("input_tokens")
    cached_tokens = usage.get("cached_input_tokens")
    return {
        "run_number": run_number,
        "pair_number": (run_number + 1) // 2,
        "tier": tier,
        "instance_id": metadata.get("problem", {}).get("instance_id"),
        "status": metadata.get("status"),
        "correctness": "not_evaluated",
        "setup_seconds": metadata.get("setup_seconds"),
        "solve_seconds": metadata.get("solve_seconds"),
        "tool_seconds": timing.get("tool_seconds"),
        "inference_and_orchestration_seconds_estimate": timing.get(
            "inference_and_orchestration_seconds_estimate"
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


def field_summary(rows: list[dict], field: str) -> dict:
    values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def aggregate(rows: list[dict]) -> dict:
    fields = [
        "setup_seconds",
        "solve_seconds",
        "tool_seconds",
        "inference_and_orchestration_seconds_estimate",
        "container_test_seconds",
        "input_tokens",
        "cached_input_tokens",
        "uncached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
    ]
    result = {}
    for tier in ("default", "ultrafast"):
        tier_rows = [row for row in rows if row["tier"] == tier]
        result[tier] = {
            "attempts": len(tier_rows),
            "completed_solves": sum(
                row.get("status") in {"completed", "completed_without_evaluation"}
                for row in tier_rows
            ),
            **{field: field_summary(tier_rows, field) for field in fields},
        }
    normal = result["default"]["solve_seconds"]["median"]
    ultrafast = result["ultrafast"]["solve_seconds"]["median"]
    result["comparison"] = (
        {
            "normal_median_divided_by_ultrafast_median": round(normal / ultrafast, 3),
            "ultrafast_median_time_reduction_percent": round(
                (1 - ultrafast / normal) * 100, 2
            ),
        }
        if normal and ultrafast
        else None
    )
    return result


def render_markdown(batch: dict) -> str:
    lines = [
        "# Problem 1: ten interleaved normal/ultrafast pairs",
        "",
        "Order: normal → ultrafast, repeated ten times  ",
        "Correctness: not evaluated",
        "",
        "| Run | Pair | Tier | Status | Solve s | Tool s | Inference + orchestration s* | Test s | Input | Output | Reasoning |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in batch["runs"]:
        def value(key: str) -> str:
            item = row.get(key)
            return "" if item is None else str(item)

        lines.append(
            f"| {row['run_number']} | {row['pair_number']} | {row['tier']} | "
            f"{value('status')} | {value('solve_seconds')} | {value('tool_seconds')} | "
            f"{value('inference_and_orchestration_seconds_estimate')} | "
            f"{value('container_test_seconds')} | {value('input_tokens')} | "
            f"{value('output_tokens')} | {value('reasoning_output_tokens')} |"
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
            "*Pure backend inference time is unavailable. The estimate is solve wall time minus measured Codex tool intervals.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
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
    batch_id = f"codex_problem1_interleaved_20_{stamp}"
    output = (args.output or ROOT / "runs" / batch_id).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "batch_metadata.json"
    manifest = {
        "schema_version": 1,
        "batch_id": batch_id,
        "instance_id": "astropy__astropy-12907",
        "model": MODEL,
        "reasoning": REASONING,
        "sequence": SEQUENCE,
        "pairs": 10,
        "total_runs": 20,
        "exclusive_sequential_solves": True,
        "official_evaluation_enabled": False,
        "created_at": iso_now(),
        "status": "running",
        "run_dirs": [],
        "returncodes": [],
    }
    write_json(manifest_path, manifest)

    print(f"[setup] Pre-pulling {PROBLEM_IMAGE}", flush=True)
    pull = subprocess.run(
        ["docker", "pull", PROBLEM_IMAGE],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output / "docker_pull.log").write_text(pull.stdout)
    if pull.returncode != 0:
        raise SystemExit(f"BLOCKER: image pull failed; see {output / 'docker_pull.log'}")

    run_dirs = []
    for run_number, tier in enumerate(SEQUENCE, 1):
        active = running_containers(env)
        if active and not args.allow_running_containers:
            manifest["status"] = "blocked"
            manifest["blocker"] = active
            write_json(manifest_path, manifest)
            break

        run_dir = output / f"{batch_id}_{run_number:02d}_{tier}"
        run_dirs.append(run_dir)
        manifest["run_dirs"].append(str(run_dir))
        write_json(manifest_path, manifest)
        print(
            f"\n=== RUN {run_number}/20 — PAIR {(run_number + 1) // 2}/10 — {tier} ===",
            flush=True,
        )
        command = [
            str(PYTHON),
            str(ADAPTER),
            "--index",
            "0",
            "--model",
            MODEL,
            "--reasoning",
            REASONING,
            "--service-tier",
            tier,
            "--output",
            str(run_dir),
            "--skip-evaluation",
            "--skip-pull",
        ]
        returncode = run_streaming(command, output / f"run_{run_number:02d}.log", env)
        manifest["returncodes"].append(returncode)
        write_json(manifest_path, manifest)

    rows = [
        result_row(run_number, tier, run_dir, load_metadata(run_dir))
        for run_number, (tier, run_dir) in enumerate(zip(SEQUENCE, run_dirs), 1)
    ]
    batch = {
        **manifest,
        "status": "completed" if len(rows) == 20 else manifest["status"],
        "completed_at": iso_now(),
        "runs": rows,
        "aggregate": aggregate(rows),
    }
    write_json(manifest_path, batch)

    csv_path = output / "interleaved_results.csv"
    if rows:
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    markdown_path = output / "batch.md"
    markdown_path.write_text(render_markdown(batch))

    print(f"\nRESULTS_CSV={csv_path}")
    print(f"BATCH_JSON={manifest_path}")
    print(f"BATCH_MARKDOWN={markdown_path}")
    successful = {"completed", "completed_without_evaluation"}
    return 0 if len(rows) == 20 and all(row["status"] in successful for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
