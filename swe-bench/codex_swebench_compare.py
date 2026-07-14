#!/usr/bin/env python3
"""Run a sequential ultrafast-vs-normal SWE-bench speed comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "mini-swe-agent" / ".venv" / "bin" / "python"
ADAPTER = ROOT / "codex_swebench_problem1.py"
SEQUENCE = ["ultrafast", "default"]
PROBLEM_IMAGE = "docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"


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


def host_snapshot(env: dict[str, str]) -> dict:
    try:
        load_average = list(os.getloadavg())
    except OSError:
        load_average = None
    return {
        "captured_at": iso_now(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "load_average_1_5_15": load_average,
        "codex_version": command_output(["codex", "--version"], env),
        "docker_server": command_output(
            ["docker", "version", "--format", "{{.Server.Version}} {{.Server.Arch}}"], env
        ),
    }


def load_metadata(run_dir: Path) -> dict:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return {"status": "missing_metadata", "output_dir": str(run_dir)}
    return json.loads(path.read_text())


def digest_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def row_for_run(order: int, tier: str, run_dir: Path, metadata: dict) -> dict:
    usage = metadata.get("codex", {}).get("usage") or {}
    timing = metadata.get("timing_breakdown") or {}
    container = metadata.get("container_commands") or {}
    patch = metadata.get("patch") or {}
    input_tokens = usage.get("input_tokens")
    cached_tokens = usage.get("cached_input_tokens")
    return {
        "order": order,
        "tier": tier,
        "run_dir": str(run_dir),
        "status": metadata.get("status"),
        "resolved": metadata.get("resolved"),
        "created_at": metadata.get("created_at"),
        "solve_started_at": metadata.get("solve_started_at"),
        "solve_ended_at": metadata.get("solve_ended_at"),
        "setup_seconds": metadata.get("setup_seconds"),
        "solve_seconds": metadata.get("solve_seconds"),
        "tool_seconds": timing.get("tool_seconds"),
        "inference_and_orchestration_seconds_estimate": timing.get(
            "inference_and_orchestration_seconds_estimate"
        ),
        "command_tool_seconds": (timing.get("by_type_seconds") or {}).get(
            "command_execution", 0
        ),
        "file_change_tool_seconds": (timing.get("by_type_seconds") or {}).get(
            "file_change", 0
        ),
        "container_test_seconds": container.get("test_seconds"),
        "container_test_count": container.get("test_count"),
        "evaluation_seconds": metadata.get("evaluation_seconds"),
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "uncached_input_tokens": (
            input_tokens - cached_tokens
            if isinstance(input_tokens, int) and isinstance(cached_tokens, int)
            else None
        ),
        "cached_input_percent": (
            round(cached_tokens / input_tokens * 100, 2)
            if isinstance(input_tokens, int)
            and input_tokens
            and isinstance(cached_tokens, int)
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
        "evaluation_returncode": metadata.get("evaluation_returncode"),
        "prompt_sha256": digest_file(run_dir / "codex_prompt.md"),
        "patch_sha256": digest_file(run_dir / "model.patch"),
        "base_commit": metadata.get("problem", {}).get("base_commit"),
        "image": metadata.get("problem", {}).get("image"),
        "prepared_head": metadata.get("prepared_head"),
        "codex_version": metadata.get("codex_version"),
    }


def numeric_summary(rows: list[dict], field: str) -> dict:
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
    result = {}
    fields = [
        "solve_seconds",
        "setup_seconds",
        "tool_seconds",
        "inference_and_orchestration_seconds_estimate",
        "container_test_seconds",
        "input_tokens",
        "cached_input_tokens",
        "cached_input_percent",
        "uncached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
    ]
    for tier in ["default", "ultrafast"]:
        tier_rows = [row for row in rows if row["tier"] == tier]
        result[tier] = {
            "attempts": len(tier_rows),
            "evaluated": sum(isinstance(row.get("resolved"), bool) for row in tier_rows),
            "resolved": sum(row.get("resolved") is True for row in tier_rows),
            **{field: numeric_summary(tier_rows, field) for field in fields},
        }
    normal_median = result["default"]["solve_seconds"]["median"]
    ultra_median = result["ultrafast"]["solve_seconds"]["median"]
    if normal_median and ultra_median:
        result["comparison"] = {
            "median_speedup_x": round(normal_median / ultra_median, 3),
            "median_time_reduction_percent": round(
                (1 - ultra_median / normal_median) * 100, 2
            ),
        }
    else:
        result["comparison"] = None
    return result


def render_markdown(batch: dict) -> str:
    lines = [
        "# Codex SWE-bench normal vs ultrafast",
        "",
        f"Batch: `{batch['batch_id']}`  ",
        f"Instance: `{batch['instance_id']}`  ",
        "Order: ultrafast → normal",
        "",
        "| # | Tier | Correctness | Solve s | Tool s | Inference + orchestration s* | Test s | Input | Cached | Output | Reasoning |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in batch["runs"]:
        def value(key: str) -> str:
            item = row.get(key)
            return "" if item is None else str(item)

        lines.append(
            f"| {row['order']} | {row['tier']} | "
            f"{'not evaluated' if row.get('resolved') is None else value('resolved')} | "
            f"{value('solve_seconds')} | {value('tool_seconds')} | "
            f"{value('inference_and_orchestration_seconds_estimate')} | "
            f"{value('container_test_seconds')} | {value('input_tokens')} | "
            f"{value('cached_input_tokens')} | {value('output_tokens')} | "
            f"{value('reasoning_output_tokens')} |"
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
            "*The Codex CLI does not expose pure backend inference time. This value is solve wall time minus measured JSONL tool intervals and therefore includes orchestration and event-stream latency.",
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
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_id = f"codex_problem1_comparison_{stamp}"
    output = (args.output or ROOT / "runs" / batch_id).resolve()

    containers = running_containers(env)
    if containers and not args.allow_running_containers:
        raise SystemExit(
            "BLOCKER: running Docker containers would contaminate timing:\n"
            + "\n".join(containers)
        )
    output.mkdir(parents=True, exist_ok=False)

    manifest = {
        "schema_version": 1,
        "batch_id": batch_id,
        "created_at": iso_now(),
        "status": "solving",
        "instance_id": "astropy__astropy-12907",
        "sequence": SEQUENCE,
        "model": "koffing-updated",
        "reasoning": "medium",
        "exclusive_sequential_solves": True,
        "official_evaluation_enabled": False,
        "host": host_snapshot(env),
        "run_dirs": [],
    }
    manifest_path = output / "comparison_metadata.json"
    write_json(manifest_path, manifest)

    print(f"[setup] Pre-pulling {PROBLEM_IMAGE}", flush=True)
    pull_result = subprocess.run(
        ["docker", "pull", PROBLEM_IMAGE],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output / "docker_pull.log").write_text(pull_result.stdout)
    if pull_result.returncode != 0:
        raise SystemExit(f"BLOCKER: image pull failed; see {output / 'docker_pull.log'}")

    run_dirs = []
    for order, tier in enumerate(SEQUENCE, 1):
        active = running_containers(env)
        if active and not args.allow_running_containers:
            raise SystemExit(
                f"BLOCKER: containers appeared before solve {order}:\n" + "\n".join(active)
            )
        run_dir = output / f"{batch_id}_{order:02d}_{tier}"
        run_dirs.append(run_dir)
        manifest["run_dirs"].append(str(run_dir))
        manifest.setdefault("pre_solve_host", []).append(host_snapshot(env))
        write_json(manifest_path, manifest)
        print(f"\n=== SOLVE {order}/{len(SEQUENCE)}: {tier} ===", flush=True)
        command = [
            str(PYTHON),
            str(ADAPTER),
            "--index",
            "0",
            "--model",
            "koffing-updated",
            "--reasoning",
            "medium",
            "--service-tier",
            tier,
            "--output",
            str(run_dir),
            "--skip-evaluation",
            "--skip-pull",
        ]
        returncode = run_streaming(command, output / f"solve_{order:02d}.log", env)
        manifest.setdefault("solve_returncodes", []).append(returncode)
        write_json(manifest_path, manifest)

    rows = [
        row_for_run(order, tier, run_dir, load_metadata(run_dir))
        for order, (tier, run_dir) in enumerate(zip(SEQUENCE, run_dirs), 1)
    ]
    comparison = {
        **manifest,
        "status": "completed",
        "completed_at": iso_now(),
        "runs": rows,
        "aggregate": aggregate(rows),
        "fairness_checks": {
            "same_prompt": None not in {row["prompt_sha256"] for row in rows}
            and len({row["prompt_sha256"] for row in rows}) == 1,
            "same_base_commit": None not in {row["base_commit"] for row in rows}
            and len({row["base_commit"] for row in rows}) == 1,
            "same_image": None not in {row["image"] for row in rows}
            and len({row["image"] for row in rows}) == 1,
            "same_prepared_head": None not in {row["prepared_head"] for row in rows}
            and len({row["prepared_head"] for row in rows}) == 1,
            "same_codex_version": None not in {row["codex_version"] for row in rows}
            and len({row["codex_version"] for row in rows}) == 1,
            "only_requested_tier_varied": all(
                load_metadata(run_dir).get("requested_model") == "koffing-updated"
                and load_metadata(run_dir).get("requested_reasoning_effort") == "medium"
                and load_metadata(run_dir).get("requested_service_tier") == tier
                for tier, run_dir in zip(SEQUENCE, run_dirs)
            ),
        },
    }
    write_json(manifest_path, comparison)

    csv_path = output / "comparison.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output / "comparison.md").write_text(render_markdown(comparison))

    print(f"\nCOMPARISON_JSON={manifest_path}")
    print(f"COMPARISON_CSV={csv_path}")
    print(f"COMPARISON_MARKDOWN={output / 'comparison.md'}")
    successful_statuses = {"completed", "completed_without_evaluation"}
    return 0 if all(row.get("status") in successful_statuses for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
