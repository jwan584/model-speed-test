#!/usr/bin/env python3
"""Run SWE-bench instances strictly sequentially and record wall-clock timings."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MINI_DIR = ROOT / "mini-swe-agent"
MINI_EXTRA = MINI_DIR / ".venv" / "bin" / "mini-extra"
NEWSLETTER_ENV = ROOT.parent / "newsletter-bot" / ".env.local"
DEFAULT_MODEL = "anthropic/claude-sonnet-4-5-20250929"
SUBMISSION_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def require_setup(model: str) -> None:
    blockers: list[str] = []
    if shutil.which("docker") is None:
        blockers.append("docker is not installed or not on PATH")
    if not MINI_EXTRA.exists():
        blockers.append(f"mini-extra is missing at {MINI_EXTRA}")
    if model.startswith("anthropic/") or "claude" in model:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            blockers.append(f"ANTHROPIC_API_KEY is not set and was not found in {NEWSLETTER_ENV}")
    elif not (os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")):
        blockers.append("API_KEY or OPENAI_API_KEY must be set for the selected model")
    if blockers:
        for blocker in blockers:
            print(f"BLOCKER: {blocker}", file=sys.stderr)
        raise SystemExit(2)


def configure_env(model: str) -> dict[str, str]:
    env = os.environ.copy()
    if env.get("API_KEY") and not env.get("OPENAI_API_KEY"):
        env["OPENAI_API_KEY"] = env["API_KEY"]
    if env.get("BASE_URL"):
        env["OPENAI_API_BASE"] = env["BASE_URL"]
        env["OPENAI_BASE_URL"] = env["BASE_URL"]
    env.setdefault("MODEL", model)
    env.setdefault("MSWEA_MODEL_NAME", model)
    env.setdefault("MSWEA_CONFIGURED", "true")
    env.setdefault("MSWEA_COST_TRACKING", "ignore_errors")
    return env


def docker_output(command: list[str], env: dict[str, str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def verify_docker_config(env: dict[str, str]) -> None:
    docker_config = env.get("DOCKER_CONFIG")
    if not docker_config:
        raise SystemExit(
            "BLOCKER: DOCKER_CONFIG must point to a writable empty directory. "
            "See SWE_BENCH_RUNBOOK.md."
        )
    config_dir = Path(docker_config)
    config_dir.mkdir(parents=True, exist_ok=True)
    probe = docker_output(["docker", "version", "--format", "{{.Server.Version}}"], env)
    if probe.returncode != 0:
        raise SystemExit(f"BLOCKER: Docker daemon preflight failed:\n{probe.stdout}")
    if "warning" in probe.stdout.lower() or "operation not permitted" in probe.stdout.lower():
        raise SystemExit(f"BLOCKER: Docker emitted output that can corrupt submission detection:\n{probe.stdout}")


def verify_submission_sentinel(image: str, env: dict[str, str]) -> None:
    """Verify Docker output preserves the sentinel as the first line before spending model tokens."""
    container_name = f"swebench-preflight-{os.getpid()}"
    start = docker_output(
        ["docker", "run", "-d", "--name", container_name, "--rm", image, "sleep", "120"],
        env,
        timeout=60,
    )
    if start.returncode != 0:
        raise SystemExit(f"BLOCKER: submission preflight container failed to start:\n{start.stdout}")
    try:
        probe = docker_output(
            [
                "docker",
                "exec",
                container_name,
                "bash",
                "-c",
                f"printf '{SUBMISSION_SENTINEL}\\npreflight-patch\\n'",
            ],
            env,
            timeout=60,
        )
        lines = probe.stdout.lstrip().splitlines()
        if probe.returncode != 0 or not lines or lines[0].strip() != SUBMISSION_SENTINEL:
            raise SystemExit(
                "BLOCKER: submission sentinel was not the first captured Docker output line.\n"
                f"returncode={probe.returncode}\noutput:\n{probe.stdout}"
            )
        print(f"PREFLIGHT_OK sentinel={SUBMISSION_SENTINEL}", flush=True)
    finally:
        docker_output(["docker", "rm", "-f", container_name], env, timeout=30)


def collect_health(env: dict[str, str], started_at: float) -> dict:
    record: dict = {
        "timestamp": iso_now(),
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
    }
    try:
        daemon = docker_output(["docker", "info", "--format", "{{.ServerVersion}}"], env, timeout=10)
        record["docker_ok"] = daemon.returncode == 0
        record["docker_output"] = daemon.stdout.strip()
        containers = docker_output(
            [
                "docker",
                "ps",
                "--filter",
                "name=minisweagent",
                "--format",
                "{{.ID}}|{{.Names}}|{{.Status}}",
            ],
            env,
            timeout=10,
        )
        record["containers_ok"] = containers.returncode == 0
        record["containers"] = [line for line in containers.stdout.splitlines() if line]
        processes: dict[str, list[str]] = {}
        for line in record["containers"]:
            container_id = line.split("|", 1)[0]
            top = docker_output(
                ["docker", "top", container_id, "-eo", "pid,etime,stat,comm,args"], env, timeout=10
            )
            processes[container_id] = top.stdout.splitlines()[:20]
        record["processes"] = processes
    except Exception as exc:
        record["monitor_error"] = f"{type(exc).__name__}: {exc}"
    return record


def monitor_health(path: Path, env: dict[str, str], stop: threading.Event, interval: float) -> None:
    started_at = time.monotonic()
    while True:
        record = collect_health(env, started_at)
        append_jsonl(path, record)
        print(
            "[health] {timestamp} elapsed={elapsed_seconds:.0f}s docker_ok={docker_ok} containers={containers}".format(
                timestamp=record["timestamp"],
                elapsed_seconds=record["elapsed_seconds"],
                docker_ok=record.get("docker_ok", False),
                containers=len(record.get("containers", [])),
            ),
            flush=True,
        )
        if stop.wait(interval):
            break


def select_instances(args: argparse.Namespace) -> list[dict]:
    from datasets import load_dataset
    from minisweagent.run.benchmarks.swebench import DATASET_MAPPING

    dataset_path = DATASET_MAPPING.get(args.subset, args.subset)
    instances = list(load_dataset(dataset_path, split=args.split))
    by_id = {instance["instance_id"]: instance for instance in instances}
    if args.instance_id:
        missing = [instance_id for instance_id in args.instance_id if instance_id not in by_id]
        if missing:
            raise SystemExit(f"Unknown instance_id(s): {', '.join(missing)}")
        return [by_id[instance_id] for instance_id in args.instance_id]
    sorted_instances = sorted(instances, key=lambda instance: instance["instance_id"])
    return sorted_instances[args.start : args.start + args.count]


def image_name(instance: dict) -> str:
    from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name

    return get_swebench_docker_image_name(instance)


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "run_label",
        "mode",
        "speed",
        "model",
        "provider",
        "index",
        "instance_id",
        "problem_title",
        "status",
        "exit_status",
        "start_time",
        "end_time",
        "elapsed_seconds",
        "cumulative_seconds",
        "returncode",
        "api_calls",
        "tool_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "inference_seconds_approx",
        "tool_seconds_approx",
        "cost_usd",
        "patch_bytes",
        "output_dir",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def bar(seconds: float, scale: float) -> str:
    width = max(1, int(round(seconds / scale))) if seconds > 0 else 1
    return "#" * min(width, 80)


def write_chart(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    max_elapsed = max(row["elapsed_seconds"] for row in rows) or 1
    scale = max(max_elapsed / 50, 1)
    lines = [
        "# Sequential SWE-bench Timing",
        "",
        f"- Run: `{rows[0]['run_label']}`",
        f"- Mode: `{rows[0]['mode']}`",
        f"- Speed: `{rows[0]['speed']}`",
        f"- Model: `{rows[0]['model']}`",
        "",
        "| # | instance_id | status | start | end | elapsed | cumulative | bar |",
        "|---:|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {index} | `{instance_id}` | {status} | {start_time} | {end_time} | "
            "{elapsed_seconds:.0f}s | {cumulative_seconds:.0f}s | `{bar}` |".format(
                **row,
                bar=bar(row["elapsed_seconds"], scale),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def write_run_metadata(path: Path, args: argparse.Namespace, rows: list[dict]) -> dict:
    provider = args.model.split("/", 1)[0] if "/" in args.model else "unspecified"
    exit_statuses = sorted({row.get("exit_status") or row.get("status", "unknown") for row in rows})
    metadata = {
        "run_label": args.run_label,
        "mode": args.mode,
        "speed": args.speed,
        "model": args.model,
        "provider": provider,
        "subset": args.subset,
        "split": args.split,
        "output_dir": str(args.output),
        "started_at": rows[0]["start_time"],
        "ended_at": rows[-1]["end_time"],
        "elapsed_seconds": round(sum(row["elapsed_seconds"] for row in rows), 3),
        "result_status": ",".join(exit_statuses),
        "totals": {
            "api_calls": sum(row.get("api_calls", 0) for row in rows),
            "tool_calls": sum(row.get("tool_calls", 0) for row in rows),
            "prompt_tokens": sum(row.get("prompt_tokens", 0) for row in rows),
            "completion_tokens": sum(row.get("completion_tokens", 0) for row in rows),
            "total_tokens": sum(row.get("total_tokens", 0) for row in rows),
            "cache_read_input_tokens": sum(row.get("cache_read_input_tokens", 0) for row in rows),
            "cache_creation_input_tokens": sum(row.get("cache_creation_input_tokens", 0) for row in rows),
            "inference_seconds_approx": round(
                sum(row.get("inference_seconds_approx", 0) for row in rows), 3
            ),
            "tool_seconds_approx": round(sum(row.get("tool_seconds_approx", 0) for row in rows), 3),
            "cost_usd": round(sum(row.get("cost_usd", 0) for row in rows), 8),
            "patch_bytes": sum(row.get("patch_bytes", 0) for row in rows),
        },
        "instances": rows,
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def update_run_history(metadata: dict) -> None:
    history_jsonl = ROOT / "runs" / "timed_run_history.jsonl"
    records = []
    if history_jsonl.exists():
        records = [json.loads(line) for line in history_jsonl.read_text().splitlines() if line.strip()]
    records = [record for record in records if record.get("output_dir") != metadata["output_dir"]]
    records.append(metadata)
    records.sort(key=lambda record: record["started_at"])
    with history_jsonl.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    history_csv = ROOT / "runs" / "timed_run_history.csv"
    fields = [
        "run_label",
        "mode",
        "speed",
        "provider",
        "model",
        "instance_ids",
        "started_at",
        "ended_at",
        "elapsed_seconds",
        "result_status",
        "api_calls",
        "tool_calls",
        "total_tokens",
        "cost_usd",
        "output_dir",
    ]
    with history_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            totals = record.get("totals", {})
            flattened = {
                **record,
                "instance_ids": ";".join(x["instance_id"] for x in record.get("instances", [])),
                "api_calls": totals.get("api_calls", ""),
                "tool_calls": totals.get("tool_calls", ""),
                "total_tokens": totals.get("total_tokens", ""),
                "cost_usd": totals.get("cost_usd", ""),
            }
            writer.writerow({field: flattened.get(field, "") for field in fields})

    history_md = ROOT / "runs" / "timed_run_history.md"
    lines = [
        "# Timed SWE-bench Run History",
        "",
        "| Run | Problem | Result | Provider / model | Mode / speed | Start | End | Elapsed | Calls | Tokens | Cost | Output |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for record in records:
        output_name = Path(record["output_dir"]).name
        totals = record.get("totals", {})
        instance_ids = ", ".join(x["instance_id"] for x in record.get("instances", []))
        lines.append(
            f"| {record['run_label']} | `{instance_ids}` | {record.get('result_status', '')} | "
            f"{record.get('provider', '')} / "
            f"`{record['model']}` | {record['mode']} / {record['speed']} | "
            f"{record['started_at']} | {record['ended_at']} | {record['elapsed_seconds']:.3f}s | "
            f"{totals.get('api_calls', '')} API / {totals.get('tool_calls', '')} tool | "
            f"{totals.get('total_tokens', '')} | ${totals.get('cost_usd', 0):.4f} | "
            f"[`{output_name}`]({output_name}) |"
        )
    history_md.write_text("\n".join(lines) + "\n")


def write_aggregate_predictions(output_root: Path, rows: list[dict]) -> None:
    predictions: dict[str, dict] = {}
    for row in rows:
        preds_path = Path(row["output_dir"]) / "preds.json"
        if not preds_path.exists():
            continue
        predictions.update(json.loads(preds_path.read_text()))
    if not predictions:
        return
    json_path = output_root / "preds.json"
    jsonl_path = output_root / "preds.jsonl"
    json_path.write_text(json.dumps(predictions, indent=2, sort_keys=True))
    with jsonl_path.open("w") as f:
        for instance_id in sorted(predictions):
            f.write(json.dumps(predictions[instance_id], sort_keys=True) + "\n")


def collect_instance_metrics(instance: dict, row: dict) -> dict:
    output_dir = Path(row["output_dir"])
    trajectory_paths = list(output_dir.glob(f"{row['instance_id']}/*.traj.json"))
    metrics = {
        "provider": row["model"].split("/", 1)[0] if "/" in row["model"] else "unspecified",
        "problem_title": instance["problem_statement"].splitlines()[0].strip(),
        "problem_statement": instance["problem_statement"],
        "exit_status": "",
        "api_calls": 0,
        "tool_calls": 0,
        "observed_tool_results": 0,
        "test_tool_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "inference_seconds_approx": 0.0,
        "tool_seconds_approx": 0.0,
        "finalization_seconds_approx": 0.0,
        "cost_usd": 0.0,
        "patch_bytes": 0,
        "patch_added_lines": 0,
        "patch_removed_lines": 0,
        "commands": [],
        "tool_returncodes": {},
    }
    if trajectory_paths:
        trajectory = json.loads(trajectory_paths[0].read_text())
        messages = trajectory.get("messages", [])
        metrics["exit_status"] = trajectory.get("info", {}).get("exit_status", "")
        model_stats = trajectory.get("info", {}).get("model_stats", {})
        metrics["cost_usd"] = model_stats.get("instance_cost", 0.0)
        metrics["api_calls"] = model_stats.get("api_calls", 0)
        last_timestamp = row["start_epoch_seconds"]
        for message in messages:
            extra = message.get("extra") or {}
            response = extra.get("response")
            if response is not None and extra.get("timestamp") is not None:
                timestamp = extra["timestamp"]
                metrics["inference_seconds_approx"] += timestamp - last_timestamp
                last_timestamp = timestamp
                usage = response.get("usage") or {}
                for key in [
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                ]:
                    metrics[key] += usage.get(key) or 0
                actions = extra.get("actions") or []
                metrics["tool_calls"] += len(actions)
                for action in actions:
                    command = action.get("command", "")
                    metrics["commands"].append(command)
                    if "pytest" in command or "test_" in command or "test " in command:
                        metrics["test_tool_calls"] += 1
            elif message.get("role") == "tool" and extra.get("timestamp") is not None:
                timestamp = extra["timestamp"]
                metrics["tool_seconds_approx"] += timestamp - last_timestamp
                last_timestamp = timestamp
                metrics["observed_tool_results"] += 1
                returncode = str(extra.get("returncode", "unknown"))
                metrics["tool_returncodes"][returncode] = metrics["tool_returncodes"].get(returncode, 0) + 1
        metrics["finalization_seconds_approx"] = row["end_epoch_seconds"] - last_timestamp

    preds_path = output_dir / "preds.json"
    if preds_path.exists():
        patch = json.loads(preds_path.read_text()).get(row["instance_id"], {}).get("model_patch") or ""
        metrics["patch_bytes"] = len(patch.encode())
        metrics["patch_added_lines"] = sum(
            line.startswith("+") and not line.startswith("+++") for line in patch.splitlines()
        )
        metrics["patch_removed_lines"] = sum(
            line.startswith("-") and not line.startswith("---") for line in patch.splitlines()
        )

    for key in ["inference_seconds_approx", "tool_seconds_approx", "finalization_seconds_approx", "cost_usd"]:
        metrics[key] = round(metrics[key], 8 if key == "cost_usd" else 3)
    return metrics


def run_problem(instance: dict, args: argparse.Namespace, env: dict[str, str], index: int) -> dict:
    instance_id = instance["instance_id"]
    output_dir = args.output / f"{index:03d}_{instance_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(MINI_EXTRA),
        "swebench",
        "--subset",
        args.subset,
        "--split",
        args.split,
        "--filter",
        f"^{instance_id}$",
        "--workers",
        "1",
        "--output",
        str(output_dir),
        "--model",
        args.model,
        "--environment-class",
        args.environment_class,
        "--redo-existing",
    ]

    start_wall = time.time()
    start_mono = time.monotonic()
    start_time = iso_now()
    print(f"[{index}] START {start_time} {instance_id}", flush=True)
    print(f"[{index}] CMD {' '.join(shlex.quote(part) for part in command)}", flush=True)

    log_path = output_dir / "harness_stdout_stderr.log"
    health_path = output_dir / "health.jsonl"
    monitor_stop = threading.Event()
    monitor = threading.Thread(
        target=monitor_health,
        args=(health_path, env, monitor_stop, args.health_interval),
        daemon=True,
    )
    monitor.start()
    try:
        with log_path.open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    finally:
        monitor_stop.set()
        monitor.join(timeout=15)

    end_mono = time.monotonic()
    end_time = iso_now()
    elapsed = end_mono - start_mono
    status = "ok" if result.returncode == 0 else "error"
    print(f"[{index}] END   {end_time} {instance_id} status={status} elapsed={elapsed:.0f}s", flush=True)

    return {
        "run_label": args.run_label,
        "mode": args.mode,
        "speed": args.speed,
        "model": args.model,
        "index": index,
        "instance_id": instance_id,
        "image": image_name(instance),
        "status": status,
        "start_time": start_time,
        "start_epoch_seconds": round(start_wall, 3),
        "end_time": end_time,
        "end_epoch_seconds": round(time.time(), 3),
        "elapsed_seconds": round(elapsed, 3),
        "returncode": result.returncode,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "health_path": str(health_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", default="verified", help="SWE-bench subset or dataset path")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--start", type=int, default=0, help="Start index after sorting instance_id")
    parser.add_argument("--count", type=int, default=1, help="Number of sorted instances to run")
    parser.add_argument("--instance-id", action="append", help="Explicit instance_id; may be repeated")
    parser.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL))
    parser.add_argument("--run-label", default=os.environ.get("RUN_LABEL", "unlabeled"))
    parser.add_argument("--mode", default=os.environ.get("RUN_MODE", "unspecified"))
    parser.add_argument("--speed", default=os.environ.get("RUN_SPEED", "unspecified"))
    parser.add_argument("--environment-class", default="docker")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / f"sequential_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--prepull", action="store_true", help="Pull all instance images before timed problem runs")
    parser.add_argument(
        "--health-interval",
        type=float,
        default=60,
        help="Seconds between passive Docker health records; this monitor never terminates a run",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print selected instances and exit")
    args = parser.parse_args()
    args.output = args.output.resolve()

    load_dotenv(NEWSLETTER_ENV)
    require_setup(args.model)
    env = configure_env(args.model)
    verify_docker_config(env)

    instances = select_instances(args)
    args.output.mkdir(parents=True, exist_ok=True)

    selection_path = args.output / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "run_label": args.run_label,
                "mode": args.mode,
                "speed": args.speed,
                "subset": args.subset,
                "split": args.split,
                "model": args.model,
                "selected": [
                    {"index": i + 1, "instance_id": instance["instance_id"], "image": image_name(instance)}
                    for i, instance in enumerate(instances)
                ],
            },
            indent=2,
        )
    )

    print(f"OUTPUT_DIR={args.output}")
    print(f"MODEL={args.model}")
    print(f"RUN_LABEL={args.run_label}")
    print(f"MODE={args.mode}")
    print(f"SPEED={args.speed}")
    print(f"COUNT={len(instances)}")
    for i, instance in enumerate(instances, start=1):
        print(f"SELECTED[{i}] {instance['instance_id']} image={image_name(instance)}")
    if args.dry_run:
        return 0

    if args.prepull:
        for i, instance in enumerate(instances, start=1):
            img = image_name(instance)
            print(f"[prepull {i}/{len(instances)}] docker pull {img}", flush=True)
            subprocess.run(["docker", "pull", img], check=True, env=env)

    for instance in instances:
        verify_submission_sentinel(image_name(instance), env)

    jsonl_path = args.output / "timings.jsonl"
    csv_path = args.output / "timings.csv"
    chart_path = args.output / "timing_chart.md"
    metadata_path = args.output / "run_metadata.json"
    rows: list[dict] = []
    cumulative = 0.0

    for i, instance in enumerate(instances, start=1):
        row = run_problem(instance, args, env, i)
        row.update(collect_instance_metrics(instance, row))
        cumulative += row["elapsed_seconds"]
        row["cumulative_seconds"] = round(cumulative, 3)
        rows.append(row)
        append_jsonl(jsonl_path, row)
        write_csv(csv_path, rows)
        write_chart(chart_path, rows)
        write_aggregate_predictions(args.output, rows)
        metadata = write_run_metadata(metadata_path, args, rows)
        update_run_history(metadata)
        if row["returncode"] != 0:
            print(f"Stopping after failure on {row['instance_id']}. See {row['log_path']}", file=sys.stderr)
            return row["returncode"]

    print(f"TIMINGS_JSONL={jsonl_path}")
    print(f"TIMINGS_CSV={csv_path}")
    print(f"TIMING_CHART={chart_path}")
    print(f"RUN_METADATA={metadata_path}")
    print(f"PREDS_JSON={args.output / 'preds.json'}")
    print(f"PREDS_JSONL={args.output / 'preds.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
