#!/usr/bin/env python3
"""Run LCB problems through the native Codex agent loop with auditable timing."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from bench_runner import check_solution, load_problems, resolve_problem_refs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--problems", nargs="+", required=True)
    parser.add_argument("--release", default="release_v6")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    return parser.parse_args()


def usage_from_event(event: dict) -> dict:
    usage = event.get("usage") or {}
    if not usage and isinstance(event.get("turn"), dict):
        usage = event["turn"].get("usage") or {}
    return usage


def main() -> int:
    args = parse_args()
    ids = resolve_problem_refs(args.problems, args.release)
    problems = load_problems(ids, args.release)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    for number, problem in enumerate(problems, 1):
        run_dir = args.output_dir / f"q{number}_{problem.question_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        solution_path = run_dir / "solution.py"
        final_path = run_dir / "final.txt"
        log_path = run_dir / "events.jsonl"
        prompt = (
            "Solve the programming problem below. Use your normal Codex reasoning, "
            "planning, tools, and testing workflow. Write the complete submission to "
            "solution.py in the current working directory. Do not modify files outside "
            "the current working directory.\n\n"
            + problem.question_content
        )
        command = [
            "codex", "exec", "--json", "--ephemeral", "--skip-git-repo-check",
            "--sandbox", "workspace-write", "--model", args.model,
            "--cd", str(run_dir), "--output-last-message", str(final_path), "-",
        ]
        started_utc = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        first_agent_event_s = None
        final_usage = {}
        event_count = 0
        timed_out = False
        stderr = ""
        with log_path.open("w") as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=os.environ.copy(),
            )
            assert process.stdin and process.stdout and process.stderr
            process.stdin.write(prompt)
            process.stdin.close()
            try:
                deadline = started + args.timeout_seconds
                while True:
                    if time.perf_counter() > deadline:
                        timed_out = True
                        process.kill()
                        break
                    line = process.stdout.readline()
                    if not line:
                        if process.poll() is not None:
                            break
                        continue
                    received_s = time.perf_counter() - started
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        event = {"type": "unparsed", "raw": line.rstrip("\n")}
                    event_count += 1
                    event["harness_received_s"] = received_s
                    event_type = str(event.get("type", ""))
                    if first_agent_event_s is None and event_type.startswith("item."):
                        first_agent_event_s = received_s
                    usage = usage_from_event(event)
                    if usage:
                        final_usage = usage
                    log.write(json.dumps(event, ensure_ascii=False) + "\n")
                    log.flush()
                stderr = process.stderr.read()
                returncode = process.wait(timeout=10)
            except Exception:
                process.kill()
                process.wait()
                raise
        wall_s = time.perf_counter() - started
        output_tokens = final_usage.get("output_tokens")
        summary = {
            "started_at": started_utc,
            "cohort": "codex_native_agent",
            "model": args.model,
            "release": args.release,
            "problem_number": number,
            "problem_id": problem.question_id,
            "title": problem.question_title,
            "returncode": returncode,
            "timed_out": timed_out,
            "event_count": event_count,
            "first_agent_event_s": first_agent_event_s,
            "total_agent_wall_s": wall_s,
            "usage": final_usage,
            "output_tokens_per_total_agent_wall_s": (
                output_tokens / wall_s if output_tokens is not None and wall_s > 0 else None
            ),
            "solution_path": str(solution_path),
            "final_message_path": str(final_path),
            "events_path": str(log_path),
            "stderr": stderr[-4000:],
        }
        if solution_path.exists():
            summary["passed"] = check_solution(problem, solution_path.read_text(), 12)
        else:
            summary["passed"] = False
            summary["solution_missing"] = True
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    (args.output_dir / "summary.json").write_text(
        json.dumps({"runs": summaries}, indent=2) + "\n"
    )
    return 0 if all(item["passed"] for item in summaries) else 2


if __name__ == "__main__":
    raise SystemExit(main())
