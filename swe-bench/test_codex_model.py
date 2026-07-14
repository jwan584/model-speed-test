#!/usr/bin/env python3
"""Probe a model/configuration through ChatGPT-authenticated `codex exec`."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="koffing-updated")
    parser.add_argument("--reasoning", default="medium", choices=["minimal", "low", "medium", "high", "xhigh"])
    parser.add_argument(
        "--speed-label",
        default="ultrafast",
        help="Metadata label only; 'ultrafast' has no documented Codex CLI setting",
    )
    parser.add_argument(
        "--service-tier",
        default="fast",
        choices=["default", "fast", "flex", "ultrafast"],
        help="Codex service tier to request; 'default' omits the setting",
    )
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("runs/codex_model_probes"))
    parser.add_argument("--prompt", default="Do not use tools. Reply with exactly: CODEX_MODEL_TEST_OK")
    args = parser.parse_args()

    args.workdir = args.workdir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{stamp}_{args.model}_{args.reasoning}_{args.speed_label}".replace("/", "_").replace(" ", "_")
    jsonl_path = (args.output_dir / f"{stem}.jsonl").resolve()
    stderr_path = (args.output_dir / f"{stem}.stderr.log").resolve()
    summary_path = (args.output_dir / f"{stem}.summary.json").resolve()

    version = subprocess.run(["codex", "--version"], text=True, capture_output=True, check=False)
    command = [
        "codex",
        "exec",
        "--json",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--model",
        args.model,
        "--config",
        f'model_reasoning_effort="{args.reasoning}"',
        "--config",
        'approval_policy="on-request"',
    ]
    if args.service_tier != "default":
        command.extend(["--config", f'service_tier="{args.service_tier}"'])
    if args.service_tier == "fast":
        command.extend(["--enable", "fast_mode"])
    command.extend(["--cd", str(args.workdir), args.prompt])

    print(f"Codex: {version.stdout.strip() or version.stderr.strip()}")
    print(f"Requested model: {args.model}")
    print(f"Reasoning effort: {args.reasoning}")
    print(f"Speed label: {args.speed_label} (metadata only)")
    print(f"Requested service tier: {args.service_tier}")
    print(f"JSONL: {jsonl_path}")
    print(f"stderr: {stderr_path}")

    started_at = iso_now()
    started_mono = time.monotonic()
    thread_id = None
    final_message = None
    usage = None
    errors: list[dict | str] = []
    event_count = 0

    with jsonl_path.open("w") as jsonl_file, stderr_path.open("w") as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=args.workdir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
        )
        assert process.stdout is not None
        for line in process.stdout:
            jsonl_file.write(line)
            jsonl_file.flush()
            event_count += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                print(f"[non-json] {line.rstrip()}")
                continue

            event_type = event.get("type", "unknown")
            if event_type == "thread.started":
                thread_id = event.get("thread_id")
                print(f"[{event_type}] thread_id={thread_id}")
            elif event_type == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    final_message = item.get("text")
                    print(f"[{event_type}] agent_message={final_message!r}")
                elif item.get("type") == "error":
                    errors.append(item)
                    print(f"[{event_type}] error={item.get('message')!r}")
                else:
                    print(f"[{event_type}] item_type={item.get('type')}")
            elif event_type == "turn.completed":
                usage = event.get("usage")
                print(f"[{event_type}] usage={json.dumps(usage, sort_keys=True)}")
            elif event_type in {"turn.failed", "error"}:
                errors.append(event)
                print(f"[{event_type}] {json.dumps(event, sort_keys=True)}")
            else:
                print(f"[{event_type}]")
        returncode = process.wait()

    ended_at = iso_now()
    elapsed = round(time.monotonic() - started_mono, 3)
    accepted = returncode == 0 and final_message is not None
    exact_reply = final_message == "CODEX_MODEL_TEST_OK"
    summary = {
        "accepted": accepted,
        "exact_reply": exact_reply,
        "returncode": returncode,
        "requested_model": args.model,
        "reasoning_effort": args.reasoning,
        "speed_label": args.speed_label,
        "service_tier": args.service_tier,
        "codex_version": version.stdout.strip() or version.stderr.strip(),
        "thread_id": thread_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": elapsed,
        "event_count": event_count,
        "usage": usage,
        "final_message": final_message,
        "errors": errors,
        "jsonl_path": str(jsonl_path),
        "stderr_path": str(stderr_path),
        "command_without_prompt": command[:-1],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Summary: {summary_path}")
    print(f"Result: accepted={accepted} exact_reply={exact_reply} returncode={returncode} elapsed={elapsed}s")
    if returncode != 0:
        print("Probe failed. Inspect the stderr and summary files above.", file=sys.stderr)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
