#!/usr/bin/env python3
"""Solve and evaluate one numbered SWE-bench Verified problem with native Codex."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codex_inference_timing import (
    LIFECYCLE_TRACE_TARGET,
    binary_has_lifecycle_trace_hook,
    load_jsonl,
    parse_lifecycle_trace_line,
    summarize_lifecycle,
)


ROOT = Path(__file__).resolve().parent
MINI_PYTHON = Path(
    os.environ.get(
        "SWE_BENCH_MINI_PYTHON",
        ROOT / "mini-swe-agent" / ".venv" / "bin" / "python",
    )
)
EVAL_PYTHON = Path(
    os.environ.get(
        "SWE_BENCH_EVAL_PYTHON",
        ROOT / ".venv_eval" / "bin" / "python",
    )
)
DATASET = "princeton-nlp/SWE-Bench_Verified"
SPLIT = "test"
MODEL = "koffing-updated"
REASONING = "medium"
SERVICE_TIER = "ultrafast"


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {shlex.join(command)}\n{result.stdout}")
    return result


def run_with_heartbeat(
    command: list[str],
    *,
    label: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    """Run a quiet subprocess while making forward waiting visible."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert process.stdout is not None
    output: list[str] = []

    def drain_output() -> None:
        output.extend(process.stdout.readlines())

    drain_thread = threading.Thread(target=drain_output, daemon=True)
    drain_thread.start()
    started = time.monotonic()
    while True:
        elapsed = time.monotonic() - started
        remaining = None if timeout is None else max(0.0, timeout - elapsed)
        if remaining == 0:
            process.kill()
            process.wait()
            drain_thread.join(timeout=5)
            process.stdout.close()
            raise subprocess.TimeoutExpired(command, timeout, "".join(output))
        try:
            returncode = process.wait(
                timeout=min(30.0, remaining) if remaining is not None else 30.0
            )
            break
        except subprocess.TimeoutExpired:
            print(
                json.dumps(
                    {
                        "status": f"{label}_running",
                        "elapsed_s": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
    drain_thread.join(timeout=5)
    process.stdout.close()
    return subprocess.CompletedProcess(
        command,
        returncode,
        stdout="".join(output),
        stderr=None,
    )


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def resolve_codex_binary(value: str) -> Path:
    resolved = shutil.which(value)
    if resolved is None:
        candidate = Path(value).expanduser()
        if candidate.is_file():
            resolved = str(candidate.resolve())
    if resolved is None:
        raise SystemExit(f"BLOCKER: Codex executable not found: {value}")
    return Path(resolved).resolve()


def require_setup(
    env: dict[str, str],
    codex_bin: Path,
    *,
    require_evaluator: bool,
) -> None:
    blockers = []
    for command in ["docker", "git"]:
        if shutil.which(command) is None:
            blockers.append(f"{command} is not installed or not on PATH")
    if not MINI_PYTHON.exists():
        blockers.append(f"mini-swe-agent Python is missing at {MINI_PYTHON}")
    if require_evaluator and not EVAL_PYTHON.exists():
        blockers.append(f"SWE-bench evaluator Python is missing at {EVAL_PYTHON}")
    if not env.get("DOCKER_HOST"):
        blockers.append("DOCKER_HOST is not set; use run_codex_swebench_problem")
    if not env.get("DOCKER_CONFIG"):
        blockers.append("DOCKER_CONFIG is not set; use run_codex_swebench_problem")
    if blockers:
        raise SystemExit("\n".join(f"BLOCKER: {item}" for item in blockers))

    probe = run(["docker", "version", "--format", "{{.Server.Version}}"], env=env, check=False)
    if probe.returncode != 0 or "warning" in probe.stdout.lower():
        raise SystemExit(f"BLOCKER: Docker preflight failed or emitted a warning:\n{probe.stdout}")
    auth = run([str(codex_bin), "login", "status"], env=env, check=False)
    if auth.returncode != 0:
        raise SystemExit(f"BLOCKER: Codex is not authenticated:\n{auth.stdout}")


def load_problem(index: int) -> dict:
    load_started = time.monotonic()
    print(
        f"[dataset] Importing SWE-bench dataset support with {sys.executable}",
        flush=True,
    )
    from datasets import load_dataset

    print(f"[dataset] Loading {DATASET}/{SPLIT} from {os.environ.get('HF_HOME')}", flush=True)
    instances = sorted(load_dataset(DATASET, split=SPLIT), key=lambda item: item["instance_id"])
    print(
        f"[dataset] Loaded {len(instances)} problems in "
        f"{time.monotonic() - load_started:.1f}s",
        flush=True,
    )
    if index < 0 or index >= len(instances):
        raise SystemExit(f"Problem index {index} is outside 0..{len(instances) - 1}")
    raw = dict(instances[index])
    image = raw.get("image_name") or raw.get("docker_image")
    if image is None:
        docker_instance_id = raw["instance_id"].replace("__", "_1776_")
        image = (
            f"docker.io/swebench/sweb.eval.x86_64.{docker_instance_id}:latest"
        ).lower()
    return {
        "index": index,
        "instance_id": raw["instance_id"],
        "repo": raw["repo"],
        "base_commit": raw["base_commit"],
        "problem_statement": raw["problem_statement"],
        "image": image,
    }


def prepare_worktree(
    problem: dict,
    worktree: Path,
    env: dict[str, str],
    output: Path,
    skip_pull: bool = False,
) -> str:
    if skip_pull:
        (output / "docker_pull.log").write_text("Skipped: image pre-pulled by batch runner.\n")
    else:
        print(f"[setup] Pulling {problem['image']}", flush=True)
        pull = run_with_heartbeat(
            ["docker", "pull", problem["image"]],
            label="docker_pull",
            env=env,
        )
        (output / "docker_pull.log").write_text(pull.stdout)
        if pull.returncode != 0:
            raise RuntimeError(f"Docker pull failed:\n{pull.stdout}")

    prep_name = f"codex-swe-prep-{uuid.uuid4().hex[:10]}"
    container_id = run(
        [
            "docker",
            "create",
            "--platform",
            "linux/amd64",
            "--name",
            prep_name,
            problem["image"],
            "sleep",
            "120",
        ],
        env=env,
    ).stdout.strip().splitlines()[-1]
    try:
        worktree.mkdir(parents=True, exist_ok=True)
        print("[setup] Extracting task worktree from the image", flush=True)
        extraction = run_with_heartbeat(
            ["docker", "cp", f"{container_id}:/testbed/.", str(worktree)],
            label="worktree_extraction",
            env=env,
            timeout=300,
        )
        if extraction.returncode != 0:
            raise RuntimeError(f"Docker worktree extraction failed:\n{extraction.stdout}")
    finally:
        run(["docker", "rm", "-f", prep_name], env=env, check=False)

    if not (worktree / ".git").exists():
        raise RuntimeError("Extracted /testbed does not contain .git")
    prepared_head = run(["git", "rev-parse", "HEAD"], cwd=worktree).stdout.strip()
    merge_base = run(["git", "merge-base", "HEAD", problem["base_commit"]], cwd=worktree).stdout.strip()
    if merge_base != problem["base_commit"]:
        raise RuntimeError(
            f"Prepared image HEAD {prepared_head} does not descend from base {problem['base_commit']}"
        )
    run(["git", "clean", "-fd"], cwd=worktree)
    exclude = worktree / ".git" / "info" / "exclude"
    with exclude.open("a") as file:
        file.write("\n.codex_swebench_exec.py\n.codex_swebench_bridge/\n")
    return prepared_head


BRIDGE_CLIENT = r'''#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

parser = argparse.ArgumentParser(description="Run one command in the isolated SWE-bench task container")
parser.add_argument("command")
parser.add_argument("--timeout", type=float, default=300)
args = parser.parse_args()

bridge = Path(__file__).resolve().parent / ".codex_swebench_bridge"
request_id = uuid.uuid4().hex
request_path = bridge / "requests" / f"{request_id}.json"
response_path = bridge / "responses" / f"{request_id}.json"
temporary_path = request_path.with_suffix(".tmp")
temporary_path.write_text(json.dumps({"command": args.command, "timeout": args.timeout}))
os.replace(temporary_path, request_path)

deadline = time.monotonic() + args.timeout + 30
while not response_path.exists():
    if time.monotonic() >= deadline:
        request_path.unlink(missing_ok=True)
        raise SystemExit("Timed out waiting for the SWE-bench container bridge")
    time.sleep(0.05)
response = json.loads(response_path.read_text())
response_path.unlink(missing_ok=True)
sys.stdout.write(response.get("output", ""))
if response.get("exception"):
    print("\n[bridge exception] " + response["exception"], file=sys.stderr)
code = response.get("returncode", 1)
raise SystemExit(code if isinstance(code, int) and 0 <= code <= 125 else 1)
'''


class CommandBridge:
    def __init__(
        self,
        container_name: str,
        env: dict[str, str],
        log_path: Path,
        worktree: Path,
    ):
        self.container_name = container_name
        self.env = env
        self.log_path = log_path
        self.worktree = worktree
        self.bridge_dir = worktree / ".codex_swebench_bridge"
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def sync_worktree(self) -> list[str]:
        tracked = run(
            ["git", "diff", "--no-renames", "--name-only", "-z", "HEAD"],
            cwd=self.worktree,
        ).stdout.split("\0")
        untracked = run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=self.worktree,
        ).stdout.split("\0")
        paths = sorted({path for path in tracked + untracked if path})
        for relative in paths:
            source = self.worktree / relative
            target = f"/testbed/{relative}"
            if not source.exists() and not source.is_symlink():
                run(
                    ["docker", "exec", self.container_name, "rm", "-rf", "--", target],
                    env=self.env,
                    check=False,
                )
                continue
            parent = str(Path(target).parent)
            run(["docker", "exec", self.container_name, "mkdir", "-p", parent], env=self.env)
            run(["docker", "cp", str(source), f"{self.container_name}:{target}"], env=self.env)
        return paths

    def start(self) -> None:
        (self.bridge_dir / "requests").mkdir(parents=True, exist_ok=True)
        (self.bridge_dir / "responses").mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        requests_dir = self.bridge_dir / "requests"
        while not self._stop_event.is_set():
            request_paths = sorted(requests_dir.glob("*.json"))
            if not request_paths:
                self._stop_event.wait(0.05)
                continue
            for request_path in request_paths:
                self._handle_request(request_path)

    def _handle_request(self, request_path: Path) -> None:
        started = time.monotonic()
        started_at = iso_now()
        command = ""
        synced_paths = []
        try:
            request = json.loads(request_path.read_text())
            request_path.unlink(missing_ok=True)
            command = str(request["command"])
            timeout = min(max(float(request.get("timeout", 300)), 1), 1800)
            synced_paths = self.sync_worktree()
            result = run(
                [
                    "docker",
                    "exec",
                    "-w",
                    "/testbed",
                    "-e",
                    "BASH_ENV=/root/.bashrc",
                    self.container_name,
                    "bash",
                    "-c",
                    command,
                ],
                env=self.env,
                check=False,
                timeout=timeout,
            )
            response = {"returncode": result.returncode, "output": result.stdout[-100_000:]}
        except Exception as error:
            request_path.unlink(missing_ok=True)
            response = {
                "returncode": -1,
                "output": "",
                "exception": f"{type(error).__name__}: {error}",
            }
        record = {
            "started_at": started_at,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "command": command,
            "synced_paths": synced_paths,
            **response,
        }
        with self._lock, self.log_path.open("a") as log:
            log.write(json.dumps(record, sort_keys=True) + "\n")
        response_path = self.bridge_dir / "responses" / request_path.name
        temporary_path = response_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(response))
        os.replace(temporary_path, response_path)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        shutil.rmtree(self.bridge_dir, ignore_errors=True)


def build_prompt(problem: dict) -> str:
    return f"""You are solving SWE-bench Verified instance {problem['instance_id']}.

<problem_statement>
{problem['problem_statement']}
</problem_statement>

Work only in the current repository. Implement a general fix in production source code.
Do not modify tests, packaging, configuration, or the benchmark helper.
Do not commit changes.

Inspect and edit files normally in the host worktree. For commands that require the task's official
Linux environment, especially Python and tests, use:

  python3 .codex_swebench_exec.py --timeout 300 '<command>'

Example:

  python3 .codex_swebench_exec.py --timeout 300 'python -m pytest path/to/test.py -q'

The helper synchronizes current host changes into the isolated container before each command.
Use it for inspection and testing only; do not edit files through the helper because container-side
edits are not copied back. Make every source edit with normal tools in the host worktree.

Run focused tests, inspect `git diff`, clean up scratch files, and leave the final patch in the
working tree. In your final response, summarize the fix and tests. Do not merely describe a patch:
make the edits in the worktree.
"""


def parse_codex_jsonl(path: Path) -> dict:
    summary = {
        "thread_id": None,
        "usage": None,
        "final_message": None,
        "errors": [],
        "event_count": 0,
        "command_executions": 0,
        "commands": [],
        "file_changes": 0,
        "changed_files": [],
    }
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        summary["event_count"] += 1
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            summary["errors"].append({"type": "non_json", "line": line})
            continue
        event_type = event.get("type")
        if event_type == "thread.started":
            summary["thread_id"] = event.get("thread_id")
        elif event_type == "turn.completed":
            summary["usage"] = event.get("usage")
        elif event_type in {"turn.failed", "error"}:
            summary["errors"].append(event)
        elif event_type == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                summary["final_message"] = item.get("text")
            elif item.get("type") == "error":
                summary["errors"].append(item)
            elif item.get("type") == "command_execution":
                summary["command_executions"] += 1
                summary["commands"].append(item.get("command"))
            elif item.get("type") == "file_change":
                summary["file_changes"] += 1
                for change in item.get("changes", []):
                    if isinstance(change, dict) and change.get("path"):
                        summary["changed_files"].append(change["path"])
    summary["commands"] = [command for command in summary["commands"] if command]
    summary["changed_files"] = sorted(set(summary["changed_files"]))
    return summary


def summarize_event_timeline(path: Path, solve_seconds: float) -> dict:
    tool_item_types = {
        "command_execution",
        "file_change",
        "mcp_tool_call",
        "web_search",
    }
    starts: dict[str, dict] = {}
    intervals = []
    by_type: dict[str, float] = {}
    items = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            item_id = record.get("item_id")
            item_type = record.get("item_type")
            event_type = record.get("type")
            if not item_id or not item_type:
                continue
            if item_type not in tool_item_types:
                continue
            if event_type == "item.started":
                starts[item_id] = record
            elif event_type == "item.completed" and item_id in starts:
                start = starts.pop(item_id)
                start_seconds = float(start["elapsed_seconds"])
                end_seconds = float(record["elapsed_seconds"])
                duration = max(0.0, end_seconds - start_seconds)
                intervals.append((start_seconds, end_seconds))
                by_type[item_type] = by_type.get(item_type, 0.0) + duration
                items.append(
                    {
                        "id": item_id,
                        "type": item_type,
                        "seconds": round(duration, 3),
                    }
                )

    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    tool_seconds = sum(end - start for start, end in merged)
    return {
        "method": "Codex JSONL item.started-to-item.completed host arrival timestamps",
        "tool_seconds": round(tool_seconds, 3),
        "by_type_seconds": {key: round(value, 3) for key, value in sorted(by_type.items())},
        "inference_and_orchestration_seconds_estimate": round(
            max(0.0, solve_seconds - tool_seconds), 3
        ),
        "items": items,
        "note": (
            "Legacy wall-minus-tools estimate only. Instrumented runs report "
            "exact client-observed provider lifecycle windows separately in "
            "inference_timing; do not use this estimate for TPS."
        ),
    }


def summarize_container_commands(path: Path) -> dict:
    records = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    test_pattern = re.compile(r"(^|[ /])(pytest|tox|unittest)([ ./'\"-]|$)")
    test_records = [
        record
        for record in records
        if test_pattern.search(str(record.get("command", "")))
    ]
    return {
        "count": len(records),
        "seconds": round(sum(float(record.get("elapsed_seconds", 0)) for record in records), 3),
        "test_count": len(test_records),
        "test_seconds": round(
            sum(float(record.get("elapsed_seconds", 0)) for record in test_records), 3
        ),
    }


def capture_patch(worktree: Path, patch_path: Path) -> str:
    run(["git", "add", "-N", "."], cwd=worktree)
    patch = run(["git", "diff", "--binary", "--", "."], cwd=worktree).stdout
    patch_path.write_text(patch)
    return patch


def find_evaluation_report(report_dir: Path, instance_id: str) -> tuple[Path | None, dict | None]:
    for path in sorted(report_dir.rglob("*.json")):
        try:
            value = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(value, dict) and instance_id in value and isinstance(value[instance_id], dict):
            if "resolved" in value[instance_id]:
                return path, value[instance_id]
        if isinstance(value, dict) and value.get("schema_version") == 2:
            completed_ids = value.get("completed_ids", [])
            if instance_id in completed_ids:
                return path, {
                    "resolved": instance_id in value.get("resolved_ids", []),
                    "summary": value,
                }
    return None, None


def evaluate_run(
    output: Path,
    problem: dict,
    metadata: dict,
    metadata_path: Path,
    env: dict[str, str],
    evaluation_timeout: int,
) -> int:
    report_dir = output / "evaluation"
    report_dir.mkdir(exist_ok=True)
    run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", output.name)
    eval_command = [
        str(EVAL_PYTHON),
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        DATASET,
        "--split",
        SPLIT,
        "--instance_ids",
        problem["instance_id"],
        "--predictions_path",
        str(output / "preds.jsonl"),
        "--max_workers",
        "1",
        "--timeout",
        str(evaluation_timeout),
        "--cache_level",
        "instance",
        "--run_id",
        run_id,
        "--report_dir",
        str(report_dir),
    ]
    metadata["status"] = "evaluating"
    metadata["evaluation_started_at"] = iso_now()
    write_json(metadata_path, metadata)
    print("[evaluation] Running official SWE-bench evaluator", flush=True)
    eval_start = time.time()
    # swebench 4.1.0 accepts --report_dir but its reporting function ignores
    # it and writes relative to cwd. Run from report_dir so the report remains
    # inside this run's durable artifact directory.
    evaluation = run_with_heartbeat(
        eval_command,
        label="evaluation",
        cwd=report_dir,
        env=env,
    )
    (output / "evaluation.log").write_text(evaluation.stdout)
    metadata["evaluation_ended_at"] = iso_now()
    metadata["evaluation_seconds"] = round(time.time() - eval_start, 3)
    metadata["evaluation_returncode"] = evaluation.returncode
    report_path, report = find_evaluation_report(report_dir, problem["instance_id"])
    metadata["evaluation_report"] = str(report_path) if report_path else None
    metadata["evaluation"] = report
    metadata["resolved"] = report.get("resolved") if report else None
    metadata["status"] = "completed" if report is not None else "evaluation_failed"
    metadata["completed_at"] = iso_now()
    write_json(metadata_path, metadata)

    print(f"SOLVE_SECONDS={metadata.get('solve_seconds')}")
    print(f"EVALUATION_SECONDS={metadata['evaluation_seconds']}")
    print(f"RESOLVED={metadata['resolved']}")
    print(f"RUN_METADATA={metadata_path}")
    return 0 if report is not None else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0, help="Index after sorting Verified/test instance IDs")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--reasoning", default=REASONING)
    parser.add_argument("--service-tier", default=SERVICE_TIER)
    parser.add_argument(
        "--codex-bin",
        default="codex",
        help="Codex executable; exact inference timing requires the instrumented binary",
    )
    parser.add_argument(
        "--require-inference-timing",
        action="store_true",
        help="Return nonzero unless every target-model lifecycle call is reconciled",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--worktree-root",
        type=Path,
        default=Path("/tmp/codex-swebench-worktrees"),
        help="Temporary host root visible to Colima; durable artifacts remain under --output",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Validate setup and bridge without running Codex")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-pull", action="store_true", help="Use an already-pulled task image")
    parser.add_argument(
        "--evaluate-existing",
        type=Path,
        help="Evaluate a run directory previously created with --skip-evaluation",
    )
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument(
        "--solve-timeout",
        type=int,
        default=3600,
        help="Terminate a stuck Codex solve after this many seconds",
    )
    parser.add_argument(
        "--codex-sandbox",
        choices=["workspace-write", "danger-full-access"],
        default="workspace-write",
        help=(
            "Codex host sandbox; use danger-full-access only when an outer "
            "managed sandbox blocks nested sandbox-exec"
        ),
    )
    parser.add_argument("--evaluation-timeout", type=int, default=1800)
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("HF_HOME", "/tmp/swe-bench-hf-cache")
    codex_bin = resolve_codex_binary(args.codex_bin)
    require_evaluator = bool(
        args.evaluate_existing
        or (not args.skip_evaluation and not args.prepare_only)
    )
    require_setup(
        env,
        codex_bin,
        require_evaluator=require_evaluator,
    )
    lifecycle_hook_available = binary_has_lifecycle_trace_hook(codex_bin)
    if args.require_inference_timing and not lifecycle_hook_available:
        raise SystemExit(
            "BLOCKER: --require-inference-timing needs the instrumented Codex "
            f"binary, but the lifecycle trace hook is missing from {codex_bin}"
        )
    problem = load_problem(args.index)
    if args.evaluate_existing:
        existing_output = args.evaluate_existing.resolve()
        existing_metadata_path = existing_output / "run_metadata.json"
        metadata = json.loads(existing_metadata_path.read_text())
        if metadata.get("problem", {}).get("instance_id") != problem["instance_id"]:
            raise SystemExit("BLOCKER: existing run instance does not match --index")
        return evaluate_run(
            existing_output,
            problem,
            metadata,
            existing_metadata_path,
            env,
            args.evaluation_timeout,
        )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tier_slug = re.sub(r"[^A-Za-z0-9_.-]", "_", args.service_tier)
    problem_number = args.index + 1
    raw_machine_id = os.environ.get("SWE_BENCH_MACHINE_ID") or os.uname().nodename
    machine_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw_machine_id).strip("-")
    if not machine_id:
        raise SystemExit("BLOCKER: SWE_BENCH_MACHINE_ID is empty after sanitization")
    output = (
        args.output
        or ROOT
        / "runs"
        / machine_id
        / f"codex_problem{problem_number}_{tier_slug}_{stamp}"
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    worktree = (args.worktree_root / output.name).resolve()
    if worktree.exists():
        raise SystemExit(f"BLOCKER: temporary worktree already exists: {worktree}")
    worktree.parent.mkdir(parents=True, exist_ok=True)
    (output / "worktree_path.txt").write_text(str(worktree) + "\n")
    metadata_path = output / "run_metadata.json"
    metadata: dict[str, Any] = {
        "schema_version": 3,
        "status": "initializing",
        "dataset": DATASET,
        "split": SPLIT,
        "problem": problem,
        "requested_model": args.model,
        "requested_reasoning_effort": args.reasoning,
        "requested_service_tier": args.service_tier,
        "codex_sandbox": args.codex_sandbox,
        "resolved_model": None,
        "resolved_service_tier": None,
        "auth_mode": "chatgpt_saved_login",
        "codex_binary": str(codex_bin),
        "lifecycle_trace_hook_available": lifecycle_hook_available,
        "inference_timing_required": args.require_inference_timing,
        "machine_id": machine_id,
        "persistent_state_root": os.environ.get("SWE_BENCH_STATE_ROOT"),
        "python_runtime": {
            "adapter": sys.executable,
            "mini_swe_agent": str(MINI_PYTHON),
            "evaluator": str(EVAL_PYTHON),
            "dont_write_bytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        },
        "created_at": iso_now(),
        "output_dir": str(output),
        "worktree": str(worktree),
    }
    write_json(metadata_path, metadata)
    write_json(output / "problem.json", problem)

    previous_excepthook = sys.excepthook

    def record_uncaught_exception(exc_type, exc_value, exc_traceback) -> None:
        metadata["status"] = "failed"
        metadata["failed_at"] = iso_now()
        metadata["failure"] = {
            "type": exc_type.__name__,
            "message": str(exc_value),
        }
        write_json(metadata_path, metadata)
        previous_excepthook(exc_type, exc_value, exc_traceback)

    sys.excepthook = record_uncaught_exception

    print(f"OUTPUT_DIR={output}")
    print(f"MACHINE_ID={machine_id}")
    print(f"INSTANCE={problem['instance_id']}")
    print(f"MODEL={args.model}")
    print(f"REASONING={args.reasoning}")
    print(f"SERVICE_TIER={args.service_tier}")

    metadata["setup_started_at"] = iso_now()
    setup_start = time.time()
    prepared_head = prepare_worktree(problem, worktree, env, output, args.skip_pull)
    metadata["prepared_head"] = prepared_head
    container_name = f"codex-swe-{uuid.uuid4().hex[:10]}"
    container_id = run(
        [
            "docker",
            "run",
            "-d",
            "--platform",
            "linux/amd64",
            "--name",
            container_name,
            "--rm",
            "-w",
            "/testbed",
            problem["image"],
            "sleep",
            "2h",
        ],
        env=env,
    ).stdout.strip()
    metadata["solve_container"] = {"name": container_name, "id": container_id}
    write_json(metadata_path, metadata)

    helper_path = worktree / ".codex_swebench_exec.py"
    helper_path.write_text(BRIDGE_CLIENT)
    helper_path.chmod(0o755)
    bridge = CommandBridge(
        container_name,
        env,
        output / "container_commands.jsonl",
        worktree,
    )
    bridge.start()

    try:
        bridge_probe = run(
            [sys.executable, str(helper_path), "pwd && git rev-parse HEAD"], cwd=worktree, check=False
        )
        (output / "bridge_preflight.log").write_text(bridge_probe.stdout)
        if bridge_probe.returncode != 0 or prepared_head not in bridge_probe.stdout:
            raise RuntimeError(f"Container bridge preflight failed:\n{bridge_probe.stdout}")
        print("[preflight] Container command bridge is healthy", flush=True)

        metadata["status"] = "prepared"
        metadata["setup_ended_at"] = iso_now()
        metadata["setup_seconds"] = round(time.time() - setup_start, 3)
        write_json(metadata_path, metadata)
        if args.prepare_only:
            print("PREPARE_ONLY_OK")
            return 0

        codex_version = run([str(codex_bin), "--version"], env=env).stdout.strip()
        metadata["codex_version"] = codex_version
        prompt_path = output / "codex_prompt.md"
        prompt_path.write_text(build_prompt(problem))
        jsonl_path = output / "codex_events.jsonl"
        timeline_path = output / "codex_timeline.jsonl"
        stderr_path = output / "codex_stderr.log"
        inference_calls_path = output / "inference_calls.jsonl"
        tool_intervals_path = output / "tool_intervals.jsonl"
        codex_command = [
            str(codex_bin),
            "exec",
            "--json",
            "--ephemeral",
            "--ignore-user-config",
            "--model",
            args.model,
            "--config",
            f'model_reasoning_effort="{args.reasoning}"',
        ]
        if args.service_tier != "default":
            codex_command.extend(["--config", f'service_tier="{args.service_tier}"'])
        codex_command.extend([
            "--config",
            'approval_policy="on-request"',
            "--config",
            'web_search="disabled"',
            "--sandbox",
            args.codex_sandbox,
            "--skip-git-repo-check",
            "--cd",
            str(worktree),
            prompt_path.read_text(),
        ])
        codex_env = env.copy()
        codex_env.pop("OPENAI_API_KEY", None)
        codex_env.pop("CODEX_API_KEY", None)
        codex_env["RUST_LOG"] = f"error,{LIFECYCLE_TRACE_TARGET}=trace"
        metadata["status"] = "solving"
        metadata["solve_started_at"] = iso_now()
        write_json(metadata_path, metadata)
        solve_start = time.monotonic()
        solve_start_ns = time.time_ns()
        print(f"[solve] Starting {shlex.join(codex_command[:-1])}", flush=True)
        lifecycle_traces: list[dict[str, Any]] = []
        malformed_lifecycle_traces: list[str] = []
        with (
            jsonl_path.open("w") as jsonl_file,
            timeline_path.open("w") as timeline_file,
            stderr_path.open("w") as stderr_file,
        ):
            process = subprocess.Popen(
                codex_command,
                cwd=worktree,
                env=codex_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
            )
            assert process.stdout is not None and process.stderr is not None
            heartbeat_stop = threading.Event()

            def drain_stderr() -> None:
                for stderr_line in process.stderr:
                    stderr_file.write(stderr_line)
                    stderr_file.flush()
                    if LIFECYCLE_TRACE_TARGET not in stderr_line:
                        continue
                    trace = parse_lifecycle_trace_line(stderr_line)
                    if trace is None:
                        malformed_lifecycle_traces.append(stderr_line.rstrip())
                        continue
                    trace["lifecycle_trace_index"] = len(lifecycle_traces) + 1
                    lifecycle_traces.append(trace)
                    request_seconds = trace["request_to_completed_ms"] / 1000
                    output_tokens = trace["output_tokens"]
                    print(
                        json.dumps(
                            {
                                "status": "inference_micro_session",
                                "call_index": len(lifecycle_traces),
                                "model": trace["model"],
                                "warmup": trace["warmup"],
                                "provider_start_kind": trace["provider_start_kind"],
                                "end_to_end_inference_seconds": round(request_seconds, 6),
                                "output_tokens": output_tokens,
                                "end_to_end_billed_tps": (
                                    round(output_tokens / request_seconds, 3)
                                    if request_seconds > 0 else None
                                ),
                                "note": "tool overlap is reconciled at query end",
                            }
                        ),
                        flush=True,
                    )

            stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
            stderr_thread.start()

            def emit_running_heartbeat() -> None:
                while not heartbeat_stop.wait(30):
                    target_calls = [
                        trace
                        for trace in lifecycle_traces
                        if trace.get("model") == args.model
                        and trace.get("warmup") is not True
                    ]
                    print(
                        json.dumps(
                            {
                                "status": "running",
                                "problem_number": problem_number,
                                "instance_id": problem["instance_id"],
                                "elapsed_s": round(time.monotonic() - solve_start, 1),
                                "inference_micro_sessions_completed": len(
                                    lifecycle_traces
                                ),
                                "target_model_micro_sessions_completed": len(
                                    target_calls
                                ),
                                "target_output_tokens_completed": sum(
                                    int(trace["output_tokens"])
                                    for trace in target_calls
                                ),
                            }
                        ),
                        flush=True,
                    )

            heartbeat_thread = threading.Thread(
                target=emit_running_heartbeat, daemon=True
            )
            heartbeat_thread.start()
            solve_watchdog_stop = threading.Event()
            solve_timed_out = threading.Event()

            def enforce_solve_timeout() -> None:
                if solve_watchdog_stop.wait(args.solve_timeout):
                    return
                solve_timed_out.set()
                print(
                    f"[solve] timeout after {args.solve_timeout}s; terminating Codex",
                    flush=True,
                )
                process.terminate()
                if not solve_watchdog_stop.wait(15) and process.poll() is None:
                    process.kill()

            solve_watchdog = threading.Thread(
                target=enforce_solve_timeout, daemon=True
            )
            solve_watchdog.start()
            for line in process.stdout:
                jsonl_file.write(line)
                jsonl_file.flush()
                try:
                    received_unix_ns = time.time_ns()
                    event = json.loads(line)
                    event_type = event.get("type")
                    item = event.get("item", {})
                    timeline_file.write(
                        json.dumps(
                            {
                                "received_at": datetime.now(timezone.utc).astimezone().isoformat(
                                    timespec="milliseconds"
                                ),
                                "received_unix_ns": received_unix_ns,
                                "elapsed_seconds": round(
                                    (received_unix_ns - solve_start_ns) / 1_000_000_000,
                                    6,
                                ),
                                "type": event_type,
                                "item_id": item.get("id"),
                                "item_type": item.get("type"),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    timeline_file.flush()
                    if event_type == "thread.started":
                        print(f"[codex] thread={event.get('thread_id')}", flush=True)
                    elif event_type == "item.completed":
                        print(f"[codex] completed {item.get('type')}", flush=True)
                    elif event_type in {"turn.completed", "turn.failed", "error"}:
                        print(f"[codex] {event_type}", flush=True)
                except json.JSONDecodeError:
                    print("[codex] non-JSON output", flush=True)
            codex_returncode = process.wait()
            solve_watchdog_stop.set()
            solve_watchdog.join(timeout=1)
            solve_end_ns = time.time_ns()
            solve_elapsed_seconds = time.monotonic() - solve_start
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1)
            stderr_thread.join(timeout=10)
        metadata["solve_ended_at"] = iso_now()
        metadata["solve_seconds"] = round(solve_elapsed_seconds, 3)
        metadata["codex_returncode"] = codex_returncode
        metadata["solve_timed_out"] = solve_timed_out.is_set()
        metadata["codex"] = parse_codex_jsonl(jsonl_path)
        metadata["timing_breakdown"] = summarize_event_timeline(
            timeline_path, metadata["solve_seconds"]
        )
        inference_calls, inference_timing, tool_intervals = summarize_lifecycle(
            lifecycle_traces,
            load_jsonl(timeline_path),
            args.model,
            metadata["codex"].get("usage") or {},
            solve_start_ns,
            solve_end_ns,
            len(malformed_lifecycle_traces),
        )
        inference_calls_path.write_text(
            "".join(json.dumps(call, sort_keys=True) + "\n" for call in inference_calls)
        )
        tool_intervals_path.write_text(
            "".join(json.dumps(tool, sort_keys=True) + "\n" for tool in tool_intervals)
        )
        metadata["inference_timing"] = inference_timing
        metadata["timing_breakdown"].update(
            {
                "end_to_end_inference_seconds": inference_timing[
                    "end_to_end_inference_seconds"
                ],
                "end_to_end_billed_tps": inference_timing[
                    "end_to_end_billed_tps"
                ],
                "provider_window_inference_seconds": inference_timing[
                    "provider_window_inference_seconds"
                ],
                "provider_window_output_tps": inference_timing[
                    "provider_window_output_tps"
                ],
                "inference_timing_coverage": inference_timing["coverage"],
                "target_inference_tool_concurrency_seconds": inference_timing[
                    "target_inference_tool_concurrency_seconds"
                ],
            }
        )
        print(
            "[timing] "
            f"coverage={inference_timing['coverage']} "
            f"calls={inference_timing['primary_eligible_call_count']} "
            f"tokens={inference_timing['primary_output_tokens']} "
            f"request_s={inference_timing['end_to_end_inference_seconds']:.6f} "
            f"tool_s={inference_timing['total_tool_seconds']:.6f} "
            f"end_to_end_tps={inference_timing['end_to_end_billed_tps']}",
            flush=True,
        )

        patch = capture_patch(worktree, output / "model.patch")
        metadata["patch"] = {
            "bytes": len(patch.encode()),
            "added_lines": sum(line.startswith("+") and not line.startswith("+++") for line in patch.splitlines()),
            "removed_lines": sum(line.startswith("-") and not line.startswith("---") for line in patch.splitlines()),
            "nonempty": bool(patch.strip()),
        }
        prediction = {
            "instance_id": problem["instance_id"],
            "model_name_or_path": f"codex__{args.model}__{args.reasoning}__{args.service_tier}",
            "model_patch": patch,
        }
        predictions_path = output / "preds.jsonl"
        predictions_path.write_text(json.dumps(prediction) + "\n")
        write_json(output / "preds.json", {problem["instance_id"]: prediction})
        metadata["status"] = (
            "solve_failed"
            if codex_returncode != 0 or solve_timed_out.is_set()
            else "patch_captured"
            if patch.strip()
            else "completed_no_patch"
        )
        metadata["inference_timing_complete"] = (
            inference_timing["coverage"] == "complete"
        )
        write_json(metadata_path, metadata)
    finally:
        bridge.stop()
        helper_path.unlink(missing_ok=True)
        if not args.keep_container:
            run(["docker", "rm", "-f", container_name], env=env, check=False)
        metadata["container_commands"] = summarize_container_commands(
            output / "container_commands.jsonl"
        )
        shutil.rmtree(worktree, ignore_errors=True)
        metadata["worktree_cleaned"] = not worktree.exists()
        write_json(metadata_path, metadata)

    if metadata["status"] == "solve_failed":
        print(f"SOLVE_FAILED: see {metadata_path}", file=sys.stderr)
        return 1
    timing_incomplete = (
        args.require_inference_timing
        and not metadata.get("inference_timing_complete", False)
    )
    if args.skip_evaluation:
        metadata["status"] = (
            "completed_with_incomplete_inference_timing"
            if timing_incomplete else "completed_without_evaluation"
            if metadata["status"] == "patch_captured"
            else "completed_without_evaluation_no_patch"
        )
        write_json(metadata_path, metadata)
        print(f"RUN_METADATA={metadata_path}")
        if timing_incomplete:
            print(
                "INFERENCE_TIMING_INCOMPLETE: inspect inference_timing.coverage_reasons",
                file=sys.stderr,
            )
            return 3
        return 0

    evaluation_returncode = evaluate_run(
        output,
        problem,
        metadata,
        metadata_path,
        env,
        args.evaluation_timeout,
    )
    if timing_incomplete:
        metadata["status"] = "completed_with_incomplete_inference_timing"
        write_json(metadata_path, metadata)
        print(
            "INFERENCE_TIMING_INCOMPLETE: inspect inference_timing.coverage_reasons",
            file=sys.stderr,
        )
        return 3
    return evaluation_returncode


if __name__ == "__main__":
    raise SystemExit(main())
