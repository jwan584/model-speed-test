#!/usr/bin/env python3
"""Solve and evaluate one numbered SWE-bench Verified problem with native Claude Code."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
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

from claude_inference_timing import (
    build_inference_calls,
    build_otel_timing_records,
    build_timeline,
    load_jsonl,
    pair_tool_intervals,
    summarize_inference_timing,
    summarize_otel_inference_timing,
)
from claude_otel_receiver import ClaudeOtelTraceReceiver


ROOT = Path(__file__).resolve().parent
EVAL_PYTHON = Path(
    os.environ.get(
        "SWE_BENCH_EVAL_PYTHON",
        ROOT / ".venv_eval" / "bin" / "python",
    )
)
DATASET = "princeton-nlp/SWE-Bench_Verified"
SPLIT = "test"
MODEL = "claude-sonnet-5"
EFFORT = "high"
ALLOWED_TOOLS = "Bash,Read,Edit,Write,Glob,Grep"
POST_TERMINAL_EXIT_GRACE_SECONDS = 30.0


def terminal_result_is_success(result_event: dict[str, Any] | None) -> bool:
    return bool(
        result_event
        and result_event.get("type") == "result"
        and result_event.get("subtype") == "success"
        and result_event.get("is_error") is not True
        and result_event.get("duration_api_ms") is not None
        and result_event.get("modelUsage")
    )


def solve_process_succeeded(
    returncode: int,
    result_event: dict[str, Any] | None,
    post_terminal_teardown_forced: bool,
) -> bool:
    return returncode == 0 or (
        post_terminal_teardown_forced
        and terminal_result_is_success(result_event)
    )


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


def resolve_claude_binary(value: str) -> Path:
    resolved = shutil.which(value)
    if resolved is None:
        candidate = Path(value).expanduser()
        if candidate.is_file():
            resolved = str(candidate.resolve())
    if resolved is None:
        raise SystemExit(f"BLOCKER: Claude Code executable not found: {value}")
    return Path(resolved).resolve()


def require_setup(
    env: dict[str, str],
    claude_bin: Path,
    *,
    require_evaluator: bool,
) -> dict[str, Any]:
    blockers = []
    for command in ["docker", "git"]:
        if shutil.which(command) is None:
            blockers.append(f"{command} is not installed or not on PATH")
    if require_evaluator and not EVAL_PYTHON.exists():
        blockers.append(f"SWE-bench evaluator Python is missing at {EVAL_PYTHON}")
    if not env.get("DOCKER_HOST"):
        blockers.append("DOCKER_HOST is not set")
    if not env.get("DOCKER_CONFIG"):
        blockers.append("DOCKER_CONFIG is not set")
    if blockers:
        raise SystemExit("\n".join(f"BLOCKER: {item}" for item in blockers))

    probe = run(["docker", "version", "--format", "{{.Server.Version}}"], env=env, check=False)
    if probe.returncode != 0 or "warning" in probe.stdout.lower():
        raise SystemExit(f"BLOCKER: Docker preflight failed or emitted a warning:\n{probe.stdout}")

    auth = run([str(claude_bin), "auth", "status"], env=env, check=False)
    try:
        auth_status = json.loads(auth.stdout)
    except json.JSONDecodeError:
        auth_status = None
    if auth.returncode != 0 or not auth_status or not auth_status.get("loggedIn"):
        raise SystemExit(f"BLOCKER: Claude Code is not authenticated:\n{auth.stdout}")
    return auth_status


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

    prep_name = f"claude-swe-prep-{uuid.uuid4().hex[:10]}"
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
        file.write("\n.claude_swebench_exec.py\n.claude_swebench_bridge/\n")
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

bridge = Path(__file__).resolve().parent / ".claude_swebench_bridge"
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
        self.bridge_dir = worktree / ".claude_swebench_bridge"
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

Use your Read, Edit, and Write tools directly on files in this host worktree; those edits
are the ones that will be graded. For any command that requires the task's official Linux
environment, especially Python and tests, run it with your Bash tool as:

  python3 .claude_swebench_exec.py --timeout 300 '<command>'

Example:

  python3 .claude_swebench_exec.py --timeout 300 'python -m pytest path/to/test.py -q'

The helper synchronizes current host changes into the isolated container before each command.
Use it for inspection and testing only; do not edit files through it because container-side
edits are not copied back. Make every source edit with your normal Edit/Write tools in the host
worktree. Do not run pytest, python, or other project commands directly with Bash on the host --
the host's Python environment does not match the task's Linux environment.

Run focused tests, inspect `git diff`, clean up scratch files, and leave the final patch in the
working tree. In your final response, summarize the fix and tests. Do not merely describe a
patch: make the edits in the worktree.
"""


def parse_claude_events(events: list[dict[str, Any]]) -> dict:
    summary = {
        "session_id": None,
        "final_message": None,
        "errors": [],
        "event_count": len(events),
        "command_executions": 0,
        "commands": [],
        "file_changes": 0,
        "changed_files": [],
    }
    changed_files: set[str] = set()
    for event in events:
        event_type = event.get("type")
        if event_type == "system" and event.get("subtype") == "init":
            summary["session_id"] = event.get("session_id")
        elif event_type == "assistant":
            message = event.get("message", {}) or {}
            for block in message.get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                name = block.get("name")
                block_input = block.get("input") or {}
                if name == "Bash":
                    summary["command_executions"] += 1
                    command = block_input.get("command")
                    if command:
                        summary["commands"].append(command)
                elif name in {"Edit", "Write"}:
                    summary["file_changes"] += 1
                    file_path = block_input.get("file_path")
                    if file_path:
                        changed_files.add(file_path)
        elif event_type == "result":
            summary["final_message"] = event.get("result")
            if event.get("is_error"):
                summary["errors"].append(
                    {
                        "subtype": event.get("subtype"),
                        "terminal_reason": event.get("terminal_reason"),
                        "result": event.get("result"),
                    }
                )
    summary["changed_files"] = sorted(changed_files)
    return summary


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
    parser.add_argument("--effort", default=EFFORT, help="Reasoning effort: low, medium, high, xhigh, max")
    parser.add_argument("--claude-bin", default="claude", help="Claude Code executable")
    parser.add_argument(
        "--require-complete-inference-timing",
        action="store_true",
        help="Return nonzero unless inference_timing.coverage is 'complete'",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--worktree-root",
        type=Path,
        default=Path("/tmp/claude-swebench-worktrees"),
        help="Temporary host root visible to Colima; durable artifacts remain under --output",
    )
    parser.add_argument("--prepare-only", action="store_true", help="Validate setup and bridge without running Claude")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-pull", action="store_true", help="Use an already-pulled task image")
    parser.add_argument(
        "--evaluate-existing",
        type=Path,
        help="Evaluate a run directory previously created with --skip-evaluation",
    )
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument("--evaluation-timeout", type=int, default=1800)
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("HF_HOME", "/tmp/swe-bench-hf-cache")
    claude_bin = resolve_claude_binary(args.claude_bin)
    require_evaluator = bool(
        args.evaluate_existing
        or (not args.skip_evaluation and not args.prepare_only)
    )
    auth_status = require_setup(
        env,
        claude_bin,
        require_evaluator=require_evaluator,
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
    effort_slug = re.sub(r"[^A-Za-z0-9_.-]", "_", args.effort)
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
        / f"claude_problem{problem_number}_{effort_slug}_{stamp}"
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    worktree = (args.worktree_root / output.name).resolve()
    if worktree.exists():
        raise SystemExit(f"BLOCKER: temporary worktree already exists: {worktree}")
    worktree.parent.mkdir(parents=True, exist_ok=True)
    (output / "worktree_path.txt").write_text(str(worktree) + "\n")
    metadata_path = output / "run_metadata.json"
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "status": "initializing",
        "dataset": DATASET,
        "split": SPLIT,
        "problem": problem,
        "requested_model": args.model,
        "requested_effort": args.effort,
        "resolved_model": None,
        "resolved_permission_mode": None,
        "resolved_tools": None,
        "auth_mode": auth_status.get("authMethod"),
        "auth_account": auth_status.get("email"),
        "claude_binary": str(claude_bin),
        "inference_timing_required_complete": args.require_complete_inference_timing,
        "machine_id": machine_id,
        "persistent_state_root": os.environ.get("SWE_BENCH_STATE_ROOT"),
        "python_runtime": {
            "adapter": sys.executable,
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
    print(f"EFFORT={args.effort}")

    metadata["setup_started_at"] = iso_now()
    setup_start = time.time()
    prepared_head = prepare_worktree(problem, worktree, env, output, args.skip_pull)
    metadata["prepared_head"] = prepared_head
    container_name = f"claude-swe-{uuid.uuid4().hex[:10]}"
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

    helper_path = worktree / ".claude_swebench_exec.py"
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

        claude_version = run([str(claude_bin), "--version"], env=env).stdout.strip()
        metadata["claude_version"] = claude_version
        prompt_path = output / "claude_prompt.md"
        prompt_path.write_text(build_prompt(problem))
        jsonl_path = output / "claude_events.jsonl"
        claude_command = [
            str(claude_bin),
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            args.model,
            "--effort",
            args.effort,
            "--setting-sources",
            "",
            "--disable-slash-commands",
            "--strict-mcp-config",
            "--tools",
            ALLOWED_TOOLS,
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            prompt_path.read_text(),
        ]
        metadata["status"] = "solving"
        metadata["solve_started_at"] = iso_now()
        write_json(metadata_path, metadata)
        solve_start = time.monotonic()
        solve_start_ns = time.time_ns()
        print(f"[solve] Starting {shlex.join(claude_command[:-1])}", flush=True)
        stream_events: list[dict[str, Any]] = []
        result_event: dict[str, Any] | None = None
        otel_receiver = ClaudeOtelTraceReceiver()
        otel_receiver.start()
        claude_env = otel_receiver.telemetry_environment(env)
        metadata["otel_trace_capture"] = {
            "enabled": True,
            "transport": "loopback_otlp_http_json",
            "raw_payloads_persisted": False,
            "content_detail_flags_enabled": False,
        }
        try:
            with jsonl_path.open("w") as jsonl_file:
                process = subprocess.Popen(
                    claude_command,
                    cwd=worktree,
                    env=claude_env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=1,
                    start_new_session=True,
                )
                assert process.stdout is not None and process.stderr is not None
                heartbeat_stop = threading.Event()
                stderr_path = output / "claude_stderr.log"
                stderr_bytes = 0
                stderr_lock = threading.Lock()

                def drain_stderr() -> None:
                    nonlocal stderr_bytes
                    with stderr_path.open("w") as stderr_file:
                        for stderr_line in process.stderr:
                            stderr_file.write(stderr_line)
                            stderr_file.flush()
                            with stderr_lock:
                                stderr_bytes += len(stderr_line.encode())

                stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
                stderr_thread.start()

                def emit_running_heartbeat() -> None:
                    while not heartbeat_stop.wait(30):
                        target_calls = [
                            event
                            for event in stream_events
                            if event.get("type") == "assistant"
                            and (event.get("message") or {}).get("model") == args.model
                        ]
                        output_tokens = sum(
                            int(((event.get("message") or {}).get("usage") or {}).get("output_tokens") or 0)
                            for event in target_calls
                        )
                        with stderr_lock:
                            observed_stderr_bytes = stderr_bytes
                        otel_live = otel_receiver.diagnostics()
                        print(
                            json.dumps(
                                {
                                    "status": "running",
                                    "problem_number": problem_number,
                                    "instance_id": problem["instance_id"],
                                    "elapsed_s": round(time.monotonic() - solve_start, 1),
                                    "assistant_messages_completed": len(target_calls),
                                    "target_output_tokens_completed": output_tokens,
                                    "stderr_bytes": observed_stderr_bytes,
                                    "otel_payload_count": otel_live["payload_count"],
                                    "otel_receiver_error_count": otel_live[
                                        "receiver_error_count"
                                    ],
                                }
                            ),
                            flush=True,
                        )

                heartbeat_thread = threading.Thread(
                    target=emit_running_heartbeat, daemon=True
                )
                heartbeat_thread.start()
                terminal_seen = threading.Event()
                post_terminal_teardown_forced = threading.Event()

                def enforce_post_terminal_exit_grace() -> None:
                    if not terminal_seen.wait():
                        return
                    if heartbeat_stop.wait(POST_TERMINAL_EXIT_GRACE_SECONDS):
                        return
                    if process.poll() is None:
                        post_terminal_teardown_forced.set()
                        print(
                            json.dumps(
                                {
                                    "status": "post_terminal_teardown_forced",
                                    "grace_seconds": POST_TERMINAL_EXIT_GRACE_SECONDS,
                                }
                            ),
                            flush=True,
                        )
                        os.killpg(process.pid, signal.SIGTERM)

                teardown_watchdog = threading.Thread(
                    target=enforce_post_terminal_exit_grace, daemon=True
                )
                teardown_watchdog.start()
                try:
                    for line in process.stdout:
                        jsonl_file.write(line)
                        jsonl_file.flush()
                        received_unix_ns = time.time_ns()
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            print("[claude] non-JSON output", flush=True)
                            continue
                        event["_received_unix_ns"] = received_unix_ns
                        stream_events.append(event)
                        event_type = event.get("type")
                        if event_type == "system" and event.get("subtype") == "init":
                            metadata["resolved_model"] = event.get("model")
                            metadata["resolved_permission_mode"] = event.get("permissionMode")
                            metadata["resolved_tools"] = event.get("tools")
                            print(f"[claude] session={event.get('session_id')} model={event.get('model')}", flush=True)
                        elif event_type == "assistant":
                            message = event.get("message", {}) or {}
                            usage = message.get("usage") or {}
                            for block in message.get("content") or []:
                                if isinstance(block, dict) and block.get("type") == "tool_use":
                                    print(f"[claude] tool_use {block.get('name')}", flush=True)
                            print(
                                json.dumps(
                                    {
                                        "status": "inference_micro_session",
                                        "call_index": sum(
                                            1 for e in stream_events if e.get("type") == "assistant"
                                        ),
                                        "model": message.get("model"),
                                        "output_tokens": usage.get("output_tokens"),
                                        "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
                                        "timing": "pending_otel_span_flush",
                                    }
                                ),
                                flush=True,
                            )
                        elif event_type == "result":
                            result_event = event
                            terminal_seen.set()
                            print(f"[claude] result subtype={event.get('subtype')}", flush=True)
                    claude_returncode = process.wait()
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                    solve_end_ns = time.time_ns()
                    solve_elapsed_seconds = time.monotonic() - solve_start
                    heartbeat_stop.set()
                    heartbeat_thread.join(timeout=1)
                    stderr_thread.join(timeout=10)
                    teardown_watchdog.join(timeout=1)
        finally:
            otel_receiver.wait_for_quiet()
            otel_payloads = otel_receiver.payloads()
            otel_receiver_diagnostics = otel_receiver.diagnostics()
            otel_receiver.stop()
        metadata["solve_ended_at"] = iso_now()
        metadata["solve_seconds"] = round(solve_elapsed_seconds, 3)
        metadata["claude_returncode"] = claude_returncode
        metadata["post_terminal_teardown_forced"] = (
            post_terminal_teardown_forced.is_set()
        )
        metadata["claude"] = parse_claude_events(stream_events)

        timeline = build_timeline(stream_events, solve_start_ns)
        timeline_path = output / "claude_timeline.jsonl"
        timeline_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in timeline)
        )
        host_tools, tool_pairing = pair_tool_intervals(
            timeline, solve_start_ns, solve_end_ns
        )
        target_models = {
            str(model)
            for model in (args.model, metadata.get("resolved_model"))
            if model
        }
        otel_calls, otel_tools, normalization_diagnostics = build_otel_timing_records(
            otel_payloads,
            target_models,
            solve_start_ns,
            solve_end_ns,
        )
        otel_diagnostics = {
            **otel_receiver_diagnostics,
            **normalization_diagnostics,
            "stream_tool_interval_count": len(host_tools),
        }
        write_json(output / "otel_trace_diagnostics.json", otel_diagnostics)

        if otel_calls:
            inference_calls = otel_calls
            if otel_tools or not host_tools:
                tools = otel_tools
                tool_timing_basis = "claude_code_otel_tool_spans"
            else:
                tools = host_tools
                tool_timing_basis = "stream_json_host_arrival_fallback"
            inference_timing = summarize_otel_inference_timing(
                result_event,
                inference_calls,
                tools,
                args.model,
                solve_start_ns,
                solve_end_ns,
                otel_diagnostics,
                tool_timing_basis=tool_timing_basis,
            )
        else:
            tools = host_tools
            inference_calls = build_inference_calls(
                stream_events, args.model, solve_start_ns
            )
            inference_timing = summarize_inference_timing(
                result_event,
                inference_calls,
                tools,
                tool_pairing,
                args.model,
                solve_start_ns,
                solve_end_ns,
            )
            inference_timing["coverage"] = "partial"
            inference_timing.setdefault("coverage_reasons", []).append(
                "otel_llm_request_spans_unavailable"
            )
            inference_timing["otel"] = otel_diagnostics

        tool_intervals_path = output / "tool_intervals.jsonl"
        tool_intervals_path.write_text(
            "".join(json.dumps(tool, sort_keys=True) + "\n" for tool in tools)
        )
        inference_calls_path = output / "inference_calls.jsonl"
        inference_calls_path.write_text(
            "".join(json.dumps(call, sort_keys=True) + "\n" for call in inference_calls)
        )
        metadata["inference_timing"] = inference_timing
        if "primary_request_seconds_sum" in inference_timing:
            print(
                "[timing:target_otel_diagnostic] "
                f"coverage={inference_timing['target_otel_diagnostic_coverage']} "
                f"calls={inference_timing['primary_call_count']} "
                f"tokens={inference_timing['primary_output_tokens']} "
                f"request_s={inference_timing['primary_request_seconds_sum']:.6f} "
                f"tool_s={inference_timing['total_tool_seconds']:.6f} "
                f"tps={inference_timing['target_otel_request_output_tps_diagnostic']}",
                flush=True,
            )
            print(
                "[timing:all_model_agent_request_headline] "
                f"coverage={inference_timing['coverage']} "
                f"tokens={inference_timing['all_model_terminal_billed_output_tokens']} "
                f"request_s={inference_timing['terminal_request_active_seconds']:.6f} "
                f"tps={inference_timing['end_to_end_billed_tps']}",
                flush=True,
            )
            partition = inference_timing["wall_partition"]
            print(
                "[wall] "
                f"llm_only_s={partition['llm_request_only_seconds']:.6f} "
                f"tool_only_s={partition['tool_only_seconds']:.6f} "
                f"overlap_s={partition['llm_request_tool_overlap_seconds']:.6f} "
                f"residual_s={partition['orchestration_residual_seconds']:.6f} "
                f"total_s={inference_timing['total_wall_seconds']:.6f}",
                flush=True,
            )
        else:
            print(
                "[timing] "
                f"coverage={inference_timing['coverage']} "
                f"calls={inference_timing['primary_call_count']} "
                f"tokens={inference_timing['primary_output_tokens']} "
                f"api_s={inference_timing['cli_reported_api_seconds']} "
                f"tool_s={inference_timing['host_observed_tool_seconds']:.6f} "
                f"tps={inference_timing['cli_reported_output_tps']}",
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
            "model_name_or_path": f"claude__{args.model}__{args.effort}",
            "model_patch": patch,
        }
        predictions_path = output / "preds.jsonl"
        predictions_path.write_text(json.dumps(prediction) + "\n")
        write_json(output / "preds.json", {problem["instance_id"]: prediction})
        metadata["status"] = (
            "patch_captured"
            if patch.strip()
            and solve_process_succeeded(
                claude_returncode,
                result_event,
                post_terminal_teardown_forced.is_set(),
            )
            else "solve_failed"
        )
        metadata["inference_timing_complete"] = inference_timing["coverage"] == "complete"
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
        args.require_complete_inference_timing
        and not metadata.get("inference_timing_complete", False)
    )
    if args.skip_evaluation:
        metadata["status"] = (
            "completed_with_incomplete_inference_timing"
            if timing_incomplete else "completed_without_evaluation"
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
