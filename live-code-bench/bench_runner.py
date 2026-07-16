#!/usr/bin/env python3
"""Time and evaluate models on LiveCodeBench or self-contained custom tasks."""

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEPENDENCY_INSTALL_HINT = (
    "Install benchmark dependencies with: .venv/bin/pip install "
    "'openai>=2.0.0' 'anthropic>=0.42.0' "
    "'datasets>=3.2.0,<4.0.0' 'packaging>=24.0' 'pebble>=5.1.0'"
)

try:
    from openai import APIError, OpenAI

    from lcb_runner.benchmarks import load_code_generation_dataset
    from lcb_runner.evaluation.compute_code_generation_metrics import (
        evaluate_generations_by_problem,
    )
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.prompts import format_prompt_generation
    from lcb_runner.utils.extraction_utils import extract_code
except ModuleNotFoundError as exc:
    print(
        f"error: benchmark dependency {exc.name!r} is missing. "
        f"{DEPENDENCY_INSTALL_HINT}",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


CSV_FIELDS = [
    "timestamp", "harness_version", "config_fingerprint", "endpoint_name",
    "provider", "model", "base_url", "endpoint_region", "thinking_effort", "max_tokens",
    "release_version", "adaptive_thinking", "checker_timeout", "prompt_style",
    "request_timeout_s", "sandbox_image", "cpu_limit", "memory_limit",
    "tool_configuration", "economy_policy", "problem_id", "difficulty",
    "problem_number", "run_idx", "attempt_idx", "request_status",
    "inference_outcome", "stop_reason",
    "request_id", "input_tokens", "billed_output_tokens", "reasoning_tokens",
    "visible_output_tokens", "visible_output_tokens_approx", "ttft_s",
    "first_stream_event_s", "response_created_s",
    "provider_window_inference_time_s", "first_observable_output_s",
    "last_observable_output_s", "observable_chunk_count", "gen_time_s",
    "generation_start_s", "generation_start_event_type",
    "generation_start_event_detail", "generation_start_confidence",
    "hidden_reasoning_observability", "terminal_event_s",
    "observed_pre_generation_s", "active_generation_time_s",
    "generation_wall_s", "inference_time_s", "tool_time_s", "retry_api_time_s",
    "backoff_time_s", "harness_overhead_s", "total_wall_s",
    "provider_window_billed_tps", "active_generation_billed_tps",
    "end_to_end_billed_tps",
    "passed", "artifact_path",
    "error_type", "error_message",
]
INDEX_SELECTOR_RE = re.compile(r"^(easy|medium|hard):([1-9][0-9]*)$")
INDEX_RANGE_RE = re.compile(r"^(easy|medium|hard):([1-9][0-9]*)-([1-9][0-9]*)$")
HARNESS_VERSION = "6"
PROMPT_STYLE = "OpenAIChat"


@dataclass(frozen=True)
class EndpointConfig:
    endpoint_name: str
    provider: str
    model: str
    api_key: str
    base_url: str | None
    endpoint_region: str | None
    thinking_effort: str | None


@dataclass(frozen=True)
class CustomDifficulty:
    value: str = "custom"


@dataclass(frozen=True)
class CustomTask:
    question_id: str
    prompt: str
    required_patterns: tuple[str, ...]
    difficulty: CustomDifficulty = CustomDifficulty()


class RequestDeadlineExceeded(TimeoutError):
    """Raised when one benchmark tuple exceeds its total wall-clock deadline."""

    def __init__(self, message: str, elapsed_s: float | None = None):
        super().__init__(message)
        self.elapsed_s = elapsed_s


class WorkerRequestError(RuntimeError):
    """Preserve an isolated request worker's original exception type."""

    def __init__(
        self, remote_type: str, message: str, elapsed_s: float | None = None
    ):
        super().__init__(message)
        self.remote_type = remote_type
        self.elapsed_s = elapsed_s


@contextmanager
def request_deadline(seconds: float):
    """Enforce a whole-call deadline, independent of SDK per-read timeouts."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def raise_deadline(signum, frame):
        raise RequestDeadlineExceeded(
            f"Request exceeded the {seconds:g}s total wall-clock deadline"
        )

    signal.signal(signal.SIGALRM, raise_deadline)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def config_fingerprint(config: EndpointConfig, args: argparse.Namespace) -> str:
    """Hash all non-secret settings that affect generation or evaluation."""
    payload = {
        "harness_version": HARNESS_VERSION,
        "endpoint_name": config.endpoint_name,
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
        "endpoint_region": config.endpoint_region,
        "thinking_effort": config.thinking_effort,
        "max_tokens": args.max_tokens,
        "release_version": args.release_version,
        "adaptive_thinking": args.anthropic_adaptive_thinking,
        "checker_timeout": args.checker_timeout,
        "prompt_style": PROMPT_STYLE,
        "request_timeout_s": args.timeout_seconds,
        "sandbox_image": getattr(args, "sandbox_image", "local"),
        "cpu_limit": getattr(args, "cpu_limit", "unspecified"),
        "memory_limit": getattr(args, "memory_limit", "unspecified"),
        "tool_configuration": getattr(args, "tool_configuration", "none"),
        "economy_policy": getattr(args, "economy_policy", "none"),
        "custom_tasks_sha256": getattr(args, "custom_tasks_sha256", None),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint-name",
        action="append",
        required=True,
        help="Endpoint label; repeat with --model to interleave models",
    )
    parser.add_argument(
        "--provider",
        action="append",
        choices=["openai", "anthropic", "cerebras"],
        help="Provider per endpoint (default: openai); repeat or provide once",
    )
    parser.add_argument(
        "--base-url",
        action="append",
        help="Optional provider base URL; repeat or provide once",
    )
    parser.add_argument(
        "--endpoint-region",
        action="append",
        help="Provider region metadata; repeat per endpoint or provide once",
    )
    parser.add_argument(
        "--api-key",
        action="append",
        required=True,
        help="Provider API key; repeat per endpoint or provide once",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Responses API model; repeat with --endpoint-name to interleave models",
    )
    parser.add_argument(
        "--thinking-effort",
        action="append",
        help="Reasoning effort per endpoint; repeat or provide once",
    )
    parser.add_argument(
        "--anthropic-adaptive-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Anthropic adaptive thinking with hidden thinking output",
    )
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--isolate-requests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run each provider request in a killable child process (default: true)",
    )
    parser.add_argument(
        "--max-attempts-per-tuple",
        type=int,
        default=3,
        help="Maximum recorded attempts for one config/problem/run tuple",
    )
    task_source = parser.add_mutually_exclusive_group(required=True)
    task_source.add_argument(
        "--problem-ids",
        help=(
            "Comma-separated IDs/selectors, an inclusive range such as hard:1-80, "
            "or a file containing comma/newline-separated IDs"
        ),
    )
    task_source.add_argument(
        "--custom-tasks",
        type=Path,
        help=(
            "JSON file of custom tasks with id, prompt, and optional "
            "required_patterns fields"
        ),
    )
    parser.add_argument("--release-version", default="release_v2")
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--csv", type=Path, default=Path("bench_results.csv"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip endpoint/model/problem/run combinations already present in --csv; "
            "use only when endpoint labels retain the same configuration"
        ),
    )
    parser.add_argument("--checker-timeout", type=int, default=6)
    parser.add_argument("--sandbox-image", default="local")
    parser.add_argument("--cpu-limit", default="unspecified")
    parser.add_argument("--memory-limit", default="unspecified")
    parser.add_argument("--tool-configuration", default="none")
    parser.add_argument("--economy-policy", default="none")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="Output directory for generated deliverables (default: beside CSV)",
    )
    args = parser.parse_args()
    if args.custom_tasks:
        args.custom_tasks_sha256 = hashlib.sha256(
            args.custom_tasks.read_bytes()
        ).hexdigest()
        args.release_version = f"custom:{args.custom_tasks.name}"
    if (
        args.max_tokens < 1
        or args.runs < 1
        or args.checker_timeout < 1
        or args.timeout_seconds <= 0
        or args.max_attempts_per_tuple < 1
    ):
        parser.error(
            "token, run, checker timeout, request timeout, and max attempts must be positive"
        )
    if len(args.endpoint_name) != len(args.model):
        parser.error("--endpoint-name and --model must be provided the same number of times")
    for name in (
        "provider", "base_url", "endpoint_region", "api_key", "thinking_effort"
    ):
        values = getattr(args, name)
        if values is not None and len(values) not in {1, len(args.model)}:
            parser.error(f"--{name.replace('_', '-')} must be provided once or once per model")
    return args


def expand_arg(values: list[str] | None, count: int, default=None) -> list:
    if values is None:
        return [default] * count
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError("Endpoint arguments must be provided once or once per model")
    return values


def build_endpoint_configs(args: argparse.Namespace) -> list[EndpointConfig]:
    count = len(args.model)
    if len(set(args.endpoint_name)) != count:
        raise ValueError("Endpoint names must be unique")
    providers = expand_arg(args.provider, count, "openai")
    api_keys = expand_arg(args.api_key, count)
    base_urls = expand_arg(args.base_url, count)
    regions = expand_arg(getattr(args, "endpoint_region", None), count)
    efforts = expand_arg(args.thinking_effort, count)
    configs = []
    for endpoint_name, provider, model, api_key, base_url, region, effort in zip(
        args.endpoint_name, providers, args.model, api_keys, base_urls, regions, efforts
    ):
        if not api_key:
            raise ValueError(f"API key is required for endpoint {endpoint_name}")
        if provider == "openai" and base_url is None:
            base_url = "https://api.openai.com/v1"
        elif provider == "cerebras" and base_url is None:
            base_url = "https://api.cerebras.ai/v1"
        configs.append(
            EndpointConfig(
                endpoint_name, provider, model, api_key, base_url, region, effort
            )
        )
    return configs


def read_problem_ids(value: str) -> list[str]:
    path = Path(value)
    text = path.read_text() if path.is_file() else value
    ids = [item.strip() for line in text.splitlines() for item in line.split(",")]
    ids = [problem_id for problem_id in ids if problem_id]
    expanded = []
    for problem_id in ids:
        match = INDEX_RANGE_RE.fullmatch(problem_id)
        if not match:
            expanded.append(problem_id)
            continue
        difficulty, raw_start, raw_end = match.groups()
        start, end = int(raw_start), int(raw_end)
        if end < start:
            raise ValueError(f"Problem range must be ascending: {problem_id}")
        expanded.extend(f"{difficulty}:{index}" for index in range(start, end + 1))
    ids = expanded
    if not ids:
        raise ValueError("No problem IDs were provided")
    if len(ids) != len(set(ids)):
        raise ValueError("Problem IDs must not contain duplicates")
    return ids


def resolve_problem_refs(problem_refs: list[str], release_version: str) -> list[str]:
    selectors = [problem_ref for problem_ref in problem_refs if INDEX_SELECTOR_RE.fullmatch(problem_ref)]
    literal_ids = [problem_ref for problem_ref in problem_refs if problem_ref not in selectors]
    if not selectors:
        return literal_ids

    dataset = load_code_generation_dataset(release_version)
    resolved = []
    for selector in selectors:
        difficulty, raw_index = INDEX_SELECTOR_RE.fullmatch(selector).groups()
        index = int(raw_index)
        matching = [problem for problem in dataset if problem.difficulty.value == difficulty]
        if index > len(matching):
            raise ValueError(
                f"{selector} is out of range for {release_version}; "
                f"{difficulty} has {len(matching)} problems"
            )
        resolved.append(matching[index - 1].question_id)

    result = literal_ids + resolved
    if len(result) != len(set(result)):
        raise ValueError("Problem references resolved to duplicate problem IDs")
    return result


def build_problem_number_map(release_version: str) -> dict[str, int]:
    dataset = load_code_generation_dataset(release_version)
    counters = {"easy": 0, "medium": 0, "hard": 0}
    mapping = {}
    for problem in dataset:
        difficulty = problem.difficulty.value
        counters[difficulty] += 1
        mapping[problem.question_id] = counters[difficulty]
    return mapping


def load_problems(problem_ids: list[str], release_version: str) -> list:
    dataset = load_code_generation_dataset(release_version, problem_ids=problem_ids)
    by_id = {problem.question_id: problem for problem in dataset}
    missing = [problem_id for problem_id in problem_ids if problem_id not in by_id]
    if missing:
        raise ValueError(f"Problem IDs not found in {release_version}: {', '.join(missing)}")
    return [by_id[problem_id] for problem_id in problem_ids]


def load_custom_tasks(path: Path) -> list[CustomTask]:
    """Load auditable, provider-neutral single-turn tasks from JSON."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError("Custom task file must contain a non-empty JSON array")
    tasks = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Custom task {index} must be a JSON object")
        task_id = item.get("id")
        prompt = item.get("prompt")
        patterns = item.get("required_patterns", [])
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"Custom task {index} has no valid id")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Custom task {task_id!r} has no valid prompt")
        if not isinstance(patterns, list) or not all(
            isinstance(pattern, str) and pattern for pattern in patterns
        ):
            raise ValueError(
                f"Custom task {task_id!r} required_patterns must be strings"
            )
        tasks.append(CustomTask(task_id.strip(), prompt.strip(), tuple(patterns)))
    ids = [task.question_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Custom task IDs must be unique")
    return tasks


def check_custom_task(task: CustomTask, output: str) -> bool:
    """Apply deterministic structural checks; interactive QA remains separate."""
    stripped = output.strip()
    is_bare_html = bool(
        re.match(r"(?is)^(?:<!doctype\s+html[^>]*>\s*)?<html\b", stripped)
        and re.search(r"(?is)</html>\s*$", stripped)
    )
    return is_bare_html and all(
        re.search(pattern, output, re.IGNORECASE | re.DOTALL)
        for pattern in task.required_patterns
    )


def responses_input_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    instructions = None
    response_input = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role in {"system", "developer"}:
            if not isinstance(content, str):
                raise ValueError("Responses instructions must be text")
            instructions = content if instructions is None else f"{instructions}\n\n{content}"
            continue
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"Unsupported Responses message: role={role!r}")
        part_type = "input_text" if role == "user" else "output_text"
        response_input.append(
            {"role": role, "content": [{"type": part_type, "text": content}]}
        )
    return instructions, response_input


def anthropic_input_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, str]]]:
    system_parts = []
    anthropic_messages = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if not isinstance(content, str):
            raise ValueError("Anthropic message content must be text")
        if role in {"system", "developer"}:
            system_parts.append(content)
        elif role in {"user", "assistant"}:
            anthropic_messages.append({"role": role, "content": content})
        else:
            raise ValueError(f"Unsupported Anthropic message role: {role!r}")
    return "\n\n".join(system_parts) or None, anthropic_messages


def stream_openai_response(
    client: OpenAI,
    model: str,
    max_tokens: int,
    messages: list[dict],
    thinking_effort: str | None = None,
    include_reasoning_summary: bool = False,
):
    instructions, response_input = responses_input_from_messages(messages)
    request = {
        "model": model,
        "input": response_input,
        "max_output_tokens": max_tokens,
        "store": False,
        "stream": True,
    }
    if instructions:
        request["instructions"] = instructions
    if thinking_effort or include_reasoning_summary:
        request["reasoning"] = {}
        if thinking_effort:
            request["reasoning"]["effort"] = thinking_effort
        if include_reasoning_summary:
            request["reasoning"]["summary"] = "auto"

    request_sent_ts = time.perf_counter()
    stream = client.responses.create(**request)
    first_token_ts = None
    last_token_ts = None
    first_event_ts = None
    response_created_ts = None
    response_terminal_ts = None
    generation_start_ts = None
    generation_start_event_type = None
    generation_start_event_detail = None
    generation_start_confidence = "unavailable"
    hidden_reasoning_observability = "unavailable"
    observable_chunk_count = 0
    output_parts = []
    reasoning_summary_parts = []
    final_response = None

    for event in stream:
        now = time.perf_counter()
        first_event_ts = first_event_ts or now
        if event.type == "response.created" and response_created_ts is None:
            response_created_ts = now
        if event.type == "response.output_item.added":
            item_type = getattr(getattr(event, "item", None), "type", None)
            if item_type == "reasoning" and generation_start_ts is None:
                generation_start_ts = now
                generation_start_event_type = event.type
                generation_start_event_detail = "item.type=reasoning"
                generation_start_confidence = "provider_boundary"
                hidden_reasoning_observability = "phase_boundary_only"
        if event.type == "response.output_text.delta" and event.delta:
            if generation_start_ts is None:
                generation_start_ts = now
                generation_start_event_type = event.type
                generation_start_event_detail = "first non-empty output text delta"
                generation_start_confidence = "delta_fallback"
                hidden_reasoning_observability = "not_observed"
            first_token_ts = first_token_ts or now
            last_token_ts = now
            observable_chunk_count += 1
            output_parts.append(event.delta)
        elif event.type == "response.reasoning_summary_text.delta" and event.delta:
            if generation_start_ts is None:
                generation_start_ts = now
                generation_start_event_type = event.type
                generation_start_event_detail = "first non-empty reasoning summary delta"
                generation_start_confidence = "delta_fallback"
                hidden_reasoning_observability = "summary_only"
            reasoning_summary_parts.append(event.delta)
        elif event.type in {"response.completed", "response.incomplete"}:
            response_terminal_ts = now
            final_response = event.response
        elif event.type == "response.failed":
            error = event.response.error
            raise RuntimeError(f"Responses API generation failed: {error}")
        elif event.type == "error":
            raise RuntimeError(f"Responses API stream failed: {event}")

    response_done_ts = time.perf_counter()
    if final_response is None:
        raise RuntimeError("Responses API stream ended without a terminal response event")
    if final_response.usage is None:
        raise RuntimeError("Responses API terminal event did not include token usage")
    request_status = getattr(final_response, "status", None) or "completed"
    incomplete_details = getattr(final_response, "incomplete_details", None)
    stop_reason = getattr(incomplete_details, "reason", None) or request_status
    if include_reasoning_summary and not reasoning_summary_parts:
        for item in getattr(final_response, "output", []) or []:
            if getattr(item, "type", None) != "reasoning":
                continue
            for summary in getattr(item, "summary", []) or []:
                text = getattr(summary, "text", None)
                if text:
                    reasoning_summary_parts.append(text)
    output_token_details = getattr(final_response.usage, "output_tokens_details", None)
    return {
        "output": "".join(output_parts),
        "reasoning_summary": "".join(reasoning_summary_parts),
        "reasoning_tokens": getattr(output_token_details, "reasoning_tokens", None),
        "request_id": getattr(final_response, "id", None),
        "ttft_s": (
            first_token_ts - request_sent_ts if first_token_ts is not None else None
        ),
        "gen_time_s": (
            last_token_ts - first_token_ts if first_token_ts is not None else 0.0
        ),
        "first_stream_event_s": (
            first_event_ts - request_sent_ts if first_event_ts is not None else None
        ),
        "response_created_s": (
            response_created_ts - request_sent_ts
            if response_created_ts is not None else None
        ),
        "provider_window_inference_time_s": (
            response_terminal_ts - response_created_ts
            if response_created_ts is not None and response_terminal_ts is not None
            else None
        ),
        "first_observable_output_s": (
            first_token_ts - request_sent_ts if first_token_ts is not None else None
        ),
        "last_observable_output_s": (
            last_token_ts - request_sent_ts if last_token_ts is not None else None
        ),
        "observable_chunk_count": observable_chunk_count,
        "generation_start_s": generation_start_ts - request_sent_ts if generation_start_ts is not None else None,
        "generation_start_event_type": generation_start_event_type,
        "generation_start_event_detail": generation_start_event_detail,
        "generation_start_confidence": generation_start_confidence,
        "hidden_reasoning_observability": hidden_reasoning_observability,
        "terminal_event_s": response_done_ts - request_sent_ts,
        "observed_pre_generation_s": generation_start_ts - request_sent_ts if generation_start_ts is not None else None,
        "active_generation_time_s": response_done_ts - generation_start_ts if generation_start_ts is not None else None,
        "generation_wall_s": response_done_ts - generation_start_ts if generation_start_ts is not None else None,
        "inference_time_s": response_done_ts - request_sent_ts,
        "total_wall_s": response_done_ts - request_sent_ts,
        "input_tokens": getattr(final_response.usage, "input_tokens", None),
        "output_tokens": final_response.usage.output_tokens,
        "request_status": request_status,
        "stop_reason": stop_reason,
    }


def stream_cerebras_response(
    client: OpenAI,
    model: str,
    max_tokens: int,
    messages: list[dict],
    thinking_effort: str | None = None,
):
    """Stream Cerebras Chat Completions while timing reasoning and text."""
    request = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Keep reasoning separate from answer text while retaining default-on
        # GLM reasoning. This also exposes the first observable reasoning token.
        "extra_body": {"reasoning_format": "parsed"},
    }
    if thinking_effort:
        request["reasoning_effort"] = thinking_effort

    request_sent_ts = time.perf_counter()
    stream = client.chat.completions.create(**request)
    first_token_ts = None
    last_token_ts = None
    first_event_ts = None
    generation_start_ts = None
    generation_start_event_type = None
    generation_start_event_detail = None
    generation_start_confidence = "unavailable"
    hidden_reasoning_observability = "unavailable"
    observable_chunk_count = 0
    output_parts = []
    reasoning_parts = []
    usage = None
    stop_reason = None
    request_id = None

    for chunk in stream:
        now = time.perf_counter()
        first_event_ts = first_event_ts or now
        request_id = request_id or getattr(chunk, "id", None)
        if getattr(chunk, "usage", None) is not None:
            usage = chunk.usage
        for choice in getattr(chunk, "choices", []) or []:
            delta = getattr(choice, "delta", None)
            reasoning = getattr(delta, "reasoning", None)
            content = getattr(delta, "content", None)
            if reasoning:
                if generation_start_ts is None:
                    generation_start_ts = now
                    generation_start_event_type = "chat.completion.chunk.reasoning"
                    generation_start_event_detail = "first non-empty delta.reasoning"
                    generation_start_confidence = "provider_delta"
                    hidden_reasoning_observability = "full"
                reasoning_parts.append(reasoning)
            if content:
                if generation_start_ts is None:
                    generation_start_ts = now
                    generation_start_event_type = "chat.completion.chunk.content"
                    generation_start_event_detail = "first non-empty delta.content"
                    generation_start_confidence = "delta_fallback"
                    hidden_reasoning_observability = "not_observed"
                first_token_ts = first_token_ts or now
                last_token_ts = now
                observable_chunk_count += 1
                output_parts.append(content)
            if getattr(choice, "finish_reason", None):
                stop_reason = choice.finish_reason

    response_done_ts = time.perf_counter()
    if usage is None:
        raise RuntimeError("Cerebras stream ended without token usage")
    stop_reason = stop_reason or "unknown"
    output_tokens = getattr(usage, "completion_tokens", None)
    if output_tokens is None:
        raise RuntimeError("Cerebras terminal usage omitted completion_tokens")
    completion_details = getattr(usage, "completion_tokens_details", None)
    return {
        "output": "".join(output_parts),
        "reasoning_summary": "".join(reasoning_parts),
        "reasoning_tokens": getattr(completion_details, "reasoning_tokens", None),
        "request_id": request_id,
        "ttft_s": (
            first_token_ts - request_sent_ts if first_token_ts is not None else None
        ),
        "gen_time_s": (
            last_token_ts - first_token_ts if first_token_ts is not None else 0.0
        ),
        "first_stream_event_s": (
            first_event_ts - request_sent_ts if first_event_ts is not None else None
        ),
        "first_observable_output_s": (
            first_token_ts - request_sent_ts if first_token_ts is not None else None
        ),
        "last_observable_output_s": (
            last_token_ts - request_sent_ts if last_token_ts is not None else None
        ),
        "observable_chunk_count": observable_chunk_count,
        "generation_start_s": generation_start_ts - request_sent_ts if generation_start_ts is not None else None,
        "generation_start_event_type": generation_start_event_type,
        "generation_start_event_detail": generation_start_event_detail,
        "generation_start_confidence": generation_start_confidence,
        "hidden_reasoning_observability": hidden_reasoning_observability,
        "terminal_event_s": response_done_ts - request_sent_ts,
        "observed_pre_generation_s": generation_start_ts - request_sent_ts if generation_start_ts is not None else None,
        "active_generation_time_s": response_done_ts - generation_start_ts if generation_start_ts is not None else None,
        "generation_wall_s": response_done_ts - generation_start_ts if generation_start_ts is not None else None,
        "inference_time_s": response_done_ts - request_sent_ts,
        "total_wall_s": response_done_ts - request_sent_ts,
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": output_tokens,
        "request_status": "incomplete" if stop_reason == "length" else "completed",
        "stop_reason": stop_reason,
    }


def stream_anthropic_response(
    api_key: str,
    base_url: str | None,
    model: str,
    max_tokens: int,
    messages: list[dict],
    thinking_effort: str | None,
    adaptive_thinking: bool,
    timeout_seconds: float,
    include_reasoning_summary: bool = False,
):
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("Install anthropic to use --provider anthropic") from exc

    system, anthropic_messages = anthropic_input_from_messages(messages)
    request: dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": max_tokens,
    }
    if system:
        request["system"] = system
    if adaptive_thinking:
        request["thinking"] = {
            "type": "adaptive",
            "display": "summarized" if include_reasoning_summary else "omitted",
        }
    if thinking_effort:
        request["output_config"] = {"effort": thinking_effort}

    client_args: dict[str, Any] = {
        "api_key": api_key,
        "timeout": timeout_seconds,
        "max_retries": 0,
    }
    if base_url:
        client_args["base_url"] = base_url.rstrip("/")
    client = anthropic.Anthropic(**client_args)

    request_sent_ts = time.perf_counter()
    first_token_ts = None
    last_token_ts = None
    first_event_ts = None
    generation_start_ts = None
    generation_start_event_type = None
    generation_start_event_detail = None
    generation_start_confidence = "unavailable"
    hidden_reasoning_observability = "unavailable"
    observable_chunk_count = 0
    output_parts = []
    reasoning_summary_parts = []
    final_message = None
    try:
        with client.messages.stream(**request) as stream:
            for event in stream:
                now = time.perf_counter()
                first_event_ts = first_event_ts or now
                if event.type == "content_block_start":
                    block_type = getattr(
                        getattr(event, "content_block", None), "type", None
                    )
                    if block_type in {"thinking", "redacted_thinking", "text", "tool_use", "refusal"} and generation_start_ts is None:
                        generation_start_ts = now
                        generation_start_event_type = event.type
                        generation_start_event_detail = f"content_block.type={block_type}"
                        generation_start_confidence = "provider_boundary"
                        hidden_reasoning_observability = "phase_boundary_only" if block_type in {"thinking", "redacted_thinking"} else "not_observed"
                    continue
                if event.type != "content_block_delta":
                    continue
                delta = getattr(event, "delta", None)
                if getattr(delta, "type", None) == "thinking_delta":
                    if generation_start_ts is None:
                        generation_start_ts = now
                        generation_start_event_type = "content_block_delta.thinking_delta"
                        generation_start_event_detail = "first non-empty thinking delta"
                        generation_start_confidence = "delta_fallback"
                        hidden_reasoning_observability = "full"
                    thinking = getattr(delta, "thinking", None)
                    if thinking:
                        reasoning_summary_parts.append(thinking)
                    continue
                if getattr(delta, "type", None) != "text_delta" or not delta.text:
                    continue
                if generation_start_ts is None:
                    generation_start_ts = now
                    generation_start_event_type = "content_block_delta.text_delta"
                    generation_start_event_detail = "first non-empty text delta"
                    generation_start_confidence = "delta_fallback"
                    hidden_reasoning_observability = "not_observed"
                first_token_ts = first_token_ts or now
                last_token_ts = now
                observable_chunk_count += 1
                output_parts.append(delta.text)
            final_message = stream.get_final_message()
    except Exception as exc:
        raise RuntimeError(f"Anthropic endpoint failed: {exc}") from exc

    response_done_ts = time.perf_counter()
    usage = getattr(final_message, "usage", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if output_tokens is None:
        raise RuntimeError("Anthropic terminal message did not include token usage")
    stop_reason = getattr(final_message, "stop_reason", None) or "unknown"
    output_token_details = getattr(usage, "output_tokens_details", None)
    return {
        "output": "".join(output_parts),
        "reasoning_summary": "".join(reasoning_summary_parts),
        "reasoning_tokens": getattr(output_token_details, "thinking_tokens", None),
        "request_id": getattr(final_message, "id", None),
        "ttft_s": (
            first_token_ts - request_sent_ts if first_token_ts is not None else None
        ),
        "gen_time_s": (
            last_token_ts - first_token_ts if first_token_ts is not None else 0.0
        ),
        "first_stream_event_s": (
            first_event_ts - request_sent_ts if first_event_ts is not None else None
        ),
        "first_observable_output_s": (
            first_token_ts - request_sent_ts if first_token_ts is not None else None
        ),
        "last_observable_output_s": (
            last_token_ts - request_sent_ts if last_token_ts is not None else None
        ),
        "observable_chunk_count": observable_chunk_count,
        "generation_start_s": generation_start_ts - request_sent_ts if generation_start_ts is not None else None,
        "generation_start_event_type": generation_start_event_type,
        "generation_start_event_detail": generation_start_event_detail,
        "generation_start_confidence": generation_start_confidence,
        "hidden_reasoning_observability": hidden_reasoning_observability,
        "terminal_event_s": response_done_ts - request_sent_ts,
        "observed_pre_generation_s": generation_start_ts - request_sent_ts if generation_start_ts is not None else None,
        "active_generation_time_s": response_done_ts - generation_start_ts if generation_start_ts is not None else None,
        "generation_wall_s": response_done_ts - generation_start_ts if generation_start_ts is not None else None,
        "inference_time_s": response_done_ts - request_sent_ts,
        "total_wall_s": response_done_ts - request_sent_ts,
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": output_tokens,
        "request_status": "incomplete" if stop_reason == "max_tokens" else "completed",
        "stop_reason": stop_reason,
    }


def request_worker_main() -> int:
    """Execute one provider request in an isolated, externally killable process."""
    api_attempt_started = None
    try:
        payload = json.load(sys.stdin)
        api_key = os.environ.get("BENCH_WORKER_API_KEY")
        if not api_key:
            raise ValueError("BENCH_WORKER_API_KEY is missing")
        api_attempt_started = time.perf_counter()
        if payload["provider"] in {"openai", "cerebras"}:
            client = OpenAI(
                api_key=api_key,
                base_url=(payload["base_url"] or "https://api.openai.com/v1").rstrip("/"),
                timeout=payload["timeout_seconds"],
                max_retries=0,
            )
            stream_function = (
                stream_openai_response
                if payload["provider"] == "openai"
                else stream_cerebras_response
            )
            result = stream_function(
                client, payload["model"], payload["max_tokens"],
                payload["messages"], payload["thinking_effort"],
            )
        elif payload["provider"] == "anthropic":
            result = stream_anthropic_response(
                api_key,
                payload["base_url"],
                payload["model"],
                payload["max_tokens"],
                payload["messages"],
                payload["thinking_effort"],
                payload["adaptive_thinking"],
                payload["timeout_seconds"],
            )
        else:
            raise ValueError(f'Unsupported provider: {payload["provider"]}')
        json.dump({"ok": True, "result": result}, sys.stdout)
    except Exception as exc:
        json.dump(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc).replace("\n", " ")[:1000],
                "failed_api_time_s": (
                    time.perf_counter() - api_attempt_started
                    if api_attempt_started is not None else None
                ),
            },
            sys.stdout,
        )
    return 0


def run_request_isolated(
    config: EndpointConfig,
    args: argparse.Namespace,
    messages: list[dict],
    label: str,
) -> dict:
    """Run one API request in a subprocess with a true wall-clock deadline."""
    payload = {
        "provider": config.provider,
        "base_url": config.base_url,
        "model": config.model,
        "max_tokens": args.max_tokens,
        "messages": messages,
        "thinking_effort": config.thinking_effort,
        "adaptive_thinking": args.anthropic_adaptive_thinking,
        "timeout_seconds": args.timeout_seconds,
    }
    env = os.environ.copy()
    env["BENCH_WORKER_API_KEY"] = config.api_key
    started = time.monotonic()
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--request-worker"],
            stdin=subprocess.PIPE,
            stdout=stdout_file,
            stderr=stderr_file,
            env=env,
        )
        try:
            request_bytes = json.dumps(payload).encode()
            process.stdin.write(request_bytes)
            process.stdin.close()
            while True:
                elapsed = time.monotonic() - started
                remaining = args.timeout_seconds - elapsed
                if remaining <= 0:
                    process.kill()
                    process.wait()
                    raise RequestDeadlineExceeded(
                        f"Request exceeded the {args.timeout_seconds:g}s total "
                        f"wall-clock deadline",
                        elapsed_s=elapsed,
                    )
                try:
                    wait_seconds = min(60.0, remaining)
                    process.wait(timeout=wait_seconds)
                    break
                except subprocess.TimeoutExpired:
                    if wait_seconds >= 59.0:
                        print(
                            f"HEARTBEAT {label} elapsed={time.monotonic() - started:.0f}s "
                            f"deadline={args.timeout_seconds:g}s",
                            flush=True,
                        )
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout_text = stdout_file.read().decode(errors="replace")
        stderr_text = stderr_file.read().decode(errors="replace")
    if process.returncode != 0:
        raise WorkerRequestError(
            "WorkerProcessError",
            f"request worker exited {process.returncode}: {stderr_text[-1000:]}",
            elapsed_s=time.monotonic() - started,
        )
    try:
        response = json.loads(stdout_text)
    except json.JSONDecodeError as exc:
        raise WorkerRequestError(
            "WorkerProtocolError",
            f"invalid worker JSON: {stdout_text[-500:]} {stderr_text[-500:]}",
            elapsed_s=time.monotonic() - started,
        ) from exc
    if not response.get("ok"):
        raise WorkerRequestError(
            response.get("error_type", "WorkerRequestError"),
            response.get("error_message", "isolated request failed"),
            elapsed_s=response.get("failed_api_time_s"),
        )
    return response["result"]


def check_solution(problem, output: str, timeout: int) -> bool:
    code = extract_code(output, LMStyle.OpenAIChat)
    results, _ = evaluate_generations_by_problem(
        ([code], problem.get_evaluation_sample(), False, timeout)
    )
    return bool(results[0]) and all(result is True for result in results[0])


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    needs_separator = False
    if not needs_header:
        with csv_path.open(newline="") as existing_text:
            existing_fields = next(csv.reader(existing_text), [])
        if existing_fields != CSV_FIELDS:
            raise ValueError(
                f"CSV schema mismatch in {csv_path}; use a new output file for "
                f"harness schema {HARNESS_VERSION}"
            )
        with csv_path.open("rb") as existing:
            existing.seek(-1, os.SEEK_END)
            needs_separator = existing.read(1) not in {b"\n", b"\r"}
    with csv_path.open("a", newline="") as handle:
        if needs_separator:
            handle.write("\n")
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def write_artifact(
    args: argparse.Namespace,
    config: EndpointConfig,
    fingerprint: str,
    problem_id: str,
    run_idx: int,
    attempt_idx: int,
    output: str,
) -> str:
    """Persist the model deliverable and return its path for CSV auditability."""
    artifacts_dir = getattr(args, "artifacts_dir", None)
    if artifacts_dir is None:
        artifacts_dir = args.csv.with_name(f"{args.csv.stem}_artifacts")
    safe_problem_id = re.sub(r"[^A-Za-z0-9_.-]", "_", problem_id)
    safe_endpoint = re.sub(r"[^A-Za-z0-9_.-]", "_", config.endpoint_name)
    path = (
        Path(artifacts_dir) / safe_endpoint / fingerprint / safe_problem_id
        / f"run_{run_idx}_attempt_{attempt_idx}"
        f"{'.html' if getattr(args, 'custom_tasks', None) else '.txt'}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output)
    return str(path)


def write_aggregate_summary(
    csv_path: Path,
    expected_keys: set[tuple[str, str, str, int, str]],
    max_attempts_per_tuple: int = 3,
) -> Path:
    """Write ratio-of-sums endpoint metrics and per-attempt averages."""
    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    attempts_by_key = Counter()
    completed_keys = set()
    if csv_path.exists():
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    key = (
                        row["endpoint_name"], row["model"], row["problem_id"],
                        int(row["run_idx"]), row["config_fingerprint"],
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if key not in expected_keys:
                    continue
                attempts_by_key[key] += 1
                group_key = (row["endpoint_name"], row["model"])
                item = aggregates.setdefault(
                    group_key,
                    {
                        "endpoint_name": row["endpoint_name"],
                        "provider": row.get("provider"),
                        "model": row["model"],
                        "thinking_effort": row.get("thinking_effort"),
                        "attempted_runs": 0,
                        "completed_runs": 0,
                        "outcomes": {},
                        "billed_output_tokens": 0,
                        "reasoning_tokens": 0,
                        "reasoning_tokens_complete": True,
                        "inference_time_s": 0.0,
                        "completed_inference_time_s": 0.0,
                        "completed_active_generation_time_s": 0.0,
                        "active_generation_billed_tokens": 0,
                        "active_generation_eligible_runs": 0,
                        "completed_provider_window_inference_time_s": 0.0,
                        "provider_window_billed_tokens": 0,
                        "provider_window_eligible_runs": 0,
                        "incomplete_billed_output_tokens": 0,
                        "tool_time_s": 0.0,
                        "retry_api_time_s": 0.0,
                        "backoff_time_s": 0.0,
                        "harness_overhead_s": 0.0,
                        "total_wall_s": 0.0,
                    },
                )
                item["attempted_runs"] += 1
                status = row.get("request_status") or "unknown"
                inference_outcome = row.get("inference_outcome") or status
                item["outcomes"][status] = item["outcomes"].get(status, 0) + 1
                for field in (
                    "inference_time_s", "tool_time_s", "retry_api_time_s", "backoff_time_s",
                    "harness_overhead_s", "total_wall_s",
                ):
                    if row.get(field):
                        item[field] += float(row[field])
                if status == "completed":
                    completed_keys.add(key)
                    item["completed_runs"] += 1
                if inference_outcome != "completed":
                    if (
                        inference_outcome == "incomplete"
                        and row.get("billed_output_tokens")
                    ):
                        item["incomplete_billed_output_tokens"] += int(
                            row["billed_output_tokens"]
                        )
                    continue
                item["billed_output_tokens"] += int(row["billed_output_tokens"])
                item["completed_inference_time_s"] += float(
                    row["inference_time_s"]
                )
                if row.get("active_generation_time_s"):
                    item["completed_active_generation_time_s"] += float(row["active_generation_time_s"])
                    item["active_generation_billed_tokens"] += int(row["billed_output_tokens"])
                    item["active_generation_eligible_runs"] += 1
                if row.get("provider_window_inference_time_s"):
                    item["completed_provider_window_inference_time_s"] += float(
                        row["provider_window_inference_time_s"]
                    )
                    item["provider_window_billed_tokens"] += int(
                        row["billed_output_tokens"]
                    )
                    item["provider_window_eligible_runs"] += 1
                if row.get("reasoning_tokens"):
                    item["reasoning_tokens"] += int(row["reasoning_tokens"])
                else:
                    item["reasoning_tokens_complete"] = False

    summaries = []
    planned_by_endpoint = Counter((key[0], key[1]) for key in expected_keys)
    skipped_by_endpoint = Counter()
    missing_by_endpoint = Counter()
    for key in expected_keys:
        if key in completed_keys:
            continue
        group_key = (key[0], key[1])
        if attempts_by_key[key] >= max_attempts_per_tuple:
            skipped_by_endpoint[group_key] += 1
        else:
            missing_by_endpoint[group_key] += 1
    for group_key, planned in planned_by_endpoint.items():
        if group_key not in aggregates:
            aggregates[group_key] = {
                "endpoint_name": group_key[0], "provider": None,
                "model": group_key[1], "thinking_effort": None,
                "attempted_runs": 0, "completed_runs": 0, "outcomes": {},
                "billed_output_tokens": 0, "reasoning_tokens": 0,
                "reasoning_tokens_complete": True, "inference_time_s": 0.0,
                "completed_inference_time_s": 0.0,
                "completed_active_generation_time_s": 0.0,
                "active_generation_billed_tokens": 0,
                "active_generation_eligible_runs": 0,
                "completed_provider_window_inference_time_s": 0.0,
                "provider_window_billed_tokens": 0,
                "provider_window_eligible_runs": 0,
                "incomplete_billed_output_tokens": 0,
                "tool_time_s": 0.0, "retry_api_time_s": 0.0,
                "backoff_time_s": 0.0, "harness_overhead_s": 0.0,
                "total_wall_s": 0.0,
            }
    for item in aggregates.values():
        attempted = item["attempted_runs"]
        item["planned_runs"] = planned_by_endpoint[
            (item["endpoint_name"], item["model"])
        ]
        group_key = (item["endpoint_name"], item["model"])
        item["skipped_attempts_exhausted"] = skipped_by_endpoint[group_key]
        item["missing_runs"] = missing_by_endpoint[group_key]
        completed_inference = item["completed_inference_time_s"]
        reasoning_complete = item.pop("reasoning_tokens_complete")
        if reasoning_complete:
            item["visible_output_tokens"] = (
                item["billed_output_tokens"] - item["reasoning_tokens"]
            )
        else:
            item["reasoning_tokens"] = None
            item["visible_output_tokens"] = None
        item["end_to_end_billed_tps"] = (
            item["billed_output_tokens"] / completed_inference
            if completed_inference > 0 else None
        )
        active_seconds = item["completed_active_generation_time_s"]
        item["active_generation_billed_tps"] = (
            item["active_generation_billed_tokens"] / active_seconds
            if active_seconds > 0 else None
        )
        provider_window_seconds = item[
            "completed_provider_window_inference_time_s"
        ]
        item["provider_window_billed_tps"] = (
            item["provider_window_billed_tokens"] / provider_window_seconds
            if provider_window_seconds > 0 else None
        )
        item["provider_window_coverage"] = (
            "complete"
            if item["completed_runs"] > 0
            and item["provider_window_eligible_runs"] == item["completed_runs"]
            else "partial"
            if item["provider_window_eligible_runs"] > 0
            else "unavailable"
        )
        item["averages_per_attempted_run"] = {
            field: item[field] / attempted if attempted else None
            for field in (
                "billed_output_tokens", "inference_time_s", "tool_time_s",
                "retry_api_time_s", "backoff_time_s", "harness_overhead_s",
                "total_wall_s",
            )
        }
        item["averages_per_attempted_run"]["reasoning_tokens"] = (
            item["reasoning_tokens"] / attempted
            if attempted and item["reasoning_tokens"] is not None else None
        )
        item["averages_per_attempted_run"]["visible_output_tokens"] = (
            item["visible_output_tokens"] / attempted
            if attempted and item["visible_output_tokens"] is not None else None
        )
        summaries.append(item)

    summary_path = csv_path.with_name(f"{csv_path.stem}.summary.json")
    temporary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps({"endpoints": summaries}, indent=2) + "\n")
    os.replace(temporary_path, summary_path)
    for item in summaries:
        tps = item["active_generation_billed_tps"]
        provider_window_tps = item["provider_window_billed_tps"]
        provider_window_text = (
            f"{provider_window_tps:.3f}"
            if provider_window_tps is not None else "unavailable"
        )
        active_generation_text = f"{tps:.3f}" if tps is not None else "unavailable"
        end_to_end_tps = item["end_to_end_billed_tps"]
        end_to_end_text = (
            f"{end_to_end_tps:.3f}"
            if end_to_end_tps is not None else "unavailable"
        )
        print(
            f'SUMMARY endpoint={item["endpoint_name"]} '
            f'attempted={item["attempted_runs"]} completed={item["completed_runs"]} '
            f'billed_tokens={item["billed_output_tokens"]} '
            f'inference_s={item["completed_inference_time_s"]:.6f} '
            f'provider_window_tok/s={provider_window_text} '
            f'active_billed_tok/s={active_generation_text} '
            f'e2e_billed_tok/s={end_to_end_text}',
            flush=True,
        )
    return summary_path


def load_completed_runs(csv_path: Path) -> set[tuple[str, str, str, int, str]]:
    """Return resumable run keys already durably recorded in a result CSV."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()

    required_fields = {
        "endpoint_name", "model", "problem_id", "run_idx",
        "config_fingerprint", "request_status",
    }
    completed = set()
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required_fields - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Cannot resume from {csv_path}: missing CSV fields "
                f"{', '.join(sorted(missing))}"
            )
        for line_number, row in enumerate(reader, start=2):
            if row["request_status"] != "completed":
                continue
            try:
                run_idx = int(row["run_idx"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot resume from {csv_path}: invalid run_idx on line {line_number}"
                ) from exc
            completed.add(
                (
                    row["endpoint_name"], row["model"], row["problem_id"],
                    run_idx, row["config_fingerprint"],
                )
            )
    return completed


def load_attempt_counts(
    csv_path: Path,
) -> Counter[tuple[str, str, str, int, str]]:
    """Count durable attempts so repeated resumes cannot retry forever."""
    counts = Counter()
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return counts
    required_fields = {
        "endpoint_name", "model", "problem_id", "run_idx", "config_fingerprint",
    }
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required_fields - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Cannot count attempts in {csv_path}: missing CSV fields "
                f"{', '.join(sorted(missing))}"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                key = (
                    row["endpoint_name"], row["model"], row["problem_id"],
                    int(row["run_idx"]), row["config_fingerprint"],
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot count attempts in {csv_path}: invalid run_idx on "
                    f"line {line_number}"
                ) from exc
            counts[key] += 1
    return counts


def format_seconds(value: float | None) -> str:
    """Format optional timing values for CSV output."""
    return "" if value is None else f"{value:.6f}"


def balanced_endpoint_order(
    configs: list[EndpointConfig], problem_index: int, run_idx: int
) -> list[EndpointConfig]:
    """Rotate which endpoint runs first to balance systematic time-order effects."""
    if not configs:
        return []
    offset = (problem_index + run_idx - 1) % len(configs)
    return configs[offset:] + configs[:offset]


def build_result_row(
    args: argparse.Namespace,
    config: EndpointConfig,
    fingerprint: str,
    problem,
    problem_number: int,
    run_idx: int,
) -> dict:
    row = dict.fromkeys(CSV_FIELDS, "")
    row.update(
        timestamp=datetime.now(timezone.utc).isoformat(),
        harness_version=HARNESS_VERSION,
        config_fingerprint=fingerprint,
        endpoint_name=config.endpoint_name,
        provider=config.provider,
        model=config.model,
        base_url=config.base_url or "",
        endpoint_region=config.endpoint_region or "unspecified",
        thinking_effort=config.thinking_effort or "",
        max_tokens=args.max_tokens,
        release_version=args.release_version,
        adaptive_thinking=args.anthropic_adaptive_thinking,
        checker_timeout=args.checker_timeout,
        prompt_style=PROMPT_STYLE,
        request_timeout_s=args.timeout_seconds,
        sandbox_image=getattr(args, "sandbox_image", "local"),
        cpu_limit=getattr(args, "cpu_limit", "unspecified"),
        memory_limit=getattr(args, "memory_limit", "unspecified"),
        tool_configuration=getattr(args, "tool_configuration", "none"),
        economy_policy=getattr(args, "economy_policy", "none"),
        problem_id=problem.question_id,
        difficulty=problem.difficulty.value,
        problem_number=problem_number,
        run_idx=run_idx,
    )
    return row


def audit_results(
    csv_path: Path,
    expected_keys: set[tuple[str, str, str, int, str]],
) -> bool:
    """Print a terminal completeness audit for the requested configuration."""
    completed_counts = {key: 0 for key in expected_keys}
    incomplete = 0
    errors = 0
    timeouts = 0
    if csv_path.exists():
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    key = (
                        row["endpoint_name"], row["model"], row["problem_id"],
                        int(row["run_idx"]), row["config_fingerprint"],
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if key not in expected_keys:
                    continue
                status = row.get("request_status")
                if status == "completed":
                    completed_counts[key] += 1
                elif status == "incomplete":
                    incomplete += 1
                elif status == "error":
                    errors += 1
                elif status == "inference_timeout":
                    timeouts += 1
    completed = sum(count > 0 for count in completed_counts.values())
    duplicates = sum(max(0, count - 1) for count in completed_counts.values())
    missing = len(expected_keys) - completed
    print(
        f"AUDIT expected={len(expected_keys)} completed={completed} missing={missing} "
        f"incomplete={incomplete} inference_timeouts={timeouts} errors={errors} "
        f"duplicate_completed={duplicates}",
        flush=True,
    )
    return missing == 0 and duplicates == 0


def main() -> int:
    args = parse_args()
    endpoint_configs = build_endpoint_configs(args)
    if len(endpoint_configs) > 1:
        efforts = {config.thinking_effort for config in endpoint_configs}
        if efforts != {"xhigh"}:
            raise ValueError(
                "Comparable multi-endpoint runs require --thinking-effort xhigh "
                "for every endpoint"
            )
        regions = {config.endpoint_region for config in endpoint_configs}
        if len(regions) > 1:
            raise ValueError(
                "Comparable multi-endpoint runs require the same "
                "--endpoint-region metadata for every endpoint"
            )
    custom_tasks_path = getattr(args, "custom_tasks", None)
    if custom_tasks_path:
        problems = load_custom_tasks(custom_tasks_path)
        problem_numbers = {
            problem.question_id: index
            for index, problem in enumerate(problems, start=1)
        }
    else:
        problem_ids = resolve_problem_refs(
            read_problem_ids(args.problem_ids), args.release_version
        )
        problem_numbers = build_problem_number_map(args.release_version)
        problems = load_problems(problem_ids, args.release_version)
    completed_runs = load_completed_runs(args.csv) if args.resume else set()
    attempt_counts = load_attempt_counts(args.csv) if args.resume else Counter()
    fingerprints = {
        config.endpoint_name: config_fingerprint(config, args)
        for config in endpoint_configs
    }
    expected_keys = {
        (
            config.endpoint_name,
            config.model,
            problem.question_id,
            run_idx,
            fingerprints[config.endpoint_name],
        )
        for problem in problems
        for run_idx in range(1, args.runs + 1)
        for config in endpoint_configs
    }
    isolate_requests = getattr(args, "isolate_requests", False)
    max_attempts = getattr(args, "max_attempts_per_tuple", 3)
    openai_clients = {
        config.endpoint_name: OpenAI(
            api_key=config.api_key,
            base_url=config.base_url.rstrip("/"),
            timeout=args.timeout_seconds,
            max_retries=0,
        )
        for config in endpoint_configs
        if config.provider in {"openai", "cerebras"} and not isolate_requests
    }

    for problem_index, problem in enumerate(problems):
        messages = (
            [{"role": "user", "content": problem.prompt}]
            if isinstance(problem, CustomTask)
            else format_prompt_generation(problem, LMStyle.OpenAIChat)
        )
        for run_idx in range(1, args.runs + 1):
            for config in balanced_endpoint_order(endpoint_configs, problem_index, run_idx):
                fingerprint = fingerprints[config.endpoint_name]
                run_key = (
                    config.endpoint_name,
                    config.model,
                    problem.question_id,
                    run_idx,
                    fingerprint,
                )
                if run_key in completed_runs:
                    print(
                        f"{problem.question_id} provider={config.provider} "
                        f"model={config.model} run={run_idx} skipped=already-recorded",
                        flush=True,
                    )
                    continue
                if attempt_counts[run_key] >= max_attempts:
                    print(
                        f"{problem.question_id} provider={config.provider} "
                        f"model={config.model} run={run_idx} "
                        f"skipped=attempts-exhausted "
                        f"attempts={attempt_counts[run_key]}",
                        flush=True,
                    )
                    continue
                row = build_result_row(
                    args, config, fingerprint, problem,
                    problem_numbers[problem.question_id], run_idx,
                )
                attempt_idx = attempt_counts[run_key] + 1
                row["attempt_idx"] = attempt_idx
                task_started = time.perf_counter()
                phase = "inference"
                inference_time = 0.0
                try:
                    if isolate_requests:
                        timing = run_request_isolated(
                            config,
                            args,
                            messages,
                            label=(
                                f"{problem.question_id} provider={config.provider} "
                                f"model={config.model} run={run_idx}"
                            ),
                        )
                    else:
                        with request_deadline(args.timeout_seconds):
                            if config.provider in {"openai", "cerebras"}:
                                stream_function = (
                                    stream_openai_response
                                    if config.provider == "openai"
                                    else stream_cerebras_response
                                )
                                timing = stream_function(
                                    openai_clients[config.endpoint_name],
                                    config.model,
                                    args.max_tokens,
                                    messages,
                                    config.thinking_effort,
                                )
                            else:
                                timing = stream_anthropic_response(
                                    config.api_key,
                                    config.base_url,
                                    config.model,
                                    args.max_tokens,
                                    messages,
                                    config.thinking_effort,
                                    args.anthropic_adaptive_thinking,
                                    args.timeout_seconds,
                                )
                    output = timing.pop("output")
                    inference_time = timing.get(
                        "inference_time_s", timing["total_wall_s"]
                    )
                    reasoning_tokens = timing.get("reasoning_tokens")
                    visible_tokens = (
                        timing["output_tokens"] - reasoning_tokens
                        if reasoning_tokens is not None else ""
                    )
                    completed = timing["request_status"] == "completed"
                    end_to_end_tps = (
                        timing["output_tokens"] / inference_time
                        if completed and inference_time > 0 else None
                    )
                    active_generation_time = timing.get("active_generation_time_s")
                    active_generation_tps = (
                        timing["output_tokens"] / active_generation_time
                        if completed and active_generation_time and active_generation_time > 0
                        else None
                    )
                    provider_window_time = timing.get(
                        "provider_window_inference_time_s"
                    )
                    provider_window_tps = (
                        timing["output_tokens"] / provider_window_time
                        if completed
                        and provider_window_time
                        and provider_window_time > 0
                        else None
                    )
                    row.update(
                        request_status=timing["request_status"],
                        inference_outcome=timing["request_status"],
                        stop_reason=timing["stop_reason"],
                        request_id=timing.get("request_id") or "",
                        input_tokens=timing["input_tokens"],
                        billed_output_tokens=timing["output_tokens"],
                        reasoning_tokens=(
                            reasoning_tokens if reasoning_tokens is not None else ""
                        ),
                        visible_output_tokens=visible_tokens,
                        visible_output_tokens_approx=False if visible_tokens != "" else "",
                        ttft_s=format_seconds(timing["ttft_s"]),
                        first_stream_event_s=format_seconds(
                            timing.get("first_stream_event_s")
                        ),
                        response_created_s=format_seconds(
                            timing.get("response_created_s")
                        ),
                        provider_window_inference_time_s=format_seconds(
                            provider_window_time
                        ),
                        first_observable_output_s=format_seconds(
                            timing.get("first_observable_output_s")
                        ),
                        last_observable_output_s=format_seconds(
                            timing.get("last_observable_output_s")
                        ),
                        observable_chunk_count=timing.get(
                            "observable_chunk_count", ""
                        ),
                        gen_time_s=format_seconds(timing["gen_time_s"]),
                        generation_start_s=format_seconds(timing.get("generation_start_s")),
                        generation_start_event_type=timing.get("generation_start_event_type") or "",
                        generation_start_event_detail=timing.get("generation_start_event_detail") or "",
                        generation_start_confidence=timing.get("generation_start_confidence") or "unavailable",
                        hidden_reasoning_observability=timing.get("hidden_reasoning_observability") or "unavailable",
                        terminal_event_s=format_seconds(timing.get("terminal_event_s")),
                        observed_pre_generation_s=format_seconds(timing.get("observed_pre_generation_s")),
                        active_generation_time_s=format_seconds(active_generation_time),
                        generation_wall_s=format_seconds(active_generation_time),
                        inference_time_s=f"{inference_time:.6f}",
                        provider_window_billed_tps=(
                            f"{provider_window_tps:.3f}"
                            if provider_window_tps is not None else ""
                        ),
                        active_generation_billed_tps=(
                            f"{active_generation_tps:.3f}" if active_generation_tps is not None else ""
                        ),
                        end_to_end_billed_tps=(
                            f"{end_to_end_tps:.3f}" if end_to_end_tps is not None else ""
                        ),
                    )
                    phase = "evaluation"
                    artifact_path = write_artifact(
                        args, config, fingerprint, problem.question_id,
                        run_idx, attempt_idx, output,
                    )
                    row["artifact_path"] = artifact_path
                    passed = (
                        check_custom_task(problem, output)
                        if isinstance(problem, CustomTask)
                        else check_solution(problem, output, args.checker_timeout)
                    )
                    task_finished = time.perf_counter()
                    task_wall = max(task_finished - task_started, inference_time)
                    overhead = max(0.0, task_wall - inference_time)
                    row.update(
                        tool_time_s="0.000000",
                        retry_api_time_s="0.000000",
                        backoff_time_s="0.000000",
                        harness_overhead_s=f"{overhead:.6f}",
                        total_wall_s=f"{task_wall:.6f}",
                        passed=passed,
                    )
                except Exception as exc:
                    task_finished = time.perf_counter()
                    task_wall = task_finished - task_started
                    failed_api_time = 0.0
                    if phase == "inference":
                        failed_api_time = getattr(exc, "elapsed_s", None) or task_wall
                    task_wall = max(task_wall, inference_time + failed_api_time)
                    overhead = max(
                        0.0, task_wall - inference_time - failed_api_time
                    )
                    row.update(
                        request_status=(
                            "inference_timeout"
                            if isinstance(exc, RequestDeadlineExceeded) else "error"
                        ),
                        inference_outcome=(
                            "inference_timeout"
                            if phase == "inference"
                            and isinstance(exc, RequestDeadlineExceeded)
                            else "error" if phase == "inference"
                            else row.get("inference_outcome", "")
                        ),
                        stop_reason=(
                            "timeout"
                            if isinstance(exc, RequestDeadlineExceeded)
                            else row.get("stop_reason", "")
                        ),
                        inference_time_s=(
                            f"{inference_time:.6f}" if inference_time else ""
                        ),
                        tool_time_s="0.000000",
                        retry_api_time_s=f"{failed_api_time:.6f}",
                        backoff_time_s="0.000000",
                        harness_overhead_s=f"{overhead:.6f}",
                        total_wall_s=f"{task_wall:.6f}",
                        error_type=(
                            exc.remote_type
                            if isinstance(exc, WorkerRequestError)
                            else type(exc).__name__
                        ),
                        error_message=str(exc).replace("\n", " ")[:1000],
                    )
                append_row(args.csv, row)
                attempt_counts[run_key] += 1
                if row["request_status"] in {"error", "inference_timeout"}:
                    print(
                        f'{problem.question_id} provider={config.provider} '
                        f'model={config.model} run={run_idx} '
                        f'status={row["request_status"]} '
                        f'error={row["error_type"]}: {row["error_message"]}',
                        flush=True,
                    )
                else:
                    print(
                        f'{problem.question_id} provider={config.provider} '
                        f'model={config.model} run={run_idx} '
                        f'status={row["request_status"]} stop={row["stop_reason"]} '
                        f'passed={row["passed"]} ttft={row["ttft_s"]}s '
                        f'inference={row["inference_time_s"]}s '
                        f'tokens={row["billed_output_tokens"]} '
                        f'provider_window_tok/s={row["provider_window_billed_tps"]} '
                        f'active_billed_tok/s={row["active_generation_billed_tps"]} '
                        f'e2e_billed_tok/s={row["end_to_end_billed_tps"]}',
                        flush=True,
                    )
    audit_ok = audit_results(args.csv, expected_keys)
    write_aggregate_summary(args.csv, expected_keys, max_attempts)
    return 0 if audit_ok else 2


if __name__ == "__main__" and "--request-worker" in sys.argv:
    raise SystemExit(request_worker_main())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (APIError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
