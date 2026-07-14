#!/usr/bin/env python3
"""Client-observed Codex inference and tool accounting for SWE-bench runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


LIFECYCLE_TRACE_TARGET = "codex_api::responses_websocket_lifecycle"
TOOL_ITEM_TYPES = {
    "command_execution",
    "file_change",
    "mcp_tool_call",
    "web_search",
}


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def integer(value: Any) -> int | None:
    number = numeric(value)
    return int(number) if number is not None else None


def exact_integer(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def tracing_field(line: str, name: str) -> Any:
    """Extract one field from tracing-subscriber's compact text format."""
    match = re.search(rf"(?:^|\s){re.escape(name)}=", line)
    if match is None:
        return None
    raw = line[match.end():].lstrip()
    if not raw:
        return None
    if raw[0] in {'"', "{", "["}:
        try:
            value, _ = json.JSONDecoder().raw_decode(raw)
            return value
        except json.JSONDecodeError:
            return None
    token = raw.split(None, 1)[0].rstrip(",")
    if token == "true":
        return True
    if token == "false":
        return False
    if token == "null":
        return None
    return token


def parse_lifecycle_trace_line(line: str) -> dict[str, Any] | None:
    """Parse one instrumented Responses WebSocket lifecycle trace."""
    if LIFECYCLE_TRACE_TARGET not in line:
        return None
    parsed = {
        "model": tracing_field(line, "model"),
        "session_id": tracing_field(line, "session_id"),
        "thread_id": tracing_field(line, "thread_id"),
        "turn_id": tracing_field(line, "turn_id"),
        "previous_response_id": tracing_field(line, "previous_response_id"),
        "response_id": tracing_field(line, "response_id"),
        "warmup": tracing_field(line, "warmup") is True,
        "connection_reused": tracing_field(line, "connection_reused") is True,
        "provider_start_kind": tracing_field(line, "provider_start_kind"),
        "request_sent_unix_ns": exact_integer(
            tracing_field(line, "request_sent_unix_ns")
        ),
        "provider_event_started_unix_ns": exact_integer(
            tracing_field(line, "provider_event_started_unix_ns")
        ),
        "completed_unix_ns": exact_integer(
            tracing_field(line, "completed_unix_ns")
        ),
        "request_to_completed_ms": numeric(
            tracing_field(line, "request_to_completed_ms")
        ),
        "provider_event_window_ms": numeric(
            tracing_field(line, "provider_event_window_ms")
        ),
        "input_tokens": integer(tracing_field(line, "input_tokens")),
        "output_tokens": integer(tracing_field(line, "output_tokens")),
        "reasoning_output_tokens": integer(
            tracing_field(line, "reasoning_output_tokens")
        ),
    }
    required = (
        "model",
        "provider_event_started_unix_ns",
        "completed_unix_ns",
        "output_tokens",
    )
    return parsed if all(parsed.get(field) is not None for field in required) else None


def parse_lifecycle_traces(stderr: str) -> tuple[list[dict[str, Any]], list[str]]:
    traces = []
    malformed = []
    for line in stderr.splitlines():
        if LIFECYCLE_TRACE_TARGET not in line:
            continue
        parsed = parse_lifecycle_trace_line(line)
        if parsed is None:
            malformed.append(line)
        else:
            parsed["lifecycle_trace_index"] = len(traces) + 1
            traces.append(parsed)
    return traces, malformed


def binary_has_lifecycle_trace_hook(path: Path) -> bool:
    marker = LIFECYCLE_TRACE_TARGET.encode()
    overlap = len(marker) - 1
    previous = b""
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                data = previous + chunk
                if marker in data:
                    return True
                previous = data[-overlap:]
    except OSError:
        return False
    return False


def union_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_duration_ns(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in union_intervals(intervals))


def interval_overlap_ns(
    interval: tuple[int, int], intervals: list[tuple[int, int]]
) -> int:
    start, end = interval
    return sum(
        max(0, min(end, other_end) - max(start, other_start))
        for other_start, other_end in union_intervals(intervals)
    )


def timeline_tool_intervals(
    records: list[dict[str, Any]],
    run_start_ns: int,
    run_end_ns: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Pair host-observed Codex item events into union-ready tool intervals."""
    starts: dict[str, dict[str, Any]] = {}
    tools = []
    completed_without_start = 0
    for record in records:
        item_id = record.get("item_id")
        item_type = record.get("item_type")
        event_type = record.get("type")
        if not item_id or item_type not in TOOL_ITEM_TYPES:
            continue
        if event_type == "item.started":
            starts[str(item_id)] = record
            continue
        if event_type != "item.completed":
            continue
        start = starts.pop(str(item_id), None)
        if start is None:
            completed_without_start += 1
            continue
        start_ns = exact_integer(start.get("received_unix_ns"))
        end_ns = exact_integer(record.get("received_unix_ns"))
        if start_ns is None or end_ns is None or end_ns <= start_ns:
            continue
        clipped_start = max(run_start_ns, start_ns)
        clipped_end = min(run_end_ns, end_ns)
        if clipped_end <= clipped_start:
            continue
        tools.append(
            {
                "item_id": str(item_id),
                "item_type": item_type,
                "start_unix_ns": clipped_start,
                "end_unix_ns": clipped_end,
                "duration_seconds": (clipped_end - clipped_start) / 1_000_000_000,
            }
        )
    return tools, {
        "completed_interval_count": len(tools),
        "unmatched_started_count": len(starts),
        "completed_without_start_count": completed_without_start,
    }


def summarize_lifecycle(
    traces: list[dict[str, Any]],
    timeline_records: list[dict[str, Any]],
    target_model: str,
    turn_usage: dict[str, Any],
    run_start_ns: int,
    run_end_ns: int,
    malformed_trace_count: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Compute ratio-of-sums TPS without subtracting concurrent tool time."""
    tools, tool_pairing = timeline_tool_intervals(
        timeline_records, run_start_ns, run_end_ns
    )
    tool_union = union_intervals(
        [(tool["start_unix_ns"], tool["end_unix_ns"]) for tool in tools]
    )
    calls = []
    all_model_windows = []
    target_windows = []
    all_model_request_windows = []
    target_request_windows = []
    for trace in traces:
        request_start_ns = exact_integer(trace.get("request_sent_unix_ns"))
        start_ns = exact_integer(trace.get("provider_event_started_unix_ns"))
        end_ns = exact_integer(trace.get("completed_unix_ns"))
        output_tokens = integer(trace.get("output_tokens"))
        window_ns = (
            end_ns - start_ns
            if start_ns is not None and end_ns is not None and end_ns > start_ns
            else None
        )
        valid_window = window_ns is not None and window_ns > 0
        request_window_ns = (
            end_ns - request_start_ns
            if request_start_ns is not None
            and end_ns is not None
            and end_ns > request_start_ns
            else None
        )
        valid_request_window = (
            request_window_ns is not None and request_window_ns > 0
        )
        overlap_ns = (
            min(window_ns, interval_overlap_ns((start_ns, end_ns), tool_union))
            if valid_window
            else None
        )
        request_overlap_ns = (
            min(
                request_window_ns,
                interval_overlap_ns((request_start_ns, end_ns), tool_union),
            )
            if valid_request_window
            else None
        )
        exclusion_reason = None
        if str(trace.get("model") or "") != target_model:
            exclusion_reason = "auxiliary_model"
        elif trace.get("warmup") is True:
            exclusion_reason = "warmup"
        elif not valid_window:
            exclusion_reason = "invalid_lifecycle_window"
        elif output_tokens is None:
            exclusion_reason = "missing_output_tokens"

        window_seconds = window_ns / 1_000_000_000 if window_ns else None
        call = {
            "schema_version": "codex-swebench-inference-call-1",
            "call_index": len(calls) + 1,
            **trace,
            "provider_window_inference_seconds": window_seconds,
            "provider_window_output_tps": (
                output_tokens / window_seconds
                if output_tokens is not None and window_seconds else None
            ),
            "tool_overlap_seconds": (
                overlap_ns / 1_000_000_000 if overlap_ns is not None else None
            ),
            "request_active_seconds": (
                request_window_ns / 1_000_000_000
                if request_window_ns is not None
                else None
            ),
            "request_active_output_tps": (
                output_tokens / (request_window_ns / 1_000_000_000)
                if output_tokens is not None and valid_request_window
                else None
            ),
            "request_active_tool_overlap_seconds": (
                request_overlap_ns / 1_000_000_000
                if request_overlap_ns is not None
                else None
            ),
            "included_in_primary": exclusion_reason is None,
            "primary_exclusion_reason": exclusion_reason,
        }
        calls.append(call)
        if valid_window and trace.get("warmup") is not True:
            all_model_windows.append((start_ns, end_ns))
        if valid_request_window and trace.get("warmup") is not True:
            all_model_request_windows.append((request_start_ns, end_ns))
        if exclusion_reason is None:
            target_windows.append((start_ns, end_ns))
            if valid_request_window:
                target_request_windows.append((request_start_ns, end_ns))

    eligible = [call for call in calls if call["included_in_primary"]]
    output_tokens = sum(int(call["output_tokens"]) for call in eligible)
    inference_seconds = sum(
        float(call["provider_window_inference_seconds"]) for call in eligible
    )
    request_active_seconds = sum(
        float(call["request_active_seconds"])
        for call in eligible
        if call.get("request_active_seconds") is not None
    )
    target_tool_overlap_seconds = sum(
        float(call["tool_overlap_seconds"]) for call in eligible
    )
    turn_output_tokens = integer(turn_usage.get("output_tokens"))
    output_reconciliation = (
        "matched"
        if turn_output_tokens is not None and turn_output_tokens == output_tokens
        else "unavailable"
        if turn_output_tokens is None
        else "mismatched"
    )
    target_missing = [
        call
        for call in calls
        if str(call.get("model") or "") == target_model
        and call.get("primary_exclusion_reason") not in {None, "warmup"}
    ]
    fallback_count = sum(
        call.get("provider_start_kind") == "request_sent_fallback"
        for call in eligible
    )
    coverage_reasons = []
    if malformed_trace_count:
        coverage_reasons.append("malformed_lifecycle_trace")
    if target_missing:
        coverage_reasons.append("missing_or_invalid_target_call")
    if not eligible:
        coverage_reasons.append("no_eligible_target_calls")
    if output_reconciliation != "matched":
        coverage_reasons.append("output_tokens_not_reconciled")
    if fallback_count:
        coverage_reasons.append("request_sent_boundary_fallback")
    if any(call.get("request_active_seconds") is None for call in eligible):
        coverage_reasons.append("missing_request_sent_boundary")

    all_model_union = union_intervals(all_model_windows)
    target_union = union_intervals(target_windows)
    all_model_request_union = union_intervals(all_model_request_windows)
    target_request_union = union_intervals(target_request_windows)
    all_model_union_ns = interval_duration_ns(all_model_union)
    all_model_request_union_ns = interval_duration_ns(all_model_request_union)
    target_request_union_ns = interval_duration_ns(target_request_union)
    tool_union_ns = interval_duration_ns(tool_union)
    target_tool_concurrency_ns = sum(
        interval_overlap_ns(window, tool_union) for window in target_union
    )
    all_model_tool_concurrency_ns = sum(
        interval_overlap_ns(window, tool_union) for window in all_model_union
    )
    target_request_tool_concurrency_ns = sum(
        interval_overlap_ns(window, tool_union)
        for window in target_request_union
    )
    all_model_request_tool_concurrency_ns = sum(
        interval_overlap_ns(window, tool_union)
        for window in all_model_request_union
    )
    wall_ns = max(0, run_end_ns - run_start_ns)
    accounted_union_ns = interval_duration_ns(all_model_union + tool_union)
    unattributed_ns = max(0, wall_ns - accounted_union_ns)
    request_accounted_union_ns = interval_duration_ns(
        all_model_request_union + tool_union
    )
    request_unattributed_ns = max(0, wall_ns - request_accounted_union_ns)
    request_only_ns = max(
        0,
        all_model_request_union_ns - all_model_request_tool_concurrency_ns,
    )
    tool_only_ns = max(
        0,
        tool_union_ns - all_model_request_tool_concurrency_ns,
    )
    request_partition_error_ns = wall_ns - (
        request_only_ns
        + tool_only_ns
        + all_model_request_tool_concurrency_ns
        + request_unattributed_ns
    )

    summary = {
        "schema_version": "codex-swebench-inference-accounting-1",
        "primary_model": target_model,
        "timing_basis": "response.created_to_response.completed",
        "server_engine_equivalent": False,
        "lifecycle_trace_count": len(traces),
        "malformed_lifecycle_trace_count": malformed_trace_count,
        "primary_eligible_call_count": len(eligible),
        "primary_missing_call_count": len(target_missing),
        "primary_output_tokens": output_tokens,
        "turn_usage_output_tokens": turn_output_tokens,
        "output_token_reconciliation": output_reconciliation,
        "provider_window_inference_seconds": inference_seconds,
        "provider_window_output_tps": (
            output_tokens / inference_seconds if inference_seconds else None
        ),
        "primary_request_seconds_sum": request_active_seconds,
        "primary_request_output_tps": (
            output_tokens / request_active_seconds
            if request_active_seconds
            else None
        ),
        "primary_request_union_seconds": (
            target_request_union_ns / 1_000_000_000
        ),
        "all_model_request_union_seconds": (
            all_model_request_union_ns / 1_000_000_000
        ),
        "primary_request_tool_concurrency_seconds": (
            target_request_tool_concurrency_ns / 1_000_000_000
        ),
        "all_model_request_tool_concurrency_seconds": (
            all_model_request_tool_concurrency_ns / 1_000_000_000
        ),
        "request_active_wall_partition": {
            "llm_request_only_seconds": request_only_ns / 1_000_000_000,
            "tool_only_seconds": tool_only_ns / 1_000_000_000,
            "llm_request_tool_overlap_seconds": (
                all_model_request_tool_concurrency_ns / 1_000_000_000
            ),
            "orchestration_residual_seconds": (
                request_unattributed_ns / 1_000_000_000
            ),
            "reconciliation_error_seconds": (
                request_partition_error_ns / 1_000_000_000
            ),
        },
        "target_tool_overlap_seconds_per_call_sum": target_tool_overlap_seconds,
        "target_inference_tool_concurrency_seconds": (
            target_tool_concurrency_ns / 1_000_000_000
        ),
        "all_model_inference_tool_concurrency_seconds": (
            all_model_tool_concurrency_ns / 1_000_000_000
        ),
        "total_all_models_provider_window_union_seconds": (
            all_model_union_ns / 1_000_000_000
        ),
        "total_tool_seconds": tool_union_ns / 1_000_000_000,
        "total_unattributed_seconds": unattributed_ns / 1_000_000_000,
        "total_wall_seconds": wall_ns / 1_000_000_000,
        "tool_interval_pairing": tool_pairing,
        "fallback_boundary_call_count": fallback_count,
        "coverage": "complete" if not coverage_reasons else "partial",
        "coverage_reasons": coverage_reasons,
    }
    return calls, summary, tools


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records
