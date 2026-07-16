#!/usr/bin/env python3
"""Benchmark LCB tasks through native Codex with JSONL and local OTel capture."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import selectors
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from bench_runner import check_solution, load_problems, resolve_problem_refs
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem


TIMING_TRACE_TARGET = "codex_api::responses_websocket_timing"
TIMING_EVENT_KIND = "responsesapi.websocket_timing"
LIFECYCLE_TRACE_TARGET = "codex_api::responses_websocket_lifecycle"
TIMING_FIELDS = (
    "responses_duration_excl_engine_and_client_tool_time_ms",
    "engine_service_total_ms",
    "engine_iapi_ttft_total_ms",
    "engine_service_ttft_total_ms",
    "engine_iapi_tbt_across_engine_calls_ms",
    "engine_service_tbt_across_engine_calls_ms",
)

try:
    from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
        ExportLogsServiceRequest,
    )
    from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
        ExportMetricsServiceRequest,
    )
except ImportError as exc:  # pragma: no cover - exercised by CLI preflight
    raise SystemExit(
        "Missing native benchmark dependencies. Run: "
        ".venv/bin/python -m pip install -r requirements-codex-native.txt"
    ) from exc


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def any_value(value) -> Any:
    kind = value.WhichOneof("value")
    if kind is None:
        return None
    raw = getattr(value, kind)
    if kind == "array_value":
        return [any_value(item) for item in raw.values]
    if kind == "kvlist_value":
        return {item.key: any_value(item.value) for item in raw.values}
    if kind == "bytes_value":
        return raw.hex()
    return raw


def attributes(values) -> dict[str, Any]:
    return {item.key: any_value(item.value) for item in values}


def decode_logs(payload: bytes, received_ns: int) -> list[dict[str, Any]]:
    request = ExportLogsServiceRequest()
    request.ParseFromString(payload)
    out = []
    for resource_logs in request.resource_logs:
        resource = attributes(resource_logs.resource.attributes)
        for scope_logs in resource_logs.scope_logs:
            scope = scope_logs.scope.name
            for record in scope_logs.log_records:
                out.append(
                    {
                        "signal": "log",
                        "time_unix_nano": record.time_unix_nano or received_ns,
                        "observed_time_unix_nano": record.observed_time_unix_nano,
                        "severity": record.severity_text,
                        "body": any_value(record.body),
                        "attributes": attributes(record.attributes),
                        "resource": resource,
                        "scope": scope,
                    }
                )
    return out


def metric_points(metric, received_ns: int) -> list[dict[str, Any]]:
    kind = metric.WhichOneof("data")
    data = getattr(metric, kind) if kind else None
    points = []
    for point in getattr(data, "data_points", []):
        item = {
            "signal": "metric",
            "name": metric.name,
            "description": metric.description,
            "unit": metric.unit,
            "kind": kind,
            "time_unix_nano": getattr(point, "time_unix_nano", 0) or received_ns,
            "attributes": attributes(point.attributes),
        }
        if hasattr(data, "aggregation_temporality"):
            item["aggregation_temporality"] = int(data.aggregation_temporality)
        if kind in {"gauge", "sum"}:
            number_kind = point.WhichOneof("value")
            item["value"] = getattr(point, number_kind) if number_kind else None
        elif kind == "histogram":
            item.update(count=point.count, sum=point.sum if point.HasField("sum") else None)
        points.append(item)
    return points


def decode_metrics(payload: bytes, received_ns: int) -> list[dict[str, Any]]:
    request = ExportMetricsServiceRequest()
    request.ParseFromString(payload)
    out = []
    for resource_metrics in request.resource_metrics:
        resource = attributes(resource_metrics.resource.attributes)
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                for point in metric_points(metric, received_ns):
                    point["resource"] = resource
                    point["scope"] = scope_metrics.scope.name
                    out.append(point)
    return out


class Collector:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.raw: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                size = int(self.headers.get("content-length", "0"))
                payload = self.rfile.read(size)
                received_ns = time.time_ns()
                try:
                    if self.path.endswith("/v1/logs"):
                        decoded = decode_logs(payload, received_ns)
                    elif self.path.endswith("/v1/metrics"):
                        decoded = decode_metrics(payload, received_ns)
                    else:
                        decoded = []
                    with owner.lock:
                        owner.events.extend(decoded)
                        owner.raw.append(
                            {"path": self.path, "bytes": len(payload), "received_ns": received_ns}
                        )
                    self.send_response(200)
                    self.send_header("content-type", "application/x-protobuf")
                    self.end_headers()
                    self.wfile.write(b"")
                except Exception as exc:  # retain exporter failures for audit
                    with owner.lock:
                        owner.raw.append(
                            {"path": self.path, "bytes": len(payload), "error": repr(exc)}
                        )
                    self.send_response(400)
                    self.end_headers()

            def log_message(self, *_args) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def logs_endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1/logs"

    @property
    def metrics_endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1/metrics"

    @property
    def endpoint(self) -> str:
        """Backward-compatible alias for the logs endpoint."""
        return self.logs_endpoint

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def snapshot(self, start: int) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.events[start:])


def event_name(event: dict[str, Any]) -> str:
    attrs = event.get("attributes", {})
    candidates = [
        event.get("name"), attrs.get("event.name"), attrs.get("name"),
        attrs.get("event_name"), event.get("body"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return ""


def attr(event: dict[str, Any], *names: str) -> Any:
    attrs = event.get("attributes", {})
    for name in names:
        if name in attrs:
            return attrs[name]
    return None


def usage_from_json_event(event: dict[str, Any]) -> dict[str, int]:
    usage = event.get("usage")
    if not isinstance(usage, dict):
        return {}
    return {
        key: int(value)
        for key, value in usage.items()
        if isinstance(value, (int, float))
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


def parse_timing_trace_line(line: str) -> dict[str, Any] | None:
    """Parse one opt-in Codex Responses WebSocket timing trace."""
    if TIMING_TRACE_TARGET not in line:
        return None
    payload = tracing_field(line, "payload")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict) or payload.get("type") != TIMING_EVENT_KIND:
        return None
    timing_metrics = payload.get("timing_metrics")
    if not isinstance(timing_metrics, dict):
        return None
    parsed = {
        "model": tracing_field(line, "model"),
        "session_id": tracing_field(line, "session_id"),
        "thread_id": tracing_field(line, "thread_id"),
        "turn_id": tracing_field(line, "turn_id"),
        "traceparent": tracing_field(line, "traceparent"),
        "previous_response_id": tracing_field(line, "previous_response_id"),
        "request_start_ms": tracing_field(line, "request_start_ms"),
        "warmup": tracing_field(line, "warmup") is True,
        "connection_reused": tracing_field(line, "connection_reused") is True,
    }
    for field in TIMING_FIELDS:
        parsed[field] = numeric(timing_metrics.get(field))
    return parsed


def parse_timing_traces(stderr: str) -> tuple[list[dict[str, Any]], list[str]]:
    timings = []
    malformed = []
    for line in stderr.splitlines():
        if TIMING_TRACE_TARGET not in line:
            continue
        parsed = parse_timing_trace_line(line)
        if parsed is None:
            malformed.append(line)
        else:
            parsed["timing_trace_index"] = len(timings) + 1
            timings.append(parsed)
    return timings, malformed


def parse_lifecycle_trace_line(line: str) -> dict[str, Any] | None:
    """Parse one locally measured Responses WebSocket lifecycle trace."""
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
        "provider_event_window_ms",
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


def union_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the union of half-open nanosecond intervals."""
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


def completed_tool_intervals(
    events: list[dict[str, Any]],
    run_start_ns: int,
    run_end_ns: int,
) -> list[dict[str, Any]]:
    """Recover exact completed-tool windows from duration-bearing OTel logs."""
    tools = []
    for event in events:
        if event_name(event) != "codex.tool_result":
            continue
        duration_ms = numeric(attr(event, "duration_ms"))
        end_ns = exact_integer(event.get("time_unix_nano"))
        if duration_ms is None or duration_ms < 0 or end_ns is None:
            continue
        start_ns = end_ns - round(duration_ms * 1_000_000)
        clipped_start = max(run_start_ns, start_ns)
        clipped_end = min(run_end_ns, end_ns)
        if clipped_end <= clipped_start:
            continue
        tools.append(
            {
                "tool_name": str(attr(event, "tool_name") or "unknown"),
                "call_id": str(attr(event, "call_id") or ""),
                "model": str(attr(event, "model", "slug") or ""),
                "conversation_id": str(attr(event, "conversation.id") or ""),
                "start_unix_ns": clipped_start,
                "end_unix_ns": clipped_end,
                "duration_ms": duration_ms,
            }
        )
    return tools


def summarize_client_lifecycle(
    traces: list[dict[str, Any]],
    events: list[dict[str, Any]],
    target_model: str,
    problem_id: str,
    run_start_ns: int,
    run_end_ns: int,
    turn_usage: dict[str, Any],
    malformed_trace_count: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Calculate client-observed inference windows and tool time.

    The primary duration starts at the first provider lifecycle event and ends
    when response.completed is fully received. Tool-only gaps are excluded by
    construction because every model call is measured separately. Concurrent
    tool time is reported as overlap, but is not subtracted from an in-flight
    inference call. This is deliberately not labeled server engine time.
    """
    tools = completed_tool_intervals(events, run_start_ns, run_end_ns)
    tool_windows = [
        (tool["start_unix_ns"], tool["end_unix_ns"]) for tool in tools
    ]
    tool_union = union_intervals(tool_windows)
    calls = []
    all_model_inference_windows: list[tuple[int, int]] = []
    for trace in traces:
        start_ns = exact_integer(trace.get("provider_event_started_unix_ns"))
        end_ns = exact_integer(trace.get("completed_unix_ns"))
        reported_window_ms = numeric(trace.get("provider_event_window_ms"))
        output_tokens = integer(trace.get("output_tokens"))
        window_ms = (
            (end_ns - start_ns) / 1_000_000
            if start_ns is not None and end_ns is not None and end_ns > start_ns
            else None
        )
        valid_window = (
            start_ns is not None
            and end_ns is not None
            and end_ns > start_ns
            and window_ms is not None
            and window_ms > 0
        )
        # Auxiliary conversations inherit the parent session_id, while OTel
        # tool logs use conversation.id. Prefer thread_id so a parent tool is
        # not attributed to a concurrently running guardian response.
        conversation_identity = trace.get("thread_id") or trace.get("session_id")
        conversation_aliases = (
            {str(conversation_identity)} if conversation_identity else set()
        )
        matching_tool_windows = [
            (tool["start_unix_ns"], tool["end_unix_ns"])
            for tool in tools
            if not conversation_aliases
            or tool.get("conversation_id") in conversation_aliases
        ]
        matching_tool_union = union_intervals(matching_tool_windows)
        overlap_ms = (
            min(
                window_ms,
                interval_overlap_ns((start_ns, end_ns), matching_tool_union)
                / 1_000_000,
            )
            if valid_window
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

        call = {
            "schema_version": "native-codex-client-inference-call-2",
            "problem_id": problem_id,
            "call_index": len(calls) + 1,
            **trace,
            "provider_window_inference_ms": window_ms,
            "provider_window_inference_seconds": (
                window_ms / 1000 if window_ms is not None else None
            ),
            "provider_window_output_tps": (
                output_tokens * 1000 / window_ms
                if output_tokens is not None and window_ms and window_ms > 0
                else None
            ),
            "reported_provider_window_delta_ms": (
                reported_window_ms - window_ms
                if reported_window_ms is not None and window_ms is not None
                else None
            ),
            "tool_overlap_ms": overlap_ms,
            # Compatibility aliases. As of schema 2, these preserve the full
            # provider lifecycle window; tool overlap is diagnostic only.
            "client_active_inference_ms": window_ms,
            "client_active_inference_seconds": (
                window_ms / 1000 if window_ms is not None else None
            ),
            "client_active_output_tps": (
                output_tokens * 1000 / window_ms
                if output_tokens is not None and window_ms and window_ms > 0
                else None
            ),
            "included_in_primary": exclusion_reason is None,
            "primary_exclusion_reason": exclusion_reason,
            "timing_boundary_quality": (
                "request_sent_fallback"
                if trace.get("provider_start_kind") == "request_sent_fallback"
                else "provider_lifecycle_event"
            ),
        }
        calls.append(call)
        if valid_window and trace.get("warmup") is not True:
            all_model_inference_windows.append((start_ns, end_ns))

    eligible = [call for call in calls if call["included_in_primary"]]
    output_sum = sum(int(call["output_tokens"]) for call in eligible)
    inference_ms_sum = sum(
        float(call["provider_window_inference_ms"]) for call in eligible
    )
    overlap_ms_sum = sum(float(call["tool_overlap_ms"]) for call in eligible)
    all_model_inference_union = union_intervals(all_model_inference_windows)
    all_models_inference_ns = interval_duration_ns(all_model_inference_union)
    tool_ns = interval_duration_ns(tool_union)
    wall_ns = max(0, run_end_ns - run_start_ns)
    inference_tool_concurrency_ns = sum(
        interval_overlap_ns(window, tool_union)
        for window in all_model_inference_union
    )
    accounted_union_ns = interval_duration_ns(
        all_model_inference_union + tool_union
    )
    unattributed_ns = max(
        0,
        wall_ns - accounted_union_ns,
    )
    reconciliation_error_ns = (
        wall_ns
        - all_models_inference_ns
        - tool_ns
        + inference_tool_concurrency_ns
        - unattributed_ns
    )
    turn_output_tokens = integer(turn_usage.get("output_tokens"))
    output_reconciliation = (
        "matched"
        if turn_output_tokens is not None and turn_output_tokens == output_sum
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

    by_tool: dict[str, list[tuple[int, int]]] = {}
    for tool in tools:
        by_tool.setdefault(tool["tool_name"], []).append(
            (tool["start_unix_ns"], tool["end_unix_ns"])
        )
    tool_breakdown = {
        name: {
            "event_count": sum(tool["tool_name"] == name for tool in tools),
            "union_seconds": interval_duration_ns(windows) / 1_000_000_000,
        }
        for name, windows in sorted(by_tool.items())
    }
    summary = {
        "schema_version": "native-codex-client-accounting-2",
        "primary_model": target_model,
        "lifecycle_trace_count": len(traces),
        "malformed_lifecycle_trace_count": malformed_trace_count,
        "primary_eligible_call_count": len(eligible),
        "primary_missing_call_count": len(target_missing),
        "primary_output_tokens": output_sum,
        "primary_provider_window_inference_seconds": inference_ms_sum / 1000,
        "primary_provider_window_output_tps": (
            output_sum * 1000 / inference_ms_sum if inference_ms_sum else None
        ),
        # Compatibility aliases for schema 1 consumers.
        "primary_provider_event_window_seconds": inference_ms_sum / 1000,
        "primary_tool_overlap_seconds": overlap_ms_sum / 1000,
        "primary_client_active_inference_seconds": inference_ms_sum / 1000,
        "primary_client_active_output_tps": (
            output_sum * 1000 / inference_ms_sum if inference_ms_sum else None
        ),
        "turn_usage_output_tokens": turn_output_tokens,
        "output_token_reconciliation": output_reconciliation,
        "total_wall_seconds": wall_ns / 1_000_000_000,
        "total_all_models_provider_window_inference_seconds": (
            all_models_inference_ns / 1_000_000_000
        ),
        "total_all_models_client_active_inference_seconds": (
            all_models_inference_ns / 1_000_000_000
        ),
        "total_tool_seconds": tool_ns / 1_000_000_000,
        "inference_tool_concurrency_seconds": (
            inference_tool_concurrency_ns / 1_000_000_000
        ),
        "total_unattributed_seconds": unattributed_ns / 1_000_000_000,
        "accounting_reconciliation_error_seconds": (
            reconciliation_error_ns / 1_000_000_000
        ),
        "tool_event_count": len(tools),
        "tool_breakdown_non_additive": tool_breakdown,
        "fallback_boundary_call_count": fallback_count,
        "primary_coverage": "complete" if not coverage_reasons else "partial",
        "coverage_reasons": coverage_reasons,
        "timing_basis": (
            "first_provider_lifecycle_event_to_response_completed"
        ),
        "server_engine_equivalent": False,
        "tool_intervals_are_union_deduplicated": True,
    }
    return calls, summary


def completed_inference_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = []
    ordered = sorted(events, key=lambda event: event.get("time_unix_nano", 0))
    for event in ordered:
        if event_name(event) != "codex.sse_event":
            continue
        if attr(event, "event.kind", "kind") != "response.completed":
            continue
        completed.append(
            {
                "model": attr(event, "model", "slug"),
                "conversation_id": attr(event, "conversation.id"),
                "completed_at": attr(event, "event.timestamp"),
                "completed_time_unix_nano": event.get("time_unix_nano"),
                "input_tokens": integer(attr(event, "input_token_count")),
                "cached_input_tokens": integer(attr(event, "cached_token_count")),
                "output_tokens": integer(attr(event, "output_token_count")),
                "reasoning_output_tokens": integer(attr(event, "reasoning_token_count")),
                "tool_tokens": integer(attr(event, "tool_token_count")),
                "client_ttft_ms": numeric(attr(event, "ttft_ms")),
            }
        )
    return completed


def pair_inference_calls(
    timings: list[dict[str, Any]],
    completions: list[dict[str, Any]],
    target_model: str,
    problem_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair per-request server timing with per-request token usage.

    Responses are sequential within a Codex conversation. Prefer exact
    model+conversation ordinal pairing; permit model ordinal pairing only when
    that model has a single conversation in the captured OTel stream.
    """
    used_completions: set[int] = set()
    calls: list[dict[str, Any]] = []
    conversations_by_model: dict[str, set[str]] = {}
    for completion in completions:
        model = str(completion.get("model") or "")
        conversation_id = str(completion.get("conversation_id") or "")
        if conversation_id:
            conversations_by_model.setdefault(model, set()).add(conversation_id)

    for timing in timings:
        model = str(timing.get("model") or "")
        aliases = {
            str(value) for value in (timing.get("thread_id"), timing.get("session_id"))
            if value
        }
        exact = [
            index for index, completion in enumerate(completions)
            if index not in used_completions
            and str(completion.get("model") or "") == model
            and str(completion.get("conversation_id") or "") in aliases
        ]
        pairing = "conversation_ordinal"
        candidates = exact
        if not candidates and len(conversations_by_model.get(model, set())) <= 1:
            candidates = [
                index for index, completion in enumerate(completions)
                if index not in used_completions
                and str(completion.get("model") or "") == model
            ]
            pairing = "single_conversation_model_ordinal"
        completion = None
        if candidates:
            completion_index = candidates[0]
            used_completions.add(completion_index)
            completion = completions[completion_index]

        call = {
            "schema_version": "native-codex-inference-call-1",
            "problem_id": problem_id,
            "call_index": len(calls) + 1,
            **timing,
            "pairing": pairing if completion else "unpaired_timing",
        }
        if completion:
            call.update(completion)
        calls.append(call)

    for index, completion in enumerate(completions):
        if index in used_completions:
            continue
        calls.append(
            {
                "schema_version": "native-codex-inference-call-1",
                "problem_id": problem_id,
                "call_index": len(calls) + 1,
                **completion,
                "pairing": "unpaired_completion",
            }
        )

    for call in calls:
        output_tokens = integer(call.get("output_tokens"))
        inference_ms = numeric(call.get("engine_service_total_ms"))
        server_ttft_ms = numeric(call.get("engine_service_ttft_total_ms"))
        call["engine_inference_seconds"] = (
            inference_ms / 1000 if inference_ms is not None else None
        )
        call["inference_output_tps"] = (
            output_tokens * 1000 / inference_ms
            if output_tokens is not None and inference_ms and inference_ms > 0
            else None
        )
        decode_ms = (
            max(0.0, inference_ms - server_ttft_ms)
            if inference_ms is not None and server_ttft_ms is not None
            else None
        )
        call["engine_decode_ms_diagnostic"] = decode_ms
        call["decode_output_tps_diagnostic"] = (
            max(0, output_tokens - 1) * 1000 / decode_ms
            if output_tokens is not None and decode_ms and decode_ms > 0
            else None
        )

        exclusion_reason = None
        if str(call.get("model") or "") != target_model:
            exclusion_reason = "auxiliary_model"
        elif call.get("warmup") is True:
            exclusion_reason = "warmup"
        elif call.get("pairing") in {"unpaired_timing", "unpaired_completion"}:
            exclusion_reason = call["pairing"]
        elif output_tokens is None:
            exclusion_reason = "missing_output_tokens"
        elif inference_ms is None or inference_ms <= 0:
            exclusion_reason = "missing_engine_inference_time"
        call["included_in_primary"] = exclusion_reason is None
        call["primary_exclusion_reason"] = exclusion_reason

    eligible = [call for call in calls if call["included_in_primary"]]
    output_sum = sum(int(call["output_tokens"]) for call in eligible)
    inference_ms_sum = sum(float(call["engine_service_total_ms"]) for call in eligible)
    target_missing = [
        call for call in calls
        if str(call.get("model") or "") == target_model
        and call.get("primary_exclusion_reason") not in {None, "warmup"}
    ]
    summary = {
        "timing_event_count": len(timings),
        "malformed_timing_event_count": 0,
        "completion_event_count": len(completions),
        "paired_call_count": sum(
            call["pairing"] not in {"unpaired_timing", "unpaired_completion"}
            for call in calls
        ),
        "recorded_call_count": len(calls),
        "primary_model": target_model,
        "primary_eligible_call_count": len(eligible),
        "primary_output_tokens": output_sum,
        "primary_engine_inference_ms": inference_ms_sum,
        "primary_engine_inference_seconds": inference_ms_sum / 1000,
        "primary_inference_output_tps": (
            output_sum * 1000 / inference_ms_sum if inference_ms_sum else None
        ),
        "primary_coverage": "complete" if not target_missing and eligible else "partial",
        "primary_missing_call_count": len(target_missing),
    }
    return calls, summary


def summarize_otel(
    events: list[dict[str, Any]], target_model: str | None = None
) -> dict[str, Any]:
    ordered = sorted(events, key=lambda event: event.get("time_unix_nano", 0))
    metric_sums: dict[str, float] = {}
    target_metric_sums: dict[str, float] = {}
    target_metric_counts: dict[str, int] = {}
    inference_metric_models: set[str] = set()
    inference_metric_temporalities: set[int] = set()
    inference_metric_point_count = 0
    target_inference_metric_point_count = 0
    inference_metric_missing_model = False
    api_durations = []
    sse_events = []
    for event in ordered:
        name = event_name(event)
        if event.get("signal") == "metric":
            value = event.get("sum") if event.get("sum") is not None else event.get("value")
            if isinstance(value, (int, float)):
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value)
                model = attr(event, "model", "slug")
                matches_target = target_model is None or model == target_model
                if matches_target:
                    target_metric_sums[name] = (
                        target_metric_sums.get(name, 0.0) + float(value)
                    )
                    count = integer(event.get("count"))
                    target_metric_counts[name] = (
                        target_metric_counts.get(name, 0) + (count or 0)
                    )
                if name.endswith("responses_api_inference_time.duration_ms"):
                    inference_metric_point_count += 1
                    if model is None:
                        inference_metric_missing_model = True
                    else:
                        inference_metric_models.add(str(model))
                    temporality = integer(event.get("aggregation_temporality"))
                    if temporality is not None:
                        inference_metric_temporalities.add(temporality)
                    if matches_target:
                        target_inference_metric_point_count += 1
        if name.endswith("api_request") or name == "codex.api_request":
            duration = attr(event, "duration_ms", "duration.ms", "duration")
            if isinstance(duration, (int, float)):
                api_durations.append(float(duration) / 1000)
        if name.endswith("sse_event") or name == "codex.sse_event":
            sse_events.append(event)

    # Event-kind timing is useful but not automatically principal: current
    # Codex OTel documents the SSE kind, not necessarily item.type=reasoning.
    active_windows = []
    pending_start = None
    pending_detail = None
    pending_item_type = None
    for event in sse_events:
        kind = str(attr(event, "kind", "event.kind", "sse.kind") or "")
        timestamp = event.get("time_unix_nano", 0) / 1e9
        if pending_start is None and kind in {
            "response.output_item.added",
            "response.output_text.delta",
            "response.function_call_arguments.delta",
        }:
            pending_start = timestamp
            pending_detail = kind
            pending_item_type = attr(
                event, "item.type", "item_type", "output_item.type"
            )
        elif pending_start is not None and kind == "response.completed":
            active_windows.append(
                {
                    "start_kind": pending_detail,
                    "start_item_type": pending_item_type,
                    "seconds": max(0.0, timestamp - pending_start),
                }
            )
            pending_start = None
            pending_detail = None
            pending_item_type = None

    target_inference_ms = sum(
        value for name, value in target_metric_sums.items()
        if name.endswith("responses_api_inference_time.duration_ms")
    )
    target_inference_count = sum(
        value for name, value in target_metric_counts.items()
        if name.endswith("responses_api_inference_time.duration_ms")
    )
    if not inference_metric_point_count:
        model_filter = "unavailable"
    elif target_model is None:
        model_filter = "unfiltered"
    elif inference_metric_missing_model:
        model_filter = "partial_missing_model"
    elif target_inference_metric_point_count:
        model_filter = "exact"
    else:
        model_filter = "no_matching_model"
    scrutable = bool(active_windows) and all(
        item["start_kind"] == "response.output_item.added"
        and item["start_item_type"] == "reasoning"
        for item in active_windows
    )
    active_seconds = sum(item["seconds"] for item in active_windows) or None
    return {
        "event_count": len(events),
        "api_request_seconds": sum(api_durations) if api_durations else None,
        "responses_api_inference_seconds": (
            target_inference_ms / 1000
            if target_inference_metric_point_count else None
        ),
        "responses_api_inference_observation_count": target_inference_count,
        "responses_api_inference_metric_models": sorted(inference_metric_models),
        "responses_api_inference_metric_temporalities": sorted(
            inference_metric_temporalities
        ),
        "responses_api_inference_model_filter": model_filter,
        "sse_active_windows": active_windows,
        "sse_active_seconds": active_seconds,
        "sse_boundary_confidence": (
            "provider_boundary" if scrutable else "event_kind_only" if active_windows else "unavailable"
        ),
        "scrutable_active_generation_seconds": active_seconds if scrutable else None,
        "metric_sums": metric_sums,
        "target_metric_sums": target_metric_sums,
        "target_metric_counts": target_metric_counts,
    }


def summarize_server_aggregate(
    completions: list[dict[str, Any]],
    otel_summary: dict[str, Any],
    target_model: str,
    turn_usage: dict[str, int],
) -> dict[str, Any]:
    """Compute target-model TPS from Codex's aggregate server metric.

    The duration histogram is emitted once for each Responses inference call,
    so its ratio of sums excludes all wall-clock gaps between calls (including
    tool execution). Stock Codex does not expose the individual observations.
    """
    target_completions = [
        completion for completion in completions
        if str(completion.get("model") or "") == target_model
    ]
    completed_with_tokens = [
        completion for completion in target_completions
        if integer(completion.get("output_tokens")) is not None
    ]
    output_tokens = sum(
        integer(completion.get("output_tokens")) or 0
        for completion in completed_with_tokens
    )
    completed_call_count = len(target_completions)
    observation_count = integer(
        otel_summary.get("responses_api_inference_observation_count")
    ) or 0
    turn_output_tokens = integer(turn_usage.get("output_tokens"))
    token_reconciliation = (
        "matched"
        if turn_output_tokens is not None and turn_output_tokens == output_tokens
        else "unavailable"
        if turn_output_tokens is None
        else "mismatched"
    )
    call_reconciliation = (
        "matched"
        if completed_call_count and observation_count == completed_call_count
        else "unavailable"
        if not completed_call_count or not observation_count
        else "mismatched"
    )
    inference_seconds = numeric(
        otel_summary.get("responses_api_inference_seconds")
    )
    temporalities = otel_summary.get(
        "responses_api_inference_metric_temporalities", []
    )
    # OTLP enum 1 is DELTA. Summing multiple export batches is exact only for
    # delta histograms; Codex's metrics exporter requests delta temporality.
    temporality = "delta" if temporalities == [1] else "unsupported_or_missing"
    coverage_reasons = []
    if otel_summary.get("responses_api_inference_model_filter") != "exact":
        coverage_reasons.append("metric_model_filter_not_exact")
    if temporality != "delta":
        coverage_reasons.append("metric_temporality_not_delta")
    if completed_call_count != len(completed_with_tokens):
        coverage_reasons.append("completion_missing_output_tokens")
    if token_reconciliation != "matched":
        coverage_reasons.append("turn_output_tokens_not_reconciled")
    if call_reconciliation != "matched":
        coverage_reasons.append("metric_observations_not_reconciled")
    if inference_seconds is None or inference_seconds <= 0:
        coverage_reasons.append("missing_engine_inference_time")
    coverage = "complete" if not coverage_reasons else "partial"
    return {
        "primary_model": target_model,
        "completed_call_count": completed_call_count,
        "output_tokens": output_tokens,
        "turn_usage_output_tokens": turn_output_tokens,
        "output_token_reconciliation": token_reconciliation,
        "engine_inference_seconds": inference_seconds,
        "inference_observation_count": observation_count,
        "call_count_reconciliation": call_reconciliation,
        "metric_model_filter": otel_summary.get(
            "responses_api_inference_model_filter"
        ),
        "metric_temporality": temporality,
        "inference_output_tps": (
            output_tokens / inference_seconds
            if coverage == "complete" and inference_seconds else None
        ),
        "coverage": coverage,
        "coverage_reasons": coverage_reasons,
        "excludes_tool_time": True,
        "includes_warmup_timing": True,
        "granularity": "query_ratio_of_sums",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--codex-bin",
        default="codex",
        help="Codex executable (instrumented binary required only in micro mode).",
    )
    parser.add_argument(
        "--timing-mode",
        choices=("micro", "aggregate"),
        default="micro",
        help=(
            "micro records every client-observed inference call and removes "
            "overlapping tool time; aggregate uses the "
            "stock CLI's target-model server histogram and requires no build"
        ),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning", default="xhigh")
    problem_source = parser.add_mutually_exclusive_group(required=True)
    problem_source.add_argument("--problems", nargs="+")
    problem_source.add_argument(
        "--problem-fixture",
        type=Path,
        help="Self-contained JSON problem fixture; bypasses dataset loading.",
    )
    parser.add_argument("--release", default="release_v6")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    parser.add_argument("--checker-timeout", type=int, default=12)
    parser.add_argument(
        "--allow-partial-inference-timing",
        action="store_true",
        help="Return normally even if the selected timing mode has partial coverage.",
    )
    return parser.parse_args()


def load_problem_fixture(path: Path) -> list[CodeGenerationProblem]:
    payload = json.loads(path.read_text())
    rows = payload if isinstance(payload, list) else [payload]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Problem fixture must contain one object or a non-empty list")
    problems = []
    for raw_row in rows:
        row = dict(raw_row)
        for field in ("public_test_cases", "private_test_cases", "metadata"):
            if not isinstance(row.get(field), str):
                row[field] = json.dumps(row.get(field, [] if field != "metadata" else {}))
        problems.append(CodeGenerationProblem(**row))
    return problems


def resolve_codex_binary(value: str) -> Path:
    resolved = shutil.which(value)
    if resolved is None:
        candidate = Path(value).expanduser()
        if candidate.is_file():
            resolved = str(candidate.resolve())
    if resolved is None:
        raise SystemExit(f"Codex executable not found: {value}")
    return Path(resolved).resolve()


def binary_has_timing_trace_hook(path: Path) -> bool:
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


def run_one(args, collector: Collector, problem, number: int) -> dict[str, Any]:
    run_dir = args.output_dir / f"q{number}_{problem.question_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    solution_path = run_dir / "solution.py"
    final_path = run_dir / "final.txt"
    jsonl_path = run_dir / "codex-events.jsonl"
    otel_path = run_dir / "otel-events.jsonl"
    stderr_path = run_dir / "codex-stderr.log"
    inference_calls_path = run_dir / "inference-calls.jsonl"
    server_inference_calls_path = run_dir / "server-inference-calls.jsonl"
    tool_intervals_path = run_dir / "tool-intervals.jsonl"
    prompt = (
        "Solve the programming problem below using your normal Codex agent workflow. "
        "Write the complete Python submission to solution.py in the current working "
        "directory, test it, and do not modify files outside this directory.\n\n"
        + problem.question_content
    )
    otel_start = len(collector.events)
    command = [
        str(args.codex_bin), "exec", "--json", "--ephemeral",
        "--skip-git-repo-check",
        "--color", "never", "--enable", "runtime_metrics",
        "--sandbox", "workspace-write", "--model", args.model,
        "-c", f'model_reasoning_effort="{args.reasoning}"',
        "-c", 'otel.environment="native-benchmark"',
        "-c", "otel.log_user_prompt=false",
        "-c", (
            "otel.exporter={ otlp-http = { endpoint = \""
            + collector.logs_endpoint
            + "\", protocol = \"binary\" } }"
        ),
        "-c", (
            "otel.metrics_exporter={ otlp-http = { endpoint = \""
            + collector.metrics_endpoint
            + "\", protocol = \"binary\" } }"
        ),
        "--cd", str(run_dir), "--output-last-message", str(final_path), "-",
    ]
    process_env = os.environ.copy()
    process_env["RUST_LOG"] = (
        f"error,{TIMING_TRACE_TARGET}=trace,{LIFECYCLE_TRACE_TARGET}=trace"
    )
    started_at = utcnow()
    started = time.perf_counter()
    run_start_ns = time.time_ns()
    json_events = []
    jsonlog = jsonl_path.open("w")
    first_item_s = None
    final_usage = {}
    print(
        json.dumps(
            {
                "status": "starting",
                "problem_number": number,
                "problem_id": problem.question_id,
                "model": args.model,
            }
        ),
        flush=True,
    )
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1, env=process_env,
    )
    assert process.stdin and process.stdout and process.stderr
    stderr_parts = []
    lifecycle_heartbeat_count = 0

    def drain_stderr() -> None:
        nonlocal lifecycle_heartbeat_count
        assert process.stderr
        for stderr_line in process.stderr:
            stderr_parts.append(stderr_line)
            lifecycle = parse_lifecycle_trace_line(stderr_line)
            if lifecycle is not None:
                lifecycle_heartbeat_count += 1
                duration_ms = lifecycle.get("provider_event_window_ms")
                tokens = lifecycle.get("output_tokens")
                print(
                    json.dumps(
                        {
                            "status": "inference_micro_session",
                            "problem_number": number,
                            "problem_id": problem.question_id,
                            "call_index": lifecycle_heartbeat_count,
                            "model": lifecycle.get("model"),
                            "warmup": lifecycle.get("warmup"),
                            "provider_start_kind": lifecycle.get(
                                "provider_start_kind"
                            ),
                            "provider_event_window_s": (
                                round(duration_ms / 1000, 3)
                                if duration_ms is not None
                                else None
                            ),
                            "output_tokens": tokens,
                            "raw_window_output_tps": (
                                round(tokens * 1000 / duration_ms, 2)
                                if tokens is not None and duration_ms
                                else None
                            ),
                            "note": "tool overlap is reconciled at query end",
                        }
                    ),
                    flush=True,
                )

    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()
    process.stdin.write(prompt)
    process.stdin.close()
    deadline = started + args.timeout_seconds
    last_heartbeat = started
    timed_out = False
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    while True:
        if time.perf_counter() > deadline:
            timed_out = True
            process.kill()
            break
        ready = selector.select(timeout=0.5)
        if not ready:
            now = time.perf_counter()
            if now - last_heartbeat >= 30:
                print(
                    json.dumps(
                        {
                            "status": "running",
                            "problem_number": number,
                            "problem_id": problem.question_id,
                            "elapsed_s": round(now - started, 1),
                            "codex_events": len(json_events),
                        }
                    ),
                    flush=True,
                )
                last_heartbeat = now
            if process.poll() is not None:
                remainder = process.stdout.read()
                if remainder:
                    for line in remainder.splitlines():
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            event = {"type": "unparsed", "raw": line}
                        event["harness_received_s"] = time.perf_counter() - started
                        json_events.append(event)
                        jsonlog.write(json.dumps(event) + "\n")
                        jsonlog.flush()
                break
            continue
        line = process.stdout.readline()
        if not line:
            if process.poll() is not None:
                break
            time.sleep(0.1)
            continue
        received_s = time.perf_counter() - started
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            event = {"type": "unparsed", "raw": line.rstrip()}
        event["harness_received_s"] = received_s
        json_events.append(event)
        jsonlog.write(json.dumps(event) + "\n")
        jsonlog.flush()
        event_type = str(event.get("type", ""))
        if event_type in {"item.completed", "turn.completed", "turn.failed", "error"}:
            item = event.get("item") if isinstance(event.get("item"), dict) else {}
            print(
                json.dumps(
                    {
                        "status": "activity",
                        "problem_number": number,
                        "problem_id": problem.question_id,
                        "elapsed_s": round(received_s, 1),
                        "event": event_type,
                        "item_type": item.get("type"),
                    }
                ),
                flush=True,
            )
            last_heartbeat = time.perf_counter()
        if first_item_s is None and str(event.get("type", "")).startswith("item."):
            first_item_s = received_s
        usage = usage_from_json_event(event)
        if usage:
            final_usage = usage
        if event.get("type") == "turn.failed":
            process.terminate()
            break
    try:
        returncode = process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        returncode = process.wait(timeout=5)
    jsonlog.close()
    stderr_thread.join(timeout=5)
    stderr = "".join(stderr_parts)
    stderr_path.write_text(stderr)
    agent_wall_s = time.perf_counter() - started
    run_end_ns = run_start_ns + round(agent_wall_s * 1_000_000_000)
    # Codex exporters flush asynchronously at shutdown.
    time.sleep(0.5)
    otel_events = collector.snapshot(otel_start)
    otel_path.write_text("".join(json.dumps(e) + "\n" for e in otel_events))
    otel = summarize_otel(otel_events, args.model)
    timings, malformed_timings = parse_timing_traces(stderr)
    lifecycle_traces, malformed_lifecycle_traces = parse_lifecycle_traces(stderr)
    completions = completed_inference_events(otel_events)
    server_aggregate = summarize_server_aggregate(
        completions, otel, args.model, final_usage
    )
    server_inference_calls, server_inference_summary = pair_inference_calls(
        timings, completions, args.model, problem.question_id
    )
    server_inference_summary["malformed_timing_event_count"] = len(
        malformed_timings
    )
    if (
        malformed_timings
        and server_inference_summary["primary_coverage"] == "complete"
    ):
        server_inference_summary["primary_coverage"] = "partial"
    output_tokens = final_usage.get("output_tokens")
    inference_calls, inference_summary = summarize_client_lifecycle(
        lifecycle_traces,
        otel_events,
        args.model,
        problem.question_id,
        run_start_ns,
        run_end_ns,
        final_usage,
        len(malformed_lifecycle_traces),
    )
    inference_calls_path.write_text(
        "".join(json.dumps(call) + "\n" for call in inference_calls)
    )
    tool_intervals_path.write_text(
        "".join(
            json.dumps(tool) + "\n"
            for tool in completed_tool_intervals(
                otel_events, run_start_ns, run_end_ns
            )
        )
    )
    server_inference_calls_path.write_text(
        "".join(json.dumps(call) + "\n" for call in server_inference_calls)
    )
    sse_active_s = otel["sse_active_seconds"]
    scrutable_active_s = otel["scrutable_active_generation_seconds"]
    diagnostic_stderr = "".join(
        line
        for line in stderr_parts
        if TIMING_TRACE_TARGET not in line
        and LIFECYCLE_TRACE_TARGET not in line
    )
    result = {
        "started_at": started_at,
        "cohort": "native_codex_cli",
        "timing_mode": args.timing_mode,
        "model": args.model,
        "reasoning_effort": args.reasoning,
        "checker_scope": args.checker_scope,
        "problem_number": number,
        "problem_id": problem.question_id,
        "title": problem.question_title,
        "returncode": returncode,
        "timed_out": timed_out,
        "usage": final_usage,
        "first_codex_item_s": first_item_s,
        "agent_wall_s": agent_wall_s,
        "agent_end_to_end_output_tps": (
            output_tokens / agent_wall_s if output_tokens is not None and agent_wall_s > 0 else None
        ),
        "otel": otel,
        "inference_calls": inference_summary,
        "server_inference_calls": server_inference_summary,
        "server_aggregate_inference": server_aggregate,
        "native_provider_window_output_tps": inference_summary[
            "primary_provider_window_output_tps"
        ],
        "native_inference_output_tps": inference_summary[
            "primary_provider_window_output_tps"
        ],
        # Compatibility alias for schema 4 consumers.
        "native_client_active_output_tps": inference_summary[
            "primary_provider_window_output_tps"
        ],
        "total_tool_seconds": inference_summary["total_tool_seconds"],
        "inference_tool_concurrency_seconds": inference_summary[
            "inference_tool_concurrency_seconds"
        ],
        "total_inference_seconds": inference_summary[
            "primary_provider_window_inference_seconds"
        ],
        "total_all_models_inference_seconds": inference_summary[
            "total_all_models_provider_window_inference_seconds"
        ],
        "total_unattributed_seconds": inference_summary[
            "total_unattributed_seconds"
        ],
        "server_aggregate_inference_output_tps": server_aggregate[
            "inference_output_tps"
        ],
        # Compatibility alias for summaries produced before timing modes.
        "codex_reported_inference_output_tps": server_aggregate[
            "inference_output_tps"
        ],
        "sse_event_window_output_tps_diagnostic": (
            output_tokens / sse_active_s if output_tokens is not None and sse_active_s else None
        ),
        "active_generation_output_tps": (
            output_tokens / scrutable_active_s
            if output_tokens is not None and scrutable_active_s else None
        ),
        "active_generation_eligibility": (
            "eligible" if scrutable_active_s
            else "unavailable_until_reasoning_item_type_is_scrutable"
        ),
        "solution_path": str(solution_path),
        "jsonl_path": str(jsonl_path),
        "otel_path": str(otel_path),
        "stderr_path": str(stderr_path),
        "inference_calls_path": str(inference_calls_path),
        "server_inference_calls_path": str(server_inference_calls_path),
        "tool_intervals_path": str(tool_intervals_path),
        "stderr_tail": diagnostic_stderr[-4000:],
    }
    if returncode and not json_events and "failed to initialize in-process app-server" in stderr:
        result["setup_error"] = "codex_app_server_initialization_denied"
    if solution_path.exists():
        fenced = f"```python\n{solution_path.read_text()}\n```"
        result["passed"] = check_solution(problem, fenced, args.checker_timeout)
    else:
        result["passed"] = False
        result["solution_missing"] = True
    (run_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    args = parse_args()
    args.codex_bin = resolve_codex_binary(args.codex_bin)
    if (
        args.timing_mode == "micro"
        and not args.allow_partial_inference_timing
        and not binary_has_timing_trace_hook(args.codex_bin)
    ):
        raise SystemExit(
            "The selected Codex binary does not expose local response lifecycle "
            "timing. "
            "Run ./build_instrumented_codex, then pass "
            "--codex-bin .codex-instrumented/codex-v0.144.3."
        )
    args.output_dir = args.output_dir.resolve()
    if args.problem_fixture is not None:
        fixture_path = args.problem_fixture.expanduser().resolve()
        problems = load_problem_fixture(fixture_path)
        args.checker_scope = "fixture_public_samples_only"
    else:
        ids = resolve_problem_refs(args.problems, args.release)
        problems = load_problems(ids, args.release)
        args.checker_scope = "release_dataset_public_and_private"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    collector = Collector()
    collector.start()
    rows = []
    try:
        for number, problem in enumerate(problems, 1):
            result = run_one(args, collector, problem, number)
            rows.append(result)
            print(json.dumps(result), flush=True)
    finally:
        collector.stop()
    eligible = [row for row in rows if row.get("usage", {}).get("output_tokens") is not None]
    output_sum = sum(row["usage"]["output_tokens"] for row in eligible)
    agent_seconds = sum(row["agent_wall_s"] for row in eligible)
    native_inference_rows = [
        row for row in rows
        if row.get("inference_calls", {}).get("primary_eligible_call_count")
    ]
    native_inference_tokens = sum(
        row["inference_calls"]["primary_output_tokens"]
        for row in native_inference_rows
    )
    native_inference_seconds = sum(
        row["inference_calls"]["primary_provider_window_inference_seconds"]
        for row in native_inference_rows
    )
    total_tool_seconds = sum(
        row["inference_calls"]["total_tool_seconds"] for row in rows
    )
    total_all_models_inference_seconds = sum(
        row["inference_calls"][
            "total_all_models_provider_window_inference_seconds"
        ]
        for row in rows
    )
    total_unattributed_seconds = sum(
        row["inference_calls"]["total_unattributed_seconds"] for row in rows
    )
    inference_tool_concurrency_seconds = sum(
        row["inference_calls"]["inference_tool_concurrency_seconds"]
        for row in rows
    )
    server_aggregate_rows = [
        row for row in rows
        if row.get("server_aggregate_inference", {}).get("coverage") == "complete"
    ]
    server_aggregate_tokens = sum(
        row["server_aggregate_inference"]["output_tokens"]
        for row in server_aggregate_rows
    )
    server_aggregate_seconds = sum(
        row["server_aggregate_inference"]["engine_inference_seconds"]
        for row in server_aggregate_rows
    )
    server_aggregate_coverage = (
        "complete"
        if server_aggregate_rows and len(server_aggregate_rows) == len(rows)
        else "partial"
    )
    active_rows = [
        row for row in eligible
        if row.get("otel", {}).get("scrutable_active_generation_seconds")
    ]
    active_tokens = sum(row["usage"]["output_tokens"] for row in active_rows)
    active_seconds = sum(
        row["otel"]["scrutable_active_generation_seconds"] for row in active_rows
    )
    summary = {
        "schema_version": "native-codex-5",
        "timing_mode": args.timing_mode,
        "runs": rows,
        "aggregate_ratio_of_sums": {
            "output_tokens": output_sum,
            "agent_wall_s": agent_seconds,
            "agent_end_to_end_output_tps": output_sum / agent_seconds if agent_seconds else None,
            "server_aggregate_inference_eligible_runs": len(
                server_aggregate_rows
            ),
            "server_aggregate_inference_output_tokens": (
                server_aggregate_tokens
            ),
            "server_aggregate_engine_inference_seconds": (
                server_aggregate_seconds
            ),
            "server_aggregate_inference_output_tps": (
                server_aggregate_tokens / server_aggregate_seconds
                if server_aggregate_seconds else None
            ),
            "server_aggregate_inference_coverage": server_aggregate_coverage,
            # Compatibility aliases for pre-timing-mode consumers.
            "codex_reported_inference_eligible_runs": len(
                server_aggregate_rows
            ),
            "codex_reported_inference_output_tokens": server_aggregate_tokens,
            "codex_reported_inference_seconds": server_aggregate_seconds,
            "codex_reported_inference_output_tps": (
                server_aggregate_tokens / server_aggregate_seconds
                if server_aggregate_seconds else None
            ),
            "native_inference_eligible_runs": len(native_inference_rows),
            "native_inference_output_tokens": native_inference_tokens,
            "native_provider_window_inference_seconds": native_inference_seconds,
            # Compatibility alias for schema 4 consumers.
            "native_client_active_inference_seconds": native_inference_seconds,
            "native_engine_inference_seconds": None,
            "native_inference_output_tps": (
                native_inference_tokens / native_inference_seconds
                if native_inference_seconds else None
            ),
            "native_provider_window_output_tps": (
                native_inference_tokens / native_inference_seconds
                if native_inference_seconds else None
            ),
            "native_inference_coverage": (
                "complete"
                if native_inference_rows
                and len(native_inference_rows) == len(rows)
                and all(
                    row["inference_calls"]["primary_coverage"] == "complete"
                    for row in native_inference_rows
                )
                else "partial"
            ),
            "total_tool_seconds": total_tool_seconds,
            "inference_tool_concurrency_seconds": (
                inference_tool_concurrency_seconds
            ),
            "total_all_models_client_active_inference_seconds": (
                total_all_models_inference_seconds
            ),
            "total_all_models_provider_window_inference_seconds": (
                total_all_models_inference_seconds
            ),
            "total_unattributed_seconds": total_unattributed_seconds,
            "native_timing_basis": (
                "first_provider_lifecycle_event_to_response_completed"
            ),
            "native_server_engine_equivalent": False,
            "active_generation_eligible_runs": len(active_rows),
            "active_generation_output_tokens": active_tokens,
            "active_generation_seconds": active_seconds,
            "active_generation_output_tps": (
                active_tokens / active_seconds if active_seconds else None
            ),
            "active_generation_eligibility": (
                "eligible" if active_rows and len(active_rows) == len(eligible)
                else "requires_scrutable reasoning-item boundary in every included internal call"
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    csv_path = args.output_dir / "results.csv"
    fields = [
        "problem_number", "problem_id", "model", "reasoning_effort", "passed",
        "agent_wall_s", "first_codex_item_s", "output_tokens",
        "reasoning_output_tokens", "agent_end_to_end_output_tps",
        "native_inference_output_tps", "native_provider_window_output_tps",
        "total_inference_seconds", "total_all_models_inference_seconds",
        "total_tool_seconds",
        "inference_tool_concurrency_seconds",
        "total_unattributed_seconds",
        "server_aggregate_inference_output_tps",
        "codex_reported_inference_output_tps",
        "sse_event_window_output_tps_diagnostic", "active_generation_output_tps",
        "active_generation_eligibility", "solution_path",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in rows:
            writer.writerow(
                {
                    **{field: result.get(field) for field in fields},
                    "output_tokens": result["usage"].get("output_tokens"),
                    "reasoning_output_tokens": result["usage"].get("reasoning_output_tokens"),
                }
            )
    coverage_field = (
        "native_inference_coverage"
        if args.timing_mode == "micro"
        else "server_aggregate_inference_coverage"
    )
    timing_complete = (
        summary["aggregate_ratio_of_sums"][coverage_field] == "complete"
    )
    if not timing_complete and not args.allow_partial_inference_timing:
        hint = (
            "Inspect inference-calls.jsonl; use "
            "--allow-partial-inference-timing only for diagnostics."
            if args.timing_mode == "micro"
            else "Inspect server_aggregate_inference.coverage_reasons and "
            "otel-events.jsonl; use --allow-partial-inference-timing only "
            "for diagnostics."
        )
        print(
            json.dumps(
                {
                    "status": "inference_timing_incomplete",
                    "summary_path": str(args.output_dir / "summary.json"),
                    "timing_mode": args.timing_mode,
                    "hint": hint,
                }
            ),
            flush=True,
        )
        return 3
    return 0 if all(row["passed"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
