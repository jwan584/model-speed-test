#!/usr/bin/env python3
"""Per-request Claude Code LLM and tool accounting for SWE-bench runs.

Claude Code's official OpenTelemetry trace surface emits one
``claude_code.llm_request`` span per model request and one
``claude_code.tool`` span per tool invocation.  This module normalizes those
spans, separates target and auxiliary models, and computes overlap-safe wall
time partitions.  Stream JSON and its terminal ``duration_api_ms`` remain as
diagnostic fallbacks only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TOOL_EVENT_TYPES = {"tool_use", "tool_result"}


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


def _otel_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return None
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "intValue" in value:
        return integer(value["intValue"])
    if "doubleValue" in value:
        return numeric(value["doubleValue"])
    if "arrayValue" in value:
        array = value["arrayValue"] or {}
        return [_otel_value(item) for item in array.get("values", [])]
    if "kvlistValue" in value:
        entries = (value["kvlistValue"] or {}).get("values", [])
        return {
            str(entry.get("key")): _otel_value(entry.get("value"))
            for entry in entries
            if isinstance(entry, dict) and entry.get("key") is not None
        }
    return None


def _otel_attributes(entries: Any) -> dict[str, Any]:
    if not isinstance(entries, list):
        return {}
    return {
        str(entry["key"]): _otel_value(entry.get("value"))
        for entry in entries
        if isinstance(entry, dict) and entry.get("key") is not None
    }


def _iter_otel_spans(payloads: list[dict[str, Any]]):
    for payload in payloads:
        resource_spans = payload.get("resourceSpans") or payload.get(
            "resource_spans", []
        )
        for resource_span in resource_spans:
            resource_attributes = _otel_attributes(
                (resource_span.get("resource") or {}).get("attributes")
            )
            scope_spans = resource_span.get("scopeSpans") or resource_span.get(
                "scope_spans", []
            )
            for scope_span in scope_spans:
                for span in scope_span.get("spans", []):
                    if isinstance(span, dict):
                        yield span, resource_attributes


def build_otel_timing_records(
    payloads: list[dict[str, Any]],
    target_models: set[str],
    run_start_ns: int,
    run_end_ns: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Normalize safe timing/token fields from Claude Code OTel spans."""
    calls: list[dict[str, Any]] = []
    tools: list[dict[str, Any]] = []
    total_span_count = 0
    invalid_timing_span_count = 0
    relevant_span_count = 0

    for span, _resource_attributes in _iter_otel_spans(payloads):
        total_span_count += 1
        name = span.get("name")
        if name not in {"claude_code.llm_request", "claude_code.tool"}:
            continue
        relevant_span_count += 1
        attributes = _otel_attributes(span.get("attributes"))
        start_ns = integer(
            span.get("startTimeUnixNano") or span.get("start_time_unix_nano")
        )
        end_ns = integer(
            span.get("endTimeUnixNano") or span.get("end_time_unix_nano")
        )
        if start_ns is None or end_ns is None or end_ns <= start_ns:
            invalid_timing_span_count += 1
            continue
        clipped_start = max(run_start_ns, start_ns)
        clipped_end = min(run_end_ns, end_ns)
        if clipped_end <= clipped_start:
            invalid_timing_span_count += 1
            continue
        duration_seconds = (clipped_end - clipped_start) / 1_000_000_000
        status = span.get("status") or {}

        if name == "claude_code.llm_request":
            model = attributes.get("model") or attributes.get(
                "gen_ai.request.model"
            )
            output_tokens = integer(attributes.get("output_tokens"))
            calls.append(
                {
                    "schema_version": "claude-swebench-otel-llm-request-1",
                    "span_id": span.get("spanId") or span.get("span_id"),
                    "parent_span_id": span.get("parentSpanId")
                    or span.get("parent_span_id"),
                    "request_id": attributes.get("request_id")
                    or attributes.get("gen_ai.response.id"),
                    "model": model,
                    "is_target_model": model in target_models,
                    "start_unix_ns": clipped_start,
                    "end_unix_ns": clipped_end,
                    "request_seconds": duration_seconds,
                    "reported_duration_ms": numeric(attributes.get("duration_ms")),
                    "ttft_ms": numeric(attributes.get("ttft_ms")),
                    "input_tokens": integer(attributes.get("input_tokens")),
                    "output_tokens": output_tokens,
                    "cache_read_input_tokens": integer(
                        attributes.get("cache_read_tokens")
                    ),
                    "cache_creation_input_tokens": integer(
                        attributes.get("cache_creation_tokens")
                    ),
                    "request_output_tps": (
                        output_tokens / duration_seconds
                        if output_tokens is not None and duration_seconds > 0
                        else None
                    ),
                    "query_source": attributes.get("query_source"),
                    "agent_id": attributes.get("agent_id"),
                    "parent_agent_id": attributes.get("parent_agent_id"),
                    "request_context": attributes.get("llm_request.context"),
                    "speed": attributes.get("speed"),
                    "effort": attributes.get("effort"),
                    "attempt": integer(attributes.get("attempt")),
                    "success": attributes.get("success"),
                    "has_tool_call": attributes.get("response.has_tool_call"),
                    "stop_reason": attributes.get("stop_reason"),
                    "status_code": status.get("code"),
                }
            )
        else:
            tools.append(
                {
                    "schema_version": "claude-swebench-otel-tool-1",
                    "span_id": span.get("spanId") or span.get("span_id"),
                    "parent_span_id": span.get("parentSpanId")
                    or span.get("parent_span_id"),
                    "tool_name": attributes.get("tool_name"),
                    "start_unix_ns": clipped_start,
                    "end_unix_ns": clipped_end,
                    "duration_seconds": duration_seconds,
                    "reported_duration_ms": numeric(attributes.get("duration_ms")),
                    "result_tokens": integer(attributes.get("result_tokens")),
                    "agent_id": attributes.get("agent_id"),
                    "parent_agent_id": attributes.get("parent_agent_id"),
                    "status_code": status.get("code"),
                }
            )

    calls.sort(key=lambda call: (call["start_unix_ns"], call["end_unix_ns"]))
    tools.sort(key=lambda tool: (tool["start_unix_ns"], tool["end_unix_ns"]))
    for call_index, call in enumerate(calls, start=1):
        call["call_index"] = call_index
    return calls, tools, {
        "total_span_count": total_span_count,
        "relevant_span_count": relevant_span_count,
        "llm_request_span_count": len(calls),
        "tool_span_count": len(tools),
        "invalid_timing_span_count": invalid_timing_span_count,
        "target_model_aliases": sorted(target_models),
    }


def summarize_otel_inference_timing(
    result_event: dict[str, Any] | None,
    inference_calls: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    target_model: str,
    run_start_ns: int,
    run_end_ns: int,
    otel_diagnostics: dict[str, Any],
    *,
    tool_timing_basis: str = "claude_code_otel_tool_spans",
) -> dict[str, Any]:
    """Compute request/tool wall partitions and ratio-of-sums target TPS."""
    wall_ns = max(0, run_end_ns - run_start_ns)
    all_request_intervals = [
        (call["start_unix_ns"], call["end_unix_ns"])
        for call in inference_calls
    ]
    target_calls = [call for call in inference_calls if call["is_target_model"]]
    auxiliary_calls = [
        call
        for call in inference_calls
        if not call["is_target_model"] and call.get("model")
    ]
    target_intervals = [
        (call["start_unix_ns"], call["end_unix_ns"])
        for call in target_calls
    ]
    tool_intervals = [
        (tool["start_unix_ns"], tool["end_unix_ns"]) for tool in tools
    ]
    all_request_union = union_intervals(all_request_intervals)
    target_request_union = union_intervals(target_intervals)
    tool_union = union_intervals(tool_intervals)
    all_request_union_ns = interval_duration_ns(all_request_union)
    target_request_union_ns = interval_duration_ns(target_request_union)
    tool_union_ns = interval_duration_ns(tool_union)
    target_overlap_ns = sum(
        interval_overlap_ns(interval, tool_union)
        for interval in target_request_union
    )
    all_overlap_ns = sum(
        interval_overlap_ns(interval, tool_union)
        for interval in all_request_union
    )
    accounted_union_ns = interval_duration_ns(all_request_union + tool_union)
    residual_ns = max(0, wall_ns - accounted_union_ns)

    target_request_seconds_sum = sum(
        float(call["request_seconds"]) for call in target_calls
    )
    target_output_tokens = sum(
        int(call["output_tokens"] or 0) for call in target_calls
    )
    target_input_tokens = sum(
        int(call["input_tokens"] or 0) for call in target_calls
    )
    target_cache_read_tokens = sum(
        int(call["cache_read_input_tokens"] or 0) for call in target_calls
    )
    target_cache_creation_tokens = sum(
        int(call["cache_creation_input_tokens"] or 0) for call in target_calls
    )

    cli_reported = None
    model_usage_breakdown: dict[str, Any] = {}
    target_usage: dict[str, Any] | None = None
    coverage_reasons: list[str] = []
    if result_event is None:
        coverage_reasons.append("missing_result_event")
    else:
        cli_reported = {
            "duration_ms": result_event.get("duration_ms"),
            "duration_api_ms": result_event.get("duration_api_ms"),
            "ttft_ms": result_event.get("ttft_ms"),
            "ttft_stream_ms": result_event.get("ttft_stream_ms"),
            "time_to_request_ms": result_event.get("time_to_request_ms"),
            "num_turns": result_event.get("num_turns"),
            "total_cost_usd": result_event.get("total_cost_usd"),
            "is_error": result_event.get("is_error"),
            "subtype": result_event.get("subtype"),
            "terminal_reason": result_event.get("terminal_reason"),
            "stop_reason": result_event.get("stop_reason"),
        }
        model_usage_breakdown = result_event.get("modelUsage") or {}
        target_usage_models = [target_model]
        target_usage_models.extend(
            str(call["model"])
            for call in target_calls
            if call.get("model") and call["model"] != target_model
        )
        target_usage = next(
            (
                model_usage_breakdown[model]
                for model in target_usage_models
                if model in model_usage_breakdown
            ),
            None,
        )

    reported_target_output = (
        integer(target_usage.get("outputTokens"))
        if target_usage is not None
        else None
    )
    output_reconciliation = (
        "unavailable"
        if reported_target_output is None
        else "matched"
        if reported_target_output == target_output_tokens
        else "mismatched"
    )
    if not target_calls:
        coverage_reasons.append("no_target_model_otel_spans")
    if target_usage is None:
        coverage_reasons.append("target_model_missing_from_model_usage")
    if output_reconciliation != "matched":
        coverage_reasons.append("target_output_tokens_not_reconciled")
    observed_models = {
        str(call["model"]) for call in inference_calls if call.get("model")
    }
    missing_usage_models = sorted(set(model_usage_breakdown) - observed_models)
    if missing_usage_models:
        coverage_reasons.append("model_usage_model_missing_from_otel_spans")
    if otel_diagnostics.get("payload_count", 0) == 0:
        coverage_reasons.append("no_otel_trace_payloads")
    if otel_diagnostics.get("receiver_error_count", 0):
        coverage_reasons.append("otel_receiver_error")
    if otel_diagnostics.get("invalid_timing_span_count", 0):
        coverage_reasons.append("invalid_otel_span_timing")
    if tool_timing_basis != "claude_code_otel_tool_spans" and tools:
        coverage_reasons.append("otel_tool_spans_unavailable_host_fallback")

    api_seconds = (
        numeric(cli_reported.get("duration_api_ms")) / 1000
        if cli_reported and numeric(cli_reported.get("duration_api_ms")) is not None
        else None
    )
    llm_only_ns = max(0, all_request_union_ns - all_overlap_ns)
    tool_only_ns = max(0, tool_union_ns - all_overlap_ns)
    decomposition_error_ns = wall_ns - (
        llm_only_ns + tool_only_ns + all_overlap_ns + residual_ns
    )

    return {
        "schema_version": "claude-swebench-inference-accounting-2",
        "primary_model": target_model,
        "timing_basis": "claude_code_otel_llm_request_spans",
        "tool_timing_basis": tool_timing_basis,
        "server_engine_equivalent": False,
        "primary_call_count": len(target_calls),
        "auxiliary_call_count": len(auxiliary_calls),
        "auxiliary_models": sorted(
            {str(call["model"]) for call in auxiliary_calls}
        ),
        "primary_output_tokens": target_output_tokens,
        "primary_input_tokens": target_input_tokens,
        "primary_cache_read_input_tokens": target_cache_read_tokens,
        "primary_cache_creation_input_tokens": target_cache_creation_tokens,
        "model_usage_primary_output_tokens": reported_target_output,
        "output_token_reconciliation": output_reconciliation,
        "primary_request_seconds_sum": target_request_seconds_sum,
        "primary_request_output_tps": (
            target_output_tokens / target_request_seconds_sum
            if target_request_seconds_sum > 0
            else None
        ),
        "primary_request_union_seconds": target_request_union_ns / 1_000_000_000,
        "all_model_request_union_seconds": all_request_union_ns / 1_000_000_000,
        "primary_request_tool_concurrency_seconds": target_overlap_ns
        / 1_000_000_000,
        "all_model_request_tool_concurrency_seconds": all_overlap_ns
        / 1_000_000_000,
        "total_tool_seconds": tool_union_ns / 1_000_000_000,
        "total_unattributed_seconds": residual_ns / 1_000_000_000,
        "total_wall_seconds": wall_ns / 1_000_000_000,
        "wall_partition": {
            "llm_request_only_seconds": llm_only_ns / 1_000_000_000,
            "tool_only_seconds": tool_only_ns / 1_000_000_000,
            "llm_request_tool_overlap_seconds": all_overlap_ns / 1_000_000_000,
            "orchestration_residual_seconds": residual_ns / 1_000_000_000,
            "reconciliation_error_seconds": decomposition_error_ns
            / 1_000_000_000,
        },
        "cli_reported": cli_reported,
        "model_usage_breakdown": model_usage_breakdown,
        "cli_reported_api_seconds_diagnostic": api_seconds,
        "cli_reported_output_tps_diagnostic": (
            target_output_tokens / api_seconds if api_seconds else None
        ),
        "otel": {
            **otel_diagnostics,
            "observed_models": sorted(observed_models),
            "model_usage_models_missing_from_spans": missing_usage_models,
            "raw_payloads_persisted": False,
        },
        "coverage": "complete" if not coverage_reasons else "partial",
        "coverage_reasons": coverage_reasons,
        "note": (
            "primary_request_seconds_sum is the sum of target-model Claude "
            "Code llm_request span durations and is the TPS denominator. "
            "Wall contribution fields use interval unions so parallel calls "
            "are not double-counted. Durations include client/API latency and "
            "retries; they are not server-engine decode time. The CLI terminal "
            "duration_api_ms value is retained only as a diagnostic."
        ),
    }


def _content_blocks(message: dict[str, Any], block_type: str) -> list[dict[str, Any]]:
    content = message.get("content") or []
    if not isinstance(content, list):
        return []
    return [
        block
        for block in content
        if isinstance(block, dict) and block.get("type") == block_type
    ]


def build_timeline(
    stream_events: list[dict[str, Any]], run_start_ns: int
) -> list[dict[str, Any]]:
    """Reduce raw stream-json events into a host-timestamped timeline.

    Each raw event must carry the harness-assigned `_received_unix_ns` (the
    moment this process read the line from the child's stdout); Claude Code
    itself does not stamp assistant messages with wall-clock time.
    """
    timeline = []
    for event in stream_events:
        received_ns = event.get("_received_unix_ns")
        if received_ns is None:
            continue
        elapsed_seconds = round((received_ns - run_start_ns) / 1_000_000_000, 6)
        event_type = event.get("type")
        base = {
            "received_unix_ns": received_ns,
            "elapsed_seconds": elapsed_seconds,
        }
        if event_type == "assistant":
            message = event.get("message", {}) or {}
            timeline.append(
                {
                    **base,
                    "type": "assistant_message",
                    "item_id": message.get("id"),
                    "model": message.get("model"),
                }
            )
            for tool_use in _content_blocks(message, "tool_use"):
                timeline.append(
                    {
                        **base,
                        "type": "tool_use",
                        "item_id": tool_use.get("id"),
                        "tool_name": tool_use.get("name"),
                    }
                )
        elif event_type == "user":
            message = event.get("message", {}) or {}
            for tool_result in _content_blocks(message, "tool_result"):
                timeline.append(
                    {
                        **base,
                        "type": "tool_result",
                        "item_id": tool_result.get("tool_use_id"),
                        "is_error": tool_result.get("is_error"),
                    }
                )
        elif event_type == "result":
            timeline.append({**base, "type": "result", "item_id": None})
    return timeline


def pair_tool_intervals(
    timeline: list[dict[str, Any]],
    run_start_ns: int,
    run_end_ns: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Pair host-observed tool_use/tool_result timeline entries into intervals."""
    starts: dict[str, dict[str, Any]] = {}
    tools = []
    completed_without_start = 0
    for record in timeline:
        item_id = record.get("item_id")
        if not item_id or record["type"] not in TOOL_EVENT_TYPES:
            continue
        if record["type"] == "tool_use":
            starts[str(item_id)] = record
            continue
        start = starts.pop(str(item_id), None)
        if start is None:
            completed_without_start += 1
            continue
        start_ns = start.get("received_unix_ns")
        end_ns = record.get("received_unix_ns")
        if start_ns is None or end_ns is None or end_ns <= start_ns:
            continue
        clipped_start = max(run_start_ns, start_ns)
        clipped_end = min(run_end_ns, end_ns)
        if clipped_end <= clipped_start:
            continue
        tools.append(
            {
                "item_id": str(item_id),
                "tool_name": start.get("tool_name"),
                "start_unix_ns": clipped_start,
                "end_unix_ns": clipped_end,
                "duration_seconds": (clipped_end - clipped_start) / 1_000_000_000,
                "is_error": record.get("is_error"),
            }
        )
    return tools, {
        "completed_interval_count": len(tools),
        "unmatched_started_count": len(starts),
        "completed_without_start_count": completed_without_start,
    }


def build_inference_calls(
    stream_events: list[dict[str, Any]],
    target_model: str,
    run_start_ns: int,
) -> list[dict[str, Any]]:
    """One entry per assistant message (API turn): the token usage Claude Code reports.

    A single turn can arrive as several `assistant` stream events -- e.g. a
    "thinking" content block and a "tool_use" content block land as separate
    lines sharing one message `id` -- and each repeats the same `usage`
    snapshot rather than an incremental delta. Grouping by message id (instead
    of summing every raw event) avoids double-counting tokens per turn.
    """
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for event in stream_events:
        if event.get("type") != "assistant":
            continue
        message = event.get("message", {}) or {}
        message_id = message.get("id")
        if message_id is None:
            continue
        if message_id not in grouped:
            order.append(message_id)
            grouped[message_id] = {
                "usage": None,
                "model": None,
                "stop_reason": None,
                "request_id": None,
                "has_text": False,
                "has_tool_use": False,
                "chunk_count": 0,
                "last_received_unix_ns": None,
            }
        bucket = grouped[message_id]
        bucket["chunk_count"] += 1
        bucket["usage"] = message.get("usage") or bucket["usage"]
        bucket["model"] = message.get("model") or bucket["model"]
        bucket["stop_reason"] = message.get("stop_reason") or bucket["stop_reason"]
        bucket["request_id"] = event.get("request_id") or bucket["request_id"]
        bucket["has_text"] = bucket["has_text"] or bool(_content_blocks(message, "text"))
        bucket["has_tool_use"] = bucket["has_tool_use"] or bool(
            _content_blocks(message, "tool_use")
        )
        bucket["last_received_unix_ns"] = event.get("_received_unix_ns")

    calls = []
    for call_index, message_id in enumerate(order, start=1):
        bucket = grouped[message_id]
        usage = bucket["usage"] or {}
        received_ns = bucket["last_received_unix_ns"]
        calls.append(
            {
                "schema_version": "claude-swebench-inference-call-1",
                "call_index": call_index,
                "message_id": message_id,
                "request_id": bucket["request_id"],
                "model": bucket["model"],
                "is_target_model": bucket["model"] == target_model,
                "chunk_count": bucket["chunk_count"],
                "elapsed_seconds_at_arrival": (
                    round((received_ns - run_start_ns) / 1_000_000_000, 6)
                    if received_ns is not None
                    else None
                ),
                "input_tokens": integer(usage.get("input_tokens")),
                "output_tokens": integer(usage.get("output_tokens")),
                "cache_read_input_tokens": integer(usage.get("cache_read_input_tokens")),
                "cache_creation_input_tokens": integer(
                    usage.get("cache_creation_input_tokens")
                ),
                "service_tier": usage.get("service_tier"),
                "stop_reason": bucket["stop_reason"],
                "has_text": bucket["has_text"],
                "has_tool_use": bucket["has_tool_use"],
            }
        )
    return calls


def summarize_inference_timing(
    result_event: dict[str, Any] | None,
    inference_calls: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_pairing: dict[str, int],
    target_model: str,
    run_start_ns: int,
    run_end_ns: int,
) -> dict[str, Any]:
    """Combine the CLI's self-reported result-event timing with host-observed tool time."""
    tool_union = union_intervals(
        [(tool["start_unix_ns"], tool["end_unix_ns"]) for tool in tools]
    )
    tool_seconds = interval_duration_ns(tool_union) / 1_000_000_000
    wall_seconds = max(0, run_end_ns - run_start_ns) / 1_000_000_000

    target_calls = [call for call in inference_calls if call["is_target_model"]]
    auxiliary_calls = [
        call
        for call in inference_calls
        if not call["is_target_model"] and call.get("model")
    ]
    # Best-effort per-turn attribution. Do NOT treat these sums as the
    # authoritative token totals: per-message `usage.output_tokens` does not
    # reliably include extended-thinking token cost the way the terminal
    # result event's `modelUsage` aggregate does, so this can undercount by
    # an order of magnitude on reasoning-heavy (e.g. high/xhigh effort) turns.
    per_call_sum_output_tokens = sum(call["output_tokens"] or 0 for call in target_calls)
    per_call_sum_input_tokens = sum(call["input_tokens"] or 0 for call in target_calls)
    per_call_sum_cache_read_tokens = sum(
        call["cache_read_input_tokens"] or 0 for call in target_calls
    )
    per_call_sum_cache_creation_tokens = sum(
        call["cache_creation_input_tokens"] or 0 for call in target_calls
    )

    coverage_reasons = []
    cli_reported = None
    model_usage_breakdown: dict[str, Any] = {}
    output_token_reconciliation = "unavailable"
    target_usage: dict[str, Any] | None = None
    if result_event is not None:
        cli_reported = {
            "duration_ms": result_event.get("duration_ms"),
            "duration_api_ms": result_event.get("duration_api_ms"),
            "ttft_ms": result_event.get("ttft_ms"),
            "ttft_stream_ms": result_event.get("ttft_stream_ms"),
            "time_to_request_ms": result_event.get("time_to_request_ms"),
            "num_turns": result_event.get("num_turns"),
            "total_cost_usd": result_event.get("total_cost_usd"),
            "is_error": result_event.get("is_error"),
            "subtype": result_event.get("subtype"),
            "terminal_reason": result_event.get("terminal_reason"),
            "stop_reason": result_event.get("stop_reason"),
        }
        model_usage_breakdown = result_event.get("modelUsage") or {}
        target_usage = model_usage_breakdown.get(target_model)
        if target_usage is not None:
            reported_output = integer(target_usage.get("outputTokens"))
            output_token_reconciliation = (
                "matched"
                if reported_output == per_call_sum_output_tokens
                else "mismatched"
            )
        else:
            coverage_reasons.append("target_model_missing_from_model_usage")
    else:
        coverage_reasons.append("missing_result_event")

    if not target_calls:
        coverage_reasons.append("no_target_model_calls")
    # A per-call-sum vs modelUsage mismatch is expected on reasoning-heavy
    # turns (see output_token_reconciliation) and is not a coverage defect,
    # since primary_output_tokens is sourced from modelUsage either way.

    api_seconds = (
        cli_reported["duration_api_ms"] / 1000
        if cli_reported and cli_reported.get("duration_api_ms") is not None
        else None
    )
    # Authoritative token totals come from modelUsage (the CLI's own final
    # accounting); per-call sums are a diagnostic fallback only.
    primary_output_tokens = (
        integer(target_usage.get("outputTokens"))
        if target_usage is not None
        else per_call_sum_output_tokens
    )
    primary_input_tokens = (
        integer(target_usage.get("inputTokens"))
        if target_usage is not None
        else per_call_sum_input_tokens
    )
    primary_cache_read_tokens = (
        integer(target_usage.get("cacheReadInputTokens"))
        if target_usage is not None
        else per_call_sum_cache_read_tokens
    )
    primary_cache_creation_tokens = (
        integer(target_usage.get("cacheCreationInputTokens"))
        if target_usage is not None
        else per_call_sum_cache_creation_tokens
    )

    return {
        "schema_version": "claude-swebench-inference-accounting-1",
        "primary_model": target_model,
        "timing_basis": "claude_code_cli_result_event",
        "server_engine_equivalent": False,
        "cli_reported": cli_reported,
        "model_usage_breakdown": model_usage_breakdown,
        "primary_call_count": len(target_calls),
        "auxiliary_call_count": len(auxiliary_calls),
        "auxiliary_models": sorted(
            {call["model"] for call in auxiliary_calls if call.get("model")}
        ),
        "primary_output_tokens": primary_output_tokens,
        "primary_input_tokens": primary_input_tokens,
        "primary_cache_read_input_tokens": primary_cache_read_tokens,
        "primary_cache_creation_input_tokens": primary_cache_creation_tokens,
        "primary_output_tokens_per_call_sum": per_call_sum_output_tokens,
        "output_token_reconciliation": output_token_reconciliation,
        "cli_reported_api_seconds": api_seconds,
        "cli_reported_output_tps": (
            primary_output_tokens / api_seconds if api_seconds else None
        ),
        "host_observed_tool_seconds": round(tool_seconds, 6),
        "host_observed_wall_seconds": round(wall_seconds, 6),
        "inference_and_orchestration_seconds_estimate": round(
            max(0.0, wall_seconds - tool_seconds), 6
        ),
        "tool_interval_pairing": tool_pairing,
        "coverage": "complete" if not coverage_reasons else "partial",
        "coverage_reasons": coverage_reasons,
        "note": (
            "primary_output_tokens (and the other primary_* token fields) are "
            "read from modelUsage on the terminal result event -- the CLI's "
            "own authoritative accounting. primary_output_tokens_per_call_sum "
            "is a per-turn diagnostic summed from each assistant message's own "
            "usage field; it reliably undercounts on reasoning-heavy turns "
            "because extended-thinking token cost is not fully reflected "
            "there, so a 'mismatched' reconciliation is expected and does not "
            "indicate a coverage problem with the authoritative totals. "
            "cli_reported_api_seconds is duration_api_ms from the CLI's own "
            "result event: it sums API time across every model call in the "
            "session, including auxiliary/background model calls, and is not "
            "exclusively the primary model's time; it can exceed wall time "
            "when calls overlap. inference_and_orchestration_seconds_estimate "
            "is a legacy wall-minus-tools fallback, not a provider-exact "
            "measurement -- prefer cli_reported for TPS."
        ),
    }
