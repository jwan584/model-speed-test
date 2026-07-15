#!/usr/bin/env python3

from __future__ import annotations

import json
import unittest
import urllib.request

from claude_inference_timing import (
    build_otel_timing_records,
    summarize_otel_inference_timing,
)
from claude_otel_receiver import ClaudeOtelTraceReceiver


BASE_NS = 1_800_000_000_000_000_000


def otel_value(value):
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


def span(name: str, start_s: float, end_s: float, **attributes):
    return {
        "traceId": "01" * 16,
        "spanId": f"{int(start_s * 1000):016x}",
        "name": name,
        "startTimeUnixNano": str(BASE_NS + int(start_s * 1_000_000_000)),
        "endTimeUnixNano": str(BASE_NS + int(end_s * 1_000_000_000)),
        "attributes": [
            {"key": key, "value": otel_value(value)}
            for key, value in attributes.items()
        ],
        "status": {"code": 0},
    }


def payload(*spans):
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": []},
                "scopeSpans": [{"scope": {"name": "claude-code"}, "spans": list(spans)}],
            }
        ]
    }


class ClaudeOtelReceiverTests(unittest.TestCase):
    def test_loopback_receiver_and_safe_environment(self):
        receiver = ClaudeOtelTraceReceiver()
        receiver.start()
        try:
            environment = receiver.telemetry_environment(
                {
                    "PATH": "/bin",
                    "OTEL_LOG_USER_PROMPTS": "1",
                    "OTEL_LOG_TOOL_DETAILS": "1",
                }
            )
            self.assertEqual(environment["CLAUDE_CODE_ENABLE_TELEMETRY"], "1")
            self.assertEqual(environment["OTEL_TRACES_EXPORTER"], "otlp")
            self.assertNotIn("OTEL_LOG_USER_PROMPTS", environment)
            self.assertNotIn("OTEL_LOG_TOOL_DETAILS", environment)

            body = json.dumps(payload()).encode()
            request = urllib.request.Request(
                receiver.endpoint,
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                self.assertEqual(response.status, 200)
            receiver.wait_for_quiet(quiet_seconds=0.01, timeout=1)
            self.assertEqual(len(receiver.payloads()), 1)
            self.assertEqual(receiver.diagnostics()["receiver_error_count"], 0)
        finally:
            receiver.stop()


class ClaudeOtelAccountingTests(unittest.TestCase):
    def test_terminal_all_model_headline_survives_target_otel_gap(self):
        trace_payload = payload(
            span(
                "claude_code.llm_request",
                1,
                2,
                model="claude-fable-5",
                duration_ms=1000,
                success=True,
                output_tokens=100,
            )
        )
        calls, tools, normalization = build_otel_timing_records(
            [trace_payload],
            {"claude-fable-5"},
            BASE_NS,
            BASE_NS + 3_000_000_000,
        )
        summary = summarize_otel_inference_timing(
            {
                "duration_api_ms": 2000,
                "is_error": False,
                "modelUsage": {
                    "claude-fable-5": {"outputTokens": 120},
                    "claude-haiku-4-5": {"outputTokens": 10},
                },
            },
            calls,
            tools,
            "claude-fable-5",
            BASE_NS,
            BASE_NS + 3_000_000_000,
            {"payload_count": 1, "receiver_error_count": 0, **normalization},
        )
        self.assertEqual(summary["coverage"], "complete")
        self.assertEqual(summary["end_to_end_billed_tps"], 65.0)
        self.assertEqual(summary["output_token_reconciliation"], "mismatched")
        self.assertEqual(summary["target_otel_diagnostic_coverage"], "partial")

    def test_target_tps_and_overlap_safe_wall_partition(self):
        trace_payload = payload(
            span(
                "claude_code.llm_request",
                1,
                4,
                model="claude-sonnet-5",
                duration_ms=3000,
                ttft_ms=500,
                success=True,
                output_tokens=300,
                input_tokens=1000,
                cache_read_tokens=800,
                cache_creation_tokens=100,
                query_source="repl_main_thread",
            ),
            span(
                "claude_code.llm_request",
                2,
                3,
                model="claude-haiku-4-5",
                duration_ms=1000,
                success=True,
                output_tokens=10,
                input_tokens=20,
                query_source="compact",
            ),
            span(
                "claude_code.llm_request",
                5,
                7,
                model="claude-sonnet-5",
                duration_ms=2000,
                ttft_ms=300,
                success=True,
                output_tokens=200,
                input_tokens=1100,
                cache_read_tokens=900,
                cache_creation_tokens=0,
                query_source="repl_main_thread",
            ),
            span(
                "claude_code.tool",
                3,
                6,
                tool_name="Bash",
                duration_ms=3000,
            ),
            span(
                "claude_code.tool",
                5,
                8,
                tool_name="Read",
                duration_ms=3000,
            ),
        )
        calls, tools, normalization = build_otel_timing_records(
            [trace_payload],
            {"claude-sonnet-5"},
            BASE_NS,
            BASE_NS + 10_000_000_000,
        )
        diagnostics = {
            "payload_count": 1,
            "receiver_error_count": 0,
            **normalization,
        }
        result_event = {
            "duration_api_ms": 6000,
            "modelUsage": {
                "claude-sonnet-5": {"outputTokens": 500},
                "claude-haiku-4-5": {"outputTokens": 10},
            },
        }
        summary = summarize_otel_inference_timing(
            result_event,
            calls,
            tools,
            "claude-sonnet-5",
            BASE_NS,
            BASE_NS + 10_000_000_000,
            diagnostics,
        )

        self.assertEqual(summary["coverage"], "complete")
        self.assertEqual(summary["primary_call_count"], 2)
        self.assertEqual(summary["auxiliary_call_count"], 1)
        self.assertEqual(summary["primary_output_tokens"], 500)
        self.assertEqual(summary["primary_request_seconds_sum"], 5.0)
        self.assertEqual(summary["primary_request_output_tps"], 100.0)
        self.assertEqual(summary["end_to_end_billed_tps"], 85.0)
        self.assertEqual(summary["all_model_terminal_billed_output_tokens"], 510)
        self.assertEqual(summary["terminal_request_active_seconds"], 6.0)
        self.assertEqual(summary["target_otel_request_output_tps_diagnostic"], 100.0)
        self.assertEqual(summary["provider_reported_ttft_ms_median"], 400.0)
        self.assertEqual(summary["primary_request_union_seconds"], 5.0)
        self.assertEqual(summary["total_tool_seconds"], 5.0)
        self.assertEqual(summary["all_model_request_tool_concurrency_seconds"], 3.0)
        self.assertEqual(
            summary["wall_partition"],
            {
                "llm_request_only_seconds": 2.0,
                "tool_only_seconds": 2.0,
                "llm_request_tool_overlap_seconds": 3.0,
                "orchestration_residual_seconds": 3.0,
                "reconciliation_error_seconds": 0.0,
            },
        )

    def test_invisible_auxiliary_usage_keeps_terminal_headline_complete(self):
        trace_payload = payload(
            span(
                "claude_code.llm_request",
                1,
                2,
                model="claude-sonnet-5",
                duration_ms=1000,
                success=True,
                output_tokens=100,
            )
        )
        calls, tools, normalization = build_otel_timing_records(
            [trace_payload],
            {"claude-sonnet-5"},
            BASE_NS,
            BASE_NS + 3_000_000_000,
        )
        summary = summarize_otel_inference_timing(
            {
                "duration_api_ms": 2000,
                "modelUsage": {
                    "claude-sonnet-5": {"outputTokens": 100},
                    "claude-haiku-4-5": {"outputTokens": 4},
                }
            },
            calls,
            tools,
            "claude-sonnet-5",
            BASE_NS,
            BASE_NS + 3_000_000_000,
            {
                "payload_count": 1,
                "receiver_error_count": 0,
                **normalization,
            },
        )
        self.assertEqual(summary["coverage"], "complete")
        self.assertEqual(summary["end_to_end_billed_tps"], 52.0)
        self.assertEqual(summary["target_otel_diagnostic_coverage"], "partial")
        self.assertIn(
            "model_usage_model_missing_from_otel_spans",
            summary["target_otel_diagnostic_coverage_reasons"],
        )

    def test_failed_target_call_is_excluded_and_marks_coverage_partial(self):
        trace_payload = payload(
            span(
                "claude_code.llm_request",
                1,
                2,
                model="claude-sonnet-5",
                duration_ms=1000,
                success=False,
                output_tokens=25,
            ),
            span(
                "claude_code.llm_request",
                2,
                4,
                model="claude-sonnet-5",
                duration_ms=2000,
                success=True,
                output_tokens=100,
            ),
        )
        calls, tools, normalization = build_otel_timing_records(
            [trace_payload],
            {"claude-sonnet-5"},
            BASE_NS,
            BASE_NS + 5_000_000_000,
        )
        summary = summarize_otel_inference_timing(
            {
                "duration_api_ms": 2500,
                "modelUsage": {"claude-sonnet-5": {"outputTokens": 100}},
            },
            calls,
            tools,
            "claude-sonnet-5",
            BASE_NS,
            BASE_NS + 5_000_000_000,
            {"payload_count": 1, "receiver_error_count": 0, **normalization},
        )
        self.assertFalse(calls[0]["included_in_primary"])
        self.assertTrue(calls[1]["included_in_primary"])
        self.assertEqual(summary["primary_output_tokens"], 100)
        self.assertEqual(summary["end_to_end_inference_seconds"], 2.5)
        self.assertEqual(summary["coverage"], "complete")
        self.assertEqual(summary["target_otel_diagnostic_coverage"], "partial")


if __name__ == "__main__":
    unittest.main()
