#!/usr/bin/env python3

import sys
import unittest

from codex_inference_timing import (
    parse_lifecycle_trace_line,
    summarize_lifecycle,
)
from codex_swebench_1_10 import summarize as summarize_batch
from codex_swebench_problem1 import run_with_heartbeat


class CodexInferenceTimingTests(unittest.TestCase):
    def test_parse_lifecycle_trace_preserves_nanoseconds(self):
        line = (
            "TRACE codex_api::responses_websocket_lifecycle: lifecycle "
            'model="gpt-test" session_id="session" thread_id="thread" '
            'turn_id="turn" response_id="response" warmup=false '
            'connection_reused=true provider_start_kind="response.created" '
            'request_sent_unix_ns="1783980000000000001" '
            'provider_event_started_unix_ns="1783980000000000123" '
            'completed_unix_ns="1783980001000000456" '
            "provider_event_window_ms=1000.2 request_to_completed_ms=1100.0 "
            "input_tokens=100 output_tokens=45 reasoning_output_tokens=12"
        )
        parsed = parse_lifecycle_trace_line(line)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(
            parsed["provider_event_started_unix_ns"], 1783980000000000123
        )
        self.assertEqual(parsed["output_tokens"], 45)

    def test_fully_overlapped_call_is_counted(self):
        traces = [
            {
                "model": "gpt-test",
                "warmup": False,
                "provider_start_kind": "response.created",
                "provider_event_started_unix_ns": 2_000_000_000,
                "completed_unix_ns": 3_000_000_000,
                "output_tokens": 45,
            }
        ]
        timeline = [
            {
                "type": "item.started",
                "item_id": "tool-1",
                "item_type": "command_execution",
                "received_unix_ns": 2_000_000_000,
            },
            {
                "type": "item.completed",
                "item_id": "tool-1",
                "item_type": "command_execution",
                "received_unix_ns": 3_000_000_000,
            },
        ]
        calls, summary, tools = summarize_lifecycle(
            traces,
            timeline,
            "gpt-test",
            {"output_tokens": 45},
            1_000_000_000,
            4_000_000_000,
        )
        self.assertTrue(calls[0]["included_in_primary"])
        self.assertEqual(calls[0]["provider_window_inference_seconds"], 1.0)
        self.assertEqual(calls[0]["tool_overlap_seconds"], 1.0)
        self.assertEqual(summary["provider_window_output_tps"], 45.0)
        self.assertEqual(summary["coverage"], "complete")
        self.assertEqual(summary["total_tool_seconds"], 1.0)
        self.assertEqual(len(tools), 1)

    def test_warmup_and_auxiliary_model_are_excluded(self):
        traces = [
            {
                "model": "gpt-test",
                "warmup": True,
                "provider_start_kind": "response.created",
                "provider_event_started_unix_ns": 1_000_000_000,
                "completed_unix_ns": 1_100_000_000,
                "output_tokens": 0,
            },
            {
                "model": "codex-auto-review",
                "warmup": False,
                "provider_start_kind": "response.created",
                "provider_event_started_unix_ns": 2_000_000_000,
                "completed_unix_ns": 3_000_000_000,
                "output_tokens": 20,
            },
            {
                "model": "gpt-test",
                "warmup": False,
                "provider_start_kind": "response.created",
                "provider_event_started_unix_ns": 4_000_000_000,
                "completed_unix_ns": 6_000_000_000,
                "output_tokens": 200,
            },
        ]
        calls, summary, _ = summarize_lifecycle(
            traces,
            [],
            "gpt-test",
            {"output_tokens": 200},
            0,
            7_000_000_000,
        )
        self.assertEqual(calls[0]["primary_exclusion_reason"], "warmup")
        self.assertEqual(calls[1]["primary_exclusion_reason"], "auxiliary_model")
        self.assertEqual(summary["primary_eligible_call_count"], 1)
        self.assertEqual(summary["primary_output_tokens"], 200)
        self.assertEqual(summary["provider_window_output_tps"], 100.0)
        self.assertEqual(summary["output_token_reconciliation"], "matched")

    def test_batch_tps_is_ratio_of_sums(self):
        aggregate = summarize_batch(
            [
                {
                    "status": "completed_without_evaluation",
                    "inference_timing_coverage": "complete",
                    "provider_window_inference_seconds": 1.0,
                    "provider_window_output_tps": 100.0,
                    "provider_window_output_tokens": 100,
                    "output_tokens": 100,
                },
                {
                    "status": "completed_without_evaluation",
                    "inference_timing_coverage": "complete",
                    "provider_window_inference_seconds": 10.0,
                    "provider_window_output_tps": 10.0,
                    "provider_window_output_tokens": 100,
                    "output_tokens": 100,
                },
            ]
        )
        self.assertEqual(
            aggregate["provider_window_output_tps_ratio_of_sums"],
            200 / 11,
        )
        self.assertNotEqual(
            aggregate["provider_window_output_tps_ratio_of_sums"], 55.0
        )

    def test_heartbeat_runner_captures_output(self):
        result = run_with_heartbeat(
            [sys.executable, "-c", "print('ready')"],
            label="test",
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout.strip(), "ready")


if __name__ == "__main__":
    unittest.main()
