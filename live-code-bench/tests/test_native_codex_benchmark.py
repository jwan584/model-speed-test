import json
import tempfile
import time
import unittest
from pathlib import Path

from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import (
    ExportLogsServiceRequest,
)
from opentelemetry.proto.collector.metrics.v1.metrics_service_pb2 import (
    ExportMetricsServiceRequest,
)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue

from native_codex_benchmark import (
    LIFECYCLE_TRACE_TARGET,
    TIMING_TRACE_TARGET,
    binary_has_timing_trace_hook,
    decode_logs,
    decode_metrics,
    load_problem_fixture,
    pair_inference_calls,
    parse_lifecycle_trace_line,
    parse_timing_trace_line,
    summarize_client_lifecycle,
    summarize_otel,
    summarize_server_aggregate,
)
from bench_runner import check_solution


class NativeCodexBenchmarkTests(unittest.TestCase):
    def test_q1_fixture_loads_without_dataset(self):
        fixture = (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "lcb_release_v6_q1_abc387_f.json"
        )
        problems = load_problem_fixture(fixture)
        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0].question_id, "abc387_f")
        self.assertEqual(problems[0].difficulty.value, "hard")
        self.assertEqual(len(problems[0].public_test_cases), 3)
        self.assertEqual(len(problems[0].private_test_cases), 0)

    def test_q1_fixture_checker_supports_buffered_readline(self):
        fixture = (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "lcb_release_v6_q1_abc387_f.json"
        )
        problem = load_problem_fixture(fixture)[0]
        output = """```python
import sys
n, m = map(int, sys.stdin.buffer.readline().split())
a = list(map(int, sys.stdin.buffer.readline().split()))
if (n, m, a) == (3, 3, [2, 1, 1]): print(6)
elif (n, m, a) == (4, 9, [1, 1, 1, 1]): print(2025)
else: print(10010)
```"""
        self.assertTrue(check_solution(problem, output, 2))

    def test_binary_timing_trace_hook_detection_crosses_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "codex"
            marker = LIFECYCLE_TRACE_TARGET.encode()
            path.write_bytes(b"x" * (1024 * 1024 - 5) + marker + b"tail")
            self.assertTrue(binary_has_timing_trace_hook(path))

    def test_parse_lifecycle_trace_line_preserves_nanoseconds(self):
        line = (
            "TRACE codex_api::responses_websocket_lifecycle: "
            'responses websocket lifecycle model="gpt-test" '
            'session_id="session-1" thread_id="conversation-main" '
            'turn_id="turn-1" previous_response_id="resp-0" '
            'response_id="resp-1" warmup=false connection_reused=true '
            'request_sent_unix_ns="1783980000000000001" '
            'provider_event_started_unix_ns="1783980000000000123" '
            'completed_unix_ns="1783980002000000456" '
            'provider_start_kind="response.created" '
            "request_to_completed_ms=2100.5 provider_event_window_ms=2000.25 "
            "input_tokens=100 output_tokens=400 reasoning_output_tokens=250"
        )
        parsed = parse_lifecycle_trace_line(line)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(
            parsed["provider_event_started_unix_ns"], 1783980000000000123
        )
        self.assertEqual(parsed["output_tokens"], 400)
        self.assertEqual(parsed["provider_start_kind"], "response.created")

    def test_client_lifecycle_separates_nested_tool_union(self):
        traces = [
            {
                "model": "gpt-test",
                "warmup": False,
                "provider_start_kind": "response.created",
                "provider_event_started_unix_ns": 2_000_000_000,
                "completed_unix_ns": 8_000_000_000,
                "provider_event_window_ms": 6000.0,
                "request_to_completed_ms": 6200.0,
                "output_tokens": 300,
                "reasoning_output_tokens": 200,
            }
        ]
        events = [
            {
                "signal": "log",
                "time_unix_nano": 6_000_000_000,
                "attributes": {
                    "event.name": "codex.tool_result",
                    "tool_name": "exec",
                    "call_id": "outer",
                    "duration_ms": 3000,
                },
            },
            {
                "signal": "log",
                "time_unix_nano": 5_500_000_000,
                "attributes": {
                    "event.name": "codex.tool_result",
                    "tool_name": "exec_command",
                    "call_id": "inner",
                    "duration_ms": 2000,
                },
            },
        ]
        calls, summary = summarize_client_lifecycle(
            traces,
            events,
            "gpt-test",
            "abc387_f",
            0,
            10_000_000_000,
            {"output_tokens": 300},
        )
        self.assertEqual(summary["primary_coverage"], "complete")
        self.assertEqual(summary["total_tool_seconds"], 3.0)
        self.assertEqual(summary["primary_tool_overlap_seconds"], 3.0)
        self.assertEqual(summary["primary_provider_window_inference_seconds"], 6.0)
        self.assertEqual(summary["primary_client_active_inference_seconds"], 6.0)
        self.assertEqual(summary["total_unattributed_seconds"], 4.0)
        self.assertEqual(summary["primary_provider_window_output_tps"], 50.0)
        self.assertEqual(calls[0]["provider_window_output_tps"], 50.0)

    def test_fully_overlapped_inference_remains_eligible(self):
        traces = [
            {
                "model": "gpt-test",
                "thread_id": "main",
                "warmup": False,
                "provider_start_kind": "response.created",
                "provider_event_started_unix_ns": 2_000_000_000,
                "completed_unix_ns": 3_000_000_000,
                # The emitted float can differ slightly from the exact ns
                # boundary and must not determine coverage.
                "provider_event_window_ms": 999.998,
                "output_tokens": 45,
            }
        ]
        events = [
            {
                "signal": "log",
                "time_unix_nano": 3_000_000_000,
                "attributes": {
                    "event.name": "codex.tool_result",
                    "tool_name": "exec",
                    "call_id": "fully-overlapping-tool",
                    "conversation.id": "main",
                    "duration_ms": 1000,
                },
            }
        ]
        calls, summary = summarize_client_lifecycle(
            traces,
            events,
            "gpt-test",
            "abc388_g",
            0,
            4_000_000_000,
            {"output_tokens": 45},
        )
        self.assertTrue(calls[0]["included_in_primary"])
        self.assertIsNone(calls[0]["primary_exclusion_reason"])
        self.assertEqual(calls[0]["provider_window_inference_seconds"], 1.0)
        self.assertEqual(calls[0]["tool_overlap_ms"], 1000.0)
        self.assertEqual(summary["primary_output_tokens"], 45)
        self.assertEqual(summary["primary_provider_window_output_tps"], 45.0)
        self.assertEqual(summary["output_token_reconciliation"], "matched")
        self.assertEqual(summary["primary_coverage"], "complete")

    def test_tool_overlap_only_matches_its_conversation(self):
        traces = [
            {
                "model": "gpt-test",
                "thread_id": "main",
                "session_id": "main",
                "warmup": False,
                "provider_start_kind": "response.created",
                "provider_event_started_unix_ns": 2_000_000_000,
                "completed_unix_ns": 8_000_000_000,
                "provider_event_window_ms": 6000.0,
                "output_tokens": 300,
            },
            {
                "model": "review-model",
                "thread_id": "review",
                "session_id": "main",
                "warmup": False,
                "provider_start_kind": "response.created",
                "provider_event_started_unix_ns": 3_000_000_000,
                "completed_unix_ns": 6_000_000_000,
                "provider_event_window_ms": 3000.0,
                "output_tokens": 30,
            },
        ]
        events = [
            {
                "signal": "log",
                "time_unix_nano": 6_000_000_000,
                "attributes": {
                    "event.name": "codex.tool_result",
                    "tool_name": "exec",
                    "call_id": "main-tool",
                    "conversation.id": "main",
                    "duration_ms": 3000,
                },
            }
        ]
        calls, summary = summarize_client_lifecycle(
            traces,
            events,
            "gpt-test",
            "abc387_f",
            0,
            10_000_000_000,
            {"output_tokens": 300},
        )
        self.assertEqual(calls[0]["client_active_inference_seconds"], 6.0)
        self.assertEqual(calls[1]["client_active_inference_seconds"], 3.0)
        self.assertEqual(calls[0]["tool_overlap_ms"], 3000.0)
        self.assertEqual(calls[1]["tool_overlap_ms"], 0.0)
        self.assertEqual(
            summary["total_all_models_client_active_inference_seconds"], 6.0
        )
        self.assertEqual(summary["inference_tool_concurrency_seconds"], 3.0)
        self.assertEqual(summary["total_unattributed_seconds"], 4.0)

    def test_parse_timing_trace_line(self):
        payload = json.dumps(
            {
                "type": "responsesapi.websocket_timing",
                "timing_metrics": {
                    "responses_duration_excl_engine_and_client_tool_time_ms": 510,
                    "engine_service_total_ms": 450,
                    "engine_iapi_ttft_total_ms": 300,
                    "engine_service_ttft_total_ms": 340,
                    "engine_iapi_tbt_across_engine_calls_ms": 2.5,
                    "engine_service_tbt_across_engine_calls_ms": 3.25,
                },
            },
            separators=(",", ":"),
        )
        line = (
            "TRACE codex_api::responses_websocket_timing: responses websocket timing "
            'model="gpt-test" session_id="session-1" thread_id="thread-1" '
            'turn_id="turn-1" request_start_ms="123" warmup=false '
            f"connection_reused=true payload={json.dumps(payload)}"
        )
        parsed = parse_timing_trace_line(line)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["model"], "gpt-test")
        self.assertEqual(parsed["thread_id"], "thread-1")
        self.assertFalse(parsed["warmup"])
        self.assertTrue(parsed["connection_reused"])
        self.assertEqual(parsed["engine_service_total_ms"], 450.0)

    def test_decode_logs(self):
        request = ExportLogsServiceRequest()
        record = request.resource_logs.add().scope_logs.add().log_records.add()
        record.time_unix_nano = 123
        record.body.string_value = "codex.sse_event"
        record.attributes.append(
            KeyValue(key="kind", value=AnyValue(string_value="response.completed"))
        )
        events = decode_logs(request.SerializeToString(), time.time_ns())
        self.assertEqual(events[0]["body"], "codex.sse_event")
        self.assertEqual(events[0]["attributes"]["kind"], "response.completed")

    def test_decode_delta_histogram(self):
        request = ExportMetricsServiceRequest()
        metric = request.resource_metrics.add().scope_metrics.add().metrics.add()
        metric.name = "codex.responses_api_inference_time.duration_ms"
        metric.histogram.aggregation_temporality = 1
        point = metric.histogram.data_points.add()
        point.time_unix_nano = 123
        point.count = 2
        point.sum = 2050.0
        point.attributes.append(
            KeyValue(key="model", value=AnyValue(string_value="gpt-test"))
        )
        events = decode_metrics(request.SerializeToString(), time.time_ns())
        self.assertEqual(events[0]["aggregation_temporality"], 1)
        self.assertEqual(events[0]["count"], 2)
        self.assertEqual(events[0]["sum"], 2050.0)
        self.assertEqual(events[0]["attributes"]["model"], "gpt-test")

    def test_sse_window_is_diagnostic_not_principal(self):
        events = [
            {
                "signal": "log",
                "time_unix_nano": 1_000_000_000,
                "body": "codex.sse_event",
                "attributes": {"kind": "response.output_item.added"},
            },
            {
                "signal": "log",
                "time_unix_nano": 3_500_000_000,
                "body": "codex.sse_event",
                "attributes": {"kind": "response.completed"},
            },
        ]
        summary = summarize_otel(events)
        self.assertEqual(summary["sse_active_seconds"], 2.5)
        self.assertEqual(summary["sse_boundary_confidence"], "event_kind_only")
        self.assertIsNone(summary["scrutable_active_generation_seconds"])

    def test_reasoning_item_boundary_is_scrutable(self):
        events = [
            {
                "signal": "log", "time_unix_nano": 2_000_000_000,
                "body": "codex.sse_event",
                "attributes": {
                    "kind": "response.output_item.added", "item.type": "reasoning"
                },
            },
            {
                "signal": "log", "time_unix_nano": 6_000_000_000,
                "body": "codex.sse_event",
                "attributes": {"kind": "response.completed"},
            },
        ]
        summary = summarize_otel(events)
        self.assertEqual(summary["scrutable_active_generation_seconds"], 4.0)
        self.assertEqual(summary["sse_boundary_confidence"], "provider_boundary")

    def test_server_metric_is_filtered_to_target_model(self):
        events = [
            {
                "signal": "metric",
                "name": "codex.responses_api_inference_time.duration_ms",
                "kind": "histogram",
                "sum": 2050.0,
                "count": 2,
                "aggregation_temporality": 1,
                "attributes": {"model": "gpt-test"},
            },
            {
                "signal": "metric",
                "name": "codex.responses_api_inference_time.duration_ms",
                "kind": "histogram",
                "sum": 1000.0,
                "count": 1,
                "aggregation_temporality": 1,
                "attributes": {"model": "codex-auto-review"},
            },
        ]
        summary = summarize_otel(events, "gpt-test")
        self.assertEqual(summary["responses_api_inference_seconds"], 2.05)
        self.assertEqual(
            summary["responses_api_inference_observation_count"], 2
        )
        self.assertEqual(
            summary["responses_api_inference_metric_models"],
            ["codex-auto-review", "gpt-test"],
        )
        self.assertEqual(
            summary["responses_api_inference_model_filter"], "exact"
        )

    def test_server_aggregate_excludes_auxiliary_model_and_tool_gaps(self):
        events = [
            {
                "signal": "metric",
                "name": "codex.responses_api_inference_time.duration_ms",
                "sum": 2050.0,
                "count": 2,
                "aggregation_temporality": 1,
                "attributes": {"model": "gpt-test"},
            },
            {
                "signal": "metric",
                "name": "codex.responses_api_inference_time.duration_ms",
                "sum": 1000.0,
                "count": 1,
                "aggregation_temporality": 1,
                "attributes": {"model": "codex-auto-review"},
            },
        ]
        completions = [
            {"model": "gpt-test", "output_tokens": 0},
            {"model": "gpt-test", "output_tokens": 400},
            {"model": "codex-auto-review", "output_tokens": 50},
        ]
        otel = summarize_otel(events, "gpt-test")
        aggregate = summarize_server_aggregate(
            completions, otel, "gpt-test", {"output_tokens": 400}
        )
        self.assertEqual(aggregate["coverage"], "complete")
        self.assertEqual(aggregate["completed_call_count"], 2)
        self.assertEqual(aggregate["inference_observation_count"], 2)
        self.assertEqual(aggregate["output_tokens"], 400)
        self.assertAlmostEqual(aggregate["inference_output_tps"], 400 / 2.05)
        self.assertTrue(aggregate["excludes_tool_time"])
        self.assertTrue(aggregate["includes_warmup_timing"])

    def test_micro_inference_calls_pair_and_aggregate(self):
        timings = [
            {
                "model": "gpt-test", "thread_id": "conversation-main",
                "session_id": "", "warmup": True,
                "engine_service_total_ms": 50.0,
                "engine_service_ttft_total_ms": 40.0,
            },
            {
                "model": "gpt-test", "thread_id": "conversation-main",
                "session_id": "", "warmup": False,
                "engine_service_total_ms": 2000.0,
                "engine_service_ttft_total_ms": 500.0,
            },
            {
                "model": "codex-auto-review", "thread_id": "conversation-review",
                "session_id": "", "warmup": False,
                "engine_service_total_ms": 1000.0,
                "engine_service_ttft_total_ms": 300.0,
            },
        ]
        completions = [
            {
                "model": "gpt-test", "conversation_id": "conversation-main",
                "output_tokens": 0, "reasoning_output_tokens": 0,
            },
            {
                "model": "gpt-test", "conversation_id": "conversation-main",
                "output_tokens": 400, "reasoning_output_tokens": 250,
            },
            {
                "model": "codex-auto-review", "conversation_id": "conversation-review",
                "output_tokens": 50, "reasoning_output_tokens": 20,
            },
        ]
        calls, summary = pair_inference_calls(
            timings, completions, "gpt-test", "abc387_f"
        )
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[1]["pairing"], "conversation_ordinal")
        self.assertEqual(calls[1]["inference_output_tps"], 200.0)
        self.assertEqual(calls[1]["decode_output_tps_diagnostic"], 266.0)
        self.assertFalse(calls[0]["included_in_primary"])
        self.assertEqual(calls[0]["primary_exclusion_reason"], "warmup")
        self.assertEqual(calls[2]["primary_exclusion_reason"], "auxiliary_model")
        self.assertEqual(summary["primary_eligible_call_count"], 1)
        self.assertEqual(summary["primary_output_tokens"], 400)
        self.assertEqual(summary["primary_engine_inference_seconds"], 2.0)
        self.assertEqual(summary["primary_inference_output_tps"], 200.0)
        self.assertEqual(summary["primary_coverage"], "complete")

    def test_missing_timing_is_recorded_not_substituted(self):
        calls, summary = pair_inference_calls(
            [],
            [{
                "model": "gpt-test", "conversation_id": "conversation-main",
                "output_tokens": 400,
            }],
            "gpt-test",
            "abc387_f",
        )
        self.assertEqual(calls[0]["pairing"], "unpaired_completion")
        self.assertEqual(calls[0]["primary_exclusion_reason"], "unpaired_completion")
        self.assertIsNone(calls[0]["inference_output_tps"])
        self.assertEqual(summary["primary_coverage"], "partial")
        self.assertEqual(summary["primary_eligible_call_count"], 0)


if __name__ == "__main__":
    unittest.main()
