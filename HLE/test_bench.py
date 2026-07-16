import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bench


class FakeStream:
    def __init__(self, events, headers):
        self.events = events
        self.response = SimpleNamespace(headers=headers)

    def __iter__(self):
        return iter(self.events)


class FakeAnthropicMessageStream:
    def __init__(self, events, final_message, headers):
        self.events = events
        self.final_message = final_message
        self._raw_stream = SimpleNamespace(response=SimpleNamespace(headers=headers))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self):
        return self.final_message


class TimeoutAnthropicMessageStream(FakeAnthropicMessageStream):
    def __iter__(self):
        raise TimeoutError("read timed out")


class BenchmarkCountingTests(unittest.TestCase):
    def test_outcomes_are_explicit(self):
        self.assertEqual(bench.outcome_from_finish_reason("stop"), "completed")
        self.assertEqual(bench.outcome_from_finish_reason("end_turn"), "completed")
        self.assertEqual(bench.outcome_from_finish_reason("refusal"), "refusal")
        self.assertEqual(bench.outcome_from_finish_reason("content_filter"), "refusal")
        self.assertEqual(
            bench.outcome_from_finish_reason("max_tokens"),
            "incomplete_max_tokens",
        )
        self.assertEqual(
            bench.outcome_from_finish_reason("inference_timeout"),
            "inference_timeout",
        )

    def test_headline_tps_uses_ratio_of_sums_for_completed_calls_only(self):
        records = [
            self.record("completed", billed=100, inference=2.0, output=100),
            self.record("completed", billed=300, inference=6.0, output=300),
            self.record("incomplete_max_tokens", billed=1000, inference=10.0, output=1000),
            self.record("inference_timeout", billed=None, inference=0.0, retry=5.0),
        ]
        output = io.StringIO()
        with redirect_stdout(output):
            bench.print_run_report(endpoint_name="test", model="model", records=records)
        report = output.getvalue()
        self.assertIn("completed_inference_calls=2", report)
        self.assertIn("billed_output_tokens=400", report)
        self.assertIn("completed_inference_time_s=8.000000", report)
        self.assertIn("end_to_end_billed_tps=50.000000", report)
        self.assertIn('usage_reconciliation={"matched":2,"unavailable":2}', report)
        self.assertIn("total_inference_time_s=18.000000", report)
        self.assertIn("total_retry_api_time_s=5.000000", report)
        self.assertIn("task_success=0/4", report)

    def test_failed_call_never_invents_billed_tokens(self):
        result = bench.failed_inference_result(
            request_sent_ts=bench.now_utc(),
            request_sent_perf=bench.perf_counter(),
            response_text_parts=["partial"],
            finish_reason="inference_timeout",
            observable_chunk_count=1,
        )
        self.assertIsNone(result["output_tokens"])
        self.assertIsNone(result["billed_output_tokens"])
        self.assertEqual(result["outcome"], "inference_timeout")
        self.assertEqual(result["token_count_method"], "incomplete_no_final_usage")
        self.assertEqual(result["usage_reconciliation_status"], "unavailable")

    def test_mismatched_usage_is_excluded_from_headline(self):
        record = self.record("completed", billed=100, inference=2.0, output=100)
        record["usage_reconciliation_status"] = "mismatched"
        self.assertEqual(
            bench.headline_eligibility(record),
            (False, "usage_reconciliation:mismatched"),
        )

    def test_final_usage_reconciliation(self):
        self.assertEqual(bench.reconcile_final_usage(10, '{"output_tokens": 10}'), "matched")
        self.assertEqual(bench.reconcile_final_usage(11, '{"output_tokens": 10}'), "mismatched")
        self.assertEqual(bench.reconcile_final_usage(None, '{"output_tokens": 10}'), "unavailable")

    def test_openai_stream_uses_terminal_provider_usage(self):
        usage = {
            "input_tokens": 5,
            "output_tokens": 10,
            "total_tokens": 15,
            "output_tokens_details": {"reasoning_tokens": 4},
        }
        completed_response = SimpleNamespace(
            status="completed",
            usage=usage,
            output=[],
            incomplete_details=None,
            _request_id=None,
        )
        events = [
            SimpleNamespace(type="response.output_text.delta", delta="answer"),
            SimpleNamespace(type="response.completed", response=completed_response),
        ]
        stream = FakeStream(events, {"x-request-id": "req_test"})
        client = SimpleNamespace(
            responses=SimpleNamespace(create=lambda **kwargs: stream),
        )
        result = bench.stream_completion(
            client=client,
            model="test-model",
            max_tokens=100,
            messages=[{"role": "user", "content": "question"}],
            thinking_effort="xhigh",
            omit_temperature=True,
        )
        self.assertEqual(result["outcome"], "completed")
        self.assertEqual(result["billed_output_tokens"], 10)
        self.assertEqual(result["reasoning_tokens"], 4)
        self.assertEqual(result["visible_output_tokens"], 6)
        self.assertEqual(result["request_id"], "req_test")
        self.assertEqual(result["observable_chunk_count"], 1)
        self.assertIsNotNone(result["first_stream_event_latency_s"])

    def test_anthropic_stream_records_reported_thinking_subset(self):
        delta = SimpleNamespace(type="text_delta", text="answer")
        events = [SimpleNamespace(type="content_block_delta", delta=delta)]
        final_message = SimpleNamespace(
            usage={"input_tokens": 5, "output_tokens": 20, "thinking_tokens": 15},
            stop_reason="end_turn",
        )
        stream = FakeAnthropicMessageStream(
            events,
            final_message,
            {"request-id": "req_anthropic"},
        )
        client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: stream))
        with patch("anthropic.Anthropic", return_value=client):
            result = bench.stream_anthropic_completion(
                api_key="test",
                model="claude-test",
                max_tokens=100,
                messages=[{"role": "user", "content": "question"}],
                thinking_effort="xhigh",
                adaptive_thinking=True,
                omit_temperature=True,
            )
        self.assertEqual(result["outcome"], "completed")
        self.assertEqual(result["billed_output_tokens"], 20)
        self.assertEqual(result["reasoning_tokens"], 15)
        self.assertEqual(result["visible_output_tokens"], 5)
        self.assertEqual(result["request_id"], "req_anthropic")

    def test_anthropic_timeout_is_recorded_without_tokens(self):
        stream = TimeoutAnthropicMessageStream([], None, {"request-id": "req_timeout"})
        client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: stream))
        with patch("anthropic.Anthropic", return_value=client):
            result = bench.stream_anthropic_completion(
                api_key="test",
                model="claude-test",
                max_tokens=100,
                messages=[{"role": "user", "content": "question"}],
                thinking_effort="xhigh",
                adaptive_thinking=True,
                omit_temperature=True,
            )
        self.assertEqual(result["outcome"], "inference_timeout")
        self.assertIsNone(result["billed_output_tokens"])
        self.assertEqual(result["request_id"], "req_timeout")

    def test_historical_csv_migration_backfills_only_derivable_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            results_path = Path(directory) / "results.csv"
            responses_path = Path(directory) / "responses.jsonl"
            old_columns = bench.RESULT_COLUMNS[:21]
            old_record = {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "endpoint_name": "old",
                "model": "model",
                "sorted_index": "1",
                "question_id": "qid",
                "max_tokens": "100",
                "run_idx": "1",
                "ttft_s": "1",
                "gen_time_s": "1",
                "total_wall_s": "2",
                "output_tokens": "10",
                "token_count_method": "stream_usage",
                "tokens_per_s": "5",
                "input_tokens": "2",
                "reasoning_tokens": "4",
                "visible_output_tokens": "6",
                "total_tokens": "12",
                "usage_json": json.dumps({"output_tokens": 10}),
                "correct": "yes",
                "finish_reason": "stop",
                "refusal": "no",
            }
            with results_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=old_columns)
                writer.writeheader()
                writer.writerow(old_record)
            responses_path.write_text(
                json.dumps({**old_record, "provider": "openai", "response": "answer"}) + "\n"
            )
            with patch.object(bench, "RESULTS_CSV", results_path), patch.object(
                bench, "RESPONSES_JSONL", responses_path
            ):
                migrated = bench.load_result_records()[0]
            self.assertEqual(migrated["outcome"], "completed")
            self.assertEqual(migrated["billed_output_tokens"], "10")
            self.assertEqual(migrated["inference_time_s"], "2")
            self.assertEqual(migrated["thinking_tokens"], "4")
            self.assertEqual(migrated["request_id"], "not_reported")

    def test_main_persists_spec_fields_end_to_end(self):
        args = SimpleNamespace(
            provider="openai",
            endpoint_name="test-endpoint",
            base_url="https://example.test/v1",
            api_key="test",
            model="test-model",
            endpoint_region="test-region",
            sandbox_image="not_applicable",
            cpu_memory_limits="not_applicable",
            economy_policy="none",
            max_tokens=100,
            timeout_seconds=60.0,
            thinking_budget_tokens=None,
            thinking_effort="xhigh",
            anthropic_adaptive_thinking=True,
            omit_temperature=True,
            index=1,
            question_ids=None,
            category=None,
            raw_subject=None,
            num_questions=1,
            runs=1,
            resume=True,
            list_problems=False,
            print_response=False,
            no_print_question=True,
            judge=False,
            judge_after_run=False,
            judge_existing=False,
            judge_model="judge",
            judge_base_url=None,
            judge_api_key=None,
            judge_max_tokens=100,
            judge_timeout_seconds=60.0,
            print_judge=False,
        )
        problem = bench.Problem(
            sorted_index=1,
            question_id="qid",
            question={
                "id": "qid",
                "question": "question",
                "answer": "answer",
                "category": "Math",
                "raw_subject": "test",
                "image": "",
            },
        )
        result = {
            "request_sent_ts": bench.now_utc(),
            "first_token_ts": bench.now_utc(),
            "last_token_ts": bench.now_utc(),
            "first_stream_event_ts": bench.now_utc(),
            "response_text": "Answer: answer",
            "output_tokens": 10,
            "billed_output_tokens": 10,
            "input_tokens": 5,
            "reasoning_tokens": 4,
            "visible_output_tokens": 6,
            "total_tokens": 15,
            "usage_json": json.dumps({"output_tokens": 10}),
            "token_count_method": "stream_usage_includes_thinking",
            "ttft_s": 0.1,
            "first_stream_event_latency_s": 0.05,
            "gen_time_s": 0.2,
            "api_call_wall_s": 1.0,
            "tokens_per_s": 10.0,
            "finish_reason": "stop",
            "outcome": "completed",
            "request_id": "req_test",
            "observable_chunk_count": 2,
            "usage_reconciliation_status": "matched",
        }
        prompt_module = SimpleNamespace(
            args=None,
            format_message=lambda question: [{"role": "user", "content": "question"}],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            patches = [
                patch.object(bench, "RESULTS_CSV", root / "results.csv"),
                patch.object(bench, "RESPONSES_JSONL", root / "responses.jsonl"),
                patch.object(bench, "RUN_SUMMARIES_JSONL", root / "summaries.jsonl"),
                patch.object(bench, "JUDGMENTS_CSV", root / "judgments.csv"),
                patch.object(bench, "parse_args", return_value=args),
                patch.object(bench, "canonical_text_only_questions", return_value=[problem]),
                patch.object(bench, "select_problems", return_value=[problem]),
                patch.object(bench, "load_prompt_module", return_value=prompt_module),
                patch.object(bench, "OpenAI", return_value=SimpleNamespace()),
                patch.object(bench, "stream_completion", return_value=result),
            ]
            for context in patches:
                context.start()
            try:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(bench.main(), 0)
            finally:
                for context in reversed(patches):
                    context.stop()
            with (root / "results.csv").open(newline="") as handle:
                persisted = next(csv.DictReader(handle))
            self.assertEqual(persisted["billed_output_tokens"], "10")
            self.assertEqual(persisted["outcome"], "completed")
            self.assertEqual(persisted["request_id"], "req_test")
            self.assertEqual(persisted["endpoint_region"], "test-region")
            self.assertEqual(persisted["serial_execution"], "yes")
            self.assertEqual(persisted["usage_reconciliation_status"], "matched")
            self.assertEqual(persisted["headline_eligible"], "yes")
            self.assertEqual(persisted["attempt_index"], "1")

    @staticmethod
    def record(outcome, billed, inference, output=None, retry=0.0):
        return {
            "total_wall_s": inference + retry,
            "inference_time_s": inference,
            "tool_time_s": 0.0,
            "retry_api_time_s": retry,
            "backoff_time_s": 0.0,
            "harness_overhead_s": 0.0,
            "reasoning_tokens": 0,
            "visible_output_tokens": output,
            "output_tokens": output,
            "billed_output_tokens": billed,
            "outcome": outcome,
            "correct": "not_judged",
            "finish_reason": outcome,
            "refusal": "no",
            "usage_reconciliation_status": "matched" if outcome == "completed" else "unavailable",
            "ttft_s": 0.5 if outcome == "completed" else None,
        }


if __name__ == "__main__":
    unittest.main()
