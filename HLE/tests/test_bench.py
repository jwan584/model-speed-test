import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bench


def result_values(**overrides):
    values = {column: "" for column in bench.RESULT_COLUMNS}
    values.update(
        {
            "timestamp": "2026-07-03T00:00:00+00:00",
            "endpoint_name": "endpoint",
            "model": "model",
            "sorted_index": "7",
            "question_id": "question",
            "max_tokens": "100000",
            "run_idx": "1",
            "total_wall_s": "12.5",
            "output_tokens": "42",
            "token_count_method": "stream_usage",
            "reasoning_tokens": "not_reported",
            "correct": "pending",
            "finish_reason": "stop",
            "refusal": "no",
        }
    )
    values.update(overrides)
    return [values[column] for column in bench.RESULT_COLUMNS]


class BenchPersistenceTests(unittest.TestCase):
    def test_results_migration_backfills_refusal_and_anthropic_token_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results.csv"
            responses = root / "responses.jsonl"
            old_columns = bench.RESULT_COLUMNS[:-2]
            old_row = result_values(
                token_count_method="anthropic_stream_usage",
                visible_output_tokens="42",
                finish_reason="",
                refusal="",
            )[: len(old_columns)]
            with results.open("w", newline="") as handle:
                csv.writer(handle).writerows([old_columns, old_row])
            response = {
                "timestamp": "2026-07-03T00:00:00+00:00",
                "endpoint_name": "endpoint",
                "model": "model",
                "question_id": "question",
                "max_tokens": 100000,
                "run_idx": 1,
                "finish_reason": "refusal",
                "thinking_effort": "xhigh",
            }
            responses.write_text(json.dumps(response) + "\n")

            with patch.object(bench, "RESULTS_CSV", results), patch.object(
                bench, "RESPONSES_JSONL", responses
            ):
                bench.write_csv_header_if_needed()

            with results.open(newline="") as handle:
                migrated = list(csv.DictReader(handle))
            self.assertEqual(migrated[0]["finish_reason"], "refusal")
            self.assertEqual(migrated[0]["refusal"], "yes")
            self.assertEqual(migrated[0]["visible_output_tokens"], "")
            self.assertEqual(
                migrated[0]["token_count_method"],
                "anthropic_stream_usage_includes_omitted_thinking",
            )

    def test_completed_run_requires_matching_result_and_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results.csv"
            responses = root / "responses.jsonl"
            with results.open("w", newline="") as handle:
                csv.writer(handle).writerows([bench.RESULT_COLUMNS, result_values()])
            response = {
                "timestamp": "2026-07-03T00:00:00+00:00",
                "endpoint_name": "endpoint",
                "model": "model",
                "question_id": "question",
                "max_tokens": 100000,
                "run_idx": 1,
                "response": "answer",
            }
            responses.write_text(json.dumps(response) + "\n")

            with patch.object(bench, "RESULTS_CSV", results), patch.object(
                bench, "RESPONSES_JSONL", responses
            ):
                completed = bench.completed_run_records()

            self.assertEqual(len(completed), 1)
            result, saved_response = next(iter(completed.values()))
            self.assertEqual(result["total_wall_s"], "12.5")
            self.assertEqual(saved_response["response"], "answer")

    def test_resume_write_replaces_same_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results.csv"
            responses = root / "responses.jsonl"
            responses.write_text("")
            with results.open("w", newline="") as handle:
                csv.writer(handle).writerows(
                    [bench.RESULT_COLUMNS, result_values(total_wall_s="1"), result_values(total_wall_s="2")]
                )
            replacement = result_values(total_wall_s="3")

            with patch.object(bench, "RESULTS_CSV", results), patch.object(
                bench, "RESPONSES_JSONL", responses
            ):
                bench.write_result_row(replacement, replace_identity=True)

            with results.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["total_wall_s"], "3")

    def test_cli_defaults_to_resume_and_one_hour_timeouts(self):
        with patch.object(sys, "argv", ["bench.py"]):
            args = bench.parse_args()
        self.assertTrue(args.resume)
        self.assertEqual(args.timeout_seconds, 3600.0)
        self.assertEqual(args.judge_timeout_seconds, 3600.0)

    def test_run_report_counts_refusals(self):
        records = [
            {"total_wall_s": 1, "reasoning_tokens": None, "output_tokens": 1, "correct": "no", "refusal": "yes"},
            {"total_wall_s": 2, "reasoning_tokens": None, "output_tokens": 2, "correct": "yes", "refusal": "no"},
        ]
        self.assertEqual(bench.format_refusals(records), "1/2")

    def test_anthropic_omitted_thinking_is_not_reported_as_visible_output(self):
        captured = {}

        class FakeStream:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __iter__(self):
                return iter(
                    [
                        SimpleNamespace(
                            type="content_block_delta",
                            delta=SimpleNamespace(type="text_delta", text="Answer: A"),
                        )
                    ]
                )

            def get_final_message(self):
                return SimpleNamespace(
                    usage=SimpleNamespace(
                        model_dump=lambda: {"input_tokens": 5, "output_tokens": 100}
                    ),
                    stop_reason="end_turn",
                )

        class FakeAnthropic:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.messages = SimpleNamespace(stream=lambda **request: FakeStream())

        messages = [
            {"role": "system", "content": "format"},
            {"role": "user", "content": [{"type": "text", "text": "question"}]},
        ]
        with patch("anthropic.Anthropic", FakeAnthropic):
            result = bench.stream_anthropic_completion(
                api_key="key",
                model="model",
                max_tokens=1000,
                messages=messages,
                thinking_effort="xhigh",
                adaptive_thinking=True,
                omit_temperature=True,
                timeout_seconds=3600,
            )

        self.assertEqual(captured["timeout"], 3600)
        self.assertEqual(result["output_tokens"], 100)
        self.assertIsNone(result["reasoning_tokens"])
        self.assertIsNone(result["visible_output_tokens"])
        self.assertEqual(
            result["token_count_method"],
            "anthropic_stream_usage_includes_omitted_thinking",
        )

    def test_openai_refusal_event_is_recorded(self):
        usage = {
            "input_tokens": 5,
            "output_tokens": 3,
            "total_tokens": 8,
            "output_tokens_details": {"reasoning_tokens": 0},
        }
        response = SimpleNamespace(
            usage=usage,
            status="completed",
            output=[SimpleNamespace(content=[SimpleNamespace(type="refusal")])],
        )
        events = [
            SimpleNamespace(type="response.refusal.delta", delta="I cannot comply."),
            SimpleNamespace(type="response.completed", response=response),
        ]
        client = SimpleNamespace(
            responses=SimpleNamespace(create=lambda **request: iter(events))
        )
        messages = [
            {"role": "system", "content": "format"},
            {"role": "user", "content": [{"type": "text", "text": "question"}]},
        ]

        result = bench.stream_completion(
            client=client,
            model="model",
            max_tokens=1000,
            messages=messages,
            omit_temperature=True,
        )

        self.assertEqual(result["finish_reason"], "refusal")
        self.assertEqual(result["response_text"], "I cannot comply.")

    def test_content_filter_is_classified_as_refusal(self):
        self.assertTrue(bench.is_refusal_finish_reason("content_filter"))

    def test_only_pending_or_unjudged_results_need_judgment(self):
        self.assertFalse(bench.result_needs_judgment({"correct": "yes"}))
        self.assertFalse(bench.result_needs_judgment({"correct": "no"}))
        self.assertTrue(bench.result_needs_judgment({"correct": "pending"}))
        self.assertTrue(bench.result_needs_judgment({"correct": "not_judged"}))


if __name__ == "__main__":
    unittest.main()
