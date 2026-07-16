import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bench_runner


class BenchRunnerRequestTests(unittest.TestCase):
    def test_custom_task_requires_bare_complete_html_and_patterns(self):
        task = bench_runner.CustomTask(
            "app", "Build an app", (r"<script", r"localStorage")
        )
        document = (
            "<!doctype html><html><body><script>"
            "localStorage.setItem('x','1')</script></body></html>"
        )
        self.assertTrue(bench_runner.check_custom_task(task, document))
        self.assertFalse(
            bench_runner.check_custom_task(task, f"```html\n{document}\n```")
        )
        self.assertFalse(
            bench_runner.check_custom_task(
                task, "<!doctype html><html><body></body></html>"
            )
        )

    def test_openai_uses_responses_shape_and_visible_output_timing(self):
        captured = {}
        response = SimpleNamespace(
            id="resp_test",
            status="completed",
            incomplete_details=None,
            usage=SimpleNamespace(
                input_tokens=7,
                output_tokens=12,
                output_tokens_details=SimpleNamespace(reasoning_tokens=8),
            ),
        )
        events = [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.reasoning_text.delta", delta="hidden"),
            SimpleNamespace(type="response.output_text.delta", delta="```python\n"),
            SimpleNamespace(type="response.output_text.delta", delta="print(1)\n```"),
            SimpleNamespace(type="response.completed", response=response),
        ]

        def create(**request):
            captured.update(request)
            return iter(events)

        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        with patch.object(
            bench_runner.time,
            "perf_counter",
            side_effect=[0.0, 1.0, 2.0, 5.0, 7.0, 8.0, 10.0],
        ):
            result = bench_runner.stream_openai_response(
                client,
                "gpt-test",
                32000,
                [
                    {"role": "system", "content": "system prompt"},
                    {"role": "user", "content": "problem"},
                ],
                "high",
            )

        self.assertEqual(captured["instructions"], "system prompt")
        self.assertEqual(
            captured["input"],
            [{"role": "user", "content": [{"type": "input_text", "text": "problem"}]}],
        )
        self.assertEqual(captured["max_output_tokens"], 32000)
        self.assertEqual(captured["reasoning"], {"effort": "high"})
        self.assertNotIn("temperature", captured)
        self.assertEqual(result["ttft_s"], 5.0)
        self.assertEqual(result["generation_wall_s"], 5.0)
        self.assertEqual(result["output"], "```python\nprint(1)\n```")
        self.assertEqual(result["request_status"], "completed")
        self.assertEqual(result["stop_reason"], "completed")
        self.assertEqual(result["input_tokens"], 7)
        self.assertEqual(result["request_id"], "resp_test")
        self.assertEqual(result["reasoning_tokens"], 8)
        self.assertEqual(result["observable_chunk_count"], 2)
        self.assertEqual(result["first_stream_event_s"], 1.0)
        self.assertEqual(result["response_created_s"], 1.0)
        self.assertEqual(result["provider_window_inference_time_s"], 7.0)
        self.assertEqual(result["first_observable_output_s"], 5.0)
        self.assertEqual(result["last_observable_output_s"], 7.0)
        self.assertEqual(result["inference_time_s"], 10.0)

    def test_anthropic_uses_native_equivalent_without_temperature(self):
        captured_client = {}
        captured_request = {}

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
                            delta=SimpleNamespace(type="thinking_delta", thinking="hidden"),
                        ),
                        SimpleNamespace(
                            type="content_block_delta",
                            delta=SimpleNamespace(type="text_delta", text="```python\nprint(1)\n```"),
                        ),
                    ]
                )

            def get_final_message(self):
                return SimpleNamespace(
                    stop_reason="end_turn",
                    usage=SimpleNamespace(input_tokens=9, output_tokens=25),
                )

        class FakeAnthropic:
            def __init__(self, **kwargs):
                captured_client.update(kwargs)
                self.messages = SimpleNamespace(stream=self.stream)

            @staticmethod
            def stream(**kwargs):
                captured_request.update(kwargs)
                return FakeStream()

        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "problem"},
        ]
        with patch("anthropic.Anthropic", FakeAnthropic):
            result = bench_runner.stream_anthropic_response(
                "key", None, "claude-test", 32000, messages, "max", True, 3600
            )

        self.assertEqual(captured_client["max_retries"], 0)
        self.assertEqual(captured_client["timeout"], 3600)
        self.assertEqual(captured_request["system"], "system prompt")
        self.assertEqual(captured_request["messages"], [{"role": "user", "content": "problem"}])
        self.assertEqual(captured_request["max_tokens"], 32000)
        self.assertEqual(captured_request["thinking"], {"type": "adaptive", "display": "omitted"})
        self.assertEqual(captured_request["output_config"], {"effort": "max"})
        self.assertNotIn("temperature", captured_request)
        self.assertEqual(result["output_tokens"], 25)
        self.assertEqual(result["input_tokens"], 9)
        self.assertEqual(result["request_status"], "completed")
        self.assertEqual(result["stop_reason"], "end_turn")

    def test_cerebras_uses_streaming_chat_completions(self):
        captured = {}
        chunks = [
            SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(reasoning="think", content=None),
                    finish_reason=None,
                )],
            ),
            SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(reasoning=None, content="```python\nprint(1)\n```"),
                    finish_reason="stop",
                )],
            ),
            SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=9, completion_tokens=25),
                choices=[],
            ),
        ]

        def create(**request):
            captured.update(request)
            return iter(chunks)

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        result = bench_runner.stream_cerebras_response(
            client, "zai-glm-4.7", 32000,
            [{"role": "user", "content": "problem"}],
        )
        self.assertEqual(captured["max_completion_tokens"], 32000)
        self.assertEqual(captured["extra_body"], {"reasoning_format": "parsed"})
        self.assertNotIn("reasoning_effort", captured)
        self.assertEqual(result["reasoning_summary"], "think")
        self.assertEqual(result["output_tokens"], 25)
        self.assertEqual(result["request_status"], "completed")
        self.assertEqual(result["stop_reason"], "stop")

    def test_openai_incomplete_response_records_reason(self):
        response = SimpleNamespace(
            status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens"),
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        )
        events = [
            SimpleNamespace(type="response.output_text.delta", delta="partial"),
            SimpleNamespace(type="response.incomplete", response=response),
        ]
        client = SimpleNamespace(
            responses=SimpleNamespace(create=lambda **kwargs: iter(events))
        )
        result = bench_runner.stream_openai_response(
            client, "gpt-test", 20, [{"role": "user", "content": "problem"}]
        )
        self.assertEqual(result["request_status"], "incomplete")
        self.assertEqual(result["stop_reason"], "max_output_tokens")

    def test_openai_terminal_response_without_text_keeps_usage(self):
        response = SimpleNamespace(
            status="completed",
            incomplete_details=None,
            usage=SimpleNamespace(input_tokens=10, output_tokens=20),
        )
        client = SimpleNamespace(
            responses=SimpleNamespace(
                create=lambda **kwargs: iter(
                    [SimpleNamespace(type="response.completed", response=response)]
                )
            )
        )
        result = bench_runner.stream_openai_response(
            client, "gpt-test", 20, [{"role": "user", "content": "problem"}]
        )
        self.assertEqual(result["output"], "")
        self.assertIsNone(result["ttft_s"])
        self.assertEqual(result["output_tokens"], 20)
        self.assertEqual(result["request_status"], "completed")

    def test_anthropic_max_tokens_is_incomplete(self):
        class FakeStream:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __iter__(self):
                return iter(
                    [SimpleNamespace(
                        type="content_block_delta",
                        delta=SimpleNamespace(type="text_delta", text="partial"),
                    )]
                )

            def get_final_message(self):
                return SimpleNamespace(
                    stop_reason="max_tokens",
                    usage=SimpleNamespace(input_tokens=10, output_tokens=20),
                )

        fake_client = SimpleNamespace(
            messages=SimpleNamespace(stream=lambda **kwargs: FakeStream())
        )
        with patch("anthropic.Anthropic", return_value=fake_client):
            result = bench_runner.stream_anthropic_response(
                "key", None, "claude-test", 20,
                [{"role": "user", "content": "problem"}], "xhigh", True, 10,
            )
        self.assertEqual(result["request_status"], "incomplete")
        self.assertEqual(result["stop_reason"], "max_tokens")

    def test_anthropic_terminal_message_without_text_keeps_usage(self):
        class FakeStream:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __iter__(self):
                return iter([])

            def get_final_message(self):
                return SimpleNamespace(
                    stop_reason="end_turn",
                    usage=SimpleNamespace(input_tokens=10, output_tokens=20),
                )

        fake_client = SimpleNamespace(
            messages=SimpleNamespace(stream=lambda **kwargs: FakeStream())
        )
        with patch("anthropic.Anthropic", return_value=fake_client):
            result = bench_runner.stream_anthropic_response(
                "key", None, "claude-test", 20,
                [{"role": "user", "content": "problem"}], "xhigh", True, 10,
            )
        self.assertEqual(result["output"], "")
        self.assertIsNone(result["ttft_s"])
        self.assertEqual(result["output_tokens"], 20)
        self.assertEqual(result["request_status"], "completed")

    def test_single_provider_key_and_effort_broadcast_to_endpoints(self):
        args = argparse.Namespace(
            endpoint_name=["one", "two"],
            provider=["openai"],
            model=["model-a", "model-b"],
            api_key=["key"],
            base_url=None,
            thinking_effort=["high"],
        )
        configs = bench_runner.build_endpoint_configs(args)
        self.assertEqual([config.provider for config in configs], ["openai", "openai"])
        self.assertEqual([config.api_key for config in configs], ["key", "key"])
        self.assertEqual(
            [config.base_url for config in configs],
            ["https://api.openai.com/v1", "https://api.openai.com/v1"],
        )

    def test_cerebras_gets_official_default_base_url(self):
        args = argparse.Namespace(
            endpoint_name=["cerebras"], provider=["cerebras"],
            model=["zai-glm-4.7"], api_key=["key"], base_url=None,
            thinking_effort=None,
        )
        config = bench_runner.build_endpoint_configs(args)[0]
        self.assertEqual(config.base_url, "https://api.cerebras.ai/v1")

    def test_load_completed_runs_returns_resume_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=bench_runner.CSV_FIELDS)
                writer.writeheader()
                writer.writerow(
                    {
                        "endpoint_name": "openai-fast",
                        "model": "gpt-test",
                        "problem_id": "problem-a",
                        "run_idx": 2,
                        "config_fingerprint": "fingerprint-a",
                        "request_status": "completed",
                    }
                )

            self.assertEqual(
                bench_runner.load_completed_runs(path),
                {("openai-fast", "gpt-test", "problem-a", 2, "fingerprint-a")},
            )

    def test_load_completed_runs_accepts_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.csv"
            self.assertEqual(bench_runner.load_completed_runs(path), set())

    def test_load_attempt_counts_counts_all_durable_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=bench_runner.CSV_FIELDS)
                writer.writeheader()
                for status in ("error", "incomplete"):
                    writer.writerow(
                        {
                            "endpoint_name": "endpoint",
                            "model": "model",
                            "problem_id": "problem",
                            "run_idx": 1,
                            "config_fingerprint": "fingerprint",
                            "request_status": status,
                        }
                    )
            counts = bench_runner.load_attempt_counts(path)
            self.assertEqual(
                counts[("endpoint", "model", "problem", 1, "fingerprint")],
                2,
            )

    def test_aggregate_summary_uses_ratio_of_sums(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=bench_runner.CSV_FIELDS)
                writer.writeheader()
                for run_idx, inference_s in ((1, 1.0), (2, 9.0)):
                    row = dict.fromkeys(bench_runner.CSV_FIELDS, "")
                    row.update(
                        endpoint_name="endpoint", provider="openai", model="model",
                        problem_id=f"problem-{run_idx}", run_idx=run_idx,
                        config_fingerprint="fingerprint", request_status="completed",
                        billed_output_tokens=100, reasoning_tokens=20,
                        inference_time_s=inference_s, tool_time_s=0,
                        provider_window_inference_time_s=inference_s,
                        retry_api_time_s=0, backoff_time_s=0,
                        harness_overhead_s=0, total_wall_s=inference_s,
                    )
                    writer.writerow(row)
            expected = {
                ("endpoint", "model", "problem-1", 1, "fingerprint"),
                ("endpoint", "model", "problem-2", 2, "fingerprint"),
            }
            summary_path = bench_runner.write_aggregate_summary(path, expected)
            summary = json.loads(summary_path.read_text())["endpoints"][0]
            self.assertEqual(summary["billed_output_tokens"], 200)
            self.assertEqual(summary["reasoning_tokens"], 40)
            self.assertEqual(summary["visible_output_tokens"], 160)
            self.assertEqual(summary["end_to_end_billed_tps"], 20.0)
            self.assertEqual(summary["provider_window_billed_tps"], 20.0)
            self.assertEqual(summary["provider_window_eligible_runs"], 2)
            self.assertEqual(summary["provider_window_coverage"], "complete")
            self.assertIsNone(summary["active_generation_billed_tps"])
            self.assertEqual(
                summary["averages_per_attempted_run"]["inference_time_s"], 5.0
            )

    def test_append_row_separates_file_without_trailing_newline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            path.write_text(
                ",".join(bench_runner.CSV_FIELDS)
                + "\nold,endpoint-a,model-a,problem-a,hard,1,1,1,1,1,1,1,True"
            )
            row = dict.fromkeys(bench_runner.CSV_FIELDS, "new")
            row.update(
                endpoint_name="endpoint-b",
                model="model-b",
                problem_id="problem-b",
                run_idx=2,
            )

            bench_runner.append_row(path, row)

            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["endpoint_name"], "endpoint-b")

    def test_balanced_endpoint_order_rotates_first_endpoint(self):
        configs = [SimpleNamespace(endpoint_name="a"), SimpleNamespace(endpoint_name="b")]
        self.assertEqual(
            [c.endpoint_name for c in bench_runner.balanced_endpoint_order(configs, 0, 1)],
            ["a", "b"],
        )
        self.assertEqual(
            [c.endpoint_name for c in bench_runner.balanced_endpoint_order(configs, 0, 2)],
            ["b", "a"],
        )
        self.assertEqual(
            [c.endpoint_name for c in bench_runner.balanced_endpoint_order(configs, 1, 1)],
            ["b", "a"],
        )

    def test_problem_range_expands(self):
        self.assertEqual(
            bench_runner.read_problem_ids("hard:2-4"),
            ["hard:2", "hard:3", "hard:4"],
        )

    def test_request_deadline_enforces_total_wall_time(self):
        with self.assertRaises(bench_runner.RequestDeadlineExceeded):
            with bench_runner.request_deadline(0.01):
                bench_runner.time.sleep(0.05)

    def test_multi_endpoint_comparison_requires_xhigh_for_all(self):
        args = argparse.Namespace(
            endpoint_name=["one", "two"], provider=["openai", "anthropic"],
            base_url=None, endpoint_region=None, api_key=["a", "b"],
            model=["model-a", "model-b"], thinking_effort=["xhigh", "max"],
        )
        configs = bench_runner.build_endpoint_configs(args)
        with (
            patch.object(bench_runner, "parse_args", return_value=args),
            patch.object(bench_runner, "build_endpoint_configs", return_value=configs),
            self.assertRaisesRegex(ValueError, "require --thinking-effort xhigh"),
        ):
            bench_runner.main()

    def test_main_records_inference_timeout_without_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "results.csv"
            args = argparse.Namespace(
                endpoint_name=["endpoint"], provider=["openai"], base_url=None,
                endpoint_region=None, api_key=["key"], model=["model"],
                thinking_effort=["xhigh"], anthropic_adaptive_thinking=True,
                timeout_seconds=1.0, problem_ids="hard:1", release_version="v6",
                max_tokens=100, runs=1, csv=csv_path, resume=False,
                checker_timeout=1, isolate_requests=False,
                max_attempts_per_tuple=3, artifacts_dir=None,
            )
            configs = bench_runner.build_endpoint_configs(args)
            problem = SimpleNamespace(
                question_id="problem-a",
                difficulty=SimpleNamespace(value="hard"),
            )
            with (
                patch.object(bench_runner, "parse_args", return_value=args),
                patch.object(bench_runner, "build_endpoint_configs", return_value=configs),
                patch.object(bench_runner, "resolve_problem_refs", return_value=["problem-a"]),
                patch.object(bench_runner, "build_problem_number_map", return_value={"problem-a": 1}),
                patch.object(bench_runner, "load_problems", return_value=[problem]),
                patch.object(bench_runner, "format_prompt_generation", return_value=[]),
                patch.object(bench_runner, "OpenAI", return_value=SimpleNamespace()),
                patch.object(
                    bench_runner, "stream_openai_response",
                    side_effect=bench_runner.RequestDeadlineExceeded("deadline"),
                ),
            ):
                return_code = bench_runner.main()
            self.assertEqual(return_code, 2)
            with csv_path.open(newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["request_status"], "inference_timeout")
            self.assertEqual(row["inference_outcome"], "inference_timeout")
            self.assertEqual(row["stop_reason"], "timeout")
            self.assertEqual(row["billed_output_tokens"], "")
            self.assertEqual(row["active_generation_billed_tps"], "")
            self.assertEqual(row["end_to_end_billed_tps"], "")
            self.assertGreater(float(row["retry_api_time_s"]), 0)

    def test_main_records_endpoint_error_and_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "results.csv"
            args = argparse.Namespace(
                endpoint_name=["broken", "working"],
                provider=["openai", "anthropic"],
                base_url=None,
                api_key=["key-a", "key-b"],
                model=["model-a", "model-b"],
                thinking_effort=["xhigh"],
                anthropic_adaptive_thinking=True,
                timeout_seconds=10.0,
                problem_ids="hard:1",
                release_version="v6",
                max_tokens=100,
                runs=1,
                csv=csv_path,
                resume=False,
                checker_timeout=1,
            )
            configs = bench_runner.build_endpoint_configs(args)
            problem = SimpleNamespace(
                question_id="problem-a",
                difficulty=SimpleNamespace(value="hard"),
            )
            successful_timing = {
                "output": "```python\nprint(1)\n```",
                "request_id": "msg_test",
                "ttft_s": 1.0,
                "first_stream_event_s": 0.5,
                "first_observable_output_s": 1.0,
                "last_observable_output_s": 3.0,
                "observable_chunk_count": 4,
                "gen_time_s": 2.0,
                "generation_wall_s": 2.5,
                "inference_time_s": 3.0,
                "total_wall_s": 3.0,
                "input_tokens": 4,
                "output_tokens": 5,
                "reasoning_tokens": 2,
                "request_status": "completed",
                "stop_reason": "end_turn",
            }
            with (
                patch.object(bench_runner, "parse_args", return_value=args),
                patch.object(bench_runner, "build_endpoint_configs", return_value=configs),
                patch.object(bench_runner, "resolve_problem_refs", return_value=["problem-a"]),
                patch.object(bench_runner, "build_problem_number_map", return_value={"problem-a": 1}),
                patch.object(bench_runner, "load_problems", return_value=[problem]),
                patch.object(bench_runner, "format_prompt_generation", return_value=[]),
                patch.object(bench_runner, "OpenAI", return_value=SimpleNamespace()),
                patch.object(bench_runner, "stream_openai_response", side_effect=RuntimeError("boom")),
                patch.object(bench_runner, "stream_anthropic_response", return_value=successful_timing),
                patch.object(bench_runner, "check_solution", return_value=True),
            ):
                return_code = bench_runner.main()

            self.assertEqual(return_code, 2)
            with csv_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["request_status"] for row in rows], ["error", "completed"])
            self.assertEqual(rows[0]["error_type"], "RuntimeError")
            self.assertEqual(rows[1]["passed"], "True")
            self.assertEqual(rows[1]["request_id"], "msg_test")
            self.assertEqual(rows[1]["inference_outcome"], "completed")
            self.assertEqual(rows[1]["billed_output_tokens"], "5")
            self.assertEqual(rows[1]["reasoning_tokens"], "2")
            self.assertEqual(rows[1]["visible_output_tokens"], "3")
            self.assertEqual(rows[1]["active_generation_billed_tps"], "")
            self.assertEqual(rows[1]["end_to_end_billed_tps"], "1.667")
            self.assertTrue(Path(rows[1]["artifact_path"]).exists())
            accounted = sum(
                float(rows[1][field])
                for field in (
                    "inference_time_s", "tool_time_s", "retry_api_time_s",
                    "backoff_time_s", "harness_overhead_s",
                )
            )
            self.assertAlmostEqual(accounted, float(rows[1]["total_wall_s"]), places=5)


if __name__ == "__main__":
    unittest.main()
