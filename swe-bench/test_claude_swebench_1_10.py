import json
from pathlib import Path

from claude_swebench_1_10 import run_batch, summarize
from claude_swebench_problem1 import (
    solve_process_succeeded,
    terminal_result_is_success,
)


def test_post_terminal_forced_teardown_preserves_valid_success():
    result = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_api_ms": 1234,
        "modelUsage": {"claude-fable-5": {"outputTokens": 10}},
    }
    assert terminal_result_is_success(result)
    assert solve_process_succeeded(-15, result, True)
    assert not solve_process_succeeded(-15, result, False)
    assert not solve_process_succeeded(-15, {**result, "is_error": True}, True)


def test_summarize_uses_ratio_of_sums_and_strict_eligibility():
    base = {
        "status": "completed_without_evaluation",
        "return_code": 0,
        "timing_coverage": "complete",
    }
    rows = [
        {**base, "request_active_seconds": 2.0, "billed_output_tokens": 20},
        {**base, "request_active_seconds": 8.0, "billed_output_tokens": 40},
        {**base, "timing_coverage": "partial", "request_active_seconds": 1.0, "billed_output_tokens": 1000},
    ]
    result = summarize(rows)
    assert result["complete_timing_problems"] == 2
    assert result["end_to_end_billed_tps_ratio_of_sums"] == 6.0


def test_batch_continues_after_failure_and_preserves_required_flags(tmp_path: Path):
    commands = []

    def load_problem(index):
        return {"instance_id": f"instance-{index:02d}", "image": f"image-{index:02d}"}

    def pull(_image, _env, _log):
        return 0

    def execute(command, _log, _env, _timeout):
        commands.append(command)
        run_dir = Path(command[command.index("--output") + 1])
        run_dir.mkdir()
        index = int(command[command.index("--index") + 1])
        metadata = {
            "status": "solve_failed" if index == 2 else "completed_without_evaluation",
            "problem": {"instance_id": f"instance-{index:02d}"},
            "requested_model": "claude-fable-5",
            "requested_effort": "xhigh",
            "inference_timing": {
                "coverage": "complete",
                "coverage_reasons": [],
                "terminal_request_active_seconds": 2,
                "end_to_end_billed_tps": 5,
                "all_model_terminal_billed_output_tokens": 10,
                "primary_output_tokens": 9,
                "wall_partition": {},
            },
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata))
        return 1 if index == 2 else 0

    output = tmp_path / "batch"
    batch = run_batch(
        output=output,
        python=Path("/python"),
        model="claude-fable-5",
        effort="xhigh",
        env={"SWE_BENCH_STATE_ROOT": str(tmp_path), "SWE_BENCH_MACHINE_ID": "test"},
        problem_loader=load_problem,
        command_runner=execute,
        image_puller=pull,
    )
    assert len(commands) == 10
    assert batch["status"] == "completed_with_failures"
    assert batch["aggregate"]["successful_problems"] == 9
    assert all("--require-complete-inference-timing" in command for command in commands)
    assert all("--skip-evaluation" in command for command in commands)
    assert all("--skip-pull" in command for command in commands)
    assert (output / "batch.csv").exists()
    assert (output / "batch.md").exists()


def test_dry_run_selects_exactly_first_ten_sorted_positions(tmp_path: Path):
    seen = []

    def load_problem(index):
        seen.append(index)
        return {"instance_id": f"sorted-{index:02d}", "image": "unused"}

    batch = run_batch(
        output=tmp_path / "dry",
        python=Path("/python"),
        model="claude-fable-5",
        effort="xhigh",
        env={"SWE_BENCH_STATE_ROOT": str(tmp_path), "SWE_BENCH_MACHINE_ID": "test"},
        dry_run=True,
        problem_loader=load_problem,
    )
    assert seen == list(range(10))
    assert batch["status"] == "dry_run"
    assert [row["question_number"] for row in batch["runs"]] == list(range(1, 11))
