#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable
from time import perf_counter

from datasets import load_dataset
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(os.environ.get("HLE_OUTPUT_DIR", ROOT))
HLE_REPO = ROOT / "hle"
HLE_PROMPT_MODULE = HLE_REPO / "hle_eval" / "run_model_predictions.py"
HLE_JUDGE_MODULE = HLE_REPO / "hle_eval" / "run_judge_results.py"
TOKENS_FILE = ROOT / "tokens.txt"
RESULTS_CSV = OUTPUT_ROOT / "results.csv"
JUDGMENTS_CSV = OUTPUT_ROOT / "judgments.csv"
RESPONSES_JSONL = OUTPUT_ROOT / "responses.jsonl"
RUN_SUMMARIES_JSONL = OUTPUT_ROOT / "run_summaries.jsonl"
CACHE_DIR = ROOT / ".cache" / "huggingface"
DATASET_NAME = "cais/hle"
DEFAULT_TIMEOUT_SECONDS = 3600.0
REFUSAL_FINISH_REASONS = {"refusal", "content_filter"}

RESULT_COLUMNS = [
    "timestamp",
    "endpoint_name",
    "model",
    "sorted_index",
    "question_id",
    "max_tokens",
    "run_idx",
    "ttft_s",
    "gen_time_s",
    "total_wall_s",
    "output_tokens",
    "token_count_method",
    "tokens_per_s",
    "input_tokens",
    "reasoning_tokens",
    "visible_output_tokens",
    "total_tokens",
    "usage_json",
    "correct",
    "finish_reason",
    "refusal",
    "provider",
    "request_id",
    "outcome",
    "billed_output_tokens",
    "thinking_tokens",
    "inference_time_s",
    "tool_time_s",
    "retry_api_time_s",
    "backoff_time_s",
    "harness_overhead_s",
    "first_stream_event_latency_s",
    "first_stream_event_ts",
    "first_observable_output_ts",
    "last_observable_output_ts",
    "observable_chunk_count",
    "endpoint_base_url",
    "endpoint_region",
    "tool_configuration",
    "sandbox_image",
    "cpu_memory_limits",
    "economy_policy",
    "serial_execution",
    "generation_start_s", "generation_start_event_type",
    "generation_start_event_detail", "generation_start_confidence",
    "hidden_reasoning_observability", "terminal_event_s",
    "observed_pre_generation_s", "active_generation_time_s",
    "active_generation_billed_tps", "end_to_end_billed_tps",
    "attempt_index", "usage_reconciliation_status", "headline_eligible",
    "headline_exclusion_reason",
]

RESULT_EXPLICIT_DEFAULTS = {
    "total_wall_s": "not_recorded",
    "reasoning_tokens": "not_reported",
    "output_tokens": "not_reported",
    "correct": "not_recorded",
    "finish_reason": "not_reported",
    "refusal": "not_reported",
    "provider": "not_recorded",
    "request_id": "not_reported",
    "outcome": "historical_not_recorded",
    "billed_output_tokens": "not_recorded",
    "thinking_tokens": "not_reported",
    "inference_time_s": "not_recorded",
    "tool_time_s": "not_recorded",
    "retry_api_time_s": "not_recorded",
    "backoff_time_s": "not_recorded",
    "harness_overhead_s": "not_recorded",
    "first_stream_event_latency_s": "not_recorded",
    "first_stream_event_ts": "not_recorded",
    "first_observable_output_ts": "not_recorded",
    "last_observable_output_ts": "not_recorded",
    "observable_chunk_count": "not_recorded",
    "endpoint_base_url": "not_recorded",
    "endpoint_region": "not_reported",
    "tool_configuration": "none",
    "sandbox_image": "not_applicable",
    "cpu_memory_limits": "not_applicable",
    "economy_policy": "none",
    "serial_execution": "yes",
    "generation_start_s": "not_recorded",
    "generation_start_event_type": "not_recorded",
    "generation_start_event_detail": "not_recorded",
    "generation_start_confidence": "unavailable",
    "hidden_reasoning_observability": "unavailable",
    "terminal_event_s": "not_recorded",
    "observed_pre_generation_s": "not_recorded",
    "active_generation_time_s": "not_recorded",
    "active_generation_billed_tps": "not_recorded",
    "end_to_end_billed_tps": "not_recorded",
    "attempt_index": "not_recorded",
    "usage_reconciliation_status": "unavailable",
    "headline_eligible": "not_recorded",
    "headline_exclusion_reason": "not_recorded",
}

JUDGMENT_COLUMNS = [
    "timestamp",
    "endpoint_name",
    "model",
    "sorted_index",
    "question_id",
    "max_tokens",
    "run_idx",
    "judge_model",
    "correct_answer",
    "model_answer",
    "correct",
    "confidence",
    "judge_parse_method",
    "reasoning",
    "result_timestamp",
]


@dataclass(frozen=True)
class Problem:
    sorted_index: int
    question_id: str
    question: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-shot timing harness for HLE text-only problems.")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        default=os.getenv("PROVIDER", "openai"),
        help="Model API provider. openai uses /responses; anthropic uses native /v1/messages.",
    )
    parser.add_argument("--endpoint-name", default="smoke", help="Label written to results.csv.")
    parser.add_argument("--base-url", default=os.getenv("BASE_URL"), help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=os.getenv("API_KEY"), help="Endpoint API key.")
    parser.add_argument("--model", default=os.getenv("MODEL"), help="Model name.")
    parser.add_argument(
        "--endpoint-region",
        default=os.getenv("ENDPOINT_REGION", "not_reported"),
        help="Endpoint region label recorded for comparison control auditing.",
    )
    parser.add_argument("--sandbox-image", default=os.getenv("SANDBOX_IMAGE", "not_applicable"))
    parser.add_argument(
        "--cpu-memory-limits",
        default=os.getenv("CPU_MEMORY_LIMITS", "not_applicable"),
    )
    parser.add_argument("--economy-policy", default=os.getenv("ECONOMY_POLICY", "none"))
    parser.add_argument("--max-tokens", type=int, default=100000, help="Max completion tokens.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("BENCH_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)),
        help="Per-request model timeout in seconds (default: 3600).",
    )
    parser.add_argument(
        "--thinking-budget-tokens",
        type=int,
        default=None,
        help="Enable Anthropic extended thinking with this token budget via extra_body.",
    )
    parser.add_argument(
        "--thinking-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default=os.getenv("THINKING_EFFORT", "xhigh"),
        help="Set reasoning effort (default: xhigh); use none to disable.",
    )
    parser.add_argument(
        "--anthropic-adaptive-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For Anthropic native calls, send thinking={type: adaptive, display: omitted}.",
    )
    parser.add_argument(
        "--omit-temperature",
        action="store_true",
        help="Do not send a temperature parameter. Required by some Anthropic models.",
    )
    parser.add_argument("--index", type=int, default=None, help="0-based index into the sorted text-only list.")
    parser.add_argument(
        "--question-ids",
        nargs="+",
        default=None,
        help="One or more explicit HLE question ids.",
    )
    parser.add_argument("--category", default=None, help="Filter by exact HLE category.")
    parser.add_argument("--raw-subject", default=None, help="Filter by exact HLE raw_subject.")
    parser.add_argument(
        "--num-questions",
        type=int,
        default=1,
        help="Take the first N questions from the sorted text-only list starting at --index.",
    )
    parser.add_argument("--runs", type=int, default=1, help="Repeat count per question.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip completed run identities and repair partial checkpoints "
            "(default: enabled; use --no-resume to force new requests)."
        ),
    )
    parser.add_argument(
        "--list-problems",
        action="store_true",
        help="List selected canonical problems without sending any model requests.",
    )
    parser.add_argument(
        "--print-response",
        action="store_true",
        help="Print the full model response for each run.",
    )
    parser.add_argument(
        "--no-print-question",
        action="store_true",
        help="Suppress printing full question text during runs.",
    )
    parser.add_argument("--judge", action="store_true", help="Judge each response after timing is recorded.")
    parser.add_argument(
        "--judge-after-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After the model batch finishes, judge only unjudged completed responses generated "
            "or resumed by this invocation "
            "(default: enabled; use --no-judge-after-run to skip)."
        ),
    )
    parser.add_argument(
        "--judge-existing",
        action="store_true",
        help="Judge matching existing responses.jsonl records without running the eval model.",
    )
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "gpt-5.5"), help="Judge model.")
    parser.add_argument(
        "--judge-base-url",
        default=os.getenv("JUDGE_BASE_URL"),
        help="OpenAI-compatible judge base URL. Defaults to --base-url.",
    )
    parser.add_argument(
        "--judge-api-key",
        default=os.getenv("JUDGE_API_KEY"),
        help="Judge API key. Defaults to --api-key.",
    )
    parser.add_argument("--judge-max-tokens", type=int, default=4096, help="Max completion tokens for judging.")
    parser.add_argument(
        "--judge-timeout-seconds",
        type=float,
        default=float(os.getenv("JUDGE_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)),
        help="Per-request judge timeout in seconds (default: 3600).",
    )
    parser.add_argument("--print-judge", action="store_true", help="Print full judge reasoning.")
    return parser.parse_args()


def ensure_hf_token() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(CACHE_DIR))
    os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_DIR / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR / "transformers"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if os.getenv("HF_TOKEN"):
        return
    if not TOKENS_FILE.exists():
        raise RuntimeError(f"HF token file not found: {TOKENS_FILE}")
    raw = TOKENS_FILE.read_text().strip()
    if not raw:
        raise RuntimeError(f"HF token file is empty: {TOKENS_FILE}")
    token = raw.split("=", 1)[1].strip() if "=" in raw else raw
    if not token:
        raise RuntimeError("HF token could not be parsed from tokens.txt")
    os.environ["HF_TOKEN"] = token


def load_python_module(module_name: str, module_path: Path):
    if not module_path.exists():
        raise RuntimeError(f"Module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_prompt_module():
    return load_python_module("hle_run_model_predictions", HLE_PROMPT_MODULE)


def load_judge_module():
    return load_python_module("hle_run_judge_results", HLE_JUDGE_MODULE)


def load_cached_hle_rows() -> list[dict[str, Any]] | None:
    arrow_paths = sorted((CACHE_DIR / "datasets" / "cais___hle" / "default").glob("*/*/hle-test.arrow"))
    if not arrow_paths:
        return None

    import pyarrow as pa
    import pyarrow.ipc as ipc

    with pa.memory_map(str(arrow_paths[-1]), "r") as source:
        table = ipc.open_stream(source).read_all()
    columns = ["id", "question", "image", "answer", "category", "raw_subject"]
    return table.select(columns).to_pylist()


def canonical_text_only_questions() -> list[Problem]:
    rows = load_cached_hle_rows()
    if rows is None:
        ensure_hf_token()
        rows = load_dataset(
            DATASET_NAME,
            split="test",
            streaming=True,
            cache_dir=str(CACHE_DIR / "datasets"),
            token=os.environ["HF_TOKEN"],
        )
    text_only_rows = [row for row in rows if not row["image"]]
    text_only_rows.sort(key=lambda row: row["id"])
    return [
        Problem(sorted_index=idx, question_id=row["id"], question=row)
        for idx, row in enumerate(text_only_rows)
    ]


def preview_text(text: str, width: int = 100) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= width else normalized[: width - 3] + "..."


def print_problem_listing(problems: Iterable[Problem]) -> None:
    for problem in problems:
        print(
            f"{problem.sorted_index}\t{problem.question_id}\t"
            f"{problem.question['category']}\t{problem.question['raw_subject']}\t"
            f"{preview_text(problem.question['question'])}"
        )


def select_problems(args: argparse.Namespace, problems: list[Problem]) -> list[Problem]:
    if not problems:
        raise RuntimeError("No text-only HLE questions were loaded.")

    by_id = {problem.question_id: problem for problem in problems}
    filtered_problems = [
        problem
        for problem in problems
        if (args.category is None or problem.question["category"] == args.category)
        and (args.raw_subject is None or problem.question["raw_subject"] == args.raw_subject)
    ]
    if not filtered_problems:
        filters = []
        if args.category is not None:
            filters.append(f"category={args.category!r}")
        if args.raw_subject is not None:
            filters.append(f"raw_subject={args.raw_subject!r}")
        raise RuntimeError(f"No text-only HLE questions match {' and '.join(filters)}.")
    selected: list[Problem] = []

    base_index = args.index if args.index is not None else 0
    if base_index < 0:
        raise ValueError("--index must be >= 0")
    if args.num_questions < 1:
        raise ValueError("--num-questions must be >= 1")

    if args.index is not None or not args.question_ids:
        end_index = base_index + args.num_questions
        if end_index > len(filtered_problems):
            raise IndexError(
                f"Requested filtered indices [{base_index}, {end_index}) exceed "
                f"{len(filtered_problems)} matching text-only problems."
            )
        selected.extend(filtered_problems[base_index:end_index])

    if args.question_ids:
        missing = [qid for qid in args.question_ids if qid not in by_id]
        if missing:
            raise KeyError(f"Unknown question ids: {', '.join(missing)}")
        selected.extend(by_id[qid] for qid in args.question_ids)

    deduped: list[Problem] = []
    seen = set()
    for problem in sorted(selected, key=lambda item: item.sorted_index):
        if problem.question_id in seen:
            continue
        seen.add(problem.question_id)
        deduped.append(problem)
    return deduped


def extract_delta_text(delta_content: Any) -> str:
    if delta_content is None:
        return ""
    if isinstance(delta_content, str):
        return delta_content
    if isinstance(delta_content, list):
        parts: list[str] = []
        for item in delta_content:
            if isinstance(item, str):
                parts.append(item)
                continue
            text = getattr(item, "text", None)
            if text:
                parts.append(text)
                continue
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(item["text"])
        return "".join(parts)
    return ""


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def format_ts(ts: datetime | None) -> str:
    return ts.isoformat() if ts else ""


def seconds_between(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return (end - start).total_seconds()


def resolve_token_count(text: str, model: str) -> tuple[int | None, str]:
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text)), f"tiktoken:{encoding.name}"
    except Exception:
        pass

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        return len(tokenizer.encode(text)), f"hf_tokenizer:{model}"
    except Exception:
        return None, "missing_usage_no_local_tokenizer"


def write_csv_header_if_needed() -> None:
    if not RESULTS_CSV.exists():
        with RESULTS_CSV.open("w", newline="") as handle:
            csv.writer(handle).writerow(RESULT_COLUMNS)
        return

    with RESULTS_CSV.open(newline="") as handle:
        header = next(csv.reader(handle), None)
    if header is None:
        with RESULTS_CSV.open("w", newline="") as handle:
            csv.writer(handle).writerow(RESULT_COLUMNS)
        return
    if header != RESULT_COLUMNS and header != RESULT_COLUMNS[: len(header)]:
        raise RuntimeError(f"Unsupported results.csv header: {header!r}")

    with RESULTS_CSV.open(newline="") as handle:
        rows = list(csv.reader(handle))
    responses_by_checkpoint: dict[tuple[tuple[str, str, str, str, str], str], dict[str, Any]] = {}
    if RESPONSES_JSONL.exists():
        for response in load_response_records():
            timestamp = str(response.get("timestamp", ""))
            if timestamp:
                responses_by_checkpoint[(run_identity(response), timestamp)] = response

    migrated_rows = [RESULT_COLUMNS]
    changed = header != RESULT_COLUMNS
    for row in rows[1:]:
        if len(row) > len(RESULT_COLUMNS):
            raise RuntimeError(f"results.csv row has {len(row)} columns; expected at most {len(RESULT_COLUMNS)}")
        migrated_row = row + [""] * (len(RESULT_COLUMNS) - len(row))
        record = dict(zip(RESULT_COLUMNS, migrated_row))
        response = responses_by_checkpoint.get((run_identity(record), str(record.get("timestamp", ""))))
        if (
            response is not None
            and record.get("token_count_method") == "anthropic_stream_usage"
            and response.get("thinking_effort") is not None
        ):
            migrated_row[RESULT_COLUMNS.index("token_count_method")] = (
                "anthropic_stream_usage_includes_omitted_thinking"
            )
            migrated_row[RESULT_COLUMNS.index("visible_output_tokens")] = ""
            changed = True
        if not record.get("finish_reason") and response is not None:
            migrated_row[RESULT_COLUMNS.index("finish_reason")] = response.get("finish_reason") or "not_reported"
            changed = True
        finish_reason = str(migrated_row[RESULT_COLUMNS.index("finish_reason")] or "not_reported")
        outcome_index = RESULT_COLUMNS.index("outcome")
        if not migrated_row[outcome_index]:
            migrated_row[outcome_index] = outcome_from_finish_reason(finish_reason)
            changed = True
        if response is not None and not migrated_row[RESULT_COLUMNS.index("provider")]:
            migrated_row[RESULT_COLUMNS.index("provider")] = response.get("provider") or "not_recorded"
            changed = True
        if not migrated_row[RESULT_COLUMNS.index("inference_time_s")]:
            migrated_row[RESULT_COLUMNS.index("inference_time_s")] = record.get("total_wall_s") or "not_recorded"
            changed = True
        for zero_column in ("tool_time_s", "retry_api_time_s", "backoff_time_s", "harness_overhead_s"):
            if not migrated_row[RESULT_COLUMNS.index(zero_column)]:
                migrated_row[RESULT_COLUMNS.index(zero_column)] = "0"
                changed = True
        if not migrated_row[RESULT_COLUMNS.index("thinking_tokens")]:
            migrated_row[RESULT_COLUMNS.index("thinking_tokens")] = record.get("reasoning_tokens") or "not_reported"
            changed = True
        billed_index = RESULT_COLUMNS.index("billed_output_tokens")
        if not migrated_row[billed_index] and record.get("usage_json"):
            try:
                historical_usage = json.loads(str(record["usage_json"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                historical_usage = {}
            if not historical_usage.get("provider_error") and record.get("output_tokens"):
                migrated_row[billed_index] = record["output_tokens"]
                changed = True
        expected_refusal = (
            "yes"
            if is_refusal_finish_reason(finish_reason)
            else "no"
            if finish_reason != "not_reported"
            else "not_reported"
        )
        if migrated_row[RESULT_COLUMNS.index("refusal")] != expected_refusal:
            migrated_row[RESULT_COLUMNS.index("refusal")] = expected_refusal
            changed = True
        for column, default in RESULT_EXPLICIT_DEFAULTS.items():
            column_index = RESULT_COLUMNS.index(column)
            if not migrated_row[column_index]:
                migrated_row[column_index] = default
                changed = True
        migrated_rows.append(migrated_row)

    if not changed:
        return

    temp_path = RESULTS_CSV.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="") as handle:
        csv.writer(handle).writerows(migrated_rows)
    temp_path.replace(RESULTS_CSV)


def run_identity(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(record.get("endpoint_name", "")),
        str(record.get("model", "")),
        str(record.get("question_id", "")),
        str(record.get("max_tokens", "")),
        str(record.get("run_idx", "")),
    )


def result_row_identity(row: list[Any]) -> tuple[str, str, str, str, str]:
    return run_identity(dict(zip(RESULT_COLUMNS, row)))


def load_result_records() -> list[dict[str, str]]:
    write_csv_header_if_needed()
    with RESULTS_CSV.open(newline="") as handle:
        return list(csv.DictReader(handle))


def optional_int(value: Any) -> int | None:
    if value in {None, "", "not_reported", "not_recorded"}:
        return None
    return int(value)


def optional_float(value: Any) -> float | None:
    if value in {None, "", "not_reported", "not_recorded"}:
        return None
    return float(value)


def is_refusal_finish_reason(finish_reason: Any) -> bool:
    return str(finish_reason) in REFUSAL_FINISH_REASONS


def outcome_from_finish_reason(finish_reason: Any) -> str:
    reason = str(finish_reason or "not_reported")
    if is_refusal_finish_reason(reason):
        return "refusal"
    if reason in {"length", "max_tokens", "max_output_tokens"}:
        return "incomplete_max_tokens"
    if reason in {"stop", "end_turn", "completed"}:
        return "completed"
    if reason in {"timeout", "inference_timeout"}:
        return "inference_timeout"
    if reason in {"failed", "cancelled", "incomplete"}:
        return f"incomplete_{reason}"
    return "api_error" if reason == "api_error" else "unknown"


def headline_eligibility(record: dict[str, Any]) -> tuple[bool, str]:
    """Apply the strict request-dispatch-to-terminal aggregate rules."""
    if record.get("outcome") != "completed":
        return False, f"outcome:{record.get('outcome', 'unknown')}"
    if record.get("billed_output_tokens") is None:
        return False, "missing_final_usage"
    duration = optional_float(record.get("inference_time_s"))
    if duration is None or duration <= 0:
        return False, "missing_or_nonpositive_request_duration"
    if record.get("usage_reconciliation_status") != "matched":
        return False, f"usage_reconciliation:{record.get('usage_reconciliation_status', 'unavailable')}"
    return True, "eligible"


def result_to_report_record(record: dict[str, Any]) -> dict[str, Any]:
    inference_time_s = optional_float(record.get("inference_time_s"))
    if inference_time_s is None and record.get("outcome") == "historical_not_recorded":
        inference_time_s = optional_float(record.get("total_wall_s"))
    return {
        "total_wall_s": optional_float(record.get("total_wall_s")) or 0.0,
        "inference_time_s": inference_time_s or 0.0,
        "tool_time_s": optional_float(record.get("tool_time_s")) or 0.0,
        "retry_api_time_s": optional_float(record.get("retry_api_time_s")) or 0.0,
        "backoff_time_s": optional_float(record.get("backoff_time_s")) or 0.0,
        "harness_overhead_s": optional_float(record.get("harness_overhead_s")) or 0.0,
        "reasoning_tokens": optional_int(record.get("reasoning_tokens")),
        "visible_output_tokens": optional_int(record.get("visible_output_tokens")),
        "output_tokens": optional_int(record.get("output_tokens")),
        "billed_output_tokens": optional_int(record.get("billed_output_tokens")),
        "outcome": str(record.get("outcome", "historical_not_recorded")),
        "correct": str(record.get("correct", "not_recorded")),
        "finish_reason": str(record.get("finish_reason", "not_reported")),
        "refusal": str(record.get("refusal", "not_reported")),
        "ttft_s": optional_float(record.get("ttft_s")),
        "active_generation_time_s": optional_float(record.get("active_generation_time_s")),
        "usage_reconciliation_status": str(
            record.get("usage_reconciliation_status", "unavailable")
        ),
    }


def result_needs_judgment(record: dict[str, Any]) -> bool:
    return str(record.get("correct", "not_recorded")) not in {"yes", "no"}


def completed_run_records() -> dict[
    tuple[str, str, str, str, str], tuple[dict[str, Any], dict[str, Any]]
]:
    responses_by_checkpoint: dict[tuple[tuple[str, str, str, str, str], str], dict[str, Any]] = {}
    for response in load_response_records():
        timestamp = str(response.get("timestamp", ""))
        if timestamp:
            responses_by_checkpoint[(run_identity(response), timestamp)] = response

    completed: dict[tuple[str, str, str, str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for result in load_result_records():
        timestamp = str(result.get("timestamp", ""))
        response = responses_by_checkpoint.get((run_identity(result), timestamp))
        try:
            has_wall_time = float(result.get("total_wall_s", "")) >= 0
        except (TypeError, ValueError):
            has_wall_time = False
        outcome = str(result.get("outcome", "historical_not_recorded"))
        resumable_completion = outcome not in {"inference_timeout", "api_error"}
        if response is not None and has_wall_time and resumable_completion:
            completed[run_identity(result)] = (result, response)
    return completed


def write_result_row(row: list[Any], *, replace_identity: bool = False) -> None:
    write_csv_header_if_needed()
    if len(row) != len(RESULT_COLUMNS):
        raise RuntimeError(f"Result row has {len(row)} columns; expected {len(RESULT_COLUMNS)}")
    row = list(row)
    for column, default in RESULT_EXPLICIT_DEFAULTS.items():
        column_index = RESULT_COLUMNS.index(column)
        if row[column_index] in {None, ""}:
            row[column_index] = default
    if replace_identity:
        with RESULTS_CSV.open(newline="") as handle:
            rows = list(csv.reader(handle))
        identity = result_row_identity(row)
        rows = [rows[0], *(existing for existing in rows[1:] if result_row_identity(existing) != identity), row]
        temp_path = RESULTS_CSV.with_suffix(".csv.tmp")
        with temp_path.open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)
        temp_path.replace(RESULTS_CSV)
        return
    with RESULTS_CSV.open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(row)


def append_result_row(row: list[Any]) -> None:
    write_result_row(row)


def display_token_count(value: int | None) -> int | str:
    return value if value is not None else "not_reported"


def aggregate_token_count(records: list[dict[str, Any]], field: str) -> int | str:
    values = [record.get(field) for record in records]
    if not values or any(value is None for value in values):
        return "not_reported"
    return sum(int(value) for value in values)


def format_correctness(records: list[dict[str, Any]]) -> str:
    statuses = [str(record.get("correct", "not_recorded")) for record in records]
    judged = sum(status in {"yes", "no"} for status in statuses)
    correct = statuses.count("yes")
    if judged == len(statuses) and statuses:
        return f"{correct}/{judged}"
    if judged:
        return f"{correct}/{judged}_judged"
    if statuses and len(set(statuses)) == 1:
        return statuses[0]
    return "not_recorded"


def format_refusals(records: list[dict[str, Any]]) -> str:
    statuses = [str(record.get("refusal", "not_reported")) for record in records]
    reported = sum(status in {"yes", "no"} for status in statuses)
    refusals = statuses.count("yes")
    if reported == len(statuses) and statuses:
        return f"{refusals}/{reported}"
    if reported:
        return f"{refusals}/{reported}_reported"
    return "not_reported"


def print_run_report(*, endpoint_name: str, model: str, records: list[dict[str, Any]]) -> None:
    attempted_runs = len(records)
    total_wall_s = sum(float(record.get("total_wall_s", 0.0)) for record in records)
    inference_time_s = sum(float(record.get("inference_time_s", 0.0)) for record in records)
    tool_time_s = sum(float(record.get("tool_time_s", 0.0)) for record in records)
    retry_api_time_s = sum(float(record.get("retry_api_time_s", 0.0)) for record in records)
    backoff_time_s = sum(float(record.get("backoff_time_s", 0.0)) for record in records)
    harness_overhead_s = sum(float(record.get("harness_overhead_s", 0.0)) for record in records)
    completed = [record for record in records if headline_eligibility(record)[0]]
    billed_output_tokens = sum(int(record["billed_output_tokens"]) for record in completed)
    billed_inference_time_s = sum(float(record["inference_time_s"]) for record in completed)
    aggregate_end_to_end_tps = (
        billed_output_tokens / billed_inference_time_s if billed_inference_time_s > 0 else None
    )
    active_completed = [record for record in completed if record.get("active_generation_time_s") is not None]
    active_tokens = sum(int(record["billed_output_tokens"]) for record in active_completed)
    active_seconds = sum(float(record["active_generation_time_s"]) for record in active_completed)
    aggregate_active_tps = active_tokens / active_seconds if active_seconds > 0 else None
    ttfts = [float(record["ttft_s"]) for record in completed if record.get("ttft_s") is not None]
    median_ttft_s = statistics.median(ttfts) if ttfts else None
    reconciliation: dict[str, int] = {}
    exclusions: dict[str, int] = {}
    outcomes: dict[str, int] = {}
    for record in records:
        outcome = str(record.get("outcome", "unknown"))
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        status = str(record.get("usage_reconciliation_status", "unavailable"))
        reconciliation[status] = reconciliation.get(status, 0) + 1
        eligible, reason = headline_eligibility(record)
        if not eligible:
            exclusions[reason] = exclusions.get(reason, 0) + 1

    def average(value: float) -> float | None:
        return value / attempted_runs if attempted_runs else None

    def average_tokens(field: str) -> str:
        total = aggregate_token_count(records, field)
        if isinstance(total, str) or not attempted_runs:
            return "not_reported"
        return format_float(total / attempted_runs)

    correct_tasks = sum(str(record.get("correct")) == "yes" for record in records)

    print(
        "run_report "
        f"endpoint={endpoint_name} model={model} attempted_runs={attempted_runs} "
        f"completed_inference_calls={len(completed)} "
        f"billed_output_tokens={billed_output_tokens} "
        f"completed_inference_time_s={format_float(billed_inference_time_s)} "
        f"end_to_end_billed_tps={format_float(aggregate_end_to_end_tps)} "
        f"active_generation_billed_tps={format_float(aggregate_active_tps)} "
        f"median_ttft_s={format_float(median_ttft_s)} "
        f"usage_reconciliation={json.dumps(reconciliation, sort_keys=True, separators=(',', ':'))} "
        f"headline_exclusions={json.dumps(exclusions, sort_keys=True, separators=(',', ':'))} "
        f"reasoning_tokens={aggregate_token_count(records, 'reasoning_tokens')} "
        f"visible_output_tokens={aggregate_token_count(records, 'visible_output_tokens')} "
        f"output_tokens={aggregate_token_count(records, 'output_tokens')} "
        f"total_inference_time_s={format_float(inference_time_s)} "
        f"total_tool_time_s={format_float(tool_time_s)} "
        f"total_retry_api_time_s={format_float(retry_api_time_s)} "
        f"total_backoff_time_s={format_float(backoff_time_s)} "
        f"total_harness_overhead_s={format_float(harness_overhead_s)} "
        f"total_wall_time_s={format_float(total_wall_s)} "
        f"avg_billed_output_tokens_per_attempt={format_float(billed_output_tokens / attempted_runs if attempted_runs else None)} "
        f"avg_reasoning_tokens_per_attempt={average_tokens('reasoning_tokens')} "
        f"avg_visible_output_tokens_per_attempt={average_tokens('visible_output_tokens')} "
        f"avg_output_tokens_per_attempt={average_tokens('output_tokens')} "
        f"avg_inference_time_s={format_float(average(inference_time_s))} "
        f"avg_tool_time_s={format_float(average(tool_time_s))} "
        f"avg_retry_api_time_s={format_float(average(retry_api_time_s))} "
        f"avg_backoff_time_s={format_float(average(backoff_time_s))} "
        f"avg_harness_overhead_s={format_float(average(harness_overhead_s))} "
        f"avg_total_wall_time_s={format_float(average(total_wall_s))} "
        f"outcomes={json.dumps(outcomes, sort_keys=True, separators=(',', ':'))} "
        f"task_success={correct_tasks}/{attempted_runs} "
        f"correctness={format_correctness(records)} "
        f"refusals={format_refusals(records)}"
    )


def update_result_correctness(
    *,
    endpoint_name: str,
    model: str,
    question_id: str,
    max_tokens: int,
    run_idx: Any,
    correct: str,
    result_timestamp: str | None = None,
) -> None:
    write_csv_header_if_needed()
    with RESULTS_CSV.open(newline="") as handle:
        rows = list(csv.reader(handle))

    matches = []
    for index, row in enumerate(rows[1:], start=1):
        if (
            row[1] == str(endpoint_name)
            and row[2] == str(model)
            and row[4] == str(question_id)
            and row[5] == str(max_tokens)
            and row[6] == str(run_idx)
            and (result_timestamp is None or row[0] == result_timestamp)
        ):
            matches.append(index)
    if not matches:
        raise RuntimeError(
            "No matching results.csv row for judgment: "
            f"endpoint={endpoint_name!r} model={model!r} question_id={question_id!r} "
            f"max_tokens={max_tokens!r} run_idx={run_idx!r} timestamp={result_timestamp!r}"
        )

    rows[matches[-1]][RESULT_COLUMNS.index("correct")] = correct
    temp_path = RESULTS_CSV.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    temp_path.replace(RESULTS_CSV)


def usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "dict"):
        return usage.dict()
    if isinstance(usage, dict):
        return usage
    return {}


def nested_get(data: dict[str, Any], *path: str) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def extract_usage_fields(usage: Any) -> dict[str, Any]:
    data = usage_to_dict(usage)
    input_tokens = data.get("prompt_tokens") or data.get("input_tokens")
    output_tokens = data.get("completion_tokens") or data.get("output_tokens")
    total_tokens = data.get("total_tokens")
    reasoning_tokens = (
        nested_get(data, "completion_tokens_details", "reasoning_tokens")
        or nested_get(data, "output_tokens_details", "reasoning_tokens")
        or data.get("reasoning_tokens")
    )
    visible_output_tokens = None
    if output_tokens is not None and reasoning_tokens is not None:
        visible_output_tokens = max(0, output_tokens - reasoning_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "reasoning_tokens": reasoning_tokens,
        "visible_output_tokens": visible_output_tokens,
        "usage_json": json.dumps(data, sort_keys=True) if data else "",
    }


def response_contains_refusal(response: Any) -> bool:
    for output_item in getattr(response, "output", None) or []:
        for content_item in getattr(output_item, "content", None) or []:
            content_type = getattr(content_item, "type", None)
            if content_type is None and isinstance(content_item, dict):
                content_type = content_item.get("type")
            if content_type == "refusal":
                return True
    return False


def response_input_from_messages(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    instructions: str | None = None
    input_items: list[dict[str, Any]] = []

    for message in messages:
        role = message["role"]
        content = message["content"]
        if isinstance(content, str):
            if role in {"system", "developer"} and instructions is None:
                instructions = content
                continue
            input_items.append({"role": role, "content": [{"type": "input_text", "text": content}]})
            continue

        parts: list[dict[str, Any]] = []
        for item in content:
            if item.get("type") == "text":
                parts.append({"type": "input_text", "text": item["text"]})
            elif item.get("type") == "image_url":
                parts.append({"type": "input_image", "image_url": item["image_url"]["url"]})
            else:
                raise RuntimeError(f"Unsupported message content type for Responses API: {item.get('type')}")
        input_items.append({"role": role, "content": parts})

    return instructions, input_items


def anthropic_input_from_messages(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
    system: str | None = None
    anthropic_messages: list[dict[str, Any]] = []

    for message in messages:
        role = message["role"]
        content = message["content"]
        if role in {"system", "developer"}:
            system = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            continue
        if role != "user":
            raise RuntimeError(f"Unsupported Anthropic message role: {role}")

        if isinstance(content, str):
            anthropic_messages.append({"role": "user", "content": content})
            continue

        parts: list[dict[str, Any]] = []
        for item in content:
            if item.get("type") == "text":
                parts.append({"type": "text", "text": item["text"]})
            elif item.get("type") == "image_url":
                url = item["image_url"]["url"]
                if not url.startswith("data:"):
                    raise RuntimeError("Anthropic native image inputs require data URLs in this harness.")
                header, data = url.split(",", 1)
                media_type = header.removeprefix("data:").split(";", 1)[0]
                parts.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        },
                    }
                )
            else:
                raise RuntimeError(f"Unsupported message content type for Anthropic API: {item.get('type')}")
        anthropic_messages.append({"role": "user", "content": parts})

    return system, anthropic_messages


def write_judgments_header_if_needed() -> None:
    if not JUDGMENTS_CSV.exists():
        with JUDGMENTS_CSV.open("w", newline="") as handle:
            csv.writer(handle).writerow(JUDGMENT_COLUMNS)
        return

    with JUDGMENTS_CSV.open(newline="") as handle:
        rows = list(csv.reader(handle))
    header = rows[0] if rows else []
    if header == JUDGMENT_COLUMNS:
        return
    if header != JUDGMENT_COLUMNS[: len(header)]:
        raise RuntimeError(f"Unsupported judgments.csv header: {header!r}")

    migrated_rows = [JUDGMENT_COLUMNS]
    for row in rows[1:]:
        if len(row) > len(JUDGMENT_COLUMNS):
            raise RuntimeError(
                f"judgments.csv row has {len(row)} columns; expected at most {len(JUDGMENT_COLUMNS)}"
            )
        migrated_rows.append(row + [""] * (len(JUDGMENT_COLUMNS) - len(row)))
    temp_path = JUDGMENTS_CSV.with_suffix(".csv.tmp")
    with temp_path.open("w", newline="") as handle:
        csv.writer(handle).writerows(migrated_rows)
    temp_path.replace(JUDGMENTS_CSV)


def judgment_row_identity(row: list[Any]) -> tuple[str, str, str, str, str, str, str]:
    record = dict(zip(JUDGMENT_COLUMNS, row))
    return (
        *run_identity(record),
        str(record.get("judge_model", "")),
        str(record.get("result_timestamp", "")),
    )


def append_judgment_row(row: list[Any], *, replace_identity: bool = False) -> None:
    write_judgments_header_if_needed()
    if len(row) != len(JUDGMENT_COLUMNS):
        raise RuntimeError(f"Judgment row has {len(row)} columns; expected {len(JUDGMENT_COLUMNS)}")
    if replace_identity:
        with JUDGMENTS_CSV.open(newline="") as handle:
            rows = list(csv.reader(handle))
        identity = judgment_row_identity(row)
        rows = [
            rows[0],
            *(existing for existing in rows[1:] if judgment_row_identity(existing) != identity),
            row,
        ]
        temp_path = JUDGMENTS_CSV.with_suffix(".csv.tmp")
        with temp_path.open("w", newline="") as handle:
            csv.writer(handle).writerows(rows)
        temp_path.replace(JUDGMENTS_CSV)
        return
    with JUDGMENTS_CSV.open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(row)


def append_response_record(record: dict[str, Any]) -> None:
    with RESPONSES_JSONL.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_run_summary(record: dict[str, Any]) -> None:
    with RUN_SUMMARIES_JSONL.open("a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_response_records() -> list[dict[str, Any]]:
    if not RESPONSES_JSONL.exists():
        return []
    records = []
    with RESPONSES_JSONL.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def failed_inference_result(
    *,
    request_sent_ts: datetime,
    request_sent_perf: float,
    response_text_parts: list[str],
    finish_reason: str,
    request_id: str | None = None,
    first_stream_event_ts: datetime | None = None,
    first_stream_event_perf: float | None = None,
    first_token_ts: datetime | None = None,
    last_token_ts: datetime | None = None,
    first_token_perf: float | None = None,
    last_token_perf: float | None = None,
    observable_chunk_count: int = 0,
) -> dict[str, Any]:
    ended_perf = perf_counter()
    return {
        "request_sent_ts": request_sent_ts,
        "first_token_ts": first_token_ts,
        "last_token_ts": last_token_ts,
        "first_stream_event_ts": first_stream_event_ts,
        "response_text": "".join(response_text_parts),
        "output_tokens": None,
        "billed_output_tokens": None,
        "input_tokens": None,
        "reasoning_tokens": None,
        "visible_output_tokens": None,
        "total_tokens": None,
        "usage_json": "",
        "token_count_method": "incomplete_no_final_usage",
        "ttft_s": None if first_token_perf is None else first_token_perf - request_sent_perf,
        "first_stream_event_latency_s": (
            None
            if first_stream_event_perf is None
            else first_stream_event_perf - request_sent_perf
        ),
        "gen_time_s": (
            None
            if first_token_perf is None or last_token_perf is None
            else last_token_perf - first_token_perf
        ),
        "api_call_wall_s": ended_perf - request_sent_perf,
        "tokens_per_s": None,
        "finish_reason": finish_reason,
        "outcome": outcome_from_finish_reason(finish_reason),
        "request_id": request_id,
        "observable_chunk_count": observable_chunk_count,
        "usage_reconciliation_status": "unavailable",
    }


def reconcile_final_usage(billed_output_tokens: int | None, usage_json: str) -> str:
    if billed_output_tokens is None or not usage_json:
        return "unavailable"
    try:
        authoritative = json.loads(usage_json).get("output_tokens")
        return "matched" if int(authoritative) == int(billed_output_tokens) else "mismatched"
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return "unavailable"


def stream_completion(
    client: OpenAI,
    model: str,
    max_tokens: int,
    messages: list[dict[str, Any]],
    thinking_budget_tokens: int | None = None,
    thinking_effort: str | None = None,
    omit_temperature: bool = False,
) -> dict[str, Any]:
    request_sent_ts = now_utc()
    request_sent_perf = perf_counter()
    response_text_parts: list[str] = []
    first_stream_event_ts: datetime | None = None
    first_stream_event_perf: float | None = None
    first_token_ts: datetime | None = None
    last_token_ts: datetime | None = None
    first_token_perf: float | None = None
    last_token_perf: float | None = None
    output_tokens: int | None = None
    input_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    visible_output_tokens: int | None = None
    usage_json = ""
    final_usage_received = False
    request_id: str | None = None
    observable_chunk_count = 0
    generation_start_perf = None
    generation_start_event_type = None
    generation_start_event_detail = None
    generation_start_confidence = "unavailable"
    hidden_reasoning_observability = "unavailable"
    token_count_method = (
        "stream_usage_includes_thinking"
        if thinking_budget_tokens is not None or thinking_effort is not None
        else "stream_usage"
    )
    finish_reason: str | None = None

    try:
        instructions, response_input = response_input_from_messages(messages)
        request_kwargs = {
            "model": model,
            "input": response_input,
            "max_output_tokens": max_tokens,
            "stream": True,
        }
        if instructions:
            request_kwargs["instructions"] = instructions
        if not omit_temperature:
            request_kwargs["temperature"] = 1 if thinking_budget_tokens is not None else 0
        extra_body = {}
        if thinking_budget_tokens is not None:
            extra_body["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget_tokens}
        if thinking_effort is not None:
            request_kwargs["reasoning"] = {"effort": thinking_effort}
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        # The request clock starts immediately before SDK dispatch. Prompt shaping
        # and client construction are harness overhead, not API request time.
        request_sent_ts = now_utc()
        request_sent_perf = perf_counter()
        stream = client.responses.create(**request_kwargs)
        stream_response = getattr(stream, "response", None)
        if stream_response is not None:
            request_id = stream_response.headers.get("x-request-id")
        for event in stream:
            if first_stream_event_ts is None:
                first_stream_event_ts = now_utc()
                first_stream_event_perf = perf_counter()
            event_type = getattr(event, "type", "")
            if event_type == "response.output_item.added" and generation_start_perf is None:
                item_type = getattr(getattr(event, "item", None), "type", None)
                if item_type == "reasoning":
                    generation_start_perf = perf_counter()
                    generation_start_event_type = event_type
                    generation_start_event_detail = "item.type=reasoning"
                    generation_start_confidence = "provider_boundary"
                    hidden_reasoning_observability = "phase_boundary_only"
            if event_type.startswith("response.refusal"):
                finish_reason = "refusal"
            if event_type in {"response.output_text.delta", "response.refusal.delta"}:
                text = getattr(event, "delta", "")
                if not text:
                    continue
                if generation_start_perf is None:
                    generation_start_perf = perf_counter()
                    generation_start_event_type = event_type
                    generation_start_event_detail = "first non-empty generated delta"
                    generation_start_confidence = "delta_fallback"
                    hidden_reasoning_observability = "not_observed"
                ts = now_utc()
                if first_token_ts is None:
                    first_token_ts = ts
                    first_token_perf = perf_counter()
                last_token_ts = ts
                last_token_perf = perf_counter()
                response_text_parts.append(text)
                observable_chunk_count += 1
                continue

            response = getattr(event, "response", None)
            if response is not None:
                request_id = getattr(response, "_request_id", None) or request_id
                chunk_usage = getattr(response, "usage", None)
                if chunk_usage is not None:
                    usage_fields = extract_usage_fields(chunk_usage)
                    input_tokens = usage_fields["input_tokens"]
                    output_tokens = usage_fields["output_tokens"]
                    total_tokens = usage_fields["total_tokens"]
                    reasoning_tokens = usage_fields["reasoning_tokens"]
                    visible_output_tokens = usage_fields["visible_output_tokens"]
                    usage_json = usage_fields["usage_json"]
                    if event_type in {
                        "response.completed",
                        "response.failed",
                        "response.incomplete",
                        "response.cancelled",
                    }:
                        final_usage_received = True

                if response_contains_refusal(response):
                    finish_reason = "refusal"
                status = getattr(response, "status", None)
                if status == "completed" and not is_refusal_finish_reason(finish_reason):
                    finish_reason = "stop"
                elif status in {"failed", "incomplete", "cancelled"}:
                    incomplete_details = getattr(response, "incomplete_details", None)
                    finish_reason = getattr(incomplete_details, "reason", None) or status

            request_id = getattr(event, "request_id", None) or request_id
            chunk_usage = getattr(event, "usage", None)
            if chunk_usage is not None:
                usage_fields = extract_usage_fields(chunk_usage)
                input_tokens = usage_fields["input_tokens"]
                output_tokens = usage_fields["output_tokens"]
                total_tokens = usage_fields["total_tokens"]
                reasoning_tokens = usage_fields["reasoning_tokens"]
                visible_output_tokens = usage_fields["visible_output_tokens"]
                usage_json = usage_fields["usage_json"]
                if event_type in {
                    "response.completed",
                    "response.failed",
                    "response.incomplete",
                    "response.cancelled",
                }:
                    final_usage_received = True
    except APITimeoutError:
        return failed_inference_result(
            request_sent_ts=request_sent_ts,
            request_sent_perf=request_sent_perf,
            response_text_parts=response_text_parts,
            finish_reason="inference_timeout",
            request_id=request_id,
            first_stream_event_ts=first_stream_event_ts,
            first_stream_event_perf=first_stream_event_perf,
            first_token_ts=first_token_ts,
            last_token_ts=last_token_ts,
            first_token_perf=first_token_perf,
            last_token_perf=last_token_perf,
            observable_chunk_count=observable_chunk_count,
        )
    except APIConnectionError:
        return failed_inference_result(
            request_sent_ts=request_sent_ts,
            request_sent_perf=request_sent_perf,
            response_text_parts=response_text_parts,
            finish_reason="api_error",
            request_id=request_id,
            first_stream_event_ts=first_stream_event_ts,
            first_stream_event_perf=first_stream_event_perf,
            first_token_ts=first_token_ts,
            last_token_ts=last_token_ts,
            first_token_perf=first_token_perf,
            last_token_perf=last_token_perf,
            observable_chunk_count=observable_chunk_count,
        )
    except APIStatusError as exc:
        return failed_inference_result(
            request_sent_ts=request_sent_ts,
            request_sent_perf=request_sent_perf,
            response_text_parts=response_text_parts,
            finish_reason="api_error",
            request_id=getattr(exc, "request_id", None) or request_id,
            first_stream_event_ts=first_stream_event_ts,
            first_stream_event_perf=first_stream_event_perf,
            first_token_ts=first_token_ts,
            last_token_ts=last_token_ts,
            first_token_perf=first_token_perf,
            last_token_perf=last_token_perf,
            observable_chunk_count=observable_chunk_count,
        )

    last_event_ts = now_utc()
    last_event_perf = perf_counter()
    response_text = "".join(response_text_parts)

    if output_tokens is None and finish_reason not in {"inference_timeout", "api_error"}:
        output_tokens, token_count_method = resolve_token_count(response_text, model)
        visible_output_tokens = output_tokens

    outcome = outcome_from_finish_reason(finish_reason)
    billed_output_tokens = output_tokens if final_usage_received else None
    usage_reconciliation_status = reconcile_final_usage(billed_output_tokens, usage_json)

    ttft_s = None if first_token_perf is None else first_token_perf - request_sent_perf
    gen_time_s = None if first_token_perf is None or last_token_perf is None else last_token_perf - first_token_perf
    if gen_time_s == 0 and output_tokens:
        gen_time_s = 0.0
    api_call_wall_s = last_event_perf - request_sent_perf
    tokens_per_s = None
    if outcome == "completed" and api_call_wall_s > 0 and billed_output_tokens is not None:
        tokens_per_s = billed_output_tokens / api_call_wall_s
    active_generation_time_s = last_event_perf - generation_start_perf if generation_start_perf is not None else None
    active_generation_billed_tps = billed_output_tokens / active_generation_time_s if outcome == "completed" and billed_output_tokens is not None and active_generation_time_s and active_generation_time_s > 0 else None

    return {
        "request_sent_ts": request_sent_ts,
        "first_token_ts": first_token_ts,
        "last_token_ts": last_token_ts,
        "first_stream_event_ts": first_stream_event_ts,
        "response_text": response_text,
        "output_tokens": output_tokens,
        "billed_output_tokens": billed_output_tokens,
        "input_tokens": input_tokens,
        "reasoning_tokens": reasoning_tokens,
        "visible_output_tokens": visible_output_tokens,
        "total_tokens": total_tokens,
        "usage_json": usage_json,
        "token_count_method": token_count_method,
        "ttft_s": ttft_s,
        "first_stream_event_latency_s": (
            None
            if first_stream_event_perf is None
            else first_stream_event_perf - request_sent_perf
        ),
        "gen_time_s": gen_time_s,
        "api_call_wall_s": api_call_wall_s,
        "tokens_per_s": tokens_per_s,
        "generation_start_s": generation_start_perf - request_sent_perf if generation_start_perf is not None else None,
        "generation_start_event_type": generation_start_event_type,
        "generation_start_event_detail": generation_start_event_detail,
        "generation_start_confidence": generation_start_confidence,
        "hidden_reasoning_observability": hidden_reasoning_observability,
        "terminal_event_s": api_call_wall_s,
        "observed_pre_generation_s": generation_start_perf - request_sent_perf if generation_start_perf is not None else None,
        "active_generation_time_s": active_generation_time_s,
        "active_generation_billed_tps": active_generation_billed_tps,
        "end_to_end_billed_tps": tokens_per_s,
        "finish_reason": finish_reason,
        "outcome": outcome,
        "request_id": request_id,
        "observable_chunk_count": observable_chunk_count,
        "usage_reconciliation_status": usage_reconciliation_status,
    }


def stream_anthropic_completion(
    api_key: str,
    model: str,
    max_tokens: int,
    messages: list[dict[str, Any]],
    thinking_effort: str | None = None,
    adaptive_thinking: bool = True,
    omit_temperature: bool = False,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic package is required for --provider anthropic") from exc

    request_sent_ts = now_utc()
    request_sent_perf = perf_counter()
    response_text_parts: list[str] = []
    first_stream_event_ts: datetime | None = None
    first_stream_event_perf: float | None = None
    first_token_ts: datetime | None = None
    last_token_ts: datetime | None = None
    first_token_perf: float | None = None
    last_token_perf: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    total_tokens: int | None = None
    usage_json = ""
    finish_reason: str | None = None
    final_usage_received = False
    request_id: str | None = None
    observable_chunk_count = 0
    generation_start_perf = None
    generation_start_event_type = None
    generation_start_event_detail = None
    generation_start_confidence = "unavailable"
    hidden_reasoning_observability = "unavailable"

    system, anthropic_messages = anthropic_input_from_messages(messages)
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": max_tokens,
    }
    if system:
        request_kwargs["system"] = system
    if not omit_temperature:
        request_kwargs["temperature"] = 0
    if adaptive_thinking:
        request_kwargs["thinking"] = {"type": "adaptive", "display": "omitted"}
    if thinking_effort is not None:
        request_kwargs["output_config"] = {"effort": thinking_effort}

    client = anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds, max_retries=0)
    try:
        # Match OpenAI semantics: start immediately before SDK dispatch.
        request_sent_ts = now_utc()
        request_sent_perf = perf_counter()
        with client.messages.stream(**request_kwargs) as stream:
            raw_stream = getattr(stream, "_raw_stream", None)
            stream_response = getattr(raw_stream, "response", None)
            if stream_response is not None:
                request_id = (
                    stream_response.headers.get("request-id")
                    or stream_response.headers.get("x-request-id")
                )
            for event in stream:
                if first_stream_event_ts is None:
                    first_stream_event_ts = now_utc()
                    first_stream_event_perf = perf_counter()
                event_type = getattr(event, "type", "")
                if event_type == "message_start":
                    usage = getattr(getattr(event, "message", None), "usage", None)
                    if usage is not None:
                        usage_data = usage_to_dict(usage)
                        input_tokens = usage_data.get("input_tokens")
                        output_tokens = usage_data.get("output_tokens")
                        thinking_tokens = usage_data.get("thinking_tokens")
                elif event_type == "content_block_start" and generation_start_perf is None:
                    block_type = getattr(getattr(event, "content_block", None), "type", None)
                    if block_type in {"thinking", "redacted_thinking", "text", "tool_use", "refusal"}:
                        generation_start_perf = perf_counter()
                        generation_start_event_type = event_type
                        generation_start_event_detail = f"content_block.type={block_type}"
                        generation_start_confidence = "provider_boundary"
                        hidden_reasoning_observability = "phase_boundary_only" if block_type in {"thinking", "redacted_thinking"} else "not_observed"
                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if getattr(delta, "type", None) != "text_delta":
                        continue
                    text = getattr(delta, "text", "")
                    if not text:
                        continue
                    if generation_start_perf is None:
                        generation_start_perf = perf_counter()
                        generation_start_event_type = "content_block_delta.text_delta"
                        generation_start_event_detail = "first non-empty text delta"
                        generation_start_confidence = "delta_fallback"
                        hidden_reasoning_observability = "not_observed"
                    ts = now_utc()
                    if first_token_ts is None:
                        first_token_ts = ts
                        first_token_perf = perf_counter()
                    last_token_ts = ts
                    last_token_perf = perf_counter()
                    response_text_parts.append(text)
                    observable_chunk_count += 1
                elif event_type == "message_delta":
                    delta = getattr(event, "delta", None)
                    if getattr(delta, "stop_reason", None):
                        finish_reason = delta.stop_reason
                    usage = getattr(event, "usage", None)
                    if usage is not None:
                        usage_data = usage_to_dict(usage)
                        if usage_data.get("output_tokens") is not None:
                            output_tokens = usage_data["output_tokens"]

            final_message = stream.get_final_message()
            usage = getattr(final_message, "usage", None)
            if usage is not None:
                usage_data = usage_to_dict(usage)
                input_tokens = usage_data.get("input_tokens", input_tokens)
                output_tokens = usage_data.get("output_tokens", output_tokens)
                thinking_tokens = usage_data.get("thinking_tokens", thinking_tokens)
                usage_json = json.dumps(usage_data, sort_keys=True)
                final_usage_received = True
            finish_reason = getattr(final_message, "stop_reason", None) or finish_reason
    except anthropic.APIStatusError as exc:
        error_message = str(exc)
        if "content filtering policy" not in error_message.lower():
            return failed_inference_result(
                request_sent_ts=request_sent_ts,
                request_sent_perf=request_sent_perf,
                response_text_parts=response_text_parts,
                finish_reason="api_error",
                request_id=getattr(exc, "request_id", None) or request_id,
                first_stream_event_ts=first_stream_event_ts,
                first_stream_event_perf=first_stream_event_perf,
                first_token_ts=first_token_ts,
                last_token_ts=last_token_ts,
                first_token_perf=first_token_perf,
                last_token_perf=last_token_perf,
                observable_chunk_count=observable_chunk_count,
            )
        finish_reason = "content_filter"
        request_id = getattr(exc, "request_id", None) or request_id
        usage_json = json.dumps(
            {
                "provider_error": "content_filter",
                "request_id": getattr(exc, "request_id", None),
                "status_code": getattr(exc, "status_code", None),
            },
            sort_keys=True,
        )
    except Exception as exc:
        failure_reason = (
            "inference_timeout"
            if "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower()
            else "api_error"
        )
        return failed_inference_result(
            request_sent_ts=request_sent_ts,
            request_sent_perf=request_sent_perf,
            response_text_parts=response_text_parts,
            finish_reason=failure_reason,
            request_id=getattr(exc, "request_id", None) or request_id,
            first_stream_event_ts=first_stream_event_ts,
            first_stream_event_perf=first_stream_event_perf,
            first_token_ts=first_token_ts,
            last_token_ts=last_token_ts,
            first_token_perf=first_token_perf,
            last_token_perf=last_token_perf,
            observable_chunk_count=observable_chunk_count,
        )

    last_event_perf = perf_counter()
    response_text = "".join(response_text_parts)
    if output_tokens is None and finish_reason != "content_filter":
        output_tokens, token_count_method = resolve_token_count(response_text, model)
        visible_output_tokens = output_tokens
    else:
        token_count_method = (
            "anthropic_stream_usage_includes_omitted_thinking"
            if adaptive_thinking
            else "anthropic_stream_usage"
        )
        visible_output_tokens = (
            max(0, output_tokens - thinking_tokens)
            if thinking_tokens is not None
            else None
            if adaptive_thinking
            else output_tokens
        )
    if input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    outcome = outcome_from_finish_reason(finish_reason)
    billed_output_tokens = output_tokens if final_usage_received else None
    usage_reconciliation_status = reconcile_final_usage(billed_output_tokens, usage_json)
    ttft_s = None if first_token_perf is None else first_token_perf - request_sent_perf
    gen_time_s = None if first_token_perf is None or last_token_perf is None else last_token_perf - first_token_perf
    api_call_wall_s = last_event_perf - request_sent_perf
    tokens_per_s = None
    if outcome == "completed" and api_call_wall_s > 0 and billed_output_tokens is not None:
        tokens_per_s = billed_output_tokens / api_call_wall_s
    active_generation_time_s = last_event_perf - generation_start_perf if generation_start_perf is not None else None
    active_generation_billed_tps = billed_output_tokens / active_generation_time_s if outcome == "completed" and billed_output_tokens is not None and active_generation_time_s and active_generation_time_s > 0 else None

    return {
        "request_sent_ts": request_sent_ts,
        "first_token_ts": first_token_ts,
        "last_token_ts": last_token_ts,
        "first_stream_event_ts": first_stream_event_ts,
        "response_text": response_text,
        "output_tokens": output_tokens,
        "billed_output_tokens": billed_output_tokens,
        "input_tokens": input_tokens,
        "reasoning_tokens": thinking_tokens,
        "visible_output_tokens": visible_output_tokens,
        "total_tokens": total_tokens,
        "usage_json": usage_json,
        "token_count_method": token_count_method,
        "ttft_s": ttft_s,
        "first_stream_event_latency_s": (
            None
            if first_stream_event_perf is None
            else first_stream_event_perf - request_sent_perf
        ),
        "gen_time_s": gen_time_s,
        "api_call_wall_s": api_call_wall_s,
        "tokens_per_s": tokens_per_s,
        "generation_start_s": generation_start_perf - request_sent_perf if generation_start_perf is not None else None,
        "generation_start_event_type": generation_start_event_type,
        "generation_start_event_detail": generation_start_event_detail,
        "generation_start_confidence": generation_start_confidence,
        "hidden_reasoning_observability": hidden_reasoning_observability,
        "terminal_event_s": api_call_wall_s,
        "observed_pre_generation_s": generation_start_perf - request_sent_perf if generation_start_perf is not None else None,
        "active_generation_time_s": active_generation_time_s,
        "active_generation_billed_tps": active_generation_billed_tps,
        "end_to_end_billed_tps": tokens_per_s,
        "finish_reason": finish_reason,
        "outcome": outcome,
        "request_id": request_id,
        "observable_chunk_count": observable_chunk_count,
        "usage_reconciliation_status": usage_reconciliation_status,
    }


def validate_endpoint_args(args: argparse.Namespace) -> None:
    required = [("api-key", args.api_key), ("model", args.model)]
    if args.provider == "openai":
        required.append(("base-url", args.base_url))
    missing = [name for name, value in required if not value]
    if missing:
        raise RuntimeError(f"Missing required endpoint args/env: {', '.join(missing)}")
    if args.provider == "anthropic" and args.thinking_budget_tokens is not None:
        raise RuntimeError("--thinking-budget-tokens is only supported for OpenAI-compatible Anthropic endpoints.")
    if args.thinking_budget_tokens is not None and args.thinking_effort is not None:
        raise RuntimeError("Use either --thinking-budget-tokens or --thinking-effort, not both.")
    if args.timeout_seconds <= 0:
        raise RuntimeError("--timeout-seconds must be greater than zero.")


def validate_judge_args(args: argparse.Namespace) -> None:
    if args.judge and args.judge_after_run:
        args.judge_after_run = False
    if not args.judge and not args.judge_existing and not args.judge_after_run:
        return
    if not args.judge_base_url:
        args.judge_base_url = args.base_url
    if not args.judge_api_key:
        args.judge_api_key = args.api_key
    missing = [
        name
        for name, value in [
            ("judge-base-url", args.judge_base_url),
            ("judge-api-key", args.judge_api_key),
            ("judge-model", args.judge_model),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required judge args/env: {', '.join(missing)}")
    if args.judge_timeout_seconds <= 0:
        raise RuntimeError("--judge-timeout-seconds must be greater than zero.")


def prime_openai_env(args: argparse.Namespace) -> None:
    api_key = args.api_key or args.judge_api_key
    base_url = args.base_url or args.judge_base_url
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url


def write_judgment(
    args: argparse.Namespace,
    judge_client: OpenAI,
    judge_module: Any,
    problem: Problem,
    response_text: str,
    run_idx: Any,
    endpoint_name: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    result_timestamp: str | None = None,
    prefix: str = "judge",
) -> dict[str, Any]:
    result_endpoint_name = endpoint_name or args.endpoint_name
    result_model = model or args.model
    result_max_tokens = max_tokens if max_tokens is not None else args.max_tokens
    judgment = judge_completion(
        judge_client=judge_client,
        judge_module=judge_module,
        judge_model=args.judge_model,
        judge_max_tokens=args.judge_max_tokens,
        problem=problem,
        response_text=response_text,
        omit_temperature=args.omit_temperature,
    )
    append_judgment_row(
        [
            format_ts(now_utc()),
            result_endpoint_name,
            result_model,
            problem.sorted_index,
            problem.question_id,
            result_max_tokens,
            run_idx,
            args.judge_model,
            judgment["correct_answer"],
            judgment["model_answer"],
            judgment["correct"],
            judgment["confidence"],
            judgment["judge_parse_method"],
            judgment["reasoning"],
            result_timestamp or "",
        ],
        replace_identity=True,
    )
    update_result_correctness(
        endpoint_name=result_endpoint_name,
        model=result_model,
        question_id=problem.question_id,
        max_tokens=result_max_tokens,
        run_idx=run_idx,
        correct=judgment["correct"],
        result_timestamp=result_timestamp,
    )
    print(
        f"{prefix}="
        f"{args.judge_model} index={problem.sorted_index} question_id={problem.question_id} "
        f"correct={judgment['correct']} confidence={judgment['confidence']} "
        f"model_answer={judgment['model_answer']!r}"
    )
    if args.print_judge:
        print("judge_reasoning:")
        print(judgment["reasoning"])
        print()
    return judgment


def parse_judge_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped).strip()

    try:
        data = json.loads(stripped)
        return {
            "model_answer": str(data.get("extracted_final_answer", "")),
            "reasoning": str(data.get("reasoning", "")),
            "correct": str(data.get("correct", "")).lower(),
            "confidence": int(data.get("confidence", 100)),
            "judge_parse_method": "json_fallback",
        }
    except Exception:
        pass

    fields: dict[str, str] = {}
    pattern = re.compile(
        r"(?ms)^(extracted_final_answer|reasoning|correct|confidence):\s*(.*?)(?=^\w[\w_]*:\s*|\Z)"
    )
    for match in pattern.finditer(stripped):
        fields[match.group(1)] = match.group(2).strip()

    confidence_text = fields.get("confidence", "100")
    confidence_match = re.search(r"\d+", confidence_text)
    return {
        "model_answer": fields.get("extracted_final_answer", ""),
        "reasoning": fields.get("reasoning", ""),
        "correct": fields.get("correct", "").lower(),
        "confidence": int(confidence_match.group(0)) if confidence_match else 100,
        "judge_parse_method": "text_fallback",
    }


def judge_completion(
    judge_client: OpenAI,
    judge_module: Any,
    judge_model: str,
    judge_max_tokens: int,
    problem: Problem,
    response_text: str,
    omit_temperature: bool = False,
) -> dict[str, Any]:
    prompt = judge_module.JUDGE_PROMPT.format(
        question=problem.question["question"],
        correct_answer=problem.question["answer"],
        response=response_text,
    )

    try:
        request_kwargs = {
            "model": judge_model,
            "max_completion_tokens": judge_max_tokens,
            "n": 1,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": judge_module.ExtractedAnswer,
        }
        if not omit_temperature:
            request_kwargs["temperature"] = 0
        response = judge_client.beta.chat.completions.parse(**request_kwargs)
        content = response.choices[0].message.parsed
        return {
            "correct_answer": problem.question["answer"],
            "model_answer": content.extracted_final_answer,
            "reasoning": content.reasoning,
            "correct": content.correct,
            "confidence": content.confidence,
            "judge_parse_method": "openai_parse",
        }
    except APIConnectionError as exc:
        raise RuntimeError(f"Judge endpoint unreachable: {exc}") from exc
    except APIStatusError as exc:
        body = exc.response.text if exc.response is not None else str(exc)
        print(f"WARNING: structured judge parse failed with HTTP {exc.status_code}; falling back to text parse.")
        print(body)
    except Exception as exc:
        print(f"WARNING: structured judge parse failed: {exc}; falling back to text parse.")

    fallback_prompt = (
        f"{prompt}\n\n"
        "Return only valid JSON with keys extracted_final_answer, reasoning, correct, confidence, strict. "
        "The correct value must be exactly \"yes\" or \"no\". The strict value must be true."
    )
    try:
        request_kwargs = {
            "model": judge_model,
            "max_completion_tokens": judge_max_tokens,
            "n": 1,
            "messages": [{"role": "user", "content": fallback_prompt}],
        }
        if not omit_temperature:
            request_kwargs["temperature"] = 0
        response = judge_client.chat.completions.create(**request_kwargs)
    except APIConnectionError as exc:
        raise RuntimeError(f"Judge endpoint unreachable: {exc}") from exc
    except APIStatusError as exc:
        body = exc.response.text if exc.response is not None else str(exc)
        raise RuntimeError(f"Judge endpoint returned HTTP {exc.status_code}: {body}") from exc

    parsed = parse_judge_text(response.choices[0].message.content or "")
    parsed["correct_answer"] = problem.question["answer"]
    return parsed


def main() -> int:
    args = parse_args()
    if args.thinking_effort == "none":
        args.thinking_effort = None
    problems = canonical_text_only_questions()
    selected = select_problems(args, problems)

    if args.list_problems:
        print_problem_listing(selected)
        return 0

    if not args.judge_existing:
        validate_endpoint_args(args)
    validate_judge_args(args)
    prime_openai_env(args)
    prompt_module = None if args.judge_existing else load_prompt_module()
    judge_module = load_judge_module() if args.judge or args.judge_existing or args.judge_after_run else None
    client = (
        OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout_seconds, max_retries=0)
        if args.provider == "openai"
        else None
    )
    judge_client = (
        OpenAI(
            api_key=args.judge_api_key,
            base_url=args.judge_base_url,
            timeout=args.judge_timeout_seconds,
            max_retries=0,
        )
        if args.judge or args.judge_existing or args.judge_after_run
        else None
    )
    completed_records = completed_run_records()

    if args.judge_existing:
        selected_by_id = {problem.question_id: problem for problem in selected}
        matching_records = []
        for result_record, response_record in completed_records.values():
            if response_record.get("question_id") not in selected_by_id:
                continue
            if args.model is not None and response_record.get("model") != args.model:
                continue
            if response_record.get("max_tokens") != args.max_tokens:
                continue
            if not result_needs_judgment(result_record):
                continue
            matching_records.append((result_record, response_record))
        if not matching_records:
            print("No matching unjudged completed responses found.")
            return 0
        for result_record, record in matching_records:
            problem = selected_by_id[record["question_id"]]
            write_judgment(
                args=args,
                judge_client=judge_client,
                judge_module=judge_module,
                problem=problem,
                response_text=record["response"],
                run_idx=record.get("run_idx", ""),
                endpoint_name=record.get("endpoint_name", args.endpoint_name),
                model=record.get("model", args.model),
                max_tokens=record.get("max_tokens", args.max_tokens),
                result_timestamp=record.get("timestamp"),
                prefix="judge_existing",
            )
        return 0

    records_to_judge: list[tuple[Problem, dict[str, Any], dict[str, Any]]] = []
    report_records: list[dict[str, Any]] = []
    for problem in selected:
        print(f"Question [{problem.sorted_index}] {problem.question_id}")
        if not args.no_print_question:
            print(problem.question["question"])
        print()

        prompt_module.args = SimpleNamespace(model=args.model)
        messages = prompt_module.format_message(problem.question)

        for run_idx in range(1, args.runs + 1):
            identity = run_identity(
                {
                    "endpoint_name": args.endpoint_name,
                    "model": args.model,
                    "question_id": problem.question_id,
                    "max_tokens": args.max_tokens,
                    "run_idx": run_idx,
                }
            )
            existing = completed_records.get(identity) if args.resume else None
            if existing is not None:
                result_record, response_record = existing
                report_record = result_to_report_record(result_record)
                report_records.append(report_record)
                print(
                    "resume_skip="
                    f"{run_idx} index={problem.sorted_index} question_id={problem.question_id} "
                    f"timestamp={result_record['timestamp']} "
                    f"finish_reason={result_record.get('finish_reason', 'not_reported')} "
                    f"refusal={result_record.get('refusal', 'not_reported')} "
                    f"correct={result_record.get('correct', 'not_recorded')}"
                )
                if result_needs_judgment(result_record):
                    if args.judge:
                        judgment = write_judgment(
                            args=args,
                            judge_client=judge_client,
                            judge_module=judge_module,
                            problem=problem,
                            response_text=response_record["response"],
                            run_idx=run_idx,
                            endpoint_name=response_record["endpoint_name"],
                            model=response_record["model"],
                            max_tokens=response_record["max_tokens"],
                            result_timestamp=response_record.get("timestamp"),
                            prefix="judge_resumed",
                        )
                        report_record["correct"] = judgment["correct"]
                    elif args.judge_after_run:
                        records_to_judge.append((problem, response_record, report_record))
                continue

            task_dispatch_perf = perf_counter()
            if args.provider == "anthropic":
                result = stream_anthropic_completion(
                    api_key=args.api_key,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    messages=messages,
                    thinking_effort=args.thinking_effort,
                    adaptive_thinking=args.anthropic_adaptive_thinking,
                    omit_temperature=args.omit_temperature,
                    timeout_seconds=args.timeout_seconds,
                )
            else:
                result = stream_completion(
                    client=client,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    messages=messages,
                    thinking_budget_tokens=args.thinking_budget_tokens,
                    thinking_effort=args.thinking_effort,
                    omit_temperature=args.omit_temperature,
                )
            api_call_wall_s = float(result["api_call_wall_s"])
            outcome = str(result["outcome"])
            inference_time_s = (
                0.0 if outcome in {"inference_timeout", "api_error"} else api_call_wall_s
            )
            retry_api_time_s = (
                api_call_wall_s if outcome in {"inference_timeout", "api_error"} else 0.0
            )
            tool_time_s = 0.0
            backoff_time_s = 0.0
            total_wall_s = perf_counter() - task_dispatch_perf
            harness_overhead_s = max(
                0.0,
                total_wall_s
                - inference_time_s
                - tool_time_s
                - retry_api_time_s
                - backoff_time_s,
            )
            if "stream_usage" not in result["token_count_method"]:
                print(
                    f"WARNING: streaming usage not returned for [{problem.sorted_index}] {problem.question_id}; "
                    f"fallback={result['token_count_method']}"
                )
            if result["finish_reason"] in {"length", "max_tokens", "max_output_tokens"}:
                print(f"WARNING: output truncation detected for [{problem.sorted_index}] {problem.question_id}")

            judgeable = outcome == "completed"
            correctness_status = (
                "pending"
                if judgeable and (args.judge or args.judge_after_run)
                else "not_judged"
            )
            finish_reason = result["finish_reason"] or "not_reported"
            refusal = (
                "yes"
                if is_refusal_finish_reason(finish_reason)
                else "no"
                if finish_reason != "not_reported"
                else "not_reported"
            )
            endpoint_base_url = (
                args.base_url if args.provider == "openai" else "https://api.anthropic.com/v1/messages"
            )
            headline_eligible, headline_exclusion_reason = headline_eligibility(
                {
                    "outcome": outcome,
                    "billed_output_tokens": result.get("billed_output_tokens"),
                    "inference_time_s": inference_time_s,
                    "usage_reconciliation_status": result.get(
                        "usage_reconciliation_status", "unavailable"
                    ),
                }
            )

            write_result_row(
                [
                    format_ts(result["request_sent_ts"]),
                    args.endpoint_name,
                    args.model,
                    problem.sorted_index,
                    problem.question_id,
                    args.max_tokens,
                    run_idx,
                    format_float(result["ttft_s"]),
                    format_float(result["gen_time_s"]),
                    format_float(total_wall_s),
                    display_token_count(result["output_tokens"]),
                    result["token_count_method"],
                    format_float(result["tokens_per_s"]),
                    "" if result["input_tokens"] is None else result["input_tokens"],
                    display_token_count(result["reasoning_tokens"]),
                    "" if result["visible_output_tokens"] is None else result["visible_output_tokens"],
                    "" if result["total_tokens"] is None else result["total_tokens"],
                    result["usage_json"],
                    correctness_status,
                    finish_reason,
                    refusal,
                    args.provider,
                    result["request_id"] or "not_reported",
                    outcome,
                    display_token_count(result["billed_output_tokens"]),
                    display_token_count(result["reasoning_tokens"]),
                    format_float(inference_time_s),
                    format_float(tool_time_s),
                    format_float(retry_api_time_s),
                    format_float(backoff_time_s),
                    format_float(harness_overhead_s),
                    format_float(result["first_stream_event_latency_s"]),
                    format_ts(result["first_stream_event_ts"]),
                    format_ts(result["first_token_ts"]),
                    format_ts(result["last_token_ts"]),
                    result["observable_chunk_count"],
                    endpoint_base_url,
                    args.endpoint_region,
                    "none",
                    args.sandbox_image,
                    args.cpu_memory_limits,
                    args.economy_policy,
                    "yes",
                    format_float(result.get("generation_start_s")),
                    result.get("generation_start_event_type") or "not_recorded",
                    result.get("generation_start_event_detail") or "not_recorded",
                    result.get("generation_start_confidence") or "unavailable",
                    result.get("hidden_reasoning_observability") or "unavailable",
                    format_float(result.get("terminal_event_s")),
                    format_float(result.get("observed_pre_generation_s")),
                    format_float(result.get("active_generation_time_s")),
                    format_float(result.get("active_generation_billed_tps")),
                    format_float(result.get("end_to_end_billed_tps")),
                    1,
                    result.get("usage_reconciliation_status", "unavailable"),
                    "yes" if headline_eligible else "no",
                    headline_exclusion_reason,
                ],
                replace_identity=args.resume,
            )
            response_record = {
                "timestamp": format_ts(result["request_sent_ts"]),
                "endpoint_name": args.endpoint_name,
                "model": args.model,
                "sorted_index": problem.sorted_index,
                "question_id": problem.question_id,
                "max_tokens": args.max_tokens,
                "run_idx": run_idx,
                "finish_reason": result["finish_reason"],
                "refusal": refusal,
                "outcome": outcome,
                "provider": args.provider,
                "request_id": result["request_id"],
                "endpoint_base_url": endpoint_base_url,
                "endpoint_region": args.endpoint_region,
                "tool_configuration": "none",
                "sandbox_image": args.sandbox_image,
                "cpu_memory_limits": args.cpu_memory_limits,
                "economy_policy": args.economy_policy,
                "serial_execution": True,
                "timeout_seconds": args.timeout_seconds,
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "billed_output_tokens": result["billed_output_tokens"],
                "attempt_index": 1,
                "usage_reconciliation_status": result.get(
                    "usage_reconciliation_status", "unavailable"
                ),
                "headline_eligible": headline_eligible,
                "headline_exclusion_reason": headline_exclusion_reason,
                "active_generation_time_s": result.get("active_generation_time_s"),
                "reasoning_tokens": result["reasoning_tokens"],
                "thinking_tokens": result["reasoning_tokens"],
                "visible_output_tokens": result["visible_output_tokens"],
                "total_tokens": result["total_tokens"],
                "usage": json.loads(result["usage_json"]) if result["usage_json"] else None,
                "thinking_budget_tokens": args.thinking_budget_tokens,
                "thinking_effort": args.thinking_effort,
                "anthropic_adaptive_thinking": (
                    args.anthropic_adaptive_thinking if args.provider == "anthropic" else None
                ),
                "temperature": None if args.omit_temperature else (1 if args.thinking_budget_tokens is not None else 0),
                "omit_temperature": args.omit_temperature,
                "inference_time_s": inference_time_s,
                "tool_time_s": tool_time_s,
                "retry_api_time_s": retry_api_time_s,
                "backoff_time_s": backoff_time_s,
                "harness_overhead_s": harness_overhead_s,
                "total_wall_s": total_wall_s,
                "first_stream_event_latency_s": result["first_stream_event_latency_s"],
                "first_stream_event_ts": format_ts(result["first_stream_event_ts"]),
                "first_observable_output_ts": format_ts(result["first_token_ts"]),
                "last_observable_output_ts": format_ts(result["last_token_ts"]),
                "observable_chunk_count": result["observable_chunk_count"],
                "response": result["response_text"],
            }
            append_response_record(response_record)
            report_record = {
                "total_wall_s": total_wall_s,
                "inference_time_s": inference_time_s,
                "tool_time_s": tool_time_s,
                "retry_api_time_s": retry_api_time_s,
                "backoff_time_s": backoff_time_s,
                "harness_overhead_s": harness_overhead_s,
                "reasoning_tokens": result["reasoning_tokens"],
                "thinking_tokens": result["reasoning_tokens"],
                "visible_output_tokens": result["visible_output_tokens"],
                "output_tokens": result["output_tokens"],
                "billed_output_tokens": result["billed_output_tokens"],
                "outcome": outcome,
                "active_generation_time_s": result.get("active_generation_time_s"),
                "ttft_s": result.get("ttft_s"),
                "usage_reconciliation_status": result.get(
                    "usage_reconciliation_status", "unavailable"
                ),
                "headline_eligible": headline_eligible,
                "headline_exclusion_reason": headline_exclusion_reason,
                "correct": correctness_status,
                "finish_reason": finish_reason,
                "refusal": refusal,
            }
            report_records.append(report_record)
            summary_record = {
                "timestamp": format_ts(result["request_sent_ts"]),
                "endpoint_name": args.endpoint_name,
                "model": args.model,
                "sorted_index": problem.sorted_index,
                "question_id": problem.question_id,
                "category": problem.question["category"],
                "raw_subject": problem.question["raw_subject"],
                "max_tokens": args.max_tokens,
                "run_idx": run_idx,
                "ttft_s": result["ttft_s"],
                "gen_time_s": result["gen_time_s"],
                "total_wall_s": total_wall_s,
                "inference_time_s": inference_time_s,
                "tool_time_s": tool_time_s,
                "retry_api_time_s": retry_api_time_s,
                "backoff_time_s": backoff_time_s,
                "harness_overhead_s": harness_overhead_s,
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "billed_output_tokens": result["billed_output_tokens"],
                "attempt_index": 1,
                "usage_reconciliation_status": result.get(
                    "usage_reconciliation_status", "unavailable"
                ),
                "headline_eligible": headline_eligible,
                "headline_exclusion_reason": headline_exclusion_reason,
                "reasoning_tokens": result["reasoning_tokens"],
                "visible_output_tokens": result["visible_output_tokens"],
                "total_tokens": result["total_tokens"],
                "token_count_method": result["token_count_method"],
                "tokens_per_s": result["tokens_per_s"],
                "generation_start_s": result.get("generation_start_s"),
                "generation_start_event_type": result.get("generation_start_event_type"),
                "generation_start_event_detail": result.get("generation_start_event_detail"),
                "generation_start_confidence": result.get("generation_start_confidence"),
                "hidden_reasoning_observability": result.get("hidden_reasoning_observability"),
                "terminal_event_s": result.get("terminal_event_s"),
                "observed_pre_generation_s": result.get("observed_pre_generation_s"),
                "active_generation_time_s": result.get("active_generation_time_s"),
                "active_generation_billed_tps": result.get("active_generation_billed_tps"),
                "end_to_end_billed_tps": result.get("end_to_end_billed_tps"),
                "finish_reason": result["finish_reason"],
                "outcome": outcome,
                "refusal": refusal,
                "provider": args.provider,
                "request_id": result["request_id"],
                "endpoint_base_url": endpoint_base_url,
                "endpoint_region": args.endpoint_region,
                "tool_configuration": "none",
                "sandbox_image": args.sandbox_image,
                "cpu_memory_limits": args.cpu_memory_limits,
                "economy_policy": args.economy_policy,
                "serial_execution": True,
                "timeout_seconds": args.timeout_seconds,
                "thinking_budget_tokens": args.thinking_budget_tokens,
                "thinking_effort": args.thinking_effort,
                "anthropic_adaptive_thinking": (
                    args.anthropic_adaptive_thinking if args.provider == "anthropic" else None
                ),
                "temperature": None if args.omit_temperature else (1 if args.thinking_budget_tokens is not None else 0),
                "omit_temperature": args.omit_temperature,
                "first_stream_event_latency_s": result["first_stream_event_latency_s"],
                "first_stream_event_ts": format_ts(result["first_stream_event_ts"]),
                "first_observable_output_ts": format_ts(result["first_token_ts"]),
                "last_observable_output_ts": format_ts(result["last_token_ts"]),
                "observable_chunk_count": result["observable_chunk_count"],
            }

            print(
                "run="
                f"{run_idx} index={problem.sorted_index} question_id={problem.question_id} "
                f"ttft_s={format_float(result['ttft_s'])} "
                f"gen_time_s={format_float(result['gen_time_s'])} "
                f"inference_time_s={format_float(inference_time_s)} "
                f"retry_api_time_s={format_float(retry_api_time_s)} "
                f"total_wall_s={format_float(total_wall_s)} "
                f"input_tokens={result['input_tokens']} "
                f"output_tokens={display_token_count(result['output_tokens'])} "
                f"billed_output_tokens={display_token_count(result['billed_output_tokens'])} "
                f"reasoning_tokens={display_token_count(result['reasoning_tokens'])} "
                f"visible_output_tokens={result['visible_output_tokens']} "
                f"total_tokens={result['total_tokens']} "
                f"finish_reason={finish_reason} "
                f"outcome={outcome} "
                f"refusal={refusal} "
                f"request_id={result['request_id'] or 'not_reported'} "
                f"token_count_method={result['token_count_method']} "
                f"usage_reconciliation_status={result.get('usage_reconciliation_status', 'unavailable')} "
                f"headline_eligible={'yes' if headline_eligible else 'no'} "
                f"end_to_end_billed_tps={format_float(result.get('end_to_end_billed_tps'))}"
            )
            if args.print_response:
                print("response:")
                print(result["response_text"])
                print()

            if args.judge and judgeable:
                judgment = write_judgment(
                    args=args,
                    judge_client=judge_client,
                    judge_module=judge_module,
                    problem=problem,
                    response_text=result["response_text"],
                    run_idx=run_idx,
                    result_timestamp=summary_record["timestamp"],
                    prefix="judge",
                )
                summary_record["judge"] = {
                    "judge_model": args.judge_model,
                    "correct_answer": judgment["correct_answer"],
                    "model_answer": judgment["model_answer"],
                    "correct": judgment["correct"],
                    "confidence": judgment["confidence"],
                    "judge_parse_method": judgment["judge_parse_method"],
                    "reasoning": judgment["reasoning"],
                }
                report_record["correct"] = judgment["correct"]
            elif args.judge_after_run and judgeable:
                records_to_judge.append((problem, response_record, report_record))
            append_run_summary(summary_record)

    if args.judge_after_run:
        for problem, record, report_record in records_to_judge:
            judgment = write_judgment(
                args=args,
                judge_client=judge_client,
                judge_module=judge_module,
                problem=problem,
                response_text=record["response"],
                run_idx=record["run_idx"],
                endpoint_name=record["endpoint_name"],
                model=record["model"],
                max_tokens=record["max_tokens"],
                result_timestamp=record.get("timestamp"),
                prefix="judge_after_run",
            )
            report_record["correct"] = judgment["correct"]

    print_run_report(endpoint_name=args.endpoint_name, model=args.model, records=report_records)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
