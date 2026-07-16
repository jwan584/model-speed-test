#!/usr/bin/env python3
"""Export the user-selected HLE runs as a self-contained CSV package."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bench


PACKAGE_NAME = "hle-selected-20-runs-2026-07-02"
OUTPUT_DIR = ROOT / "exports" / PACKAGE_NAME
ARCHIVE_PATH = ROOT / "exports" / f"{PACKAGE_NAME}.zip"
SIMPLE_EXPORT_PATH = ROOT / "exports" / "hle-selected-20-runs-simple.csv"

DATASET_NAME = "cais/hle"
DATASET_SPLIT = "test"
DATASET_REVISION = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"
HLE100_FINGERPRINT = "599a49520013d7b4893dfaf19f0727e414ef991b628b8723803fda50391f5ad6"

SELECTIONS = [
    {
        "selection_batch": "opus-hle100-mini-1-10",
        "endpoint_name": "hle-mini-100-v1-q001-010-opus-max",
        "model": "claude-opus-4-8",
        "display_model": "claude-opus-4-8",
        "provider": "anthropic",
        "indexes": [12, 19, 62, 83, 88, 135, 156, 249, 266, 355],
        "question_ids": [
            "66e8784d70625d8c7700315a",
            "66e8a1833aa94517d4573b0d",
            "66e95faf8451a9b41f307932",
            "66ea2cc3c602e2b991ae8aba",
            "66ea3d3fa715c6c835b25764",
            "66ec02c52ec65d6153428744",
            "66ed5f1e85adbeda9f978022",
            "66f2cda3b508188b6e7328a8",
            "66f402add1c77d20ca3338ef",
            "66fc23cfa7be4edbe85cf177",
        ],
    },
    {
        "selection_batch": "gpt-hle100-mini-1-10",
        "endpoint_name": "hle-mini-100-v1-q001-010-xhigh",
        "model": "gpt-5.5-koffing-castor-kosmo-0527-test-ev3",
        "display_model": "gpt-5.6-ultrafast",
        "provider": "openai",
        "indexes": [12, 19, 62, 83, 88, 135, 156, 249, 266, 355],
        "question_ids": [
            "66e8784d70625d8c7700315a",
            "66e8a1833aa94517d4573b0d",
            "66e95faf8451a9b41f307932",
            "66ea2cc3c602e2b991ae8aba",
            "66ea3d3fa715c6c835b25764",
            "66ec02c52ec65d6153428744",
            "66ed5f1e85adbeda9f978022",
            "66f2cda3b508188b6e7328a8",
            "66f402add1c77d20ca3338ef",
            "66fc23cfa7be4edbe85cf177",
        ],
    },
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def numeric(value: Any) -> int | float | None:
    if value in {None, "", "not_reported", "not_recorded"}:
        return None
    text = str(value)
    return float(text) if "." in text else int(text)


def infer_answer_type(question: str) -> str:
    return "multipleChoice" if "Answer Choices:" in question else "exactMatch"


def require_one(records: list[dict[str, Any]], description: str) -> dict[str, Any]:
    if len(records) != 1:
        raise RuntimeError(f"Expected exactly one {description}; found {len(records)}")
    return records[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = read_csv(ROOT / "results.csv")
    judgments = read_csv(ROOT / "judgments.csv")
    responses = load_jsonl(ROOT / "responses.jsonl")
    summaries = load_jsonl(ROOT / "run_summaries.jsonl")
    questions = {problem.question_id: problem.question for problem in bench.canonical_text_only_questions()}

    detailed_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []

    for selection in SELECTIONS:
        if len(selection["indexes"]) != len(selection["question_ids"]):
            raise RuntimeError(f"Malformed selection: {selection['selection_batch']}")

        for position, (sorted_index, question_id) in enumerate(
            zip(selection["indexes"], selection["question_ids"], strict=True), start=1
        ):
            question = questions[question_id]
            base_match = {
                "endpoint_name": selection["endpoint_name"],
                "model": selection["model"],
                "question_id": question_id,
            }

            result = require_one(
                [
                    row
                    for row in results
                    if all(row[key] == str(value) for key, value in base_match.items())
                    and int(row["sorted_index"]) == sorted_index
                ],
                f"result row for {selection['model']} / {question_id}",
            )
            response = require_one(
                [
                    row
                    for row in responses
                    if all(str(row.get(key)) == str(value) for key, value in base_match.items())
                    and str(row.get("timestamp")) == result["timestamp"]
                    and str(row.get("run_idx")) == result["run_idx"]
                ],
                f"response row for {selection['model']} / {question_id}",
            )
            summary = require_one(
                [
                    row
                    for row in summaries
                    if all(str(row.get(key)) == str(value) for key, value in base_match.items())
                    and str(row.get("timestamp")) == result["timestamp"]
                    and str(row.get("run_idx")) == result["run_idx"]
                ],
                f"summary row for {selection['model']} / {question_id}",
            )
            judgment = require_one(
                [
                    row
                    for row in judgments
                    if all(row[key] == str(value) for key, value in base_match.items())
                    and row["run_idx"] == result["run_idx"]
                    and row["max_tokens"] == result["max_tokens"]
                ],
                f"judgment row for {selection['model']} / {question_id}",
            )

            if question["answer"] != judgment["correct_answer"]:
                raise RuntimeError(f"Answer mismatch for {question_id}")
            if result["correct"] != judgment["correct"]:
                raise RuntimeError(f"Correctness mismatch for {question_id}")

            hle100_mini_number = position if "hle100-mini-1-10" in selection["selection_batch"] else ""
            question_row = {
                "selection_batch": selection["selection_batch"],
                "selection_position": position,
                "hle100_mini_number": hle100_mini_number,
                "sorted_index": sorted_index,
                "question_id": question_id,
                "category": question["category"],
                "raw_subject": question["raw_subject"],
                "answer_type": infer_answer_type(question["question"]),
                "question": question["question"],
                "correct_answer": question["answer"],
                "image": question["image"],
            }
            question_rows.append(question_row)

            detailed_rows.append(
                {
                    "selection_batch": selection["selection_batch"],
                    "selection_position": position,
                    "hle100_mini_number": hle100_mini_number,
                    "dataset_name": DATASET_NAME,
                    "dataset_split": DATASET_SPLIT,
                    "dataset_revision": DATASET_REVISION,
                    "hle100_ordered_id_fingerprint_sha256": (
                        HLE100_FINGERPRINT if hle100_mini_number else ""
                    ),
                    "provider": selection["provider"],
                    "display_model": selection["display_model"],
                    "model": selection["model"],
                    "endpoint_name": selection["endpoint_name"],
                    "sorted_index": sorted_index,
                    "question_id": question_id,
                    "category": question["category"],
                    "raw_subject": question["raw_subject"],
                    "answer_type": infer_answer_type(question["question"]),
                    "question": question["question"],
                    "correct_answer": question["answer"],
                    "image": question["image"],
                    "result_timestamp": result["timestamp"],
                    "max_tokens": numeric(result["max_tokens"]),
                    "run_idx": numeric(result["run_idx"]),
                    "thinking_effort": summary.get("thinking_effort"),
                    "thinking_budget_tokens": summary.get("thinking_budget_tokens"),
                    "temperature": summary.get("temperature"),
                    "omit_temperature": summary.get("omit_temperature"),
                    "finish_reason": summary.get("finish_reason"),
                    "ttft_s": numeric(result["ttft_s"]),
                    "gen_time_s": numeric(result["gen_time_s"]),
                    "total_time_s": numeric(result["total_wall_s"]),
                    "input_tokens": numeric(result["input_tokens"]),
                    "output_tokens": numeric(result["output_tokens"]),
                    "reasoning_tokens": numeric(result["reasoning_tokens"]),
                    "reasoning_tokens_status": (
                        "reported" if numeric(result["reasoning_tokens"]) is not None else result["reasoning_tokens"]
                    ),
                    "visible_output_tokens": numeric(result["visible_output_tokens"]),
                    "total_tokens": numeric(result["total_tokens"]),
                    "token_count_method": result["token_count_method"],
                    "tokens_per_s": numeric(result["tokens_per_s"]),
                    "usage_json": result["usage_json"],
                    "correct": result["correct"],
                    "judgment_timestamp": judgment["timestamp"],
                    "judge_model": judgment["judge_model"],
                    "model_answer": judgment["model_answer"],
                    "judge_confidence": numeric(judgment["confidence"]),
                    "judge_parse_method": judgment["judge_parse_method"],
                    "judge_reasoning": judgment["reasoning"],
                    "response_text": response["response"],
                }
            )

    detailed_fields = list(detailed_rows[0])
    question_fields = list(question_rows[0])
    write_csv(OUTPUT_DIR / "runs_detailed.csv", detailed_fields, detailed_rows)
    write_csv(OUTPUT_DIR / "questions.csv", question_fields, question_rows)
    simple_rows = [
        {
            "model": row["model"],
            "question_num": row["sorted_index"],
            "category": row["category"],
            "question_content": row["question"],
            "correctness": row["correct"],
            "total_wall_clock_s": row["total_time_s"],
        }
        for row in detailed_rows
    ]
    write_csv(SIMPLE_EXPORT_PATH, list(simple_rows[0]), simple_rows)

    by_batch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detailed_rows:
        by_batch[row["selection_batch"]].append(row)

    summary_rows: list[dict[str, Any]] = []
    for selection in SELECTIONS:
        rows = by_batch[selection["selection_batch"]]
        times = [float(row["total_time_s"]) for row in rows]
        reasoning = [row["reasoning_tokens"] for row in rows]
        summary_rows.append(
            {
                "selection_batch": selection["selection_batch"],
                "provider": selection["provider"],
                "display_model": selection["display_model"],
                "model": selection["model"],
                "endpoint_name": selection["endpoint_name"],
                "thinking_effort": rows[0]["thinking_effort"],
                "questions": len(rows),
                "correct": sum(row["correct"] == "yes" for row in rows),
                "incorrect": sum(row["correct"] == "no" for row in rows),
                "accuracy": sum(row["correct"] == "yes" for row in rows) / len(rows),
                "total_time_s": sum(times),
                "average_time_s": statistics.mean(times),
                "median_time_s": statistics.median(times),
                "minimum_time_s": min(times),
                "maximum_time_s": max(times),
                "input_tokens": sum(int(row["input_tokens"] or 0) for row in rows),
                "output_tokens": sum(int(row["output_tokens"] or 0) for row in rows),
                "reasoning_tokens": (
                    sum(int(value) for value in reasoning) if all(value is not None for value in reasoning) else "not_reported"
                ),
                "visible_output_tokens": sum(int(row["visible_output_tokens"] or 0) for row in rows),
                "total_tokens": sum(int(row["total_tokens"] or 0) for row in rows),
                "output_tokens_per_total_wall_s": (
                    sum(int(row["output_tokens"] or 0) for row in rows) / sum(times)
                ),
                "judge_model": rows[0]["judge_model"],
            }
        )

    expected = {
        "opus-hle100-mini-1-10": {"correct": 4, "total_time_s": 3584.500383, "output_tokens": 289974},
        "gpt-hle100-mini-1-10": {"correct": 3, "total_time_s": 246.707280, "output_tokens": 55409},
    }
    for row in summary_rows:
        target = expected[row["selection_batch"]]
        if row["correct"] != target["correct"]:
            raise RuntimeError(f"Unexpected correctness total for {row['selection_batch']}")
        if abs(row["total_time_s"] - target["total_time_s"]) > 1e-6:
            raise RuntimeError(f"Unexpected time total for {row['selection_batch']}")
        if row["output_tokens"] != target["output_tokens"]:
            raise RuntimeError(f"Unexpected output-token total for {row['selection_batch']}")

    write_csv(OUTPUT_DIR / "summary.csv", list(summary_rows[0]), summary_rows)

    generated_at = datetime.now(timezone.utc).isoformat()
    readme = f"""# HLE selected-run CSV export

Generated: {generated_at}

## Scope

- 10 judged `claude-opus-4-8` runs from endpoint `hle-mini-100-v1-q001-010-opus-max`.
- 10 judged `gpt-5.5-koffing-castor-kosmo-0527-test-ev3` runs from endpoint `hle-mini-100-v1-q001-010-xhigh`; display alias `gpt-5.6-ultrafast`.
- Both models were evaluated on the same HLE-100 Mini #1–10 question IDs, so these records form a paired comparison.
- The rejected Q10 `reasoning.effort=max` probe produced no result and is not included.

## Files

- `runs_detailed.csv`: one row per inference run, including the full question, reference answer, response, judgment, timing, usage, and request configuration metadata.
- `questions.csv`: the 20 selected questions and reference answers, without duplicated run metadata.
- `summary.csv`: aggregate correctness, timing, and token totals by selected batch.
- `package_manifest.csv`: row counts, byte sizes, and SHA-256 checksums for package files.

## Definitions

- `total_time_s` is the model request's recorded `total_wall_s`; it excludes later judge latency.
- `correct` is the batch judge's answer-key agreement (`yes` or `no`), using the recorded `judge_model`.
- Anthropic usage does not expose a separate reasoning-token count here, so that field is `not_reported` in the summary and blank at run level with an explicit status column.
- Dataset: `{DATASET_NAME}`, split `{DATASET_SPLIT}`, pinned revision `{DATASET_REVISION}`.
- HLE-100 ordered-ID fingerprint: `sha256:{HLE100_FINGERPRINT}`.
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    manifest_rows = []
    for filename, description in [
        ("runs_detailed.csv", "Joined run-level export"),
        ("questions.csv", "Selected question and reference-answer manifest"),
        ("summary.csv", "Aggregate results by selected batch"),
        ("README.md", "Scope, provenance, and field definitions"),
    ]:
        path = OUTPUT_DIR / filename
        row_count = ""
        if path.suffix == ".csv":
            row_count = len(read_csv(path))
        manifest_rows.append(
            {
                "filename": filename,
                "description": description,
                "row_count": row_count,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_csv(OUTPUT_DIR / "package_manifest.csv", list(manifest_rows[0]), manifest_rows)

    with zipfile.ZipFile(ARCHIVE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT_DIR.iterdir()):
            archive.write(path, arcname=f"{PACKAGE_NAME}/{path.name}")

    print(f"package_dir={OUTPUT_DIR}")
    print(f"archive={ARCHIVE_PATH}")
    print(f"simple_csv={SIMPLE_EXPORT_PATH}")
    print(f"runs={len(detailed_rows)} questions={len(question_rows)} summaries={len(summary_rows)}")


if __name__ == "__main__":
    main()
