#!/usr/bin/env python3
"""Build the deterministic HLE-Verified Gold 100 benchmark manifests."""

from __future__ import annotations

import csv
import glob
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]

VERIFIED_DATASET = "skylenage-ai/HLE-Verified"
VERIFIED_REVISION = "0bc83643672d4f68a5f89998617a639d85e7318b"
VERIFIED_CACHE = ROOT / ".cache" / "hle-verified-gold" / VERIFIED_REVISION

RAW_DATASET = "cais/hle"
RAW_REVISION = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"
RAW_ARROW = (
    ROOT
    / ".cache"
    / "huggingface"
    / "datasets"
    / "cais___hle"
    / "default"
    / "0.0.0"
    / RAW_REVISION
    / "hle-test.arrow"
)

SELECTION_SEED = "hle-verified-gold-mini-bench-v1-2026-07-02:"
BENCHMARK_NAME = "hle-verified-gold-100-v1"
CSV_PATH = ROOT / "hle-verified-gold-100-questions.csv"
DOC_PATH = ROOT / "docs" / "hle-verified-gold-100.md"

# Keep the same category quotas as hle-mini-100-v1 so category mix is
# comparable. One hundred cannot divide evenly across eight categories.
QUOTAS = {
    "Biology/Medicine": 13,
    "Chemistry": 12,
    "Computer Science/AI": 13,
    "Engineering": 12,
    "Humanities/Social Science": 12,
    "Math": 13,
    "Other": 12,
    "Physics": 13,
}


def load_gold_records() -> list[dict]:
    paths = sorted(VERIFIED_CACHE.glob("Gold_subset.part*.parquet"))
    if len(paths) != 5:
        raise RuntimeError(
            f"Expected five pinned Gold parquet shards under {VERIFIED_CACHE}; found {len(paths)}"
        )

    rows: list[dict] = []
    for path in paths:
        rows.extend(pq.read_table(path).to_pylist())

    if len(rows) != 668:
        raise RuntimeError(f"Expected 668 Gold rows; found {len(rows)}")
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("Gold source contains duplicate question IDs")
    if {row["Verified_Classes"] for row in rows} != {"Gold subset"}:
        raise RuntimeError("Pinned Gold shards contain a non-Gold class")

    records = [json.loads(row["json"]) for row in rows]
    for record in records:
        if record.get("Verified_Classes") != "Gold subset":
            raise RuntimeError(f"Non-Gold JSON record: {record['id']}")
    strict_count = sum(is_strict_gold(record) for record in records)
    if strict_count != 641:
        raise RuntimeError(f"Expected 641 component-valid Gold records; found {strict_count}")
    return records


def is_strict_gold(record: dict) -> bool:
    verify = record.get("verify_meta_info", {})
    return all(
        verify.get(component, {}).get("is_valid") == 1
        for component in ("problem_verify", "answer_verify", "rationale_verify")
    )


def load_raw_hle() -> tuple[list[dict], dict[str, int]]:
    if not RAW_ARROW.exists():
        raise RuntimeError(f"Pinned raw HLE Arrow file not found: {RAW_ARROW}")
    with pa.memory_map(str(RAW_ARROW), "r") as source:
        rows = ipc.open_stream(source).read_all().to_pylist()
    if len(rows) != 2500:
        raise RuntimeError(f"Expected 2,500 raw HLE rows; found {len(rows)}")

    text_rows = sorted((row for row in rows if not row["image"]), key=lambda row: row["id"])
    return rows, {row["id"]: index for index, row in enumerate(text_rows)}


def select_records(gold_records: list[dict], raw_rows: list[dict], raw_indices: dict[str, int]) -> list[dict]:
    text_only = [record for record in gold_records if is_strict_gold(record) and not record.get("image")]
    if len(text_only) != 554:
        raise RuntimeError(f"Expected 554 strict text-only Gold records; found {len(text_only)}")

    by_category: dict[str, list[dict]] = defaultdict(list)
    for record in text_only:
        by_category[record["category"]].append(record)
    if set(by_category) != set(QUOTAS):
        raise RuntimeError(f"Unexpected Gold categories: {sorted(set(by_category) - set(QUOTAS))}")

    selected: list[dict] = []
    for category, quota in QUOTAS.items():
        ranked = sorted(
            by_category[category],
            key=lambda record: (
                hashlib.sha256((SELECTION_SEED + record["id"]).encode()).hexdigest(),
                record["id"],
            ),
        )
        if len(ranked) < quota:
            raise RuntimeError(f"Category {category!r} has only {len(ranked)} candidates for quota {quota}")
        selected.extend(ranked[:quota])

    if len(selected) != 100 or len({record["id"] for record in selected}) != 100:
        raise RuntimeError("Selection is not exactly 100 unique records")

    raw_by_id = {row["id"]: row for row in raw_rows}
    for record in selected:
        raw = raw_by_id.get(record["id"])
        if raw is None:
            raise RuntimeError(f"Gold selection ID missing from pinned raw HLE: {record['id']}")
        for field in ("question", "answer", "image", "answer_type", "category", "raw_subject"):
            if record.get(field) != raw.get(field):
                raise RuntimeError(
                    f"Pinned Gold and raw HLE disagree for selected ID {record['id']} field {field}"
                )
        if record["id"] not in raw_indices:
            raise RuntimeError(f"Selected ID is not text-only in pinned raw HLE: {record['id']}")

    selected.sort(key=lambda record: raw_indices[record["id"]])
    return selected


def ordered_id_fingerprint(selected: list[dict]) -> str:
    payload = "".join(record["id"] + "\n" for record in selected).encode()
    return hashlib.sha256(payload).hexdigest()


def content_fingerprint(selected: list[dict]) -> str:
    core = [
        {
            "id": record["id"],
            "question": record["question"],
            "answer": record["answer"],
            "answer_type": record["answer_type"],
            "category": record["category"],
            "raw_subject": record["raw_subject"],
            "image": record["image"],
        }
        for record in selected
    ]
    payload = json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_csv(selected: list[dict], raw_indices: dict[str, int]) -> None:
    columns = [
        "gold100_number",
        "sorted_index",
        "id",
        "category",
        "raw_subject",
        "answer_type",
        "question",
        "correct_answer",
        "image",
        "verified_class",
        "problem_is_valid",
        "answer_is_valid",
        "rationale_is_valid",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for number, record in enumerate(selected, 1):
            verify = record["verify_meta_info"]
            writer.writerow(
                {
                    "gold100_number": number,
                    "sorted_index": raw_indices[record["id"]],
                    "id": record["id"],
                    "category": record["category"],
                    "raw_subject": record["raw_subject"],
                    "answer_type": record["answer_type"],
                    "question": record["question"],
                    "correct_answer": record["answer"],
                    "image": record["image"],
                    "verified_class": record["Verified_Classes"],
                    "problem_is_valid": verify["problem_verify"]["is_valid"],
                    "answer_is_valid": verify["answer_verify"]["is_valid"],
                    "rationale_is_valid": verify["rationale_verify"]["is_valid"],
                }
            )


def write_doc(
    selected: list[dict],
    gold_records: list[dict],
    raw_indices: dict[str, int],
    id_fingerprint: str,
    content_sha: str,
) -> None:
    strict_gold = [record for record in gold_records if is_strict_gold(record)]
    gold_text_counts = Counter(record["category"] for record in strict_gold if not record.get("image"))
    lines = [
        "# HLE-Verified Gold 100 mini-benchmark",
        "",
        f"This manifest defines `{BENCHMARK_NAME}`, a fixed 100-question text-only subset of HLE-Verified Gold for repeatable model comparisons.",
        "It is separate from the existing raw-HLE `hle-mini-100-v1` benchmark and does not replace its files or results.",
        "",
        "## Selection contract",
        "",
        f"- Verification source: `{VERIFIED_DATASET}` at commit `{VERIFIED_REVISION}`.",
        f"- Compatibility source: `{RAW_DATASET}`, `test` split, local revision `{RAW_REVISION}`.",
        "- Conservative Gold definition: records classified `Gold subset` with `is_valid=1` for the problem, answer, and rationale components and no revision to the original HLE item.",
        "- Source consistency safeguard: the five Gold shards contain 668 rows, but 27 rows have at least one component marked invalid. Those 27 are excluded, leaving 641 strict Gold records—the count reported in the paper abstract.",
        "- Candidate pool: 554 text-only records from the 641 strict Gold records; the remaining 87 strict Gold records have a non-empty input `image` and are excluded.",
        "- Balance: 100 questions across all eight broad HLE categories using the same quotas as `hle-mini-100-v1`.",
        f"- Deterministic selection: within each category, rank candidates by `SHA-256(\"{SELECTION_SEED}\" + question_id)` and take the category quota.",
        "- Final run order: ascending zero-based `sorted_index` from the pinned `cais/hle` text-only pool.",
        "- Compatibility check: every selected question, answer, image field, answer type, category, and raw subject exactly matches the pinned `cais/hle` record with the same ID.",
        "- Dataset changes: use `id` as the durable identifier; `sorted_index` is specific to the pinned raw HLE revision.",
        f"- Ordered-ID fingerprint: `sha256:{id_fingerprint}` (newline-delimited IDs with a trailing newline).",
        f"- Content fingerprint: `sha256:{content_sha}` (canonical JSON over ordered prompt/answer metadata).",
        "",
        "## Category quotas",
        "",
        "| Category | Gold text-only available | Selected |",
        "|---|---:|---:|",
    ]
    for category in QUOTAS:
        lines.append(f"| {category} | {gold_text_counts[category]} | {QUOTAS[category]} |")
    lines.extend(
        [
            f"| **Total** | **{sum(gold_text_counts.values())}** | **{sum(QUOTAS.values())}** |",
            "",
            "## Running the subset",
            "",
            "The CSV is the canonical content manifest. The command below extracts its IDs and runs the matching records from the pinned local `cais/hle` cache. Batch correctness judging runs after the response batch by default.",
            "",
            "```bash",
            "python3 -c 'import csv,sys; print(*(r[\"id\"] for r in csv.DictReader(open(sys.argv[1]))))' \\",
            "  hle-verified-gold-100-questions.csv | \\",
            "  xargs .venv/bin/python bench.py \\",
            f"    --endpoint-name {BENCHMARK_NAME} \\",
            "    --runs 1 \\",
            "    --no-print-question \\",
            "    --max-tokens 100000 \\",
            "    --omit-temperature \\",
            "    --question-ids",
            "```",
            "",
            "To run only Gold100 questions 1–10, use the same command with this extraction line:",
            "",
            "```bash",
            "python3 -c 'import csv,itertools,sys; print(*(r[\"id\"] for r in itertools.islice(csv.DictReader(open(sys.argv[1])), 10)))' \\",
            "  hle-verified-gold-100-questions.csv | \\",
            "  xargs .venv/bin/python bench.py \\",
            f"    --endpoint-name {BENCHMARK_NAME}-q001-010 \\",
            "    --runs 1 \\",
            "    --no-print-question \\",
            "    --max-tokens 100000 \\",
            "    --omit-temperature \\",
            "    --question-ids",
            "```",
            "",
            "## Sources",
            "",
            "- HLE-Verified dataset: https://huggingface.co/datasets/skylenage-ai/HLE-Verified",
            "- HLE-Verified paper: https://arxiv.org/abs/2602.13964",
            "- Original HLE dataset: https://huggingface.co/datasets/cais/hle",
            "",
            "## Question manifest",
            "",
            "| Gold100 # | sorted_index | question_id | Category | Raw subject | Answer type |",
            "|---:|---:|---|---|---|---|",
        ]
    )
    for number, record in enumerate(selected, 1):
        lines.append(
            f"| {number} | {raw_indices[record['id']]} | {record['id']} | "
            f"{record['category']} | {record['raw_subject']} | {record['answer_type']} |"
        )
    lines.append("")
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    gold_records = load_gold_records()
    raw_rows, raw_indices = load_raw_hle()
    selected = select_records(gold_records, raw_rows, raw_indices)
    id_fingerprint = ordered_id_fingerprint(selected)
    content_sha = content_fingerprint(selected)
    write_csv(selected, raw_indices)
    write_doc(selected, gold_records, raw_indices, id_fingerprint, content_sha)
    print(f"wrote {CSV_PATH.relative_to(ROOT)}")
    print(f"wrote {DOC_PATH.relative_to(ROOT)}")
    print(f"ordered_id_sha256={id_fingerprint}")
    print(f"content_sha256={content_sha}")


if __name__ == "__main__":
    main()
