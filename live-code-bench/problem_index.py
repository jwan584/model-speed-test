#!/usr/bin/env python3
"""List and resolve stable problem indexes for a LiveCodeBench release."""

import argparse
import csv
import re
import sys
from pathlib import Path

from lcb_runner.benchmarks import load_code_generation_dataset


INDEX_SELECTOR_RE = re.compile(r"^(easy|medium|hard):([1-9][0-9]*)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-version", default="v6")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"])
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--resolve",
        help="Comma-separated selectors like hard:1,medium:2 to resolve to question IDs",
    )
    return parser.parse_args()


def build_index(release_version: str, difficulty: str | None) -> list[dict[str, str | int]]:
    problems = load_code_generation_dataset(release_version)
    rows = []
    counters = {"easy": 0, "medium": 0, "hard": 0}
    for problem in problems:
        level = problem.difficulty.value
        counters[level] += 1
        if difficulty and level != difficulty:
            continue
        rows.append(
            {
                "index": counters[level],
                "question_id": problem.question_id,
                "difficulty": level,
                "title": problem.question_title,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["index", "question_id", "difficulty", "title"]
        )
        writer.writeheader()
        writer.writerows(rows)


def resolve_selectors(release_version: str, selectors: list[str]) -> list[str]:
    indexed = {
        level: build_index(release_version, level)
        for level in ("easy", "medium", "hard")
    }
    resolved = []
    for selector in selectors:
        match = INDEX_SELECTOR_RE.fullmatch(selector)
        if not match:
            raise ValueError(f"Invalid selector: {selector}")
        difficulty, raw_index = match.groups()
        index = int(raw_index)
        rows = indexed[difficulty]
        if index > len(rows):
            raise ValueError(
                f"{selector} is out of range for {release_version}; "
                f"{difficulty} has {len(rows)} problems"
            )
        resolved.append(rows[index - 1]["question_id"])
    return resolved


def main() -> int:
    args = parse_args()
    if args.resolve:
        selectors = [item.strip() for item in args.resolve.split(",") if item.strip()]
        for question_id in resolve_selectors(args.release_version, selectors):
            print(question_id)
        return 0

    rows = build_index(args.release_version, args.difficulty)
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.csv:
        write_csv(args.csv, rows)
    for row in rows:
        print(
            f"{row['index']},{row['question_id']},{row['difficulty']},{row['title']}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
