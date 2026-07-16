#!/usr/bin/env python3
import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)


def load_env(path: str) -> dict[str, str]:
    values = {}
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


values = load_env("../.env")
env = os.environ.copy()
env["API_KEY"] = values["ANTHROPIC_API_KEY"]
env["JUDGE_API_KEY"] = values["OpenAI-key"]
env["JUDGE_BASE_URL"] = "https://api.openai.com/v1"
env["JUDGE_MODEL"] = values.get("model_default", "gpt-5.5")

endpoint_pairs = [
    (
        "hle-verified-gold-100-v1-q001-010-model-fast-xhigh",
        "hle-verified-gold-100-v1-q001-010-opus-4-8-xhigh",
    ),
    (
        "hle-verified-gold-100-v1-q011-100-model-fast-xhigh",
        "hle-verified-gold-100-v1-q011-100-opus-4-8-xhigh",
    ),
]

with open("results.csv", newline="") as results_file:
    result_rows = list(csv.DictReader(results_file))

all_ids = []
for source_endpoint, target_endpoint in endpoint_pairs:
    selected = [row for row in result_rows if row["endpoint_name"] == source_endpoint]
    selected.sort(key=lambda row: int(row["sorted_index"]))
    question_ids = list(dict.fromkeys(row["question_id"] for row in selected))
    all_ids.extend(question_ids)
    print(f"launch endpoint={target_endpoint} questions={len(question_ids)}", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-u",
            "bench.py",
            "--provider",
            "anthropic",
            "--endpoint-name",
            target_endpoint,
            "--model",
            "claude-opus-4-8",
            "--question-ids",
            *question_ids,
            "--runs",
            "1",
            "--max-tokens",
            "100000",
            "--thinking-effort",
            "xhigh",
            "--omit-temperature",
            "--no-print-question",
        ],
        env=env,
        check=True,
    )

print(f"complete unique_questions={len(set(all_ids))}", flush=True)
