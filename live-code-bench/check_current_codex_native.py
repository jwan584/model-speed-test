#!/usr/bin/env python3
import json
import time
from pathlib import Path

import bench_runner


def main() -> None:
    root = Path("current_codex_native_q1-3_2026-07-13")
    ids = bench_runner.resolve_problem_refs(
        ["hard:1", "hard:2", "hard:3"], "release_v6"
    )
    problems = bench_runner.load_problems(ids, "release_v6")
    rows = []
    for number, problem in enumerate(problems, 1):
        path = root / f"q{number}_{problem.question_id}" / "solution.py"
        started = time.perf_counter()
        # The standard chat checker expects a fenced model response. Native
        # Codex wrote a raw submission file, so adapt only the serialization.
        submission = f"```python\n{path.read_text()}\n```"
        passed = bench_runner.check_solution(problem, submission, 12)
        row = {
            "question": number,
            "problem_id": problem.question_id,
            "passed": passed,
            "checker_wall_s": time.perf_counter() - started,
            "solution": str(path),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "cohort": "current Codex model; native conversation agent turn",
                "timer_start_utc": "2026-07-13T17:29:06.131400000Z",
                "runs": rows,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
