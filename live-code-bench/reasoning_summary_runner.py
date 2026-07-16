#!/usr/bin/env python3
"""Capture provider reasoning summaries for a small LiveCodeBench comparison."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import bench_runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openai-api-key", required=True)
    parser.add_argument("--openai-model", required=True)
    parser.add_argument("--anthropic-api-key", required=True)
    parser.add_argument("--anthropic-model", default="claude-fable-5")
    parser.add_argument("--problem-ids", default="hard:1-2")
    parser.add_argument("--release-version", default="v6")
    parser.add_argument("--thinking-effort", default="xhigh")
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--checker-timeout", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    problem_ids = bench_runner.resolve_problem_refs(
        bench_runner.read_problem_ids(args.problem_ids), args.release_version
    )
    problems = bench_runner.load_problems(problem_ids, args.release_version)
    problem_numbers = bench_runner.build_problem_number_map(args.release_version)
    openai_client = bench_runner.OpenAI(
        api_key=args.openai_api_key,
        base_url="https://api.openai.com/v1",
        timeout=args.timeout_seconds,
        max_retries=0,
    )
    results = []
    for problem_index, problem in enumerate(problems):
        messages = bench_runner.format_prompt_generation(
            problem, bench_runner.LMStyle.OpenAIChat
        )
        providers = ["openai", "anthropic"]
        if problem_index % 2:
            providers.reverse()
        for provider in providers:
            if provider == "openai":
                model = args.openai_model
                result = bench_runner.stream_openai_response(
                    openai_client,
                    model,
                    args.max_tokens,
                    messages,
                    args.thinking_effort,
                    include_reasoning_summary=True,
                )
            else:
                model = args.anthropic_model
                result = bench_runner.stream_anthropic_response(
                    args.anthropic_api_key,
                    None,
                    model,
                    args.max_tokens,
                    messages,
                    args.thinking_effort,
                    True,
                    args.timeout_seconds,
                    include_reasoning_summary=True,
                )
            output = result.pop("output")
            result.update(
                timestamp=datetime.now(timezone.utc).isoformat(),
                provider=provider,
                model=model,
                problem_id=problem.question_id,
                difficulty=problem.difficulty.value,
                problem_number=problem_numbers[problem.question_id],
                thinking_effort=args.thinking_effort,
                max_tokens=args.max_tokens,
                passed=bench_runner.check_solution(
                    problem, output, args.checker_timeout
                ),
            )
            results.append(result)
            print(
                f"Q{result['problem_number']} provider={provider} "
                f"status={result['request_status']} passed={result['passed']} "
                f"reasoning_tokens={result['reasoning_tokens']} "
                f"summary_chars={len(result['reasoning_summary'])}",
                flush=True,
            )
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {len(results)} records to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
