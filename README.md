# Model speed test

Reproducible harnesses for measuring model performance in coding-agent workflows.

## SWE-bench

The [`swe-bench/`](swe-bench/) directory contains native Codex and Claude Code
SWE-bench Verified harnesses, per-request model timing, overlap-safe tool and
wall accounting, pinned Python dependencies, tests, and operating
documentation.

Clone with submodules so the legacy mini-swe-agent workflows are available:

```bash
git clone --recurse-submodules https://github.com/jwan584/model-speed-test.git
cd model-speed-test/swe-bench
```

The current one-problem native harness accepts any one-based Verified question
number and model:

```bash
bash ./run_codex_swebench_problem Q2 \
  --model gpt-5.6-sol-ultrafast \
  --reasoning high \
  --skip-evaluation
```

It requires an instrumented Codex binary that emits response lifecycle traces.
Set `CODEX_INSTRUMENTED_BIN` when the companion LiveCodeBench checkout is not
located at `../live-code-bench` relative to `swe-bench/`.

Generated runs, local virtual environments, caches, credentials, and machine
state are intentionally not versioned.
