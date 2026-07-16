# HLE selected-run CSV export

Generated: 2026-07-02T23:47:38.027198+00:00

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
- Dataset: `cais/hle`, split `test`, pinned revision `5a81a4c7271a2a2a312b9a690f0c2fde837e4c29`.
- HLE-100 ordered-ID fingerprint: `sha256:599a49520013d7b4893dfaf19f0727e414ef991b628b8723803fda50391f5ad6`.
