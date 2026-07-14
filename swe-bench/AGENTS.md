# Repository instructions

When asked to run or time SWE-bench, read and follow
`SWE_BENCH_RUNBOOK.md` before starting. The default Docker and
Hugging Face paths are not writable in the managed Codex environment. Do not
spend time retrying Docker Desktop, the default Colima profile, or the default
Hugging Face cache.

`DOCKER_CONFIG` must point to the writable empty directory documented in the
runbook. This is correctness-critical: the warning emitted when Docker cannot
read `~/.docker/config.json` is prepended to command output and prevents
mini-swe-agent from recognizing its submission sentinel.

Use `run_sequential_speed.sh` for timed runs. For "the first problem", use the
defaults plus `--count 1`; this selects the lexicographically first instance in
SWE-bench Verified/test (`astropy__astropy-12907` at the time this runbook was
written). Use `--prepull` so image download time is excluded from the recorded
problem time.

Do not impose an arbitrary wall-clock timeout and do not terminate a run merely
because it is quiet or exceeds a previous timing. SWE-bench duration varies
substantially by task, model behavior, and emulation speed. After preflight,
wait for the harness's own terminal result unless there is concrete evidence
of infrastructure failure.
