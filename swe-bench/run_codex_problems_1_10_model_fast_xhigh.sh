#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_ROOT=${SWE_BENCH_STATE_ROOT:-/tmp/swe-bench-runtime-claude-batch}
RUNNER_PYTHON=${SWE_BENCH_RUNNER_PYTHON:-$STATE_ROOT/runner-venv/bin/python}
CODEX_HOME=${CODEX_HOME:-$ROOT/../../LiveCodeBench/.codex-benchmark-home}
CODEX_BIN=${CODEX_INSTRUMENTED_BIN:-$ROOT/../../LiveCodeBench/.codex-instrumented/codex-v0.144.3}
DOCKER_HOST=${SWE_BENCH_DOCKER_HOST:-unix://$HOME/.colima/default/docker.sock}
DOCKER_CONFIG=${DOCKER_CONFIG:-$STATE_ROOT/docker-config}
HF_HOME=${HF_HOME:-$STATE_ROOT/hf-cache}

[[ -x "$RUNNER_PYTHON" ]] || {
  echo "BLOCKER: runner Python missing: $RUNNER_PYTHON" >&2
  exit 2
}
[[ -x "$CODEX_BIN" ]] || {
  echo "BLOCKER: instrumented Codex missing: $CODEX_BIN" >&2
  exit 2
}

export STATE_ROOT RUNNER_PYTHON CODEX_HOME CODEX_BIN
export SWE_BENCH_STATE_ROOT="$STATE_ROOT"
export SWE_BENCH_MACHINE_ID=${SWE_BENCH_MACHINE_ID:-macbook-nostrano}
export SWE_BENCH_MINI_PYTHON="$RUNNER_PYTHON"
export DOCKER_HOST DOCKER_CONFIG HF_HOME
export SSL_CERT_FILE=${SSL_CERT_FILE:-/etc/ssl/cert.pem}
export PYTHONDONTWRITEBYTECODE=1

"$CODEX_BIN" login status
docker version --format 'server={{.Server.Version}} arch={{.Server.Arch}}'
if [[ -n "$(docker ps --format '{{.ID}} {{.Names}}')" ]]; then
  echo "BLOCKER: running Docker containers would contaminate timing" >&2
  docker ps --format '{{.ID}} {{.Names}}' >&2
  exit 2
fi

stamp=$(date +%Y%m%d_%H%M%S)
output="$ROOT/runs/$SWE_BENCH_MACHINE_ID/codex_verified_q01_q10_model_fast_xhigh_terminal_$stamp"

exec "$RUNNER_PYTHON" "$ROOT/codex_swebench_1_10.py" \
  --tier default \
  --model gpt-5.6-sol-ultrafast \
  --reasoning xhigh \
  --codex-sandbox workspace-write \
  --python "$RUNNER_PYTHON" \
  --codex-bin "$CODEX_BIN" \
  --output "$output"
