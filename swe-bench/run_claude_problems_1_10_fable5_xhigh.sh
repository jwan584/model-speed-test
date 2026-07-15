#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_ROOT=${SWE_BENCH_STATE_ROOT:-$HOME/.swe-bench-runtime}
COLIMA_HOME="$STATE_ROOT/colima-home"
DOCKER_CONFIG_ROOT="$STATE_ROOT/docker-config"
HF_CACHE_ROOT="$STATE_ROOT/hf-cache"
PIP_CACHE_ROOT="$STATE_ROOT/pip-cache"
RUNNER_VENV=${SWE_BENCH_RUNNER_VENV:-$STATE_ROOT/runner-venv}
RUNNER_REQUIREMENTS="$ROOT/swebench_runner_requirements.txt"

usage() {
  cat <<'EOF'
Usage: bash ./run_claude_problems_1_10_fable5_xhigh.sh [--dry-run] [batch options]

Runs lexicographically sorted SWE-bench Verified/test Q1-Q10 sequentially
with Claude Code model claude-fable-5, effort xhigh, strict inference timing, and no
official correctness evaluation. Persistent runtime state lives under
~/.swe-bench-runtime; compact artifacts are written to runs/<machine>/.
EOF
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
  usage
  "$RUNNER_VENV/bin/python" "$ROOT/claude_swebench_1_10.py" --help 2>/dev/null || true
  exit 0
fi

mkdir -p "$STATE_ROOT" "$COLIMA_HOME" "$DOCKER_CONFIG_ROOT" \
  "$HF_CACHE_ROOT" "$PIP_CACHE_ROOT" "$STATE_ROOT/worktrees" \
  "$STATE_ROOT/build" "$STATE_ROOT/claude-config"

fingerprint=$({ shasum -a 256 "$RUNNER_REQUIREMENTS"; python3 -VV; uname -m; } | shasum -a 256 | awk '{print $1}')
marker="$RUNNER_VENV/.swe-bench-runtime-fingerprint"
if [[ ! -x "$RUNNER_VENV/bin/python" || ! -f "$marker" || "$(<"$marker")" != "$fingerprint" ]]; then
  echo "[bootstrap] Installing pinned runner under $STATE_ROOT"
  build_dir=$(mktemp -d "$STATE_ROOT/build/claude-runner.XXXXXX")
  trap 'rm -rf -- "${build_dir:-}"' EXIT
  python3 -m venv "$build_dir/venv"
  PIP_CACHE_DIR="$PIP_CACHE_ROOT" "$build_dir/venv/bin/python" -m pip install \
    --disable-pip-version-check --requirement "$RUNNER_REQUIREMENTS"
  printf '%s\n' "$fingerprint" > "$build_dir/venv/.swe-bench-runtime-fingerprint"
  rm -rf -- "$RUNNER_VENV"
  mv "$build_dir/venv" "$RUNNER_VENV"
  rmdir "$build_dir"
  build_dir=""
fi

if [[ -n ${SWE_BENCH_MACHINE_ID:-} ]]; then
  raw_machine=$SWE_BENCH_MACHINE_ID
elif command -v scutil >/dev/null 2>&1; then
  raw_machine=$(scutil --get LocalHostName 2>/dev/null || hostname -s)
else
  raw_machine=$(hostname -s)
fi
machine=$(printf '%s' "$raw_machine" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9_.-]+/-/g; s/^-+|-+$//g')
[[ -n "$machine" ]] || { echo "BLOCKER: invalid machine identifier" >&2; exit 2; }

export SWE_BENCH_STATE_ROOT="$STATE_ROOT"
export SWE_BENCH_MACHINE_ID="$machine"
export SWE_BENCH_EVAL_PYTHON="$STATE_ROOT/eval-venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export DOCKER_HOST=${SWE_BENCH_DOCKER_HOST:-"unix://$COLIMA_HOME/.colima/default/docker.sock"}
export DOCKER_CONFIG="$DOCKER_CONFIG_ROOT"
export HF_HOME="$HF_CACHE_ROOT"
export CLAUDE_CONFIG_DIR="$STATE_ROOT/claude-config"

# Use the shared benchmark credential when the caller has not already
# exported one. Parse only the requested POSIX-style assignment because the
# shared file may also contain unrelated, non-shell-compatible entries.
shared_env="$ROOT/../../../.env"
if [[ -z ${ANTHROPIC_API_KEY:-} && -f "$shared_env" ]]; then
  while IFS='=' read -r key value; do
    if [[ "$key" == ANTHROPIC_API_KEY && -n "$value" ]]; then
      export ANTHROPIC_API_KEY="$value"
      break
    fi
  done < "$shared_env"
fi

dry_run=0
for argument in "$@"; do
  [[ "$argument" == --dry-run ]] && dry_run=1
done
if [[ $dry_run -eq 0 ]]; then
  command -v claude >/dev/null || { echo "BLOCKER: claude is not on PATH" >&2; exit 2; }
  command -v colima >/dev/null || { echo "BLOCKER: colima is not on PATH" >&2; exit 2; }
  command -v docker >/dev/null || { echo "BLOCKER: docker is not on PATH" >&2; exit 2; }
  if ! claude auth status | "$RUNNER_VENV/bin/python" -c \
      'import json,sys; raise SystemExit(0 if json.load(sys.stdin).get("loggedIn") else 1)'; then
    echo "BLOCKER: Claude Code is not authenticated" >&2
    exit 2
  fi
  if docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
    echo "[bootstrap] Reusing running Docker/Colima VM at $DOCKER_HOST"
  elif ! HOME="$COLIMA_HOME" colima status >/dev/null 2>&1; then
    HOME="$COLIMA_HOME" colima start --cpus 4 --memory 8 --runtime docker --vm-type qemu --mount-type 9p
  else
    echo "[bootstrap] Reusing running Colima VM"
  fi
  docker version --format 'server={{.Server.Version}} arch={{.Server.Arch}}'
fi

exec "$RUNNER_VENV/bin/python" "$ROOT/claude_swebench_1_10.py" \
  --python "$RUNNER_VENV/bin/python" --model claude-fable-5 --effort xhigh "$@"
