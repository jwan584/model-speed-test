#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLIMA_HOME="/tmp/swe-bench-home"

mkdir -p "$COLIMA_HOME" /tmp/swe-bench-hf-cache /tmp/swe-bench-docker-config

export DOCKER_HOST="unix://$COLIMA_HOME/.colima/default/docker.sock"
export DOCKER_CONFIG="/tmp/swe-bench-docker-config"
export HF_HOME="/tmp/swe-bench-hf-cache"

if ! HOME="$COLIMA_HOME" colima status >/dev/null 2>&1; then
  HOME="$COLIMA_HOME" colima start \
    --cpus 4 \
    --memory 8 \
    --runtime docker \
    --vm-type qemu \
    --mount-type 9p
fi

exec "$ROOT/mini-swe-agent/.venv/bin/python" \
  "$ROOT/codex_swebench_problem1.py" \
  --index 0 \
  --model koffing-updated \
  --reasoning medium \
  --service-tier ultrafast \
  "$@"
