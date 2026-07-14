#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/mini-swe-agent/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
  echo "BLOCKER: mini-swe-agent venv python is missing at $PYTHON" >&2
  echo "Run: cd \"$ROOT/mini-swe-agent\" && python3 -m venv .venv && .venv/bin/python -m pip install -e ." >&2
  exit 2
fi

exec "$PYTHON" "$ROOT/sequential_speed_harness.py" "$@"
