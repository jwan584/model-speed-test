#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINI_DIR="$ROOT/mini-swe-agent"
MINI_EXTRA="$MINI_DIR/.venv/bin/mini-extra"
NEWSLETTER_ENV="$ROOT/../newsletter-bot/.env.local"

INSTANCE_ID="astropy__astropy-12907"
DOCKER_IMAGE="docker.io/swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
MODEL_NAME="${MODEL:-anthropic/claude-sonnet-4-5-20250929}"
OUTPUT_DIR="$ROOT/runs/$INSTANCE_ID"

if [ -f "$NEWSLETTER_ENV" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$NEWSLETTER_ENV"
  set +a
fi

echo "INSTANCE_ID=$INSTANCE_ID"
echo "DOCKER_IMAGE=$DOCKER_IMAGE"
echo "MODEL=$MODEL_NAME"

BLOCKERS=0

if ! command -v docker >/dev/null 2>&1; then
  echo "BLOCKER: docker is not installed or not on PATH; cannot pre-pull or run SWE-bench Docker image." >&2
  BLOCKERS=1
fi

if [ ! -x "$MINI_EXTRA" ]; then
  echo "BLOCKER: mini-swe-agent is not installed at $MINI_EXTRA. Run: cd \"$MINI_DIR\" && python3 -m venv .venv && .venv/bin/python -m pip install -e ." >&2
  BLOCKERS=1
fi

if [[ "$MODEL_NAME" == anthropic/* || "$MODEL_NAME" == *claude* ]]; then
  if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "BLOCKER: ANTHROPIC_API_KEY must be set, or present in $NEWSLETTER_ENV." >&2
    BLOCKERS=1
  fi
elif [ -z "${API_KEY:-${OPENAI_API_KEY:-}}" ]; then
  echo "BLOCKER: API_KEY or OPENAI_API_KEY must be set for the hosted coding model endpoint." >&2
  BLOCKERS=1
fi

if [ "$BLOCKERS" -ne 0 ]; then
  exit 2
fi

if [ -n "${API_KEY:-${OPENAI_API_KEY:-}}" ]; then
  export OPENAI_API_KEY="${API_KEY:-${OPENAI_API_KEY:-}}"
fi
if [ -n "${BASE_URL:-}" ]; then
  export OPENAI_API_BASE="$BASE_URL"
  export OPENAI_BASE_URL="$BASE_URL"
fi

export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"

echo "Pre-pulling $DOCKER_IMAGE"
docker pull "$DOCKER_IMAGE"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

"$MINI_EXTRA" swebench \
  --subset verified \
  --split test \
  --filter "^${INSTANCE_ID}$" \
  --workers 1 \
  --output "$OUTPUT_DIR" \
  --model "$MODEL_NAME" \
  --environment-class docker \
  --redo-existing

TRAJ_FILE="$OUTPUT_DIR/$INSTANCE_ID/$INSTANCE_ID.traj.json"
PREDS_FILE="$OUTPUT_DIR/preds.json"

echo
echo "===== TRAJECTORY: $TRAJ_FILE ====="
cat "$TRAJ_FILE"

echo
echo "===== FINAL PATCH ====="
"$MINI_DIR/.venv/bin/python" - "$PREDS_FILE" "$INSTANCE_ID" <<'PY'
import json
import sys
from pathlib import Path

preds_path = Path(sys.argv[1])
instance_id = sys.argv[2]
preds = json.loads(preds_path.read_text())
print(preds[instance_id]["model_patch"])
PY
