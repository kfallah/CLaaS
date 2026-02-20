#!/usr/bin/env bash
# Run the full Tinker integration test suite.
#
# Usage:
#   TINKER_API_KEY=... ./scripts/run_integration_tests.sh
#
# Requires: docker, uv
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker/docker-compose.local.yml"

export MODEL="${MODEL:-meta-llama/Llama-3.2-1B}"

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  echo "ERROR: TINKER_API_KEY is not set" >&2
  exit 1
fi

export TINKER_API_KEY
export FEEDBACK_BATCH_SIZE=1

OPENCLAW_TOKEN="openclaw-local-dev-token"

compose() {
  docker compose -f "$COMPOSE_FILE" --profile tinker "$@"
}

cleanup() {
  echo "==> Tearing down tinker stack ..."
  compose down -v || true
}
trap cleanup EXIT

# ── 1. Build and start ─────────────────────────────────────────────
echo "==> Starting tinker stack (model=$MODEL) ..."
compose up -d --build

# ── 2. Wait for health ─────────────────────────────────────────────
echo "==> Waiting for tinker-proxy ..."
timeout 180 bash -c 'until curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; do sleep 5; done'

echo "==> Waiting for CLaaS API ..."
timeout 120 bash -c 'until curl -sf http://127.0.0.1:8080/ >/dev/null 2>&1; do sleep 5; done'

echo "==> Waiting for OpenClaw gateway ..."
timeout 180 bash -c '
  until curl -sf \
    -H "Authorization: Bearer '"$OPENCLAW_TOKEN"'" \
    -H "Content-Type: application/json" \
    -H "x-openclaw-agent-id: main" \
    -X POST http://127.0.0.1:18789/v1/chat/completions \
    -d "{\"model\":\"openclaw\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" \
    -o /dev/null 2>&1; do sleep 5; done'

echo "==> All services healthy."

# ── 3. Run tests ───────────────────────────────────────────────────
echo "==> Running integration tests ..."
cd "$PROJECT_ROOT"
uv run pytest tests/integration/test_tinker_stack_integration.py -v --log-cli-level=INFO
