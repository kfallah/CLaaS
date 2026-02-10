#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-18789}"
VLLM_HEALTH_URL="${VLLM_HEALTH_URL:-http://127.0.0.1:8000/health}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-180}"

echo "Waiting for vLLM health at ${VLLM_HEALTH_URL} (timeout: ${VLLM_WAIT_SECONDS}s)..."
for ((i=1; i<=VLLM_WAIT_SECONDS; i++)); do
  if curl -fsS "$VLLM_HEALTH_URL" >/dev/null 2>&1; then
    echo "vLLM is ready."
    break
  fi
  if (( i == VLLM_WAIT_SECONDS )); then
    echo "Timed out waiting for vLLM health endpoint: ${VLLM_HEALTH_URL}" >&2
    exit 1
  fi
  sleep 1
done

exec openclaw gateway --port "$PORT" --verbose
