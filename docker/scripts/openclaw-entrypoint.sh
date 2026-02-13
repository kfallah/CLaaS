#!/usr/bin/env bash
set -euo pipefail

VLLM_BASE="${VLLM_BASE_URL:-http://vllm:8000}"
# Build health URL from base endpoint; supports both ...:8000 and ...:8000/v1
VLLM_BASE="${VLLM_BASE%/}"
VLLM_BASE="${VLLM_BASE%/v1}"
VLLM_HEALTH_URL="${VLLM_BASE}/health"

echo "Waiting for vLLM at ${VLLM_HEALTH_URL} ..."
until curl -sf "${VLLM_HEALTH_URL}" > /dev/null 2>&1; do
    sleep 5
done
echo "vLLM is ready."

exec openclaw gateway --port 18789 --bind lan --allow-unconfigured --verbose
