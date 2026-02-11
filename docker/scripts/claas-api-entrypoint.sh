#!/usr/bin/env bash
set -euo pipefail

VLLM_HEALTH_URL="${VLLM_BASE_URL:-http://vllm:8000}/health"
# Strip /v1 suffix if present for health check
VLLM_HEALTH_URL="${VLLM_HEALTH_URL%/v1}/health"

echo "Waiting for vLLM at ${VLLM_HEALTH_URL} ..."
until curl -sf "${VLLM_HEALTH_URL}" > /dev/null 2>&1; do
    sleep 5
done
echo "vLLM is ready."

exec uvicorn claas.api:web_app --host 0.0.0.0 --port 8080
