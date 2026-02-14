#!/usr/bin/env bash
set -euo pipefail

WAIT_FOR_BACKEND="${WAIT_FOR_BACKEND:-1}"
if [[ "${WAIT_FOR_BACKEND}" == "1" ]]; then
    BACKEND_BASE="${VLLM_BASE_URL:-http://vllm:8000}"
    BACKEND_BASE="${BACKEND_BASE%/}"
    BACKEND_BASE="${BACKEND_BASE%/v1}"
    BACKEND_HEALTH_URL="${BACKEND_HEALTH_URL:-${BACKEND_BASE}/health}"

    echo "Waiting for backend at ${BACKEND_HEALTH_URL} ..."
    until curl -sf --connect-timeout 5 --max-time 10 "${BACKEND_HEALTH_URL}" > /dev/null 2>&1; do
        sleep 5
    done
    echo "Backend is ready."
else
    echo "Skipping backend readiness check (WAIT_FOR_BACKEND=${WAIT_FOR_BACKEND})."
fi

exec uvicorn claas.api:web_app --host 0.0.0.0 --port 8080
