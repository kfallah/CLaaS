#!/usr/bin/env bash
set -euo pipefail

WAIT_FOR_BACKEND="${WAIT_FOR_BACKEND:-1}"
if [[ "${WAIT_FOR_BACKEND}" == "1" ]]; then
    BACKEND_HEALTH_URL="${BACKEND_HEALTH_URL:-http://vllm:8000/health}"

    echo "Waiting for backend at ${BACKEND_HEALTH_URL} ..."
    until curl -sf --connect-timeout 5 --max-time 10 "${BACKEND_HEALTH_URL}" > /dev/null 2>&1; do
        sleep 5
    done
    echo "Backend is ready."
else
    echo "Skipping backend readiness check (WAIT_FOR_BACKEND=${WAIT_FOR_BACKEND})."
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config-name>" >&2
    exit 2
fi

CONFIG_NAME="$1"
shift

exec python -m claas.api --config-name "${CONFIG_NAME}" "$@"
