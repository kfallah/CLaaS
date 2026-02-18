#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/.run-logs}"
mkdir -p "$LOG_DIR"

VLLM_LOG="${VLLM_LOG:-$HOME/.openclaw-vllm-8000-sleep.log}"
VLLM_PID_FILE="${VLLM_PID_FILE:-$HOME/.openclaw-vllm-8000-sleep.pid}"
GATEWAY_LOG="${GATEWAY_LOG:-$LOG_DIR/gateway.log}"
GATEWAY_PID_FILE="${GATEWAY_PID_FILE:-$LOG_DIR/gateway.pid}"
CLAAS_API_LOG="${CLAAS_API_LOG:-$LOG_DIR/claas-api.log}"
CLAAS_API_PID_FILE="${CLAAS_API_PID_FILE:-$LOG_DIR/claas-api.pid}"

VLLM_START_SCRIPT="${VLLM_START_SCRIPT:-$ROOT_DIR/scripts/openclaw-local/start_vllm_qwen3_8b.sh}"
GATEWAY_START_SCRIPT="${GATEWAY_START_SCRIPT:-$ROOT_DIR/scripts/openclaw-local/start_openclaw_gateway_local.sh}"
CONFIGURE_SCRIPT="${CONFIGURE_SCRIPT:-$ROOT_DIR/scripts/openclaw-local/configure_openclaw_local_models.py}"

# Shared lora root — used by init_lora.py, vLLM, and the CLaaS API
export CLAAS_LORA_ROOT="${CLAAS_LORA_ROOT:-$ROOT_DIR/.local_loras}"

STACK_NAME="openclaw-local"
RESTART_BACKOFF_SECONDS="${RESTART_BACKOFF_SECONDS:-3}"
VLLM_HEALTH_URL="${VLLM_HEALTH_URL:-http://127.0.0.1:8000/health}"
VLLM_MODELS_URL="${VLLM_MODELS_URL:-http://127.0.0.1:8000/v1/models}"
CLAAS_API_HEALTH_URL="${CLAAS_API_HEALTH_URL:-http://127.0.0.1:${CLAAS_API_PORT:-8080}/v1/health}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-180}"
CLAAS_API_WAIT_SECONDS="${CLAAS_API_WAIT_SECONDS:-60}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-30}"

VLLM_OOM_STATE_FILE="${VLLM_OOM_STATE_FILE:-$LOG_DIR/vllm-log.offset}"
CLAAS_API_OOM_STATE_FILE="${CLAAS_API_OOM_STATE_FILE:-$LOG_DIR/claas-api-log.offset}"

LORA_NAME="${LORA_NAME:-openclaw/assistant}"
LORA_MODEL="${LORA_MODEL:-openclaw-assistant-latest}"
MODEL_IDS="${MODEL_IDS:-qwen3-8b,$LORA_MODEL}"
PRIMARY_MODEL="${PRIMARY_MODEL:-$LORA_MODEL}"

INIT_SCRIPT="${INIT_SCRIPT:-$ROOT_DIR/scripts/openclaw-local/init_lora.py}"

cleanup_old_pid() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$pid_file"
    fi
  fi
}

wait_for_health() {
  local url="$1"
  local timeout="$2"
  for ((i=1; i<=timeout; i++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

wait_for_models() {
  local url="$1"
  local timeout="$2"
  for ((i=1; i<=timeout; i++)); do
    if curl -fsS -H "Authorization: Bearer ${API_KEY:-sk-local}" "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

initialize_log_offset_state() {
  local log_file="$1"
  local state_file="$2"

  local size=0
  local offset=0

  if [[ -f "$log_file" ]]; then
    size="$(wc -c < "$log_file")"
  fi

  if [[ -f "$state_file" ]]; then
    offset="$(cat "$state_file")"
    if [[ ! "$offset" =~ ^[0-9]+$ ]]; then
      offset="$size"
    fi
  else
    offset="$size"
  fi

  if (( offset > size )); then
    offset="$size"
  fi

  printf '%s' "$offset" > "$state_file"
}

check_log_for_patterns() {
  local log_file="$1"
  local state_file="$2"
  local label="$3"

  if [[ ! -f "$log_file" ]]; then
    return 1
  fi

  local size=0
  local offset=0
  local chunk_start=1
  local chunk=""

  size="$(wc -c < "$log_file")"
  if [[ -f "$state_file" ]]; then
    offset="$(cat "$state_file")"
    if [[ ! "$offset" =~ ^[0-9]+$ ]]; then
      offset=0
    fi
  fi

  if (( offset > size )); then
    offset=0
  fi

  chunk_start=$((offset + 1))
  chunk="$(tail -c +"$chunk_start" "$log_file")"
  printf '%s' "$size" > "$state_file"

  if [[ -z "$chunk" ]]; then
    return 1
  fi

  if printf '%s' "$chunk" | rg -qi "out of memory|cuda out of memory|engine process failed|killed"; then
    echo "[$STACK_NAME] detected $label failure signature in $log_file, restarting..."
    return 0
  fi

  return 1
}

stop_pid_if_running() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  fi
}

start_stack_once() {
  cleanup_old_pid "$VLLM_PID_FILE"
  cleanup_old_pid "$GATEWAY_PID_FILE"

  # Ensure LoRA adapter exists before vLLM tries to load it
  echo "[$STACK_NAME] initializing LoRA adapter..."
  CLAAS_STORAGE_BACKEND=local_fs \
  LORA_NAME="$LORA_NAME" \
  python3 "$INIT_SCRIPT" 2>&1 | sed "s/^/[$STACK_NAME] /"

  BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}" \
  API_KEY="${API_KEY:-sk-local}" \
  MODEL_IDS="$MODEL_IDS" \
  PRIMARY_MODEL="$PRIMARY_MODEL" \
  python3 "$CONFIGURE_SCRIPT" >/dev/null 2>&1 || true

  echo "[$STACK_NAME] starting vLLM..."
  nohup "$VLLM_START_SCRIPT" >>"$VLLM_LOG" 2>&1 &
  local vllm_pid=$!
  echo "$vllm_pid" >"$VLLM_PID_FILE"
  initialize_log_offset_state "$VLLM_LOG" "$VLLM_OOM_STATE_FILE"

  if ! wait_for_health "$VLLM_HEALTH_URL" "$VLLM_WAIT_SECONDS"; then
    echo "[$STACK_NAME] vLLM failed health check; see $VLLM_LOG"
    return 1
  fi
  echo "[$STACK_NAME] vLLM healthy at $VLLM_HEALTH_URL"

  if ! wait_for_models "$VLLM_MODELS_URL" "$VLLM_WAIT_SECONDS"; then
    echo "[$STACK_NAME] vLLM models endpoint failed readiness check; see $VLLM_LOG"
    return 1
  fi
  echo "[$STACK_NAME] vLLM models endpoint ready at $VLLM_MODELS_URL"

  echo "[$STACK_NAME] starting CLaaS feedback API..."
  cleanup_old_pid "$CLAAS_API_PID_FILE"
  CLAAS_STORAGE_BACKEND=local_fs \
  CLAAS_DISTILL_EXECUTION_MODE=local \
  VLLM_BASE_URL="http://127.0.0.1:8000" \
  VLLM_API_KEY="${API_KEY:-sk-local}" \
  nohup uvicorn claas.api:web_app --host 0.0.0.0 --port "${CLAAS_API_PORT:-8080}" >>"$CLAAS_API_LOG" 2>&1 &
  local api_pid=$!
  echo "$api_pid" >"$CLAAS_API_PID_FILE"
  initialize_log_offset_state "$CLAAS_API_LOG" "$CLAAS_API_OOM_STATE_FILE"

  if ! wait_for_health "$CLAAS_API_HEALTH_URL" "$CLAAS_API_WAIT_SECONDS"; then
    echo "[$STACK_NAME] CLaaS API failed health check; see $CLAAS_API_LOG"
    return 1
  fi
  echo "[$STACK_NAME] CLaaS API healthy at $CLAAS_API_HEALTH_URL"

  echo "[$STACK_NAME] starting OpenClaw gateway..."
  nohup "$GATEWAY_START_SCRIPT" >>"$GATEWAY_LOG" 2>&1 &
  local gw_pid=$!
  echo "$gw_pid" >"$GATEWAY_PID_FILE"

  echo "[$STACK_NAME] started (vLLM pid=$vllm_pid, CLaaS API pid=$api_pid, gateway pid=$gw_pid)"
  return 0
}

monitor_loop() {
  while true; do
    local vpid=""
    local gpid=""
    local apid=""
    vpid="$(cat "$VLLM_PID_FILE" 2>/dev/null || true)"
    gpid="$(cat "$GATEWAY_PID_FILE" 2>/dev/null || true)"
    apid="$(cat "$CLAAS_API_PID_FILE" 2>/dev/null || true)"

    if [[ -z "$vpid" || -z "$gpid" || -z "$apid" ]]; then
      echo "[$STACK_NAME] missing pid file(s), restarting..."
      return 1
    fi
    if ! kill -0 "$vpid" 2>/dev/null; then
      echo "[$STACK_NAME] vLLM process exited (pid=$vpid), restarting..."
      return 1
    fi
    if ! kill -0 "$apid" 2>/dev/null; then
      echo "[$STACK_NAME] CLaaS API process exited (pid=$apid), restarting..."
      return 1
    fi
    if ! kill -0 "$gpid" 2>/dev/null; then
      echo "[$STACK_NAME] gateway process exited (pid=$gpid), restarting..."
      return 1
    fi
    if ! curl -fsS "$VLLM_HEALTH_URL" >/dev/null 2>&1; then
      echo "[$STACK_NAME] vLLM health check failed, restarting..."
      return 1
    fi
    if ! curl -fsS -H "Authorization: Bearer ${API_KEY:-sk-local}" "$VLLM_MODELS_URL" >/dev/null 2>&1; then
      echo "[$STACK_NAME] vLLM models endpoint failed, restarting..."
      return 1
    fi
    if ! curl -fsS "$CLAAS_API_HEALTH_URL" >/dev/null 2>&1; then
      echo "[$STACK_NAME] CLaaS API health check failed, restarting..."
      return 1
    fi
    if check_log_for_patterns "$VLLM_LOG" "$VLLM_OOM_STATE_FILE" "vLLM OOM"; then
      return 1
    fi
    if check_log_for_patterns "$CLAAS_API_LOG" "$CLAAS_API_OOM_STATE_FILE" "CLaaS API OOM"; then
      return 1
    fi

    sleep "$MONITOR_INTERVAL_SECONDS"
  done
}

stop_all() {
  stop_pid_if_running "$GATEWAY_PID_FILE"
  stop_pid_if_running "$CLAAS_API_PID_FILE"
  stop_pid_if_running "$VLLM_PID_FILE"
}

trap 'stop_all; exit 0' INT TERM

while true; do
  if ! start_stack_once; then
    stop_all
    sleep "$RESTART_BACKOFF_SECONDS"
    continue
  fi
  if ! monitor_loop; then
    stop_all
    sleep "$RESTART_BACKOFF_SECONDS"
    continue
  fi
done
