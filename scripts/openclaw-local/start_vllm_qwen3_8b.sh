#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-sk-local}"
SERVED_MODEL_NAMES="${SERVED_MODEL_NAMES:-qwen3-8b}"
ENABLE_SLEEP_MODE="${ENABLE_SLEEP_MODE:-1}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
LORA_MODULES="${LORA_MODULES:-}"
LORA_ALIAS_FILE="${LORA_ALIAS_FILE:-$ROOT_DIR/.local_loras/.aliases.json}"
LORA_ROOT="${LORA_ROOT:-$ROOT_DIR/.local_loras}"
INCLUDE_ALIAS_LORAS="${INCLUDE_ALIAS_LORAS:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-${ENABLE_SLEEP_MODE}}"
export VLLM_SERVER_DEV_MODE

declare -a CMD=()
declare -a MODULES=()
declare -a MODEL_NAME_ARR=()

IFS=',' read -r -a MODEL_NAME_ARR <<<"$SERVED_MODEL_NAMES"
for name in "${MODEL_NAME_ARR[@]}"; do
  trimmed="$(echo "$name" | xargs)"
  if [[ -n "$trimmed" ]]; then
    CMD+=(--served-model-name "$trimmed")
  fi
done

IFS=',' read -r -a EXPLICIT_MODULES <<<"$LORA_MODULES"
for module in "${EXPLICIT_MODULES[@]}"; do
  trimmed="$(echo "$module" | xargs)"
  if [[ -n "$trimmed" ]]; then
    MODULES+=("$trimmed")
  fi
done

if [[ "$INCLUDE_ALIAS_LORAS" == "1" && -f "$LORA_ALIAS_FILE" ]]; then
  while IFS= read -r module; do
    [[ -n "$module" ]] && MODULES+=("$module")
  done < <(
    python3 -c '
import json
import os
import re
import sys

alias_file = sys.argv[1]
lora_root = sys.argv[2]

def normalize(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-")
    return normalized or "lora"

with open(alias_file, "r", encoding="utf-8") as fh:
    aliases = json.load(fh)

for alias, target in sorted(aliases.items()):
    if not isinstance(alias, str) or not isinstance(target, str):
        continue
    path = os.path.join(lora_root, target.strip("/"))
    if os.path.isdir(path):
        print(f"{normalize(alias)}={path}")
' "$LORA_ALIAS_FILE" "$LORA_ROOT"
  )
fi

declare -a FINAL_MODULES=()
declare -A SEEN_MODULES=()
for module in "${MODULES[@]+"${MODULES[@]}"}"; do
  if [[ -z "${SEEN_MODULES[$module]+x}" ]]; then
    SEEN_MODULES["$module"]=1
    FINAL_MODULES+=("$module")
  fi
done

BASE_CMD=(
  vllm serve "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --api-key "$API_KEY"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
)

if [[ "$ENABLE_SLEEP_MODE" == "1" ]]; then
  BASE_CMD+=(--enable-sleep-mode)
fi

if [[ "$ENABLE_AUTO_TOOL_CHOICE" == "1" ]]; then
  BASE_CMD+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER")
fi

if [[ "${#FINAL_MODULES[@]}" -gt 0 ]]; then
  BASE_CMD+=(--enable-lora)
  for module in "${FINAL_MODULES[@]}"; do
    BASE_CMD+=(--lora-modules "$module")
  done
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_ARGS)
  BASE_CMD+=("${EXTRA_ARR[@]}")
fi

BASE_CMD+=("${CMD[@]}")

printf 'Starting vLLM with command:\n%s\n' "${BASE_CMD[*]}"
if [[ "${#FINAL_MODULES[@]}" -gt 0 ]]; then
  printf 'Loaded LoRA modules (%d):\n' "${#FINAL_MODULES[@]}"
  printf '  - %s\n' "${FINAL_MODULES[@]}"
fi
exec "${BASE_CMD[@]}"
