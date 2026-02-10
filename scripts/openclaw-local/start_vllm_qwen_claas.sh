#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-Coder-Next}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-coder-next-8b}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-sk-local}"
LORA_DIR="${LORA_DIR:-}"
LORA_NAME="${LORA_NAME:-coder}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CMD=(
  vllm serve "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --api-key "$API_KEY"
)

# CLaaS README local serving mode uses LoRA.
if [[ -n "$LORA_DIR" ]]; then
  CMD+=(--enable-lora --lora-modules "${LORA_NAME}=${LORA_DIR}")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_ARGS)
  CMD+=("${EXTRA_ARR[@]}")
fi

printf 'Starting vLLM with command:\n%s\n' "${CMD[*]}"
exec "${CMD[@]}"
