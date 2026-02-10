#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen3-8B"
SERVED_MODEL_NAME="qwen3-8b"

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name "$SERVED_MODEL_NAME" \
  --api-key sk-local \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.95
