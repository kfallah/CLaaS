#!/usr/bin/env bash
set -euo pipefail

# L40S-friendly fallback that still exposes the CLaaS served model name.
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
SERVED_MODEL_NAME="qwen3-coder-next-8b"

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name "$SERVED_MODEL_NAME" \
  --api-key sk-local \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.95
