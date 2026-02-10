#!/usr/bin/env bash
set -euo pipefail

# Local vLLM + CLaaS distill orchestration:
# 1) warm up local vLLM
# 2) put vLLM into sleep mode (CPU offload)
# 3) run one CLaaS distill step with realistic mock data (no rollout_logprobs)
# 4) optionally wake vLLM

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_API_KEY="${VLLM_API_KEY:-sk-local}"
VLLM_MODEL="${VLLM_MODEL:-qwen3-8b}"

CLAAS_API_BASE_URL="${CLAAS_API_BASE_URL:-https://kfallah--claas-distill-fastapi-app.modal.run}"
LORA_ID="${LORA_ID:-demo/local-sleep-distill-init}"
TEACHER_MODE="${TEACHER_MODE:-self}"  # self or remote
WAKE_AFTER="${WAKE_AFTER:-1}"         # 1 wakes vLLM after distill

echo "Using local vLLM at http://${VLLM_HOST}:${VLLM_PORT}"
echo "Using CLaaS API at ${CLAAS_API_BASE_URL}"
echo "Using LoRA id ${LORA_ID}"

echo "1) Checking vLLM health..."
curl -sS -m 30 "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null
echo "vLLM is healthy."

echo "2) Warmup request to compile/capture kernels..."
curl -sS -m 90 "http://${VLLM_HOST}:${VLLM_PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${VLLM_API_KEY}" \
  -d "{
    \"model\": \"${VLLM_MODEL}\",
    \"messages\": [{\"role\":\"user\",\"content\":\"Say hello in one short sentence.\"}],
    \"max_tokens\": 32,
    \"temperature\": 0
  }" >/dev/null
echo "Warmup complete."

echo "3) Sleeping vLLM (offload weights/KV)..."
curl -sS -m 30 -X POST "http://${VLLM_HOST}:${VLLM_PORT}/sleep?level=1" >/dev/null
echo "vLLM is in sleep mode."

echo "4) Ensuring LoRA exists (idempotent if already present)..."
curl -sS -m 60 -X POST "${CLAAS_API_BASE_URL}/v1/lora/init" \
  -H "Content-Type: application/json" \
  -d "{
    \"lora_id\": \"${LORA_ID%-init}\",
    \"base_model\": \"Qwen/Qwen3-8B\"
  }" >/dev/null || true

DISTILL_PAYLOAD="$(cat <<'JSON'
{
  "lora_id": "__LORA_ID__",
  "prompt": "Write a Python function `unique_preserve_order(items)` that returns unique items while preserving their first-seen order.",
  "response": "def unique_preserve_order(items):\n    seen = set()\n    out = []\n    for x in items:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out",
  "feedback": "Solid implementation with correct order preservation and linear complexity.",
  "training": {
    "learning_rate": 0.0001,
    "alpha": 0.5,
    "teacher_mode": "__TEACHER_MODE__",
    "teacher_top_k": 20
  }
}
JSON
)"
DISTILL_PAYLOAD="${DISTILL_PAYLOAD/__LORA_ID__/${LORA_ID}}"
DISTILL_PAYLOAD="${DISTILL_PAYLOAD/__TEACHER_MODE__/${TEACHER_MODE}}"

echo "5) Running one distill step (no rollout_logprobs)..."
DISTILL_RESP="$(curl -sS -m 300 -X POST "${CLAAS_API_BASE_URL}/v1/distill" \
  -H "Content-Type: application/json" \
  -d "${DISTILL_PAYLOAD}")"
echo "${DISTILL_RESP}"

if [[ "${WAKE_AFTER}" == "1" ]]; then
  echo "6) Waking vLLM..."
  curl -sS -m 30 -X POST "http://${VLLM_HOST}:${VLLM_PORT}/wake_up" >/dev/null
  echo "vLLM wake_up complete."
fi

echo "Done."
