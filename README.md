# CLaaS: Continual Learning as a Service

SDPO-style continual distillation API for per-request model adaptation. Runs locally on a single GPU with optional Modal remote backend.

## Overview

CLaaS turns every user interaction into an online learning step via Self-Distillation Policy Optimization (SDPO). Each call:

1. Loads a user's LoRA adapter from local storage (or Modal Volume)
2. Runs the student model forward pass (Qwen3-8B + LoRA)
3. Gets teacher logprobs:
   - `self` (default): frozen base model conditioned on feedback
   - `remote`: 30B teacher model via vLLM on Modal
4. Computes SDPO loss (Generalized JSD + KL regularization)
5. Updates the LoRA parameters
6. Saves the adapter back to storage

This enables real-time model personalization where the model learns from each interaction.

## Installation

```bash
# Local GPU workflow (vLLM + local distillation)
uv sync --extra local

# Tinker workflow (no GPU deps; install CPU torch wheel)
uv sync --extra tinker
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch
```

**Prerequisites:** Python 3.11+ and `uv`.
- Local mode also requires a CUDA GPU and `vllm`.
- For remote execution, also run `uv run modal token new`.

## Quick Start

```bash
# 1. Start vLLM with LoRA support
vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8000 \
  --enable-lora --lora-modules my-lora=/loras/user/my-lora-init

# 2. Start the CLaaS API
uv run uvicorn claas.api:web_app --host 0.0.0.0 --port 8080

# 3. Initialize a LoRA adapter
curl -X POST http://localhost:8080/v1/lora/init \
  -H "Content-Type: application/json" \
  -d '{"lora_id": "user/my-lora"}'

# 4. Send a feedback update
curl -X POST http://localhost:8080/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "lora_id": "user/my-lora-init",
    "prompt": "Write a function to calculate factorial",
    "response": "def factorial(n): ...",
    "feedback": "Good recursive solution"
  }'
```

For the full local stack (vLLM + gateway + auto-restart), see [Local vLLM + OpenClaw](#local-vllm--openclaw).

## POST /v1/feedback

The primary endpoint. Runs one online update transaction for a served adapter.

**Orchestration lifecycle:**

1. Acquire per-LoRA lock (prevents concurrent updates)
2. `POST /sleep?level=1` to local vLLM (frees GPU memory)
3. Distill one SDPO step in-place via CLaaS worker
4. `POST /wake_up` to local vLLM (reloads adapter)
5. Write structured JSON log to `FEEDBACK_LOG_DIR`

**Request body:**

```json
{
  "lora_id": "user/my-lora-init",
  "prompt": "User prompt text",
  "response": "Model response to learn from",
  "feedback": "Feedback about response quality",
  "rollout_logprobs": [-0.5, -1.2, -0.8],
  "training": { "learning_rate": 1e-4, "alpha": 0.5, "teacher_mode": "self" },
  "orchestration": { "sleep_before": true, "wake_after": true, "sleep_level": 1 }
}
```

`rollout_logprobs` is optional; when provided, enables proper off-policy IS correction. `training` and `orchestration` have sensible defaults.

**Response:** returns `status`, `request_id`, updated `lora_id`, `distill_result` (loss metrics), `vllm` state, and `timing_ms`. See `/docs` for full schema.

## Other Endpoints

- **POST /v1/distill** — Run a single SDPO distillation step (low-level; returns versioned `lora_id` and training metrics).
- **POST /v1/lora/init** — Initialize a new LoRA adapter (`lora_id`, optional `base_model`, `lora_r`, `lora_alpha`, `target_modules`).
- **GET /v1/lora** — List all LoRA adapters and aliases (optional `prefix` query param).
- **GET /v1/lora/export** — Download a LoRA as a zip archive (`?lora_id=...`).
- **GET /v1/health** — Health check for the API and backing services.

## Execution Modes

Set `CLAAS_DISTILL_EXECUTION_MODE` to control which training engine implementation handles distillation:

- **`local`** (default) — Runs on the same machine. Requires a GPU with enough VRAM for Qwen3-8B + LoRA training.
- **`modal`** — Runs the distill step remotely on Modal (L40S) and keeps teacher scoring on Modal.
- **`tinker`** — Uses a Tinker-hosted CLaaS backend for distillation **and** LoRA lifecycle operations (`/v1/lora/init`, `/v1/lora`, `/v1/lora/export`). This mode is designed for large hosted teachers/models such as Qwen3 235B MoE and requires `CLAAS_TINKER_BASE_URL` and `CLAAS_TINKER_API_KEY`.

## Storage

- **Local** (default): adapters are stored under `CLAAS_LORA_ROOT` (default `/loras`). Path format: `/loras/{user}/{model}`.
- **Remote**: Modal Volume `claas-loras`, same path layout. Used automatically when `CLAAS_DISTILL_EXECUTION_MODE=modal`.

`/v1/feedback` updates adapters in-place (same `lora_id`). `/v1/distill` creates versioned checkpoints with timestamps.

## Docker Compose (Recommended)

The fastest way to get the full stack running — vLLM, CLaaS API, and OpenClaw with Telegram:

```bash
cd docker
cp .env.example .env
# Edit .env — set TELEGRAM_BOT_TOKEN (from @BotFather)
docker compose up --build
```

This brings up four services: an init container (creates the LoRA + config), vLLM with Qwen3-8B and LoRA serving, the CLaaS feedback API, and OpenClaw's Telegram gateway. See [`docker/README.md`](docker/README.md) for details.

For hosted Tinker instead of local vLLM, use the dedicated compose file:

```bash
cd docker
cp .env.tinker.example .env.tinker
docker compose -f docker-compose.tinker.yml --env-file .env.tinker up --build
```

## Local vLLM + OpenClaw

See [`scripts/openclaw-local/README.md`](scripts/openclaw-local/README.md) for the full supervised local stack (vLLM + gateway + auto-restart, multi-LoRA, Telegram integration).

## Modal Deployment

To deploy the service on Modal instead of running locally:

```bash
# Set HF_TOKEN if using gated models (e.g. Qwen/Qwen3-Coder-Next-8B)
export HF_TOKEN=...
export CLAAS_BASE_MODEL_ID=Qwen/Qwen3-8B

uv run modal deploy -m claas.deploy
```

The deployed app exposes the same API at `https://your-app--claas-distill-fastapi-app.modal.run`. LoRAs are stored in the `claas-loras` Modal Volume.

## Claude Code Setup

If you use [Claude Code](https://claude.ai/claude-code), you can set up the full local stack automatically:

```
/setup-local <TELEGRAM_BOT_TOKEN>
```

This skill installs all dependencies (CLaaS, vLLM, OpenClaw), initializes the LoRA adapter, and starts the full stack (vLLM + CLaaS API + Telegram gateway). See [`.claude/skills/setup-local/SKILL.md`](.claude/skills/setup-local/SKILL.md) for what it does under the hood.

## Development

```bash
uv sync --extra dev --extra local
uv run ruff check claas/ tests/
uv run ty check
uv run pytest -q
```

## References

1. Hübotter et al. (2026). "Reinforcement Learning via Self-Distillation." arXiv:2601.20802
2. SDPO Reference Implementation: https://github.com/lasgroup/SDPO
3. Modal GPU Memory Snapshots: https://modal.com/blog/gpu-mem-snapshots
4. vLLM: https://github.com/vllm-project/vllm
5. PEFT/LoRA: https://github.com/huggingface/peft
6. Tinker SDPO training reference (continualcode): https://github.com/sdan/continualcode

## License

MIT
