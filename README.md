# CLaaS: Continual Learning as a Service

Continual learning via self-distillation for OpenClaw. Personalize the model weights of your OpenClaw assistant using text feedback.

## Hybrid engine

The locally hosted request path is driven by a hybrid engine that switches between:

- **Serving mode**: route request traffic through vLLM (local or remote) for low-latency generation.
- **Update mode**: run a single SDPO LoRA step using the provided feedback to adapt the adapter.

In practice, the flow is: request is answered by vLLM, then the engine performs (or schedules) the training step, and subsequent requests can use the updated adapter. The engine can prefer local or remote teacher inference depending on `teacher_mode`.

![Hybrid engine diagram](assets/image.png)

## Installation

**Prerequisites:** Python 3.11+, `uv`, and [Docker](https://docs.docker.com/get-docker/).

### Local (GPU)

Requires an NVIDIA GPU with >= 24 GB VRAM (L40S, A100, RTX 4090, etc.) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

**Docker Compose (recommended):**

```bash
cd docker
cp .env.example .env
# Edit .env — set TELEGRAM_BOT_TOKEN (from @BotFather)
docker compose --profile local up --build
```

This brings up vLLM with Qwen3-8B, the CLaaS feedback API, and OpenClaw's Telegram gateway. See [`docker/README.md`](docker/README.md) for details.

**Manual install:**

```bash
uv sync --extra local
```

Then start vLLM and the API yourself. See [Quick Start](#quick-start) and [`scripts/openclaw-local/README.md`](scripts/openclaw-local/README.md) for the full supervised local stack.

If you use [Claude Code](https://claude.ai/claude-code), `/setup-local <TELEGRAM_BOT_TOKEN>` installs all deps and starts the full local stack automatically.

### Tinker (no GPU)

Uses the Tinker SDK for hosted distillation and inference. Requires a `TINKER_API_KEY`.

**Docker Compose (recommended):**

```bash
cd docker
cp .env.tinker.example .env.tinker
# Edit .env.tinker — set TELEGRAM_BOT_TOKEN and TINKER_API_KEY
docker compose -f docker-compose.tinker.yml --env-file .env.tinker up --build
```

**Manual install:**

```bash
uv sync --extra tinker
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch
```

If you use [Claude Code](https://claude.ai/claude-code), `/setup-tinker` deploys the Tinker Docker stack. Use `/clear-tinker-storage` to delete all Tinker checkpoints and free storage.

### Modal (remote GPU)

Runs distillation remotely on Modal (L40S). Requires a Modal account.

```bash
uv sync --extra local
uv run modal token new
```

Deploy:

```bash
# Set HF_TOKEN if using gated models
export HF_TOKEN=...
export CLAAS_BASE_MODEL_ID=Qwen/Qwen3-8B
uv run modal deploy -m claas.deploy
```

The deployed app exposes the same API at `https://your-app--claas-distill-fastapi-app.modal.run`. LoRAs are stored in the `claas-loras` Modal Volume.

If you use [Claude Code](https://claude.ai/claude-code), `/setup-modal` deploys the CLaaS distillation service to Modal.

## Quick Start

For manual (non-Docker) local setup:

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

For the full supervised local stack (vLLM + gateway + auto-restart, multi-LoRA, Telegram), see [`scripts/openclaw-local/README.md`](scripts/openclaw-local/README.md).

## Configuration

All configuration is via environment variables. Docker Compose users set these in `.env` / `.env.tinker`; manual installs export them directly.

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAAS_DISTILL_EXECUTION_MODE` | `local` | Training engine: `local`, `modal`, or `tinker` |
| `CLAAS_BASE_MODEL_ID` | `Qwen/Qwen3-8B` | Base model for LoRA training |
| `CLAAS_LORA_ROOT` | `/loras` | Root directory for LoRA adapter storage |
| `CLAAS_STORAGE_BACKEND` | `modal_volume` | Storage backend: `local_fs`, `modal_volume`, or `tinker_json` |
| `CLAAS_ALLOWED_INIT_BASE_MODELS` | `Qwen/Qwen3-8B` | Comma-separated allowlist of base models for `/v1/lora/init` |
| `HF_TOKEN` | | Hugging Face token (required for gated models) |
| `FEEDBACK_LOG_DIR` | `./feedback_logs` | Directory for structured feedback JSON logs |

### vLLM (local mode)

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_BASE_URL` | `http://127.0.0.1:8000` | vLLM server URL |
| `VLLM_API_KEY` | `sk-local` | API key for vLLM |
| `CLAAS_ATTN_IMPLEMENTATION` | `sdpa` | Attention backend (`sdpa`, `flash_attention_2`) |
| `FEEDBACK_LOCK_TIMEOUT_S` | `120` | Per-LoRA lock timeout (seconds) |
| `FEEDBACK_WAKE_ON_FAILURE` | `true` | Wake vLLM if the distill step fails |
| `FEEDBACK_MIN_FREE_VRAM_GB` | `20` | Minimum free VRAM before training |
| `FEEDBACK_SLEEP_VERIFY_TIMEOUT_S` | `30` | Timeout waiting for vLLM to sleep |
| `FEEDBACK_DRAIN_TIMEOUT_S` | `30` | Timeout draining vLLM queue before sleep |

### Tinker mode

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAAS_TINKER_API_KEY` | | Tinker SDK API key (required) |
| `CLAAS_TINKER_BASE_MODEL` | `gpt-oss/GPT-OSS-120B` | Hosted model for distillation |
| `CLAAS_TINKER_STATE_PATH` | `~/.claas/tinker_state.json` | Local path for Tinker LoRA state |
| `CLAAS_COMPLETION_CACHE_SIZE` | `100` | Inference proxy completion cache size |

### Modal mode

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAAS_HF_SECRET_NAME` | | Name of Modal Secret containing HF credentials |

## References

1. Hübotter et al. (2026). "Reinforcement Learning via Self-Distillation." arXiv:2601.20802
2. SDPO Reference Implementation: https://github.com/lasgroup/SDPO
3. Modal GPU Memory Snapshots: https://modal.com/blog/gpu-mem-snapshots
4. vLLM: https://github.com/vllm-project/vllm
5. PEFT/LoRA: https://github.com/huggingface/peft
6. Tinker SDPO training reference (continualcode): https://github.com/sdan/continualcode

## License

MIT
