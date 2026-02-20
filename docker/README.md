# CLaaS Docker Compose Stack

One-command setup for the full CLaaS + vLLM + OpenClaw stack with Telegram integration. Two **profiles** are available in a single `docker-compose.yml`:

| Profile | GPU required | Inference backend |
|---------|-------------|-------------------|
| `local` | Yes (>= 24 GB VRAM) | Local vLLM |
| `tinker` | No | Hosted Tinker SDK |

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- For local `docker-compose.yml` only: NVIDIA GPU with >= 24 GB VRAM and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Quick Start

### Local profile (GPU)

```bash
cd docker
cp .env.local.example .env
# Edit .env and set TELEGRAM_BOT_TOKEN
docker compose --profile local up --build
```

The first run downloads Qwen3-8B (~16 GB) — expect the vLLM health check to take 10-20 minutes. Subsequent runs use the cached model.

## Tinker Compose Stack

Use this when you want CLaaS distillation + inference to run against hosted Tinker instead of local vLLM:

```bash
cd docker
cp .env.tinker.example .env.tinker
# Edit .env.tinker (set TELEGRAM_BOT_TOKEN + TINKER_API_KEY)
docker compose -f docker-compose.tinker.yml --env-file .env.tinker up --build
```

`docker-compose.tinker.yml` does not require a local GPU. The images install CPU-only PyTorch wheels and use Tinker-hosted inference/training.

This stack brings up:
- `tinker-proxy` (OpenAI-compatible `/v1/chat/completions` + `/v1/completions`)
- `claas-api` in `CLAAS_DISTILL_EXECUTION_MODE=tinker`
- `init` (creates `{LORA_NAME}-latest` through the API + writes OpenClaw config)
- `openclaw` (Telegram gateway pointed at `tinker-proxy`)


## Services

### Local profile

| Service | Port | Description |
|---------|------|-------------|
| `vllm` | 8000 | Qwen3-8B with LoRA serving and sleep/wake support |
| `claas-api` | 8080 | CLaaS feedback API and distill worker |
| `openclaw-local` | 18789 | OpenClaw gateway with Telegram bot |
| `init-local` | — | One-shot: creates LoRA adapter + writes OpenClaw config |

### Tinker profile

| Service | Port | Description |
|---------|------|-------------|
| `tinker-proxy` | 8000 | OpenAI-compatible proxy backed by Tinker SDK |
| `claas-api-tinker` | 8080 | CLaaS feedback API in Tinker execution mode |
| `openclaw-tinker` | 18789 | OpenClaw gateway with Telegram bot |
| `init-tinker` | — | One-shot: creates LoRA via API + writes OpenClaw config |

> **Note:** `docker-compose.tinker.yml` uses the base service names `claas-api`, `openclaw`, and `init` (no `-tinker` suffix).

## Architecture

```
                              ┌──────────────┐
                              │  init (exit)  │
                              │  Creates LoRA │
                              │  + OC config  │
                              └──────┬───────┘
                                     │ writes to volumes
          ┌──────────────────────────┼──────────────────────┐
          ▼                          ▼                      ▼
 ┌────────────────┐        ┌─────────────────┐    ┌────────────────┐
 │  vllm (:8000)  │◄───────│ claas-api(:8080) │   │ openclaw(:18789)│
 │  Qwen3-8B +    │  sleep │  Feedback API    │   │  Telegram bot  │
 │  LoRA serving  │  /wake │  Distill worker  │   │  Uses LoRA     │
 └────────────────┘        └─────────────────┘    └────────────────┘
       │                          │                       │
 [hf-cache vol]           [lora-storage vol]      [openclaw-config vol]
 [lora-storage vol]       [feedback-logs vol]
```

(In tinker profile, `tinker-proxy` replaces `vllm` and training runs via Tinker SDK instead of locally.)

## Verification

```bash
# Check vLLM / tinker-proxy models
curl http://localhost:8000/v1/models -H "Authorization: Bearer sk-local"

# Check CLaaS API
curl http://localhost:8080/

# List LoRA adapters
curl http://localhost:8080/v1/lora

# Test feedback loop
curl -X POST http://localhost:8080/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "lora_id": "openclaw/assistant-latest",
    "prompt": "hi",
    "response": "hello",
    "feedback": "good",
    "training": {"teacher_mode": "self"}
  }'
```

Send a DM to your Telegram bot — it should respond using the `openclaw-assistant-latest` LoRA model.

## Volumes

| Volume | Purpose |
|--------|---------|
| `hf-cache` | HuggingFace model cache (local profile only) |
| `lora-storage` | LoRA adapters and aliases (local profile only) |
| `openclaw-config` | OpenClaw configuration (`~/.openclaw`) |
| `claas-feedback-logs` | Feedback log JSON files |
| `tinker-state` | Tinker LoRA state (tinker profile only) |

## Configuration

Settings live in `.env` (local profile) and `.env.tinker` (tinker stack).

### Docker Compose variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | *(required)* | Bot token from @BotFather |
| `TINKER_API_KEY` | *(tinker only)* | API key for Tinker SDK |
| `HF_TOKEN` | — | HuggingFace token for gated models (local only) |
| `MODEL` | `Qwen/Qwen3-8B` (local) / `gpt-oss/GPT-OSS-120B` (tinker) | Base model ID |
| `MAX_MODEL_LEN` | `32768` | Max sequence length (local only) |
| `GPU_MEMORY_UTILIZATION` | `0.70` | GPU VRAM fraction (local only) |
| `LORA_NAME` | `openclaw/assistant` | LoRA adapter identity |
| `CLAAS_API_PORT` | `8080` | Host port for CLaaS API |
| `OPENCLAW_PORT` | `18789` | Host port for OpenClaw gateway |
| `TINKER_PROXY_PORT` | `8000` | Host port for Tinker proxy (tinker only) |
| `FEEDBACK_BATCH_SIZE` | `4` | Samples per feedback batch before triggering distill |

### CLaaS API environment variables

These are set inside the containers (via `docker-compose.yml`) and generally don't need to be changed, but can be overridden for advanced use.

#### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAAS_DISTILL_EXECUTION_MODE` | `local` | Training engine: `local`, `modal`, or `tinker` |
| `CLAAS_BASE_MODEL_ID` | `Qwen/Qwen3-8B` | Base model for LoRA training |
| `CLAAS_LORA_ROOT` | `/loras` | Root directory for LoRA adapter storage |
| `CLAAS_STORAGE_BACKEND` | `modal_volume` | Storage backend: `local_fs`, `modal_volume`, or `tinker_json` |
| `CLAAS_ALLOWED_INIT_BASE_MODELS` | `Qwen/Qwen3-8B` | Comma-separated allowlist of base models for `/v1/lora/init` |
| `FEEDBACK_LOG_DIR` | `./feedback_logs` | Directory for structured feedback JSON logs |

#### vLLM (local mode)

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

#### Tinker mode

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAAS_TINKER_API_KEY` | | Tinker SDK API key (required) |
| `CLAAS_TINKER_BASE_MODEL` | `gpt-oss/GPT-OSS-120B` | Hosted model for distillation |
| `CLAAS_TINKER_STATE_PATH` | `~/.claas/tinker_state.json` | Local path for Tinker LoRA state |
| `CLAAS_COMPLETION_CACHE_SIZE` | `100` | Inference proxy completion cache size |

#### Modal mode

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAAS_HF_SECRET_NAME` | | Name of Modal Secret containing HF credentials |

## How the Feedback Loop Works

1. User sends a Telegram message to the bot
2. OpenClaw routes it to vLLM (or tinker-proxy) using the `openclaw-assistant-latest` LoRA
3. You submit feedback via `POST /v1/feedback` with the conversation
4. CLaaS runs an SDPO distillation step, saves the updated LoRA
5. The next inference uses the updated adapter

## Troubleshooting

**vLLM takes forever to start**: The first run downloads Qwen3-8B. Check progress with `docker compose --profile local logs -f vllm`.

**Out of GPU memory**: Lower `GPU_MEMORY_UTILIZATION` in `.env` (e.g., `0.85`). The sleep/wake mechanism ensures vLLM and CLaaS don't use GPU simultaneously.

**Telegram bot doesn't respond**: Verify `TELEGRAM_BOT_TOKEN` is set and the bot has been started in Telegram (send `/start`). Check logs: `docker compose logs -f openclaw`.

**Feedback plugin cannot reach API (`ECONNREFUSED 127.0.0.1:8080`)**: In Docker, plugin traffic must target `http://claas-api:8080` (service DNS), not `localhost`. Re-run init and restart OpenClaw so the plugin config is refreshed.
