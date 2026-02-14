# CLaaS Docker Compose Stack

One-command setup for the full CLaaS + vLLM + OpenClaw stack with Telegram integration.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- For local `docker-compose.yml` only: NVIDIA GPU with >= 24 GB VRAM and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## Quick Start

```bash
cd docker
cp .env.example .env
# Edit .env and set TELEGRAM_BOT_TOKEN
docker compose up --build
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

| Service | Port | Description |
|---------|------|-------------|
| `vllm` | 8000 | Qwen3-8B with LoRA serving and sleep/wake support |
| `claas-api` | 8080 | CLaaS feedback API and distill worker |
| `openclaw` | 18789 | OpenClaw gateway with Telegram bot |
| `init` | — | One-shot: creates LoRA adapter + writes OpenClaw config |

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

## Verification

```bash
# Check vLLM models
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
| `hf-cache` | HuggingFace model cache (persists across restarts) |
| `lora-storage` | LoRA adapters and aliases |
| `openclaw-config` | OpenClaw configuration (`~/.openclaw`) |
| `claas-feedback-logs` | Feedback log JSON files |

## Configuration

All settings are in `.env`. Only `TELEGRAM_BOT_TOKEN` is required:

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | *(required)* | Bot token from @BotFather |
| `HF_TOKEN` | — | HuggingFace token for gated models |
| `MODEL` | `Qwen/Qwen3-8B` | Base model ID |
| `MAX_MODEL_LEN` | `32768` | Max sequence length |
| `GPU_MEMORY_UTILIZATION` | `0.90` | GPU VRAM fraction |
| `LORA_NAME` | `openclaw/assistant` | LoRA adapter identity |
| `CLAAS_API_PORT` | `8080` | Host port for CLaaS API |
| `OPENCLAW_PORT` | `18789` | Host port for OpenClaw gateway |

For `.env.tinker`, key variables are:

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | *(required)* | Bot token from @BotFather |
| `TINKER_API_KEY` | *(required)* | API key for Tinker SDK calls |
| `MODEL` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Hosted base model for proxy and distillation |
| `LORA_NAME` | `openclaw/assistant` | LoRA adapter identity |
| `TINKER_PROXY_PORT` | `8000` | Host port for the local OpenAI-compatible proxy |
| `CLAAS_API_PORT` | `8080` | Host port for CLaaS API |
| `OPENCLAW_PORT` | `18789` | Host port for OpenClaw gateway |

## How the Feedback Loop Works

1. User sends a Telegram message to the bot
2. OpenClaw routes it to vLLM using the `openclaw-assistant-latest` LoRA
3. You submit feedback via `POST /v1/feedback` with the conversation
4. CLaaS puts vLLM to sleep, runs an SDPO distillation step, saves the updated LoRA, and wakes vLLM
5. The next inference uses the updated adapter

## Troubleshooting

**vLLM takes forever to start**: The first run downloads Qwen3-8B. Check progress with `docker compose logs -f vllm`.

**Out of GPU memory**: Lower `GPU_MEMORY_UTILIZATION` in `.env` (e.g., `0.85`). The sleep/wake mechanism ensures vLLM and CLaaS don't use GPU simultaneously.

**Telegram bot doesn't respond**: Verify `TELEGRAM_BOT_TOKEN` is set and the bot has been started in Telegram (send `/start`). Check logs: `docker compose logs -f openclaw`.

**Feedback plugin cannot reach API (`ECONNREFUSED 127.0.0.1:8080`)**: In Docker, plugin traffic must target `http://claas-api:8080` (service DNS), not `localhost`. Re-run init and restart OpenClaw so the plugin config is refreshed.
