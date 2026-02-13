# CLaaS Docker Compose Stack

One-command setup for the full CLaaS + vLLM + OpenClaw stack with Telegram integration.

## Prerequisites

- NVIDIA GPU with >= 24 GB VRAM
- [Docker](https://docs.docker.com/get-docker/) with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- A Telegram bot token from [@BotFather](https://t.me/BotFather)

## Quick Start

```bash
cd docker
cp .env.example .env
# Edit .env and set TELEGRAM_BOT_TOKEN
docker compose up --build
```

The first run downloads Qwen3-8B (~16 GB) — expect the vLLM health check to take 10-20 minutes. Subsequent runs use the cached model.

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
