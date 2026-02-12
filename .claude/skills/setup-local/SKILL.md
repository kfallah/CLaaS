---
name: setup-local
description: Set up the full CLaaS stack (vLLM + API + OpenClaw/Telegram) directly on the host without Docker. Use when Docker is unavailable or you want a native setup.
---

# Setup Local (Dockerless)

Run the full CLaaS stack natively — vLLM, CLaaS feedback API, and OpenClaw Telegram gateway — without Docker.

## Prerequisites

- Python 3.11+, pip
- Node.js 22+, npm
- NVIDIA GPU with >= 24 GB VRAM and CUDA drivers
- A Telegram bot token from @BotFather

## Instructions

When this skill is invoked, perform the following steps. The user may pass a Telegram bot token as an argument; if not, ask for it.

### 1. Install dependencies

```bash
# Install CLaaS (from repo root)
pip install -e .

# Install vLLM
pip install vllm

# Install OpenClaw
npm install -g openclaw@latest
```

If Node.js is missing or < v22, install it:
```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs
```

### 2. Run the init container script

Create the LoRA adapter and OpenClaw config. Set `VLLM_BASE_URL` to `http://localhost:8000/v1` (not the Docker service name).

```bash
mkdir -p /loras /openclaw-config

CLAAS_STORAGE_BACKEND=local_fs \
CLAAS_LORA_ROOT=/loras \
LORA_NAME=openclaw/assistant \
MODEL=Qwen/Qwen3-8B \
VLLM_BASE_URL=http://localhost:8000/v1 \
API_KEY=sk-local \
TELEGRAM_BOT_TOKEN=<token> \
OPENCLAW_HOME=/openclaw-config \
  python3 docker/scripts/init-stack.py
```

Verify: `/loras/.aliases.json` should exist with an `openclaw/assistant-latest` entry.

### 2b. Install feedback plugin

Copy the CLaaS feedback plugin into the OpenClaw extensions directory so it loads on startup:

```bash
mkdir -p /openclaw-config/extensions
cp -r plugins/claas-feedback /openclaw-config/extensions/claas-feedback
```

### 3. Start vLLM (background)

```bash
export MODEL=Qwen/Qwen3-8B HOST=0.0.0.0 PORT=8000 API_KEY=sk-local
export SERVED_MODEL_NAMES=qwen3-8b MAX_MODEL_LEN=32768 GPU_MEMORY_UTILIZATION=0.90
export ENABLE_SLEEP_MODE=1 VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
export ENABLE_AUTO_TOOL_CHOICE=1 TOOL_CALL_PARSER=qwen3_xml
export LORA_ROOT=/loras LORA_ALIAS_FILE=/loras/.aliases.json INCLUDE_ALIAS_LORAS=1

bash scripts/openclaw-local/start_vllm_qwen3_8b.sh >> /tmp/vllm.log 2>&1 &
```

Wait for health check:
```bash
until curl -sf http://localhost:8000/health; do sleep 5; done
```

First run downloads Qwen3-8B (~16 GB); expect 5-20 minutes.

### 4. Start CLaaS API (background)

```bash
CLAAS_STORAGE_BACKEND=local_fs \
CLAAS_LORA_ROOT=/loras \
CLAAS_DISTILL_EXECUTION_MODE=local \
VLLM_BASE_URL=http://localhost:8000 \
VLLM_API_KEY=sk-local \
FEEDBACK_LOG_DIR=/tmp/feedback-logs \
  uvicorn claas.api:web_app --host 0.0.0.0 --port 8080 >> /tmp/claas-api.log 2>&1 &
```

Verify: `curl http://localhost:8080/v1/health`

### 5. Start OpenClaw gateway (background)

The init script writes config to `OPENCLAW_HOME/.openclaw/openclaw.json`. Point `HOME` there so OpenClaw finds it:

```bash
TELEGRAM_BOT_TOKEN=<token> \
VLLM_BASE_URL=http://localhost:8000 \
HOME=/openclaw-config \
OPENCLAW_GATEWAY_TOKEN=openclaw-local-dev-token \
  openclaw gateway --port 18789 --bind lan --allow-unconfigured --verbose >> /tmp/openclaw.log 2>&1 &
```

### 6. Verify the stack

```bash
# vLLM models
curl -s http://localhost:8000/v1/models -H "Authorization: Bearer sk-local"

# CLaaS health
curl -s http://localhost:8080/v1/health

# LoRA adapters
curl -s http://localhost:8080/v1/lora

# OpenClaw logs
tail -5 /tmp/openclaw.log
```

Report the status of all four components and the Telegram bot username.

## Troubleshooting

- **vLLM OOM on startup**: Lower `GPU_MEMORY_UTILIZATION` (e.g. `0.85`). The sleep/wake mechanism ensures vLLM and CLaaS don't use GPU simultaneously.
- **OpenClaw exits immediately**: Ensure the config at `/openclaw-config/.openclaw/openclaw.json` has `channels.telegram.enabled: true`, `channels.telegram.botToken` set, and `gateway.mode: "local"`. Run `HOME=/openclaw-config openclaw doctor --fix` if needed.
- **Node.js too old**: OpenClaw requires Node 22+. Install via NodeSource (see step 1).

## Log files

| Service | Log |
|---------|-----|
| vLLM | `/tmp/vllm.log` |
| CLaaS API | `/tmp/claas-api.log` |
| OpenClaw | `/tmp/openclaw.log` |
