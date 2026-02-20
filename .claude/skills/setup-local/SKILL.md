---
name: setup-local
description: Set up the full CLaaS stack (vLLM + API + OpenClaw/Telegram) directly on the host without Docker. Use when Docker is unavailable or you want a native setup.
---

# Setup Local (Dockerless)

> **Docker is strongly preferred.** If Docker is available, use it instead:
> ```bash
> cd docker
> cp .env.local.example .env
> # Set TELEGRAM_BOT_TOKEN in .env
> docker compose --profile local up --build
> ```
> See `docker/README.md`. Only continue below if Docker is unavailable.

---

Run the full CLaaS stack natively without Docker. **Tested on an NVIDIA RTX 5090.**

## Prerequisites

- `uv`, Node.js 22+, npm
- NVIDIA GPU with >= 24 GB VRAM, CUDA drivers and toolkit (`nvidia-smi` should work)
- A Telegram bot token from @BotFather

## Instructions

Ask for the Telegram bot token if not provided as an argument, then work through each step.

### 1. Install dependencies

```bash
# Python deps (from repo root)
uv sync --extra local --extra teacher --extra dev

# pyproject.toml pins torch to CPU — reinstall with CUDA
# Match cu1XX to your CUDA version from nvidia-smi (e.g. cu124, cu128)
uv pip install "torch>=2.1.0+cu128" torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128 --reinstall
uv pip install "numpy<2.3"  # numba compatibility

# OpenClaw
npm install -g openclaw@latest
```

> **Python headers:** Triton JIT-compiles a CUDA extension at first vLLM startup and needs `Python.h`. System Python often lacks dev headers. Use a uv-managed Python (which ships with headers) to avoid this:
> ```bash
> uv python install 3.12
> uv venv --python 3.12 --clear .venv
> # Re-run the installs above
> ```

### 2. Initialize LoRA and OpenClaw config

```bash
LORA_ROOT="${HOME}/.local/share/claas/loras"
OPENCLAW_HOME="${HOME}/.local/share/claas/openclaw-config"
mkdir -p "$LORA_ROOT" "$OPENCLAW_HOME"

CLAAS_STORAGE_BACKEND=local_fs \
CLAAS_LORA_ROOT="$LORA_ROOT" \
LORA_NAME=openclaw/assistant \
MODEL=Qwen/Qwen3-8B \
VLLM_BASE_URL=http://localhost:8000/v1 \
API_KEY=sk-local \
TELEGRAM_BOT_TOKEN=<token> \
OPENCLAW_HOME="$OPENCLAW_HOME" \
  uv run python3 docker/scripts/init-stack.py
```

The init script writes config to `$OPENCLAW_HOME/` but OpenClaw (run with `HOME="$OPENCLAW_HOME"`) reads from `$OPENCLAW_HOME/.openclaw/`. Copy and fix:

```bash
cp "$OPENCLAW_HOME/openclaw.json" "$OPENCLAW_HOME/.openclaw/openclaw.json"
cp "$OPENCLAW_HOME/agents/main/agent/models.json" \
   "$OPENCLAW_HOME/.openclaw/agents/main/agent/models.json"

# Replace Docker service hostname with localhost
sed -i 's|http://claas-api:8080|http://localhost:8080|g' \
  "$OPENCLAW_HOME/.openclaw/openclaw.json"

# Feedback plugin
mkdir -p "$OPENCLAW_HOME/.openclaw/extensions"
cp -r plugins/claas-feedback "$OPENCLAW_HOME/.openclaw/extensions/claas-feedback"

# Auth credentials for the local vLLM provider
cat > "$OPENCLAW_HOME/.openclaw/agents/main/agent/auth-profiles.json" << 'EOF'
{
  "version": 1,
  "profiles": {
    "local-default": {
      "type": "api_key",
      "provider": "local",
      "key": "sk-local"
    }
  }
}
EOF
```

### 3. Start vLLM

```bash
LORA_ROOT="${HOME}/.local/share/claas/loras"
export PATH="$(pwd)/.venv/bin:$PATH"  # puts 'vllm' on PATH
export MODEL=Qwen/Qwen3-8B HOST=0.0.0.0 PORT=8000 API_KEY=sk-local
export SERVED_MODEL_NAMES=qwen3-8b MAX_MODEL_LEN=32768 GPU_MEMORY_UTILIZATION=0.70
export ENABLE_SLEEP_MODE=1 VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
export ENABLE_AUTO_TOOL_CHOICE=1 TOOL_CALL_PARSER=qwen3_xml
export LORA_ROOT="$LORA_ROOT" LORA_ALIAS_FILE="$LORA_ROOT/.aliases.json" INCLUDE_ALIAS_LORAS=1

bash scripts/openclaw-local/start_vllm_qwen3_8b.sh >> /tmp/vllm.log 2>&1 &

# First run downloads Qwen3-8B (~16 GB) — expect 5-20 min
until curl -sf http://localhost:8000/health; do sleep 5; done && echo "vLLM ready"
```

### 4. Start CLaaS API

```bash
CLAAS_STORAGE_BACKEND=local_fs \
CLAAS_LORA_ROOT="${HOME}/.local/share/claas/loras" \
CLAAS_DISTILL_EXECUTION_MODE=local \
VLLM_BASE_URL=http://localhost:8000 \
VLLM_API_KEY=sk-local \
FEEDBACK_LOG_DIR=/tmp/feedback-logs \
  uv run uvicorn claas.api:web_app --host 0.0.0.0 --port 8080 >> /tmp/claas-api.log 2>&1 &

curl -sf http://localhost:8080/v1/health
```

### 5. Start OpenClaw

```bash
OPENCLAW_HOME="${HOME}/.local/share/claas/openclaw-config"

TELEGRAM_BOT_TOKEN=<token> \
VLLM_BASE_URL=http://localhost:8000 \
HOME="$OPENCLAW_HOME" \
OPENCLAW_GATEWAY_TOKEN=openclaw-local-dev-token \
  openclaw gateway --port 18789 --bind lan --allow-unconfigured --verbose >> /tmp/openclaw.log 2>&1 &
```

### 6. Verify and approve pairing

```bash
curl -s http://localhost:8000/v1/models -H "Authorization: Bearer sk-local"
curl -s http://localhost:8080/v1/health
curl -s http://localhost:8080/v1/lora
tail -10 /tmp/openclaw.log  # should show "agent model: local/openclaw-assistant-latest"
```

When the user first messages the bot they'll receive a pairing code. Approve it:
```bash
HOME="${HOME}/.local/share/claas/openclaw-config" openclaw pairing approve telegram <CODE>
```

Report the status of all four components and the Telegram bot username.

## Troubleshooting

| Error | Fix |
|-------|-----|
| `vllm: not found` | Prepend `.venv/bin` to PATH |
| `libtorch_cuda.so not found` | Reinstall torch with CUDA index (step 1) |
| `Numba needs NumPy 2.2 or less` | `uv pip install "numpy<2.3"` |
| `Python.h: No such file or directory` | Recreate venv with uv-managed Python (step 1 note) |
| `No API key found for provider "local"` | Create `auth-profiles.json` (step 2) |
| vLLM OOM | Lower `GPU_MEMORY_UTILIZATION` to `0.60` |

## Logs

| Service | Log |
|---------|-----|
| vLLM | `/tmp/vllm.log` |
| CLaaS API | `/tmp/claas-api.log` |
| OpenClaw | `/tmp/openclaw.log` |
