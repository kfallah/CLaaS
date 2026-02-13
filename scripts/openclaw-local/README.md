# OpenClaw Local Stack: Qwen3-8B + vLLM + LoRA

Run OpenClaw with a local vLLM server serving Qwen3-8B and CLaaS LoRA adapters.

## Prerequisites

- NVIDIA GPU with >= 24 GB VRAM (L40S, A100, RTX 4090, etc.)
- `uv` installed
- OpenClaw CLI (`openclaw` binary in PATH)
- Python 3.10+

Install project dependencies (including vLLM) from repo root:

```bash
uv sync --extra teacher --extra dev
```

Verify your setup:

```bash
nvidia-smi                 # GPU visible
vllm --version             # vLLM installed
openclaw --version         # OpenClaw CLI available
```

## Quick Start

```bash
cd /path/to/CLaaS

# 1. Create your env config from the template
cp scripts/openclaw-local/openclaw-local.env.example .env.openclaw-local

# 2. Source it
set -a; source .env.openclaw-local; set +a

# 3. Run the supervised stack (auto-restarts on crash)
uv run bash scripts/openclaw-local/run_openclaw_local_stack.sh
```

This starts vLLM + the OpenClaw gateway and monitors both. If either
process exits or fails a health check, the stack restarts automatically.

## Step-by-Step (Manual)

If you prefer to run each component separately:

### 1. Configure environment

```bash
cp scripts/openclaw-local/openclaw-local.env.example .env.openclaw-local
# Edit .env.openclaw-local to taste (model, port, LoRA modules, etc.)
set -a; source .env.openclaw-local; set +a
```

### 2. Start vLLM

```bash
uv run bash scripts/openclaw-local/start_vllm_qwen3_8b.sh
```

This runs `vllm serve Qwen/Qwen3-8B` with:
- LoRA serving enabled (reads aliases from `.local_loras/.aliases.json`)
- Sleep mode for CLaaS feedback orchestration
- Tool calling support (qwen3_xml parser)

Wait for the health check to pass:

```bash
curl http://127.0.0.1:8000/health
```

Verify the models are loaded:

```bash
curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer sk-local"
```

### 3. Configure OpenClaw to use local vLLM

```bash
uv run python scripts/openclaw-local/configure_openclaw_local_models.py
```

This writes `~/.openclaw/openclaw.json` and
`~/.openclaw/agents/main/agent/models.json` with a `local` provider
pointing at the vLLM server.

### 4. Start the OpenClaw gateway

```bash
openclaw gateway --port 18789 --verbose
```

Or use the script (waits for vLLM health first):

```bash
uv run bash scripts/openclaw-local/start_openclaw_gateway_local.sh
```

### 5. Verify

```bash
openclaw status
```

You should see:
- Gateway: reachable
- Telegram (or other channels): ON / OK
- Sessions using model `qwen3-8b`

## Environment Variables

All variables have sensible defaults. Edit `.env.openclaw-local` to override.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `Qwen/Qwen3-8B` | HuggingFace model ID |
| `HOST` | `127.0.0.1` | vLLM bind address |
| `PORT` | `8000` | vLLM port |
| `API_KEY` | `sk-local` | vLLM API key |
| `SERVED_MODEL_NAMES` | `qwen3-8b` | Comma-separated model names for the OpenAI-compatible API |
| `MAX_MODEL_LEN` | `32768` | Maximum sequence length |
| `GPU_MEMORY_UTILIZATION` | `0.70` | Fraction of GPU VRAM to use |
| `ENABLE_SLEEP_MODE` | `1` | Enable vLLM sleep/wake for CLaaS feedback |
| `ENABLE_AUTO_TOOL_CHOICE` | `1` | Enable tool calling |
| `TOOL_CALL_PARSER` | `qwen3_xml` | Tool call format parser |
| `LORA_MODULES` | *(empty)* | Explicit LoRA modules (`alias=/path,alias2=/path2`) |
| `INCLUDE_ALIAS_LORAS` | `1` | Auto-load LoRAs from `.local_loras/.aliases.json` |
| `MODEL_IDS` | `qwen3-8b` | Model IDs for OpenClaw config |
| `PRIMARY_MODEL` | `qwen3-8b` | Default model for OpenClaw agents |
| `BASE_URL` | `http://127.0.0.1:8000/v1` | vLLM base URL for OpenClaw |

## LoRA Adapters

### Directory structure

LoRA adapters live in `CLaaS/.local_loras/`:

```
.local_loras/
  .aliases.json                          # alias -> directory mapping
  openclaw/
    telegram-conversational-style-init/
      adapter_config.json
      adapter_model.safetensors
```

### Adding a LoRA

**From CLaaS API (export a trained adapter):**

```bash
curl -L "http://localhost:8080/v1/lora/export?lora_id=user/my-lora-init" \
  -o my-lora.zip
unzip -o my-lora.zip -d .local_loras/user/my-lora-init
```

**From HuggingFace:**

```bash
git clone https://huggingface.co/your-org/your-lora .local_loras/your-org/your-lora
```

Then add an alias in `.local_loras/.aliases.json`:

```json
{
  "your-org/your-lora-latest": "your-org/your-lora"
}
```

Restart vLLM to pick up the new adapter.

### Using LoRA models

Once loaded, LoRA adapters appear as separate models in the vLLM API.
Reference them by their alias name in API calls:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local" \
  -d '{
    "model": "openclaw-telegram-conversational-style-latest",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'
```

## Troubleshooting

### Telegram 409 conflict

```
Telegram getUpdates conflict: terminated by other getUpdates request
```

Another bot instance is polling with the same token. Stop the other
instance, then restart the gateway:

```bash
# Kill the gateway
pkill -f openclaw-gateway

# Restart
openclaw gateway --port 18789 --verbose
```

### vLLM OOM

If vLLM runs out of GPU memory, reduce `MAX_MODEL_LEN` or
`GPU_MEMORY_UTILIZATION` in `.env.openclaw-local`:

```bash
MAX_MODEL_LEN=16384
GPU_MEMORY_UTILIZATION=0.85
```

### Gateway can't reach vLLM

The gateway startup script waits up to 180s for vLLM health. If vLLM
is slow to load (large model, cold HF cache), increase `VLLM_WAIT_SECONDS`:

```bash
VLLM_WAIT_SECONDS=300 uv run bash scripts/openclaw-local/run_openclaw_local_stack.sh
```

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `run_openclaw_local_stack.sh` | Supervised launcher: starts vLLM + gateway, auto-restarts on failure |
| `start_vllm_qwen3_8b.sh` | Starts vLLM with Qwen3-8B, LoRA modules, sleep mode, tool calling |
| `configure_openclaw_local_models.py` | Writes OpenClaw config JSON pointing at local vLLM |
| `start_openclaw_gateway_local.sh` | Waits for vLLM health, then starts `openclaw gateway` |
| `openclaw-local.env.example` | Template for environment variables |
