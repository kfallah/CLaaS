---
name: setup-tinker-no-docker
description: Run the CLaaS Tinker stack (tinker-proxy + API + OpenClaw/Telegram) natively without Docker. No GPU required.
---

# Setup Tinker Stack (No Docker)

Run the full CLaaS stack natively — Tinker inference proxy, CLaaS feedback API, and OpenClaw Telegram gateway — without Docker. Inference runs on Tinker's cloud, so no local GPU is needed.

## Prerequisites

- Python 3.11+, `uv`
- Node.js 22+, npm
- A Telegram bot token from @BotFather
- A Tinker API key

## Instructions

When this skill is invoked, perform the following steps. The user may pass a Telegram bot token, Tinker API key, and/or model name as arguments; if not, ask for them.

**Default model**: `Qwen/Qwen3-30B-A3B` (Tinker model names differ from HuggingFace names).

### 1. Kill any existing stack

Check for and kill any running tinker-proxy, claas-api, or openclaw processes — including old tmux sessions:

```bash
tmux kill-session -t tinker-proxy 2>/dev/null
tmux kill-session -t claas-api 2>/dev/null
tmux kill-session -t openclaw 2>/dev/null
```

### 2. Install dependencies

```bash
# Python deps (from repo root)
uv sync --extra tinker --extra dev

# OpenClaw (if not installed or outdated)
npm install -g openclaw@latest
```

If Node.js is missing or < v22, install it:
```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
apt-get install -y nodejs
```

### 3. Set up directory structure

```bash
OPENCLAW_HOME="${HOME}/.local/share/claas/openclaw-config"
mkdir -p "$OPENCLAW_HOME/.openclaw/extensions" /tmp/feedback-logs
```

### 4. Start Tinker proxy (tmux)

Replace `<TINKER_API_KEY>` and `<MODEL>` with actual values.

```bash
tmux new-session -d -s tinker-proxy \
  "CLAAS_TINKER_API_KEY='<TINKER_API_KEY>' \
   CLAAS_TINKER_BASE_MODEL='<MODEL>' \
   uv run uvicorn claas.proxy.tinker_inference_proxy:app --host 0.0.0.0 --port 8000 \
   2>&1 | tee /tmp/tinker-proxy.log"
```

### 5. Start CLaaS API (tmux)

```bash
tmux new-session -d -s claas-api \
  "CLAAS_DISTILL_EXECUTION_MODE=tinker \
   CLAAS_TINKER_API_KEY='<TINKER_API_KEY>' \
   CLAAS_TINKER_BASE_MODEL='<MODEL>' \
   CLAAS_TINKER_STATE_PATH='${HOME}/.claas/tinker_state.json' \
   VLLM_BASE_URL='http://localhost:8000' \
   VLLM_API_KEY=sk-local \
   FEEDBACK_LOG_DIR=/tmp/feedback-logs \
   uv run uvicorn claas.api:web_app --host 0.0.0.0 --port 8080 \
   2>&1 | tee /tmp/claas-api.log"
```

### 6. Wait for health

Poll both services (timeout after 60 seconds):

```bash
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 \
     && curl -sf http://localhost:8080/ >/dev/null 2>&1; then
    echo "Both services healthy"
    break
  fi
  sleep 1
done
```

### 7. Initialize LoRA adapter

Force-initialize the LoRA via the CLaaS API. Always use `force: true` to ensure the checkpoint is fresh for this Tinker session (stale checkpoints from previous sessions will cause 400 errors):

```bash
curl -s -X POST http://localhost:8080/v1/lora/init \
  -H 'Content-Type: application/json' \
  -d '{"lora_id": "openclaw/assistant-latest", "base_model": "<MODEL>", "force": true}'
```

Verify:
```bash
curl -s http://localhost:8080/v1/lora
```

### 8. Write OpenClaw config

**IMPORTANT**: Write config to `$OPENCLAW_HOME/.openclaw/openclaw.json`. OpenClaw runs with `HOME=$OPENCLAW_HOME` and reads its config from `$HOME/.openclaw/openclaw.json`. The init-stack.py script writes to `$OPENCLAW_HOME/openclaw.json` (without the `.openclaw/` subdirectory) because in Docker, `OPENCLAW_HOME` is mounted directly as the `.openclaw` volume. Locally the paths differ, so we write to the correct location directly.

Normalize the LoRA alias for model IDs: replace non-alphanumeric characters (except `.`, `-`, `_`) with `-`. For `openclaw/assistant-latest` this produces `openclaw-assistant-latest`.

Write `$OPENCLAW_HOME/.openclaw/openclaw.json` with this structure (substitute `<MODEL_ID>` with normalized base model name, e.g. `qwen3-30b-a3b`, and `<TELEGRAM_BOT_TOKEN>` with the actual token):

```json
{
  "models": {
    "mode": "merge",
    "providers": {
      "local": {
        "baseUrl": "http://localhost:8000/v1",
        "apiKey": "sk-local",
        "api": "openai-completions",
        "models": [
          {
            "id": "<MODEL_ID>",
            "name": "<MODEL_ID>",
            "api": "openai-completions",
            "reasoning": false,
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 32768,
            "maxTokens": 8192
          },
          {
            "id": "openclaw-assistant-latest",
            "name": "openclaw-assistant-latest",
            "api": "openai-completions",
            "reasoning": false,
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 32768,
            "maxTokens": 8192
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "compaction": {"mode": "safeguard"},
      "model": {"primary": "local/openclaw-assistant-latest"},
      "models": {
        "local/<MODEL_ID>": {"alias": "<MODEL_ID> (Tinker proxy)"},
        "local/openclaw-assistant-latest": {"alias": "openclaw-assistant-latest (Tinker proxy)"}
      },
      "maxConcurrent": 4,
      "subagents": {"maxConcurrent": 8}
    }
  },
  "tools": {
    "byProvider": {
      "local/<MODEL_ID>": {"deny": ["*"]},
      "local/openclaw-assistant-latest": {"deny": ["*"]}
    }
  },
  "messages": {"ackReactionScope": "group-mentions"},
  "commands": {"native": "auto", "nativeSkills": "auto"},
  "channels": {
    "telegram": {
      "enabled": true,
      "dmPolicy": "open",
      "allowFrom": ["*"],
      "groupPolicy": "open",
      "groups": {"*": {"requireMention": false}},
      "streamMode": "off",
      "actions": {"reactions": true, "sendMessage": true, "deleteMessage": true},
      "botToken": "<TELEGRAM_BOT_TOKEN>"
    }
  },
  "gateway": {
    "port": 18789,
    "mode": "local",
    "bind": "loopback",
    "http": {"endpoints": {"chatCompletions": {"enabled": true}}},
    "auth": {"mode": "token", "token": "openclaw-local-dev-token"}
  },
  "plugins": {
    "entries": {
      "telegram": {"enabled": true},
      "claas-feedback": {
        "enabled": true,
        "config": {
          "claasApiUrl": "http://localhost:8080",
          "loraId": "openclaw/assistant-latest",
          "feedbackBatchSize": 4
        }
      }
    }
  }
}
```

Also write the agent-level models config to `$OPENCLAW_HOME/.openclaw/agents/main/agent/models.json` (create parent dirs). This should contain only the `providers` object from above (same `local` provider block with both models).

### 9. Install feedback plugin

Copy the CLaaS feedback plugin into OpenClaw's **runtime** extensions directory (NOT `$OPENCLAW_HOME/extensions/`):

```bash
mkdir -p "$OPENCLAW_HOME/.openclaw/extensions"
cp -r plugins/claas-feedback "$OPENCLAW_HOME/.openclaw/extensions/claas-feedback"
```

### 10. Start OpenClaw gateway (tmux)

**IMPORTANT**: Pass `CLAAS_API_URL` and `CLAAS_TINKER_PROXY_URL` as environment variables. The feedback plugin checks these env vars as fallback — the plugin config in `openclaw.json` may not be passed through by all OpenClaw versions.

```bash
tmux new-session -d -s openclaw \
  "TELEGRAM_BOT_TOKEN='<TELEGRAM_BOT_TOKEN>' \
   CLAAS_VLLM_BASE_URL='http://localhost:8000' \
   CLAAS_TINKER_PROXY_URL='http://localhost:8000' \
   CLAAS_API_URL='http://localhost:8080' \
   HOME='$OPENCLAW_HOME' \
   OPENCLAW_GATEWAY_TOKEN='openclaw-local-dev-token' \
   openclaw gateway --port 18789 --bind lan --allow-unconfigured --verbose \
   2>&1 | tee /tmp/openclaw.log"
```

### 11. Approve Telegram pairing

Wait ~5 seconds, then check the OpenClaw logs for the Telegram bot username and any pairing requests:

```bash
sleep 5
grep -E 'agent model:|listening|telegram.*starting|error|invalid|plugin' /tmp/openclaw.log | tail -10
```

If the user has already paired before, their Telegram account is remembered. If they report a pairing code, approve it:

```bash
HOME="$OPENCLAW_HOME" openclaw pairing approve telegram <CODE>
```

### 12. Verify the full stack

```bash
# Tinker proxy
curl -s http://localhost:8000/v1/models

# CLaaS API health
curl -s http://localhost:8080/v1/health

# LoRA adapters
curl -s http://localhost:8080/v1/lora

# OpenClaw — confirm agent model and feedback plugin
grep -E 'agent model:|Registered plugin command: /feedback' /tmp/openclaw.log
```

Confirm:
- Tinker proxy serves the expected model
- CLaaS API is healthy
- `openclaw/assistant-latest` LoRA exists
- OpenClaw agent model is `local/openclaw-assistant-latest` (NOT `anthropic/claude-*`)
- `/feedback` command is registered

Report status to the user with the Telegram bot username.

## Troubleshooting

- **OpenClaw uses `anthropic/claude-*` instead of local model**: Config was written to the wrong path. OpenClaw reads `$HOME/.openclaw/openclaw.json` (where `HOME=$OPENCLAW_HOME`). Make sure config is at `$OPENCLAW_HOME/.openclaw/openclaw.json`, not `$OPENCLAW_HOME/openclaw.json`.
- **Feedback plugin "not found"**: Plugin must be in `$OPENCLAW_HOME/.openclaw/extensions/claas-feedback/`, not `$OPENCLAW_HOME/extensions/claas-feedback/`.
- **Feedback fails with "fetch failed"**: The feedback plugin is using Docker service names (`http://claas-api:8080`). Ensure `CLAAS_API_URL=http://localhost:8080` and `CLAAS_TINKER_PROXY_URL=http://localhost:8000` are set as env vars on the OpenClaw tmux session.
- **Feedback fails with 500 / "Invalid checkpoint tinker path"**: The LoRA adapter has a stale Tinker checkpoint. Re-init with `force: true`: `curl -X POST http://localhost:8080/v1/lora/init -H 'Content-Type: application/json' -d '{"lora_id": "openclaw/assistant-latest", "base_model": "<MODEL>", "force": true}'`
- **Node.js too old**: OpenClaw requires Node 22+. Install via NodeSource (see step 2).

## Log files

| Service | Log | tmux session |
|---------|-----|--------------|
| Tinker proxy | `/tmp/tinker-proxy.log` | `tinker-proxy` |
| CLaaS API | `/tmp/claas-api.log` | `claas-api` |
| OpenClaw | `/tmp/openclaw.log` | `openclaw` |

Attach to any session with `tmux attach -t <name>`. All processes persist if the Claude Code session ends.
