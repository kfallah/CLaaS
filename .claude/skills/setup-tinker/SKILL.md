---
name: setup-tinker
description: Deploy the CLaaS Tinker stack (API + OpenClaw/Telegram) via Docker Compose. No GPU required.
---

# Setup Tinker Stack

Deploy the full CLaaS stack using Docker Compose with Tinker-hosted inference — no local GPU needed.

## Prerequisites

- Docker running
- A Telegram bot token from @BotFather
- A Tinker API key

## Instructions

When this skill is invoked, perform the following steps. The user may pass a Telegram bot token and/or Tinker API key as arguments; if not, ask for them.

### 1. Check prerequisites

Verify Docker is running:

```bash
docker info > /dev/null 2>&1
```

If Docker is not running, tell the user to start Docker Desktop (or the Docker daemon) and re-run.

### 2. Configure `.env.tinker`

From the repo root, check if `docker/.env.tinker` exists. If not, copy the example:

```bash
cp docker/.env.tinker.example docker/.env.tinker
```

Set the required variables in `docker/.env.tinker`:

| Variable | Required | Default |
|----------|----------|---------|
| `TELEGRAM_BOT_TOKEN` | Yes | — |
| `TINKER_API_KEY` | Yes | — |
| `MODEL` | No | `Qwen/Qwen3-30B-A3B` |
| `LORA_NAME` | No | `openclaw/assistant` |
| `CLAAS_API_PORT` | No | `8080` |
| `OPENCLAW_PORT` | No | `18789` |

Prompt the user for `TELEGRAM_BOT_TOKEN` and `TINKER_API_KEY` if they were not passed as arguments. Update the file using sed or the Edit tool.

### 3. Start the stack

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env.tinker --profile tinker up --build -d
```

### 4. Wait for health

Poll the services until they are ready (timeout after 3 minutes):

```bash
# claas-api (serves both inference and training)
until curl -sf http://localhost:8080/ > /dev/null 2>&1; do sleep 5; done
```

Then wait for the `init-tinker` container to complete:

```bash
docker compose -f docker/docker-compose.yml --env-file docker/.env.tinker --profile tinker logs -f init-tinker 2>&1 | head -50
```

### 5. Verify the stack

```bash
# Models (served by CLaaS API)
curl -s http://localhost:8080/v1/models

# CLaaS health
curl -s http://localhost:8080/v1/health

# LoRA adapters
curl -s http://localhost:8080/v1/lora

# OpenClaw logs — look for Telegram bot username
docker compose -f docker/docker-compose.yml --env-file docker/.env.tinker --profile tinker logs openclaw-tinker 2>&1 | tail -20
```

Report the status of all services and the Telegram bot username from the OpenClaw logs.

### 6. Confirm

Tell the user the stack is running and they can send a DM to their Telegram bot. The bot will respond using the `openclaw/assistant-latest` LoRA adapter via Tinker-hosted inference.

## Troubleshooting

- **Docker not running**: Start Docker Desktop or the Docker daemon and re-run `/setup-tinker`.
- **Invalid bot token (401 from Telegram)**: Double-check the `TELEGRAM_BOT_TOKEN` in `docker/.env.tinker`. Generate a new token from @BotFather if needed.
- **Model not supported by Tinker**: The `MODEL` value must match a model ID supported by Tinker (e.g. `Qwen/Qwen3-30B-A3B`). Check available models via Tinker's `get_server_capabilities` endpoint.
- **Plugin ECONNREFUSED on port 8080**: The OpenClaw feedback plugin must reach `http://claas-api:8080` (Docker service DNS), not `localhost`. Re-run the init container: `docker compose -f docker/docker-compose.yml --env-file docker/.env.tinker --profile tinker up init-tinker`.
- **Containers keep restarting**: Check logs with `docker compose -f docker/docker-compose.yml --env-file docker/.env.tinker --profile tinker logs -f <service>`.
