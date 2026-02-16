# CLaaS Plugins

OpenClaw plugins that connect chat frontends (Telegram, etc.) to the CLaaS training loop.

## claas-feedback

Adds a `/feedback` command to OpenClaw. When a user sends `/feedback <text>`, the plugin:

1. Looks up the cached conversation context for that user.
2. Fetches the raw prompt/completion from the inference proxy.
3. Buffers feedback items and, once a batch is full, submits them to the CLaaS API for a distillation step.

### Configuration

Set via `openclaw.plugin.json` config or environment variables:

| Option | Env var | Default | Description |
|--------|---------|---------|-------------|
| `claasApiUrl` | `CLAAS_API_URL` | `http://claas-api:8080` | CLaaS API base URL |
| `proxyUrl` | `CLAAS_TINKER_PROXY_URL` / `CLAAS_VLLM_BASE_URL` | `http://tinker-proxy:8000` | Inference proxy URL |
| `loraId` | — | `openclaw/assistant-latest` | LoRA adapter ID to update |
| `debug` | `CLAAS_FEEDBACK_DEBUG` | `false` | Enable debug logging |
| `feedbackBatchSize` | — | `4` | Number of feedback items to buffer before submitting a batch |
