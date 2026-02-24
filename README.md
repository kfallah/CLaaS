<p align="center">
  <img src="assets/logo.png" alt="CLaaS logo" width="200">
</p>
<p align="center">
  🚀 <a href="https://openclaas.com">Project website</a> |
  📚 <a href="https://docs.openclaas.com">Docs</a> |
  💬 <a href="https://discord.gg/eCgjGXAc">Discord</a>
</p>

# CLaaS: Continual Learning as a Service

Continual learning as-a-service (CLaaS) via self-distillation for OpenClaw. Personalize the model weights of your OpenClaw assistant using text feedback without model collapse.

<p align="center">
  <img src="assets/telegram.png" alt="Telegram demo" width="400">
</p>

## Installation

**Prerequisites:** Python 3.11+, `uv`, and [Docker](https://docs.docker.com/get-docker/).

CLaaS supports three execution engines. Pick the one that matches your setup:

| Engine | GPU required | Status |
|--------|-------------|--------|
| **Local** | Yes (>= 24 GB VRAM) | Available |
| **Tinker** (recommended) | No | Available |
| **Modal** | No (remote GPU) | Coming soon |

### Local (GPU)

Requires an NVIDIA GPU with >= 24 GB VRAM (L40S, A100, RTX 4090, RTX 5090, etc.) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Local GPU setup is pinned to CUDA **12.8** (`cu128` / NGC PyTorch `25.01`), so your NVIDIA driver must support CUDA 12.8. The native (non-Docker) setup is tested on an RTX 5090.

**Docker Compose (recommended):**

```bash
cd docker
cp .env.local.example .env
# Edit .env — set TELEGRAM_BOT_TOKEN (from @BotFather)
docker compose --profile local up --build
```

This brings up vLLM with Qwen3-8B, the CLaaS feedback API, and OpenClaw's Telegram gateway. See [`docker/README.md`](docker/README.md) for details.

**Manual install:**

```bash
uv sync --extra local
```

Then start vLLM and the API yourself. See [Quick Start](#quick-start) and [`docker/README.md`](docker/README.md) for the full supervised local stack.

If you use [Claude Code](https://claude.ai/claude-code), `/setup-local <TELEGRAM_BOT_TOKEN>` installs all deps and starts the full local stack automatically.

### Tinker (no GPU)

Uses the Tinker SDK for hosted distillation and inference. Requires a `TINKER_API_KEY`.

**Docker Compose (recommended):**

```bash
cd docker
cp .env.tinker.example .env.tinker
# Edit .env.tinker — set TELEGRAM_BOT_TOKEN and TINKER_API_KEY
docker compose --env-file .env.tinker --profile tinker up --build
```

**Manual install:**

```bash
uv sync --extra tinker
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch
```

If you use [Claude Code](https://claude.ai/claude-code), `/setup-tinker` deploys the Tinker Docker stack. Use `/clear-tinker-storage` to delete all Tinker checkpoints and free storage.

### Modal (remote GPU) — Coming Soon

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
uv run modal deploy -m claas.modal.deploy
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
uv run python -m claas.api --config-name local

# 3. Initialize a LoRA adapter
curl -X POST http://localhost:8080/v1/lora/init \
  -H "Content-Type: application/json" \
  -d '{"lora_id": "user/my-lora"}'

# 4. Send a feedback update
curl -X POST http://localhost:8080/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [{
      "lora_id": "user/my-lora",
      "prompt": "Write a function to calculate factorial",
      "response": "def factorial(n): ...",
      "feedback": "Good recursive solution"
    }]
  }'
```

For the full supervised local stack (vLLM + gateway + Telegram), see [`docker/README.md`](docker/README.md).

## Hybrid engine

The locally hosted request path is driven by a hybrid engine that switches between:

- **Serving mode**: route request traffic through vLLM (local or remote) for low-latency generation.
- **Update mode**: run a single self-distillation LoRA step using the provided feedback to adapt the adapter.

In practice, the flow is: request is answered by vLLM, then the engine performs (or schedules) the training step, and subsequent requests can use the updated adapter. The engine can prefer local or remote teacher inference depending on `teacher_mode`.

![Hybrid engine diagram](assets/image.png)

## Eval Harness

The eval harness runs automated feedback loops against a live CLaaS stack and measures whether training shifts the model toward preferred behaviours without collapsing. Configuration uses [Hydra](https://hydra.cc/) with YAML configs.

```bash
# Install tinker + dev dependencies
uv sync --extra tinker --extra dev

# Run conciseness eval for 20 steps (Tinker mode, no GPU)
CLAAS_TINKER_API_KEY="tml-..." \
  uv run python -m claas.eval 'preferences=[concise]' num_steps=20
```

Override any config field via Hydra's `key=value` syntax. The default config is in [`claas/eval/configs/base.yaml`](claas/eval/configs/base.yaml). See [`claas/eval/README.md`](claas/eval/README.md) for full documentation including metrics, config reference, setup steps, and known gotchas.

**Important**: When using Tinker mode, `base_model` must use Tinker's model name (e.g. `Qwen/Qwen3-30B-A3B`), not the HuggingFace name (`Qwen/Qwen3-Coder-30B-A3B-Instruct`). Tinker's sampling API accepts either, but the LoRA training API only accepts the Tinker name.

## Dashboard

The CLaaS API serves a built-in dashboard at `/v1/dashboard` showing recent feedback batches, training metrics, and timing breakdowns. Each row is a batch — expand it to see individual samples and detailed metrics.

<p align="center">
  <img src="assets/dashboard.png" alt="CLaaS Dashboard" width="600">
</p>

The eval dashboard at `/v1/eval` displays results from running the eval harness against your model.

<p align="center">
  <img src="assets/eval.png" alt="Eval Dashboard" width="600">
</p>

## References

1. Kleine Buening et al. (2026). "Aligning Language Models from User Interactions." [Paper](https://self-distillation.github.io/user_interactions.pdf) · [Code](https://github.com/lasgroup/user_interactions) · [Hindsight template](https://github.com/lasgroup/user_interactions/blob/main/online_sdpo_trainer.py)
2. Hübotter et al. (2026). "Reinforcement Learning via Self-Distillation." arXiv:2601.20802
3. SDPO Reference Implementation: https://github.com/lasgroup/SDPO
4. Modal GPU Memory Snapshots: https://modal.com/blog/gpu-mem-snapshots
5. vLLM: https://github.com/vllm-project/vllm
6. PEFT/LoRA: https://github.com/huggingface/peft
7. Tinker SDPO training reference (continualcode): https://github.com/sdan/continualcode

## License

MIT
