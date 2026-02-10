# CLaaS: Continual Learning as a Service

SDPO-style continual distillation API for per-request model adaptation.

## Overview

CLaaS provides a serverless API for continual learning via Self-Distillation Policy Optimization (SDPO). Each API call:

1. Loads a user's LoRA adapter from Modal Volume
2. Runs the student model (with LoRA) on the provided prompt/response
3. Gets dense teacher logprobs from a 30B teacher model via vLLM
4. Computes SDPO loss (Generalized JSD + KL regularization to base policy)
5. Updates the LoRA parameters
6. Saves the updated adapter back to Modal Volume

This enables real-time model personalization where the model learns from each interaction.

## Architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│  Modal App: claas-distill                                        │
│                                                                  │
│  ┌─────────────────────────────┐  ┌────────────────────────────┐│
│  │  DistillWorker (L40S)       │  │  TeacherService (H100)     ││
│  │  GPU memory snapshot        │  │  GPU memory snapshot        ││
│  │                             │  │                            ││
│  │  • Qwen3-Coder-Next         │  │  • vLLM                    ││
│  │  • LoRA training            │◄─►│  • Qwen3-Coder-30B-A3B    ││
│  │  • SDPO loss                │  │  • prompt_logprobs=100     ││
│  │  Cold start: ~2s            │  │  Cold start: ~3-5s         ││
│  └─────────────────────────────┘  └────────────────────────────┘│
│                                                                  │
│  Modal Volume: claas-loras (LoRA storage)                        │
│  FastAPI endpoint: /v1/distill                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -e .
```

## Prerequisites

1. **Modal account**: Sign up at [modal.com](https://modal.com)
2. **Modal CLI**: `pip install modal && modal token new`

That's it! No AWS/S3 credentials needed - LoRAs are stored in Modal Volumes.

## Quick Start

### 1. Deploy the service

```bash
modal deploy claas.api
```

### 2. Initialize a LoRA adapter

```bash
curl -X POST https://your-app--claas-distill-fastapi-app.modal.run/v1/lora/init \
  -H "Content-Type: application/json" \
  -d '{"lora_id": "user123/coder-v1"}'
```

### 3. Call the distillation API

```bash
curl -X POST https://your-app--claas-distill-fastapi-app.modal.run/v1/distill \
  -H "Content-Type: application/json" \
  -d '{
    "lora_id": "user123/coder-v1-init",
    "prompt": "Write a function to calculate factorial",
    "response": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
    "feedback": "Good recursive solution",
    "rollout_logprobs": [-0.5, -1.2, -0.8, -0.3],
    "training": {
      "learning_rate": 1e-4,
      "alpha": 0.5
    }
  }'
```

## API Reference

### POST /v1/distill

Run a single SDPO distillation step.

**Request body:**
```json
{
  "lora_id": "user123/coder-v1",
  "prompt": "User prompt text",
  "response": "Model response to learn from",
  "feedback": "Feedback about response quality (required)",
  "rollout_logprobs": [-0.5, -1.2, -0.8, ...],
  "training": {
    "learning_rate": 1e-4,
    "alpha": 0.5,
    "is_clip": 5.0,
    "max_grad_norm": 1.0,
    "kl_reg_weight": 0.1,
    "teacher_top_k": 100
  }
}
```

**Top-level parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `lora_id` | Yes | LoRA identifier (e.g., "user123/coder-v1") |
| `prompt` | Yes | User prompt that generated the response |
| `response` | Yes | Model response to learn from |
| `feedback` | Yes | Feedback about response quality |
| `rollout_logprobs` | No | Logprobs from inference server for off-policy IS correction |

**Training parameters** (nested under `training`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-4 | AdamW learning rate |
| `alpha` | 0.5 | GJS interpolation (0.5 = symmetric JSD, 1.0 = reverse KL) |
| `is_clip` | 5.0 | Importance sampling ratio clip (exp space) |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `kl_reg_weight` | 0.1 | Weight for KL regularization to base policy |
| `teacher_top_k` | 100 | Top-K logprobs from teacher |

**Response:**
```json
{
  "lora_id": "user123/coder-v1-20250209-123456",
  "metadata": {
    "total_loss": 0.234,
    "distill_loss": 0.156,
    "kl_reg": 0.078,
    "mean_is_ratio": 1.001,
    "clip_fraction": 0.05,
    "grad_norm": 0.89,
    "tokens_processed": 128
  }
}
```

### POST /v1/lora/init

Initialize a new LoRA adapter.

**Request body:**
```json
{
  "lora_id": "user123/coder-v1",
  "base_model": "Qwen/Qwen3-Coder-Next",
  "lora_r": 16,
  "lora_alpha": 32,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}
```

### GET /v1/lora

List all LoRA adapters (with optional `prefix` query parameter).

### GET /v1/health

Health check for all services.

## Storage

LoRA adapters are stored in **Modal Volumes** - no external storage needed:

- Volume name: `claas-loras`
- Path format: `/loras/{user_id}/{lora_name}`
- Automatic versioning with timestamps

Benefits:
- No AWS credentials to manage
- Integrated with Modal infrastructure
- ~100MB/s read/write throughput
- Persists across container restarts

## Off-Policy Learning

For proper off-policy learning (when the response was generated by a different model checkpoint), pass `rollout_logprobs` at the top level with the log-probabilities from the inference server that generated the rollout:

```json
{
  "lora_id": "user123/coder-v1",
  "prompt": "...",
  "response": "...",
  "feedback": "...",
  "rollout_logprobs": [-0.5, -1.2, -0.8, ...]
}
```

If not provided, logprobs are computed from the current model with a warning logged. This is incorrect for off-policy learning but acceptable for on-policy scenarios.

## Configuration

### LoRA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA scaling factor |
| `target_modules` | attention + MLP | Modules to apply LoRA to |

## Development

### Run locally

```bash
modal serve claas.api
```

### Run tests

```bash
pytest tests/ -v
```

## Cold Start Performance

| Worker | Without Snapshots | With GPU Snapshots |
|--------|-------------------|-------------------|
| Student (Qwen3-Coder-Next) | ~15-20s | ~2s |
| Teacher (Qwen3-Coder-30B) | ~45-60s | ~3-5s |

## Cost Estimate

| Calls/day | Monthly cost | Note |
|-----------|--------------|------|
| 50 | ~$15 | Light use, personal |
| 500 | ~$150 | CI pipeline |
| 5000 | ~$1,500 | Production agent loop |

## References

1. Hübotter et al. (2026). "Reinforcement Learning via Self-Distillation." arXiv:2601.20802
2. SDPO Reference Implementation: https://github.com/lasgroup/SDPO
3. Modal GPU Memory Snapshots: https://modal.com/blog/gpu-mem-snapshots
4. vLLM: https://github.com/vllm-project/vllm
5. PEFT/LoRA: https://github.com/huggingface/peft

## License

MIT
