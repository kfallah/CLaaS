# CLaaS: Continual Learning as a Service

SDPO-style continual distillation API for per-request model adaptation.

## Overview

CLaaS provides a serverless API for continual learning via Self-Distillation Policy Optimization (SDPO). Each API call:

1. Loads a user's LoRA adapter from S3
2. Runs the student model on the provided prompt/response
3. Gets dense teacher logprobs from a 32B teacher model
4. Computes SDPO loss (JSD-based policy gradient with per-token advantages)
5. Updates the LoRA parameters
6. Saves the updated adapter back to S3

This enables real-time model personalization where the model learns from each interaction.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Modal App: claas-distill                                        │
│                                                                  │
│  ┌─────────────────────────────┐  ┌────────────────────────────┐│
│  │  DistillWorker (L40S)       │  │  TeacherService            ││
│  │  GPU memory snapshot        │  │  (A100-80GB)               ││
│  │                             │  │  GPU memory snapshot        ││
│  │  • Qwen2.5-Coder-3B        │  │  • vLLM(Qwen2.5-32B)       ││
│  │  • LoRA training           │◄─►│  • prompt_logprobs=100     ││
│  │  • SDPO loss               │  │                            ││
│  │  Cold start: ~2s           │  │  Cold start: ~3-5s         ││
│  └─────────────────────────────┘  └────────────────────────────┘│
│                                                                  │
│  FastAPI endpoint: /v1/distill                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -e .
```

## Prerequisites

1. **Modal account**: Sign up at [modal.com](https://modal.com)
2. **AWS credentials**: For S3 LoRA storage
3. **Modal secrets**: Configure AWS credentials in Modal

```bash
modal secret create aws-credentials \
  AWS_ACCESS_KEY_ID=<your-key> \
  AWS_SECRET_ACCESS_KEY=<your-secret> \
  AWS_REGION=us-east-1
```

## Quick Start

### 1. Initialize a LoRA adapter

```bash
claas init-lora --output-uri s3://my-bucket/loras/user123/coder-v1/
```

### 2. Deploy the service

```bash
claas deploy
```

### 3. Call the API

```bash
curl -X POST https://your-app--claas-distill-fastapi-app.modal.run/v1/distill \
  -H "Content-Type: application/json" \
  -d '{
    "lora_uri": "s3://my-bucket/loras/user123/coder-v1/",
    "prompt": "Write a function to calculate factorial",
    "response": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
    "feedback": "Good recursive solution",
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
  "lora_uri": "s3://bucket/loras/user/model/",
  "prompt": "User prompt text",
  "response": "Model response to learn from",
  "feedback": "Optional feedback about quality",
  "training": {
    "learning_rate": 1e-4,
    "alpha": 0.5,
    "clip_eps": 0.2,
    "max_grad_norm": 1.0,
    "jsd_reg_weight": 0.5,
    "teacher_top_k": 100
  }
}
```

**Response:**
```json
{
  "lora_uri": "s3://bucket/loras/user/model-20250209-123456/",
  "metadata": {
    "total_loss": 0.234,
    "pg_loss": 0.156,
    "jsd_reg": 0.078,
    "mean_advantage": 0.045,
    "frac_positive_advantage": 0.62,
    "mean_is_ratio": 1.001,
    "clip_fraction": 0.05,
    "grad_norm": 0.89,
    "tokens_processed": 128
  }
}
```

### POST /v1/distill/lite

Lightweight distillation with pre-computed teacher logprobs (for use with Fireworks API).

### POST /v1/lora/init

Initialize a new LoRA adapter.

### GET /v1/health

Health check for all services.

## Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-4 | AdamW learning rate |
| `alpha` | 0.5 | JSD interpolation (0.5 = symmetric JSD) |
| `clip_eps` | 0.2 | PPO clip range |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `jsd_reg_weight` | 0.5 | Weight for logit-level JSD regularizer |
| `teacher_top_k` | 100 | Top-K logprobs from teacher |

### LoRA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA scaling factor |

## Development

### Run locally

```bash
claas serve
```

### Run tests

```bash
pytest tests/ -v
```

### Check health

```bash
claas health
```

## Cost Estimate

| Calls/day | Monthly cost | Note |
|-----------|--------------|------|
| 50 | ~$15 | Light use, personal |
| 500 | ~$150 | CI pipeline |
| 5000 | ~$1,500 | Production agent loop |

## References

1. Hübotter et al. (2026). "Reinforcement Learning via Self-Distillation." arXiv:2601.20802
2. Modal GPU Memory Snapshots: https://modal.com/blog/gpu-mem-snapshots
3. vLLM: https://github.com/vllm-project/vllm
4. PEFT/LoRA: https://github.com/huggingface/peft

## License

MIT
