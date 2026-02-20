# scripts/

## `eval/` -- SDPO Evaluation Harness

Measures how SDPO continual learning affects model behavior over multiple
feedback steps. Tracks four metrics:

| Metric | What it measures |
|--------|-----------------|
| **logprob** | Logprob margin between preferred/dispreferred responses (deterministic, no generation) |
| **compliance** | Whether the model follows the trained preference (generates + verifies) |
| **general** | General capability retention -- coding, instruction following |
| **collapse** | Token entropy, self-ROUGE-L diversity, logprob drift from baseline |

### Prerequisites

A running Docker stack (either local or Tinker) with OpenClaw, CLaaS API,
and the inference backend.

### Tinker stack (no GPU)

```bash
MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
  docker compose -f docker/docker-compose.tinker.yml up -d

python -m scripts.eval \
    --openclaw-url http://localhost:18789 \
    --proxy-url http://localhost:8000 \
    --claas-url http://localhost:8080 \
    --base-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --preferences no_emoji \
    --metrics logprob \
    --num-steps 10
```

Notes:
- `--proxy-url` enables Tinker mode (proxy scoring via `/v1/score`)
- Entropy is unavailable in Tinker mode (proxy doesn't support `top_logprobs`)
- Rollout logprobs are fetched from the proxy cache

### Local stack (GPU + vLLM)

```bash
docker compose -f docker/docker-compose.yml --profile local up -d

python -m scripts.eval \
    --openclaw-url http://localhost:18789 \
    --vllm-url http://localhost:8000 \
    --claas-url http://localhost:8080 \
    --base-model Qwen/Qwen3-8B \
    --preferences no_emoji \
    --metrics all \
    --num-steps 20
```

Notes:
- `--vllm-url` is used for logprob scoring and rollout logprob collection
- All metrics including entropy are available
- LoRA adapters are loaded/reloaded into vLLM each step

### Output

Results are written to `--output-dir` (default `./eval_results/`):

```
eval_results/
  summary.json              # Pass/marginal/fail verdicts per preference
  no_emoji/
    metadata.json           # Config and LoRA ID
    baseline.json           # Baseline metrics (step 0)
    steps.jsonl             # Per-step results (append-only, resumable)
  concise/
    ...
```

### Metric presets

- `--metrics logprob` -- Fast, deterministic (default)
- `--metrics all` -- logprob + compliance + general + collapse
- `--metrics logprob,compliance` -- Comma-separated custom selection

---

## `openclaw-local/` -- Manual Local Stack Setup

Scripts for setting up the local stack (vLLM + CLaaS API + OpenClaw)
directly on the host without Docker. Useful when Docker is unavailable.

---

## `run_integration_tests.sh` -- Integration Test Runner

Runs the full integration test suite against a running Docker stack.
Typically invoked via CI (`workflow_dispatch`) or manually before merging
changes that touch the training engine, proxy, or API feedback flow.
