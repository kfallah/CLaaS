# CLaaS Training Package

Self-Distillation Policy Optimization (SDPO) training pipeline with pluggable execution backends.

A frozen copy of the base model acts as the **teacher**, receiving the original prompt plus natural-language feedback about a prior attempt. The student (a LoRA adapter on the same base model) learns to match the teacher's corrected distribution via a JSD-based policy gradient, with KL regularization to prevent drift from the base policy.

Reference: Huebotter et al. (2026), *Reinforcement Learning via Self-Distillation* ([arXiv:2601.20802](https://arxiv.org/abs/2601.20802)).

## Package layout

```
training/
├── distillation.py          # DistillationTrainer — shared SDPO update logic (Local & Modal)
├── sdpo_loss.py             # compute_sdpo_loss() — GJS divergence + IS correction + KL reg
├── storage.py               # LoRA lifecycle: create, load, save, alias, export
├── teacher_helpers.py       # Pure prompt construction for teacher signals
└── engine/
    ├── base.py               # TrainingEngine abstract interface
    ├── local/engine.py       # On-box GPU execution
    ├── modal/engine.py       # Modal cloud RPC
    └── tinker/engine.py      # Tinker SDK (cloud-native, no local GPU)
```

## Core algorithm

`compute_sdpo_loss` combines three components:

1. **Generalized Jensen-Shannon Divergence** over the teacher's top-K logprobs. The `alpha` parameter interpolates between symmetric GJS (0.5) and pure reverse KL (1.0).
2. **Importance-sampling correction** for off-policy updates, with a configurable clip threshold.
3. **KL regularization** to the base policy (Schulman k3 estimator) to limit adapter drift.

All losses are masked to completion tokens only — prompt tokens contribute zero gradient.

## Training engines

All engines implement the `TrainingEngine` ABC (async methods). Select via the factory:

```python
from claas.training.engine import get_training_engine
engine = get_training_engine("local", cfg)   # or "modal", "tinker"
```

### Local

Runs on the host GPU. The `DistillationTrainer` loads the base model onto CUDA, runs the full forward-backward-optimize cycle, then offloads. LoRA adapters are stored on the local filesystem and can be hot-loaded into a local vLLM instance via `lora_runtime_ref()`.

Best for: development and single-GPU setups.

### Modal

Same `DistillationTrainer` logic, but the distill call is an RPC to a Modal serverless GPU. LoRA storage uses a Modal networked volume. Init and storage operations run locally against the shared volume.

Best for: scaling to larger models or bursting GPU capacity without managing hardware.

### Tinker

Cloud-native — no `DistillationTrainer` or local GPU. The entire training loop (forward, backward, optimizer step, checkpointing) runs via Tinker SDK calls. Key differences from the other engines:

- **Adaptive KL scaling**: dynamically adjusts the KL coefficient each step to keep advantage magnitude stable, rather than using a fixed `kl_reg_weight`.
- **No `storage.py`**: checkpoints live in the Tinker cloud. A local JSON state file (`~/.claas/tinker_state.json`) maps `lora_id` to Tinker checkpoint paths.
- **No vLLM reload**: `lora_runtime_ref()` raises — inference goes through the Tinker sampling API instead.

Best for: production distillation on large models (e.g., Qwen3-30B-A3B) without managing GPUs.

## Environment variables

| Variable | Engine | Purpose |
|---|---|---|
| `CLAAS_LORA_ROOT` | Local, Modal | Root directory for LoRA storage |
| `CLAAS_TINKER_API_KEY` | Tinker | Tinker SDK credentials |
| `CLAAS_TINKER_STATE_PATH` | Tinker | JSON state file path (default `~/.claas/tinker_state.json`) |
| `HF_HOME` | Local, Modal | HuggingFace model cache |
