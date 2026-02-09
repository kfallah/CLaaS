# CLaaS - Claude Code Guidelines

## Code Quality Rules

### After Every Change

Run lint and type checking after every code change:

```bash
# Lint and auto-fix
uvx ruff check claas/ tests/ --fix

# Type check (import errors for modal/torch/vllm are expected)
uvx ty check
```

Note: GPU dependencies (modal, torch, vllm, transformers, peft) are not installed locally. `ty check` will report `unresolved-import` errors for these - this is expected and can be ignored.

### Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
claas/
├── api.py           # FastAPI endpoints
├── cli.py           # Command-line interface
├── config.py        # Configuration dataclasses
├── sdpo_loss.py     # SDPO loss computation (core algorithm)
├── storage.py       # Modal Volume storage for LoRAs
├── teacher.py       # vLLM teacher service (Modal)
└── worker.py        # Training worker (Modal)
```

## Modal Deployment

Deploy to Modal:

```bash
modal deploy claas.api
```

Run locally for development:

```bash
modal serve claas.api
```

## Key Patterns

### Storage

LoRAs are stored in Modal Volumes at `/loras/{user}/{model}`:

```python
from claas.storage import load_lora, save_lora, create_initial_lora

# Initialize new LoRA
lora_id = create_initial_lora("user/model", base_model_name="...")

# Load/save
local_path = load_lora("user/model")
new_id = save_lora(local_path, "user/model")
```

### SDPO Loss

The core algorithm uses JSD-based policy gradient:

```python
from claas.sdpo_loss import compute_sdpo_loss

loss_dict = compute_sdpo_loss(
    student_logits=...,
    teacher_logprobs=...,
    teacher_indices=...,
    response_mask=...,
    old_student_logprobs=...,
    response_ids=...,
    alpha=0.5,  # JSD interpolation
)
```

## Dependencies

Heavy dependencies (torch, vllm, transformers) are only available inside Modal containers. Type checking will show "missing import" errors for these - this is expected.

## Ruff Rules

Using default ruff rules plus:
- I: isort (import sorting)
- E/W: pycodestyle
- F: Pyflakes
