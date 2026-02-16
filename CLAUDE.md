# CLaaS - Claude Code Guidelines

## Code Quality Rules

### After Every Change

Run lint and type checking after every code change:

```bash
uv sync --extra dev

# Lint and auto-fix
uv run ruff check claas/ tests/ --fix

# Type check (import errors for modal/torch/vllm are expected)
uv run ty check
```

Note: GPU dependencies (modal, torch, vllm, transformers, peft) are not installed locally. `ty check` will report `unresolved-import` errors for these - this is expected and can be ignored.

### Run Tests

```bash
uv run pytest tests/ -v -m "not integration"
```

## Project Structure

```text
claas/
├── __init__.py
├── api.py                               # FastAPI endpoints (entrypoint)
├── cli.py                               # Command-line interface (entrypoint)
├── deploy.py                            # Modal deployment entrypoint
├── index.html                           # Dashboard template
│
├── core/                                # Shared types & config
│   ├── __init__.py
│   ├── types.py                         # Pydantic models, TypedDicts (ChatMessage, etc.)
│   └── config.py                        # Configuration dataclasses
│
├── training/                            # Training pipeline
│   ├── __init__.py
│   ├── sdpo_loss.py                     # SDPO loss computation (core algorithm)
│   ├── worker.py                        # Distill worker (local/Modal)
│   ├── storage.py                       # LoRA storage (Modal Volume or local fs)
│   ├── teacher_helpers.py               # Pure teacher prompt functions
│   ├── teacher_service.py               # Modal TeacherService class
│   └── engine/                          # Pluggable training backends
│       ├── __init__.py                  # get_training_engine() factory
│       ├── base.py                      # TrainingEngine abstract interface
│       ├── local/engine.py              # Local GPU execution
│       ├── modal/engine.py              # Modal remote execution
│       └── tinker/engine.py, state.py   # Tinker SDK execution
│
└── proxy/                               # Inference proxy
    ├── __init__.py
    └── tinker_inference_proxy.py         # Tinker SDK -> OpenAI-compatible proxy
```

## Modal Deployment

Deploy to Modal:

```bash
modal deploy -m claas.deploy
```

Run locally for development:

```bash
modal serve -m claas.deploy
```

## Key Patterns

### Storage

LoRA storage is engine-dependent: local filesystem (`CLAAS_LORA_ROOT`), Modal Volume, or Tinker JSON state. The core functions in `training/storage.py` handle local and Modal paths:

```python
from claas.training.storage import load_lora, save_lora, create_initial_lora

# Initialize new LoRA
lora_id = create_initial_lora("user/model", base_model_name="...")

# Load/save
local_path = load_lora("user/model")
new_id = save_lora(local_path, "user/model")
```

### SDPO Loss

The core algorithm uses JSD-based policy gradient:

```python
from claas.training.sdpo_loss import compute_sdpo_loss

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

Heavy dependencies (torch, vllm, transformers, tinker) are not installed locally. They run inside Docker containers, Modal containers, or the Tinker cloud. `ty check` will report `unresolved-import` errors for these — this is expected.

## Architecture Rules

### DO NOT manage vLLM as a subprocess from the CLaaS API

Never add code to kill, restart, or spawn vLLM from within the API process. vLLM is managed externally (by the user, systemd, Docker, etc.). The API communicates with vLLM only via its HTTP API (sleep/wake, load/unload LoRA). Adding process management (pkill, subprocess.Popen, etc.) to the API is fragile, creates tight coupling, and is not how this system is designed.

## Development Workflow

All features are developed on branches and merged via GitHub PRs. Every PR must pass CI before merging.

CI has two jobs (`lint-and-test` runs on every PR, `integration` runs on manual dispatch):

```bash
# Check PR status
gh pr checks <pr-number>

# Trigger the full CI suite (including integration tests)
gh workflow run ci.yml --ref <branch-name>

# Watch a run
gh run watch <run-id> --exit-status
```

Before opening or merging a PR, verify locally:

```bash
uv run ruff check claas/ tests/ --fix
uv run ty check
uv run pytest tests/ -q -m "not integration"
```

Gate merges on all CI checks passing. Run the integration test (`workflow_dispatch`) before merging any change that touches the training engine, proxy, or API feedback flow.

## Ruff Rules

Using default ruff rules plus:
- I: isort (import sorting)
- E/W: pycodestyle
- F: Pyflakes
