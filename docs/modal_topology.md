# Execution Modes

CLaaS supports three training engine backends, selected via
`CLAAS_DISTILL_EXECUTION_MODE`:

## `local` (default)

- All training runs on the host GPU (Qwen3-8B + LoRA).
- Teacher logprobs come from vLLM (local or remote, controlled by
  `VLLM_BASE_URL`).
- LoRA adapters stored on the local filesystem (`CLAAS_LORA_ROOT`).

```bash
uv run uvicorn claas.api:web_app --host 0.0.0.0 --port 8080
```

## `modal`

- Distillation runs remotely on Modal (L40S for student, H100 for teacher).
- LoRA adapters stored in Modal Volume `claas-loras`.
- GPU memory snapshots enable sub-second cold starts.

```bash
modal deploy claas.api
```

## `tinker`

- Uses the Tinker Python SDK for LoRA lifecycle and SDPO-style distillation.
- No local GPU required; all compute happens on Tinker's hosted infrastructure.
- An OpenAI-compatible inference proxy (`claas.tinker_inference_proxy`) exposes
  `/v1/chat/completions` and `/v1/completions` backed by a Tinker
  `SamplingClient`.
- LoRA state tracked in a local JSON file (`CLAAS_TINKER_STATE_PATH`).

```bash
# API
CLAAS_DISTILL_EXECUTION_MODE=tinker CLAAS_TINKER_API_KEY=... \
    uv run uvicorn claas.api:web_app --host 0.0.0.0 --port 8080

# Inference proxy (separate process)
CLAAS_TINKER_API_KEY=... \
    uv run uvicorn claas.tinker_inference_proxy:app --host 0.0.0.0 --port 8000
```

## Architecture

All three engines implement the same `TrainingEngine` abstract interface
(`claas/training_engines/base.py`), which defines:

- `init_lora` / `list_loras` / `lora_exists` / `export_lora`
- `lora_runtime_ref` (local/modal only)
- `distill` (single SDPO step)
- `health`

The `claas.api` module routes requests to whichever engine is active. Shared
code (teacher prompt building, types) lives in `claas.teacher` and
`claas.types`.
