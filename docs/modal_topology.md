# Modal-First Topology

This project is intentionally Modal-native.

## Default experiment mode

- Local process: `modal serve claas.api` (FastAPI development surface).
- Remote Modal worker: `DistillWorker` for student-side distillation/training.
- Remote Modal worker: `TeacherService` for teacher logprob scoring.

Even in local API development mode, distillation compute does **not** run locally;
`DistillWorker().distill.remote(...)` executes on Modal.

## Why this mode

- Keeps teacher and student compute close to the model volumes and GPUs.
- Preserves Modal RPC integration (no extra HTTP transport layer to maintain).
- Supports quick local endpoint iteration while exercising real remote training flow.

## Commands

```bash
# start local API shell with Modal-backed workers
modal serve claas.api

# deploy API and workers remotely
modal deploy claas.api

# check lint/type/tests
uv sync --extra dev
uv run ruff check .
uv run ty check
uv run pytest -q
```
