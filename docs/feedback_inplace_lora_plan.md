# Feedback API In-Place LoRA Update Plan

## Goal

Use a single served LoRA ID that is updated in place by a dedicated feedback endpoint.
Each feedback request should orchestrate:

1. sleep vLLM
2. run one distillation update
3. write updated LoRA to the same path/ID
4. wake vLLM

This supports local experiment loops where teacher scoring is optional (default self-distillation).

## Topology

- API runtime: local Modal runtime (`modal serve claas.api`)
- Distill worker: Modal GPU worker (L40S)
- Teacher service: optional, remote when `teacher_mode=remote`
- Inference server: local vLLM process with sleep/wake enabled

## API Contract

`POST /v1/feedback`

Required fields:

- `lora_id`
- `prompt`
- `response`
- `feedback`

Optional fields:

- `rollout_logprobs`
- `training` (defaults from `TrainingConfig`, `teacher_mode=self` by default)
- `orchestration`:
  - `sleep_before` (default `true`)
  - `wake_after` (default `true`)
  - `wake_on_failure` (default `true`)
  - `sleep_level` (default `1`)

## Persistence Rules

- `/v1/feedback` writes LoRA in place to fixed `lora_id`.
- `/v1/distill` keeps existing versioned behavior for compatibility.
- Every `/v1/feedback` call writes one JSON log entry to disk.

## Feedback Logging

Environment variables:

- `FEEDBACK_LOG_DIR` (default `./feedback_logs`)
- `FEEDBACK_LOCK_TIMEOUT_S` (default `120`)
- `FEEDBACK_WAKE_ON_FAILURE` (default `true`)

Each log record includes:

- request id and timestamp
- request payload
- phase status (`sleep`, `distill`, `wake`, error phase)
- vLLM sleep/wake flags
- per-phase and total timing
- distill result and error message (if any)

## Concurrency and Failure Handling

- Per-LoRA lock serializes feedback updates.
- Lock timeout returns conflict (`409`).
- If distill/update fails after sleep, API attempts wake-up before returning error.
- Logging is best effort and should not hide core training result.

## Validation

- `ruff check`
- `ty check`
- unit tests for API feedback endpoint and in-place storage writes

