"""CLaaS API: FastAPI web endpoint for SDPO continual distillation.

This module provides the REST API for the distillation service.

Endpoints:
- POST /v1/distill: Run a single SDPO distillation step
- POST /v1/lora/init: Initialize a new LoRA adapter
- GET /v1/lora: List all LoRA adapters
- GET /v1/health: Health check

Example usage:
    curl -X POST https://your-modal-app.modal.run/v1/distill \\
        -H "Content-Type: application/json" \\
        -d '{
            "lora_id": "user123/coder-v1",
            "prompt": "Write a function to calculate factorial",
            "response": "def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n-1)",
            "feedback": "Good recursive solution",
            "training": {
                "learning_rate": 1e-4,
                "alpha": 0.5
            }
        }'
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import modal
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .storage import (
    LORA_MOUNT_PATH,
    create_initial_lora,
    export_lora_zip_bytes,
    list_loras,
    lora_exists,
    lora_volume,
)
from .types import TrainingConfig

# Modal app for API surface; worker/teacher are resolved by name at runtime.
app = modal.App("claas-distill")

# FastAPI app
web_app = FastAPI(
    title="CLaaS API",
    description="Continual Learning as a Service - SDPO-style distillation",
    version="0.1.0",
)

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
FEEDBACK_LOG_DIR = os.environ.get("FEEDBACK_LOG_DIR", "./feedback_logs")
FEEDBACK_LOCK_TIMEOUT_S = float(os.environ.get("FEEDBACK_LOCK_TIMEOUT_S", "120"))
DISTILL_EXECUTION_MODE = os.environ.get("CLAAS_DISTILL_EXECUTION_MODE", "modal_rpc").strip().lower()

_feedback_locks: dict[str, asyncio.Lock] = {}
_feedback_locks_guard = asyncio.Lock()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


FEEDBACK_WAKE_ON_FAILURE = _env_flag("FEEDBACK_WAKE_ON_FAILURE", True)


def _format_teacher_prompt(
    user_prompt: str,
    feedback: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Format prompt for teacher scoring without importing training deps."""
    if system_prompt is None:
        system_prompt = (
            "You are an expert coding assistant. Provide high-quality, "
            "correct, and well-explained code solutions."
        )

    parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]
    if feedback:
        parts.append(
            f"<|im_start|>user\n{user_prompt}\n\n"
            f"[Feedback on previous attempt: {feedback}]<|im_end|>"
        )
    else:
        parts.append(f"<|im_start|>user\n{user_prompt}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


# Request/Response Models


class DistillRequest(BaseModel):
    """Request for a distillation step."""

    lora_id: str = Field(
        ...,
        description="LoRA identifier (e.g., 'user123/coder-v1')",
    )
    prompt: str = Field(
        ...,
        min_length=1,
        description="User prompt that generated the response",
    )
    response: str = Field(
        ...,
        min_length=1,
        description="Student's response to learn from",
    )
    feedback: str = Field(
        ...,
        min_length=1,
        description="Feedback about response quality",
    )
    rollout_logprobs: list[float] | None = Field(
        default=None,
        description="Log-probabilities from the inference server that generated the rollout. "
        "Required for proper off-policy IS correction. If not provided, logprobs are computed "
        "from the current model (which is incorrect for off-policy learning).",
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training configuration",
    )


class DistillResponse(BaseModel):
    """Response from a distillation step."""

    lora_id: str = Field(
        ...,
        description="Updated LoRA identifier",
    )
    metadata: dict[str, Any] = Field(
        ...,
        description="Training metrics and diagnostics",
    )


class FeedbackOrchestration(BaseModel):
    """Runtime orchestration options for feedback updates."""

    sleep_before: bool = True
    wake_after: bool = True
    wake_on_failure: bool = True
    sleep_level: int = Field(default=1, ge=1, le=2)


class FeedbackRequest(BaseModel):
    """Request for a feedback-triggered in-place LoRA update."""

    lora_id: str = Field(
        ...,
        description="Fixed LoRA identifier to update in place",
    )
    prompt: str = Field(..., min_length=1)
    response: str = Field(..., min_length=1)
    feedback: str = Field(..., min_length=1)
    rollout_logprobs: list[float] | None = Field(default=None)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    orchestration: FeedbackOrchestration = Field(default_factory=FeedbackOrchestration)


class FeedbackResponse(BaseModel):
    """Response from feedback-triggered LoRA update orchestration."""

    status: str
    request_id: str
    lora_id: str
    distill_result: DistillResponse | None = None
    vllm: FeedbackLogVllmState
    feedback_log_path: str
    timing_ms: FeedbackTimingMs


class FeedbackLogRequest(BaseModel):
    """Redacted feedback request fields persisted to logs."""

    lora_id: str
    training: TrainingConfig
    orchestration: FeedbackOrchestration
    prompt_chars: int
    response_chars: int
    feedback_chars: int
    rollout_logprobs_count: int | None = None


class FeedbackLogVllmState(BaseModel):
    """vLLM orchestration state persisted to logs."""

    slept: bool
    woke: bool


class FeedbackTimingMs(BaseModel):
    """Timing breakdown for feedback orchestration."""

    sleep: int = 0
    distill: int = 0
    save: int = 0
    wake: int = 0
    total: int = 0


class FeedbackLogRecord(BaseModel):
    """Structured log record for feedback orchestration."""

    request_id: str
    timestamp_utc: str
    status: str
    phase: str
    lora_id: str
    teacher_mode: str
    request: FeedbackLogRequest
    vllm: FeedbackLogVllmState
    timing_ms: FeedbackTimingMs
    distill_result: DistillResponse | None = None
    error: str | None = None


class LoraInitRequest(BaseModel):
    """Request to initialize a new LoRA adapter."""

    lora_id: str = Field(
        ...,
        description="LoRA identifier (e.g., 'user123/coder-v1')",
    )
    base_model: str = Field(
        default="Qwen/Qwen3-8B",
        description="Base model the LoRA will be applied to",
    )
    lora_r: int = Field(
        default=16,
        ge=4,
        le=128,
        description="LoRA rank",
    )
    lora_alpha: int = Field(
        default=32,
        ge=8,
        le=256,
        description="LoRA alpha scaling factor",
    )
    target_modules: list[str] | None = Field(
        default=None,
        description="Modules to apply LoRA to (defaults to attention + MLP)",
    )


class LoraInitResponse(BaseModel):
    """Response from LoRA initialization."""

    lora_id: str = Field(
        ...,
        description="LoRA identifier of the initialized adapter",
    )


class LoraListResponse(BaseModel):
    """Response listing all LoRA adapters."""

    loras: list[str] = Field(
        ...,
        description="List of LoRA identifiers",
    )


class ServiceHealth(BaseModel):
    """Health status for a backing service (worker or teacher)."""

    status: str
    error: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    worker: ServiceHealth | None = None
    teacher: ServiceHealth | None = None


async def _get_feedback_lock(lora_id: str) -> asyncio.Lock:
    """Return a per-LoRA lock used to serialize feedback updates."""
    key = lora_id.strip("/")
    async with _feedback_locks_guard:
        if key not in _feedback_locks:
            _feedback_locks[key] = asyncio.Lock()
        return _feedback_locks[key]


async def _vllm_post(path: str, *, params: dict[str, Any] | None = None, timeout_s: float = 30.0) -> None:
    """Call a vLLM control endpoint and raise on non-success."""
    async with httpx.AsyncClient(base_url=VLLM_BASE_URL, timeout=timeout_s) as client:
        resp = await client.post(path, params=params)
    resp.raise_for_status()


def _write_feedback_log(record: dict[str, Any] | FeedbackLogRecord) -> str:
    """Persist a feedback lifecycle record to disk and return its path."""
    if isinstance(record, FeedbackLogRecord):
        payload = record.model_dump(mode="json")
        request_id = record.request_id
    else:
        payload = record
        request_id = str(payload.get("request_id", ""))

    log_root = Path(FEEDBACK_LOG_DIR)
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    request_id = request_id or uuid.uuid4().hex
    path = log_root / f"{timestamp}-{request_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return str(path)


def _redact_feedback_request(request: FeedbackRequest) -> FeedbackLogRequest:
    """Return metadata-only request fields for feedback logs."""
    return FeedbackLogRequest(
        lora_id=request.lora_id,
        training=request.training,
        orchestration=request.orchestration,
        prompt_chars=len(request.prompt),
        response_chars=len(request.response),
        feedback_chars=len(request.feedback),
        rollout_logprobs_count=(
            None if request.rollout_logprobs is None else len(request.rollout_logprobs)
        ),
    )


def _safe_export_name(lora_id: str) -> str:
    """Sanitize LoRA identifier for Content-Disposition filename."""
    base = lora_id.strip("/").replace("/", "__")
    safe = "".join(c for c in base if c.isalnum() or c in "-_.")
    return safe or "lora_export"


async def _run_distill(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute a distill request via configured execution backend."""
    if DISTILL_EXECUTION_MODE == "local":
        from .worker import DistillWorker

        worker = DistillWorker()
        try:
            result = await asyncio.to_thread(worker.distill.local, payload)
            return DistillResponse.model_validate(result).model_dump()
        finally:
            # Release worker refs so vLLM can reclaim VRAM on wake.
            del worker
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    distill_fn = modal.Function.from_name("claas-distill", "DistillWorker.distill")
    result = await distill_fn.remote.aio(payload)
    return DistillResponse.model_validate(result).model_dump()


# API Endpoints


@web_app.post("/v1/distill", response_model=DistillResponse)
async def distill(request: DistillRequest) -> DistillResponse:
    """Run a single SDPO distillation step.

    This endpoint:
    1. Loads the user's LoRA from Modal Volume
    2. Runs the student model forward pass
    3. Gets teacher logprobs from configured source
       - self (default): detached student logits
       - remote: vLLM TeacherService
    4. Computes SDPO loss (JSD-based policy gradient)
    5. Updates LoRA parameters
    6. Saves the updated LoRA back to Modal Volume

    Returns the new LoRA ID and training metrics.
    """
    try:
        # Validate LoRA exists (run sync function in thread pool to avoid blocking)
        exists = await asyncio.to_thread(lora_exists, request.lora_id)
        if not exists:
            raise HTTPException(
                status_code=404,
                detail=f"LoRA not found: {request.lora_id}",
            )

        payload = request.model_dump()

        # Remote teacher is optional; self-distillation is the default path.
        if request.training.teacher_mode == "remote":
            teacher_score_fn = modal.Function.from_name("claas-distill", "TeacherService.score_tokens")
            teacher_prompt = _format_teacher_prompt(request.prompt, request.feedback)
            teacher_scored = await teacher_score_fn.remote.aio(
                prompts=[teacher_prompt],
                completions=[request.response],
                top_k=request.training.teacher_top_k,
            )
            if not teacher_scored or not teacher_scored[0]:
                raise HTTPException(
                    status_code=502,
                    detail="Remote teacher returned empty scores",
                )
            payload["teacher_result"] = teacher_scored[0]

        result = await _run_distill(payload)

        return DistillResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Distillation failed: {str(e)}",
        ) from e


@web_app.post("/v1/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Run feedback orchestration: sleep vLLM, distill in-place, wake vLLM."""
    request_id = uuid.uuid4().hex
    lock_acquired = False
    slept = False
    woke = False
    phase = "validate"
    distill_result: DistillResponse | None = None
    error_message: str | None = None
    log_path = ""
    timing_ms = FeedbackTimingMs()
    started_total = time.perf_counter()

    lock = await _get_feedback_lock(request.lora_id)
    try:
        # Validate LoRA exists before attempting orchestration.
        exists = await asyncio.to_thread(lora_exists, request.lora_id)
        if not exists:
            raise HTTPException(status_code=404, detail=f"LoRA not found: {request.lora_id}")

        await asyncio.wait_for(lock.acquire(), timeout=FEEDBACK_LOCK_TIMEOUT_S)
        lock_acquired = True

        if request.orchestration.sleep_before:
            phase = "sleep"
            sleep_start = time.perf_counter()
            await _vllm_post(
                "/sleep",
                params={"level": request.orchestration.sleep_level},
            )
            timing_ms.sleep = int((time.perf_counter() - sleep_start) * 1000)
            slept = True

        phase = "distill"
        distill_start = time.perf_counter()
        payload = request.model_dump()
        payload["save_in_place"] = True

        if request.training.teacher_mode == "remote":
            teacher_score_fn = modal.Function.from_name("claas-distill", "TeacherService.score_tokens")
            teacher_prompt = _format_teacher_prompt(request.prompt, request.feedback)
            teacher_scored = await teacher_score_fn.remote.aio(
                prompts=[teacher_prompt],
                completions=[request.response],
                top_k=request.training.teacher_top_k,
            )
            if not teacher_scored or not teacher_scored[0]:
                raise HTTPException(
                    status_code=502,
                    detail="Remote teacher returned empty scores",
                )
            payload["teacher_result"] = teacher_scored[0]

        distill_result = DistillResponse.model_validate(await _run_distill(payload))
        timing_ms.distill = int((time.perf_counter() - distill_start) * 1000)

        if request.orchestration.wake_after:
            phase = "wake"
            wake_start = time.perf_counter()
            await _vllm_post("/wake_up")
            timing_ms.wake = int((time.perf_counter() - wake_start) * 1000)
            woke = True

        timing_ms.total = int((time.perf_counter() - started_total) * 1000)
    except asyncio.TimeoutError:
        phase = "lock"
        error_message = f"Timed out waiting for lock on LoRA '{request.lora_id}'"
        raise HTTPException(status_code=409, detail=error_message) from None
    except HTTPException as e:
        error_message = str(e.detail)
        raise
    except Exception as e:
        error_message = str(e)
        if (
            slept
            and request.orchestration.wake_after
            and (request.orchestration.wake_on_failure or FEEDBACK_WAKE_ON_FAILURE)
        ):
            try:
                await _vllm_post("/wake_up")
                woke = True
            except httpx.HTTPError:
                pass
        raise HTTPException(
            status_code=500,
            detail=f"Feedback update failed in phase '{phase}': {error_message}",
        ) from e
    finally:
        timing_ms.total = int((time.perf_counter() - started_total) * 1000)
        if lock_acquired:
            lock.release()

        log_record = FeedbackLogRecord(
            request_id=request_id,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            status="ok" if error_message is None else "error",
            phase=phase,
            lora_id=request.lora_id,
            teacher_mode=request.training.teacher_mode,
            request=_redact_feedback_request(request),
            vllm=FeedbackLogVllmState(slept=slept, woke=woke),
            timing_ms=timing_ms,
            distill_result=distill_result,
            error=error_message,
        )
        try:
            log_path = await asyncio.to_thread(
                _write_feedback_log,
                log_record.model_dump(mode="json"),
            )
        except (OSError, TypeError, ValueError):
            # Logging failures should not hide the training result.
            log_path = ""

    return FeedbackResponse(
        status="ok",
        request_id=request_id,
        lora_id=distill_result.lora_id if distill_result else request.lora_id,
        distill_result=distill_result,
        vllm=FeedbackLogVllmState(slept=slept, woke=woke),
        feedback_log_path=log_path,
        timing_ms=timing_ms,
    )


@web_app.post("/v1/lora/init", response_model=LoraInitResponse)
async def init_lora(request: LoraInitRequest) -> LoraInitResponse:
    """Initialize a new LoRA adapter.

    Creates a new LoRA adapter configuration in the Modal Volume.
    The adapter will have zero weights initially and will be trained
    through distill calls.
    """
    try:
        # Run sync function in thread pool to avoid blocking
        lora_id = await asyncio.to_thread(
            create_initial_lora,
            lora_id=request.lora_id,
            base_model_name=request.base_model,
            lora_r=request.lora_r,
            lora_alpha=request.lora_alpha,
            target_modules=request.target_modules,
        )
        return LoraInitResponse(lora_id=lora_id)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LoRA initialization failed: {str(e)}",
        ) from e


@web_app.get("/v1/lora", response_model=LoraListResponse)
async def list_lora_adapters(prefix: str = "") -> LoraListResponse:
    """List all LoRA adapters.

    Args:
        prefix: Optional prefix to filter by (e.g., 'user123/')
    """
    try:
        # Run sync function in thread pool to avoid blocking
        loras = await asyncio.to_thread(list_loras, prefix)
        return LoraListResponse(loras=loras)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list LoRAs: {str(e)}",
        ) from e


@web_app.get("/v1/lora/export")
async def export_lora_adapter(lora_id: str) -> Response:
    """Export a LoRA adapter as a zip archive for local inference servers."""
    try:
        exists = await asyncio.to_thread(lora_exists, lora_id)
        if not exists:
            raise HTTPException(
                status_code=404,
                detail=f"LoRA not found: {lora_id}",
            )

        zip_bytes = await asyncio.to_thread(export_lora_zip_bytes, lora_id)
        safe_name = _safe_export_name(lora_id)
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{safe_name}.zip"',
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LoRA export failed: {str(e)}",
        ) from e


@web_app.get("/v1/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check health of the API and backing services."""
    status = "healthy"
    worker: ServiceHealth | None = None
    teacher: ServiceHealth | None = None

    try:
        worker_health_fn = modal.Function.from_name("claas-distill", "DistillWorker.health_check")
        data = await asyncio.wait_for(worker_health_fn.remote.aio(), timeout=15)
        worker = ServiceHealth.model_validate(data)
    except Exception as e:
        worker = ServiceHealth(status="unhealthy", error=str(e))
        status = "degraded"

    try:
        teacher_health_fn = modal.Function.from_name("claas-distill", "TeacherService.health_check")
        data = await asyncio.wait_for(teacher_health_fn.remote.aio(), timeout=15)
        teacher = ServiceHealth.model_validate(data)
    except Exception as e:
        teacher = ServiceHealth(status="unhealthy", error=str(e))
        status = "degraded"

    return HealthResponse(status=status, worker=worker, teacher=teacher)


@web_app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "CLaaS API",
        "version": "0.1.0",
        "description": "Continual Learning as a Service - SDPO-style distillation",
        "docs": "/docs",
    }


# Mount FastAPI to Modal
@app.function(
    image=modal.Image.debian_slim(python_version="3.11").pip_install(
        "modal>=1.0.0",
        "fastapi>=0.110.0",
        "pydantic>=2.6.0",
        "httpx>=0.27.0",
    ),
    volumes={LORA_MOUNT_PATH: lora_volume},
)
@modal.asgi_app()
def fastapi_app():
    """Modal ASGI app entry point."""
    return web_app
