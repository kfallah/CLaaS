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
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import modal
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from .storage import (
    LORA_MOUNT_PATH,
    create_initial_lora,
    export_lora_zip_bytes,
    list_loras,
    lora_exists,
    lora_volume,
    resolve_lora_id,
)
from .teacher import format_teacher_prompt
from .types import (
    DistillRequest,
    DistillResponse,
    FeedbackLogRecord,
    FeedbackLogVllmState,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackTimingMs,
    HealthResponse,
    LoraInitRequest,
    LoraInitResponse,
    LoraListResponse,
    ServiceHealth,
)

logger = logging.getLogger(__name__)

# Modal app for API surface; worker/teacher are resolved by name at runtime.
app = modal.App("claas-distill")

# FastAPI app
web_app = FastAPI(
    title="CLaaS API",
    description="Continual Learning as a Service - SDPO-style distillation",
    version="0.1.0",
)

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "sk-local")
FEEDBACK_LOG_DIR = os.environ.get("FEEDBACK_LOG_DIR", "./feedback_logs")
FEEDBACK_LOCK_TIMEOUT_S = float(os.environ.get("FEEDBACK_LOCK_TIMEOUT_S", "120"))
# Default "local" assumes GPU deps (torch, etc.) are available on this machine.
# Set to "modal_rpc" for Modal deployments where the API image is CPU-only.
DISTILL_EXECUTION_MODE = os.environ.get("CLAAS_DISTILL_EXECUTION_MODE", "local").strip().lower()

_feedback_locks: dict[str, asyncio.Lock] = {}
_feedback_locks_guard = asyncio.Lock()


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


FEEDBACK_WAKE_ON_FAILURE = _env_flag("FEEDBACK_WAKE_ON_FAILURE", True)

# Minimum free GPU memory (GB) required before starting training.
FEEDBACK_MIN_FREE_VRAM_GB = float(os.environ.get("FEEDBACK_MIN_FREE_VRAM_GB", "20"))
# Maximum seconds to wait for GPU memory after vLLM sleep.
FEEDBACK_SLEEP_VERIFY_TIMEOUT_S = float(os.environ.get("FEEDBACK_SLEEP_VERIFY_TIMEOUT_S", "30"))
# Maximum seconds to wait for in-flight vLLM requests to drain before sleeping.
FEEDBACK_DRAIN_TIMEOUT_S = float(os.environ.get("FEEDBACK_DRAIN_TIMEOUT_S", "30"))
# vLLM model name for fetching rollout logprobs.
# None = auto-derive from LoRA ID.  "" = disabled.
VLLM_ROLLOUT_MODEL = os.environ.get("VLLM_ROLLOUT_MODEL")

ALLOWED_INIT_BASE_MODELS = {
    model.strip()
    for model in os.environ.get("CLAAS_ALLOWED_INIT_BASE_MODELS", "Qwen/Qwen3-8B").split(",")
    if model.strip()
}


def _validate_init_base_model(base_model: str) -> None:
    if base_model in ALLOWED_INIT_BASE_MODELS:
        return

    logger.warning("Rejected /v1/lora/init for disallowed base_model: %s", base_model)
    allowed = ", ".join(sorted(ALLOWED_INIT_BASE_MODELS))
    raise HTTPException(
        status_code=403,
        detail=(
            f"base_model '{base_model}' is not allowed for initialization. "
            f"Allowed models: {allowed}"
        ),
    )





async def _get_feedback_lock(lora_id: str) -> asyncio.Lock:
    """Return a per-LoRA lock used to serialize feedback updates."""
    key = lora_id.strip("/")
    async with _feedback_locks_guard:
        if key not in _feedback_locks:
            _feedback_locks[key] = asyncio.Lock()
        return _feedback_locks[key]


async def _vllm_post(
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout_s: float = 30.0,
) -> None:
    """Call a vLLM control endpoint and raise on non-success."""
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"} if VLLM_API_KEY else {}
    async with httpx.AsyncClient(base_url=VLLM_BASE_URL, timeout=timeout_s) as client:
        resp = await client.post(path, params=params, json=json_body, headers=headers)
    resp.raise_for_status()


async def _wait_for_vllm_idle(
    timeout_s: float = FEEDBACK_DRAIN_TIMEOUT_S,
) -> None:
    """Poll vLLM ``/metrics`` until no requests are running or waiting.

    Raises :class:`TimeoutError` if vLLM is still busy after *timeout_s*.
    """
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"} if VLLM_API_KEY else {}
    deadline = time.perf_counter() + timeout_s

    while True:
        async with httpx.AsyncClient(base_url=VLLM_BASE_URL, timeout=10) as client:
            resp = await client.get("/metrics", headers=headers)
        resp.raise_for_status()

        running = 0
        waiting = 0
        for line in resp.text.splitlines():
            if line.startswith("vllm:num_requests_running"):
                running += int(float(line.split()[-1]))
            elif line.startswith("vllm:num_requests_waiting"):
                waiting += int(float(line.split()[-1]))

        if running == 0 and waiting == 0:
            logger.info("vLLM idle (0 running, 0 waiting) — safe to sleep")
            return

        if time.perf_counter() >= deadline:
            raise TimeoutError(
                f"vLLM still busy after {timeout_s}s: "
                f"{running} running, {waiting} waiting"
            )

        logger.info(
            "vLLM busy (%d running, %d waiting) — polling again in 0.5s",
            running,
            waiting,
        )
        await asyncio.sleep(0.5)


def _resolve_vllm_model_name(lora_id: str) -> str | None:
    """Derive the vLLM model name for a LoRA, matching _vllm_reload_lora logic.

    Returns ``None`` when rollout-logprob fetching is explicitly disabled
    (``VLLM_ROLLOUT_MODEL=""``) or when *lora_id* is empty.
    """
    import re

    if VLLM_ROLLOUT_MODEL == "":
        return None
    if VLLM_ROLLOUT_MODEL is not None:
        return VLLM_ROLLOUT_MODEL
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", lora_id.strip("/")).strip("-") or None


async def _fetch_rollout_logprobs(
    prompt: str,
    response: str,
    model: str,
    timeout_s: float = 60.0,
) -> list[float]:
    """Fetch per-token logprobs for *response* from the vLLM completions API.

    1. Tokenize the prompt to learn its token count.
    2. Submit ``prompt + response`` with ``max_tokens=0`` and ``prompt_logprobs=1``.
    3. Strip the prompt portion and extract the log-probability for each
       response token.
    """
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"} if VLLM_API_KEY else {}
    async with httpx.AsyncClient(base_url=VLLM_BASE_URL, timeout=timeout_s) as client:
        tok_resp = await client.post(
            "/tokenize",
            json={"model": model, "prompt": prompt},
            headers=headers,
        )
        tok_resp.raise_for_status()
        prompt_token_count = tok_resp.json()["count"]

        comp_resp = await client.post(
            "/v1/completions",
            json={
                "model": model,
                "prompt": prompt + response,
                "max_tokens": 0,
                "prompt_logprobs": 1,
            },
            headers=headers,
        )
        comp_resp.raise_for_status()

    raw_logprobs = comp_resp.json()["choices"][0]["prompt_logprobs"]

    # Skip prompt tokens and null entries; extract logprob values.
    logprobs: list[float] = []
    for entry in raw_logprobs[prompt_token_count:]:
        if entry is None:
            continue
        top = next(iter(entry.values()))
        logprobs.append(top["logprob"])
    return logprobs


async def _verify_gpu_ready(
    min_free_gb: float = FEEDBACK_MIN_FREE_VRAM_GB,
    timeout_s: float = FEEDBACK_SLEEP_VERIFY_TIMEOUT_S,
) -> None:
    """Poll GPU memory until *min_free_gb* is available or *timeout_s* expires.

    Called after ``POST /sleep`` so training only starts once vLLM has
    actually released its VRAM.  If torch is not installed (CPU-only API
    image) the check is skipped silently.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
    except ImportError:
        return

    deadline = time.perf_counter() + timeout_s
    while True:
        free_bytes, _total = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024**3)
        if free_gb >= min_free_gb:
            logger.info("GPU has %.1f GB free — ready for training", free_gb)
            return
        if time.perf_counter() >= deadline:
            logger.warning(
                "Only %.1f GB GPU memory free after waiting %.0fs for vLLM sleep. "
                "Training may OOM.",
                free_gb,
                timeout_s,
            )
            return
        logger.info(
            "GPU has %.1f GB free (need %.1f GB) — waiting for vLLM to release memory…",
            free_gb,
            min_free_gb,
        )
        await asyncio.sleep(1.0)


async def _vllm_reload_lora(lora_id: str) -> None:
    """Unload and reload a LoRA adapter so vLLM picks up on-disk changes.

    Requires VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 on the vLLM server.
    """
    import re

    resolved = resolve_lora_id(lora_id)
    vllm_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", resolved.strip("/")).strip("-") or "lora"
    lora_path = os.path.join(LORA_MOUNT_PATH, resolved)

    try:
        await _vllm_post(
            "/v1/unload_lora_adapter",
            json_body={"lora_name": vllm_name},
        )
    except httpx.HTTPStatusError as e:
        # 404 = adapter not found (first run or already unloaded) — safe to ignore
        if e.response.status_code != 404:
            raise
    await _vllm_post(
        "/v1/load_lora_adapter",
        json_body={"lora_name": vllm_name, "lora_path": lora_path},
    )


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
            # Offload the base model to CPU so vLLM can reclaim full VRAM.
            # This must happen even on failure — otherwise the 16GB base model
            # stays on GPU and vLLM's wake triggers CUDA memory conflicts.
            try:
                logger.info("_run_distill: calling _offload_base_model")
                await asyncio.to_thread(worker._offload_base_model)
                logger.info("_run_distill: offload complete")
            except Exception as offload_err:
                logger.error("_run_distill: offload FAILED: %s", offload_err)
            del worker
            gc.collect()

    distill_fn = modal.Function.from_name("claas-distill", "DistillWorker.distill")
    result = await distill_fn.remote.aio(payload)
    return DistillResponse.model_validate(result).model_dump()


# API Endpoints


@web_app.post("/v1/distill", response_model=DistillResponse)
async def distill(request: DistillRequest) -> DistillResponse:
    """Run a single SDPO distillation step.

    This endpoint:
    1. Loads the user's LoRA from local storage (or Modal Volume in remote mode)
    2. Runs the student model forward pass
    3. Gets teacher logprobs from configured source
       - self (default): base model conditioned on feedback
       - remote: vLLM TeacherService
    4. Computes SDPO loss (JSD-based policy gradient)
    5. Updates LoRA parameters
    6. Saves the updated LoRA back to local storage (or Modal Volume in remote mode)

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
            teacher_prompt = format_teacher_prompt(request.prompt, request.feedback)
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
    except (ValueError, RuntimeError, OSError) as e:
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

    resolved_id = await asyncio.to_thread(resolve_lora_id, request.lora_id)
    lock = await _get_feedback_lock(resolved_id)
    try:
        # Validate LoRA exists before attempting orchestration.
        exists = await asyncio.to_thread(lora_exists, request.lora_id)
        if not exists:
            raise HTTPException(status_code=404, detail=f"LoRA not found: {request.lora_id}")

        await asyncio.wait_for(lock.acquire(), timeout=FEEDBACK_LOCK_TIMEOUT_S)
        lock_acquired = True

        if request.orchestration.sleep_before:
            phase = "drain"
            try:
                await _wait_for_vllm_idle()
            except (TimeoutError, httpx.HTTPError) as e:
                raise HTTPException(status_code=503, detail=f"vLLM not idle: {e}") from e

            if request.rollout_logprobs is None:
                phase = "logprobs"
                logprobs_start = time.perf_counter()
                vllm_model = _resolve_vllm_model_name(request.lora_id)
                if vllm_model:
                    try:
                        fetched = await _fetch_rollout_logprobs(
                            request.prompt, request.response, vllm_model,
                        )
                        request = request.model_copy(update={"rollout_logprobs": fetched})
                    except (httpx.HTTPError, ValueError, KeyError) as e:
                        logger.warning("Failed to fetch rollout logprobs: %s", e)
                timing_ms.logprobs = int((time.perf_counter() - logprobs_start) * 1000)

            phase = "sleep"
            sleep_start = time.perf_counter()
            await _vllm_post(
                "/sleep",
                params={"level": request.orchestration.sleep_level},
            )
            await _verify_gpu_ready()
            timing_ms.sleep = int((time.perf_counter() - sleep_start) * 1000)
            slept = True

        phase = "distill"
        distill_start = time.perf_counter()
        payload = request.model_dump()
        payload["save_in_place"] = True

        if request.training.teacher_mode == "remote":
            teacher_score_fn = modal.Function.from_name("claas-distill", "TeacherService.score_tokens")
            teacher_prompt = format_teacher_prompt(request.prompt, request.feedback)
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
            await _vllm_reload_lora(request.lora_id)
            timing_ms.wake = int((time.perf_counter() - wake_start) * 1000)
            woke = True

        timing_ms.total = int((time.perf_counter() - started_total) * 1000)
    except asyncio.TimeoutError:
        phase = "lock"
        error_message = f"Timed out waiting for lock on LoRA '{request.lora_id}'"
        raise HTTPException(status_code=409, detail=error_message) from None
    except HTTPException as e:
        error_message = str(e.detail)
        logger.error("Feedback %s failed in phase '%s' (HTTP %d): %s", request_id, phase, e.status_code, error_message)
        raise
    except (ValueError, RuntimeError, ImportError, OSError, httpx.HTTPError) as e:
        error_message = str(e)
        logger.error("Feedback %s failed in phase '%s': %s", request_id, phase, error_message, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Feedback update failed in phase '{phase}': {error_message}",
        ) from e
    finally:
        # Always attempt to wake vLLM if we slept it and haven't woken it yet.
        # This lives in `finally` so it runs regardless of exception type.
        if (
            slept
            and not woke
            and request.orchestration.wake_after
            and (request.orchestration.wake_on_failure or FEEDBACK_WAKE_ON_FAILURE)
        ):
            try:
                await _vllm_post("/wake_up")
                woke = True
                logger.info("Feedback %s: woke vLLM after failure in phase '%s'", request_id, phase)
            except httpx.HTTPError as wake_err:
                logger.warning("Feedback %s: failed to wake vLLM after error: %s", request_id, wake_err)

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
            request=request,
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
            logger.warning("Failed to write feedback log for request %s", request_id, exc_info=True)
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
    _validate_init_base_model(request.base_model)

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

    except (ValueError, RuntimeError, OSError) as e:
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
    except (ValueError, OSError) as e:
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
    except (ValueError, OSError) as e:
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

    if DISTILL_EXECUTION_MODE == "local":
        worker = ServiceHealth(status="healthy", error=None)
        teacher = ServiceHealth(status="healthy", error=None)
    else:
        try:
            worker_health_fn = modal.Function.from_name("claas-distill", "DistillWorker.health_check")
            data = await asyncio.wait_for(worker_health_fn.remote.aio(), timeout=15)
            worker = ServiceHealth.model_validate(data)
        except (asyncio.TimeoutError, ConnectionError, OSError, ValueError, RuntimeError) as e:
            worker = ServiceHealth(status="unhealthy", error=str(e))
            status = "degraded"

        try:
            teacher_health_fn = modal.Function.from_name("claas-distill", "TeacherService.health_check")
            data = await asyncio.wait_for(teacher_health_fn.remote.aio(), timeout=15)
            teacher = ServiceHealth.model_validate(data)
        except (asyncio.TimeoutError, ConnectionError, OSError, ValueError, RuntimeError) as e:
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
