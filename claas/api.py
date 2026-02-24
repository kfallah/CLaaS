"""CLaaS API: FastAPI web endpoint for SDPO continual distillation and inference.

This module provides the REST API for the distillation service and
inference proxy. Runtime backend config is selected via Hydra config name
(``local``, ``modal``, or ``tinker``) at process startup.

Inference is proxied through the API rather than served directly so that
chain-of-thought (thinking) tags can be stripped from user-facing responses
while the raw completion is cached for the training pipeline. This cache
also lets the ``/feedback`` endpoint retrieve the on-policy rollout (including
thinking) needed for self-distillation.

Endpoints:
- POST /v1/chat/completions: Chat completion (forwarded to inference backend)
- POST /v1/completions: Text completion (forwarded to inference backend)
- POST /v1/score: Score a completion by computing per-token logprobs
- GET  /v1/models: List available models
- POST /v1/feedback: Run one online update transaction (primary endpoint)
- POST /v1/lora/init: Initialize a new LoRA adapter
- GET  /v1/lora: List all LoRA adapters
- GET  /v1/lora/export: Download a LoRA as a zip archive
- GET  /v1/health: Health check

Example usage (feedback)::

    curl -X POST http://localhost:8080/v1/feedback \\
        -H "Content-Type: application/json" \\
        -d '{
            "lora_id": "user/my-lora-init",
            "prompt": "Write a function to calculate factorial",
            "response": "def factorial(n): ...",
            "feedback": "Good recursive solution"
        }'
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
import uuid
from pathlib import Path

import httpx
import hydra
import modal
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from omegaconf import OmegaConf

from .core.config import (
    CoreConfig,
    LocalConfig,
    ModalConfig,
    TinkerConfig,
    get_engine_kind,
    load_core_config,
    register_config_schemas,
)
from .core.types import (
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionUsage,
    DistillBatchItem,
    DistillBatchRequestPayload,
    DistillResponse,
    FeedbackBatchRequest,
    FeedbackLogRecord,
    FeedbackLogVllmState,
    FeedbackResponse,
    FeedbackTimingMs,
    HealthResponse,
    LoraDeleteResponse,
    LoraInitRequest,
    LoraInitResponse,
    LoraListResponse,
    ScoreRequest,
    ScoreResponse,
    ServiceHealth,
    TextCompletionChoice,
    TextCompletionResponse,
)
from .dashboard import feedback_log as feedback_log_mod, rendering as dashboard_rendering
from .inference import get_inference_backend, vllm_control
from .inference.base import InferenceBackend
from .inference.cache import CompletionCacheEntry, completion_cache
from .inference.helpers import (
    coerce_content,
    extract_final_channel,
    normalize_for_hash,
    stream_chat_response,
    stream_completion_response,
    strip_thinking,
)
from .training.engine import get_training_engine
from .training.engine.base import EngineKind, TrainingEngine
from .training.storage import (
    LORA_MOUNT_PATH,
    configure_storage_backend,
    configure_storage_root,
    lora_volume,
)

logger = logging.getLogger(__name__)
register_config_schemas()

# Modal app for API surface; worker/teacher are resolved by name at runtime.
app = modal.App("claas-distill")


# FastAPI app
web_app = FastAPI(
    title="CLaaS API",
    description="Continual Learning as a Service - SDPO-style distillation",
    version="0.1.0",
)


def configure_web_app(cfg: CoreConfig) -> None:
    """Inject runtime config into the process-local FastAPI app."""
    web_app.state.runtime_config = cfg
    configure_storage_root(cfg.lora_root)
    backend = cfg.storage_backend
    if backend == "local_fs":
        configure_storage_backend("local_fs")
    elif backend == "modal_volume":
        configure_storage_backend("modal_volume")
    else:
        raise ValueError(f"Unsupported storage backend: {backend!r}")
    # Inference backend
    inference = get_inference_backend(get_engine_kind(cfg), cfg=cfg)
    completion_cache._max_size = cfg.completion_cache_size
    inference.register_routes(web_app)
    web_app.state.inference_backend = inference


def _runtime_config() -> CoreConfig:
    cfg = getattr(web_app.state, "runtime_config", None)
    if isinstance(cfg, (LocalConfig, ModalConfig, TinkerConfig)):
        return cfg
    raise TypeError("Hydra did not produce a supported CLaaS runtime config")


# ---------------------------------------------------------------------------
# Engine / backend helpers
# ---------------------------------------------------------------------------


def _get_engine_kind() -> EngineKind:
    """Validate and return the configured engine kind."""
    return get_engine_kind(_runtime_config())

def _get_training_engine() -> TrainingEngine:
    """Build the configured training engine instance."""
    cfg = _runtime_config()
    return get_training_engine(_get_engine_kind(), cfg)


_feedback_locks: dict[str, asyncio.Lock] = {}
_feedback_locks_guard = asyncio.Lock()


def _validate_init_base_model(base_model: str) -> None:
    cfg = _runtime_config()
    if base_model in cfg.allowed_init_base_models:
        return

    logger.warning("Rejected /v1/lora/init for disallowed base_model: %s", base_model)
    allowed = ", ".join(sorted(cfg.allowed_init_base_models))
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


async def _get_feedback_lock_key(lora_id: str) -> str:
    """Resolve the lock key to a canonical LoRA runtime identifier when possible."""
    key = lora_id.strip("/")
    if _get_engine_kind() not in {"local", "modal"}:
        return key

    runtime_ref = await _get_training_engine().lora_runtime_ref(lora_id)
    return runtime_ref.vllm_name


async def _run_distill(payload: DistillBatchRequestPayload) -> DistillResponse:
    """Execute a distill request via configured execution backend."""
    engine = _get_training_engine()
    result = await engine.distill(payload)
    return result


def _get_inference_backend(request: Request) -> InferenceBackend:
    """Retrieve the inference backend from FastAPI app state."""
    return request.app.state.inference_backend


# ---------------------------------------------------------------------------
# Inference endpoints
# ---------------------------------------------------------------------------


@web_app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    req: ChatCompletionRequest,
    request: Request,
) -> ChatCompletionResponse | StreamingResponse:
    """Chat completion endpoint (forwards to configured inference backend)."""
    backend = _get_inference_backend(request)

    messages = [
        {"role": m.role, "content": coerce_content(m.content)}
        for m in req.messages
    ]
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    system_prompt = "\n".join(system_parts) if system_parts else None
    result = await backend.chat_completion(
        messages=messages,
        model=req.model or "default",
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=req.stop,
        logprobs=req.logprobs,
        top_logprobs=req.top_logprobs,
    )

    # Strip thinking and channel tags from visible content
    visible = extract_final_channel(result.content)
    visible = strip_thinking(visible)

    # Cache raw completion for training pipeline retrieval
    content_hash = hashlib.sha256(
        normalize_for_hash(visible).encode("utf-8"),
    ).hexdigest()
    completion_cache.put(
        content_hash,
        CompletionCacheEntry(
            prompt=result.raw_prompt,
            response=result.raw_response,
            response_token_ids=result.response_token_ids,
            prompt_token_ids=result.prompt_token_ids,
            response_logprobs=result.response_logprobs,
            system_prompt=system_prompt,
        ),
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if req.stream:
        return stream_chat_response(completion_id, created, req.model or "default", visible)

    choice = ChatCompletionChoice(
        message=ChatCompletionChoiceMessage(content=visible),
    )
    if req.logprobs and result.logprobs_content is not None:
        choice.logprobs = result.logprobs_content

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=req.model or "default",
        choices=[choice],
        usage=CompletionUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )


@web_app.post("/v1/completions", response_model=None)
async def completions(
    req: CompletionRequest,
    request: Request,
) -> TextCompletionResponse | StreamingResponse:
    """Text completion endpoint (forwards to configured inference backend)."""
    backend = _get_inference_backend(request)

    result = await backend.text_completion(
        prompt=req.prompt,
        model=req.model or "default",
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=req.stop,
    )

    completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if req.stream:
        return stream_completion_response(completion_id, created, req.model or "default", result.text)

    return TextCompletionResponse(
        id=completion_id,
        created=created,
        model=req.model or "default",
        choices=[
            TextCompletionChoice(text=result.text),
        ],
        usage=CompletionUsage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )



@web_app.get("/v1/models", response_model=None)
async def list_models(request: Request) -> dict[str, object] | Response:
    """List available models from the inference backend."""
    backend = _get_inference_backend(request)
    return await backend.list_models()


@web_app.post("/v1/score", response_model=ScoreResponse)
async def score_completion(req: ScoreRequest, request: Request) -> ScoreResponse:
    """Score a completion by computing per-token logprobs."""
    backend = _get_inference_backend(request)
    messages = [
        {"role": m.role, "content": coerce_content(m.content)}
        for m in req.messages
    ]
    result = await backend.score(model=req.model, messages=messages, completion=req.completion)
    return result


# ---------------------------------------------------------------------------
# Training API endpoints
# ---------------------------------------------------------------------------


@web_app.post("/v1/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackBatchRequest) -> FeedbackResponse:
    """Run one explicit batched feedback update without API-side buffering."""
    request_id = uuid.uuid4().hex
    slept = False
    woke = False
    phase = "validate"
    distill_result: DistillResponse | None = None
    error_message: str | None = None
    log_path = ""
    timing_ms = FeedbackTimingMs()
    started_total = time.perf_counter()
    batch_requests = request.requests

    if len(batch_requests) == 0:
        raise HTTPException(status_code=422, detail="requests must contain at least one item")

    first_request = batch_requests[0]
    lora_id = first_request.lora_id
    training_ref = first_request.training.model_dump(mode="json")

    for req in batch_requests[1:]:
        if req.lora_id != lora_id:
            raise HTTPException(status_code=400, detail="all requests must use the same lora_id")
        if req.training.model_dump(mode="json") != training_ref:
            raise HTTPException(status_code=400, detail="all requests must use the same training config")

    # Resolve cache entries before acquiring lock or doing orchestration.
    batch_samples: list[DistillBatchItem] = []
    for req in batch_requests:
        content_hash = hashlib.sha256(
            normalize_for_hash(req.response).encode("utf-8"),
        ).hexdigest()
        entry = completion_cache.get(content_hash)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail=f"No cached completion for content_hash={content_hash[:16]}…",
            )
        if entry.response_logprobs is None:
            raise HTTPException(
                status_code=422,
                detail=f"Cached completion has no logprobs (content_hash={content_hash[:16]}…)",
            )
        if entry.system_prompt is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Cached completion has no system prompt (content_hash={content_hash[:16]}…). "
                    "The chat completion request must include a system message so the teacher "
                    "can score under the same context as the student."
                ),
            )
        batch_samples.append(
            DistillBatchItem(
                prompt=entry.prompt,
                response=entry.response,
                feedback=req.feedback,
                response_logprobs=entry.response_logprobs,
                prompt_token_ids=entry.prompt_token_ids,
                response_token_ids=entry.response_token_ids,
                user_prompt=req.prompt,
                system_prompt=entry.system_prompt,
            )
        )

    try:
        exists_payload = await _get_training_engine().lora_exists(lora_id)
        if not exists_payload.exists:
            raise HTTPException(status_code=404, detail=f"LoRA not found: {lora_id}")

        lock_key = await _get_feedback_lock_key(lora_id)
        lock = await _get_feedback_lock(lock_key)
        cfg = _runtime_config()
        lock_timeout = getattr(cfg, "feedback_lock_timeout_s", 120.0)
        await asyncio.wait_for(lock.acquire(), timeout=lock_timeout)

        try:
            engine_kind = _get_engine_kind()
            base_url, api_key = vllm_control.vllm_connection(cfg)

            if request.orchestration.sleep_before and engine_kind != "tinker":
                phase = "drain"
                drain_timeout = getattr(cfg, "feedback_drain_timeout_s", 30.0)
                try:
                    await vllm_control.wait_for_vllm_idle(base_url, api_key, drain_timeout)
                except (TimeoutError, httpx.HTTPError) as e:
                    raise HTTPException(status_code=503, detail=f"vLLM not idle: {e}") from e

                phase = "sleep"
                sleep_start = time.perf_counter()
                await vllm_control.vllm_post(
                    base_url, api_key,
                    "/pause",
                    params={"level": request.orchestration.sleep_level},
                )
                min_free_gb = getattr(cfg, "feedback_min_free_vram_gb", 20.0)
                verify_timeout = getattr(cfg, "feedback_sleep_verify_timeout_s", 30.0)
                await vllm_control.verify_gpu_ready(min_free_gb, verify_timeout)
                timing_ms.sleep = int((time.perf_counter() - sleep_start) * 1000)
                slept = True

            phase = "distill"
            distill_start = time.perf_counter()
            payload = DistillBatchRequestPayload(
                lora_id=lora_id,
                training=first_request.training,
                samples=batch_samples,
                save_in_place=True,
            )

            distill_result = await _run_distill(payload)
            timing_ms.distill = int((time.perf_counter() - distill_start) * 1000)

            # Always reload LoRA so vLLM picks up new weights
            if engine_kind not in {"tinker"}:
                engine = _get_training_engine()
                await vllm_control.vllm_reload_lora(engine, base_url, api_key, lora_id)

            # Resume vLLM from sleep (separate concern — GPU memory management)
            if request.orchestration.wake_after and engine_kind != "tinker":
                phase = "wake"
                wake_start = time.perf_counter()
                await vllm_control.vllm_post(base_url, api_key, "/resume")
                timing_ms.wake = int((time.perf_counter() - wake_start) * 1000)
                woke = True

            if engine_kind == "tinker" and distill_result is not None:
                sampler_path = distill_result.metadata.get("sampler_weights_path")
                if sampler_path:
                    phase = "wake"
                    wake_start = time.perf_counter()
                    backend: InferenceBackend = web_app.state.inference_backend
                    from .inference.tinker import TinkerBackend

                    if isinstance(backend, TinkerBackend):
                        backend.refresh_sampler(str(sampler_path))
                    timing_ms.wake = int((time.perf_counter() - wake_start) * 1000)
                    woke = True
        finally:
            lock.release()

        timing_ms.total = int((time.perf_counter() - started_total) * 1000)
    except asyncio.TimeoutError:
        phase = "lock"
        error_message = f"Timed out waiting for lock on LoRA '{lora_id}'"
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
        if (
            slept
            and not woke
            and request.orchestration.wake_after
            and (
                request.orchestration.wake_on_failure
                or getattr(_runtime_config(), "feedback_wake_on_failure", True)
            )
        ):
            try:
                base_url, api_key = vllm_control.vllm_connection(_runtime_config())
                await vllm_control.vllm_post(base_url, api_key, "/resume")
                woke = True
                logger.info("Feedback %s: woke vLLM after failure in phase '%s'", request_id, phase)
            except httpx.HTTPError as wake_err:
                logger.warning("Feedback %s: failed to wake vLLM after error: %s", request_id, wake_err)

        timing_ms.total = int((time.perf_counter() - started_total) * 1000)

        log_record = FeedbackLogRecord(
            request_id=request_id,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            status="ok" if error_message is None else "error",
            phase=phase,
            lora_id=lora_id,
            requests=batch_requests,
            vllm=FeedbackLogVllmState(slept=slept, woke=woke),
            timing_ms=timing_ms,
            batch_samples=batch_samples,
            distill_result=distill_result,
            error=error_message,
        )
        try:
            log_dir = _runtime_config().feedback_log_dir
            log_path = await asyncio.to_thread(
                feedback_log_mod.write_feedback_log, log_record.model_dump(mode="json"), log_dir,
            )
        except (OSError, TypeError, ValueError):
            logger.warning("Failed to write feedback log for request %s", request_id, exc_info=True)
            log_path = ""

    return FeedbackResponse(
        status="ok",
        request_id=request_id,
        lora_id=distill_result.lora_id if distill_result else lora_id,
        distill_result=distill_result,
        vllm=FeedbackLogVllmState(slept=slept, woke=woke),
        feedback_log_path=log_path,
        timing_ms=timing_ms,
        batch_size=len(batch_requests),
    )


@web_app.post("/v1/lora/init", response_model=LoraInitResponse)
async def init_lora(request: LoraInitRequest) -> LoraInitResponse:
    """Initialize a new LoRA adapter.

    Creates a new LoRA adapter configuration in the Modal Volume.
    The adapter will have zero weights initially and will be trained
    through distill calls.
    """
    engine_kind = _get_engine_kind()
    if engine_kind in {"local", "modal"}:
        _validate_init_base_model(request.base_model)

    try:
        result = await _get_training_engine().init_lora(request)
        if engine_kind in {"local", "modal"}:
            try:
                cfg = _runtime_config()
                base_url, api_key = vllm_control.vllm_connection(cfg)
                engine = _get_training_engine()
                await vllm_control.vllm_reload_lora(engine, base_url, api_key, result.lora_id)
            except Exception:
                logger.warning("Failed to load LoRA into vLLM after init", exc_info=True)
        return result
    except (ValueError, RuntimeError, OSError, httpx.HTTPError) as e:
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
        return await _get_training_engine().list_loras(prefix)
    except (ValueError, OSError, httpx.HTTPError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list LoRAs: {str(e)}",
        ) from e


@web_app.delete("/v1/lora", response_model=LoraDeleteResponse)
async def delete_lora_adapter(lora_id: str) -> LoraDeleteResponse:
    """Delete a LoRA adapter.

    Returns {"deleted": false} if not found (idempotent, no 404).
    """
    try:
        return await _get_training_engine().delete_lora(lora_id)
    except (ValueError, RuntimeError, OSError, httpx.HTTPError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"LoRA deletion failed: {str(e)}",
        ) from e


@web_app.get("/v1/lora/export")
async def export_lora_adapter(lora_id: str) -> Response:
    """Export a LoRA adapter as a zip archive for local inference servers."""
    try:
        exists_payload = await _get_training_engine().lora_exists(lora_id)
        if not exists_payload.exists:
            raise HTTPException(
                status_code=404,
                detail=f"LoRA not found: {lora_id}",
            )

        export_payload = await _get_training_engine().export_lora(lora_id)
        safe_filename = re.sub(r'[^\w._-]', '_', export_payload.filename)
        return Response(
            content=export_payload.content,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{safe_filename}"',
            },
        )
    except HTTPException:
        raise
    except (ValueError, OSError, httpx.HTTPError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"LoRA export failed: {str(e)}",
        ) from e


@web_app.get("/v1/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check health of the API and backing services."""
    status = "healthy"

    try:
        worker = await get_training_engine(_get_engine_kind(), _runtime_config()).health()
    except (asyncio.TimeoutError, ConnectionError, OSError, ValueError, RuntimeError, httpx.HTTPError) as e:
        worker = ServiceHealth(status="unhealthy", error=str(e))
        status = "degraded"

    return HealthResponse(status=status, worker=worker)


@web_app.get("/v1/dashboard", response_class=HTMLResponse)
async def dashboard(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=200),
) -> HTMLResponse:
    """Serve a paginated dashboard of recent feedback records.

    Args:
        page: Page number (1-indexed).
        per_page: Number of records per page (max 200).

    Returns:
        HTML dashboard containing recent feedback details and metrics.
    """
    from .dashboard.pagination import paginate, render_pagination_nav

    log_dir = _runtime_config().feedback_log_dir

    # Two-step: get total first so paginate() can clamp the page,
    # then fetch the correct slice.
    _, total = await asyncio.to_thread(feedback_log_mod.read_recent_feedback_logs, log_dir, 0, 0)
    info = paginate(total, page, per_page)
    records, _ = await asyncio.to_thread(
        feedback_log_mod.read_recent_feedback_logs, log_dir, info.offset, per_page
    )
    nav = render_pagination_nav(info, "/v1/dashboard")
    html_content = dashboard_rendering.feedback_dashboard_html(records, pagination_nav=nav)
    return HTMLResponse(content=html_content)


@web_app.get("/v1/eval", response_class=HTMLResponse)
async def eval_dashboard(
    results_dir: str = Query(default="./data/evals"),
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=200),
) -> HTMLResponse:
    """Serve a paginated dashboard of evaluation results.

    Args:
        results_dir: Path to the eval results directory.
        page: Page number (1-indexed).
        per_page: Number of runs per page (max 200).

    Returns:
        HTML dashboard with summary and per-preference step details.
    """
    from .dashboard.eval_dashboard import eval_dashboard_html

    base_dir = Path("./data/evals").resolve()
    requested_dir = Path(results_dir).resolve()
    if not requested_dir.is_relative_to(base_dir):
        raise HTTPException(
            status_code=400,
            detail="results_dir must be within ./data/evals",
        )
    content = await asyncio.to_thread(
        eval_dashboard_html, str(requested_dir), page=page, per_page=per_page
    )
    return HTMLResponse(content=content)


@web_app.get("/health")
async def health_check_root() -> HealthResponse:
    """Health check at root path (alias for /v1/health)."""
    return await health_check()


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
    configure_web_app(load_core_config("modal"))
    return web_app


@hydra.main(version_base=None, config_path="core/configs", config_name="local")
def main(cfg: LocalConfig | ModalConfig | TinkerConfig) -> None:
    """Hydra entry point for running the API locally with explicit config profile."""
    parsed_cfg = OmegaConf.to_object(cfg)
    if not isinstance(parsed_cfg, (LocalConfig, ModalConfig, TinkerConfig)):
        raise TypeError("Hydra did not produce a supported CLaaS runtime config")
    configure_web_app(parsed_cfg)
    host = os.environ.get("CLAAS_API_HOST", "0.0.0.0")
    port = int(os.environ.get("CLAAS_API_PORT", "8080"))
    uvicorn.run(web_app, host=host, port=port)


if __name__ == "__main__":
    main()
