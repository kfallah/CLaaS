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
- GET  /v1/completions/raw: Retrieve cached raw completion by content hash
- GET  /v1/models: List available models
- POST /v1/feedback: Run one online update transaction (primary endpoint)
- POST /v1/distill: Run a single SDPO distillation step (low-level)
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
import html
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import hydra
import modal
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from omegaconf import OmegaConf

from .core.config import (
    CoreConfig,
    LocalConfig,
    ModalConfig,
    TinkerConfig,
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
    DistillRequest,
    DistillRequestPayload,
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
    RawCompletionResponse,
    ServiceHealth,
    TextCompletionChoice,
    TextCompletionResponse,
)
from .inference import get_inference_backend
from .inference.base import InferenceBackend
from .inference.cache import CompletionCacheEntry, completion_cache
from .inference.helpers import (
    coerce_content,
    extract_final_channel,
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
from .training.teacher_helpers import format_teacher_prompt

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

FEEDBACK_DASHBOARD_TEMPLATE = Path(__file__).resolve().parent / "index.html"
EVAL_DASHBOARD_TEMPLATE = Path(__file__).resolve().parent / "eval_dashboard.html"


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
    inference = get_inference_backend(_get_engine_kind_from_cfg(cfg), cfg=cfg)
    completion_cache._max_size = cfg.completion_cache_size
    inference.register_routes(web_app)
    web_app.state.inference_backend = inference


def _runtime_config() -> CoreConfig:
    cfg = getattr(web_app.state, "runtime_config", None)
    if isinstance(cfg, (LocalConfig, ModalConfig, TinkerConfig)):
        return cfg
    raise TypeError("Hydra did not produce a supported CLaaS runtime config")


def _get_engine_kind_from_cfg(cfg: CoreConfig) -> EngineKind:
    """Return the engine kind from a config object (used during init)."""
    mode = cfg.mode
    if mode == "local":
        return "local"
    if mode == "modal":
        return "modal"
    if mode == "tinker":
        return "tinker"
    raise ValueError(f"Unsupported runtime config mode: {mode}")


# ---------------------------------------------------------------------------
# Engine / backend helpers
# ---------------------------------------------------------------------------


def _get_engine_kind() -> EngineKind:
    """Validate and return the configured engine kind."""
    return _get_engine_kind_from_cfg(_runtime_config())

def _uses_modal_teacher() -> bool:
    """Return whether API should fetch teacher scores from Modal TeacherService."""
    return _get_engine_kind() == "modal"


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


def _vllm_api_key() -> str:
    raw = os.environ.get("VLLM_API_KEY")
    return raw.strip() if raw is not None else ""


def _vllm_connection(cfg: CoreConfig | None = None) -> tuple[str, str]:
    """Return (base_url, api_key) from the current config."""
    if cfg is None:
        cfg = _runtime_config()
    base_url: str = getattr(cfg, "vllm_base_url", "http://127.0.0.1:8000")
    api_key = _vllm_api_key()
    return base_url, api_key


async def _vllm_post(
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout_s: float = 30.0,
) -> None:
    """Call a vLLM control endpoint and raise on non-success."""
    base_url, api_key = _vllm_connection()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        resp = await client.post(path, params=params, json=json_body, headers=headers)
    resp.raise_for_status()


async def _tinker_sampler_refresh(model_path: str) -> None:
    """Refresh the in-process Tinker sampler to the latest checkpoint."""
    backend: InferenceBackend = web_app.state.inference_backend
    from .inference.tinker import TinkerBackend

    if isinstance(backend, TinkerBackend):
        backend.holder.refresh(model_path=model_path)
        logger.info("Tinker sampler refreshed to checkpoint: %s", model_path)
    else:
        logger.warning("_tinker_sampler_refresh called but backend is not TinkerBackend")


async def _wait_for_vllm_idle(
    timeout_s: float | None = None,
) -> None:
    """Poll vLLM ``/metrics`` until no requests are running or waiting.

    Raises :class:`TimeoutError` if vLLM is still busy after *timeout_s*.
    """
    cfg = _runtime_config()
    if timeout_s is None:
        timeout_s = getattr(cfg, "feedback_drain_timeout_s", 30.0)
    base_url, api_key = _vllm_connection(cfg)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    deadline = time.perf_counter() + timeout_s

    while True:
        async with httpx.AsyncClient(base_url=base_url, timeout=10) as client:
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


async def _verify_gpu_ready(
    min_free_gb: float | None = None,
    timeout_s: float | None = None,
) -> None:
    """Poll GPU memory until *min_free_gb* is available or *timeout_s* expires.

    Called after ``POST /sleep`` so training only starts once vLLM has
    actually released its VRAM.  If torch is not installed (CPU-only API
    image) the check is skipped silently.
    """
    cfg = _runtime_config()
    if min_free_gb is None:
        min_free_gb = getattr(cfg, "feedback_min_free_vram_gb", 20.0)
    if timeout_s is None:
        timeout_s = getattr(cfg, "feedback_sleep_verify_timeout_s", 30.0)

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
    runtime_ref = await _get_training_engine().lora_runtime_ref(lora_id)

    try:
        await _vllm_post(
            "/v1/unload_lora_adapter",
            json_body={"lora_name": runtime_ref.vllm_name},
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 404:
            raise
    await _vllm_post(
        "/v1/load_lora_adapter",
        json_body={"lora_name": runtime_ref.vllm_name, "lora_path": runtime_ref.lora_path},
    )


def _write_feedback_log(record: dict[str, Any] | FeedbackLogRecord) -> str:
    """Persist a feedback lifecycle record to disk and return its path."""
    if isinstance(record, FeedbackLogRecord):
        payload = record.model_dump(mode="json")
        request_id = record.request_id
    else:
        payload = record
        request_id = str(payload.get("request_id", ""))

    log_root = Path(_runtime_config().feedback_log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    request_id = request_id or uuid.uuid4().hex
    path = log_root / f"{timestamp}-{request_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return str(path)


def _read_recent_feedback_logs(
    offset: int = 0, limit: int = 20
) -> tuple[list[FeedbackLogRecord], int]:
    """Load recent feedback records from disk.

    Args:
        offset: Number of records to skip (for pagination).
        limit: Maximum number of records to load.

    Returns:
        A tuple of (records, total) where *records* is ordered newest-first
        and *total* is the full count of log files on disk.
    """
    log_root = Path(_runtime_config().feedback_log_dir)
    if not log_root.exists():
        return [], 0

    log_paths = sorted(log_root.glob("*.json"), reverse=True)
    total = len(log_paths)
    selected_paths = log_paths[offset : offset + limit]
    records: list[FeedbackLogRecord] = []
    for path in selected_paths:
        with path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        try:
            records.append(FeedbackLogRecord.model_validate(payload))
        except Exception:
            logger.warning("Skipping invalid feedback log: %s", path)
    return records, total


def _feedback_prompt_preview(prompt: str, limit: int = 140) -> str:
    """Build a single-line prompt preview for table display.

    Args:
        prompt: Full prompt text.
        limit: Maximum preview length.

    Returns:
        Prompt preview trimmed to the requested length.
    """
    normalized = " ".join(prompt.splitlines())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}…"


def _feedback_dashboard_rows(records: list[FeedbackLogRecord]) -> str:
    """Render dashboard table rows for feedback records, grouped by batch.

    Each batch gets a single summary row.  Expanding it reveals per-sample
    details (prompt / response / feedback) plus the batch-level timing,
    training metrics, and orchestration info.
    """
    rows: list[str] = []
    for idx, record in enumerate(records):
        metrics_payload: dict[str, object] = {}
        if record.distill_result is not None:
            metrics_payload = record.distill_result.metadata
        timing_json = json.dumps(record.timing_ms.model_dump(mode="json"), indent=2, sort_keys=True)
        metrics_json = json.dumps(metrics_payload, indent=2, sort_keys=True)
        vllm_json = json.dumps(record.vllm.model_dump(mode="json"), indent=2, sort_keys=True)
        error_value = record.error or ""
        batch_size = len(record.batch_samples)
        detail_row_id = f"feedback-detail-{idx}"

        # -- Batch summary row --
        rows.append(
            """
            <tr>
              <td>{request_id}<br><small>{timestamp}</small></td>
              <td>{status} ({phase})</td>
              <td>{lora_id}</td>
              <td>{batch_size} sample{plural}</td>
              <td>{distill_ms}</td>
              <td>{total_ms}</td>
              <td><button type="button" onclick="toggleDetails('{detail_row_id}', this)">Expand</button></td>
            </tr>
            """.format(
                request_id=html.escape(record.request_id),
                timestamp=html.escape(record.timestamp_utc),
                status=html.escape(record.status),
                phase=html.escape(record.phase),
                lora_id=html.escape(record.lora_id),
                batch_size=batch_size,
                plural="s" if batch_size != 1 else "",
                distill_ms=record.timing_ms.distill,
                total_ms=record.timing_ms.total,
                detail_row_id=detail_row_id,
            )
        )

        # -- Expandable detail row --
        sample_sections: list[str] = []
        for item_index, sample in enumerate(record.batch_samples):
            sample_sections.append(
                """
                <details{open_attr}>
                  <summary>Sample {item_number}/{batch_size} &mdash; {prompt_preview}</summary>
                  <div class="detail-panel">
                    <section><h3>Prompt</h3><pre>{prompt}</pre></section>
                    <section><h3>Response</h3><pre>{response}</pre></section>
                    <section><h3>Feedback</h3><pre>{feedback}</pre></section>
                  </div>
                </details>
                """.format(
                    open_attr=" open" if batch_size == 1 else "",
                    item_number=item_index + 1,
                    batch_size=batch_size,
                    prompt_preview=html.escape(_feedback_prompt_preview(sample.prompt, limit=80)),
                    prompt=html.escape(sample.prompt),
                    response=html.escape(sample.response),
                    feedback=html.escape(sample.feedback),
                )
            )

        rows.append(
            """
            <tr id="{detail_row_id}" class="detail-row">
              <td colspan="7">
                {samples}
                <div class="detail-panel" style="margin-top: 0.75rem">
                  <section><h3>Timing (ms)</h3><pre>{timing_json}</pre></section>
                  <section><h3>Training metrics</h3><pre>{metrics_json}</pre></section>
                  <section><h3>vLLM orchestration</h3><pre>{vllm_json}</pre></section>
                  <section><h3>Error</h3><pre>{error_value}</pre></section>
                </div>
              </td>
            </tr>
            """.format(
                detail_row_id=detail_row_id,
                samples="\n".join(sample_sections),
                timing_json=html.escape(timing_json),
                metrics_json=html.escape(metrics_json),
                vllm_json=html.escape(vllm_json),
                error_value=html.escape(error_value),
            )
        )

    if not rows:
        return '<tr><td colspan="7">No feedback records found.</td></tr>'
    return "\n".join(rows)


def _feedback_dashboard_html(
    records: list[FeedbackLogRecord], pagination_nav: str = ""
) -> str:
    """Render feedback records into the dashboard HTML template.

    Args:
        records: Feedback records to display.
        pagination_nav: Pre-rendered pagination HTML to inject.

    Returns:
        Rendered HTML content.
    """
    template = FEEDBACK_DASHBOARD_TEMPLATE.read_text(encoding="utf-8")
    table_rows = _feedback_dashboard_rows(records)
    return template.replace("{{TABLE_ROWS}}", table_rows).replace(
        "{{PAGINATION}}", pagination_nav
    )



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
    result = await backend.chat_completion(
        messages=messages,
        model=req.model or "default",
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=req.stop,
    )

    # Strip thinking and channel tags from visible content
    visible = extract_final_channel(result.content)
    visible = strip_thinking(visible)

    # Cache raw completion for training pipeline retrieval
    content_hash = hashlib.sha256(
        strip_thinking(visible).encode("utf-8"),
    ).hexdigest()
    completion_cache.put(
        content_hash,
        CompletionCacheEntry(
            prompt=result.raw_prompt,
            response=result.raw_response,
            token_ids=result.token_ids,
            prompt_token_ids=result.prompt_token_ids,
            logprobs=result.logprobs,
        ),
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if req.stream:
        return stream_chat_response(completion_id, created, req.model or "default", visible)

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=req.model or "default",
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionChoiceMessage(content=visible),
            )
        ],
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


@web_app.get("/v1/completions/raw", response_model=None)
async def get_raw_completion(content_hash: str) -> RawCompletionResponse | JSONResponse:
    """Retrieve cached raw completion by SHA-256 hash of parsed content text."""
    entry = completion_cache.get(content_hash)
    if entry is None:
        return JSONResponse(
            status_code=404,
            content={"error": "No cached completion found for this content hash"},
        )
    return RawCompletionResponse(
        prompt=entry.prompt,
        response=entry.response,
        token_ids=entry.token_ids,
        prompt_token_ids=entry.prompt_token_ids,
        logprobs=entry.logprobs,
    )


@web_app.get("/v1/models", response_model=None)
async def list_models(request: Request) -> dict[str, object] | Response:
    """List available models from the inference backend."""
    backend = _get_inference_backend(request)
    return await backend.list_models()


# ---------------------------------------------------------------------------
# Training API endpoints
# ---------------------------------------------------------------------------


@web_app.post("/v1/distill", response_model=DistillResponse)
async def distill(request: DistillRequest) -> DistillResponse:
    """Run a single SDPO distillation step.

    This endpoint:
    1. Loads the user's LoRA from local storage (or Modal Volume in modal mode)
    2. Runs the student model forward pass
    3. Gets teacher logprobs from configured source
       - self (default): base model conditioned on feedback
       - remote: vLLM TeacherService
    4. Computes SDPO loss (JSD-based policy gradient)
    5. Updates LoRA parameters
    6. Saves the updated LoRA back to local storage (or Modal Volume in modal mode)

    Returns the new LoRA ID and training metrics.
    """
    try:
        exists_payload = await _get_training_engine().lora_exists(request.lora_id)
        if not exists_payload.exists:
            raise HTTPException(
                status_code=404,
                detail=f"LoRA not found: {request.lora_id}",
            )

        single_payload = DistillRequestPayload.model_validate(request.model_dump())
        payload = DistillBatchRequestPayload(
            lora_id=single_payload.lora_id,
            training=single_payload.training,
            save_in_place=single_payload.save_in_place,
            samples=[
                DistillBatchItem(
                    prompt=single_payload.prompt,
                    response=single_payload.response,
                    feedback=single_payload.feedback,
                    rollout_logprobs=single_payload.rollout_logprobs,
                    teacher_result=single_payload.teacher_result,
                    prompt_token_ids=single_payload.prompt_token_ids,
                    response_token_ids=single_payload.response_token_ids,
                    user_prompt=single_payload.user_prompt,
                )
            ],
        )

        # Remote teacher is optional; self-distillation is the default path.
        if request.training.teacher_mode == "remote" and _uses_modal_teacher():
            teacher_score_fn = modal.Function.from_name("claas-distill", "TeacherService.score_tokens")
            teacher_prompt = format_teacher_prompt(request.user_prompt or request.prompt, request.feedback)
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
            payload.samples[0] = payload.samples[0].model_copy(update={"teacher_result": teacher_scored[0]})

        result = await _run_distill(payload)

        return result

    except HTTPException:
        raise
    except (ValueError, RuntimeError, OSError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Distillation failed: {str(e)}",
        ) from e


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
        entry = completion_cache.get(req.content_hash)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail=f"No cached completion for content_hash={req.content_hash[:16]}…",
            )
        if entry.logprobs is None:
            raise HTTPException(
                status_code=422,
                detail=f"Cached completion has no logprobs (content_hash={req.content_hash[:16]}…)",
            )
        batch_samples.append(
            DistillBatchItem(
                prompt=entry.prompt,
                response=entry.response,
                feedback=req.feedback,
                rollout_logprobs=entry.logprobs,
                prompt_token_ids=entry.prompt_token_ids,
                response_token_ids=entry.token_ids,
                user_prompt=req.user_prompt,
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
            if request.orchestration.sleep_before and _get_engine_kind() != "tinker":
                phase = "drain"
                try:
                    await _wait_for_vllm_idle()
                except (TimeoutError, httpx.HTTPError) as e:
                    raise HTTPException(status_code=503, detail=f"vLLM not idle: {e}") from e

                phase = "sleep"
                sleep_start = time.perf_counter()
                await _vllm_post(
                    "/pause",
                    params={"level": request.orchestration.sleep_level},
                )
                await _verify_gpu_ready()
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

            if first_request.training.teacher_mode == "remote" and _uses_modal_teacher():
                teacher_score_fn = modal.Function.from_name("claas-distill", "TeacherService.score_tokens")
                teacher_scored = await teacher_score_fn.remote.aio(
                    prompts=[format_teacher_prompt(req.user_prompt, s.feedback) for req, s in zip(batch_requests, batch_samples, strict=True)],
                    completions=[s.response for s in batch_samples],
                    top_k=first_request.training.teacher_top_k,
                )
                if not teacher_scored or len(teacher_scored) != len(batch_samples):
                    raise HTTPException(status_code=502, detail="Remote teacher returned invalid scores")
                payload.samples = [
                    sample_item.model_copy(update={"teacher_result": teacher_scored[idx]})
                    for idx, sample_item in enumerate(payload.samples)
                ]

            distill_result = await _run_distill(payload)
            timing_ms.distill = int((time.perf_counter() - distill_start) * 1000)

            if request.orchestration.wake_after and _get_engine_kind() != "tinker":
                phase = "wake"
                wake_start = time.perf_counter()
                await _vllm_post("/resume")
                await _vllm_reload_lora(lora_id)
                timing_ms.wake = int((time.perf_counter() - wake_start) * 1000)
                woke = True

            if _get_engine_kind() == "tinker" and distill_result is not None:
                sampler_path = distill_result.metadata.get("sampler_weights_path")
                if sampler_path:
                    phase = "wake"
                    wake_start = time.perf_counter()
                    await _tinker_sampler_refresh(str(sampler_path))
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
                await _vllm_post("/resume")
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
            teacher_mode=first_request.training.teacher_mode,
            requests=batch_requests,
            vllm=FeedbackLogVllmState(slept=slept, woke=woke),
            timing_ms=timing_ms,
            batch_samples=batch_samples,
            distill_result=distill_result,
            error=error_message,
        )
        try:
            log_path = await asyncio.to_thread(_write_feedback_log, log_record.model_dump(mode="json"))
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
    if _get_engine_kind() in {"local", "modal"}:
        _validate_init_base_model(request.base_model)

    try:
        return await _get_training_engine().init_lora(request)
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

    if _uses_modal_teacher():
        try:
            teacher_health_fn = modal.Function.from_name("claas-distill", "TeacherService.health_check")
            data = await asyncio.wait_for(teacher_health_fn.remote.aio(), timeout=15)
            teacher = ServiceHealth.model_validate(data)
        except (asyncio.TimeoutError, ConnectionError, OSError, ValueError, RuntimeError) as e:
            teacher = ServiceHealth(status="unhealthy", error=str(e))
            status = "degraded"
    else:
        teacher = ServiceHealth(status="healthy", error=None)

    return HealthResponse(status=status, worker=worker, teacher=teacher)


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
    from .pagination import paginate, render_pagination_nav

    # Two-step: get total first so paginate() can clamp the page,
    # then fetch the correct slice.
    _, total = await asyncio.to_thread(_read_recent_feedback_logs, 0, 0)
    info = paginate(total, page, per_page)
    records, _ = await asyncio.to_thread(
        _read_recent_feedback_logs, info.offset, per_page
    )
    nav = render_pagination_nav(info, "/v1/dashboard")
    html_content = _feedback_dashboard_html(records, pagination_nav=nav)
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
    from .eval.dashboard import eval_dashboard_html

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
