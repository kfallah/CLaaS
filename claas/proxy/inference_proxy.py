"""Unified OpenAI-compatible inference proxy (Tinker SDK or local vLLM).

Exposes ``/v1/chat/completions``, ``/v1/completions``, and
``/v1/completions/raw`` so that any OpenAI-compatible client (e.g. OpenClaw)
can talk to a model without caring whether it is hosted on Tinker or a local
vLLM instance.

The proxy mode is determined by the ``CLAAS_PROXY_MODE`` env var (falls back
to ``CLAAS_DISTILL_EXECUTION_MODE``, default ``"local"``).

Usage::

    # Tinker mode
    CLAAS_PROXY_MODE=tinker TINKER_API_KEY=... CLAAS_TINKER_BASE_MODEL=gpt-oss/GPT-OSS-120B \
        uvicorn claas.proxy.inference_proxy:app --host 0.0.0.0 --port 8000

    # Local / vLLM mode
    CLAAS_PROXY_MODE=local CLAAS_PROXY_VLLM_BACKEND_URL=http://vllm:8000 \
        uvicorn claas.proxy.inference_proxy:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from claas.core.config import get_proxy_config

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
else:
    PreTrainedTokenizerBase = Any

app = FastAPI(title="CLaaS Inference Proxy")


# ---------------------------------------------------------------------------
# Mode helpers
# ---------------------------------------------------------------------------

def _mode() -> str:
    """Return ``"tinker"`` or ``"local"``."""
    return get_proxy_config().mode


def _base_model() -> str:
    return get_proxy_config().tinker_base_model


# ---------------------------------------------------------------------------
# Fallback renderer using the tokenizer's built-in chat template
# ---------------------------------------------------------------------------


def _apply_chat_template_ids(tokenizer: Any, dicts: list[dict[str, str]]) -> list[int]:
    """Tokenize messages via apply_chat_template, returning token ids.

    The tokenizer parameter is typed as ``Any`` so that ty doesn't try to
    narrow the complex union return type of ``apply_chat_template``.
    """
    result = tokenizer.apply_chat_template(
        dicts, add_generation_prompt=True, tokenize=True,
    )
    if isinstance(result, list):
        return result
    return result["input_ids"]


class _TokenizerChatRenderer:
    """Minimal renderer that delegates to ``tokenizer.apply_chat_template``.

    Used when *tinker_cookbook* doesn't have a dedicated renderer for the model.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    def build_generation_prompt(self, messages: list[dict[str, str]]) -> Any:
        dicts = [{"role": m["role"], "content": m.get("content", "")} for m in messages]
        token_ids = _apply_chat_template_ids(self._tokenizer, dicts)
        # Lazy import — only available in Tinker mode
        import tinker.types as T  # noqa: N812
        return T.ModelInput.from_ints(token_ids)

    def parse_response(self, tokens: list[int]) -> tuple[dict[str, str], list[Any]]:
        text = self._tokenizer.decode(tokens, skip_special_tokens=True)
        return {"role": "assistant", "content": text}, []

    def get_stop_sequences(self) -> list[str]:
        seqs: list[str] = []
        for attr in ("eos_token",):
            tok = getattr(self._tokenizer, attr, None)
            if tok:
                seqs.append(tok)
        return seqs


# ---------------------------------------------------------------------------
# Lazy singleton for the Tinker sampling client (Tinker mode only)
# ---------------------------------------------------------------------------

def _make_renderer(
    base_model: str, tokenizer: PreTrainedTokenizerBase
) -> Any:
    """Return a tinker_cookbook renderer, falling back to the tokenizer's chat template."""
    from tinker_cookbook import model_info
    from tinker_cookbook.renderers import get_renderer

    try:
        renderer_name = model_info.get_recommended_renderer_name(base_model)
        return get_renderer(renderer_name, tokenizer=tokenizer)
    except (ValueError, KeyError):
        logger.warning(
            "No tinker_cookbook renderer for %s; falling back to tokenizer chat template",
            base_model,
        )
        return _TokenizerChatRenderer(tokenizer)


class _SamplerHolder:
    """Holds a lazily-initialized Tinker SamplingClient and tokenizer.

    Only used in Tinker mode.
    """

    def __init__(self) -> None:
        self._service: Any | None = None
        self._sampler: Any | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._renderer: Any | None = None
        self._model_path: str | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        with self._lock:
            if (
                self._sampler is not None
                and self._tokenizer is not None
                and self._renderer is not None
            ):
                return
            import tinker

            proxy_cfg = get_proxy_config()
            api_key = proxy_cfg.tinker_api_key
            if api_key:
                os.environ["TINKER_API_KEY"] = api_key
            base_model = proxy_cfg.tinker_base_model
            self._service = tinker.ServiceClient()
            self._sampler = self._service.create_sampling_client(base_model=base_model)
            self._tokenizer = self._sampler.get_tokenizer()

            self._renderer = _make_renderer(base_model, self._tokenizer)

    @property
    def sampler(self) -> Any:
        self._ensure()
        assert self._sampler is not None
        return self._sampler

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        self._ensure()
        assert self._tokenizer is not None
        return self._tokenizer

    @property
    def renderer(self) -> Any:
        self._ensure()
        assert self._renderer is not None
        return self._renderer

    def refresh(self, model_path: str | None = None) -> None:
        """Refresh the sampling client (e.g. after a distillation step)."""
        import tinker

        with self._lock:
            proxy_cfg = get_proxy_config()
            base_model = proxy_cfg.tinker_base_model
            if self._service is None:
                api_key = proxy_cfg.tinker_api_key
                if api_key:
                    os.environ["TINKER_API_KEY"] = api_key
                self._service = tinker.ServiceClient()
            if model_path:
                self._sampler = self._service.create_sampling_client(model_path=model_path)
            else:
                self._sampler = self._service.create_sampling_client(base_model=base_model)
            self._tokenizer = self._sampler.get_tokenizer()
            self._renderer = _make_renderer(base_model, self._tokenizer)
            self._model_path = model_path


_holder = _SamplerHolder()


# ---------------------------------------------------------------------------
# Raw completion cache (shared by both modes)
# ---------------------------------------------------------------------------

_CACHE_TTL_SECS = 3600  # 1 hour


class _CompletionCacheEntry:
    __slots__ = ("prompt", "response", "token_ids", "logprobs", "created_at")

    def __init__(
        self,
        prompt: str,
        response: str,
        token_ids: list[int],
        logprobs: list[float] | None,
    ) -> None:
        self.prompt = prompt
        self.response = response
        self.token_ids = token_ids
        self.logprobs = logprobs
        self.created_at = time.monotonic()

    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > _CACHE_TTL_SECS


class _CompletionCache:
    """FIFO cache keyed by SHA-256 of parsed content text."""

    def __init__(self, max_size: int | None = None) -> None:
        if max_size is None:
            max_size = get_proxy_config().completion_cache_size
        self._store: OrderedDict[str, _CompletionCacheEntry] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def put(self, content_hash: str, entry: _CompletionCacheEntry) -> None:
        with self._lock:
            if content_hash in self._store:
                self._store.move_to_end(content_hash)
                self._store[content_hash] = entry
            else:
                self._store[content_hash] = entry
                while len(self._store) > self._max_size:
                    self._store.popitem(last=False)

    def get(self, content_hash: str) -> _CompletionCacheEntry | None:
        with self._lock:
            entry = self._store.get(content_hash)
            if entry is None:
                return None
            if entry.is_expired():
                del self._store[content_hash]
                return None
            return entry

_completion_cache = _CompletionCache()

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove thinking blocks before hashing content.

    Handles two cases:
    1. Proper ``<think>...</think>`` blocks.
    2. Orphaned ``</think>`` when the opening ``<think>`` was consumed as a
       special token by the tokenizer (Qwen3).  Everything before the first
       orphaned ``</think>`` is thinking text and is stripped.
    """
    text = _THINK_RE.sub("", text)
    idx = text.find("</think>")
    if idx >= 0:
        text = text[idx + len("</think>"):]
    return text.strip()


async def _sample_async(
    sampler: Any,
    prompt: Any,
    sampling_params: Any,
) -> Any:
    """Run blocking Tinker sampling in a worker thread."""
    return await asyncio.to_thread(
        lambda: sampler.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=sampling_params,
        ).result()
    )


# ---------------------------------------------------------------------------
# Local mode: forward to vLLM
# ---------------------------------------------------------------------------

_vllm_client: httpx.AsyncClient | None = None


def _get_vllm_client() -> httpx.AsyncClient:
    global _vllm_client  # noqa: PLW0603
    if _vllm_client is None:
        _vllm_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
    return _vllm_client


def _vllm_backend_url() -> str:
    """Return the upstream vLLM base URL (no trailing slash)."""
    return get_proxy_config().vllm_backend_url.rstrip("/")


def _vllm_api_key() -> str:
    return get_proxy_config().vllm_api_key


def _build_chatml_prompt(messages: list[dict[str, str]]) -> str:
    """Reconstruct ChatML prompt from messages for cache storage."""
    parts: list[str] = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


async def _forward_to_vllm(req: ChatCompletionRequest) -> dict[str, Any]:
    """Forward a chat completion request to the upstream vLLM and extract results."""
    client = _get_vllm_client()

    messages_dicts = [
        {"role": m.role, "content": _coerce_content(m.content)}
        for m in req.messages
    ]

    body: dict[str, Any] = {
        "model": req.model,
        "messages": messages_dicts,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 1,
    }
    if req.max_tokens is not None:
        body["max_tokens"] = req.max_tokens
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if req.stop:
        body["stop"] = req.stop

    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = _vllm_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = await client.post(
        f"{_vllm_backend_url()}/v1/chat/completions",
        json=body,
        headers=headers,
    )
    resp.raise_for_status()
    data = resp.json()

    choice = data["choices"][0]
    content = choice["message"]["content"]

    # Extract logprobs if available
    logprobs: list[float] | None = None
    lp_data = choice.get("logprobs")
    if lp_data and lp_data.get("content"):
        logprobs = [entry["logprob"] for entry in lp_data["content"]]

    raw_prompt = _build_chatml_prompt(messages_dicts)
    raw_response = content

    usage = data.get("usage", {})

    return {
        "content": content,
        "raw_prompt": raw_prompt,
        "raw_response": raw_response,
        "token_ids": [],
        "logprobs": logprobs,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatCompletionMessage(BaseModel):
    role: str
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatCompletionMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    stop: list[str] | None = None


class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    stop: list[str] | None = None


class ScoreRequest(BaseModel):
    prompt: str
    completion: str


class RefreshRequest(BaseModel):
    model_path: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(req: ChatCompletionRequest) -> dict[str, object] | StreamingResponse:
    if _mode() == "local":
        return await _chat_completions_local(req)
    return await _chat_completions_tinker(req)


async def _chat_completions_local(
    req: ChatCompletionRequest,
) -> dict[str, object] | StreamingResponse:
    """Handle chat completions by forwarding to upstream vLLM."""
    if not req.model:
        # Use the first model from the upstream vLLM, or a sensible default
        req.model = "default"

    result = await _forward_to_vllm(req)

    content: str = result["content"]
    raw_prompt: str = result["raw_prompt"]
    raw_response: str = result["raw_response"]
    token_ids: list[int] = result["token_ids"]
    logprobs: list[float] | None = result["logprobs"]
    prompt_tokens: int = result["prompt_tokens"]
    completion_tokens: int = result["completion_tokens"]

    # Strip thinking and channel tags from visible content
    visible = _extract_final_channel(content)
    visible = _strip_thinking(visible)

    # Cache raw completion for training pipeline retrieval
    content_hash = hashlib.sha256(_strip_thinking(visible).encode("utf-8")).hexdigest()
    _completion_cache.put(
        content_hash,
        _CompletionCacheEntry(
            prompt=raw_prompt,
            response=raw_response,
            token_ids=token_ids,
            logprobs=logprobs,
        ),
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if req.stream:
        return _stream_chat_response(completion_id, created, req.model, visible)

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": visible},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def _chat_completions_tinker(
    req: ChatCompletionRequest,
) -> dict[str, object] | StreamingResponse:
    """Handle chat completions via Tinker SDK (original code path)."""
    import tinker.types as T  # noqa: N812
    from tinker_cookbook.renderers import Message

    if not req.model:
        req.model = _base_model()
    renderer = _holder.renderer
    sampler = _holder.sampler

    messages: list[Message] = [
        Message(role=m.role, content=_coerce_content(m.content))
        for m in req.messages
    ]
    model_input = renderer.build_generation_prompt(messages)

    stop_seqs = req.stop or renderer.get_stop_sequences()
    max_tokens = _bounded_int(req.max_tokens, default=2048, minimum=1, maximum=32768)
    temperature = _bounded_float(req.temperature, default=0.7, minimum=0.0, maximum=2.0)
    top_p = _bounded_float(req.top_p, default=1.0, minimum=0.0, maximum=1.0)
    sampling_params = T.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=0,
        seed=0,
        stop=stop_seqs,
    )

    resp = await _sample_async(sampler, model_input, sampling_params)

    seq = resp.sequences[0]
    text_msg, _ = renderer.parse_response(seq.tokens)
    content = text_msg.get("content", "") if isinstance(text_msg, dict) else str(text_msg)
    content = _extract_final_channel(content)
    content = _strip_thinking(content)

    # Cache raw completion for training pipeline retrieval
    tokenizer = _holder.tokenizer
    raw_completion_text = tokenizer.decode(seq.tokens, skip_special_tokens=False)
    prompt_text = tokenizer.decode(model_input.to_ints(), skip_special_tokens=False)
    content_hash = hashlib.sha256(_strip_thinking(content).encode("utf-8")).hexdigest()
    _completion_cache.put(
        content_hash,
        _CompletionCacheEntry(
            prompt=prompt_text,
            response=raw_completion_text,
            token_ids=list(seq.tokens),
            logprobs=list(seq.logprobs) if seq.logprobs is not None else None,
        ),
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    prompt_tokens = model_input.length
    completion_tokens = len(seq.tokens)

    if req.stream:
        return _stream_chat_response(completion_id, created, req.model, content)

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.post("/v1/completions", response_model=None)
async def completions(req: CompletionRequest) -> dict[str, object] | StreamingResponse:
    if _mode() == "local":
        return await _completions_local(req)
    return await _completions_tinker(req)


async def _completions_local(
    req: CompletionRequest,
) -> dict[str, object] | StreamingResponse:
    """Forward text completions to upstream vLLM."""
    client = _get_vllm_client()
    body: dict[str, Any] = {
        "model": req.model or "default",
        "prompt": req.prompt,
        "stream": False,
    }
    if req.max_tokens is not None:
        body["max_tokens"] = req.max_tokens
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.top_p is not None:
        body["top_p"] = req.top_p
    if req.stop:
        body["stop"] = req.stop

    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = _vllm_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = await client.post(
        f"{_vllm_backend_url()}/v1/completions",
        json=body,
        headers=headers,
    )
    resp.raise_for_status()
    data = resp.json()

    text: str = data["choices"][0]["text"]
    usage = data.get("usage", {})

    completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if req.stream:
        return _stream_completion_response(completion_id, created, req.model or "default", text)

    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": req.model or "default",
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


async def _completions_tinker(
    req: CompletionRequest,
) -> dict[str, object] | StreamingResponse:
    """Handle text completions via Tinker SDK."""
    import tinker.types as T  # noqa: N812

    if not req.model:
        req.model = _base_model()
    tokenizer = _holder.tokenizer
    sampler = _holder.sampler

    tokens: list[int] = tokenizer.encode(req.prompt)
    model_input = T.ModelInput.from_ints(tokens)

    stop_seqs = req.stop or []
    max_tokens = _bounded_int(req.max_tokens, default=2048, minimum=1, maximum=32768)
    temperature = _bounded_float(req.temperature, default=0.7, minimum=0.0, maximum=2.0)
    top_p = _bounded_float(req.top_p, default=1.0, minimum=0.0, maximum=1.0)
    sampling_params = T.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=0,
        seed=0,
        stop=stop_seqs,
    )

    resp = await _sample_async(sampler, model_input, sampling_params)

    seq = resp.sequences[0]
    text: str = tokenizer.decode(seq.tokens, skip_special_tokens=True)

    completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if req.stream:
        return _stream_completion_response(completion_id, created, req.model, text)

    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "text": text,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(tokens),
            "completion_tokens": len(seq.tokens),
            "total_tokens": len(tokens) + len(seq.tokens),
        },
    }


@app.get("/v1/completions/raw")
async def get_raw_completion(content_hash: str) -> dict[str, object]:
    """Retrieve cached raw completion by SHA-256 hash of parsed content text."""
    entry = _completion_cache.get(content_hash)
    if entry is None:
        from fastapi.responses import JSONResponse

        return JSONResponse(  # type: ignore[return-value]
            status_code=404,
            content={"error": "No cached completion found for this content hash"},
        )
    return {
        "prompt": entry.prompt,
        "response": entry.response,
        "token_ids": entry.token_ids,
        "logprobs": entry.logprobs,
    }


@app.get("/v1/models")
async def list_models() -> dict[str, object] | Response:
    if _mode() == "local":
        # Forward to upstream vLLM
        client = _get_vllm_client()
        headers: dict[str, str] = {}
        api_key = _vllm_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = await client.get(f"{_vllm_backend_url()}/v1/models", headers=headers)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type="application/json",
        )
    return {
        "object": "list",
        "data": [
            {
                "id": _base_model(),
                "object": "model",
                "owned_by": "tinker",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Tinker-only endpoints (registered conditionally)
# ---------------------------------------------------------------------------

if _mode() == "tinker":
    @app.post("/v1/sampler/refresh")
    async def refresh_sampler(body: RefreshRequest) -> dict[str, object]:
        """Refresh the sampling client, optionally pointing at a new checkpoint."""
        _holder.refresh(model_path=body.model_path)
        return {"status": "ok", "model_path": body.model_path}

    @app.get("/v1/sampler/status")
    async def sampler_status() -> dict[str, object]:
        """Return the currently loaded model path (null = base model only)."""
        return {"model_path": _holder._model_path, "base_model": _base_model()}

    @app.post("/v1/score")
    async def score_completion(req: ScoreRequest) -> dict[str, object]:
        """Score a prompt+completion pair and return per-token completion logprobs."""
        tokenizer = _holder.tokenizer
        sampler = _holder.sampler

        prompt_tokens: list[int] = list(tokenizer.encode(req.prompt, add_special_tokens=True))
        completion_tokens: list[int] = tokenizer.encode(
            req.completion, add_special_tokens=False,
        )
        full_tokens = prompt_tokens + completion_tokens
        prompt_len = len(prompt_tokens)
        completion_len = len(completion_tokens)

        import tinker.types as T  # noqa: N812
        model_input = T.ModelInput.from_ints(full_tokens)
        logprobs_full = await asyncio.to_thread(
            lambda: sampler.compute_logprobs(model_input).result(),
        )

        raw = logprobs_full[prompt_len : prompt_len + completion_len]
        logprobs = [lp if lp is not None else 0.0 for lp in raw]

        token_strings = [tokenizer.decode([t]) for t in completion_tokens]
        logprob_sum = sum(logprobs)

        return {
            "logprobs": logprobs,
            "tokens": token_strings,
            "prompt_tokens": prompt_len,
            "completion_tokens": completion_len,
            "logprob_sum": logprob_sum,
        }


# ---------------------------------------------------------------------------
# Local-mode catch-all reverse proxy (forwards /health, etc. to vLLM)
# ---------------------------------------------------------------------------

if _mode() == "local":
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def _vllm_catch_all(request: Request, path: str) -> Response:
        """Forward unhandled requests to upstream vLLM."""
        client = _get_vllm_client()
        url = f"{_vllm_backend_url()}/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"

        headers = dict(request.headers)
        # Remove host header so upstream gets its own
        headers.pop("host", None)
        # Inject auth if configured
        api_key = _vllm_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body = await request.body()

        resp = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body if body else None,
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
            media_type=resp.headers.get("content-type"),
        )


# ---------------------------------------------------------------------------
# SSE streaming helpers
# ---------------------------------------------------------------------------

def _stream_chat_response(
    completion_id: str,
    created: int,
    model: str,
    content: str,
) -> StreamingResponse:
    """Wrap a complete response as an SSE stream for OpenAI-compatible clients."""

    def _generate() -> Iterator[str]:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


_FINAL_CHANNEL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)",
    re.DOTALL,
)
_ANALYSIS_CHANNEL_RE = re.compile(
    r"<\|channel\|>analysis<\|message\|>",
)


def _extract_final_channel(text: str) -> str:
    """Extract the ``final`` channel content from GPT-OSS style output.

    GPT-OSS generates ``<|channel|>analysis<|message|>...<|end|>
    <|start|>assistant<|channel|>final<|message|>...``.  Only the *final*
    channel should be shown to the user.  If the model ran out of tokens
    before producing a ``final`` channel, return an empty string rather
    than leaking the raw analysis text.
    """
    m = _FINAL_CHANNEL_RE.search(text)
    if m:
        return m.group(1).strip()
    # If the text contains an analysis channel but no final channel,
    # the model ran out of tokens mid-reasoning — return empty.
    if _ANALYSIS_CHANNEL_RE.search(text):
        return ""
    return text


def _coerce_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                    continue
                if isinstance(part.get("content"), str):
                    parts.append(part["content"])
                    continue
        return "\n".join(parts)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
    return str(content)


def _bounded_int(value: int | None, *, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    return max(minimum, min(maximum, int(value)))


def _bounded_float(
    value: float | None,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    if value is None:
        return default
    return max(minimum, min(maximum, float(value)))


def _stream_completion_response(
    completion_id: str,
    created: int,
    model: str,
    text: str,
) -> StreamingResponse:
    """Wrap a complete text-completion response as an SSE stream."""

    def _generate() -> Iterator[str]:
        chunk = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        final = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")
