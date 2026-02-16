"""OpenAI-compatible inference proxy backed by Tinker's SamplingClient.

Exposes ``/v1/chat/completions`` and ``/v1/completions`` so that any
OpenAI-compatible client (e.g. OpenClaw) can talk to a Tinker-hosted model
without a local GPU.

Usage::

    TINKER_API_KEY=... CLAAS_TINKER_BASE_MODEL=gpt-oss/GPT-OSS-120B \
        uvicorn claas.proxy.tinker_inference_proxy:app --host 0.0.0.0 --port 8000
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

import tinker
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, model_validator
from tinker import types as T
from tinker_cookbook import model_info
from tinker_cookbook.renderers import Message, Renderer, get_renderer

from claas.core.config import get_proxy_config

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
else:
    PreTrainedTokenizerBase = Any

app = FastAPI(title="CLaaS Tinker Inference Proxy")


def _base_model() -> str:
    return get_proxy_config().tinker_base_model


# ---------------------------------------------------------------------------
# Fallback renderer using the tokenizer's built-in chat template
# ---------------------------------------------------------------------------

class _TokenizerChatRenderer:
    """Minimal renderer that delegates to ``tokenizer.apply_chat_template``.

    Used when *tinker_cookbook* doesn't have a dedicated renderer for the model.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    def build_generation_prompt(self, messages: list[Message]) -> T.ModelInput:
        dicts = [{"role": m["role"], "content": m.get("content", "")} for m in messages]
        result = self._tokenizer.apply_chat_template(
            dicts, add_generation_prompt=True, tokenize=True,
        )
        # apply_chat_template may return a plain list[int] or a BatchEncoding
        if isinstance(result, list):
            token_ids: list[int] = result  # type: ignore[assignment]
        else:
            token_ids = result["input_ids"]  # type: ignore[index]
        return T.ModelInput.from_ints(token_ids)  # type: ignore[arg-type]

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
# Lazy singleton for the sampling client
# ---------------------------------------------------------------------------

def _make_renderer(
    base_model: str, tokenizer: PreTrainedTokenizerBase
) -> Renderer | _TokenizerChatRenderer:
    """Return a tinker_cookbook renderer, falling back to the tokenizer's chat template."""
    try:
        renderer_name = model_info.get_recommended_renderer_name(base_model)
        return get_renderer(renderer_name, tokenizer=tokenizer)  # type: ignore[arg-type]
    except (ValueError, KeyError):
        logger.warning(
            "No tinker_cookbook renderer for %s; falling back to tokenizer chat template",
            base_model,
        )
        return _TokenizerChatRenderer(tokenizer)


class _SamplerHolder:
    """Holds a lazily-initialized Tinker SamplingClient and tokenizer."""

    def __init__(self) -> None:
        self._service: tinker.ServiceClient | None = None
        self._sampler: tinker.SamplingClient | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._renderer: Renderer | _TokenizerChatRenderer | None = None
        self._model_path: str | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        with self._lock:
            if self._sampler is not None:
                return
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
    def sampler(self) -> tinker.SamplingClient:
        self._ensure()
        assert self._sampler is not None
        return self._sampler

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        self._ensure()
        assert self._tokenizer is not None
        return self._tokenizer

    @property
    def renderer(self) -> Renderer | _TokenizerChatRenderer:
        self._ensure()
        assert self._renderer is not None
        return self._renderer

    def refresh(self, model_path: str | None = None) -> None:
        """Refresh the sampling client (e.g. after a distillation step)."""
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
# Raw completion cache
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


async def _sample_async(
    sampler: tinker.SamplingClient,
    prompt: T.ModelInput,
    sampling_params: T.SamplingParams,
) -> T.SampleResponse:
    """Run blocking sampling in a worker thread to avoid blocking the event loop."""
    return await asyncio.to_thread(
        lambda: sampler.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=sampling_params,
        ).result()
    )


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
    prompt: str | None = None
    messages: list[ChatCompletionMessage] | None = None
    completion: str

    @model_validator(mode="before")
    @classmethod
    def _require_prompt_or_messages(cls, values: dict) -> dict:
        prompt = values.get("prompt")
        messages = values.get("messages")
        if not prompt and not messages:
            raise ValueError("At least one of 'prompt' or 'messages' is required")
        return values


class RefreshRequest(BaseModel):
    model_path: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(req: ChatCompletionRequest) -> dict[str, object] | StreamingResponse:
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

    # Cache raw completion for training pipeline retrieval
    tokenizer = _holder.tokenizer
    raw_completion_text = tokenizer.decode(seq.tokens, skip_special_tokens=False)
    prompt_text = tokenizer.decode(model_input.to_ints(), skip_special_tokens=False)
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
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


@app.post("/v1/sampler/refresh")
async def refresh_sampler(body: RefreshRequest) -> dict[str, object]:
    """Refresh the sampling client, optionally pointing at a new checkpoint."""
    _holder.refresh(model_path=body.model_path)
    return {"status": "ok", "model_path": body.model_path}


@app.get("/v1/sampler/status")
async def sampler_status() -> dict[str, object]:
    """Return the currently loaded model path (null = base model only)."""
    return {"model_path": _holder._model_path, "base_model": _base_model()}


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


@app.post("/v1/score")
async def score_completion(req: ScoreRequest) -> dict[str, object]:
    """Score a prompt+completion pair and return per-token completion logprobs."""
    tokenizer = _holder.tokenizer
    sampler = _holder.sampler

    if req.messages is not None:
        dicts = [{"role": m.role, "content": m.content or ""} for m in req.messages]
        result = tokenizer.apply_chat_template(
            dicts, add_generation_prompt=True, tokenize=True,
        )
        if isinstance(result, list):
            prompt_tokens: list[int] = result
        else:
            prompt_tokens = result["input_ids"]
    else:
        assert req.prompt is not None
        prompt_tokens = tokenizer.encode(req.prompt, add_special_tokens=True)
    completion_tokens: list[int] = tokenizer.encode(
        req.completion, add_special_tokens=False,
    )
    full_tokens = prompt_tokens + completion_tokens
    prompt_len = len(prompt_tokens)
    completion_len = len(completion_tokens)

    model_input = T.ModelInput.from_ints(full_tokens)
    logprobs_full = await asyncio.to_thread(
        lambda: sampler.compute_logprobs(model_input).result(),
    )

    # Slice completion logprobs (same pattern as engine.py _slice_completion_logprobs)
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


@app.get("/v1/models")
async def list_models() -> dict[str, object]:
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
