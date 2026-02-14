"""OpenAI-compatible inference proxy backed by Tinker's SamplingClient.

Exposes ``/v1/chat/completions`` and ``/v1/completions`` so that any
OpenAI-compatible client (e.g. OpenClaw) can talk to a Tinker-hosted model
without a local GPU.

Usage::

    TINKER_API_KEY=... CLAAS_TINKER_BASE_MODEL=Qwen/Qwen3-235B-A22B \
        uvicorn claas.tinker_inference_proxy:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Generator

import tinker
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from tinker import types as T
from tinker_cookbook import model_info
from tinker_cookbook.renderers import Renderer, get_renderer
from transformers import PreTrainedTokenizerBase

app = FastAPI(title="CLaaS Tinker Inference Proxy")

_BASE_MODEL = os.environ.get("CLAAS_TINKER_BASE_MODEL", "Qwen/Qwen3-235B-A22B")


# ---------------------------------------------------------------------------
# Lazy singleton for the sampling client
# ---------------------------------------------------------------------------

class _SamplerHolder:
    """Holds a lazily-initialized Tinker SamplingClient and tokenizer."""

    def __init__(self) -> None:
        self._service: tinker.ServiceClient | None = None
        self._sampler: tinker.SamplingClient | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._renderer: Renderer | None = None
        self._model_path: str | None = None

    def _ensure(self) -> None:
        if self._sampler is not None:
            return
        api_key = os.environ.get("CLAAS_TINKER_API_KEY", "")
        if api_key:
            os.environ["TINKER_API_KEY"] = api_key
        self._service = tinker.ServiceClient()
        self._sampler = self._service.create_sampling_client(base_model=_BASE_MODEL)
        self._tokenizer = self._sampler.get_tokenizer()

        renderer_name = model_info.get_recommended_renderer_name(_BASE_MODEL)
        self._renderer = get_renderer(renderer_name, tokenizer=self._tokenizer)

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
    def renderer(self) -> Renderer:
        self._ensure()
        assert self._renderer is not None
        return self._renderer

    def refresh(self, model_path: str | None = None) -> None:
        """Refresh the sampling client (e.g. after a distillation step)."""
        if self._service is None:
            self._ensure()
            return
        if model_path:
            self._sampler = self._service.create_sampling_client(model_path=model_path)
        else:
            self._sampler = self._service.create_sampling_client(base_model=_BASE_MODEL)
        self._model_path = model_path


_holder = _SamplerHolder()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = _BASE_MODEL
    messages: list[ChatMessage]
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: list[str] | None = None


class CompletionRequest(BaseModel):
    model: str = _BASE_MODEL
    prompt: str
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: list[str] | None = None


class RefreshRequest(BaseModel):
    model_path: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> dict[str, object] | StreamingResponse:
    renderer = _holder.renderer
    sampler = _holder.sampler

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    model_input = renderer.build_generation_prompt(messages)

    stop_seqs = req.stop or renderer.get_stop_sequences()
    sampling_params = T.SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=0,
        seed=0,
        stop=stop_seqs,
    )

    resp = sampler.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    ).result()

    seq = resp.sequences[0]
    text_msg, _ = renderer.parse_response(seq.tokens)
    content = text_msg.get("content", "") if isinstance(text_msg, dict) else str(text_msg)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    prompt_tokens = model_input.length()
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


@app.post("/v1/completions")
async def completions(req: CompletionRequest) -> dict[str, object] | StreamingResponse:
    tokenizer = _holder.tokenizer
    sampler = _holder.sampler

    tokens: list[int] = tokenizer.encode(req.prompt)
    model_input = T.ModelInput.from_ints(tokens)

    stop_seqs = req.stop or []
    sampling_params = T.SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=0,
        seed=0,
        stop=stop_seqs,
    )

    resp = sampler.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    ).result()

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


@app.get("/v1/models")
async def list_models() -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {
                "id": _BASE_MODEL,
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

    def _generate() -> Generator[str]:
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


def _stream_completion_response(
    completion_id: str,
    created: int,
    model: str,
    text: str,
) -> StreamingResponse:
    """Wrap a complete text-completion response as an SSE stream."""

    def _generate() -> Generator[str]:
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
