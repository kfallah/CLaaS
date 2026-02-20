"""Tinker SDK inference backend.

Uses the Tinker sampling API for chat and text completions.
Only imported when ``CLAAS_DISTILL_EXECUTION_MODE=tinker``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from claas.core.config import get_config

from .base import CompletionResult, InferenceBackend, TextCompletionResult
from .helpers import bounded_float, bounded_int, coerce_content

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
else:
    PreTrainedTokenizerBase = Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fallback renderer using the tokenizer's built-in chat template
# ---------------------------------------------------------------------------


def _apply_chat_template_ids(tokenizer: Any, dicts: list[dict[str, str]]) -> list[int]:
    """Tokenize messages via apply_chat_template, returning token ids."""
    result = tokenizer.apply_chat_template(
        dicts, add_generation_prompt=True, tokenize=True,
    )
    if isinstance(result, list):
        return result
    return result["input_ids"]


class _TokenizerChatRenderer:
    """Minimal renderer that delegates to ``tokenizer.apply_chat_template``."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    def build_generation_prompt(self, messages: list[dict[str, str]]) -> Any:
        dicts = [{"role": m["role"], "content": m.get("content", "")} for m in messages]
        token_ids = _apply_chat_template_ids(self._tokenizer, dicts)
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


# ---------------------------------------------------------------------------
# Lazy singleton for the Tinker sampling client
# ---------------------------------------------------------------------------


class _SamplerHolder:
    """Holds a lazily-initialized Tinker SamplingClient and tokenizer."""

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

            cfg = get_config()
            api_key = getattr(cfg, "tinker_api_key", "")
            if api_key:
                os.environ["TINKER_API_KEY"] = api_key
            base_model = getattr(cfg, "tinker_base_model", "")
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
            cfg = get_config()
            base_model = getattr(cfg, "tinker_base_model", "")
            if self._service is None:
                api_key = getattr(cfg, "tinker_api_key", "")
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
# Pydantic models for Tinker-specific endpoints
# ---------------------------------------------------------------------------


class RefreshRequest(BaseModel):
    model_path: str | None = None


class ScoreRequest(BaseModel):
    prompt: str
    completion: str


# ---------------------------------------------------------------------------
# TinkerBackend
# ---------------------------------------------------------------------------


class TinkerBackend(InferenceBackend):
    """Inference backend backed by the Tinker SDK."""

    def __init__(self) -> None:
        self._holder = _SamplerHolder()

    @property
    def holder(self) -> _SamplerHolder:
        return self._holder

    def _base_model(self) -> str:
        return getattr(get_config(), "tinker_base_model", "")

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
    ) -> CompletionResult:
        import tinker.types as T  # noqa: N812
        from tinker_cookbook.renderers import Message

        if not model:
            model = self._base_model()
        renderer = self._holder.renderer
        sampler = self._holder.sampler

        msgs: list[Message] = [
            Message(role=m["role"], content=coerce_content(m.get("content", "")))
            for m in messages
        ]
        model_input = renderer.build_generation_prompt(msgs)

        stop_seqs = stop or renderer.get_stop_sequences()
        max_tok = bounded_int(max_tokens, default=2048, minimum=1, maximum=32768)
        temp = bounded_float(temperature, default=0.7, minimum=0.0, maximum=2.0)
        tp = bounded_float(top_p, default=1.0, minimum=0.0, maximum=1.0)
        sampling_params = T.SamplingParams(
            max_tokens=max_tok,
            temperature=temp,
            top_p=tp,
            top_k=0,
            seed=0,
            stop=stop_seqs,
        )

        resp = await _sample_async(sampler, model_input, sampling_params)

        seq = resp.sequences[0]
        text_msg, _ = renderer.parse_response(seq.tokens)
        content = text_msg.get("content", "") if isinstance(text_msg, dict) else str(text_msg)

        tokenizer = self._holder.tokenizer
        raw_completion_text = tokenizer.decode(seq.tokens, skip_special_tokens=False)
        prompt_text = tokenizer.decode(model_input.to_ints(), skip_special_tokens=False)

        return CompletionResult(
            content=content,
            raw_prompt=prompt_text,
            raw_response=raw_completion_text,
            token_ids=list(seq.tokens),
            logprobs=list(seq.logprobs) if seq.logprobs is not None else None,
            prompt_tokens=model_input.length,
            completion_tokens=len(seq.tokens),
        )

    async def text_completion(
        self,
        *,
        prompt: str,
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
    ) -> TextCompletionResult:
        import tinker.types as T  # noqa: N812

        if not model:
            model = self._base_model()
        tokenizer = self._holder.tokenizer
        sampler = self._holder.sampler

        tokens: list[int] = tokenizer.encode(prompt)
        model_input = T.ModelInput.from_ints(tokens)

        stop_seqs = stop or []
        max_tok = bounded_int(max_tokens, default=2048, minimum=1, maximum=32768)
        temp = bounded_float(temperature, default=0.7, minimum=0.0, maximum=2.0)
        tp = bounded_float(top_p, default=1.0, minimum=0.0, maximum=1.0)
        sampling_params = T.SamplingParams(
            max_tokens=max_tok,
            temperature=temp,
            top_p=tp,
            top_k=0,
            seed=0,
            stop=stop_seqs,
        )

        resp = await _sample_async(sampler, model_input, sampling_params)

        seq = resp.sequences[0]
        text: str = tokenizer.decode(seq.tokens, skip_special_tokens=True)

        return TextCompletionResult(
            text=text,
            prompt_tokens=len(tokens),
            completion_tokens=len(seq.tokens),
        )

    async def list_models(self) -> dict[str, object] | Response:
        return {
            "object": "list",
            "data": [
                {
                    "id": self._base_model(),
                    "object": "model",
                    "owned_by": "tinker",
                }
            ],
        }

    def register_routes(self, app: FastAPI) -> None:
        """Register Tinker-specific endpoints: refresh, status, score."""
        holder = self._holder

        @app.post("/v1/sampler/refresh")
        async def refresh_sampler(body: RefreshRequest) -> dict[str, object]:
            """Refresh the sampling client, optionally pointing at a new checkpoint."""
            holder.refresh(model_path=body.model_path)
            return {"status": "ok", "model_path": body.model_path}

        @app.get("/v1/sampler/status")
        async def sampler_status() -> dict[str, object]:
            """Return the currently loaded model path (null = base model only)."""
            return {"model_path": holder._model_path, "base_model": self._base_model()}

        @app.post("/v1/score")
        async def score_completion(req: ScoreRequest) -> dict[str, object]:
            """Score a prompt+completion pair and return per-token completion logprobs."""
            tokenizer = holder.tokenizer
            sampler = holder.sampler

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
