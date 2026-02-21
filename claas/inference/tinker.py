"""Tinker SDK inference backend.

Uses the Tinker sampling API for chat and text completions.
Only imported when running in tinker mode.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from claas.core.config import CoreConfig

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


def _coerce_template_ids(result: Any) -> list[int]:
    """Normalize ``tokenizer.apply_chat_template`` output to token IDs."""
    if isinstance(result, list):
        return [int(tok) for tok in result]
    if isinstance(result, dict):
        maybe_ids = result.get("input_ids")
        if isinstance(maybe_ids, list):
            return [int(tok) for tok in maybe_ids]
    if hasattr(result, "tolist"):
        maybe_ids = result.tolist()
        if isinstance(maybe_ids, list):
            return [int(tok) for tok in maybe_ids]
    raise TypeError("Unsupported apply_chat_template result shape")


def _apply_chat_template_ids(
    tokenizer: Any,
    dicts: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    """Tokenize messages via apply_chat_template, returning token ids."""
    result = tokenizer.apply_chat_template(
        dicts,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
    )
    return _coerce_template_ids(result)


class _TokenizerChatRenderer:
    """Minimal renderer that delegates to ``tokenizer.apply_chat_template``."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    def build_generation_prompt(self, messages: list[dict[str, str]]) -> Any:
        dicts = [{"role": m["role"], "content": m.get("content", "")} for m in messages]
        token_ids = _apply_chat_template_ids(
            self._tokenizer, dicts, add_generation_prompt=True,
        )
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
        return get_renderer(renderer_name, tokenizer=tokenizer)  # type: ignore[arg-type]
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

    def __init__(self, cfg: CoreConfig | None = None) -> None:
        self._cfg = cfg
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

            api_key = os.environ.get("CLAAS_TINKER_API_KEY", "")
            if api_key:
                os.environ["TINKER_API_KEY"] = api_key
            base_model = getattr(self._cfg, "tinker_base_model", "") if self._cfg else ""
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
            base_model = getattr(self._cfg, "tinker_base_model", "") if self._cfg else ""
            if self._service is None:
                api_key = os.environ.get("CLAAS_TINKER_API_KEY", "")
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
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    completion: str


# ---------------------------------------------------------------------------
# TinkerBackend
# ---------------------------------------------------------------------------


class TinkerBackend(InferenceBackend):
    """Inference backend backed by the Tinker SDK."""

    def __init__(self, cfg: CoreConfig | None = None) -> None:
        self._cfg = cfg
        self._holder = _SamplerHolder(cfg=cfg)

    @property
    def holder(self) -> _SamplerHolder:
        return self._holder

    def _base_model(self) -> str:
        return getattr(self._cfg, "tinker_base_model", "") if self._cfg else ""

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
        prompt_token_ids = list(model_input.to_ints())
        prompt_len = model_input.length

        # seq.tokens contains only the generated completion tokens
        # (the Tinker sampler does not echo the prompt).
        response_token_ids = list(seq.tokens)
        response_logprobs = list(seq.logprobs) if seq.logprobs is not None else None

        raw_completion_text = tokenizer.decode(response_token_ids, skip_special_tokens=False)
        prompt_text = tokenizer.decode(prompt_token_ids, skip_special_tokens=False)

        return CompletionResult(
            content=content,
            raw_prompt=prompt_text,
            raw_response=raw_completion_text,
            token_ids=response_token_ids,
            prompt_token_ids=prompt_token_ids,
            logprobs=response_logprobs,
            prompt_tokens=prompt_len,
            completion_tokens=len(response_token_ids),
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
        def _current_backend() -> TinkerBackend:
            backend = getattr(app.state, "inference_backend", None)
            if isinstance(backend, TinkerBackend):
                return backend
            return self

        def _current_holder() -> _SamplerHolder:
            return _current_backend().holder

        @app.post("/v1/sampler/refresh")
        async def refresh_sampler(body: RefreshRequest) -> dict[str, object]:
            """Refresh the sampling client, optionally pointing at a new checkpoint."""
            holder = _current_holder()
            holder.refresh(model_path=body.model_path)
            return {"status": "ok", "model_path": body.model_path}

        @app.get("/v1/sampler/status")
        async def sampler_status() -> dict[str, object]:
            """Return the currently loaded model path (null = base model only)."""
            backend = _current_backend()
            holder = backend.holder
            return {"model_path": holder._model_path, "base_model": backend._base_model()}

        @app.post("/v1/score")
        async def score_completion(req: ScoreRequest) -> dict[str, object]:
            """Score a prompt/completion pair or chat messages/completion pair."""
            if req.messages is None and req.prompt is None:
                raise HTTPException(
                    status_code=422,
                    detail="either prompt or messages must be provided",
                )

            holder = _current_holder()
            tokenizer = holder.tokenizer
            sampler = holder.sampler

            if req.messages is not None:
                if not req.messages:
                    raise HTTPException(status_code=422, detail="messages must be non-empty")
                prompt_messages = [
                    {
                        "role": str(m.get("role", "user")),
                        "content": coerce_content(m.get("content", "")),
                    }
                    for m in req.messages
                ]
                prompt_tokens = _apply_chat_template_ids(
                    tokenizer,
                    prompt_messages,
                    add_generation_prompt=True,
                )
                full_tokens = _apply_chat_template_ids(
                    tokenizer,
                    prompt_messages + [{"role": "assistant", "content": req.completion}],
                    add_generation_prompt=False,
                )
                completion_tokens = full_tokens[len(prompt_tokens):]
            else:
                assert req.prompt is not None
                prompt_tokens = list(tokenizer.encode(req.prompt, add_special_tokens=True))
                completion_tokens = tokenizer.encode(
                    req.completion, add_special_tokens=False,
                )
                full_tokens = prompt_tokens + completion_tokens

            prompt_len = len(prompt_tokens)
            completion_len = len(completion_tokens)

            if completion_len == 0:
                logprobs = []
            else:
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
