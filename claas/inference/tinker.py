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

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from claas.core.config import TinkerConfig
from claas.core.types import ScoreResponse

from .base import CompletionResult, InferenceBackend, TextCompletionResult
from .helpers import apply_chat_template_ids, bounded_float, bounded_int, coerce_content

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
else:
    PreTrainedTokenizerBase = Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy singleton for the Tinker sampling client
# ---------------------------------------------------------------------------


def _load_stop_token_ids(tokenizer: PreTrainedTokenizerBase) -> set[int]:
    """Load stop token IDs from the model's GenerationConfig."""
    from transformers import GenerationConfig

    gen_config = GenerationConfig.from_pretrained(tokenizer.name_or_path)
    eos = gen_config.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    return set(eos)


class SamplerHolder:
    """Holds a lazily-initialized Tinker SamplingClient and tokenizer."""

    def __init__(self, cfg: TinkerConfig) -> None:
        self._cfg = cfg
        self._service: Any | None = None
        self._sampler: Any | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._stop_token_ids: set[int] = set()
        self._model_path: str | None = None
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        with self._lock:
            if self._sampler is not None and self._tokenizer is not None:
                return
            import tinker

            api_key = os.environ.get("CLAAS_TINKER_API_KEY", "")
            if api_key:
                os.environ["TINKER_API_KEY"] = api_key
            self._service = tinker.ServiceClient()
            self._sampler = self._service.create_sampling_client(
                base_model=self._cfg.tinker_base_model,
            )
            self._tokenizer = self._sampler.get_tokenizer()
            self._stop_token_ids = _load_stop_token_ids(self._tokenizer)

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
    def stop_token_ids(self) -> set[int]:
        self._ensure()
        return self._stop_token_ids

    def refresh(self, model_path: str | None = None) -> None:
        """Refresh the sampling client (e.g. after a distillation step)."""
        import tinker

        with self._lock:
            if self._service is None:
                api_key = os.environ.get("CLAAS_TINKER_API_KEY", "")
                if api_key:
                    os.environ["TINKER_API_KEY"] = api_key
                self._service = tinker.ServiceClient()
            if model_path:
                self._sampler = self._service.create_sampling_client(model_path=model_path)
            else:
                self._sampler = self._service.create_sampling_client(
                    base_model=self._cfg.tinker_base_model,
                )
            self._tokenizer = self._sampler.get_tokenizer()
            self._stop_token_ids = _load_stop_token_ids(self._tokenizer)
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


# ---------------------------------------------------------------------------
# TinkerBackend
# ---------------------------------------------------------------------------


class TinkerBackend(InferenceBackend):
    """Inference backend backed by the Tinker SDK."""

    def __init__(self, cfg: TinkerConfig) -> None:
        self._cfg = cfg
        self._holder = SamplerHolder(cfg=cfg)

    @property
    def holder(self) -> SamplerHolder:
        return self._holder

    def _base_model(self) -> str:
        return self._cfg.tinker_base_model

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        logprobs: bool = False,
        top_logprobs: int = 1,
    ) -> CompletionResult:
        import tinker.types as T  # noqa: N812

        if not model:
            model = self._base_model()
        tokenizer = self._holder.tokenizer
        sampler = self._holder.sampler
        stop_token_ids = self._holder.stop_token_ids

        dicts = [
            {"role": m["role"], "content": coerce_content(m.get("content", ""))}
            for m in messages
        ]
        prompt_token_ids = apply_chat_template_ids(
            tokenizer, dicts, add_generation_prompt=True,
        )
        model_input = T.ModelInput.from_ints(prompt_token_ids)

        stop_strs = stop if stop else [
            tokenizer.decode([tid]) for tid in stop_token_ids
        ]
        max_tok = bounded_int(max_tokens, default=2048, minimum=1, maximum=32768)
        temp = bounded_float(temperature, default=0.7, minimum=0.0, maximum=2.0)
        tp = bounded_float(top_p, default=1.0, minimum=0.0, maximum=1.0)
        sampling_params = T.SamplingParams(
            max_tokens=max_tok,
            temperature=temp,
            top_p=tp,
            top_k=0,
            seed=0,
            stop=stop_strs,
        )

        resp = await _sample_async(sampler, model_input, sampling_params)

        seq = resp.sequences[0]
        response_token_ids = list(seq.tokens)
        if seq.logprobs is None:
            raise RuntimeError("Tinker sampler returned no logprobs — distillation requires logprobs")
        response_logprobs = list(seq.logprobs)

        # Strip the stop token (and its logprob) if the sampler included it
        if response_token_ids and response_token_ids[-1] in stop_token_ids:
            response_token_ids = response_token_ids[:-1]
            response_logprobs = response_logprobs[:-1]

        content = tokenizer.decode(response_token_ids, skip_special_tokens=False)

        prompt_len = len(prompt_token_ids)

        raw_completion_text = tokenizer.decode(response_token_ids, skip_special_tokens=False)
        prompt_text = tokenizer.decode(prompt_token_ids, skip_special_tokens=False)

        return CompletionResult(
            content=content,
            raw_prompt=prompt_text,
            raw_response=raw_completion_text,
            response_token_ids=response_token_ids,
            prompt_token_ids=list(prompt_token_ids),
            response_logprobs=response_logprobs,
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

    async def score(
        self,
        *,
        model: str,  # noqa: ARG002 — Tinker uses its current sampler implicitly
        messages: list[dict[str, str]],
        completion: str,
    ) -> ScoreResponse:
        import tinker.types as T  # noqa: N812

        tokenizer = self._holder.tokenizer
        sampler = self._holder.sampler

        dicts = [
            {"role": m["role"], "content": coerce_content(m.get("content", ""))}
            for m in messages
        ]

        prompt_token_ids = apply_chat_template_ids(
            tokenizer, dicts, add_generation_prompt=True,
        )

        full_messages = dicts + [{"role": "assistant", "content": completion}]
        full_token_ids = apply_chat_template_ids(
            tokenizer, full_messages, add_generation_prompt=False,
        )

        completion_token_ids = full_token_ids[len(prompt_token_ids):]

        if len(completion_token_ids) == 0:
            return ScoreResponse(
                logprobs=[],
                tokens=[],
                prompt_tokens=len(prompt_token_ids),
                completion_tokens=0,
                logprob_sum=0.0,
            )

        model_input = T.ModelInput.from_ints(full_token_ids)
        logprobs_full = await asyncio.to_thread(
            lambda: sampler.compute_logprobs(model_input).result()
        )
        prompt_len = len(prompt_token_ids)
        completion_len = len(completion_token_ids)
        completion_logprobs = [
            lp if lp is not None else 0.0
            for lp in logprobs_full[prompt_len : prompt_len + completion_len]
        ]

        completion_tokens_str = [
            tokenizer.decode([tid]) for tid in completion_token_ids
        ]

        return ScoreResponse(
            logprobs=completion_logprobs,
            tokens=completion_tokens_str,
            prompt_tokens=len(prompt_token_ids),
            completion_tokens=completion_len,
            logprob_sum=sum(completion_logprobs),
        )

    def refresh_sampler(self, model_path: str) -> None:
        """Refresh the sampling client to a new checkpoint."""
        self._holder.refresh(model_path=model_path)
        logger.info("Tinker sampler refreshed to checkpoint: %s", model_path)

    def register_routes(self, app: FastAPI) -> None:
        """Register Tinker-specific endpoints: refresh, status."""
        def _current_backend() -> TinkerBackend:
            backend = getattr(app.state, "inference_backend", None)
            if isinstance(backend, TinkerBackend):
                return backend
            return self

        def _current_holder() -> SamplerHolder:
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
