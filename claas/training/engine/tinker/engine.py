"""Tinker SDK training engine implementation.

Uses the native Tinker Python SDK (``tinker``, ``tinker_cookbook``) for all
LoRA lifecycle operations and SDPO-style distillation.  No fake HTTP
endpoints — every call goes through the Tinker gRPC/REST client.

Reference implementation:
  https://github.com/sdan/continualcode/blob/master/continualcode/train.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import httpx
import tinker
import torch
from tinker import types as T
from tinker.types.tensor_data import TensorData

from claas.core.config import TinkerConfig, get_config
from claas.core.types import (
    DistillBatchItem,
    DistillBatchRequestPayload,
    DistillResponse,
    LoraDeleteResponse,
    LoraExistsPayload,
    LoraExportPayload,
    LoraInitRequest,
    LoraInitResponse,
    LoraListResponse,
    LoraRuntimeRef,
    ServiceHealth,
)
from claas.training.engine.base import TrainingEngine
from claas.training.engine.tinker.state import (
    LoraEntry,
    all_checkpoint_paths,
    delete_entry,
    get_entry,
    list_loras as state_list_loras,
    lora_exists as state_lora_exists,
    set_tinker_path,
)
from claas.training.teacher_helpers import build_teacher_messages, teacher_messages_to_chat_template

logger = logging.getLogger(__name__)

# Adaptive KL scaling defaults (from continualcode reference).
_TARGET_ADV_ABS_MEAN = 0.03
_MAX_KL_GAIN = 4.0


class TinkerTrainingEngine(TrainingEngine):
    """Executes training and LoRA management through the Tinker Python SDK."""

    def __init__(self) -> None:
        cfg = get_config()
        api_key = cfg.tinker_api_key if isinstance(cfg, TinkerConfig) else ""
        if api_key:
            os.environ["TINKER_API_KEY"] = api_key
        self._base_model = cfg.tinker_base_model if isinstance(cfg, TinkerConfig) else "gpt-oss/GPT-OSS-120B"
        self._service: tinker.ServiceClient | None = None

    @property
    def service(self) -> tinker.ServiceClient:
        if self._service is None:
            self._service = tinker.ServiceClient()
        return self._service

    # ------------------------------------------------------------------
    # LoRA lifecycle
    # ------------------------------------------------------------------

    async def init_lora(self, request: LoraInitRequest) -> LoraInitResponse:
        base_model = request.base_model or self._base_model
        rank = request.lora_r

        tc = await self.service.create_lora_training_client_async(
            base_model=base_model,
            rank=rank,
        )
        save_resp = await _await_api_future(await tc.save_state_async("init"))
        set_tinker_path(
            lora_id=request.lora_id,
            tinker_path=save_resp.path,
            base_model=base_model,
            rank=rank,
            step=0,
        )
        return LoraInitResponse(lora_id=request.lora_id)

    async def list_loras(self, prefix: str) -> LoraListResponse:
        loras = state_list_loras(prefix)
        return LoraListResponse(loras=loras)

    async def delete_lora(self, lora_id: str) -> LoraDeleteResponse:
        """Delete a LoRA adapter and its Tinker checkpoints.

        Removes all tracked checkpoint paths from the Tinker service,
        then deletes the local state entry.  Returns ``deleted=False``
        if the LoRA was not found (idempotent).
        """
        entry = get_entry(lora_id)
        if entry is None:
            return LoraDeleteResponse(deleted=False)
        rest = self.service.create_rest_client()
        for ckpt_path in all_checkpoint_paths(entry):
            try:
                await rest.delete_checkpoint_from_tinker_path_async(ckpt_path)
            except Exception:
                logger.warning("Failed to delete checkpoint %s", ckpt_path, exc_info=True)
        delete_entry(lora_id)
        return LoraDeleteResponse(deleted=True)

    async def lora_exists(self, lora_id: str) -> LoraExistsPayload:
        return LoraExistsPayload(exists=state_lora_exists(lora_id))

    async def export_lora(self, lora_id: str) -> LoraExportPayload:
        entry = _require_entry(lora_id)

        def _export() -> LoraExportPayload:
            rest = self.service.create_rest_client()
            archive_resp = rest.get_checkpoint_archive_url(entry.tinker_path)
            url = archive_resp.result()
            resp = httpx.get(str(url), follow_redirects=True, timeout=120)
            resp.raise_for_status()
            filename = f"{quote(lora_id, safe='')}.zip"
            return LoraExportPayload(filename=filename, content=resp.content)

        return await asyncio.to_thread(_export)

    async def lora_runtime_ref(self, lora_id: str) -> LoraRuntimeRef:
        raise ValueError(
            f"tinker backend does not expose local runtime LoRA paths for vLLM reload (lora_id={lora_id})"
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health(self) -> ServiceHealth:
        try:
            await self.service.get_server_capabilities_async()
        except (ConnectionError, TimeoutError, OSError) as exc:
            return ServiceHealth(status="unhealthy", error=str(exc))
        return ServiceHealth(status="healthy", error=None)

    # ------------------------------------------------------------------
    # Distillation
    # ------------------------------------------------------------------

    async def distill(
        self,
        payload: DistillBatchRequestPayload,
    ) -> DistillResponse:
        """Run one SDPO distillation step entirely through the Tinker SDK.

        Supports batched samples: all samples are processed concurrently,
        then a single forward_backward + optim_step is applied.

        Per-sample flow:
        1. Tokenize prompt + response directly (no chat wrapping)
        2. Compute student (rollout) logprobs via SamplingClient
        3. Build teacher prompt using build_teacher_messages (feedback reprompt)
        4. Compute teacher logprobs (base model conditioned on feedback)
        5. Derive advantages with adaptive KL scaling
        6. Build Datum with right-shifted alignment

        Batch flow:
        7. forward_backward with importance_sampling loss (all datums)
        8. optim_step with AdamW
        9. Save checkpoint and update state
        """
        entry = _require_entry(payload.lora_id)

        base_model = entry.base_model
        lr = payload.training.learning_rate
        kl_coef = payload.training.alpha

        # ── Phase 1: Setup (once per batch) ──
        training_client = await self.service.create_training_client_from_state_async(
            entry.tinker_path
        )
        tokenizer = training_client.get_tokenizer()

        teacher_sampling = await self.service.create_sampling_client_async(
            base_model=base_model
        )

        # ── Phase 2: Per-sample processing (concurrent) ──
        tasks = [
            _build_sample_datum(
                sample, tokenizer, teacher_sampling, kl_coef
            )
            for sample in payload.samples
        ]
        results = await asyncio.gather(*tasks)
        datums = [r[0] for r in results]
        sample_metrics = [r[1] for r in results]

        # ── Phase 3: Training step (once per batch) ──
        fwd_bwd = await training_client.forward_backward_async(
            datums, "importance_sampling"
        )
        await training_client.optim_step_async(
            T.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95)
        )

        # ── Phase 4: Save & return (once per batch) ──
        new_step = entry.step + 1
        checkpoint_name = f"step-{new_step}"
        save_result = await _await_api_future(await training_client.save_state_async(checkpoint_name))

        sampler_save = await _await_api_future(
            await training_client.save_weights_for_sampler_async(checkpoint_name)
        )
        sampler_weights_path = sampler_save.path

        set_tinker_path(
            lora_id=payload.lora_id,
            tinker_path=save_result.path,
            base_model=base_model,
            rank=entry.rank,
            step=new_step,
            sampler_weights_path=sampler_weights_path,
        )

        # Aggregate metrics across samples.
        n = len(sample_metrics)
        total_completion_len = sum(m["completion_len"] for m in sample_metrics)
        avg = lambda key: sum(m[key] for m in sample_metrics) / n  # noqa: E731

        metadata: dict[str, object] = {
            "step": new_step,
            "tinker_path": save_result.path,
            "sampler_weights_path": sampler_weights_path,
            "batch_size": n,
            "completion_len": total_completion_len,
            "effective_kl_coef": avg("effective_kl_coef"),
            "kl_gain": avg("kl_gain"),
            "adv_mean": avg("adv_mean"),
            "adv_abs_mean": avg("adv_abs_mean"),
            "kl_mean": avg("kl_mean"),
            "adv_abs_mean_raw": avg("adv_abs_mean_raw"),
            "lr": lr,
            "loss_fn": "importance_sampling",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if hasattr(fwd_bwd, "metrics") and fwd_bwd.metrics:
            metadata["tinker_fwd_metrics"] = fwd_bwd.metrics

        return DistillResponse(lora_id=payload.lora_id, metadata=metadata)


async def _build_sample_datum(
    sample: DistillBatchItem,
    tokenizer: Any,
    teacher_sampling: Any,
    kl_coef: float,
) -> tuple[T.Datum, dict[str, float]]:
    """Process a single sample into a Datum and per-sample metrics.

    This is the per-sample logic extracted from ``distill()`` so that
    multiple samples can be processed concurrently via ``asyncio.gather``.
    """
    # ── Tokenize prompt + response directly (matching local worker) ──
    # Prefer pre-tokenized IDs when available to avoid decode/re-encode mismatch.
    if sample.prompt_token_ids is not None:
        prompt_tokens = list(sample.prompt_token_ids)
    else:
        prompt_tokens = tokenizer.encode(sample.prompt, add_special_tokens=True)
    if sample.response_token_ids is not None:
        response_tokens = list(sample.response_token_ids)
    else:
        response_tokens = tokenizer.encode(sample.response, add_special_tokens=False)
    full_tokens = prompt_tokens + response_tokens
    prompt_len = len(prompt_tokens)
    completion_len = len(response_tokens)

    # ── Validate and use provided rollout logprobs ──
    if len(sample.rollout_logprobs) != completion_len:
        raise ValueError(
            f"rollout_logprobs length ({len(sample.rollout_logprobs)}) != "
            f"completion_len ({completion_len})"
        )
    student_logprobs = list(sample.rollout_logprobs)

    # ── Build teacher prompt (matching local worker: build_teacher_messages) ──
    # Use clean user_prompt (without chat template decoration) when available.
    teacher_prompt_source = sample.user_prompt or sample.prompt
    teacher_messages = build_teacher_messages(teacher_prompt_source, sample.feedback)
    template_messages = teacher_messages_to_chat_template(teacher_messages)
    teacher_prompt_text = tokenizer.apply_chat_template(
        template_messages,
        add_generation_prompt=False,
        tokenize=False,
    )
    teacher_prompt_tokens: list[int] = tokenizer.encode(
        teacher_prompt_text,
        add_special_tokens=False,
    )
    teacher_full_tokens = teacher_prompt_tokens + response_tokens
    teacher_prompt_len = len(teacher_prompt_tokens)
    teacher_full = T.ModelInput.from_ints(teacher_full_tokens)

    # ── Compute teacher logprobs (base model = self-distillation) ──
    teacher_logprobs_full = await teacher_sampling.compute_logprobs_async(teacher_full)
    teacher_logprobs = _slice_completion_logprobs(
        teacher_logprobs_full, teacher_prompt_len, completion_len
    )

    # ── Compute advantages with adaptive KL scaling ──
    raw_kl_deltas = [t - s for s, t in zip(student_logprobs, teacher_logprobs, strict=True)]
    adv_abs_mean_raw = sum(abs(d) for d in raw_kl_deltas) / max(len(raw_kl_deltas), 1)

    gain = 1.0
    if adv_abs_mean_raw > 0:
        gain = min(max(_TARGET_ADV_ABS_MEAN / adv_abs_mean_raw, 1.0), _MAX_KL_GAIN)
    effective_kl_coef = kl_coef * gain

    advantages = [
        effective_kl_coef * (t - s)
        for s, t in zip(student_logprobs, teacher_logprobs, strict=True)
    ]

    # ── Build Datum with right-shifted alignment ──
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    input_model_input = T.ModelInput.from_ints(input_tokens)

    full_logprobs = [0.0] * prompt_len + student_logprobs
    full_advantages = [0.0] * prompt_len + advantages
    shifted_logprobs = full_logprobs[1:]
    shifted_advantages = full_advantages[1:]

    datum = T.Datum(
        model_input=input_model_input,
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(
                torch.tensor(target_tokens, dtype=torch.int64)
            ),
            "logprobs": TensorData.from_torch(
                torch.tensor(shifted_logprobs, dtype=torch.float32)
            ),
            "advantages": TensorData.from_torch(
                torch.tensor(shifted_advantages, dtype=torch.float32)
            ),
        },
    )

    adv_mean = sum(advantages) / max(len(advantages), 1)
    adv_abs_mean = sum(abs(a) for a in advantages) / max(len(advantages), 1)
    kl_mean = sum(raw_kl_deltas) / max(len(raw_kl_deltas), 1)

    metrics = {
        "completion_len": completion_len,
        "effective_kl_coef": effective_kl_coef,
        "kl_gain": gain,
        "adv_mean": adv_mean,
        "adv_abs_mean": adv_abs_mean,
        "kl_mean": kl_mean,
        "adv_abs_mean_raw": adv_abs_mean_raw,
    }

    return datum, metrics


def _require_entry(lora_id: str) -> LoraEntry:
    """Return the state entry for *lora_id* or raise ``FileNotFoundError``."""
    entry = get_entry(lora_id)
    if entry is None:
        raise FileNotFoundError(f"LoRA not found in Tinker state: {lora_id}")
    return entry


def _slice_completion_logprobs(
    logprobs_full: list[float | None],
    prompt_len: int,
    completion_len: int,
) -> list[float]:
    """Extract completion-only logprobs from a full-sequence logprobs list.

    Positions inside the prompt are discarded.  ``None`` entries (e.g. the
    very first token which has no conditioning context) are replaced with 0.0.
    """
    raw = logprobs_full[prompt_len : prompt_len + completion_len]
    return [lp if lp is not None else 0.0 for lp in raw]


async def _await_api_future(obj):
    """Resolve Tinker future-like wrappers into concrete results."""
    result_async = getattr(obj, "result_async", None)
    if callable(result_async):
        return await result_async()
    result = getattr(obj, "result", None)
    if callable(result):
        return result()
    return obj
