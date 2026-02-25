"""Tinker SDK training engine implementation.

Uses the native Tinker Python SDK (``tinker``) for all LoRA lifecycle
operations and SDPO-style distillation.  No fake HTTP endpoints — every
call goes through the Tinker gRPC/REST client.

Reference implementation:
  https://github.com/sdan/continualcode/blob/master/continualcode/train.py
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, TypedDict
from urllib.parse import quote

import httpx
import tinker
from tinker import types as T
from tinker.types.tensor_data import TensorData

from claas.core.config import TinkerConfig
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

    def __init__(self, cfg: TinkerConfig) -> None:
        api_key = os.environ.get("CLAAS_TINKER_API_KEY", "").strip()
        if api_key:
            os.environ["TINKER_API_KEY"] = api_key
        self._base_model = cfg.tinker_base_model
        self._state_path = cfg.tinker_state_path
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
            path=self._state_path,
        )
        return LoraInitResponse(lora_id=request.lora_id)

    async def list_loras(self, prefix: str) -> LoraListResponse:
        loras = state_list_loras(prefix, path=self._state_path)
        return LoraListResponse(loras=loras)

    async def delete_lora(self, lora_id: str) -> LoraDeleteResponse:
        """Delete a LoRA adapter and its Tinker checkpoints.

        Removes all tracked checkpoint paths from the Tinker service,
        then deletes the local state entry.  Returns ``deleted=False``
        if the LoRA was not found (idempotent).
        """
        entry = get_entry(lora_id, path=self._state_path)
        if entry is None:
            return LoraDeleteResponse(deleted=False)
        rest = self.service.create_rest_client()
        for ckpt_path in all_checkpoint_paths(entry):
            try:
                await rest.delete_checkpoint_from_tinker_path_async(ckpt_path)
            except Exception:
                logger.warning("Failed to delete checkpoint %s", ckpt_path, exc_info=True)
        delete_entry(lora_id, path=self._state_path)
        return LoraDeleteResponse(deleted=True)

    async def lora_exists(self, lora_id: str) -> LoraExistsPayload:
        return LoraExistsPayload(exists=state_lora_exists(lora_id, path=self._state_path))

    async def export_lora(self, lora_id: str) -> LoraExportPayload:
        entry = _require_entry(lora_id, self._state_path)

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

        Supports batched samples with configurable multi-step updates.
        Sample preparation (teacher signals) runs once, then each optimizer
        step rebuilds importance-weighted datums using current behavior logprobs.

        Per-sample flow:
        1. Tokenize prompt + response directly (no chat wrapping)
        2. Read initial student (rollout) logprobs from cached response
        3. Build teacher prompt using build_teacher_messages (feedback reprompt)
        4. Compute teacher logprobs (base model conditioned on feedback)
        5. Derive advantages with adaptive KL scaling
        6. Build Datum with right-shifted alignment

        Batch flow:
        7. Repeat `steps_per_batch` times:
           - forward_backward with importance_sampling loss
           - optim_step with AdamW
           - recompute behavior logprobs for next step (if needed)
        8. Save checkpoint and update state
        """
        entry = _require_entry(payload.lora_id, self._state_path)

        base_model = entry.base_model
        lr = payload.training.learning_rate
        kl_coef = payload.training.alpha
        steps_per_batch = payload.training.steps_per_batch
        feedback_repetitions = payload.training.feedback_repetitions

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
            _prepare_sample_inputs(
                sample=sample,
                tokenizer=tokenizer,
                teacher_sampling=teacher_sampling,
                feedback_repetitions=feedback_repetitions,
            )
            for sample in payload.samples
        ]
        results = await asyncio.gather(*tasks)
        prepared_samples = [r[0] for r in results]
        behavior_logprobs = [r[1] for r in results]

        # ── Phase 3: Multi-step training ──
        step_metrics: list[dict[str, float | int]] = []
        final_fwd_metrics: dict[str, object] | None = None
        for step_idx in range(steps_per_batch):
            datum_metrics = [
                _build_sample_datum(
                    prepared=prepared,
                    student_logprobs=behavior_logprobs[sample_idx],
                    kl_coef=kl_coef,
                )
                for sample_idx, prepared in enumerate(prepared_samples)
            ]
            datums = [dm[0] for dm in datum_metrics]
            sample_metrics = [dm[1] for dm in datum_metrics]

            fwd_bwd = await training_client.forward_backward_async(
                datums, "importance_sampling"
            )
            await training_client.optim_step_async(
                T.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95)
            )
            if hasattr(fwd_bwd, "metrics") and fwd_bwd.metrics:
                final_fwd_metrics = dict(fwd_bwd.metrics)

            n = len(sample_metrics)
            total_completion_len = sum(m["completion_len"] for m in sample_metrics)
            avg = lambda key: sum(m[key] for m in sample_metrics) / n  # noqa: E731
            step_metrics.append(
                {
                    "step": step_idx + 1,
                    "batch_size": n,
                    "completion_len": total_completion_len,
                    "effective_kl_coef": avg("effective_kl_coef"),
                    "kl_gain": avg("kl_gain"),
                    "adv_mean": avg("adv_mean"),
                    "adv_abs_mean": avg("adv_abs_mean"),
                    "kl_mean": avg("kl_mean"),
                    "adv_abs_mean_raw": avg("adv_abs_mean_raw"),
                }
            )

            if step_idx < steps_per_batch - 1:
                student_sampling = await training_client.save_weights_and_get_sampling_client_async()
                behavior_logprobs = await _compute_student_logprobs_for_batch(
                    student_sampling=student_sampling,
                    prepared_samples=prepared_samples,
                )

        # ── Phase 4: Save & return (once per request) ──
        final_step = step_metrics[-1]
        new_step = entry.step + steps_per_batch
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
            path=self._state_path,
        )

        metadata: dict[str, object] = {
            "step": new_step,
            "tinker_path": save_result.path,
            "sampler_weights_path": sampler_weights_path,
            "batch_size": final_step["batch_size"],
            "completion_len": final_step["completion_len"],
            "effective_kl_coef": final_step["effective_kl_coef"],
            "kl_gain": final_step["kl_gain"],
            "adv_mean": final_step["adv_mean"],
            "adv_abs_mean": final_step["adv_abs_mean"],
            "kl_mean": final_step["kl_mean"],
            "adv_abs_mean_raw": final_step["adv_abs_mean_raw"],
            "lr": lr,
            "loss_fn": "importance_sampling",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "teacher_scored_texts": [p["teacher_scored_text"] for p in prepared_samples],
            "steps_per_batch_applied": steps_per_batch,
            "per_step_metrics": step_metrics,
        }

        if final_fwd_metrics is not None:
            metadata["tinker_fwd_metrics"] = final_fwd_metrics

        return DistillResponse(lora_id=payload.lora_id, metadata=metadata)


class PreparedSample(TypedDict):
    full_tokens: list[int]
    input_tokens: list[int]
    target_tokens: list[int]
    prompt_len: int
    completion_len: int
    teacher_logprobs: list[float]
    teacher_scored_text: str


async def _prepare_sample_inputs(
    *,
    sample: DistillBatchItem,
    tokenizer: Any,
    teacher_sampling: Any,
    feedback_repetitions: int,
) -> tuple[PreparedSample, list[float]]:
    """Prepare sample-invariant tensors and initial behavior logprobs."""
    prompt_tokens = list(sample.prompt_token_ids)
    response_tokens = list(sample.response_token_ids)
    completion_len = len(response_tokens)
    if len(sample.response_logprobs) != completion_len:
        raise ValueError(
            f"response_logprobs length ({len(sample.response_logprobs)}) != "
            f"completion_len ({completion_len})"
        )

    full_tokens = prompt_tokens + response_tokens
    prompt_len = len(prompt_tokens)
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]

    repeated_feedback = " ".join([sample.feedback] * feedback_repetitions)
    teacher_messages = build_teacher_messages(
        sample.user_prompt,
        repeated_feedback,
        system_prompt=sample.system_prompt,
    )
    template_messages = teacher_messages_to_chat_template(teacher_messages)
    teacher_prompt_text = tokenizer.apply_chat_template(
        template_messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    teacher_prompt_tokens: list[int] = tokenizer.encode(
        teacher_prompt_text,
        add_special_tokens=False,
    )
    teacher_full_tokens = teacher_prompt_tokens + response_tokens
    teacher_prompt_len = len(teacher_prompt_tokens)
    teacher_full = T.ModelInput.from_ints(teacher_full_tokens)
    teacher_scored_text = tokenizer.decode(teacher_full_tokens, skip_special_tokens=False)

    teacher_logprobs_full = await teacher_sampling.compute_logprobs_async(teacher_full)
    teacher_logprobs = _slice_completion_logprobs(
        teacher_logprobs_full,
        teacher_prompt_len,
        completion_len,
    )

    prepared = PreparedSample(
        full_tokens=full_tokens,
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        prompt_len=prompt_len,
        completion_len=completion_len,
        teacher_logprobs=teacher_logprobs,
        teacher_scored_text=teacher_scored_text,
    )
    return prepared, list(sample.response_logprobs)


def _build_sample_datum(
    *,
    prepared: PreparedSample,
    student_logprobs: list[float],
    kl_coef: float,
) -> tuple[T.Datum, dict[str, float]]:
    """Build a Tinker datum from prepared teacher signals + current behavior policy."""
    completion_len = prepared["completion_len"]
    if len(student_logprobs) != completion_len:
        raise ValueError(
            f"student_logprobs length ({len(student_logprobs)}) != "
            f"completion_len ({completion_len})"
        )

    teacher_logprobs = prepared["teacher_logprobs"]
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

    full_logprobs = [0.0] * prepared["prompt_len"] + student_logprobs
    full_advantages = [0.0] * prepared["prompt_len"] + advantages
    shifted_logprobs = full_logprobs[1:]
    shifted_advantages = full_advantages[1:]

    datum = T.Datum(
        model_input=T.ModelInput.from_ints(prepared["input_tokens"]),
        loss_fn_inputs={
            "target_tokens": TensorData(data=prepared["target_tokens"], dtype="int64"),
            "logprobs": TensorData(data=shifted_logprobs, dtype="float32"),
            "advantages": TensorData(data=shifted_advantages, dtype="float32"),
        },
    )

    metrics = {
        "completion_len": completion_len,
        "effective_kl_coef": effective_kl_coef,
        "kl_gain": gain,
        "adv_mean": sum(advantages) / max(len(advantages), 1),
        "adv_abs_mean": sum(abs(a) for a in advantages) / max(len(advantages), 1),
        "kl_mean": sum(raw_kl_deltas) / max(len(raw_kl_deltas), 1),
        "adv_abs_mean_raw": adv_abs_mean_raw,
    }
    return datum, metrics


async def _compute_student_logprobs_for_batch(
    *,
    student_sampling: Any,
    prepared_samples: list[PreparedSample],
) -> list[list[float]]:
    """Recompute behavior logprobs under the updated student policy."""
    tasks = [
        _compute_student_logprobs_for_sample(
            student_sampling=student_sampling,
            prepared=prepared,
        )
        for prepared in prepared_samples
    ]
    return await asyncio.gather(*tasks)


async def _compute_student_logprobs_for_sample(
    *,
    student_sampling: Any,
    prepared: PreparedSample,
) -> list[float]:
    """Compute completion logprobs for one sample under current student weights."""
    student_full = T.ModelInput.from_ints(prepared["full_tokens"])
    student_logprobs_full = await student_sampling.compute_logprobs_async(student_full)
    return _slice_completion_logprobs(
        student_logprobs_full,
        prepared["prompt_len"],
        prepared["completion_len"],
    )


def _require_entry(lora_id: str, state_path: str) -> LoraEntry:
    """Return the state entry for *lora_id* or raise ``FileNotFoundError``."""
    entry = get_entry(lora_id, path=state_path)
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
