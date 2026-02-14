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
from urllib.parse import quote

import httpx
import tinker
from tinker import types as T
from tinker.types.tensor_data import TensorData

from claas.training_engines.base import TrainingEngine
from claas.training_engines.tinker.state import (
    get_entry,
    set_tinker_path,
)
from claas.training_engines.tinker.state import (
    list_loras as state_list_loras,
)
from claas.training_engines.tinker.state import (
    lora_exists as state_lora_exists,
)
from claas.types import (
    DistillRequestPayload,
    DistillResponse,
    LoraExistsPayload,
    LoraExportPayload,
    LoraInitRequest,
    LoraInitResponse,
    LoraListResponse,
    LoraRuntimeRef,
    ServiceHealth,
)

logger = logging.getLogger(__name__)

# Default model for the Tinker engine.
_DEFAULT_BASE_MODEL = os.environ.get("CLAAS_TINKER_BASE_MODEL", "Qwen/Qwen3-235B-A22B")

# Adaptive KL scaling defaults (from continualcode reference).
_TARGET_ADV_ABS_MEAN = 0.03
_MAX_KL_GAIN = 4.0


def _import_cookbook():  # noqa: ANN202
    """Lazy import of tinker_cookbook to avoid hard failure when not installed."""
    from tinker_cookbook import model_info  # noqa: F811
    from tinker_cookbook.renderers import get_renderer  # noqa: F811
    from tinker_cookbook.supervised.common import (
        create_rightshifted_model_input_and_leftshifted_targets,  # noqa: F811
    )

    return get_renderer, create_rightshifted_model_input_and_leftshifted_targets, model_info


class TinkerTrainingEngine(TrainingEngine):
    """Executes training and LoRA management through the Tinker Python SDK."""

    def __init__(self) -> None:
        api_key = os.environ.get("CLAAS_TINKER_API_KEY", "")
        if api_key:
            os.environ["TINKER_API_KEY"] = api_key
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
        base_model = request.base_model or _DEFAULT_BASE_MODEL
        rank = request.lora_r

        def _init() -> LoraInitResponse:
            tc = self.service.create_lora_training_client(
                base_model=base_model,
                rank=rank,
            )
            save_resp = tc.save_state("init").result()
            set_tinker_path(
                lora_id=request.lora_id,
                tinker_path=save_resp.path,
                base_model=base_model,
                rank=rank,
                step=0,
            )
            return LoraInitResponse(lora_id=request.lora_id)

        return await asyncio.to_thread(_init)

    async def list_loras(self, prefix: str) -> LoraListResponse:
        loras = state_list_loras(prefix)
        return LoraListResponse(loras=loras)

    async def lora_exists(self, lora_id: str) -> LoraExistsPayload:
        return LoraExistsPayload(exists=state_lora_exists(lora_id))

    async def export_lora(self, lora_id: str) -> LoraExportPayload:
        entry = get_entry(lora_id)
        if entry is None:
            raise FileNotFoundError(f"LoRA not found in Tinker state: {lora_id}")

        def _export() -> LoraExportPayload:
            rest = self.service.create_rest_client()
            archive_resp = rest.get_checkpoint_archive_url(entry.tinker_path)
            url = archive_resp.result() if hasattr(archive_resp, "result") else archive_resp
            resp = httpx.get(str(url), follow_redirects=True, timeout=120)
            resp.raise_for_status()
            filename = f"{quote(lora_id, safe='')}.zip"
            return LoraExportPayload(filename=filename, content=resp.content)

        return await asyncio.to_thread(_export)

    async def lora_runtime_ref(self, lora_id: str) -> LoraRuntimeRef:
        raise ValueError(
            "tinker backend does not expose local runtime LoRA paths for vLLM reload"
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health(self) -> ServiceHealth:
        def _health() -> ServiceHealth:
            try:
                self.service.get_server_capabilities()
                return ServiceHealth(status="healthy", error=None)
            except Exception as exc:
                return ServiceHealth(status="unhealthy", error=str(exc))

        return await asyncio.to_thread(_health)

    # ------------------------------------------------------------------
    # Distillation
    # ------------------------------------------------------------------

    async def distill(self, payload: DistillRequestPayload) -> DistillResponse:
        """Run one SDPO distillation step entirely through the Tinker SDK.

        Follows the continualcode reference implementation:
        1. Tokenize prompt+completion with the renderer
        2. Compute student (rollout) logprobs via SamplingClient
        3. Compute teacher logprobs (base model conditioned on feedback)
        4. Derive advantages with adaptive KL scaling
        5. Build Datum with right-shifted alignment
        6. forward_backward with importance_sampling loss
        7. optim_step with AdamW
        8. Save checkpoint → update state
        """

        def _run() -> DistillResponse:
            return self._distill_sync(payload)

        return await asyncio.to_thread(_run)

    def _distill_sync(self, payload: DistillRequestPayload) -> DistillResponse:
        """Synchronous distillation core — runs in a worker thread."""
        get_renderer, create_rightshifted, model_info = _import_cookbook()

        lora_id = payload.lora_id
        entry = get_entry(lora_id)
        if entry is None:
            raise FileNotFoundError(f"LoRA not found in Tinker state: {lora_id}")

        base_model = entry.base_model
        tinker_path = entry.tinker_path
        step = entry.step
        lr = payload.training.learning_rate
        kl_coef = payload.training.alpha  # reuse alpha as kl_coef

        # ── Restore training client from checkpoint ──
        training_client = self.service.create_training_client_from_state(tinker_path)
        tokenizer = training_client.get_tokenizer()

        # ── Set up renderer ──
        renderer_name = model_info.get_recommended_renderer_name(base_model)
        renderer = get_renderer(renderer_name, tokenizer=tokenizer)

        # ── Build student prompt + completion as ModelInput ──
        student_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": payload.prompt},
        ]
        student_model_input = renderer.build_generation_prompt(student_messages)
        prompt_len = student_model_input.length()

        completion_tokens = tokenizer.encode(payload.response, add_special_tokens=False)
        student_full = student_model_input.append(
            T.EncodedTextChunk(tokens=completion_tokens)
        )
        completion_len = len(completion_tokens)

        # ── Compute student (rollout) logprobs ──
        if payload.rollout_logprobs is not None and len(payload.rollout_logprobs) == completion_len:
            student_logprobs = list(payload.rollout_logprobs)
        else:
            sampling_client = training_client.save_weights_and_get_sampling_client("current")
            student_logprobs_full = sampling_client.compute_logprobs(student_full).result()
            student_logprobs = _slice_completion_logprobs(
                student_logprobs_full, prompt_len, completion_len
            )

        # ── Build teacher prompt (conversation + feedback reprompt) ──
        teacher_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": payload.prompt},
            {"role": "assistant", "content": payload.response},
            {
                "role": "user",
                "content": (
                    f"{payload.prompt}\n\nFeedback:\n{payload.feedback}\n\n"
                    "Correctly solve the original question."
                ),
            },
        ]
        teacher_model_input = renderer.build_generation_prompt(teacher_messages)
        teacher_prompt_len = teacher_model_input.length()
        teacher_full = teacher_model_input.append(
            T.EncodedTextChunk(tokens=completion_tokens)
        )

        # ── Compute teacher logprobs (base model = self-distillation) ──
        teacher_sampling = self.service.create_sampling_client(base_model=base_model)
        teacher_logprobs_full = teacher_sampling.compute_logprobs(teacher_full).result()
        teacher_logprobs = _slice_completion_logprobs(
            teacher_logprobs_full, teacher_prompt_len, completion_len
        )

        # ── Compute advantages with adaptive KL scaling ──
        raw_kl_deltas = [t - s for s, t in zip(student_logprobs, teacher_logprobs)]
        adv_abs_mean_raw = sum(abs(d) for d in raw_kl_deltas) / max(len(raw_kl_deltas), 1)

        gain = 1.0
        if adv_abs_mean_raw > 0:
            gain = min(max(_TARGET_ADV_ABS_MEAN / adv_abs_mean_raw, 1.0), _MAX_KL_GAIN)
        effective_kl_coef = kl_coef * gain

        advantages = [effective_kl_coef * (t - s) for s, t in zip(student_logprobs, teacher_logprobs)]

        # ── Build Datum using right-shifted alignment ──
        input_model_input, target_tokens = create_rightshifted(student_full.chunks)

        # Pad prompt positions with 0.0, real values for completion only.
        # After right-shift the sequence is 1 shorter; we drop the first position.
        full_logprobs = [0.0] * prompt_len + student_logprobs
        full_advantages = [0.0] * prompt_len + advantages
        # Right-shift: drop position 0 (it becomes position -1 in targets)
        shifted_logprobs = full_logprobs[1:]
        shifted_advantages = full_advantages[1:]

        import torch

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

        # ── Train with importance_sampling loss ──
        fwd_future = training_client.forward_backward([datum], "importance_sampling")
        optim_future = training_client.optim_step(
            T.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95)
        )
        fwd_bwd = fwd_future.result()
        optim_future.result()

        # ── Save checkpoint & update state ──
        new_step = step + 1
        checkpoint_name = f"step-{new_step}"
        save_result = training_client.save_state(checkpoint_name).result()

        set_tinker_path(
            lora_id=lora_id,
            tinker_path=save_result.path,
            base_model=base_model,
            rank=entry.rank,
            step=new_step,
        )

        # ── Return metrics ──
        adv_mean = sum(advantages) / max(len(advantages), 1)
        adv_abs_mean = sum(abs(a) for a in advantages) / max(len(advantages), 1)
        kl_mean = sum(raw_kl_deltas) / max(len(raw_kl_deltas), 1)

        metadata = {
            "step": new_step,
            "tinker_path": save_result.path,
            "completion_len": completion_len,
            "effective_kl_coef": effective_kl_coef,
            "kl_gain": gain,
            "adv_mean": adv_mean,
            "adv_abs_mean": adv_abs_mean,
            "kl_mean": kl_mean,
            "adv_abs_mean_raw": adv_abs_mean_raw,
            "lr": lr,
            "loss_fn": "importance_sampling",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Merge metrics from Tinker's forward_backward output if available.
        if hasattr(fwd_bwd, "metrics") and fwd_bwd.metrics:
            metadata["tinker_fwd_metrics"] = fwd_bwd.metrics

        return DistillResponse(lora_id=lora_id, metadata=metadata)


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
