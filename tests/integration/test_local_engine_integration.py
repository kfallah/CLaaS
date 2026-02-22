"""Integration tests for local training engine behavior."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from claas.core.config import LocalConfig  # noqa: E402
from claas.core.types import (  # noqa: E402
    DistillBatchItem,
    DistillBatchRequestPayload,
    DistillResponse,
    LoraInitRequest,
    TrainingConfig,
)
from claas.training import storage  # noqa: E402
from claas.training.engine.local.engine import LocalTrainingEngine  # noqa: E402


@dataclass
class TrainerState:
    """Tracks calls made to the fake trainer."""

    payload: DistillBatchRequestPayload | None = None
    loaded: bool = False
    cleaned_up: bool = False


class _TrainerStub:
    """Distillation trainer test double with deterministic behavior."""

    def __init__(self, base_model_id: str, attn_implementation: str, state: TrainerState):
        self.base_model_id = base_model_id
        self.attn_implementation = attn_implementation
        self._state = state

    def load_base_model(self) -> None:
        self._state.loaded = True

    def distill(self, payload: DistillBatchRequestPayload) -> DistillResponse:
        self._state.payload = payload
        return DistillResponse.model_validate(
            {"lora_id": payload.lora_id, "metadata": {"tokens_processed": 5}}
        )

    def offload_base_model(self) -> None:
        self._state.cleaned_up = True


class _TrainerWithCleanupFailure(_TrainerStub):
    """Distillation trainer test double that raises on cleanup."""

    def offload_base_model(self) -> None:
        raise RuntimeError("cleanup failure")


def test_local_engine_integration_paths(monkeypatch, tmp_path):
    """Exercise the full local engine API surface against local storage."""
    from claas.training.engine.local import engine as engine_module

    monkeypatch.setenv("CLAAS_LORA_ROOT", str(tmp_path))
    monkeypatch.setenv("CLAAS_BASE_MODEL_ID", "Qwen/Qwen3-8B")
    monkeypatch.setenv("CLAAS_ATTN_IMPLEMENTATION", "sdpa")
    monkeypatch.setattr(storage, "LORA_MOUNT_PATH", str(tmp_path))

    state = TrainerState()
    monkeypatch.setattr(
        engine_module,
        "DistillationTrainer",
        lambda base_model_id, attn_implementation: _TrainerStub(
            base_model_id,
            attn_implementation,
            state,
        ),
    )
    cfg = LocalConfig(base_model_id="Qwen/Qwen3-8B", attn_implementation="sdpa")
    local_engine = LocalTrainingEngine(cfg)

    init_response = asyncio.run(local_engine.init_lora(LoraInitRequest(lora_id="user/integration")))
    lora_id = init_response.lora_id
    assert lora_id.startswith("user/integration")

    exists_response = asyncio.run(local_engine.lora_exists(lora_id))
    assert exists_response.exists is True

    lora_list = asyncio.run(local_engine.list_loras("user"))
    assert lora_id in lora_list.loras

    runtime_ref = asyncio.run(local_engine.lora_runtime_ref(lora_id))
    assert runtime_ref.lora_path.endswith(lora_id)

    distill_response = asyncio.run(
        local_engine.distill(
            DistillBatchRequestPayload(
                lora_id=lora_id,
                training=TrainingConfig(),
                samples=[
                    DistillBatchItem(
                        prompt="prompt",
                        response="response",
                        feedback="feedback",
                        response_logprobs=[-0.1],
                        prompt_token_ids=[1, 2],
                        response_token_ids=[3],
                        user_prompt="prompt",
                    )
                ],
            )
        )
    )
    assert distill_response.lora_id == lora_id
    assert state.payload is not None
    assert state.loaded is True
    assert state.cleaned_up is True

    export_payload = asyncio.run(local_engine.export_lora(lora_id))
    assert export_payload.filename.endswith(".zip")
    assert len(export_payload.content) > 0

    health = asyncio.run(local_engine.health())
    assert health.status == "healthy"


def test_local_engine_cleanup_failure_propagates(monkeypatch):
    """Cleanup errors fail distillation requests."""
    from claas.training.engine.local import engine as engine_module

    monkeypatch.setenv("CLAAS_BASE_MODEL_ID", "Qwen/Qwen3-8B")
    monkeypatch.setenv("CLAAS_ATTN_IMPLEMENTATION", "sdpa")
    state = TrainerState()
    monkeypatch.setattr(
        engine_module,
        "DistillationTrainer",
        lambda base_model_id, attn_implementation: _TrainerWithCleanupFailure(
            base_model_id,
            attn_implementation,
            state,
        ),
    )
    cfg = LocalConfig(base_model_id="Qwen/Qwen3-8B", attn_implementation="sdpa")

    with pytest.raises(RuntimeError, match="cleanup failure"):
        asyncio.run(
            LocalTrainingEngine(cfg).distill(
                DistillBatchRequestPayload(
                    lora_id="user/integration",
                    training=TrainingConfig(),
                    samples=[
                        DistillBatchItem(
                            prompt="prompt",
                            response="response",
                            feedback="feedback",
                            response_logprobs=[-0.1],
                            prompt_token_ids=[1, 2],
                            response_token_ids=[3],
                            user_prompt="prompt",
                        )
                    ],
                )
            )
        )
