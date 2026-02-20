from __future__ import annotations

import asyncio

import pytest

torch = pytest.importorskip("torch")

from claas.core.types import (  # noqa: E402
    DistillBatchItem,
    DistillBatchRequestPayload,
    DistillResponse,
    TrainingConfig,
)
from claas.training.engine.local.engine import LocalTrainingEngine  # noqa: E402


class _Trainer:
    def __init__(self, base_model_id: str, attn_implementation: str):
        self.base_model_id = base_model_id
        self.attn_implementation = attn_implementation

    def load_base_model(self) -> None:
        return None

    def distill(self, _payload: DistillBatchRequestPayload) -> DistillResponse:
        return DistillResponse(lora_id="user/model", metadata={})

    def offload_base_model(self) -> None:
        raise OSError("cleanup failed")


def test_local_engine_distill_propagates_cleanup_error(monkeypatch):
    from claas.training.engine.local import engine as local_engine

    monkeypatch.setenv("CLAAS_BASE_MODEL_ID", "Qwen/Qwen3-8B")
    monkeypatch.setenv("CLAAS_ATTN_IMPLEMENTATION", "sdpa")
    monkeypatch.setattr(local_engine, "DistillationTrainer", _Trainer)

    with pytest.raises(OSError, match="cleanup failed"):
        asyncio.run(
            LocalTrainingEngine().distill(
                DistillBatchRequestPayload(
                    lora_id="user/model",
                    training=TrainingConfig(),
                    samples=[
                        DistillBatchItem(
                            prompt="p",
                            response="r",
                            feedback="f",
                            rollout_logprobs=[-0.1],
                        )
                    ],
                )
            )
        )
