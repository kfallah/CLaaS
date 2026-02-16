from __future__ import annotations

import asyncio

import pytest

torch = pytest.importorskip("torch")

from claas.core.types import (  # noqa: E402
    DistillBatchItem,
    DistillBatchRequestPayload,
    TrainingConfig,
)
from claas.training.engine.local.engine import LocalTrainingEngine  # noqa: E402


class _Worker:
    def __init__(self):
        self.distill = self

    def local(self, _payload):
        return {"lora_id": "user/model", "metadata": {}}

    def _offload_base_model(self):
        raise OSError("cleanup failed")


def test_local_engine_distill_ignores_cleanup_error(monkeypatch):
    from claas.training.engine.local import engine as local_engine

    monkeypatch.setattr(local_engine, "DistillWorker", _Worker)

    result = asyncio.run(
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

    assert result.lora_id == "user/model"
