from __future__ import annotations

import asyncio

import pytest

torch = pytest.importorskip("torch")

from claas.training_engines.local.engine import LocalTrainingEngine  # noqa: E402
from claas.types import DistillRequestPayload, TrainingConfig  # noqa: E402


class _Worker:
    def __init__(self):
        self.distill = self

    def local(self, _payload):
        return {"lora_id": "user/model", "metadata": {}}

    def _offload_base_model(self):
        raise OSError("cleanup failed")


def test_local_engine_distill_ignores_cleanup_error(monkeypatch):
    from claas.training_engines.local import engine as local_engine

    monkeypatch.setattr(local_engine, "DistillWorker", _Worker)

    result = asyncio.run(
        LocalTrainingEngine().distill(
            DistillRequestPayload(
                lora_id="user/model",
                prompt="p",
                response="r",
                feedback="f",
                training=TrainingConfig(),
            )
        )
    )

    assert result.lora_id == "user/model"
