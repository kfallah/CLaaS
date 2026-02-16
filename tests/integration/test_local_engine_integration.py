"""Integration tests for local training engine behavior."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from claas.core.types import (  # noqa: E402
    DistillBatchItem,
    DistillBatchRequestPayload,
    LoraInitRequest,
    TrainingConfig,
)
from claas.training import storage  # noqa: E402
from claas.training.engine.local.engine import LocalTrainingEngine  # noqa: E402


@dataclass
class WorkerState:
    """Tracks calls made to the fake worker.

    Attributes:
        payload: Distillation payload forwarded by the engine.
        cleaned_up: Whether cleanup was triggered.
    """

    payload: dict[str, object] | None = None
    cleaned_up: bool = False


class _WorkerStub:
    """Distill worker test double with deterministic behavior."""

    def __init__(self, state: WorkerState):
        self.distill = self
        self._state = state

    def local(self, payload: dict[str, object]) -> dict[str, object]:
        """Return a valid distill response payload.

        Args:
            payload: Incoming request payload.

        Returns:
            Distill response payload.
        """
        self._state.payload = payload
        return {"lora_id": payload["lora_id"], "metadata": {"tokens_processed": 5}}

    def _offload_base_model(self) -> None:
        """Track cleanup calls from the engine."""
        self._state.cleaned_up = True


class _WorkerWithCleanupFailure(_WorkerStub):
    """Distill worker test double that raises on cleanup."""

    def _offload_base_model(self) -> None:
        """Raise a deterministic cleanup error."""
        raise RuntimeError("cleanup failure")


def test_local_engine_integration_paths(monkeypatch, tmp_path):
    """Exercise the full local engine API surface against local storage."""
    from claas.training.engine.local import engine as engine_module

    monkeypatch.setenv("CLAAS_LORA_ROOT", str(tmp_path))
    monkeypatch.setenv("CLAAS_STORAGE_BACKEND", "local_fs")
    monkeypatch.setattr(storage, "LORA_MOUNT_PATH", str(tmp_path))

    state = WorkerState()
    monkeypatch.setattr(engine_module, "DistillWorker", lambda: _WorkerStub(state))
    local_engine = LocalTrainingEngine()

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
                        rollout_logprobs=[-0.1],
                    )
                ],
            )
        )
    )
    assert distill_response.lora_id == lora_id
    assert state.payload is not None
    assert state.cleaned_up is True

    export_payload = asyncio.run(local_engine.export_lora(lora_id))
    assert export_payload.filename.endswith(".zip")
    assert len(export_payload.content) > 0

    health = asyncio.run(local_engine.health())
    assert health.status == "healthy"


def test_local_engine_cleanup_failure_is_ignored(monkeypatch):
    """Cleanup errors do not fail distillation requests."""
    from claas.training.engine.local import engine as engine_module

    state = WorkerState()
    monkeypatch.setattr(
        engine_module,
        "DistillWorker",
        lambda: _WorkerWithCleanupFailure(state),
    )

    result = asyncio.run(
        LocalTrainingEngine().distill(
            DistillBatchRequestPayload(
                lora_id="user/integration",
                training=TrainingConfig(),
                samples=[
                    DistillBatchItem(
                        prompt="prompt",
                        response="response",
                        feedback="feedback",
                        rollout_logprobs=[-0.1],
                    )
                ],
            )
        )
    )

    assert result.lora_id == "user/integration"
