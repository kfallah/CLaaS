"""Integration tests for local training engine behavior."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from claas.training_engines.local.engine import LocalTrainingEngine
from claas.types import DistillRequestPayload, LoraInitRequest


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
    from claas import config
    from claas.training_engines.local import engine as engine_module

    monkeypatch.setenv("CLAAS_LORA_ROOT", str(tmp_path))
    monkeypatch.setenv("CLAAS_STORAGE_BACKEND", "local_fs")
    monkeypatch.setattr(config, "LORA_ROOT", str(tmp_path))

    state = WorkerState()
    monkeypatch.setattr(engine_module, "DistillWorker", lambda: _WorkerStub(state))
    local_engine = LocalTrainingEngine()

    init_response = asyncio.run(local_engine.init_lora(LoraInitRequest(lora_id="user/integration")))
    assert init_response.lora_id == "user/integration"

    exists_response = asyncio.run(local_engine.lora_exists("user/integration"))
    assert exists_response.exists is True

    lora_list = asyncio.run(local_engine.list_loras("user"))
    assert "user/integration" in lora_list.loras

    runtime_ref = asyncio.run(local_engine.lora_runtime_ref("user/integration"))
    assert runtime_ref.vllm_name == "user-integration"
    assert runtime_ref.lora_path.endswith("user/integration")

    distill_response = asyncio.run(
        local_engine.distill(
            DistillRequestPayload(
                lora_id="user/integration",
                prompt="prompt",
                response="response",
                feedback="feedback",
                training={},
            )
        )
    )
    assert distill_response.lora_id == "user/integration"
    assert state.payload is not None
    assert state.cleaned_up is True

    export_payload = asyncio.run(local_engine.export_lora("user/integration"))
    assert export_payload.filename == "user__integration.zip"
    assert len(export_payload.content) > 0

    health = asyncio.run(local_engine.health())
    assert health.status == "healthy"


def test_local_engine_cleanup_failure_is_ignored(monkeypatch):
    """Cleanup errors do not fail distillation requests."""
    from claas.training_engines.local import engine as engine_module

    state = WorkerState()
    monkeypatch.setattr(
        engine_module,
        "DistillWorker",
        lambda: _WorkerWithCleanupFailure(state),
    )

    result = asyncio.run(
        LocalTrainingEngine().distill(
            DistillRequestPayload(
                lora_id="user/integration",
                prompt="prompt",
                response="response",
                feedback="feedback",
                training={},
            )
        )
    )

    assert result.lora_id == "user/integration"
