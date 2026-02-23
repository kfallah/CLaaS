from __future__ import annotations

import asyncio

import pytest

torch = pytest.importorskip("torch")

from claas.core.config import LocalConfig  # noqa: E402
from claas.core.types import (  # noqa: E402
    DistillBatchItem,
    DistillBatchRequestPayload,
    DistillResponse,
    TrainingConfig,
)
from claas.training.engine.local.cache import (  # noqa: E402
    DistillStepResult,
    LoraAdapterConfig,
    LoraCacheEntry,
)
from claas.training.engine.local.engine import LocalTrainingEngine  # noqa: E402

_DUMMY_CACHE_ENTRY = LoraCacheEntry(
    lora_state_dict={"w": torch.zeros(2)},
    optimizer_state_dict={"state": {}, "param_groups": []},
    adapter_config=LoraAdapterConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    ),
)


def _make_payload(lora_id: str = "user/model") -> DistillBatchRequestPayload:
    return DistillBatchRequestPayload(
        lora_id=lora_id,
        training=TrainingConfig(),
        samples=[
            DistillBatchItem(
                prompt="p",
                response="r",
                feedback="f",
                response_logprobs=[-0.1],
                prompt_token_ids=[1, 2],
                response_token_ids=[3],
                user_prompt="p",
            )
        ],
    )


class _Trainer:
    """Fake trainer that records method calls."""

    def __init__(self, base_model_id: str, attn_implementation: str):
        self.base_model_id = base_model_id
        self.attn_implementation = attn_implementation
        self.load_base_model_count = 0
        self.reload_count = 0
        self.offload_count = 0
        self.distill_calls: list[dict] = []

    def load_base_model(self) -> None:
        self.load_base_model_count += 1

    def reload_base_model(self) -> None:
        self.reload_count += 1

    def distill(
        self,
        _payload: DistillBatchRequestPayload,
        *,
        cached: LoraCacheEntry | None = None,
    ) -> DistillStepResult:
        self.distill_calls.append({"cached": cached})
        return DistillStepResult(
            response=DistillResponse(lora_id="user/model", metadata={}),
            cache_entry=_DUMMY_CACHE_ENTRY,
        )

    def offload_base_model(self) -> None:
        self.offload_count += 1


class _FailingOffloadTrainer(_Trainer):
    """Trainer whose offload raises to test error propagation."""

    def offload_base_model(self) -> None:
        raise OSError("cleanup failed")


def _build_engine(monkeypatch, trainer_cls=_Trainer):
    from claas.training.engine.local import engine as local_engine

    monkeypatch.setattr(local_engine, "DistillationTrainer", trainer_cls)
    monkeypatch.setattr(local_engine, "resolve_lora_id", lambda lid: lid.strip("/"))
    cfg = LocalConfig(base_model_id="Qwen/Qwen3-8B", attn_implementation="sdpa")
    return LocalTrainingEngine(cfg)


def test_trainer_created_eagerly_in_init(monkeypatch):
    """Trainer is created in __init__, not lazily on first distill()."""
    engine = _build_engine(monkeypatch)
    assert isinstance(engine._trainer, _Trainer)
    assert engine._model_loaded is False


def test_load_base_model_called_once(monkeypatch):
    """load_base_model is called exactly once across multiple distill() calls."""
    engine = _build_engine(monkeypatch)

    asyncio.run(engine.distill(_make_payload()))
    asyncio.run(engine.distill(_make_payload()))

    assert engine._trainer.load_base_model_count == 1


def test_reload_called_every_distill(monkeypatch):
    """reload_base_model is called on every distill() call."""
    engine = _build_engine(monkeypatch)

    asyncio.run(engine.distill(_make_payload()))
    asyncio.run(engine.distill(_make_payload()))

    assert engine._trainer.reload_count == 2


def test_cache_miss_then_hit(monkeypatch):
    """First call has cached=None, second call uses the cached entry."""
    engine = _build_engine(monkeypatch)

    asyncio.run(engine.distill(_make_payload()))
    # First call: no cache
    assert engine._trainer.distill_calls[0]["cached"] is None

    asyncio.run(engine.distill(_make_payload()))
    # Second call: cache hit
    assert engine._trainer.distill_calls[1]["cached"] is _DUMMY_CACHE_ENTRY


def test_cache_evicted_on_delete(monkeypatch):
    """delete_lora() evicts the cache entry for that lora_id."""
    from claas.training.engine.local import engine as local_engine

    monkeypatch.setattr(local_engine, "DistillationTrainer", _Trainer)
    monkeypatch.setattr(local_engine, "resolve_lora_id", lambda lid: lid.strip("/"))
    monkeypatch.setattr(local_engine, "delete_lora", lambda lid: True)
    cfg = LocalConfig(base_model_id="Qwen/Qwen3-8B", attn_implementation="sdpa")
    engine = LocalTrainingEngine(cfg)

    asyncio.run(engine.distill(_make_payload()))
    assert "user/model" in engine._lora_cache

    asyncio.run(engine.delete_lora("user/model"))
    assert "user/model" not in engine._lora_cache


def test_offload_error_propagates(monkeypatch):
    """Errors from offload_base_model propagate to the caller."""
    engine = _build_engine(monkeypatch, trainer_cls=_FailingOffloadTrainer)

    with pytest.raises(OSError, match="cleanup failed"):
        asyncio.run(engine.distill(_make_payload()))
