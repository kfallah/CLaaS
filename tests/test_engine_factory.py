"""Tests for training engine factory."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from claas.training.engine import get_training_engine  # noqa: E402
from claas.training.engine.local.engine import LocalTrainingEngine  # noqa: E402
from claas.training.engine.modal.engine import ModalTrainingEngine  # noqa: E402


def test_get_local_engine():
    engine = get_training_engine("local")
    assert isinstance(engine, LocalTrainingEngine)


def test_get_modal_engine():
    engine = get_training_engine("modal")
    assert isinstance(engine, ModalTrainingEngine)


def test_get_tinker_engine():
    from claas.training.engine.tinker.engine import TinkerTrainingEngine

    engine = get_training_engine("tinker")
    assert isinstance(engine, TinkerTrainingEngine)


def test_unsupported_engine_raises():
    with pytest.raises(ValueError, match="Unsupported training engine"):
        get_training_engine("nonexistent")  # type: ignore[arg-type]
