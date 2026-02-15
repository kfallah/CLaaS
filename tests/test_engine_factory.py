"""Tests for training engine factory."""

from __future__ import annotations

import pytest

from claas.training_engines import get_training_engine
from claas.training_engines.local.engine import LocalTrainingEngine
from claas.training_engines.modal.engine import ModalTrainingEngine


def test_get_local_engine():
    engine = get_training_engine("local")
    assert isinstance(engine, LocalTrainingEngine)


def test_get_modal_engine():
    engine = get_training_engine("modal")
    assert isinstance(engine, ModalTrainingEngine)


def test_get_tinker_engine():
    from claas.training_engines.tinker.engine import TinkerTrainingEngine

    engine = get_training_engine("tinker")
    assert isinstance(engine, TinkerTrainingEngine)


def test_unsupported_engine_raises():
    with pytest.raises(ValueError, match="Unsupported training engine"):
        get_training_engine("nonexistent")  # type: ignore[arg-type]
