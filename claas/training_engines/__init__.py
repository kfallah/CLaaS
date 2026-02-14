"""Training engine selection for CLaaS."""

from __future__ import annotations

from claas.training_engines.base import EngineKind, TrainingEngine
from claas.training_engines.local.engine import LocalTrainingEngine
from claas.training_engines.modal.engine import ModalTrainingEngine
from claas.training_engines.tinker.engine import TinkerTrainingEngine


def get_training_engine(kind: EngineKind) -> TrainingEngine:
    """Build a concrete engine for the requested execution kind."""
    if kind == "local":
        return LocalTrainingEngine()
    if kind == "modal":
        return ModalTrainingEngine()
    if kind == "tinker":
        return TinkerTrainingEngine()
    raise ValueError(f"Unsupported training engine: {kind}")
