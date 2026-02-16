"""Training engine selection for CLaaS."""

from __future__ import annotations

from typing import TYPE_CHECKING

from claas.training_engines.base import EngineKind, TrainingEngine

if TYPE_CHECKING:
    from claas.training_engines.local.engine import LocalTrainingEngine as LocalTrainingEngine
    from claas.training_engines.modal.engine import ModalTrainingEngine as ModalTrainingEngine


def get_training_engine(kind: EngineKind) -> TrainingEngine:
    """Build a concrete engine for the requested execution kind."""
    if kind == "local":
        from claas.training_engines.local.engine import LocalTrainingEngine

        return LocalTrainingEngine()
    if kind == "modal":
        from claas.training_engines.modal.engine import ModalTrainingEngine

        return ModalTrainingEngine()
    if kind == "tinker":
        from claas.training_engines.tinker.engine import TinkerTrainingEngine

        return TinkerTrainingEngine()
    raise ValueError(f"Unsupported training engine: {kind}")
