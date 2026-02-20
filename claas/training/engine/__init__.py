"""Training engine selection for CLaaS."""

from __future__ import annotations

from typing import TYPE_CHECKING

from claas.core.config import CoreConfig, TinkerConfig
from claas.training.engine.base import EngineKind, TrainingEngine

if TYPE_CHECKING:
    from claas.training.engine.local.engine import LocalTrainingEngine as LocalTrainingEngine
    from claas.training.engine.modal.engine import ModalTrainingEngine as ModalTrainingEngine


def get_training_engine(kind: EngineKind, cfg: CoreConfig) -> TrainingEngine:
    """Build a concrete engine for the requested execution kind."""
    if kind == "local":
        from claas.training.engine.local.engine import LocalTrainingEngine

        return LocalTrainingEngine()
    if kind == "modal":
        from claas.training.engine.modal.engine import ModalTrainingEngine

        return ModalTrainingEngine()
    if kind == "tinker":
        from claas.training.engine.tinker.engine import TinkerTrainingEngine

        if not isinstance(cfg, TinkerConfig):
            raise ValueError("Tinker engine requires a TinkerConfig instance")
        return TinkerTrainingEngine(cfg)
    raise ValueError(f"Unsupported training engine: {kind}")
