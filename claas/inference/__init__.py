"""Inference backend abstraction.

Factory function to get the appropriate backend based on execution mode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claas.core.config import CoreConfig

    from .base import BackendKind, InferenceBackend


def get_inference_backend(kind: BackendKind, cfg: CoreConfig | None = None) -> InferenceBackend:
    """Return the appropriate inference backend for the given execution mode."""
    if kind in ("local", "modal"):
        from .vllm import VllmBackend

        return VllmBackend(cfg=cfg)
    if kind == "tinker":
        from claas.core.config import TinkerConfig

        from .tinker import TinkerBackend

        assert isinstance(cfg, TinkerConfig), f"tinker backend requires TinkerConfig, got {type(cfg)}"
        return TinkerBackend(cfg=cfg)
    raise ValueError(f"Unsupported inference backend: {kind!r}")
