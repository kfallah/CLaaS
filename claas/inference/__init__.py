"""Inference backend abstraction.

Factory function to get the appropriate backend based on execution mode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BackendKind, InferenceBackend


def get_inference_backend(kind: BackendKind) -> InferenceBackend:
    """Return the appropriate inference backend for the given execution mode."""
    if kind in ("local", "modal"):
        from .vllm import VllmBackend

        return VllmBackend()
    if kind == "tinker":
        from .tinker import TinkerBackend

        return TinkerBackend()
    raise ValueError(f"Unsupported inference backend: {kind!r}")
