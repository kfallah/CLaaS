"""Canonical SDPO metric normalization across execution backends.

Different engines populate ``DistillResponse.metadata`` with different keys.
This helper maps those variants into one stable shape so downstream consumers
(API logs, eval harness, dashboards) do not need backend-specific branching.
"""

from __future__ import annotations

from typing import Any


CANONICAL_SDPO_KEYS = (
    "distill_loss",
    "kl_reg",
    "mean_is_ratio",
    "clip_fraction",
)


def _as_float(value: object) -> float | None:
    """Best-effort float conversion for numeric metric values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def normalize_sdpo_metrics(metadata: dict[str, Any] | None) -> dict[str, float | None]:
    """Normalize backend-specific distill metadata to canonical SDPO metrics.

    Returns a dict containing exactly:
    - ``distill_loss``
    - ``kl_reg``
    - ``mean_is_ratio``
    - ``clip_fraction``

    Values are ``float`` when available, otherwise ``None``.
    """
    md: dict[str, Any] = metadata or {}
    tinker_fwd = md.get("tinker_fwd_metrics")
    tinker_fwd = tinker_fwd if isinstance(tinker_fwd, dict) else {}

    # Local/Modal engines already emit canonical keys directly.
    distill_loss = _as_float(md.get("distill_loss"))
    kl_reg = _as_float(md.get("kl_reg"))
    mean_is_ratio = _as_float(md.get("mean_is_ratio"))
    clip_fraction = _as_float(md.get("clip_fraction"))

    # Tinker fallback: only use semantically strong aliases.
    if distill_loss is None:
        distill_loss = _as_float(tinker_fwd.get("loss"))

    return {
        "distill_loss": distill_loss,
        "kl_reg": kl_reg,
        "mean_is_ratio": mean_is_ratio,
        "clip_fraction": clip_fraction,
    }

