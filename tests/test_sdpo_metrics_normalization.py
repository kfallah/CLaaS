from __future__ import annotations

from claas.core.sdpo_metrics import CANONICAL_SDPO_KEYS, normalize_sdpo_metrics


def test_normalize_sdpo_metrics_local_modal_metadata() -> None:
    metadata = {
        "distill_loss": 1.25,
        "kl_reg": 0.03,
        "mean_is_ratio": 0.92,
        "clip_fraction": 0.11,
        "other": "ignored",
    }
    got = normalize_sdpo_metrics(metadata)
    assert tuple(got.keys()) == CANONICAL_SDPO_KEYS
    assert got == {
        "distill_loss": 1.25,
        "kl_reg": 0.03,
        "mean_is_ratio": 0.92,
        "clip_fraction": 0.11,
    }


def test_normalize_sdpo_metrics_tinker_metadata() -> None:
    metadata = {
        "step": 2,
        "kl_mean": -0.8,
        "tinker_fwd_metrics": {"loss": 0.42},
    }
    got = normalize_sdpo_metrics(metadata)
    assert tuple(got.keys()) == CANONICAL_SDPO_KEYS
    assert got["distill_loss"] == 0.42
    assert got["kl_reg"] is None
    assert got["mean_is_ratio"] is None
    assert got["clip_fraction"] is None


def test_normalize_sdpo_metrics_missing_values() -> None:
    got = normalize_sdpo_metrics({})
    assert got == {
        "distill_loss": None,
        "kl_reg": None,
        "mean_is_ratio": None,
        "clip_fraction": None,
    }

