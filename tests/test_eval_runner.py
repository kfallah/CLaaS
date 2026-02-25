"""Tests for eval metric types and deserialization."""

from __future__ import annotations

from claas.core.types import TrainingConfig
from claas.eval.types import (
    EvalConfig,
    LocalDistillMetrics,
    StepResult,
    TinkerDistillMetrics,
    step_result_from_dict,
)

# ── TinkerDistillMetrics ─────────────────────────────────────────────


def test_tinker_distill_metrics_fields():
    """TinkerDistillMetrics preserves native Tinker field names."""
    m = TinkerDistillMetrics(
        adv_mean=-0.42,
        kl_mean=0.015,
        effective_kl_coef=0.1,
        kl_gain=0.003,
        adv_abs_mean=0.55,
        adv_abs_mean_raw=0.60,
        completion_len=128,
        batch_size=4,
    )
    assert m.adv_mean == -0.42
    assert m.kl_mean == 0.015
    assert m.effective_kl_coef == 0.1
    assert m.kl_gain == 0.003
    assert m.adv_abs_mean == 0.55
    assert m.adv_abs_mean_raw == 0.60
    assert m.completion_len == 128
    assert m.batch_size == 4


def test_tinker_distill_metrics_defaults():
    """Optional fields default to 0."""
    m = TinkerDistillMetrics(
        adv_mean=1.0,
        kl_mean=2.0,
        effective_kl_coef=3.0,
        kl_gain=4.0,
        adv_abs_mean=5.0,
        adv_abs_mean_raw=6.0,
    )
    assert m.completion_len == 0
    assert m.batch_size == 0


# ── step_result_from_dict deserialization ─────────────────────────────


_BASE_STEP = {
    "preference": "no_emoji",
    "step": 3,
    "timestamp": "2026-02-18T00:00:00Z",
    "feedback_given": "no emojis",
    "prompt_used": "Hello",
    "eval": {},
}


def test_step_result_from_dict_tinker_metrics():
    """Tinker metadata (adv_mean present) deserializes to TinkerDistillMetrics."""
    data = {
        **_BASE_STEP,
        "sdpo_metrics": {
            "adv_mean": -0.5,
            "kl_mean": 0.01,
            "effective_kl_coef": 0.1,
            "kl_gain": 0.002,
            "adv_abs_mean": 0.5,
            "adv_abs_mean_raw": 0.55,
            "completion_len": 64,
            "batch_size": 4,
        },
    }
    result = step_result_from_dict(data)
    assert isinstance(result.sdpo_metrics, TinkerDistillMetrics)
    assert result.sdpo_metrics.adv_mean == -0.5


def test_step_result_from_dict_local_metrics():
    """Local metadata (distill_loss present) deserializes to LocalDistillMetrics."""
    data = {
        **_BASE_STEP,
        "sdpo_metrics": {
            "distill_loss": 0.35,
            "kl_reg": 0.02,
            "mean_is_ratio": 1.01,
            "clip_fraction": 0.05,
        },
    }
    result = step_result_from_dict(data)
    assert isinstance(result.sdpo_metrics, LocalDistillMetrics)
    assert result.sdpo_metrics.distill_loss == 0.35


def test_step_result_from_dict_no_metrics():
    """Missing sdpo_metrics deserializes to None."""
    result = step_result_from_dict(_BASE_STEP)
    assert result.sdpo_metrics is None


# ── EvalConfig defaults ──────────────────────────────────────────────


def test_steps_per_batch_default():
    """EvalConfig.steps_per_batch defaults to 4."""
    config = EvalConfig()
    assert config.steps_per_batch == 4


def test_steps_per_batch_custom():
    """EvalConfig.steps_per_batch can be set."""
    config = EvalConfig(steps_per_batch=3)
    assert config.steps_per_batch == 3


def test_feedback_repetitions_default():
    """EvalConfig.feedback_repetitions defaults to 1."""
    config = EvalConfig()
    assert config.feedback_repetitions == 1


def test_feedback_repetitions_custom():
    """EvalConfig.feedback_repetitions can be set."""
    config = EvalConfig(feedback_repetitions=4)
    assert config.feedback_repetitions == 4


def test_training_config_default_type():
    """EvalConfig.training defaults to typed TrainingConfig."""
    config = EvalConfig()
    assert isinstance(config.training, TrainingConfig)
    assert config.training.is_clip == 5.0


# ── StepResult sub_step_count ────────────────────────────────────────


def test_sub_step_count_default():
    """StepResult.sub_step_count defaults to 1."""
    from claas.eval.types import EvalMetrics

    sr = StepResult(
        preference="no_emoji",
        step=0,
        timestamp="2026-02-19T00:00:00Z",
        feedback_given="no emojis",
        sdpo_metrics=None,
        eval=EvalMetrics(),
        prompt_used="Hello",
    )
    assert sr.sub_step_count == 1


def test_sub_step_count_set():
    """StepResult.sub_step_count can be set explicitly."""
    from claas.eval.types import EvalMetrics

    sr = StepResult(
        preference="no_emoji",
        step=0,
        timestamp="2026-02-19T00:00:00Z",
        feedback_given="no emojis",
        sdpo_metrics=None,
        eval=EvalMetrics(),
        prompt_used="Hello",
        sub_step_count=3,
    )
    assert sr.sub_step_count == 3
