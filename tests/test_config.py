"""Tests for claas.config module."""

from __future__ import annotations

from claas.config import DEFAULT_CONFIG, CLaaSConfig, get_config


def test_config_defaults_and_accessors():
    """Verify config hierarchy instantiates correctly with sensible defaults."""
    cfg = get_config()
    assert cfg is DEFAULT_CONFIG
    assert isinstance(cfg, CLaaSConfig)

    # Spot-check a few values across each sub-config
    assert cfg.models.student_model_id == "Qwen/Qwen3-8B"
    assert cfg.models.lora_r == 32
    assert len(cfg.models.lora_target_modules) == 7

    assert cfg.training.alpha == 0.5
    assert cfg.training.learning_rate == 1e-4
    assert cfg.training.teacher_top_k == 100

    assert cfg.infra.teacher_gpu == "H100"
    assert cfg.infra.model_volume_name == "claas-models"

    # Custom overrides work
    custom = CLaaSConfig()
    custom.models.lora_r = 8
    assert custom.models.lora_r == 8
