"""Tests for Hydra-backed core config loading."""

from __future__ import annotations

import pytest

from claas.core.config import (
    LocalConfig,
    ModalConfig,
    TinkerConfig,
    load_core_config,
)


def test_load_local_config() -> None:
    cfg = load_core_config("local")
    assert isinstance(cfg, LocalConfig)
    assert cfg.mode == "local"
    assert cfg.storage_backend == "local_fs"
    assert cfg.vllm_base_url == "http://127.0.0.1:8000"
    assert cfg.base_model_id == "Qwen/Qwen3-8B"


def test_load_modal_config() -> None:
    cfg = load_core_config("modal")
    assert isinstance(cfg, ModalConfig)
    assert cfg.mode == "modal"
    assert cfg.storage_backend == "modal_volume"
    assert cfg.feedback_lock_timeout_s == 120.0


def test_load_tinker_config() -> None:
    cfg = load_core_config("tinker")
    assert isinstance(cfg, TinkerConfig)
    assert cfg.mode == "tinker"
    assert cfg.storage_backend == "local_fs"
    assert cfg.tinker_base_model == "Qwen/Qwen3-30B-A3B"
    assert cfg.tinker_state_path == "/data/tinker_state.json"


def test_core_config_includes_shared_defaults() -> None:
    cfg = load_core_config("local")
    assert cfg.feedback_log_dir == "./data/feedback"
    assert cfg.lora_root == "/loras"
    assert "Qwen/Qwen3-8B" in cfg.allowed_init_base_models


def test_core_config_name_is_case_insensitive() -> None:
    cfg = load_core_config("LOCAL")
    assert isinstance(cfg, LocalConfig)


def test_unknown_core_config_raises() -> None:
    with pytest.raises(ValueError, match="bogus"):
        load_core_config("bogus")


def test_load_core_config_returns_new_instances() -> None:
    first = load_core_config("local")
    second = load_core_config("local")
    assert first is not second


def test_core_config_includes_completion_cache_size() -> None:
    cfg = load_core_config("local")
    assert cfg.completion_cache_size == 100
