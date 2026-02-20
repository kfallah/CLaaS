"""Tests for Hydra schema-backed loading in claas.core.config."""

from __future__ import annotations

import pytest

from claas.core.config import (
    LocalConfig,
    ModalConfig,
    ProxyConfig,
    TinkerConfig,
    get_proxy_config,
    load_core_config,
    load_proxy_config,
)


class TestLoadCoreConfig:
    def test_load_local(self):
        cfg = load_core_config("local")
        assert isinstance(cfg, LocalConfig)
        assert cfg.mode == "local"
        assert cfg.storage_backend == "local_fs"
        assert cfg.vllm_base_url == "http://127.0.0.1:8000"

    def test_load_tinker(self):
        cfg = load_core_config("tinker")
        assert isinstance(cfg, TinkerConfig)
        assert cfg.mode == "tinker"
        assert cfg.storage_backend == "local_fs"
        assert cfg.tinker_base_model == "gpt-oss/GPT-OSS-120B"

    def test_load_modal(self):
        cfg = load_core_config("modal")
        assert isinstance(cfg, ModalConfig)
        assert cfg.mode == "modal"
        assert cfg.storage_backend == "modal_volume"
        assert cfg.feedback_lock_timeout_s == 120.0

    def test_load_proxy(self):
        cfg = load_proxy_config()
        assert isinstance(cfg, ProxyConfig)
        assert cfg.tinker_base_model == "gpt-oss/GPT-OSS-120B"
        assert cfg.completion_cache_size == 100

    def test_load_nonexistent_raises(self):
        with pytest.raises(ValueError, match="nonexistent_config_xyz"):
            load_core_config("nonexistent_config_xyz")


class TestSelfContainedConfigs:
    def test_local_has_all_shared_fields(self):
        cfg = load_core_config("local")
        assert cfg.lora_root == "/loras"
        assert "Qwen/Qwen3-8B" in cfg.allowed_init_base_models
        assert cfg.feedback_log_dir == "./data/feedback"

    def test_tinker_has_all_shared_fields(self):
        cfg = load_core_config("tinker")
        assert cfg.feedback_log_dir == "./data/feedback"
        assert cfg.lora_root == "/loras"

    def test_modal_has_all_shared_fields(self):
        cfg = load_core_config("modal")
        assert cfg.feedback_log_dir == "./data/feedback"
        assert cfg.lora_root == "/loras"


class TestGetConfig:
    def test_get_local_config_name(self):
        cfg = load_core_config("local")
        assert isinstance(cfg, LocalConfig)
        assert cfg.mode == "local"
        assert cfg.storage_backend == "local_fs"

    def test_get_tinker_config_name(self):
        cfg = load_core_config("tinker")
        assert isinstance(cfg, TinkerConfig)
        assert cfg.mode == "tinker"
        assert cfg.tinker_base_model == "gpt-oss/GPT-OSS-120B"

    def test_get_modal_config_name(self):
        cfg = load_core_config("modal")
        assert isinstance(cfg, ModalConfig)
        assert cfg.mode == "modal"


class TestYamlValues:
    def test_local_yaml_provides_vllm_base_url(self):
        cfg = load_core_config("local")
        assert isinstance(cfg, LocalConfig)
        assert cfg.vllm_base_url == "http://127.0.0.1:8000"

    def test_local_yaml_provides_base_model_id(self):
        cfg = load_core_config("local")
        assert isinstance(cfg, LocalConfig)
        assert cfg.base_model_id == "Qwen/Qwen3-8B"

    def test_local_yaml_provides_attn_implementation(self):
        cfg = load_core_config("local")
        assert isinstance(cfg, LocalConfig)
        assert cfg.attn_implementation == "flash_attention_2"

    def test_tinker_yaml_provides_state_path(self):
        cfg = load_core_config("tinker")
        assert isinstance(cfg, TinkerConfig)
        assert cfg.tinker_state_path == "/data/tinker_state.json"


class TestProxyYamlConfig:
    def test_proxy_defaults_from_yaml(self):
        cfg = get_proxy_config()
        assert isinstance(cfg, ProxyConfig)
        assert cfg.tinker_base_model == "gpt-oss/GPT-OSS-120B"
        assert cfg.completion_cache_size == 100


class TestInvalidConfigName:
    def test_unknown_config_name_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            load_core_config("bogus")
