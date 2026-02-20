"""Tests for the Hydra YAML config loading in claas.core.config."""

from __future__ import annotations

import pytest

from claas.core.config import (
    LocalConfig,
    ModalConfig,
    ProxyConfig,
    TinkerConfig,
    _load_yaml_config,
    get_config,
    get_proxy_config,
)

# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


class TestLoadYamlConfig:
    def test_load_base(self):
        cfg = _load_yaml_config("base")
        assert cfg["mode"] == "local"
        assert cfg["feedback_log_dir"] == "./data/feedback"

    def test_load_local(self):
        cfg = _load_yaml_config("local")
        assert cfg["mode"] == "local"
        assert cfg["storage_backend"] == "local_fs"
        assert cfg["vllm_base_url"] == "http://127.0.0.1:8000"

    def test_load_tinker(self):
        cfg = _load_yaml_config("tinker")
        assert cfg["mode"] == "tinker"
        assert cfg["storage_backend"] == "local_fs"
        assert cfg["tinker_base_model"] == "gpt-oss/GPT-OSS-120B"

    def test_load_modal(self):
        cfg = _load_yaml_config("modal")
        assert cfg["mode"] == "modal"
        assert cfg["feedback_lock_timeout_s"] == 120.0

    def test_load_proxy(self):
        cfg = _load_yaml_config("proxy")
        assert cfg["tinker_base_model"] == "gpt-oss/GPT-OSS-120B"
        assert cfg["completion_cache_size"] == 100

    def test_load_nonexistent_returns_empty(self):
        cfg = _load_yaml_config("nonexistent_config_xyz")
        assert cfg == {}


# ---------------------------------------------------------------------------
# Inheritance: child YAMLs inherit base values
# ---------------------------------------------------------------------------


class TestYamlInheritance:
    def test_local_inherits_base_defaults(self):
        cfg = _load_yaml_config("local")
        assert cfg["lora_root"] == "/loras"
        assert "Qwen/Qwen3-8B" in cfg["allowed_init_base_models"]  # type: ignore[operator]

    def test_tinker_inherits_base_defaults(self):
        cfg = _load_yaml_config("tinker")
        assert cfg["feedback_log_dir"] == "./data/feedback"

    def test_local_overrides_base_storage_backend(self):
        base = _load_yaml_config("base")
        local = _load_yaml_config("local")
        assert base["storage_backend"] == "modal_volume"
        assert local["storage_backend"] == "local_fs"


# ---------------------------------------------------------------------------
# get_config() with CLAAS_CONFIG_NAME
# ---------------------------------------------------------------------------


class TestGetConfigWithConfigName:
    def test_local_config_name(self, monkeypatch):
        monkeypatch.setenv("CLAAS_CONFIG_NAME", "local")
        cfg = get_config()
        assert isinstance(cfg, LocalConfig)
        assert cfg.mode == "local"
        assert cfg.storage_backend == "local_fs"

    def test_tinker_config_name(self, monkeypatch):
        monkeypatch.setenv("CLAAS_CONFIG_NAME", "tinker")
        cfg = get_config()
        assert isinstance(cfg, TinkerConfig)
        assert cfg.mode == "tinker"
        assert cfg.tinker_base_model == "gpt-oss/GPT-OSS-120B"

    def test_modal_config_name(self, monkeypatch):
        monkeypatch.setenv("CLAAS_CONFIG_NAME", "modal")
        cfg = get_config()
        assert isinstance(cfg, ModalConfig)
        assert cfg.mode == "modal"


# ---------------------------------------------------------------------------
# Env var overrides take precedence over YAML
# ---------------------------------------------------------------------------


class TestEnvVarOverridesYaml:
    def test_vllm_base_url_env_overrides_yaml(self, monkeypatch):
        monkeypatch.setenv("CLAAS_CONFIG_NAME", "local")
        monkeypatch.setenv("VLLM_BASE_URL", "http://custom:9000")
        cfg = get_config()
        assert isinstance(cfg, LocalConfig)
        assert cfg.vllm_base_url == "http://custom:9000"

    def test_storage_backend_env_overrides_yaml(self, monkeypatch):
        monkeypatch.setenv("CLAAS_CONFIG_NAME", "local")
        monkeypatch.setenv("CLAAS_STORAGE_BACKEND", "modal_volume")
        cfg = get_config()
        assert cfg.storage_backend == "modal_volume"

    def test_base_model_id_env_overrides_yaml(self, monkeypatch):
        monkeypatch.setenv("CLAAS_CONFIG_NAME", "local")
        monkeypatch.setenv("CLAAS_BASE_MODEL_ID", "my/custom-model")
        cfg = get_config()
        assert isinstance(cfg, LocalConfig)
        assert cfg.base_model_id == "my/custom-model"

    def test_tinker_base_model_env_overrides_yaml(self, monkeypatch):
        monkeypatch.setenv("CLAAS_CONFIG_NAME", "tinker")
        monkeypatch.setenv("CLAAS_TINKER_BASE_MODEL", "my/other-model")
        cfg = get_config()
        assert isinstance(cfg, TinkerConfig)
        assert cfg.tinker_base_model == "my/other-model"


# ---------------------------------------------------------------------------
# YAML values are used when env vars are absent
# ---------------------------------------------------------------------------


class TestYamlDefaults:
    def test_local_yaml_provides_vllm_base_url(self, monkeypatch):
        monkeypatch.setenv("CLAAS_CONFIG_NAME", "local")
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        cfg = get_config()
        assert isinstance(cfg, LocalConfig)
        assert cfg.vllm_base_url == "http://127.0.0.1:8000"

    def test_local_yaml_provides_base_model_id(self, monkeypatch):
        monkeypatch.setenv("CLAAS_CONFIG_NAME", "local")
        monkeypatch.delenv("CLAAS_BASE_MODEL_ID", raising=False)
        cfg = get_config()
        assert isinstance(cfg, LocalConfig)
        assert cfg.base_model_id == "Qwen/Qwen3-8B"

    def test_local_yaml_provides_attn_implementation(self, monkeypatch):
        monkeypatch.setenv("CLAAS_CONFIG_NAME", "local")
        monkeypatch.delenv("CLAAS_ATTN_IMPLEMENTATION", raising=False)
        cfg = get_config()
        assert isinstance(cfg, LocalConfig)
        assert cfg.attn_implementation == "sdpa"

    def test_tinker_yaml_provides_tinker_state_path(self, monkeypatch):
        monkeypatch.setenv("CLAAS_CONFIG_NAME", "tinker")
        monkeypatch.delenv("CLAAS_TINKER_STATE_PATH", raising=False)
        cfg = get_config()
        assert isinstance(cfg, TinkerConfig)
        assert cfg.tinker_state_path == "~/.claas/tinker_state.json"


# ---------------------------------------------------------------------------
# Proxy config
# ---------------------------------------------------------------------------


class TestProxyYamlConfig:
    def test_proxy_defaults_from_yaml(self):
        cfg = get_proxy_config()
        assert isinstance(cfg, ProxyConfig)
        assert cfg.tinker_base_model == "gpt-oss/GPT-OSS-120B"
        assert cfg.completion_cache_size == 100

    def test_proxy_env_override(self, monkeypatch):
        monkeypatch.setenv("CLAAS_COMPLETION_CACHE_SIZE", "42")
        cfg = get_proxy_config()
        assert cfg.completion_cache_size == 42


# ---------------------------------------------------------------------------
# Invalid config name
# ---------------------------------------------------------------------------


class TestInvalidConfigName:
    def test_unknown_config_name_raises(self, monkeypatch):
        monkeypatch.setenv("CLAAS_CONFIG_NAME", "bogus")
        with pytest.raises(ValueError, match="bogus"):
            get_config()
