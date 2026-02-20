"""Tests for the centralized config module."""

from __future__ import annotations

import pytest

from claas.core.config import (
    LocalConfig,
    ModalConfig,
    ProxyConfig,
    TinkerConfig,
    get_config,
    get_proxy_config,
)

# ---------------------------------------------------------------------------
# Factory returns correct subclass
# ---------------------------------------------------------------------------

class TestGetConfig:
    def test_default_is_local(self):
        cfg = get_config()
        assert isinstance(cfg, LocalConfig)
        assert cfg.mode == "local"

    def test_local_mode(self, monkeypatch):
        monkeypatch.setenv("CLAAS_CONFIG_NAME", "local")
        cfg = get_config()
        assert isinstance(cfg, LocalConfig)

    def test_modal_mode(self, monkeypatch):
        monkeypatch.setenv("CLAAS_CONFIG_NAME", "modal")
        cfg = get_config()
        assert isinstance(cfg, ModalConfig)

    def test_tinker_mode(self, monkeypatch):
        monkeypatch.setenv("CLAAS_CONFIG_NAME", "tinker")
        cfg = get_config()
        assert isinstance(cfg, TinkerConfig)

    def test_unknown_mode_raises(self, monkeypatch):
        monkeypatch.setenv("CLAAS_CONFIG_NAME", "bogus")
        with pytest.raises(ValueError, match="bogus"):
            get_config()

    def test_config_name_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("CLAAS_CONFIG_NAME", "LOCAL")
        cfg = get_config()
        assert isinstance(cfg, LocalConfig)


# ---------------------------------------------------------------------------
# Secret env vars still work
# ---------------------------------------------------------------------------

class TestSecretEnvVars:
    def test_tinker_api_key(self, monkeypatch):
        monkeypatch.setenv("CLAAS_CONFIG_NAME", "tinker")
        monkeypatch.setenv("CLAAS_TINKER_API_KEY", "tk-secret")
        cfg = get_config()
        assert isinstance(cfg, TinkerConfig)
        assert cfg.tinker_api_key == "tk-secret"

    def test_vllm_api_key(self, monkeypatch):
        monkeypatch.setenv("VLLM_API_KEY", "sk-test")
        cfg = get_config()
        assert isinstance(cfg, LocalConfig)
        assert cfg.vllm_api_key == "sk-test"

    def test_hf_token(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_abc")
        cfg = get_config()
        assert cfg.hf_token == "hf_abc"


# ---------------------------------------------------------------------------
# Frozen enforcement
# ---------------------------------------------------------------------------

class TestFrozen:
    def test_base_config_frozen(self):
        cfg = get_config()
        with pytest.raises(AttributeError):
            cfg.mode = "modal"  # type: ignore[misc]

    def test_proxy_config_frozen(self):
        cfg = get_proxy_config()
        with pytest.raises(AttributeError):
            cfg.tinker_api_key = "new"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Proxy config
# ---------------------------------------------------------------------------

class TestProxyConfig:
    def test_defaults(self):
        cfg = get_proxy_config()
        assert isinstance(cfg, ProxyConfig)
        assert cfg.tinker_base_model == "gpt-oss/GPT-OSS-120B"
        assert cfg.completion_cache_size == 100


# ---------------------------------------------------------------------------
# Cache clearing
# ---------------------------------------------------------------------------

class TestCacheClearing:
    def test_get_config_returns_same_instance(self):
        a = get_config()
        b = get_config()
        assert a is b

    def test_cache_clear_returns_new_instance(self):
        a = get_config()
        get_config.cache_clear()
        b = get_config()
        assert a is not b
