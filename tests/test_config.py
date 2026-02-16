"""Tests for the centralized config module."""

from __future__ import annotations

import pytest

from claas.core.config import (
    LocalConfig,
    ModalConfig,
    ProxyConfig,
    TinkerConfig,
    _env_bool,
    _env_set,
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
        monkeypatch.setenv("CLAAS_DISTILL_EXECUTION_MODE", "local")
        cfg = get_config()
        assert isinstance(cfg, LocalConfig)

    def test_modal_mode(self, monkeypatch):
        monkeypatch.setenv("CLAAS_DISTILL_EXECUTION_MODE", "modal")
        cfg = get_config()
        assert isinstance(cfg, ModalConfig)

    def test_tinker_mode(self, monkeypatch):
        monkeypatch.setenv("CLAAS_DISTILL_EXECUTION_MODE", "tinker")
        cfg = get_config()
        assert isinstance(cfg, TinkerConfig)

    def test_unknown_mode_raises(self, monkeypatch):
        monkeypatch.setenv("CLAAS_DISTILL_EXECUTION_MODE", "bogus")
        with pytest.raises(ValueError, match="bogus"):
            get_config()


# ---------------------------------------------------------------------------
# Env var overrides
# ---------------------------------------------------------------------------

class TestEnvVarOverrides:
    def test_feedback_log_dir(self, monkeypatch):
        monkeypatch.setenv("FEEDBACK_LOG_DIR", "/tmp/logs")
        cfg = get_config()
        assert cfg.feedback_log_dir == "/tmp/logs"

    def test_vllm_base_url_local(self, monkeypatch):
        monkeypatch.setenv("VLLM_BASE_URL", "http://gpu:9000")
        cfg = get_config()
        assert isinstance(cfg, LocalConfig)
        assert cfg.vllm_base_url == "http://gpu:9000"

    def test_tinker_api_key(self, monkeypatch):
        monkeypatch.setenv("CLAAS_DISTILL_EXECUTION_MODE", "tinker")
        monkeypatch.setenv("CLAAS_TINKER_API_KEY", "tk-secret")
        cfg = get_config()
        assert isinstance(cfg, TinkerConfig)
        assert cfg.tinker_api_key == "tk-secret"

    def test_allowed_models_override(self, monkeypatch):
        monkeypatch.setenv("CLAAS_ALLOWED_INIT_BASE_MODELS", "A/B,C/D")
        cfg = get_config()
        assert cfg.allowed_init_base_models == frozenset({"A/B", "C/D"})


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
# Env helpers
# ---------------------------------------------------------------------------

class TestEnvBool:
    @pytest.mark.parametrize("val,expected", [
        ("1", True),
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
        ("random", False),
    ])
    def test_values(self, monkeypatch, val, expected):
        monkeypatch.setenv("_TEST_BOOL", val)
        assert _env_bool("_TEST_BOOL", False) is expected

    def test_default_when_missing(self):
        assert _env_bool("_NONEXISTENT_BOOL_VAR", True) is True
        assert _env_bool("_NONEXISTENT_BOOL_VAR", False) is False


class TestEnvSet:
    def test_comma_separated(self, monkeypatch):
        monkeypatch.setenv("_TEST_SET", "a, b ,c")
        result = _env_set("_TEST_SET", "default")
        assert result == frozenset({"a", "b", "c"})

    def test_empty_items_filtered(self, monkeypatch):
        monkeypatch.setenv("_TEST_SET", "a,,b,")
        result = _env_set("_TEST_SET", "default")
        assert result == frozenset({"a", "b"})

    def test_default(self):
        result = _env_set("_NONEXISTENT_SET_VAR", "x,y")
        assert result == frozenset({"x", "y"})


# ---------------------------------------------------------------------------
# Proxy config
# ---------------------------------------------------------------------------

class TestProxyConfig:
    def test_defaults(self):
        cfg = get_proxy_config()
        assert isinstance(cfg, ProxyConfig)
        assert cfg.tinker_base_model == "gpt-oss/GPT-OSS-120B"
        assert cfg.completion_cache_size == 100

    def test_cache_size_override(self, monkeypatch):
        monkeypatch.setenv("CLAAS_COMPLETION_CACHE_SIZE", "50")
        cfg = get_proxy_config()
        assert cfg.completion_cache_size == 50


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
