"""Tests for the centralized Hydra config module."""

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


class TestGetConfig:
    def test_local_mode(self):
        cfg = get_config("local")
        assert isinstance(cfg, LocalConfig)
        assert cfg.mode == "local"

    def test_modal_mode(self):
        cfg = get_config("modal")
        assert isinstance(cfg, ModalConfig)
        assert cfg.mode == "modal"

    def test_tinker_mode(self):
        cfg = get_config("tinker")
        assert isinstance(cfg, TinkerConfig)
        assert cfg.mode == "tinker"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            get_config("bogus")

    def test_config_name_case_insensitive(self):
        cfg = get_config("LOCAL")
        assert isinstance(cfg, LocalConfig)


class TestMutability:
    def test_base_config_mutable(self):
        cfg = get_config("local")
        cfg.mode = "modal"
        assert cfg.mode == "modal"

    def test_proxy_config_mutable(self):
        cfg = get_proxy_config()
        cfg.tinker_base_model = "gpt-oss/GPT-OSS-20B"
        assert cfg.tinker_base_model == "gpt-oss/GPT-OSS-20B"


class TestProxyConfig:
    def test_defaults(self):
        cfg = get_proxy_config()
        assert isinstance(cfg, ProxyConfig)
        assert cfg.tinker_base_model == "gpt-oss/GPT-OSS-120B"
        assert cfg.completion_cache_size == 100


class TestCacheClearing:
    def test_get_config_returns_same_instance(self):
        a = get_config("local")
        b = get_config("local")
        assert a is b

    def test_cache_clear_returns_new_instance(self):
        a = get_config("local")
        get_config.cache_clear()
        b = get_config("local")
        assert a is not b

