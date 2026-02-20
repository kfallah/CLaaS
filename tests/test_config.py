"""Tests for the centralized Hydra config module."""

from __future__ import annotations

import pytest

from claas.core.config import (
    LocalConfig,
    ModalConfig,
    ProxyConfig,
    TinkerConfig,
    get_proxy_config,
    load_core_config,
)


class TestLoadCoreConfig:
    def test_local_mode(self):
        cfg = load_core_config("local")
        assert isinstance(cfg, LocalConfig)
        assert cfg.mode == "local"

    def test_modal_mode(self):
        cfg = load_core_config("modal")
        assert isinstance(cfg, ModalConfig)
        assert cfg.mode == "modal"

    def test_tinker_mode(self):
        cfg = load_core_config("tinker")
        assert isinstance(cfg, TinkerConfig)
        assert cfg.mode == "tinker"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            load_core_config("bogus")

    def test_config_name_case_insensitive(self):
        cfg = load_core_config("LOCAL")
        assert isinstance(cfg, LocalConfig)


class TestMutability:
    def test_base_config_mutable(self):
        cfg = load_core_config("local")
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


class TestLoadSemantics:
    def test_load_core_config_returns_new_instances(self):
        a = load_core_config("local")
        b = load_core_config("local")
        assert a is not b
