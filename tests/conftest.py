"""Shared test fixtures for CLaaS."""

from __future__ import annotations

import pytest

from claas.core.config import get_config, get_proxy_config


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """Clear cached config before and after each test.

    This ensures monkeypatched env vars take effect and don't leak
    between tests.
    """
    get_config.cache_clear()
    get_proxy_config.cache_clear()
    yield
    get_config.cache_clear()
    get_proxy_config.cache_clear()
