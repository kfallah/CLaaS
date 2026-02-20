"""Shared test fixtures for CLaaS."""

from __future__ import annotations

import pytest

from claas.api import configure_web_app
from claas.core.config import get_proxy_config, load_core_config
from claas.training.storage import configure_storage_backend


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """Reset process-local runtime config and proxy config cache around each test."""
    configure_storage_backend("local_fs")
    configure_web_app(load_core_config("local"))
    get_proxy_config.cache_clear()
    yield
    configure_storage_backend("local_fs")
    configure_web_app(load_core_config("local"))
    get_proxy_config.cache_clear()
