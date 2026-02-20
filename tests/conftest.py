"""Shared test fixtures for CLaaS."""

from __future__ import annotations

import pytest

from claas.core.config import get_config, get_proxy_config
from claas.api import configure_web_app
from claas.core.config import load_core_config
from claas.training.storage import configure_storage_backend


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """Clear cached config before and after each test.

    This ensures cached Hydra configs don't leak between tests.
    """
    configure_storage_backend("local_fs")
    configure_web_app(load_core_config("local"))
    get_config.cache_clear()
    get_proxy_config.cache_clear()
    yield
    configure_storage_backend("local_fs")
    configure_web_app(load_core_config("local"))
    get_config.cache_clear()
    get_proxy_config.cache_clear()
