"""Shared test fixtures for CLaaS."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from claas.api import configure_web_app
from claas.core.config import load_core_config
from claas.training.storage import configure_storage_backend


def _configure():
    """Configure the web app with tokenizer loading disabled.

    Patches AutoTokenizer.from_pretrained so VllmBackend.__init__
    does not hit the HuggingFace API during unit tests.
    """
    with patch(
        "transformers.AutoTokenizer.from_pretrained",
        side_effect=ImportError("mocked for tests"),
    ):
        configure_storage_backend("local_fs")
        configure_web_app(load_core_config("local"))


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """Reset process-local runtime config around each test."""
    _configure()
    yield
    _configure()
