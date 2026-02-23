"""Shared test fixtures for CLaaS."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from claas.api import configure_web_app
from claas.core.config import load_core_config
from claas.training.storage import configure_storage_backend


def _configure():
    """Configure the web app with tokenizer loading disabled.

    Patches AutoTokenizer.from_pretrained so VllmBackend.__init__
    does not hit the HuggingFace API during unit tests.

    When ``transformers`` is not installed (GPU-only dependency), we
    inject a lightweight stub module so that ``unittest.mock.patch``
    can resolve the dotted target path.
    """
    _need_stub = "transformers" not in sys.modules
    if _need_stub:
        stub = MagicMock()
        sys.modules["transformers"] = stub

    try:
        with patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=ImportError("mocked for tests"),
        ):
            configure_storage_backend("local_fs")
            configure_web_app(load_core_config("local"))
    finally:
        if _need_stub:
            sys.modules.pop("transformers", None)


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """Reset process-local runtime config around each test."""
    _configure()
    yield
    _configure()
