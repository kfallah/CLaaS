"""Shared fixtures for integration tests.

The Docker stack is managed by ``tests/integration/run_tinker_stack_integration.sh``,
not by pytest.  This conftest just provides connection details.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class TinkerStack:
    """Connection details for a running Tinker Docker stack."""

    proxy_url: str
    claas_url: str
    openclaw_url: str
    openclaw_token: str
    model: str


@pytest.fixture(scope="session")
def tinker_stack():
    """Provide connection details for the Tinker Docker stack."""
    return TinkerStack(
        proxy_url="http://127.0.0.1:8000",
        claas_url="http://127.0.0.1:8080",
        openclaw_url="http://127.0.0.1:18789",
        openclaw_token="openclaw-local-dev-token",
        model="meta-llama/Llama-3.2-1B",
    )
