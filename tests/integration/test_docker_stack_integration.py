"""Integration tests against the real docker stack services."""

from __future__ import annotations

import os

import httpx
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.gpu]

VLLM_API_KEY = "sk-local"


def _require_env(name: str) -> str:
    """Read a required environment variable.

    Args:
        name: Variable name.

    Returns:
        Variable value.

    Raises:
        RuntimeError: If the variable is missing.
    """
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def test_stack_services_are_reachable() -> None:
    """Validate that vLLM and CLaaS API endpoints are reachable."""
    vllm_base = _require_env("INTEGRATION_VLLM_BASE_URL")
    claas_base = _require_env("INTEGRATION_CLAAS_BASE_URL")

    with httpx.Client(timeout=10.0) as client:
        vllm_health = client.get(f"{vllm_base}/health")
        assert vllm_health.status_code == 200

        vllm_models = client.get(
            f"{vllm_base}/v1/models",
            headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
        )
        assert vllm_models.status_code == 200

        claas_root = client.get(f"{claas_base}/")
        assert claas_root.status_code == 200

        lora_list = client.get(f"{claas_base}/v1/lora")
        assert lora_list.status_code == 200


def test_stack_feedback_round_trip() -> None:
    """Validate CLaaS feedback behavior using the real stack."""
    claas_base = _require_env("INTEGRATION_CLAAS_BASE_URL")

    with httpx.Client(timeout=20.0) as client:
        response = client.post(
            f"{claas_base}/v1/feedback",
            json={
                "lora_id": "openclaw/assistant-latest",
                "prompt": "hello",
                "response": "hi there",
                "feedback": "good",
                "training": {"teacher_mode": "self"},
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["lora_id"] == "openclaw/assistant-latest"
