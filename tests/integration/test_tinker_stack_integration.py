"""Integration tests against the real Tinker docker stack services."""

from __future__ import annotations

import os
import uuid

import httpx
import pytest

pytestmark = pytest.mark.integration


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------


class TestTinkerStackReachability:
    """Verify that all Tinker stack services are reachable."""

    def test_tinker_proxy_models(self) -> None:
        proxy_url = _require_env("INTEGRATION_TINKER_PROXY_URL")
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{proxy_url}/v1/models")
        assert resp.status_code == 200

    def test_claas_api_root(self) -> None:
        claas_url = _require_env("INTEGRATION_CLAAS_BASE_URL")
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{claas_url}/")
        assert resp.status_code == 200

    def test_claas_api_lora_list(self) -> None:
        claas_url = _require_env("INTEGRATION_CLAAS_BASE_URL")
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{claas_url}/v1/lora")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Full lifecycle round-trip
# ---------------------------------------------------------------------------


class TestTinkerStackRoundTrip:
    """End-to-end LoRA lifecycle: init → list → inference → distill → delete."""

    def test_full_lifecycle(self) -> None:
        proxy_url = _require_env("INTEGRATION_TINKER_PROXY_URL")
        claas_url = _require_env("INTEGRATION_CLAAS_BASE_URL")

        suffix = uuid.uuid4().hex[:8]
        lora_id = f"test/integration-{suffix}"

        with httpx.Client(timeout=300.0) as client:
            try:
                # 1. Init LoRA
                init_resp = client.post(
                    f"{claas_url}/v1/lora/init",
                    json={
                        "lora_id": lora_id,
                        "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                        "lora_r": 8,
                        "lora_alpha": 16,
                    },
                    timeout=60.0,
                )
                assert init_resp.status_code == 200, init_resp.text
                assert init_resp.json()["lora_id"] == lora_id

                # 2. Verify it appears in list
                list_resp = client.get(
                    f"{claas_url}/v1/lora",
                    params={"prefix": "test/"},
                    timeout=15.0,
                )
                assert list_resp.status_code == 200
                assert lora_id in list_resp.json()["loras"]

                # 3. Inference through tinker-proxy
                chat_resp = client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json={
                        "model": "Qwen/Qwen3-4B-Instruct-2507",
                        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
                        "max_tokens": 64,
                    },
                    timeout=120.0,
                )
                assert chat_resp.status_code == 200, chat_resp.text
                choices = chat_resp.json()["choices"]
                assert len(choices) > 0
                assert len(choices[0]["message"]["content"]) > 0

                # 4. Distill via feedback endpoint (teacher_mode=self)
                feedback_resp = client.post(
                    f"{claas_url}/v1/feedback",
                    json={
                        "lora_id": lora_id,
                        "prompt": "Say hello in one sentence.",
                        "response": choices[0]["message"]["content"],
                        "feedback": "Good, concise greeting.",
                        "training": {"teacher_mode": "self"},
                        "orchestration": {
                            "sleep_before": False,
                            "wake_after": False,
                        },
                    },
                    timeout=300.0,
                )
                assert feedback_resp.status_code == 200, feedback_resp.text
                fb_data = feedback_resp.json()
                assert fb_data["status"] == "ok"
                assert fb_data["distill_result"] is not None
                step = fb_data["distill_result"]["metadata"].get("step")
                assert step is not None and step >= 1

            finally:
                # 5. Cleanup — always delete even on assertion failure
                delete_resp = client.delete(
                    f"{claas_url}/v1/lora",
                    params={"lora_id": lora_id},
                    timeout=30.0,
                )
                assert delete_resp.status_code == 200, delete_resp.text
                assert delete_resp.json()["deleted"] is True

                # Verify it's gone
                verify_resp = client.get(
                    f"{claas_url}/v1/lora",
                    params={"prefix": "test/"},
                    timeout=15.0,
                )
                if verify_resp.status_code == 200:
                    assert lora_id not in verify_resp.json()["loras"]
