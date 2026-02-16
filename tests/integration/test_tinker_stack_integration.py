"""Integration tests against the real Tinker docker stack services.

All tests receive the ``tinker_stack`` session fixture (see conftest.py)
which brings up and tears down the Docker stack automatically.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING

import httpx
import pytest

from claas.teacher import build_teacher_messages

if TYPE_CHECKING:
    from conftest import TinkerStack

pytestmark = pytest.mark.integration

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------


class TestTinkerStackReachability:
    """Verify that all Tinker stack services are reachable."""

    def test_tinker_proxy_models(self, tinker_stack: TinkerStack) -> None:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{tinker_stack.proxy_url}/v1/models")
        assert resp.status_code == 200

    def test_claas_api_root(self, tinker_stack: TinkerStack) -> None:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{tinker_stack.claas_url}/")
        assert resp.status_code == 200

    def test_claas_api_lora_list(self, tinker_stack: TinkerStack) -> None:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{tinker_stack.claas_url}/v1/lora")
        assert resp.status_code == 200

    def test_openclaw_reachable(self, tinker_stack: TinkerStack) -> None:
        """Verify OpenClaw responds to a chat completion via its HTTP API."""
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{tinker_stack.openclaw_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {tinker_stack.openclaw_token}",
                    "x-openclaw-agent-id": "main",
                },
                json={
                    "model": "openclaw",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
            )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Full lifecycle round-trip (direct via tinker-proxy)
# ---------------------------------------------------------------------------


class TestTinkerStackRoundTrip:
    """End-to-end LoRA lifecycle: init -> list -> inference -> distill -> delete."""

    def test_full_lifecycle(self, tinker_stack: TinkerStack) -> None:
        proxy_url = tinker_stack.proxy_url
        claas_url = tinker_stack.claas_url

        suffix = uuid.uuid4().hex[:8]
        lora_id = f"test/integration-{suffix}"
        user_prompt = "Say hello in one sentence."
        feedback_text = "Good, concise greeting."

        with httpx.Client(timeout=300.0) as client:
            created = False
            try:
                # 1. Init LoRA
                init_payload = {
                    "lora_id": lora_id,
                    "base_model": tinker_stack.model,
                    "lora_r": 8,
                    "lora_alpha": 16,
                }
                logger.info("POST /v1/lora/init:\n%s", json.dumps(init_payload, indent=2))
                init_resp = client.post(
                    f"{claas_url}/v1/lora/init",
                    json=init_payload,
                    timeout=60.0,
                )
                assert init_resp.status_code == 200, init_resp.text
                assert init_resp.json()["lora_id"] == lora_id
                logger.info("Init response: %s", init_resp.text)
                created = True

                # 2. Verify it appears in list
                list_resp = client.get(
                    f"{claas_url}/v1/lora",
                    params={"prefix": "test/"},
                    timeout=15.0,
                )
                assert list_resp.status_code == 200
                assert lora_id in list_resp.json()["loras"]

                # 3. Inference through tinker-proxy
                chat_messages = [{"role": "user", "content": user_prompt}]
                chat_payload = {
                    "model": tinker_stack.model,
                    "messages": chat_messages,
                    "max_tokens": 64,
                }
                logger.info(
                    "POST /v1/chat/completions:\n%s", json.dumps(chat_payload, indent=2)
                )
                chat_resp = client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json=chat_payload,
                    timeout=120.0,
                )
                assert chat_resp.status_code == 200, chat_resp.text
                choices = chat_resp.json()["choices"]
                assert len(choices) > 0
                response_content = choices[0]["message"]["content"]
                assert len(response_content) > 0
                logger.info("Model response: %s", response_content)

                # 4. Distill via feedback endpoint (teacher_mode=self)
                teacher_messages = build_teacher_messages(user_prompt, feedback_text)
                logger.info(
                    "Teacher messages (built by engine for self-distillation):\n%s",
                    json.dumps(teacher_messages, indent=2),
                )

                feedback_payload = {
                    "lora_id": lora_id,
                    "prompt": user_prompt,
                    "response": response_content,
                    "feedback": feedback_text,
                    "training": {"teacher_mode": "self"},
                    "orchestration": {
                        "sleep_before": False,
                        "wake_after": False,
                    },
                }
                logger.info(
                    "POST /v1/feedback:\n%s", json.dumps(feedback_payload, indent=2)
                )
                feedback_resp = client.post(
                    f"{claas_url}/v1/feedback",
                    json=feedback_payload,
                    timeout=300.0,
                )
                assert feedback_resp.status_code == 200, feedback_resp.text
                fb_data = feedback_resp.json()
                logger.info(
                    "Feedback response:\n%s", json.dumps(fb_data, indent=2)
                )
                assert fb_data["status"] == "ok"
                assert fb_data["distill_result"] is not None
                step = fb_data["distill_result"]["metadata"].get("step")
                assert step is not None and step >= 1

                # Verify tinker-proxy was refreshed with new weights
                assert fb_data["vllm"]["woke"] is True, (
                    "Expected tinker-proxy refresh after distillation"
                )

                # 5. Verify the proxy switched to the distilled LoRA
                expected_path = fb_data["distill_result"]["metadata"][
                    "sampler_weights_path"
                ]
                status_resp = client.get(
                    f"{proxy_url}/v1/sampler/status",
                    timeout=15.0,
                )
                assert status_resp.status_code == 200
                actual_path = status_resp.json()["model_path"]
                logger.info("Post-distillation sampler model_path: %s", actual_path)
                assert actual_path == expected_path, (
                    f"Proxy should be serving distilled LoRA {expected_path!r}, "
                    f"got {actual_path!r}"
                )

                # 6. Post-distillation health: inference still works
                post_payload = {
                    "model": tinker_stack.model,
                    "messages": [{"role": "user", "content": "Say hi."}],
                    "max_tokens": 32,
                }
                logger.info("Post-distillation inference check ...")
                post_resp = client.post(
                    f"{proxy_url}/v1/chat/completions",
                    json=post_payload,
                    timeout=120.0,
                )
                assert post_resp.status_code == 200, post_resp.text
                post_choices = post_resp.json()["choices"]
                assert len(post_choices) > 0
                assert len(post_choices[0]["message"]["content"]) > 0
                logger.info(
                    "Post-distillation response: %s",
                    post_choices[0]["message"]["content"],
                )

            finally:
                # 7. Cleanup — best-effort to avoid masking root failures
                if created:
                    try:
                        delete_resp = client.delete(
                            f"{claas_url}/v1/lora",
                            params={"lora_id": lora_id},
                            timeout=30.0,
                        )
                        if delete_resp.status_code == 200 and delete_resp.json().get("deleted"):
                            logger.info("Deleted %s: %s", lora_id, delete_resp.text)
                        else:
                            logger.warning("Cleanup failed for %s: %s", lora_id, delete_resp.text)
                    except Exception:
                        logger.warning("Cleanup request failed for %s", lora_id, exc_info=True)


# ---------------------------------------------------------------------------
# End-to-end through OpenClaw gateway
# ---------------------------------------------------------------------------


class TestOpenClawEndToEnd:
    """Chat via OpenClaw HTTP API -> tinker-proxy -> feedback -> distillation.

    OpenClaw exposes an OpenAI-compatible ``/v1/chat/completions`` HTTP
    endpoint (enabled via ``gateway.http.endpoints.chatCompletions``).
    Requests go through the full context engine (persona, skills,
    workspace files) just like a Telegram session.
    """

    def test_chat_and_feedback_via_openclaw(self, tinker_stack: TinkerStack) -> None:
        claas_url = tinker_stack.claas_url

        suffix = uuid.uuid4().hex[:8]
        lora_id = f"test/openclaw-e2e-{suffix}"
        user_prompt = "What is 2 + 2? Answer in one word."
        feedback_text = "Correct and concise."

        openclaw_headers = {
            "Authorization": f"Bearer {tinker_stack.openclaw_token}",
            "x-openclaw-agent-id": "main",
        }

        with httpx.Client(timeout=300.0) as client:
            created = False
            try:
                # 1. Init LoRA via CLaaS API
                init_payload = {
                    "lora_id": lora_id,
                    "base_model": tinker_stack.model,
                    "lora_r": 8,
                    "lora_alpha": 16,
                }
                logger.info("POST /v1/lora/init:\n%s", json.dumps(init_payload, indent=2))
                init_resp = client.post(
                    f"{claas_url}/v1/lora/init",
                    json=init_payload,
                    timeout=60.0,
                )
                assert init_resp.status_code == 200, init_resp.text
                logger.info("Init response: %s", init_resp.text)
                created = True

                # 2. Chat through OpenClaw (HTTP API -> context engine -> tinker-proxy)
                chat_payload = {
                    "model": "openclaw",
                    "messages": [{"role": "user", "content": user_prompt}],
                    "max_tokens": 64,
                }
                logger.info(
                    "POST %s/v1/chat/completions (via OpenClaw):\n%s",
                    tinker_stack.openclaw_url,
                    json.dumps(chat_payload, indent=2),
                )
                chat_resp = client.post(
                    f"{tinker_stack.openclaw_url}/v1/chat/completions",
                    json=chat_payload,
                    headers=openclaw_headers,
                    timeout=120.0,
                )
                assert chat_resp.status_code == 200, chat_resp.text
                choices = chat_resp.json()["choices"]
                assert len(choices) > 0
                response_content = choices[0]["message"]["content"]
                assert len(response_content) > 0
                logger.info("OpenClaw response: %s", response_content[:500])

                # 3. Distill via feedback endpoint
                teacher_messages = build_teacher_messages(user_prompt, feedback_text)
                logger.info(
                    "Teacher messages:\n%s",
                    json.dumps(teacher_messages, indent=2),
                )

                feedback_payload = {
                    "lora_id": lora_id,
                    "prompt": user_prompt,
                    "response": response_content,
                    "feedback": feedback_text,
                    "training": {"teacher_mode": "self"},
                    "orchestration": {
                        "sleep_before": False,
                        "wake_after": False,
                    },
                }
                logger.info(
                    "POST /v1/feedback:\n%s", json.dumps(feedback_payload, indent=2)
                )
                feedback_resp = client.post(
                    f"{claas_url}/v1/feedback",
                    json=feedback_payload,
                    timeout=300.0,
                )
                assert feedback_resp.status_code == 200, feedback_resp.text
                fb_data = feedback_resp.json()
                logger.info(
                    "Feedback response:\n%s", json.dumps(fb_data, indent=2)
                )
                assert fb_data["status"] == "ok"
                assert fb_data["distill_result"] is not None
                step = fb_data["distill_result"]["metadata"].get("step")
                assert step is not None and step >= 1

                # Verify tinker-proxy was refreshed with new weights
                assert fb_data["vllm"]["woke"] is True, (
                    "Expected tinker-proxy refresh after distillation"
                )

                # Verify the proxy switched to the distilled LoRA
                expected_path = fb_data["distill_result"]["metadata"][
                    "sampler_weights_path"
                ]
                post_status = client.get(
                    f"{tinker_stack.proxy_url}/v1/sampler/status",
                    timeout=15.0,
                )
                assert post_status.status_code == 200
                actual_path = post_status.json()["model_path"]
                logger.info("Post-distillation sampler model_path: %s", actual_path)
                assert actual_path == expected_path, (
                    f"Proxy should be serving distilled LoRA {expected_path!r}, "
                    f"got {actual_path!r}"
                )

                # 4. Post-distillation health: OpenClaw still works
                logger.info("Post-distillation OpenClaw inference check ...")
                post_resp = client.post(
                    f"{tinker_stack.openclaw_url}/v1/chat/completions",
                    json={
                        "model": "openclaw",
                        "messages": [{"role": "user", "content": "Say goodbye."}],
                        "max_tokens": 32,
                    },
                    headers=openclaw_headers,
                    timeout=120.0,
                )
                assert post_resp.status_code == 200, post_resp.text
                post_choices = post_resp.json()["choices"]
                assert len(post_choices) > 0
                assert len(post_choices[0]["message"]["content"]) > 0
                logger.info(
                    "Post-distillation OpenClaw response: %s",
                    post_choices[0]["message"]["content"][:200],
                )

            finally:
                # 5. Cleanup — best-effort to avoid masking root failures
                if created:
                    try:
                        delete_resp = client.delete(
                            f"{claas_url}/v1/lora",
                            params={"lora_id": lora_id},
                            timeout=30.0,
                        )
                        if delete_resp.status_code == 200 and delete_resp.json().get("deleted"):
                            logger.info("Deleted %s: %s", lora_id, delete_resp.text)
                        else:
                            logger.warning("Cleanup failed for %s: %s", lora_id, delete_resp.text)
                    except Exception:
                        logger.warning("Cleanup request failed for %s", lora_id, exc_info=True)
