"""Integration tests against the real Tinker docker stack services.

All tests receive the ``tinker_stack`` session fixture (see conftest.py)
which brings up and tears down the Docker stack automatically.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from typing import TYPE_CHECKING

import httpx
import pytest

from claas.training.teacher_helpers import build_teacher_messages

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

if TYPE_CHECKING:
    from conftest import TinkerStack

pytestmark = pytest.mark.integration

logger = logging.getLogger(__name__)


def _content_hash(response_content: str) -> str:
    """Compute the SHA-256 content hash matching the API's cache key.

    Mirrors the OpenClaw feedback plugin: strip ``<think>`` blocks,
    strip orphaned ``</think>`` prefixes, then SHA-256 the visible text.
    """
    visible = _THINK_RE.sub("", response_content)
    idx = visible.find("</think>")
    if idx >= 0:
        visible = visible[idx + len("</think>"):]
    visible = visible.strip()
    return hashlib.sha256(visible.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Reachability
# ---------------------------------------------------------------------------


class TestTinkerStackReachability:
    """Verify that all Tinker stack services are reachable."""

    def test_claas_api_models(self, tinker_stack: TinkerStack) -> None:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{tinker_stack.claas_url}/v1/models")
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
# Full lifecycle round-trip (direct via CLaaS API)
# ---------------------------------------------------------------------------


class TestTinkerStackRoundTrip:
    """End-to-end LoRA lifecycle: init -> list -> inference -> distill -> delete."""

    def test_full_lifecycle(self, tinker_stack: TinkerStack) -> None:
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
                    "lora_r": 32,
                    "lora_alpha": 64,
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

                # 3. Inference through CLaaS API
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
                    f"{claas_url}/v1/chat/completions",
                    json=chat_payload,
                    timeout=120.0,
                )
                assert chat_resp.status_code == 200, chat_resp.text
                choices = chat_resp.json()["choices"]
                assert len(choices) > 0
                response_content = choices[0]["message"]["content"]
                assert len(response_content) > 0
                logger.info("Model response: %s", response_content)

                # 3b. Compute content hash (the API resolves cache internally)
                ch = _content_hash(response_content)
                logger.info("Content hash: %s", ch)

                # Verify the cache entry exists (debugging aid)
                raw_resp = client.get(
                    f"{claas_url}/v1/completions/raw",
                    params={"content_hash": ch},
                    timeout=15.0,
                )
                assert raw_resp.status_code == 200, (
                    f"Cache miss for content_hash={ch[:16]}…: {raw_resp.text}"
                )
                logger.info(
                    "Cache hit: %d rollout logprobs available",
                    len(raw_resp.json()["logprobs"]),
                )

                # 4. Distill via feedback endpoint (teacher_mode=self)
                #    The API resolves prompt, response, and logprobs from
                #    the completion cache using content_hash.
                teacher_messages = build_teacher_messages(user_prompt, feedback_text)
                logger.info(
                    "Teacher messages (built by engine for self-distillation):\n%s",
                    json.dumps(teacher_messages, indent=2),
                )

                feedback_payload = {
                    "requests": [
                        {
                            "lora_id": lora_id,
                            "content_hash": ch,
                            "feedback": feedback_text,
                            "user_prompt": user_prompt,
                            "training": {"teacher_mode": "self"},
                        },
                    ],
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

                # Verify inference backend was refreshed with new weights
                assert fb_data["vllm"]["woke"] is True, (
                    "Expected inference backend refresh after distillation"
                )

                # 5. Verify the API switched to the distilled LoRA
                expected_path = fb_data["distill_result"]["metadata"][
                    "sampler_weights_path"
                ]
                status_resp = client.get(
                    f"{claas_url}/v1/sampler/status",
                    timeout=15.0,
                )
                assert status_resp.status_code == 200
                actual_path = status_resp.json()["model_path"]
                logger.info("Post-distillation sampler model_path: %s", actual_path)
                assert actual_path == expected_path, (
                    f"API should be serving distilled LoRA {expected_path!r}, "
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
                    f"{claas_url}/v1/chat/completions",
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
