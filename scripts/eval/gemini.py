"""Gemini simulated user for natural feedback variation.

Optional — requires --gemini-api-key.
Falls back to hardcoded feedback_string when unavailable.

Uses the google-genai SDK (>=1.0), which communicates via REST/httpx
and has no protobuf dependency conflict with vLLM.
"""

from __future__ import annotations

import json
import logging

import httpx

from .types import GeminiEvalResult

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are testing an AI chatbot. Have a natural conversation about coding, \
daily life, or general knowledge. Keep messages short (1-2 sentences). \
Your hidden preference: {preference_description}. \
After the chatbot responds, evaluate whether it followed your preference.
Reply with ONLY a JSON object: {{"satisfied": true/false, "feedback": "string or null"}}
If satisfied, set feedback to null.
If not satisfied, feedback should be a natural rephrasing of your preference."""


class GeminiUser:
    """Simulated user powered by Google Gemini."""

    def __init__(self, api_key: str, preference_description: str):
        self._api_key = api_key
        self._preference_description = preference_description
        self._client = None

    def _get_client(self):
        """Lazily initialize the Gemini client."""
        if self._client is None:
            try:
                from google import genai

                self._client = genai.Client(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "google-genai is required for Gemini user. "
                    "Install with: pip install google-genai"
                )
        return self._client

    async def evaluate_response(
        self,
        chatbot_response: str,
        user_prompt: str,
    ) -> GeminiEvalResult:
        """Ask Gemini to evaluate the chatbot's response."""
        client = self._get_client()
        system = _SYSTEM_PROMPT.format(
            preference_description=self._preference_description,
        )
        prompt = (
            f"User message: {user_prompt}\n"
            f"Chatbot response: {chatbot_response}\n\n"
            f"Evaluate:"
        )

        try:
            response = await client.aio.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config={"system_instruction": system},
            )
            text = response.text.strip()
            # Handle cases where Gemini wraps JSON in markdown code blocks
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
            return GeminiEvalResult(
                satisfied=parsed.get("satisfied", False),
                feedback=parsed.get("feedback"),
            )
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.warning("Failed to parse Gemini response: %s", e)
            return GeminiEvalResult(satisfied=False)
        except (httpx.HTTPError, ConnectionError, OSError, RuntimeError) as e:
            logger.warning("Gemini API call failed: %s", e)
            return GeminiEvalResult(satisfied=False)


def get_feedback(
    gemini_user: GeminiUser | None,
    default_feedback: str,
    chatbot_response: str | None = None,
    user_prompt: str | None = None,
) -> str:
    """Get feedback from Gemini user or fall back to hardcoded string.

    This is synchronous — the runner calls gemini_user.evaluate_response() directly
    for the async path.
    """
    return default_feedback
