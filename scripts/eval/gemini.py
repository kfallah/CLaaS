"""Gemini simulated user for natural feedback variation.

Phase 2+ only. Optional — requires --gemini-api-key.
Falls back to hardcoded feedback_string when unavailable.
"""

from __future__ import annotations

import json
import logging

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
        self._model = None

    def _get_model(self):
        """Lazily initialize the Gemini model."""
        if self._model is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self._api_key)
                self._model = genai.GenerativeModel("gemini-1.5-flash")
            except ImportError:
                raise ImportError(
                    "google-generativeai is required for Gemini user. "
                    "Install with: pip install google-generativeai"
                )
        return self._model

    async def evaluate_response(
        self,
        chatbot_response: str,
        user_prompt: str,
    ) -> dict:
        """Ask Gemini to evaluate the chatbot's response.

        Returns:
            {"satisfied": bool, "feedback": str | None}
        """
        model = self._get_model()
        system = _SYSTEM_PROMPT.format(
            preference_description=self._preference_description,
        )
        prompt = (
            f"{system}\n\n"
            f"User message: {user_prompt}\n"
            f"Chatbot response: {chatbot_response}\n\n"
            f"Evaluate:"
        )

        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            # Try to parse JSON from the response
            # Handle cases where Gemini wraps JSON in markdown code blocks
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.warning("Failed to parse Gemini response: %s", e)
            return {"satisfied": False, "feedback": None}
        except Exception as e:
            logger.warning("Gemini API call failed: %s", e)
            return {"satisfied": False, "feedback": None}


def get_feedback(
    gemini_user: GeminiUser | None,
    default_feedback: str,
    chatbot_response: str | None = None,
    user_prompt: str | None = None,
) -> str:
    """Get feedback from Gemini user or fall back to hardcoded string.

    This is synchronous — for Phase 2+ we'd use the async version in the runner.
    """
    # Gemini integration is async; this sync wrapper is for Phase 1 compatibility.
    # In Phase 2+, the runner calls gemini_user.evaluate_response() directly.
    return default_feedback
