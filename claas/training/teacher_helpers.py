"""Pure helper functions for teacher prompt building.

These functions have no Modal dependency and can be imported anywhere.
"""

from __future__ import annotations

from claas.core.config import DEFAULT_SYSTEM_PROMPT
from claas.core.types import ChatMessage


def build_teacher_messages(
    prompt: str,
    feedback: str | None = None,
    system_prompt: str | None = None,
) -> list[ChatMessage]:
    """Build chat messages for teacher prompt (veRL-compatible template).

    Template matches veRL's reprompt_template structure:
    {prompt}{feedback}\\n\\nCorrectly solve the original question.

    Args:
        prompt: The original user prompt
        feedback: Optional feedback about the response quality
        system_prompt: Optional system prompt

    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    messages: list[ChatMessage] = [{"role": "system", "content": system_prompt}]

    # Build user content with veRL-style template
    if feedback:
        feedback_section = (
            "\n\nThe following is positive or negative feedback from your earlier attempt:"
            f"\n\n{feedback}\n"
        )
        user_content = f"{prompt}{feedback_section}\n\nCorrectly solve the original question.\n"
    else:
        user_content = prompt

    messages.append({"role": "user", "content": user_content})
    return messages


def teacher_messages_to_chat_template(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Convert typed chat messages to transformers chat-template input."""
    return [{"role": msg["role"], "content": msg["content"]} for msg in messages]


