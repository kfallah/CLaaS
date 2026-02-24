"""Pure helper functions for teacher prompt building.

These functions have no Modal dependency and can be imported anywhere.

Reference:
    Kleine Buening et al. (2026), "Aligning Language Models from User Interactions"
    https://github.com/lasgroup/user_interactions/blob/main/online_sdpo_trainer.py
"""

from __future__ import annotations

from claas.core.types import ChatMessage


def build_teacher_messages(
    prompt: str,
    feedback: str | None = None,
    *,
    system_prompt: str,
) -> list[ChatMessage]:
    """Build chat messages for the hindsight policy (SDPO teacher).

    Appends a hindsight context block to the user prompt so the frozen base
    model can score the student's response tokens with awareness of the user's
    follow-up.  Template follows the online SDPO trainer from:
    https://github.com/lasgroup/user_interactions/blob/main/online_sdpo_trainer.py

    The system_prompt must match the one the student saw so the teacher scores
    under the same context.

    Args:
        prompt: The original user prompt
        feedback: Optional user follow-up / feedback about the response
        system_prompt: System prompt (must match the student's)

    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    messages: list[ChatMessage] = [{"role": "system", "content": system_prompt}]

    if feedback:
        block = (
            "\n\n=== HINDSIGHT CONTEXT ===\n"
            "[The following is a future user message. "
            "Use this to guide your answer to the user prompt.]\n"
            f"{feedback.strip()}"
        )
        user_content = f"{prompt}{block}"
    else:
        user_content = prompt

    messages.append({"role": "user", "content": user_content})
    return messages


def teacher_messages_to_chat_template(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Convert typed chat messages to transformers chat-template input."""
    return [{"role": msg["role"], "content": msg["content"]} for msg in messages]


