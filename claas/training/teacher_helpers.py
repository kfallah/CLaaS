"""Pure helper functions for teacher prompt building and result parsing.

These functions have no Modal dependency and can be imported anywhere.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from claas.core.types import ChatMessage

if TYPE_CHECKING:
    import torch

    from claas.core.types import TeacherTokenLogprobs


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
        system_prompt = (
            "You are an expert coding assistant. Provide high-quality, "
            "correct, and well-explained code solutions."
        )
    messages: list[ChatMessage] = [{"role": "system", "content": system_prompt}]

    # Build user content with veRL-style template
    if feedback:
        feedback_section = (
            "\n\nThe following is feedback from your unsuccessful earlier attempt:"
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


def messages_to_chatml(messages: list[ChatMessage]) -> str:
    """Convert chat messages to a ChatML string ending with assistant prompt."""
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def format_teacher_prompt(
    user_prompt: str,
    feedback: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Format prompt for the teacher model as a ChatML string."""
    messages = build_teacher_messages(user_prompt, feedback, system_prompt)
    return messages_to_chatml(messages)


def parse_teacher_result(
    result: list[dict] | list["TeacherTokenLogprobs"],
    device: str = "cuda",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Parse teacher scoring result into tensors.

    Args:
        result: List of dicts or TeacherTokenLogprobs (Pydantic)
        device: Device to place tensors on

    Returns:
        Tuple of (teacher_logprobs, teacher_indices) tensors
            - teacher_logprobs: (T, K) tensor of log-probabilities
            - teacher_indices: (T, K) tensor of token indices
    """
    if not result:
        raise ValueError("Empty teacher result")

    import torch

    def _get(pos: object, key: str) -> list[int] | list[float]:
        """Access field by key (dict) or attribute (Pydantic)."""
        if isinstance(pos, dict):
            return pos[key]  # type: ignore[return-value]
        return getattr(pos, key)

    # Find max K across positions
    max_k = max(len(_get(pos, "indices")) for pos in result)
    if max_k == 0:
        raise ValueError("All teacher positions have empty top-K results")

    T = len(result)

    # Initialize tensors with padding
    teacher_logprobs = torch.full((T, max_k), -100.0, device=device)
    teacher_indices = torch.zeros((T, max_k), dtype=torch.long, device=device)

    for t, pos in enumerate(result):
        indices = _get(pos, "indices")
        logprobs = _get(pos, "logprobs")
        k = len(indices)
        if k > 0:
            teacher_indices[t, :k] = torch.tensor(indices, dtype=torch.long, device=device)
            teacher_logprobs[t, :k] = torch.tensor(logprobs, dtype=torch.float, device=device)

    return teacher_logprobs, teacher_indices
