"""Shared utilities for inference backends."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

from fastapi.responses import StreamingResponse

# ---------------------------------------------------------------------------
# Content processing
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_WHITESPACE_RUN_RE = re.compile(r"\s+")


def strip_thinking(text: str) -> str:
    """Remove thinking blocks before hashing content.

    Handles two cases:
    1. Proper ``<think>...</think>`` blocks.
    2. Orphaned ``</think>`` when the opening ``<think>`` was consumed as a
       special token by the tokenizer (Qwen3).  Everything before the first
       orphaned ``</think>`` is thinking text and is stripped.
    """
    text = _THINK_RE.sub("", text)
    idx = text.find("</think>")
    if idx >= 0:
        text = text[idx + len("</think>"):]
    return text.strip()


def normalize_for_hash(text: str) -> str:
    """Normalize text for content hashing.

    Strips thinking tags and collapses all whitespace runs to a single space.
    This makes the hash resilient to downstream whitespace mutations
    (e.g. OpenClaw flattening newlines).
    """
    return _WHITESPACE_RUN_RE.sub(" ", strip_thinking(text)).strip()


_FINAL_CHANNEL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)",
    re.DOTALL,
)
_ANALYSIS_CHANNEL_RE = re.compile(
    r"<\|channel\|>analysis<\|message\|>",
)


def extract_final_channel(text: str) -> str:
    """Extract the ``final`` channel content from GPT-OSS style output.

    GPT-OSS generates ``<|channel|>analysis<|message|>...<|end|>
    <|start|>assistant<|channel|>final<|message|>...``.  Only the *final*
    channel should be shown to the user.  If the model ran out of tokens
    before producing a ``final`` channel, return an empty string rather
    than leaking the raw analysis text.
    """
    m = _FINAL_CHANNEL_RE.search(text)
    if m:
        return m.group(1).strip()
    # If the text contains an analysis channel but no final channel,
    # the model ran out of tokens mid-reasoning -- return empty.
    if _ANALYSIS_CHANNEL_RE.search(text):
        return ""
    return text


def coerce_content(content: Any) -> str:
    """Coerce various content formats to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                    continue
                if isinstance(part.get("content"), str):
                    parts.append(part["content"])
                    continue
        return "\n".join(parts)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
    return str(content)


# ---------------------------------------------------------------------------
# Bounded parameter helpers
# ---------------------------------------------------------------------------


def bounded_int(value: int | None, *, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    return max(minimum, min(maximum, int(value)))


def bounded_float(
    value: float | None,
    *,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    if value is None:
        return default
    return max(minimum, min(maximum, float(value)))


# ---------------------------------------------------------------------------
# Chat template utilities
# ---------------------------------------------------------------------------


def coerce_template_ids(result: Any) -> list[int]:
    """Normalize ``tokenizer.apply_chat_template`` output to a plain list[int]."""
    if isinstance(result, list):
        return [int(tok) for tok in result]
    if isinstance(result, dict):
        maybe_ids = result.get("input_ids")
        if isinstance(maybe_ids, list):
            return [int(tok) for tok in maybe_ids]
    if hasattr(result, "tolist"):
        maybe_ids = result.tolist()
        if isinstance(maybe_ids, list):
            return [int(tok) for tok in maybe_ids]
    raise TypeError("Unsupported apply_chat_template result shape")


def apply_chat_template_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    """Tokenize messages via apply_chat_template, returning token IDs."""
    result = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
    )
    return coerce_template_ids(result)


# ---------------------------------------------------------------------------
# SSE streaming helpers
# ---------------------------------------------------------------------------


def stream_chat_response(
    completion_id: str,
    created: int,
    model: str,
    content: str,
) -> StreamingResponse:
    """Wrap a complete response as an SSE stream for OpenAI-compatible clients."""

    def _generate() -> Iterator[str]:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": content},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


def stream_completion_response(
    completion_id: str,
    created: int,
    model: str,
    text: str,
) -> StreamingResponse:
    """Wrap a complete text-completion response as an SSE stream."""

    def _generate() -> Iterator[str]:
        chunk = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

        final = {
            "id": completion_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": "",
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")
