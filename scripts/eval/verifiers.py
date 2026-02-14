"""Programmatic compliance verifiers for preference types.

Each verifier returns a score in [0.0, 1.0].
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove Qwen3 ``<think>...</think>`` blocks from model output."""
    return _THINK_RE.sub("", text).strip()


def _is_emoji(char: str) -> bool:
    """Check if a character is an emoji using Unicode category and codepoint ranges."""
    # 'So' (Symbol, Other) covers most emoji in the BMP and supplementary planes
    if unicodedata.category(char) == "So":
        return True
    # Supplementary plane emoji blocks not always categorized as 'So'
    cp = ord(char)
    return (
        0x1F600 <= cp <= 0x1F64F   # emoticons
        or 0x1F300 <= cp <= 0x1F5FF  # misc symbols & pictographs
        or 0x1F680 <= cp <= 0x1F6FF  # transport & map
        or 0x1F900 <= cp <= 0x1F9FF  # supplemental symbols
        or 0x1FA00 <= cp <= 0x1FAFF  # symbols extended-A
        or 0x1F1E0 <= cp <= 0x1F1FF  # flags
    )


def _count_emoji(text: str) -> int:
    """Count emoji characters in text."""
    return sum(1 for char in text if _is_emoji(char))


def verify_no_emoji(response: str) -> float:
    """Return 1.0 if no emoji chars, 0.0 otherwise."""
    return 1.0 if _count_emoji(response) == 0 else 0.0


def _count_sentences(text: str) -> int:
    """Count sentences by splitting on sentence-ending punctuation."""
    # Split on period, exclamation, question mark followed by space or end of string.
    sentences = re.split(r'[.!?]+(?:\s|$)', text.strip())
    # Filter empty strings from split
    return len([s for s in sentences if s.strip()])


def verify_concise(response: str) -> float:
    """Return 1.0 if <=3 sentences, linear decay to 0.0 at 9+."""
    n = _count_sentences(response)
    if n <= 3:
        return 1.0
    if n >= 9:
        return 0.0
    return max(0.0, 1.0 - (n - 3) / 6.0)


def verify_identity(response: str) -> float:
    """Return 1.0 if 'kuro' appears in the response (case-insensitive)."""
    return 1.0 if "kuro" in response.lower() else 0.0


# Registry mapping verifier names to functions.
VERIFIERS: dict[str, Callable[[str], float]] = {
    "no_emoji": verify_no_emoji,
    "concise": verify_concise,
    "identity": verify_identity,
}


def run_verifier(name: str, response: str) -> float:
    """Run a named verifier on a response (thinking blocks stripped)."""
    fn = VERIFIERS.get(name)
    if fn is None:
        logger.warning("Unknown verifier: %s", name)
        return 0.0
    return fn(strip_thinking(response))


def explain_verifier(name: str, response: str) -> dict[str, object]:
    """Return score plus verifier-specific diagnostics for auditing."""
    clean = strip_thinking(response)
    score = run_verifier(name, response)

    if name == "no_emoji":
        emoji_count = _count_emoji(clean)
        return {"score": score, "emoji_count": emoji_count, "pass": emoji_count == 0}

    if name == "concise":
        sentence_count = _count_sentences(clean)
        return {
            "score": score,
            "sentence_count": sentence_count,
            "pass": sentence_count <= 3,
        }

    if name == "identity":
        contains_kuro = "kuro" in clean.lower()
        return {"score": score, "contains_kuro": contains_kuro, "pass": contains_kuro}

    return {"score": score}
