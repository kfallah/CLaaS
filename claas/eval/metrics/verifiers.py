"""Programmatic compliance verifiers for preference types.

Each verifier is a callable class returning a VerifierResult.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class VerifierResult:
    score: float
    passed: bool


class Verifier(Protocol):
    def __call__(self, response: str) -> VerifierResult: ...


def strip_thinking(text: str) -> str:
    """Remove thinking blocks from model output.

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


def _count_sentences(text: str) -> int:
    """Count sentences by splitting on sentence-ending punctuation."""
    # Split on period, exclamation, question mark followed by space or end of string.
    sentences = re.split(r'[.!?]+(?:\s|$)', text.strip())
    # Filter empty strings from split
    return len([s for s in sentences if s.strip()])


class NoEmojiVerifier:
    def __call__(self, response: str) -> VerifierResult:
        passed = _count_emoji(response) == 0
        return VerifierResult(score=1.0 if passed else 0.0, passed=passed)


class ConciseVerifier:
    def __call__(self, response: str) -> VerifierResult:
        n = _count_sentences(response)
        if n <= 3:
            score = 1.0
        elif n >= 9:
            score = 0.0
        else:
            score = max(0.0, 1.0 - (n - 3) / 6.0)
        return VerifierResult(score=score, passed=n <= 3)


class IdentityVerifier:
    def __call__(self, response: str) -> VerifierResult:
        passed = "kuro" in response.lower()
        return VerifierResult(score=1.0 if passed else 0.0, passed=passed)


def run_verifier(verifier: Verifier, response: str) -> VerifierResult:
    """Run a verifier on a response (thinking blocks stripped)."""
    return verifier(strip_thinking(response))
