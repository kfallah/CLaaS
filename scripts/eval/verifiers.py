"""Programmatic compliance verifiers for preference types.

Each verifier returns a score in [0.0, 1.0].
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Unicode categories that cover emoji characters.
_EMOJI_CATEGORIES = {"So"}  # Symbol, Other

# Regex pattern for common emoji sequences (supplementary plane + variation selectors).
_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f1e0-\U0001f1ff"  # flags
    "\U00002702-\U000027b0"  # dingbats
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "\U0001f900-\U0001f9ff"  # supplemental symbols
    "\U0001fa00-\U0001fa6f"  # chess symbols
    "\U0001fa70-\U0001faff"  # symbols extended-A
    "\U00002600-\U000026ff"  # misc symbols
    "\U0000200d"             # zero width joiner
    "\U00002b50"             # star
    "\U00002728"             # sparkles
    "\U00002764"             # heart
    "]+",
)


def _count_emoji(text: str) -> int:
    """Count emoji characters in text."""
    count = 0
    for char in text:
        if unicodedata.category(char) in _EMOJI_CATEGORIES:
            count += 1
        elif _EMOJI_RE.match(char):
            count += 1
    return count


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
    """Run a named verifier on a response."""
    fn = VERIFIERS.get(name)
    if fn is None:
        logger.warning("Unknown verifier: %s", name)
        return 0.0
    return fn(response)
