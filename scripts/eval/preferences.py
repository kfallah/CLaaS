"""Preference configurations for the evaluation harness.

Each preference defines:
- feedback_string: hardcoded feedback for Phase 1
- logprob_pairs: (positive, negative) examples in Qwen3 ChatML format
- probe_prompts: chat prompts for generative eval (Phase 2+)
- verifier_name: maps to function in verifiers.py
"""

from __future__ import annotations

from dataclasses import dataclass

CHATML_PREFIX = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


@dataclass
class LogprobPair:
    """A positive/negative response pair for logprob margin measurement."""

    prompt_chatml: str
    positive_response: str
    negative_response: str


@dataclass
class PreferenceConfig:
    """Full configuration for a single preference type."""

    name: str
    feedback_string: str
    logprob_pairs: list[LogprobPair]
    probe_prompts: list[str]
    verifier_name: str


def _chatml(prompt: str) -> str:
    """Format a prompt as Qwen3 ChatML prefix."""
    return CHATML_PREFIX.format(prompt=prompt)


def get_preference_configs() -> dict[str, PreferenceConfig]:
    """Return all preference configurations keyed by name."""
    return {
        "no_emoji": _no_emoji_config(),
        "concise": _concise_config(),
        "identity": _identity_config(),
    }


def _no_emoji_config() -> PreferenceConfig:
    prompt = "Hey, what's up?"
    return PreferenceConfig(
        name="no_emoji",
        feedback_string="Don't use any emojis in your responses",
        logprob_pairs=[
            LogprobPair(
                prompt_chatml=_chatml(prompt),
                positive_response=(
                    "Hello! How can I help you today? "
                    "I'm happy to assist with anything you need."
                ),
                negative_response=(
                    "Hello! \U0001f60a How can I help you today? "
                    "\U0001f389 I'm happy to assist! \u2728"
                ),
            ),
        ],
        probe_prompts=[
            "Hey, what's up?",
            "Tell me something interesting about space.",
            "How are you today?",
            "Write an enthusiastic greeting!",
            "What's your favorite thing to talk about?",
        ],
        verifier_name="no_emoji",
    )


def _concise_config() -> PreferenceConfig:
    prompt = "Tell me about Python programming."
    return PreferenceConfig(
        name="concise",
        feedback_string="Be more concise, keep responses under 3 sentences",
        logprob_pairs=[
            LogprobPair(
                prompt_chatml=_chatml(prompt),
                positive_response=(
                    "Python is a versatile, high-level programming language known "
                    "for its readable syntax. It's widely used in web development, "
                    "data science, and automation."
                ),
                negative_response=(
                    "Python is a high-level, interpreted programming language that "
                    "was created by Guido van Rossum and first released in 1991. "
                    "It emphasizes code readability with its notable use of "
                    "significant whitespace. Python supports multiple programming "
                    "paradigms, including structured, object-oriented, and functional "
                    "programming. It has a comprehensive standard library that "
                    "provides tools for many tasks. The language is commonly used "
                    "in web development, scientific computing, data analysis, "
                    "artificial intelligence, and system scripting."
                ),
            ),
        ],
        probe_prompts=[
            "Tell me about Python programming.",
            "What is machine learning?",
            "Explain how the internet works.",
            "Describe the water cycle in detail.",
            "What are the benefits of exercise?",
        ],
        verifier_name="concise",
    )


def _identity_config() -> PreferenceConfig:
    prompt = "Who are you?"
    return PreferenceConfig(
        name="identity",
        feedback_string="Your name is Kuro, always introduce yourself as Kuro",
        logprob_pairs=[
            LogprobPair(
                prompt_chatml=_chatml(prompt),
                positive_response=(
                    "I'm Kuro! I'm here to help you with whatever you need. "
                    "Feel free to ask me anything."
                ),
                negative_response=(
                    "I'm an AI assistant created to be helpful, harmless, and "
                    "honest. How can I assist you today?"
                ),
            ),
        ],
        probe_prompts=[
            "Who are you?",
            "Introduce yourself.",
            "What should I call you?",
            "Hey, what's your name?",
            "Tell me about yourself.",
        ],
        verifier_name="identity",
    )
