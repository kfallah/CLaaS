"""Tests for pure helper functions in claas.training.teacher_helpers."""

from __future__ import annotations

import pytest

from claas.core.types import ChatMessage
from claas.training.teacher_helpers import (
    build_teacher_messages,
    teacher_messages_to_chat_template,
)


class TestBuildTeacherMessages:
    def test_with_feedback(self):
        msgs = build_teacher_messages("What is 2+2?", "Too verbose", system_prompt="You are helpful.")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful."
        assert msgs[1]["role"] == "user"
        assert "What is 2+2?" in msgs[1]["content"]
        assert "Too verbose" in msgs[1]["content"]
        assert "=== HINDSIGHT CONTEXT ===" in msgs[1]["content"]
        assert "Use this to guide your answer to the user prompt" in msgs[1]["content"]

    def test_without_feedback(self):
        msgs = build_teacher_messages("What is 2+2?", system_prompt="You are helpful.")
        assert len(msgs) == 2
        assert msgs[1]["content"] == "What is 2+2?"

    def test_without_feedback_explicit_none(self):
        msgs = build_teacher_messages("Hello", None, system_prompt="You are helpful.")
        assert msgs[1]["content"] == "Hello"

    @pytest.mark.parametrize("system_prompt", [
        "Be concise.",
        "You are a pirate assistant who speaks in pirate lingo.",
    ])
    def test_system_prompt_matches_input(self, system_prompt: str):
        msgs = build_teacher_messages("x", system_prompt=system_prompt)
        assert msgs[0]["content"] == system_prompt


class TestTeacherMessagesToChatTemplate:
    def test_converts_typed_messages(self):
        msgs: list[ChatMessage] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        result = teacher_messages_to_chat_template(msgs)
        assert result == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        # Returns new list (not the same object)
        assert result is not msgs
