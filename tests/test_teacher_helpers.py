"""Tests for pure helper functions in claas.teacher."""

from __future__ import annotations

import pytest

from claas.teacher import (
    ChatMessage,
    TokenLogprobs,
    build_teacher_messages,
    format_teacher_prompt,
    messages_to_chatml,
    teacher_messages_to_chat_template,
)


class TestBuildTeacherMessages:
    def test_with_feedback(self):
        msgs = build_teacher_messages("What is 2+2?", "Too verbose")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "What is 2+2?" in msgs[1]["content"]
        assert "Too verbose" in msgs[1]["content"]
        assert "Correctly solve the original question" in msgs[1]["content"]

    def test_without_feedback(self):
        msgs = build_teacher_messages("What is 2+2?")
        assert len(msgs) == 2
        assert msgs[1]["content"] == "What is 2+2?"

    def test_without_feedback_explicit_none(self):
        msgs = build_teacher_messages("Hello", None)
        assert msgs[1]["content"] == "Hello"

    def test_custom_system_prompt(self):
        msgs = build_teacher_messages("x", system_prompt="Be concise.")
        assert msgs[0]["content"] == "Be concise."

    def test_default_system_prompt(self):
        msgs = build_teacher_messages("x")
        assert "expert coding assistant" in msgs[0]["content"]


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


class TestMessagesToChatml:
    def test_basic_conversion(self):
        msgs: list[ChatMessage] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = messages_to_chatml(msgs)
        assert "<|im_start|>system\nYou are helpful.<|im_end|>" in result
        assert "<|im_start|>user\nHi<|im_end|>" in result
        assert result.endswith("<|im_start|>assistant\n")

    def test_empty_messages(self):
        result = messages_to_chatml([])
        assert result == "<|im_start|>assistant\n"


class TestFormatTeacherPrompt:
    def test_combines_build_and_chatml(self):
        result = format_teacher_prompt("What is 1+1?", feedback="Wrong answer")
        assert "<|im_start|>system" in result
        assert "What is 1+1?" in result
        assert "Wrong answer" in result
        assert result.endswith("<|im_start|>assistant\n")

    def test_without_feedback(self):
        result = format_teacher_prompt("Hello")
        assert "<|im_start|>user\nHello<|im_end|>" in result

    def test_custom_system_prompt(self):
        result = format_teacher_prompt("x", system_prompt="Custom.")
        assert "<|im_start|>system\nCustom.<|im_end|>" in result


class TestParseTeacherResult:
    def test_parses_result(self):
        pytest.importorskip("torch")
        from claas.teacher import parse_teacher_result

        result: list[TokenLogprobs] = [
            {"indices": [10, 20, 30], "logprobs": [-0.1, -0.5, -1.0]},
            {"indices": [40, 50], "logprobs": [-0.2, -0.3]},
        ]
        logprobs, indices = parse_teacher_result(result, device="cpu")
        assert logprobs.shape == (2, 3)
        assert indices.shape == (2, 3)
        assert indices[0].tolist() == [10, 20, 30]
        assert indices[1, :2].tolist() == [40, 50]
        # Padding for shorter row
        assert logprobs[1, 2].item() == pytest.approx(-100.0)

    def test_empty_result_raises(self):
        pytest.importorskip("torch")
        from claas.teacher import parse_teacher_result

        with pytest.raises(ValueError, match="Empty teacher result"):
            parse_teacher_result([])

    def test_all_empty_raises(self):
        pytest.importorskip("torch")
        from claas.teacher import parse_teacher_result

        with pytest.raises(ValueError, match="empty top-K"):
            parse_teacher_result([{"indices": [], "logprobs": []}])
