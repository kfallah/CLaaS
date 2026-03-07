"""Tinker engine data types for SDPO distillation."""

from __future__ import annotations

from dataclasses import dataclass

from claas.training.gjs import SequenceTopK


@dataclass(frozen=True)
class SampleCore:
    """Fields shared across both scalar and top-K prepared samples."""

    full_tokens: list[int]
    input_tokens: list[int]
    target_tokens: list[int]
    prompt_len: int
    completion_len: int
    teacher_scored_text: str


@dataclass(frozen=True)
class ScalarPreparedSample(SampleCore):
    """Sample with scalar teacher logprobs (current approach)."""

    teacher_logprobs: list[float]


@dataclass(frozen=True)
class TopKPreparedSample(SampleCore):
    """Sample with top-K teacher distributions (GJS approach)."""

    teacher_logprobs: list[float]
    teacher_topk: SequenceTopK


PreparedSample = ScalarPreparedSample | TopKPreparedSample


@dataclass(frozen=True)
class ScalarBehavior:
    """Scalar student logprobs for IS correction."""

    logprobs: list[float]


@dataclass(frozen=True)
class TopKBehavior:
    """Top-K student distributions + scalar logprobs for IS correction."""

    logprobs: list[float]
    topk: SequenceTopK


BehaviorSignal = ScalarBehavior | TopKBehavior
