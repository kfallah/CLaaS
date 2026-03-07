"""Tests for the pure-Python GJS divergence module."""

from __future__ import annotations

import math

import pytest

from claas.training.gjs import (
    compute_topk_gjs,
    extract_token_logprobs,
    slice_completion_topk,
)

# ── compute_topk_gjs ────────────────────────────────────────────────


def _make_topk(probs: list[float], start_id: int = 0) -> list[tuple[int, float]]:
    """Build a TokenTopK from raw probabilities (converts to logprobs)."""
    return [(start_id + i, math.log(p)) for i, p in enumerate(probs)]


def test_gjs_identical_distributions_is_zero():
    """GJS(p, p) = 0 for any alpha in (0, 1)."""
    topk = [_make_topk([0.5, 0.3, 0.2])]
    for alpha in [0.1, 0.5, 0.9]:
        result = compute_topk_gjs(topk, topk, alpha=alpha)
        assert len(result) == 1
        assert abs(result[0]) < 1e-10, f"GJS(p,p) should be 0, got {result[0]} for alpha={alpha}"


def test_gjs_symmetric_at_half():
    """GJS_0.5(p, q) == GJS_0.5(q, p) — Jensen-Shannon is symmetric at alpha=0.5."""
    p_topk = [_make_topk([0.7, 0.2, 0.1])]
    q_topk = [_make_topk([0.3, 0.4, 0.3])]

    gjs_pq = compute_topk_gjs(p_topk, q_topk, alpha=0.5)
    gjs_qp = compute_topk_gjs(q_topk, p_topk, alpha=0.5)

    assert abs(gjs_pq[0] - gjs_qp[0]) < 1e-10, (
        f"JSD should be symmetric: {gjs_pq[0]} != {gjs_qp[0]}"
    )


def test_gjs_non_negative():
    """GJS >= 0 for all inputs and alpha values."""
    p_topk = [_make_topk([0.6, 0.3, 0.1])]
    q_topk = [_make_topk([0.2, 0.5, 0.3])]

    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        result = compute_topk_gjs(p_topk, q_topk, alpha=alpha)
        assert result[0] >= -1e-10, f"GJS should be >= 0, got {result[0]} for alpha={alpha}"


def test_gjs_floor_logprob_used_for_missing_tokens():
    """When student's top-K doesn't contain a teacher token, floor is applied."""
    # Teacher has tokens 0, 1, 2
    teacher = [[(0, math.log(0.5)), (1, math.log(0.3)), (2, math.log(0.2))]]
    # Student only has tokens 0, 3, 4 — tokens 1 and 2 will use floor
    student = [[(0, math.log(0.6)), (3, math.log(0.3)), (4, math.log(0.1))]]

    result = compute_topk_gjs(teacher, student, alpha=0.5, floor_logprob=-20.0)
    assert len(result) == 1
    # Should be positive (distributions differ significantly)
    assert result[0] > 0.01


def test_gjs_alpha_one_degenerates():
    """alpha=1.0 path returns scalar KL-like values (teacher_lp - student_lp)."""
    teacher_lp = math.log(0.8)
    student_lp = math.log(0.3)
    # With alpha=1.0, result should be teacher_lp[0] - student_lp[0]
    teacher = [[(0, teacher_lp), (1, math.log(0.2))]]
    student = [[(0, student_lp), (1, math.log(0.7))]]

    result = compute_topk_gjs(teacher, student, alpha=1.0)
    expected = teacher_lp - student_lp
    assert abs(result[0] - expected) < 1e-10


def test_gjs_multi_position():
    """Verify per-position output for a 3-position sequence."""
    teacher = [
        _make_topk([0.5, 0.3, 0.2]),
        _make_topk([0.8, 0.1, 0.1]),
        _make_topk([0.4, 0.4, 0.2]),
    ]
    student = [
        _make_topk([0.5, 0.3, 0.2]),  # identical → 0
        _make_topk([0.3, 0.4, 0.3]),  # different → positive
        _make_topk([0.1, 0.1, 0.8]),  # very different → larger positive
    ]

    result = compute_topk_gjs(teacher, student, alpha=0.5)
    assert len(result) == 3
    assert abs(result[0]) < 1e-10  # identical
    assert result[1] > 0.01       # different
    assert result[2] > result[1]  # more different


def test_gjs_length_mismatch_raises():
    """Mismatched teacher/student lengths raise ValueError."""
    with pytest.raises(ValueError, match="length"):
        compute_topk_gjs(
            [_make_topk([0.5, 0.5])],
            [_make_topk([0.5, 0.5]), _make_topk([0.3, 0.7])],
            alpha=0.5,
        )


# ── extract_token_logprobs ──────────────────────────────────────────


def test_extract_token_logprobs_found():
    """Token in top-K returns correct logprob."""
    topk = [[(10, -1.5), (20, -2.0), (30, -3.0)]]
    result = extract_token_logprobs(topk, [20])
    assert result == [-2.0]


def test_extract_token_logprobs_missing():
    """Token not in top-K returns floor."""
    topk = [[(10, -1.5), (20, -2.0)]]
    result = extract_token_logprobs(topk, [99], floor_logprob=-20.0)
    assert result == [-20.0]


def test_extract_token_logprobs_multi_position():
    """Multi-position extraction with mixed hits and misses."""
    topk = [
        [(10, -1.0), (20, -2.0)],
        [(30, -0.5), (40, -1.5)],
        [(50, -3.0)],
    ]
    result = extract_token_logprobs(topk, [10, 99, 50], floor_logprob=-20.0)
    assert result == [-1.0, -20.0, -3.0]


def test_extract_token_logprobs_length_mismatch():
    """Mismatched lengths raise ValueError."""
    with pytest.raises(ValueError, match="length"):
        extract_token_logprobs([[(1, -1.0)]], [1, 2])


# ── slice_completion_topk ───────────────────────────────────────────


def test_slice_completion_topk():
    """Correct slicing with None handling."""
    topk_full = [
        None,                         # position 0 (prompt, skipped)
        [(1, -1.0)],                  # position 1 (prompt, skipped)
        [(2, -2.0), (3, -2.5)],       # position 2 (completion start)
        None,                         # position 3 (None → empty list)
        [(4, -3.0)],                  # position 4
    ]
    result = slice_completion_topk(topk_full, prompt_len=2, completion_len=3)
    assert len(result) == 3
    assert result[0] == [(2, -2.0), (3, -2.5)]
    assert result[1] == []  # None replaced
    assert result[2] == [(4, -3.0)]


def test_slice_completion_topk_all_none():
    """All None entries become empty lists."""
    topk_full = [None, None, None, None]
    result = slice_completion_topk(topk_full, prompt_len=1, completion_len=2)
    assert result == [[], []]
