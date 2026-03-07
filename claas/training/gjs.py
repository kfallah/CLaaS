"""Pure-Python GJS divergence for the Tinker SDPO engine.

Mirrors the torch-based GJS computation in ``sdpo_loss.py`` but uses only
the Python standard library (``math``).  This is the non-GPU code path
used when the Tinker engine fetches top-K distributions via
``topk_prompt_logprobs`` on ``sample_async``.

Reference: veRL ``core_algos.py:1120-1122`` for the renormalize-over-top-K
pattern and Hübotter et al. (2026) for the GJS formulation.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

TokenTopK = list[tuple[int, float]]
"""K (token_id, logprob) pairs for one position."""

SequenceTopK = list[TokenTopK]
"""One TokenTopK per token in the sequence."""

# ---------------------------------------------------------------------------
# Default floor log-probability (matches ``sdpo_loss._lookup_token_in_topk``)
# ---------------------------------------------------------------------------

_DEFAULT_FLOOR_LOGPROB = -20.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_topk_gjs(
    teacher_topk: SequenceTopK,
    student_topk: SequenceTopK,
    alpha: float,
    floor_logprob: float = _DEFAULT_FLOOR_LOGPROB,
) -> list[float]:
    """Per-token GJS divergence over top-K distributions.

    For each position the teacher's top-K token IDs define the shared
    support (following the veRL reference).  Student logprobs are looked
    up in its own top-K set — missing entries fall back to *floor_logprob*.

    When ``alpha == 1.0`` the function degenerates to reverse-KL style
    scalar advantages (``teacher_logprob - student_logprob`` at the
    *first* token in the teacher's top-K set — the most-probable token).
    This is kept for backward-compatibility with the scalar path.

    Args:
        teacher_topk: Per-position teacher top-K distributions.
        student_topk: Per-position student top-K distributions.
        alpha: GJS interpolation weight (0 < alpha <= 1).
        floor_logprob: Logprob to use when a token is absent from a top-K set.

    Returns:
        List of per-token GJS divergence values (length = len(teacher_topk)).
    """
    if len(teacher_topk) != len(student_topk):
        raise ValueError(
            f"teacher_topk length ({len(teacher_topk)}) != "
            f"student_topk length ({len(student_topk)})"
        )

    result: list[float] = []
    for t_topk, s_topk in zip(teacher_topk, student_topk):
        result.append(_gjs_at_position(t_topk, s_topk, alpha, floor_logprob))
    return result


def extract_token_logprobs(
    topk: SequenceTopK,
    token_ids: list[int],
    floor_logprob: float = _DEFAULT_FLOOR_LOGPROB,
) -> list[float]:
    """Look up specific token logprobs in a per-position top-K set.

    Pure-Python counterpart to ``sdpo_loss._lookup_token_in_topk``.

    Args:
        topk: Per-position top-K distributions.
        token_ids: Token IDs to look up (one per position).
        floor_logprob: Value returned when the token is absent.

    Returns:
        List of logprobs (length = len(token_ids)).
    """
    if len(topk) != len(token_ids):
        raise ValueError(
            f"topk length ({len(topk)}) != token_ids length ({len(token_ids)})"
        )

    result: list[float] = []
    for pos_topk, tid in zip(topk, token_ids):
        lp = floor_logprob
        for tok_id, tok_lp in pos_topk:
            if tok_id == tid:
                lp = tok_lp
                break
        result.append(lp)
    return result


def slice_completion_topk(
    topk_full: list[TokenTopK | None],
    prompt_len: int,
    completion_len: int,
) -> SequenceTopK:
    """Extract the completion portion of a full-sequence top-K list.

    Analogous to ``_slice_completion_logprobs`` but for top-K data.
    ``None`` entries (e.g. the first unconditional position) are replaced
    with empty lists.

    Args:
        topk_full: Full-sequence top-K data (may contain ``None``).
        prompt_len: Number of prompt tokens to skip.
        completion_len: Number of completion tokens to extract.

    Returns:
        SequenceTopK of length *completion_len*.
    """
    raw = topk_full[prompt_len : prompt_len + completion_len]
    return [entry if entry is not None else [] for entry in raw]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gjs_at_position(
    teacher_topk: TokenTopK,
    student_topk: TokenTopK,
    alpha: float,
    floor_logprob: float,
) -> float:
    """GJS divergence at a single position over the teacher's top-K support."""
    if not teacher_topk:
        return 0.0

    # Build student lookup
    student_lookup: dict[int, float] = {tid: lp for tid, lp in student_topk}

    # Shared support = teacher's top-K token IDs
    teacher_ids = [tid for tid, _ in teacher_topk]
    teacher_lps = [lp for _, lp in teacher_topk]

    student_lps = [
        student_lookup.get(tid, floor_logprob) for tid in teacher_ids
    ]

    # Renormalize both distributions over the shared support
    teacher_probs = _renormalize(teacher_lps)
    student_probs = _renormalize(student_lps)

    if alpha >= 1.0:
        # Degenerate to reverse KL: scalar teacher_lp - student_lp
        # for the highest-probability teacher token
        return teacher_lps[0] - student_lps[0]

    # Mixture: M = alpha * teacher + (1 - alpha) * student
    mixture = [
        alpha * tp + (1.0 - alpha) * sp
        for tp, sp in zip(teacher_probs, student_probs)
    ]

    kl_teacher_m = _kl_divergence(teacher_probs, mixture)
    kl_student_m = _kl_divergence(student_probs, mixture)

    return alpha * kl_teacher_m + (1.0 - alpha) * kl_student_m


def _renormalize(logprobs: list[float]) -> list[float]:
    """Convert logprobs to normalized probabilities over the support."""
    max_lp = max(logprobs) if logprobs else 0.0
    # exp(lp - max_lp) for numerical stability
    exps = [math.exp(lp - max_lp) for lp in logprobs]
    total = sum(exps)
    if total < 1e-30:
        # Uniform fallback to avoid division by zero
        n = len(exps)
        return [1.0 / n] * n
    return [e / total for e in exps]


def _kl_divergence(p: list[float], q: list[float]) -> float:
    """KL(p || q) where p and q are probability vectors."""
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0 and qi > 0:
            total += pi * math.log(pi / qi)
    return total
