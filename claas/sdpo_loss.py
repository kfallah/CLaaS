"""SDPO Loss: Self-Distillation Policy Optimization.

This module implements the Self-Distillation Policy Optimization (SDPO) loss
as described in Hübotter et al. (2026), "Reinforcement Learning via Self-Distillation".

Reference implementation: https://github.com/lasgroup/SDPO

SDPO is a policy gradient algorithm that:
1. Uses teacher logprobs to derive per-token advantages (distillation signal)
2. Applies PPO-style clipped importance sampling for stable updates
3. Regularizes with KL divergence to base policy to prevent drift

The loss combines a clipped policy gradient objective (like PPO) with KL
regularization to the base policy to prevent excessive drift from the
pre-trained model.
"""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn.functional as F


class SDPOLossResult(TypedDict):
    """Result from SDPO loss computation."""

    loss: torch.Tensor
    pg_loss: float
    kl_reg: float
    mean_advantage: float
    frac_positive_advantage: float
    mean_is_ratio: float
    clip_fraction: float


def compute_sdpo_loss(
    student_logits: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    teacher_indices: torch.Tensor,
    base_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    old_student_logprobs: torch.Tensor,
    response_ids: torch.Tensor,
    clip_eps_lower: float = 0.2,
    clip_eps_upper: float = 0.2,
    kl_reg_weight: float = 0.1,
) -> SDPOLossResult:
    """Compute SDPO loss: policy gradient with teacher-derived advantages + KL regularization.

    This follows the reference implementation from https://github.com/lasgroup/SDPO.
    The loss consists of:
    1. Policy gradient loss with PPO clipping (advantages from teacher-student KL)
    2. KL divergence from student to base policy (drift regularization)

    Args:
        student_logits: (B, T, V) student model logits at response positions (WITH grad)
        teacher_logprobs: (B, T, K) teacher's top-K log-probabilities (no grad)
        teacher_indices: (B, T, K) teacher's top-K token indices
        base_logprobs: (B, T) base model log-prob of response tokens (no grad, no LoRA)
        response_mask: (B, T) binary mask, 1 for response tokens
        old_student_logprobs: (B, T) student log-prob of chosen token at rollout time
        response_ids: (B, T) actual token ids in the response
        clip_eps_lower: lower bound for PPO clip range (ratio >= 1 - clip_eps_lower)
        clip_eps_upper: upper bound for PPO clip range (ratio <= 1 + clip_eps_upper)
        kl_reg_weight: weight for KL regularization to base policy

    Returns:
        SDPOLossResult with loss tensor and diagnostic metrics
    """
    _B, _T, _V = student_logits.shape

    # Step 1: Student log-probabilities (full, with gradient)
    student_log_probs = F.log_softmax(student_logits, dim=-1)  # (B, T, V)

    # Current student log-prob of the actually-generated token
    current_logprob_chosen = student_log_probs.gather(
        -1, response_ids.unsqueeze(-1)
    ).squeeze(-1)  # (B, T)

    # Step 2: Compute per-token advantages from teacher-student KL divergence
    # Teacher's probability of the chosen token
    teacher_logprob_chosen = _lookup_token_in_topk(
        response_ids,
        teacher_indices,
        teacher_logprobs,
        floor_logprob=-20.0,
    )  # (B, T)

    # Advantage: how much the teacher prefers this token over the student
    # Positive advantage = teacher wants this token more = encourage this token
    advantages = teacher_logprob_chosen - current_logprob_chosen.detach()  # (B, T)

    # Step 3: PPO-style clipped policy gradient
    # Importance sampling ratio (current policy / old policy)
    log_ratio_is = current_logprob_chosen - old_student_logprobs
    ratio = log_ratio_is.exp().clamp(max=10.0)  # prevent explosion

    # Clipped surrogate objective with separate lower/upper bounds
    # advantages are NOT differentiated through (detached)
    surr1 = ratio * advantages.detach()
    surr2 = torch.clamp(ratio, 1.0 - clip_eps_lower, 1.0 + clip_eps_upper) * advantages.detach()
    pg_loss = -torch.min(surr1, surr2)

    # Apply response mask and average
    mask_sum = response_mask.sum().clamp(min=1.0)
    pg_loss = (pg_loss * response_mask).sum() / mask_sum

    # Step 4: KL regularization to base policy
    # Prevents the student from drifting too far from the pre-trained model
    # KL(student || base) = student_logprob - base_logprob (per token)
    kl_to_base_per_token = current_logprob_chosen - base_logprobs  # (B, T)
    kl_reg = (kl_to_base_per_token * response_mask).sum() / mask_sum

    total_loss = pg_loss + kl_reg_weight * kl_reg

    # Diagnostics
    with torch.no_grad():
        mean_advantage = (advantages.abs() * response_mask).sum() / mask_sum
        frac_positive = ((advantages > 0).float() * response_mask).sum() / mask_sum
        mean_ratio = (ratio * response_mask).sum() / mask_sum
        # Check if ratio is outside the clip bounds
        clipped_lower = (ratio < 1.0 - clip_eps_lower).float()
        clipped_upper = (ratio > 1.0 + clip_eps_upper).float()
        clip_frac = ((clipped_lower + clipped_upper) * response_mask).sum() / mask_sum

    return {
        "loss": total_loss,
        "pg_loss": pg_loss.item(),
        "kl_reg": kl_reg.item(),
        "mean_advantage": mean_advantage.item(),
        "frac_positive_advantage": frac_positive.item(),
        "mean_is_ratio": mean_ratio.item(),
        "clip_fraction": clip_frac.item(),
    }


def _lookup_token_in_topk(
    token_ids: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_logprobs: torch.Tensor,
    floor_logprob: float = -20.0,
) -> torch.Tensor:
    """Look up log-probability of specific tokens in a top-K set.

    If a token is not in the top-K, returns floor_logprob.

    Args:
        token_ids: (B, T) target token ids to look up
        topk_indices: (B, T, K) top-K token indices
        topk_logprobs: (B, T, K) corresponding log-probabilities
        floor_logprob: value to use when token is not in top-K

    Returns:
        (B, T) log-probabilities of the target tokens
    """
    # Expand token_ids for comparison: (B, T, 1)
    target = token_ids.unsqueeze(-1)
    # Check which top-K position matches: (B, T, K)
    match_mask = topk_indices == target
    # If matched, extract the logprob; otherwise use floor
    matched_logprob = (topk_logprobs * match_mask.float()).sum(dim=-1)  # (B, T)
    has_match = match_mask.any(dim=-1).float()  # (B, T)
    result = matched_logprob * has_match + floor_logprob * (1 - has_match)
    return result
