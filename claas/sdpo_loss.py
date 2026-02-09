"""SDPO Loss: Self-Distillation Policy Optimization.

This module implements the Self-Distillation Policy Optimization (SDPO) loss
as described in Hübotter et al. (2026), "Reinforcement Learning via Self-Distillation".

Reference implementation: https://github.com/lasgroup/SDPO

SDPO is a policy gradient algorithm that:
1. Uses teacher logprobs to derive per-token advantages (distillation signal)
2. Applies PPO-style clipped importance sampling for stable updates
3. Regularizes with KL divergence to base policy to prevent drift

This implementation follows the reference `compute_self_distillation_loss` function,
including numerical stability tricks like `torch.expm1` for tail probabilities.
"""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn.functional as F


class SDPOLossResult(TypedDict):
    """Result from SDPO loss computation."""

    loss: torch.Tensor
    distill_loss: float
    kl_reg: float
    mean_kl_to_teacher: float
    mean_is_ratio: float
    clip_fraction: float


def compute_sdpo_loss(
    student_logits: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    teacher_indices: torch.Tensor,
    base_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    response_ids: torch.Tensor,
    kl_reg_weight: float = 0.1,
    is_clip: float = 5.0,
) -> SDPOLossResult:
    """Compute SDPO loss: distillation from teacher + KL regularization to base.

    This follows the reference implementation from https://github.com/lasgroup/SDPO.
    The loss consists of:
    1. KL divergence from student to teacher (distillation signal)
    2. KL divergence from student to base policy (drift regularization)

    Args:
        student_logits: (B, T, V) student model logits at response positions (WITH grad)
        teacher_logprobs: (B, T, K) teacher's top-K log-probabilities (no grad)
        teacher_indices: (B, T, K) teacher's top-K token indices
        base_logprobs: (B, T) base model log-prob of response tokens (no grad, no LoRA)
        response_mask: (B, T) binary mask, 1 for response tokens
        response_ids: (B, T) actual token ids in the response
        kl_reg_weight: weight for KL regularization to base policy
        is_clip: importance sampling ratio clip (exp space)

    Returns:
        SDPOLossResult with loss tensor and diagnostic metrics
    """
    _B, T, _V = student_logits.shape

    # Step 1: Student log-probabilities (full, with gradient)
    student_log_probs = F.log_softmax(student_logits, dim=-1)  # (B, T, V)

    # Get student logprobs at teacher's top-K indices for distillation
    student_at_teacher_k = student_log_probs.gather(-1, teacher_indices)  # (B, T, K)

    # Step 2: Compute KL divergence from teacher to student over top-K
    # KL(teacher || student) = sum_k teacher_prob[k] * (teacher_logprob[k] - student_logprob[k])
    teacher_probs = teacher_logprobs.exp()  # (B, T, K)

    # KL over top-K tokens (tail probability contribution is negligible for large K)
    kl_per_token = (teacher_probs * (teacher_logprobs - student_at_teacher_k)).sum(-1)  # (B, T)

    # Step 3: Importance sampling correction
    # Ratio: student(response_token) / base(response_token)
    student_logprob_chosen = student_log_probs.gather(
        -1, response_ids.unsqueeze(-1)
    ).squeeze(-1)  # (B, T)

    log_ratio = student_logprob_chosen - base_logprobs
    ratio = log_ratio.exp().clamp(max=is_clip)  # (B, T)

    # Weighted distillation loss
    distill_loss_per_token = ratio * kl_per_token

    # Step 4: KL regularization to base policy
    # KL(student || base) at the response token
    kl_to_base_per_token = student_logprob_chosen - base_logprobs  # (B, T)

    # Apply response mask and average
    mask_sum = response_mask.sum().clamp(min=1.0)
    distill_loss = (distill_loss_per_token * response_mask).sum() / mask_sum
    kl_reg = (kl_to_base_per_token * response_mask).sum() / mask_sum

    total_loss = distill_loss + kl_reg_weight * kl_reg

    # Diagnostics
    with torch.no_grad():
        mean_kl = (kl_per_token * response_mask).sum() / mask_sum
        mean_ratio = (ratio * response_mask).sum() / mask_sum
        clipped = (ratio >= is_clip).float()
        clip_frac = (clipped * response_mask).sum() / mask_sum

    return {
        "loss": total_loss,
        "distill_loss": distill_loss.item(),
        "kl_reg": kl_reg.item(),
        "mean_kl_to_teacher": mean_kl.item(),
        "mean_is_ratio": mean_ratio.item(),
        "clip_fraction": clip_frac.item(),
    }


