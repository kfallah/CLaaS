"""SDPO Loss: Self-Distillation Policy Optimization.

This module implements the Self-Distillation Policy Optimization (SDPO) loss
as described in Hübotter et al. (2026), "Reinforcement Learning via Self-Distillation".

Reference implementation: https://github.com/lasgroup/SDPO

SDPO uses Generalized Jensen-Shannon Divergence (GJS) for distillation, with:
1. Alpha-interpolated mixture distribution between student and teacher
2. IS ratio clipping for off-policy correction
3. KL regularization to base policy to prevent drift
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from .types import SDPOLossInput, SDPOLossResult

logger = logging.getLogger(__name__)


def compute_sdpo_loss(loss_input: SDPOLossInput) -> SDPOLossResult:
    """Compute SDPO loss: Generalized JSD distillation + KL regularization to base.

    This follows the reference implementation from https://github.com/lasgroup/SDPO.
    The loss consists of:
    1. Generalized Jensen-Shannon Divergence between student and teacher
    2. IS ratio clipping for off-policy correction
    3. KL divergence from student to base policy (drift regularization)

    Args:
        loss_input: SDPOLossInput containing all required tensors and hyperparameters

    Returns:
        SDPOLossResult with loss tensor and diagnostic metrics
    """
    # Extract from typed input
    student_logits = loss_input.student_logits
    teacher_logprobs = loss_input.teacher_logprobs
    teacher_indices = loss_input.teacher_indices
    base_logprobs = loss_input.base_logprobs
    response_mask = loss_input.response_mask
    old_student_logprobs = loss_input.old_student_logprobs
    response_ids = loss_input.response_ids
    alpha = loss_input.alpha
    is_clip = loss_input.is_clip
    kl_reg_weight = loss_input.kl_reg_weight

    _B, _T, _V = student_logits.shape

    # Step 1: Student log-probabilities (full, with gradient)
    student_log_probs = F.log_softmax(student_logits, dim=-1)  # (B, T, V)

    # Get student logprobs at teacher's top-K indices
    student_at_teacher_k = student_log_probs.gather(-1, teacher_indices)  # (B, T, K)

    # Step 2: Compute Generalized Jensen-Shannon Divergence over top-K
    if alpha < 1.0:
        # Full GJS with mixture distribution
        # log(alpha * teacher + (1-alpha) * student) via logsumexp
        alpha_t = torch.tensor(alpha, dtype=student_log_probs.dtype, device=student_log_probs.device)
        log_alpha = torch.log(alpha_t)
        log_one_minus_alpha = torch.log(1 - alpha_t)

        # Mixture: M = alpha * teacher + (1-alpha) * student
        mixture_log_probs = torch.logsumexp(
            torch.stack([
                student_at_teacher_k + log_one_minus_alpha,
                teacher_logprobs + log_alpha,
            ]),
            dim=0,
        )  # (B, T, K)

        # KL(teacher || M) and KL(student || M)
        # Using F.kl_div with log_target=True: KL(P||Q) = sum(exp(log_P) * (log_P - log_Q))
        teacher_probs = teacher_logprobs.exp()
        student_probs_k = student_at_teacher_k.exp()

        kl_teacher_M = (teacher_probs * (teacher_logprobs - mixture_log_probs)).sum(-1)  # (B, T)
        kl_student_M = (student_probs_k * (student_at_teacher_k - mixture_log_probs)).sum(-1)  # (B, T)

        # GJS = alpha * KL(teacher||M) + (1-alpha) * KL(student||M)
        per_token_loss = (1.0 - alpha_t) * kl_student_M + alpha_t * kl_teacher_M  # (B, T)
    else:
        # Reverse KL (alpha = 1.0): student learns from teacher
        # per_token_loss = (student_logprob - teacher_logprob).detach() * student_logprob
        teacher_logprob_chosen = _lookup_token_in_topk(
            response_ids,
            teacher_indices,
            teacher_logprobs,
            floor_logprob=-20.0,
        )
        student_logprob_chosen = student_log_probs.gather(
            -1, response_ids.unsqueeze(-1)
        ).squeeze(-1)

        log_ratio = student_logprob_chosen - teacher_logprob_chosen
        per_token_loss = log_ratio.detach() * student_logprob_chosen  # (B, T)

    # Step 3: IS ratio clipping for off-policy correction
    student_logprob_chosen = student_log_probs.gather(
        -1, response_ids.unsqueeze(-1)
    ).squeeze(-1)  # (B, T)

    negative_approx_kl = (student_logprob_chosen - old_student_logprobs).detach()
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)

    per_token_loss = per_token_loss * ratio

    # Step 4: KL regularization to base policy
    kl_to_base_per_token = student_logprob_chosen - base_logprobs  # (B, T)

    # Apply response mask and average
    mask_sum = response_mask.sum().clamp(min=1.0)
    distill_loss = (per_token_loss * response_mask).sum() / mask_sum
    kl_reg = (kl_to_base_per_token * response_mask).sum() / mask_sum

    total_loss = distill_loss + kl_reg_weight * kl_reg

    # Diagnostics
    with torch.no_grad():
        mean_ratio = (ratio * response_mask).sum() / mask_sum
        clipped = (ratio >= is_clip).float()
        clip_frac = (clipped * response_mask).sum() / mask_sum

    return {
        "loss": total_loss,
        "distill_loss": distill_loss.item(),
        "kl_reg": kl_reg.item(),
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
