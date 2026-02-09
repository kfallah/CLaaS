"""SDPO Loss: Policy Gradient with JSD-Derived Per-Token Advantages.

This module implements the Self-Distillation Policy Optimization (SDPO) loss
as described in Hübotter et al. (2026), "Reinforcement Learning via Self-Distillation".

SDPO is a policy gradient algorithm where per-token advantages are derived from
a divergence between student and teacher distributions. The key insight is that
SDPO does NOT directly minimize a KL divergence as a supervised loss. Instead,
it computes per-token divergences that serve as advantages in a standard policy
gradient framework.

The loss uses α-interpolated Jensen-Shannon divergence (α=0.5 by default) for
stability, and includes PPO-style clipped importance sampling for off-policy
correction when doing multiple gradient steps.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_sdpo_loss(
    student_logits: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    teacher_indices: torch.Tensor,
    response_mask: torch.Tensor,
    old_student_logprobs: torch.Tensor,
    response_ids: torch.Tensor,
    alpha: float = 0.5,
    clip_eps: float = 0.2,
    jsd_reg_weight: float = 0.5,
) -> dict:
    """Compute SDPO loss: policy gradient with JSD-derived per-token advantages.

    Args:
        student_logits: (B, T, V) student model logits at response positions (WITH grad)
        teacher_logprobs: (B, T, K_t) teacher's top-K log-probabilities (no grad)
        teacher_indices: (B, T, K_t) teacher's top-K token indices
        response_mask: (B, T) binary mask, 1 for response tokens
        old_student_logprobs: (B, T) student log-prob of chosen token at rollout time
        response_ids: (B, T) actual token ids in the response
        alpha: interpolation parameter (0.5 = symmetric JSD, SDPO default)
        clip_eps: PPO clip range for importance sampling
        jsd_reg_weight: weight for the logit-level JSD regularizer

    Returns:
        dict with 'loss' and diagnostic tensors
    """
    B, T, V = student_logits.shape

    # Step 1: Student log-probabilities (full, with gradient)
    student_log_probs = F.log_softmax(student_logits, dim=-1)  # (B, T, V)

    # Current student log-prob of the actually-generated token
    current_logprob_chosen = student_log_probs.gather(
        -1, response_ids.unsqueeze(-1)
    ).squeeze(-1)  # (B, T)

    # Step 2: Reconstruct sparse teacher distribution
    teacher_probs_sparse = teacher_logprobs.exp()  # (B, T, K_t)

    # Teacher probability of the actually-generated token
    # Check if response_ids[b, t] is in teacher_indices[b, t]
    teacher_logprob_chosen = _lookup_token_in_topk(
        response_ids,
        teacher_indices,
        teacher_logprobs,
        floor_logprob=-20.0,
    )  # (B, T)

    # Step 3: Compute per-token advantages via α-interpolated divergence
    #
    # For the generated token y_t at position t, we compute the JSD advantage:
    #   A(y_t, t) = -(log π_student(y_t) - log M(y_t))
    # where M(y_t) = α * π_student(y_t) + (1-α) * q_teacher(y_t)
    #
    # For α = 0.5 (JSD): M = 0.5 * (p_s + p_t)

    student_p_chosen = current_logprob_chosen.exp()  # (B, T)
    teacher_p_chosen = teacher_logprob_chosen.exp()  # (B, T)

    if alpha == 0.5:
        # JSD advantage for the chosen token
        M_chosen = 0.5 * (student_p_chosen + teacher_p_chosen)
        log_M_chosen = M_chosen.log().clamp(min=-20.0)  # numerical stability
        # Advantage: how much the student diverges from the mixture
        # Positive when student >> teacher → student should decrease
        # We negate so positive = "teacher wants this more" = student should increase
        advantages = -(current_logprob_chosen - log_M_chosen)
    else:
        # General α-interpolated divergence
        M_chosen = alpha * student_p_chosen + (1 - alpha) * teacher_p_chosen
        log_M_chosen = M_chosen.log().clamp(min=-20.0)
        # KL(student || M) contribution: student_logprob - log_M
        # KL(teacher || M) contribution: teacher_logprob - log_M
        # Combined with α weighting
        advantages = -(
            alpha * (current_logprob_chosen - log_M_chosen)
            + (1 - alpha) * (teacher_logprob_chosen - log_M_chosen)
        )

    # Step 4: PPO-style clipped policy gradient
    # Importance sampling ratio (current policy / old policy)
    log_ratio_is = current_logprob_chosen - old_student_logprobs
    ratio = log_ratio_is.exp().clamp(max=10.0)  # prevent explosion

    # Clipped surrogate objective
    # advantages are NOT differentiated through (detached)
    surr1 = ratio * advantages.detach()
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages.detach()
    pg_loss = -torch.min(surr1, surr2)

    # Apply response mask and average
    mask_sum = response_mask.sum().clamp(min=1.0)
    pg_loss = (pg_loss * response_mask).sum() / mask_sum

    # Step 5: Logit-level JSD regularizer
    # In addition to the token-level policy gradient, add a direct JSD loss
    # over the student's full distribution vs teacher's sparse distribution.
    # This provides the "logit-level" credit assignment SDPO describes.
    jsd_reg = _sparse_logit_jsd(
        student_log_probs,
        teacher_logprobs,
        teacher_probs_sparse,
        teacher_indices,
        response_mask,
        alpha,
    )

    total_loss = pg_loss + jsd_reg_weight * jsd_reg

    # Diagnostics
    with torch.no_grad():
        mean_advantage = (advantages.abs() * response_mask).sum() / mask_sum
        frac_positive = ((advantages > 0).float() * response_mask).sum() / mask_sum
        mean_ratio = (ratio * response_mask).sum() / mask_sum
        clip_frac = (
            ((ratio - 1.0).abs() > clip_eps).float() * response_mask
        ).sum() / mask_sum

    return {
        "loss": total_loss,
        "pg_loss": pg_loss.item(),
        "jsd_reg": jsd_reg.item(),
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


def _sparse_logit_jsd(
    student_log_probs: torch.Tensor,
    teacher_topk_logprobs: torch.Tensor,
    teacher_topk_probs: torch.Tensor,
    teacher_topk_indices: torch.Tensor,
    response_mask: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Compute JSD over the teacher's top-K token set only.

    This is the "logit-level" credit assignment that SDPO uses to provide
    dense supervision signal beyond just the chosen token.

    Args:
        student_log_probs: (B, T, V) full student log-probabilities
        teacher_topk_logprobs: (B, T, K) teacher's top-K log-probabilities
        teacher_topk_probs: (B, T, K) teacher's top-K probabilities
        teacher_topk_indices: (B, T, K) teacher's top-K token indices
        response_mask: (B, T) binary mask for response positions
        alpha: interpolation parameter for JSD

    Returns:
        Scalar JSD loss averaged over response positions
    """
    # Gather student probs/logprobs at teacher's top-K indices
    # student_log_probs: (B, T, V), teacher_topk_indices: (B, T, K)
    student_at_teacher_topk = student_log_probs.gather(
        -1, teacher_topk_indices
    )  # (B, T, K)
    student_probs_at_topk = student_at_teacher_topk.exp()

    # M = mixture distribution
    M_probs = alpha * student_probs_at_topk + (1 - alpha) * teacher_topk_probs
    M_log_probs = M_probs.log().clamp(min=-20.0)

    # JSD = α * KL(student || M) + (1-α) * KL(teacher || M)
    # KL(P || Q) = sum_v P(v) * (log P(v) - log Q(v))
    kl_student_M = (
        student_probs_at_topk * (student_at_teacher_topk - M_log_probs)
    ).sum(-1)
    kl_teacher_M = (
        teacher_topk_probs * (teacher_topk_logprobs - M_log_probs)
    ).sum(-1)
    jsd = alpha * kl_student_M + (1 - alpha) * kl_teacher_M

    # Clamp negative values (numerical precision issues)
    jsd = jsd.clamp(min=0.0)

    # Average over response positions
    mask_sum = response_mask.sum().clamp(min=1.0)
    jsd_loss = (jsd * response_mask).sum() / mask_sum
    return jsd_loss


def compute_token_level_only_loss(
    student_logits: torch.Tensor,
    teacher_logprob_chosen: torch.Tensor,
    response_mask: torch.Tensor,
    old_student_logprobs: torch.Tensor,
    response_ids: torch.Tensor,
    alpha: float = 0.5,
    clip_eps: float = 0.2,
) -> dict:
    """Simplified SDPO loss using only token-level advantages.

    This is the "lite" mode for use with Fireworks' K=5 logprobs,
    where we don't have enough teacher logprobs for the full
    logit-level JSD regularizer.

    Args:
        student_logits: (B, T, V) student logits at response positions
        teacher_logprob_chosen: (B, T) teacher log-prob of chosen token
        response_mask: (B, T) binary mask for response tokens
        old_student_logprobs: (B, T) student log-prob at rollout time
        response_ids: (B, T) actual token ids
        alpha: interpolation for advantage computation
        clip_eps: PPO clip range

    Returns:
        dict with 'loss' and diagnostics
    """
    B, T, V = student_logits.shape

    # Student log-prob of chosen token
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    current_logprob_chosen = student_log_probs.gather(
        -1, response_ids.unsqueeze(-1)
    ).squeeze(-1)

    student_p_chosen = current_logprob_chosen.exp()
    teacher_p_chosen = teacher_logprob_chosen.exp()

    # Compute advantage
    if alpha == 0.5:
        M_chosen = 0.5 * (student_p_chosen + teacher_p_chosen)
        log_M_chosen = M_chosen.log().clamp(min=-20.0)
        advantages = -(current_logprob_chosen - log_M_chosen)
    else:
        M_chosen = alpha * student_p_chosen + (1 - alpha) * teacher_p_chosen
        log_M_chosen = M_chosen.log().clamp(min=-20.0)
        advantages = -(
            alpha * (current_logprob_chosen - log_M_chosen)
            + (1 - alpha) * (teacher_logprob_chosen - log_M_chosen)
        )

    # PPO-style clipped objective
    log_ratio_is = current_logprob_chosen - old_student_logprobs
    ratio = log_ratio_is.exp().clamp(max=10.0)

    surr1 = ratio * advantages.detach()
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages.detach()
    pg_loss = -torch.min(surr1, surr2)

    mask_sum = response_mask.sum().clamp(min=1.0)
    pg_loss = (pg_loss * response_mask).sum() / mask_sum

    with torch.no_grad():
        mean_advantage = (advantages.abs() * response_mask).sum() / mask_sum
        frac_positive = ((advantages > 0).float() * response_mask).sum() / mask_sum
        mean_ratio = (ratio * response_mask).sum() / mask_sum
        clip_frac = (
            ((ratio - 1.0).abs() > clip_eps).float() * response_mask
        ).sum() / mask_sum

    return {
        "loss": pg_loss,
        "pg_loss": pg_loss.item(),
        "mean_advantage": mean_advantage.item(),
        "frac_positive_advantage": frac_positive.item(),
        "mean_is_ratio": mean_ratio.item(),
        "clip_fraction": clip_frac.item(),
    }
