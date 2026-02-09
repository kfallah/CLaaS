"""Tests for SDPO loss computation."""

from __future__ import annotations

import pytest
import torch

from claas.sdpo_loss import compute_sdpo_loss


@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cpu")


@pytest.fixture
def sample_data(device):
    """Create sample data for testing."""
    B, T, V = 1, 5, 100  # batch, sequence length, vocab size
    K = 10  # teacher top-K

    # Student logits (random, with gradient)
    student_logits = torch.randn(B, T, V, device=device, requires_grad=True)

    # Teacher top-K (random)
    teacher_logprobs = torch.randn(B, T, K, device=device)
    teacher_logprobs = torch.log_softmax(teacher_logprobs, dim=-1)  # normalize to valid logprobs
    teacher_indices = torch.randint(0, V, (B, T, K), device=device)

    # Response tokens (random)
    response_ids = torch.randint(0, V, (B, T), device=device)

    # Response mask (all 1s)
    response_mask = torch.ones(B, T, device=device)

    # Base model logprobs (detached, no LoRA)
    with torch.no_grad():
        base_logprobs = torch.log_softmax(
            torch.randn(B, T, V, device=device), dim=-1
        ).gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)

    return {
        "student_logits": student_logits,
        "teacher_logprobs": teacher_logprobs,
        "teacher_indices": teacher_indices,
        "response_ids": response_ids,
        "response_mask": response_mask,
        "base_logprobs": base_logprobs,
        "B": B,
        "T": T,
        "V": V,
        "K": K,
    }


class TestComputeSdpoLoss:
    """Tests for compute_sdpo_loss."""

    def test_returns_expected_keys(self, sample_data):
        """Loss dict contains all expected keys."""
        result = compute_sdpo_loss(
            student_logits=sample_data["student_logits"],
            teacher_logprobs=sample_data["teacher_logprobs"],
            teacher_indices=sample_data["teacher_indices"],
            base_logprobs=sample_data["base_logprobs"],
            response_mask=sample_data["response_mask"],
            response_ids=sample_data["response_ids"],
        )

        expected_keys = {
            "loss",
            "distill_loss",
            "kl_reg",
            "mean_kl_to_teacher",
            "mean_is_ratio",
            "clip_fraction",
        }
        assert set(result.keys()) == expected_keys

    def test_loss_is_differentiable(self, sample_data):
        """Loss should have gradient through student logits."""
        result = compute_sdpo_loss(
            student_logits=sample_data["student_logits"],
            teacher_logprobs=sample_data["teacher_logprobs"],
            teacher_indices=sample_data["teacher_indices"],
            base_logprobs=sample_data["base_logprobs"],
            response_mask=sample_data["response_mask"],
            response_ids=sample_data["response_ids"],
        )

        result["loss"].backward()

        assert sample_data["student_logits"].grad is not None
        assert not torch.isnan(sample_data["student_logits"].grad).any()

    def test_kl_reg_weight_affects_loss(self, sample_data):
        """Different kl_reg_weight values produce different losses."""
        results = []
        for weight in [0.0, 0.1, 0.5]:
            # Need fresh logits for each run
            logits = torch.randn_like(sample_data["student_logits"], requires_grad=True)

            result = compute_sdpo_loss(
                student_logits=logits,
                teacher_logprobs=sample_data["teacher_logprobs"],
                teacher_indices=sample_data["teacher_indices"],
                base_logprobs=sample_data["base_logprobs"],
                response_mask=sample_data["response_mask"],
                response_ids=sample_data["response_ids"],
                kl_reg_weight=weight,
            )
            results.append(result["loss"].item())

        # Losses should be different for different weights
        assert len(set(results)) > 1

    def test_clip_fraction_in_bounds(self, sample_data):
        """Clip fraction should be between 0 and 1."""
        result = compute_sdpo_loss(
            student_logits=sample_data["student_logits"],
            teacher_logprobs=sample_data["teacher_logprobs"],
            teacher_indices=sample_data["teacher_indices"],
            base_logprobs=sample_data["base_logprobs"],
            response_mask=sample_data["response_mask"],
            response_ids=sample_data["response_ids"],
        )

        assert 0.0 <= result["clip_fraction"] <= 1.0

    def test_is_ratio_with_matching_base(self, sample_data, device):
        """When student matches base, IS ratio should be ~1."""
        B, T, V = sample_data["B"], sample_data["T"], sample_data["V"]

        # Create student logits that match base
        base_logits = torch.randn(B, T, V, device=device)
        student_logits = base_logits.clone().requires_grad_(True)

        with torch.no_grad():
            base_logprobs = torch.log_softmax(base_logits, dim=-1).gather(
                -1, sample_data["response_ids"].unsqueeze(-1)
            ).squeeze(-1)

        result = compute_sdpo_loss(
            student_logits=student_logits,
            teacher_logprobs=sample_data["teacher_logprobs"],
            teacher_indices=sample_data["teacher_indices"],
            base_logprobs=base_logprobs,
            response_mask=sample_data["response_mask"],
            response_ids=sample_data["response_ids"],
        )

        # Mean IS ratio should be very close to 1 when student = base
        assert torch.isclose(
            torch.tensor(result["mean_is_ratio"]),
            torch.tensor(1.0),
            atol=1e-4,
        )

    def test_distill_loss_is_non_negative(self, sample_data):
        """Distillation loss (KL divergence) should be non-negative."""
        result = compute_sdpo_loss(
            student_logits=sample_data["student_logits"],
            teacher_logprobs=sample_data["teacher_logprobs"],
            teacher_indices=sample_data["teacher_indices"],
            base_logprobs=sample_data["base_logprobs"],
            response_mask=sample_data["response_mask"],
            response_ids=sample_data["response_ids"],
        )

        # KL divergence is always >= 0
        assert result["distill_loss"] >= 0

    def test_response_mask_applied(self, sample_data, device):
        """Loss should only consider masked positions."""
        B, T = sample_data["B"], sample_data["T"]

        # Mask out last 2 positions
        partial_mask = torch.ones(B, T, device=device)
        partial_mask[:, -2:] = 0

        result_full = compute_sdpo_loss(
            student_logits=sample_data["student_logits"],
            teacher_logprobs=sample_data["teacher_logprobs"],
            teacher_indices=sample_data["teacher_indices"],
            base_logprobs=sample_data["base_logprobs"],
            response_mask=sample_data["response_mask"],
            response_ids=sample_data["response_ids"],
        )

        result_partial = compute_sdpo_loss(
            student_logits=sample_data["student_logits"],
            teacher_logprobs=sample_data["teacher_logprobs"],
            teacher_indices=sample_data["teacher_indices"],
            base_logprobs=sample_data["base_logprobs"],
            response_mask=partial_mask,
            response_ids=sample_data["response_ids"],
        )

        # Losses should differ when mask is different
        assert result_full["loss"].item() != result_partial["loss"].item()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
