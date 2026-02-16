"""Tests for SDPO loss computation."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from claas.core.types import SDPOLossInput  # noqa: E402
from claas.training.sdpo_loss import _lookup_token_in_topk, compute_sdpo_loss  # noqa: E402


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

    # Teacher top-K: realistic slices from full-vocabulary softmax
    full_logprobs = torch.log_softmax(torch.randn(B, T, V, device=device), dim=-1)
    teacher_logprobs, teacher_indices = full_logprobs.topk(K, dim=-1)

    # Response tokens (random)
    response_ids = torch.randint(0, V, (B, T), device=device)

    # Response mask (all 1s)
    response_mask = torch.ones(B, T, device=device)

    # Base model logprobs (detached, no LoRA)
    with torch.no_grad():
        base_logprobs = torch.log_softmax(
            torch.randn(B, T, V, device=device), dim=-1
        ).gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)

    # Old student logprobs (detached snapshot)
    with torch.no_grad():
        old_student_logprobs = torch.log_softmax(student_logits.detach(), dim=-1).gather(
            -1, response_ids.unsqueeze(-1)
        ).squeeze(-1)

    return {
        "student_logits": student_logits,
        "teacher_logprobs": teacher_logprobs,
        "teacher_indices": teacher_indices,
        "response_ids": response_ids,
        "response_mask": response_mask,
        "base_logprobs": base_logprobs,
        "old_student_logprobs": old_student_logprobs,
        "B": B,
        "T": T,
        "V": V,
        "K": K,
    }


class TestLookupTokenInTopk:
    """Tests for _lookup_token_in_topk."""

    def test_finds_token_in_topk(self, device):
        """Token in top-K returns correct logprob."""
        token_ids = torch.tensor([[5]], device=device)
        topk_indices = torch.tensor([[[5, 10, 15]]], device=device)
        topk_logprobs = torch.tensor([[[-1.0, -2.0, -3.0]]], device=device)

        result = _lookup_token_in_topk(token_ids, topk_indices, topk_logprobs)

        assert result.shape == (1, 1)
        assert torch.isclose(result[0, 0], torch.tensor(-1.0))

    def test_missing_token_uses_floor(self, device):
        """Token not in top-K returns floor value."""
        token_ids = torch.tensor([[99]], device=device)
        topk_indices = torch.tensor([[[5, 10, 15]]], device=device)
        topk_logprobs = torch.tensor([[[-1.0, -2.0, -3.0]]], device=device)

        result = _lookup_token_in_topk(
            token_ids, topk_indices, topk_logprobs, floor_logprob=-20.0
        )

        assert result.shape == (1, 1)
        assert torch.isclose(result[0, 0], torch.tensor(-20.0))

    def test_batch_processing(self, device):
        """Handles batch dimension correctly."""
        B, T, K = 2, 3, 4
        token_ids = torch.randint(0, 10, (B, T), device=device)
        topk_indices = torch.randint(0, 10, (B, T, K), device=device)
        topk_logprobs = torch.randn(B, T, K, device=device)

        result = _lookup_token_in_topk(token_ids, topk_indices, topk_logprobs)

        assert result.shape == (B, T)


class TestComputeSdpoLoss:
    """Tests for compute_sdpo_loss."""

    @staticmethod
    def _run_loss(sample_data: dict, **overrides):
        payload = {
            "student_logits": sample_data["student_logits"],
            "teacher_logprobs": sample_data["teacher_logprobs"],
            "teacher_indices": sample_data["teacher_indices"],
            "base_logprobs": sample_data["base_logprobs"],
            "response_mask": sample_data["response_mask"],
            "old_student_logprobs": sample_data["old_student_logprobs"],
            "response_ids": sample_data["response_ids"],
        }
        payload.update(overrides)
        return compute_sdpo_loss(SDPOLossInput(**payload))

    def test_returns_expected_keys(self, sample_data):
        """Loss dict contains all expected keys."""
        result = self._run_loss(sample_data)

        expected_keys = {
            "loss",
            "distill_loss",
            "kl_reg",
            "mean_is_ratio",
            "clip_fraction",
        }
        assert set(result.keys()) == expected_keys

    def test_loss_is_differentiable(self, sample_data):
        """Loss should have gradient through student logits."""
        result = self._run_loss(sample_data)

        result["loss"].backward()

        assert sample_data["student_logits"].grad is not None
        assert not torch.isnan(sample_data["student_logits"].grad).any()

    def test_alpha_affects_loss(self, sample_data):
        """Different alpha values produce different losses."""
        results = []
        for alpha in [0.3, 0.5, 0.7]:
            # Need fresh logits for each run
            logits = torch.randn_like(sample_data["student_logits"], requires_grad=True)

            result = self._run_loss(
                sample_data,
                student_logits=logits,
                alpha=alpha,
            )
            results.append(result["loss"].item())

        # Losses should be different for different alphas
        assert len(set(results)) > 1

    def test_kl_reg_weight_affects_loss(self, sample_data):
        """Different kl_reg_weight values produce different losses."""
        results = []
        for weight in [0.0, 0.1, 0.5]:
            # Need fresh logits for each run
            logits = torch.randn_like(sample_data["student_logits"], requires_grad=True)

            result = self._run_loss(
                sample_data,
                student_logits=logits,
                kl_reg_weight=weight,
            )
            results.append(result["loss"].item())

        # Losses should be different for different weights
        assert len(set(results)) > 1

    def test_clip_fraction_in_bounds(self, sample_data):
        """Clip fraction should be between 0 and 1."""
        result = self._run_loss(sample_data)

        assert 0.0 <= result["clip_fraction"] <= 1.0

    def test_on_policy_ratio_is_one(self, sample_data):
        """When old_logprobs match current, IS ratio should be ~1."""
        # Compute current logprobs to use as "old" (truly on-policy)
        with torch.no_grad():
            log_probs = torch.log_softmax(sample_data["student_logits"], dim=-1)
            old_logprobs = log_probs.gather(
                -1, sample_data["response_ids"].unsqueeze(-1)
            ).squeeze(-1)

        result = self._run_loss(
            sample_data,
            old_student_logprobs=old_logprobs,
        )

        # Mean IS ratio should be very close to 1 for on-policy
        assert torch.isclose(
            torch.tensor(result["mean_is_ratio"]),
            torch.tensor(1.0),
            atol=1e-4,
        )

    def test_is_clip_affects_ratio(self, sample_data):
        """Lower is_clip should increase clip fraction when off-policy."""
        # Use a highly off-policy situation
        with torch.no_grad():
            # Old logprobs very different from current
            off_policy_old = sample_data["old_student_logprobs"] - 3.0

        result_high = self._run_loss(
            sample_data,
            old_student_logprobs=off_policy_old,
            is_clip=10.0,
        )

        result_low = self._run_loss(
            sample_data,
            old_student_logprobs=off_policy_old,
            is_clip=2.0,
        )

        # Lower clip should result in more clipping
        assert result_low["clip_fraction"] >= result_high["clip_fraction"]

    def test_response_mask_applied(self, sample_data, device):
        """Loss should only consider masked positions."""
        B, T = sample_data["B"], sample_data["T"]

        # Mask out last 2 positions
        partial_mask = torch.ones(B, T, device=device)
        partial_mask[:, -2:] = 0

        result_full = self._run_loss(sample_data)

        result_partial = self._run_loss(
            sample_data,
            response_mask=partial_mask,
        )

        # Losses should differ when mask is different
        assert result_full["loss"].item() != result_partial["loss"].item()

    def test_reverse_kl_mode(self, sample_data):
        """alpha=1.0 should use reverse KL mode."""
        result = self._run_loss(sample_data, alpha=1.0)

        # Should still produce valid loss
        assert not torch.isnan(result["loss"])
        assert result["distill_loss"] is not None

    def test_kl_reg_non_negative(self, device):
        """KL regularization should be >= 0 (Schulman k3 estimator property)."""
        B, T, V, K = 2, 8, 100, 10

        for _ in range(5):
            student_logits = torch.randn(B, T, V, device=device, requires_grad=True)
            full_logprobs = torch.log_softmax(torch.randn(B, T, V, device=device), dim=-1)
            teacher_logprobs, teacher_indices = full_logprobs.topk(K, dim=-1)
            response_ids = torch.randint(0, V, (B, T), device=device)
            response_mask = torch.ones(B, T, device=device)

            with torch.no_grad():
                base_logprobs = torch.log_softmax(
                    torch.randn(B, T, V, device=device), dim=-1
                ).gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
                old_student_logprobs = torch.log_softmax(
                    student_logits.detach(), dim=-1
                ).gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)

            result = compute_sdpo_loss(SDPOLossInput(
                student_logits=student_logits,
                teacher_logprobs=teacher_logprobs,
                teacher_indices=teacher_indices,
                base_logprobs=base_logprobs,
                response_mask=response_mask,
                old_student_logprobs=old_student_logprobs,
                response_ids=response_ids,
            ))

            assert result["kl_reg"] >= -1e-6, (
                f"kl_reg should be non-negative, got {result['kl_reg']}"
            )

    def test_gjs_invariant_to_prenormalization(self, device):
        """GJS output should be the same whether teacher logprobs are pre-normalized or not."""
        B, T, V, K = 1, 5, 100, 10

        student_logits = torch.randn(B, T, V, device=device, requires_grad=True)
        response_ids = torch.randint(0, V, (B, T), device=device)
        response_mask = torch.ones(B, T, device=device)

        # Create an un-normalized top-K variant (raw logits)
        full_logits = torch.randn(B, T, V, device=device)
        teacher_logprobs_raw, teacher_indices = full_logits.topk(K, dim=-1)

        # Create normalized top-K variant from full-vocab log-softmax
        full_logprobs = torch.log_softmax(full_logits, dim=-1)
        teacher_logprobs_norm = full_logprobs.gather(-1, teacher_indices)

        with torch.no_grad():
            base_logprobs = torch.log_softmax(
                torch.randn(B, T, V, device=device), dim=-1
            ).gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
            old_student_logprobs = torch.log_softmax(
                student_logits.detach(), dim=-1
            ).gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)

        result_norm = compute_sdpo_loss(
            SDPOLossInput(
                student_logits=student_logits,
                teacher_logprobs=teacher_logprobs_norm,
                teacher_indices=teacher_indices,
                base_logprobs=base_logprobs,
                response_mask=response_mask,
                old_student_logprobs=old_student_logprobs,
                response_ids=response_ids,
            )
        )
        result_raw = compute_sdpo_loss(
            SDPOLossInput(
                student_logits=student_logits,
                teacher_logprobs=teacher_logprobs_raw,
                teacher_indices=teacher_indices,
                base_logprobs=base_logprobs,
                response_mask=response_mask,
                old_student_logprobs=old_student_logprobs,
                response_ids=response_ids,
            )
        )

        assert torch.isclose(
            torch.tensor(result_norm["distill_loss"]),
            torch.tensor(result_raw["distill_loss"]),
            atol=1e-5,
        ), (
            f"GJS should be invariant to pre-normalization: "
            f"{result_norm['distill_loss']} vs {result_raw['distill_loss']}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
