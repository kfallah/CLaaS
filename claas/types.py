"""Shared Pydantic models for CLaaS.

All typed data structures are defined here to avoid duplication
and ensure consistency across the codebase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import torch


class TrainingConfig(BaseModel):
    """Training configuration for distillation."""

    learning_rate: float = Field(
        default=1e-4,
        ge=1e-6,
        le=1e-2,
        description="Learning rate for LoRA parameter updates",
    )
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="GJS interpolation (0.5 = symmetric JSD, 1.0 = reverse KL)",
    )
    is_clip: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="Importance sampling ratio clip (exp space)",
    )
    max_grad_norm: float = Field(
        default=1.0,
        ge=0.0,
        description="Maximum gradient norm for clipping",
    )
    kl_reg_weight: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Weight for KL regularization to base policy",
    )
    teacher_top_k: int = Field(
        default=100,
        ge=10,
        le=100,
        description="Number of top logprobs to request from teacher",
    )
    teacher_mode: Literal["self", "remote"] = Field(
        default="self",
        description="Teacher source: 'self' uses detached student logits; "
        "'remote' scores with TeacherService.",
    )


class SDPOLossInput(BaseModel):
    """Typed input for SDPO loss computation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    student_logits: Any  # torch.Tensor (B, T, V)
    teacher_logprobs: Any  # torch.Tensor (B, T, K)
    teacher_indices: Any  # torch.Tensor (B, T, K)
    base_logprobs: Any  # torch.Tensor (B, T)
    response_mask: Any  # torch.Tensor (B, T)
    old_student_logprobs: Any  # torch.Tensor (B, T)
    response_ids: Any  # torch.Tensor (B, T)
    alpha: float = 0.5
    is_clip: float = 5.0
    kl_reg_weight: float = 0.1


class SDPOLossResult(TypedDict):
    """Result from SDPO loss computation.

    Uses TypedDict (not Pydantic) for dict-like subscript access:
    result["loss"].backward() works directly.
    """

    loss: "torch.Tensor"
    distill_loss: float
    kl_reg: float
    mean_is_ratio: float
    clip_fraction: float
