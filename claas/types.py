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
        description="Teacher source: 'self' uses base model conditioned on feedback; "
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


# --- API Request/Response Models ---


class DistillRequest(BaseModel):
    """Request for a distillation step."""

    lora_id: str = Field(
        ...,
        description="LoRA identifier (e.g., 'user123/coder-v1')",
    )
    prompt: str = Field(
        ...,
        min_length=1,
        description="User prompt that generated the response",
    )
    response: str = Field(
        ...,
        min_length=1,
        description="Student's response to learn from",
    )
    feedback: str = Field(
        ...,
        min_length=1,
        description="Feedback about response quality",
    )
    rollout_logprobs: list[float] | None = Field(
        default=None,
        description="Log-probabilities from the inference server that generated the rollout. "
        "Required for proper off-policy IS correction. If not provided, logprobs are computed "
        "from the current model (which is incorrect for off-policy learning).",
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training configuration",
    )


class DistillResponse(BaseModel):
    """Response from a distillation step."""

    lora_id: str = Field(
        ...,
        description="Updated LoRA identifier",
    )
    metadata: dict[str, Any] = Field(
        ...,
        description="Training metrics and diagnostics",
    )


class FeedbackOrchestration(BaseModel):
    """Runtime orchestration options for feedback updates."""

    sleep_before: bool = True
    wake_after: bool = True
    wake_on_failure: bool = True
    sleep_level: int = Field(default=1, ge=1, le=2)


class FeedbackRequest(BaseModel):
    """Request for a feedback-triggered in-place LoRA update."""

    lora_id: str = Field(
        ...,
        description="Fixed LoRA identifier to update in place",
    )
    prompt: str = Field(..., min_length=1)
    response: str = Field(..., min_length=1)
    feedback: str = Field(..., min_length=1)
    rollout_logprobs: list[float] | None = Field(default=None)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    orchestration: FeedbackOrchestration = Field(default_factory=FeedbackOrchestration)


class FeedbackLogVllmState(BaseModel):
    """vLLM orchestration state persisted to logs."""

    slept: bool
    woke: bool


class FeedbackTimingMs(BaseModel):
    """Timing breakdown for feedback orchestration."""

    sleep: int = 0
    distill: int = 0
    save: int = 0
    wake: int = 0
    logprobs: int = 0
    total: int = 0


class FeedbackResponse(BaseModel):
    """Response from feedback-triggered LoRA update orchestration."""

    status: str
    request_id: str
    lora_id: str
    distill_result: DistillResponse | None = None
    vllm: FeedbackLogVllmState
    feedback_log_path: str
    timing_ms: FeedbackTimingMs


class FeedbackLogRecord(BaseModel):
    """Structured log record for feedback orchestration."""

    request_id: str
    timestamp_utc: str
    status: str
    phase: str
    lora_id: str
    teacher_mode: str
    request: FeedbackRequest
    vllm: FeedbackLogVllmState
    timing_ms: FeedbackTimingMs
    distill_result: DistillResponse | None = None
    error: str | None = None


class LoraInitRequest(BaseModel):
    """Request to initialize a new LoRA adapter."""

    lora_id: str = Field(
        ...,
        description="LoRA identifier (e.g., 'user123/coder-v1')",
    )
    base_model: str = Field(
        default="Qwen/Qwen3-8B",
        description="Base model the LoRA will be applied to",
    )
    lora_r: int = Field(
        default=16,
        ge=4,
        le=128,
        description="LoRA rank",
    )
    lora_alpha: int = Field(
        default=32,
        ge=8,
        le=256,
        description="LoRA alpha scaling factor",
    )
    target_modules: list[str] | None = Field(
        default=None,
        description="Modules to apply LoRA to (defaults to attention + MLP)",
    )


class LoraInitResponse(BaseModel):
    """Response from LoRA initialization."""

    lora_id: str = Field(
        ...,
        description="LoRA identifier of the initialized adapter",
    )


class LoraListResponse(BaseModel):
    """Response listing all LoRA adapters."""

    loras: list[str] = Field(
        ...,
        description="List of LoRA identifiers",
    )


class ServiceHealth(BaseModel):
    """Health status for a backing service (worker or teacher)."""

    status: str
    error: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    worker: ServiceHealth | None = None
    teacher: ServiceHealth | None = None
