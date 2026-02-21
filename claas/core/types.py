"""Shared Pydantic models for CLaaS.

All typed data structures are defined here to avoid duplication
and ensure consistency across the codebase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import torch


class ChatMessage(TypedDict):
    """Typed chat message invariant for teacher prompt formatting."""

    role: Literal["system", "user", "assistant"]
    content: str


class TrainingConfig(BaseModel):
    """Training configuration for distillation."""

    learning_rate: float = Field(
        default=5e-5,
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
        default=0.001,
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
    kl_reg_weight: float = 0.001


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
    rollout_logprobs: list[float] = Field(
        ...,
        min_length=1,
        description="Log-probabilities from the inference server that generated the rollout.",
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training configuration",
    )
    prompt_token_ids: list[int] | None = Field(
        default=None,
        description="Pre-tokenized prompt token IDs (avoids decode/re-encode mismatch).",
    )
    response_token_ids: list[int] | None = Field(
        default=None,
        description="Pre-tokenized response token IDs (avoids decode/re-encode mismatch).",
    )
    user_prompt: str | None = Field(
        default=None,
        description="Clean user prompt for teacher construction (without chat template decoration).",
    )


class TeacherTokenLogprobs(BaseModel):
    """Top-k teacher token log-probabilities at one response position."""

    indices: list[int]
    logprobs: list[float]


class DistillRequestPayload(BaseModel):
    """Typed payload forwarded to the configured training engine."""

    lora_id: str
    prompt: str
    response: str
    feedback: str
    rollout_logprobs: list[float]
    training: TrainingConfig
    teacher_result: list[TeacherTokenLogprobs] | None = None
    prompt_token_ids: list[int] | None = None
    response_token_ids: list[int] | None = None
    user_prompt: str | None = None
    save_in_place: bool = False


class DistillBatchItem(BaseModel):
    """One prompt/response/feedback sample used in batched distillation."""

    prompt: str
    response: str
    feedback: str
    rollout_logprobs: list[float]
    teacher_result: list[TeacherTokenLogprobs] | None = None
    prompt_token_ids: list[int] | None = None
    response_token_ids: list[int] | None = None
    user_prompt: str | None = None


class DistillBatchRequestPayload(BaseModel):
    """Typed batched payload forwarded to the training engine."""

    lora_id: str
    training: TrainingConfig
    samples: list[DistillBatchItem] = Field(min_length=1)
    save_in_place: bool = False



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


class LoraExportPayload(BaseModel):
    """Typed LoRA export payload used by training engines."""

    filename: str
    content: bytes


class LoraExistsPayload(BaseModel):
    """Typed LoRA existence payload used by training engines."""

    exists: bool



class LoraRuntimeRef(BaseModel):
    """Runtime LoRA reference for vLLM adapter reload operations."""

    vllm_name: str
    lora_path: str



class FeedbackOrchestration(BaseModel):
    """Runtime orchestration options for feedback updates."""

    sleep_before: bool = True
    wake_after: bool = True
    wake_on_failure: bool = True
    sleep_level: int = Field(default=1, ge=1, le=2)


class FeedbackBatchRequest(BaseModel):
    """Request for a feedback-triggered batched LoRA update."""

    requests: list[DistillRequest] = Field(min_length=1)
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
    batch_size: int


class FeedbackLogRecord(BaseModel):
    """Structured log record for feedback orchestration."""

    request_id: str
    timestamp_utc: str
    status: str
    phase: str
    lora_id: str
    teacher_mode: str
    requests: list[DistillRequest]
    vllm: FeedbackLogVllmState
    timing_ms: FeedbackTimingMs
    batch_samples: list[DistillBatchItem]
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
        default=32,
        ge=4,
        le=128,
        description="LoRA rank",
    )
    lora_alpha: int = Field(
        default=64,
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


class LoraDeleteResponse(BaseModel):
    """Response from LoRA deletion."""

    deleted: bool


class ServiceHealth(BaseModel):
    """Health status for a backing service (worker or teacher)."""

    status: str
    error: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    worker: ServiceHealth | None = None
    teacher: ServiceHealth | None = None


# --- Inference Request Models ---


class ChatCompletionMessage(BaseModel):
    """A single message in a chat completion request."""

    role: str
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = ""
    messages: list[ChatCompletionMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    stop: list[str] | None = None


class CompletionRequest(BaseModel):
    """OpenAI-compatible text completion request."""

    model: str = ""
    prompt: str
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    stop: list[str] | None = None


# --- Inference Response Models ---


class ChatCompletionChoiceMessage(BaseModel):
    """Message within a chat completion choice."""

    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    """A single choice in a chat completion response."""

    index: int = 0
    message: ChatCompletionChoiceMessage
    finish_reason: str = "stop"


class CompletionUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage


class TextCompletionChoice(BaseModel):
    """A single choice in a text completion response."""

    index: int = 0
    text: str
    finish_reason: str = "stop"


class TextCompletionResponse(BaseModel):
    """OpenAI-compatible text completion response."""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[TextCompletionChoice]
    usage: CompletionUsage


class RawCompletionResponse(BaseModel):
    """Cached raw completion for the training pipeline."""

    prompt: str
    response: str
    token_ids: list[int] | None = None
    prompt_token_ids: list[int] | None = None
    logprobs: list[float] | None = None
