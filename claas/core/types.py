"""Shared Pydantic models for CLaaS.

All typed data structures are defined here to avoid duplication
and ensure consistency across the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    import torch


class ChatMessage(TypedDict):
    """Typed chat message invariant for teacher prompt formatting."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class TrainingConfig:
    """Training hyperparameters (dataclass for Hydra structured-config compatibility)."""

    learning_rate: float = 3e-5
    alpha: float = 0.5
    is_clip: float = 5.0
    max_grad_norm: float = 1.0
    kl_reg_weight: float = 0.0
    teacher_top_k: int = 100


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
    training: TrainingConfig = Field(default_factory=TrainingConfig)


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


class DistillBatchItem(BaseModel):
    """Cache-enriched training sample, constructed by the API feedback endpoint.

    The API resolves a FeedbackItem against the completion cache to produce this.
    FeedbackItem carries what the client knows (clean prompt, visible response,
    feedback text); DistillBatchItem adds what training needs (token IDs,
    logprobs) from the cached CompletionCacheEntry.

    Fields sourced from CompletionCacheEntry:
        prompt, response, response_logprobs, prompt_token_ids, response_token_ids
    Fields sourced from FeedbackItem:
        feedback, user_prompt
    """

    prompt: str = Field(description="Chat-template-decorated prompt text from the completion cache.")
    response: str = Field(description="Raw model response text from the completion cache.")
    feedback: str = Field(description="User's feedback text (from the client's FeedbackItem).")
    response_logprobs: list[float] = Field(description="Per-token log-probabilities of the student rollout.")
    prompt_token_ids: list[int] = Field(description="Tokenized prompt (chat template applied).")
    response_token_ids: list[int] = Field(description="Tokenized response (no special token stripping).")
    user_prompt: str = Field(
        description=(
            "Clean user prompt without chat-template decoration (from FeedbackItem.prompt). "
            "Used by teacher prompt construction so the teacher sees the original question, "
            "not a nested chat template."
        ),
    )
    system_prompt: str = Field(
        description=(
            "System prompt from the chat completion request. "
            "Passed to the teacher so it scores under the same context as the student."
        ),
    )


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



class FeedbackItem(BaseModel):
    """Client-facing feedback request for one prompt/response pair.

    This is the external API schema — the client sends this to POST /v1/feedback.
    The API resolves each FeedbackItem against the completion cache (keyed by
    a SHA-256 hash of the visible response text) to produce a DistillBatchItem
    with token IDs and logprobs attached.
    """

    lora_id: str = Field(
        ...,
        description="LoRA identifier (e.g., 'user123/coder-v1')",
    )
    prompt: str = Field(
        ...,
        min_length=1,
        description=(
            "Clean user prompt text (no chat-template decoration). "
            "Becomes DistillBatchItem.user_prompt for teacher prompt construction."
        ),
    )
    response: str = Field(
        ...,
        min_length=1,
        description=(
            "Visible assistant response text. Used as the cache lookup key "
            "(SHA-256 hash) to retrieve token IDs and logprobs."
        ),
    )
    feedback: str = Field(
        ...,
        min_length=1,
        description="User's feedback about response quality, forwarded to the teacher.",
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training configuration for this distillation step.",
    )


class FeedbackOrchestration(BaseModel):
    """Runtime orchestration options for feedback updates."""

    sleep_before: bool = True
    wake_after: bool = True
    wake_on_failure: bool = True
    sleep_level: int = Field(default=1, ge=1, le=2)


class FeedbackBatchRequest(BaseModel):
    """Request for a feedback-triggered batched LoRA update."""

    requests: list[FeedbackItem] = Field(min_length=1)
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
    requests: list[FeedbackItem]
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


# --- Inference Request Models ---


class ChatCompletionMessage(BaseModel):
    """A single message in a chat completion request."""

    role: str
    content: Any = ""


class ScoreRequest(BaseModel):
    """Request to score a completion by computing per-token logprobs."""

    model: str
    messages: list[ChatCompletionMessage]
    completion: str


class ScoreResponse(BaseModel):
    """Response from scoring a completion."""

    logprobs: list[float]
    tokens: list[str]
    prompt_tokens: int
    completion_tokens: int
    logprob_sum: float


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = ""
    messages: list[ChatCompletionMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    stop: list[str] | None = None
    logprobs: bool = False
    top_logprobs: int = 1


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


class TopLogprob(BaseModel):
    """A single top logprob entry."""

    token: str
    logprob: float
    bytes: list[int] | None = None


class TokenLogprob(BaseModel):
    """Logprob info for a single generated token."""

    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprob] = []


class ChoiceLogprobs(BaseModel):
    """Logprobs attached to a chat completion choice."""

    content: list[TokenLogprob] = []


class ChatCompletionChoiceMessage(BaseModel):
    """Message within a chat completion choice."""

    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    """A single choice in a chat completion response."""

    index: int = 0
    message: ChatCompletionChoiceMessage
    finish_reason: str = "stop"
    logprobs: ChoiceLogprobs | None = None


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
