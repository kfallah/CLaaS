"""Data structures for the SDPO continual learning evaluation harness."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from claas.core.config import DEFAULT_SYSTEM_PROMPT
from claas.core.types import ChatMessage


@dataclass
class ChatRequestParams:
    """Resolved parameters for a /v1/chat/completions request."""

    base_url: str
    headers: dict[str, str]
    model: str
    messages: list[dict[str, str]]


def openclaw_chat_params(
    openclaw_url: str,
    openclaw_api_key: str,
    prompt: str,
) -> ChatRequestParams:
    """Build chat completion params routed through OpenClaw.

    OpenClaw injects the full agent system prompt and context, so we only
    send the bare user message.
    """
    return ChatRequestParams(
        base_url=openclaw_url,
        headers={"Authorization": f"Bearer {openclaw_api_key}"},
        model="openclaw",
        messages=[{"role": "user", "content": prompt}],
    )


def direct_vllm_chat_params(
    vllm_url: str,
    vllm_api_key: str,
    model: str,
    prompt: str,
) -> ChatRequestParams:
    """Build chat completion params for direct vLLM communication.

    Manually prepends the default system prompt since there is no
    gateway to inject it.
    """
    return ChatRequestParams(
        base_url=vllm_url,
        headers={"Authorization": f"Bearer {vllm_api_key}"} if vllm_api_key else {},
        model=model,
        messages=[
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )


@dataclass
class EvalRollout:
    """Logged prompt/response transcript with metric-specific metadata."""

    metric: str
    messages: list[ChatMessage]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class HarnessConfig:
    """Top-level configuration for an evaluation run."""

    mode: str = "local"  # "local" (GPU + vLLM) or "tinker" (no GPU, Tinker proxy)
    claas_url: str = "http://localhost:8080"
    vllm_url: str = "http://localhost:8000"
    vllm_api_key: str = "sk-local"
    vllm_model_name: str = "qwen3-8b"
    preferences: list[str] = field(default_factory=lambda: ["no_emoji", "concise", "identity"])
    num_steps: int = 20
    output_dir: str = "./data/evals"
    gemini_api_key: str | None = None
    metrics: list[str] = field(default_factory=lambda: ["logprob"])
    plots: bool = True
    collapse_steps: set[int] | None = None
    lora_id_prefix: str = "eval"
    seed: int = 42
    openclaw_url: str | None = None
    openclaw_api_key: str = "openclaw-local-dev-token"
    proxy_url: str | None = None
    base_model: str = "Qwen/Qwen3-8B"
    batch_size: int = 4
    steps_per_batch: int = 1  # gradient steps per feedback batch (>=1)


@dataclass
class LogprobMargin:
    """Logprob margin between positive and negative examples."""

    positive_logprob: float
    negative_logprob: float
    margin: float
    margin_delta_from_baseline: float


@dataclass
class LocalDistillMetrics:
    """Metrics returned by a single SDPO distillation step."""

    distill_loss: float | None
    kl_reg: float | None
    mean_is_ratio: float | None
    clip_fraction: float | None


@dataclass
class TinkerDistillMetrics:
    """Metrics returned by the Tinker engine's importance-sampling training step."""

    adv_mean: float
    kl_mean: float
    effective_kl_coef: float
    kl_gain: float
    adv_abs_mean: float
    adv_abs_mean_raw: float
    completion_len: int = 0
    batch_size: int = 0


@dataclass
class CollapseMetrics:
    """Collapse detection metrics."""

    # None in Tinker mode — proxy doesn't expose the full token distribution
    mean_entropy: float | None = None
    mean_logprob: float = 0.0
    entropy_ratio_to_baseline: float | None = None
    self_rouge_l: float = 0.0
    mean_logprob_drift: float = 0.0
    alert: bool = False


@dataclass
class GeneralCapability:
    """General capability preservation metrics."""

    coding_correct: bool = False
    coding_has_docstring: bool = False
    ifeval_pass_rate: float = 0.0
    general_score: float = 0.0


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics for a single step."""

    logprob_margin: LogprobMargin | None = None
    preference_compliance: float | None = None
    general: GeneralCapability | None = None
    collapse: CollapseMetrics | None = None
    rollouts: list[EvalRollout] = field(default_factory=list)


@dataclass
class StepResult:
    """Result from a single feedback + evaluation step."""

    preference: str
    step: int
    timestamp: str
    feedback_given: str
    sdpo_metrics: LocalDistillMetrics | TinkerDistillMetrics | None
    eval: EvalMetrics
    prompt_used: str
    response_text: str | None = None
    timing_s: float = 0.0
    sub_step_count: int = 1  # number of gradient sub-steps taken


@dataclass
class ExperimentResult:
    """Full result for one preference experiment."""

    preference: str
    lora_id: str
    baseline: EvalMetrics
    steps: list[StepResult] = field(default_factory=list)


@dataclass
class ExperimentSummary:
    """Summary entry for a single preference experiment."""

    preference: str
    lora_id: str
    logprob_margin_delta: float | None = None
    final_compliance: float | None = None
    capability_ratio: float | None = None


@dataclass
class GeminiEvalResult:
    """Result from Gemini's evaluation of a chatbot response."""

    satisfied: bool
    feedback: str | None = None


@dataclass
class MetricContext:
    """Bundled arguments passed to each metric's measure() method."""

    vllm_url: str
    vllm_api_key: str
    vllm_model: str
    step: int
    pref: Any  # PreferenceConfig (forward ref to avoid circular import)
    baseline: EvalMetrics
    response_text: str | None = None
    generate: Callable[[str], Awaitable[str]] | None = None
    openclaw_url: str | None = None
    openclaw_api_key: str = "openclaw-local-dev-token"
    proxy_url: str | None = None


def step_result_from_dict(data: dict[str, object]) -> StepResult:
    """Deserialize a StepResult from a parsed JSON dict (e.g. JSONL line)."""
    eval_data = data.get("eval") or {}
    eval_metrics = EvalMetrics()

    lm = eval_data.get("logprob_margin")  # type: ignore[union-attr]
    if lm:
        eval_metrics.logprob_margin = LogprobMargin(**lm)

    pc = eval_data.get("preference_compliance")  # type: ignore[union-attr]
    if pc is not None:
        eval_metrics.preference_compliance = pc

    gen = eval_data.get("general")  # type: ignore[union-attr]
    if gen:
        eval_metrics.general = GeneralCapability(**gen)

    col = eval_data.get("collapse")  # type: ignore[union-attr]
    if col:
        eval_metrics.collapse = CollapseMetrics(**col)

    rollouts = eval_data.get("rollouts")  # type: ignore[union-attr]
    if isinstance(rollouts, list):
        eval_metrics.rollouts = [
            EvalRollout(
                metric=item.get("metric", ""),
                messages=item.get("messages", []),
                metadata=item.get("metadata", {}),
            )
            for item in rollouts
            if isinstance(item, dict)
        ]

    sdpo: LocalDistillMetrics | TinkerDistillMetrics | None = None
    sdpo_data = data.get("sdpo_metrics")
    if sdpo_data:
        if "adv_mean" in sdpo_data:  # type: ignore[operator]
            sdpo = TinkerDistillMetrics(**sdpo_data)  # type: ignore[arg-type]
        else:
            sdpo = LocalDistillMetrics(**sdpo_data)  # type: ignore[arg-type]

    return StepResult(
        preference=data["preference"],  # type: ignore[arg-type]
        step=data["step"],  # type: ignore[arg-type]
        timestamp=data["timestamp"],  # type: ignore[arg-type]
        feedback_given=data["feedback_given"],  # type: ignore[arg-type]
        sdpo_metrics=sdpo,
        eval=eval_metrics,
        prompt_used=data["prompt_used"],  # type: ignore[arg-type]
        response_text=data.get("response_text"),  # type: ignore[arg-type]
        timing_s=data.get("timing_s", 0.0),  # type: ignore[arg-type]
    )
