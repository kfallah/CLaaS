"""Data structures for the SDPO continual learning evaluation harness."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from claas.core.types import ChatMessage


@dataclass
class EvalRollout:
    """Logged prompt/response transcript with metric-specific metadata."""

    metric: str
    messages: list[ChatMessage]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class HarnessConfig:
    """Top-level configuration for an evaluation run."""

    claas_url: str = "http://localhost:8080"
    vllm_url: str = "http://localhost:8000"
    vllm_api_key: str = "sk-local"
    vllm_model_name: str = "qwen3-8b"
    preferences: list[str] = field(default_factory=lambda: ["no_emoji", "concise", "identity"])
    num_steps: int = 20
    output_dir: str = "./eval_results"
    gemini_api_key: str | None = None
    metrics: list[str] = field(default_factory=lambda: ["logprob"])
    plots: bool = False
    collapse_steps: set[int] | None = None
    lora_id_prefix: str = "eval"
    seed: int = 42
    system_prompt: str | None = None
    prompt_preamble: list[ChatMessage] = field(default_factory=list)
    openclaw_url: str | None = None
    openclaw_api_key: str = "openclaw-local-dev-token"
    proxy_url: str | None = None
    base_model: str = "Qwen/Qwen3-8B"
    batch_size: int = 1


@dataclass
class LogprobMargin:
    """Logprob margin between positive and negative examples."""

    positive_logprob: float
    negative_logprob: float
    margin: float
    margin_delta_from_baseline: float


@dataclass
class SDPOMetrics:
    """Metrics returned by a single SDPO distillation step."""

    distill_loss: float
    kl_reg: float
    mean_is_ratio: float
    clip_fraction: float


@dataclass
class CollapseMetrics:
    """Collapse detection metrics."""

    mean_entropy: float = 0.0
    mean_logprob: float = 0.0
    entropy_ratio_to_baseline: float = 1.0
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
    sdpo_metrics: SDPOMetrics | None
    eval: EvalMetrics
    prompt_used: str
    response_text: str | None = None
    timing_s: float = 0.0


@dataclass
class ExperimentResult:
    """Full result for one preference experiment."""

    preference: str
    lora_id: str
    baseline: EvalMetrics
    steps: list[StepResult] = field(default_factory=list)


@dataclass
class GeminiEvalResult:
    """Result from Gemini's evaluation of a chatbot response."""

    satisfied: bool
    feedback: str | None = None


@dataclass
class CriteriaResult:
    """Pass/marginal/fail verdicts for each success criterion."""

    logprob_margin_increase: str | None = None
    preference_compliance: str | None = None
    capability_retention: str | None = None
    entropy_ratio: str | None = None
    self_rouge_l: str | None = None

    def verdicts(self) -> list[str]:
        """Return all non-None verdict values."""
        return [
            v
            for v in [
                self.logprob_margin_increase,
                self.preference_compliance,
                self.capability_retention,
                self.entropy_ratio,
                self.self_rouge_l,
            ]
            if v is not None
        ]


@dataclass
class PreferenceSummary:
    """Summary result for a single preference experiment."""

    preference: str
    lora_id: str
    criteria: CriteriaResult = field(default_factory=CriteriaResult)
    logprob_margin_delta: float | None = None
    final_compliance: float | None = None
    capability_ratio: float | None = None
    overall: str = "pending"


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
    system_prompt: str | None = None
    prompt_preamble: list[ChatMessage] = field(default_factory=list)
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

    sdpo = None
    sdpo_data = data.get("sdpo_metrics")
    if sdpo_data:
        sdpo = SDPOMetrics(**sdpo_data)  # type: ignore[arg-type]

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
