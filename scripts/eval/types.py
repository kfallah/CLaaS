"""Data structures for the SDPO continual learning evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field


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
    phase: int = 1
    lora_id_prefix: str = "eval"
    seed: int = 42


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
