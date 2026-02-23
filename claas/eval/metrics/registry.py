"""Metric classes and registry for the evaluation harness.

Each metric wraps an existing module function and exposes a uniform
``measure(ctx) -> None`` interface that sets results on an EvalMetrics instance.
"""

from __future__ import annotations

import logging
from typing import Protocol

import httpx

from claas.core.config import DEFAULT_SYSTEM_PROMPT
from claas.core.types import ChatMessage
from claas.eval.types import (
    EvalMetrics,
    EvalRollout,
    LogprobMargin,
    MetricContext,
)

from .capability import evaluate_general_capability
from .collapse import measure_collapse
from .logprob import measure_logprob_margin
from .verifiers import run_verifier

logger = logging.getLogger(__name__)

DEFAULT_COLLAPSE_STEPS = [0, 5, 10, 15, 19]


def _prefixed_messages(
    prompt: str,
    response_text: str | None,
    include_default_system_prompt: bool,
) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    if include_default_system_prompt:
        messages.append(ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT))
    messages.append(ChatMessage(role="user", content=prompt))
    if response_text is not None:
        messages.append(ChatMessage(role="assistant", content=response_text))
    return messages


class Metric(Protocol):
    """Protocol for evaluation metrics."""

    name: str
    needs_generation: bool

    async def measure(self, ctx: MetricContext, metrics: EvalMetrics) -> None: ...


class LogprobMetric:
    name = "logprob"
    needs_generation = False

    async def measure(self, ctx: MetricContext, metrics: EvalMetrics) -> None:
        baseline_margin = ctx.baseline.logprob_margin.margin if ctx.baseline.logprob_margin else None
        margins: list[LogprobMargin] = []
        for pair in ctx.pref.logprob_pairs:
            margin = await measure_logprob_margin(
                ctx.claas_url, ctx.model, pair, baseline_margin,
                use_default_system_prompt=ctx.openclaw_url is None,
            )
            margins.append(margin)
        if margins:
            n = len(margins)
            metrics.logprob_margin = LogprobMargin(
                positive_logprob=sum(m.positive_logprob for m in margins) / n,
                negative_logprob=sum(m.negative_logprob for m in margins) / n,
                margin=sum(m.margin for m in margins) / n,
                margin_delta_from_baseline=sum(m.margin_delta_from_baseline for m in margins) / n,
            )


class ComplianceMetric:
    name = "compliance"
    needs_generation = True

    async def measure(self, ctx: MetricContext, metrics: EvalMetrics) -> None:
        if ctx.generate is None:
            return
        scores: list[float] = []
        for probe_prompt in ctx.pref.probe_prompts[:3]:
            try:
                response_text = await ctx.generate(probe_prompt)
                result = run_verifier(ctx.pref.verifier, response_text)
                scores.append(result.score)
                metrics.rollouts.append(
                    EvalRollout(
                        metric="compliance",
                        messages=_prefixed_messages(
                            prompt=probe_prompt,
                            response_text=response_text,
                            include_default_system_prompt=ctx.openclaw_url is None,
                        ),
                        metadata={
                            "verifier": ctx.pref.name,
                            "score": result.score,
                            "passed": result.passed,
                        },
                    )
                )
            except (httpx.HTTPError, KeyError) as e:
                logger.warning("Probe generation failed: %s", e)
                metrics.rollouts.append(
                    EvalRollout(
                        metric="compliance",
                        messages=_prefixed_messages(
                            prompt=probe_prompt,
                            response_text=None,
                            include_default_system_prompt=ctx.openclaw_url is None,
                        ),
                        metadata={"error": str(e), "verifier": ctx.pref.name},
                    )
                )
        if scores:
            metrics.preference_compliance = sum(scores) / len(scores)


class GeneralMetric:
    name = "general"
    needs_generation = True

    async def measure(self, ctx: MetricContext, metrics: EvalMetrics) -> None:
        try:
            rollout_log: list[EvalRollout] = []
            metrics.general = await evaluate_general_capability(
                ctx.claas_url,
                ctx.model,
                rollout_log=rollout_log,
                openclaw_url=ctx.openclaw_url,
                openclaw_api_key=ctx.openclaw_api_key,
            )
            metrics.rollouts.extend(rollout_log)
        except (httpx.HTTPError, KeyError) as e:
            logger.warning("General capability eval failed: %s", e)


class CollapseMetric:
    name = "collapse"
    needs_generation = True

    def __init__(self, check_steps: list[int] | None = None) -> None:
        self.check_steps = check_steps if check_steps is not None else DEFAULT_COLLAPSE_STEPS

    async def measure(self, ctx: MetricContext, metrics: EvalMetrics) -> None:
        if ctx.step not in self.check_steps:
            return
        baseline_entropy = ctx.baseline.collapse.mean_entropy if ctx.baseline.collapse else None
        baseline_mean_logprob = ctx.baseline.collapse.mean_logprob if ctx.baseline.collapse else None
        try:
            rollout_log: list[EvalRollout] = []
            metrics.collapse = await measure_collapse(
                ctx.claas_url, ctx.model,
                baseline_entropy=baseline_entropy,
                baseline_mean_logprob=baseline_mean_logprob,
                rollout_log=rollout_log,
                openclaw_url=ctx.openclaw_url,
                openclaw_api_key=ctx.openclaw_api_key,
            )
            metrics.rollouts.extend(rollout_log)
        except (httpx.HTTPError, KeyError) as e:
            logger.warning("Collapse detection failed: %s", e)


METRIC_PRESETS: dict[str, list[str]] = {
    "all": ["logprob", "compliance", "general", "collapse"],
    "quick": ["logprob"],
}

METRIC_REGISTRY: dict[str, type] = {
    "logprob": LogprobMetric,
    "compliance": ComplianceMetric,
    "general": GeneralMetric,
    "collapse": CollapseMetric,
}


def build_metrics(names: list[str], collapse_steps: list[int] | None = None) -> list[Metric]:
    """Expand presets, deduplicate, and instantiate metric objects."""
    expanded: list[str] = []
    for name in names:
        if name in METRIC_PRESETS:
            expanded.extend(METRIC_PRESETS[name])
        else:
            expanded.append(name)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for name in expanded:
        if name not in seen:
            seen.add(name)
            unique.append(name)

    result: list[Metric] = []
    for name in unique:
        cls = METRIC_REGISTRY.get(name)
        if cls is None:
            logger.warning("Unknown metric: %s (skipping)", name)
            continue
        if cls is CollapseMetric:
            result.append(cls(check_steps=collapse_steps))
        else:
            result.append(cls())
    return result
