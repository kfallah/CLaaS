"""Main orchestration loop for the evaluation harness.

Runs feedback steps, measures metrics, writes results to disk.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time

import httpx

from claas.core.types import (
    FeedbackBatchRequest,
    FeedbackItem,
    FeedbackOrchestration,
)

from .metrics import Metric, build_metrics, derive_model_name
from .plotting import generate_plots
from .preferences import PreferenceConfig, get_preference_configs
from .types import (
    EvalConfig,
    EvalMetrics,
    ExperimentResult,
    ExperimentSummary,
    LocalDistillMetrics,
    MetricContext,
    StepResult,
    TinkerDistillMetrics,
    claas_proxy_chat_params,
    openclaw_chat_params,
)

logger = logging.getLogger(__name__)


async def _init_lora(config: EvalConfig, lora_id: str) -> str:
    """Initialize a fresh LoRA adapter via CLaaS API."""
    # LoRA init can exceed two minutes when the remote trainer is cold-starting.
    async with httpx.AsyncClient(base_url=config.claas_url, timeout=300.0) as client:
        resp = await client.post(
            "/v1/lora/init",
            json={"lora_id": lora_id, "base_model": config.base_model},
        )
        resp.raise_for_status()
        return resp.json()["lora_id"]


async def _submit_feedback(
    config: EvalConfig,
    lora_id: str,
    samples: list[FeedbackItem],
) -> LocalDistillMetrics | TinkerDistillMetrics | None:
    """Submit batched feedback via CLaaS API and return SDPO metrics."""
    payload = FeedbackBatchRequest(
        requests=samples,
        orchestration=FeedbackOrchestration(sleep_before=False, wake_after=False),
    )
    async with httpx.AsyncClient(base_url=config.claas_url, timeout=180.0) as client:
        resp = await client.post("/v1/feedback", json=payload.model_dump())
        resp.raise_for_status()
        result = resp.json()

    distill_result = result.get("distill_result")
    if not distill_result:
        return None

    metadata = distill_result["metadata"]

    if config.mode == "tinker" and "adv_mean" in metadata:
        return TinkerDistillMetrics(
            adv_mean=metadata["adv_mean"],
            kl_mean=metadata["kl_mean"],
            effective_kl_coef=metadata["effective_kl_coef"],
            kl_gain=metadata["kl_gain"],
            adv_abs_mean=metadata["adv_abs_mean"],
            adv_abs_mean_raw=metadata["adv_abs_mean_raw"],
            completion_len=metadata["completion_len"],
            batch_size=metadata["batch_size"],
        )

    return LocalDistillMetrics(
        distill_loss=metadata.get("distill_loss"),
        kl_reg=metadata.get("kl_reg"),
        mean_is_ratio=metadata.get("mean_is_ratio"),
        clip_fraction=metadata.get("clip_fraction"),
    )


async def _generate_response(
    config: EvalConfig,
    model: str,
    prompt: str,
    temperature: float = 0,
    max_tokens: int = 2048,
) -> str:
    """Generate a response via OpenClaw or CLaaS proxy.

    All completions route through the CLaaS API proxy so that token IDs
    and logprobs are cached for the subsequent feedback call.
    """
    if config.openclaw_url:
        api_key = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")
        params = openclaw_chat_params(config.openclaw_url, api_key, prompt)
    else:
        params = claas_proxy_chat_params(config.claas_url, model, prompt)

    async with httpx.AsyncClient(base_url=params.base_url, timeout=120.0) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": params.model,
                "messages": params.messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            headers=params.headers,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]




async def _measure_eval_metrics(
    config: EvalConfig,
    pref: PreferenceConfig,
    model_name: str,
    step: int,
    baseline: EvalMetrics,
    enabled_metrics: list[Metric],
    *,
    openclaw_api_key: str = "",
    response_text: str | None = None,
) -> EvalMetrics:
    """Run all enabled metrics and return aggregated results."""
    metrics = EvalMetrics()

    async def generate(prompt: str) -> str:
        return await _generate_response(config, model_name, prompt)

    ctx = MetricContext(
        claas_url=config.claas_url,
        model=model_name,
        step=step,
        pref=pref,
        baseline=baseline,
        response_text=response_text,
        generate=generate,
        openclaw_url=config.openclaw_url,
        openclaw_api_key=openclaw_api_key,
    )

    for metric in enabled_metrics:
        await metric.measure(ctx, metrics)

    return metrics


def _load_completed_steps(output_dir: str, preference: str) -> list[StepResult]:
    """Load completed steps from existing JSONL for resumability."""
    path = os.path.join(output_dir, preference, "steps.jsonl")
    if not os.path.exists(path):
        return []

    steps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            steps.append(StepResult(
                preference=data["preference"],
                step=data["step"],
                timestamp=data["timestamp"],
                feedback_given=data["feedback_given"],
                sdpo_metrics=(
                    TinkerDistillMetrics(**data["sdpo_metrics"])
                    if data.get("sdpo_metrics") and "adv_mean" in data["sdpo_metrics"]
                    else LocalDistillMetrics(**data["sdpo_metrics"])
                    if data.get("sdpo_metrics")
                    else None
                ),
                eval=EvalMetrics(),  # Not needed for resumability check
                prompt_used=data["prompt_used"],
                response_text=data.get("response_text"),
                timing_s=data.get("timing_s", 0.0),
            ))
    return steps


def _append_step_jsonl(output_dir: str, preference: str, step: StepResult) -> None:
    """Append a step result to the JSONL file."""
    pref_dir = os.path.join(output_dir, preference)
    os.makedirs(pref_dir, exist_ok=True)
    path = os.path.join(pref_dir, "steps.jsonl")

    data = dataclasses.asdict(step)
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")


def _write_metadata(output_dir: str, preference: str, config: EvalConfig, lora_id: str) -> None:
    """Write experiment metadata."""
    pref_dir = os.path.join(output_dir, preference)
    os.makedirs(pref_dir, exist_ok=True)
    path = os.path.join(pref_dir, "metadata.json")

    config_dict = dataclasses.asdict(config)

    data = {
        "config": config_dict,
        "lora_id": lora_id,
        "preference": preference,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_baseline(output_dir: str, preference: str, baseline: EvalMetrics) -> None:
    """Write baseline metrics."""
    pref_dir = os.path.join(output_dir, preference)
    os.makedirs(pref_dir, exist_ok=True)
    path = os.path.join(pref_dir, "baseline.json")

    with open(path, "w") as f:
        json.dump(dataclasses.asdict(baseline), f, indent=2)


def _write_summary(output_dir: str, results: list[ExperimentResult]) -> None:
    """Write aggregated summary with raw metric values (no verdicts)."""
    summaries: dict[str, dict[str, object]] = {}

    for result in results:
        entry = ExperimentSummary(
            preference=result.preference,
            lora_id=result.lora_id,
        )

        # Final logprob margin delta
        if result.steps and result.steps[-1].eval.logprob_margin:
            entry.logprob_margin_delta = (
                result.steps[-1].eval.logprob_margin.margin_delta_from_baseline
            )

        # Preference compliance @ final step
        if result.steps and result.steps[-1].eval.preference_compliance is not None:
            entry.final_compliance = result.steps[-1].eval.preference_compliance

        # General capability retention ratio
        if (
            result.baseline.general
            and result.steps
            and result.steps[-1].eval.general
        ):
            baseline_score = result.baseline.general.general_score
            final_score = result.steps[-1].eval.general.general_score
            entry.capability_ratio = (
                final_score / baseline_score if baseline_score > 0 else 1.0
            )

        summaries[result.preference] = dataclasses.asdict(entry)

    path = os.path.join(output_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summaries, f, indent=2)
    logger.info("Summary written to %s", path)


async def run_preference_experiment(
    config: EvalConfig,
    pref: PreferenceConfig,
    enabled_metrics: list[Metric] | None = None,
    needs_generation: bool = False,
) -> ExperimentResult:
    """Run the full experiment loop for a single preference."""
    openclaw_api_key = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")

    lora_id = f"{config.lora_id_prefix}/{pref.name}"

    # Check for resumed steps
    completed_steps = _load_completed_steps(config.output_dir, pref.name)
    completed_step_nums = {s.step for s in completed_steps}
    resume_from = max(completed_step_nums) + 1 if completed_step_nums else 0

    if resume_from > 0:
        logger.info(
            "[%s] Resuming from step %d (%d steps already completed)",
            pref.name, resume_from, len(completed_steps),
        )
        # Load the existing LoRA ID from metadata if available
        meta_path = os.path.join(config.output_dir, pref.name, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            actual_lora_id = meta.get("lora_id", lora_id)
        else:
            actual_lora_id = lora_id
    else:
        # Fresh start: init LoRA
        logger.info("[%s] Initializing fresh LoRA: %s", pref.name, lora_id)
        actual_lora_id = await _init_lora(config, lora_id)
        logger.info("[%s] LoRA created: %s", pref.name, actual_lora_id)

    model_name = derive_model_name(actual_lora_id)

    # Write metadata
    _write_metadata(config.output_dir, pref.name, config, actual_lora_id)

    # Measure baseline
    if enabled_metrics is None:
        enabled_metrics = []
    if resume_from == 0:
        logger.info("[%s] Measuring baseline...", pref.name)
        baseline = await _measure_eval_metrics(
            config, pref, model_name, step=0,
            baseline=EvalMetrics(),  # No baseline for the baseline itself
            enabled_metrics=enabled_metrics,
            openclaw_api_key=openclaw_api_key,
        )
        _write_baseline(config.output_dir, pref.name, baseline)
    else:
        # Load baseline from disk (restore all metric fields)
        baseline_path = os.path.join(config.output_dir, pref.name, "baseline.json")
        if os.path.exists(baseline_path):
            with open(baseline_path) as f:
                baseline_data = json.loads(f.read())
            from .types import CollapseMetrics, GeneralCapability, LogprobMargin
            baseline = EvalMetrics()
            if baseline_data.get("logprob_margin"):
                baseline.logprob_margin = LogprobMargin(**baseline_data["logprob_margin"])
            if baseline_data.get("general"):
                baseline.general = GeneralCapability(**baseline_data["general"])
            if baseline_data.get("collapse"):
                baseline.collapse = CollapseMetrics(**baseline_data["collapse"])
            if baseline_data.get("preference_compliance") is not None:
                baseline.preference_compliance = baseline_data["preference_compliance"]
        else:
            baseline = EvalMetrics()

    logger.info(
        "[%s] Baseline margin: %.3f",
        pref.name,
        baseline.logprob_margin.margin if baseline.logprob_margin else 0.0,
    )

    result = ExperimentResult(
        preference=pref.name,
        lora_id=actual_lora_id,
        baseline=baseline,
        steps=list(completed_steps),
    )

    # Main loop
    training_cfg = config.training

    for step in range(resume_from, config.num_steps):
        step_start = time.perf_counter()

        # Determine feedback string
        feedback_str = " ".join([pref.feedback_string] * config.feedback_repetitions)

        # Collect samples for this step (batch_size >= 1)
        samples: list[FeedbackItem] = []
        response_text: str | None = None

        for i in range(config.batch_size):
            prompt = pref.probe_prompts[
                (step * config.batch_size + i) % len(pref.probe_prompts)
            ]

            # Generate via CLaaS proxy (caches completion for feedback lookup)
            try:
                temperature = 0.7 if config.mode == "tinker" else 0
                content = await _generate_response(
                    config, model_name, prompt, temperature=temperature,
                )
                if response_text is None:
                    response_text = content
                samples.append(FeedbackItem(
                    lora_id=actual_lora_id,
                    prompt=prompt,
                    response=content,
                    feedback=feedback_str,
                    training=training_cfg,
                ))
            except (httpx.HTTPError, KeyError, ValueError) as e:
                logger.warning(
                    "[%s] Step %d sample %d generation failed: %s",
                    pref.name, step, i, e,
                )

        if response_text is None:
            response_text = "I'd be happy to help you with that."

        # Submit feedback — possibly multiple gradient steps on same batch
        sdpo_metrics = None
        sub_steps_completed = 0
        if samples:
            for sub_step in range(config.steps_per_batch):
                try:
                    sdpo_metrics = await _submit_feedback(
                        config, actual_lora_id, samples,
                    )
                    sub_steps_completed += 1
                except (httpx.HTTPError, KeyError) as e:
                    logger.warning(
                        "[%s] Step %d sub-step %d feedback failed: %s",
                        pref.name, step, sub_step, e,
                    )
                    break

            if config.steps_per_batch > 1:
                logger.info(
                    "[%s] Step %d: %d sub-steps completed",
                    pref.name, step, sub_steps_completed,
                )

        # Measure eval
        try:
            eval_metrics = await _measure_eval_metrics(
                config, pref, model_name, step, baseline,
                enabled_metrics=enabled_metrics,
                openclaw_api_key=openclaw_api_key,
                response_text=response_text if needs_generation else None,
            )
        except (httpx.HTTPError, KeyError) as e:
            logger.warning("[%s] Step %d eval failed: %s", pref.name, step, e)
            eval_metrics = EvalMetrics()

        timing_s = time.perf_counter() - step_start

        step_result = StepResult(
            preference=pref.name,
            step=step,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            feedback_given=feedback_str,
            sdpo_metrics=sdpo_metrics,
            eval=eval_metrics,
            prompt_used=pref.probe_prompts[
                (step * config.batch_size) % len(pref.probe_prompts)
            ],
            response_text=response_text if needs_generation else None,
            timing_s=timing_s,
            sub_step_count=sub_steps_completed if sub_steps_completed > 0 else 1,
        )

        result.steps.append(step_result)
        _append_step_jsonl(config.output_dir, pref.name, step_result)

        # Log progress
        margin_str = (
            f"margin={eval_metrics.logprob_margin.margin:.3f} "
            f"(delta={eval_metrics.logprob_margin.margin_delta_from_baseline:+.3f})"
            if eval_metrics.logprob_margin
            else "no margin"
        )
        compliance_str = (
            f"compliance={eval_metrics.preference_compliance:.2f}"
            if eval_metrics.preference_compliance is not None
            else ""
        )
        logger.info(
            "[%s] Step %d/%d (batch=%d): %s %s (%.1fs)",
            pref.name, step, config.num_steps - 1, len(samples),
            margin_str, compliance_str, timing_s,
        )

    return result


async def run_harness(config: EvalConfig) -> None:
    """Run the full evaluation harness."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pref_configs = get_preference_configs()
    selected = [pref_configs[name] for name in config.preferences if name in pref_configs]

    if not selected:
        logger.error("No valid preferences selected: %s", config.preferences)
        return

    # Build metric objects from config
    enabled_metrics = build_metrics(config.metrics, config.collapse_steps)
    needs_generation = any(m.needs_generation for m in enabled_metrics)

    logger.info(
        "Starting eval harness: metrics=%s, preferences=%s, steps=%d",
        [m.name for m in enabled_metrics], [p.name for p in selected], config.num_steps,
    )

    results: list[ExperimentResult] = []
    for pref in selected:
        result = await run_preference_experiment(
            config, pref,
            enabled_metrics=enabled_metrics,
            needs_generation=needs_generation,
        )
        results.append(result)

    # Write summary
    _write_summary(config.output_dir, results)

    # Generate plots
    if config.plots:
        generate_plots(config.output_dir, config.preferences)

    logger.info("Evaluation complete. Results in %s", config.output_dir)
