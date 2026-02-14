"""Main orchestration loop for the evaluation harness.

Runs feedback steps, measures metrics, writes results to disk.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import time

import httpx

from .capability import evaluate_general_capability
from .collapse import measure_collapse
from .gemini import GeminiUser
from .logprob import derive_vllm_model_name, measure_logprob_margin
from .plotting import generate_plots
from .preferences import PreferenceConfig, get_preference_configs
from .types import (
    EvalMetrics,
    ExperimentResult,
    HarnessConfig,
    SDPOMetrics,
    StepResult,
)
from .verifiers import run_verifier

logger = logging.getLogger(__name__)

# Steps at which to run collapse checks (Phase 3).
COLLAPSE_CHECK_STEPS = {0, 5, 10, 15, 19}


async def _api_post(
    url: str,
    path: str,
    json_body: dict | None = None,
    timeout_s: float = 120.0,
) -> dict:
    """POST to an API endpoint and return the JSON response."""
    async with httpx.AsyncClient(base_url=url, timeout=timeout_s) as client:
        resp = await client.post(path, json=json_body)
        resp.raise_for_status()
        return resp.json()


async def _init_lora(config: HarnessConfig, lora_id: str) -> str:
    """Initialize a fresh LoRA adapter via CLaaS API."""
    result = await _api_post(
        config.claas_url,
        "/v1/lora/init",
        json_body={"lora_id": lora_id, "base_model": "Qwen/Qwen3-8B"},
    )
    return result["lora_id"]


async def _load_lora_into_vllm(
    config: HarnessConfig,
    lora_id: str,
    lora_path: str,
) -> None:
    """Load a LoRA adapter into vLLM (unload first if present)."""
    vllm_name = derive_vllm_model_name(lora_id)
    headers = {"Authorization": f"Bearer {config.vllm_api_key}"} if config.vllm_api_key else {}

    async with httpx.AsyncClient(base_url=config.vllm_url, timeout=30.0) as client:
        # Unload (ignore 404)
        try:
            resp = await client.post(
                "/v1/unload_lora_adapter",
                json={"lora_name": vllm_name},
                headers=headers,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise

        # Load
        resp = await client.post(
            "/v1/load_lora_adapter",
            json={"lora_name": vllm_name, "lora_path": lora_path},
            headers=headers,
        )
        resp.raise_for_status()


async def _submit_feedback(
    config: HarnessConfig,
    lora_id: str,
    prompt: str,
    response: str,
    feedback: str,
) -> SDPOMetrics | None:
    """Submit feedback via CLaaS API and return SDPO metrics."""
    result = await _api_post(
        config.claas_url,
        "/v1/feedback",
        json_body={
            "lora_id": lora_id,
            "prompt": prompt,
            "response": response,
            "feedback": feedback,
        },
        timeout_s=180.0,
    )

    distill_result = result.get("distill_result")
    if not distill_result:
        return None

    metadata = distill_result.get("metadata", {})
    return SDPOMetrics(
        distill_loss=metadata.get("distill_loss", 0.0),
        kl_reg=metadata.get("kl_reg", 0.0),
        mean_is_ratio=metadata.get("mean_is_ratio", 0.0),
        clip_fraction=metadata.get("clip_fraction", 0.0),
    )


async def _generate_response(
    config: HarnessConfig,
    model: str,
    prompt: str,
    temperature: float = 0,
    max_tokens: int = 256,
) -> str:
    """Generate a response via vLLM chat completions."""
    headers = {"Authorization": f"Bearer {config.vllm_api_key}"} if config.vllm_api_key else {}
    async with httpx.AsyncClient(base_url=config.vllm_url, timeout=60.0) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def _measure_eval(
    config: HarnessConfig,
    pref: PreferenceConfig,
    vllm_model: str,
    step: int,
    baseline: EvalMetrics,
    gemini_user: GeminiUser | None = None,
) -> EvalMetrics:
    """Run the appropriate evaluation metrics for the current phase."""
    metrics = EvalMetrics()

    # Phase 1+: logprob margins
    baseline_margin = baseline.logprob_margin.margin if baseline.logprob_margin else None
    for pair in pref.logprob_pairs:
        margin = await measure_logprob_margin(
            config.vllm_url, config.vllm_api_key, vllm_model, pair, baseline_margin,
        )
        metrics.logprob_margin = margin

    if config.phase >= 2:
        # Phase 2+: generative preference compliance
        scores = []
        for probe_prompt in pref.probe_prompts[:3]:
            try:
                response_text = await _generate_response(config, vllm_model, probe_prompt)
                score = run_verifier(pref.verifier_name, response_text)
                scores.append(score)
            except (httpx.HTTPError, KeyError) as e:
                logger.warning("Probe generation failed: %s", e)
        if scores:
            metrics.preference_compliance = sum(scores) / len(scores)

        # Phase 2+: general capability
        try:
            metrics.general = await evaluate_general_capability(
                config.vllm_url, config.vllm_api_key, vllm_model,
            )
        except (httpx.HTTPError, KeyError) as e:
            logger.warning("General capability eval failed: %s", e)

    if config.phase >= 3 and step in COLLAPSE_CHECK_STEPS:
        # Phase 3: collapse detection
        baseline_entropy = (
            baseline.collapse.mean_entropy if baseline.collapse else None
        )
        try:
            metrics.collapse = await measure_collapse(
                config.vllm_url, config.vllm_api_key, vllm_model,
                baseline_entropy=baseline_entropy,
            )
        except (httpx.HTTPError, KeyError) as e:
            logger.warning("Collapse detection failed: %s", e)

    return metrics


def _lora_path_on_disk(lora_id: str) -> str:
    """Derive the on-disk LoRA path from the LoRA ID."""
    lora_root = os.environ.get("CLAAS_LORA_ROOT", "/loras")
    return os.path.join(lora_root, lora_id.strip("/"))


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
                    SDPOMetrics(**data["sdpo_metrics"])
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


def _write_metadata(output_dir: str, preference: str, config: HarnessConfig, lora_id: str) -> None:
    """Write experiment metadata."""
    pref_dir = os.path.join(output_dir, preference)
    os.makedirs(pref_dir, exist_ok=True)
    path = os.path.join(pref_dir, "metadata.json")

    data = {
        "config": dataclasses.asdict(config),
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
    """Write aggregated summary with pass/fail per success criteria."""
    summary = {}
    for result in results:
        pref = result.preference
        entry: dict = {"preference": pref, "lora_id": result.lora_id, "criteria": {}}

        # Final logprob margin increase
        if result.steps and result.steps[-1].eval.logprob_margin:
            delta = result.steps[-1].eval.logprob_margin.margin_delta_from_baseline
            if delta > 2.0:
                entry["criteria"]["logprob_margin_increase"] = "pass"
            elif delta >= 0.5:
                entry["criteria"]["logprob_margin_increase"] = "marginal"
            else:
                entry["criteria"]["logprob_margin_increase"] = "fail"
            entry["logprob_margin_delta"] = delta

        # Preference compliance @ step 20
        if result.steps and result.steps[-1].eval.preference_compliance is not None:
            compliance = result.steps[-1].eval.preference_compliance
            if compliance >= 0.8:
                entry["criteria"]["preference_compliance"] = "pass"
            elif compliance >= 0.5:
                entry["criteria"]["preference_compliance"] = "marginal"
            else:
                entry["criteria"]["preference_compliance"] = "fail"
            entry["final_compliance"] = compliance

        # General capability retention
        if (
            result.baseline.general
            and result.steps
            and result.steps[-1].eval.general
        ):
            baseline_score = result.baseline.general.general_score
            final_score = result.steps[-1].eval.general.general_score
            if baseline_score > 0:
                ratio = final_score / baseline_score
            else:
                ratio = 1.0
            if ratio > 0.9:
                entry["criteria"]["capability_retention"] = "pass"
            elif ratio >= 0.7:
                entry["criteria"]["capability_retention"] = "marginal"
            else:
                entry["criteria"]["capability_retention"] = "fail"
            entry["capability_ratio"] = ratio

        # Collapse
        collapse_entries = [
            s for s in result.steps if s.eval.collapse
        ]
        if collapse_entries:
            last_collapse = collapse_entries[-1].eval.collapse
            if last_collapse is not None:
                entry["criteria"]["entropy_ratio"] = (
                    "pass" if last_collapse.entropy_ratio_to_baseline > 0.6
                    else "marginal" if last_collapse.entropy_ratio_to_baseline > 0.4
                    else "fail"
                )
                entry["criteria"]["self_rouge_l"] = (
                    "pass" if last_collapse.self_rouge_l < 0.85
                    else "marginal" if last_collapse.self_rouge_l < 0.95
                    else "fail"
                )

        # Overall pass
        criteria_values = list(entry["criteria"].values())
        entry["overall"] = (
            "pass" if criteria_values and all(v == "pass" for v in criteria_values)
            else "fail" if any(v == "fail" for v in criteria_values)
            else "marginal"
        )

        summary[pref] = entry

    path = os.path.join(output_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary written to %s", path)


async def run_preference_experiment(
    config: HarnessConfig,
    pref: PreferenceConfig,
    gemini_user: GeminiUser | None = None,
) -> ExperimentResult:
    """Run the full experiment loop for a single preference."""
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

    vllm_model = derive_vllm_model_name(actual_lora_id)
    lora_path = _lora_path_on_disk(actual_lora_id)

    # Load LoRA into vLLM
    logger.info("[%s] Loading LoRA into vLLM as '%s'", pref.name, vllm_model)
    await _load_lora_into_vllm(config, actual_lora_id, lora_path)

    # Write metadata
    _write_metadata(config.output_dir, pref.name, config, actual_lora_id)

    # Measure baseline
    if resume_from == 0:
        logger.info("[%s] Measuring baseline...", pref.name)
        baseline = await _measure_eval(
            config, pref, vllm_model, step=0,
            baseline=EvalMetrics(),  # No baseline for the baseline itself
        )
        _write_baseline(config.output_dir, pref.name, baseline)
    else:
        # Load baseline from disk
        baseline_path = os.path.join(config.output_dir, pref.name, "baseline.json")
        if os.path.exists(baseline_path):
            with open(baseline_path) as f:
                baseline_data = json.loads(f.read())
            from .types import LogprobMargin
            baseline = EvalMetrics()
            if baseline_data.get("logprob_margin"):
                baseline.logprob_margin = LogprobMargin(**baseline_data["logprob_margin"])
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
    for step in range(resume_from, config.num_steps):
        step_start = time.perf_counter()

        prompt = pref.probe_prompts[step % len(pref.probe_prompts)]
        # Phase 1: fixed response; Phase 2+: generate from model
        response_text = "I'd be happy to help you with that."

        # Determine feedback string
        feedback_str = pref.feedback_string
        if config.phase >= 2 and gemini_user:
            try:
                gemini_result = await gemini_user.evaluate_response(response_text, prompt)
                if gemini_result.get("feedback"):
                    feedback_str = gemini_result["feedback"]
            except Exception as e:
                logger.warning("[%s] Gemini feedback failed, using default: %s", pref.name, e)

        # Submit feedback via CLaaS API.
        # The feedback endpoint handles vLLM sleep/wake orchestration internally.
        sdpo_metrics = None
        try:
            sdpo_metrics = await _submit_feedback(
                config, actual_lora_id, prompt, response_text, feedback_str,
            )
        except (httpx.HTTPError, KeyError) as e:
            logger.warning("[%s] Step %d feedback failed: %s — retrying in 5s", pref.name, step, e)
            await asyncio.sleep(5)
            try:
                sdpo_metrics = await _submit_feedback(
                    config, actual_lora_id, prompt, response_text, feedback_str,
                )
            except (httpx.HTTPError, KeyError) as e2:
                logger.error("[%s] Step %d feedback failed on retry: %s", pref.name, step, e2)

        # Reload LoRA into vLLM (picks up any on-disk changes from distillation)
        try:
            await _load_lora_into_vllm(config, actual_lora_id, lora_path)
        except (httpx.HTTPError, KeyError) as e:
            logger.warning("[%s] LoRA reload failed: %s", pref.name, e)

        # Measure eval
        try:
            eval_metrics = await _measure_eval(
                config, pref, vllm_model, step, baseline, gemini_user,
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
            prompt_used=prompt,
            response_text=response_text if config.phase >= 2 else None,
            timing_s=timing_s,
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
            "[%s] Step %d/%d: %s %s (%.1fs)",
            pref.name, step, config.num_steps - 1,
            margin_str, compliance_str, timing_s,
        )

    return result


async def run_harness(config: HarnessConfig) -> None:
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

    logger.info(
        "Starting eval harness: phase=%d, preferences=%s, steps=%d",
        config.phase, [p.name for p in selected], config.num_steps,
    )

    results: list[ExperimentResult] = []
    for pref in selected:
        gemini = None
        if config.phase >= 2 and config.gemini_api_key:
            gemini = GeminiUser(config.gemini_api_key, pref.feedback_string)

        result = await run_preference_experiment(config, pref, gemini)
        results.append(result)

    # Write summary
    _write_summary(config.output_dir, results)

    # Generate plots (Phase 3+)
    if config.phase >= 3:
        generate_plots(config.output_dir, config.preferences)

    logger.info("Evaluation complete. Results in %s", config.output_dir)
