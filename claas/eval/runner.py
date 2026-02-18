"""Main orchestration loop for the evaluation harness.

Runs feedback steps, measures metrics, writes results to disk.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import os
import time

import httpx

from .gemini import GeminiUser
from .logprob import derive_vllm_model_name
from .metrics import Metric, build_metrics
from .plotting import generate_plots
from .preferences import PreferenceConfig, get_preference_configs
from .types import (
    DEFAULT_SYSTEM_PROMPT,
    ChatMessage,
    EvalMetrics,
    ExperimentResult,
    HarnessConfig,
    MetricContext,
    SDPOMetrics,
    StepResult,
)
from .verifiers import strip_thinking

logger = logging.getLogger(__name__)


def _build_messages(
    prompt: str,
) -> list[ChatMessage]:
    """Build direct-vLLM messages with a default system prompt."""
    return [
        ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT),
        ChatMessage(role="user", content=prompt),
    ]


async def _init_lora(config: HarnessConfig, lora_id: str) -> str:
    """Initialize a fresh LoRA adapter via CLaaS API."""
    async with httpx.AsyncClient(base_url=config.claas_url, timeout=120.0) as client:
        resp = await client.post(
            "/v1/lora/init",
            json={"lora_id": lora_id, "base_model": config.base_model},
        )
        resp.raise_for_status()
        return resp.json()["lora_id"]


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
    samples: list[dict[str, object]],
) -> SDPOMetrics | None:
    """Submit batched feedback via CLaaS API and return SDPO metrics.

    Each sample dict must contain: prompt, response, feedback, rollout_logprobs.
    """
    payload: dict[str, object] = {
        "requests": [
            {
                "lora_id": lora_id,
                "prompt": s["prompt"],
                "response": s["response"],
                "feedback": s["feedback"],
                "rollout_logprobs": s["rollout_logprobs"],
                "training": {"teacher_mode": "self"},
            }
            for s in samples
        ],
        "orchestration": {"sleep_before": False, "wake_after": False},
    }
    async with httpx.AsyncClient(base_url=config.claas_url, timeout=180.0) as client:
        resp = await client.post("/v1/feedback", json=payload)
        resp.raise_for_status()
        result = resp.json()

    distill_result = result.get("distill_result")
    if not distill_result:
        return None

    metadata = distill_result.get("metadata") or {}
    return SDPOMetrics(
        distill_loss=metadata.get("distill_loss"),
        kl_reg=metadata.get("kl_reg"),
        mean_is_ratio=metadata.get("mean_is_ratio"),
        clip_fraction=metadata.get("clip_fraction"),
    )


async def _generate_response(
    config: HarnessConfig,
    model: str,
    prompt: str,
    temperature: float = 0,
    max_tokens: int = 2048,
) -> str:
    """Generate a response via OpenClaw gateway (if configured) or direct vLLM.

    When ``config.openclaw_url`` is set the request is routed through the
    OpenClaw ``/v1/chat/completions`` endpoint which prepends the full agent
    system prompt and context automatically.  Otherwise the request goes
    directly to vLLM with manually-constructed messages.
    """
    if config.openclaw_url:
        headers = {"Authorization": f"Bearer {config.openclaw_api_key}"}
        base_url = config.openclaw_url
        body: dict[str, object] = {
            "model": "openclaw",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
    else:
        headers = (
            {"Authorization": f"Bearer {config.vllm_api_key}"}
            if config.vllm_api_key
            else {}
        )
        base_url = config.vllm_url
        messages = _build_messages(prompt=prompt)
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json=body,
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def _fetch_cached_completion(
    proxy_url: str,
    visible_content: str,
) -> tuple[str, str, list[float]]:
    """Fetch the cached raw completion from the proxy by content hash.

    The proxy caches ``{prompt, response, logprobs}`` keyed by
    ``SHA-256(stripped_content)``.  Since the proxy now strips thinking
    before returning content, ``visible_content`` already matches the
    cache key.  We apply ``strip_thinking`` defensively in case the caller
    passes un-stripped text.

    Returns ``(real_prompt, raw_response, rollout_logprobs)``.
    """
    content_hash = hashlib.sha256(
        strip_thinking(visible_content).encode("utf-8"),
    ).hexdigest()
    async with httpx.AsyncClient(base_url=proxy_url, timeout=60.0) as client:
        resp = await client.get(
            "/v1/completions/raw",
            params={"content_hash": content_hash},
        )
        resp.raise_for_status()

    data = resp.json()
    return data["prompt"], data["response"], data["logprobs"] or []


async def _generate_and_collect(
    config: HarnessConfig,
    model: str,
    prompt: str,
) -> tuple[str, str, str, list[float]]:
    """Generate via OpenClaw, then fetch cached raw completion from the proxy.

    When ``config.openclaw_url`` is set, generation routes through the OpenClaw
    gateway which prepends the full agent system prompt and context.  The proxy
    strips thinking from the returned content and caches the raw completion
    (with the full OpenClaw-templated prompt and generation-time logprobs)
    keyed by ``SHA-256(visible_content)``.

    Returns ``(visible_content, real_prompt, raw_response, rollout_logprobs)``.
    """
    assert config.proxy_url is not None
    content = await _generate_response(config, model, prompt, temperature=0.7)

    real_prompt, raw_response, rollout_lps = await _fetch_cached_completion(
        config.proxy_url, content,
    )
    return content, real_prompt, raw_response, rollout_lps


async def _fetch_rollout_logprobs_vllm(
    config: HarnessConfig,
    model: str,
    prompt: str,
    response_text: str,
) -> list[float]:
    """Score a prompt+response via vLLM and return per-token logprobs.

    Uses vLLM's message-based /tokenize endpoint so the real tokenizer
    chat template is applied server-side (no manual ChatML construction).
    """
    headers = {"Authorization": f"Bearer {config.vllm_api_key}"} if config.vllm_api_key else {}
    messages = _build_messages(prompt=prompt)

    async with httpx.AsyncClient(base_url=config.vllm_url, timeout=60.0) as client:
        # Tokenize prompt messages to learn token count
        tok_resp = await client.post(
            "/tokenize",
            json={"model": model, "messages": messages},
            headers=headers,
        )
        tok_resp.raise_for_status()
        prompt_token_count: int = tok_resp.json()["count"]

        # Tokenize full conversation (prompt + response) to get token IDs
        full_messages = list(messages) + [
            {"role": "assistant", "content": response_text},
        ]
        full_tok_resp = await client.post(
            "/tokenize",
            json={
                "model": model,
                "messages": full_messages,
                "add_generation_prompt": False,
            },
            headers=headers,
        )
        full_tok_resp.raise_for_status()
        full_token_ids = full_tok_resp.json()["tokens"]

        # Get logprobs for the full sequence
        comp_resp = await client.post(
            "/v1/completions",
            json={
                "model": model,
                "prompt": full_token_ids,
                "max_tokens": 1,
                "prompt_logprobs": 1,
            },
            headers=headers,
        )
        comp_resp.raise_for_status()

    raw_logprobs = comp_resp.json()["choices"][0]["prompt_logprobs"]
    logprobs: list[float] = []
    for entry in raw_logprobs[prompt_token_count:]:
        if entry is None:
            logprobs.append(0.0)
            continue
        top = next(iter(entry.values()))
        logprobs.append(top["logprob"])

    return logprobs


async def _measure_eval_metrics(
    config: HarnessConfig,
    pref: PreferenceConfig,
    vllm_model: str,
    step: int,
    baseline: EvalMetrics,
    enabled_metrics: list[Metric],
    response_text: str | None = None,
) -> EvalMetrics:
    """Run all enabled metrics and return aggregated results."""
    metrics = EvalMetrics()

    async def generate(prompt: str) -> str:
        return await _generate_response(config, vllm_model, prompt)

    ctx = MetricContext(
        vllm_url=config.vllm_url,
        vllm_api_key=config.vllm_api_key,
        vllm_model=vllm_model,
        step=step,
        pref=pref,
        baseline=baseline,
        response_text=response_text,
        generate=generate,
        openclaw_url=config.openclaw_url,
        openclaw_api_key=config.openclaw_api_key,
        proxy_url=config.proxy_url,
    )

    for metric in enabled_metrics:
        await metric.measure(ctx, metrics)

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

    config_dict = dataclasses.asdict(config)
    # Convert set to sorted list for JSON serialization
    if config_dict.get("collapse_steps") is not None:
        config_dict["collapse_steps"] = sorted(config_dict["collapse_steps"])

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
        entry: dict[str, object] = {
            "preference": result.preference,
            "lora_id": result.lora_id,
        }

        # Final logprob margin delta
        if result.steps and result.steps[-1].eval.logprob_margin:
            entry["logprob_margin_delta"] = (
                result.steps[-1].eval.logprob_margin.margin_delta_from_baseline
            )

        # Preference compliance @ final step
        if result.steps and result.steps[-1].eval.preference_compliance is not None:
            entry["final_compliance"] = result.steps[-1].eval.preference_compliance

        # General capability retention ratio
        if (
            result.baseline.general
            and result.steps
            and result.steps[-1].eval.general
        ):
            baseline_score = result.baseline.general.general_score
            final_score = result.steps[-1].eval.general.general_score
            entry["capability_ratio"] = (
                final_score / baseline_score if baseline_score > 0 else 1.0
            )

        summaries[result.preference] = entry

    path = os.path.join(output_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summaries, f, indent=2)
    logger.info("Summary written to %s", path)


async def run_preference_experiment(
    config: HarnessConfig,
    pref: PreferenceConfig,
    gemini_user: GeminiUser | None = None,
    enabled_metrics: list[Metric] | None = None,
    needs_generation: bool = False,
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

    # Load LoRA into vLLM (skip for Tinker — proxy auto-refreshes)
    if not config.proxy_url:
        logger.info("[%s] Loading LoRA into vLLM as '%s'", pref.name, vllm_model)
        await _load_lora_into_vllm(config, actual_lora_id, lora_path)

    # Write metadata
    _write_metadata(config.output_dir, pref.name, config, actual_lora_id)

    # Measure baseline
    if enabled_metrics is None:
        enabled_metrics = []
    if resume_from == 0:
        logger.info("[%s] Measuring baseline...", pref.name)
        baseline = await _measure_eval_metrics(
            config, pref, vllm_model, step=0,
            baseline=EvalMetrics(),  # No baseline for the baseline itself
            enabled_metrics=enabled_metrics,
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
    for step in range(resume_from, config.num_steps):
        step_start = time.perf_counter()

        # Determine feedback string
        feedback_str = pref.feedback_string

        # Collect samples for this step (batch_size >= 1)
        samples: list[dict[str, object]] = []
        response_text: str | None = None

        for i in range(config.batch_size):
            prompt = pref.probe_prompts[
                (step * config.batch_size + i) % len(pref.probe_prompts)
            ]

            if config.proxy_url:
                # Tinker mode: generate via OpenClaw, fetch cached completion
                try:
                    content, real_prompt, raw_response, rollout_lps = (
                        await _generate_and_collect(config, vllm_model, prompt)
                    )
                    if response_text is None:
                        response_text = content
                    samples.append({
                        "prompt": real_prompt,
                        "response": raw_response,
                        "feedback": feedback_str,
                        "rollout_logprobs": rollout_lps,
                    })
                except (httpx.HTTPError, KeyError, ValueError) as e:
                    logger.warning(
                        "[%s] Step %d sample %d proxy generation failed: %s",
                        pref.name, step, i, e,
                    )
            else:
                # vLLM mode: generate, score logprobs via vLLM
                try:
                    gen_text = await _generate_response(
                        config, vllm_model, prompt, temperature=0,
                    )
                    if response_text is None:
                        response_text = gen_text
                    rollout_lps = await _fetch_rollout_logprobs_vllm(
                        config, vllm_model, prompt, gen_text,
                    )
                    samples.append({
                        "prompt": prompt,
                        "response": gen_text,
                        "feedback": feedback_str,
                        "rollout_logprobs": rollout_lps,
                    })
                except (httpx.HTTPError, KeyError, ValueError) as e:
                    logger.warning(
                        "[%s] Step %d sample %d vLLM generation failed: %s",
                        pref.name, step, i, e,
                    )

        if response_text is None:
            response_text = "I'd be happy to help you with that."

        # Gemini feedback override (optional)
        if needs_generation and gemini_user:
            try:
                gemini_result = await gemini_user.evaluate_response(response_text, prompt)
                if gemini_result.feedback:
                    feedback_str = gemini_result.feedback
                    # Update feedback in all samples
                    for s in samples:
                        s["feedback"] = feedback_str
            except (httpx.HTTPError, KeyError, ValueError, ImportError) as e:
                logger.warning("[%s] Gemini feedback failed, using default: %s", pref.name, e)

        # Submit feedback via CLaaS API
        sdpo_metrics = None
        if samples:
            try:
                sdpo_metrics = await _submit_feedback(
                    config, actual_lora_id, samples,
                )
            except (httpx.HTTPError, KeyError) as e:
                logger.warning("[%s] Step %d feedback failed: %s — retrying in 5s", pref.name, step, e)
                await asyncio.sleep(5)
                try:
                    sdpo_metrics = await _submit_feedback(
                        config, actual_lora_id, samples,
                    )
                except (httpx.HTTPError, KeyError) as e2:
                    logger.error("[%s] Step %d feedback failed on retry: %s", pref.name, step, e2)

        # Reload LoRA into vLLM (skip for Tinker — proxy auto-refreshes)
        if not config.proxy_url:
            try:
                await _load_lora_into_vllm(config, actual_lora_id, lora_path)
            except (httpx.HTTPError, KeyError) as e:
                logger.warning("[%s] LoRA reload failed: %s", pref.name, e)

        # Measure eval
        try:
            eval_metrics = await _measure_eval_metrics(
                config, pref, vllm_model, step, baseline,
                enabled_metrics=enabled_metrics,
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

    # Build metric objects from config
    enabled_metrics = build_metrics(config.metrics, config.collapse_steps)
    needs_generation = any(m.needs_generation for m in enabled_metrics)

    logger.info(
        "Starting eval harness: metrics=%s, preferences=%s, steps=%d",
        [m.name for m in enabled_metrics], [p.name for p in selected], config.num_steps,
    )

    results: list[ExperimentResult] = []
    for pref in selected:
        gemini = None
        if needs_generation and config.gemini_api_key:
            gemini = GeminiUser(config.gemini_api_key, pref.feedback_string)

        result = await run_preference_experiment(
            config, pref, gemini,
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
