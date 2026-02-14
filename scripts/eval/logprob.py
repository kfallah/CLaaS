"""Logprob margin computation via vLLM /v1/completions API.

Mirrors the _fetch_rollout_logprobs() pattern from claas/api.py.
"""

from __future__ import annotations

import logging
import re

import httpx

from .preferences import LogprobPair
from .types import LogprobMargin

logger = logging.getLogger(__name__)


def derive_vllm_model_name(lora_id: str) -> str:
    """Derive the vLLM adapter model name from a LoRA ID.

    Matches the sanitization logic in claas/api.py _resolve_vllm_model_name.
    """
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", lora_id.strip("/")).strip("-") or "lora"


async def fetch_response_logprob_sum(
    vllm_url: str,
    vllm_api_key: str,
    model: str,
    chatml_prefix: str,
    response_text: str,
    timeout_s: float = 60.0,
) -> float:
    """Fetch the total log-probability of response_text given chatml_prefix.

    Steps:
    1. POST /tokenize to get prompt token count
    2. POST /v1/completions with prompt+response, max_tokens=0, prompt_logprobs=1
    3. Extract and sum logprobs for response tokens
    """
    headers = {"Authorization": f"Bearer {vllm_api_key}"} if vllm_api_key else {}

    async with httpx.AsyncClient(base_url=vllm_url, timeout=timeout_s) as client:
        # Step 1: tokenize the prompt to learn its length
        tok_resp = await client.post(
            "/tokenize",
            json={"model": model, "prompt": chatml_prefix},
            headers=headers,
        )
        tok_resp.raise_for_status()
        prompt_token_count = tok_resp.json()["count"]

        # Step 2: get logprobs for prompt + response
        comp_resp = await client.post(
            "/v1/completions",
            json={
                "model": model,
                "prompt": chatml_prefix + response_text,
                "max_tokens": 1,
                "prompt_logprobs": 1,
            },
            headers=headers,
        )
        comp_resp.raise_for_status()

    raw_logprobs = comp_resp.json()["choices"][0]["prompt_logprobs"]

    # Step 3: skip prompt tokens, extract logprob values, sum
    logprob_sum = 0.0
    for entry in raw_logprobs[prompt_token_count:]:
        if entry is None:
            continue
        top = next(iter(entry.values()))
        logprob_sum += top["logprob"]

    return logprob_sum


async def measure_logprob_margin(
    vllm_url: str,
    vllm_api_key: str,
    model: str,
    pair: LogprobPair,
    baseline_margin: float | None = None,
) -> LogprobMargin:
    """Measure the logprob margin between positive and negative examples."""
    positive_lp = await fetch_response_logprob_sum(
        vllm_url, vllm_api_key, model, pair.prompt_chatml, pair.positive_response,
    )
    negative_lp = await fetch_response_logprob_sum(
        vllm_url, vllm_api_key, model, pair.prompt_chatml, pair.negative_response,
    )

    margin = positive_lp - negative_lp
    delta = margin - baseline_margin if baseline_margin is not None else 0.0

    return LogprobMargin(
        positive_logprob=positive_lp,
        negative_logprob=negative_lp,
        margin=margin,
        margin_delta_from_baseline=delta,
    )
