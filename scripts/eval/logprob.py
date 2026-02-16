"""Logprob margin computation via vLLM /v1/completions API.

**What is "logprob margin"?**

The logprob margin is the difference between the sum of per-token log
probabilities for a preferred (positive) response and a dispreferred
(negative) response, given the same prompt.  Concretely:

    margin = sum(logp(token) for token in positive_response)
           - sum(logp(token) for token in negative_response)

A positive margin indicates the model assigns higher probability to the
preferred response than to the dispreferred one.

**How it works**

For each (prompt, response) pair the pipeline:

1. Tokenizes the ChatML prompt prefix via ``POST /tokenize`` to learn
   where the prompt ends and the response begins.
2. Sends the full sequence (prefix + response) to ``POST /v1/completions``
   with ``max_tokens=1`` and ``prompt_logprobs=1``, which returns the
   per-token log probabilities assigned by the model to every token in
   the input without generating any new tokens.
3. Discards the logprobs that correspond to the prompt tokens (using the
   count from step 1) and sums the logprobs for only the response tokens.

The margin is then ``positive_logprob_sum - negative_logprob_sum``.

**Why this metric?**

Logprob margin directly measures whether SDPO training is shifting the
model's distribution toward preferred behaviours *without* needing to
generate text.  Because the model only scores existing tokens (no
sampling), the measurement is fast and fully deterministic, making it
ideal for automated regression checks across training checkpoints.

**Connection to claas/api.py**

This module mirrors the ``_fetch_rollout_logprobs()`` helper in
``claas/api.py``, which uses the same tokenize-then-completions pattern
during the distillation rollout to collect logprobs from the student
model.  The eval version is factored out here so it can run stand-alone
against any vLLM endpoint without importing the training code.
"""

from __future__ import annotations

import logging
import re

import httpx

from claas.core.types import ChatMessage
from claas.training.teacher_helpers import messages_to_chatml

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
    messages: list[ChatMessage],
    response_text: str,
    timeout_s: float = 60.0,
) -> float:
    """Fetch the total log-probability of response_text given prompt messages.

    Steps:
    1. POST /tokenize to get prompt token count
    2. POST /v1/completions with prompt+response, max_tokens=1, prompt_logprobs=1
    3. Extract and sum logprobs for response tokens
    """
    headers = {"Authorization": f"Bearer {vllm_api_key}"} if vllm_api_key else {}
    chatml_prefix = messages_to_chatml(messages)

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


async def fetch_response_logprob_sum_via_proxy(
    proxy_url: str,
    messages: list[ChatMessage],
    response_text: str,
    timeout_s: float = 60.0,
) -> float:
    """Fetch the total log-probability via the Tinker proxy /v1/score endpoint."""
    async with httpx.AsyncClient(base_url=proxy_url, timeout=timeout_s) as client:
        resp = await client.post(
            "/v1/score",
            json={"messages": messages, "completion": response_text},
        )
        resp.raise_for_status()
        return resp.json()["logprob_sum"]


async def measure_logprob_margin(
    vllm_url: str,
    vllm_api_key: str,
    model: str,
    pair: LogprobPair,
    baseline_margin: float | None = None,
    proxy_url: str | None = None,
) -> LogprobMargin:
    """Measure the logprob margin between positive and negative examples."""
    if proxy_url:
        positive_lp = await fetch_response_logprob_sum_via_proxy(
            proxy_url, pair.prompt_messages, pair.positive_response,
        )
        negative_lp = await fetch_response_logprob_sum_via_proxy(
            proxy_url, pair.prompt_messages, pair.negative_response,
        )
    else:
        positive_lp = await fetch_response_logprob_sum(
            vllm_url, vllm_api_key, model, pair.prompt_messages, pair.positive_response,
        )
        negative_lp = await fetch_response_logprob_sum(
            vllm_url, vllm_api_key, model, pair.prompt_messages, pair.negative_response,
        )

    margin = positive_lp - negative_lp
    delta = margin - baseline_margin if baseline_margin is not None else 0.0

    return LogprobMargin(
        positive_logprob=positive_lp,
        negative_logprob=negative_lp,
        margin=margin,
        margin_delta_from_baseline=delta,
    )
