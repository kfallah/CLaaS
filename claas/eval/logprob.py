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

1. Tokenizes the prompt messages via ``POST /tokenize`` with
   ``{"messages": [...]}`` so vLLM applies the real tokenizer chat
   template server-side. This gives the prompt token count.
2. Tokenizes the full conversation (prompt + assistant response) the
   same way with ``add_generation_prompt: false`` to get the full token
   IDs.
3. Sends the full token IDs to ``POST /v1/completions`` with
   ``max_tokens=1`` and ``prompt_logprobs=1``, which returns per-token
   log probabilities for every token in the input.
4. Discards the logprobs that correspond to the prompt tokens (using the
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

    Uses vLLM's message-based /tokenize endpoint so the real tokenizer chat
    template is applied server-side (no manual ChatML construction).

    Steps:
    1. POST /tokenize {"messages": [...]} to get prompt token count
    2. POST /tokenize {"messages": [..., assistant], "add_generation_prompt": false}
       to get full token IDs
    3. POST /v1/completions with full token IDs, max_tokens=1, prompt_logprobs=1
    4. Extract and sum logprobs for response tokens
    """
    headers = {"Authorization": f"Bearer {vllm_api_key}"} if vllm_api_key else {}

    async with httpx.AsyncClient(base_url=vllm_url, timeout=timeout_s) as client:
        # Step 1: tokenize prompt messages to learn token count
        tok_resp = await client.post(
            "/tokenize",
            json={"model": model, "messages": messages},
            headers=headers,
        )
        tok_resp.raise_for_status()
        prompt_token_count = tok_resp.json()["count"]

        # Step 2: tokenize full conversation (prompt + response) to get token IDs
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

        # Step 3: get logprobs for the full sequence
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

    # Step 4: skip prompt tokens, extract logprob values, sum
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
    system_prompt: str | None = None,
) -> LogprobMargin:
    """Measure the logprob margin between positive and negative examples.

    When ``system_prompt`` is provided it is prepended to the pair's prompt
    messages so that scoring is consistent with what the model sees through
    OpenClaw during generation.
    """
    messages = list(pair.prompt_messages)
    if system_prompt and not any(
        m.get("role") == "system" and m.get("content") == system_prompt
        for m in messages
    ):
        messages.insert(0, ChatMessage(role="system", content=system_prompt))

    if proxy_url:
        positive_lp = await fetch_response_logprob_sum_via_proxy(
            proxy_url, messages, pair.positive_response,
        )
        negative_lp = await fetch_response_logprob_sum_via_proxy(
            proxy_url, messages, pair.negative_response,
        )
    else:
        positive_lp = await fetch_response_logprob_sum(
            vllm_url, vllm_api_key, model, messages, pair.positive_response,
        )
        negative_lp = await fetch_response_logprob_sum(
            vllm_url, vllm_api_key, model, messages, pair.negative_response,
        )

    margin = positive_lp - negative_lp
    delta = margin - baseline_margin if baseline_margin is not None else 0.0

    return LogprobMargin(
        positive_logprob=positive_lp,
        negative_logprob=negative_lp,
        margin=margin,
        margin_delta_from_baseline=delta,
    )
