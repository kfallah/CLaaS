"""Logprob margin: sum(logp(positive)) - sum(logp(negative)) via vLLM.

Scores existing token sequences through vLLM's prompt_logprobs endpoint
(no generation needed), giving a fast, deterministic measure of whether
training is shifting the model toward preferred responses.
"""

from __future__ import annotations

import logging
import re

import httpx

from claas.core.types import ChatMessage

from .preferences import LogprobPair
from .types import DEFAULT_SYSTEM_PROMPT, LogprobMargin

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
    """Fetch the total log-probability via the Tinker proxy /v1/score/chat endpoint.

    Sends structured chat messages so the proxy can apply the model's chat
    template, matching how the model sees inputs during generation and training.
    """
    msg_dicts = [{"role": m["role"], "content": m.get("content", "")} for m in messages]
    async with httpx.AsyncClient(base_url=proxy_url, timeout=timeout_s) as client:
        resp = await client.post(
            "/v1/score/chat",
            json={"messages": msg_dicts, "completion": response_text},
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
    use_default_system_prompt: bool = True,
) -> LogprobMargin:
    """Measure the logprob margin between positive and negative examples.

    In direct-vLLM mode, prepend the default system prompt so scoring
    matches non-OpenClaw generation inputs.
    """
    messages = list(pair.prompt_messages)
    if use_default_system_prompt and not any(
        m.get("role") == "system" and m.get("content") == DEFAULT_SYSTEM_PROMPT
        for m in messages
    ):
        messages.insert(0, ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT))

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
