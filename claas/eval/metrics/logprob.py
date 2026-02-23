"""Logprob margin: sum(logp(positive)) - sum(logp(negative)) via CLaaS /v1/score.

Scores existing token sequences through the CLaaS API's /v1/score endpoint
(no generation needed), giving a fast, deterministic measure of whether
training is shifting the model toward preferred responses.
"""

from __future__ import annotations

import logging
import re

import httpx

from claas.core.config import DEFAULT_SYSTEM_PROMPT
from claas.core.types import ChatMessage
from claas.eval.types import LogprobMargin, LogprobPair

logger = logging.getLogger(__name__)


def derive_model_name(lora_id: str) -> str:
    """Derive the vLLM adapter model name from a LoRA ID.

    Matches the sanitization logic used for vLLM adapter names.
    """
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", lora_id.strip("/")).strip("-") or "lora"


async def fetch_response_logprob_sum_via_score(
    base_url: str,
    model: str,
    messages: list[ChatMessage],
    response_text: str,
    timeout_s: float = 60.0,
) -> float:
    """Fetch the total log-probability via the CLaaS API /v1/score endpoint.

    Sends structured chat messages so the API can apply the model's chat
    template, matching how the model sees inputs during generation and training.
    """
    msg_dicts = [{"role": m["role"], "content": m.get("content", "")} for m in messages]
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        resp = await client.post(
            "/v1/score",
            json={"model": model, "messages": msg_dicts, "completion": response_text},
        )
        resp.raise_for_status()
        return resp.json()["logprob_sum"]


async def measure_logprob_margin(
    claas_url: str,
    model: str,
    pair: LogprobPair,
    baseline_margin: float | None = None,
    use_default_system_prompt: bool = True,
) -> LogprobMargin:
    """Measure the logprob margin between positive and negative examples.

    Always uses the CLaaS API ``/v1/score`` endpoint.
    """
    messages = list(pair.prompt_messages)
    if use_default_system_prompt and not any(
        m.get("role") == "system" and m.get("content") == DEFAULT_SYSTEM_PROMPT
        for m in messages
    ):
        messages.insert(0, ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT))

    positive_lp = await fetch_response_logprob_sum_via_score(
        claas_url, model, messages, pair.positive_response,
    )
    negative_lp = await fetch_response_logprob_sum_via_score(
        claas_url, model, messages, pair.negative_response,
    )

    margin = positive_lp - negative_lp
    delta = margin - baseline_margin if baseline_margin is not None else 0.0

    return LogprobMargin(
        positive_logprob=positive_lp,
        negative_logprob=negative_lp,
        margin=margin,
        margin_delta_from_baseline=delta,
    )
