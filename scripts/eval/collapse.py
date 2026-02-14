"""Collapse detection: token entropy, self-ROUGE-L, logprob drift.

Phase 3. Run at steps 0, 5, 10, 15, 20 for speed.
"""

from __future__ import annotations

import logging
import math

import httpx

from .types import CollapseMetrics

logger = logging.getLogger(__name__)

COLLAPSE_PROBE = "Tell me something interesting about the ocean."


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    # Space-optimized DP
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l_score(text_a: str, text_b: str) -> float:
    """Compute ROUGE-L F1 between two texts (word-level)."""
    words_a = text_a.split()
    words_b = text_b.split()
    if not words_a or not words_b:
        return 0.0

    lcs = _lcs_length(words_a, words_b)
    precision = lcs / len(words_b) if words_b else 0.0
    recall = lcs / len(words_a) if words_a else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


async def measure_token_entropy(
    vllm_url: str,
    vllm_api_key: str,
    model: str,
    prompt: str = COLLAPSE_PROBE,
    timeout_s: float = 60.0,
) -> float:
    """Generate response with top_logprobs=20 and compute mean token entropy."""
    headers = {"Authorization": f"Bearer {vllm_api_key}"} if vllm_api_key else {}

    async with httpx.AsyncClient(base_url=vllm_url, timeout=timeout_s) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 128,
                "logprobs": True,
                "top_logprobs": 20,
            },
            headers=headers,
        )
        resp.raise_for_status()

    choice = resp.json()["choices"][0]
    logprobs_content = choice.get("logprobs", {}).get("content", [])

    if not logprobs_content:
        return 0.0

    entropies = []
    for token_info in logprobs_content:
        top_lps = token_info.get("top_logprobs", [])
        if not top_lps:
            continue
        # Convert logprobs to probabilities and compute entropy
        log_probs = [entry["logprob"] for entry in top_lps]
        probs = [math.exp(lp) for lp in log_probs]
        # Renormalize over the top-k we have
        total = sum(probs)
        if total <= 0:
            continue
        probs = [p / total for p in probs]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        entropies.append(entropy)

    return sum(entropies) / len(entropies) if entropies else 0.0


async def measure_self_rouge_l(
    vllm_url: str,
    vllm_api_key: str,
    model: str,
    prompt: str = COLLAPSE_PROBE,
    n_samples: int = 3,
    timeout_s: float = 60.0,
) -> float:
    """Generate n_samples responses at temperature=0.7 and compute mean pairwise ROUGE-L."""
    headers = {"Authorization": f"Bearer {vllm_api_key}"} if vllm_api_key else {}
    responses = []

    async with httpx.AsyncClient(base_url=vllm_url, timeout=timeout_s) as client:
        for _ in range(n_samples):
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7,
                    "max_tokens": 256,
                },
                headers=headers,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            responses.append(text)

    if len(responses) < 2:
        return 0.0

    # Pairwise ROUGE-L
    scores = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            scores.append(rouge_l_score(responses[i], responses[j]))

    return sum(scores) / len(scores) if scores else 0.0


async def measure_collapse(
    vllm_url: str,
    vllm_api_key: str,
    model: str,
    baseline_entropy: float | None = None,
    baseline_mean_logprob: float | None = None,
) -> CollapseMetrics:
    """Run all collapse detection checks and return metrics."""
    mean_entropy = await measure_token_entropy(vllm_url, vllm_api_key, model)
    self_rouge = await measure_self_rouge_l(vllm_url, vllm_api_key, model)

    # Entropy ratio
    entropy_ratio = 1.0
    if baseline_entropy and baseline_entropy > 0:
        entropy_ratio = mean_entropy / baseline_entropy

    # Logprob drift (simplified: use entropy as proxy since we don't separately track mean logprob)
    drift = 0.0
    if baseline_mean_logprob is not None:
        drift = abs(mean_entropy - baseline_mean_logprob)

    # Alert conditions
    alert = False
    if entropy_ratio < 0.6:
        logger.warning("COLLAPSE ALERT: entropy ratio %.2f < 0.6", entropy_ratio)
        alert = True
    if self_rouge > 0.85:
        logger.warning("COLLAPSE ALERT: self-ROUGE-L %.2f > 0.85", self_rouge)
        alert = True
    if drift > 2.0:
        logger.warning("COLLAPSE ALERT: logprob drift %.2f > 2.0", drift)
        alert = True

    return CollapseMetrics(
        mean_entropy=mean_entropy,
        entropy_ratio_to_baseline=entropy_ratio,
        self_rouge_l=self_rouge,
        mean_logprob_drift=drift,
        alert=alert,
    )
