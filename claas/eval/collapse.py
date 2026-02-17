"""Collapse detection: token entropy, self-ROUGE-L, logprob drift.

Run at configurable steps (default: 0, 5, 10, 15, 19) for speed.

This module implements three complementary algorithms to detect model collapse
during continual distillation. Model collapse occurs when a student model
degenerates during training -- it may start producing repetitive, low-diversity,
or overly confident outputs instead of the nuanced behavior learned from the
teacher. Each metric captures a different symptom of collapse:

1. **Token Entropy** (``measure_token_entropy``)
   - *What it measures*: The Shannon entropy of the predicted token distribution
     at each generation step, averaged over all tokens in a response. We request
     the top-20 logprobs from the model, convert them to probabilities, and
     compute H = -sum(p * log(p)) for each position.
   - *Why we chose it*: A healthy language model maintains a spread of plausible
     next tokens at each step. When a model collapses, it becomes degenerate --
     always predicting the same few tokens with near-certainty, which drives
     entropy toward zero. Tracking mean token entropy gives a direct, per-token
     measure of output diversity.
   - *Threshold*: We compare current entropy to a baseline measurement taken at
     step 0. An entropy ratio below 0.6 (i.e., a >40% drop from baseline)
     triggers a collapse alert. This threshold was chosen to tolerate normal
     training fluctuations while catching the steep entropy drops that
     characterize true collapse.

2. **Self-ROUGE-L** (``measure_self_rouge_l``)
   - *What it measures*: The mean pairwise ROUGE-L F1 score across multiple
     stochastic generations from the same prompt. ROUGE-L uses the longest
     common subsequence (LCS) between two texts to compute precision and recall
     at the word level, then combines them into an F1 score. We generate
     ``n_samples`` (default 3) responses at temperature 0.7 and compute all
     pairwise ROUGE-L scores.
   - *Why we chose it*: A collapsed model produces near-identical outputs even
     when sampling with temperature, because its probability mass is
     concentrated on a single sequence. Self-ROUGE-L directly measures this
     repetitiveness: if different samples are essentially the same text, ROUGE-L
     between them will approach 1.0. Unlike simple string matching, ROUGE-L is
     robust to minor word-order variations and partial overlaps.
   - *Threshold*: A mean pairwise ROUGE-L score above 0.85 triggers an alert.
     Normal stochastic generations from a healthy model typically have ROUGE-L
     in the 0.3--0.6 range (similar topic, different phrasing). Scores above
     0.85 indicate the model is producing nearly identical outputs regardless
     of sampling randomness, a clear sign of mode collapse.

3. **Logprob Drift** (``measure_collapse`` -- drift component)
   - *What it measures*: The absolute shift in mean per-token log-probability
     between the current checkpoint and a baseline. This captures how much the
     model's overall confidence level has changed since the start of training.
   - *Why we chose it*: During collapse, models often become pathologically
     overconfident (logprobs shift toward 0) or underconfident (logprobs become
     very negative). Tracking the drift in mean logprob from a known-good
     baseline detects these confidence divergences even when entropy or ROUGE-L
     have not yet crossed their thresholds. It serves as an early warning for
     distribution shift.
   - *Threshold*: An absolute drift exceeding 2.0 nats triggers an alert. This
     corresponds to a roughly 7x change in mean token probability (e^2 ~= 7.4),
     which is far outside normal training variance and indicates the model's
     output distribution has shifted substantially from its baseline.
"""

from __future__ import annotations

import logging
import math

import httpx

from .types import DEFAULT_SYSTEM_PROMPT, ChatMessage, CollapseMetrics, EvalRollout
from .verifiers import strip_thinking

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
) -> float | None:
    """Generate response with top_logprobs=20 and compute mean token entropy."""
    mean_entropy, _mean_logprob = await measure_entropy_and_mean_logprob(
        vllm_url=vllm_url,
        vllm_api_key=vllm_api_key,
        model=model,
        prompt=prompt,
        timeout_s=timeout_s,
    )
    return mean_entropy


async def measure_entropy_and_mean_logprob(
    vllm_url: str,
    vllm_api_key: str,
    model: str,
    prompt: str = COLLAPSE_PROBE,
    timeout_s: float = 60.0,
    rollout_log: list[EvalRollout] | None = None,
    openclaw_url: str | None = None,
    openclaw_api_key: str = "openclaw-local-dev-token",
    proxy_url: str | None = None,
) -> tuple[float | None, float]:
    """Generate one response and return (mean_entropy, mean_logprob).

    Routing:
    - **OpenClaw + local vLLM** (openclaw_url set, proxy_url None): Generate
      through OpenClaw which forwards to vLLM. vLLM returns logprobs and
      top_logprobs in chat completion responses, so entropy works as before.
    - **OpenClaw + Tinker proxy** (openclaw_url and proxy_url set): Generate
      through OpenClaw which forwards to the proxy. The proxy doesn't return
      top_logprobs, so entropy is None. After generation, score the response
      via proxy /v1/score to get mean_logprob for drift detection.
    - **Direct vLLM** (fallback, no openclaw_url): Legacy direct-to-vLLM path.
    """
    if openclaw_url and proxy_url:
        # Tinker mode: generate through OpenClaw, score via proxy /v1/score
        return await _entropy_and_logprob_tinker(
            openclaw_url=openclaw_url,
            openclaw_api_key=openclaw_api_key,
            proxy_url=proxy_url,
            prompt=prompt,
            timeout_s=timeout_s,
            rollout_log=rollout_log,
        )

    if openclaw_url:
        # Local mode via OpenClaw: route through OpenClaw → vLLM
        base_url = openclaw_url
        headers = {"Authorization": f"Bearer {openclaw_api_key}"}
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        req_model = "openclaw"
    else:
        # Fallback: direct to vLLM
        base_url = vllm_url
        headers = {"Authorization": f"Bearer {vllm_api_key}"} if vllm_api_key else {}
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        req_model = model

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": req_model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 2048,
                "logprobs": True,
                "top_logprobs": 20,
            },
            headers=headers,
        )
        resp.raise_for_status()

    choice = resp.json()["choices"][0]
    message_body = choice.get("message", {})
    response_text = message_body.get("content", "")
    logprobs_content = choice.get("logprobs", {}).get("content", [])

    if not logprobs_content:
        return (0.0, 0.0)

    entropies = []
    selected_logprobs = []
    for token_info in logprobs_content:
        token_logprob = token_info.get("logprob")
        if token_logprob is not None:
            selected_logprobs.append(token_logprob)
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

    mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    mean_logprob = (
        sum(selected_logprobs) / len(selected_logprobs)
        if selected_logprobs
        else 0.0
    )
    if rollout_log is not None:
        rollout_msgs: list[ChatMessage] = []
        if not openclaw_url:
            rollout_msgs.append(ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT))
        rollout_msgs.append(ChatMessage(role="user", content=prompt))
        rollout_msgs.append(ChatMessage(role="assistant", content=response_text))
        rollout_log.append(
            EvalRollout(
                metric="collapse",
                messages=rollout_msgs,
                metadata={
                    "task": "entropy_probe",
                    "mean_entropy": mean_entropy,
                    "mean_logprob": mean_logprob,
                },
            )
        )
    return (mean_entropy, mean_logprob)


async def _entropy_and_logprob_tinker(
    openclaw_url: str,
    openclaw_api_key: str,
    proxy_url: str,
    prompt: str = COLLAPSE_PROBE,
    timeout_s: float = 60.0,
    rollout_log: list[EvalRollout] | None = None,
) -> tuple[None, float]:
    """Tinker mode: generate through OpenClaw, score via proxy /v1/score.

    The proxy doesn't support top_logprobs, so entropy is unavailable (None).
    Mean logprob is obtained by scoring the generated response via /v1/score.
    """
    headers = {"Authorization": f"Bearer {openclaw_api_key}"}

    # Generate through OpenClaw
    async with httpx.AsyncClient(base_url=openclaw_url, timeout=timeout_s) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "openclaw",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 2048,
            },
            headers=headers,
        )
        resp.raise_for_status()

    response_text = resp.json()["choices"][0]["message"]["content"]

    # Score via proxy /v1/score to get logprob sum
    async with httpx.AsyncClient(base_url=proxy_url, timeout=timeout_s) as client:
        score_resp = await client.post(
            "/v1/score",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "completion": response_text,
            },
        )
        score_resp.raise_for_status()

    score_data = score_resp.json()
    logprob_sum = score_data["logprob_sum"]
    completion_tokens = score_data.get("completion_tokens", 1)
    mean_logprob = logprob_sum / max(completion_tokens, 1)

    if rollout_log is not None:
        rollout_log.append(
            EvalRollout(
                metric="collapse",
                messages=[
                    ChatMessage(role="user", content=prompt),
                    ChatMessage(role="assistant", content=response_text),
                ],
                metadata={
                    "task": "entropy_probe",
                    "mean_entropy": None,
                    "mean_logprob": mean_logprob,
                    "mode": "tinker",
                },
            )
        )
    return (None, mean_logprob)


async def measure_self_rouge_l(
    vllm_url: str,
    vllm_api_key: str,
    model: str,
    prompt: str = COLLAPSE_PROBE,
    n_samples: int = 3,
    timeout_s: float = 60.0,
    rollout_log: list[EvalRollout] | None = None,
    openclaw_url: str | None = None,
    openclaw_api_key: str = "openclaw-local-dev-token",
) -> float:
    """Generate n_samples responses at temperature=0.7 and compute mean pairwise ROUGE-L."""
    # Route through OpenClaw when configured (injects full agent context)
    if openclaw_url:
        base_url = openclaw_url
        headers = {"Authorization": f"Bearer {openclaw_api_key}"}
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        req_model = "openclaw"
    else:
        base_url = vllm_url
        headers = {"Authorization": f"Bearer {vllm_api_key}"} if vllm_api_key else {}
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        req_model = model

    responses = []

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        for _ in range(n_samples):
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": req_model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 2048,
                },
                headers=headers,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            responses.append(text)

    if len(responses) < 2:
        return 0.0

    # Pairwise ROUGE-L (compare visible text only, strip thinking)
    clean_responses = [strip_thinking(r) for r in responses]
    scores = []
    for i in range(len(clean_responses)):
        for j in range(i + 1, len(clean_responses)):
            scores.append(rouge_l_score(clean_responses[i], clean_responses[j]))

    if rollout_log is not None:
        base_msgs: list[ChatMessage] = []
        if not openclaw_url:
            base_msgs.append(ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT))
        for idx, response in enumerate(responses):
            sample_msgs = list(base_msgs)
            sample_msgs.append(ChatMessage(role="user", content=prompt))
            sample_msgs.append(ChatMessage(role="assistant", content=response))
            rollout_log.append(
                EvalRollout(
                    metric="collapse",
                    messages=sample_msgs,
                    metadata={
                        "task": "diversity_probe",
                        "sample_index": idx,
                        "temperature": 0.7,
                    },
                )
            )
        rollout_log.append(
            EvalRollout(
                metric="collapse",
                messages=[ChatMessage(role="user", content=prompt)],
                metadata={"task": "diversity_scores", "pairwise_scores": scores},
            )
        )
    return sum(scores) / len(scores) if scores else 0.0


async def measure_collapse(
    vllm_url: str,
    vllm_api_key: str,
    model: str,
    baseline_entropy: float | None = None,
    baseline_mean_logprob: float | None = None,
    rollout_log: list[EvalRollout] | None = None,
    openclaw_url: str | None = None,
    openclaw_api_key: str = "openclaw-local-dev-token",
    proxy_url: str | None = None,
) -> CollapseMetrics:
    """Run all collapse detection checks and return metrics.

    Both entropy/logprob probes and self-ROUGE-L route through OpenClaw
    when configured. In Tinker mode (proxy_url set), entropy is unavailable
    and mean_logprob is obtained via proxy /v1/score.
    """
    mean_entropy, mean_logprob = await measure_entropy_and_mean_logprob(
        vllm_url=vllm_url,
        vllm_api_key=vllm_api_key,
        model=model,
        rollout_log=rollout_log,
        openclaw_url=openclaw_url,
        openclaw_api_key=openclaw_api_key,
        proxy_url=proxy_url,
    )
    self_rouge = await measure_self_rouge_l(
        vllm_url,
        vllm_api_key,
        model,
        rollout_log=rollout_log,
        openclaw_url=openclaw_url,
        openclaw_api_key=openclaw_api_key,
    )

    # Entropy ratio (None when entropy unavailable, e.g. Tinker mode)
    entropy_ratio: float | None = None
    if mean_entropy is not None:
        entropy_ratio = 1.0
        if baseline_entropy and baseline_entropy > 0:
            entropy_ratio = mean_entropy / baseline_entropy

    # Logprob drift from baseline mean token logprob.
    drift = 0.0
    if baseline_mean_logprob is not None:
        drift = abs(mean_logprob - baseline_mean_logprob)

    # Alert conditions
    alert = False
    if entropy_ratio is not None and entropy_ratio < 0.6:
        logger.warning("COLLAPSE ALERT: entropy ratio %.2f < 0.6", entropy_ratio)
        alert = True
    if self_rouge > 0.85:
        logger.warning("COLLAPSE ALERT: self-ROUGE-L %.2f > 0.85", self_rouge)
        alert = True
    if drift > 2.0:
        logger.warning("COLLAPSE ALERT: logprob drift %.2f > 2.0", drift)
        alert = True

    if rollout_log is not None:
        rollout_log.append(
            EvalRollout(
                metric="collapse",
                messages=[],
                metadata={
                    "task": "summary",
                    "entropy_ratio_to_baseline": entropy_ratio,
                    "self_rouge_l": self_rouge,
                    "mean_logprob_drift": drift,
                    "alert": alert,
                },
            )
        )

    return CollapseMetrics(
        mean_entropy=mean_entropy,
        mean_logprob=mean_logprob,
        entropy_ratio_to_baseline=entropy_ratio,
        self_rouge_l=self_rouge,
        mean_logprob_drift=drift,
        alert=alert,
    )
