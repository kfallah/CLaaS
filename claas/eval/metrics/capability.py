"""General capability evaluation: coding task + IFEval-style probes.

Measures whether SDPO training preserves general ability.
"""

from __future__ import annotations

import logging
import re
import signal
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from claas.core.config import DEFAULT_SYSTEM_PROMPT
from claas.core.types import ChatMessage
from claas.eval.types import EvalRollout, GeneralCapability

from .verifiers import _count_sentences, strip_thinking

logger = logging.getLogger(__name__)

# Max tokens for capability probes.  Qwen3 emits ``<think>`` reasoning
# tokens before the visible answer so we need a generous budget.  The
# fallback list is tried in order when the backend rejects a budget as
# exceeding the context window.
GENERAL_MAX_TOKENS = 2048
GENERAL_MAX_TOKENS_FALLBACKS = (1024, 512, 256)

CODING_PROMPT = (
    "Write a Python function called `fibonacci` that takes an integer n "
    "and returns the nth Fibonacci number. Use iterative approach, not recursive. "
    "Include a docstring."
)


@dataclass
class CodingResult:
    """Result from coding task verification."""

    correct: bool
    has_docstring: bool


@dataclass
class IFEvalProbe:
    """A single IFEval-style instruction-following probe."""

    prompt: str
    verify: Callable[[str], bool]



IFEVAL_PROBES: list[IFEvalProbe] = [
    IFEvalProbe(
        prompt="Write exactly 3 sentences about Python.",
        verify=lambda resp: _count_sentences(strip_thinking(resp)) == 3,
    ),
    IFEvalProbe(
        prompt="List 5 benefits of exercise. Use numbered list.",
        verify=lambda resp: len(
            re.findall(r"^\s*\d+[.)]\s+", strip_thinking(resp), re.MULTILINE)
        )
        >= 5,
    ),
    IFEvalProbe(
        prompt="Explain recursion without using the word 'function'.",
        verify=lambda resp: "function" not in strip_thinking(resp).lower(),
    ),
]


def _extract_code_block(response: str) -> str:
    """Extract the first fenced code block from a response."""
    # Try ```python ... ``` first, then ``` ... ```
    match = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: if no code fence, try to find a def statement
    lines = response.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith("def "):
            in_code = True
        if in_code:
            code_lines.append(line)
            if line.strip() and not line.strip().startswith(("#", "def", " ", "\t")):
                if code_lines:
                    break
    return "\n".join(code_lines).strip()


def verify_coding(response: str, timeout_s: float = 5.0) -> CodingResult:
    """Extract code from response, exec it, verify fibonacci correctness."""
    code = _extract_code_block(strip_thinking(response))
    if not code:
        return CodingResult(correct=False, has_docstring=False)

    has_docstring = '"""' in code or "'''" in code

    try:
        def _timeout_handler(_signum: int, _frame: object) -> None:
            raise TimeoutError("coding verification timed out")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        try:
            safe_builtins = {
                "abs": abs,
                "int": int,
                "len": len,
                "max": max,
                "min": min,
                "range": range,
                "sum": sum,
            }
            namespace: dict[str, object] = {"__builtins__": safe_builtins}
            exec(code, namespace, namespace)
            fn = namespace.get("fibonacci")
            if callable(fn):
                correct = (
                    fn(0) == 0 and  # type: ignore[call-arg]
                    fn(1) == 1 and  # type: ignore[call-arg]
                    fn(10) == 55  # type: ignore[call-arg]
                )
            else:
                correct = False
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
    except (TimeoutError, OSError, ValueError, TypeError, NameError, SyntaxError):
        correct = False

    return CodingResult(correct=correct, has_docstring=has_docstring)


async def evaluate_general_capability(
    claas_url: str,
    model: str,
    timeout_s: float = 60.0,
    rollout_log: list[EvalRollout] | None = None,
    openclaw_url: str | None = None,
    openclaw_api_key: str = "openclaw-local-dev-token",
) -> GeneralCapability:
    """Run coding task + IFEval probes and return capability metrics."""
    # Route through OpenClaw when configured (injects full agent context)
    if openclaw_url:
        base_url = openclaw_url
        headers = {"Authorization": f"Bearer {openclaw_api_key}"}
    else:
        base_url = claas_url
        headers = {}

    async def _chat_completion_with_budget(
        client: httpx.AsyncClient,
        prompt: str,
        max_tokens: int = GENERAL_MAX_TOKENS,
    ) -> tuple[str, int]:
        token_budgets = (max_tokens, *GENERAL_MAX_TOKENS_FALLBACKS)
        last_exc: Exception | None = None

        # When going through OpenClaw the gateway injects system prompt/context
        if openclaw_url:
            messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
            req_model = "openclaw"
        else:
            messages = [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            req_model = model

        for budget in token_budgets:
            try:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": req_model,
                        "messages": messages,
                        "temperature": 0,
                        "max_tokens": budget,
                    },
                    headers=headers,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"], budget
            except httpx.HTTPStatusError as e:
                last_exc = e
                # Retry only for context-window budget failures.
                msg = e.response.text if e.response is not None else ""
                if e.response.status_code != 400 or "maximum context length" not in msg:
                    raise
                logger.warning(
                    "Capability request exceeded context budget (max_tokens=%d), retrying",
                    budget,
                )
            except (KeyError, ValueError) as e:
                last_exc = e
                raise

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("capability request failed without a captured exception")

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        # Coding task
        coding_text, coding_budget = await _chat_completion_with_budget(client, CODING_PROMPT)

    coding_result = verify_coding(coding_text)
    if rollout_log is not None:
        coding_msgs: list[ChatMessage] = []
        if not openclaw_url:
            coding_msgs.append(ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT))
        coding_msgs.append(ChatMessage(role="user", content=CODING_PROMPT))
        coding_msgs.append(ChatMessage(role="assistant", content=coding_text))
        rollout_log.append(
            EvalRollout(
                metric="general",
                messages=coding_msgs,
                metadata={
                    "task": "coding",
                    "max_tokens_used": coding_budget,
                    "verdict": {
                        "correct": coding_result.correct,
                        "has_docstring": coding_result.has_docstring,
                    },
                },
            )
        )

    # IFEval probes
    passes = 0
    total = len(IFEVAL_PROBES)

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        for probe in IFEVAL_PROBES:
            try:
                text, budget_used = await _chat_completion_with_budget(client, probe.prompt)
                passed = probe.verify(text)
                if passed:
                    passes += 1
                if rollout_log is not None:
                    probe_msgs: list[ChatMessage] = []
                    if not openclaw_url:
                        probe_msgs.append(ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT))
                    probe_msgs.append(ChatMessage(role="user", content=probe.prompt))
                    probe_msgs.append(ChatMessage(role="assistant", content=text))
                    rollout_log.append(
                        EvalRollout(
                            metric="general",
                            messages=probe_msgs,
                            metadata={
                                "task": "ifeval",
                                "max_tokens_used": budget_used,
                                "passed": passed,
                            },
                        )
                    )
            except (httpx.HTTPError, KeyError, ValueError) as e:
                logger.warning("IFEval probe failed: %s", e)
                if rollout_log is not None:
                    err_msgs: list[ChatMessage] = []
                    if not openclaw_url:
                        err_msgs.append(ChatMessage(role="system", content=DEFAULT_SYSTEM_PROMPT))
                    err_msgs.append(ChatMessage(role="user", content=probe.prompt))
                    rollout_log.append(
                        EvalRollout(
                            metric="general",
                            messages=err_msgs,
                            metadata={"task": "ifeval", "error": str(e), "passed": False},
                        )
                    )

    ifeval_pass_rate = passes / total if total > 0 else 0.0
    general_score = 0.5 * float(coding_result.correct) + 0.5 * ifeval_pass_rate

    return GeneralCapability(
        coding_correct=coding_result.correct,
        coding_has_docstring=coding_result.has_docstring,
        ifeval_pass_rate=ifeval_pass_rate,
        general_score=general_score,
    )
