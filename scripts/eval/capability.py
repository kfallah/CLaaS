"""General capability evaluation: coding task + IFEval-style probes.

Measures whether SDPO training preserves general ability.
"""

from __future__ import annotations

import logging
import re
import subprocess
import textwrap
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from .types import GeneralCapability

logger = logging.getLogger(__name__)

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


def _count_sentences(text: str) -> int:
    """Count sentences in text."""
    sentences = re.split(r"[.!?]+(?:\s|$)", text.strip())
    return len([s for s in sentences if s.strip()])


IFEVAL_PROBES: list[IFEvalProbe] = [
    IFEvalProbe(
        prompt="Write exactly 3 sentences about Python.",
        verify=lambda resp: _count_sentences(resp) == 3,
    ),
    IFEvalProbe(
        prompt="List 5 benefits of exercise. Use numbered list.",
        verify=lambda resp: len(re.findall(r"^\s*\d+[.)]\s+", resp, re.MULTILINE)) >= 5,
    ),
    IFEvalProbe(
        prompt="Explain recursion without using the word 'function'.",
        verify=lambda resp: "function" not in resp.lower(),
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
    code = _extract_code_block(response)
    if not code:
        return CodingResult(correct=False, has_docstring=False)

    has_docstring = '"""' in code or "'''" in code

    # Run in subprocess for isolation and timeout
    test_script = textwrap.dedent(f"""\
        {code}

        assert fibonacci(0) == 0, f"fibonacci(0) = {{fibonacci(0)}}"
        assert fibonacci(1) == 1, f"fibonacci(1) = {{fibonacci(1)}}"
        assert fibonacci(10) == 55, f"fibonacci(10) = {{fibonacci(10)}}"
        print("PASS")
    """)

    try:
        result = subprocess.run(
            ["python3", "-c", test_script],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        correct = result.returncode == 0 and "PASS" in result.stdout
    except (subprocess.TimeoutExpired, OSError):
        correct = False

    return CodingResult(correct=correct, has_docstring=has_docstring)


async def evaluate_general_capability(
    vllm_url: str,
    vllm_api_key: str,
    model: str,
    timeout_s: float = 60.0,
) -> GeneralCapability:
    """Run coding task + IFEval probes and return capability metrics."""
    headers = {"Authorization": f"Bearer {vllm_api_key}"} if vllm_api_key else {}

    async with httpx.AsyncClient(base_url=vllm_url, timeout=timeout_s) as client:
        # Coding task
        coding_resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": CODING_PROMPT}],
                "temperature": 0,
                "max_tokens": 512,
            },
            headers=headers,
        )
        coding_resp.raise_for_status()
        coding_text = coding_resp.json()["choices"][0]["message"]["content"]

    coding_result = verify_coding(coding_text)

    # IFEval probes
    passes = 0
    total = len(IFEVAL_PROBES)

    async with httpx.AsyncClient(base_url=vllm_url, timeout=timeout_s) as client:
        for probe in IFEVAL_PROBES:
            try:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": probe.prompt}],
                        "temperature": 0,
                        "max_tokens": 512,
                    },
                    headers=headers,
                )
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
                if probe.verify(text):
                    passes += 1
            except (httpx.HTTPError, KeyError, ValueError) as e:
                logger.warning("IFEval probe failed: %s", e)

    ifeval_pass_rate = passes / total if total > 0 else 0.0
    general_score = 0.5 * float(coding_result.correct) + 0.5 * ifeval_pass_rate

    return GeneralCapability(
        coding_correct=coding_result.correct,
        coding_has_docstring=coding_result.has_docstring,
        ifeval_pass_rate=ifeval_pass_rate,
        general_score=general_score,
    )
