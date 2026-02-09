"""TeacherService: vLLM-based teacher model for SDPO logprob scoring.

This Modal service hosts a Qwen3-Coder-30B teacher model via vLLM to provide
dense logprob supervision for the SDPO loss. The teacher scores student-
generated tokens and returns top-K log-probabilities at each position.

Key features:
- GPU memory snapshots for sub-second cold starts (~3-5s vs ~45-60s)
- prompt_logprobs=100 for dense teacher signal (vs Fireworks' K=5 limit)
- Stateless service that can be shared across users/LoRAs
"""

from __future__ import annotations

from typing import TypedDict

import modal
import torch
from vllm import LLM, SamplingParams


class TokenLogprobs(TypedDict):
    """Log-probabilities for a single token position."""

    indices: list[int]
    logprobs: list[float]


class HealthCheckResult(TypedDict):
    """Result from health check."""

    status: str
    model: str
    ready: bool

# Modal app (shared with training worker)
app = modal.App("claas-distill")

# Volume for model weights
model_volume = modal.Volume.from_name("claas-models", create_if_missing=True)

# vLLM image with dependencies
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.4.0",
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "huggingface_hub",
    )
    .env({"HF_HOME": "/models/hf_cache"})
)


@app.cls(
    gpu="H100",
    image=vllm_image,
    volumes={"/models": model_volume},
    keep_warm=1,
    container_idle_timeout=600,
    enable_memory_snapshot=True,
    timeout=300,
)
class TeacherService:
    """vLLM-based teacher model service for SDPO logprob scoring."""

    model_id: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    max_model_len: int = 8192
    default_top_k: int = 100

    @modal.enter(snap=True)
    def start_vllm(self):
        """Initialize vLLM engine with Qwen3-Coder-30B.

        The entire state — model weights, KV cache allocations, CUDA graphs,
        compiled kernels — is captured in the GPU memory snapshot.

        Without snapshot: ~45-60s cold start
        With snapshot: ~3-5s cold start
        """
        print(f"Initializing vLLM with {self.model_id}...")

        self.llm = LLM(
            model=self.model_id,
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
            download_dir="/models",
        )

        # Get tokenizer for prompt length calculation
        self.tokenizer = self.llm.get_tokenizer()

        # Warm up: run a dummy prompt to trigger all CUDA graph captures
        # and kernel compilations before the snapshot is taken
        warmup_params = SamplingParams(max_tokens=1, temperature=0)
        _ = self.llm.generate(["Hello, world!"], warmup_params)

        print("vLLM engine initialized and warmed up. Snapshot will capture this state.")

    @modal.method()
    def score_tokens(
        self,
        prompts: list[str],
        completions: list[str],
        top_k: int | None = None,
    ) -> list[list[TokenLogprobs]]:
        """Get teacher logprobs on student-generated tokens.

        Args:
            prompts: List of prompts (teacher-formatted)
            completions: List of student responses to score
            top_k: Number of top logprobs to return per position

        Returns:
            List of completion results, each containing:
                - List of TokenLogprobs with 'indices' and 'logprobs'
        """

        if top_k is None:
            top_k = self.default_top_k

        # Concatenate prompt + completion as a single prompt
        # Use prompt_logprobs to get logprobs at every position
        full_texts = [p + c for p, c in zip(prompts, completions, strict=True)]
        prompt_lengths = [len(self.tokenizer.encode(p)) for p in prompts]

        params = SamplingParams(
            max_tokens=1,  # don't generate new tokens
            temperature=0,
            prompt_logprobs=top_k,  # vLLM supports arbitrary K here
        )

        outputs = self.llm.generate(full_texts, params)

        results = []
        for output, plen in zip(outputs, prompt_lengths, strict=True):
            # Extract logprobs at completion token positions (after the prompt)
            completion_logprobs = []

            if output.prompt_logprobs is None:
                # No logprobs available (shouldn't happen with prompt_logprobs set)
                results.append([])
                continue

            for pos in range(plen, len(output.prompt_logprobs)):
                token_logprobs = output.prompt_logprobs[pos]
                if token_logprobs is None:
                    continue

                # token_logprobs is a dict: {token_id: Logprob(logprob, rank, decoded)}
                top_k_items = sorted(
                    token_logprobs.items(),
                    key=lambda x: x[1].logprob,
                    reverse=True,
                )[:top_k]

                completion_logprobs.append({
                    "indices": [item[0] for item in top_k_items],
                    "logprobs": [item[1].logprob for item in top_k_items],
                })

            results.append(completion_logprobs)

        return results

    @modal.method()
    def health_check(self) -> HealthCheckResult:
        """Check if the service is ready."""
        return HealthCheckResult(
            status="healthy",
            model=self.model_id,
            ready=self.llm is not None,
        )


def format_teacher_prompt(
    user_prompt: str,
    feedback: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Format prompt for the teacher model.

    The teacher evaluates the student's response in context, providing
    logprobs that indicate how likely the teacher would have generated
    each token.

    Args:
        user_prompt: The original user prompt
        feedback: Optional feedback about the response quality
        system_prompt: Optional system prompt

    Returns:
        Formatted prompt string for the teacher
    """
    if system_prompt is None:
        system_prompt = (
            "You are an expert coding assistant. Provide high-quality, "
            "correct, and well-explained code solutions."
        )

    # For scoring, we want the teacher to see the same context the student saw
    # The teacher's logprobs on the response tokens indicate agreement/disagreement
    parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]

    # Include feedback in the prompt if provided
    if feedback:
        parts.append(
            f"<|im_start|>user\n{user_prompt}\n\n"
            f"[Feedback on previous attempt: {feedback}]<|im_end|>"
        )
    else:
        parts.append(f"<|im_start|>user\n{user_prompt}<|im_end|>")

    parts.append("<|im_start|>assistant\n")

    return "".join(parts)


def parse_teacher_result(
    result: list[TokenLogprobs],
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse teacher scoring result into tensors.

    Args:
        result: List of TokenLogprobs from score_tokens
        device: Device to place tensors on

    Returns:
        Tuple of (teacher_logprobs, teacher_indices) tensors
            - teacher_logprobs: (T, K) tensor of log-probabilities
            - teacher_indices: (T, K) tensor of token indices
    """
    if not result:
        raise ValueError("Empty teacher result")

    # Find max K across positions
    max_k = max(len(pos["indices"]) for pos in result)
    T = len(result)

    # Initialize tensors with padding
    teacher_logprobs = torch.full((T, max_k), -100.0, device=device)
    teacher_indices = torch.zeros((T, max_k), dtype=torch.long, device=device)

    for t, pos in enumerate(result):
        k = len(pos["indices"])
        teacher_indices[t, :k] = torch.tensor(pos["indices"], device=device)
        teacher_logprobs[t, :k] = torch.tensor(pos["logprobs"], device=device)

    return teacher_logprobs, teacher_indices


# For local testing
if __name__ == "__main__":
    with app.run():
        teacher = TeacherService()
        result = teacher.health_check.remote()
        print(f"Health check: {result}")

        # Test scoring
        prompts = ["What is 2 + 2?"]
        completions = [" The answer is 4."]
        scores = teacher.score_tokens.remote(prompts, completions, top_k=10)
        print(f"Score result positions: {len(scores[0])}")
