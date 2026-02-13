"""TeacherService: vLLM-based teacher model for SDPO logprob scoring.

This Modal service hosts a Qwen3-Coder-30B teacher model via vLLM to provide
dense logprob supervision for the SDPO loss. The teacher scores student-
generated tokens and returns top-K log-probabilities at each position.

Key features:
- GPU memory snapshots for sub-second cold starts (~3-5s vs ~45-60s)
- prompt_logprobs up to 100 (matching SDPO reference)
- Stateless service that can be shared across users/LoRAs
"""

from __future__ import annotations

import importlib
import os
import warnings
from typing import TypedDict

import modal
import torch


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
hf_secret_name = os.environ.get("CLAAS_HF_SECRET_NAME", "").strip()
teacher_secrets = [modal.Secret.from_name(hf_secret_name)] if hf_secret_name else []

# vLLM image with dependencies
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.11.0",
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "huggingface_hub",
    )
    .env({
        "HF_HOME": "/models/hf_cache",
        "VLLM_SERVER_DEV_MODE": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    })
)


@app.cls(
    gpu="H100",
    image=vllm_image,
    volumes={"/models": model_volume},
    secrets=teacher_secrets,
    min_containers=1,
    scaledown_window=600,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    startup_timeout=1800,
    timeout=900,
)
class TeacherService:
    """vLLM-based teacher model service for SDPO logprob scoring."""

    model_id: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    max_model_len: int = 8192
    top_k: int = 100

    @modal.enter(snap=True)
    def start_vllm(self):
        """Initialize vLLM engine with Qwen3-Coder-30B.

        The entire state — model weights, KV cache allocations, CUDA graphs,
        compiled kernels — is captured in the GPU memory snapshot.

        Without snapshot: ~45-60s cold start
        With snapshot: ~3-5s cold start
        """
        # vLLM model inspection subprocess expects a numeric CUDA device ID.
        # Modal can expose "none" during early init in some snapshots.
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").lower() == "none":
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        # Ensure runtime-injected tokens are visible to huggingface_hub/vLLM.
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        vllm_module = importlib.import_module("vllm")
        LLM = getattr(vllm_module, "LLM")
        SamplingParams = getattr(vllm_module, "SamplingParams")

        print(f"Initializing vLLM with {self.model_id}...")

        self.llm = LLM(
            model=self.model_id,
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=0.90,
            enable_sleep_mode=True,
            trust_remote_code=True,
            download_dir="/models",
        )

        # Get tokenizer for prompt length calculation
        self.tokenizer = self.llm.get_tokenizer()

        # Warm up: run a dummy prompt to trigger all CUDA graph captures
        # and kernel compilations before the snapshot is taken
        warmup_params = SamplingParams(max_tokens=1, temperature=0)
        _ = self.llm.generate(["Hello, world!"], warmup_params)
        self._sampling_params_cls = SamplingParams

        # Follow Modal's snapshot pattern: warmup then sleep before snapshotting.
        self.llm.sleep(level=1)
        print("vLLM initialized, warmed up, and put to sleep for snapshot capture.")

    @modal.enter(snap=False)
    def wake_up(self):
        """Wake the engine after a snapshot restore."""
        try:
            self.llm.wake_up()
        except Exception as exc:
            warnings.warn(f"Teacher wake_up failed: {exc!r}", stacklevel=1)
            raise

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
            top_k = self.top_k

        # Concatenate prompt + completion as a single prompt
        # Use prompt_logprobs to get logprobs at every position
        full_texts = [p + c for p, c in zip(prompts, completions, strict=True)]
        prompt_lengths = [len(self.tokenizer.encode(p)) for p in prompts]

        params = self._sampling_params_cls(
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


def build_teacher_messages(
    prompt: str,
    feedback: str | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for teacher prompt (veRL-compatible template).

    Template matches veRL's reprompt_template structure:
    {prompt}{feedback}\\n\\nCorrectly solve the original question.

    Args:
        prompt: The original user prompt
        feedback: Optional feedback about the response quality
        system_prompt: Optional system prompt

    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    if system_prompt is None:
        system_prompt = (
            "You are an expert coding assistant. Provide high-quality, "
            "correct, and well-explained code solutions."
        )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Build user content with veRL-style template
    if feedback:
        feedback_section = (
            "\n\nThe following is feedback from your unsuccessful earlier attempt:"
            f"\n\n{feedback}\n"
        )
        user_content = f"{prompt}{feedback_section}\n\nCorrectly solve the original question.\n"
    else:
        user_content = prompt

    messages.append({"role": "user", "content": user_content})
    return messages


def messages_to_chatml(messages: list[dict[str, str]]) -> str:
    """Convert a list of chat messages to ChatML string format.

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        ChatML-formatted string ending with assistant generation prompt
    """
    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def format_teacher_prompt(
    user_prompt: str,
    feedback: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Format prompt for the teacher model as a ChatML string.

    Thin wrapper around build_teacher_messages() for the remote teacher path
    (api.py) which sends raw prompt strings to vLLM.

    Args:
        user_prompt: The original user prompt
        feedback: Optional feedback about the response quality
        system_prompt: Optional system prompt

    Returns:
        Formatted ChatML prompt string for the teacher
    """
    messages = build_teacher_messages(user_prompt, feedback, system_prompt)
    return messages_to_chatml(messages)


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
    if max_k == 0:
        raise ValueError("All teacher positions have empty top-K results")

    T = len(result)

    # Initialize tensors with padding
    teacher_logprobs = torch.full((T, max_k), -100.0, device=device)
    teacher_indices = torch.zeros((T, max_k), dtype=torch.long, device=device)

    for t, pos in enumerate(result):
        k = len(pos["indices"])
        if k > 0:
            teacher_indices[t, :k] = torch.tensor(pos["indices"], dtype=torch.long, device=device)
            teacher_logprobs[t, :k] = torch.tensor(pos["logprobs"], dtype=torch.float, device=device)

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
