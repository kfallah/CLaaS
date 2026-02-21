"""Modal TeacherService for SDPO token-level logprob scoring."""

from __future__ import annotations

import importlib
import os

import modal
from pydantic import BaseModel


class HealthCheckResult(BaseModel):
    """Health check payload for teacher service."""

    status: str
    model: str
    ready: bool


class TokenTopKResult(BaseModel):
    """Top-k token logprob result for one generated position."""

    indices: list[int]
    logprobs: list[float]


app = modal.App("claas-distill")
model_volume = modal.Volume.from_name("claas-models", create_if_missing=True)
hf_secret_name = os.environ.get("CLAAS_HF_SECRET_NAME")
teacher_secrets = [modal.Secret.from_name(hf_secret_name)] if hf_secret_name else []
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.11.0",
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "huggingface_hub",
    )
    .env(
        {
            "HF_HOME": "/models/hf_cache",
            "VLLM_SERVER_DEV_MODE": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
        }
    )
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
    """vLLM-backed teacher service for SDPO supervision."""

    model_id: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    max_model_len: int = 8192
    top_k: int = 100

    @modal.enter(snap=True)
    def start_vllm(self) -> None:
        """Initialize the vLLM engine and capture a warm snapshot."""
        if os.environ.get("CUDA_VISIBLE_DEVICES", "").lower() == "none":
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        vllm_module = importlib.import_module("vllm")
        llm_cls = getattr(vllm_module, "LLM")
        sampling_params_cls = getattr(vllm_module, "SamplingParams")

        self.llm = llm_cls(
            model=self.model_id,
            dtype="bfloat16",
            tensor_parallel_size=1,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=0.90,
            enable_sleep_mode=True,
            trust_remote_code=True,
            download_dir="/models",
        )
        self.tokenizer = self.llm.get_tokenizer()
        warmup_params = sampling_params_cls(max_tokens=1, temperature=0)
        _ = self.llm.generate(["Hello, world!"], warmup_params)
        self._sampling_params_cls = sampling_params_cls
        self.llm.sleep(level=1)

    @modal.enter(snap=False)
    def wake_up(self) -> None:
        """Wake vLLM after a snapshot restore."""
        self.llm.wake_up()

    @modal.method()
    def score_tokens(
        self,
        prompts: list[str],
        completions: list[str],
        top_k: int,
    ) -> list[list[TokenTopKResult]]:
        """Get teacher top-k logprobs on provided completion tokens.

        Args:
            prompts: Teacher-formatted prompt strings.
            completions: Student completion strings.
            top_k: Number of token candidates to return per position.

        Returns:
            Nested list of token logprob results per completion position.
        """
        full_texts = [prompt + completion for prompt, completion in zip(prompts, completions, strict=True)]
        prompt_lengths = [len(self.tokenizer.encode(prompt)) for prompt in prompts]
        params = self._sampling_params_cls(max_tokens=1, temperature=0, prompt_logprobs=top_k)
        outputs = self.llm.generate(full_texts, params)

        results: list[list[TokenTopKResult]] = []
        for output, prompt_length in zip(outputs, prompt_lengths, strict=True):
            if output.prompt_logprobs is None:
                raise RuntimeError("Teacher service returned empty prompt_logprobs")

            completion_logprobs: list[TokenTopKResult] = []
            for pos in range(prompt_length, len(output.prompt_logprobs)):
                token_logprobs = output.prompt_logprobs[pos]
                if token_logprobs is None:
                    raise RuntimeError("Teacher service returned empty token logprobs")

                top_k_items = sorted(
                    token_logprobs.items(),
                    key=lambda item: item[1].logprob,
                    reverse=True,
                )[:top_k]
                completion_logprobs.append(
                    TokenTopKResult(
                        indices=[item[0] for item in top_k_items],
                        logprobs=[item[1].logprob for item in top_k_items],
                    )
                )
            results.append(completion_logprobs)

        return results

    @modal.method()
    def health_check(self) -> HealthCheckResult:
        """Check whether the teacher service is ready."""
        return HealthCheckResult(status="healthy", model=self.model_id, ready=self.llm is not None)


if __name__ == "__main__":
    with app.run():
        teacher = TeacherService()
        result = teacher.health_check.remote()
        print(f"Health check: {result}")
