"""CLI entry point for the evaluation harness.

Usage:
    python -m scripts.eval \
        --claas-url http://localhost:8080 \
        --vllm-url http://localhost:8000 \
        --preferences no_emoji concise identity \
        --num-steps 20 \
        --output-dir ./eval_results \
        --phase 1
"""

from __future__ import annotations

import argparse
import asyncio

from .runner import run_harness
from .types import HarnessConfig


def parse_args() -> HarnessConfig:
    parser = argparse.ArgumentParser(
        description="SDPO Continual Learning Evaluation Harness",
    )
    parser.add_argument(
        "--claas-url",
        default="http://localhost:8080",
        help="CLaaS API base URL (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
        help="vLLM API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--vllm-api-key",
        default="sk-local",
        help="vLLM API key (default: sk-local)",
    )
    parser.add_argument(
        "--vllm-model-name",
        default="qwen3-8b",
        help="vLLM base model name (default: qwen3-8b)",
    )
    parser.add_argument(
        "--preferences",
        nargs="+",
        default=["no_emoji", "concise", "identity"],
        choices=["no_emoji", "concise", "identity"],
        help="Preference types to evaluate (default: all three)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Number of feedback steps per preference (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        default="./eval_results",
        help="Output directory for results (default: ./eval_results)",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="Google Gemini API key for simulated user (Phase 2+, optional)",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Evaluation phase: 1=logprob only, 2=+generative, 3=+collapse+plots (default: 1)",
    )
    parser.add_argument(
        "--lora-id-prefix",
        default="eval",
        help="Prefix for LoRA IDs (default: eval)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    return HarnessConfig(
        claas_url=args.claas_url,
        vllm_url=args.vllm_url,
        vllm_api_key=args.vllm_api_key,
        vllm_model_name=args.vllm_model_name,
        preferences=args.preferences,
        num_steps=args.num_steps,
        output_dir=args.output_dir,
        gemini_api_key=args.gemini_api_key,
        phase=args.phase,
        lora_id_prefix=args.lora_id_prefix,
        seed=args.seed,
    )


def main() -> None:
    config = parse_args()
    asyncio.run(run_harness(config))


if __name__ == "__main__":
    main()
