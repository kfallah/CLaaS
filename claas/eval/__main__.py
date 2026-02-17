"""CLI entry point for the evaluation harness.

Both Docker stacks (local GPU + vLLM, and Tinker no-GPU) route generation
through OpenClaw, which injects the real agent system prompt and context.

Tinker stack (no GPU)::

    python -m claas.eval \\
        --openclaw-url http://localhost:18789 \\
        --proxy-url http://localhost:8000 \\
        --claas-url http://localhost:8080 \\
        --base-model Qwen/Qwen3-Coder-30B-A3B-Instruct \\
        --preferences no_emoji --metrics logprob --num-steps 10

Local stack (GPU)::

    python -m claas.eval \\
        --openclaw-url http://localhost:18789 \\
        --vllm-url http://localhost:8000 \\
        --claas-url http://localhost:8080 \\
        --base-model Qwen/Qwen3-8B \\
        --preferences no_emoji --metrics all --num-steps 20

Can also be invoked via the CLI::

    claas eval --preferences no_emoji --metrics logprob --num-steps 10
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime, timezone

from .runner import run_harness
from .types import HarnessConfig


def add_eval_arguments(parser: argparse.ArgumentParser) -> None:
    """Add all eval harness arguments to an argparse parser."""
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
        default=None,
        help=(
            "Output directory for results "
            "(default: auto ./data/evals/<UTC timestamp>)"
        ),
    )
    parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="Google Gemini API key for simulated user (optional, used with generative metrics)",
    )
    parser.add_argument(
        "--metrics",
        default="logprob",
        help=(
            "Comma-separated metric names or preset "
            "(default: logprob). Presets: all=logprob,compliance,general,collapse; quick=logprob"
        ),
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        default=False,
        help="Generate summary plots after evaluation (default: off)",
    )
    parser.add_argument(
        "--collapse-steps",
        default=None,
        help="Comma-separated step numbers for collapse checks (default: 0,5,10,15,19)",
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
    parser.add_argument(
        "--openclaw-url",
        default=None,
        help=(
            "OpenClaw gateway URL (e.g. http://localhost:18789). Primary generation endpoint "
            "for both local and Tinker stacks. All chat completions route through OpenClaw's "
            "/v1/chat/completions endpoint, which injects the full agent system prompt and "
            "context. OpenClaw forwards to the backend (vLLM or Tinker proxy) which applies "
            "the real tokenizer chat template."
        ),
    )
    parser.add_argument(
        "--openclaw-api-key",
        default=None,
        help="OpenClaw gateway API key (default: from OPENCLAW_GATEWAY_TOKEN env)",
    )
    parser.add_argument(
        "--proxy-url",
        default=None,
        help="Tinker proxy URL (enables Tinker mode, e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-8B",
        help="Base model identifier for LoRA init (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Samples per feedback step (default: 1 = progressive, 4 = batched)",
    )


def build_config(args: argparse.Namespace) -> HarnessConfig:
    """Build a HarnessConfig from parsed CLI arguments."""
    # Parse --metrics comma string
    metrics_list = [m.strip() for m in args.metrics.split(",") if m.strip()]

    # Parse --collapse-steps
    collapse_steps: set[int] | None = None
    if args.collapse_steps is not None:
        collapse_steps = {int(s.strip()) for s in args.collapse_steps.split(",") if s.strip()}

    openclaw_url = args.openclaw_url
    openclaw_api_key = args.openclaw_api_key or os.environ.get(
        "OPENCLAW_GATEWAY_TOKEN", "openclaw-local-dev-token"
    )
    output_dir = args.output_dir
    if not output_dir:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
        output_dir = os.path.join("./data/evals", run_id)

    return HarnessConfig(
        claas_url=args.claas_url,
        vllm_url=args.vllm_url,
        vllm_api_key=args.vllm_api_key,
        vllm_model_name=args.vllm_model_name,
        preferences=args.preferences,
        num_steps=args.num_steps,
        output_dir=output_dir,
        gemini_api_key=args.gemini_api_key,
        metrics=metrics_list,
        plots=args.plots,
        collapse_steps=collapse_steps,
        lora_id_prefix=args.lora_id_prefix,
        seed=args.seed,
        openclaw_url=openclaw_url,
        openclaw_api_key=openclaw_api_key,
        proxy_url=args.proxy_url,
        base_model=args.base_model,
        batch_size=args.batch_size,
    )


def parse_args() -> HarnessConfig:
    parser = argparse.ArgumentParser(
        description="SDPO Continual Learning Evaluation Harness",
    )
    add_eval_arguments(parser)
    args = parser.parse_args()
    return build_config(args)


def main() -> None:
    config = parse_args()
    asyncio.run(run_harness(config))


if __name__ == "__main__":
    main()
