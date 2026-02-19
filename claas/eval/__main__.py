"""CLI entry point for the evaluation harness.

Both Docker stacks (local GPU + vLLM, and Tinker no-GPU) route generation
through OpenClaw, which injects the real agent system prompt and context.

Tinker stack (no GPU)::

    python -m claas.eval --mode tinker \\
        --openclaw-url http://localhost:18789 \\
        --proxy-url http://localhost:8000 \\
        --claas-url http://localhost:8080 \\
        --base-model Qwen/Qwen3-Coder-30B-A3B-Instruct \\
        --preferences no_emoji --metrics logprob --num-steps 10

Local stack (GPU)::

    python -m claas.eval --mode local \\
        --openclaw-url http://localhost:18789 \\
        --vllm-url http://localhost:8000 \\
        --claas-url http://localhost:8080 \\
        --base-model Qwen/Qwen3-8B \\
        --preferences no_emoji --metrics all --num-steps 20

Can also be invoked via the CLI::

    claas eval --mode local --preferences no_emoji --metrics logprob --num-steps 10

YAML config file::

    python -m claas.eval --config claas/eval/configs/example.yaml --num-steps 5
"""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime, timezone

from .config import build_config_from_yaml
from .runner import run_harness
from .types import HarnessConfig


def add_eval_arguments(parser: argparse.ArgumentParser) -> None:
    """Add all eval harness arguments to an argparse parser."""
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (CLI args override YAML values)",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "tinker"],
        default="local",
        help="Execution mode: 'local' (GPU + vLLM) or 'tinker' (no GPU, Tinker proxy) (default: local)",
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
        default="./data/evals",
        help="Base output directory for results (default: ./data/evals)",
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate summary plots after evaluation (default: on; disable with --no-plots)",
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
        default=4,
        help="Samples per feedback step (default: 4 = batched)",
    )
    parser.add_argument(
        "--steps-per-batch",
        type=int,
        default=1,
        help="Gradient steps per feedback batch; IS ratios recomputed each sub-step (default: 1)",
    )


def _get_explicit_cli_overrides(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> dict:
    """Return a dict of only the CLI args that were explicitly provided (not defaults).

    Uses argparse default-detection: compares each arg value against the parser
    defaults. Only args whose value differs from the default (i.e., explicitly set)
    are included, so YAML values are not clobbered by CLI defaults.
    """
    defaults = vars(parser.parse_args([]))
    overrides: dict = {}
    args_dict = vars(args)

    # Map of CLI dest names → HarnessConfig field names (where they differ)
    dest_to_field = {
        "claas_url": "claas_url",
        "vllm_url": "vllm_url",
        "vllm_api_key": "vllm_api_key",
        "vllm_model_name": "vllm_model_name",
        "num_steps": "num_steps",
        "output_dir": "output_dir",
        "gemini_api_key": "gemini_api_key",
        "lora_id_prefix": "lora_id_prefix",
        "openclaw_url": "openclaw_url",
        "openclaw_api_key": "openclaw_api_key",
        "proxy_url": "proxy_url",
        "base_model": "base_model",
        "batch_size": "batch_size",
        "steps_per_batch": "steps_per_batch",
        "mode": "mode",
        "preferences": "preferences",
        "seed": "seed",
        "plots": "plots",
    }

    for dest, field_name in dest_to_field.items():
        if dest in args_dict and args_dict[dest] != defaults.get(dest):
            overrides[field_name] = args_dict[dest]

    # Special handling for metrics (needs comma-split coercion)
    if "metrics" in args_dict and args_dict["metrics"] != defaults.get("metrics"):
        raw = args_dict["metrics"]
        overrides["metrics"] = [m.strip() for m in raw.split(",") if m.strip()]

    # Special handling for collapse_steps (needs comma-split → set coercion)
    if "collapse_steps" in args_dict and args_dict["collapse_steps"] != defaults.get(
        "collapse_steps"
    ):
        raw = args_dict["collapse_steps"]
        if raw is not None:
            overrides["collapse_steps"] = {int(s.strip()) for s in raw.split(",") if s.strip()}

    return overrides


def build_config(args: argparse.Namespace, parser: argparse.ArgumentParser | None = None) -> HarnessConfig:
    """Build a HarnessConfig from parsed CLI arguments.

    If ``args.config`` is set, loads the YAML as the base config and overlays
    only explicitly-provided CLI args on top.
    """
    if args.config:
        cli_overrides = _get_explicit_cli_overrides(args, parser) if parser else {}
        return build_config_from_yaml(args.config, cli_overrides)

    mode: str = args.mode

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

    # Tinker mode: default proxy_url to vllm_url if not explicitly set
    proxy_url = args.proxy_url
    if mode == "tinker" and not proxy_url:
        proxy_url = args.vllm_url

    # Timestamped subdir under the base output directory
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    output_dir = os.path.join(args.output_dir, run_id)

    return HarnessConfig(
        mode=mode,
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
        proxy_url=proxy_url,
        base_model=args.base_model,
        batch_size=args.batch_size,
        steps_per_batch=args.steps_per_batch,
    )


def parse_args() -> HarnessConfig:
    parser = argparse.ArgumentParser(
        description="SDPO Continual Learning Evaluation Harness",
    )
    add_eval_arguments(parser)
    args = parser.parse_args()
    return build_config(args, parser)


def main() -> None:
    config = parse_args()
    asyncio.run(run_harness(config))


if __name__ == "__main__":
    main()
