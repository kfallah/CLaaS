"""CLaaS CLI: Command-line interface for CLaaS operations.

Usage:
    claas init-lora --lora-id user123/coder-v1
    claas deploy
    claas health
"""

from __future__ import annotations

import argparse
import json
import sys


def cmd_init_lora(args: argparse.Namespace) -> int:
    """Initialize a new LoRA adapter."""
    from .storage import create_initial_lora

    try:
        lora_id = create_initial_lora(
            lora_id=args.lora_id,
            base_model_name=args.base_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules.split(",") if args.target_modules else None,
        )
        print(f"LoRA initialized: {lora_id}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_list_loras(args: argparse.Namespace) -> int:
    """List all LoRA adapters."""
    from .storage import list_loras

    try:
        loras = list_loras(args.prefix or "")
        if loras:
            print("LoRA adapters:")
            for lora in loras:
                print(f"  - {lora}")
        else:
            print("No LoRA adapters found.")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_deploy(args: argparse.Namespace) -> int:
    """Deploy the Modal app."""
    import subprocess

    cmd = ["modal", "deploy", "claas.api"]
    if args.name:
        cmd.extend(["--name", args.name])

    result = subprocess.run(cmd, check=False)
    return result.returncode


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the Modal app locally for development."""
    import subprocess

    cmd = ["modal", "serve", "claas.api"]
    result = subprocess.run(cmd, check=False)
    return result.returncode


def cmd_health(args: argparse.Namespace) -> int:
    """Check health of deployed services."""
    try:
        import modal

        from .teacher import TeacherService
        from .worker import DistillWorker

        with modal.enable_output():
            print("Checking DistillWorker...")
            worker = DistillWorker()
            worker_health = worker.health_check.remote()
            print(f"  Worker: {json.dumps(worker_health, indent=2)}")

            print("\nChecking TeacherService...")
            teacher = TeacherService()
            teacher_health = teacher.health_check.remote()
            print(f"  Teacher: {json.dumps(teacher_health, indent=2)}")

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_distill(args: argparse.Namespace) -> int:
    """Run a single distillation step (for testing)."""
    try:
        import modal

        from .worker import DistillWorker

        request = {
            "lora_id": args.lora_id,
            "prompt": args.prompt,
            "response": args.response,
            "feedback": args.feedback,
            "training": {
                "learning_rate": args.learning_rate,
                "alpha": args.alpha,
            },
        }

        with modal.enable_output():
            worker = DistillWorker()
            result = worker.distill.remote(request)

        print(f"Result: {json.dumps(result, indent=2)}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="claas",
        description="CLaaS: Continual Learning as a Service",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init-lora command
    init_parser = subparsers.add_parser("init-lora", help="Initialize a new LoRA adapter")
    init_parser.add_argument(
        "--lora-id",
        required=True,
        help="LoRA identifier (e.g., 'user123/coder-v1')",
    )
    init_parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-Coder-Next",
        help="Base model name",
    )
    init_parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    init_parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    init_parser.add_argument(
        "--target-modules",
        default=None,
        help="Comma-separated target modules",
    )
    init_parser.set_defaults(func=cmd_init_lora)

    # list-loras command
    list_parser = subparsers.add_parser("list-loras", help="List all LoRA adapters")
    list_parser.add_argument("--prefix", default="", help="Filter by prefix")
    list_parser.set_defaults(func=cmd_list_loras)

    # deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy to Modal")
    deploy_parser.add_argument("--name", help="Deployment name")
    deploy_parser.set_defaults(func=cmd_deploy)

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Run locally with Modal")
    serve_parser.set_defaults(func=cmd_serve)

    # health command
    health_parser = subparsers.add_parser("health", help="Check service health")
    health_parser.set_defaults(func=cmd_health)

    # distill command (for testing)
    distill_parser = subparsers.add_parser("distill", help="Run a distillation step")
    distill_parser.add_argument("--lora-id", required=True, help="LoRA identifier")
    distill_parser.add_argument("--prompt", required=True, help="Prompt text")
    distill_parser.add_argument("--response", required=True, help="Response text")
    distill_parser.add_argument("--feedback", default=None, help="Feedback text")
    distill_parser.add_argument("--learning-rate", type=float, default=1e-4)
    distill_parser.add_argument("--alpha", type=float, default=0.5)
    distill_parser.set_defaults(func=cmd_distill)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
