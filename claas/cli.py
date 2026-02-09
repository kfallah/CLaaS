"""CLaaS CLI: Command-line interface for CLaaS operations.

Usage:
    claas init-lora --output-uri s3://bucket/loras/user/model/
    claas deploy
    claas health
"""

from __future__ import annotations

import argparse
import json
import sys


def cmd_init_lora(args: argparse.Namespace) -> int:
    """Initialize a new LoRA adapter."""
    from .s3_utils import initialize_lora_from_base

    try:
        uri = initialize_lora_from_base(
            base_model_name=args.base_model,
            output_uri=args.output_uri,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules.split(",") if args.target_modules else None,
        )
        print(f"LoRA initialized at: {uri}")
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
            "lora_uri": args.lora_uri,
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
        "--output-uri",
        required=True,
        help="S3 URI for the LoRA output",
    )
    init_parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-Coder-3B-Instruct",
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
    distill_parser.add_argument("--lora-uri", required=True, help="LoRA S3 URI")
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
