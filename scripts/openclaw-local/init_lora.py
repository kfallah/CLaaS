#!/usr/bin/env python3
"""Initialize a LoRA adapter for the local stack.

Idempotent — skips creation if the adapter already exists with weights.

Usage:
    uv run python3 scripts/openclaw-local/init_lora.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_LORA_ROOT = str(Path(__file__).resolve().parents[2] / ".local_loras")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize local CLaaS LoRA storage")
    parser.add_argument("--lora-name", default=os.environ.get("LORA_NAME", "openclaw/assistant"))
    parser.add_argument("--base-model", default=os.environ.get("MODEL", "Qwen/Qwen3-8B"))
    parser.add_argument("--lora-root", default=DEFAULT_LORA_ROOT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    lora_name = args.lora_name
    base_model = args.base_model
    lora_root = str(Path(args.lora_root).expanduser())
    alias_key = f"{lora_name}-latest"
    aliases_path = Path(lora_root) / ".aliases.json"

    # Check if already initialized
    if aliases_path.exists():
        try:
            aliases = json.loads(aliases_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            aliases = {}

        if alias_key in aliases:
            target = aliases[alias_key]
            adapter_dir = Path(lora_root) / target
            if (adapter_dir / "adapter_config.json").exists() and (
                adapter_dir / "adapter_model.safetensors"
            ).exists():
                print(f"LoRA '{alias_key}' already exists -> {target}, skipping.")
                return

    from claas.training.storage import (
        configure_storage_backend,
        configure_storage_root,
        create_initial_lora,
    )

    configure_storage_root(lora_root)
    configure_storage_backend("local_fs")

    lora_r = int(os.environ.get("LORA_R", "32"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", "64"))
    full_id = create_initial_lora(
        lora_name, base_model_name=base_model, lora_r=lora_r, lora_alpha=lora_alpha,
    )
    print(f"Created initial LoRA: {full_id}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1) from e
