#!/usr/bin/env python3
"""Initialize a LoRA adapter for the local stack.

Idempotent — skips creation if the adapter already exists with weights.

Usage:
    CLAAS_STORAGE_BACKEND=local_fs python3 scripts/openclaw-local/init_lora.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

LORA_NAME = os.environ.get("LORA_NAME", "openclaw/assistant")
BASE_MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
LORA_ROOT = os.environ.get(
    "CLAAS_LORA_ROOT",
    str(Path(__file__).resolve().parents[2] / ".local_loras"),
)

# Ensure claas uses local_fs storage at our lora root
os.environ.setdefault("CLAAS_STORAGE_BACKEND", "local_fs")
os.environ["CLAAS_LORA_ROOT"] = LORA_ROOT


def main() -> None:
    alias_key = f"{LORA_NAME}-latest"
    aliases_path = Path(LORA_ROOT) / ".aliases.json"

    # Check if already initialized
    if aliases_path.exists():
        try:
            aliases = json.loads(aliases_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            aliases = {}

        if alias_key in aliases:
            target = aliases[alias_key]
            adapter_dir = Path(LORA_ROOT) / target
            if (adapter_dir / "adapter_config.json").exists() and (
                adapter_dir / "adapter_model.safetensors"
            ).exists():
                print(f"LoRA '{alias_key}' already exists -> {target}, skipping.")
                return

    from claas.training.storage import create_initial_lora

    lora_r = int(os.environ.get("LORA_R", "32"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", "64"))
    full_id = create_initial_lora(
        LORA_NAME, base_model_name=BASE_MODEL, lora_r=lora_r, lora_alpha=lora_alpha,
    )
    print(f"Created initial LoRA: {full_id}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1) from e
