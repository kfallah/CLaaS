#!/usr/bin/env python3
"""One-shot init container: create initial LoRA + write OpenClaw config.

Idempotent — skips LoRA creation if the alias already exists.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
LORA_NAME = os.environ.get("LORA_NAME", "openclaw/assistant")
BASE_MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
LORA_ROOT = os.environ.get("CLAAS_LORA_ROOT", "/loras")
OPENCLAW_HOME = Path(os.environ.get("OPENCLAW_HOME", "/openclaw-config"))
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://vllm:8000/v1")
API_KEY = os.environ.get("API_KEY", "sk-local")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# Ensure claas picks up local_fs mode and our lora root
os.environ["CLAAS_STORAGE_BACKEND"] = "local_fs"
os.environ["CLAAS_LORA_ROOT"] = LORA_ROOT


def _aliases_path() -> Path:
    return Path(LORA_ROOT) / ".aliases.json"


def _read_aliases() -> dict[str, str]:
    p = _aliases_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Step 1: Create initial LoRA (idempotent)
# ---------------------------------------------------------------------------
def create_lora() -> None:
    alias_key = f"{LORA_NAME}-latest"
    aliases = _read_aliases()

    if alias_key in aliases:
        target = aliases[alias_key]
        adapter_cfg = Path(LORA_ROOT) / target / "adapter_config.json"
        if adapter_cfg.exists():
            print(f"LoRA alias '{alias_key}' already exists → {target}, skipping creation.")
            return

    from claas.storage import create_initial_lora

    full_id = create_initial_lora(LORA_NAME, base_model_name=BASE_MODEL)
    print(f"Created initial LoRA: {full_id}")


# ---------------------------------------------------------------------------
# Step 2: Write OpenClaw config
# ---------------------------------------------------------------------------
def _model_entry(model_id: str) -> dict:
    return {
        "id": model_id,
        "name": model_id,
        "api": "openai-completions",
        "reasoning": False,
        "input": ["text"],
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
        "contextWindow": 32768,
        "maxTokens": 8192,
    }


def _normalize_lora_alias(name: str) -> str:
    """Slash-separated LoRA alias → vLLM-safe name (matches start_vllm_qwen3_8b.sh)."""
    import re

    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-")
    return normalized or "lora"


def write_openclaw_config() -> None:
    lora_alias = f"{LORA_NAME}-latest"
    vllm_model_name = _normalize_lora_alias(lora_alias)

    model_ids = ["qwen3-8b", vllm_model_name]
    primary_model = vllm_model_name

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    cfg = {
        "meta": {
            "lastTouchedAt": now,
        },
        "models": {
            "mode": "merge",
            "providers": {
                "local": {
                    "baseUrl": VLLM_BASE_URL,
                    "apiKey": API_KEY,
                    "api": "openai-completions",
                    "models": [_model_entry(mid) for mid in model_ids],
                },
            },
        },
        "agents": {
            "defaults": {
                "model": {
                    "primary": f"local/{primary_model}",
                },
                "models": {
                    f"local/{mid}": {"alias": f"{mid} (local vLLM)"}
                    for mid in model_ids
                },
                "maxConcurrent": 4,
                "subagents": {
                    "maxConcurrent": 8,
                },
            },
        },
        "tools": {
            "byProvider": {
                f"local/{mid}": {"deny": ["*"]} for mid in model_ids
            },
        },
        "commands": {
            "native": "auto",
            "nativeSkills": "auto",
        },
        "channels": {
            "telegram": {
                "enabled": True,
                "dmPolicy": "open",
                "allowFrom": ["*"],
                "groupPolicy": "open",
                "groups": {
                    "*": {"requireMention": False},
                },
                "streamMode": "off",
                "actions": {
                    "reactions": True,
                    "sendMessage": True,
                    "deleteMessage": True,
                },
            },
        },
        "gateway": {
            "port": 18789,
            "mode": "local",
            "bind": "loopback",
            "auth": {
                "mode": "token",
                "token": "openclaw-local-dev-token",
            },
        },
        "messages": {
            "ackReactionScope": "group-mentions",
        },
        "plugins": {
            "entries": {
                "telegram": {"enabled": True},
            },
        },
    }

    if TELEGRAM_BOT_TOKEN:
        cfg["channels"]["telegram"]["botToken"] = TELEGRAM_BOT_TOKEN

    # Write openclaw.json
    config_path = OPENCLAW_HOME / "openclaw.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {config_path}")

    # Write agents/main/agent/models.json
    agent_models_path = OPENCLAW_HOME / "agents" / "main" / "agent" / "models.json"
    agent_models_path.parent.mkdir(parents=True, exist_ok=True)
    agent_models = {
        "providers": {
            "local": {
                "baseUrl": VLLM_BASE_URL,
                "apiKey": API_KEY,
                "api": "openai-completions",
                "models": [_model_entry(mid) for mid in model_ids],
            },
        },
    }
    agent_models_path.write_text(json.dumps(agent_models, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {agent_models_path}")


# ---------------------------------------------------------------------------
# Step 3: Fix permissions so OpenClaw's node user can read everything
# ---------------------------------------------------------------------------
def fix_permissions() -> None:
    for root, dirs, files in os.walk(str(OPENCLAW_HOME)):
        for d in dirs:
            p = os.path.join(root, d)
            os.chmod(p, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        for f in files:
            p = os.path.join(root, f)
            os.chmod(p, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== CLaaS init container ===")
    print(f"  LORA_NAME       = {LORA_NAME}")
    print(f"  BASE_MODEL      = {BASE_MODEL}")
    print(f"  LORA_ROOT       = {LORA_ROOT}")
    print(f"  OPENCLAW_HOME   = {OPENCLAW_HOME}")
    print(f"  VLLM_BASE_URL   = {VLLM_BASE_URL}")
    print()

    create_lora()
    write_openclaw_config()
    fix_permissions()

    print("\n=== Init complete ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        raise SystemExit(1) from e
