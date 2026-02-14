#!/usr/bin/env python3
"""One-shot init container: create initial LoRA + write OpenClaw config.

Idempotent — skips LoRA creation if the alias already exists.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
LORA_NAME = os.environ.get("LORA_NAME", "openclaw/assistant")
BASE_MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
DISTILL_MODE = os.environ.get("CLAAS_DISTILL_EXECUTION_MODE", "local").strip().lower()
LORA_ROOT = os.environ.get("CLAAS_LORA_ROOT", "/loras")
OPENCLAW_HOME = Path(os.environ.get("OPENCLAW_HOME", "/openclaw-config"))
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://vllm:8000/v1")
API_KEY = os.environ.get("API_KEY", "sk-local")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CLAAS_API_URL = os.environ.get("CLAAS_API_URL", "http://localhost:8080")

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
        raw_aliases = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(raw_aliases, dict):
        return {}

    aliases: dict[str, str] = {}
    for alias, target in raw_aliases.items():
        if not isinstance(alias, str) or not isinstance(target, str):
            continue
        aliases[alias.strip("/")] = target.strip("/")
    return aliases


# ---------------------------------------------------------------------------
# Step 1: Create initial LoRA (idempotent)
# ---------------------------------------------------------------------------
def create_lora() -> None:
    alias_key = f"{LORA_NAME}-latest"
    if DISTILL_MODE == "tinker":
        create_tinker_lora(alias_key)
        return

    aliases = _read_aliases()

    if alias_key in aliases:
        target = aliases[alias_key]
        adapter_dir = Path(LORA_ROOT) / target
        has_config = (adapter_dir / "adapter_config.json").exists()
        has_weights = (adapter_dir / "adapter_model.safetensors").exists()
        if has_config and has_weights:
            print(f"LoRA alias '{alias_key}' already exists → {target}, skipping creation.")
            return

    from claas.storage import create_initial_lora

    full_id = create_initial_lora(LORA_NAME, base_model_name=BASE_MODEL)
    print(f"Created initial LoRA: {full_id}")


def _wait_for_api(timeout_s: float = 120.0) -> None:
    deadline = time.time() + timeout_s
    while True:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{CLAAS_API_URL.rstrip('/')}/")
                if resp.status_code < 500:
                    return
        except httpx.HTTPError:
            pass
        if time.time() >= deadline:
            raise RuntimeError(f"Timed out waiting for CLaaS API at {CLAAS_API_URL}")
        time.sleep(2)


def create_tinker_lora(alias_key: str) -> None:
    """Initialize LoRA in Tinker mode via the CLaaS API."""
    _wait_for_api()
    api_url = CLAAS_API_URL.rstrip("/")
    with httpx.Client(timeout=30.0) as client:
        listed = client.get(f"{api_url}/v1/lora", params={"prefix": LORA_NAME})
        listed.raise_for_status()
        loras = listed.json().get("loras", [])
        if alias_key in loras:
            print(f"LoRA '{alias_key}' already exists in Tinker state, skipping creation.")
            return

        init_resp = client.post(
            f"{api_url}/v1/lora/init",
            json={"lora_id": alias_key, "base_model": BASE_MODEL},
        )
        init_resp.raise_for_status()
        print(f"Created initial Tinker LoRA: {alias_key}")


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
                "claas-feedback": {
                    "enabled": True,
                    "config": {
                        "claasApiUrl": CLAAS_API_URL,
                        "loraId": f"{LORA_NAME}-latest",
                    },
                },
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
# Step 2b: Install feedback plugin into OpenClaw extensions
# ---------------------------------------------------------------------------
PLUGIN_SOURCE = Path("/app/plugins/claas-feedback")


def install_feedback_plugin() -> None:
    dest = OPENCLAW_HOME / "extensions" / "claas-feedback"
    if not PLUGIN_SOURCE.is_dir():
        print(f"Plugin source {PLUGIN_SOURCE} not found, skipping feedback plugin.")
        return

    import shutil

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(PLUGIN_SOURCE, dest)
    print(f"Installed feedback plugin → {dest}")


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
    print(f"  DISTILL_MODE    = {DISTILL_MODE}")
    print(f"  VLLM_BASE_URL   = {VLLM_BASE_URL}")
    print()

    create_lora()
    write_openclaw_config()
    install_feedback_plugin()
    fix_permissions()

    print("\n=== Init complete ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        raise SystemExit(1) from e
