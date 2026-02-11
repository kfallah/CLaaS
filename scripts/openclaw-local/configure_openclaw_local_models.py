#!/usr/bin/env python3
"""Write OpenClaw config JSON pointing at a local vLLM server.

Usage:
    python scripts/openclaw-local/configure_openclaw_local_models.py

Reads from environment variables (see openclaw-local.env.example).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

OPENCLAW_HOME = Path(os.environ.get("OPENCLAW_HOME", Path.home() / ".openclaw"))
OPENCLAW_CONFIG = Path(os.environ.get("OPENCLAW_CONFIG", OPENCLAW_HOME / "openclaw.json"))
OPENCLAW_AGENT_MODELS = Path(
    os.environ.get(
        "OPENCLAW_AGENT_MODELS",
        OPENCLAW_HOME / "agents" / "main" / "agent" / "models.json",
    )
)
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("API_KEY", "sk-local")
MODEL_IDS = [m.strip() for m in os.environ.get("MODEL_IDS", "qwen3-8b").split(",") if m.strip()]
PRIMARY_MODEL = os.environ.get("PRIMARY_MODEL", "qwen3-8b")


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


def main() -> None:
    if not MODEL_IDS:
        MODEL_IDS.append("qwen3-8b")
    if PRIMARY_MODEL not in MODEL_IDS:
        MODEL_IDS.append(PRIMARY_MODEL)

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    # --- openclaw.json ---
    OPENCLAW_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    cfg: dict = {}
    if OPENCLAW_CONFIG.exists():
        try:
            cfg = json.loads(OPENCLAW_CONFIG.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cfg = {}

    cfg.setdefault("models", {})
    cfg["models"]["mode"] = "merge"
    cfg["models"].setdefault("providers", {})
    cfg["models"]["providers"]["local"] = {
        "baseUrl": BASE_URL,
        "apiKey": API_KEY,
        "api": "openai-completions",
        "models": [_model_entry(mid) for mid in MODEL_IDS],
    }

    cfg.setdefault("agents", {})
    cfg["agents"].setdefault("defaults", {})
    cfg["agents"]["defaults"].setdefault("model", {})
    cfg["agents"]["defaults"]["model"]["primary"] = f"local/{PRIMARY_MODEL}"

    agent_aliases = cfg["agents"]["defaults"].setdefault("models", {})
    for mid in MODEL_IDS:
        agent_aliases.setdefault(f"local/{mid}", {"alias": f"{mid} (local vLLM)"})

    cfg.setdefault("tools", {})
    cfg["tools"].setdefault("byProvider", {})
    for mid in MODEL_IDS:
        cfg["tools"]["byProvider"].setdefault(f"local/{mid}", {"deny": ["*"]})

    cfg.setdefault("meta", {})
    cfg["meta"]["lastTouchedAt"] = now

    OPENCLAW_CONFIG.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    # --- agent models.json ---
    OPENCLAW_AGENT_MODELS.parent.mkdir(parents=True, exist_ok=True)
    agent_models = {
        "providers": {
            "local": {
                "baseUrl": BASE_URL,
                "apiKey": API_KEY,
                "api": "openai-completions",
                "models": [_model_entry(mid) for mid in MODEL_IDS],
            }
        }
    }
    OPENCLAW_AGENT_MODELS.write_text(json.dumps(agent_models, indent=2) + "\n", encoding="utf-8")

    print(f"Updated {OPENCLAW_CONFIG}")
    print(f"Updated {OPENCLAW_AGENT_MODELS}")
    print(f"Primary model: local/{PRIMARY_MODEL}")
    print("Models: " + ", ".join(MODEL_IDS))


if __name__ == "__main__":
    main()
