#!/usr/bin/env bash
set -euo pipefail

OPENCLAW_HOME="${OPENCLAW_HOME:-$HOME/.openclaw}"
OPENCLAW_CONFIG="${OPENCLAW_CONFIG:-$OPENCLAW_HOME/openclaw.json}"
OPENCLAW_AGENT_MODELS="${OPENCLAW_AGENT_MODELS:-$OPENCLAW_HOME/agents/main/agent/models.json}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
API_KEY="${API_KEY:-sk-local}"
MODEL_IDS="${MODEL_IDS:-qwen3-8b}"
PRIMARY_MODEL="${PRIMARY_MODEL:-qwen3-8b}"

mkdir -p "$(dirname "$OPENCLAW_CONFIG")"
mkdir -p "$(dirname "$OPENCLAW_AGENT_MODELS")"

python3 -c '
import json
import os
import sys
from datetime import datetime, timezone

config_path = sys.argv[1]
agent_models_path = sys.argv[2]
base_url = sys.argv[3]
api_key = sys.argv[4]
model_ids = [m.strip() for m in sys.argv[5].split(",") if m.strip()]
primary_model = sys.argv[6]

if not model_ids:
    model_ids = ["qwen3-8b"]
if primary_model not in model_ids:
    model_ids.append(primary_model)

now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def base_model_entry(model_id: str) -> dict:
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

def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

cfg = load_json(config_path)
cfg.setdefault("models", {})
cfg["models"]["mode"] = "merge"
cfg["models"].setdefault("providers", {})
cfg["models"]["providers"]["local"] = {
    "baseUrl": base_url,
    "apiKey": api_key,
    "api": "openai-completions",
    "models": [base_model_entry(mid) for mid in model_ids],
}

cfg.setdefault("agents", {})
cfg["agents"].setdefault("defaults", {})
cfg["agents"]["defaults"].setdefault("model", {})
cfg["agents"]["defaults"]["model"]["primary"] = f"local/{primary_model}"

agent_aliases = cfg["agents"]["defaults"].setdefault("models", {})
for mid in model_ids:
    agent_aliases.setdefault(f"local/{mid}", {"alias": f"{mid} (local vLLM)"})

cfg.setdefault("tools", {})
cfg["tools"].setdefault("byProvider", {})
for mid in model_ids:
    cfg["tools"]["byProvider"].setdefault(f"local/{mid}", {"deny": ["*"]})

cfg.setdefault("meta", {})
cfg["meta"]["lastTouchedAt"] = now

with open(config_path, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")

agent_models = {
    "providers": {
        "local": {
            "baseUrl": base_url,
            "apiKey": api_key,
            "api": "openai-completions",
            "models": [base_model_entry(mid) for mid in model_ids],
        }
    }
}
with open(agent_models_path, "w", encoding="utf-8") as f:
    json.dump(agent_models, f, indent=2)
    f.write("\n")

print(f"Updated {config_path}")
print(f"Updated {agent_models_path}")
print(f"Primary model: local/{primary_model}")
print("Models: " + ", ".join(model_ids))
' "$OPENCLAW_CONFIG" "$OPENCLAW_AGENT_MODELS" "$BASE_URL" "$API_KEY" "$MODEL_IDS" "$PRIMARY_MODEL"
