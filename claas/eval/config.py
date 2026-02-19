"""YAML configuration loading for the evaluation harness."""

from __future__ import annotations

import dataclasses
import os
from datetime import datetime, timezone

from .types import HarnessConfig

# Valid keys: field names of HarnessConfig
_VALID_KEYS = {f.name for f in dataclasses.fields(HarnessConfig)}


def load_yaml_config(path: str) -> dict:
    """Read a YAML file and return a raw dict.

    Validates that no unknown keys are present (compared to HarnessConfig fields).
    Raises ValueError for unknown keys.
    """
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping, got {type(data).__name__}")

    unknown = set(data.keys()) - _VALID_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")

    return data


def build_config_from_yaml(
    yaml_path: str,
    cli_overrides: dict | None = None,
) -> HarnessConfig:
    """Load a YAML config, apply CLI overrides, and return a HarnessConfig.

    CLI overrides take priority over YAML values. Handles type coercions for
    metrics (str or list), collapse_steps (str or list → set[int]),
    and preferences (list).
    """
    data = load_yaml_config(yaml_path)

    # Apply CLI overrides on top
    if cli_overrides:
        data.update(cli_overrides)

    # Type coercions
    # metrics: accept comma-separated string or list
    if "metrics" in data:
        val = data["metrics"]
        if isinstance(val, str):
            data["metrics"] = [m.strip() for m in val.split(",") if m.strip()]
        elif isinstance(val, list):
            data["metrics"] = [str(m).strip() for m in val]

    # collapse_steps: accept comma-separated string or list → set[int]
    if "collapse_steps" in data:
        val = data["collapse_steps"]
        if isinstance(val, str):
            data["collapse_steps"] = {int(s.strip()) for s in val.split(",") if s.strip()}
        elif isinstance(val, list):
            data["collapse_steps"] = {int(s) for s in val}

    # preferences: ensure list of strings
    if "preferences" in data and isinstance(data["preferences"], list):
        data["preferences"] = [str(p) for p in data["preferences"]]

    # Tinker defaults: proxy_url fallback
    mode = data.get("mode", "local")
    if mode == "tinker" and not data.get("proxy_url"):
        data["proxy_url"] = data.get("vllm_url", "http://localhost:8000")

    # openclaw_api_key: env var fallback
    if "openclaw_api_key" not in data:
        token = os.environ.get("OPENCLAW_GATEWAY_TOKEN")
        if token:
            data["openclaw_api_key"] = token

    # Timestamped subdir under the base output directory
    base_output = data.get("output_dir", "./data/evals")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    data["output_dir"] = os.path.join(base_output, run_id)

    return HarnessConfig(**data)
