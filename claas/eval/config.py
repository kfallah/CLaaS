"""Hydra-based configuration for the evaluation harness."""

from __future__ import annotations

import dataclasses
import os
from datetime import datetime, timezone
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from .types import EvalConfig, HarnessConfig

# Register EvalConfig as the structured config schema
cs = ConfigStore.instance()
cs.store(name="_eval_schema", node=EvalConfig)

# Default config directory (relative to this module)
_DEFAULT_CONFIG_DIR = str(Path(__file__).parent / "configs")


def load_config(
    config_dir: str | None = None,
    config_name: str = "base",
    overrides: list[str] | None = None,
) -> HarnessConfig:
    """Load config via Hydra Compose API and return a HarnessConfig.

    Secrets (API keys) are NOT stored on the config object — they are
    resolved from env vars at call sites in the runner.
    """
    abs_dir = os.path.abspath(config_dir or _DEFAULT_CONFIG_DIR)

    with initialize_config_dir(version_base=None, config_dir=abs_dir):
        cfg = compose(config_name=config_name, overrides=overrides or [])

    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    eval_cfg = EvalConfig(**container)  # type: ignore[arg-type]
    return build_harness_config(eval_cfg)


def build_harness_config(eval_cfg: EvalConfig) -> HarnessConfig:
    """Post-process EvalConfig → HarnessConfig (no secrets)."""
    fields = dataclasses.asdict(eval_cfg)

    # Tinker defaults: proxy_url fallback to vllm_url
    if fields["mode"] == "tinker" and not fields.get("proxy_url"):
        fields["proxy_url"] = fields["vllm_url"]

    # Timestamped output subdir
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    fields["output_dir"] = os.path.join(fields["output_dir"], run_id)

    return HarnessConfig(**fields)
