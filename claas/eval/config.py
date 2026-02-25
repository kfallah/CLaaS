"""Hydra-based configuration for the evaluation harness."""

from __future__ import annotations

import dataclasses
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from hydra.core.config_store import ConfigStore

from claas.core.types import TrainingConfig

from .types import EvalConfig

# Pattern matching the timestamped run-id suffix (e.g. 20260220-012345Z)
_RUN_ID_RE = re.compile(r"\d{8}-\d{6}Z$")

ConfigStore.instance().store(name="_eval_schema", node=EvalConfig)


def build_harness_config(eval_cfg: EvalConfig) -> EvalConfig:
    """Post-process EvalConfig while preserving strict runtime types."""
    if not isinstance(eval_cfg.training, TrainingConfig):
        raise TypeError(
            f"EvalConfig.training must be TrainingConfig, got {type(eval_cfg.training)!r}"
        )

    config = dataclasses.replace(
        eval_cfg,
        training=dataclasses.replace(eval_cfg.training),
    )

    # Timestamped output subdir (skip if output_dir already ends with a run-id,
    # which allows resuming an existing run by passing its directory).
    output_dir = config.output_dir
    if not _RUN_ID_RE.search(Path(output_dir).name):
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
        config = dataclasses.replace(
            config,
            output_dir=os.path.join(output_dir, run_id),
        )

    return config
