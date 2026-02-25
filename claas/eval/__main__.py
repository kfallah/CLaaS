"""CLI entry point for the evaluation harness.

Usage::

    python -m claas.eval                              # base config
    python -m claas.eval mode=tinker num_steps=5      # overrides
"""

from __future__ import annotations

import asyncio

import hydra
from omegaconf import OmegaConf

from . import config as _config  # noqa: F401
from .types import EvalConfig


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: EvalConfig) -> None:
    from .runner import run_harness

    eval_cfg = OmegaConf.to_object(cfg)
    if not isinstance(eval_cfg, EvalConfig):
        raise TypeError("Hydra did not produce an EvalConfig instance")

    asyncio.run(run_harness(eval_cfg))


if __name__ == "__main__":
    main()
