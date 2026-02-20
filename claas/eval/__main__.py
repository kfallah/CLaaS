"""CLI entry point for the evaluation harness.

Usage::

    python -m claas.eval                              # base config
    python -m claas.eval mode=tinker num_steps=5      # overrides
    python -m claas.eval --config-dir ./my_configs    # custom config dir
"""

from __future__ import annotations

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from .types import EvalConfig

cs = ConfigStore.instance()
cs.store(name="base", node=EvalConfig)


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: EvalConfig) -> None:
    import asyncio

    from .config import build_harness_config
    from .runner import run_harness

    eval_cfg = EvalConfig(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[invalid-argument-type]
    config = build_harness_config(eval_cfg)
    asyncio.run(run_harness(config))


if __name__ == "__main__":
    main()
