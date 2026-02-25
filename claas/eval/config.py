"""Hydra schema registration for the evaluation harness."""

from hydra.core.config_store import ConfigStore

from .types import EvalConfig

ConfigStore.instance().store(name="_eval_schema", node=EvalConfig)
