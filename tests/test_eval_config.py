"""Tests for Hydra-based eval config loading."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from claas.eval.config import build_harness_config
from claas.eval.types import EvalConfig, HarnessConfig


def _make_config_dir(yaml_content: str) -> str:
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "base.yaml")
    with open(path, "w") as f:
        f.write(yaml_content)
    return tmpdir


def _compose_eval_config(
    overrides: list[str] | None = None,
    *,
    config_dir: str | None = None,
    config_name: str = "base",
) -> EvalConfig:
    if config_dir is None:
        config_dir = str(Path(__file__).resolve().parents[1] / "claas" / "eval" / "configs")

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=overrides or [])

    typed_cfg = OmegaConf.merge(OmegaConf.structured(EvalConfig), cfg)
    obj = OmegaConf.to_object(typed_cfg)
    if not isinstance(obj, EvalConfig):
        raise TypeError("Hydra did not produce an EvalConfig instance")
    return obj


def test_hydra_config_defaults() -> None:
    config = build_harness_config(_compose_eval_config())
    assert isinstance(config, HarnessConfig)
    assert config.mode == "tinker"
    assert config.num_steps == 20
    assert config.batch_size == 4
    assert config.seed == 42
    assert config.lora_id_prefix == "eval"
    assert config.plots is True


def test_hydra_config_with_overrides() -> None:
    config = build_harness_config(_compose_eval_config(["num_steps=5", "seed=99"]))
    assert config.num_steps == 5
    assert config.seed == 99


def test_unknown_key_rejected() -> None:
    tmpdir = _make_config_dir("mode: local\nbogus_key: 42\n")
    with pytest.raises(Exception):
        _compose_eval_config(config_dir=tmpdir)


def test_timestamped_output_dir() -> None:
    config = build_harness_config(_compose_eval_config(["output_dir=/tmp/test-evals"]))
    assert config.output_dir.startswith("/tmp/test-evals/")
    assert len(config.output_dir) > len("/tmp/test-evals/")


def test_collapse_steps_list() -> None:
    config = build_harness_config(_compose_eval_config(["collapse_steps=[0,5,10]"]))
    assert config.collapse_steps == [0, 5, 10]


def test_steps_per_batch_from_config() -> None:
    config = build_harness_config(_compose_eval_config(["steps_per_batch=3"]))
    assert config.steps_per_batch == 3


def test_hydra_config_custom_dir() -> None:
    tmpdir = _make_config_dir("mode: local\nnum_steps: 3\npreferences:\n  - no_emoji\n")
    config = build_harness_config(_compose_eval_config(config_dir=tmpdir))
    assert config.mode == "local"
    assert config.num_steps == 3
    assert config.preferences == ["no_emoji"]
