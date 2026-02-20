"""Tests for Hydra-based eval config loading (claas.eval.config)."""

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
    """Write YAML to a temp directory as base.yaml and return the directory."""
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
        eval_config_dir = Path(__file__).resolve().parents[1] / "claas" / "eval" / "configs"
        resolved_config_dir = str(eval_config_dir)
    else:
        resolved_config_dir = config_dir

    with initialize_config_dir(version_base=None, config_dir=resolved_config_dir):
        cfg = compose(config_name=config_name, overrides=overrides or [])

    typed_cfg = OmegaConf.merge(OmegaConf.structured(EvalConfig), cfg)
    obj = OmegaConf.to_object(typed_cfg)
    if not isinstance(obj, EvalConfig):
        raise TypeError("Hydra did not produce an EvalConfig instance")
    return obj


# ── Hydra defaults ───────────────────────────────────────────────────


def test_hydra_config_defaults():
    """Load the shipped base.yaml, verify key field values."""
    config = build_harness_config(_compose_eval_config())
    assert isinstance(config, HarnessConfig)
    assert config.mode == "tinker"  # base.yaml ships with tinker
    assert config.num_steps == 20
    assert config.batch_size == 4
    assert config.seed == 42
    assert config.lora_id_prefix == "eval"
    assert config.plots is True


# ── overrides ────────────────────────────────────────────────────────


def test_hydra_config_with_overrides():
    """Hydra key=value overrides change config values."""
    config = build_harness_config(_compose_eval_config(["num_steps=5", "seed=99"]))
    assert config.num_steps == 5
    assert config.seed == 99


# ── unknown key rejected ─────────────────────────────────────────────


def test_unknown_key_rejected():
    """Unknown keys in YAML cause an error at load time."""
    tmpdir = _make_config_dir("mode: local\nbogus_key: 42\n")
    with pytest.raises(Exception):
        _compose_eval_config(config_dir=tmpdir)


# ── tinker proxy default ─────────────────────────────────────────────


def test_tinker_proxy_default():
    """In tinker mode, proxy_url defaults to vllm_url."""
    config = build_harness_config(_compose_eval_config(["mode=tinker", "proxy_url=null"]))
    assert config.proxy_url == config.vllm_url


# ── timestamped output_dir ───────────────────────────────────────────


def test_timestamped_output_dir():
    """Output dir gets a timestamped run_id subdir."""
    config = build_harness_config(_compose_eval_config(["output_dir=/tmp/test-evals"]))
    assert config.output_dir.startswith("/tmp/test-evals/")
    assert len(config.output_dir) > len("/tmp/test-evals/")


# ── collapse_steps as list ───────────────────────────────────────────


def test_collapse_steps_list():
    """collapse_steps loaded as list[int]."""
    config = build_harness_config(_compose_eval_config(["collapse_steps=[0,5,10]"]))
    assert config.collapse_steps == [0, 5, 10]


# ── steps_per_batch ──────────────────────────────────────────────────


def test_steps_per_batch_from_config():
    """steps_per_batch is read correctly."""
    config = build_harness_config(_compose_eval_config(["steps_per_batch=3"]))
    assert config.steps_per_batch == 3


# ── build_harness_config ─────────────────────────────────────────────


def test_build_harness_config_tinker_proxy():
    """build_harness_config sets proxy_url for tinker mode."""
    eval_cfg = EvalConfig(mode="tinker")
    config = build_harness_config(eval_cfg)
    assert config.proxy_url == config.vllm_url


def test_build_harness_config_local_no_proxy():
    """build_harness_config does NOT set proxy_url for local mode."""
    eval_cfg = EvalConfig(mode="local")
    config = build_harness_config(eval_cfg)
    assert config.proxy_url is None


# ── custom config dir ────────────────────────────────────────────────


def test_hydra_config_custom_dir():
    """Loading from a custom config directory works."""
    tmpdir = _make_config_dir(
        "mode: local\nnum_steps: 3\npreferences:\n  - no_emoji\n"
    )
    config = build_harness_config(_compose_eval_config(config_dir=tmpdir))
    assert config.mode == "local"
    assert config.num_steps == 3
    assert config.preferences == ["no_emoji"]
