"""Tests for YAML config loading (claas.eval.config)."""

from __future__ import annotations

import os
import tempfile

import pytest

from claas.eval.config import build_config_from_yaml, load_yaml_config
from claas.eval.types import HarnessConfig


def _write_yaml(content: str) -> str:
    """Write YAML content to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


# ── load_yaml_config ─────────────────────────────────────────────────


def test_yaml_config_load():
    """Round-trip a YAML config, verify all fields are loaded."""
    path = _write_yaml(
        """\
mode: tinker
claas_url: http://example.com:8080
base_model: Qwen/Qwen3-Coder-30B-A3B-Instruct
preferences:
  - no_emoji
  - concise
metrics:
  - logprob
  - compliance
collapse_steps: [0, 5, 10]
plots: true
num_steps: 10
batch_size: 2
steps_per_batch: 3
seed: 123
lora_id_prefix: test
output_dir: /tmp/evals
"""
    )
    try:
        data = load_yaml_config(path)
        assert data["mode"] == "tinker"
        assert data["claas_url"] == "http://example.com:8080"
        assert data["preferences"] == ["no_emoji", "concise"]
        assert data["metrics"] == ["logprob", "compliance"]
        assert data["collapse_steps"] == [0, 5, 10]
        assert data["num_steps"] == 10
        assert data["batch_size"] == 2
        assert data["steps_per_batch"] == 3
        assert data["seed"] == 123
    finally:
        os.unlink(path)


def test_yaml_unknown_keys_raises():
    """Unknown YAML keys raise ValueError."""
    path = _write_yaml("mode: local\nbogus_key: 42\n")
    try:
        with pytest.raises(ValueError, match="Unknown config keys.*bogus_key"):
            load_yaml_config(path)
    finally:
        os.unlink(path)


def test_yaml_non_mapping_raises():
    """Non-mapping YAML raises ValueError."""
    path = _write_yaml("- item1\n- item2\n")
    try:
        with pytest.raises(ValueError, match="must be a mapping"):
            load_yaml_config(path)
    finally:
        os.unlink(path)


# ── build_config_from_yaml ───────────────────────────────────────────


def test_build_config_from_yaml_basic():
    """Build a HarnessConfig from YAML, verify fields and output_dir timestamping."""
    path = _write_yaml(
        """\
mode: tinker
num_steps: 5
batch_size: 2
steps_per_batch: 2
seed: 99
output_dir: /tmp/test-evals
"""
    )
    try:
        config = build_config_from_yaml(path)
        assert isinstance(config, HarnessConfig)
        assert config.mode == "tinker"
        assert config.num_steps == 5
        assert config.batch_size == 2
        assert config.steps_per_batch == 2
        assert config.seed == 99
        # output_dir should have a timestamped subdir appended
        assert config.output_dir.startswith("/tmp/test-evals/")
        assert len(config.output_dir) > len("/tmp/test-evals/")
        # Tinker mode: proxy_url should default to vllm_url
        assert config.proxy_url == config.vllm_url
    finally:
        os.unlink(path)


def test_yaml_config_with_cli_overrides():
    """CLI overrides take priority over YAML values."""
    path = _write_yaml(
        """\
mode: local
num_steps: 20
seed: 42
batch_size: 4
"""
    )
    try:
        config = build_config_from_yaml(path, cli_overrides={"num_steps": 5, "seed": 123})
        assert config.num_steps == 5
        assert config.seed == 123
        # Non-overridden values come from YAML
        assert config.mode == "local"
        assert config.batch_size == 4
    finally:
        os.unlink(path)


def test_yaml_metrics_string_coercion():
    """Metrics as a comma-separated string are split into a list."""
    path = _write_yaml('metrics: "logprob,compliance"\n')
    try:
        config = build_config_from_yaml(path)
        assert config.metrics == ["logprob", "compliance"]
    finally:
        os.unlink(path)


def test_yaml_metrics_list():
    """Metrics as a YAML list are preserved."""
    path = _write_yaml("metrics:\n  - logprob\n  - collapse\n")
    try:
        config = build_config_from_yaml(path)
        assert config.metrics == ["logprob", "collapse"]
    finally:
        os.unlink(path)


def test_yaml_collapse_steps_string_coercion():
    """collapse_steps as a comma string are parsed into a set of ints."""
    path = _write_yaml('collapse_steps: "0,5,10"\n')
    try:
        config = build_config_from_yaml(path)
        assert config.collapse_steps == {0, 5, 10}
    finally:
        os.unlink(path)


def test_yaml_collapse_steps_list():
    """collapse_steps as a YAML list are parsed into a set of ints."""
    path = _write_yaml("collapse_steps: [0, 10, 19]\n")
    try:
        config = build_config_from_yaml(path)
        assert config.collapse_steps == {0, 10, 19}
    finally:
        os.unlink(path)


def test_steps_per_batch_in_yaml():
    """steps_per_batch is read correctly from YAML."""
    path = _write_yaml("steps_per_batch: 4\n")
    try:
        config = build_config_from_yaml(path)
        assert config.steps_per_batch == 4
    finally:
        os.unlink(path)
