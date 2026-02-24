"""Tests for Hydra-based eval config loading."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from claas.eval.config import build_harness_config
from claas.eval.metrics import VerifierResult
from claas.eval.preferences import get_preference_configs
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


# --- Preference YAML loading tests ---


def test_preference_configs_load_from_yaml() -> None:
    configs = get_preference_configs()
    assert {"no_emoji", "concise", "identity"}.issubset(configs.keys())
    for name, cfg in configs.items():
        assert cfg.name == name
        assert isinstance(cfg.feedback_string, str)
        assert len(cfg.feedback_string) > 0
        assert len(cfg.probe_prompts) > 0
        assert len(cfg.logprob_pairs) > 0


def test_preference_verifier_callable() -> None:
    configs = get_preference_configs()

    # no_emoji: clean text should pass
    result = configs["no_emoji"].verifier("Hello, how are you?")
    assert isinstance(result, VerifierResult)
    assert result.score == 1.0
    assert result.passed is True

    # no_emoji: text with emoji should fail
    result = configs["no_emoji"].verifier("Hello! \U0001f60a")
    assert result.score == 0.0
    assert result.passed is False

    # concise: short text should pass (>=10 words, <=3 sentences)
    result = configs["concise"].verifier(
        "Python is a versatile, high-level programming language known for its readable syntax."
    )
    assert result.score == 1.0
    assert result.passed is True

    # concise: degenerate text (too few words) should fail
    result = configs["concise"].verifier("Just a dot.")
    assert result.score == 0.0
    assert result.passed is False

    # concise: verbose text should fail
    result = configs["concise"].verifier(
        "One thing to know. Two things to know. Three things to know. "
        "Four things to know. Five things to know. Six things to know. "
        "Seven things to know. Eight things to know. Nine things to know. Ten things."
    )
    assert result.score < 1.0
    assert result.passed is False

    # identity: text with 'kuro' should pass
    result = configs["identity"].verifier("I'm Kuro, nice to meet you!")
    assert result.score == 1.0
    assert result.passed is True

    # identity: text without 'kuro' should fail
    result = configs["identity"].verifier("I'm an AI assistant.")
    assert result.score == 0.0
    assert result.passed is False


def test_preference_logprob_pairs_structure() -> None:
    configs = get_preference_configs()
    for cfg in configs.values():
        for pair in cfg.logprob_pairs:
            assert len(pair.prompt_messages) > 0
            for msg in pair.prompt_messages:
                assert "role" in msg
                assert "content" in msg
                assert msg["role"] in ("system", "user", "assistant")
            assert isinstance(pair.positive_response, str)
            assert isinstance(pair.negative_response, str)
            assert len(pair.positive_response) > 0
            assert len(pair.negative_response) > 0
