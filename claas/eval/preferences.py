"""Preference configurations for the evaluation harness.

Each preference is defined in a YAML file under configs/preference/.
This module loads them and instantiates verifier classes via Hydra.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from hydra.utils import instantiate

from claas.core.types import ChatMessage
from claas.eval.types import LogprobPair

if TYPE_CHECKING:
    from claas.eval.metrics.verifiers import Verifier


@dataclass
class PreferenceConfig:
    """Full configuration for a single preference type."""

    name: str
    feedback_string: str
    logprob_pairs: list[LogprobPair]
    probe_prompts: list[str]
    verifier: Verifier


_PREFERENCE_DIR = Path(__file__).resolve().parent / "configs" / "preference"


def get_preference_configs() -> dict[str, PreferenceConfig]:
    """Return all preference configurations keyed by name."""
    configs: dict[str, PreferenceConfig] = {}
    for yaml_path in sorted(_PREFERENCE_DIR.glob("*.yaml")):
        try:
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            name = data["name"]
        except (KeyError, TypeError, yaml.YAMLError) as exc:
            raise ValueError(f"Invalid preference YAML {yaml_path.name}: {exc}") from exc
        if name in configs:
            raise ValueError(
                f"Duplicate preference name '{name}' in {yaml_path.name} "
                f"(already defined by another config file)"
            )
        verifier = instantiate(data["verifier"])
        logprob_pairs = [
            LogprobPair(
                prompt_messages=[ChatMessage(**m) for m in pair["prompt_messages"]],
                positive_response=pair["positive_response"],
                negative_response=pair["negative_response"],
            )
            for pair in data["logprob_pairs"]
        ]
        configs[name] = PreferenceConfig(
            name=name,
            feedback_string=data["feedback_string"],
            logprob_pairs=logprob_pairs,
            probe_prompts=data["probe_prompts"],
            verifier=verifier,
        )
    return configs
