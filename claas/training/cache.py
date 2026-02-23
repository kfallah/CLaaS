"""Typed cache structures for CPU-resident LoRA state between training steps."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from claas.core.types import DistillResponse


@dataclass(frozen=True, slots=True)
class LoraAdapterConfig:
    """Typed representation of LoRA adapter configuration."""

    r: int
    lora_alpha: int
    target_modules: list[str]
    lora_dropout: float
    bias: str
    task_type: str


@dataclass(frozen=True, slots=True)
class LoraCacheEntry:
    """CPU-resident snapshot of LoRA adapter state between training steps."""

    lora_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, object]
    adapter_config: LoraAdapterConfig


@dataclass(frozen=True, slots=True)
class DistillStepResult:
    """Result of a distillation step with both response and cache entry."""

    response: DistillResponse
    cache_entry: LoraCacheEntry
