"""Typed cache structures and helpers for CPU-resident LoRA state between training steps."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import cast

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


def cpu_optimizer_state(state_dict: dict[str, object]) -> dict[str, object]:
    """Deep-copy optimizer state with all tensors moved to CPU."""
    result: dict[str, object] = {}
    for key, value in state_dict.items():
        if key == "state":
            param_states = cast("dict[int, dict[str, object]]", value)
            cpu_states: dict[int, dict[str, object]] = {}
            for param_id, param_state in param_states.items():
                cpu_param: dict[str, object] = {}
                for k, v in param_state.items():
                    if isinstance(v, torch.Tensor):
                        cpu_param[k] = v.detach().cpu().clone()
                    else:
                        cpu_param[k] = copy.deepcopy(v)
                cpu_states[param_id] = cpu_param
            result[key] = cpu_states
        else:
            result[key] = copy.deepcopy(value)
    return result


def gpu_optimizer_state(
    state_dict: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    """Deep-copy optimizer state with all tensors moved to a target device."""
    result: dict[str, object] = {}
    for key, value in state_dict.items():
        if key == "state":
            param_states = cast("dict[int, dict[str, object]]", value)
            gpu_states: dict[int, dict[str, object]] = {}
            for param_id, param_state in param_states.items():
                gpu_param: dict[str, object] = {}
                for k, v in param_state.items():
                    if isinstance(v, torch.Tensor):
                        gpu_param[k] = v.detach().to(device).clone()
                    else:
                        gpu_param[k] = copy.deepcopy(v)
                gpu_states[param_id] = gpu_param
            result[key] = gpu_states
        else:
            result[key] = copy.deepcopy(value)
    return result
