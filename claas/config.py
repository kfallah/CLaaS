"""Configuration for CLaaS.

Centralizes all configuration settings including model IDs, training defaults,
and infrastructure settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration."""

    # Student model (Qwen3-8B)
    student_model_id: str = "Qwen/Qwen3-8B"
    student_dtype: str = "bfloat16"
    student_attn_implementation: str = "flash_attention_2"

    # Teacher model (Qwen3-Coder-30B-A3B)
    teacher_model_id: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    teacher_dtype: str = "bfloat16"
    teacher_max_model_len: int = 8192

    # LoRA defaults
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class TrainingDefaults:
    """Default training hyperparameters."""

    # SDPO loss parameters
    learning_rate: float = 1e-4
    alpha: float = 0.5  # GJS interpolation (0.5 = symmetric JSD)
    is_clip: float = 5.0  # IS ratio clip
    max_grad_norm: float = 1.0
    kl_reg_weight: float = 0.1  # KL regularization to base policy

    # Teacher scoring
    teacher_top_k: int = 20  # Number of top logprobs from teacher

    # Optimizer
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999


@dataclass
class InfraConfig:
    """Infrastructure configuration for Modal."""

    # Training worker (student)
    student_gpu: str = "L40S"
    student_scaledown_window: int = 300  # 5 minutes
    student_timeout: int = 120  # 2 minutes per request

    # Teacher worker (vLLM)
    teacher_gpu: str = "H100"
    teacher_min_containers: int = 1
    teacher_scaledown_window: int = 600  # 10 minutes
    teacher_gpu_memory_utilization: float = 0.90

    # Volumes
    model_volume_name: str = "claas-models"
    model_cache_dir: str = "/models/hf_cache"


@dataclass
class CLaaSConfig:
    """Complete CLaaS configuration."""

    models: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingDefaults = field(default_factory=TrainingDefaults)
    infra: InfraConfig = field(default_factory=InfraConfig)


# Global default config
DEFAULT_CONFIG = CLaaSConfig()


def get_config() -> CLaaSConfig:
    """Get the current configuration."""
    return DEFAULT_CONFIG
