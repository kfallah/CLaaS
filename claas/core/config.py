"""Centralized Hydra-backed configuration for CLaaS.

All profile YAML is validated against structured dataclass schemas registered in
Hydra's ``ConfigStore``. Runtime profile selection is explicit via config name.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypeVar

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

_CONFIG_DIR = str(Path(__file__).parent / "configs")
_DEFAULT_ALLOWED_MODELS = ("Qwen/Qwen3-8B",)
_SCHEMAS_REGISTERED = False
T = TypeVar("T")

CoreConfigName = Literal["local", "modal", "tinker"]


@dataclass
class CLaaSConfig:
    """Base configuration shared by all execution modes."""

    mode: str = "local"
    feedback_log_dir: str = "./data/feedback"
    lora_root: str = "/loras"
    storage_backend: str = "local_fs"
    allowed_init_base_models: list[str] = field(
        default_factory=lambda: list(_DEFAULT_ALLOWED_MODELS),
    )


@dataclass
class LocalConfig(CLaaSConfig):
    """Configuration for local GPU execution."""

    mode: str = "local"
    storage_backend: str = "local_fs"
    vllm_base_url: str = "http://127.0.0.1:8000"
    feedback_lock_timeout_s: float = 120.0
    feedback_wake_on_failure: bool = True
    feedback_min_free_vram_gb: float = 20.0
    feedback_sleep_verify_timeout_s: float = 30.0
    feedback_drain_timeout_s: float = 30.0
    base_model_id: str = "Qwen/Qwen3-8B"
    attn_implementation: str = "flash_attention_2"


@dataclass
class ModalConfig(CLaaSConfig):
    """Configuration for Modal remote execution."""

    mode: str = "modal"
    storage_backend: str = "modal_volume"
    vllm_base_url: str = "http://127.0.0.1:8000"
    feedback_lock_timeout_s: float = 120.0
    feedback_wake_on_failure: bool = True
    feedback_min_free_vram_gb: float = 20.0
    feedback_sleep_verify_timeout_s: float = 30.0
    feedback_drain_timeout_s: float = 30.0
    hf_secret_name: str = ""


@dataclass
class TinkerConfig(CLaaSConfig):
    """Configuration for Tinker SDK execution."""

    mode: str = "tinker"
    storage_backend: str = "local_fs"
    tinker_base_model: str = "gpt-oss/GPT-OSS-120B"
    tinker_state_path: str = "~/.claas/tinker_state.json"
    vllm_base_url: str = "http://127.0.0.1:8000"


CoreConfig = LocalConfig | ModalConfig | TinkerConfig


@dataclass
class ProxyConfig:
    """Standalone configuration for the Tinker inference proxy."""

    tinker_base_model: str = "gpt-oss/GPT-OSS-120B"
    completion_cache_size: int = 100


def register_config_schemas() -> None:
    """Register all core config schemas in Hydra's ConfigStore."""
    global _SCHEMAS_REGISTERED
    if _SCHEMAS_REGISTERED:
        return

    cs = ConfigStore.instance()
    cs.store(name="_core_local_schema", node=LocalConfig)
    cs.store(name="_core_modal_schema", node=ModalConfig)
    cs.store(name="_core_tinker_schema", node=TinkerConfig)
    cs.store(name="_core_proxy_schema", node=ProxyConfig)
    _SCHEMAS_REGISTERED = True


def _compose_structured(
    *,
    config_name: str,
    schema_type: type[T],
    overrides: list[str] | None = None,
    config_dir: str | None = None,
) -> T:
    register_config_schemas()
    abs_dir = os.path.abspath(config_dir or _CONFIG_DIR)
    with initialize_config_dir(version_base=None, config_dir=abs_dir):
        cfg = compose(config_name=config_name, overrides=overrides or [])

    typed_cfg = OmegaConf.merge(OmegaConf.structured(schema_type), cfg)
    obj = OmegaConf.to_object(typed_cfg)
    if not isinstance(obj, schema_type):
        raise TypeError(
            f"Hydra compose for {config_name!r} did not produce {schema_type.__name__}",
        )
    return obj


def _normalize_runtime_fields(cfg: CoreConfig) -> CoreConfig:
    if isinstance(cfg, TinkerConfig):
        cfg.tinker_state_path = os.path.expanduser(cfg.tinker_state_path)
    return cfg


def load_core_config(
    config_name: str,
    overrides: list[str] | None = None,
    config_dir: str | None = None,
) -> CoreConfig:
    """Compose and return a typed core runtime config for a specific profile."""
    normalized = config_name.strip().lower()
    if normalized == "local":
        return _normalize_runtime_fields(
            _compose_structured(
                config_name="local",
                schema_type=LocalConfig,
                overrides=overrides,
                config_dir=config_dir,
            ),
        )
    if normalized == "modal":
        return _normalize_runtime_fields(
            _compose_structured(
                config_name="modal",
                schema_type=ModalConfig,
                overrides=overrides,
                config_dir=config_dir,
            ),
        )
    if normalized == "tinker":
        return _normalize_runtime_fields(
            _compose_structured(
                config_name="tinker",
                schema_type=TinkerConfig,
                overrides=overrides,
                config_dir=config_dir,
            ),
        )

    raise ValueError(f"Unsupported core config profile: {config_name!r}")


@lru_cache(maxsize=3)
def get_config(config_name: str) -> CoreConfig:
    """Cached explicit-profile core config accessor."""
    return load_core_config(config_name=config_name)


def load_proxy_config(
    *,
    overrides: list[str] | None = None,
    config_dir: str | None = None,
) -> ProxyConfig:
    """Compose and return the standalone proxy config."""
    return _compose_structured(
        config_name="proxy",
        schema_type=ProxyConfig,
        overrides=overrides,
        config_dir=config_dir,
    )


@lru_cache(maxsize=1)
def get_proxy_config() -> ProxyConfig:
    """Cached standalone proxy config accessor."""
    return load_proxy_config()
