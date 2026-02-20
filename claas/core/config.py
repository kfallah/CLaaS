"""Centralized Hydra-backed configuration for CLaaS.

All profile YAML is validated against structured dataclass schemas registered in
Hydra's ``ConfigStore``. Runtime profile selection is explicit via config name.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

_CONFIG_DIR = str(Path(__file__).parent / "configs")
_SCHEMAS_REGISTERED = False


@dataclass
class CLaaSConfig:
    """Base configuration shared by all execution modes."""

    mode: str = "local"
    feedback_log_dir: str = "./data/feedback"
    lora_root: str = "/loras"
    storage_backend: str = "local_fs"
    allowed_init_base_models: list[str] = field(default_factory=list)


@dataclass
class LocalConfig(CLaaSConfig):
    """Configuration for local GPU execution."""

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
    tinker_base_model: str = "Qwen/Qwen3-30B-A3B"
    tinker_state_path: str = "~/.claas/tinker_state.json"
    vllm_base_url: str = "http://127.0.0.1:8000"


CoreConfig = LocalConfig | ModalConfig | TinkerConfig


@dataclass
class ProxyConfig:
    """Standalone configuration for the Tinker inference proxy."""

    tinker_base_model: str = "Qwen/Qwen3-30B-A3B"
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


def load_core_config(
    config_name: str,
    overrides: list[str] | None = None,
    config_dir: str | None = None,
) -> CoreConfig:
    """Compose and return a typed core runtime config for a specific profile."""
    normalized = config_name.strip().lower()
    if normalized not in {"local", "modal", "tinker"}:
        raise ValueError(f"Unsupported core config profile: {config_name!r}")

    register_config_schemas()
    abs_dir = os.path.abspath(config_dir or _CONFIG_DIR)
    with initialize_config_dir(version_base=None, config_dir=abs_dir):
        cfg = compose(config_name=normalized, overrides=overrides or [])

    if normalized == "local":
        typed_cfg = OmegaConf.merge(OmegaConf.structured(LocalConfig), cfg)
        obj = OmegaConf.to_object(typed_cfg)
        if not isinstance(obj, LocalConfig):
            raise TypeError("Hydra compose for 'local' did not produce LocalConfig")
        return obj

    if normalized == "modal":
        typed_cfg = OmegaConf.merge(OmegaConf.structured(ModalConfig), cfg)
        obj = OmegaConf.to_object(typed_cfg)
        if not isinstance(obj, ModalConfig):
            raise TypeError("Hydra compose for 'modal' did not produce ModalConfig")
        return obj

    typed_cfg = OmegaConf.merge(OmegaConf.structured(TinkerConfig), cfg)
    obj = OmegaConf.to_object(typed_cfg)
    if not isinstance(obj, TinkerConfig):
        raise TypeError("Hydra compose for 'tinker' did not produce TinkerConfig")
    obj.tinker_state_path = os.path.expanduser(obj.tinker_state_path)
    return obj


def load_proxy_config(
    *,
    overrides: list[str] | None = None,
    config_dir: str | None = None,
) -> ProxyConfig:
    """Compose and return the standalone proxy config."""
    register_config_schemas()
    abs_dir = os.path.abspath(config_dir or _CONFIG_DIR)
    all_overrides = list(overrides or [])
    base_model_env = os.environ.get("CLAAS_TINKER_BASE_MODEL")
    if base_model_env:
        all_overrides.append(f"tinker_base_model={base_model_env}")
    with initialize_config_dir(version_base=None, config_dir=abs_dir):
        cfg = compose(config_name="proxy", overrides=all_overrides)

    typed_cfg = OmegaConf.merge(OmegaConf.structured(ProxyConfig), cfg)
    obj = OmegaConf.to_object(typed_cfg)
    if not isinstance(obj, ProxyConfig):
        raise TypeError("Hydra compose for 'proxy' did not produce ProxyConfig")
    return obj
