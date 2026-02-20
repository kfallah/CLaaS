"""Centralized Hydra-backed configuration for CLaaS.

Configuration is resolved from:
1. YAML profiles in ``claas/core/configs/``
2. Secrets in environment variables

All YAML is validated against structured schemas registered in Hydra's
``ConfigStore``. Invalid or unknown keys fail fast during composition.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

_CONFIG_DIR = str(Path(__file__).parent / "configs")

_DEFAULT_ALLOWED_MODELS = ("Qwen/Qwen3-8B",)
_SCHEMAS_REGISTERED = False
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Runtime config dataclasses (consumed by app code)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CLaaSConfig:
    """Base configuration shared by all execution modes."""

    mode: str = "local"
    feedback_log_dir: str = "./data/feedback"
    hf_token: str = ""
    lora_root: str = "/loras"
    storage_backend: str = "local_fs"
    allowed_init_base_models: frozenset[str] = field(
        default_factory=lambda: frozenset(_DEFAULT_ALLOWED_MODELS),
    )


@dataclass(frozen=True)
class LocalConfig(CLaaSConfig):
    """Configuration for local GPU execution."""

    mode: str = "local"
    storage_backend: str = "local_fs"
    vllm_base_url: str = "http://127.0.0.1:8000"
    vllm_api_key: str = ""
    feedback_lock_timeout_s: float = 120.0
    feedback_wake_on_failure: bool = True
    feedback_min_free_vram_gb: float = 20.0
    feedback_sleep_verify_timeout_s: float = 30.0
    feedback_drain_timeout_s: float = 30.0
    base_model_id: str = "Qwen/Qwen3-8B"
    attn_implementation: str = "flash_attention_2"


@dataclass(frozen=True)
class ModalConfig(CLaaSConfig):
    """Configuration for Modal remote execution."""

    mode: str = "modal"
    storage_backend: str = "modal_volume"
    vllm_base_url: str = "http://127.0.0.1:8000"
    vllm_api_key: str = ""
    feedback_lock_timeout_s: float = 120.0
    feedback_wake_on_failure: bool = True
    feedback_min_free_vram_gb: float = 20.0
    feedback_sleep_verify_timeout_s: float = 30.0
    feedback_drain_timeout_s: float = 30.0
    hf_secret_name: str = ""


@dataclass(frozen=True)
class TinkerConfig(CLaaSConfig):
    """Configuration for Tinker SDK execution."""

    mode: str = "tinker"
    storage_backend: str = "local_fs"
    tinker_api_key: str = ""
    tinker_base_model: str = "gpt-oss/GPT-OSS-120B"
    tinker_state_path: str = "~/.claas/tinker_state.json"
    vllm_base_url: str = "http://127.0.0.1:8000"
    vllm_api_key: str = ""


@dataclass(frozen=True)
class ProxyConfig:
    """Standalone configuration for the Tinker inference proxy."""

    tinker_api_key: str = ""
    tinker_base_model: str = "gpt-oss/GPT-OSS-120B"
    completion_cache_size: int = 100


# ---------------------------------------------------------------------------
# Structured schema dataclasses (registered in Hydra ConfigStore)
# ---------------------------------------------------------------------------

@dataclass
class CLaaSConfigSchema:
    mode: str = "local"
    feedback_log_dir: str = "./data/feedback"
    lora_root: str = "/loras"
    storage_backend: str = "local_fs"
    allowed_init_base_models: list[str] = field(
        default_factory=lambda: list(_DEFAULT_ALLOWED_MODELS),
    )


@dataclass
class LocalConfigSchema(CLaaSConfigSchema):
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
class ModalConfigSchema(CLaaSConfigSchema):
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
class TinkerConfigSchema(CLaaSConfigSchema):
    mode: str = "tinker"
    storage_backend: str = "local_fs"
    tinker_base_model: str = "gpt-oss/GPT-OSS-120B"
    tinker_state_path: str = "~/.claas/tinker_state.json"
    vllm_base_url: str = "http://127.0.0.1:8000"


@dataclass
class ProxyConfigSchema:
    tinker_base_model: str = "gpt-oss/GPT-OSS-120B"
    completion_cache_size: int = 100


# ---------------------------------------------------------------------------
# Hydra helpers
# ---------------------------------------------------------------------------

def register_config_schemas() -> None:
    """Register all core config schemas in Hydra's ConfigStore."""
    global _SCHEMAS_REGISTERED
    if _SCHEMAS_REGISTERED:
        return

    cs = ConfigStore.instance()
    cs.store(name="_core_local_schema", node=LocalConfigSchema)
    cs.store(name="_core_modal_schema", node=ModalConfigSchema)
    cs.store(name="_core_tinker_schema", node=TinkerConfigSchema)
    cs.store(name="_core_proxy_schema", node=ProxyConfigSchema)
    _SCHEMAS_REGISTERED = True


def _compose_structured(config_name: str, schema_type: type[T]) -> T:
    register_config_schemas()
    abs_dir = os.path.abspath(_CONFIG_DIR)
    with initialize_config_dir(version_base=None, config_dir=abs_dir):
        cfg = compose(config_name=config_name)

    typed_cfg = OmegaConf.merge(OmegaConf.structured(schema_type), cfg)
    obj = OmegaConf.to_object(typed_cfg)
    if not isinstance(obj, schema_type):
        raise TypeError(
            f"Hydra compose for {config_name!r} did not produce {schema_type.__name__}",
        )
    return obj


def _secret(name: str, default: str = "") -> str:
    raw = os.environ.get(name)
    return raw.strip() if raw is not None else default


def _base_runtime_fields(schema: CLaaSConfigSchema) -> dict[str, object]:
    return {
        "mode": schema.mode,
        "feedback_log_dir": schema.feedback_log_dir,
        "hf_token": _secret("HF_TOKEN"),
        "lora_root": schema.lora_root,
        "storage_backend": schema.storage_backend,
        "allowed_init_base_models": frozenset(schema.allowed_init_base_models),
    }


def _load_yaml_config(config_name: str) -> dict[str, object]:
    """Compose a core YAML profile into a schema-validated dict."""
    if config_name == "local":
        return asdict(_compose_structured(config_name, LocalConfigSchema))
    if config_name == "modal":
        return asdict(_compose_structured(config_name, ModalConfigSchema))
    if config_name == "tinker":
        return asdict(_compose_structured(config_name, TinkerConfigSchema))
    if config_name == "proxy":
        return asdict(_compose_structured(config_name, ProxyConfigSchema))
    raise ValueError(f"Unsupported config name: {config_name!r}")


@lru_cache(maxsize=1)
def get_config() -> CLaaSConfig:
    """Return the application runtime config selected by ``CLAAS_CONFIG_NAME``."""
    config_name = os.environ.get("CLAAS_CONFIG_NAME", "local").strip().lower()
    if config_name == "local":
        schema = _compose_structured("local", LocalConfigSchema)
        return LocalConfig(
            **_base_runtime_fields(schema),  # type: ignore[arg-type]
            vllm_base_url=schema.vllm_base_url,
            vllm_api_key=_secret("VLLM_API_KEY"),
            feedback_lock_timeout_s=schema.feedback_lock_timeout_s,
            feedback_wake_on_failure=schema.feedback_wake_on_failure,
            feedback_min_free_vram_gb=schema.feedback_min_free_vram_gb,
            feedback_sleep_verify_timeout_s=schema.feedback_sleep_verify_timeout_s,
            feedback_drain_timeout_s=schema.feedback_drain_timeout_s,
            base_model_id=schema.base_model_id,
            attn_implementation=schema.attn_implementation,
        )

    if config_name == "modal":
        schema = _compose_structured("modal", ModalConfigSchema)
        return ModalConfig(
            **_base_runtime_fields(schema),  # type: ignore[arg-type]
            vllm_base_url=schema.vllm_base_url,
            vllm_api_key=_secret("VLLM_API_KEY"),
            feedback_lock_timeout_s=schema.feedback_lock_timeout_s,
            feedback_wake_on_failure=schema.feedback_wake_on_failure,
            feedback_min_free_vram_gb=schema.feedback_min_free_vram_gb,
            feedback_sleep_verify_timeout_s=schema.feedback_sleep_verify_timeout_s,
            feedback_drain_timeout_s=schema.feedback_drain_timeout_s,
            hf_secret_name=schema.hf_secret_name,
        )

    if config_name == "tinker":
        schema = _compose_structured("tinker", TinkerConfigSchema)
        return TinkerConfig(
            **_base_runtime_fields(schema),  # type: ignore[arg-type]
            tinker_api_key=_secret("CLAAS_TINKER_API_KEY"),
            tinker_base_model=schema.tinker_base_model,
            tinker_state_path=os.path.expanduser(schema.tinker_state_path),
            vllm_base_url=schema.vllm_base_url,
            vllm_api_key=_secret("VLLM_API_KEY"),
        )

    raise ValueError(f"Unsupported CLAAS_CONFIG_NAME: {config_name!r}")


@lru_cache(maxsize=1)
def get_proxy_config() -> ProxyConfig:
    """Return the standalone proxy config."""
    schema = _compose_structured("proxy", ProxyConfigSchema)
    return ProxyConfig(
        tinker_api_key=_secret("CLAAS_TINKER_API_KEY"),
        tinker_base_model=schema.tinker_base_model,
        completion_cache_size=schema.completion_cache_size,
    )
