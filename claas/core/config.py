"""Centralized configuration for CLaaS.

Configuration is resolved from two sources:

1. **YAML config** — loaded via Hydra from ``claas/core/configs/``
2. **Environment variables** — used only for secrets and the config selector

The YAML profile is selected by ``CLAAS_CONFIG_NAME`` (default: ``"local"``).
Use Hydra CLI overrides (``key=value``) to customize non-secret values.

Usage::

    from claas.core.config import get_config, get_proxy_config

    cfg = get_config()          # CLaaSConfig subclass based on mode
    proxy_cfg = get_proxy_config()  # standalone proxy config
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

logger = logging.getLogger(__name__)

_CONFIG_DIR = str(Path(__file__).parent / "configs")

# ---------------------------------------------------------------------------
# YAML loading via Hydra Compose API
# ---------------------------------------------------------------------------

def _load_yaml_config(config_name: str) -> dict[str, object]:
    """Load a YAML config via Hydra Compose API.

    Returns an empty dict if Hydra is unavailable or the config file is missing.
    """
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        abs_dir = os.path.abspath(_CONFIG_DIR)
        with initialize_config_dir(version_base=None, config_dir=abs_dir):
            cfg = compose(config_name=config_name)
        container = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(container, dict):
            return container  # type: ignore[return-value]
        return {}
    except Exception:
        logger.debug("Failed to load YAML config %r, falling back to defaults", config_name)
        return {}


# ---------------------------------------------------------------------------
# YAML value helpers
# ---------------------------------------------------------------------------

def _yaml_str(yaml: dict[str, object], key: str, default: str = "") -> str:
    val = yaml.get(key)
    return str(val) if val is not None else default


def _yaml_float(yaml: dict[str, object], key: str, default: float = 0.0) -> float:
    val = yaml.get(key)
    return float(str(val)) if val is not None else default


def _yaml_bool(yaml: dict[str, object], key: str, default: bool = False) -> bool:
    val = yaml.get(key)
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _yaml_set(
    yaml: dict[str, object], key: str, default: frozenset[str] = frozenset(),
) -> frozenset[str]:
    val = yaml.get(key)
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        return frozenset(str(item).strip() for item in val if str(item).strip())
    return frozenset(
        item.strip() for item in str(val).split(",") if item.strip()
    )


def _secret(name: str, default: str = "") -> str:
    """Read a secret from an environment variable."""
    raw = os.environ.get(name)
    return raw.strip() if raw is not None else default


# ---------------------------------------------------------------------------
# Config hierarchy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CLaaSConfig:
    """Base configuration shared by all execution modes."""

    mode: str = ""
    feedback_log_dir: str = "./data/feedback"
    hf_token: str = ""
    lora_root: str = ""
    storage_backend: str = ""
    allowed_init_base_models: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class LocalConfig(CLaaSConfig):
    """Configuration for local GPU execution."""

    vllm_base_url: str = ""
    vllm_api_key: str = ""
    feedback_lock_timeout_s: float = 120.0
    feedback_wake_on_failure: bool = True
    feedback_min_free_vram_gb: float = 20.0
    feedback_sleep_verify_timeout_s: float = 30.0
    feedback_drain_timeout_s: float = 30.0
    base_model_id: str = ""
    attn_implementation: str = ""


@dataclass(frozen=True)
class ModalConfig(CLaaSConfig):
    """Configuration for Modal remote execution."""

    vllm_base_url: str = ""
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

    tinker_api_key: str = ""
    tinker_base_model: str = ""
    tinker_state_path: str = ""
    vllm_base_url: str = ""
    vllm_api_key: str = ""


@dataclass(frozen=True)
class ProxyConfig:
    """Standalone configuration for the Tinker inference proxy."""

    tinker_api_key: str = ""
    tinker_base_model: str = ""
    completion_cache_size: int = 100


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def _build_base_fields(yaml: dict[str, object]) -> dict[str, object]:
    return {
        "feedback_log_dir": _yaml_str(yaml, "feedback_log_dir", "./data/feedback"),
        "hf_token": _secret("HF_TOKEN"),
        "lora_root": _yaml_str(yaml, "lora_root", "/loras"),
        "storage_backend": _yaml_str(yaml, "storage_backend"),
        "allowed_init_base_models": _yaml_set(yaml, "allowed_init_base_models"),
    }


def _build_vllm_feedback_fields(yaml: dict[str, object]) -> dict[str, object]:
    return {
        "vllm_base_url": _yaml_str(yaml, "vllm_base_url", "http://127.0.0.1:8000"),
        "vllm_api_key": _secret("VLLM_API_KEY"),
        "feedback_lock_timeout_s": _yaml_float(yaml, "feedback_lock_timeout_s", 120.0),
        "feedback_wake_on_failure": _yaml_bool(yaml, "feedback_wake_on_failure", True),
        "feedback_min_free_vram_gb": _yaml_float(
            yaml, "feedback_min_free_vram_gb", 20.0,
        ),
        "feedback_sleep_verify_timeout_s": _yaml_float(
            yaml, "feedback_sleep_verify_timeout_s", 30.0,
        ),
        "feedback_drain_timeout_s": _yaml_float(yaml, "feedback_drain_timeout_s", 30.0),
    }


@lru_cache(maxsize=1)
def get_config() -> CLaaSConfig:
    """Return the application config for the current execution mode.

    The mode is determined by ``CLAAS_CONFIG_NAME`` (default: ``"local"``),
    which selects a YAML config from ``claas/core/configs/``.

    The result is cached; call ``get_config.cache_clear()`` to re-read
    (useful in tests).
    """
    config_name = os.environ.get("CLAAS_CONFIG_NAME", "local").strip().lower()
    yaml = _load_yaml_config(config_name)

    mode = _yaml_str(yaml, "mode", config_name)

    base = _build_base_fields(yaml)

    if mode == "local":
        return LocalConfig(
            mode=mode,
            **base,  # type: ignore[arg-type]
            **_build_vllm_feedback_fields(yaml),
            base_model_id=_yaml_str(yaml, "base_model_id"),
            attn_implementation=_yaml_str(yaml, "attn_implementation"),
        )

    if mode == "modal":
        return ModalConfig(
            mode=mode,
            **base,  # type: ignore[arg-type]
            **_build_vllm_feedback_fields(yaml),
            hf_secret_name=_yaml_str(yaml, "hf_secret_name"),
        )

    if mode == "tinker":
        raw_state_path = _yaml_str(
            yaml, "tinker_state_path",
            os.path.join("~", ".claas", "tinker_state.json"),
        )
        return TinkerConfig(
            mode=mode,
            **base,  # type: ignore[arg-type]
            tinker_api_key=_secret("CLAAS_TINKER_API_KEY"),
            tinker_base_model=_yaml_str(yaml, "tinker_base_model"),
            tinker_state_path=os.path.expanduser(raw_state_path),
            vllm_base_url=_yaml_str(yaml, "vllm_base_url", "http://127.0.0.1:8000"),
            vllm_api_key=_secret("VLLM_API_KEY"),
        )

    raise ValueError(f"Unsupported CLAAS_CONFIG_NAME: {mode!r}")


@lru_cache(maxsize=1)
def get_proxy_config() -> ProxyConfig:
    """Return the standalone proxy config.

    Cached; call ``get_proxy_config.cache_clear()`` to re-read.
    """
    yaml = _load_yaml_config("proxy")
    return ProxyConfig(
        tinker_api_key=_secret("CLAAS_TINKER_API_KEY"),
        tinker_base_model=_yaml_str(yaml, "tinker_base_model"),
        completion_cache_size=int(
            _yaml_float(yaml, "completion_cache_size", 100),
        ),
    )
