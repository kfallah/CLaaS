"""Centralized configuration for CLaaS.

Configuration is resolved in three tiers (highest priority first):

1. **Environment variables** — always win (for Docker/CI overrides and secrets)
2. **YAML config** — loaded via Hydra from ``claas/core/configs/``
3. **Dataclass defaults** — hardcoded fallbacks

The YAML config is selected by ``CLAAS_CONFIG_NAME`` (default: ``"local"``).

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
# Env var helpers (with optional YAML fallback)
# ---------------------------------------------------------------------------

def _env(name: str, default: str, yaml_val: object = None) -> str:
    raw = os.environ.get(name)
    if raw is not None:
        return raw.strip()
    if yaml_val is not None:
        return str(yaml_val)
    return default


def _env_int(name: str, default: int, yaml_val: object = None) -> int:
    raw = os.environ.get(name)
    if raw is not None:
        return int(raw.strip())
    if yaml_val is not None:
        return int(str(yaml_val))
    return default


def _env_float(name: str, default: float, yaml_val: object = None) -> float:
    raw = os.environ.get(name)
    if raw is not None:
        return float(raw.strip())
    if yaml_val is not None:
        return float(str(yaml_val))
    return default


def _env_bool(name: str, default: bool, yaml_val: object = None) -> bool:
    raw = os.environ.get(name)
    if raw is not None:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if yaml_val is not None:
        if isinstance(yaml_val, bool):
            return yaml_val
        return str(yaml_val).strip().lower() in {"1", "true", "yes", "on"}
    return default


def _env_set(name: str, default: str, yaml_val: object = None) -> frozenset[str]:
    raw = os.environ.get(name)
    if raw is not None:
        return frozenset(item.strip() for item in raw.split(",") if item.strip())
    if yaml_val is not None:
        if isinstance(yaml_val, (list, tuple)):
            return frozenset(str(item).strip() for item in yaml_val if str(item).strip())
        return frozenset(
            item.strip() for item in str(yaml_val).split(",") if item.strip()
        )
    return frozenset(item.strip() for item in default.split(",") if item.strip())


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
        "feedback_log_dir": _env(
            "FEEDBACK_LOG_DIR", "./data/feedback", yaml.get("feedback_log_dir"),
        ),
        "hf_token": _env("HF_TOKEN", ""),
        "lora_root": _env("CLAAS_LORA_ROOT", "/loras", yaml.get("lora_root")),
        "storage_backend": _env(
            "CLAAS_STORAGE_BACKEND", "modal_volume", yaml.get("storage_backend"),
        ),
        "allowed_init_base_models": _env_set(
            "CLAAS_ALLOWED_INIT_BASE_MODELS",
            "Qwen/Qwen3-8B",
            yaml.get("allowed_init_base_models"),
        ),
    }


def _build_vllm_feedback_fields(yaml: dict[str, object]) -> dict[str, object]:
    return {
        "vllm_base_url": _env(
            "VLLM_BASE_URL", "http://127.0.0.1:8000", yaml.get("vllm_base_url"),
        ),
        "vllm_api_key": _env("VLLM_API_KEY", "sk-local"),
        "feedback_lock_timeout_s": _env_float(
            "FEEDBACK_LOCK_TIMEOUT_S", 120.0, yaml.get("feedback_lock_timeout_s"),
        ),
        "feedback_wake_on_failure": _env_bool(
            "FEEDBACK_WAKE_ON_FAILURE", True, yaml.get("feedback_wake_on_failure"),
        ),
        "feedback_min_free_vram_gb": _env_float(
            "FEEDBACK_MIN_FREE_VRAM_GB", 20.0, yaml.get("feedback_min_free_vram_gb"),
        ),
        "feedback_sleep_verify_timeout_s": _env_float(
            "FEEDBACK_SLEEP_VERIFY_TIMEOUT_S", 30.0,
            yaml.get("feedback_sleep_verify_timeout_s"),
        ),
        "feedback_drain_timeout_s": _env_float(
            "FEEDBACK_DRAIN_TIMEOUT_S", 30.0, yaml.get("feedback_drain_timeout_s"),
        ),
    }


@lru_cache(maxsize=1)
def get_config() -> CLaaSConfig:
    """Return the application config for the current execution mode.

    The mode is determined by ``CLAAS_CONFIG_NAME`` (default: ``"local"``),
    which selects a YAML config from ``claas/core/configs/``.

    The result is cached; call ``get_config.cache_clear()`` to re-read
    (useful in tests).
    """
    config_name = _env("CLAAS_CONFIG_NAME", "local")
    yaml = _load_yaml_config(config_name)

    mode = _env("CLAAS_CONFIG_NAME", "local").lower()
    # The YAML `mode` field is authoritative when present
    yaml_mode = yaml.get("mode")
    if yaml_mode is not None:
        mode = str(yaml_mode).lower()

    base = _build_base_fields(yaml)

    if mode == "local":
        return LocalConfig(
            mode=mode,
            **base,  # type: ignore[arg-type]
            **_build_vllm_feedback_fields(yaml),
            base_model_id=_env(
                "CLAAS_BASE_MODEL_ID", "Qwen/Qwen3-8B", yaml.get("base_model_id"),
            ),
            attn_implementation=_env(
                "CLAAS_ATTN_IMPLEMENTATION", "sdpa", yaml.get("attn_implementation"),
            ),
        )

    if mode == "modal":
        return ModalConfig(
            mode=mode,
            **base,  # type: ignore[arg-type]
            **_build_vllm_feedback_fields(yaml),
            hf_secret_name=_env("CLAAS_HF_SECRET_NAME", ""),
        )

    if mode == "tinker":
        return TinkerConfig(
            mode=mode,
            **base,  # type: ignore[arg-type]
            tinker_api_key=_env("CLAAS_TINKER_API_KEY", ""),
            tinker_base_model=_env(
                "CLAAS_TINKER_BASE_MODEL", "gpt-oss/GPT-OSS-120B",
                yaml.get("tinker_base_model"),
            ),
            tinker_state_path=_env(
                "CLAAS_TINKER_STATE_PATH",
                os.path.join(os.path.expanduser("~"), ".claas", "tinker_state.json"),
                yaml.get("tinker_state_path"),
            ),
            vllm_base_url=_env(
                "VLLM_BASE_URL", "http://127.0.0.1:8000", yaml.get("vllm_base_url"),
            ),
            vllm_api_key=_env("VLLM_API_KEY", "sk-local"),
        )

    raise ValueError(f"Unsupported CLAAS_CONFIG_NAME: {mode!r}")


@lru_cache(maxsize=1)
def get_proxy_config() -> ProxyConfig:
    """Return the standalone proxy config.

    Cached; call ``get_proxy_config.cache_clear()`` to re-read.
    """
    yaml = _load_yaml_config("proxy")
    return ProxyConfig(
        tinker_api_key=_env("CLAAS_TINKER_API_KEY", ""),
        tinker_base_model=_env(
            "CLAAS_TINKER_BASE_MODEL", "gpt-oss/GPT-OSS-120B",
            yaml.get("tinker_base_model"),
        ),
        completion_cache_size=_env_int(
            "CLAAS_COMPLETION_CACHE_SIZE", 100, yaml.get("completion_cache_size"),
        ),
    )
