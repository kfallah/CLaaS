"""Centralized configuration for CLaaS.

All environment variable reads are concentrated here so the rest of the
codebase uses typed, validated config objects instead of ad-hoc
``os.environ.get`` calls.

Usage::

    from claas.core.config import get_config, get_proxy_config

    cfg = get_config()          # CLaaSConfig subclass based on mode
    proxy_cfg = get_proxy_config()  # standalone proxy config
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

# ---------------------------------------------------------------------------
# Env var helpers
# ---------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return int(raw.strip())


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw.strip())


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_set(name: str, default: str) -> frozenset[str]:
    raw = os.environ.get(name, default)
    return frozenset(item.strip() for item in raw.split(",") if item.strip())


# ---------------------------------------------------------------------------
# Config hierarchy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CLaaSConfig:
    """Base configuration shared by all execution modes."""

    mode: str = ""
    feedback_log_dir: str = ""
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

def _build_base_fields() -> dict[str, object]:
    return {
        "feedback_log_dir": _env("FEEDBACK_LOG_DIR", "./data/feedback"),
        "hf_token": _env("HF_TOKEN", ""),
        "lora_root": _env("CLAAS_LORA_ROOT", "/loras"),
        "storage_backend": _env("CLAAS_STORAGE_BACKEND", "modal_volume"),
        "allowed_init_base_models": _env_set("CLAAS_ALLOWED_INIT_BASE_MODELS", "Qwen/Qwen3-8B"),
    }


def _build_vllm_feedback_fields() -> dict[str, object]:
    return {
        "vllm_base_url": _env("VLLM_BASE_URL", "http://127.0.0.1:8000"),
        "vllm_api_key": _env("VLLM_API_KEY", "sk-local"),
        "feedback_lock_timeout_s": _env_float("FEEDBACK_LOCK_TIMEOUT_S", 120.0),
        "feedback_wake_on_failure": _env_bool("FEEDBACK_WAKE_ON_FAILURE", True),
        "feedback_min_free_vram_gb": _env_float("FEEDBACK_MIN_FREE_VRAM_GB", 20.0),
        "feedback_sleep_verify_timeout_s": _env_float("FEEDBACK_SLEEP_VERIFY_TIMEOUT_S", 30.0),
        "feedback_drain_timeout_s": _env_float("FEEDBACK_DRAIN_TIMEOUT_S", 30.0),
    }


@lru_cache(maxsize=1)
def get_config() -> CLaaSConfig:
    """Return the application config for the current execution mode.

    The mode is determined by ``CLAAS_DISTILL_EXECUTION_MODE`` (default: ``"local"``).
    The result is cached; call ``get_config.cache_clear()`` to re-read env vars
    (useful in tests).
    """
    mode = _env("CLAAS_DISTILL_EXECUTION_MODE", "local").lower()
    base = _build_base_fields()

    if mode == "local":
        return LocalConfig(
            mode=mode,
            **base,  # type: ignore[arg-type]
            **_build_vllm_feedback_fields(),
            base_model_id=_env("CLAAS_BASE_MODEL_ID", "Qwen/Qwen3-8B"),
            attn_implementation=_env("CLAAS_ATTN_IMPLEMENTATION", "sdpa"),
        )

    if mode == "modal":
        return ModalConfig(
            mode=mode,
            **base,  # type: ignore[arg-type]
            **_build_vllm_feedback_fields(),
            hf_secret_name=_env("CLAAS_HF_SECRET_NAME", ""),
        )

    if mode == "tinker":
        return TinkerConfig(
            mode=mode,
            **base,  # type: ignore[arg-type]
            tinker_api_key=_env("CLAAS_TINKER_API_KEY", ""),
            tinker_base_model=_env("CLAAS_TINKER_BASE_MODEL", "gpt-oss/GPT-OSS-120B"),
            tinker_state_path=_env(
                "CLAAS_TINKER_STATE_PATH",
                os.path.join(os.path.expanduser("~"), ".claas", "tinker_state.json"),
            ),
            vllm_base_url=_env("VLLM_BASE_URL", "http://127.0.0.1:8000"),
            vllm_api_key=_env("VLLM_API_KEY", "sk-local"),
        )

    raise ValueError(f"Unsupported CLAAS_DISTILL_EXECUTION_MODE: {mode!r}")


@lru_cache(maxsize=1)
def get_proxy_config() -> ProxyConfig:
    """Return the standalone proxy config.

    Cached; call ``get_proxy_config.cache_clear()`` to re-read.
    """
    return ProxyConfig(
        tinker_api_key=_env("CLAAS_TINKER_API_KEY", ""),
        tinker_base_model=_env("CLAAS_TINKER_BASE_MODEL", "gpt-oss/GPT-OSS-120B"),
        completion_cache_size=_env_int("CLAAS_COMPLETION_CACHE_SIZE", 100),
    )
