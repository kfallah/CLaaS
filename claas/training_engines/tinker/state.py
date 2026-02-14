"""Persistent mapping of CLaaS lora_id strings to Tinker checkpoint paths.

The state is persisted as a JSON file on disk.  Writes use the atomic
tempfile-then-replace pattern (same as ``claas/storage.py``).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from typing import Any

_DEFAULT_STATE_PATH = os.environ.get(
    "CLAAS_TINKER_STATE_PATH",
    os.path.join(os.path.expanduser("~"), ".claas", "tinker_state.json"),
)


@dataclass
class LoraEntry:
    """Metadata for a single LoRA stored in Tinker."""

    tinker_path: str
    base_model: str
    rank: int
    step: int = 0


def _state_path() -> str:
    return _DEFAULT_STATE_PATH


def _read_state(path: str | None = None) -> dict[str, Any]:
    path = path or _state_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _write_state(data: dict[str, Any], path: str | None = None) -> None:
    path = path or _state_path()
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)

    fd, tmp = tempfile.mkstemp(dir=parent, prefix=".tinker_state.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(dict(sorted(data.items())), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(parent, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def get_entry(lora_id: str, path: str | None = None) -> LoraEntry | None:
    """Return the stored entry for *lora_id*, or ``None``."""
    state = _read_state(path)
    raw = state.get(lora_id)
    if raw is None or not isinstance(raw, dict):
        return None
    try:
        return LoraEntry(**raw)
    except TypeError:
        return None


def get_tinker_path(lora_id: str, path: str | None = None) -> str | None:
    """Return the ``tinker://`` checkpoint path for *lora_id*."""
    entry = get_entry(lora_id, path)
    return entry.tinker_path if entry else None


def set_tinker_path(
    lora_id: str,
    tinker_path: str,
    base_model: str,
    rank: int,
    step: int = 0,
    path: str | None = None,
) -> None:
    """Create or update the mapping for *lora_id*."""
    state = _read_state(path)
    state[lora_id] = asdict(
        LoraEntry(tinker_path=tinker_path, base_model=base_model, rank=rank, step=step)
    )
    _write_state(state, path)


def lora_exists(lora_id: str, path: str | None = None) -> bool:
    """Return ``True`` if *lora_id* is tracked in state."""
    return get_entry(lora_id, path) is not None


def list_loras(prefix: str = "", path: str | None = None) -> list[str]:
    """Return all tracked lora_id strings, optionally filtered by *prefix*."""
    state = _read_state(path)
    if not prefix:
        return sorted(state.keys())
    return sorted(k for k in state if k.startswith(prefix))
