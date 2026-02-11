"""Storage utilities for LoRA adapter persistence.

Uses Modal Volumes for simple, serverless storage without external dependencies.
LoRA adapters are stored at: /loras/{user_id}/{lora_id}/

Modal Volumes:
- Persist data across container restarts
- Accessible from all Modal functions in the same app
- No AWS credentials or external services needed
- ~100MB/s read/write throughput
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import modal

# Modal volume for LoRA storage
lora_volume = modal.Volume.from_name("claas-loras", create_if_missing=True)

# Mount path inside containers (or local filesystem root in local mode)
LORA_MOUNT_PATH = os.environ.get("CLAAS_LORA_ROOT", "/loras")
ALIASES_FILE_NAME = ".aliases.json"


def _storage_backend() -> str:
    """Return configured storage backend."""
    return os.environ.get("CLAAS_STORAGE_BACKEND", "modal_volume").strip().lower()


def _commit_storage() -> None:
    """Commit storage when using Modal Volumes.

    Local filesystem mode is a no-op commit.
    """
    if _storage_backend() == "local_fs":
        return
    lora_volume.commit()


def _aliases_file_path() -> str:
    """Get path to alias mapping file in the LoRA volume."""
    return os.path.join(LORA_MOUNT_PATH, ALIASES_FILE_NAME)


def _read_aliases() -> dict[str, str]:
    """Read alias -> lora_id mapping from disk."""
    aliases_path = _aliases_file_path()
    if not os.path.exists(aliases_path):
        return {}

    try:
        with open(aliases_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(data, dict):
        return {}

    aliases: dict[str, str] = {}
    for alias, target in data.items():
        if isinstance(alias, str) and isinstance(target, str):
            aliases[alias.strip("/")] = target.strip("/")
    return aliases


def _write_aliases(aliases: dict[str, str]) -> None:
    """Persist alias -> lora_id mapping to disk."""
    os.makedirs(LORA_MOUNT_PATH, exist_ok=True)
    aliases_path = _aliases_file_path()
    aliases_dir = os.path.dirname(aliases_path)
    fd, temp_path = tempfile.mkstemp(
        dir=aliases_dir,
        prefix=".aliases.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(dict(sorted(aliases.items())), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, aliases_path)
        dir_fd = os.open(aliases_dir, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def resolve_lora_id(lora_id: str) -> str:
    """Resolve a LoRA identifier or alias to a concrete LoRA identifier."""
    sanitized = lora_id.strip("/")
    try:
        direct_path = get_lora_path(sanitized)
        if os.path.exists(os.path.join(direct_path, "adapter_config.json")):
            return sanitized
    except ValueError:
        return sanitized

    aliases = _read_aliases()
    target = aliases.get(sanitized)
    if target:
        return target
    return sanitized


def get_lora_path(lora_id: str) -> str:
    """Get the full path to a LoRA directory.

    Args:
        lora_id: LoRA identifier (e.g., "user123/coder-v1")

    Returns:
        Full path like "/loras/user123/coder-v1"

    Raises:
        ValueError: If lora_id contains path traversal sequences
    """
    sanitized = lora_id.strip("/")

    # Check for path traversal attempts
    if ".." in sanitized.split("/"):
        raise ValueError(f"Invalid lora_id: path traversal detected in '{lora_id}'")

    full_path = os.path.normpath(os.path.join(LORA_MOUNT_PATH, sanitized))

    # Verify the resolved path is within LORA_MOUNT_PATH
    if not full_path.startswith(LORA_MOUNT_PATH + "/"):
        raise ValueError("Invalid lora_id: resolves outside storage root")

    return full_path


def lora_exists(lora_id: str) -> bool:
    """Check if a LoRA exists.

    Args:
        lora_id: LoRA identifier

    Returns:
        True if LoRA directory exists with adapter_config.json
    """
    resolved_lora_id = resolve_lora_id(lora_id)
    lora_path = get_lora_path(resolved_lora_id)
    config_path = os.path.join(lora_path, "adapter_config.json")
    return os.path.exists(config_path)


def load_lora(lora_id: str, local_dir: str | None = None) -> str:
    """Load a LoRA adapter to a local directory.

    Args:
        lora_id: LoRA identifier
        local_dir: Local directory to copy to (creates temp dir if None)

    Returns:
        Path to the local LoRA directory
    """
    resolved_lora_id = resolve_lora_id(lora_id)
    lora_path = get_lora_path(resolved_lora_id)

    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA not found: {lora_id}")

    # Create local directory
    if local_dir is None:
        local_dir = tempfile.mkdtemp(prefix="lora_")
    else:
        os.makedirs(local_dir, exist_ok=True)

    # Copy files from volume to local
    for item in os.listdir(lora_path):
        src = os.path.join(lora_path, item)
        dst = os.path.join(local_dir, item)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            shutil.copytree(src, dst)

    return local_dir


def save_lora(
    local_dir: str,
    lora_id: str,
    version_suffix: str | None = None,
    update_latest_alias: bool = True,
) -> str:
    """Save a LoRA adapter from local directory to storage.

    Args:
        local_dir: Local directory containing the LoRA files
        lora_id: Base LoRA identifier
        version_suffix: Optional version suffix (auto-generates timestamp if None)

    Returns:
        The full lora_id of the saved adapter
    """
    base_lora_id = lora_id.strip("/")
    if base_lora_id.endswith("-latest"):
        # Keep a stable base when callers send the latest alias.
        base_lora_id = base_lora_id[: -len("-latest")]

    # Add version suffix
    if version_suffix:
        full_lora_id = f"{base_lora_id}-{version_suffix}"
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        full_lora_id = f"{base_lora_id}-{timestamp}"

    lora_path = get_lora_path(full_lora_id)

    # Create directory
    os.makedirs(lora_path, exist_ok=True)

    # Copy files from local to volume
    local_path = Path(local_dir)
    for item in local_path.iterdir():
        src = str(item)
        dst = os.path.join(lora_path, item.name)
        if item.is_file():
            shutil.copy2(src, dst)
        elif item.is_dir():
            shutil.copytree(src, dst)

    if update_latest_alias:
        aliases = _read_aliases()
        aliases[f"{base_lora_id}-latest"] = full_lora_id
        _write_aliases(aliases)

    # Commit changes to the volume (LoRA files + aliases map)
    _commit_storage()

    return full_lora_id


def save_lora_inplace(local_dir: str, lora_id: str) -> str:
    """Save a LoRA adapter by replacing a fixed LoRA path in place.

    This uses a staged directory and rename swaps to avoid exposing
    partially written adapter files.
    """
    sanitized_id = lora_id.strip("/")
    resolved_id = resolve_lora_id(sanitized_id)
    target_path = get_lora_path(resolved_id)
    parent = os.path.dirname(target_path)
    os.makedirs(parent, exist_ok=True)

    stage_path = tempfile.mkdtemp(prefix=".stage_", dir=parent)
    backup_path = f"{target_path}.bak"

    try:
        # Copy files into stage directory first.
        local_path = Path(local_dir)
        for item in local_path.iterdir():
            src = str(item)
            dst = os.path.join(stage_path, item.name)
            if item.is_file():
                shutil.copy2(src, dst)
            elif item.is_dir():
                shutil.copytree(src, dst)

        config_path = os.path.join(stage_path, "adapter_config.json")
        if not os.path.exists(config_path):
            raise ValueError("adapter_config.json is required for LoRA save")

        # Clear old backup if present from a prior interrupted run.
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)

        had_existing = os.path.exists(target_path)
        if had_existing:
            os.replace(target_path, backup_path)

        try:
            os.replace(stage_path, target_path)
        except OSError:
            # Roll back prior adapter on swap failure.
            if had_existing and os.path.exists(backup_path):
                os.replace(backup_path, target_path)
            raise

        if had_existing and os.path.exists(backup_path):
            shutil.rmtree(backup_path)
    finally:
        if os.path.exists(stage_path):
            shutil.rmtree(stage_path)

    _commit_storage()
    return resolved_id


def create_initial_lora(
    lora_id: str,
    base_model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    target_modules: list[str] | None = None,
) -> str:
    """Create a new LoRA adapter with initial configuration.

    Args:
        lora_id: LoRA identifier
        base_model_name: Name/path of the base model
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        target_modules: List of module names to apply LoRA to

    Returns:
        The lora_id of the created adapter
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    config = {
        "base_model_name_or_path": base_model_name,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": False,
        "init_lora_weights": True,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": lora_r,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
    }

    # Create temp directory with config
    with tempfile.TemporaryDirectory(prefix="lora_init_") as temp_dir:
        config_path = os.path.join(temp_dir, "adapter_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Save to storage
        full_lora_id = save_lora(temp_dir, lora_id, version_suffix="init")

    return full_lora_id


def delete_lora(lora_id: str) -> bool:
    """Delete a LoRA adapter.

    Args:
        lora_id: LoRA identifier

    Returns:
        True if deleted, False if not found
    """
    aliases = _read_aliases()
    sanitized = lora_id.strip("/")

    # Deleting an alias only removes the alias mapping.
    if sanitized in aliases:
        aliases.pop(sanitized, None)
        _write_aliases(aliases)
        _commit_storage()
        return True

    resolved_lora_id = resolve_lora_id(lora_id)
    lora_path = get_lora_path(resolved_lora_id)

    if not os.path.exists(lora_path):
        return False

    shutil.rmtree(lora_path)

    # Remove aliases pointing to this LoRA.
    dangling_aliases = [alias for alias, target in aliases.items() if target == resolved_lora_id]
    for alias in dangling_aliases:
        aliases.pop(alias, None)
    _write_aliases(aliases)

    _commit_storage()
    return True


def list_loras(prefix: str = "") -> list[str]:
    """List all LoRA adapters.

    Args:
        prefix: Optional prefix to filter by (e.g., "user123/")

    Returns:
        List of lora_ids
    """
    loras = []
    base_path = LORA_MOUNT_PATH

    if not os.path.exists(base_path):
        return loras

    # Sanitize prefix to prevent path traversal
    sanitized_prefix = os.path.normpath(prefix.strip("/"))
    if sanitized_prefix == ".":
        sanitized_prefix = ""
    elif ".." in sanitized_prefix.split("/"):
        # Reject prefixes with path traversal
        return loras

    for root, _dirs, files in os.walk(base_path):
        if "adapter_config.json" in files:
            # This is a LoRA directory
            rel_path = os.path.relpath(root, base_path)
            if not sanitized_prefix or rel_path.startswith(sanitized_prefix):
                loras.append(rel_path)

    aliases = _read_aliases()
    for alias, target in aliases.items():
        if lora_exists(target):
            if not sanitized_prefix or alias.startswith(sanitized_prefix):
                loras.append(alias)

    return sorted(set(loras))


def export_lora_zip_bytes(lora_id: str) -> bytes:
    """Export a LoRA adapter directory as a zip archive.

    Args:
        lora_id: LoRA identifier

    Returns:
        Zip archive bytes containing adapter files.
    """
    resolved_lora_id = resolve_lora_id(lora_id)
    lora_path = get_lora_path(resolved_lora_id)
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA not found: {lora_id}")

    with tempfile.NamedTemporaryFile(suffix=".zip") as temp_zip:
        with zipfile.ZipFile(temp_zip.name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _dirs, files in os.walk(lora_path):
                for file_name in files:
                    src = os.path.join(root, file_name)
                    arcname = os.path.relpath(src, lora_path)
                    zf.write(src, arcname=arcname)
        with open(temp_zip.name, "rb") as f:
            return f.read()


def cleanup_local_lora(local_dir: str) -> None:
    """Remove local LoRA directory after use.

    Args:
        local_dir: Local directory to remove
    """
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
