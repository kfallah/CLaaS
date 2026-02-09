"""S3 utilities for LoRA adapter storage and retrieval.

LoRA adapters are stored in S3 with the following structure:
    s3://bucket/loras/{user_id}/{lora_id}/
        adapter_config.json
        adapter_model.safetensors

Each distill request specifies a lora_uri like:
    s3://my-bucket/loras/user123/coder-v1/

After training, the updated LoRA is uploaded back to S3 with a new version suffix.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config


def get_s3_client():
    """Get configured S3 client."""
    config = Config(
        retries={"max_attempts": 3, "mode": "adaptive"},
        connect_timeout=5,
        read_timeout=30,
    )
    return boto3.client("s3", config=config)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse S3 URI into bucket and key prefix.

    Args:
        uri: S3 URI like 's3://bucket/path/to/lora/'

    Returns:
        Tuple of (bucket, key_prefix)
    """
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Invalid S3 URI scheme: {parsed.scheme}")
    bucket = parsed.netloc
    key_prefix = parsed.path.lstrip("/")
    return bucket, key_prefix


def download_lora_from_s3(
    lora_uri: str,
    local_dir: str | None = None,
) -> str:
    """Download LoRA adapter from S3 to local directory.

    Args:
        lora_uri: S3 URI of the LoRA adapter
        local_dir: Local directory to download to (creates temp dir if None)

    Returns:
        Path to the local LoRA directory
    """
    s3 = get_s3_client()
    bucket, key_prefix = parse_s3_uri(lora_uri)

    # Create local directory
    if local_dir is None:
        local_dir = tempfile.mkdtemp(prefix="lora_")
    else:
        os.makedirs(local_dir, exist_ok=True)

    # List and download all files in the LoRA directory
    response = s3.list_objects_v2(Bucket=bucket, Prefix=key_prefix)

    if "Contents" not in response:
        raise FileNotFoundError(f"No LoRA found at {lora_uri}")

    for obj in response["Contents"]:
        key = obj["Key"]
        # Get relative path from prefix
        rel_path = key[len(key_prefix) :].lstrip("/")
        if not rel_path:
            continue

        local_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        s3.download_file(bucket, key, local_path)

    return local_dir


def upload_lora_to_s3(
    local_dir: str,
    base_uri: str,
    version_suffix: str | None = None,
) -> str:
    """Upload LoRA adapter to S3.

    Args:
        local_dir: Local directory containing the LoRA adapter files
        base_uri: Base S3 URI for the LoRA
        version_suffix: Optional version suffix (auto-generates timestamp if None)

    Returns:
        Full S3 URI of the uploaded LoRA
    """
    s3 = get_s3_client()
    bucket, key_prefix = parse_s3_uri(base_uri)

    # Add version suffix if provided or generate timestamp
    if version_suffix:
        key_prefix = key_prefix.rstrip("/") + f"-{version_suffix}/"
    else:
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        key_prefix = key_prefix.rstrip("/") + f"-{timestamp}/"

    # Upload all files
    local_path = Path(local_dir)
    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(local_path)
            s3_key = f"{key_prefix}{rel_path}"
            s3.upload_file(str(file_path), bucket, s3_key)

    return f"s3://{bucket}/{key_prefix}"


def copy_lora_in_s3(
    source_uri: str,
    dest_uri: str,
) -> str:
    """Copy a LoRA from one S3 location to another.

    Args:
        source_uri: Source S3 URI
        dest_uri: Destination S3 URI

    Returns:
        The destination URI
    """
    s3 = get_s3_client()
    src_bucket, src_prefix = parse_s3_uri(source_uri)
    dst_bucket, dst_prefix = parse_s3_uri(dest_uri)

    # List source files
    response = s3.list_objects_v2(Bucket=src_bucket, Prefix=src_prefix)

    if "Contents" not in response:
        raise FileNotFoundError(f"No LoRA found at {source_uri}")

    for obj in response["Contents"]:
        src_key = obj["Key"]
        rel_path = src_key[len(src_prefix) :].lstrip("/")
        if not rel_path:
            continue

        dst_key = f"{dst_prefix}{rel_path}"
        copy_source = {"Bucket": src_bucket, "Key": src_key}
        s3.copy_object(CopySource=copy_source, Bucket=dst_bucket, Key=dst_key)

    return dest_uri


def lora_exists(lora_uri: str) -> bool:
    """Check if a LoRA exists at the given S3 URI.

    Args:
        lora_uri: S3 URI to check

    Returns:
        True if LoRA exists
    """
    s3 = get_s3_client()
    bucket, key_prefix = parse_s3_uri(lora_uri)

    response = s3.list_objects_v2(Bucket=bucket, Prefix=key_prefix, MaxKeys=1)
    return "Contents" in response and len(response["Contents"]) > 0


def create_initial_lora_config(
    base_model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    target_modules: list[str] | None = None,
) -> dict:
    """Create initial LoRA adapter configuration.

    Args:
        base_model_name: Name/path of the base model
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        target_modules: List of module names to apply LoRA to

    Returns:
        LoRA configuration dict compatible with PEFT
    """
    if target_modules is None:
        # Default target modules for Qwen models
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    return {
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


def initialize_lora_from_base(
    base_model_name: str,
    output_uri: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    target_modules: list[str] | None = None,
) -> str:
    """Initialize a new LoRA adapter with zero weights.

    This creates the adapter_config.json and a minimal adapter that can be
    loaded with PEFT. The actual adapter weights will be initialized when
    first loaded in training.

    Args:
        base_model_name: Name/path of the base model
        output_uri: S3 URI to save the LoRA to
        lora_r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        target_modules: List of module names to apply LoRA to

    Returns:
        S3 URI of the created LoRA
    """
    config = create_initial_lora_config(
        base_model_name=base_model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
    )

    # Create temp directory with config
    with tempfile.TemporaryDirectory(prefix="lora_init_") as temp_dir:
        config_path = os.path.join(temp_dir, "adapter_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Upload to S3
        uri = upload_lora_to_s3(temp_dir, output_uri, version_suffix="init")

    return uri


def cleanup_local_lora(local_dir: str) -> None:
    """Remove local LoRA directory after upload.

    Args:
        local_dir: Local directory to remove
    """
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
