"""Tests for S3 utilities."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from claas.s3_utils import (
    create_initial_lora_config,
    parse_s3_uri,
)


class TestParseS3Uri:
    """Tests for parse_s3_uri."""

    def test_valid_uri(self):
        """Parses valid S3 URI correctly."""
        bucket, prefix = parse_s3_uri("s3://my-bucket/path/to/lora/")

        assert bucket == "my-bucket"
        assert prefix == "path/to/lora/"

    def test_uri_without_trailing_slash(self):
        """Handles URI without trailing slash."""
        bucket, prefix = parse_s3_uri("s3://bucket/loras/user/model")

        assert bucket == "bucket"
        assert prefix == "loras/user/model"

    def test_root_path(self):
        """Handles root-level path."""
        bucket, prefix = parse_s3_uri("s3://bucket/")

        assert bucket == "bucket"
        assert prefix == ""

    def test_invalid_scheme(self):
        """Raises error for non-S3 URI."""
        with pytest.raises(ValueError, match="Invalid S3 URI scheme"):
            parse_s3_uri("https://bucket.s3.amazonaws.com/path")


class TestCreateInitialLoraConfig:
    """Tests for create_initial_lora_config."""

    def test_default_config(self):
        """Creates config with defaults."""
        config = create_initial_lora_config("test-model")

        assert config["base_model_name_or_path"] == "test-model"
        assert config["r"] == 16
        assert config["lora_alpha"] == 32
        assert config["peft_type"] == "LORA"
        assert config["task_type"] == "CAUSAL_LM"
        assert "q_proj" in config["target_modules"]

    def test_custom_rank(self):
        """Accepts custom LoRA rank."""
        config = create_initial_lora_config("test-model", lora_r=32)

        assert config["r"] == 32

    def test_custom_alpha(self):
        """Accepts custom LoRA alpha."""
        config = create_initial_lora_config("test-model", lora_alpha=64)

        assert config["lora_alpha"] == 64

    def test_custom_target_modules(self):
        """Accepts custom target modules."""
        modules = ["q_proj", "v_proj"]
        config = create_initial_lora_config("test-model", target_modules=modules)

        assert config["target_modules"] == modules

    def test_config_is_json_serializable(self):
        """Config can be serialized to JSON."""
        config = create_initial_lora_config("test-model")

        # Should not raise
        json_str = json.dumps(config)
        assert isinstance(json_str, str)


class TestDownloadLoraFromS3:
    """Tests for download_lora_from_s3 with mocked S3."""

    @patch("claas.s3_utils.get_s3_client")
    def test_downloads_all_files(self, mock_get_client):
        """Downloads all files in the LoRA directory."""
        # Setup mock
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "loras/user/model/adapter_config.json"},
                {"Key": "loras/user/model/adapter_model.safetensors"},
            ]
        }

        # Track download calls
        downloaded = []

        def mock_download(bucket, key, path):
            downloaded.append((bucket, key, path))
            # Create empty file
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write("{}")

        mock_client.download_file.side_effect = mock_download

        from claas.s3_utils import download_lora_from_s3

        with tempfile.TemporaryDirectory() as temp_dir:
            result = download_lora_from_s3(
                "s3://bucket/loras/user/model/",
                local_dir=temp_dir,
            )

            assert result == temp_dir
            assert len(downloaded) == 2

    @patch("claas.s3_utils.get_s3_client")
    def test_raises_on_missing_lora(self, mock_get_client):
        """Raises FileNotFoundError when LoRA doesn't exist."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.list_objects_v2.return_value = {}  # No Contents

        from claas.s3_utils import download_lora_from_s3

        with pytest.raises(FileNotFoundError):
            download_lora_from_s3("s3://bucket/nonexistent/")


class TestUploadLoraToS3:
    """Tests for upload_lora_to_s3 with mocked S3."""

    @patch("claas.s3_utils.get_s3_client")
    def test_uploads_all_files(self, mock_get_client):
        """Uploads all files from local directory."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        uploaded = []

        def mock_upload(path, bucket, key):
            uploaded.append((path, bucket, key))

        mock_client.upload_file.side_effect = mock_upload

        from claas.s3_utils import upload_lora_to_s3

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            with open(os.path.join(temp_dir, "adapter_config.json"), "w") as f:
                f.write("{}")
            with open(os.path.join(temp_dir, "adapter_model.safetensors"), "w") as f:
                f.write("data")

            result = upload_lora_to_s3(
                temp_dir,
                "s3://bucket/loras/user/model/",
                version_suffix="v1",
            )

            assert "s3://bucket/loras/user/model-v1/" in result
            assert len(uploaded) == 2

    @patch("claas.s3_utils.get_s3_client")
    def test_generates_timestamp_suffix(self, mock_get_client):
        """Auto-generates timestamp suffix when not provided."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.upload_file.side_effect = lambda *args: None

        from claas.s3_utils import upload_lora_to_s3

        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "config.json"), "w") as f:
                f.write("{}")

            result = upload_lora_to_s3(temp_dir, "s3://bucket/loras/model/")

            # Should have timestamp suffix like -20250209-123456/
            assert "s3://bucket/loras/model-" in result
            assert len(result) > len("s3://bucket/loras/model-")


class TestLoraExists:
    """Tests for lora_exists."""

    @patch("claas.s3_utils.get_s3_client")
    def test_returns_true_when_exists(self, mock_get_client):
        """Returns True when LoRA exists."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.list_objects_v2.return_value = {
            "Contents": [{"Key": "loras/user/model/config.json"}]
        }

        from claas.s3_utils import lora_exists

        assert lora_exists("s3://bucket/loras/user/model/") is True

    @patch("claas.s3_utils.get_s3_client")
    def test_returns_false_when_missing(self, mock_get_client):
        """Returns False when LoRA doesn't exist."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.list_objects_v2.return_value = {}

        from claas.s3_utils import lora_exists

        assert lora_exists("s3://bucket/nonexistent/") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
