"""Tests for Modal Volume storage utilities."""

from __future__ import annotations

import json

import pytest


class TestGetLoraPath:
    """Tests for get_lora_path."""

    def test_simple_path(self):
        """Returns correct path for simple ID."""
        from claas.storage import LORA_MOUNT_PATH, get_lora_path

        result = get_lora_path("user123/coder-v1")
        assert result == f"{LORA_MOUNT_PATH}/user123/coder-v1"

    def test_strips_slashes(self):
        """Strips leading/trailing slashes."""
        from claas.storage import LORA_MOUNT_PATH, get_lora_path

        result = get_lora_path("/user123/coder-v1/")
        assert result == f"{LORA_MOUNT_PATH}/user123/coder-v1"

    def test_nested_path(self):
        """Handles nested paths."""
        from claas.storage import LORA_MOUNT_PATH, get_lora_path

        result = get_lora_path("org/team/project/model-v1")
        assert result == f"{LORA_MOUNT_PATH}/org/team/project/model-v1"


class TestLoraExists:
    """Tests for lora_exists (requires mocking filesystem)."""

    def test_exists_when_config_present(self, tmp_path, monkeypatch):
        """Returns True when adapter_config.json exists."""
        from claas import storage

        # Create mock LoRA directory
        lora_dir = tmp_path / "test-user" / "test-model"
        lora_dir.mkdir(parents=True)
        (lora_dir / "adapter_config.json").write_text("{}")

        # Monkeypatch the mount path
        monkeypatch.setattr(storage, "LORA_MOUNT_PATH", str(tmp_path))

        assert storage.lora_exists("test-user/test-model") is True

    def test_not_exists_when_no_config(self, tmp_path, monkeypatch):
        """Returns False when adapter_config.json is missing."""
        from claas import storage

        # Create empty directory
        lora_dir = tmp_path / "test-user" / "test-model"
        lora_dir.mkdir(parents=True)

        monkeypatch.setattr(storage, "LORA_MOUNT_PATH", str(tmp_path))

        assert storage.lora_exists("test-user/test-model") is False

    def test_not_exists_when_no_directory(self, tmp_path, monkeypatch):
        """Returns False when directory doesn't exist."""
        from claas import storage

        monkeypatch.setattr(storage, "LORA_MOUNT_PATH", str(tmp_path))

        assert storage.lora_exists("nonexistent/model") is False


class TestLoadLora:
    """Tests for load_lora."""

    def test_copies_files_to_local(self, tmp_path, monkeypatch):
        """Copies all files from volume to local directory."""
        from claas import storage

        # Create mock LoRA in "volume"
        volume_path = tmp_path / "volume"
        lora_dir = volume_path / "user" / "model"
        lora_dir.mkdir(parents=True)
        (lora_dir / "adapter_config.json").write_text('{"r": 16}')
        (lora_dir / "adapter_model.safetensors").write_bytes(b"weights")

        monkeypatch.setattr(storage, "LORA_MOUNT_PATH", str(volume_path))

        # Load to local
        local_dir = tmp_path / "local"
        result = storage.load_lora("user/model", str(local_dir))

        assert result == str(local_dir)
        assert (local_dir / "adapter_config.json").exists()
        assert (local_dir / "adapter_model.safetensors").exists()
        assert json.loads((local_dir / "adapter_config.json").read_text()) == {"r": 16}

    def test_raises_when_not_found(self, tmp_path, monkeypatch):
        """Raises FileNotFoundError when LoRA doesn't exist."""
        from claas import storage

        monkeypatch.setattr(storage, "LORA_MOUNT_PATH", str(tmp_path))

        with pytest.raises(FileNotFoundError):
            storage.load_lora("nonexistent/model")


class TestSaveLora:
    """Tests for save_lora."""

    def test_copies_files_to_volume(self, tmp_path, monkeypatch):
        """Copies files from local to volume with version suffix."""
        from claas import storage

        # Mock volume commit
        class MockVolume:
            def commit(self):
                pass

        monkeypatch.setattr(storage, "lora_volume", MockVolume())

        volume_path = tmp_path / "volume"
        volume_path.mkdir()
        monkeypatch.setattr(storage, "LORA_MOUNT_PATH", str(volume_path))

        # Create local LoRA
        local_dir = tmp_path / "local"
        local_dir.mkdir()
        (local_dir / "adapter_config.json").write_text('{"r": 16}')

        result = storage.save_lora(str(local_dir), "user/model", version_suffix="v1")

        assert result == "user/model-v1"
        assert (volume_path / "user" / "model-v1" / "adapter_config.json").exists()

    def test_auto_generates_timestamp(self, tmp_path, monkeypatch):
        """Auto-generates timestamp suffix when not provided."""
        from claas import storage

        class MockVolume:
            def commit(self):
                pass

        monkeypatch.setattr(storage, "lora_volume", MockVolume())

        volume_path = tmp_path / "volume"
        volume_path.mkdir()
        monkeypatch.setattr(storage, "LORA_MOUNT_PATH", str(volume_path))

        local_dir = tmp_path / "local"
        local_dir.mkdir()
        (local_dir / "config.json").write_text("{}")

        result = storage.save_lora(str(local_dir), "user/model")

        # Should have timestamp format like user/model-20250209-123456
        assert result.startswith("user/model-")
        assert len(result) > len("user/model-")

    def test_inplace_overwrites_fixed_path(self, tmp_path, monkeypatch):
        """In-place save keeps a stable lora_id and replaces adapter files."""
        from claas import storage

        class MockVolume:
            def commit(self):
                pass

        monkeypatch.setattr(storage, "lora_volume", MockVolume())

        volume_path = tmp_path / "volume"
        volume_path.mkdir()
        monkeypatch.setattr(storage, "LORA_MOUNT_PATH", str(volume_path))

        initial = tmp_path / "initial"
        initial.mkdir()
        (initial / "adapter_config.json").write_text('{"r": 16}')
        (initial / "weights.txt").write_text("old")
        storage.save_lora_inplace(str(initial), "user/model")

        updated = tmp_path / "updated"
        updated.mkdir()
        (updated / "adapter_config.json").write_text('{"r": 16}')
        (updated / "weights.txt").write_text("new")
        result = storage.save_lora_inplace(str(updated), "user/model")

        assert result == "user/model"
        fixed_path = volume_path / "user" / "model"
        assert fixed_path.exists()
        assert (fixed_path / "weights.txt").read_text() == "new"
        assert not (volume_path / "user" / "model.bak").exists()


class TestCreateInitialLora:
    """Tests for create_initial_lora."""

    def test_creates_config_file(self, tmp_path, monkeypatch):
        """Creates adapter_config.json with correct content."""
        from claas import storage

        class MockVolume:
            def commit(self):
                pass

        monkeypatch.setattr(storage, "lora_volume", MockVolume())

        volume_path = tmp_path / "volume"
        volume_path.mkdir()
        monkeypatch.setattr(storage, "LORA_MOUNT_PATH", str(volume_path))

        result = storage.create_initial_lora(
            lora_id="user/model",
            base_model_name="test-model",
            lora_r=32,
            lora_alpha=64,
        )

        assert "user/model-init" in result

        # Find the created config
        config_path = volume_path / "user" / "model-init" / "adapter_config.json"
        assert config_path.exists()

        config = json.loads(config_path.read_text())
        assert config["r"] == 32
        assert config["lora_alpha"] == 64
        assert config["base_model_name_or_path"] == "test-model"
        assert config["peft_type"] == "LORA"


class TestListLoras:
    """Tests for list_loras."""

    def test_finds_all_loras(self, tmp_path, monkeypatch):
        """Finds all directories with adapter_config.json."""
        from claas import storage

        monkeypatch.setattr(storage, "LORA_MOUNT_PATH", str(tmp_path))

        # Create multiple LoRAs
        for name in ["user1/model-a", "user1/model-b", "user2/model-c"]:
            path = tmp_path / name
            path.mkdir(parents=True)
            (path / "adapter_config.json").write_text("{}")

        result = storage.list_loras()

        assert len(result) == 3
        assert "user1/model-a" in result
        assert "user1/model-b" in result
        assert "user2/model-c" in result

    def test_filters_by_prefix(self, tmp_path, monkeypatch):
        """Filters results by prefix."""
        from claas import storage

        monkeypatch.setattr(storage, "LORA_MOUNT_PATH", str(tmp_path))

        for name in ["user1/model-a", "user1/model-b", "user2/model-c"]:
            path = tmp_path / name
            path.mkdir(parents=True)
            (path / "adapter_config.json").write_text("{}")

        result = storage.list_loras(prefix="user1/")

        assert len(result) == 2
        assert all(r.startswith("user1/") for r in result)

    def test_returns_empty_when_no_loras(self, tmp_path, monkeypatch):
        """Returns empty list when no LoRAs exist."""
        from claas import storage

        monkeypatch.setattr(storage, "LORA_MOUNT_PATH", str(tmp_path))

        result = storage.list_loras()
        assert result == []


class TestCleanupLocalLora:
    """Tests for cleanup_local_lora."""

    def test_removes_directory(self, tmp_path):
        """Removes the specified directory."""
        from claas.storage import cleanup_local_lora

        local_dir = tmp_path / "lora"
        local_dir.mkdir()
        (local_dir / "config.json").write_text("{}")

        cleanup_local_lora(str(local_dir))

        assert not local_dir.exists()


class TestLoraAliases:
    """Tests for latest alias behavior."""

    def test_save_lora_creates_latest_alias(self, tmp_path, monkeypatch):
        """Saving a LoRA should create/update a latest alias."""
        from claas import storage

        class MockVolume:
            def commit(self):
                pass

        monkeypatch.setattr(storage, "lora_volume", MockVolume())
        monkeypatch.setattr(storage, "LORA_MOUNT_PATH", str(tmp_path))

        local_dir = tmp_path / "local"
        local_dir.mkdir()
        (local_dir / "adapter_config.json").write_text("{}")

        saved_id = storage.save_lora(str(local_dir), "user/model", version_suffix="v1")
        assert saved_id == "user/model-v1"

        aliases = storage._read_aliases()
        assert aliases["user/model-latest"] == "user/model-v1"

    def test_alias_resolves_for_exists_and_export(self, tmp_path, monkeypatch):
        """Alias IDs should resolve for existence and export operations."""
        from claas import storage

        class MockVolume:
            def commit(self):
                pass

        monkeypatch.setattr(storage, "lora_volume", MockVolume())
        monkeypatch.setattr(storage, "LORA_MOUNT_PATH", str(tmp_path))

        lora_dir = tmp_path / "user" / "model-v2"
        lora_dir.mkdir(parents=True)
        (lora_dir / "adapter_config.json").write_text("{}")

        storage._write_aliases({"user/model-latest": "user/model-v2"})

        assert storage.lora_exists("user/model-latest") is True
        exported = storage.export_lora_zip_bytes("user/model-latest")
        assert isinstance(exported, bytes)
        assert len(exported) > 0

    def test_delete_lora_alias_only_removes_mapping(self, tmp_path, monkeypatch):
        """Deleting an alias should not delete target LoRA files."""
        from claas import storage

        class MockVolume:
            def commit(self):
                pass

        monkeypatch.setattr(storage, "lora_volume", MockVolume())
        monkeypatch.setattr(storage, "LORA_MOUNT_PATH", str(tmp_path))

        lora_dir = tmp_path / "user" / "model-v3"
        lora_dir.mkdir(parents=True)
        (lora_dir / "adapter_config.json").write_text("{}")

        storage._write_aliases({"user/model-latest": "user/model-v3"})

        assert storage.delete_lora("user/model-latest") is True
        assert (lora_dir / "adapter_config.json").exists()
        assert "user/model-latest" not in storage._read_aliases()

    def test_no_error_when_missing(self, tmp_path):
        """Doesn't raise when directory doesn't exist."""
        from claas.storage import cleanup_local_lora

        # Should not raise
        cleanup_local_lora(str(tmp_path / "nonexistent"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
