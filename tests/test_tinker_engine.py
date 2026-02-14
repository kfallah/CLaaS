"""Tests for Tinker training engine state management and engine methods."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claas.training_engines.tinker.state import (
    delete_entry,
    get_entry,
    get_tinker_path,
    list_loras,
    lora_exists,
    set_tinker_path,
)
from claas.types import (
    LoraDeleteResponse,
    LoraExistsPayload,
    LoraInitRequest,
    LoraInitResponse,
    LoraListResponse,
    ServiceHealth,
)

# ── State management tests ──────────────────────────────────────────


@pytest.fixture()
def state_file(tmp_path):
    """Yield a temporary state file path, cleaned up automatically."""
    return str(tmp_path / "tinker_state.json")


def test_state_roundtrip(state_file):
    """set_tinker_path then get_tinker_path returns the stored path."""
    set_tinker_path(
        lora_id="user/model",
        tinker_path="tinker://checkpoints/abc123",
        base_model="Qwen/Qwen3-235B-A22B",
        rank=32,
        step=0,
        path=state_file,
    )

    assert get_tinker_path("user/model", path=state_file) == "tinker://checkpoints/abc123"


def test_state_get_entry(state_file):
    """get_entry returns a full LoraEntry."""
    set_tinker_path(
        lora_id="user/model",
        tinker_path="tinker://checkpoints/abc",
        base_model="Qwen/Qwen3-235B-A22B",
        rank=16,
        step=5,
        path=state_file,
    )
    entry = get_entry("user/model", path=state_file)
    assert entry is not None
    assert entry.tinker_path == "tinker://checkpoints/abc"
    assert entry.base_model == "Qwen/Qwen3-235B-A22B"
    assert entry.rank == 16
    assert entry.step == 5


def test_lora_exists_false_for_missing(state_file):
    """lora_exists returns False when the lora_id is not tracked."""
    assert lora_exists("nonexistent/model", path=state_file) is False


def test_lora_exists_true(state_file):
    """lora_exists returns True for a tracked lora_id."""
    set_tinker_path(
        lora_id="user/model",
        tinker_path="tinker://x",
        base_model="m",
        rank=8,
        path=state_file,
    )
    assert lora_exists("user/model", path=state_file) is True


def test_list_loras_all(state_file):
    """list_loras with no prefix returns all entries."""
    set_tinker_path("a/one", "tinker://1", "m", 8, path=state_file)
    set_tinker_path("b/two", "tinker://2", "m", 8, path=state_file)
    set_tinker_path("a/three", "tinker://3", "m", 8, path=state_file)

    result = list_loras(path=state_file)
    assert result == ["a/one", "a/three", "b/two"]


def test_list_loras_with_prefix(state_file):
    """list_loras with a prefix filters correctly."""
    set_tinker_path("a/one", "tinker://1", "m", 8, path=state_file)
    set_tinker_path("b/two", "tinker://2", "m", 8, path=state_file)
    set_tinker_path("a/three", "tinker://3", "m", 8, path=state_file)

    result = list_loras(prefix="a/", path=state_file)
    assert result == ["a/one", "a/three"]


def test_get_tinker_path_missing(state_file):
    """get_tinker_path returns None for unknown lora_id."""
    assert get_tinker_path("missing", path=state_file) is None


def test_state_update_step(state_file):
    """Updating a lora_id overwrites the previous entry."""
    set_tinker_path("u/m", "tinker://v1", "model", 32, step=0, path=state_file)
    set_tinker_path("u/m", "tinker://v2", "model", 32, step=1, path=state_file)

    entry = get_entry("u/m", path=state_file)
    assert entry is not None
    assert entry.tinker_path == "tinker://v2"
    assert entry.step == 1


def test_delete_entry_existing(state_file):
    """delete_entry removes an existing entry and returns True."""
    set_tinker_path("user/model", "tinker://ckpt", "m", 8, path=state_file)
    assert delete_entry("user/model", path=state_file) is True
    assert get_entry("user/model", path=state_file) is None


def test_delete_entry_missing(state_file):
    """delete_entry returns False for a nonexistent entry."""
    assert delete_entry("nonexistent/model", path=state_file) is False



def test_state_corrupt_json(state_file):
    """Corrupt JSON file results in empty state (no crash)."""
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, "w") as f:
        f.write("{bad json")

    assert get_tinker_path("anything", path=state_file) is None
    assert list_loras(path=state_file) == []


# ── Engine tests (mocked Tinker SDK) ────────────────────────────────


def _make_engine_with_mocks(state_file):
    """Create a TinkerTrainingEngine with mocked Tinker SDK."""
    pytest.importorskip("tinker")
    with patch.dict(os.environ, {
        "CLAAS_TINKER_API_KEY": "fake-key",
        "CLAAS_TINKER_STATE_PATH": state_file,
    }):
        from claas.training_engines.tinker.engine import TinkerTrainingEngine

        engine = TinkerTrainingEngine()

    mock_service = MagicMock()
    engine._service = mock_service
    return engine, mock_service


def test_engine_init_lora(state_file):
    """init_lora creates a training client, saves state, and stores the path."""
    engine, mock_service = _make_engine_with_mocks(state_file)

    # Use spec=[] so _await_api_future doesn't find auto-generated result_async.
    mock_save_result = MagicMock(spec=[])
    mock_save_result.path = "tinker://checkpoints/init-abc"

    mock_tc = MagicMock()
    mock_tc.save_state_async = AsyncMock(return_value=mock_save_result)
    mock_service.create_lora_training_client_async = AsyncMock(return_value=mock_tc)

    request = LoraInitRequest(lora_id="test/lora", base_model="Qwen/Qwen3-235B-A22B", lora_r=32)

    with patch("claas.training_engines.tinker.engine.set_tinker_path"):
        result = asyncio.run(engine.init_lora(request))

    assert isinstance(result, LoraInitResponse)
    assert result.lora_id == "test/lora"
    mock_service.create_lora_training_client_async.assert_called_once_with(
        base_model="Qwen/Qwen3-235B-A22B", rank=32
    )
    mock_tc.save_state_async.assert_called_once_with("init")


def test_engine_list_loras(state_file):
    """list_loras delegates to state module."""
    engine, _ = _make_engine_with_mocks(state_file)

    with patch("claas.training_engines.tinker.engine.state_list_loras", return_value=["a", "b"]):
        result = asyncio.run(engine.list_loras(""))

    assert isinstance(result, LoraListResponse)
    assert result.loras == ["a", "b"]


def test_engine_lora_exists(state_file):
    """lora_exists delegates to state module."""
    engine, _ = _make_engine_with_mocks(state_file)

    with patch("claas.training_engines.tinker.engine.state_lora_exists", return_value=True):
        result = asyncio.run(engine.lora_exists("test/lora"))

    assert isinstance(result, LoraExistsPayload)
    assert result.exists is True


def test_engine_lora_runtime_ref_raises(state_file):
    """lora_runtime_ref raises ValueError for tinker backend."""
    engine, _ = _make_engine_with_mocks(state_file)

    with pytest.raises(ValueError, match="tinker backend"):
        asyncio.run(engine.lora_runtime_ref("test/lora"))


def test_engine_health_ok(state_file):
    """health returns healthy when service responds."""
    engine, mock_service = _make_engine_with_mocks(state_file)
    mock_service.get_server_capabilities_async = AsyncMock(return_value=MagicMock())

    result = asyncio.run(engine.health())
    assert isinstance(result, ServiceHealth)
    assert result.status == "healthy"


def test_engine_health_error(state_file):
    """health returns unhealthy when service raises."""
    engine, mock_service = _make_engine_with_mocks(state_file)
    mock_service.get_server_capabilities_async = AsyncMock(
        side_effect=ConnectionError("down")
    )

    result = asyncio.run(engine.health())
    assert isinstance(result, ServiceHealth)
    assert result.status == "unhealthy"
    assert "down" in result.error


def test_engine_delete_lora(state_file):
    """delete_lora removes state entry and returns deleted=True."""
    from claas.training_engines.tinker.state import LoraEntry

    engine, mock_service = _make_engine_with_mocks(state_file)
    mock_service.delete_checkpoint_from_tinker_path_async = AsyncMock(return_value=None)

    mock_entry = LoraEntry(tinker_path="tinker://ckpt-del", base_model="m", rank=8, step=0)
    with (
        patch("claas.training_engines.tinker.engine.get_entry", return_value=mock_entry),
        patch("claas.training_engines.tinker.engine.delete_entry") as mock_delete,
    ):
        result = asyncio.run(engine.delete_lora("test/del"))

    assert isinstance(result, LoraDeleteResponse)
    assert result.deleted is True
    mock_service.delete_checkpoint_from_tinker_path_async.assert_called_once_with("tinker://ckpt-del")
    mock_delete.assert_called_once_with("test/del")


def test_engine_delete_lora_missing(state_file):
    """delete_lora returns deleted=False for nonexistent LoRA."""
    engine, _ = _make_engine_with_mocks(state_file)

    with patch("claas.training_engines.tinker.engine.get_entry", return_value=None):
        result = asyncio.run(engine.delete_lora("missing/lora"))

    assert isinstance(result, LoraDeleteResponse)
    assert result.deleted is False
