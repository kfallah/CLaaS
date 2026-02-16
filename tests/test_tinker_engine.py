"""Tests for Tinker training engine state management and engine methods."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claas.training_engines.tinker.state import (
    all_checkpoint_paths,
    delete_entry,
    get_entry,
    get_tinker_path,
    list_loras,
    lora_exists,
    set_tinker_path,
)
from claas.types import (
    DistillBatchItem,
    DistillBatchRequestPayload,
    DistillResponse,
    LoraDeleteResponse,
    LoraExistsPayload,
    LoraInitRequest,
    LoraInitResponse,
    LoraListResponse,
    ServiceHealth,
    TrainingConfig,
)

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture()
def state_file(tmp_path):
    """Yield a temporary state file path, cleaned up automatically."""
    return str(tmp_path / "tinker_state.json")


@pytest.fixture()
def tinker_env(state_file):
    """Point all tinker state operations at a temp file for the test duration.

    Patches ``_DEFAULT_STATE_PATH`` so every call to ``get_entry``,
    ``set_tinker_path``, etc. (even without an explicit ``path=``)
    reads/writes the temp file.
    """
    with patch("claas.training_engines.tinker.state._DEFAULT_STATE_PATH", state_file):
        yield state_file


@pytest.fixture()
def tinker_engine(tinker_env):
    """Create a TinkerTrainingEngine backed by a mocked ServiceClient.

    Returns ``(engine, mock_service)`` tuple.  All state operations are
    routed to the temp state file via ``tinker_env``.
    """
    pytest.importorskip("tinker")
    with patch.dict(os.environ, {"CLAAS_TINKER_API_KEY": "fake-key"}):
        from claas.training_engines.tinker.engine import TinkerTrainingEngine

        engine = TinkerTrainingEngine()
    mock_service = MagicMock()
    engine._service = mock_service
    return engine, mock_service


@pytest.fixture()
def mock_tokenizer():
    """Tokenizer mock: each character becomes one token (sequential IDs)."""
    tok = MagicMock()
    tok.encode.side_effect = lambda text, **kw: list(range(len(text)))
    tok.apply_chat_template.return_value = "chat-template-output"
    return tok


@pytest.fixture()
def mock_training_client(mock_tokenizer):
    """Fully wired mock training client for distill/init tests.

    Provides tokenizer, save_state, save_weights_for_sampler,
    forward_backward, optim_step, and a student sampling client.
    """
    tc = MagicMock()
    tc.get_tokenizer.return_value = mock_tokenizer

    save_result = MagicMock(spec=[])
    save_result.path = "tinker://checkpoints/step-1"
    tc.save_state_async = AsyncMock(return_value=save_result)

    sampler_save = MagicMock(spec=[])
    sampler_save.path = "tinker://weights/step-1-sampler"
    tc.save_weights_for_sampler_async = AsyncMock(return_value=sampler_save)

    fwd_bwd = MagicMock()
    fwd_bwd.metrics = {"loss": 0.42}
    tc.forward_backward_async = AsyncMock(return_value=fwd_bwd)
    tc.optim_step_async = AsyncMock()

    student_sampler = MagicMock()
    student_sampler.compute_logprobs_async = AsyncMock(
        return_value=[-0.1] * 100
    )
    tc.save_weights_and_get_sampling_client_async = AsyncMock(
        return_value=student_sampler
    )

    return tc


# ── State management tests ──────────────────────────────────────────


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


def test_lora_exists(state_file):
    """lora_exists returns False for missing and True for tracked IDs."""
    assert lora_exists("nonexistent/model", path=state_file) is False

    set_tinker_path("user/model", "tinker://x", "m", 8, path=state_file)
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


def test_old_paths_accumulate_across_steps(state_file):
    """Successive set_tinker_path calls accumulate superseded paths in old_paths."""
    # Step 0: init checkpoint
    set_tinker_path("u/m", "tinker://init", "model", 8, step=0, path=state_file)
    entry = get_entry("u/m", path=state_file)
    assert entry is not None
    assert entry.old_paths is None

    # Step 1: distill — init path moves to old_paths, sampler weights tracked
    set_tinker_path(
        "u/m", "tinker://step-1", "model", 8, step=1,
        sampler_weights_path="tinker://step-1-sampler", path=state_file,
    )
    entry = get_entry("u/m", path=state_file)
    assert entry is not None
    assert entry.tinker_path == "tinker://step-1"
    assert entry.sampler_weights_path == "tinker://step-1-sampler"
    assert "tinker://init" in (entry.old_paths or [])

    # Step 2: distill again — step-1 and step-1-sampler move to old_paths
    set_tinker_path(
        "u/m", "tinker://step-2", "model", 8, step=2,
        sampler_weights_path="tinker://step-2-sampler", path=state_file,
    )
    entry = get_entry("u/m", path=state_file)
    assert entry is not None
    assert entry.tinker_path == "tinker://step-2"
    old = entry.old_paths or []
    assert "tinker://init" in old
    assert "tinker://step-1" in old
    assert "tinker://step-1-sampler" in old

    # all_checkpoint_paths returns current + sampler + all old
    paths = all_checkpoint_paths(entry)
    assert "tinker://step-2" in paths
    assert "tinker://step-2-sampler" in paths
    assert "tinker://step-1" in paths
    assert "tinker://init" in paths
    assert len(paths) == len(set(paths))  # no duplicates


def test_state_corrupt_json(state_file):
    """Corrupt JSON file results in empty state (no crash)."""
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, "w") as f:
        f.write("{bad json")

    assert get_tinker_path("anything", path=state_file) is None
    assert list_loras(path=state_file) == []


# ── Engine lifecycle tests (using fixtures) ──────────────────────────


def test_engine_init_lora(tinker_engine):
    """init_lora creates a training client, saves checkpoint, and persists state."""
    engine, mock_service = tinker_engine

    mock_save_result = MagicMock(spec=[])
    mock_save_result.path = "tinker://checkpoints/init-abc"

    mock_tc = MagicMock()
    mock_tc.save_state_async = AsyncMock(return_value=mock_save_result)
    mock_service.create_lora_training_client_async = AsyncMock(return_value=mock_tc)

    request = LoraInitRequest(lora_id="test/lora", base_model="Qwen/Qwen3-235B-A22B", lora_r=32)
    result = asyncio.run(engine.init_lora(request))

    assert isinstance(result, LoraInitResponse)
    assert result.lora_id == "test/lora"
    mock_service.create_lora_training_client_async.assert_called_once_with(
        base_model="Qwen/Qwen3-235B-A22B", rank=32
    )
    mock_tc.save_state_async.assert_called_once_with("init")

    # State was persisted to the temp file (via tinker_env)
    entry = get_entry("test/lora")
    assert entry is not None
    assert entry.tinker_path == "tinker://checkpoints/init-abc"
    assert entry.base_model == "Qwen/Qwen3-235B-A22B"
    assert entry.rank == 32
    assert entry.step == 0


def test_engine_list_and_exists(tinker_engine):
    """list_loras and lora_exists read from the shared state file."""
    engine, _ = tinker_engine

    # Pre-populate state
    set_tinker_path("a/one", "tinker://1", "m", 8)
    set_tinker_path("b/two", "tinker://2", "m", 8)

    result = asyncio.run(engine.list_loras(""))
    assert isinstance(result, LoraListResponse)
    assert result.loras == ["a/one", "b/two"]

    exists_yes = asyncio.run(engine.lora_exists("a/one"))
    assert isinstance(exists_yes, LoraExistsPayload)
    assert exists_yes.exists is True

    exists_no = asyncio.run(engine.lora_exists("missing/lora"))
    assert exists_no.exists is False


def test_engine_lora_runtime_ref_raises(tinker_engine):
    """lora_runtime_ref raises ValueError for tinker backend."""
    engine, _ = tinker_engine

    with pytest.raises(ValueError, match="tinker backend"):
        asyncio.run(engine.lora_runtime_ref("test/lora"))


def test_engine_health_ok(tinker_engine):
    """health returns healthy when service responds."""
    engine, mock_service = tinker_engine
    mock_service.get_server_capabilities_async = AsyncMock(return_value=MagicMock())

    result = asyncio.run(engine.health())
    assert isinstance(result, ServiceHealth)
    assert result.status == "healthy"


def test_engine_health_error(tinker_engine):
    """health returns unhealthy when service raises."""
    engine, mock_service = tinker_engine
    mock_service.get_server_capabilities_async = AsyncMock(
        side_effect=ConnectionError("down")
    )

    result = asyncio.run(engine.health())
    assert isinstance(result, ServiceHealth)
    assert result.status == "unhealthy"
    assert "down" in result.error


def test_engine_delete_lora_cleans_up_checkpoints(tinker_engine):
    """delete_lora removes all tracked checkpoints from Tinker and state."""
    engine, mock_service = tinker_engine

    # Pre-populate state with 2 distillation steps (accumulates old_paths)
    set_tinker_path("test/del", "tinker://init", "m", 8, step=0)
    set_tinker_path(
        "test/del", "tinker://step-1", "m", 8, step=1,
        sampler_weights_path="tinker://step-1-sampler",
    )

    mock_rest = MagicMock()
    mock_rest.delete_checkpoint_from_tinker_path_async = AsyncMock(return_value=None)
    mock_service.create_rest_client = MagicMock(return_value=mock_rest)

    result = asyncio.run(engine.delete_lora("test/del"))

    assert isinstance(result, LoraDeleteResponse)
    assert result.deleted is True
    assert get_entry("test/del") is None

    # All checkpoints were deleted: current + sampler + old
    deleted_paths = [
        call.args[0]
        for call in mock_rest.delete_checkpoint_from_tinker_path_async.call_args_list
    ]
    assert "tinker://step-1" in deleted_paths
    assert "tinker://step-1-sampler" in deleted_paths
    assert "tinker://init" in deleted_paths


def test_engine_delete_lora_continues_on_checkpoint_error(tinker_engine):
    """delete_lora logs a warning but completes when a checkpoint delete fails."""
    engine, mock_service = tinker_engine
    set_tinker_path("test/del", "tinker://ckpt", "m", 8, step=0)

    mock_rest = MagicMock()
    mock_rest.delete_checkpoint_from_tinker_path_async = AsyncMock(
        side_effect=RuntimeError("remote error")
    )
    mock_service.create_rest_client = MagicMock(return_value=mock_rest)

    result = asyncio.run(engine.delete_lora("test/del"))

    # Still returns deleted=True (state is removed even if remote delete fails)
    assert result.deleted is True
    assert get_entry("test/del") is None


# ── Distill flow tests ──────────────────────────────────────────────


def test_engine_distill_full_flow(tinker_engine, mock_training_client):
    """End-to-end distill: restore → tokenize → logprobs → train → save.

    Exercises the complete distill method with rollout_logprobs=None,
    forcing the engine to compute student logprobs via sampling client.
    """
    engine, mock_service = tinker_engine

    # Pre-populate state with an initialized LoRA
    set_tinker_path("test/lora", "tinker://init-ckpt", "gpt-oss/GPT-OSS-120B", 32, step=0)

    # Wire the mock training client into the service
    mock_service.create_training_client_from_state_async = AsyncMock(
        return_value=mock_training_client
    )

    # Teacher sampling client
    teacher_sampler = MagicMock()
    teacher_sampler.compute_logprobs_async = AsyncMock(
        return_value=[-0.2] * 100
    )
    mock_service.create_sampling_client_async = AsyncMock(
        return_value=teacher_sampler
    )

    payload = DistillBatchRequestPayload(
        lora_id="test/lora",
        training=TrainingConfig(),
        samples=[
            DistillBatchItem(
                prompt="Hello",
                response="World",
                feedback="Good job",
                rollout_logprobs=[],
            )
        ],
    )

    result = asyncio.run(engine.distill(payload))

    # Verify response structure
    assert isinstance(result, DistillResponse)
    assert result.lora_id == "test/lora"
    assert result.metadata["step"] == 1
    assert result.metadata["tinker_path"] == "tinker://checkpoints/step-1"
    assert result.metadata["sampler_weights_path"] == "tinker://weights/step-1-sampler"
    assert result.metadata["completion_len"] == len("World")
    assert "effective_kl_coef" in result.metadata
    assert "kl_mean" in result.metadata
    assert "adv_mean" in result.metadata
    assert result.metadata["loss_fn"] == "importance_sampling"
    assert result.metadata["tinker_fwd_metrics"] == {"loss": 0.42}

    # Verify the training client was restored from the init checkpoint
    mock_service.create_training_client_from_state_async.assert_called_once_with(
        "tinker://init-ckpt"
    )

    # Verify student logprobs were computed via sampling client (not provided)
    mock_training_client.save_weights_and_get_sampling_client_async.assert_called_once_with(
        "current"
    )

    # Verify teacher was queried with base model
    mock_service.create_sampling_client_async.assert_called_once_with(
        base_model="gpt-oss/GPT-OSS-120B"
    )

    # Verify training step happened
    mock_training_client.forward_backward_async.assert_called_once()
    mock_training_client.optim_step_async.assert_called_once()

    # Verify checkpoint was saved
    mock_training_client.save_state_async.assert_called_once_with("step-1")
    mock_training_client.save_weights_for_sampler_async.assert_called_once_with("step-1")

    # Verify state was updated
    entry = get_entry("test/lora")
    assert entry is not None
    assert entry.step == 1
    assert entry.tinker_path == "tinker://checkpoints/step-1"
    assert entry.sampler_weights_path == "tinker://weights/step-1-sampler"
    assert "tinker://init-ckpt" in (entry.old_paths or [])


def test_engine_distill_uses_provided_rollout_logprobs(tinker_engine, mock_training_client):
    """When rollout_logprobs are provided, the engine skips student sampling."""
    engine, mock_service = tinker_engine

    set_tinker_path("test/lora", "tinker://ckpt", "gpt-oss/GPT-OSS-120B", 32, step=0)
    mock_service.create_training_client_from_state_async = AsyncMock(
        return_value=mock_training_client
    )

    teacher_sampler = MagicMock()
    teacher_sampler.compute_logprobs_async = AsyncMock(return_value=[-0.3] * 100)
    mock_service.create_sampling_client_async = AsyncMock(return_value=teacher_sampler)

    # Provide rollout_logprobs matching the response length (5 chars = 5 tokens)
    payload = DistillBatchRequestPayload(
        lora_id="test/lora",
        training=TrainingConfig(),
        samples=[
            DistillBatchItem(
                prompt="Hello",
                response="World",
                feedback="Nice",
                rollout_logprobs=[-0.1, -0.2, -0.3, -0.4, -0.5],
            )
        ],
    )

    result = asyncio.run(engine.distill(payload))

    assert isinstance(result, DistillResponse)
    assert result.metadata["step"] == 1

    # Student sampling client was NOT used
    mock_training_client.save_weights_and_get_sampling_client_async.assert_not_called()


# ── Helper function tests ────────────────────────────────────────────


def test_require_entry(tinker_env):
    """_require_entry raises for missing and returns entry for existing LoRA."""
    from claas.training_engines.tinker.engine import _require_entry

    with pytest.raises(FileNotFoundError, match="not found"):
        _require_entry("nonexistent/lora")

    set_tinker_path("test/lora", "tinker://ckpt", "m", 8, step=3)
    entry = _require_entry("test/lora")
    assert entry.tinker_path == "tinker://ckpt"
    assert entry.step == 3


def test_slice_completion_logprobs():
    """_slice_completion_logprobs extracts the completion portion, replacing None."""
    pytest.importorskip("tinker")
    from claas.training_engines.tinker.engine import _slice_completion_logprobs

    logprobs_full = [None, -1.0, -2.0, -3.0, -4.0, -5.0]
    # prompt_len=2, completion_len=3 → slice [2:5] = [-2.0, -3.0, -4.0]
    result = _slice_completion_logprobs(logprobs_full, prompt_len=2, completion_len=3)
    assert result == [-2.0, -3.0, -4.0]

    # None values in the completion portion are replaced with 0.0
    logprobs_with_none = [None, -1.0, None, -3.0, -4.0]
    result2 = _slice_completion_logprobs(logprobs_with_none, prompt_len=1, completion_len=3)
    assert result2 == [-1.0, 0.0, -3.0]


def test_await_api_future():
    """_await_api_future: result_async → result() → passthrough."""
    pytest.importorskip("tinker")
    from claas.training_engines.tinker.engine import _await_api_future

    # Branch 1: has result_async
    obj_async = MagicMock()
    obj_async.result_async = AsyncMock(return_value="async-value")
    assert asyncio.run(_await_api_future(obj_async)) == "async-value"

    # Branch 2: has result() but no result_async
    obj_sync = MagicMock(spec=[])
    obj_sync.result = MagicMock(return_value="sync-value")
    assert asyncio.run(_await_api_future(obj_sync)) == "sync-value"

    # Branch 3: no result method — passthrough
    obj_plain = MagicMock(spec=[])
    assert asyncio.run(_await_api_future(obj_plain)) is obj_plain


def test_engine_export_lora(tinker_engine):
    """export_lora downloads the checkpoint archive via REST client."""
    import httpx

    engine, mock_service = tinker_engine
    set_tinker_path("test/export", "tinker://ckpt-export", "m", 8)

    mock_archive_future = MagicMock()
    mock_archive_future.result.return_value = "https://example.com/archive.zip"

    mock_rest = MagicMock()
    mock_rest.get_checkpoint_archive_url.return_value = mock_archive_future
    mock_service.create_rest_client.return_value = mock_rest

    fake_zip = b"PK\x03\x04fakecontent"
    with patch.object(
        httpx, "get",
        return_value=MagicMock(status_code=200, content=fake_zip, raise_for_status=lambda: None),
    ):
        result = asyncio.run(engine.export_lora("test/export"))

    assert result.filename == "test%2Fexport.zip"
    assert result.content == fake_zip
