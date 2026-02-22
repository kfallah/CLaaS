"""Tests for FastAPI endpoint behavior."""

from __future__ import annotations

import asyncio
import hashlib

import pytest
from fastapi.testclient import TestClient

from claas.api import web_app
from claas.core.config import LocalConfig, ModalConfig, TinkerConfig
from claas.core.types import (
    DistillResponse,
    LoraDeleteResponse,
    LoraExistsPayload,
    LoraExportPayload,
    LoraInitResponse,
    LoraListResponse,
    LoraRuntimeRef,
    ServiceHealth,
)
from claas.inference.cache import CompletionCacheEntry, completion_cache
from claas.inference.helpers import normalize_for_hash


def _mock_config(monkeypatch, mode: str, **overrides):
    """Patch API runtime-config accessor to return a config for the given mode."""
    from claas import api

    log_dir = str(overrides.get("feedback_log_dir", "./feedback_logs"))
    root = str(overrides.get("lora_root", "/loras"))
    backend = str(overrides.get("storage_backend", "modal_volume"))
    allowed = list(overrides.get("allowed_init_base_models", ["Qwen/Qwen3-8B"]))

    if mode == "modal":
        cfg = ModalConfig(
            mode="modal",
            feedback_log_dir=log_dir,
            lora_root=root,
            storage_backend=backend,
            allowed_init_base_models=allowed,
            vllm_base_url="http://127.0.0.1:8000",
        )
    elif mode == "tinker":
        cfg = TinkerConfig(
            mode="tinker",
            feedback_log_dir=log_dir,
            lora_root=root,
            storage_backend=backend,
            allowed_init_base_models=allowed,
            tinker_base_model="gpt-oss/GPT-OSS-120B",
            tinker_state_path="",
            vllm_base_url="http://127.0.0.1:8000",
        )
    else:
        cfg = LocalConfig(
            mode="local",
            feedback_log_dir=log_dir,
            lora_root=root,
            storage_backend=backend,
            allowed_init_base_models=allowed,
            vllm_base_url="http://127.0.0.1:8000",
        )

    monkeypatch.setattr(api, "_runtime_config", lambda: cfg)


@pytest.fixture()
def api_client():
    """FastAPI TestClient for the web_app."""
    return TestClient(web_app)



def _set_engine(monkeypatch, engine):
    """Wire a mock engine into the API."""
    from claas import api

    monkeypatch.setattr(api, "_get_training_engine", lambda: engine)


class _RemoteCall:
    def __init__(self, payload, capture=None):
        self.payload = payload
        self.capture = capture

    async def aio(self, request):
        if self.capture is not None:
            self.capture["request"] = request
        return self.payload


class _FunctionStub:
    def __init__(self, payload, capture=None):
        self.remote = _RemoteCall(payload, capture=capture)


class _RemoteCallFailure:
    async def aio(self, _request):
        raise RuntimeError("modal rpc failure")


class _FunctionFailureStub:
    def __init__(self):
        self.remote = _RemoteCallFailure()


def _seed_cache(visible_response: str, *, logprobs: list[float] | None = None) -> None:
    """Pre-populate the completion cache keyed by normalized SHA-256 of visible_response."""
    content_hash = hashlib.sha256(normalize_for_hash(visible_response).encode("utf-8")).hexdigest()
    completion_cache.put(
        content_hash,
        CompletionCacheEntry(
            prompt="cached-prompt",
            response="cached-response",
            response_token_ids=[10, 20],
            prompt_token_ids=[1, 2, 3],
            response_logprobs=logprobs if logprobs is not None else [-0.1, -0.2],
        ),
    )


class _EngineStub:
    """Minimal TrainingEngine mock for API tests."""

    def __init__(self, exists=True):
        self._exists = exists

    async def lora_exists(self, _lora_id):

        return LoraExistsPayload(exists=self._exists)

    async def lora_runtime_ref(self, lora_id):

        return LoraRuntimeRef(vllm_name=lora_id, lora_path=f"/loras/{lora_id}")

    async def distill(self, payload):
        import modal


        distill_fn = modal.Function.from_name("claas-distill", "DistillWorker.distill")
        result = await distill_fn.remote.aio(payload.model_dump())
        return DistillResponse.model_validate(result)


def test_feedback_success_inplace_flow(monkeypatch, tmp_path):
    from claas import api

    _mock_config(monkeypatch, "modal")
    calls = []
    captured = {}
    log_records = []
    log_path = str(tmp_path / "feedback-log.json")
    _seed_cache("test response")

    def fake_from_name(_app, fn_name):
        if fn_name == "DistillWorker.distill":
            return _FunctionStub(
                {"lora_id": "user/model", "metadata": {"tokens_processed": 2}},
                capture=captured,
            )
        raise AssertionError(f"unexpected modal function: {fn_name}")

    async def fake_vllm_post(path, *, params=None, json_body=None, timeout_s=30.0):
        calls.append((path, params))

    def fake_write_feedback_log(record):
        log_records.append(record)
        return log_path

    async def fake_wait_idle():
        calls.append(("_wait_for_vllm_idle",))

    monkeypatch.setattr(api, "_get_training_engine", lambda: _EngineStub(exists=True))
    monkeypatch.setattr(api.modal.Function, "from_name", fake_from_name)
    monkeypatch.setattr(api, "_vllm_post", fake_vllm_post)
    monkeypatch.setattr(api, "_wait_for_vllm_idle", fake_wait_idle)
    monkeypatch.setattr(api, "_write_feedback_log", fake_write_feedback_log)

    client = TestClient(web_app)
    response = client.post(
        "/v1/feedback",
        json={
            "requests": [
                {
                    "lora_id": "user/model",
                    "prompt": "clean prompt",
                    "response": "test response",
                    "feedback": "f",
                    "training": {},
                }
            ],
            "orchestration": {"sleep_before": True, "wake_after": True, "wake_on_failure": True, "sleep_level": 1},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["lora_id"] == "user/model"
    assert body["feedback_log_path"] == log_path
    # drain → pause → wake → unload old LoRA → load updated LoRA
    assert calls[0] == ("_wait_for_vllm_idle",)
    assert calls[1] == ("/pause", {"level": 1})
    assert calls[2] == ("/resume", None)
    assert calls[3] == ("/v1/unload_lora_adapter", None)
    assert calls[4] == ("/v1/load_lora_adapter", None)
    assert captured["request"]["save_in_place"] is True
    assert log_records and log_records[0]["status"] == "ok"


def test_feedback_returns_500_and_logs_error(monkeypatch, tmp_path):
    from claas import api

    _mock_config(monkeypatch, "modal")
    log_records = []
    log_path = str(tmp_path / "feedback-log.json")
    _seed_cache("error response")

    def fake_from_name(_app, fn_name):
        if fn_name == "DistillWorker.distill":
            return _FunctionFailureStub()
        raise AssertionError(f"unexpected modal function: {fn_name}")

    async def fake_vllm_post(_path, *, params=None, json_body=None, timeout_s=30.0):
        return None

    def fake_write_feedback_log(record):
        log_records.append(record)
        return log_path

    async def fake_wait_idle():
        pass

    monkeypatch.setattr(api, "_get_training_engine", lambda: _EngineStub(exists=True))
    monkeypatch.setattr(api.modal.Function, "from_name", fake_from_name)
    monkeypatch.setattr(api, "_vllm_post", fake_vllm_post)
    monkeypatch.setattr(api, "_wait_for_vllm_idle", fake_wait_idle)
    monkeypatch.setattr(api, "_write_feedback_log", fake_write_feedback_log)

    client = TestClient(web_app)
    response = client.post(
        "/v1/feedback",
        json={
            "requests": [
                {
                    "lora_id": "user/model",
                    "prompt": "clean prompt",
                    "response": "error response",
                    "feedback": "f",
                    "training": {},
                }
            ],
            "orchestration": {"sleep_before": True, "wake_after": True, "wake_on_failure": True, "sleep_level": 1},
        },
    )

    assert response.status_code == 500
    assert "Feedback update failed" in response.json()["detail"]
    assert log_records and log_records[0]["status"] == "error"


def test_export_404_when_missing(monkeypatch):
    from claas import api

    monkeypatch.setattr(api, "_get_training_engine", lambda: _EngineStub(exists=False))
    client = TestClient(web_app)

    response = client.get("/v1/lora/export", params={"lora_id": "user/missing"})
    assert response.status_code == 404


def test_feedback_calls_drain_before_pause(monkeypatch, tmp_path):
    """_wait_for_vllm_idle is called before /pause."""
    from claas import api

    _mock_config(monkeypatch, "modal")
    order = []
    _seed_cache("drain response")

    def fake_from_name(_app, fn_name):
        if fn_name == "DistillWorker.distill":
            return _FunctionStub(
                {"lora_id": "user/model", "metadata": {"tokens_processed": 1}},
            )
        raise AssertionError(f"unexpected modal function: {fn_name}")

    async def fake_wait_idle():
        order.append("drain")

    async def fake_vllm_post(path, *, params=None, json_body=None, timeout_s=30.0):
        order.append(path)

    monkeypatch.setattr(api, "_get_training_engine", lambda: _EngineStub(exists=True))
    monkeypatch.setattr(api.modal.Function, "from_name", fake_from_name)
    monkeypatch.setattr(api, "_wait_for_vllm_idle", fake_wait_idle)
    monkeypatch.setattr(api, "_vllm_post", fake_vllm_post)
    monkeypatch.setattr(api, "_write_feedback_log", lambda _r: str(tmp_path / "log.json"))

    client = TestClient(web_app)
    response = client.post(
        "/v1/feedback",
        json={
            "requests": [
                {
                    "lora_id": "user/model",
                    "prompt": "clean prompt",
                    "response": "drain response",
                    "feedback": "f",
                    "training": {},
                }
            ],
            "orchestration": {"sleep_before": True, "wake_after": True, "wake_on_failure": True, "sleep_level": 1},
        },
    )

    assert response.status_code == 200
    # drain must come before /pause
    assert order[0] == "drain"
    assert order[1] == "/pause"



def test_feedback_drain_timeout_returns_503(monkeypatch, tmp_path):
    """A drain timeout produces a 503 response."""
    from claas import api

    _mock_config(monkeypatch, "modal")
    _seed_cache("timeout response")

    async def fake_wait_idle():
        raise TimeoutError("still busy")

    async def fake_vllm_post(_path, *, params=None, json_body=None, timeout_s=30.0):
        pass

    monkeypatch.setattr(api, "_get_training_engine", lambda: _EngineStub(exists=True))
    monkeypatch.setattr(api, "_wait_for_vllm_idle", fake_wait_idle)
    monkeypatch.setattr(api, "_vllm_post", fake_vllm_post)
    monkeypatch.setattr(api, "_write_feedback_log", lambda _r: str(tmp_path / "log.json"))

    client = TestClient(web_app)
    response = client.post(
        "/v1/feedback",
        json={
            "requests": [
                {
                    "lora_id": "user/model",
                    "prompt": "clean prompt",
                    "response": "timeout response",
                    "feedback": "f",
                    "training": {},
                }
            ],
            "orchestration": {"sleep_before": True, "wake_after": True, "wake_on_failure": True, "sleep_level": 1},
        },
    )

    assert response.status_code == 503
    assert "vLLM not idle" in response.json()["detail"]




def test_feedback_resolves_logprobs_from_cache(monkeypatch, tmp_path):
    """Rollout logprobs are resolved from the completion cache, not sent by client."""
    from claas import api

    _mock_config(monkeypatch, "modal")
    _seed_cache("logprob response", logprobs=[-0.1, -0.2])
    captured = {}

    def fake_from_name(_app, fn_name):
        if fn_name == "DistillWorker.distill":
            return _FunctionStub(
                {"lora_id": "user/model", "metadata": {"tokens_processed": 1}},
                capture=captured,
            )
        raise AssertionError(f"unexpected modal function: {fn_name}")


    monkeypatch.setattr(api, "_get_training_engine", lambda: _EngineStub(exists=True))
    monkeypatch.setattr(api.modal.Function, "from_name", fake_from_name)
    monkeypatch.setattr(api, "_vllm_post", lambda *_a, **_kw: _noop_coro())
    monkeypatch.setattr(api, "_wait_for_vllm_idle", lambda: _noop_coro())
    monkeypatch.setattr(api, "_write_feedback_log", lambda _r: str(tmp_path / "log.json"))

    client = TestClient(web_app)
    response = client.post(
        "/v1/feedback",
        json={
            "requests": [
                {
                    "lora_id": "user/model",
                    "prompt": "clean prompt",
                    "response": "logprob response",
                    "feedback": "f",
                    "training": {},
                }
            ],
            "orchestration": {"sleep_before": True, "wake_after": True, "wake_on_failure": True, "sleep_level": 1},
        },
    )

    assert response.status_code == 200
    assert captured["request"]["samples"][0]["response_logprobs"] == [-0.1, -0.2]


def test_feedback_resolves_token_ids_from_cache(monkeypatch, tmp_path):
    """Token IDs are resolved from cache, not sent by client."""
    from claas import api

    _mock_config(monkeypatch, "tinker")
    _seed_cache("token id response")
    captured_payload = {}

    class _Engine:
        async def lora_exists(self, _lora_id):
            return LoraExistsPayload(exists=True)

    async def fake_run_distill(payload):
        captured_payload["samples"] = [s.model_dump() for s in payload.samples]
        return DistillResponse(lora_id="user/model", metadata={})

    monkeypatch.setattr(api, "_get_training_engine", lambda: _Engine())
    monkeypatch.setattr(api, "_run_distill", fake_run_distill)
    monkeypatch.setattr(api, "_write_feedback_log", lambda _r: str(tmp_path / "log.json"))

    client = TestClient(web_app)
    response = client.post(
        "/v1/feedback",
        json={
            "requests": [
                {
                    "lora_id": "user/model",
                    "prompt": "clean prompt",
                    "response": "token id response",
                    "feedback": "good",
                    "training": {},
                }
            ],
            "orchestration": {"sleep_before": False, "wake_after": False},
        },
    )

    assert response.status_code == 200
    sample = captured_payload["samples"][0]
    assert sample["prompt_token_ids"] == [1, 2, 3]
    assert sample["response_token_ids"] == [10, 20]
    assert sample["user_prompt"] == "clean prompt"
    assert sample["prompt"] == "cached-prompt"
    assert sample["response"] == "cached-response"


def test_feedback_cache_miss_returns_404(monkeypatch, tmp_path):
    """Missing cache entry returns 404 before any orchestration."""
    from claas import api

    _mock_config(monkeypatch, "tinker")

    class _Engine:
        async def lora_exists(self, _lora_id):
            return LoraExistsPayload(exists=True)

    monkeypatch.setattr(api, "_get_training_engine", lambda: _Engine())
    monkeypatch.setattr(api, "_write_feedback_log", lambda _r: str(tmp_path / "log.json"))

    client = TestClient(web_app)
    response = client.post(
        "/v1/feedback",
        json={
            "requests": [
                {
                    "lora_id": "user/model",
                    "prompt": "clean prompt",
                    "response": "nonexistent response",
                    "feedback": "f",
                    "training": {},
                }
            ],
            "orchestration": {"sleep_before": False, "wake_after": False},
        },
    )

    assert response.status_code == 404
    assert "No cached completion" in response.json()["detail"]


def test_feedback_missing_logprobs_returns_422(monkeypatch, tmp_path):
    """Cache entry without logprobs returns 422."""
    from claas import api

    _mock_config(monkeypatch, "tinker")
    # Seed cache with logprobs=None, keyed by hash of the response text
    no_logprobs_response = "no logprobs response"
    content_hash = hashlib.sha256(normalize_for_hash(no_logprobs_response).encode("utf-8")).hexdigest()
    completion_cache.put(
        content_hash,
        CompletionCacheEntry(
            prompt="p",
            response="r",
            response_token_ids=[10],
            prompt_token_ids=[1],
            response_logprobs=None,
        ),
    )

    class _Engine:
        async def lora_exists(self, _lora_id):
            return LoraExistsPayload(exists=True)

    monkeypatch.setattr(api, "_get_training_engine", lambda: _Engine())
    monkeypatch.setattr(api, "_write_feedback_log", lambda _r: str(tmp_path / "log.json"))

    client = TestClient(web_app)
    response = client.post(
        "/v1/feedback",
        json={
            "requests": [
                {
                    "lora_id": "user/model",
                    "prompt": "clean prompt",
                    "response": no_logprobs_response,
                    "feedback": "f",
                    "training": {},
                }
            ],
            "orchestration": {"sleep_before": False, "wake_after": False},
        },
    )

    assert response.status_code == 422
    assert "no logprobs" in response.json()["detail"]


async def _noop_coro():
    pass



def test_feedback_uses_resolved_lock_key(monkeypatch, tmp_path):
    from claas import api

    _mock_config(monkeypatch, "modal")
    _seed_cache("lock key response")

    class _Engine:
        async def lora_exists(self, _lora_id):
            return LoraExistsPayload(exists=True)

    captured = {}

    async def fake_lock_key(_lora_id):
        return "user-model-v1"

    async def fake_get_lock(key):
        captured["key"] = key
        return asyncio.Lock()

    async def fake_run_distill(_payload):
        return DistillResponse(lora_id="user/model", metadata={})

    monkeypatch.setattr(api, "_get_training_engine", lambda: _Engine())
    monkeypatch.setattr(api, "_get_feedback_lock_key", fake_lock_key)
    monkeypatch.setattr(api, "_get_feedback_lock", fake_get_lock)
    monkeypatch.setattr(api, "_run_distill", fake_run_distill)
    monkeypatch.setattr(api, "_write_feedback_log", lambda _r: str(tmp_path / "log.json"))

    client = TestClient(web_app)
    response = client.post(
        "/v1/feedback",
        json={
            "requests": [
                {
                    "lora_id": "user/model-latest",
                    "prompt": "clean prompt",
                    "response": "lock key response",
                    "feedback": "f",
                    "training": {},
                }
            ],
            "orchestration": {"sleep_before": False, "wake_after": False, "wake_on_failure": True, "sleep_level": 1},
        },
    )

    assert response.status_code == 200
    assert captured["key"] == "user-model-v1"


def test_feedback_tinker_accepts_explicit_orchestration(monkeypatch, tmp_path):
    from claas import api

    _mock_config(monkeypatch, "tinker")
    _seed_cache("tinker orch response")

    class _Engine:
        async def lora_exists(self, _lora_id):
            return LoraExistsPayload(exists=True)

    async def fake_run_distill(_payload):
        return DistillResponse(lora_id="user/model", metadata={})

    monkeypatch.setattr(api, "_get_training_engine", lambda: _Engine())
    monkeypatch.setattr(api, "_run_distill", fake_run_distill)
    monkeypatch.setattr(api, "_write_feedback_log", lambda _r: str(tmp_path / "log.json"))

    client = TestClient(web_app)
    response = client.post(
        "/v1/feedback",
        json={
            "requests": [
                {
                    "lora_id": "user/model",
                    "prompt": "clean prompt",
                    "response": "tinker orch response",
                    "feedback": "f",
                    "training": {},
                }
            ],
            "orchestration": {"sleep_before": True, "wake_after": True, "wake_on_failure": True, "sleep_level": 1},
        },
    )

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Additional endpoint tests
# ---------------------------------------------------------------------------


def test_root_endpoint():
    client = TestClient(web_app)
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "CLaaS API"
    assert "docs" in body


def test_init_lora_success(monkeypatch):
    from claas import api

    class _InitEngine:
        async def init_lora(self, request):
            return LoraInitResponse(lora_id=request.lora_id)

    _mock_config(monkeypatch, "tinker")
    monkeypatch.setattr(api, "_get_training_engine", lambda: _InitEngine())

    client = TestClient(web_app)
    resp = client.post(
        "/v1/lora/init",
        json={"lora_id": "test/new-lora", "base_model": "meta-llama/Llama-3.2-1B"},
    )
    assert resp.status_code == 200
    assert resp.json()["lora_id"] == "test/new-lora"


def test_init_lora_rejects_disallowed_base_model(monkeypatch):
    from claas import api

    _mock_config(monkeypatch, "local", allowed_init_base_models=frozenset({"Qwen/Qwen3-8B"}))
    monkeypatch.setattr(api, "_get_training_engine", lambda: _EngineStub(exists=True))

    client = TestClient(web_app)
    resp = client.post(
        "/v1/lora/init",
        json={"lora_id": "test/x", "base_model": "evil/model"},
    )
    assert resp.status_code == 403
    assert "not allowed" in resp.json()["detail"]


def test_list_lora_success(monkeypatch):
    from claas import api

    class _ListEngine:
        async def list_loras(self, prefix):
            return LoraListResponse(loras=["a/lora-1", "a/lora-2"])

    monkeypatch.setattr(api, "_get_training_engine", lambda: _ListEngine())

    client = TestClient(web_app)
    resp = client.get("/v1/lora", params={"prefix": "a/"})
    assert resp.status_code == 200
    assert resp.json()["loras"] == ["a/lora-1", "a/lora-2"]


def test_delete_lora_success(monkeypatch):
    from claas import api

    class _DeleteEngine:
        async def delete_lora(self, lora_id):
            return LoraDeleteResponse(deleted=True)

    monkeypatch.setattr(api, "_get_training_engine", lambda: _DeleteEngine())

    client = TestClient(web_app)
    resp = client.delete("/v1/lora", params={"lora_id": "test/x"})
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True


def test_list_lora_error(monkeypatch):
    from claas import api

    class _FailEngine:
        async def list_loras(self, prefix):
            raise ValueError("storage error")

    monkeypatch.setattr(api, "_get_training_engine", lambda: _FailEngine())

    client = TestClient(web_app)
    resp = client.get("/v1/lora")
    assert resp.status_code == 500
    assert "Failed to list" in resp.json()["detail"]


def test_init_lora_error(monkeypatch):
    from claas import api

    class _FailEngine:
        async def init_lora(self, request):
            raise RuntimeError("init failed")

    _mock_config(monkeypatch, "tinker")
    monkeypatch.setattr(api, "_get_training_engine", lambda: _FailEngine())

    client = TestClient(web_app)
    resp = client.post(
        "/v1/lora/init",
        json={"lora_id": "test/x", "base_model": "m"},
    )
    assert resp.status_code == 500
    assert "initialization failed" in resp.json()["detail"]


def test_export_lora_success(monkeypatch):
    from claas import api

    class _ExportEngine:
        async def lora_exists(self, lora_id):
            return LoraExistsPayload(exists=True)

        async def export_lora(self, lora_id):
            return LoraExportPayload(filename="test.zip", content=b"PK\x03\x04fake")

    monkeypatch.setattr(api, "_get_training_engine", lambda: _ExportEngine())

    client = TestClient(web_app)
    resp = client.get("/v1/lora/export", params={"lora_id": "test/x"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert b"PK" in resp.content


def test_health_check_healthy(monkeypatch):
    """Health endpoint returns healthy when engine responds."""
    from claas import api

    _mock_config(monkeypatch, "tinker")

    class _HealthyEngine:
        async def health(self):
            return ServiceHealth(status="healthy", error=None)

    monkeypatch.setattr(api, "get_training_engine", lambda _kind, _cfg: _HealthyEngine())

    client = TestClient(web_app)
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["worker"]["status"] == "healthy"


def test_health_check_degraded(monkeypatch):
    """Health endpoint returns degraded when engine health fails."""
    from claas import api

    _mock_config(monkeypatch, "tinker")

    class _UnhealthyEngine:
        async def health(self):
            raise ConnectionError("service down")

    monkeypatch.setattr(api, "get_training_engine", lambda _kind, _cfg: _UnhealthyEngine())

    client = TestClient(web_app)
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["worker"]["status"] == "unhealthy"
    assert "service down" in body["worker"]["error"]


def test_dashboard_renders_latest_records(monkeypatch, tmp_path):

    _mock_config(monkeypatch, "local", feedback_log_dir=str(tmp_path))

    (tmp_path / "20240101T000001-a.json").write_text(
        """
{
  "request_id": "a",
  "timestamp_utc": "2024-01-01T00:00:01Z",
  "status": "ok",
  "phase": "done",
  "lora_id": "user/model",
  "requests": [
    {
      "lora_id": "user/model",
      "prompt": "prompt-a",
      "response": "response-a",
      "feedback": "feedback-a"
    }
  ],
  "batch_samples": [
    {
      "prompt": "prompt-a",
      "response": "response-a",
      "feedback": "feedback-a",
      "response_logprobs": [-0.1],
      "prompt_token_ids": [1],
      "response_token_ids": [2],
      "user_prompt": "prompt-a"
    }
  ],
  "vllm": {
    "slept": true,
    "woke": true
  },
  "timing_ms": {
    "sleep": 1,
    "distill": 2,
    "save": 0,
    "wake": 1,
    "logprobs": 0,
    "total": 4
  },
  "distill_result": {
    "lora_id": "user/model",
    "metadata": {
      "loss": 0.1
    }
  },
  "error": null
}
""".strip(),
        encoding="utf-8",
    )

    (tmp_path / "20240101T000002-b.json").write_text(
        (tmp_path / "20240101T000001-a.json").read_text(encoding="utf-8").replace("\"a\"", "\"b\"", 1).replace("prompt-a", "prompt-b"),
        encoding="utf-8",
    )

    client = TestClient(web_app)
    response = client.get("/v1/dashboard", params={"page": 1, "per_page": 1})

    assert response.status_code == 200
    assert "CLaaS Dashboard" in response.text
    assert "<table>" in response.text
    assert "Expand" in response.text
    assert "prompt-b" in response.text
    assert "prompt-a" not in response.text
    assert "Page 1 of 2" in response.text


def test_dashboard_skips_invalid_log(monkeypatch, tmp_path):

    _mock_config(monkeypatch, "local", feedback_log_dir=str(tmp_path))
    (tmp_path / "20240101T000001-a.json").write_text("{}", encoding="utf-8")

    client = TestClient(web_app)
    response = client.get("/v1/dashboard")

    assert response.status_code == 200


def test_dashboard_truncates_prompt_preview(monkeypatch, tmp_path):

    _mock_config(monkeypatch, "local", feedback_log_dir=str(tmp_path))
    long_prompt = "x" * 400
    (tmp_path / "20240101T000003-c.json").write_text(
        f'''
{{
  "request_id": "c",
  "timestamp_utc": "2024-01-01T00:00:03Z",
  "status": "ok",
  "phase": "done",
  "lora_id": "user/model",
  "requests": [
    {{
      "lora_id": "user/model",
      "prompt": "{long_prompt}",
      "response": "response-c",
      "feedback": "feedback-c"
    }}
  ],
  "batch_samples": [
    {{
      "prompt": "{long_prompt}",
      "response": "response-c",
      "feedback": "feedback-c",
      "response_logprobs": [-0.1],
      "prompt_token_ids": [1],
      "response_token_ids": [2],
      "user_prompt": "{long_prompt}"
    }}
  ],
  "vllm": {{
    "slept": true,
    "woke": true
  }},
  "timing_ms": {{
    "sleep": 1,
    "distill": 2,
    "save": 0,
    "wake": 1,
    "logprobs": 0,
    "total": 4
  }},
  "distill_result": {{
    "lora_id": "user/model",
    "metadata": {{
      "loss": 0.1
    }}
  }},
  "error": null
}}
'''.strip(),
        encoding="utf-8",
    )

    client = TestClient(web_app)
    response = client.get("/v1/dashboard", params={"page": 1, "per_page": 1})

    assert response.status_code == 200
    assert "…" in response.text
    assert long_prompt in response.text


def test_dashboard_renders_one_row_per_batch_item(monkeypatch, tmp_path):

    _mock_config(monkeypatch, "local", feedback_log_dir=str(tmp_path))
    (tmp_path / "20240101T000004-d.json").write_text(
        """
{
  "request_id": "d",
  "timestamp_utc": "2024-01-01T00:00:04Z",
  "status": "ok",
  "phase": "done",
  "lora_id": "user/model",
  "requests": [
    {
      "lora_id": "user/model",
      "prompt": "prompt-d1",
      "response": "response-d1",
      "feedback": "feedback-d1"
    },
    {
      "lora_id": "user/model",
      "prompt": "prompt-d2",
      "response": "response-d2",
      "feedback": "feedback-d2"
    }
  ],
  "batch_samples": [
    {
      "prompt": "prompt-d1",
      "response": "response-d1",
      "feedback": "feedback-d1",
      "response_logprobs": [-0.1],
      "prompt_token_ids": [1],
      "response_token_ids": [2],
      "user_prompt": "prompt-d1"
    },
    {
      "prompt": "prompt-d2",
      "response": "response-d2",
      "feedback": "feedback-d2",
      "response_logprobs": [-0.2],
      "prompt_token_ids": [1],
      "response_token_ids": [2],
      "user_prompt": "prompt-d2"
    }
  ],
  "vllm": {
    "slept": true,
    "woke": true
  },
  "timing_ms": {
    "sleep": 1,
    "distill": 2,
    "save": 0,
    "wake": 1,
    "logprobs": 0,
    "total": 4
  },
  "distill_result": {
    "lora_id": "user/model",
    "metadata": {
      "loss": 0.1
    }
  },
  "error": null
}
""".strip(),
        encoding="utf-8",
    )

    client = TestClient(web_app)
    response = client.get("/v1/dashboard", params={"page": 1, "per_page": 1})

    assert response.status_code == 200
    assert "prompt-d1" in response.text
    assert "prompt-d2" in response.text
    # Batch is rendered as a single row with samples inside <details>
    assert "Sample 1/2" in response.text
    assert "Sample 2/2" in response.text
    assert 'id="feedback-detail-0"' in response.text
    assert "2 samples" in response.text


def test_feedback_recent_route_is_removed():
    client = TestClient(web_app)
    response = client.get("/v1/feedback/recent")
    assert response.status_code == 404
