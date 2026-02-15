"""Tests for FastAPI endpoint behavior."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from claas.api import web_app
from claas.types import (
    DistillResponse,
    LoraDeleteResponse,
    LoraExistsPayload,
    LoraExportPayload,
    LoraInitResponse,
    LoraListResponse,
    LoraRuntimeRef,
    ServiceHealth,
)


@pytest.fixture()
def api_client():
    """FastAPI TestClient for the web_app."""
    return TestClient(web_app)


@pytest.fixture()
def tinker_mode(monkeypatch):
    """Set the distill execution mode to tinker (skips vLLM orchestration)."""
    from claas import api

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "tinker")


@pytest.fixture()
def modal_mode(monkeypatch):
    """Set the distill execution mode to modal."""
    from claas import api

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "modal")


@pytest.fixture()
def noop_feedback_log(monkeypatch, tmp_path):
    """Stub out feedback log writing."""
    from claas import api

    log_path = str(tmp_path / "feedback-log.json")
    monkeypatch.setattr(api, "_write_feedback_log", lambda _r: log_path)
    return log_path


@pytest.fixture()
def noop_vllm(monkeypatch):
    """Stub out all vLLM interactions."""
    from claas import api

    async def noop_post(_path, *, params=None, json_body=None, timeout_s=30.0):
        pass

    async def noop_wait():
        pass

    async def noop_fetch(_prompt, _response, _model, _timeout_s=60.0):
        return [-0.1, -0.2]

    monkeypatch.setattr(api, "_vllm_post", noop_post)
    monkeypatch.setattr(api, "_wait_for_vllm_idle", noop_wait)
    monkeypatch.setattr(api, "_fetch_rollout_logprobs", noop_fetch)


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


async def _noop_fetch_logprobs(_prompt, _response, _model, _timeout_s=60.0):
    return [-0.1, -0.2]


class _FunctionFailureStub:
    def __init__(self):
        self.remote = _RemoteCallFailure()


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


def test_distill_404_when_lora_missing(monkeypatch):
    from claas import api

    monkeypatch.setattr(api, "_get_training_engine", lambda: _EngineStub(exists=False))
    client = TestClient(web_app)

    response = client.post(
        "/v1/distill",
        json={
            "lora_id": "user/model",
            "prompt": "p",
            "response": "r",
            "feedback": "f",
            "training": {},
        },
    )

    assert response.status_code == 404
    assert "LoRA not found" in response.json()["detail"]


def test_distill_success(monkeypatch):
    from claas import api

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "modal")
    monkeypatch.setattr(api, "_get_training_engine", lambda: _EngineStub(exists=True))
    monkeypatch.setattr(
        api.modal.Function,
        "from_name",
        lambda *_args, **_kwargs: _FunctionStub(
            {"lora_id": "user/model-v2", "metadata": {"tokens_processed": 3}}
        ),
    )
    client = TestClient(web_app)

    response = client.post(
        "/v1/distill",
        json={
            "lora_id": "user/model",
            "prompt": "p",
            "response": "r",
            "feedback": "f",
            "training": {},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["lora_id"] == "user/model-v2"
    assert body["metadata"]["tokens_processed"] == 3


def test_feedback_success_inplace_flow(monkeypatch, tmp_path):
    from claas import api

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "modal")
    calls = []
    captured = {}
    log_records = []
    log_path = str(tmp_path / "feedback-log.json")

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
    monkeypatch.setattr(api, "_fetch_rollout_logprobs", _noop_fetch_logprobs)

    client = TestClient(web_app)
    response = client.post(
        "/v1/feedback",
        json={
            "lora_id": "user/model",
            "prompt": "p",
            "response": "r",
            "feedback": "f",
            "training": {"teacher_mode": "self"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["lora_id"] == "user/model"
    assert body["feedback_log_path"] == log_path
    # drain → sleep → wake → unload old LoRA → load updated LoRA
    assert calls[0] == ("_wait_for_vllm_idle",)
    assert calls[1] == ("/sleep", {"level": 1})
    assert calls[2] == ("/wake_up", None)
    assert calls[3] == ("/v1/unload_lora_adapter", None)
    assert calls[4] == ("/v1/load_lora_adapter", None)
    assert captured["request"]["save_in_place"] is True
    assert log_records and log_records[0]["status"] == "ok"


def test_feedback_returns_500_and_logs_error(monkeypatch, tmp_path):
    from claas import api

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "modal")
    log_records = []
    log_path = str(tmp_path / "feedback-log.json")

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
    monkeypatch.setattr(api, "_fetch_rollout_logprobs", _noop_fetch_logprobs)

    client = TestClient(web_app)
    response = client.post(
        "/v1/feedback",
        json={
            "lora_id": "user/model",
            "prompt": "p",
            "response": "r",
            "feedback": "f",
            "training": {"teacher_mode": "self"},
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


def test_distill_returns_500_on_worker_failure(monkeypatch):
    from claas import api

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "modal")
    monkeypatch.setattr(api, "_get_training_engine", lambda: _EngineStub(exists=True))
    monkeypatch.setattr(
        api.modal.Function,
        "from_name",
        lambda *_args, **_kwargs: _FunctionFailureStub(),
    )
    client = TestClient(web_app)

    response = client.post(
        "/v1/distill",
        json={
            "lora_id": "user/model",
            "prompt": "p",
            "response": "r",
            "feedback": "f",
            "training": {},
        },
    )
    assert response.status_code == 500
    assert "Distillation failed" in response.json()["detail"]


def test_feedback_drain_timeout_returns_503(monkeypatch, tmp_path):
    """A drain timeout produces a 503 response."""
    from claas import api

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "modal")

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
            "lora_id": "user/model",
            "prompt": "p",
            "response": "r",
            "feedback": "f",
            "training": {"teacher_mode": "self"},
        },
    )

    assert response.status_code == 503
    assert "vLLM not idle" in response.json()["detail"]


def test_feedback_fetches_rollout_logprobs(monkeypatch, tmp_path):
    """Rollout logprobs are fetched and forwarded to the distill worker."""
    from claas import api

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "modal")
    captured = {}

    def fake_from_name(_app, fn_name):
        if fn_name == "DistillWorker.distill":
            return _FunctionStub(
                {"lora_id": "user/model", "metadata": {"tokens_processed": 1}},
                capture=captured,
            )
        raise AssertionError(f"unexpected modal function: {fn_name}")

    async def fake_fetch(_prompt, _response, _model, _timeout_s=60.0):
        return [-0.5, -1.2, -0.3]

    monkeypatch.setattr(api, "_get_training_engine", lambda: _EngineStub(exists=True))
    monkeypatch.setattr(api.modal.Function, "from_name", fake_from_name)
    monkeypatch.setattr(api, "_vllm_post", lambda *_a, **_kw: _noop_coro())
    monkeypatch.setattr(api, "_wait_for_vllm_idle", lambda: _noop_coro())
    monkeypatch.setattr(api, "_write_feedback_log", lambda _r: str(tmp_path / "log.json"))
    monkeypatch.setattr(api, "_fetch_rollout_logprobs", fake_fetch)

    client = TestClient(web_app)
    response = client.post(
        "/v1/feedback",
        json={
            "lora_id": "user/model",
            "prompt": "p",
            "response": "r",
            "feedback": "f",
            "training": {"teacher_mode": "self"},
        },
    )

    assert response.status_code == 200
    assert captured["request"]["rollout_logprobs"] == [-0.5, -1.2, -0.3]


def test_feedback_logprobs_fetch_failure_continues(monkeypatch, tmp_path):
    """A failing logprobs fetch logs a warning but returns 200."""
    import httpx as _httpx

    from claas import api

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "modal")
    captured = {}

    def fake_from_name(_app, fn_name):
        if fn_name == "DistillWorker.distill":
            return _FunctionStub(
                {"lora_id": "user/model", "metadata": {"tokens_processed": 1}},
                capture=captured,
            )
        raise AssertionError(f"unexpected modal function: {fn_name}")

    async def failing_fetch(_prompt, _response, _model, _timeout_s=60.0):
        raise _httpx.HTTPError("connection refused")

    monkeypatch.setattr(api, "_get_training_engine", lambda: _EngineStub(exists=True))
    monkeypatch.setattr(api.modal.Function, "from_name", fake_from_name)
    monkeypatch.setattr(api, "_vllm_post", lambda *_a, **_kw: _noop_coro())
    monkeypatch.setattr(api, "_wait_for_vllm_idle", lambda: _noop_coro())
    monkeypatch.setattr(api, "_write_feedback_log", lambda _r: str(tmp_path / "log.json"))
    monkeypatch.setattr(api, "_fetch_rollout_logprobs", failing_fetch)

    client = TestClient(web_app)
    response = client.post(
        "/v1/feedback",
        json={
            "lora_id": "user/model",
            "prompt": "p",
            "response": "r",
            "feedback": "f",
            "training": {"teacher_mode": "self"},
        },
    )

    assert response.status_code == 200
    assert captured["request"]["rollout_logprobs"] is None


def test_feedback_skips_logprobs_when_provided(monkeypatch, tmp_path):
    """When rollout_logprobs is already provided, the fetch is skipped."""
    from claas import api

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "modal")
    captured = {}
    fetch_called = []

    def fake_from_name(_app, fn_name):
        if fn_name == "DistillWorker.distill":
            return _FunctionStub(
                {"lora_id": "user/model", "metadata": {"tokens_processed": 1}},
                capture=captured,
            )
        raise AssertionError(f"unexpected modal function: {fn_name}")

    async def spy_fetch(_prompt, _response, _model, _timeout_s=60.0):
        fetch_called.append(True)
        return [-9.9]

    monkeypatch.setattr(api, "_get_training_engine", lambda: _EngineStub(exists=True))
    monkeypatch.setattr(api.modal.Function, "from_name", fake_from_name)
    monkeypatch.setattr(api, "_vllm_post", lambda *_a, **_kw: _noop_coro())
    monkeypatch.setattr(api, "_wait_for_vllm_idle", lambda: _noop_coro())
    monkeypatch.setattr(api, "_write_feedback_log", lambda _r: str(tmp_path / "log.json"))
    monkeypatch.setattr(api, "_fetch_rollout_logprobs", spy_fetch)

    client = TestClient(web_app)
    response = client.post(
        "/v1/feedback",
        json={
            "lora_id": "user/model",
            "prompt": "p",
            "response": "r",
            "feedback": "f",
            "rollout_logprobs": [-0.1, -0.2],
            "training": {"teacher_mode": "self"},
        },
    )

    assert response.status_code == 200
    assert captured["request"]["rollout_logprobs"] == [-0.1, -0.2]
    assert fetch_called == []


async def _noop_coro():
    pass



def test_feedback_uses_resolved_lock_key(monkeypatch, tmp_path):
    from claas import api

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "modal")

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
            "lora_id": "user/model-latest",
            "prompt": "p",
            "response": "r",
            "feedback": "f",
            "training": {"teacher_mode": "self"},
            "orchestration": {"sleep_before": False, "wake_after": False},
        },
    )

    assert response.status_code == 200
    assert captured["key"] == "user-model-v1"


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

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "tinker")
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

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "local")
    monkeypatch.setattr(api, "ALLOWED_INIT_BASE_MODELS", {"Qwen/Qwen3-8B"})
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

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "tinker")
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

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "tinker")

    class _HealthyEngine:
        async def health(self):
            return ServiceHealth(status="healthy", error=None)

    monkeypatch.setattr(api, "get_training_engine", lambda _kind: _HealthyEngine())

    client = TestClient(web_app)
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["worker"]["status"] == "healthy"
    assert body["teacher"]["status"] == "healthy"


def test_health_check_degraded(monkeypatch):
    """Health endpoint returns degraded when engine health fails."""
    from claas import api

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "tinker")

    class _UnhealthyEngine:
        async def health(self):
            raise ConnectionError("service down")

    monkeypatch.setattr(api, "get_training_engine", lambda _kind: _UnhealthyEngine())

    client = TestClient(web_app)
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["worker"]["status"] == "unhealthy"
    assert "service down" in body["worker"]["error"]
