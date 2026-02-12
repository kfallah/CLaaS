"""Tests for FastAPI endpoint behavior."""

from __future__ import annotations

from fastapi.testclient import TestClient

from claas.api import web_app


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


def test_distill_404_when_lora_missing(monkeypatch):
    from claas import api

    monkeypatch.setattr(api, "lora_exists", lambda _lora_id: False)
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

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "modal_rpc")
    monkeypatch.setattr(api, "lora_exists", lambda _lora_id: True)
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


def test_feedback_success_inplace_flow(monkeypatch):
    from claas import api

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "modal_rpc")
    calls = []
    captured = {}
    log_records = []

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
        return "/tmp/feedback-log.json"

    monkeypatch.setattr(api, "lora_exists", lambda _lora_id: True)
    monkeypatch.setattr(api.modal.Function, "from_name", fake_from_name)
    monkeypatch.setattr(api, "_vllm_post", fake_vllm_post)
    monkeypatch.setattr(api, "_write_feedback_log", fake_write_feedback_log)

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
    assert body["feedback_log_path"] == "/tmp/feedback-log.json"
    # sleep → wake → unload old LoRA → load updated LoRA
    assert calls[0] == ("/sleep", {"level": 1})
    assert calls[1] == ("/wake_up", None)
    assert calls[2] == ("/v1/unload_lora_adapter", None)
    assert calls[3] == ("/v1/load_lora_adapter", None)
    assert captured["request"]["save_in_place"] is True
    assert log_records and log_records[0]["status"] == "ok"


def test_feedback_returns_500_and_logs_error(monkeypatch):
    from claas import api

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "modal_rpc")
    log_records = []

    def fake_from_name(_app, fn_name):
        if fn_name == "DistillWorker.distill":
            return _FunctionFailureStub()
        raise AssertionError(f"unexpected modal function: {fn_name}")

    async def fake_vllm_post(_path, *, params=None, json_body=None, timeout_s=30.0):
        return None

    def fake_write_feedback_log(record):
        log_records.append(record)
        return "/tmp/feedback-log.json"

    monkeypatch.setattr(api, "lora_exists", lambda _lora_id: True)
    monkeypatch.setattr(api.modal.Function, "from_name", fake_from_name)
    monkeypatch.setattr(api, "_vllm_post", fake_vllm_post)
    monkeypatch.setattr(api, "_write_feedback_log", fake_write_feedback_log)

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

    monkeypatch.setattr(api, "lora_exists", lambda _lora_id: False)
    client = TestClient(web_app)

    response = client.get("/v1/lora/export", params={"lora_id": "user/missing"})
    assert response.status_code == 404


def test_distill_returns_500_on_worker_failure(monkeypatch):
    from claas import api

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "modal_rpc")
    monkeypatch.setattr(api, "lora_exists", lambda _lora_id: True)
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
