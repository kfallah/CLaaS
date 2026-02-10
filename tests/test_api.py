"""Tests for FastAPI endpoint behavior."""

from __future__ import annotations

from fastapi.testclient import TestClient

from claas.api import web_app


class _RemoteCall:
    def __init__(self, payload):
        self.payload = payload

    async def aio(self, _request):
        return self.payload


class _WorkerSuccess:
    def __init__(self):
        self.distill = type("DistillMethod", (), {"remote": _RemoteCall({"lora_id": "user/model-v2", "metadata": {"tokens_processed": 3}})})()

    async def health_check(self):
        return {"status": "healthy"}


class _RemoteCallFailure:
    async def aio(self, _request):
        raise RuntimeError("modal rpc failure")


class _WorkerFailure:
    def __init__(self):
        self.distill = type("DistillMethod", (), {"remote": _RemoteCallFailure()})()


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

    monkeypatch.setattr(api, "lora_exists", lambda _lora_id: True)
    monkeypatch.setattr(api, "DistillWorker", _WorkerSuccess)
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


def test_export_404_when_missing(monkeypatch):
    from claas import api

    monkeypatch.setattr(api, "lora_exists", lambda _lora_id: False)
    client = TestClient(web_app)

    response = client.get("/v1/lora/export", params={"lora_id": "user/missing"})
    assert response.status_code == 404


def test_distill_returns_500_on_worker_failure(monkeypatch):
    from claas import api

    monkeypatch.setattr(api, "lora_exists", lambda _lora_id: True)
    monkeypatch.setattr(api, "DistillWorker", _WorkerFailure)
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
