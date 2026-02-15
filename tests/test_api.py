"""Tests for FastAPI endpoint behavior."""

from __future__ import annotations

import asyncio

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
        from claas.types import LoraExistsPayload

        return LoraExistsPayload(exists=self._exists)

    async def lora_runtime_ref(self, lora_id):
        from claas.types import LoraRuntimeRef

        return LoraRuntimeRef(vllm_name=lora_id, lora_path=f"/loras/{lora_id}")

    async def distill(self, payload):
        import modal

        from claas.types import DistillResponse

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


def test_feedback_calls_drain_before_sleep(monkeypatch, tmp_path):
    """_wait_for_vllm_idle is called before /sleep."""
    from claas import api

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "modal")
    order = []

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
    # drain must come before /pause
    assert order[0] == "drain"
    assert order[1] == "/pause"


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
    from claas.types import DistillResponse, LoraExistsPayload

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


def test_feedback_tinker_accepts_default_orchestration(monkeypatch, tmp_path):
    from claas import api
    from claas.types import DistillResponse, LoraExistsPayload

    monkeypatch.setattr(api, "DISTILL_EXECUTION_MODE", "tinker")

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
            "lora_id": "user/model",
            "prompt": "p",
            "response": "r",
            "feedback": "f",
            "training": {"teacher_mode": "self"},
        },
    )

    assert response.status_code == 200


def test_recent_feedback_dashboard_renders_latest_records(monkeypatch, tmp_path):
    from claas import api

    monkeypatch.setattr(api, "FEEDBACK_LOG_DIR", str(tmp_path))

    (tmp_path / "20240101T000001-a.json").write_text(
        """
{
  "request_id": "a",
  "timestamp_utc": "2024-01-01T00:00:01Z",
  "status": "ok",
  "phase": "done",
  "lora_id": "user/model",
  "teacher_mode": "self",
  "request": {
    "lora_id": "user/model",
    "prompt": "prompt-a",
    "response": "response-a",
    "feedback": "feedback-a",
    "rollout_logprobs": null,
    "training": {
      "learning_rate": 0.0001,
      "alpha": 0.5,
      "is_clip": 5.0,
      "max_grad_norm": 1.0,
      "kl_reg_weight": 0.001,
      "teacher_top_k": 100,
      "teacher_mode": "self"
    },
    "orchestration": {
      "sleep_before": true,
      "wake_after": true,
      "wake_on_failure": true,
      "sleep_level": 1
    }
  },
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
    response = client.get("/v1/feedback/recent", params={"limit": 1})

    assert response.status_code == 200
    assert "CLaaS Recent Feedback" in response.text
    assert "<table>" in response.text
    assert "Expand" in response.text
    assert "prompt-b" in response.text
    assert "prompt-a" not in response.text


def test_recent_feedback_dashboard_rejects_invalid_log(monkeypatch, tmp_path):
    from claas import api

    monkeypatch.setattr(api, "FEEDBACK_LOG_DIR", str(tmp_path))
    (tmp_path / "20240101T000001-a.json").write_text("{}", encoding="utf-8")

    client = TestClient(web_app)
    response = client.get("/v1/feedback/recent")

    assert response.status_code == 500


def test_recent_feedback_dashboard_truncates_prompt_preview(monkeypatch, tmp_path):
    from claas import api

    monkeypatch.setattr(api, "FEEDBACK_LOG_DIR", str(tmp_path))
    long_prompt = "x" * 400
    (tmp_path / "20240101T000003-c.json").write_text(
        f'''
{{
  "request_id": "c",
  "timestamp_utc": "2024-01-01T00:00:03Z",
  "status": "ok",
  "phase": "done",
  "lora_id": "user/model",
  "teacher_mode": "self",
  "request": {{
    "lora_id": "user/model",
    "prompt": "{long_prompt}",
    "response": "response-c",
    "feedback": "feedback-c",
    "rollout_logprobs": null,
    "training": {{
      "learning_rate": 0.0001,
      "alpha": 0.5,
      "is_clip": 5.0,
      "max_grad_norm": 1.0,
      "kl_reg_weight": 0.001,
      "teacher_top_k": 100,
      "teacher_mode": "self"
    }},
    "orchestration": {{
      "sleep_before": true,
      "wake_after": true,
      "wake_on_failure": true,
      "sleep_level": 1
    }}
  }},
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
    response = client.get("/v1/feedback/recent", params={"limit": 1})

    assert response.status_code == 200
    assert "…" in response.text
    assert long_prompt in response.text
