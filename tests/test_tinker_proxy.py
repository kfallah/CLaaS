"""Tests for the Tinker inference proxy FastAPI endpoints.

The tinker SDK is available but we mock the ServiceClient/SamplingClient
so tests don't need a real Tinker API key or GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

tinker = pytest.importorskip("tinker")  # noqa: F841

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_holder():
    """Reset the proxy singleton between tests."""
    from claas.proxy.tinker_inference_proxy import _holder

    _holder._service = None
    _holder._sampler = None
    _holder._tokenizer = None
    _holder._renderer = None
    _holder._model_path = None
    yield
    _holder._service = None
    _holder._sampler = None
    _holder._tokenizer = None
    _holder._renderer = None
    _holder._model_path = None


@pytest.fixture()
def proxy_client():
    """TestClient for the proxy FastAPI app."""
    from claas.proxy.tinker_inference_proxy import app

    return TestClient(app)


class TestModelsEndpoint:
    def test_list_models(self, proxy_client):
        resp = proxy_client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["owned_by"] == "tinker"


class TestSamplerStatus:
    def test_status_before_refresh(self, proxy_client):
        resp = proxy_client.get("/v1/sampler/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_path"] is None
        assert "base_model" in body

    def test_status_after_manual_set(self, proxy_client):
        from claas.proxy.tinker_inference_proxy import _holder

        _holder._model_path = "tinker://run-123/weights/step-1"
        resp = proxy_client.get("/v1/sampler/status")
        assert resp.status_code == 200
        assert resp.json()["model_path"] == "tinker://run-123/weights/step-1"


class TestSamplerRefresh:
    def test_refresh_calls_holder(self, proxy_client):
        from claas.proxy.tinker_inference_proxy import _holder

        with patch.object(_holder, "refresh") as mock_refresh:
            resp = proxy_client.post(
                "/v1/sampler/refresh",
                json={"model_path": "tinker://run-1/weights/0001"},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        mock_refresh.assert_called_once_with(model_path="tinker://run-1/weights/0001")

class TestHolderInternals:
    """Test _ensure() and refresh() with mocked Tinker SDK."""

    def test_ensure_initializes_once(self):
        from claas.proxy.tinker_inference_proxy import _SamplerHolder

        holder = _SamplerHolder()
        mock_sampler = MagicMock()
        mock_sampler.get_tokenizer.return_value = MagicMock()
        mock_service = MagicMock()
        mock_service.create_sampling_client.return_value = mock_sampler

        with patch("claas.proxy.tinker_inference_proxy.tinker.ServiceClient", return_value=mock_service), \
             patch("claas.proxy.tinker_inference_proxy.model_info.get_recommended_renderer_name", return_value="chatml"), \
             patch("claas.proxy.tinker_inference_proxy.get_renderer", return_value=MagicMock()):
            # First call initializes
            holder._ensure()
            assert holder._sampler is mock_sampler
            assert holder._service is mock_service

            # Second call is a no-op (already initialized)
            mock_service.create_sampling_client.reset_mock()
            holder._ensure()
            mock_service.create_sampling_client.assert_not_called()

    def test_refresh_with_model_path(self):
        from claas.proxy.tinker_inference_proxy import _SamplerHolder

        holder = _SamplerHolder()
        mock_sampler = MagicMock()
        mock_sampler.get_tokenizer.return_value = MagicMock()
        mock_service = MagicMock()
        mock_service.create_sampling_client.return_value = mock_sampler
        holder._service = mock_service

        with patch("claas.proxy.tinker_inference_proxy.model_info.get_recommended_renderer_name", return_value="chatml"), \
             patch("claas.proxy.tinker_inference_proxy.get_renderer", return_value=MagicMock()):
            holder.refresh(model_path="tinker://run-1/weights/step-1")

        assert holder._model_path == "tinker://run-1/weights/step-1"
        mock_service.create_sampling_client.assert_called_once_with(
            model_path="tinker://run-1/weights/step-1"
        )

    def test_refresh_without_model_path_uses_base(self):
        from claas.proxy.tinker_inference_proxy import _base_model, _SamplerHolder

        holder = _SamplerHolder()
        mock_sampler = MagicMock()
        mock_sampler.get_tokenizer.return_value = MagicMock()
        mock_service = MagicMock()
        mock_service.create_sampling_client.return_value = mock_sampler
        holder._service = mock_service

        with patch("claas.proxy.tinker_inference_proxy.model_info.get_recommended_renderer_name", return_value="chatml"), \
             patch("claas.proxy.tinker_inference_proxy.get_renderer", return_value=MagicMock()):
            holder.refresh(model_path=None)

        assert holder._model_path is None
        mock_service.create_sampling_client.assert_called_once_with(
            base_model=_base_model()
        )

    def test_refresh_creates_service_if_missing(self):
        from claas.proxy.tinker_inference_proxy import _SamplerHolder

        holder = _SamplerHolder()
        assert holder._service is None

        mock_sampler = MagicMock()
        mock_sampler.get_tokenizer.return_value = MagicMock()
        mock_service = MagicMock()
        mock_service.create_sampling_client.return_value = mock_sampler

        with patch("claas.proxy.tinker_inference_proxy.tinker.ServiceClient", return_value=mock_service), \
             patch("claas.proxy.tinker_inference_proxy.model_info.get_recommended_renderer_name", return_value="chatml"), \
             patch("claas.proxy.tinker_inference_proxy.get_renderer", return_value=MagicMock()):
            holder.refresh(model_path="tinker://x")

        assert holder._service is mock_service
        assert holder._model_path == "tinker://x"


def _make_mock_sampler(text_output="Hello!"):
    """Create a mock sampler that returns a fixed response."""
    import tinker.types as T

    mock_seq = MagicMock()
    mock_seq.tokens = [1, 2, 3]

    mock_response = MagicMock(spec=T.SampleResponse)
    mock_response.sequences = [mock_seq]

    mock_future = MagicMock()
    mock_future.result.return_value = mock_response

    mock_sampler = MagicMock(spec_set=["sample", "get_tokenizer"])
    mock_sampler.sample.return_value = mock_future

    return mock_sampler, mock_response


def _make_mock_tokenizer():
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2, 3]
    mock_tok.decode.return_value = "decoded text"
    return mock_tok


def _make_mock_renderer(content="Mocked response"):

    mock_renderer = MagicMock()
    mock_input = MagicMock()
    mock_input.length = 5
    mock_renderer.build_generation_prompt.return_value = mock_input
    mock_renderer.get_stop_sequences.return_value = ["<|im_end|>"]
    mock_renderer.parse_response.return_value = ({"content": content}, None)
    return mock_renderer


def _patch_holder(holder, sampler, tokenizer, renderer):
    holder._sampler = sampler
    holder._tokenizer = tokenizer
    holder._renderer = renderer
    holder._service = MagicMock()  # prevent _ensure() from running


class TestChatCompletions:
    def test_non_streaming(self, proxy_client):
        from claas.proxy.tinker_inference_proxy import _holder

        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer()
        renderer = _make_mock_renderer("Hello world")
        _patch_holder(_holder, sampler, tokenizer, renderer)

        resp = proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 32,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"] == "Hello world"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["prompt_tokens"] == 5
        assert body["usage"]["completion_tokens"] == 3

    def test_streaming(self, proxy_client):
        from claas.proxy.tinker_inference_proxy import _holder

        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer()
        renderer = _make_mock_renderer("Streamed")
        _patch_holder(_holder, sampler, tokenizer, renderer)

        resp = proxy_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        text = resp.text
        assert "data:" in text
        assert "[DONE]" in text

class TestCompletions:
    def test_non_streaming(self, proxy_client):
        from claas.proxy.tinker_inference_proxy import _holder

        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer()
        renderer = _make_mock_renderer()
        _patch_holder(_holder, sampler, tokenizer, renderer)

        resp = proxy_client.post(
            "/v1/completions",
            json={"prompt": "Once upon a time", "max_tokens": 32},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "text_completion"
        assert body["choices"][0]["text"] == "decoded text"
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_streaming(self, proxy_client):
        from claas.proxy.tinker_inference_proxy import _holder

        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer()
        renderer = _make_mock_renderer()
        _patch_holder(_holder, sampler, tokenizer, renderer)

        resp = proxy_client.post(
            "/v1/completions",
            json={"prompt": "Hello", "stream": True},
        )
        assert resp.status_code == 200
        assert "[DONE]" in resp.text


class TestHelperFunctions:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("hello", "hello"),
            (None, ""),
            (["a", "b"], "a\nb"),
            ([{"type": "text", "text": "hello"}], "hello"),
            ([{"content": "world"}], "world"),
            ({"text": "hi"}, "hi"),
            (42, "42"),
        ],
    )
    def test_coerce_content(self, raw, expected):
        from claas.proxy.tinker_inference_proxy import _coerce_content

        assert _coerce_content(raw) == expected

    def test_bounded_int_and_float(self):
        from claas.proxy.tinker_inference_proxy import _bounded_float, _bounded_int

        assert _bounded_int(None, default=10, minimum=1, maximum=100) == 10
        assert _bounded_int(999, default=10, minimum=1, maximum=100) == 100
        assert _bounded_int(-5, default=10, minimum=1, maximum=100) == 1
        assert _bounded_float(None, default=0.5, minimum=0.0, maximum=1.0) == 0.5
        assert _bounded_float(5.0, default=0.5, minimum=0.0, maximum=1.0) == 1.0
        assert _bounded_float(-1.0, default=0.5, minimum=0.0, maximum=1.0) == 0.0
