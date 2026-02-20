"""Tests for the inference backend abstraction (Tinker mode).

The tinker SDK is available but we mock the ServiceClient/SamplingClient
so tests don't need a real Tinker API key or GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

tinker = pytest.importorskip("tinker")  # noqa: F841

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset the inference backend and config between tests."""
    from claas.api import configure_web_app
    from claas.core.config import load_core_config

    configure_web_app(load_core_config("tinker"))
    yield
    configure_web_app(load_core_config("tinker"))


@pytest.fixture()
def api_client():
    """TestClient for the CLaaS API FastAPI app."""
    from claas.api import web_app

    return TestClient(web_app)


def _make_mock_sampler(text_output="Hello!"):
    """Create a mock sampler that returns a fixed response."""
    import tinker.types as T

    mock_seq = MagicMock()
    mock_seq.tokens = [1, 2, 3]

    mock_response = MagicMock(spec=T.SampleResponse)
    mock_response.sequences = [mock_seq]

    mock_future = MagicMock()
    mock_future.result.return_value = mock_response

    mock_sampler = MagicMock(spec_set=["sample", "get_tokenizer", "compute_logprobs"])
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


def _get_tinker_backend(api_client):
    """Force backend creation and return the TinkerBackend instance."""
    from claas.api import web_app
    from claas.inference.tinker import TinkerBackend

    backend = web_app.state.inference_backend
    assert isinstance(backend, TinkerBackend)
    return backend


class TestModelsEndpoint:
    def test_list_models(self, api_client):
        _get_tinker_backend(api_client)
        resp = api_client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["owned_by"] == "tinker"


class TestSamplerStatus:
    def test_status_before_refresh(self, api_client):
        _get_tinker_backend(api_client)
        resp = api_client.get("/v1/sampler/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["model_path"] is None
        assert "base_model" in body

    def test_status_after_manual_set(self, api_client):
        backend = _get_tinker_backend(api_client)
        backend.holder._model_path = "tinker://run-123/weights/step-1"
        resp = api_client.get("/v1/sampler/status")
        assert resp.status_code == 200
        assert resp.json()["model_path"] == "tinker://run-123/weights/step-1"


class TestSamplerRefresh:
    def test_refresh_calls_holder(self, api_client):
        backend = _get_tinker_backend(api_client)
        with patch.object(backend.holder, "refresh") as mock_refresh:
            resp = api_client.post(
                "/v1/sampler/refresh",
                json={"model_path": "tinker://run-1/weights/0001"},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        mock_refresh.assert_called_once_with(model_path="tinker://run-1/weights/0001")


class TestHolderInternals:
    """Test _ensure() and refresh() with mocked Tinker SDK."""

    def test_ensure_initializes_once(self):
        from claas.inference.tinker import _SamplerHolder

        holder = _SamplerHolder()
        mock_sampler = MagicMock()
        mock_sampler.get_tokenizer.return_value = MagicMock()
        mock_service = MagicMock()
        mock_service.create_sampling_client.return_value = mock_sampler

        with patch("tinker.ServiceClient", return_value=mock_service), \
             patch("tinker_cookbook.model_info.get_recommended_renderer_name", return_value="chatml"), \
             patch("tinker_cookbook.renderers.get_renderer", return_value=MagicMock()):
            holder._ensure()
            assert holder._sampler is mock_sampler
            assert holder._service is mock_service

            mock_service.create_sampling_client.reset_mock()
            holder._ensure()
            mock_service.create_sampling_client.assert_not_called()

    def test_refresh_with_model_path(self):
        from claas.inference.tinker import _SamplerHolder

        holder = _SamplerHolder()
        mock_sampler = MagicMock()
        mock_sampler.get_tokenizer.return_value = MagicMock()
        mock_service = MagicMock()
        mock_service.create_sampling_client.return_value = mock_sampler
        holder._service = mock_service

        with patch("tinker_cookbook.model_info.get_recommended_renderer_name", return_value="chatml"), \
             patch("tinker_cookbook.renderers.get_renderer", return_value=MagicMock()):
            holder.refresh(model_path="tinker://run-1/weights/step-1")

        assert holder._model_path == "tinker://run-1/weights/step-1"
        mock_service.create_sampling_client.assert_called_once_with(
            model_path="tinker://run-1/weights/step-1"
        )

    def test_refresh_without_model_path_uses_base(self):
        from claas.inference.tinker import _SamplerHolder

        holder = _SamplerHolder()
        mock_sampler = MagicMock()
        mock_sampler.get_tokenizer.return_value = MagicMock()
        mock_service = MagicMock()
        mock_service.create_sampling_client.return_value = mock_sampler
        holder._service = mock_service

        with patch("tinker_cookbook.model_info.get_recommended_renderer_name", return_value="chatml"), \
             patch("tinker_cookbook.renderers.get_renderer", return_value=MagicMock()):
            holder.refresh(model_path=None)

        assert holder._model_path is None

    def test_refresh_creates_service_if_missing(self):
        from claas.inference.tinker import _SamplerHolder

        holder = _SamplerHolder()
        assert holder._service is None

        mock_sampler = MagicMock()
        mock_sampler.get_tokenizer.return_value = MagicMock()
        mock_service = MagicMock()
        mock_service.create_sampling_client.return_value = mock_sampler

        with patch("tinker.ServiceClient", return_value=mock_service), \
             patch("tinker_cookbook.model_info.get_recommended_renderer_name", return_value="chatml"), \
             patch("tinker_cookbook.renderers.get_renderer", return_value=MagicMock()):
            holder.refresh(model_path="tinker://x")

        assert holder._service is mock_service
        assert holder._model_path == "tinker://x"


class TestChatCompletions:
    def test_non_streaming(self, api_client):
        backend = _get_tinker_backend(api_client)
        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer()
        renderer = _make_mock_renderer("Hello world")
        _patch_holder(backend.holder, sampler, tokenizer, renderer)

        resp = api_client.post(
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

    def test_streaming(self, api_client):
        backend = _get_tinker_backend(api_client)
        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer()
        renderer = _make_mock_renderer("Streamed")
        _patch_holder(backend.holder, sampler, tokenizer, renderer)

        resp = api_client.post(
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
    def test_non_streaming(self, api_client):
        backend = _get_tinker_backend(api_client)
        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer()
        renderer = _make_mock_renderer()
        _patch_holder(backend.holder, sampler, tokenizer, renderer)

        resp = api_client.post(
            "/v1/completions",
            json={"prompt": "Once upon a time", "max_tokens": 32},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "text_completion"
        assert body["choices"][0]["text"] == "decoded text"
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_streaming(self, api_client):
        backend = _get_tinker_backend(api_client)
        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer()
        renderer = _make_mock_renderer()
        _patch_holder(backend.holder, sampler, tokenizer, renderer)

        resp = api_client.post(
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
        from claas.inference.helpers import coerce_content

        assert coerce_content(raw) == expected

    def test_bounded_int_and_float(self):
        from claas.inference.helpers import bounded_float, bounded_int

        assert bounded_int(None, default=10, minimum=1, maximum=100) == 10
        assert bounded_int(999, default=10, minimum=1, maximum=100) == 100
        assert bounded_int(-5, default=10, minimum=1, maximum=100) == 1
        assert bounded_float(None, default=0.5, minimum=0.0, maximum=1.0) == 0.5
        assert bounded_float(5.0, default=0.5, minimum=0.0, maximum=1.0) == 1.0
        assert bounded_float(-1.0, default=0.5, minimum=0.0, maximum=1.0) == 0.0


class TestScoreEndpoint:
    def test_score_returns_logprobs(self, api_client):
        backend = _get_tinker_backend(api_client)

        mock_sampler = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = [None, -1.0, -2.0, -0.5, -0.3]
        mock_sampler.compute_logprobs.return_value = mock_future

        mock_tokenizer = MagicMock()

        def _encode(text, add_special_tokens=True):
            if add_special_tokens:
                return [10, 20, 30]
            return [40, 50]

        mock_tokenizer.encode.side_effect = _encode
        mock_tokenizer.decode.side_effect = lambda ids: f"tok{ids[0]}"

        mock_renderer = _make_mock_renderer()
        _patch_holder(backend.holder, mock_sampler, mock_tokenizer, mock_renderer)

        resp = api_client.post(
            "/v1/score",
            json={"prompt": "prompt text", "completion": "completion text"},
        )
        assert resp.status_code == 200
        body = resp.json()

        assert body["prompt_tokens"] == 3
        assert body["completion_tokens"] == 2
        assert body["logprobs"] == pytest.approx([-0.5, -0.3])
        assert body["tokens"] == ["tok40", "tok50"]
        assert body["logprob_sum"] == pytest.approx(-0.8)

    def test_score_handles_none_logprobs(self, api_client):
        backend = _get_tinker_backend(api_client)

        mock_sampler = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = [None, -1.0, None]
        mock_sampler.compute_logprobs.return_value = mock_future

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text, add_special_tokens=True: (
            [10] if add_special_tokens else [20, 30]
        )
        mock_tokenizer.decode.side_effect = lambda ids: f"t{ids[0]}"

        _patch_holder(backend.holder, mock_sampler, mock_tokenizer, _make_mock_renderer())

        resp = api_client.post(
            "/v1/score",
            json={"prompt": "p", "completion": "c"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["logprobs"] == pytest.approx([-1.0, 0.0])
        assert body["logprob_sum"] == pytest.approx(-1.0)

    def test_score_rejects_missing_prompt(self, api_client):
        _get_tinker_backend(api_client)
        resp = api_client.post(
            "/v1/score",
            json={"completion": "hello"},
        )
        assert resp.status_code == 422


class TestCompletionCache:
    """Verify that the completion cache strips <think> blocks before hashing."""

    def _populate_and_lookup(self, api_client, content, visible_text):
        """Helper: populate cache with content, look up by hash of visible_text."""
        import hashlib

        from claas.inference.cache import completion_cache

        backend = _get_tinker_backend(api_client)

        completion_cache._store.clear()

        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer()
        renderer = _make_mock_renderer(content)
        _patch_holder(backend.holder, sampler, tokenizer, renderer)

        resp = api_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 200

        visible_hash = hashlib.sha256(visible_text.encode("utf-8")).hexdigest()
        return api_client.get(
            "/v1/completions/raw",
            params={"content_hash": visible_hash},
        )

    def test_proper_think_tags(self, api_client):
        raw_resp = self._populate_and_lookup(
            api_client,
            content="<think>thinking</think>The answer",
            visible_text="The answer",
        )
        assert raw_resp.status_code == 200

    def test_orphaned_close_tag(self, api_client):
        raw_resp = self._populate_and_lookup(
            api_client,
            content="internal reasoning\n</think>\n\nThe answer",
            visible_text="The answer",
        )
        assert raw_resp.status_code == 200

    def test_no_thinking(self, api_client):
        raw_resp = self._populate_and_lookup(
            api_client,
            content="Just a plain answer",
            visible_text="Just a plain answer",
        )
        assert raw_resp.status_code == 200

    def test_full_content_hash_misses(self, api_client):
        import hashlib

        from claas.inference.cache import completion_cache

        backend = _get_tinker_backend(api_client)
        completion_cache._store.clear()
        content = "thinking\n</think>\n\nThe answer"

        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer()
        renderer = _make_mock_renderer(content)
        _patch_holder(backend.holder, sampler, tokenizer, renderer)

        api_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )

        full_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        miss_resp = api_client.get(
            "/v1/completions/raw",
            params={"content_hash": full_hash},
        )
        assert miss_resp.status_code == 404


class TestContentStripping:
    """Verify that returned content has thinking stripped."""

    def test_orphaned_think_tag_stripped_from_response(self, api_client):
        backend = _get_tinker_backend(api_client)
        raw_content = "Some internal reasoning\n</think>\n\nThe actual answer"
        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer()
        renderer = _make_mock_renderer(raw_content)
        _patch_holder(backend.holder, sampler, tokenizer, renderer)

        resp = api_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "The actual answer"

    def test_proper_think_block_stripped_from_response(self, api_client):
        backend = _get_tinker_backend(api_client)
        raw_content = "<think>Let me think about this...</think>Here is my answer"
        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer()
        renderer = _make_mock_renderer(raw_content)
        _patch_holder(backend.holder, sampler, tokenizer, renderer)

        resp = api_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "Here is my answer"

    def test_no_thinking_content_unchanged(self, api_client):
        backend = _get_tinker_backend(api_client)
        raw_content = "Just a plain answer"
        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer()
        renderer = _make_mock_renderer(raw_content)
        _patch_holder(backend.holder, sampler, tokenizer, renderer)

        resp = api_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "Just a plain answer"


class TestCacheEndToEnd:
    """Test cache through both API and eval paths for Qwen3 and GPT-OSS models."""

    def _generate_and_lookup(self, api_client, renderer_content):
        import hashlib

        from claas.inference.cache import completion_cache
        from claas.inference.helpers import strip_thinking

        backend = _get_tinker_backend(api_client)
        completion_cache._store.clear()

        sampler, mock_resp = _make_mock_sampler()
        mock_resp.sequences[0].logprobs = [-0.1, -0.2, -0.3]
        tokenizer = _make_mock_tokenizer()
        renderer = _make_mock_renderer(renderer_content)
        _patch_holder(backend.holder, sampler, tokenizer, renderer)

        api_resp = api_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert api_resp.status_code == 200
        api_content = api_resp.json()["choices"][0]["message"]["content"]

        content_hash = hashlib.sha256(
            strip_thinking(api_content).encode("utf-8"),
        ).hexdigest()
        cache_resp = api_client.get(
            "/v1/completions/raw",
            params={"content_hash": content_hash},
        )
        return api_content, cache_resp

    def test_qwen3_api_returns_answer_only(self, api_client):
        api_content, _ = self._generate_and_lookup(
            api_client,
            "Let me think step by step...\n</think>\n\nThe answer is 42",
        )
        assert api_content == "The answer is 42"
        assert "</think>" not in api_content
        assert "step by step" not in api_content

    def test_qwen3_cache_hit_from_api_content(self, api_client):
        _, cache_resp = self._generate_and_lookup(
            api_client,
            "Let me think step by step...\n</think>\n\nThe answer is 42",
        )
        assert cache_resp.status_code == 200
        data = cache_resp.json()
        assert data["prompt"] is not None
        assert data["response"] is not None
        assert data["logprobs"] == pytest.approx([-0.1, -0.2, -0.3])

    def test_qwen3_proper_think_block_cache_hit(self, api_client):
        api_content, cache_resp = self._generate_and_lookup(
            api_client,
            "<think>reasoning here</think>Final answer",
        )
        assert api_content == "Final answer"
        assert cache_resp.status_code == 200

    def test_gptoss_api_returns_final_channel(self, api_client):
        gptoss_content = (
            "<|channel|>analysis<|message|>internal analysis<|end|>"
            "<|start|>assistant<|channel|>final<|message|>The final answer<|end|>"
        )
        api_content, _ = self._generate_and_lookup(api_client, gptoss_content)
        assert api_content == "The final answer"
        assert "analysis" not in api_content

    def test_gptoss_cache_hit_from_api_content(self, api_client):
        gptoss_content = (
            "<|channel|>analysis<|message|>internal analysis<|end|>"
            "<|start|>assistant<|channel|>final<|message|>The final answer<|end|>"
        )
        _, cache_resp = self._generate_and_lookup(api_client, gptoss_content)
        assert cache_resp.status_code == 200
        data = cache_resp.json()
        assert data["logprobs"] == pytest.approx([-0.1, -0.2, -0.3])

    def test_plain_model_cache_hit(self, api_client):
        api_content, cache_resp = self._generate_and_lookup(
            api_client, "Hello, I can help with that!",
        )
        assert api_content == "Hello, I can help with that!"
        assert cache_resp.status_code == 200

    def test_eval_fetch_cached_completion_qwen3(self, api_client):
        import hashlib

        from claas.inference.cache import completion_cache

        backend = _get_tinker_backend(api_client)
        completion_cache._store.clear()

        sampler, mock_resp = _make_mock_sampler()
        mock_resp.sequences[0].logprobs = [-0.5, -0.4, -0.6]

        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        templated_prompt = (
            "<|im_start|>system\nYou are Kuro, a helpful assistant."
            "<|im_end|>\n<|im_start|>user\ntest<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        raw_response = "<think>thinking about it</think>\n\nHere is my response"
        tokenizer.decode.side_effect = [raw_response, templated_prompt]

        renderer = _make_mock_renderer(
            "thinking about it\n</think>\n\nHere is my response",
        )
        _patch_holder(backend.holder, sampler, tokenizer, renderer)

        api_resp = api_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]},
        )
        assert api_resp.status_code == 200
        visible = api_resp.json()["choices"][0]["message"]["content"]
        assert visible == "Here is my response"
        assert "<think>" not in visible
        assert "</think>" not in visible

        content_hash = hashlib.sha256(visible.encode("utf-8")).hexdigest()
        raw_resp = api_client.get(
            "/v1/completions/raw",
            params={"content_hash": content_hash},
        )
        assert raw_resp.status_code == 200
        data = raw_resp.json()

        assert data["prompt"] == templated_prompt
        assert "<|im_start|>system" in data["prompt"]
        assert data["prompt"] != "test"
        assert data["response"] == raw_response
        assert "<think>" in data["response"]
        assert data["logprobs"] == pytest.approx([-0.5, -0.4, -0.6])

    def test_eval_fetch_cached_completion_gptoss(self, api_client):
        import hashlib

        from claas.inference.cache import completion_cache

        backend = _get_tinker_backend(api_client)
        completion_cache._store.clear()

        sampler, mock_resp = _make_mock_sampler()
        mock_resp.sequences[0].logprobs = [-1.0, -0.8, -0.9]

        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        templated_prompt = (
            "<|system|>You are GPT-OSS, a helpful AI.<|end|>\n"
            "<|user|>test<|end|>\n<|assistant|>"
        )
        raw_response = (
            "<|channel|>analysis<|message|>deep analysis here<|end|>"
            "<|start|>assistant<|channel|>final<|message|>GPT-OSS says hello<|end|>"
        )
        tokenizer.decode.side_effect = [raw_response, templated_prompt]

        renderer = _make_mock_renderer(raw_response)
        _patch_holder(backend.holder, sampler, tokenizer, renderer)

        api_resp = api_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]},
        )
        assert api_resp.status_code == 200
        visible = api_resp.json()["choices"][0]["message"]["content"]
        assert visible == "GPT-OSS says hello"
        assert "analysis" not in visible

        content_hash = hashlib.sha256(visible.encode("utf-8")).hexdigest()
        raw_resp = api_client.get(
            "/v1/completions/raw",
            params={"content_hash": content_hash},
        )
        assert raw_resp.status_code == 200
        data = raw_resp.json()

        assert data["prompt"] == templated_prompt
        assert "<|system|>" in data["prompt"]
        assert data["prompt"] != "test"
        assert data["response"] == raw_response
        assert "<|channel|>analysis" in data["response"]
        assert data["logprobs"] == pytest.approx([-1.0, -0.8, -0.9])


class TestChatScoreEndpoint:
    """Tests for /v1/score with messages (chat template path)."""

    def _setup_holder_with_chat_template(self, holder, logprobs_result):
        """Set up holder with a tokenizer that supports apply_chat_template."""
        mock_sampler = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = logprobs_result
        mock_sampler.compute_logprobs.return_value = mock_future

        mock_tokenizer = MagicMock()

        # apply_chat_template returns different lengths depending on
        # add_generation_prompt: True → prompt only, False → prompt + completion
        def _apply_template(dicts, add_generation_prompt=True, tokenize=True):
            if add_generation_prompt:
                # Prompt tokens only
                return [10, 20, 30]
            # Full conversation including assistant reply
            return [10, 20, 30, 40, 50]

        mock_tokenizer.apply_chat_template.side_effect = _apply_template
        mock_tokenizer.decode.side_effect = lambda ids: f"tok{ids[0]}"

        _patch_holder(holder, mock_sampler, mock_tokenizer, _make_mock_renderer())
        return mock_sampler, mock_tokenizer

    def test_chat_score_returns_logprobs(self, api_client):
        """Happy path: structured messages are templated and scored."""
        from claas.api import web_app
        _holder = web_app.state.inference_backend.holder

        # 3 prompt tokens + 2 completion tokens = 5 total
        # logprobs for all 5 positions
        self._setup_holder_with_chat_template(
            _holder, [None, -1.0, -2.0, -0.5, -0.3],
        )

        resp = api_client.post(
            "/v1/score",
            json={
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                ],
                "completion": "Hello!",
            },
        )
        assert resp.status_code == 200
        body = resp.json()

        assert body["prompt_tokens"] == 3
        assert body["completion_tokens"] == 2
        # Completion logprobs at positions 3 and 4 → [-0.5, -0.3]
        assert body["logprobs"] == pytest.approx([-0.5, -0.3])
        assert body["tokens"] == ["tok40", "tok50"]
        assert body["logprob_sum"] == pytest.approx(-0.8)

    def test_chat_score_handles_none_logprobs(self, api_client):
        """None in completion region is treated as 0.0."""
        from claas.api import web_app
        _holder = web_app.state.inference_backend.holder

        self._setup_holder_with_chat_template(
            _holder, [None, -1.0, -2.0, None, -0.3],
        )

        resp = api_client.post(
            "/v1/score",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "completion": "Hello!",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        # Position 3 = None → 0.0, position 4 = -0.3
        assert body["logprobs"] == pytest.approx([0.0, -0.3])
        assert body["logprob_sum"] == pytest.approx(-0.3)

    def test_chat_score_empty_completion(self, api_client):
        """When completion produces no extra tokens, return empty results."""
        from claas.api import web_app
        _holder = web_app.state.inference_backend.holder

        mock_sampler = MagicMock()
        mock_tokenizer = MagicMock()

        # Both calls return the same tokens → completion_len = 0
        mock_tokenizer.apply_chat_template.return_value = [10, 20, 30]
        mock_tokenizer.decode.side_effect = lambda ids: f"tok{ids[0]}"

        _patch_holder(_holder, mock_sampler, mock_tokenizer, _make_mock_renderer())

        resp = api_client.post(
            "/v1/score",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "completion": "",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["logprobs"] == []
        assert body["tokens"] == []
        assert body["completion_tokens"] == 0
        assert body["logprob_sum"] == 0.0

    def test_chat_score_rejects_missing_fields(self, api_client):
        """Missing messages or completion should return 422."""
        # Missing completion
        resp = api_client.post(
            "/v1/score",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 422

        # Missing messages
        resp = api_client.post(
            "/v1/score",
            json={"completion": "Hello!"},
        )
        assert resp.status_code == 422
