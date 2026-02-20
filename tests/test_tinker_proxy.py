"""Tests for the inference proxy FastAPI endpoints (Tinker mode).

The tinker SDK is available but we mock the ServiceClient/SamplingClient
so tests don't need a real Tinker API key or GPU.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# Force tinker mode so that tinker-only endpoints are registered at import time.
os.environ["CLAAS_PROXY_MODE"] = "tinker"

tinker = pytest.importorskip("tinker")  # noqa: F841

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_holder():
    """Reset the proxy singleton and config cache between tests."""
    from claas.core.config import get_proxy_config
    from claas.proxy.inference_proxy import _holder

    get_proxy_config.cache_clear()

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

    get_proxy_config.cache_clear()


@pytest.fixture()
def proxy_client():
    """TestClient for the proxy FastAPI app."""
    from claas.proxy.inference_proxy import app

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
        from claas.proxy.inference_proxy import _holder

        _holder._model_path = "tinker://run-123/weights/step-1"
        resp = proxy_client.get("/v1/sampler/status")
        assert resp.status_code == 200
        assert resp.json()["model_path"] == "tinker://run-123/weights/step-1"


class TestSamplerRefresh:
    def test_refresh_calls_holder(self, proxy_client):
        from claas.proxy.inference_proxy import _holder

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
        from claas.proxy.inference_proxy import _SamplerHolder

        holder = _SamplerHolder()
        mock_sampler = MagicMock()
        mock_sampler.get_tokenizer.return_value = MagicMock()
        mock_service = MagicMock()
        mock_service.create_sampling_client.return_value = mock_sampler

        with patch("tinker.ServiceClient", return_value=mock_service), \
             patch("tinker_cookbook.model_info.get_recommended_renderer_name", return_value="chatml"), \
             patch("tinker_cookbook.renderers.get_renderer", return_value=MagicMock()):
            # First call initializes
            holder._ensure()
            assert holder._sampler is mock_sampler
            assert holder._service is mock_service

            # Second call is a no-op (already initialized)
            mock_service.create_sampling_client.reset_mock()
            holder._ensure()
            mock_service.create_sampling_client.assert_not_called()

    def test_refresh_with_model_path(self):
        from claas.proxy.inference_proxy import _SamplerHolder

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
        from claas.proxy.inference_proxy import _base_model, _SamplerHolder

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
        mock_service.create_sampling_client.assert_called_once_with(
            base_model=_base_model()
        )

    def test_refresh_creates_service_if_missing(self):
        from claas.proxy.inference_proxy import _SamplerHolder

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
        from claas.proxy.inference_proxy import _holder

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
        from claas.proxy.inference_proxy import _holder

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
        from claas.proxy.inference_proxy import _holder

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
        from claas.proxy.inference_proxy import _holder

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
        from claas.proxy.inference_proxy import _coerce_content

        assert _coerce_content(raw) == expected

    def test_bounded_int_and_float(self):
        from claas.proxy.inference_proxy import _bounded_float, _bounded_int

        assert _bounded_int(None, default=10, minimum=1, maximum=100) == 10
        assert _bounded_int(999, default=10, minimum=1, maximum=100) == 100
        assert _bounded_int(-5, default=10, minimum=1, maximum=100) == 1
        assert _bounded_float(None, default=0.5, minimum=0.0, maximum=1.0) == 0.5
        assert _bounded_float(5.0, default=0.5, minimum=0.0, maximum=1.0) == 1.0
        assert _bounded_float(-1.0, default=0.5, minimum=0.0, maximum=1.0) == 0.0


class TestScoreEndpoint:
    def test_score_returns_logprobs(self, proxy_client):
        from claas.proxy.inference_proxy import _holder

        # Set up mock sampler with compute_logprobs
        mock_sampler = MagicMock()
        # Prompt tokens [10, 20, 30] + completion tokens [40, 50] = 5 tokens total
        # compute_logprobs returns a Future whose .result() gives logprobs
        mock_future = MagicMock()
        mock_future.result.return_value = [None, -1.0, -2.0, -0.5, -0.3]
        mock_sampler.compute_logprobs.return_value = mock_future

        mock_tokenizer = MagicMock()
        # encode("prompt text", add_special_tokens=True) -> prompt tokens
        # encode("completion text", add_special_tokens=False) -> completion tokens
        def _encode(text, add_special_tokens=True):
            if add_special_tokens:
                return [10, 20, 30]  # prompt
            return [40, 50]  # completion

        mock_tokenizer.encode.side_effect = _encode
        mock_tokenizer.decode.side_effect = lambda ids: f"tok{ids[0]}"

        mock_renderer = _make_mock_renderer()
        _patch_holder(_holder, mock_sampler, mock_tokenizer, mock_renderer)

        resp = proxy_client.post(
            "/v1/score",
            json={"prompt": "prompt text", "completion": "completion text"},
        )
        assert resp.status_code == 200
        body = resp.json()

        assert body["prompt_tokens"] == 3
        assert body["completion_tokens"] == 2
        # Logprobs at positions 3 and 4 -> [-0.5, -0.3]
        assert body["logprobs"] == pytest.approx([-0.5, -0.3])
        assert body["tokens"] == ["tok40", "tok50"]
        assert body["logprob_sum"] == pytest.approx(-0.8)

    def test_score_handles_none_logprobs(self, proxy_client):
        from claas.proxy.inference_proxy import _holder

        mock_sampler = MagicMock()
        # None in completion region should become 0.0
        mock_future = MagicMock()
        mock_future.result.return_value = [None, -1.0, None]
        mock_sampler.compute_logprobs.return_value = mock_future

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text, add_special_tokens=True: (
            [10] if add_special_tokens else [20, 30]
        )
        mock_tokenizer.decode.side_effect = lambda ids: f"t{ids[0]}"

        _patch_holder(_holder, mock_sampler, mock_tokenizer, _make_mock_renderer())

        resp = proxy_client.post(
            "/v1/score",
            json={"prompt": "p", "completion": "c"},
        )
        assert resp.status_code == 200
        body = resp.json()
        # Position 1 = -1.0, position 2 = None -> 0.0
        assert body["logprobs"] == pytest.approx([-1.0, 0.0])
        assert body["logprob_sum"] == pytest.approx(-1.0)

    def test_score_rejects_missing_prompt(self, proxy_client):
        resp = proxy_client.post(
            "/v1/score",
            json={"completion": "hello"},
        )
        assert resp.status_code == 422


class TestCompletionCache:
    """Verify that the completion cache strips <think> blocks before hashing."""

    def _populate_and_lookup(self, proxy_client, content, visible_text):
        """Helper: populate cache with content, look up by hash of visible_text."""
        import hashlib

        from claas.proxy.inference_proxy import _completion_cache, _holder

        # Clear cache between sub-tests
        _completion_cache._store.clear()

        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer()
        renderer = _make_mock_renderer(content)
        _patch_holder(_holder, sampler, tokenizer, renderer)

        resp = proxy_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 200

        visible_hash = hashlib.sha256(visible_text.encode("utf-8")).hexdigest()
        return proxy_client.get(
            "/v1/completions/raw",
            params={"content_hash": visible_hash},
        )

    def test_proper_think_tags(self, proxy_client):
        """Proper <think>...</think> blocks are stripped before hashing."""
        raw_resp = self._populate_and_lookup(
            proxy_client,
            content="<think>thinking</think>The answer",
            visible_text="The answer",
        )
        assert raw_resp.status_code == 200

    def test_orphaned_close_tag(self, proxy_client):
        """Orphaned </think> (opening tag consumed as special token) is stripped."""
        raw_resp = self._populate_and_lookup(
            proxy_client,
            content="internal reasoning\n</think>\n\nThe answer",
            visible_text="The answer",
        )
        assert raw_resp.status_code == 200

    def test_no_thinking(self, proxy_client):
        """Content without thinking blocks hashes as-is."""
        raw_resp = self._populate_and_lookup(
            proxy_client,
            content="Just a plain answer",
            visible_text="Just a plain answer",
        )
        assert raw_resp.status_code == 200

    def test_full_content_hash_misses(self, proxy_client):
        """Hash of full content (with thinking) should NOT match the cache."""
        import hashlib

        from claas.proxy.inference_proxy import _completion_cache, _holder

        _completion_cache._store.clear()
        content = "thinking\n</think>\n\nThe answer"

        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer()
        renderer = _make_mock_renderer(content)
        _patch_holder(_holder, sampler, tokenizer, renderer)

        proxy_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )

        full_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        miss_resp = proxy_client.get(
            "/v1/completions/raw",
            params={"content_hash": full_hash},
        )
        assert miss_resp.status_code == 404


class TestContentStripping:
    """Verify that returned content has thinking stripped."""

    def test_orphaned_think_tag_stripped_from_response(self, proxy_client):
        """Renderer returns content with orphaned </think>; response should be answer-only."""
        from claas.proxy.inference_proxy import _holder

        raw_content = "Some internal reasoning\n</think>\n\nThe actual answer"
        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer()
        renderer = _make_mock_renderer(raw_content)
        _patch_holder(_holder, sampler, tokenizer, renderer)

        resp = proxy_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "The actual answer"

    def test_proper_think_block_stripped_from_response(self, proxy_client):
        """Proper <think>...</think> block is stripped from response content."""
        from claas.proxy.inference_proxy import _holder

        raw_content = "<think>Let me think about this...</think>Here is my answer"
        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer()
        renderer = _make_mock_renderer(raw_content)
        _patch_holder(_holder, sampler, tokenizer, renderer)

        resp = proxy_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "Here is my answer"

    def test_no_thinking_content_unchanged(self, proxy_client):
        """Content without thinking blocks is returned as-is."""
        from claas.proxy.inference_proxy import _holder

        raw_content = "Just a plain answer"
        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer()
        renderer = _make_mock_renderer(raw_content)
        _patch_holder(_holder, sampler, tokenizer, renderer)

        resp = proxy_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "Just a plain answer"


class TestCacheEndToEnd:
    """Test cache through both API and eval paths for Qwen3 and GPT-OSS models."""

    def _generate_and_lookup(self, proxy_client, renderer_content):
        """Generate via /v1/chat/completions and return (api_content, cache_resp)."""
        import hashlib

        from claas.proxy.inference_proxy import _completion_cache, _holder, _strip_thinking

        _completion_cache._store.clear()

        sampler, mock_resp = _make_mock_sampler()
        mock_resp.sequences[0].logprobs = [-0.1, -0.2, -0.3]
        tokenizer = _make_mock_tokenizer()
        renderer = _make_mock_renderer(renderer_content)
        _patch_holder(_holder, sampler, tokenizer, renderer)

        api_resp = proxy_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert api_resp.status_code == 200
        api_content = api_resp.json()["choices"][0]["message"]["content"]

        # Look up cache using the same hash the eval runner would compute
        content_hash = hashlib.sha256(
            _strip_thinking(api_content).encode("utf-8"),
        ).hexdigest()
        cache_resp = proxy_client.get(
            "/v1/completions/raw",
            params={"content_hash": content_hash},
        )
        return api_content, cache_resp

    # --- Qwen3 (thinking model) ---

    def test_qwen3_api_returns_answer_only(self, proxy_client):
        """Qwen3 orphaned </think> is stripped from the API response."""
        api_content, _ = self._generate_and_lookup(
            proxy_client,
            "Let me think step by step...\n</think>\n\nThe answer is 42",
        )
        assert api_content == "The answer is 42"
        assert "</think>" not in api_content
        assert "step by step" not in api_content

    def test_qwen3_cache_hit_from_api_content(self, proxy_client):
        """Eval runner can look up cache using the stripped content from API."""
        _, cache_resp = self._generate_and_lookup(
            proxy_client,
            "Let me think step by step...\n</think>\n\nThe answer is 42",
        )
        assert cache_resp.status_code == 200
        data = cache_resp.json()
        assert data["prompt"] is not None
        assert data["response"] is not None
        assert data["logprobs"] == pytest.approx([-0.1, -0.2, -0.3])

    def test_qwen3_proper_think_block_cache_hit(self, proxy_client):
        """Proper <think>...</think> blocks also produce matching cache keys."""
        api_content, cache_resp = self._generate_and_lookup(
            proxy_client,
            "<think>reasoning here</think>Final answer",
        )
        assert api_content == "Final answer"
        assert cache_resp.status_code == 200

    # --- GPT-OSS (channel model) ---

    def test_gptoss_api_returns_final_channel(self, proxy_client):
        """GPT-OSS analysis channel is stripped; only final channel is returned."""
        gptoss_content = (
            "<|channel|>analysis<|message|>internal analysis<|end|>"
            "<|start|>assistant<|channel|>final<|message|>The final answer<|end|>"
        )
        api_content, _ = self._generate_and_lookup(proxy_client, gptoss_content)
        assert api_content == "The final answer"
        assert "analysis" not in api_content

    def test_gptoss_cache_hit_from_api_content(self, proxy_client):
        """Eval runner can look up cache using the extracted final channel content."""
        gptoss_content = (
            "<|channel|>analysis<|message|>internal analysis<|end|>"
            "<|start|>assistant<|channel|>final<|message|>The final answer<|end|>"
        )
        _, cache_resp = self._generate_and_lookup(proxy_client, gptoss_content)
        assert cache_resp.status_code == 200
        data = cache_resp.json()
        assert data["logprobs"] == pytest.approx([-0.1, -0.2, -0.3])

    # --- Plain model (no thinking, no channels) ---

    def test_plain_model_cache_hit(self, proxy_client):
        """Plain content without thinking or channels caches and looks up correctly."""
        api_content, cache_resp = self._generate_and_lookup(
            proxy_client, "Hello, I can help with that!",
        )
        assert api_content == "Hello, I can help with that!"
        assert cache_resp.status_code == 200

    # --- Eval path simulation ---

    def test_eval_fetch_cached_completion_qwen3(self, proxy_client):
        """Full round-trip: proxy strips thinking, cache returns it back.

        Flow:
        1. Renderer produces content with orphaned </think> (Qwen3 behavior)
        2. Real proxy code strips thinking -> API returns answer-only
        3. Caller hashes the visible content (with defensive strip_thinking)
        4. Cache lookup returns the RAW response (with <think> block) +
           the full templated prompt + generation-time logprobs
        """
        import hashlib

        from claas.proxy.inference_proxy import _completion_cache, _holder

        _completion_cache._store.clear()

        sampler, mock_resp = _make_mock_sampler()
        mock_resp.sequences[0].logprobs = [-0.5, -0.4, -0.6]

        # Tokenizer.decode is called twice in the cache path:
        #   1. decode(seq.tokens, skip_special_tokens=False)  -> raw response
        #   2. decode(model_input.to_ints(), skip_special_tokens=False)  -> templated prompt
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        templated_prompt = (
            "<|im_start|>system\nYou are Kuro, a helpful assistant."
            "<|im_end|>\n<|im_start|>user\ntest<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        raw_response = "<think>thinking about it</think>\n\nHere is my response"
        tokenizer.decode.side_effect = [raw_response, templated_prompt]

        # Renderer simulates Qwen3: <think> consumed as special token,
        # orphaned </think> remains in parsed content
        renderer = _make_mock_renderer(
            "thinking about it\n</think>\n\nHere is my response",
        )
        _patch_holder(_holder, sampler, tokenizer, renderer)

        # Step 1: POST /v1/chat/completions — proxy strips thinking
        api_resp = proxy_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]},
        )
        assert api_resp.status_code == 200
        visible = api_resp.json()["choices"][0]["message"]["content"]
        assert visible == "Here is my response"
        assert "<think>" not in visible
        assert "</think>" not in visible

        # Step 2: GET /v1/completions/raw — eval runner hashes visible content
        # (strip_thinking is a no-op here since proxy already stripped)
        content_hash = hashlib.sha256(visible.encode("utf-8")).hexdigest()
        raw_resp = proxy_client.get(
            "/v1/completions/raw",
            params={"content_hash": content_hash},
        )
        assert raw_resp.status_code == 200
        data = raw_resp.json()

        # Cached prompt is the full template, NOT the bare probe "test"
        assert data["prompt"] == templated_prompt
        assert "<|im_start|>system" in data["prompt"]
        assert data["prompt"] != "test"

        # Cached response is the RAW output WITH thinking (for training)
        assert data["response"] == raw_response
        assert "<think>" in data["response"]

        assert data["logprobs"] == pytest.approx([-0.5, -0.4, -0.6])

    def test_eval_fetch_cached_completion_gptoss(self, proxy_client):
        """Full round-trip: proxy extracts final channel, cache returns raw.

        Flow:
        1. Renderer produces GPT-OSS multi-channel output (analysis + final)
        2. Real proxy code extracts final channel -> API returns "GPT-OSS says hello"
        3. Caller hashes the visible content
        4. Cache lookup returns the RAW response (with analysis channel) +
           the full templated prompt + generation-time logprobs
        """
        import hashlib

        from claas.proxy.inference_proxy import _completion_cache, _holder

        _completion_cache._store.clear()

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

        # Renderer returns the full multi-channel output (as if from the model)
        renderer = _make_mock_renderer(raw_response)
        _patch_holder(_holder, sampler, tokenizer, renderer)

        # Step 1: POST /v1/chat/completions — proxy extracts final channel
        api_resp = proxy_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]},
        )
        assert api_resp.status_code == 200
        visible = api_resp.json()["choices"][0]["message"]["content"]
        assert visible == "GPT-OSS says hello"
        assert "analysis" not in visible

        # Step 2: GET /v1/completions/raw — eval runner hashes visible content
        content_hash = hashlib.sha256(visible.encode("utf-8")).hexdigest()
        raw_resp = proxy_client.get(
            "/v1/completions/raw",
            params={"content_hash": content_hash},
        )
        assert raw_resp.status_code == 200
        data = raw_resp.json()

        # Cached prompt is the full template, NOT the bare probe "test"
        assert data["prompt"] == templated_prompt
        assert "<|system|>" in data["prompt"]
        assert data["prompt"] != "test"

        # Cached response is the RAW output WITH analysis channel (for training)
        assert data["response"] == raw_response
        assert "<|channel|>analysis" in data["response"]

        assert data["logprobs"] == pytest.approx([-1.0, -0.8, -0.9])
