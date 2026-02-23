"""Tests for the inference backend abstraction (Tinker mode).

The tinker SDK is available but we mock the ServiceClient/SamplingClient
so tests don't need a real Tinker API key or GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

tinker = pytest.importorskip("tinker")  # noqa: F841

from fastapi.testclient import TestClient  # noqa: E402

from claas.core.config import TinkerConfig  # noqa: E402


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


def _make_mock_sampler():
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


def _make_mock_tokenizer_for_chat(content="Mocked response"):
    """Create a mock tokenizer that handles apply_chat_template and decode."""
    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2, 3]
    mock_tok.decode.return_value = content
    mock_tok.eos_token = "<|im_end|>"
    # apply_chat_template returns token IDs [10, 20, 30, 40, 50] (prompt)
    mock_tok.apply_chat_template.return_value = [10, 20, 30, 40, 50]
    return mock_tok


def _patch_holder(holder, sampler, tokenizer, stop_token_ids=None):
    holder._sampler = sampler
    holder._tokenizer = tokenizer
    holder._stop_token_ids = stop_token_ids if stop_token_ids is not None else {99}
    holder._service = MagicMock()  # prevent _ensure() from running


def _get_tinker_backend(api_client):
    """Force backend creation and return the TinkerBackend instance."""
    from claas.api import web_app
    from claas.inference.tinker import TinkerBackend

    backend = web_app.state.inference_backend
    assert isinstance(backend, TinkerBackend)
    return backend


class TestHolderInternals:
    """Test _ensure() and refresh() with mocked Tinker SDK."""

    def test_ensure_initializes_once(self):
        from claas.inference.tinker import SamplerHolder

        holder = SamplerHolder(cfg=TinkerConfig())
        mock_sampler = MagicMock()
        mock_tok = MagicMock()
        mock_tok.name_or_path = "Qwen/Qwen3-30B-A3B"
        mock_sampler.get_tokenizer.return_value = mock_tok
        mock_service = MagicMock()
        mock_service.create_sampling_client.return_value = mock_sampler

        mock_gen_config = MagicMock()
        mock_gen_config.eos_token_id = [151645, 151643]

        with (
            patch("tinker.ServiceClient", return_value=mock_service),
            patch(
                "transformers.GenerationConfig.from_pretrained",
                return_value=mock_gen_config,
            ),
        ):
            holder._ensure()
            assert holder._sampler is mock_sampler
            assert holder._stop_token_ids == {151645, 151643}

            mock_service.create_sampling_client.reset_mock()
            holder._ensure()
            mock_service.create_sampling_client.assert_not_called()

    def test_refresh_with_model_path(self):
        from claas.inference.tinker import SamplerHolder

        holder = SamplerHolder(cfg=TinkerConfig())
        mock_sampler = MagicMock()
        mock_tok = MagicMock()
        mock_tok.name_or_path = "Qwen/Qwen3-30B-A3B"
        mock_sampler.get_tokenizer.return_value = mock_tok
        mock_service = MagicMock()
        mock_service.create_sampling_client.return_value = mock_sampler
        holder._service = mock_service

        mock_gen_config = MagicMock()
        mock_gen_config.eos_token_id = 151645

        with patch(
            "transformers.GenerationConfig.from_pretrained",
            return_value=mock_gen_config,
        ):
            holder.refresh(model_path="tinker://run-1/weights/step-1")

        assert holder._model_path == "tinker://run-1/weights/step-1"
        assert holder._stop_token_ids == {151645}
        mock_service.create_sampling_client.assert_called_once_with(
            model_path="tinker://run-1/weights/step-1"
        )


class TestChatCompletions:
    def test_non_streaming(self, api_client):
        backend = _get_tinker_backend(api_client)
        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer_for_chat("Hello world")
        _patch_holder(backend.holder, sampler, tokenizer)

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


class TestStopTokenStripping:
    """Verify stop tokens are stripped from generated output."""

    def test_stop_token_stripped_from_end(self, api_client):
        backend = _get_tinker_backend(api_client)
        sampler, mock_resp = _make_mock_sampler()
        # Last token (3) is a stop token
        mock_resp.sequences[0].tokens = [1, 2, 3]
        tokenizer = _make_mock_tokenizer_for_chat("decoded")
        _patch_holder(backend.holder, sampler, tokenizer, stop_token_ids={3})

        resp = api_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 200
        # Verify decode was called with stop token removed
        decode_calls = [
            c for c in tokenizer.decode.call_args_list
            if c[0][0] == [1, 2]
        ]
        assert len(decode_calls) > 0

    def test_logprobs_aligned_after_stop_token_strip(self, api_client):
        """When a stop token is stripped, response_logprobs must shrink to match."""
        import hashlib

        from claas.inference.cache import completion_cache
        from claas.inference.helpers import normalize_for_hash

        backend = _get_tinker_backend(api_client)
        completion_cache._store.clear()

        sampler, mock_resp = _make_mock_sampler()
        # 3 response tokens, last one is a stop token
        mock_resp.sequences[0].tokens = [1, 2, 3]
        mock_resp.sequences[0].logprobs = [-0.1, -0.2, -0.3]

        tokenizer = _make_mock_tokenizer_for_chat("decoded")
        # Stop token 3 → will be stripped
        _patch_holder(backend.holder, sampler, tokenizer, stop_token_ids={3})

        resp = api_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 200
        visible = resp.json()["choices"][0]["message"]["content"]

        # Look up cached entry directly
        content_hash = hashlib.sha256(
            normalize_for_hash(visible).encode("utf-8"),
        ).hexdigest()
        entry = completion_cache.get(content_hash)
        assert entry is not None

        # After stripping stop token 3: tokens=[1,2], logprobs=[-0.1,-0.2]
        assert len(entry.response_token_ids) == 2
        assert entry.response_logprobs is not None
        assert len(entry.response_logprobs) == 2
        assert entry.response_logprobs == pytest.approx([-0.1, -0.2])

    def test_non_stop_token_preserved(self, api_client):
        backend = _get_tinker_backend(api_client)
        sampler, mock_resp = _make_mock_sampler()
        mock_resp.sequences[0].tokens = [1, 2, 3]
        tokenizer = _make_mock_tokenizer_for_chat("decoded")
        # Stop token is 99, not in the sequence
        _patch_holder(backend.holder, sampler, tokenizer, stop_token_ids={99})

        resp = api_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 200
        # Verify decode was called with all tokens preserved
        decode_calls = [
            c for c in tokenizer.decode.call_args_list
            if c[0][0] == [1, 2, 3]
        ]
        assert len(decode_calls) > 0

    def test_stop_token_ids_used_in_sampling_params(self, api_client):
        backend = _get_tinker_backend(api_client)
        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer_for_chat("decoded")
        tokenizer.decode.side_effect = lambda ids, **kw: {
            (42,): "<|im_end|>",
            (43,): "<|endoftext|>",
        }.get(tuple(ids), "decoded")
        _patch_holder(backend.holder, sampler, tokenizer, stop_token_ids={42, 43})

        resp = api_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 200
        sp = sampler.sample.call_args[1]["sampling_params"]
        assert set(sp.stop) == {"<|im_end|>", "<|endoftext|>"}


class TestContentStripping:
    """Verify that returned content has thinking stripped (API layer)."""

    def test_orphaned_think_tag_stripped_from_response(self, api_client):
        backend = _get_tinker_backend(api_client)
        raw_content = "Some internal reasoning\n</think>\n\nThe actual answer"
        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer_for_chat(raw_content)
        _patch_holder(backend.holder, sampler, tokenizer)

        resp = api_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "The actual answer"

    def test_gptoss_channel_extraction(self, api_client):
        backend = _get_tinker_backend(api_client)
        raw_content = (
            "<|channel|>analysis<|message|>internal analysis<|end|>"
            "<|start|>assistant<|channel|>final<|message|>The final answer<|end|>"
        )
        sampler, _ = _make_mock_sampler()
        tokenizer = _make_mock_tokenizer_for_chat(raw_content)
        _patch_holder(backend.holder, sampler, tokenizer)

        resp = api_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "The final answer"


class TestCacheEndToEnd:
    """Verify cache stores raw tokens/logprobs and retrieves by content hash."""

    def test_cache_stores_and_retrieves(self, api_client):
        import hashlib

        from claas.inference.cache import completion_cache
        from claas.inference.helpers import normalize_for_hash

        backend = _get_tinker_backend(api_client)
        completion_cache._store.clear()

        sampler, mock_resp = _make_mock_sampler()
        mock_resp.sequences[0].logprobs = [-0.5, -0.4, -0.6]

        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1, 2, 3]
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.apply_chat_template.return_value = [10, 20, 30, 40, 50]

        content_with_thinking = "thinking about it\n</think>\n\nHere is my response"
        raw_response = "thinking about it\n\nHere is my response"
        raw_prompt = "<|im_start|>user\ntest<|im_end|>\n<|im_start|>assistant\n"
        # decode is called for: stop token str conversion, content, raw_response, raw_prompt
        tokenizer.decode.side_effect = [
            "<|stop|>",  # stop token ID → string
            content_with_thinking,  # content
            raw_response,  # raw_response
            raw_prompt,  # raw_prompt
        ]
        _patch_holder(backend.holder, sampler, tokenizer, stop_token_ids={99})

        api_resp = api_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]},
        )
        assert api_resp.status_code == 200
        visible = api_resp.json()["choices"][0]["message"]["content"]
        assert visible == "Here is my response"

        content_hash = hashlib.sha256(
            normalize_for_hash(visible).encode("utf-8"),
        ).hexdigest()
        entry = completion_cache.get(content_hash)
        assert entry is not None
        assert entry.prompt == raw_prompt
        assert entry.response == raw_response
        assert entry.response_logprobs == pytest.approx([-0.5, -0.4, -0.6])
        assert entry.prompt_token_ids == [10, 20, 30, 40, 50]
