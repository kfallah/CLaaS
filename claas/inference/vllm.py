"""vLLM forwarding inference backend.

Forwards requests to an upstream vLLM instance via HTTP.
Used for both ``local`` and ``modal`` execution modes.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response

from claas.core.config import get_config

from .base import CompletionResult, InferenceBackend, TextCompletionResult
from .helpers import build_chatml_prompt, coerce_content

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared httpx client
# ---------------------------------------------------------------------------

_vllm_client: httpx.AsyncClient | None = None


def _get_vllm_client() -> httpx.AsyncClient:
    global _vllm_client  # noqa: PLW0603
    if _vllm_client is None:
        _vllm_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0))
    return _vllm_client


# ---------------------------------------------------------------------------
# VllmBackend
# ---------------------------------------------------------------------------


class VllmBackend(InferenceBackend):
    """Inference backend that forwards to an upstream vLLM instance."""

    def _backend_url(self) -> str:
        """Return the upstream vLLM base URL (no trailing slash)."""
        cfg = get_config()
        url: str = getattr(cfg, "vllm_base_url", "http://127.0.0.1:8000")
        return url.rstrip("/")

    def _api_key(self) -> str:
        cfg = get_config()
        return getattr(cfg, "vllm_api_key", "")

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        api_key = self._api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
    ) -> CompletionResult:
        client = _get_vllm_client()

        messages_dicts = [
            {"role": m["role"], "content": coerce_content(m.get("content", ""))}
            for m in messages
        ]

        body: dict[str, Any] = {
            "model": model,
            "messages": messages_dicts,
            "stream": False,
            "logprobs": True,
            "top_logprobs": 1,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if stop:
            body["stop"] = stop

        headers: dict[str, str] = {"Content-Type": "application/json"}
        headers.update(self._auth_headers())

        resp = await client.post(
            f"{self._backend_url()}/v1/chat/completions",
            json=body,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        content = choice["message"]["content"]

        logprobs: list[float] | None = None
        lp_data = choice.get("logprobs")
        if lp_data and lp_data.get("content"):
            logprobs = [entry["logprob"] for entry in lp_data["content"]]

        raw_prompt = build_chatml_prompt(messages_dicts)
        usage = data.get("usage", {})

        return CompletionResult(
            content=content,
            raw_prompt=raw_prompt,
            raw_response=content,
            token_ids=[],
            logprobs=logprobs,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    async def text_completion(
        self,
        *,
        prompt: str,
        model: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
    ) -> TextCompletionResult:
        client = _get_vllm_client()
        body: dict[str, Any] = {
            "model": model or "default",
            "prompt": prompt,
            "stream": False,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if stop:
            body["stop"] = stop

        headers: dict[str, str] = {"Content-Type": "application/json"}
        headers.update(self._auth_headers())

        resp = await client.post(
            f"{self._backend_url()}/v1/completions",
            json=body,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()

        text: str = data["choices"][0]["text"]
        usage = data.get("usage", {})

        return TextCompletionResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    async def list_models(self) -> dict[str, object] | Response:
        client = _get_vllm_client()
        headers = self._auth_headers()
        resp = await client.get(f"{self._backend_url()}/v1/models", headers=headers)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type="application/json",
        )

    def register_routes(self, app: FastAPI) -> None:
        """Register catch-all reverse proxy to upstream vLLM."""
        backend = self

        @app.api_route(
            "/v1/vllm/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        )
        async def _vllm_catch_all(request: Request, path: str) -> Response:
            """Forward unhandled requests to upstream vLLM."""
            client = _get_vllm_client()
            url = f"{backend._backend_url()}/{path}"
            if request.url.query:
                url = f"{url}?{request.url.query}"

            headers = dict(request.headers)
            headers.pop("host", None)
            headers.update(backend._auth_headers())

            body = await request.body()

            resp = await client.request(
                method=request.method,
                url=url,
                headers=headers,
                content=body if body else None,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
                media_type=resp.headers.get("content-type"),
            )
