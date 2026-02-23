"""vLLM server lifecycle orchestration.

Functions for sleep/wake, drain, GPU readiness checks, and LoRA
reload — used by the feedback endpoint when running with local or
modal backends.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx

from claas.core.config import CoreConfig
from claas.training.engine.base import TrainingEngine

logger = logging.getLogger(__name__)


def vllm_api_key() -> str:
    """Return the vLLM API key from the environment."""
    raw = os.environ.get("VLLM_API_KEY")
    return raw.strip() if raw is not None else ""


def vllm_connection(cfg: CoreConfig) -> tuple[str, str]:
    """Return ``(base_url, api_key)`` for the current config."""
    base_url: str = getattr(cfg, "vllm_base_url", "http://127.0.0.1:8000")
    api_key = vllm_api_key()
    return base_url, api_key


async def vllm_post(
    base_url: str,
    api_key: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout_s: float = 30.0,
) -> None:
    """Call a vLLM control endpoint and raise on non-success."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        resp = await client.post(path, params=params, json=json_body, headers=headers)
    resp.raise_for_status()


async def wait_for_vllm_idle(
    base_url: str,
    api_key: str,
    timeout_s: float,
) -> None:
    """Poll vLLM ``/metrics`` until no requests are running or waiting.

    Raises :class:`TimeoutError` if vLLM is still busy after *timeout_s*.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    deadline = time.perf_counter() + timeout_s

    while True:
        async with httpx.AsyncClient(base_url=base_url, timeout=10) as client:
            resp = await client.get("/metrics", headers=headers)
        resp.raise_for_status()

        running = 0
        waiting = 0
        for line in resp.text.splitlines():
            if line.startswith("vllm:num_requests_running"):
                running += int(float(line.split()[-1]))
            elif line.startswith("vllm:num_requests_waiting"):
                waiting += int(float(line.split()[-1]))

        if running == 0 and waiting == 0:
            logger.info("vLLM idle (0 running, 0 waiting) — safe to sleep")
            return

        if time.perf_counter() >= deadline:
            raise TimeoutError(
                f"vLLM still busy after {timeout_s}s: "
                f"{running} running, {waiting} waiting"
            )

        logger.info(
            "vLLM busy (%d running, %d waiting) — polling again in 0.5s",
            running,
            waiting,
        )
        await asyncio.sleep(0.5)


async def verify_gpu_ready(
    min_free_gb: float,
    timeout_s: float,
) -> None:
    """Poll GPU memory until *min_free_gb* is available or *timeout_s* expires.

    Called after ``POST /sleep`` so training only starts once vLLM has
    actually released its VRAM.  If torch is not installed (CPU-only API
    image) the check is skipped silently.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
    except ImportError:
        return

    deadline = time.perf_counter() + timeout_s
    while True:
        free_bytes, _total = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024**3)
        if free_gb >= min_free_gb:
            logger.info("GPU has %.1f GB free — ready for training", free_gb)
            return
        if time.perf_counter() >= deadline:
            logger.warning(
                "Only %.1f GB GPU memory free after waiting %.0fs for vLLM sleep. "
                "Training may OOM.",
                free_gb,
                timeout_s,
            )
            return
        logger.info(
            "GPU has %.1f GB free (need %.1f GB) — waiting for vLLM to release memory…",
            free_gb,
            min_free_gb,
        )
        await asyncio.sleep(1.0)


async def vllm_reload_lora(
    engine: TrainingEngine,
    base_url: str,
    api_key: str,
    lora_id: str,
) -> None:
    """Unload and reload a LoRA adapter so vLLM picks up on-disk changes.

    Requires VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 on the vLLM server.
    """
    runtime_ref = await engine.lora_runtime_ref(lora_id)

    try:
        await vllm_post(
            base_url,
            api_key,
            "/v1/unload_lora_adapter",
            json_body={"lora_name": runtime_ref.vllm_name},
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 404:
            raise
    await vllm_post(
        base_url,
        api_key,
        "/v1/load_lora_adapter",
        json_body={"lora_name": runtime_ref.vllm_name, "lora_path": runtime_ref.lora_path},
    )
