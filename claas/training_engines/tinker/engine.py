"""Tinker API training engine implementation."""

from __future__ import annotations

import os
from urllib.parse import quote

import httpx
from pydantic import BaseModel

from claas.training_engines.base import TrainingEngine
from claas.types import (
    DistillRequestPayload,
    DistillResponse,
    LoraExistsPayload,
    LoraExportPayload,
    LoraInitRequest,
    LoraInitResponse,
    LoraListResponse,
    LoraRuntimeRef,
    ServiceHealth,
)


class TinkerEngineSettings(BaseModel):
    """Required settings for Tinker-backed execution."""

    base_url: str
    api_key: str


class TinkerTrainingEngine(TrainingEngine):
    """Executes training and LoRA management through strict Tinker HTTP APIs."""

    def __init__(self) -> None:
        """Load mandatory Tinker settings from environment."""
        self.settings = TinkerEngineSettings(
            base_url=os.environ["CLAAS_TINKER_BASE_URL"],
            api_key=os.environ["CLAAS_TINKER_API_KEY"],
        )

    def _headers(self) -> dict[str, str]:
        """Build authenticated request headers.

        Returns:
            HTTP headers for Tinker calls.
        """
        return {"Authorization": f"Bearer {self.settings.api_key}"}

    async def distill(self, payload: DistillRequestPayload) -> DistillResponse:
        """Call Tinker distillation endpoint.

        Args:
            payload: Distillation payload.

        Returns:
            Distillation response.
        """
        async with httpx.AsyncClient(base_url=self.settings.base_url, timeout=120) as client:
            response = await client.post(
                "/v1/claas/distill",
                json=payload.model_dump(mode="json"),
                headers=self._headers(),
            )
            response.raise_for_status()
        return DistillResponse.model_validate(response.json())

    async def init_lora(self, request: LoraInitRequest) -> LoraInitResponse:
        """Call Tinker LoRA initialization endpoint.

        Args:
            request: LoRA initialization request.

        Returns:
            Initialization response.
        """
        async with httpx.AsyncClient(base_url=self.settings.base_url, timeout=60) as client:
            response = await client.post(
                "/v1/claas/lora/init",
                json=request.model_dump(mode="json"),
                headers=self._headers(),
            )
            response.raise_for_status()
        return LoraInitResponse.model_validate(response.json())

    async def list_loras(self, prefix: str) -> LoraListResponse:
        """Call Tinker LoRA listing endpoint.

        Args:
            prefix: Prefix filter.

        Returns:
            LoRA list response.
        """
        async with httpx.AsyncClient(base_url=self.settings.base_url, timeout=30) as client:
            response = await client.get(
                "/v1/claas/lora",
                params={"prefix": prefix},
                headers=self._headers(),
            )
            response.raise_for_status()
        return LoraListResponse.model_validate(response.json())

    async def export_lora(self, lora_id: str) -> LoraExportPayload:
        """Call Tinker LoRA export endpoint.

        Args:
            lora_id: LoRA identifier.

        Returns:
            Archive filename and bytes.
        """
        async with httpx.AsyncClient(base_url=self.settings.base_url, timeout=60) as client:
            response = await client.get(
                "/v1/claas/lora/export",
                params={"lora_id": lora_id},
                headers=self._headers(),
            )
            response.raise_for_status()
        filename = f"{quote(lora_id, safe='')}.zip"
        return LoraExportPayload(filename=filename, content=response.content)

    async def lora_exists(self, lora_id: str) -> LoraExistsPayload:
        """Check LoRA existence on Tinker backend.

        Args:
            lora_id: LoRA identifier.

        Returns:
            Existence response.
        """
        async with httpx.AsyncClient(base_url=self.settings.base_url, timeout=30) as client:
            response = await client.get(
                "/v1/claas/lora/exists",
                params={"lora_id": lora_id},
                headers=self._headers(),
            )
            response.raise_for_status()
        return LoraExistsPayload.model_validate(response.json())


    async def lora_runtime_ref(self, lora_id: str) -> LoraRuntimeRef:
        """Reject runtime LoRA-path resolution for Tinker backend.

        Args:
            lora_id: LoRA identifier.

        Raises:
            ValueError: Tinker mode has no local filesystem LoRA path.
        """
        raise ValueError(
            "tinker backend does not expose local runtime LoRA paths for vLLM reload"
        )

    async def health(self) -> ServiceHealth:
        """Call Tinker health endpoint.

        Returns:
            Health response.
        """
        async with httpx.AsyncClient(base_url=self.settings.base_url, timeout=30) as client:
            response = await client.get("/v1/claas/health", headers=self._headers())
            response.raise_for_status()
        return ServiceHealth.model_validate(response.json())
