"""Base interfaces for CLaaS training engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel

from claas.types import (
    DistillBatchRequestPayload,
    DistillResponse,
    LoraExistsPayload,
    LoraExportPayload,
    LoraInitRequest,
    LoraInitResponse,
    LoraListResponse,
    LoraRuntimeRef,
    ServiceHealth,
)

EngineKind = Literal["local", "modal", "tinker"]


class TrainingEngineConfig(BaseModel):
    """Configuration used to select a concrete training engine."""

    kind: EngineKind


class TrainingEngine(ABC):
    """Abstract engine contract for distillation and LoRA lifecycle operations."""

    @abstractmethod
    async def distill(
        self,
        payload: DistillBatchRequestPayload,
    ) -> DistillResponse:
        """Run one distillation step.

        Args:
            payload: Typed distillation request payload.

        Returns:
            Distill response payload.
        """

    @abstractmethod
    async def init_lora(self, request: LoraInitRequest) -> LoraInitResponse:
        """Initialize a LoRA adapter.

        Args:
            request: Typed LoRA initialization request.

        Returns:
            LoRA initialization response.
        """

    @abstractmethod
    async def list_loras(self, prefix: str) -> LoraListResponse:
        """List LoRA identifiers.

        Args:
            prefix: Prefix filter.

        Returns:
            LoRA list response.
        """

    @abstractmethod
    async def export_lora(self, lora_id: str) -> LoraExportPayload:
        """Export a LoRA adapter archive.

        Args:
            lora_id: LoRA identifier.

        Returns:
            LoRA archive payload.
        """

    @abstractmethod
    async def lora_exists(self, lora_id: str) -> LoraExistsPayload:
        """Check whether a LoRA exists.

        Args:
            lora_id: LoRA identifier.

        Returns:
            Existence payload.
        """

    @abstractmethod
    async def lora_runtime_ref(self, lora_id: str) -> LoraRuntimeRef:
        """Resolve runtime LoRA path/name information.

        Args:
            lora_id: LoRA identifier.

        Returns:
            Runtime LoRA reference.
        """

    @abstractmethod
    async def health(self) -> ServiceHealth:
        """Return backend health.

        Returns:
            Backend health status.
        """
