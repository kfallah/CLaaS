"""Modal RPC training engine implementation."""

from __future__ import annotations

import asyncio
import re

import modal

from claas.core.types import (
    DistillBatchRequestPayload,
    DistillResponse,
    LoraDeleteResponse,
    LoraExistsPayload,
    LoraExportPayload,
    LoraInitRequest,
    LoraInitResponse,
    LoraListResponse,
    LoraRuntimeRef,
    ServiceHealth,
)
from claas.training.engine.base import TrainingEngine
from claas.training.storage import (
    configure_storage_backend,
    create_initial_lora,
    delete_lora,
    export_lora_zip_bytes,
    get_lora_path,
    list_loras,
    lora_exists,
    resolve_lora_id,
)


class ModalTrainingEngine(TrainingEngine):
    """Executes training on Modal while using shared LoRA storage."""

    def __init__(self) -> None:
        configure_storage_backend("modal_volume")

    async def distill(
        self,
        payload: DistillBatchRequestPayload,
    ) -> DistillResponse:
        """Run distillation using Modal RPC.

        Args:
            payload: Typed distillation payload.

        Returns:
            Distillation response.
        """
        distill_fn = modal.Function.from_name("claas-distill", "DistillWorker.distill")
        result = await distill_fn.remote.aio(payload)
        return DistillResponse.model_validate(result)

    async def init_lora(self, request: LoraInitRequest) -> LoraInitResponse:
        """Initialize a LoRA adapter in shared storage.

        Args:
            request: Initialization request.

        Returns:
            Initialization response.
        """
        lora_id = await asyncio.to_thread(
            create_initial_lora,
            lora_id=request.lora_id,
            base_model_name=request.base_model,
            lora_r=request.lora_r,
            lora_alpha=request.lora_alpha,
            target_modules=request.target_modules,
        )
        return LoraInitResponse(lora_id=lora_id)

    async def delete_lora(self, lora_id: str) -> LoraDeleteResponse:
        deleted = await asyncio.to_thread(delete_lora, lora_id)
        return LoraDeleteResponse(deleted=deleted)

    async def list_loras(self, prefix: str) -> LoraListResponse:
        """List LoRA identifiers from shared storage.

        Args:
            prefix: Prefix filter.

        Returns:
            LoRA list response.
        """
        loras = await asyncio.to_thread(list_loras, prefix)
        return LoraListResponse(loras=loras)

    async def export_lora(self, lora_id: str) -> LoraExportPayload:
        """Export a LoRA archive from shared storage.

        Args:
            lora_id: LoRA identifier.

        Returns:
            Export payload.
        """
        zip_bytes = await asyncio.to_thread(export_lora_zip_bytes, lora_id)
        safe_name = lora_id.strip("/").replace("/", "__")
        return LoraExportPayload(filename=f"{safe_name or 'lora_export'}.zip", content=zip_bytes)

    async def lora_exists(self, lora_id: str) -> LoraExistsPayload:
        """Check whether a LoRA exists in shared storage.

        Args:
            lora_id: LoRA identifier.

        Returns:
            Existence response.
        """
        exists = await asyncio.to_thread(lora_exists, lora_id)
        return LoraExistsPayload(exists=exists)

    async def lora_runtime_ref(self, lora_id: str) -> LoraRuntimeRef:
        """Resolve runtime LoRA path and vLLM adapter name.

        Args:
            lora_id: LoRA identifier.

        Returns:
            Runtime LoRA reference.
        """
        resolved_id = await asyncio.to_thread(resolve_lora_id, lora_id)
        lora_path = await asyncio.to_thread(get_lora_path, resolved_id)
        vllm_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", resolved_id.strip("/")).strip("-") or "lora"
        return LoraRuntimeRef(vllm_name=vllm_name, lora_path=lora_path)

    async def health(self) -> ServiceHealth:
        """Query remote distillation worker health.

        Returns:
            Health response.
        """
        worker_health_fn = modal.Function.from_name("claas-distill", "DistillWorker.health_check")
        data = await asyncio.wait_for(worker_health_fn.remote.aio(), timeout=15)
        return ServiceHealth.model_validate(data)
