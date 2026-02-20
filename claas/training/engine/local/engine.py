"""Local training engine implementation."""

from __future__ import annotations

import asyncio
import os
import re

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
from claas.training.distillation import DistillationTrainer
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


class LocalTrainingEngine(TrainingEngine):
    """Executes training and LoRA operations on local infrastructure."""

    def __init__(self) -> None:
        configure_storage_backend("local_fs")

    async def distill(
        self,
        payload: DistillBatchRequestPayload,
    ) -> DistillResponse:
        """Run distillation against the local worker implementation.

        Args:
            payload: Typed distillation payload.

        Returns:
            Distillation response.
        """
        trainer = DistillationTrainer(
            base_model_id=os.environ["CLAAS_BASE_MODEL_ID"],
            attn_implementation=os.environ["CLAAS_ATTN_IMPLEMENTATION"],
        )
        await asyncio.to_thread(trainer.load_base_model)
        try:
            return await asyncio.to_thread(trainer.distill, payload)
        finally:
            await asyncio.to_thread(trainer.offload_base_model)

    async def init_lora(self, request: LoraInitRequest) -> LoraInitResponse:
        """Initialize a LoRA adapter locally.

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
        """List local LoRA identifiers.

        Args:
            prefix: Prefix filter.

        Returns:
            LoRA list response.
        """
        loras = await asyncio.to_thread(list_loras, prefix)
        return LoraListResponse(loras=loras)

    async def export_lora(self, lora_id: str) -> LoraExportPayload:
        """Export a local LoRA archive.

        Args:
            lora_id: LoRA identifier.

        Returns:
            Export payload with archive bytes.
        """
        zip_bytes = await asyncio.to_thread(export_lora_zip_bytes, lora_id)
        safe_name = lora_id.strip("/").replace("/", "__")
        return LoraExportPayload(filename=f"{safe_name or 'lora_export'}.zip", content=zip_bytes)

    async def lora_exists(self, lora_id: str) -> LoraExistsPayload:
        """Check local LoRA existence.

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
        """Return local engine health.

        Returns:
            Health response.
        """
        return ServiceHealth(status="healthy", error=None)
