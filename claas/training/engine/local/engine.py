"""Local training engine implementation."""

from __future__ import annotations

import asyncio
import logging
import re
import threading

from claas.core.config import LocalConfig
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
from claas.training.engine.local.cache import LoraCacheEntry
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

logger = logging.getLogger(__name__)


class LocalTrainingEngine(TrainingEngine):
    """Executes training and LoRA operations on local infrastructure."""

    _trainer: DistillationTrainer
    _lora_cache: dict[str, LoraCacheEntry]
    _cache_lock: threading.Lock
    _model_loaded: bool

    def __init__(self, cfg: LocalConfig) -> None:
        configure_storage_backend("local_fs")
        self._base_model_id = cfg.base_model_id
        self._attn_implementation = cfg.attn_implementation
        self._trainer = DistillationTrainer(
            base_model_id=cfg.base_model_id,
            attn_implementation=cfg.attn_implementation,
        )
        self._lora_cache = {}
        self._cache_lock = threading.Lock()
        self._model_loaded = False

    async def _ensure_model_loaded(self) -> None:
        """One-time base model load on first distill() call."""
        if not self._model_loaded:
            await asyncio.to_thread(self._trainer.load_base_model)
            self._model_loaded = True

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
        await self._ensure_model_loaded()
        await asyncio.to_thread(self._trainer.reload_base_model)

        resolved_id = await asyncio.to_thread(resolve_lora_id, payload.lora_id)
        with self._cache_lock:
            cached = self._lora_cache.get(resolved_id)

        try:
            result = await asyncio.to_thread(
                self._trainer.distill, payload, cached=cached
            )
        finally:
            await asyncio.to_thread(self._trainer.offload_base_model)

        with self._cache_lock:
            self._lora_cache[resolved_id] = result.cache_entry

        return result.response

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
        resolved_id = await asyncio.to_thread(resolve_lora_id, lora_id)
        deleted = await asyncio.to_thread(delete_lora, lora_id)
        if deleted:
            with self._cache_lock:
                self._lora_cache.pop(resolved_id, None)
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
