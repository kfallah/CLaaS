"""Modal worker wrapper around shared SDPO distillation logic."""

from __future__ import annotations

import os

import modal
from pydantic import BaseModel

from claas.core.types import DistillBatchRequestPayload, DistillResponse
from claas.training.distillation import DistillationTrainer
from claas.training.storage import LORA_MOUNT_PATH, lora_volume


class WorkerHealthResponse(BaseModel):
    """Health response for the distillation worker."""

    status: str
    model: str
    device: str
    vram_allocated_gb: float
    vram_reserved_gb: float


app = modal.App("claas-distill")
model_volume = modal.Volume.from_name("claas-models", create_if_missing=True)
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "transformers>=4.40.0,<5.0.0",
        "peft>=0.10.0",
        "accelerate>=0.27.0",
        "bitsandbytes>=0.42.0",
        "safetensors>=0.4.0",
        "pydantic>=2.6.0",
        "packaging>=24.0",
    )
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
        "flash_attn-2.8.3%2Bcu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    )
    .env(
        {
            "HF_HOME": "/models/hf_cache",
            "TRANSFORMERS_CACHE": "/models/hf_cache",
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "HUGGING_FACE_HUB_TOKEN": os.environ.get("HF_TOKEN", ""),
        }
    )
)


@app.cls(
    gpu="L40S",
    image=training_image,
    volumes={
        "/models": model_volume,
        LORA_MOUNT_PATH: lora_volume,
    },
    scaledown_window=300,
    timeout=600,
    enable_memory_snapshot=True,
)
class DistillWorker:
    """Modal worker that delegates training to ``DistillationTrainer``."""

    # Defaults are captured at deploy/import time. Move env reads into runtime
    # initialization if container-level overrides are required.
    base_model_id: str = os.environ.get("CLAAS_BASE_MODEL_ID", "Qwen/Qwen3-8B")
    attn_implementation: str = os.environ.get("CLAAS_ATTN_IMPLEMENTATION", "sdpa")

    @modal.enter(snap=True)
    def load_base_model(self) -> None:
        """Load base model into the reusable trainer for snapshotting."""
        self.trainer = DistillationTrainer(
            base_model_id=self.base_model_id,
            attn_implementation=self.attn_implementation,
        )
        self.trainer.load_base_model()

    @modal.method()
    def distill(self, request: DistillBatchRequestPayload) -> DistillResponse:
        """Run one SDPO distillation update.

        Args:
            request: Distillation payload.

        Returns:
            Distillation response payload.
        """
        try:
            return self.trainer.distill(request)
        finally:
            self.trainer.offload_base_model()

    @modal.method()
    def health_check(self) -> WorkerHealthResponse:
        """Return worker health and current CUDA memory metrics."""
        import torch

        return WorkerHealthResponse(
            status="healthy",
            model=self.base_model_id,
            device=str(self.trainer.device),
            vram_allocated_gb=torch.cuda.memory_allocated() / 1e9,
            vram_reserved_gb=torch.cuda.memory_reserved() / 1e9,
        )


if __name__ == "__main__":
    with app.run():
        worker = DistillWorker()
        result = worker.health_check.remote()
        print(f"Health check: {result}")
