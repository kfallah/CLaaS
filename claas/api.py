"""CLaaS API: FastAPI web endpoint for SDPO continual distillation.

This module provides the REST API for the distillation service.

Endpoints:
- POST /v1/distill: Run a single SDPO distillation step
- POST /v1/lora/init: Initialize a new LoRA adapter
- GET /v1/lora: List all LoRA adapters
- GET /v1/health: Health check

Example usage:
    curl -X POST https://your-modal-app.modal.run/v1/distill \\
        -H "Content-Type: application/json" \\
        -d '{
            "lora_id": "user123/coder-v1",
            "prompt": "Write a function to calculate factorial",
            "response": "def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n-1)",
            "feedback": "Good recursive solution",
            "training": {
                "learning_rate": 1e-4,
                "alpha": 0.5
            }
        }'
"""

from __future__ import annotations

import asyncio
from typing import Any

import modal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .storage import LORA_MOUNT_PATH, create_initial_lora, list_loras, lora_exists, lora_volume
from .types import TrainingConfig
from .worker import DistillWorker
from .worker import app as modal_app

# FastAPI app
web_app = FastAPI(
    title="CLaaS API",
    description="Continual Learning as a Service - SDPO-style distillation",
    version="0.1.0",
)


# Request/Response Models


class DistillRequest(BaseModel):
    """Request for a distillation step."""

    lora_id: str = Field(
        ...,
        description="LoRA identifier (e.g., 'user123/coder-v1')",
    )
    prompt: str = Field(
        ...,
        min_length=1,
        description="User prompt that generated the response",
    )
    response: str = Field(
        ...,
        min_length=1,
        description="Student's response to learn from",
    )
    feedback: str = Field(
        ...,
        min_length=1,
        description="Feedback about response quality",
    )
    rollout_logprobs: list[float] | None = Field(
        default=None,
        description="Log-probabilities from the inference server that generated the rollout. "
        "Required for proper off-policy IS correction. If not provided, logprobs are computed "
        "from the current model (which is incorrect for off-policy learning).",
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training configuration",
    )


class DistillResponse(BaseModel):
    """Response from a distillation step."""

    lora_id: str = Field(
        ...,
        description="Updated LoRA identifier",
    )
    metadata: dict[str, Any] = Field(
        ...,
        description="Training metrics and diagnostics",
    )


class LoraInitRequest(BaseModel):
    """Request to initialize a new LoRA adapter."""

    lora_id: str = Field(
        ...,
        description="LoRA identifier (e.g., 'user123/coder-v1')",
    )
    base_model: str = Field(
        default="Qwen/Qwen3-Coder-Next",
        description="Base model the LoRA will be applied to",
    )
    lora_r: int = Field(
        default=16,
        ge=4,
        le=128,
        description="LoRA rank",
    )
    lora_alpha: int = Field(
        default=32,
        ge=8,
        le=256,
        description="LoRA alpha scaling factor",
    )
    target_modules: list[str] | None = Field(
        default=None,
        description="Modules to apply LoRA to (defaults to attention + MLP)",
    )


class LoraInitResponse(BaseModel):
    """Response from LoRA initialization."""

    lora_id: str = Field(
        ...,
        description="LoRA identifier of the initialized adapter",
    )


class LoraListResponse(BaseModel):
    """Response listing all LoRA adapters."""

    loras: list[str] = Field(
        ...,
        description="List of LoRA identifiers",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    worker: dict[str, Any] | None = None
    teacher: dict[str, Any] | None = None


# API Endpoints


@web_app.post("/v1/distill", response_model=DistillResponse)
async def distill(request: DistillRequest) -> DistillResponse:
    """Run a single SDPO distillation step.

    This endpoint:
    1. Loads the user's LoRA from Modal Volume
    2. Runs the student model forward pass
    3. Gets teacher logprobs from the vLLM teacher service
    4. Computes SDPO loss (JSD-based policy gradient)
    5. Updates LoRA parameters
    6. Saves the updated LoRA back to Modal Volume

    Returns the new LoRA ID and training metrics.
    """
    try:
        # Validate LoRA exists (run sync function in thread pool to avoid blocking)
        exists = await asyncio.to_thread(lora_exists, request.lora_id)
        if not exists:
            raise HTTPException(
                status_code=404,
                detail=f"LoRA not found: {request.lora_id}",
            )

        # Call the Modal worker (use .remote.aio() in async context)
        worker = DistillWorker()
        result = await worker.distill.remote.aio(request.model_dump())

        return DistillResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Distillation failed: {str(e)}",
        ) from e


@web_app.post("/v1/lora/init", response_model=LoraInitResponse)
async def init_lora(request: LoraInitRequest) -> LoraInitResponse:
    """Initialize a new LoRA adapter.

    Creates a new LoRA adapter configuration in the Modal Volume.
    The adapter will have zero weights initially and will be trained
    through distill calls.
    """
    try:
        # Run sync function in thread pool to avoid blocking
        lora_id = await asyncio.to_thread(
            create_initial_lora,
            lora_id=request.lora_id,
            base_model_name=request.base_model,
            lora_r=request.lora_r,
            lora_alpha=request.lora_alpha,
            target_modules=request.target_modules,
        )
        return LoraInitResponse(lora_id=lora_id)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LoRA initialization failed: {str(e)}",
        ) from e


@web_app.get("/v1/lora", response_model=LoraListResponse)
async def list_lora_adapters(prefix: str = "") -> LoraListResponse:
    """List all LoRA adapters.

    Args:
        prefix: Optional prefix to filter by (e.g., 'user123/')
    """
    try:
        # Run sync function in thread pool to avoid blocking
        loras = await asyncio.to_thread(list_loras, prefix)
        return LoraListResponse(loras=loras)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list LoRAs: {str(e)}",
        ) from e


@web_app.get("/v1/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check health of the API and backing services."""
    result: dict[str, Any] = {"status": "healthy"}

    try:
        worker = DistillWorker()
        result["worker"] = await worker.health_check.remote.aio()
    except Exception as e:
        result["worker"] = {"status": "unhealthy", "error": str(e)}
        result["status"] = "degraded"

    try:
        from .teacher import TeacherService

        teacher = TeacherService()
        result["teacher"] = await teacher.health_check.remote.aio()
    except Exception as e:
        result["teacher"] = {"status": "unhealthy", "error": str(e)}
        result["status"] = "degraded"

    return HealthResponse(**result)


@web_app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "CLaaS API",
        "version": "0.1.0",
        "description": "Continual Learning as a Service - SDPO-style distillation",
        "docs": "/docs",
    }


# Mount FastAPI to Modal
@modal_app.function(
    image=modal.Image.debian_slim(python_version="3.11").pip_install(
        "modal>=1.0.0",
        "fastapi>=0.110.0",
        "pydantic>=2.6.0",
    ),
    volumes={LORA_MOUNT_PATH: lora_volume},
)
@modal.asgi_app()
def fastapi_app():
    """Modal ASGI app entry point."""
    return web_app
