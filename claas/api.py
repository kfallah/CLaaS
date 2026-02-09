"""CLaaS API: FastAPI web endpoint for SDPO continual distillation.

This module provides the REST API for the distillation service.

Endpoints:
- POST /v1/distill: Run a single SDPO distillation step
- POST /v1/distill/lite: Lightweight distillation with pre-computed teacher logprobs
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

from typing import Any

import modal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .storage import create_initial_lora, list_loras, lora_exists, lora_volume, LORA_MOUNT_PATH
from .worker import DistillWorker, app as modal_app

# FastAPI app
web_app = FastAPI(
    title="CLaaS API",
    description="Continual Learning as a Service - SDPO-style distillation",
    version="0.1.0",
)


# Request/Response Models


class TrainingConfig(BaseModel):
    """Training configuration for distillation."""

    learning_rate: float = Field(
        default=1e-4,
        ge=1e-6,
        le=1e-2,
        description="Learning rate for LoRA parameter updates",
    )
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="JSD interpolation (0.5 = symmetric JSD, SDPO default)",
    )
    clip_eps: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="PPO clip range for importance sampling",
    )
    max_grad_norm: float = Field(
        default=1.0,
        ge=0.0,
        description="Maximum gradient norm for clipping",
    )
    jsd_reg_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Weight for logit-level JSD regularizer",
    )
    teacher_top_k: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Number of top logprobs to request from teacher",
    )


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
    feedback: str | None = Field(
        default=None,
        description="Optional feedback about response quality",
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Training configuration",
    )


class DistillLiteRequest(BaseModel):
    """Request for lightweight distillation with pre-computed teacher logprobs."""

    lora_id: str = Field(
        ...,
        description="LoRA identifier",
    )
    prompt: str = Field(
        ...,
        min_length=1,
        description="User prompt",
    )
    response: str = Field(
        ...,
        min_length=1,
        description="Student's response",
    )
    teacher_logprobs: list[float] = Field(
        ...,
        description="Pre-computed teacher log-probs for each response token",
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
        default="Qwen/Qwen2.5-Coder-3B-Instruct",
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
        # Validate LoRA exists
        if not lora_exists(request.lora_id):
            raise HTTPException(
                status_code=404,
                detail=f"LoRA not found: {request.lora_id}",
            )

        # Call the Modal worker
        worker = DistillWorker()
        result = worker.distill.remote(request.model_dump())

        return DistillResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Distillation failed: {str(e)}",
        ) from e


@web_app.post("/v1/distill/lite", response_model=DistillResponse)
async def distill_lite(request: DistillLiteRequest) -> DistillResponse:
    """Run a lightweight distillation step with pre-computed teacher logprobs.

    This is the "lite" mode for use with external teacher APIs (e.g., Fireworks)
    that have limited logprob support (K=5). It uses only the token-level
    advantage computation, without the logit-level JSD regularizer.

    The caller must pre-compute teacher log-probabilities for each response
    token using their own teacher API.
    """
    try:
        if not lora_exists(request.lora_id):
            raise HTTPException(
                status_code=404,
                detail=f"LoRA not found: {request.lora_id}",
            )

        worker = DistillWorker()
        result = worker.distill_lite.remote(request.model_dump())

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
        lora_id = create_initial_lora(
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
        loras = list_loras(prefix)
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
        result["worker"] = worker.health_check.remote()
    except Exception as e:
        result["worker"] = {"status": "unhealthy", "error": str(e)}
        result["status"] = "degraded"

    try:
        from .teacher import TeacherService

        teacher = TeacherService()
        result["teacher"] = teacher.health_check.remote()
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
        "fastapi>=0.110.0",
        "pydantic>=2.6.0",
    ),
    volumes={LORA_MOUNT_PATH: lora_volume},
)
@modal.asgi_app()
def fastapi_app():
    """Modal ASGI app entry point."""
    return web_app
