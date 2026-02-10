"""DistillWorker: Modal-based training worker for SDPO continual distillation.

This Modal service handles the core training loop:
1. Load user's LoRA adapter from Modal Volume
2. Run base model forward pass (for KL regularization)
3. Run student (base + LoRA) forward pass
4. Get teacher logprobs from TeacherService
5. Compute SDPO loss (distillation + KL to base)
6. Backward pass and optimizer step on LoRA params only
7. Save updated LoRA back to Modal Volume

Key features:
- GPU memory snapshots for sub-second cold starts (~2s vs ~15-20s)
- Frozen base model with per-request LoRA loading
- SDPO loss: KL distillation from teacher + KL regularization to base
- Simple Modal Volume storage (no S3/AWS needed)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile

import modal

from .sdpo_loss import compute_sdpo_loss
from .storage import (
    LORA_MOUNT_PATH,
    cleanup_local_lora,
    load_lora,
    lora_volume,
    save_lora,
)
from .teacher import parse_teacher_result
from .types import SDPOLossInput, TrainingConfig

logger = logging.getLogger(__name__)

# Modal app (shared with teacher)
app = modal.App("claas-distill")

# Volume for model weights
model_volume = modal.Volume.from_name("claas-models", create_if_missing=True)

# Training image with all dependencies
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
    .env({
        "HF_HOME": "/models/hf_cache",
        "TRANSFORMERS_CACHE": "/models/hf_cache",
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "HUGGING_FACE_HUB_TOKEN": os.environ.get("HF_TOKEN", ""),
    })
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
    """Training worker for SDPO continual distillation."""

    base_model_id: str = os.environ.get(
        "CLAAS_BASE_MODEL_ID",
        "Qwen/Qwen3-8B",
    )
    attn_implementation: str = os.environ.get(
        "CLAAS_ATTN_IMPLEMENTATION",
        "flash_attention_2",
    )

    @modal.enter(snap=True)
    def load_base_model(self):
        """Load base model directly to GPU.

        The entire GPU state — model weights in VRAM, CUDA context, flash
        attention kernels — will be captured in the snapshot. Subsequent
        cold boots restore from snapshot in ~2s instead of re-loading (~15-20s).
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading base model {self.base_model_id}...")

        self.device = torch.device("cuda")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            trust_remote_code=True,
            cache_dir="/models/hf_cache",
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation=self.attn_implementation,
            cache_dir="/models/hf_cache",
        )

        # Freeze all base parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Pre-warm: run a dummy forward pass to trigger CUDA kernel compilation
        # This gets captured in the snapshot so future boots skip it
        dummy_ids = self.tokenizer.encode("Hello", return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.base_model(input_ids=dummy_ids)
        del dummy_ids
        torch.cuda.empty_cache()

        # Store classes/modules for later use (avoids lazy imports in methods)
        self.optimizer_cls = torch.optim.AdamW
        self.F = torch.nn.functional

        vram_gb = torch.cuda.memory_allocated() / 1e9
        print(f"Base model loaded. VRAM: {vram_gb:.2f} GB")
        print("Snapshot will capture this state.")

    def _load_or_create_lora(self, lora_path: str):
        """Load existing LoRA weights or create fresh LoRA from config.

        For newly initialized LoRAs (config only, no weights), creates a fresh
        LoRA using get_peft_model() with the config. For trained LoRAs with
        saved weights, loads them with PeftModel.from_pretrained().

        Args:
            lora_path: Path to local LoRA directory

        Returns:
            PeftModel with LoRA applied to base model
        """
        from peft import LoraConfig, PeftModel, get_peft_model

        # Check if adapter weights exist
        weights_safetensors = os.path.join(lora_path, "adapter_model.safetensors")
        weights_bin = os.path.join(lora_path, "adapter_model.bin")
        has_weights = os.path.exists(weights_safetensors) or os.path.exists(weights_bin)

        if has_weights:
            # Load existing trained LoRA
            model = PeftModel.from_pretrained(
                self.base_model,
                lora_path,
                is_trainable=True,
            )
        else:
            # Fresh LoRA - create from config using get_peft_model()
            config_path = os.path.join(lora_path, "adapter_config.json")
            with open(config_path) as f:
                config_dict = json.load(f)

            lora_config = LoraConfig(
                r=config_dict.get("r", 16),
                lora_alpha=config_dict.get("lora_alpha", 32),
                target_modules=config_dict.get("target_modules"),
                lora_dropout=config_dict.get("lora_dropout", 0.0),
                bias=config_dict.get("bias", "none"),
                task_type=config_dict.get("task_type", "CAUSAL_LM"),
            )

            model = get_peft_model(self.base_model, lora_config)

        return model

    def _build_self_teacher_topk(self, student_logits, top_k: int):
        """Build top-K teacher distribution from detached student logits."""
        import torch

        with torch.no_grad():
            student_log_probs = self.F.log_softmax(student_logits.detach(), dim=-1)
            vocab_size = student_log_probs.shape[-1]
            k = min(max(1, top_k), vocab_size)
            top_logprobs, top_indices = torch.topk(student_log_probs[0], k=k, dim=-1)
        return top_logprobs, top_indices

    @modal.method()
    def distill(self, request: dict) -> dict:
        """Run a single SDPO distillation step.

        Args:
            request: Distillation request with:
                - lora_id: LoRA identifier (e.g., "user123/coder-v1")
                - prompt: User prompt
                - response: Student's response to learn from
                - feedback: Feedback about the response quality
                - training: Training config (learning_rate, alpha, clip_eps, etc.)

        Returns:
            dict with:
                - lora_id: Updated LoRA identifier
                - metadata: Training metrics
        """
        import torch

        F = self.F  # Use pre-imported functional module
        torch.cuda.empty_cache()

        # Validate request
        if "lora_id" not in request:
            raise ValueError("Missing required field: lora_id")
        if "prompt" not in request:
            raise ValueError("Missing required field: prompt")
        if "response" not in request:
            raise ValueError("Missing required field: response")
        if "feedback" not in request:
            raise ValueError("Missing required field: feedback")

        # Parse training config with typed defaults
        config = TrainingConfig.model_validate(request.get("training", {}))

        # 1. Load LoRA from Modal Volume
        lora_local_path = load_lora(request["lora_id"])

        try:
            model = self._load_or_create_lora(lora_local_path)
            model.train()
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            cleanup_local_lora(lora_local_path)
            raise RuntimeError(f"Failed to load LoRA: {e}") from e
        except ValueError as e:
            # PEFT config validation errors
            cleanup_local_lora(lora_local_path)
            raise RuntimeError(f"Invalid LoRA configuration: {e}") from e

        # Set up optimizer on LoRA params only
        lora_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self.optimizer_cls(
            lora_params,
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # 2. Tokenize
        prompt_ids = self.tokenizer.encode(
            request["prompt"],
            add_special_tokens=True,
            return_tensors="pt",
        ).to(self.device)

        response_ids = self.tokenizer.encode(
            request["response"],
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)

        full_ids = torch.cat([prompt_ids, response_ids], dim=-1)
        response_start = prompt_ids.shape[-1]
        T_resp = response_ids.shape[-1]

        # Response mask
        response_mask = torch.zeros(1, full_ids.shape[-1], device=self.device)
        response_mask[:, response_start:] = 1.0

        # 3. Base model forward pass (for KL regularization, no grad)
        with torch.no_grad():
            base_output = self.base_model(input_ids=full_ids)
            base_logits = base_output.logits[:, response_start - 1 : -1, :]  # (1, T_resp, V)
            base_logprobs = F.log_softmax(base_logits, dim=-1).gather(
                -1, response_ids[:, :T_resp].unsqueeze(-1)
            ).squeeze(-1)  # (1, T_resp)

        # 4. Student forward pass (WITH gradient)
        student_output = model(input_ids=full_ids)
        # Logits at positions [response_start-1, ..., end-1] predict tokens at [response_start, ..., end]
        student_logits = student_output.logits[:, response_start - 1 : -1, :]  # (1, T_resp, V)

        # Old logprobs for importance sampling ratio
        # These should come from the inference server that generated the rollout
        rollout_logprobs = request.get("rollout_logprobs")
        if rollout_logprobs is not None:
            old_student_logprobs = torch.tensor(
                rollout_logprobs,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)  # Add batch dimension
            # Truncate/pad to match response length
            if old_student_logprobs.shape[1] > T_resp:
                old_student_logprobs = old_student_logprobs[:, :T_resp]
            elif old_student_logprobs.shape[1] < T_resp:
                # Pad with zeros (log(1) = 0, neutral for IS ratio)
                pad_size = T_resp - old_student_logprobs.shape[1]
                old_student_logprobs = F.pad(old_student_logprobs, (0, pad_size), value=0.0)
        else:
            # Fallback: compute from current model (incorrect for off-policy)
            logger.warning(
                "rollout_logprobs not provided; computing from current model. "
                "For proper off-policy learning, pass logprobs from the inference server."
            )
            with torch.no_grad():
                old_student_logprobs = F.log_softmax(
                    student_logits.detach(), dim=-1
                ).gather(-1, response_ids[:, :T_resp].unsqueeze(-1)).squeeze(-1)

        # 5. Build/parse teacher logprobs
        teacher_result = request.get("teacher_result")
        if teacher_result:
            teacher_logprobs, teacher_indices = parse_teacher_result(
                teacher_result, str(self.device)
            )
            teacher_mode = "remote"
        else:
            teacher_logprobs, teacher_indices = self._build_self_teacher_topk(
                student_logits, config.teacher_top_k
            )
            teacher_mode = "self"

        # Ensure dimensions match
        if teacher_logprobs.shape[0] != T_resp:
            # Truncate or pad to match
            if teacher_logprobs.shape[0] > T_resp:
                teacher_logprobs = teacher_logprobs[:T_resp]
                teacher_indices = teacher_indices[:T_resp]
            else:
                # Pad with zeros (shouldn't happen normally)
                pad_size = T_resp - teacher_logprobs.shape[0]
                teacher_logprobs = F.pad(teacher_logprobs, (0, 0, 0, pad_size), value=-100.0)
                teacher_indices = F.pad(teacher_indices, (0, 0, 0, pad_size), value=0)

        # Add batch dimension
        teacher_logprobs = teacher_logprobs.unsqueeze(0)
        teacher_indices = teacher_indices.unsqueeze(0)

        # 6. Compute SDPO loss
        loss_input = SDPOLossInput(
            student_logits=student_logits,
            teacher_logprobs=teacher_logprobs,
            teacher_indices=teacher_indices,
            base_logprobs=base_logprobs,
            response_mask=response_mask[:, response_start:],
            old_student_logprobs=old_student_logprobs,
            response_ids=response_ids[:, :T_resp],
            alpha=config.alpha,
            is_clip=config.is_clip,
            kl_reg_weight=config.kl_reg_weight,
        )
        loss_dict = compute_sdpo_loss(loss_input)

        # 7. Backward + clip + step
        loss_dict["loss"].backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # 8. Save LoRA to Modal Volume
        save_dir = tempfile.mkdtemp(prefix="lora_updated_")
        try:
            model.save_pretrained(save_dir)
            new_lora_id = save_lora(save_dir, request["lora_id"])
        finally:
            cleanup_local_lora(save_dir)

        # 9. Cleanup
        del model, optimizer, student_output, student_logits, base_output
        cleanup_local_lora(lora_local_path)
        torch.cuda.empty_cache()

        return {
            "lora_id": new_lora_id,
            "metadata": {
                "total_loss": loss_dict["loss"].item(),
                "distill_loss": loss_dict["distill_loss"],
                "kl_reg": loss_dict["kl_reg"],
                "mean_is_ratio": loss_dict["mean_is_ratio"],
                "clip_fraction": loss_dict["clip_fraction"],
                "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
                "tokens_processed": T_resp,
                "teacher_mode": teacher_mode,
            },
        }

    @modal.method()
    def health_check(self) -> dict:
        """Check if the worker is ready."""
        import torch

        return {
            "status": "healthy",
            "model": self.base_model_id,
            "device": str(self.device),
            "vram_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "vram_reserved_gb": torch.cuda.memory_reserved() / 1e9,
        }


# For local testing
if __name__ == "__main__":
    with app.run():
        worker = DistillWorker()
        result = worker.health_check.remote()
        print(f"Health check: {result}")
