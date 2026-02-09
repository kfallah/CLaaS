"""DistillWorker: Modal-based training worker for SDPO continual distillation.

This Modal service handles the core training loop:
1. Load user's LoRA adapter from Modal Volume
2. Run student forward pass
3. Get teacher logprobs from TeacherService
4. Compute SDPO loss (JSD-based policy gradient)
5. Backward pass and optimizer step on LoRA params only
6. Save updated LoRA back to Modal Volume

Key features:
- GPU memory snapshots for sub-second cold starts (~2s vs ~15-20s)
- Frozen base model with per-request LoRA loading
- Full SDPO loss with logit-level JSD regularizer
- Simple Modal Volume storage (no S3/AWS needed)
"""

from __future__ import annotations

import tempfile

import modal

from .sdpo_loss import compute_sdpo_loss, compute_token_level_only_loss
from .storage import (
    LORA_MOUNT_PATH,
    cleanup_local_lora,
    load_lora,
    lora_volume,
    save_lora,
)
from .teacher import TeacherService, format_teacher_prompt, parse_teacher_result

# Modal app (shared with teacher)
app = modal.App("claas-distill")

# Volume for model weights
model_volume = modal.Volume.from_name("claas-models", create_if_missing=True)

# Training image with all dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "accelerate>=0.27.0",
        "bitsandbytes>=0.42.0",
        "flash-attn>=2.5.0",
        "safetensors>=0.4.0",
    )
    .env({
        "HF_HOME": "/models/hf_cache",
        "TRANSFORMERS_CACHE": "/models/hf_cache",
    })
)


@app.cls(
    gpu="L40S",
    image=training_image,
    volumes={
        "/models": model_volume,
        LORA_MOUNT_PATH: lora_volume,
    },
    container_idle_timeout=300,
    timeout=120,
    enable_memory_snapshot=True,
)
class DistillWorker:
    """Training worker for SDPO continual distillation."""

    base_model_id: str = "Qwen/Qwen2.5-Coder-3B-Instruct"

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
            attn_implementation="flash_attention_2",
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

        # Store optimizer class for later
        self.optimizer_cls = torch.optim.AdamW

        vram_gb = torch.cuda.memory_allocated() / 1e9
        print(f"Base model loaded. VRAM: {vram_gb:.2f} GB")
        print("Snapshot will capture this state.")

    @modal.method()
    def distill(self, request: dict) -> dict:
        """Run a single SDPO distillation step.

        Args:
            request: Distillation request with:
                - lora_id: LoRA identifier (e.g., "user123/coder-v1")
                - prompt: User prompt
                - response: Student's response to learn from
                - feedback: Optional feedback about the response
                - training: Training config (learning_rate, alpha, clip_eps, etc.)

        Returns:
            dict with:
                - lora_id: Updated LoRA identifier
                - metadata: Training metrics
        """
        import torch
        import torch.nn.functional as F
        from peft import PeftModel

        torch.cuda.empty_cache()

        # Validate request
        if "lora_id" not in request:
            raise ValueError("Missing required field: lora_id")
        if "prompt" not in request:
            raise ValueError("Missing required field: prompt")
        if "response" not in request:
            raise ValueError("Missing required field: response")

        training_config = request.get("training", {})
        lr = training_config.get("learning_rate", 1e-4)
        alpha = training_config.get("alpha", 0.5)
        clip_eps = training_config.get("clip_eps", 0.2)
        max_grad_norm = training_config.get("max_grad_norm", 1.0)
        jsd_reg_weight = training_config.get("jsd_reg_weight", 0.5)
        teacher_top_k = training_config.get("teacher_top_k", 100)

        # 1. Load LoRA from Modal Volume
        lora_local_path = load_lora(request["lora_id"])

        try:
            model = PeftModel.from_pretrained(
                self.base_model,
                lora_local_path,
                is_trainable=True,
            )
            model.train()
        except Exception as e:
            cleanup_local_lora(lora_local_path)
            raise RuntimeError(f"Failed to load LoRA: {e}") from e

        # Set up optimizer on LoRA params only
        lora_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self.optimizer_cls(
            lora_params,
            lr=lr,
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

        # 3. Student forward pass (WITH gradient)
        student_output = model(input_ids=full_ids)
        # Logits at positions [response_start-1, ..., end-1] predict tokens at [response_start, ..., end]
        student_logits = student_output.logits[:, response_start - 1 : -1, :]  # (1, T_resp, V)

        # Old logprobs (for importance sampling - detached snapshot)
        with torch.no_grad():
            old_student_logprobs = F.log_softmax(
                student_logits.detach(), dim=-1
            ).gather(-1, response_ids[:, :T_resp].unsqueeze(-1)).squeeze(-1)

        # 4. Get teacher logprobs from vLLM sidecar
        teacher_prompt = format_teacher_prompt(
            request["prompt"],
            request["response"],
            request.get("feedback"),
        )

        teacher_service = TeacherService()
        teacher_result = teacher_service.score_tokens.remote(
            prompts=[teacher_prompt],
            completions=[request["response"]],
            top_k=teacher_top_k,
        )

        if not teacher_result or not teacher_result[0]:
            cleanup_local_lora(lora_local_path)
            raise RuntimeError("Failed to get teacher logprobs")

        teacher_logprobs, teacher_indices = parse_teacher_result(
            teacher_result[0], self.device
        )

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

        # 5. Compute SDPO loss
        loss_dict = compute_sdpo_loss(
            student_logits=student_logits,
            teacher_logprobs=teacher_logprobs,
            teacher_indices=teacher_indices,
            response_mask=response_mask[:, response_start:],
            old_student_logprobs=old_student_logprobs,
            response_ids=response_ids[:, :T_resp],
            alpha=alpha,
            clip_eps=clip_eps,
            jsd_reg_weight=jsd_reg_weight,
        )

        # 6. Backward + clip + step
        loss_dict["loss"].backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # 7. Save LoRA to Modal Volume
        save_dir = tempfile.mkdtemp(prefix="lora_updated_")
        model.save_pretrained(save_dir)
        new_lora_id = save_lora(save_dir, request["lora_id"])
        cleanup_local_lora(save_dir)

        # 8. Cleanup
        del model, optimizer, student_output, student_logits
        cleanup_local_lora(lora_local_path)
        torch.cuda.empty_cache()

        return {
            "lora_id": new_lora_id,
            "metadata": {
                "total_loss": loss_dict["loss"].item(),
                "pg_loss": loss_dict["pg_loss"],
                "jsd_reg": loss_dict["jsd_reg"],
                "mean_advantage": loss_dict["mean_advantage"],
                "frac_positive_advantage": loss_dict["frac_positive_advantage"],
                "mean_is_ratio": loss_dict["mean_is_ratio"],
                "clip_fraction": loss_dict["clip_fraction"],
                "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
                "tokens_processed": T_resp,
            },
        }

    @modal.method()
    def distill_lite(self, request: dict) -> dict:
        """Run a lightweight distillation step using token-level loss only.

        This is the "lite" mode for use with Fireworks' K=5 logprobs,
        where we don't have enough teacher logprobs for the full
        logit-level JSD regularizer.

        Args:
            request: Same as distill(), but teacher_logprobs should be
                     pre-computed externally (e.g., from Fireworks)

        Returns:
            dict with lora_id and metadata
        """
        import torch
        import torch.nn.functional as F
        from peft import PeftModel

        torch.cuda.empty_cache()

        # Validate request
        if "lora_id" not in request:
            raise ValueError("Missing required field: lora_id")
        if "prompt" not in request:
            raise ValueError("Missing required field: prompt")
        if "response" not in request:
            raise ValueError("Missing required field: response")
        if "teacher_logprobs" not in request:
            raise ValueError("Missing required field: teacher_logprobs for lite mode")

        training_config = request.get("training", {})
        lr = training_config.get("learning_rate", 1e-4)
        alpha = training_config.get("alpha", 0.5)
        clip_eps = training_config.get("clip_eps", 0.2)
        max_grad_norm = training_config.get("max_grad_norm", 1.0)

        # Load LoRA
        lora_local_path = load_lora(request["lora_id"])

        try:
            model = PeftModel.from_pretrained(
                self.base_model,
                lora_local_path,
                is_trainable=True,
            )
            model.train()
        except Exception as e:
            cleanup_local_lora(lora_local_path)
            raise RuntimeError(f"Failed to load LoRA: {e}") from e

        lora_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self.optimizer_cls(lora_params, lr=lr, weight_decay=0.01)

        # Tokenize
        prompt_ids = self.tokenizer.encode(
            request["prompt"], add_special_tokens=True, return_tensors="pt"
        ).to(self.device)
        response_ids = self.tokenizer.encode(
            request["response"], add_special_tokens=False, return_tensors="pt"
        ).to(self.device)

        full_ids = torch.cat([prompt_ids, response_ids], dim=-1)
        response_start = prompt_ids.shape[-1]
        T_resp = response_ids.shape[-1]

        response_mask = torch.zeros(1, T_resp, device=self.device)
        response_mask[:, :] = 1.0

        # Student forward
        student_output = model(input_ids=full_ids)
        student_logits = student_output.logits[:, response_start - 1 : -1, :]

        with torch.no_grad():
            old_student_logprobs = F.log_softmax(
                student_logits.detach(), dim=-1
            ).gather(-1, response_ids[:, :T_resp].unsqueeze(-1)).squeeze(-1)

        # Teacher logprobs (pre-computed, from Fireworks)
        teacher_logprob_chosen = torch.tensor(
            request["teacher_logprobs"], device=self.device
        ).unsqueeze(0)

        # Compute token-level only loss
        loss_dict = compute_token_level_only_loss(
            student_logits=student_logits,
            teacher_logprob_chosen=teacher_logprob_chosen,
            response_mask=response_mask,
            old_student_logprobs=old_student_logprobs,
            response_ids=response_ids[:, :T_resp],
            alpha=alpha,
            clip_eps=clip_eps,
        )

        # Backward + step
        loss_dict["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # Save LoRA
        save_dir = tempfile.mkdtemp(prefix="lora_updated_")
        model.save_pretrained(save_dir)
        new_lora_id = save_lora(save_dir, request["lora_id"])
        cleanup_local_lora(save_dir)

        # Cleanup
        del model, optimizer, student_output, student_logits
        cleanup_local_lora(lora_local_path)
        torch.cuda.empty_cache()

        return {
            "lora_id": new_lora_id,
            "metadata": {
                "total_loss": loss_dict["loss"].item(),
                "pg_loss": loss_dict["pg_loss"],
                "mean_advantage": loss_dict["mean_advantage"],
                "frac_positive_advantage": loss_dict["frac_positive_advantage"],
                "mean_is_ratio": loss_dict["mean_is_ratio"],
                "clip_fraction": loss_dict["clip_fraction"],
                "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
                "tokens_processed": T_resp,
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
