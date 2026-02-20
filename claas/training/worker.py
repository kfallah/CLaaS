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
from typing import TYPE_CHECKING, cast

import modal

from claas.core.types import DistillBatchRequestPayload, SDPOLossInput
from claas.training.sdpo_loss import compute_sdpo_loss
from claas.training.storage import (
    LORA_MOUNT_PATH,
    cleanup_local_lora,
    load_lora,
    lora_volume,
    save_lora,
    save_lora_inplace,
)
from claas.training.teacher_helpers import (
    build_teacher_messages,
    parse_teacher_result,
    teacher_messages_to_chat_template,
)

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

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

    device: "torch.device"
    tokenizer: "PreTrainedTokenizerBase"
    base_model: "PreTrainedModel"
    base_model_id: str = os.environ.get(
        "CLAAS_BASE_MODEL_ID",
        "Qwen/Qwen3-8B",
    )
    attn_implementation: str = os.environ.get(
        "CLAAS_ATTN_IMPLEMENTATION",
        "sdpa",
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

        # Resolve HF cache: respect user env, fall back to Modal volume path.
        hf_cache = os.environ.get(
            "HF_HOME", os.environ.get("TRANSFORMERS_CACHE", "/models/hf_cache")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            trust_remote_code=True,
            cache_dir=hf_cache,
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation=self.attn_implementation,
            cache_dir=hf_cache,
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
                r=config_dict.get("r", 32),
                lora_alpha=config_dict.get("lora_alpha", 64),
                target_modules=config_dict.get("target_modules"),
                lora_dropout=config_dict.get("lora_dropout", 0.0),
                bias=config_dict.get("bias", "none"),
                task_type=config_dict.get("task_type", "CAUSAL_LM"),
            )

            model = get_peft_model(self.base_model, lora_config)

        return model

    def _build_self_teacher_topk(
        self,
        prompt: str,
        feedback: str,
        response_ids,
        top_k: int,
    ):
        """Build top-K teacher from base model conditioned on feedback.

        The SDPO teacher is the base student model with the feedback
        incorporated into its prompt.  A separate forward pass through
        the frozen base model produces logits that reflect what the model
        would generate *given* the feedback, creating a meaningful
        distillation signal (non-zero log-ratio vs. the student).

        Reference: https://arxiv.org/pdf/2601.20802, Section 3.
        """
        import torch

        messages = build_teacher_messages(prompt, feedback)
        template_messages = teacher_messages_to_chat_template(messages)
        teacher_prompt_ids_raw = self.tokenizer.apply_chat_template(
            template_messages,
            add_generation_prompt=False,
            return_tensors="pt",
            tokenize=True,
        )
        teacher_prompt_ids = cast("torch.Tensor", teacher_prompt_ids_raw).to(self.device)
        teacher_full_ids = torch.cat([teacher_prompt_ids, response_ids], dim=-1)
        teacher_resp_start = teacher_prompt_ids.shape[-1]
        T_resp = response_ids.shape[-1]

        with torch.no_grad():
            teacher_output = self.base_model(input_ids=teacher_full_ids)
            teacher_logits = teacher_output.logits[
                :, teacher_resp_start - 1 : -1, :
            ]  # (1, T_resp, V)
            log_probs = self.F.log_softmax(teacher_logits, dim=-1)
            vocab_size = log_probs.shape[-1]
            k = min(max(1, top_k), vocab_size)
            top_logprobs, top_indices = torch.topk(log_probs[0, :T_resp], k=k, dim=-1)

        del teacher_output, teacher_logits, log_probs
        del teacher_full_ids, teacher_prompt_ids
        torch.cuda.empty_cache()

        return top_logprobs, top_indices

    def _offload_base_model(self):
        """Move base model to CPU and release all GPU memory."""
        import torch

        torch.nn.Module.to(self.base_model, "cpu")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @modal.method()
    def distill(self, request: dict) -> dict:
        """Run one SDPO distillation update on a batch of feedback samples.

        Args:
            request: Distillation request payload containing ``lora_id``,
                ``training``, and ``samples``.

        Returns:
            Dictionary containing the updated LoRA ID and training metadata.
        """
        import torch

        F = self.F
        torch.cuda.empty_cache()

        if next(self.base_model.parameters()).device.type != "cuda":
            torch.nn.Module.to(self.base_model, self.device)

        payload = DistillBatchRequestPayload.model_validate(request)
        config = payload.training

        if len(payload.samples) == 0:
            raise ValueError("samples must contain at least one item")

        lora_local_path = load_lora(payload.lora_id)

        try:
            model = self._load_or_create_lora(lora_local_path)
            model.train()
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            cleanup_local_lora(lora_local_path)
            raise RuntimeError(f"Failed to load LoRA: {e}") from e
        except ValueError as e:
            cleanup_local_lora(lora_local_path)
            raise RuntimeError(f"Invalid LoRA configuration: {e}") from e

        lora_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self.optimizer_cls(
            lora_params,
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        batch_loss_tensors: list[torch.Tensor] = []
        batch_distill_loss: list[float] = []
        batch_kl_reg: list[float] = []
        batch_mean_is_ratio: list[float] = []
        batch_clip_fraction: list[float] = []
        tokens_processed = 0
        teacher_mode = config.teacher_mode

        for sample in payload.samples:
            if sample.prompt_token_ids is not None:
                prompt_ids = torch.tensor(
                    [sample.prompt_token_ids], device=self.device,
                )
            else:
                prompt_ids = self.tokenizer.encode(
                    sample.prompt,
                    add_special_tokens=True,
                    return_tensors="pt",
                ).to(self.device)
            if sample.response_token_ids is not None:
                response_ids = torch.tensor(
                    [sample.response_token_ids], device=self.device,
                )
            else:
                response_ids = self.tokenizer.encode(
                    sample.response,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).to(self.device)

            full_ids = torch.cat([prompt_ids, response_ids], dim=-1)
            response_start = prompt_ids.shape[-1]
            t_resp = response_ids.shape[-1]
            tokens_processed += int(t_resp)

            response_mask = torch.zeros(1, full_ids.shape[-1], device=self.device)
            response_mask[:, response_start:] = 1.0

            with torch.no_grad():
                base_output = self.base_model(input_ids=full_ids)
                base_logits = base_output.logits[:, response_start - 1 : -1, :]
                base_logprobs = F.log_softmax(base_logits, dim=-1).gather(
                    -1, response_ids[:, :t_resp].unsqueeze(-1)
                ).squeeze(-1)

            del base_output, base_logits
            torch.cuda.empty_cache()

            student_output = model(input_ids=full_ids)
            student_logits = student_output.logits[:, response_start - 1 : -1, :].contiguous()
            del student_output

            old_student_logprobs = torch.tensor(
                sample.rollout_logprobs,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            if old_student_logprobs.shape[1] > t_resp:
                old_student_logprobs = old_student_logprobs[:, :t_resp]
            elif old_student_logprobs.shape[1] < t_resp:
                raise ValueError(
                    "rollout_logprobs length must match response token length"
                )

            if config.teacher_mode == "remote":
                if sample.teacher_result is None:
                    raise ValueError("teacher_mode='remote' requires teacher_result")
                teacher_logprobs, teacher_indices = parse_teacher_result(
                    sample.teacher_result,
                    str(self.device),
                )
            else:
                teacher_prompt_source = sample.user_prompt or sample.prompt
                teacher_logprobs, teacher_indices = self._build_self_teacher_topk(
                    teacher_prompt_source,
                    sample.feedback,
                    response_ids,
                    config.teacher_top_k,
                )

            if teacher_logprobs.shape[0] != t_resp:
                raise ValueError("teacher logprob sequence length must match response length")

            loss_input = SDPOLossInput(
                student_logits=student_logits,
                teacher_logprobs=teacher_logprobs.unsqueeze(0),
                teacher_indices=teacher_indices.unsqueeze(0),
                base_logprobs=base_logprobs,
                response_mask=response_mask[:, response_start:],
                old_student_logprobs=old_student_logprobs,
                response_ids=response_ids[:, :t_resp],
                alpha=config.alpha,
                is_clip=config.is_clip,
                kl_reg_weight=config.kl_reg_weight,
            )
            loss_dict = compute_sdpo_loss(loss_input)
            batch_loss_tensors.append(loss_dict["loss"])
            batch_distill_loss.append(loss_dict["distill_loss"])
            batch_kl_reg.append(loss_dict["kl_reg"])
            batch_mean_is_ratio.append(loss_dict["mean_is_ratio"])
            batch_clip_fraction.append(loss_dict["clip_fraction"])

            del full_ids, prompt_ids, response_ids, response_mask
            del student_logits, base_logprobs, old_student_logprobs
            del teacher_logprobs, teacher_indices, loss_input

        mean_loss = torch.stack(batch_loss_tensors).mean()
        mean_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        model.gradient_checkpointing_disable()

        save_dir = tempfile.mkdtemp(prefix="lora_updated_")
        try:
            model.save_pretrained(save_dir)
            if payload.save_in_place:
                new_lora_id = save_lora_inplace(save_dir, payload.lora_id)
            else:
                new_lora_id = save_lora(save_dir, payload.lora_id)
        finally:
            cleanup_local_lora(save_dir)

        total_loss = mean_loss.item()
        distill_loss = sum(batch_distill_loss) / len(batch_distill_loss)
        kl_reg = sum(batch_kl_reg) / len(batch_kl_reg)
        mean_is_ratio = sum(batch_mean_is_ratio) / len(batch_mean_is_ratio)
        clip_fraction = sum(batch_clip_fraction) / len(batch_clip_fraction)
        grad_norm_val = grad_norm.item() if hasattr(grad_norm, "item") else grad_norm

        del model, optimizer, batch_loss_tensors
        cleanup_local_lora(lora_local_path)
        torch.cuda.empty_cache()

        self._offload_base_model()

        return {
            "lora_id": new_lora_id,
            "metadata": {
                "total_loss": total_loss,
                "distill_loss": distill_loss,
                "kl_reg": kl_reg,
                "mean_is_ratio": mean_is_ratio,
                "clip_fraction": clip_fraction,
                "grad_norm": grad_norm_val,
                "tokens_processed": tokens_processed,
                "teacher_mode": teacher_mode,
                "batch_size": len(payload.samples),
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
