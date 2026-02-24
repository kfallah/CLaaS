"""Shared SDPO distillation trainer for local and Modal runtimes."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import TYPE_CHECKING, cast

import torch

from claas.core.types import DistillBatchRequestPayload, DistillResponse, SDPOLossInput
from claas.training.sdpo_loss import compute_sdpo_loss
from claas.training.storage import (
    cleanup_local_lora,
    has_optimizer_state,
    load_lora,
    load_optimizer_state,
    save_lora,
    save_lora_inplace,
    save_optimizer_state,
)
from claas.training.teacher_helpers import (
    build_teacher_messages,
    teacher_messages_to_chat_template,
)

if TYPE_CHECKING:
    from peft import PeftMixedModel, PeftModel
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


logger = logging.getLogger(__name__)


class DistillationTrainer:
    """Runs one SDPO distillation update using a loaded base model."""

    device: "torch.device"
    tokenizer: "PreTrainedTokenizerBase"
    base_model: "PreTrainedModel"

    def __init__(self, base_model_id: str, attn_implementation: str) -> None:
        """Create a trainer bound to one base model configuration.

        Args:
            base_model_id: Hugging Face model identifier.
            attn_implementation: Attention backend identifier.
        """
        self.base_model_id = base_model_id
        self.attn_implementation = attn_implementation

    def load_base_model(self) -> None:
        """Load and freeze the base model on CUDA."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device("cuda")
        hf_cache = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_id,
            trust_remote_code=True,
            cache_dir=hf_cache,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation=self.attn_implementation,
            cache_dir=hf_cache,
        )
        for param in self.base_model.parameters():
            param.requires_grad = False

        dummy_ids = self.tokenizer.encode("Hello", return_tensors="pt").to(self.device)
        with torch.no_grad():
            _ = self.base_model(input_ids=dummy_ids)
        del dummy_ids
        torch.cuda.empty_cache()

        self.optimizer_cls = torch.optim.AdamW
        self.functional = torch.nn.functional

    def offload_base_model(self) -> None:
        """Move base model to CPU and release CUDA memory."""

        self.base_model.to(torch.device("cpu"))  # type: ignore[arg-type]  # functools.wraps confuses ty
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def _load_or_create_lora(self, lora_path: str) -> "PeftModel | PeftMixedModel":
        """Load existing LoRA weights or instantiate from config.

        Args:
            lora_path: Path to LoRA adapter directory.

        Returns:
            Trainable PEFT model.
        """
        from peft import LoraConfig, PeftModel, get_peft_model

        weights_safetensors = os.path.join(lora_path, "adapter_model.safetensors")
        weights_bin = os.path.join(lora_path, "adapter_model.bin")
        has_weights = os.path.exists(weights_safetensors) or os.path.exists(weights_bin)

        if has_weights:
            return PeftModel.from_pretrained(
                self.base_model,
                lora_path,
                is_trainable=True,
            )

        config_path = os.path.join(lora_path, "adapter_config.json")
        with open(config_path, encoding="utf-8") as file_obj:
            config_dict = json.load(file_obj)

        lora_config = LoraConfig(
            r=config_dict["r"],
            lora_alpha=config_dict["lora_alpha"],
            target_modules=config_dict["target_modules"],
            lora_dropout=config_dict["lora_dropout"],
            bias=config_dict["bias"],
            task_type=config_dict["task_type"],
        )
        return get_peft_model(self.base_model, lora_config)

    def _load_optimizer_state(
        self,
        lora_path: str,
        optimizer: "torch.optim.Optimizer",
    ) -> None:
        """Load optimizer state from local storage into an optimizer.

        Args:
            lora_path: Local LoRA directory path.
            optimizer: Optimizer instance to hydrate.
        """
        if not has_optimizer_state(lora_path):
            logger.warning("Optimizer state file missing for lora_path=%s", lora_path)
            return

        state_obj = load_optimizer_state(lora_path)
        optimizer.load_state_dict(state_obj)
        logger.info("Loaded optimizer state from lora_path=%s", lora_path)

    def _save_optimizer_state(
        self,
        optimizer: "torch.optim.Optimizer",
        save_dir: str,
    ) -> None:
        """Persist optimizer state into the local LoRA directory.

        Args:
            optimizer: Optimizer with current state.
            save_dir: Local LoRA directory path.
        """
        save_optimizer_state(save_dir, optimizer.state_dict())

    def _build_self_teacher_topk(
        self,
        prompt: str,
        feedback: str,
        response_ids: "torch.Tensor",
        top_k: int,
    ) -> tuple["torch.Tensor", "torch.Tensor", str]:
        """Build top-k teacher logits from the frozen base model.

        Args:
            prompt: User prompt.
            feedback: Critique text.
            response_ids: Tokenized sampled response.
            top_k: Number of logits to retain per token.

        Returns:
            Pair of top-k logprobs and indices for each response token.

        Reference:
            Kleine Buening et al. (2026), "Aligning Language Models from User Interactions"
            https://github.com/lasgroup/user_interactions/blob/main/online_sdpo_trainer.py
        """

        messages = build_teacher_messages(prompt, feedback)
        template_messages = teacher_messages_to_chat_template(messages)
        teacher_prompt_ids_raw = self.tokenizer.apply_chat_template(
            template_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        )
        # transformers >=5.x returns BatchEncoding; extract input_ids tensor
        if hasattr(teacher_prompt_ids_raw, "input_ids"):
            teacher_prompt_ids_raw = teacher_prompt_ids_raw.input_ids
        teacher_prompt_ids = cast("torch.Tensor", teacher_prompt_ids_raw).to(self.device)
        teacher_full_ids = torch.cat([teacher_prompt_ids, response_ids], dim=-1)
        teacher_scored_text = self.tokenizer.decode(teacher_full_ids[0].tolist(), skip_special_tokens=False)
        teacher_resp_start = teacher_prompt_ids.shape[-1]
        response_token_count = response_ids.shape[-1]

        with torch.no_grad():
            teacher_output = self.base_model(input_ids=teacher_full_ids)
            teacher_logits = teacher_output.logits[:, teacher_resp_start - 1 : -1, :]
            log_probs = self.functional.log_softmax(teacher_logits, dim=-1)
            vocab_size = log_probs.shape[-1]
            k = min(max(1, top_k), vocab_size)
            top_logprobs, top_indices = torch.topk(log_probs[0, :response_token_count], k=k, dim=-1)

        del teacher_output, teacher_logits, log_probs
        del teacher_full_ids, teacher_prompt_ids
        torch.cuda.empty_cache()
        return top_logprobs, top_indices, teacher_scored_text

    def distill(self, payload: DistillBatchRequestPayload) -> DistillResponse:
        """Run one SDPO distillation step.

        Args:
            payload: Distillation request payload.

        Returns:
            Distillation response with metrics.
        """

        torch.cuda.empty_cache()
        if next(self.base_model.parameters()).device.type != "cuda":
            torch.nn.Module.to(self.base_model, self.device)

        config = payload.training
        if len(payload.samples) == 0:
            raise ValueError("samples must contain at least one item")

        lora_local_path = load_lora(payload.lora_id)
        try:
            try:
                model = self._load_or_create_lora(lora_local_path)
                model.train()
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
            except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as error:
                raise RuntimeError(f"Failed to initialize LoRA adapter: {error}") from error

            lora_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
            optimizer = self.optimizer_cls(
                lora_params,
                lr=config.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.01,
            )
            self._load_optimizer_state(lora_local_path, optimizer)

            batch_loss_tensors: list[torch.Tensor] = []
            batch_distill_loss: list[float] = []
            batch_kl_reg: list[float] = []
            batch_mean_is_ratio: list[float] = []
            batch_clip_fraction: list[float] = []
            batch_teacher_scored_texts: list[str] = []
            tokens_processed = 0

            for sample in payload.samples:
                prompt_ids = torch.tensor(
                    [sample.prompt_token_ids],
                    device=self.device,
                    dtype=torch.int64,
                )
                response_ids = torch.tensor(
                    [sample.response_token_ids],
                    device=self.device,
                    dtype=torch.int64,
                )

                full_ids = torch.cat([prompt_ids, response_ids], dim=-1)
                response_start = prompt_ids.shape[-1]
                response_token_count = response_ids.shape[-1]
                tokens_processed += int(response_token_count)

                response_mask = torch.zeros(1, full_ids.shape[-1], device=self.device)
                response_mask[:, response_start:] = 1.0

                with torch.no_grad():
                    base_output = self.base_model(input_ids=full_ids)
                    base_logits = base_output.logits[:, response_start - 1 : -1, :]
                    base_logprobs = self.functional.log_softmax(base_logits, dim=-1).gather(
                        -1, response_ids[:, :response_token_count].unsqueeze(-1)
                    ).squeeze(-1)

                del base_output, base_logits
                torch.cuda.empty_cache()

                student_output = model(input_ids=full_ids)
                student_logits = student_output.logits[:, response_start - 1 : -1, :].contiguous()
                del student_output

                old_student_logprobs = torch.tensor(
                    sample.response_logprobs,
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                if old_student_logprobs.shape[1] > response_token_count:
                    old_student_logprobs = old_student_logprobs[:, :response_token_count]
                elif old_student_logprobs.shape[1] < response_token_count:
                    raise ValueError("response_logprobs length must match response token length")

                teacher_logprobs, teacher_indices, teacher_scored_text = self._build_self_teacher_topk(
                    sample.user_prompt,
                    sample.feedback,
                    response_ids,
                    config.teacher_top_k,
                )

                if teacher_logprobs.shape[0] != response_token_count:
                    raise ValueError("teacher logprob sequence length must match response length")

                loss_input = SDPOLossInput(
                    student_logits=student_logits,
                    teacher_logprobs=teacher_logprobs.unsqueeze(0),
                    teacher_indices=teacher_indices.unsqueeze(0),
                    base_logprobs=base_logprobs,
                    response_mask=response_mask[:, response_start:],
                    old_student_logprobs=old_student_logprobs,
                    response_ids=response_ids[:, :response_token_count],
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
                batch_teacher_scored_texts.append(teacher_scored_text)

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
                self._save_optimizer_state(optimizer, save_dir)
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
            grad_norm_value = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)

            del model, optimizer, batch_loss_tensors
            torch.cuda.empty_cache()

            return DistillResponse.model_validate(
                {
                    "lora_id": new_lora_id,
                    "metadata": {
                        "total_loss": total_loss,
                        "distill_loss": distill_loss,
                        "kl_reg": kl_reg,
                        "mean_is_ratio": mean_is_ratio,
                        "clip_fraction": clip_fraction,
                        "grad_norm": grad_norm_value,
                        "tokens_processed": tokens_processed,
                        "batch_size": len(payload.samples),
                        "teacher_scored_texts": batch_teacher_scored_texts,
                    },
                }
            )
        finally:
            cleanup_local_lora(lora_local_path)
