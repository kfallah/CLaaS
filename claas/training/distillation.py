"""Shared SDPO distillation trainer for local and Modal runtimes."""

from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
from typing import TYPE_CHECKING, cast

import torch

from claas.core.types import DistillBatchRequestPayload, DistillResponse, SDPOLossInput
from claas.training.cache import (
    DistillStepResult,
    LoraAdapterConfig,
    LoraCacheEntry,
)
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


def _cpu_optimizer_state(state_dict: dict[str, object]) -> dict[str, object]:
    """Deep-copy optimizer state with all tensors moved to CPU."""
    result: dict[str, object] = {}
    for key, value in state_dict.items():
        if key == "state":
            param_states = cast("dict[int, dict[str, object]]", value)
            cpu_states: dict[int, dict[str, object]] = {}
            for param_id, param_state in param_states.items():
                cpu_param: dict[str, object] = {}
                for k, v in param_state.items():
                    if isinstance(v, torch.Tensor):
                        cpu_param[k] = v.detach().cpu().clone()
                    else:
                        cpu_param[k] = copy.deepcopy(v)
                cpu_states[param_id] = cpu_param
            result[key] = cpu_states
        else:
            result[key] = copy.deepcopy(value)
    return result


def _gpu_optimizer_state(
    state_dict: dict[str, object],
    device: torch.device,
) -> dict[str, object]:
    """Deep-copy optimizer state with all tensors moved to a target device."""
    result: dict[str, object] = {}
    for key, value in state_dict.items():
        if key == "state":
            param_states = cast("dict[int, dict[str, object]]", value)
            gpu_states: dict[int, dict[str, object]] = {}
            for param_id, param_state in param_states.items():
                gpu_param: dict[str, object] = {}
                for k, v in param_state.items():
                    if isinstance(v, torch.Tensor):
                        gpu_param[k] = v.detach().to(device).clone()
                    else:
                        gpu_param[k] = copy.deepcopy(v)
                gpu_states[param_id] = gpu_param
            result[key] = gpu_states
        else:
            result[key] = copy.deepcopy(value)
    return result


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

    def reload_base_model(self) -> None:
        """Move base model from CPU back to CUDA."""
        self.base_model.to(self.device)  # type: ignore[arg-type]  # functools.wraps confuses ty

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

    def _load_lora_from_cache(
        self,
        cached: LoraCacheEntry,
    ) -> "PeftModel | PeftMixedModel":
        """Restore a LoRA adapter from a CPU cache entry.

        Args:
            cached: CPU-resident snapshot of adapter state.

        Returns:
            Trainable PEFT model with cached weights loaded.
        """
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

        cfg = cached.adapter_config
        lora_config = LoraConfig(
            r=cfg.r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.target_modules,
            lora_dropout=cfg.lora_dropout,
            bias=cfg.bias,
            task_type=cfg.task_type,
        )
        model = get_peft_model(self.base_model, lora_config)
        set_peft_model_state_dict(model, cached.lora_state_dict)
        return model

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
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Build top-k teacher logits from the frozen base model.

        Args:
            prompt: User prompt.
            feedback: Critique text.
            response_ids: Tokenized sampled response.
            top_k: Number of logits to retain per token.

        Returns:
            Pair of top-k logprobs and indices for each response token.

        Reference:
            Huebotter et al. (2026), "Reinforcement Learning via Self-Distillation"
            (arXiv:2601.20802), Section 3.
        """

        messages = build_teacher_messages(prompt, feedback)
        template_messages = teacher_messages_to_chat_template(messages)
        teacher_prompt_ids_raw = self.tokenizer.apply_chat_template(
            template_messages,
            add_generation_prompt=False,
            return_tensors="pt",
            tokenize=True,
        )
        # transformers >=5.x returns BatchEncoding; extract input_ids tensor
        if hasattr(teacher_prompt_ids_raw, "input_ids"):
            teacher_prompt_ids_raw = teacher_prompt_ids_raw.input_ids
        teacher_prompt_ids = cast("torch.Tensor", teacher_prompt_ids_raw).to(self.device)
        teacher_full_ids = torch.cat([teacher_prompt_ids, response_ids], dim=-1)
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
        return top_logprobs, top_indices

    def _build_cache_entry(
        self,
        model: "PeftModel | PeftMixedModel",
        optimizer: "torch.optim.Optimizer",
    ) -> LoraCacheEntry:
        """Snapshot current model + optimizer state into a CPU-resident cache entry."""
        from peft import PeftModel as PeftModelCls

        peft_config = model.peft_config["default"]
        adapter_config = LoraAdapterConfig(
            r=peft_config.r,
            lora_alpha=peft_config.lora_alpha,
            target_modules=list(peft_config.target_modules),
            lora_dropout=peft_config.lora_dropout,
            bias=peft_config.bias,
            task_type=peft_config.task_type,
        )

        # Determine state dict — use PEFT's adapter-only extraction if available
        if isinstance(model, PeftModelCls):
            from peft import get_peft_model_state_dict

            raw_state = get_peft_model_state_dict(model)
        else:
            raw_state = model.state_dict()

        lora_state = {k: v.detach().cpu().clone() for k, v in raw_state.items()}
        opt_state = _cpu_optimizer_state(optimizer.state_dict())

        return LoraCacheEntry(
            lora_state_dict=lora_state,
            optimizer_state_dict=opt_state,
            adapter_config=adapter_config,
        )

    def distill(
        self,
        payload: DistillBatchRequestPayload,
        *,
        cached: LoraCacheEntry | None = None,
    ) -> DistillStepResult:
        """Run one SDPO distillation step.

        Args:
            payload: Distillation request payload.
            cached: When provided, skip disk reads and load LoRA + optimizer
                state from this CPU-resident cache entry. When ``None``,
                load from disk (cold start).

        Returns:
            Result containing both the distillation response and a cache
            entry for the post-step state.
        """

        torch.cuda.empty_cache()
        if next(self.base_model.parameters()).device.type != "cuda":
            torch.nn.Module.to(self.base_model, self.device)

        config = payload.training
        if len(payload.samples) == 0:
            raise ValueError("samples must contain at least one item")

        # Disk path (cold start) or cache path
        lora_local_path: str | None = None
        if cached is None:
            lora_local_path = load_lora(payload.lora_id)

        try:
            try:
                if cached is not None:
                    model = self._load_lora_from_cache(cached)
                else:
                    assert lora_local_path is not None
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

            if cached is not None:
                optimizer.load_state_dict(
                    _gpu_optimizer_state(cached.optimizer_state_dict, self.device)
                )
            elif lora_local_path is not None:
                self._load_optimizer_state(lora_local_path, optimizer)

            batch_loss_tensors: list[torch.Tensor] = []
            batch_distill_loss: list[float] = []
            batch_kl_reg: list[float] = []
            batch_mean_is_ratio: list[float] = []
            batch_clip_fraction: list[float] = []
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

                teacher_logprobs, teacher_indices = self._build_self_teacher_topk(
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

            cache_entry = self._build_cache_entry(model, optimizer)

            del model, optimizer, batch_loss_tensors
            torch.cuda.empty_cache()

            response = DistillResponse.model_validate(
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
                    },
                }
            )
            return DistillStepResult(response=response, cache_entry=cache_entry)
        finally:
            if lora_local_path is not None:
                cleanup_local_lora(lora_local_path)
