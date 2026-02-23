from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from claas.training.cache import LoraAdapterConfig, LoraCacheEntry  # noqa: E402
from claas.training.distillation import (  # noqa: E402
    DistillationTrainer,
    _cpu_optimizer_state,
    _gpu_optimizer_state,
)


class _SimpleLoraModel(torch.nn.Module):
    """Minimal model exposing trainable LoRA-like parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Parameter(torch.zeros(2, 2))
        self.second = torch.nn.Parameter(torch.zeros(2, 2))


@pytest.fixture
def trainer() -> DistillationTrainer:
    """Return a distillation trainer bound to CPU for state tests."""
    inst = DistillationTrainer(base_model_id="test", attn_implementation="sdpa")
    inst.device = torch.device("cpu")
    return inst


def test_optimizer_state_loaded_between_steps(trainer: DistillationTrainer, tmp_path: Path) -> None:
    """Optimizer momentum carries over when state is saved and loaded between steps."""
    # Step 1: create model, run optimizer step, save state
    model_1 = _SimpleLoraModel()
    optimizer_1 = torch.optim.AdamW(model_1.parameters(), lr=1e-3)
    loss_1 = model_1.first.sum() + model_1.second.sum()
    loss_1.backward()
    optimizer_1.step()
    optimizer_1.zero_grad()

    step1_dir = str(tmp_path / "step1")
    Path(step1_dir).mkdir()
    trainer._save_optimizer_state(optimizer_1, step1_dir)

    step1_exp_avg = optimizer_1.state[model_1.first]["exp_avg"].clone()
    step1_step_count = optimizer_1.state[model_1.first]["step"]

    # Step 2: create fresh model + optimizer, load state from step 1, run another step
    model_2 = _SimpleLoraModel()
    optimizer_2 = torch.optim.AdamW(model_2.parameters(), lr=1e-3)
    trainer._load_optimizer_state(step1_dir, optimizer_2)

    # Verify state was restored before step 2
    assert optimizer_2.state[model_2.first]["step"] == step1_step_count
    assert torch.equal(optimizer_2.state[model_2.first]["exp_avg"], step1_exp_avg)

    # Run step 2
    loss_2 = model_2.first.sum() + model_2.second.sum()
    loss_2.backward()
    optimizer_2.step()
    optimizer_2.zero_grad()

    # Verify step count advanced and momentum updated
    assert optimizer_2.state[model_2.first]["step"] == step1_step_count + 1
    assert not torch.equal(optimizer_2.state[model_2.first]["exp_avg"], step1_exp_avg)

    # Save step 2, load into step 3, verify continuity
    step2_dir = str(tmp_path / "step2")
    Path(step2_dir).mkdir()
    trainer._save_optimizer_state(optimizer_2, step2_dir)

    model_3 = _SimpleLoraModel()
    optimizer_3 = torch.optim.AdamW(model_3.parameters(), lr=1e-3)
    trainer._load_optimizer_state(step2_dir, optimizer_3)

    assert optimizer_3.state[model_3.first]["step"] == step1_step_count + 1
    assert torch.equal(
        optimizer_3.state[model_3.first]["exp_avg"],
        optimizer_2.state[model_2.first]["exp_avg"],
    )
    assert torch.equal(
        optimizer_3.state[model_3.first]["exp_avg_sq"],
        optimizer_2.state[model_2.first]["exp_avg_sq"],
    )


def test_optimizer_state_missing_gracefully_skips(trainer: DistillationTrainer, tmp_path: Path) -> None:
    """When no optimizer state file exists, loading is a no-op (first step)."""
    model = _SimpleLoraModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    trainer._load_optimizer_state(str(tmp_path), optimizer)

    assert len(optimizer.state) == 0


def test_cpu_optimizer_state_moves_tensors_to_cpu() -> None:
    """_cpu_optimizer_state produces a state dict with all tensors on CPU."""
    model = _SimpleLoraModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model.first.sum()
    loss.backward()
    optimizer.step()

    original = optimizer.state_dict()
    cpu_state = _cpu_optimizer_state(original)

    for param_state in cpu_state["state"].values():
        for v in param_state.values():
            if isinstance(v, torch.Tensor):
                assert v.device == torch.device("cpu")


def test_cpu_gpu_optimizer_state_roundtrip() -> None:
    """_cpu_optimizer_state / _gpu_optimizer_state round-trip preserves values."""
    model = _SimpleLoraModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model.first.sum()
    loss.backward()
    optimizer.step()

    original = optimizer.state_dict()
    cpu_state = _cpu_optimizer_state(original)
    roundtripped = _gpu_optimizer_state(cpu_state, torch.device("cpu"))

    # Step counts match
    for param_id in original["state"]:
        assert roundtripped["state"][param_id]["step"] == original["state"][param_id]["step"]

    # Tensor values match
    for param_id in original["state"]:
        for key in ("exp_avg", "exp_avg_sq"):
            orig_tensor = original["state"][param_id][key]
            rt_tensor = roundtripped["state"][param_id][key]
            assert torch.equal(orig_tensor, rt_tensor)


def test_cpu_optimizer_state_does_not_mutate_original() -> None:
    """_cpu_optimizer_state deep-copies — mutating the copy leaves the original intact."""
    model = _SimpleLoraModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model.first.sum()
    loss.backward()
    optimizer.step()

    original = optimizer.state_dict()
    original_exp_avg = original["state"][0]["exp_avg"].clone()

    cpu_state = _cpu_optimizer_state(original)
    # Mutate the copy
    cpu_state["state"][0]["exp_avg"].zero_()

    # Original is unchanged
    assert torch.equal(original["state"][0]["exp_avg"], original_exp_avg)


def test_lora_cache_entry_is_frozen() -> None:
    """LoraCacheEntry is immutable — attribute assignment raises."""
    entry = LoraCacheEntry(
        lora_state_dict={"w": torch.zeros(2)},
        optimizer_state_dict={"state": {}, "param_groups": []},
        adapter_config=LoraAdapterConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    with pytest.raises(AttributeError):
        entry.lora_state_dict = {}  # type: ignore[misc]
