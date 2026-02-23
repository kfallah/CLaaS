from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from claas.training.distillation import DistillationTrainer


class _SimpleLoraModel(torch.nn.Module):
    """Minimal model exposing trainable LoRA-like parameters."""

    def __init__(self, include_second: bool = True) -> None:
        """Initialize synthetic trainable parameters.

        Args:
            include_second: Whether to include the second trainable parameter.
        """
        super().__init__()
        self.first = torch.nn.Parameter(torch.zeros(2, 2))
        if include_second:
            self.second = torch.nn.Parameter(torch.zeros(2, 2))


@pytest.fixture
def trainer() -> DistillationTrainer:
    """Return a distillation trainer bound to CPU for state tests."""
    inst = DistillationTrainer(base_model_id="test", attn_implementation="sdpa")
    inst.device = torch.device("cpu")
    return inst


def _seed_optimizer_state(model: _SimpleLoraModel) -> torch.optim.Optimizer:
    """Create AdamW optimizer with initialized momentum buffers.

    Args:
        model: Model that provides trainable parameters.

    Returns:
        Optimizer with non-empty state.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model.first.sum()
    if hasattr(model, "second"):
        loss = loss + model.second.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return optimizer


def test_optimizer_state_roundtrip(trainer: DistillationTrainer, tmp_path: Path) -> None:
    """Saves then loads optimizer state through torch state dictionaries."""
    model_a = _SimpleLoraModel()
    optimizer_a = _seed_optimizer_state(model_a)

    trainer._save_optimizer_state(optimizer_a, str(tmp_path))

    model_b = _SimpleLoraModel()
    optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)
    trainer._load_optimizer_state(str(tmp_path), optimizer_b)

    state_a = optimizer_a.state[model_a.first]["exp_avg"]
    state_b = optimizer_b.state[model_b.first]["exp_avg"]
    assert torch.equal(state_a, state_b)


def test_optimizer_state_invalid_schema_fails(trainer: DistillationTrainer, tmp_path: Path) -> None:
    """Raises when optimizer artifact does not deserialize to a valid state dict."""
    artifact_path = tmp_path / "optimizer_state.pt"
    torch.save(["not", "a", "dict"], artifact_path)

    model = _SimpleLoraModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="dictionary"):
        trainer._load_optimizer_state(str(tmp_path), optimizer)
