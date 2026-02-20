"""Regression tests for optional environment variables in Modal/training modules."""

from __future__ import annotations

import importlib
import sys
import types


def _install_torch_stub(monkeypatch):
    torch_stub = types.ModuleType("torch")
    torch_stub.bfloat16 = object()
    torch_stub.device = lambda *_args, **_kwargs: "cuda"
    torch_stub.optim = types.SimpleNamespace(AdamW=object)

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *_exc):
            return False

    torch_stub.no_grad = lambda: _NoGrad()
    torch_stub.cuda = types.SimpleNamespace(empty_cache=lambda: None, synchronize=lambda: None)

    nn_module = types.ModuleType("torch.nn")
    nn_functional_module = types.ModuleType("torch.nn.functional")
    nn_module.functional = nn_functional_module
    nn_module.Module = types.SimpleNamespace(to=lambda *_args: None)
    torch_stub.nn = nn_module

    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torch.nn", nn_module)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", nn_functional_module)


def _reload(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_teacher_service_import_allows_missing_secret(monkeypatch):
    """Teacher module should tolerate missing CLAAS_HF_SECRET_NAME."""
    monkeypatch.delenv("CLAAS_HF_SECRET_NAME", raising=False)

    teacher_service = _reload("claas.modal.teacher_service")

    assert teacher_service.teacher_secrets == []


def test_worker_import_allows_missing_tokens_and_model_env(monkeypatch):
    """Worker module should use defaults when optional env vars are absent."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("CLAAS_BASE_MODEL_ID", raising=False)
    monkeypatch.delenv("CLAAS_ATTN_IMPLEMENTATION", raising=False)

    _install_torch_stub(monkeypatch)
    worker = _reload("claas.modal.worker")

    assert worker.DistillWorker.base_model_id == "Qwen/Qwen3-8B"
    assert worker.DistillWorker.attn_implementation == "sdpa"


class _DummyTensor:
    def to(self, _device):
        return self


class _DummyModel:
    def parameters(self):
        return [types.SimpleNamespace(requires_grad=True)]

    def __call__(self, **_kwargs):
        return object()


class _DummyTokenizer:
    pad_token = None
    eos_token = "<eos>"

    def encode(self, *_args, **_kwargs):
        return _DummyTensor()


def test_distillation_uses_transformers_cache_fallback(monkeypatch):
    """DistillationTrainer should fall back to TRANSFORMERS_CACHE."""
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("TRANSFORMERS_CACHE", "/tmp/transformers-cache")

    recorded: dict[str, str | None] = {}

    transformers_stub = types.ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **kwargs):
            recorded["tokenizer_cache_dir"] = kwargs.get("cache_dir")
            return _DummyTokenizer()

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*_args, **kwargs):
            recorded["model_cache_dir"] = kwargs.get("cache_dir")
            return _DummyModel()

    transformers_stub.AutoTokenizer = _AutoTokenizer
    transformers_stub.AutoModelForCausalLM = _AutoModelForCausalLM

    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)
    _install_torch_stub(monkeypatch)

    from claas.training.distillation import DistillationTrainer

    trainer = DistillationTrainer(base_model_id="dummy/model", attn_implementation="sdpa")
    trainer.load_base_model()

    assert recorded["tokenizer_cache_dir"] == "/tmp/transformers-cache"
    assert recorded["model_cache_dir"] == "/tmp/transformers-cache"
