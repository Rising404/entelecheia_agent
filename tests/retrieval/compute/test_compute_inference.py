from __future__ import annotations

from types import SimpleNamespace
import sys

import pytest

from personagraph.retrieval.compute.inference import (
    classify_device_error,
    guard_device_errors,
    release_device_cache,
)
from personagraph.retrieval.ports import RetrievalCancelled


@pytest.mark.parametrize(
    ("device", "message", "reason"),
    [
        ("mps", "MPS backend out of memory", "device_out_of_memory"),
        ("cuda:0", "CUDA out of memory", "device_out_of_memory"),
        ("mps", "operation not implemented for the MPS device", "device_unsupported"),
        ("cuda:0", "CUDA error: no kernel image is available for execution on the device", "device_unsupported"),
        ("mps", "MPS is not available", "device_unavailable"),
        ("cpu", "out of memory", None),
        ("cuda:0", "CUDA error: device-side assert triggered", None),
        ("mps", "shape mismatch", None),
    ],
)
def test_device_failures_use_a_narrow_whitelist(device, message, reason):
    assert classify_device_error(RuntimeError(message), device) == reason


def test_device_classifier_follows_wrapped_causes_not_arbitrary_text():
    cause = RuntimeError("MPS backend out of memory")
    wrapper = RuntimeError("model_unavailable")
    wrapper.__cause__ = cause
    assert classify_device_error(wrapper, "mps") == "device_out_of_memory"
    assert classify_device_error(ValueError("MPS backend out of memory"), "mps") is None


@pytest.mark.parametrize("method_name", ["forward", "to", "half"])
@pytest.mark.parametrize("message", ["MPS backend out of memory", "shape mismatch"])
def test_vendor_runtime_retry_cannot_swallow_device_or_unknown_errors(method_name, message):
    calls = []
    failure = RuntimeError(message)

    def fail(*args, **kwargs):
        calls.append(1)
        raise failure

    module = SimpleNamespace(**{method_name: fail})
    model = SimpleNamespace(model=module)
    with pytest.raises(RuntimeError) as observed:
        with guard_device_errors(model, "mps"):
            for _ in range(5):
                try:
                    getattr(module, method_name)()
                except RuntimeError:
                    continue
    assert observed.value is failure
    assert calls == [1]
    assert getattr(module, method_name) is fail


def test_guard_restores_class_methods_without_leaving_instance_overrides():
    class Module:
        def forward(self):
            return 7

    module = Module()
    with guard_device_errors(SimpleNamespace(model=module), "mps"):
        assert module.forward() == 7
    assert "forward" not in vars(module)


def test_cancel_is_not_classified_or_replaced():
    failure = RetrievalCancelled("cancelled")

    def cancel():
        raise failure

    model = SimpleNamespace(model=SimpleNamespace(forward=cancel))
    with pytest.raises(RetrievalCancelled) as observed:
        with guard_device_errors(model, "mps"):
            model.model.forward()
    assert observed.value is failure
    assert classify_device_error(failure, "mps") is None


def test_input_transfer_error_also_escapes_vendor_batch_reduction():
    failure = RuntimeError("MPS backend out of memory")
    calls = []

    class Batch(dict):
        def to(self, device):
            calls.append(device)
            raise failure

    tokenizer = SimpleNamespace(pad=lambda: Batch(input_ids=[1]))
    model = SimpleNamespace(model=SimpleNamespace(), tokenizer=tokenizer)
    with pytest.raises(RuntimeError) as observed:
        with guard_device_errors(model, "mps"):
            for _ in range(3):
                try:
                    model.tokenizer.pad().to("mps")
                except RuntimeError:
                    continue
    assert observed.value is failure
    assert calls == ["mps"]
    assert model.tokenizer is tokenizer


def test_successful_transfer_returns_original_batch_not_a_new_tensor_format():
    class Batch(dict):
        def to(self, device):
            return self

    batch = Batch(input_ids=[1])
    tokenizer = SimpleNamespace(pad=lambda: batch)
    model = SimpleNamespace(model=SimpleNamespace(), tokenizer=tokenizer)
    with guard_device_errors(model, "mps"):
        assert model.tokenizer.pad().to("mps") is batch
    assert model.tokenizer is tokenizer


@pytest.mark.parametrize("device", ["mps", "cuda:0"])
def test_cleanup_never_calls_unavailable_native_backend(monkeypatch, device):
    def unsafe():
        raise AssertionError("unavailable native backend must never be called")

    backend = SimpleNamespace(is_available=lambda: False, empty_cache=unsafe)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=backend, mps=backend, backends=SimpleNamespace(mps=backend),
    ))
    assert release_device_cache(device) == "device_unavailable"


@pytest.mark.parametrize(
    ("failure", "reason"),
    [(RuntimeError("MPS is not available"), "device_unavailable"),
     (RuntimeError("MPS backend out of memory"), "device_out_of_memory")],
)
def test_known_cleanup_failure_is_reported_without_blocking_cpu(monkeypatch, failure, reason):
    def fail():
        raise failure

    backend = SimpleNamespace(is_available=lambda: True, empty_cache=fail)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        mps=backend, backends=SimpleNamespace(mps=backend),
    ))
    assert release_device_cache("mps") == reason


def test_unknown_cleanup_bug_is_not_silently_ignored(monkeypatch):
    def fail():
        raise ValueError("cleanup bug")

    backend = SimpleNamespace(is_available=lambda: True, empty_cache=fail)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        mps=backend, backends=SimpleNamespace(mps=backend),
    ))
    with pytest.raises(ValueError, match="cleanup bug"):
        release_device_cache("mps")
