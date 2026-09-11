from __future__ import annotations

from types import SimpleNamespace
import sys

import pytest

from personagraph.retrieval.compute import devices
from personagraph.retrieval.compute.resources import (
    compute_resource_lock,
    normalize_device,
)


def _torch(monkeypatch, *, cuda=False, mps=False, probe_failure=None):
    calls = []

    class Tensor:
        def sum(self):
            return self

        def item(self):
            return 1.0

    def ones(shape, *, device):
        calls.append(device)
        if probe_failure is not None:
            raise probe_failure
        return Tensor()

    module = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            device_count=lambda: 2 if cuda else 0,
            synchronize=lambda device: None,
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        mps=SimpleNamespace(synchronize=lambda: None),
        ones=ones,
    )
    monkeypatch.setitem(sys.modules, "torch", module)
    return calls


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [("cuda", "cuda:0"), (" CUDA:1 ", "cuda:1"), ("mps:0", "mps"), ("cpu", "cpu")],
)
def test_concrete_device_names_have_one_lock_identity(raw, canonical):
    assert normalize_device(raw) == canonical


@pytest.mark.parametrize("raw", ["auto", "gpu", "cuda:-1", "cuda:x", "mps:1", ""])
def test_unresolved_or_invalid_names_cannot_enter_model_backend(raw):
    with pytest.raises(ValueError):
        normalize_device(raw)


def test_explicit_cpu_never_imports_or_probes_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    selection = devices.select_device("cpu")
    assert selection.ready
    assert selection.device == "cpu"
    assert selection.fallback_reason is None


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda:0"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_selects_one_device_in_priority_order(monkeypatch, cuda, mps, expected):
    calls = _torch(monkeypatch, cuda=cuda, mps=mps)
    selection = devices.select_device("auto")
    assert selection.requested_device == "auto"
    assert selection.device == expected
    assert selection.ready
    assert calls == ([] if expected == "cpu" else [expected])
    assert selection.diagnostic_snapshot()["model_verified"] is False


def test_auto_probe_device_failure_falls_back_with_safe_reason(monkeypatch):
    _torch(
        monkeypatch, mps=True, probe_failure=RuntimeError("MPS backend out of memory")
    )
    selection = devices.select_device("auto")
    assert selection.device == "cpu"
    assert selection.fallback_reason == "device_out_of_memory"
    assert "device_out_of_memory" in repr(selection.probe_reasons)


def test_explicit_unavailable_device_is_not_silently_cpu(monkeypatch):
    _torch(monkeypatch)
    selection = devices.select_device("mps")
    assert not selection.ready
    assert selection.device is None


def test_auto_does_not_mask_unknown_probe_bug(monkeypatch):
    _torch(monkeypatch, mps=True, probe_failure=ValueError("bad tensor arguments"))
    with pytest.raises(ValueError, match="bad tensor"):
        devices.select_device("auto")


def test_missing_torch_allows_auto_cpu_but_not_explicit_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert devices.select_device("auto").device == "cpu"
    assert not devices.select_device("cuda:0").ready


def test_compute_locks_share_canonical_device_identity():
    assert compute_resource_lock("mps") is compute_resource_lock("mps:0")
    assert compute_resource_lock("cuda") is compute_resource_lock("cuda:0")
    assert compute_resource_lock("cuda:0") is not compute_resource_lock("cpu")


def test_probe_holds_inference_device_lock_for_tensor_work_and_synchronization(
    monkeypatch,
):
    _torch(monkeypatch, mps=True)
    module = sys.modules["torch"]
    lock = compute_resource_lock("mps")
    stages = []

    def check(stage):
        assert lock.locked(), f"device lock missing at {stage}"
        stages.append(stage)

    class Tensor:
        def sum(self):
            check("sum")
            return self

        def item(self):
            check("item")
            return 1.0

    def ones(shape, *, device):
        check("ones")
        return Tensor()

    module.ones = ones
    module.mps.synchronize = lambda: check("synchronize")
    assert devices.select_device("mps").device == "mps"
    assert stages == ["ones", "sum", "item", "synchronize"]
    assert not lock.locked()


@pytest.mark.parametrize("stage", ["ones", "sum", "item", "synchronize"])
def test_failed_device_probe_releases_shared_lock(monkeypatch, stage):
    _torch(monkeypatch, mps=True)
    module = sys.modules["torch"]

    def fail():
        raise RuntimeError("MPS backend out of memory")

    class Tensor:
        def sum(self):
            if stage == "sum":
                fail()
            return self

        def item(self):
            if stage == "item":
                fail()
            return 1.0

    def ones(shape, *, device):
        if stage == "ones":
            fail()
        return Tensor()

    module.ones = ones
    if stage == "synchronize":
        module.mps.synchronize = fail
    assert devices.select_device("auto").device == "cpu"
    assert not compute_resource_lock("mps").locked()
