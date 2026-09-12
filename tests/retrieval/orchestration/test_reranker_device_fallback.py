"""重排器设备回退不改变输入、取消语义或模型精度。"""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from personagraph.retrieval.execution import (
    RetrievalExecution,
    RerankingDeadlineReached,
    execution_scope,
)
from personagraph.retrieval.orchestration import reranking
from personagraph.retrieval.orchestration import rerank_execution
from personagraph.retrieval.ports import RetrievalCancelled
from tests.retrieval.orchestration.test_execution_cancellation import (
    Cancellation,
    HookedForward,
)


PAIRS = (("question b", "passage b"), ("question a", "passage a"))


def _fake_backend(monkeypatch, tmp_path, *, gpu_error=None, cpu_error=None,
                  gpu_load_error=None, invalid_scores=False):
    """只模拟第三方调用；所有 GPU 算子和权重加载均为本地替身。"""

    loads = []
    calls = []
    released = []
    locks = {"mps": threading.Lock(), "cpu": threading.Lock()}
    monkeypatch.setattr(reranking, "compute_resource_lock", locks.__getitem__)
    monkeypatch.setattr(
        reranking.BgeM3Reranker, "_local_model_path", lambda _self: tmp_path,
    )
    monkeypatch.setattr(
        reranking, "synchronize_device", lambda _device: None, raising=False,
    )
    monkeypatch.setattr(
        rerank_execution, "synchronize_device", lambda _device: None, raising=False,
    )

    def release(device):
        assert not locks[device].locked(), "GPU resource must be released before cleanup"
        released.append(device)

    monkeypatch.setattr(reranking, "release_device_cache", release, raising=False)

    class Native(HookedForward):
        def forward(self, pairs):
            self.run(len(pairs))
            return pairs

        def to(self, *_args, **_kwargs):
            return self

        def half(self):
            return self

    class Model:
        def __init__(self, _path, **kwargs):
            device = kwargs["devices"]
            assert locks[device].locked(), "model loading must hold its actual resource"
            if device == "cpu":
                assert not locks["mps"].locked()
                assert kwargs["use_fp16"] is False
            loads.append(dict(kwargs))
            if device == "mps" and gpu_load_error is not None:
                raise gpu_load_error
            self.device = device
            self.instance_ordinal = len(loads)
            self.model = Native()

        def compute_score(self, pairs, **_kwargs):
            assert locks[self.device].locked()
            calls.append((self.device, tuple(pairs), self.instance_ordinal))
            self.model.forward(pairs)
            failure = gpu_error if self.device == "mps" else cpu_error
            if failure is not None:
                raise failure
            return [0.75] if invalid_scores else [0.75, -0.25]

    monkeypatch.setitem(sys.modules, "FlagEmbedding", SimpleNamespace(FlagReranker=Model))
    return SimpleNamespace(loads=loads, calls=calls, released=released, locks=locks)


def test_known_gpu_failure_retries_complete_pairs_once_with_fresh_cpu_model(
    monkeypatch, tmp_path,
):
    backend = _fake_backend(
        monkeypatch, tmp_path, gpu_error=RuntimeError("MPS backend out of memory"),
    )
    scorer = reranking.BgeM3Reranker(device="mps", allow_cpu_fallback=True)
    configured_fingerprint = scorer.fingerprint()
    assert ";cpu_fallback=true" in configured_fingerprint

    assert scorer.score(PAIRS) == (0.75, -0.25)
    assert [(device, pairs) for device, pairs, _ in backend.calls] == [
        ("mps", PAIRS), ("cpu", PAIRS),
    ]
    assert backend.calls[0][2] != backend.calls[1][2]
    assert [item["devices"] for item in backend.loads] == ["mps", "cpu"]
    assert backend.released == ["mps"]
    assert not any(lock.locked() for lock in backend.locks.values())
    snapshot = scorer.diagnostic_snapshot()
    assert snapshot["device"] == "cpu"
    assert snapshot["requested_device"] == "mps"
    assert snapshot["cpu_fallback_reason"]
    assert scorer.fingerprint() == configured_fingerprint

    assert scorer.score(PAIRS) == (0.75, -0.25)
    assert [item["devices"] for item in backend.loads] == ["mps", "cpu"]
    assert [device for device, _, _ in backend.calls] == ["mps", "cpu", "cpu"]


def test_known_gpu_loading_failure_preserves_cause_for_cpu_fallback(monkeypatch, tmp_path):
    backend = _fake_backend(
        monkeypatch, tmp_path, gpu_load_error=RuntimeError("MPS backend out of memory"),
    )
    scorer = reranking.BgeM3Reranker(device="mps", allow_cpu_fallback=True)

    assert scorer.score(PAIRS) == (0.75, -0.25)
    assert [item["devices"] for item in backend.loads] == ["mps", "cpu"]
    assert [device for device, _, _ in backend.calls] == ["cpu"]
    assert scorer.diagnostic_snapshot()["load_failure_reason"] is None


@pytest.mark.parametrize("error", [
    RuntimeError("shape mismatch in model weights"),
    ValueError("invalid checkpoint contents"),
    OSError("missing local model file"),
])
def test_unknown_or_asset_failure_never_falls_back(monkeypatch, tmp_path, error):
    backend = _fake_backend(monkeypatch, tmp_path, gpu_load_error=error)
    scorer = reranking.BgeM3Reranker(device="mps", allow_cpu_fallback=True)

    with pytest.raises(reranking.RerankerUnavailable):
        scorer.score(PAIRS)
    assert [item["devices"] for item in backend.loads] == ["mps"]
    assert backend.released == []


@pytest.mark.parametrize("error", [
    RetrievalCancelled("execution_cancelled"),
    RerankingDeadlineReached("reranker_projection_reserve_reached"),
])
def test_cancellation_and_deadline_never_trigger_cpu_retry(monkeypatch, tmp_path, error):
    backend = _fake_backend(monkeypatch, tmp_path, gpu_error=error)
    scorer = reranking.BgeM3Reranker(device="mps", allow_cpu_fallback=True)

    with pytest.raises(type(error)):
        scorer.score(PAIRS)
    assert [item["devices"] for item in backend.loads] == ["mps"]
    assert len(backend.calls) == 1
    assert backend.released == []


def test_cpu_failure_does_not_reenter_gpu_or_retry_again(monkeypatch, tmp_path):
    backend = _fake_backend(
        monkeypatch, tmp_path,
        gpu_error=RuntimeError("MPS backend out of memory"),
        cpu_error=RuntimeError("CPU allocation failed"),
    )
    scorer = reranking.BgeM3Reranker(device="mps", allow_cpu_fallback=True)

    with pytest.raises(reranking.RerankerUnavailable):
        scorer.score(PAIRS)
    assert [device for device, _, _ in backend.calls] == ["mps", "cpu"]
    assert [item["devices"] for item in backend.loads] == ["mps", "cpu"]


def test_cancellation_between_device_attempts_prevents_cpu_load(monkeypatch, tmp_path):
    backend = _fake_backend(
        monkeypatch, tmp_path, gpu_error=RuntimeError("MPS backend out of memory"),
    )

    def check():
        if backend.released:
            raise RetrievalCancelled("execution_cancelled")

    monkeypatch.setattr(reranking, "checkpoint", check)
    scorer = reranking.BgeM3Reranker(device="mps", allow_cpu_fallback=True)

    with pytest.raises(RetrievalCancelled, match="execution_cancelled"):
        scorer.score(PAIRS)
    assert [item["devices"] for item in backend.loads] == ["mps"]
    assert [device for device, _, _ in backend.calls] == ["mps"]


def test_known_gpu_cleanup_failure_is_recorded_and_cpu_retries_same_call(monkeypatch, tmp_path):
    backend = _fake_backend(
        monkeypatch, tmp_path, gpu_error=RuntimeError("MPS backend out of memory"),
    )
    release = reranking.release_device_cache

    def known_cleanup_failure(device):
        release(device)
        return "device_unavailable"

    monkeypatch.setattr(reranking, "release_device_cache", known_cleanup_failure)
    scorer = reranking.BgeM3Reranker(device="mps", allow_cpu_fallback=True)
    execution = RetrievalExecution(Cancellation())

    with execution_scope(execution):
        assert scorer.score(PAIRS) == (0.75, -0.25)

    assert [device for device, _, _ in backend.calls] == ["mps", "cpu"]
    assert scorer.diagnostic_snapshot()["gpu_cleanup_reason"] == "device_unavailable"
    assert execution.snapshot()["metrics"]["reranker_gpu_cleanup_reason"] == "device_unavailable"
    assert execution.snapshot()["metrics"]["reranker_cpu_fallback_count"] == 1


def test_unknown_cleanup_error_does_not_relabel_or_run_cpu(monkeypatch, tmp_path):
    backend = _fake_backend(
        monkeypatch, tmp_path, gpu_error=RuntimeError("MPS backend out of memory"),
    )

    def unknown_cleanup_failure(_device):
        raise RuntimeError("unrecognized cleanup failure")

    monkeypatch.setattr(reranking, "release_device_cache", unknown_cleanup_failure)
    scorer = reranking.BgeM3Reranker(device="mps", allow_cpu_fallback=True)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="unrecognized cleanup failure"):
            scorer.score(PAIRS)
        assert scorer.diagnostic_snapshot()["device"] == "mps"
        assert scorer.diagnostic_snapshot()["loaded"] is False
        assert scorer.diagnostic_snapshot()["cpu_fallback_reason"] is None

    assert [item["devices"] for item in backend.loads] == ["mps", "mps"]


def test_cancellation_during_cleanup_does_not_relabel_or_run_cpu(monkeypatch, tmp_path):
    backend = _fake_backend(
        monkeypatch, tmp_path, gpu_error=RuntimeError("MPS backend out of memory"),
    )

    def cancelled_cleanup(_device):
        raise RetrievalCancelled("execution_cancelled")

    monkeypatch.setattr(reranking, "release_device_cache", cancelled_cleanup)
    scorer = reranking.BgeM3Reranker(device="mps", allow_cpu_fallback=True)

    with pytest.raises(RetrievalCancelled, match="execution_cancelled"):
        scorer.score(PAIRS)
    assert [item["devices"] for item in backend.loads] == ["mps"]
    assert scorer.diagnostic_snapshot()["device"] == "mps"
    assert scorer.diagnostic_snapshot()["loaded"] is False


def test_one_instance_serializes_device_switch_with_concurrent_call(monkeypatch, tmp_path):
    backend = _fake_backend(
        monkeypatch, tmp_path, gpu_error=RuntimeError("MPS backend out of memory"),
    )
    cleanup_entered = threading.Event()
    allow_cleanup = threading.Event()
    second_waiting = threading.Event()
    second_finished = threading.Event()
    results = []
    errors = []
    original_release = reranking.release_device_cache
    original_wait = reranking.wait_for_lock

    def release(device):
        original_release(device)
        cleanup_entered.set()
        assert allow_cleanup.wait(timeout=2.0)

    @contextmanager
    def traced_wait(lock, stage, **kwargs):
        if threading.current_thread().name == "second-scorer" and stage == "reranker_lifecycle_queue":
            second_waiting.set()
        with original_wait(lock, stage, **kwargs):
            yield

    monkeypatch.setattr(reranking, "release_device_cache", release)
    monkeypatch.setattr(reranking, "wait_for_lock", traced_wait)
    scorer = reranking.BgeM3Reranker(device="mps", allow_cpu_fallback=True)

    def score():
        try:
            results.append(scorer.score(PAIRS))
        except BaseException as exc:
            errors.append(exc)
        finally:
            if threading.current_thread().name == "second-scorer":
                second_finished.set()

    first = threading.Thread(target=score, name="first-scorer")
    second = threading.Thread(target=score, name="second-scorer")
    first.start()
    try:
        assert cleanup_entered.wait(timeout=2.0)
        second.start()
        assert second_waiting.wait(timeout=2.0)
        assert not second_finished.is_set()
        assert [item["devices"] for item in backend.loads] == ["mps"]
    finally:
        allow_cleanup.set()
        first.join(timeout=2.0)
        if second.ident is not None:
            second.join(timeout=2.0)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert results == [(0.75, -0.25), (0.75, -0.25)]
    assert [item["devices"] for item in backend.loads] == ["mps", "cpu"]
    assert [device for device, _, _ in backend.calls] == ["mps", "cpu", "cpu"]


def test_explicit_device_stays_fail_closed_without_fallback_opt_in(monkeypatch, tmp_path):
    backend = _fake_backend(
        monkeypatch, tmp_path, gpu_error=RuntimeError("MPS backend out of memory"),
    )
    scorer = reranking.BgeM3Reranker(device="mps")

    with pytest.raises(reranking.RerankerUnavailable):
        scorer.score(PAIRS)
    assert [item["devices"] for item in backend.loads] == ["mps"]
    assert len(backend.calls) == 1


def test_invalid_score_alignment_is_not_a_gpu_failure(monkeypatch, tmp_path):
    backend = _fake_backend(monkeypatch, tmp_path, invalid_scores=True)
    scorer = reranking.BgeM3Reranker(device="mps", allow_cpu_fallback=True)

    with pytest.raises(reranking.RerankerUnavailable, match="unaligned"):
        scorer.score(PAIRS)
    assert [item["devices"] for item in backend.loads] == ["mps"]
    assert len(backend.calls) == 1


def test_half_precision_profile_cannot_opt_into_cpu_fallback():
    with pytest.raises(ValueError, match="fp16|FP32|precision"):
        reranking.BgeM3Reranker(
            device="mps", use_fp16=True, allow_cpu_fallback=True,
        )


def test_gpu_batch_timing_includes_device_synchronization(monkeypatch):
    now = [0.0]
    events = []
    native = HookedForward(lambda: events.append("forward_dispatched"))

    def synchronize(device):
        assert device == "mps"
        events.append("gpu_completed")
        now[0] += 0.4

    monkeypatch.setattr(
        rerank_execution, "synchronize_device", synchronize, raising=False,
    )
    execution = RetrievalExecution(Cancellation(), clock=lambda: now[0])

    with execution_scope(execution):
        with rerank_execution.observe_reranker_batches(
            SimpleNamespace(model=native), device="mps",
        ):
            native.run(2)

    assert events == ["forward_dispatched", "gpu_completed"]
    assert execution.snapshot()["stages"]["reranker_score"] == {
        "calls": 1, "duration_ms": 400,
    }
    assert execution.snapshot()["metrics"]["reranker_max_observed_batch_ms"] == 400
    assert native.pre_hooks == native.post_hooks == []


def test_async_gpu_error_in_post_hook_escapes_vendor_runtime_retry(monkeypatch, tmp_path):
    instances = []
    swallowed = []

    class Native(HookedForward):
        # Torch dispatches pre/forward/post hooks inside _call_impl. Keep that
        # boundary so a post-hook failure tests the guard without importing Torch.
        def __call__(self, *, input_ids):
            return self._call_impl(input_ids=input_ids)

        def _call_impl(self, *, input_ids):
            self.run(input_ids.shape[0])
            return [1.0] * input_ids.shape[0]

    class Scorer:
        def __init__(self, _path, **kwargs):
            self.device = kwargs["devices"]
            self.model = Native()
            instances.append(self)

        def compute_score(self, pairs, **_kwargs):
            # 模拟供应商吞 RuntimeError 的首批探测，但测试本身不能无限循环。
            for _ in range(3):
                try:
                    output = self.model(input_ids=SimpleNamespace(shape=(len(pairs), 12)))
                except RuntimeError:
                    swallowed.append(self.device)
                else:
                    return output
            raise AssertionError("vendor swallowed the asynchronous device failure")

    def synchronize(device):
        if device == "mps":
            raise RuntimeError("MPS backend out of memory")

    monkeypatch.setitem(sys.modules, "FlagEmbedding", SimpleNamespace(FlagReranker=Scorer))
    monkeypatch.setattr(reranking.BgeM3Reranker, "_local_model_path", lambda _self: tmp_path)
    monkeypatch.setattr(reranking, "release_device_cache", lambda _device: None)
    monkeypatch.setattr(reranking, "synchronize_device", synchronize)
    monkeypatch.setattr(rerank_execution, "synchronize_device", synchronize)
    scorer = reranking.BgeM3Reranker(device="mps", allow_cpu_fallback=True)
    execution = RetrievalExecution(Cancellation())

    with execution_scope(execution):
        assert scorer.score(PAIRS) == (1.0, 1.0)

    assert swallowed == []
    assert [model.device for model in instances] == ["mps", "cpu"]
    assert all(not model.model.post_hooks for model in instances)
    assert all(not model.model.pre_hooks for model in instances)
    assert all("_call_impl" not in vars(model.model) for model in instances)
    assert execution.snapshot()["metrics"]["reranker_cpu_fallback_count"] == 1
