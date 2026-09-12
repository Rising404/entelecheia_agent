"""编码器的设备身份不可在同一索引代内因计算失败而切换。"""

from __future__ import annotations

from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

from personagraph.retrieval.execution import (
    RetrievalExecution,
    execution_scope,
)
from personagraph.retrieval.compute.resources import compute_resource_lock
from personagraph.retrieval.indexing.encoder import BgeM3Encoder, RetrievalMethodUnavailable
from personagraph.retrieval.indexing.model_assets import LocalModelAssetRef
from personagraph.retrieval.ports import RetrievalCancelled


class FakeCudaOutOfMemoryError(RuntimeError):
    pass


class FakeForward:
    def __init__(self) -> None:
        self.error: Exception | None = None
        self.calls = 0

    def forward(self, texts: list[str]) -> dict[str, object]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return {"dense_vecs": [[0.25] * 1024 for _text in texts]}

    def __call__(self, texts: list[str]) -> dict[str, object]:
        return self.forward(texts)


class FakeFlagEncoder:
    """复现供应商吞 RuntimeError 的边界，不加载 Torch 或真实权重。"""

    def __init__(self) -> None:
        self.model = FakeForward()
        self.tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [1, 2])
        self.encode_calls = 0
        self.swallowed_runtime_errors = 0

    def encode(self, texts: list[str], **_kwargs: object) -> dict[str, object]:
        self.encode_calls += 1
        for _attempt in range(3):
            try:
                return self.model(texts)
            except RuntimeError:
                self.swallowed_runtime_errors += 1
        raise ValueError("vendor_exhausted_batch_reduction")


class NoCancellation:
    def checkpoint(self) -> None:
        pass

    def remaining_seconds(self) -> None:
        return None

    def snapshot(self) -> dict[str, object]:
        return {}


@pytest.fixture
def encoder_asset(tmp_path: Path) -> LocalModelAssetRef:
    model_path = tmp_path / "encoder-fixture"
    model_path.mkdir()
    for filename in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors",
    ):
        (model_path / filename).write_bytes(b"fake-model-never-loaded")
    return LocalModelAssetRef.path(model_path, canonical_identity="test:encoder-device")


@pytest.fixture(autouse=True)
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                OutOfMemoryError=FakeCudaOutOfMemoryError,
                is_available=lambda: True,
                device_count=lambda: 1,
                synchronize=lambda *_args, **_kwargs: None,
            ),
            mps=SimpleNamespace(synchronize=lambda: None),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        ),
    )


def make_encoder(
    asset: LocalModelAssetRef, *, device: str = "mps"
) -> tuple[BgeM3Encoder, FakeFlagEncoder]:
    encoder = BgeM3Encoder(asset=asset, device=device, use_fp16=False)
    model = FakeFlagEncoder()
    encoder._model = model
    encoder._tokenizer = model.tokenizer
    return encoder, model


def test_encoder_rejects_auto_before_a_model_or_generation_can_be_bound(encoder_asset):
    with pytest.raises(ValueError, match="resolve auto"):
        BgeM3Encoder(asset=encoder_asset, device="auto", use_fp16=False)


def test_encoder_module_default_stays_cpu_and_preserves_identity(encoder_asset):
    encoder = BgeM3Encoder(asset=encoder_asset, use_fp16=False)
    model = FakeFlagEncoder()
    encoder._model = model
    encoder._tokenizer = model.tokenizer
    fingerprint = encoder.fingerprint()

    first = encoder.encode_query("same question")
    second = encoder.encode_query("same question")

    assert first is second
    assert model.encode_calls == 1
    assert encoder.diagnostic_snapshot()["device"] == "cpu"
    assert "device=cpu;" in fingerprint
    assert encoder.fingerprint() == fingerprint


@pytest.mark.parametrize(
    ("device", "error"),
    [
        ("mps", RuntimeError("MPS backend out of memory")),
        ("cuda:0", FakeCudaOutOfMemoryError("CUDA out of memory")),
    ],
)
def test_gpu_failure_requires_cpu_rebuild_without_changing_identity_or_retrying(
    encoder_asset, device, error
):
    encoder, model = make_encoder(encoder_asset, device=device)
    fingerprint = encoder.fingerprint()
    original_forward = model.model.forward
    model.model.error = error

    with pytest.raises(RetrievalMethodUnavailable, match="cpu_rebuild_required") as caught:
        encoder.encode_query("uncached question")

    assert caught.value.stage == "encode"
    assert model.encode_calls == model.model.calls == 1
    assert model.swallowed_runtime_errors == 0
    assert model.model.forward == original_forward
    assert encoder.diagnostic_snapshot()["device"] == device
    assert encoder.fingerprint() == fingerprint
    assert not encoder._query_cache


def test_device_failure_cannot_be_hidden_by_an_earlier_cached_query(encoder_asset):
    encoder, model = make_encoder(encoder_asset)
    encoder.encode_query("previous question")
    fingerprint = encoder.fingerprint()
    model.model.error = RuntimeError("MPS backend out of memory")

    with pytest.raises(RetrievalMethodUnavailable, match="cpu_rebuild_required"):
        encoder.encode_query("new question")
    model.model.error = None
    calls_after_failure = model.encode_calls

    with pytest.raises(RetrievalMethodUnavailable, match="cpu_rebuild_required"):
        encoder.encode_query("previous question")

    assert not encoder._query_cache
    assert model.encode_calls == calls_after_failure
    assert encoder.fingerprint() == fingerprint


def test_inflight_query_cannot_publish_cache_after_another_call_requires_cpu_rebuild(
    encoder_asset, monkeypatch
):
    encoder, model = make_encoder(encoder_asset)
    fingerprint = encoder.fingerprint()
    old_result_encoded = threading.Event()
    allow_old_result_to_finish = threading.Event()
    finished = threading.Event()
    results = []
    errors = []

    def pause_old_result(text, **_kwargs):
        if text == "old question":
            # 原始向量已产生且设备锁已释放；暂停在 encode_query 发布缓存之前。
            assert model.model.calls == 1
            assert not compute_resource_lock("mps").locked()
            old_result_encoded.set()
            assert allow_old_result_to_finish.wait(timeout=2.0)
        return [1, 2]

    monkeypatch.setattr(model.tokenizer, "encode", pause_old_result)

    def encode_old_question():
        try:
            results.append(encoder.encode_query("old question"))
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=encode_old_question, name="encoder-late-publication")
    thread.start()
    try:
        assert old_result_encoded.wait(timeout=2.0)
        assert not finished.is_set()
        assert not encoder._query_cache
        model.model.error = RuntimeError("MPS backend out of memory")

        with pytest.raises(RetrievalMethodUnavailable, match="cpu_rebuild_required"):
            encoder.encode_query("new question")

        assert not finished.is_set()
        assert encoder.diagnostic_snapshot()["cpu_rebuild_required"] is True
    finally:
        allow_old_result_to_finish.set()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert results == []
    assert len(errors) == 1
    assert isinstance(errors[0], RetrievalMethodUnavailable)
    assert errors[0].safe_error_code == "bge_m3_cpu_rebuild_required:device_out_of_memory"
    assert not encoder._query_cache
    assert model.model.calls == 2
    assert encoder.fingerprint() == fingerprint


def test_gpu_load_failure_does_not_reload_or_change_device(encoder_asset, monkeypatch):
    # Exercise the vendor's load failure independently of optional-package preflight.
    monkeypatch.setattr(
        BgeM3Encoder, "_local_model_path", lambda _self: Path(encoder_asset.identifier),
    )
    encoder = BgeM3Encoder(asset=encoder_asset, device="mps", use_fp16=False)
    fingerprint = encoder.fingerprint()
    constructor_devices = []

    def failing_constructor(_path, **kwargs):
        assert Path(_path) == Path(encoder_asset.identifier)
        constructor_devices.append(kwargs["devices"])
        raise RuntimeError("MPS backend out of memory")

    monkeypatch.setitem(
        sys.modules, "FlagEmbedding", SimpleNamespace(BGEM3FlagModel=failing_constructor)
    )

    for _attempt in range(2):
        with pytest.raises(RetrievalMethodUnavailable, match="cpu_rebuild_required"):
            encoder.encode_query("question")

    assert constructor_devices == ["mps"]
    assert not encoder._query_cache
    assert encoder.fingerprint() == fingerprint


def test_gpu_synchronization_failure_invalidates_the_whole_encoded_batch(
    encoder_asset, monkeypatch
):
    encoder, model = make_encoder(encoder_asset)

    def failing_synchronize():
        raise RuntimeError("MPS backend out of memory")

    monkeypatch.setattr(sys.modules["torch"].mps, "synchronize", failing_synchronize)

    with pytest.raises(RetrievalMethodUnavailable, match="cpu_rebuild_required"):
        encoder.encode(("first question", "second question"))

    assert model.model.calls == 1
    assert not encoder._query_cache
    assert not compute_resource_lock("mps").locked()
    assert encoder.diagnostic_snapshot()["cpu_rebuild_required"] is True


@pytest.mark.parametrize("device", ["mps", "cuda:0"])
def test_unknown_runtime_error_escapes_vendor_retries_without_claiming_device_failure(
    encoder_asset, device
):
    encoder, model = make_encoder(encoder_asset, device=device)
    original = RuntimeError("tensor shape is incompatible")
    model.model.error = original
    original_forward = model.model.forward

    with pytest.raises(RetrievalMethodUnavailable) as caught:
        encoder.encode(("question",))

    assert caught.value.safe_error_code == "bge_m3_encode_failed:RuntimeError"
    assert caught.value.__cause__ is original
    assert model.model.calls == 1
    assert model.swallowed_runtime_errors == 0
    assert model.model.forward == original_forward
    assert encoder.diagnostic_snapshot()["cpu_rebuild_required"] is False
    model.model.error = None
    assert encoder.encode_query("question").token_ids == (1, 2)


def test_cpu_error_with_gpu_words_does_not_request_a_new_generation(encoder_asset):
    encoder, model = make_encoder(encoder_asset, device="cpu")
    model.model.error = RuntimeError("MPS backend out of memory")

    with pytest.raises(RetrievalMethodUnavailable) as caught:
        encoder.encode(("question",))

    # 本轮只约束 GPU 异常边界，CPU 保留供应商既有的缩批和最终报错行为。
    assert caught.value.safe_error_code == "bge_m3_encode_failed:ValueError"
    assert model.model.calls == model.swallowed_runtime_errors == 3
    assert encoder.diagnostic_snapshot()["cpu_rebuild_required"] is False


def test_cancellation_propagates_without_device_poisoning_or_result_cache(encoder_asset):
    encoder, model = make_encoder(encoder_asset)
    cancelled = RetrievalCancelled("execution_timeout")
    model.model.error = cancelled
    original_forward = model.model.forward

    with pytest.raises(RetrievalCancelled) as caught:
        encoder.encode_query("question")

    assert caught.value is cancelled
    assert not encoder._query_cache
    assert model.model.calls == 1
    assert model.model.forward == original_forward
    model.model.error = None
    assert encoder.encode_query("question").token_ids == (1, 2)


def test_gpu_completion_wait_is_inside_resource_lock_and_encode_timing(
    encoder_asset, monkeypatch
):
    encoder, model = make_encoder(encoder_asset)
    now = [10.0]
    synchronization_calls = []
    resource_lock = compute_resource_lock("mps")

    def synchronize() -> None:
        assert resource_lock.locked()
        synchronization_calls.append(model.model.calls)
        now[0] += 2.0

    monkeypatch.setattr(sys.modules["torch"].mps, "synchronize", synchronize)
    execution = RetrievalExecution(NoCancellation(), clock=lambda: now[0])

    with execution_scope(execution):
        encoder.encode(("question",))

    assert synchronization_calls
    assert synchronization_calls[-1] == 1
    assert execution.snapshot()["stages"]["encoder_encode"]["duration_ms"] >= 2000
    assert not resource_lock.locked()
