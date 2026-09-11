"""显式 opt-in 的本地 BGE 设备验收；不调用远端 API，不读取用户文档。"""

from __future__ import annotations

import gc
import json
import math
import os
import time

import pytest


@pytest.fixture
def smoke_execution():
    from personagraph.retrieval.execution import RetrievalExecution, execution_scope

    class Control:
        def checkpoint(self):
            pass

        def remaining_seconds(self):
            return None

        def snapshot(self):
            return {}

    execution = RetrievalExecution(Control())
    with execution_scope(execution):
        yield execution


@pytest.mark.skipif(
    os.environ.get("PERSONAGRAPH_RUN_LOCAL_DEVICE_SMOKE") != "1",
    reason="requires explicit local model/device smoke opt-in",
)
def test_real_bge_models_execute_on_the_selected_device(smoke_execution):
    from personagraph.retrieval.compute.inference import release_device_cache
    from personagraph.retrieval.profile import (
        DocumentRetrievalProfile,
        build_document_retrieval_runtime,
    )

    requested = os.environ.get("PERSONAGRAPH_SMOKE_DEVICE", "auto")
    runtime = build_document_retrieval_runtime(DocumentRetrievalProfile(device=requested))
    device = runtime.effective_profile.device
    expected = os.environ.get("PERSONAGRAPH_SMOKE_EXPECT_DEVICE")
    if expected:
        assert device == expected
    encoder = runtime.encoder
    reranker = runtime.reranker
    assert reranker is not None
    query = "What is the capital of France?"
    texts = ("Paris is the capital of France.", "Photosynthesis converts light into energy.")
    pairs = ((query, texts[0]), (query, texts[1]))
    try:
        start = time.perf_counter()
        encoded = encoder.encode((query, *texts))
        encoder_cold_s = time.perf_counter() - start
        start = time.perf_counter()
        hot_encoded = encoder.encode((query, *texts))
        encoder_hot_s = time.perf_counter() - start
        assert len(encoded) == 3
        assert encoder.learned_sparse_available
        for item in encoded:
            assert len(item.dense_vector) == 1024
            assert all(math.isfinite(value) for value in item.dense_vector)
            assert item.learned_sparse_weights
        assert all(
            math.isclose(a, b, rel_tol=1e-4, abs_tol=1e-5)
            for a, b in zip(encoded[0].dense_vector, hot_encoded[0].dense_vector, strict=True)
        )
        start = time.perf_counter()
        scores = reranker.score(pairs)
        reranker_cold_s = time.perf_counter() - start
        start = time.perf_counter()
        hot_scores = reranker.score(pairs)
        reranker_hot_s = time.perf_counter() - start
        assert len(scores) == 2 and all(math.isfinite(value) for value in scores)
        assert scores[0] > scores[1]
        assert hot_scores == pytest.approx(scores, rel=1e-4, abs=1e-4)
        audit = smoke_execution.snapshot()
        assert audit["metrics"]["reranker_device"] == device
        assert audit["metrics"]["reranker_forward_batches"] >= 2
        assert audit["stages"]["reranker_score"]["duration_ms"] > 0
        model_parameters = {}
        for name, component in (("encoder", encoder), ("reranker", reranker)):
            # 实机验收不能只相信配置字符串：检查已加载模型中全部参数的真实设备/精度。
            parameters = tuple(component._model.model.parameters())
            observed_devices = sorted({str(parameter.device) for parameter in parameters})
            dtypes = sorted({str(parameter.dtype) for parameter in parameters})
            assert observed_devices == ["mps:0" if device == "mps" else device]
            assert dtypes == ["torch.float32"]
            model_parameters[name] = {"devices": observed_devices, "dtypes": dtypes}
        print(json.dumps({
            "requested_device": requested,
            "effective_device": device,
            "device_selection": runtime.capability.device_selection.diagnostic_snapshot(),
            "model_parameters": model_parameters,
            "encoder_cold_s": round(encoder_cold_s, 4),
            "encoder_hot_s": round(encoder_hot_s, 4),
            "reranker_cold_s": round(reranker_cold_s, 4),
            "reranker_hot_s": round(reranker_hot_s, 4),
            "scores": scores,
            "dense_first_8": encoded[0].dense_vector[:8],
            "reranker_cpu_fallback_reason": reranker.diagnostic_snapshot()["cpu_fallback_reason"],
            "execution_audit": audit,
        }, ensure_ascii=False), flush=True)
    finally:
        encoder._model = None
        reranker._model = None
        gc.collect()
        release_device_cache(device)
