from __future__ import annotations

import threading
import sys
from types import SimpleNamespace

import pytest

from personagraph.retrieval.execution import (
    RetrievalExecution,
    RerankingDeadlineReached,
    execution_scope,
    wait_for_lock,
)
from personagraph.retrieval.ports import RetrievalCancelled
from personagraph.retrieval.orchestration.recovery import RetrievalRecoveryController
from personagraph.retrieval.orchestration.reranking import rerank_verified_candidates


class Cancellation:
    def __init__(self):
        self.cancelled = threading.Event()
        self.checked = threading.Event()

    def checkpoint(self):
        self.checked.set()
        if self.cancelled.is_set():
            raise RetrievalCancelled("execution_cancelled")

    def remaining_seconds(self):
        return None

    def snapshot(self):
        return {"cancellation_requested": self.cancelled.is_set()}


def test_cancelled_waiter_never_enters_the_scoring_resource():
    control = Cancellation()
    lock = threading.Lock()
    lock.acquire()
    errors = []
    entered = threading.Event()
    execution = RetrievalExecution(control)

    def waiter():
        try:
            with execution_scope(execution), wait_for_lock(lock, "reranker_queue"):
                entered.set()
        except RetrievalCancelled as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=waiter)
    thread.start()
    assert control.checked.wait(timeout=1.0)
    control.cancelled.set()
    thread.join(timeout=1.0)
    lock.release()

    assert not thread.is_alive()
    assert not entered.is_set()
    assert errors == ["execution_cancelled"]
    assert execution.snapshot()["stages"]["reranker_queue"]["calls"] == 1


def test_recovery_does_not_translate_cancellation_to_failure_or_retry():
    calls = []

    def cancelled():
        calls.append(1)
        raise RetrievalCancelled("execution_timeout")

    with pytest.raises(RetrievalCancelled, match="execution_timeout"):
        RetrievalRecoveryController(max_attempts=3).run(("query",), cancelled)
    assert calls == [1]


def test_cancelled_reranker_does_not_return_successful_fusion_fallback():
    from personagraph.retrieval.contracts import (
        RetrievalCandidate,
        RetrievalMethod,
        SourceType,
        SourceUnit,
        SourceUnitRef,
    )

    class CancelledReranker:
        def fingerprint(self):
            return "cancelled-reranker"

        def score(self, pairs):
            raise RetrievalCancelled("execution_timeout")

    ref = SourceUnitRef(SourceType.DOCUMENT, "one", "r1", "hash-one")
    candidate = RetrievalCandidate(ref, RetrievalMethod.DENSE, 0, 1, 1.0)
    with pytest.raises(RetrievalCancelled, match="execution_timeout"):
        rerank_verified_candidates(
            queries=("question",),
            resolved_by_source={
                SourceType.DOCUMENT: ((candidate, SourceUnit(ref, "text")),)
            },
            reranker=CancelledReranker(),
            candidate_limit_per_source=32,
        )


def test_stage_timings_use_injected_clock_and_remain_outside_result_payload():
    now = [10.0]
    execution = RetrievalExecution(Cancellation(), clock=lambda: now[0])
    with execution.measure("reranker_score"):
        now[0] = 12.5
    assert execution.snapshot()["stages"] == {
        "reranker_score": {"calls": 1, "duration_ms": 2500}
    }


class HookedForward:
    def __init__(self, after_forward=lambda: None):
        self.pre_hooks = []
        self.post_hooks = []
        self.calls = 0
        self.after_forward = after_forward

    def register_forward_pre_hook(self, hook, *, with_kwargs):
        assert with_kwargs is True
        self.pre_hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.pre_hooks.remove(hook))

    def register_forward_hook(self, hook, *, with_kwargs):
        assert with_kwargs is True
        self.post_hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.post_hooks.remove(hook))

    def run(self, count):
        inputs = {"input_ids": SimpleNamespace(shape=(count, 12))}
        for hook in self.pre_hooks:
            hook(self, (), inputs)
        self.calls += 1
        self.after_forward()
        for hook in self.post_hooks:
            hook(self, (), inputs, None)


class HookedScorer:
    def __init__(self, forward):
        self.model = forward

    def compute_score(self, pairs, *, batch_size, **kwargs):
        for offset in range(0, len(pairs), batch_size):
            self.model.run(len(pairs[offset : offset + batch_size]))
        return list(range(len(pairs)))


def test_cancelled_tokenizer_stops_remaining_batches_and_restores_original_object():
    from personagraph.retrieval.orchestration.reranking import BgeM3Reranker

    control = Cancellation()

    class Tokenizer:
        calls = 0

        def __call__(self, values, **kwargs):
            self.calls += 1
            control.cancelled.set()
            return {"input_ids": [[1] for _ in values]}

    class TokenizingScorer(HookedScorer):
        def compute_score(self, pairs, *, batch_size, **kwargs):
            for offset in range(0, len(pairs), batch_size):
                self.tokenizer(
                    [query for query, _ in pairs[offset : offset + batch_size]]
                )
            return super().compute_score(pairs, batch_size=batch_size, **kwargs)

    original_tokenizer = Tokenizer()
    native = HookedForward()
    model = TokenizingScorer(native)
    model.tokenizer = original_tokenizer
    reranker = BgeM3Reranker(batch_size=2)
    reranker._model = model
    execution = RetrievalExecution(control)

    with execution_scope(execution), pytest.raises(RetrievalCancelled):
        reranker.score(tuple(("q", str(index)) for index in range(8)))

    assert original_tokenizer.calls == 1
    assert native.calls == 0
    assert model.tokenizer is original_tokenizer
    assert native.pre_hooks == native.post_hooks == []
    assert execution.snapshot()["stages"]["reranker_tokenize"]["calls"] == 1


def test_bge_cancellation_after_native_batch_stops_next_batch_and_removes_hooks():
    from personagraph.retrieval.orchestration.reranking import BgeM3Reranker

    control = Cancellation()
    native = HookedForward(control.cancelled.set)
    reranker = BgeM3Reranker(batch_size=2)
    reranker._model = HookedScorer(native)
    execution = RetrievalExecution(control)

    with execution_scope(execution), pytest.raises(RetrievalCancelled):
        reranker.score(tuple(("q", str(index)) for index in range(8)))

    assert native.calls == 1
    assert native.pre_hooks == native.post_hooks == []
    assert execution.snapshot()["metrics"]["reranker_forward_pairs"] == 2
    assert "reranker_scored_pairs" not in execution.snapshot()["metrics"]


def test_bge_soft_deadline_never_returns_partial_scores():
    from personagraph.retrieval.orchestration.reranking import BgeM3Reranker

    control = Cancellation()
    remaining = [10.0]
    control.remaining_seconds = lambda: remaining[0]
    native = HookedForward(lambda: remaining.__setitem__(0, 1.0))
    reranker = BgeM3Reranker(batch_size=2)
    reranker._model = HookedScorer(native)
    execution = RetrievalExecution(control)

    with execution_scope(execution), pytest.raises(RerankingDeadlineReached):
        reranker.score(tuple(("q", str(index)) for index in range(8)))

    assert native.calls == 1
    assert native.pre_hooks == native.post_hooks == []
    assert execution.snapshot()["metrics"]["reranker_deadline_degraded"] is True


def test_bge_next_batch_requires_observed_compute_time_plus_projection_reserve():
    from personagraph.retrieval.orchestration.reranking import BgeM3Reranker

    now = [0.0]
    control = Cancellation()
    control.remaining_seconds = lambda: 13.0 - now[0]
    native = HookedForward(lambda: now.__setitem__(0, now[0] + 8.0))
    reranker = BgeM3Reranker(batch_size=2)
    reranker._model = HookedScorer(native)
    execution = RetrievalExecution(control, clock=lambda: now[0])

    with execution_scope(execution), pytest.raises(RerankingDeadlineReached):
        reranker.score(tuple(("q", str(index)) for index in range(8)))

    assert native.calls == 1
    assert now[0] == 8.0
    assert execution.snapshot()["metrics"]["reranker_max_observed_batch_ms"] == 8000


def test_bge_forward_timings_and_pair_alignment_preserve_the_existing_scorer():
    from personagraph.retrieval.orchestration.reranking import BgeM3Reranker

    now = [0.0]
    native = HookedForward(lambda: now.__setitem__(0, now[0] + 0.25))
    reranker = BgeM3Reranker(batch_size=2)
    reranker._model = HookedScorer(native)
    execution = RetrievalExecution(Cancellation(), clock=lambda: now[0])

    with execution_scope(execution):
        scores = reranker.score(
            (("query two", "b"), ("query one", "a"), ("query two", "a"))
        )

    assert scores == (0.0, 1.0, 2.0)
    assert native.calls == 2
    assert execution.snapshot()["stages"]["reranker_score"] == {
        "calls": 2,
        "duration_ms": 500,
    }
    assert execution.snapshot()["metrics"]["reranker_scored_pairs"] == 3


def test_cancelled_cold_load_stops_before_scoring_and_is_not_reloaded(monkeypatch, tmp_path):
    from personagraph.retrieval.indexing.model_assets import LocalModelAssetRef
    from personagraph.retrieval.orchestration.reranking import BgeM3Reranker

    now = [0.0]
    control = Cancellation()
    native = HookedForward()
    loads = []

    def load(*args, **kwargs):
        assert args == (str(tmp_path),)
        loads.append(1)
        now[0] = 2.0
        control.cancelled.set()
        return HookedScorer(native)

    monkeypatch.setitem(
        sys.modules, "FlagEmbedding", SimpleNamespace(FlagReranker=load)
    )
    monkeypatch.setattr(BgeM3Reranker, "_local_model_path", lambda _self: tmp_path)
    reranker = BgeM3Reranker(
        asset=LocalModelAssetRef.path(tmp_path, canonical_identity="test:cancelled-load"),
        batch_size=2,
    )
    execution = RetrievalExecution(control, clock=lambda: now[0])
    with execution_scope(execution), pytest.raises(RetrievalCancelled):
        reranker.score((("q", "a"),))
    assert native.calls == 0
    assert reranker._load_failure_reason is None
    assert execution.snapshot()["stages"]["reranker_model_load"] == {
        "calls": 1,
        "duration_ms": 2000,
    }

    next_execution = RetrievalExecution(Cancellation())
    with execution_scope(next_execution):
        assert reranker.score((("q", "a"),)) == (0.0,)
    assert loads == [1]
    assert "reranker_model_load" not in next_execution.snapshot()["stages"]
    assert next_execution.snapshot()["metrics"]["reranker_model_cache_hits"] == 1


def test_file_service_cancellation_is_audited_without_success_evidence(monkeypatch):
    from personagraph.retrieval.contracts import CorpusKey
    from personagraph.retrieval.tooling.contracts import (
        FileRetrievalScope,
        RetrievalCorpus,
        RetrievalToolRequest,
    )
    from personagraph.retrieval.tooling.service import facade

    def cancel(*args, **kwargs):
        raise RetrievalCancelled("execution_timeout")

    foundation = SimpleNamespace(
        corpus_key=CorpusKey.FILE,
        service=SimpleNamespace(
            retrieve_context=cancel,
            retrieve_context_readonly=cancel,
            retrieve_file_query_batch=cancel,
        ),
        data_version_provider=SimpleNamespace(
            active_retrieval_data_version_id=lambda: "generation"
        ),
    )
    request = RetrievalToolRequest(
        request_id="cancel-query",
        corpus=RetrievalCorpus.FILES,
        limit=96,
        context_token_limit=96000,
        query="question",
        session_id="session",
        scope_snapshot_id="scope",
        retrieval_data_version="generation",
        file_scope=FileRetrievalScope.SESSION_CORPUS,
    )
    audit = []
    monkeypatch.setattr(
        facade, "_record_query_audit", lambda **kwargs: audit.append(kwargs)
    )
    port = facade.RetrievalServiceToolPort(
        file_foundation=foundation, file_readonly=True
    )

    with pytest.raises(RetrievalCancelled):
        port.retrieve_file_readonly_unrecorded(request)
    assert len(audit) == 1
    assert audit[0]["final_projection"]["evidence"] == []
    assert audit[0]["diagnostics"][0]["code"] == "retrieval_cancelled"


def test_token_count_cancellation_is_not_cached_as_fallback():
    from personagraph.retrieval.indexing.token_estimation import (
        BgeM3RetrievalTokenEstimator,
    )

    def cancel(_text):
        raise RetrievalCancelled("execution_timeout")

    estimator = BgeM3RetrievalTokenEstimator(SimpleNamespace(token_ids=cancel))
    with pytest.raises(RetrievalCancelled):
        estimator("not a tokenizer failure")
    assert estimator.diagnostic_snapshot().fallback_count == 0
    assert estimator.diagnostic_snapshot().cache_entries == 0
