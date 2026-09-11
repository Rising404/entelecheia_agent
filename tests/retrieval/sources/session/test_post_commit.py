from __future__ import annotations

from inspect import signature
from types import SimpleNamespace

from personagraph.retrieval.sources.session import post_commit as session_post_commit
from personagraph.retrieval.sources.session.post_commit import (
    SessionRetrievalPostCommitStore,
    process_session_retrieval_index_job,
)
from personagraph.retrieval.indexing.encoder import DeterministicLexicalEncoder
from personagraph.retrieval.profile import DocumentRetrievalProfile
from personagraph.retrieval.sources.session.composition import build_session_retrieval_composition
from personagraph.retrieval.sources.session.lifecycle import (
    ensure_committed_session_pair_retrieval_ready,
)
from personagraph.runtime.post_commit.runner import process_due_turn_post_commit_jobs
from personagraph.session import store
from tests.helpers.session_records import (
    complete_test_turn_execution as _complete_turn,
    completed_test_turn_id as _turn_id,
)


def test_session_retrieval_job_indexes_the_exact_committed_pair_and_releases_window():
    session_id = store.create_session("Entelecheia")
    completed = _complete_turn(
        session_id,
        1,
        post_commit_job_kinds=("session_retrieval_index",),
    )
    indexed: list[dict[str, object]] = []

    result = process_due_turn_post_commit_jobs(
        session_id=session_id,
        store=store,
        worker_id="session-index-worker",
        session_retrieval_indexer=lambda pair: indexed.append(dict(pair)) or "v1",
    )

    assert result.applied_job_count == 1
    assert result.released_window is True
    assert indexed[0]["turn_id"] == _turn_id(completed)
    job = store.list_turn_post_commit_jobs(_turn_id(completed))[0]
    assert job["status"] == "applied"
    assert job["attempts"] == 1


def test_session_retrieval_job_persists_stable_failure_category_without_raw_error():
    session_id = store.create_session("Entelecheia")
    completed = _complete_turn(
        session_id,
        1,
        post_commit_job_kinds=("session_retrieval_index",),
    )

    def fail_index(_pair):
        raise RuntimeError("private encoder worker=batch-7 exploded")

    result = process_due_turn_post_commit_jobs(
        session_id=session_id,
        store=store,
        worker_id="session-index-worker",
        session_retrieval_indexer=fail_index,
    )

    job = store.list_turn_post_commit_jobs(_turn_id(completed))[0]
    assert result.failed_job_count == 1
    assert result.released_window is False
    assert job["status"] == "retryable_failed"
    assert job["reason_code"] == "SESSION_RETRIEVAL_INTERNAL_FAILURE"
    assert job["attempts"] == 1
    assert "private encoder" not in str(job)


def test_session_retrieval_post_commit_store_declares_default_builder_reads():
    assert tuple(signature(SessionRetrievalPostCommitStore.get_session).parameters) == (
        "self",
        "session_id",
    )
    assert tuple(
        signature(SessionRetrievalPostCommitStore.get_committed_turn_pair).parameters
    ) == ("self", "session_id", "run_id")
    assert tuple(
        signature(SessionRetrievalPostCommitStore.list_committed_turn_pairs).parameters
    ) == ("self", "session_id", "limit")


def test_session_retrieval_job_builds_the_default_composition_when_no_indexer_is_injected(
    monkeypatch,
):
    pair = {"turn_id": "turn-default-indexer"}
    composition = SimpleNamespace(name="default-session-retrieval")
    observed: dict[str, object] = {}

    class FakeStore:
        def get_committed_turn_pair_for_post_commit(self, *, session_id, turn_id):
            observed["pair_lookup"] = (session_id, turn_id)
            return pair

        def mark_turn_post_commit_job_applied(self, *, job_id, worker_id):
            observed["applied"] = (job_id, worker_id)
            return {"status": "applied"}

        def mark_turn_post_commit_job_failed(self, **kwargs):
            raise AssertionError(f"unexpected failure: {kwargs}")

    fake_store = FakeStore()

    def build_default_composition(*, store):
        observed["composition_store"] = store
        return composition

    monkeypatch.setattr(
        session_post_commit,
        "_build_default_composition",
        build_default_composition,
    )
    monkeypatch.setattr(
        session_post_commit,
        "_ensure_pair_ready",
        lambda actual_composition, *, pair: observed.update(
            indexed=(actual_composition, pair)
        )
        or "generation-v1",
    )

    applied = process_session_retrieval_index_job(
        session_id="session-default-indexer",
        job={"job_id": "job-default-indexer", "turn_id": "turn-default-indexer"},
        worker_id="worker-default-indexer",
        store=fake_store,
        index_pair=None,
    )

    assert applied is True
    assert observed == {
        "pair_lookup": ("session-default-indexer", "turn-default-indexer"),
        "composition_store": fake_store,
        "indexed": (composition, pair),
        "applied": ("job-default-indexer", "worker-default-indexer"),
    }


def test_session_retrieval_job_never_records_applied_after_source_read_failure(
    tmp_path, monkeypatch,
):
    session_id = store.create_session("Entelecheia")
    completed = _complete_turn(
        session_id, 1, post_commit_job_kinds=("session_retrieval_index",),
    )
    composition = build_session_retrieval_composition(
        retrieval_db_path=tmp_path / "session_retrieval.sqlite",
        profile=DocumentRetrievalProfile.lexical(),
        encoder=DeterministicLexicalEncoder(),
        reranker=None,
        store=store,
    )
    job = store.claim_due_turn_post_commit_jobs(
        session_id=session_id, worker_id="index-source-failure", lease_seconds=420,
    )[0]

    def failed_source(_session_id):
        raise RuntimeError("metadata reader unavailable")

    monkeypatch.setattr(store, "get_session", failed_source)
    applied = process_session_retrieval_index_job(
        session_id=session_id, job=job, worker_id="index-source-failure", store=store,
        index_pair=lambda pair: ensure_committed_session_pair_retrieval_ready(
            composition, pair=pair,
        ),
    )

    persisted = store.list_turn_post_commit_jobs(_turn_id(completed))[0]
    assert applied is False
    assert persisted["status"] == "retryable_failed"
    assert persisted["reason_code"] == "SESSION_RETRIEVAL_INDEX_UNAVAILABLE"
    assert composition.foundation.catalog.active_data_version() is None
