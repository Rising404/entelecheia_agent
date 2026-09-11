from __future__ import annotations

from dataclasses import dataclass

import pytest

from personagraph.retrieval.contracts import (
    CURRENT_SESSION_TURN_CUTOFF_SCOPE_KEY,
    QueryProposal,
    RetrievalBudget,
    RetrievalRequest,
    SourceFilter,
    SourceType,
    TrustedRetrievalBoundary,
)
from personagraph.retrieval.indexing.encoder import DeterministicLexicalEncoder
from personagraph.retrieval.profile import DocumentRetrievalProfile
from personagraph.retrieval.sources.session.composition import (
    build_session_retrieval_composition,
    prepare_session_retrieval_binding,
)
from personagraph.retrieval.sources.session.contracts import SessionRetrievalNotReady
from personagraph.retrieval.sources.session.lifecycle import (
    ensure_committed_session_pair_retrieval_ready,
    ensure_session_retrieval_ready,
    purge_session_retrieval_index,
)
from personagraph.retrieval.sources.session.projection import (
    SESSION_MAX_TOKENS,
)


@dataclass
class FakeSessionStore:
    pairs: list[dict[str, object]]
    status: str = "active"

    def get_session(self, session_id: str):
        return {"id": session_id, "status": self.status}

    def get_committed_turn_pair(self, session_id: str, run_id: str):
        return next(
            (
                pair
                for pair in self.pairs
                if pair["session_id"] == session_id and pair["run_id"] == run_id
            ),
            None,
        )

    def list_committed_turn_pairs(self, session_id: str, *, limit=None):
        selected = [pair for pair in self.pairs if pair["session_id"] == session_id]
        return selected if limit is None else selected[-limit:]


def _pair(
    run_id: str,
    *,
    assistant_turn_idx: int,
    user: str,
    assistant: str,
):
    return {
        "session_id": "session-1",
        "run_id": run_id,
        "created_at": f"2026-09-01T00:00:0{assistant_turn_idx}Z",
        "user_turn_idx": assistant_turn_idx - 1,
        "assistant_turn_idx": assistant_turn_idx,
        "user_content": user,
        "assistant_content": assistant,
    }


def _composition(tmp_path, store):
    return build_session_retrieval_composition(
        retrieval_db_path=tmp_path / "session_retrieval.sqlite",
        profile=DocumentRetrievalProfile.lexical(),
        encoder=DeterministicLexicalEncoder(),
        reranker=None,
        store=store,
    )


def _filter(cutoff: int):
    return SourceFilter.from_mapping(
        SourceType.CURRENT_SESSION,
        {
            "session_id": "session-1",
            CURRENT_SESSION_TURN_CUTOFF_SCOPE_KEY: str(cutoff),
        },
    )


def test_short_turn_pair_remains_one_atomic_retrieval_unit(tmp_path):
    store = FakeSessionStore(
        [_pair("run-1", assistant_turn_idx=1, user="短问题", assistant="短回答")]
    )
    composition = _composition(tmp_path, store)

    units = composition.source_adapter.list_indexable_units(_filter(1))

    assert len(units) == 1
    assert units[0].content == "用户：短问题\n\n助手：短回答"
    assert composition.chunking_profile.tokens_of(units[0].content) < 80


def test_long_turn_splits_by_role_and_never_crosses_role_overlap(tmp_path):
    user = " ".join(f"userterm{i}" for i in range(1_200))
    store = FakeSessionStore(
        [_pair("run-1", assistant_turn_idx=1, user=user, assistant="brief answer")]
    )
    composition = _composition(tmp_path, store)

    units = composition.source_adapter.list_indexable_units(_filter(1))
    user_units = [unit for unit in units if unit.content.startswith("用户：")]
    assistant_units = [unit for unit in units if unit.content.startswith("助手：")]

    assert len(user_units) > 1
    assert len(assistant_units) == 1
    assert assistant_units[0].content == "助手：brief answer"
    assert all(
        composition.chunking_profile.tokens_of(unit.content) <= SESSION_MAX_TOKENS
        for unit in units
    )
    previous_tail = user_units[0].content.split()[-1]
    assert previous_tail in user_units[1].content
    assert "userterm" not in assistant_units[0].content


def test_cutoff_excludes_later_committed_pairs_before_backfill(tmp_path):
    store = FakeSessionStore(
        [
            _pair("run-1", assistant_turn_idx=1, user="first", assistant="one"),
            _pair("run-2", assistant_turn_idx=3, user="second", assistant="two"),
        ]
    )
    composition = _composition(tmp_path, store)

    units = composition.source_adapter.list_indexable_units(_filter(1))

    assert len(units) == 1
    assert "first" in units[0].content
    assert "second" not in units[0].content


def test_session_generation_backfills_and_retrieves_through_shared_kernel(tmp_path):
    store = FakeSessionStore(
        [
            _pair(
                "run-1",
                assistant_turn_idx=1,
                user="Where is the aurora ledger?",
                assistant="The aurora ledger is in cabinet seven.",
            )
        ]
    )
    composition = _composition(tmp_path, store)

    generation_id = ensure_session_retrieval_ready(
        composition,
        session_id="session-1",
        assistant_turn_cutoff=1,
    )
    result = composition.foundation.service.retrieve_context(
        RetrievalRequest(
            request_id="request-1",
            model_call_purpose="test_session_retrieval",
            query_proposal=QueryProposal(
                ("aurora ledger",),
                source_hints=frozenset({SourceType.CURRENT_SESSION}),
            ),
            boundary=TrustedRetrievalBoundary(
                {SourceType.CURRENT_SESSION: _filter(1)}
            ),
        ),
        RetrievalBudget(
            candidate_limit_per_source=8,
            context_token_limit=1_000,
            max_items=4,
        ),
    )

    assert generation_id == composition.generation_spec.version_id
    assert result.retrieval_data_version == generation_id
    assert result.items
    assert "cabinet seven" in result.items[0].content


def test_turn_binding_freezes_latest_complete_assistant_cutoff(tmp_path):
    store = FakeSessionStore(
        [
            _pair("run-1", assistant_turn_idx=1, user="first", assistant="one"),
            _pair("run-2", assistant_turn_idx=3, user="second", assistant="two"),
        ]
    )
    composition = _composition(tmp_path, store)

    binding = prepare_session_retrieval_binding(
        session_id="session-1",
        store=store,
        composition=composition,
    )

    assert binding.assistant_turn_cutoff == 3
    assert binding.data_version_id == composition.generation_spec.version_id


def test_committed_pair_indexing_is_incremental_and_idempotent(tmp_path):
    first = _pair(
        "run-1",
        assistant_turn_idx=1,
        user="Where is the amber key?",
        assistant="The amber key is under the north lamp.",
    )
    second = _pair(
        "run-2",
        assistant_turn_idx=3,
        user="Where is the blue key?",
        assistant="The blue key is under the south lamp.",
    )
    store = FakeSessionStore([first])
    composition = _composition(tmp_path, store)
    ensure_committed_session_pair_retrieval_ready(composition, pair=first)
    store.pairs.append(second)

    first_apply = ensure_committed_session_pair_retrieval_ready(
        composition,
        pair=second,
    )
    replay = ensure_committed_session_pair_retrieval_ready(
        composition,
        pair=second,
    )

    assert first_apply == composition.generation_spec.version_id
    assert replay == first_apply
    assert len(composition.foundation.catalog.list_stored_units(first_apply)) == 2


def test_failed_initial_generation_can_reopen_and_complete(tmp_path):
    pair = _pair(
        "run-1",
        assistant_turn_idx=1,
        user="Remember the brass token.",
        assistant="The brass token is stored in drawer nine.",
    )
    store = FakeSessionStore([pair])
    composition = _composition(tmp_path, store)
    spec = composition.generation_spec
    composition.foundation.catalog.ensure_staging_data_version(
        version_id=spec.version_id,
        fingerprint=spec.fingerprint,
    )
    composition.foundation.catalog.mark_data_version_failed(spec.version_id)

    generation_id = ensure_session_retrieval_ready(
        composition,
        session_id="session-1",
        assistant_turn_cutoff=1,
    )

    active = composition.foundation.catalog.active_data_version()
    assert generation_id == spec.version_id
    assert active is not None
    assert active.id == spec.version_id
    assert active.state.value == "ready"


def test_session_index_purge_removes_units_and_applied_receipts_do_not_block_restore(
    tmp_path,
):
    pair = _pair(
        "run-1",
        assistant_turn_idx=1,
        user="Remember the silver token.",
        assistant="The silver token is stored in drawer four.",
    )
    store = FakeSessionStore([pair])
    composition = _composition(tmp_path, store)
    generation_id = ensure_committed_session_pair_retrieval_ready(
        composition,
        pair=pair,
    )
    assert composition.foundation.catalog.list_stored_units(generation_id)

    purged = purge_session_retrieval_index(
        "session-1",
        retrieval_db_path=tmp_path / "session_retrieval.sqlite",
    )
    assert purged == 1
    assert composition.foundation.catalog.list_stored_units(generation_id) == ()

    restored = ensure_committed_session_pair_retrieval_ready(
        composition,
        pair=pair,
    )
    assert restored == generation_id
    units = composition.foundation.catalog.list_stored_units(generation_id)
    assert len(units) == 1
    assert units[0].index_state.value == "ready"


@pytest.mark.parametrize("unavailable_kind", ["exception", "missing", "trashed"])
@pytest.mark.parametrize("initial", [True, False])
def test_session_index_does_not_publish_unavailable_source_as_empty(
    tmp_path, monkeypatch, unavailable_kind, initial,
):
    pair = _pair("run-1", assistant_turn_idx=1, user="question", assistant="answer")
    store = FakeSessionStore([pair])
    composition = _composition(tmp_path, store)
    if not initial:
        ensure_committed_session_pair_retrieval_ready(composition, pair=pair)

    def unavailable_session(_session_id):
        if unavailable_kind == "exception":
            raise RuntimeError("metadata reader unavailable")
        if unavailable_kind == "missing":
            return None
        return {"id": "session-1", "status": "trashed"}

    monkeypatch.setattr(store, "get_session", unavailable_session)

    with pytest.raises(SessionRetrievalNotReady):
        ensure_committed_session_pair_retrieval_ready(composition, pair=pair)

    if initial:
        assert composition.foundation.catalog.active_data_version() is None


def test_ready_staging_generation_is_reopened_and_indexes_exact_pair(tmp_path):
    pair = _pair("run-1", assistant_turn_idx=1, user="question", assistant="answer")
    composition = _composition(tmp_path, FakeSessionStore([pair]))
    catalog = composition.foundation.catalog
    spec = composition.generation_spec
    catalog.ensure_staging_data_version(
        version_id=spec.version_id, fingerprint=spec.fingerprint,
    )
    catalog.mark_data_version_ready(spec.version_id)

    generation = ensure_committed_session_pair_retrieval_ready(composition, pair=pair)

    assert generation == spec.version_id
    assert len(catalog.list_stored_units(generation)) == 1
    assert catalog.active_data_version().id == generation


@pytest.mark.parametrize("prior_generation", [False, True])
def test_bootstrap_requires_the_requested_pair_not_only_any_ready_generation(
    tmp_path, monkeypatch, prior_generation,
):
    pair = _pair("run-1", assistant_turn_idx=1, user="question", assistant="answer")
    store = FakeSessionStore([pair])
    composition = _composition(tmp_path, store)
    catalog = composition.foundation.catalog
    if prior_generation:
        catalog.ensure_staging_data_version(version_id="old-index", fingerprint="old")
        catalog.mark_data_version_ready("old-index")
        catalog.activate_data_version("old-index")
    monkeypatch.setattr(store, "list_committed_turn_pairs", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(store, "get_committed_turn_pair", lambda *_args: None)

    with pytest.raises(SessionRetrievalNotReady):
        ensure_committed_session_pair_retrieval_ready(composition, pair=pair)

    active = catalog.active_data_version()
    assert (active.id if active is not None else None) == (
        "old-index" if prior_generation else None
    )


def test_applied_receipt_does_not_hide_missing_method_representation(
    tmp_path, monkeypatch,
):
    pair = _pair("run-1", assistant_turn_idx=1, user="silver key", assistant="drawer four")
    composition = _composition(tmp_path, FakeSessionStore([pair]))
    generation = ensure_committed_session_pair_retrieval_ready(composition, pair=pair)
    foundation = composition.foundation
    stored = foundation.catalog.list_stored_units(generation)[0]
    foundation.method_store.purge(stored)
    indexed = []
    original_index = foundation.method_store.index

    def count_index(unit, content):
        indexed.append(unit.unit_id)
        return original_index(unit, content)

    monkeypatch.setattr(foundation.method_store, "index", count_index)

    ensure_committed_session_pair_retrieval_ready(composition, pair=pair)
    ensure_committed_session_pair_retrieval_ready(composition, pair=pair)

    assert indexed == [stored.unit_id]
    assert any(
        item.method.value == "bm25" and item.representation_present
        for item in foundation.method_store.method_index_health(stored.unit_id)
    )
