from __future__ import annotations

import pytest

from personagraph.retrieval.indexing.encoder import DeterministicLexicalEncoder
from personagraph.retrieval.profile import DocumentRetrievalProfile
from personagraph.retrieval.sources.session.composition import (
    build_session_retrieval_composition,
    prepare_session_retrieval_binding,
)
from personagraph.retrieval.sources.session.lifecycle import (
    ensure_committed_session_pair_retrieval_ready,
)
from personagraph.runtime.post_commit.session_retrieval_recovery import (
    reconcile_session_retrieval_before_turn,
)
from personagraph.session import store
from tests.helpers.session_records import (
    complete_test_turn_execution as _complete_turn,
    completed_test_turn_id as _turn_id,
)


def test_pre_turn_recovery_settles_index_job_left_under_a_crashed_worker_lease(
    tmp_path,
):
    session_id = store.create_session("Entelecheia")
    completed = _complete_turn(
        session_id,
        1,
        post_commit_job_kinds=("session_retrieval_index",),
    )
    store.claim_due_turn_post_commit_jobs(
        session_id=session_id,
        worker_id="crashed-index-worker",
        lease_seconds=420,
    )
    composition = build_session_retrieval_composition(
        retrieval_db_path=tmp_path / "session_retrieval.sqlite",
        profile=DocumentRetrievalProfile.lexical(),
        encoder=DeterministicLexicalEncoder(),
        reranker=None,
        store=store,
    )
    binding = prepare_session_retrieval_binding(
        session_id=session_id,
        store=store,
        composition=composition,
    )

    reconcile_session_retrieval_before_turn(
        session_id=session_id,
        store=store,
        binding=binding,
    )

    job = store.list_turn_post_commit_jobs(_turn_id(completed))[0]
    assert job["status"] == "applied"
    assert (
        store.get_turn_execution_window(session_id)["window_state"]  # type: ignore[index]
        == "empty"
    )
    units = composition.foundation.catalog.list_stored_units(
        composition.generation_spec.version_id
    )
    assert len(units) == 1


@pytest.mark.parametrize("damage", ["unit", "method_representation"])
def test_pre_turn_recovery_repairs_older_partial_pair_without_reencoding_healthy_units(
    tmp_path, monkeypatch, damage,
):
    session_id = store.create_session("Entelecheia")
    first = _complete_turn(
        session_id, 1,
        user_content=" ".join(f"term{i}" for i in range(1_200)),
        assistant_content="first answer",
    )["pair"]
    second = _complete_turn(
        session_id, 2, user_content="latest question", assistant_content="latest answer",
    )["pair"]
    composition = build_session_retrieval_composition(
        retrieval_db_path=tmp_path / "session_retrieval.sqlite",
        profile=DocumentRetrievalProfile.lexical(),
        encoder=DeterministicLexicalEncoder(),
        reranker=None,
        store=store,
    )
    generation = ensure_committed_session_pair_retrieval_ready(composition, pair=second)
    foundation = composition.foundation
    old_refs = {unit.ref for unit in composition.source_adapter.project_committed_pair(first)}
    original = foundation.catalog.list_stored_units(generation)
    missing = next(unit for unit in original if unit.unit.ref in old_refs)
    assert len(old_refs) > 1
    if damage == "unit":
        foundation.catalog.delete_unit(missing.unit_id)
    else:
        foundation.method_store.purge(missing)
    indexed = []
    original_index = foundation.method_store.index

    def count_index(unit, content):
        indexed.append(unit.unit.ref)
        return original_index(unit, content)

    monkeypatch.setattr(foundation.method_store, "index", count_index)
    binding = prepare_session_retrieval_binding(
        session_id=session_id, store=store, composition=composition,
    )

    reconcile_session_retrieval_before_turn(
        session_id=session_id, store=store, binding=binding,
    )
    reconcile_session_retrieval_before_turn(
        session_id=session_id, store=store, binding=binding,
    )

    assert len(foundation.catalog.list_stored_units(generation)) == len(original)
    assert foundation.catalog.get_unit(missing.unit.ref, generation) is not None
    assert indexed == [missing.unit.ref]
