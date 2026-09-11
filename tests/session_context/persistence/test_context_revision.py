import pytest

from personagraph.session.context import store as context_store, evidence, repair
from personagraph.session import store as session_store
from personagraph.session.context.models import (
    ObservationCandidate,
    Operation,
    SessionDomain,
    SourceKind,
)
from personagraph.session.context.reducer import reduce_candidate
from tests.helpers.session_records import append_test_turn


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "DB_PATH", tmp_path / "sessions.sqlite")


def _candidate(session_id: str, evidence_id: str):
    return ObservationCandidate(
        candidate_id="revision-candidate",
        session_id=session_id,
        domain=SessionDomain.USER,
        state_type="temporary_preference",
        key="response_depth",
        proposed_value="detailed",
        operation=Operation.SET,
        source_kind=SourceKind.EXPLICIT,
        derived_from=(evidence_id,),
        extractor_version="session-context-v1:extractor-v1",
    )


def test_revision_increases_for_turn_candidate_view_and_correction_writes():
    session_id = session_store.create_session("Entelecheia", "revision")
    assert context_store.context_revision(session_id) == 0

    turn_idx = append_test_turn(session_id, "user", "answer in detail")
    after_turn = context_store.context_revision(session_id)
    assert after_turn > 0

    candidate = _candidate(session_id, evidence.turn_evidence_id(session_id, turn_idx))
    context_store.save_candidates((candidate,))
    after_candidate = context_store.context_revision(session_id)
    assert after_candidate > after_turn

    context_store.apply_transition(
        reduce_candidate(candidate, None, now="2026-07-12T10:00:00+00:00")
    )
    after_view = context_store.context_revision(session_id)
    assert after_view > after_candidate

    repair.record_correction(
        session_id,
        SessionDomain.USER,
        "temporary_preference",
        "response_depth",
        "brief",
    )
    assert context_store.context_revision(session_id) > after_view


def test_replace_compare_and_swap_rejects_a_stale_revision_without_writes():
    session_id = session_store.create_session("Entelecheia", "revision cas")
    expected = context_store.context_revision(session_id)
    turn_idx = append_test_turn(session_id, "user", "new evidence")
    candidate = _candidate(session_id, evidence.turn_evidence_id(session_id, turn_idx))
    context_store.save_candidates((candidate,))

    with pytest.raises(context_store.ContextRevisionConflict) as exc:
        context_store.replace_session_context(
            session_id,
            candidates=(),
            state_items=(),
            audits=(),
            expected_revision=expected,
        )

    assert exc.value.actual > exc.value.expected
    assert context_store.list_candidates(session_id) == [candidate]


def test_repair_rejects_preview_after_any_new_authoritative_turn():
    session_id = session_store.create_session("Entelecheia", "stale preview")
    turn_idx = append_test_turn(session_id, "user", "answer in detail")
    candidate = _candidate(session_id, evidence.turn_evidence_id(session_id, turn_idx))
    context_store.save_candidates((candidate,))
    preview = repair.repair_session(session_id, dry_run=True)

    append_test_turn(session_id, "user", "newer instruction")

    with pytest.raises(repair.RepairError, match="preview_stale"):
        repair.repair_session(
            session_id,
            dry_run=False,
            expected_preview_token=preview.preview_token,
        )
    assert context_store.list_state_items(session_id) == []
