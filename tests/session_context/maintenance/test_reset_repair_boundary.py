import pytest

from personagraph.session.context import evidence, extractor, repair
from personagraph.session.context import reset as context_reset
from personagraph.session.context import store as context_store
from personagraph.session import store as session_store
from personagraph.session.context.models import (
    EvidenceKind,
    ObservationCandidate,
    Operation,
    SessionDomain,
    SessionExtractionResult,
    SourceKind,
)
from tests.helpers.session_records import append_test_turn


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "DB_PATH", tmp_path / "sessions.sqlite")


def _candidate(session_id: str, candidate_id: str, evidence_id: str, value: str):
    record = evidence.get_evidence(evidence_id)
    return ObservationCandidate(
        candidate_id=candidate_id,
        session_id=session_id,
        domain=SessionDomain.USER,
        state_type="temporary_preference",
        key="response_depth",
        proposed_value=value,
        operation=Operation.SET,
        source_kind=SourceKind.EXPLICIT,
        derived_from=(evidence_id,),
        extractor_version=extractor.EXTRACTOR_VERSION,
        valid_from=record.created_at,
    )


def test_repair_cannot_rebuild_from_pre_reset_candidate_or_evidence():
    session_id = session_store.create_session("Entelecheia", "repair reset")
    pre_idx = append_test_turn(session_id, "user", "answer in detail")
    pre_evidence_id = evidence.turn_evidence_id(session_id, pre_idx)
    context_store.save_candidates((
        _candidate(session_id, "candidate-before-reset", pre_evidence_id, "detailed"),
    ))

    context_reset.clear_session_context(
        session_id,
        request_id="reset-before-repair",
        actor="user",
        reason="forget derived context",
    )
    preview = repair.repair_session(session_id, dry_run=True)

    assert preview.candidate_count == 0
    assert preview.reset_id is not None
    assert preview.cutoff_turn_idx == pre_idx
    assert preview.diff == {
        "added": [], "removed": [], "changed": [], "unchanged_count": 0,
    }
    assert context_store.list_state_items(session_id) == []
    assert evidence.get_evidence(pre_evidence_id) is not None


def test_repair_accepts_post_reset_state_and_force_reextract_sees_only_new_evidence():
    session_id = session_store.create_session("Entelecheia", "repair post reset")
    old_idx = append_test_turn(session_id, "user", "old preference")
    old_evidence_id = evidence.turn_evidence_id(session_id, old_idx)
    old_event = evidence.append_event(
        session_id,
        EvidenceKind.TOOL_RESULT,
        "tool:before-reset",
        "old event with a future business timestamp",
        created_at="2099-01-01T00:00:00+00:00",
    )
    context_reset.clear_session_context(
        session_id,
        request_id="boundary",
        actor="user",
        reason="start fresh",
    )
    new_idx = append_test_turn(session_id, "user", "new preference")
    new_evidence_id = evidence.turn_evidence_id(session_id, new_idx)
    new_event = evidence.append_event(
        session_id,
        EvidenceKind.TOOL_RESULT,
        "tool:after-reset",
        "new event with an old business timestamp",
        created_at="2000-01-01T00:00:00+00:00",
    )
    post_candidate = _candidate(
        session_id,
        "candidate-after-reset",
        new_evidence_id,
        "brief",
    )
    context_store.save_candidates((post_candidate,))

    preview = repair.repair_session(session_id, dry_run=True)
    applied = repair.repair_session(
        session_id,
        dry_run=False,
        expected_preview_token=preview.preview_token,
    )
    item = context_store.get_state_item(
        session_id,
        SessionDomain.USER,
        "temporary_preference",
        "response_depth",
    )
    assert applied.candidate_count == 1
    assert item.value_json == "brief"

    captured_ids = []

    def capture(_session_id, records):
        captured_ids.extend(record.id for record in records)
        return SessionExtractionResult((post_candidate,))

    repair.repair_session(
        session_id,
        dry_run=True,
        force_reextract=True,
        candidate_extractor=capture,
    )
    assert new_evidence_id in captured_ids
    assert new_event.id in captured_ids
    assert old_evidence_id not in captured_ids
    assert old_event.id not in captured_ids
