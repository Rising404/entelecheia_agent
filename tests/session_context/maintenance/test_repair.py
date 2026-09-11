from dataclasses import replace

import pytest

from personagraph.session.context import store as context_store, evidence, repair
from personagraph.session import store as ss
from personagraph.session.context.models import (
    ObservationCandidate,
    Operation,
    SessionDomain,
    SessionExtractionResult,
    SourceKind,
)
from tests.helpers.session_records import complete_test_turn_execution


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "DB_PATH", tmp_path / "sessions.sqlite")


def _session_with_candidate():
    sid = ss.create_session("Entelecheia", "repair")
    completed = complete_test_turn_execution(
        sid,
        1,
        user_content="请详细解释",
        assistant_content="好的",
    )
    commit = completed["pair"]
    evidence_id = evidence.turn_evidence_id(sid, int(commit["user_turn_idx"]))
    record = evidence.get_evidence(evidence_id)
    candidate = ObservationCandidate(
        candidate_id="candidate-repair",
        session_id=sid,
        domain=SessionDomain.USER,
        state_type="temporary_preference",
        key="response_depth",
        proposed_value="detailed",
        operation=Operation.SET,
        source_kind=SourceKind.EXPLICIT,
        derived_from=(evidence_id,),
        extractor_version="session-context-v1:extractor-v1",
        valid_from=record.created_at,
    )
    context_store.save_candidates([candidate])
    return sid, candidate


def test_repair_dry_run_has_zero_writes_then_apply_is_repeatable():
    sid, _candidate = _session_with_candidate()

    preview = repair.repair_session(sid, dry_run=True)
    assert preview.applied is False
    assert preview.candidate_source == "stored"
    assert len(preview.diff["added"]) == 1
    assert context_store.list_state_items(sid) == []
    assert context_store.list_transition_audits(sid) == []

    applied = repair.repair_session(
        sid, dry_run=False, expected_preview_token=preview.preview_token
    )
    assert applied.applied is True
    first_items = context_store.list_state_items(sid)
    first_audits = context_store.list_transition_audits(sid)
    repeated = repair.repair_session(sid, dry_run=True)
    assert repeated.diff == {
        "added": [], "removed": [], "changed": [], "unchanged_count": 1,
    }
    assert context_store.list_state_items(sid) == first_items
    assert context_store.list_transition_audits(sid) == first_audits


def test_atomic_replace_failure_preserves_old_view(monkeypatch):
    sid, candidate = _session_with_candidate()
    initial_preview = repair.repair_session(sid, dry_run=True)
    repair.repair_session(
        sid, dry_run=False, expected_preview_token=initial_preview.preview_token
    )
    before = context_store.list_state_items(sid)
    new_candidate = replace(
        candidate,
        candidate_id="candidate-repair-2",
        proposed_value="brief",
    )
    context_store.save_candidates([new_candidate])
    original_insert = context_store._insert_audit

    def fail_after_delete(*_args, **_kwargs):
        raise RuntimeError("injected replace failure")

    monkeypatch.setattr(context_store, "_insert_audit", fail_after_delete)
    preview = repair.repair_session(sid, dry_run=True)
    with pytest.raises(RuntimeError, match="injected"):
        repair.repair_session(
            sid, dry_run=False, expected_preview_token=preview.preview_token
        )
    monkeypatch.setattr(context_store, "_insert_audit", original_insert)
    assert context_store.list_state_items(sid) == before
    assert context_store.get_repair_apply(preview.preview_token) is None


def test_correction_rebuilds_one_slot_without_rewriting_turn_evidence():
    sid, _candidate = _session_with_candidate()
    initial_preview = repair.repair_session(sid, dry_run=True)
    repair.repair_session(
        sid, dry_run=False, expected_preview_token=initial_preview.preview_token
    )
    original_turns = ss.get_turns(sid)

    correction = repair.record_correction(
        sid,
        SessionDomain.USER,
        "temporary_preference",
        "response_depth",
        "brief",
    )
    preview = repair.repair_session(
        sid,
        dry_run=True,
        slot=(SessionDomain.USER, "temporary_preference", "response_depth"),
    )
    result = repair.repair_session(
        sid,
        dry_run=False,
        slot=(SessionDomain.USER, "temporary_preference", "response_depth"),
        expected_preview_token=preview.preview_token,
    )

    item = context_store.get_state_item(
        sid, SessionDomain.USER, "temporary_preference", "response_depth"
    )
    assert item.value_json == "brief"
    assert correction.id in item.derived_from
    assert ss.get_turns(sid) == original_turns
    assert result.applied is True


def test_force_reextract_accepts_injected_candidate_source():
    sid, candidate = _session_with_candidate()
    updated = replace(candidate, candidate_id="candidate-new", proposed_value="brief")

    result = repair.repair_session(
        sid,
        dry_run=True,
        force_reextract=True,
        candidate_extractor=lambda _sid, _records: SessionExtractionResult((updated,)),
    )

    assert result.candidate_source == "reextracted"
    assert len(result.diff["added"]) == 1
    assert context_store.list_state_items(sid) == []
