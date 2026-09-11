from dataclasses import replace

import pytest

from personagraph.session.context import store as context_store
from personagraph.session import store as ss
from personagraph.session.context.models import (
    ObservationCandidate,
    Operation,
    ReasonCode,
    SessionDomain,
    SourceKind,
    TransitionDecision,
    TransitionResult,
)
from personagraph.session.context.reducer import reduce_candidate


NOW = "2026-07-12T10:00:00+00:00"


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "DB_PATH", tmp_path / "sessions.sqlite")


def _candidate(session_id: str, **overrides):
    values = {
        "candidate_id": "candidate-1",
        "session_id": session_id,
        "domain": SessionDomain.USER,
        "state_type": "temporary_preference",
        "key": "response_depth",
        "proposed_value": "detailed",
        "operation": Operation.SET,
        "source_kind": SourceKind.EXPLICIT,
        "derived_from": (f"turn:{session_id}:0",),
        "extractor_version": "fixture-v1",
    }
    values.update(overrides)
    return ObservationCandidate(**values)


def _session():
    return ss.create_session("Entelecheia", "context store")


def test_applied_transition_round_trips_item_and_audit():
    sid = _session()
    result = reduce_candidate(_candidate(sid), None, now=NOW)
    context_store.apply_transition(result)

    stored = context_store.get_state_item(
        sid, SessionDomain.USER, "temporary_preference", "response_depth"
    )
    assert stored == result.state_item
    assert context_store.list_transition_audits(sid) == [result.audit]


def test_exact_transition_retry_is_idempotent():
    sid = _session()
    result = reduce_candidate(_candidate(sid), None, now=NOW)
    context_store.apply_transition(result)
    context_store.apply_transition(result)
    assert len(context_store.list_state_items(sid)) == 1
    assert len(context_store.list_transition_audits(sid)) == 1


def test_candidate_log_round_trips_versions_and_detects_collisions():
    sid = _session()
    candidate = _candidate(sid)
    context_store.save_candidates([candidate])
    context_store.save_candidates([candidate])
    assert context_store.list_candidates(sid) == [candidate]
    assert context_store.list_candidates(sid, extractor_version="fixture-v1") == [candidate]
    assert context_store.list_candidates(sid, extractor_version="other") == []

    with pytest.raises(ValueError, match="candidate id collision"):
        context_store.save_candidates([replace(candidate, proposed_value="brief")])


def test_rejected_candidate_persists_audit_without_state_item():
    sid = _session()
    candidate = _candidate(sid, source_kind=SourceKind.INFERRED, confidence_hint=0.99)
    result = reduce_candidate(candidate, None, now=NOW)
    assert result.audit.reason_code == ReasonCode.SOURCE_REQUIRES_VERIFICATION
    context_store.apply_transition(result)
    assert context_store.list_state_items(sid) == []
    assert context_store.list_transition_audits(sid)[0].decision == TransitionDecision.REJECTED


def test_invalid_cross_session_result_is_rejected_before_partial_write():
    first_sid = _session()
    second_sid = _session()
    result = reduce_candidate(_candidate(first_sid), None, now=NOW)
    bad_audit = replace(result.audit, session_id=second_sid)

    with pytest.raises(ValueError, match="different sessions"):
        context_store.apply_transition(TransitionResult(result.state_item, bad_audit))
    assert context_store.list_state_items(first_sid) == []
    assert context_store.list_transition_audits(second_sid) == []


def test_transition_batch_validates_all_results_before_any_write():
    first_sid = _session()
    second_sid = _session()
    valid = reduce_candidate(_candidate(first_sid), None, now=NOW)
    invalid = TransitionResult(
        valid.state_item,
        replace(valid.audit, transition_id="bad-batch", session_id=second_sid),
    )

    with pytest.raises(ValueError, match="different sessions"):
        context_store.apply_transitions((valid, invalid))
    assert context_store.list_state_items(first_sid) == []
    assert context_store.list_transition_audits(first_sid) == []


def test_transition_id_collision_is_detected_not_silently_ignored():
    sid = _session()
    result = reduce_candidate(_candidate(sid), None, now=NOW)
    context_store.apply_transition(result)
    collision = replace(result.audit, reason_code=ReasonCode.APPLIED_UPDATE)
    with pytest.raises(ValueError, match="transition id collision"):
        context_store.apply_transition(TransitionResult(result.state_item, collision))


def test_list_filters_preserve_owner_boundaries():
    sid = _session()
    candidates = [
        _candidate(sid),
        _candidate(
            sid,
            candidate_id="candidate-2",
            domain=SessionDomain.TASK,
            state_type="progress_delta",
            key="progress-1",
            proposed_value="SC0 done",
            operation=Operation.APPEND,
        ),
        _candidate(
            sid,
            candidate_id="candidate-3",
            domain=SessionDomain.INTERACTION,
            state_type="user_correction",
            key="correction-1",
            proposed_value="current directory",
            operation=Operation.APPEND,
        ),
    ]
    for candidate in candidates:
        context_store.apply_transition(reduce_candidate(candidate, None, now=NOW))

    user_items = context_store.list_state_items(sid, domain=SessionDomain.USER)
    assert [item.domain for item in user_items] == [SessionDomain.USER]


def test_purge_removes_views_and_audits_for_only_target_session():
    first_sid = _session()
    second_sid = _session()
    for sid in (first_sid, second_sid):
        context_store.apply_transition(reduce_candidate(_candidate(sid), None, now=NOW))

    assert ss.purge_session(first_sid)
    assert context_store.list_state_items(first_sid) == []
    assert context_store.list_transition_audits(first_sid) == []
    assert len(context_store.list_state_items(second_sid)) == 1
