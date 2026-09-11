from dataclasses import replace

import pytest

from personagraph.session.context.models import (
    ObservationCandidate,
    Operation,
    ReasonCode,
    SessionDomain,
    SourceKind,
    StateStatus,
    TransitionDecision,
)
from personagraph.session.context.reducer import REDUCER_VERSION, expire_state_item, reduce_candidate


NOW = "2026-07-12T10:00:00+00:00"
LATER = "2026-07-12T11:00:00+00:00"


def _candidate(**overrides):
    values = {
        "candidate_id": "candidate-1",
        "session_id": "session-a",
        "domain": SessionDomain.USER,
        "state_type": "temporary_preference",
        "key": "response_depth",
        "proposed_value": "detailed",
        "operation": Operation.SET,
        "source_kind": SourceKind.EXPLICIT,
        "derived_from": ("turn:session-a:4",),
        "extractor_version": "fixture-v1",
        "confidence_hint": 1.0,
    }
    values.update(overrides)
    return ObservationCandidate(**values)


def test_explicit_set_creates_evidence_backed_active_item():
    result = reduce_candidate(_candidate(), None, now=NOW)
    assert result.audit.decision == TransitionDecision.APPLIED
    assert result.audit.reason_code == ReasonCode.APPLIED_NEW
    assert result.state_item.status == StateStatus.ACTIVE
    assert result.state_item.value_json == "detailed"
    assert result.state_item.derived_from == ("turn:session-a:4",)
    assert result.state_item.reducer_version == REDUCER_VERSION


def test_replaying_same_candidate_is_idempotent_and_ids_are_stable():
    first = reduce_candidate(_candidate(), None, now=NOW)
    second = reduce_candidate(_candidate(), first.state_item, now=LATER)
    assert second.state_item == first.state_item
    assert second.audit.reason_code == ReasonCode.DUPLICATE_NOOP
    assert second.audit.transition_id == reduce_candidate(
        _candidate(), first.state_item, now=LATER
    ).audit.transition_id


def test_new_explicit_evidence_updates_singleton_and_preserves_provenance():
    first = reduce_candidate(_candidate(), None, now=NOW)
    update = _candidate(
        candidate_id="candidate-2",
        proposed_value="concise",
        derived_from=("turn:session-a:8",),
    )
    second = reduce_candidate(update, first.state_item, now=LATER)
    assert second.audit.reason_code == ReasonCode.APPLIED_UPDATE
    assert second.state_item.value_json == "concise"
    assert second.state_item.derived_from == (
        "turn:session-a:4", "turn:session-a:8"
    )


def test_inferred_candidate_cannot_silently_override_explicit_item():
    first = reduce_candidate(_candidate(), None, now=NOW)
    inferred = _candidate(
        candidate_id="candidate-2",
        proposed_value="concise",
        source_kind=SourceKind.INFERRED,
        confidence_hint=0.99,
    )
    result = reduce_candidate(inferred, first.state_item, now=LATER)
    assert result.state_item == first.state_item
    assert result.audit.decision == TransitionDecision.REJECTED
    assert result.audit.reason_code == ReasonCode.SOURCE_REQUIRES_VERIFICATION


def test_high_confidence_affect_inference_applies_but_low_confidence_does_not():
    inferred = _candidate(
        domain=SessionDomain.USER,
        state_type="affect_signal",
        key="confusion",
        proposed_value={"signal": "confusion", "intensity": "low"},
        source_kind=SourceKind.INFERRED,
        confidence_hint=0.86,
    )
    accepted = reduce_candidate(inferred, None, now=NOW)
    rejected = reduce_candidate(
        replace(inferred, candidate_id="candidate-2", confidence_hint=0.84),
        None,
        now=NOW,
    )
    assert accepted.audit.reason_code == ReasonCode.APPLIED_NEW
    assert rejected.state_item is None
    assert rejected.audit.reason_code == ReasonCode.CONFIDENCE_BELOW_AUTO_APPLY


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (_candidate(derived_from=()), ReasonCode.MISSING_EVIDENCE),
        (_candidate(confidence_hint=1.1), ReasonCode.INVALID_CONFIDENCE),
        (_candidate(key=" "), ReasonCode.INVALID_KEY),
        (_candidate(proposed_value=object()), ReasonCode.INVALID_VALUE),
        (_candidate(source_kind=SourceKind.TOOL), ReasonCode.SOURCE_NOT_ALLOWED),
        (_candidate(state_type="unknown"), ReasonCode.INVALID_TYPE),
    ],
)
def test_invalid_or_unsupported_candidates_fail_closed(candidate, reason):
    result = reduce_candidate(candidate, None, now=NOW)
    assert result.state_item is None
    assert result.audit.decision == TransitionDecision.REJECTED
    assert result.audit.reason_code == reason


def test_event_append_is_idempotent_but_key_collision_is_conflicted():
    progress = _candidate(
        domain=SessionDomain.TASK,
        state_type="progress_delta",
        key="progress-1",
        proposed_value="完成类型目录",
        operation=Operation.APPEND,
    )
    first = reduce_candidate(progress, None, now=NOW)
    duplicate = reduce_candidate(progress, first.state_item, now=LATER)
    collision = reduce_candidate(
        replace(progress, candidate_id="candidate-2", proposed_value="完成 Graph 接入"),
        first.state_item,
        now=LATER,
    )
    assert first.audit.reason_code == ReasonCode.APPLIED_APPEND
    assert duplicate.audit.reason_code == ReasonCode.DUPLICATE_NOOP
    assert collision.state_item == first.state_item
    assert collision.audit.decision == TransitionDecision.CONFLICTED
    assert collision.audit.reason_code == ReasonCode.APPEND_KEY_CONFLICT


def test_resolve_requires_target_and_retracts_existing_event():
    question = _candidate(
        domain=SessionDomain.TASK,
        state_type="open_question",
        key="q1",
        proposed_value="是否引入 verifier",
        operation=Operation.APPEND,
    )
    resolve = replace(
        question,
        candidate_id="candidate-2",
        proposed_value=None,
        operation=Operation.RESOLVE,
        derived_from=("turn:session-a:9",),
    )
    missing = reduce_candidate(resolve, None, now=NOW)
    active = reduce_candidate(question, None, now=NOW)
    resolved = reduce_candidate(resolve, active.state_item, now=LATER)
    assert missing.audit.reason_code == ReasonCode.TARGET_MISSING
    assert resolved.state_item.status == StateStatus.RETRACTED
    assert resolved.audit.reason_code == ReasonCode.RESOLVED


def test_scope_mismatch_rejects_cross_session_or_cross_slot_update():
    first = reduce_candidate(_candidate(), None, now=NOW)
    other_session = _candidate(session_id="session-b", derived_from=("turn:session-b:1",))
    result = reduce_candidate(other_session, first.state_item, now=LATER)
    assert result.state_item == first.state_item
    assert result.audit.reason_code == ReasonCode.SCOPE_MISMATCH


def test_touch_merges_new_evidence_without_changing_value():
    first = reduce_candidate(_candidate(), None, now=NOW)
    touch = _candidate(
        candidate_id="candidate-2",
        proposed_value=None,
        operation=Operation.TOUCH,
        derived_from=("turn:session-a:10",),
    )
    result = reduce_candidate(touch, first.state_item, now=LATER)
    assert result.state_item.value_json == "detailed"
    assert result.state_item.updated_at == LATER
    assert result.state_item.derived_from == (
        "turn:session-a:4", "turn:session-a:10"
    )
    assert result.audit.reason_code == ReasonCode.TOUCHED


def test_expiry_is_deterministic_and_does_not_mutate_non_due_items():
    candidate = _candidate(expires_at="2026-07-12T10:30:00+00:00")
    item = reduce_candidate(candidate, None, now=NOW).state_item
    assert expire_state_item(item, now="2026-07-12T10:29:59+00:00") == item
    expired = expire_state_item(item, now=LATER)
    assert expired.status == StateStatus.EXPIRED
    assert expired.updated_at == LATER


def test_naive_timestamp_is_rejected_at_boundary():
    with pytest.raises(ValueError, match="UTC offset"):
        reduce_candidate(_candidate(), None, now="2026-07-12T10:00:00")
