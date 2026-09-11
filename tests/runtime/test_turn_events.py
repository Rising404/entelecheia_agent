from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from personagraph.runtime.turn_events import (
    RuntimeErrorCode,
    RuntimeStage,
    TurnEventStatus,
    TurnEvent,
    new_turn_event,
    project_turn_event,
)


def _event(**changes: object) -> TurnEvent:
    values: dict[str, object] = {
        "event_id": "turnevt-1",
        "session_id": "session-1",
        "turn_id": "turn-1",
        "stage": RuntimeStage.CLASSIFY,
        "status": TurnEventStatus.COMPLETED,
        "occurred_at": datetime(2026, 8, 3, tzinfo=timezone.utc),
        "model_call_id": "model-1",
        "model_attempt": 1,
    }
    values.update(changes)
    return TurnEvent(**values)


def test_turn_event_records_closed_host_lifecycle_facts():
    event = _event()

    assert event.schema_version == 1
    assert event.stage is RuntimeStage.CLASSIFY
    assert event.status is TurnEventStatus.COMPLETED
    assert event.model_attempt == 1


def test_failure_requires_stable_error_code_and_may_be_retryable():
    failed = _event(
        status=TurnEventStatus.FAILED,
        error_code=RuntimeErrorCode.MODEL_TIMEOUT,
        retryable=True,
    )
    assert failed.error_code is RuntimeErrorCode.MODEL_TIMEOUT

    with pytest.raises(ValidationError, match="requires error_code"):
        _event(status=TurnEventStatus.FAILED)
    with pytest.raises(ValidationError, match="error_code requires"):
        _event(error_code=RuntimeErrorCode.MODEL_TIMEOUT)
    with pytest.raises(ValidationError, match="retryable"):
        _event(retryable=True)


def test_model_attempt_and_started_duration_are_constrained():
    with pytest.raises(ValidationError, match="model_attempt requires"):
        _event(model_call_id=None, model_attempt=1)
    with pytest.raises(ValidationError, match="started events"):
        _event(status=TurnEventStatus.STARTED, duration_ms=1)


def test_public_projection_drops_private_correlation_and_diagnostics():
    event = _event(
        parent_event_id="turnevt-parent",
        insession_task_id="task-1",
        work_run_id="workrun-1",
        attempt_id="attempt-1",
        operation_id="operation-1",
        diagnostic_ref="diagnostic-1",
        duration_ms=45,
    )

    public = project_turn_event(event, sequence=7).model_dump(mode="json")

    assert public["stage"] == "CLASSIFY"
    assert public["session_id"] == "session-1"
    assert public["sequence"] == 7
    assert public["prompt_replay"] is False
    assert public["retryable"] is False
    assert public["insession_task_id"] == "task-1"
    assert public["work_run_id"] == "workrun-1"
    assert public["attempt_id"] == "attempt-1"
    assert public["operation_id"] == "operation-1"
    assert "parent_event_id" not in public
    assert "model_call_id" not in public
    assert "diagnostic_ref" not in public
    assert "duration_ms" not in public
    assert "insession_task_node_id" not in public


def test_event_rejects_unstructured_diagnostic_payload_and_self_parenting():
    with pytest.raises(ValidationError):
        _event(detail="raw model response")
    with pytest.raises(ValidationError, match="must not equal"):
        _event(parent_event_id="turnevt-1")


def test_public_projection_rejects_unsafe_correlation_identifiers():
    with pytest.raises(ValidationError, match="pattern"):
        project_turn_event(_event(operation_id="provider secret payload"), sequence=1)

    with pytest.raises(ValidationError, match="greater than or equal"):
        project_turn_event(_event(), sequence=0)


def test_factory_creates_utc_event_ids_without_runtime_side_effects():
    event = new_turn_event(
        turn_id="turn-1",
        stage=RuntimeStage.INGRESS,
        status=TurnEventStatus.STARTED,
    )

    assert event.event_id.startswith("turnevt_")
    assert event.occurred_at.tzinfo is not None
