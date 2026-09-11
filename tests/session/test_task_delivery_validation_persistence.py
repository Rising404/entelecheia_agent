from __future__ import annotations

import pytest

from personagraph.l2.auxiliary_execution.delivery.composition import (
    AuxiliaryTaskDeliveryStatus,
)
from personagraph.l2.task_execution.task_graph.controller import (
    TaskGraphWorkRunRequest,
    run_task_graph_work_runs,
)
from personagraph.session import store
from personagraph.session.l2_store import task_delivery as task_delivery_store
from personagraph.session.l2_store import task_graph as task_graph_store  # noqa: F401
from personagraph.session.l2_store import work_run as work_run_store
from tests.helpers.current_auxiliary_delivery import (
    settle_current_auxiliary_candidate,
)
from tests.runtime.test_auxiliary_task_delivery_composition import (
    _commit_task_graph,
)


def _completed_without_candidate(monkeypatch: pytest.MonkeyPatch):
    session_id, turn_id, task_id = _commit_task_graph(monkeypatch)
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    executed = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            expected_window_revision=int(window["state_version"]),
        ),
        monotonic_clock=iter(range(1, 500)).__next__,
        emit=lambda _event: None,
    )
    assert executed.status == "completed"
    delivery_id = work_run_store.get_completed_task_final_delivery_id(
        session_id=session_id,
        task_id=task_id,
    )
    return session_id, turn_id, task_id, delivery_id


def test_root_publication_requires_current_candidate_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, _task_id, delivery_id = _completed_without_candidate(
        monkeypatch
    )
    window = store.get_turn_execution_window(session_id)
    assert window is not None

    with pytest.raises(
        store.TurnExecutionFinalizationConflict,
        match="whole-Task PASS authority",
    ):
        store.finalize_verified_turn_execution(
            session_id=session_id,
            turn_id=turn_id,
            delivery_id=delivery_id,
            expected_window_revision=int(window["state_version"]),
            post_commit_job_kinds=(),
        )

    inspected = store.inspect_turn_execution(session_id)
    assert inspected["turn"]["status"] == "running"
    assert inspected["window"]["state_version"] == window["state_version"]


def test_candidate_pass_publication_is_atomic_and_replayable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, _task_id, result = settle_current_auxiliary_candidate(
        monkeypatch,
        route="pass",
    )
    assert result.status is AuxiliaryTaskDeliveryStatus.DELIVERY_READY
    assert result.final_delivery_id is not None
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    command = {
        "session_id": session_id,
        "turn_id": turn_id,
        "delivery_id": result.final_delivery_id,
        "expected_window_revision": int(window["state_version"]),
        "post_commit_job_kinds": (),
    }

    finalized = store.finalize_verified_turn_execution(**command)
    assert finalized["replayed"] is False
    assert finalized["node_delivery_ids"] == (result.final_delivery_id,)
    replayed = store.finalize_verified_turn_execution(**command)
    assert replayed["replayed"] is True
    assert replayed["node_delivery_ids"] == (result.final_delivery_id,)


def test_candidate_pass_publication_rejects_task_state_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id, result = settle_current_auxiliary_candidate(
        monkeypatch,
        route="pass",
    )
    assert result.final_delivery_id is not None
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_tasks SET state_version=state_version+1 "
            "WHERE session_id=? AND insession_task_id=?",
            (session_id, task_id),
        )
        conn.commit()
    window = store.get_turn_execution_window(session_id)
    assert window is not None

    with pytest.raises(
        store.TurnExecutionFinalizationConflict,
        match="whole-Task PASS authority",
    ):
        store.finalize_verified_turn_execution(
            session_id=session_id,
            turn_id=turn_id,
            delivery_id=result.final_delivery_id,
            expected_window_revision=int(window["state_version"]),
            post_commit_job_kinds=(),
        )


def test_root_body_cannot_bypass_reference_only_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, _task_id, _delivery_id = _completed_without_candidate(
        monkeypatch
    )
    window = store.get_turn_execution_window(session_id)
    assert window is not None

    with pytest.raises(
        store.TurnExecutionFinalizationConflict,
        match="reference-only PASS publication",
    ):
        store.finalize_turn_execution(
            session_id=session_id,
            turn_id=turn_id,
            expected_window_revision=int(window["state_version"]),
            processing_level="L2",
            assistant_content="copied body must not bypass candidate authority",
            post_commit_job_kinds=(),
        )


def test_candidate_trigger_reauthenticates_its_settlement_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, _turn_id, task_id, result = settle_current_auxiliary_candidate(
        monkeypatch,
        route="replan_task_graph",
    )
    assert result.status is AuxiliaryTaskDeliveryStatus.REVISION_REQUIRED
    trigger = task_delivery_store.get_active_task_graph_revision_trigger(
        session_id=session_id,
        task_id=task_id,
    )
    assert trigger is not None
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_task_delivery_validation_settlements "
            "SET settlement_json='{}' WHERE settlement_id=?",
            (trigger.settlement_id,),
        )
        conn.commit()

    with pytest.raises(
        task_delivery_store.TaskDeliveryValidationStoredAuthorityCorrupt
    ):
        task_delivery_store.get_active_task_graph_revision_trigger(
            session_id=session_id,
            task_id=task_id,
        )


def test_candidate_trigger_reauthenticates_its_root_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, _turn_id, task_id, result = settle_current_auxiliary_candidate(
        monkeypatch,
        route="replan_task_graph",
    )
    assert result.status is AuxiliaryTaskDeliveryStatus.REVISION_REQUIRED
    trigger = task_delivery_store.get_active_task_graph_revision_trigger(
        session_id=session_id,
        task_id=task_id,
    )
    assert trigger is not None
    with store._connect() as conn:
        delivery = conn.execute(
            "SELECT work_run_id, output_revision "
            "FROM insession_task_node_deliveries WHERE delivery_id=?",
            (trigger.root_delivery_id,),
        ).fetchone()
        assert delivery is not None
        conn.execute(
            "UPDATE insession_work_run_output_windows SET frozen_at=NULL "
            "WHERE work_run_id=? AND output_revision=?",
            (str(delivery["work_run_id"]), int(delivery["output_revision"])),
        )
        conn.commit()

    with pytest.raises(
        task_delivery_store.TaskDeliveryValidationStoredAuthorityCorrupt
    ):
        task_delivery_store.get_active_task_graph_revision_trigger(
            session_id=session_id,
            task_id=task_id,
        )


def test_legacy_delivery_settlement_api_is_absent() -> None:
    for name in (
        "SettleTaskDeliveryValidationCommandV1",
        "StoredTaskDeliveryValidationV1",
        "commit_task_delivery_validation_request",
        "get_task_delivery_validation",
        "get_task_delivery_validation_request",
        "project_task_delivery_validation_prompt",
        "settle_task_delivery_validation",
    ):
        assert not hasattr(task_delivery_store, name)
