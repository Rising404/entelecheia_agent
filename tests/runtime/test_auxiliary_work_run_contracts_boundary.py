"""冷态 AuxiliaryGraph WorkRun 契约的边界覆盖。"""

from __future__ import annotations

import pytest

from personagraph.l2.auxiliary_graph import AuxiliaryNodeExecutorKind
from personagraph.l2.auxiliary_execution.work_run import contracts
from personagraph.l2.work_run import AuxiliaryNodeSubject


def _subject(*, node_revision: int = 3) -> AuxiliaryNodeSubject:
    return AuxiliaryNodeSubject(
        task_id="task-1",
        auxiliary_graph_id="auxiliary-graph-1",
        auxiliary_graph_revision=2,
        node_id="node-1",
        node_revision=node_revision,
    )


def _id_plan() -> contracts.AuxiliaryWorkRunIdPlan:
    return contracts.derive_auxiliary_work_run_ids(
        session_id="session-1",
        subject=_subject(),
    )


def _request(
    *,
    executor_kind: AuxiliaryNodeExecutorKind = (
        AuxiliaryNodeExecutorKind.MODEL_WORK_RUN
    ),
) -> contracts.AuxiliaryWorkRunRequest:
    return contracts.AuxiliaryWorkRunRequest(
        session_id="session-1",
        turn_id="turn-1",
        subject=_subject(),
        executor_kind=executor_kind,
        initial_driver_state_guard_sha256="a" * 64,
        id_plan=_id_plan(),
    )


def test_stable_id_plan_is_deterministic_revision_bound_and_bounded() -> None:
    plan = _id_plan()
    assert plan == contracts.derive_auxiliary_work_run_ids(
        session_id="session-1",
        subject=_subject(),
    )
    assert plan != contracts.derive_auxiliary_work_run_ids(
        session_id="session-1",
        subject=_subject(node_revision=4),
    )
    assert plan.work_run_id.startswith("auxv2wr-")
    assert plan.for_attempt(plan.attempt_id, 1) == plan.attempt_id
    assert plan.for_attempt(plan.attempt_id, 2).endswith(":attempt-2")
    assert plan.tool_call_id(1, 1).endswith(":a1:tool-1")

    long_plan = plan.model_copy(update={"attempt_id": "a" * 200})
    assert len(long_plan.for_attempt(long_plan.attempt_id, 2)) <= 200
    with pytest.raises(ValueError, match="Attempt ordinal must be positive"):
        plan.for_attempt(plan.attempt_id, 0)


def test_request_and_result_keep_terminal_authority_guards() -> None:
    request = _request()
    assert request.id_plan == _id_plan()
    with pytest.raises(ValueError, match="terminal planner alone requires"):
        _request(executor_kind=AuxiliaryNodeExecutorKind.TERMINAL_PLANNER)

    waiting = contracts.AuxiliaryWorkRunResult(
        status=contracts.AuxiliaryWorkRunStatus.WAITING_EXTERNAL,
        reason_code="awaiting_model",
        subject=_subject(),
        executor_kind=AuxiliaryNodeExecutorKind.MODEL_WORK_RUN,
        window_state_version=2,
    )
    assert waiting.completion_id is None
    completed = contracts.AuxiliaryWorkRunResult(
        status=contracts.AuxiliaryWorkRunStatus.COMPLETED,
        reason_code="verification_passed",
        subject=_subject(),
        executor_kind=AuxiliaryNodeExecutorKind.MODEL_WORK_RUN,
        work_run_id="work-run-1",
        attempt_id="attempt-1",
        completion_id="completion-1",
        output_revision=3,
        window_state_version=4,
    )
    assert completed.output_revision == 3
    with pytest.raises(ValueError, match="only a completed result"):
        contracts.AuxiliaryWorkRunResult(
            status=contracts.AuxiliaryWorkRunStatus.WAITING_EXTERNAL,
            reason_code="awaiting_model",
            subject=_subject(),
            executor_kind=AuxiliaryNodeExecutorKind.MODEL_WORK_RUN,
            work_run_id="work-run-1",
            completion_id="completion-1",
            output_revision=3,
            window_state_version=2,
        )
