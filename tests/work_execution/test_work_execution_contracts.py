"""隔离式 WorkRun V2 执行领域的契约测试。"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from personagraph.l2.work_run import (
    AcceptanceProgressItem,
    AcceptanceUpdate,
    AttemptDecision,
    AttemptStatus,
    Attempt,
    AuxiliaryNodeSubject,
    CallToolsAction,
    HostAcceptedAttemptDecision,
    HostMaterializedCallToolsAction,
    HostMaterializedSubmitOutputWindowAction,
    HostMaterializedToolCall,
    OutputWindowFormat,
    RequestUserInputAction,
    SubmitOutputWindowAction,
    SupportingToolResult,
    TaskNodeSubject,
    ToolResultStatus,
    ToolResult,
    ToolCallProposal,
    WorkRunBudget,
    WorkRunBudgetDisposition,
    WorkRunBudgetTransition,
    WorkRunStatus,
    WorkRun,
    WriteOutputWindowAction,
    create_work_run,
    charge_work_run_active_seconds,
)


def _task_subject() -> TaskNodeSubject:
    return TaskNodeSubject(
        task_id="task-1",
        graph_revision=2,
        node_id="node-1",
        node_revision=3,
    )


def _proposal(index: int = 1) -> ToolCallProposal:
    return ToolCallProposal(
        tool_id="workspace.read",
        arguments={"path": f"/workspace/readme-{index}.md", "options": ["safe"]},
    )


def _materialized_call(
    call_id: str, *, modifies_environment: bool = False
) -> HostMaterializedToolCall:
    return HostMaterializedToolCall(
        tool_call_id=call_id,
        tool_id="workspace.read",
        tool_version="v2",
        arguments={"path": "/workspace/readme.md", "options": ["safe"]},
        modifies_environment=modifies_environment,
    )


def test_work_run_statuses_are_the_frozen_v2_set() -> None:
    assert {status.value for status in WorkRunStatus} == {
        "active",
        "paused",
        "waiting_user",
        "waiting_authorization",
        "waiting_external",
        "turn_limit_reached",
        "interrupted",
        "completed",
        "failed",
        "cancelled",
    }
    assert "created" not in {status.value for status in WorkRunStatus}


def test_work_run_creation_is_active_revision_one_with_host_budget_defaults() -> None:
    run = create_work_run(work_run_id="run-1", subject=_task_subject())

    assert run.status is WorkRunStatus.ACTIVE
    assert run.reason is None
    assert run.revision == 1
    assert run.budget == WorkRunBudget(
        max_attempts=32,
        soft_active_seconds=720,
        hard_active_seconds=900,
        attempts_started=0,
        active_seconds_consumed=0,
    )

    with pytest.raises(ValidationError):
        WorkRun(
            work_run_id="run-invalid",
            subject=_task_subject(),
            revision=1,
            status="paused",
        )

    waiting = WorkRun(
        work_run_id="run-waiting",
        subject=_task_subject(),
        revision=2,
        status="waiting_user",
        reason="request_user_input",
    )
    assert waiting.reason == "request_user_input"
    with pytest.raises(ValidationError):
        WorkRun(
            work_run_id="run-empty-reason",
            subject=_task_subject(),
            revision=2,
            status="waiting_user",
            reason="",
        )


def test_budget_snapshot_rejects_invalid_host_ledger_values() -> None:
    with pytest.raises(ValidationError):
        WorkRunBudget(soft_active_seconds=901, hard_active_seconds=900)
    with pytest.raises(ValidationError):
        WorkRunBudget(max_attempts=32, attempts_started=33)
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError):
            WorkRunBudget(active_seconds_consumed=value)


def test_active_time_charge_classifies_exact_soft_and_hard_boundaries() -> None:
    initial = WorkRunBudget(active_seconds_consumed=719)
    soft = charge_work_run_active_seconds(initial, active_seconds_delta=1)
    assert soft.budget_before is initial
    assert soft.budget_after.active_seconds_consumed == 720
    assert soft.disposition is WorkRunBudgetDisposition.SOFT_LIMIT_REACHED

    hard = charge_work_run_active_seconds(
        soft.budget_after,
        active_seconds_delta=180,
    )
    assert hard.budget_after.active_seconds_consumed == 900
    assert hard.disposition is WorkRunBudgetDisposition.HARD_LIMIT_REACHED

    over_hard = charge_work_run_active_seconds(
        WorkRunBudget(active_seconds_consumed=899),
        active_seconds_delta=2,
    )
    assert over_hard.budget_after.active_seconds_consumed == 901
    assert over_hard.disposition is WorkRunBudgetDisposition.HARD_LIMIT_REACHED


@pytest.mark.parametrize(
    "delta",
    [0, -1, True, float("nan"), float("inf"), float("-inf")],
)
def test_active_time_charge_rejects_nonpositive_or_nonfinite_deltas(delta) -> None:
    with pytest.raises(ValueError):
        charge_work_run_active_seconds(
            WorkRunBudget(),
            active_seconds_delta=delta,
        )


def test_active_time_transition_rejects_forged_snapshots_and_disposition() -> None:
    before = WorkRunBudget(active_seconds_consumed=10)
    with pytest.raises(ValidationError, match="budgets are inconsistent"):
        WorkRunBudgetTransition(
            active_seconds_delta=1,
            budget_before=before,
            budget_after=WorkRunBudget(active_seconds_consumed=12),
            disposition=WorkRunBudgetDisposition.WITHIN_LIMIT,
        )
    with pytest.raises(ValidationError, match="budgets are inconsistent"):
        WorkRunBudgetTransition(
            active_seconds_delta=1,
            budget_before=before,
            budget_after=WorkRunBudget(
                attempts_started=1,
                active_seconds_consumed=11,
            ),
            disposition=WorkRunBudgetDisposition.WITHIN_LIMIT,
        )
    with pytest.raises(ValidationError, match="disposition is inconsistent"):
        WorkRunBudgetTransition(
            active_seconds_delta=1,
            budget_before=before,
            budget_after=WorkRunBudget(active_seconds_consumed=11),
            disposition=WorkRunBudgetDisposition.SOFT_LIMIT_REACHED,
        )


def test_execution_subject_is_a_discriminated_task_or_auxiliary_node() -> None:
    task_run = WorkRun.model_validate(
        {
            "work_run_id": "run-task",
            "subject": {
                "kind": "task_node",
                "task_id": "task-1",
                "graph_revision": 1,
                "node_id": "node-1",
                "node_revision": 1,
            },
        }
    )
    auxiliary_run = WorkRun(
        work_run_id="run-aux",
        subject=AuxiliaryNodeSubject(
            task_id="task-1",
            auxiliary_graph_id="aux-1",
            auxiliary_graph_revision=4,
            node_id="aux-node-1",
            node_revision=2,
        ),
    )

    assert isinstance(task_run.subject, TaskNodeSubject)
    assert isinstance(auxiliary_run.subject, AuxiliaryNodeSubject)
    with pytest.raises(ValidationError):
        WorkRun.model_validate(
            {"work_run_id": "run-bad", "subject": {"kind": "unknown"}}
        )


def test_attempt_has_only_active_and_closed_states() -> None:
    assert {status.value for status in AttemptStatus} == {"active", "closed"}
    assert Attempt(attempt_id="attempt-1", work_run_id="run-1", ordinal=1).status is (
        AttemptStatus.ACTIVE
    )
    with pytest.raises(ValidationError):
        Attempt(
            attempt_id="attempt-active-submit",
            work_run_id="run-1",
            ordinal=2,
            submitted_output_revision=3,
        )
    submitted = Attempt(
        attempt_id="attempt-submit",
        work_run_id="run-1",
        ordinal=2,
        status=AttemptStatus.CLOSED,
        submitted_output_revision=3,
    )
    assert submitted.submitted_output_revision == 3


def test_attempt_decision_has_one_discriminated_action_and_no_legacy_fields() -> None:
    decision = AttemptDecision.model_validate(
        {
            "acceptance_updates": [
                {
                    "acceptance_id": "acceptance-1",
                    "model_claimed_satisfied": True,
                    "supporting_tool_result_ids": [],
                }
            ],
            "action": {
                "kind": "submit_output_window",
                "content": "Completed result",
                "format": "plain_text",
            },
        }
    )

    assert isinstance(decision.action, SubmitOutputWindowAction)
    assert set(AcceptanceUpdate.model_fields) == {
        "acceptance_id",
        "empty_support_justification",
        "model_claimed_satisfied",
        "supporting_tool_result_ids",
    }
    with pytest.raises(ValidationError):
        AttemptDecision.model_validate(
            {
                "acceptance_updates": [],
                "action": {"kind": "cannot_progress"},
            }
        )
    with pytest.raises(ValidationError):
        AttemptDecision.model_validate(
            {
                "acceptance_updates": [],
                "action": {
                    "kind": "submit_output_window",
                    "content": "result",
                    "format": "plain_text",
                },
                "progress_note": "legacy",
            }
        )

    with pytest.raises(ValidationError):
        AttemptDecision.model_validate(
            {
                "acceptance_updates": [],
                "action": {"kind": "ready_for_verification"},
            }
        )


def test_request_user_input_is_mutually_exclusive_with_tool_calls() -> None:
    decision = AttemptDecision(
        action=RequestUserInputAction(question="Which directory should I inspect?")
    )
    assert decision.action.kind == "request_user_input"
    with pytest.raises(ValidationError):
        AttemptDecision.model_validate(
            {
                "acceptance_updates": [],
                "action": {
                    "kind": "request_user_input",
                    "question": "Continue?",
                    "calls": [],
                },
            }
        )


def test_call_tools_has_no_product_count_cap_but_only_one_modifying_call() -> None:
    many_read_calls = tuple(_proposal(index) for index in range(40))
    action = CallToolsAction(calls=many_read_calls)
    assert len(action.calls) == 40

    accepted = HostAcceptedAttemptDecision(
        action=HostMaterializedCallToolsAction(
            calls=(_materialized_call("write-1", modifies_environment=True),)
        )
    )
    assert accepted.action.calls[0].tool_call_id == "write-1"

    with pytest.raises(ValidationError):
        HostMaterializedCallToolsAction(
            calls=(
                _materialized_call("write-1", modifies_environment=True),
                _materialized_call("write-2", modifies_environment=True),
            )
        )
    with pytest.raises(ValidationError):
        HostMaterializedCallToolsAction(
            calls=(_materialized_call("same"), _materialized_call("same"))
        )


def test_materialized_call_requires_host_fields_and_freezes_arguments() -> None:
    _proposal()
    assert set(ToolCallProposal.model_fields) == {"tool_id", "arguments"}
    assert "tool_call_id" not in ToolCallProposal.model_fields
    assert "tool_version" not in ToolCallProposal.model_fields
    assert "modifies_environment" not in ToolCallProposal.model_fields

    call = _materialized_call("call-1")
    assert call.tool_version == "v2"
    with pytest.raises(TypeError):
        call.arguments["path"] = "/other"  # type: ignore[index]
    with pytest.raises(TypeError):
        call.arguments["options"][0] = "unsafe"  # type: ignore[index]
    with pytest.raises(ValidationError):
        HostMaterializedToolCall.model_validate(
            {
                "tool_id": "workspace.read",
                "tool_version": "v2",
                "arguments": {},
                "modifies_environment": False,
            }
        )


def test_tool_result_preserves_precise_status_and_is_immutable() -> None:
    for status in ToolResultStatus:
        result = ToolResult(
            status=status,
            tool_result_id=f"result-{status.value}",
            tool_call_id="call-1",
            attempt_id="attempt-1",
            ordinal=1,
            output={"rows": [1, 2]},
            error_code=None if status is ToolResultStatus.SUCCEEDED else "tool_failure",
            error_message=None if status is ToolResultStatus.SUCCEEDED else "Tool did not succeed",
        )
        assert result.status is status
        with pytest.raises(TypeError):
            result.output["rows"] = []  # type: ignore[index]

    with pytest.raises(ValidationError):
        ToolResult(
            status=ToolResultStatus.FAILED,
            tool_result_id="result-1",
            tool_call_id="call-1",
            attempt_id="attempt-1",
            ordinal=1,
            output=None,
        )
    with pytest.raises(ValidationError):
        ToolResult(
            status=ToolResultStatus.SUCCEEDED,
            tool_result_id="result-1",
            tool_call_id="call-1",
            attempt_id="attempt-1",
            ordinal=1,
            output={},
            error_code="unexpected",
            error_message="unexpected",
        )


@pytest.mark.parametrize(
    "status",
    (
        ToolResultStatus.REJECTED,
        ToolResultStatus.FAILED,
        ToolResultStatus.TIMED_OUT,
        ToolResultStatus.CANCELLED,
        ToolResultStatus.COMPLETION_UNCONFIRMED,
    ),
)
def test_only_succeeded_tool_results_can_be_projected_as_acceptance_support(
    status: ToolResultStatus,
) -> None:
    with pytest.raises(ValidationError, match="succeeded"):
        SupportingToolResult(
            work_run_id="run-1",
            tool_id="workspace.read",
            tool_version="v2",
            result=ToolResult(
                status=status,
                tool_result_id=f"result-{status.value}",
                tool_call_id="call-1",
                attempt_id="attempt-1",
                ordinal=1,
                output=None,
                error_code="tool_did_not_succeed",
                error_message="The tool did not produce successful evidence.",
            ),
        )


def test_progress_item_has_only_typed_support_declaration_fields() -> None:
    AcceptanceProgressItem(
        acceptance_id="acceptance-1",
        model_claimed_satisfied=True,
        supporting_tool_result_ids=("result-1",),
    )
    assert set(AcceptanceProgressItem.model_fields) == {
        "acceptance_id",
        "empty_support_justification",
        "model_claimed_satisfied",
        "supporting_tool_result_ids",
    }
    with pytest.raises(ValidationError):
        AcceptanceProgressItem(
            acceptance_id="acceptance-1",
            model_claimed_satisfied=False,
            supporting_tool_result_ids=("result-1",),
        )


def test_contracts_are_frozen() -> None:
    decision = AttemptDecision(
        acceptance_updates=(
            AcceptanceUpdate(
                acceptance_id="acceptance-1",
                model_claimed_satisfied=False,
            ),
        ),
        action=WriteOutputWindowAction(
            content="draft",
            format=OutputWindowFormat.PLAIN_TEXT,
        ),
    )
    with pytest.raises(ValidationError):
        decision.action = CallToolsAction(calls=(_proposal(),))


def test_host_decision_and_tool_result_deep_frozen_json_remains_serializable() -> None:
    accepted = HostAcceptedAttemptDecision(
        action=HostMaterializedCallToolsAction(
            calls=(_materialized_call("call-1"),)
        )
    )
    result = ToolResult(
        status=ToolResultStatus.SUCCEEDED,
        tool_result_id="result-1",
        tool_call_id="call-1",
        attempt_id="attempt-1",
        ordinal=1,
        output={"nested": [{"ok": True}]},
    )

    accepted_json = json.loads(accepted.model_dump_json())
    result_json = json.loads(result.model_dump_json())

    assert accepted_json["action"]["calls"][0]["arguments"]["options"] == ["safe"]
    assert result_json["output"] == {"nested": [{"ok": True}]}


def test_host_materialized_output_action_contains_only_small_revision_reference() -> None:
    accepted = HostAcceptedAttemptDecision(
        action=HostMaterializedSubmitOutputWindowAction(
            work_run_id="run-1",
            output_revision=2,
            format=OutputWindowFormat.MARKDOWN,
            size_bytes=12,
        )
    )

    assert set(type(accepted.action).model_fields) == {
        "kind",
        "work_run_id",
        "output_revision",
        "format",
        "size_bytes",
    }
    assert "content" not in accepted.model_dump_json()
