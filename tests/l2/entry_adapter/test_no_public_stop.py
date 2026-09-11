"""权威非公开稳定投影的边界测试。"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType

import pytest

from personagraph.l2.entry_adapter import no_public_stop as policy_module
from personagraph.l2.entry_adapter.no_public_stop import (
    AuthoritativeNoPublicStopSettlementProjection,
    project_authoritative_no_public_stop_settlement,
)
from personagraph.runtime.turn.contracts import (
    AcceptedEntryTurn,
    EntryExecutionSnapshot,
)
from personagraph.runtime.turn_events import RuntimeErrorCode, RuntimeStage
from personagraph.runtime.entry import application as entry_application


def _accepted() -> AcceptedEntryTurn:
    return AcceptedEntryTurn(
        session_id="session-1",
        turn_id="turn-1",
        client_request_id="request-1",
        user_input="继续任务",
        attachment_ids=(),
        window_revision=17,
        replayed=False,
        execution_snapshot=EntryExecutionSnapshot.create(
            features={},
            post_commit_job_kinds=(),
        ),
    )


def _settled(
    *,
    stage: object = RuntimeStage.TOOL.value,
    error_code: object = RuntimeErrorCode.TOOL_COMPLETION_UNCONFIRMED.value,
    end_reason: object = "host_stopped",
    stop_kind: object = "waiting_external",
    state_version: object = 17,
    turn_id: object = "turn-1",
    window_state: object = "interrupted",
    window_stage: object = RuntimeStage.TOOL.value,
    interruption_reason: object = RuntimeErrorCode.TOOL_COMPLETION_UNCONFIRMED.value,
) -> dict[str, object]:
    return {
        "stage": stage,
        "error_code": error_code,
        "end_reason": end_reason,
        "stop_kind": stop_kind,
        "window": {
            "state_version": state_version,
            "turn_id": turn_id,
            "window_state": window_state,
            "stage": window_stage,
            "interruption_reason": interruption_reason,
        },
    }


@pytest.mark.parametrize(
    ("stage", "error_code", "end_reason", "stop_kind"),
    (
        (
            RuntimeStage.TOOL.value,
            RuntimeErrorCode.TOOL_COMPLETION_UNCONFIRMED.value,
            "host_stopped",
            "waiting_external",
        ),
        (
            RuntimeStage.RESPONSE.value,
            RuntimeErrorCode.TURN_DEADLINE_EXCEEDED.value,
            "host_stopped",
            "turn_limit_reached",
        ),
        (
            RuntimeStage.RESPONSE.value,
            RuntimeErrorCode.INTERNAL_FAILURE.value,
            "module_error",
            "work_run_failed",
        ),
    ),
)
def test_valid_authoritative_markers_project_immutable_stop_facts(
    stage: str,
    error_code: str,
    end_reason: str,
    stop_kind: str,
) -> None:
    settled = _settled(
        stage=stage,
        error_code=error_code,
        end_reason=end_reason,
        stop_kind=stop_kind,
        window_stage=stage,
        interruption_reason=error_code,
    )

    projection = project_authoritative_no_public_stop_settlement(
        settled,
        expected_turn_id="turn-1",
    )

    assert projection == AuthoritativeNoPublicStopSettlementProjection(
        window_revision=17,
        end_reason=end_reason,  # type: ignore[arg-type]
        error_code=RuntimeErrorCode(error_code),
        stage=RuntimeStage(stage),
    )
    with pytest.raises(FrozenInstanceError):
        projection.stage = RuntimeStage.INGRESS  # type: ignore[misc]


@pytest.mark.parametrize(
    "settled",
    (
        None,
        MappingProxyType(_settled()),
        _settled(error_code=RuntimeErrorCode.INGRESS_REJECTED.value),
        _settled(error_code="unknown_error"),
        _settled(end_reason="module_error"),
        _settled(stop_kind="unexpected_stop"),
        _settled(turn_id="another-turn"),
        _settled(window_state="active"),
        _settled(window_stage=RuntimeStage.RESPONSE.value),
        _settled(interruption_reason=RuntimeErrorCode.INTERNAL_FAILURE.value),
        _settled(state_version=0),
        _settled(state_version=object()),
    ),
)
def test_unknown_or_inconsistent_raw_settlement_is_not_projected(
    settled: object,
) -> None:
    assert (
        project_authoritative_no_public_stop_settlement(
            settled,
            expected_turn_id="turn-1",
        )
        is None
    )


def test_projection_preserves_existing_revision_coercion_and_narrow_catch() -> None:
    string_revision = project_authoritative_no_public_stop_settlement(
        _settled(state_version="18"),
        expected_turn_id="turn-1",
    )
    boolean_revision = project_authoritative_no_public_stop_settlement(
        _settled(state_version=True),
        expected_turn_id="turn-1",
    )

    assert string_revision is not None and string_revision.window_revision == 18
    assert boolean_revision is not None and boolean_revision.window_revision == 1
    with pytest.raises(OverflowError):
        project_authoritative_no_public_stop_settlement(
            _settled(state_version=float("inf")),
            expected_turn_id="turn-1",
        )


def test_entry_retains_public_result_construction_and_related_references() -> None:
    result = entry_application._entry_turn_from_authoritative_no_public_stop(
        accepted=_accepted(),
        settled=_settled(),
        related_insession_task_ids=("task-1",),
        work_run_ids=("work-run-1",),
    )

    assert result is not None
    assert result.session_id == "session-1"
    assert result.turn_id == "turn-1"
    assert result.status == "incomplete"
    assert result.processing_level == "L2"
    assert result.end_reason == "host_stopped"
    assert result.error_code == RuntimeErrorCode.TOOL_COMPLETION_UNCONFIRMED.value
    assert result.related_insession_task_ids == ("task-1",)
    assert result.work_run_ids == ("work-run-1",)
    assert result.window_state == "interrupted"
    assert result.window_revision == 17


def test_policy_has_no_entry_store_controller_or_lifecycle_effects() -> None:
    source = Path(policy_module.__file__).read_text(encoding="utf-8")
    module = ast.parse(source)
    relative_imports = {
        node.module
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.level in {1, 2}
    }
    imported_names = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    called_names = {
        node.func.id
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert relative_imports == set()
    assert "personagraph.runtime.entry" not in source
    assert not {"EntryStorePort", "EntryTurnResult"} & imported_names
    assert not {
        "_finalize_formal_reply",
        "_incomplete_turn",
        "new_turn_event",
    } & called_names
    assert not {
        "finalize_authoritative_referenced_turn_execution",
        "mark_authoritative_no_public_turn_stop",
        "advance_turn_execution_window",
        "append_runtime_turn_event",
    } & called_attributes

    entry_module = ast.parse(
        Path(entry_application.__file__).read_text(encoding="utf-8")
    )
    adapter = next(
        node
        for node in entry_module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_entry_turn_from_authoritative_no_public_stop"
    )
    adapter_calls = {
        node.func.id
        for node in ast.walk(adapter)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "project_authoritative_no_public_stop_settlement" in adapter_calls
    assert "EntryTurnResult" in adapter_calls
