"""基于原因的控制器，用于终止一个陈旧的规划目标。

控制器推导出一个稳定的 Store 应用身份，并仅返回经过身份验证的替换回执。它从不启动替换目标。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json

from personagraph.session.l2_store import planning as planning_store


class AuxiliaryGoalSupersedeControllerStatus(StrEnum):
    SUPERSEDED = "superseded"
    ALREADY_SUPERSEDED = "already_superseded"


@dataclass(frozen=True, slots=True)
class AuxiliaryGoalSupersedeRequest:
    session_id: str
    turn_id: str
    task_id: str
    reason: planning_store.PlanningGoalSupersedeReason
    auxiliary_graph_id: str
    goal_id: str
    expected_task_state_version: int
    expected_control_state_version: int
    expected_goal_state_version: int
    expected_revision_state_version: int
    expected_budget_state_version: int
    expected_current_auxiliary_graph_revision: int
    expected_base_task_graph_revision: int | None
    observed_task_graph_revision: int | None
    replacement_objective: str | None = None
    source_start: int | None = None
    source_end: int | None = None
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "session_id",
            "turn_id",
            "task_id",
            "auxiliary_graph_id",
            "goal_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or len(value) > 200:
                raise ValueError(f"{name} must be a bounded non-empty identifier")
        if not isinstance(self.reason, planning_store.PlanningGoalSupersedeReason):
            raise TypeError("reason must be PlanningGoalSupersedeReason")
        for name in (
            "expected_task_state_version",
            "expected_control_state_version",
            "expected_goal_state_version",
            "expected_revision_state_version",
            "expected_budget_state_version",
            "expected_current_auxiliary_graph_revision",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        # 在此重用 Store 命令的条件授权验证。
        _store_command(self, apply_id="goal-supersede-contract-validation")


@dataclass(frozen=True, slots=True)
class AuxiliaryGoalSupersedeControllerResult:
    status: AuxiliaryGoalSupersedeControllerStatus
    reason_code: str
    apply_id: str
    receipt_sha256: str
    store_result: planning_store.SupersedeAuxiliaryPlanningGoalResult


class AuxiliaryGoalSupersedeControllerError(RuntimeError):
    """当前请求无法安全地替换其精确目标。"""

    code = "auxiliary_v2_goal_supersede_rejected"


def derive_auxiliary_goal_supersede_apply_id(
    request: AuxiliaryGoalSupersedeRequest,
) -> str:
    if not isinstance(request, AuxiliaryGoalSupersedeRequest):
        raise TypeError("request must be AuxiliaryGoalSupersedeRequest")
    payload = {
        "schema_version": "auxiliary-v2-goal-supersede-identity-v1",
        "session_id": request.session_id,
        "task_id": request.task_id,
        "auxiliary_graph_id": request.auxiliary_graph_id,
        "goal_id": request.goal_id,
        "reason": request.reason.value,
        "expected_current_auxiliary_graph_revision": (
            request.expected_current_auxiliary_graph_revision
        ),
        "expected_base_task_graph_revision": (
            request.expected_base_task_graph_revision
        ),
        "observed_task_graph_revision": request.observed_task_graph_revision,
        "replacement_objective": request.replacement_objective,
        "source_turn_id": (
            request.turn_id
            if request.reason
            is planning_store.PlanningGoalSupersedeReason.USER_TARGET_CHANGED
            else None
        ),
        "source_start": request.source_start,
        "source_end": request.source_end,
        "source_sha256": request.source_sha256,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"auxgoalsupersede_{digest[:40]}"


def run_auxiliary_goal_supersede(
    request: AuxiliaryGoalSupersedeRequest,
) -> AuxiliaryGoalSupersedeControllerResult:
    """提交或精确重放一次与原因绑定的规划目标取代操作。"""

    if not isinstance(request, AuxiliaryGoalSupersedeRequest):
        raise TypeError("request must be AuxiliaryGoalSupersedeRequest")
    apply_id = derive_auxiliary_goal_supersede_apply_id(request)
    try:
        result = planning_store.supersede_auxiliary_planning_goal(
            command=_store_command(request, apply_id=apply_id)
        )
    except planning_store.AuxiliaryGoalSupersedePersistenceError as exc:
        raise AuxiliaryGoalSupersedeControllerError(str(exc)) from exc
    receipt = result.receipt
    if (
        receipt.apply_id != apply_id
        or receipt.session_id != request.session_id
        or receipt.invocation_turn_id != request.turn_id
        or receipt.task_id != request.task_id
        or receipt.auxiliary_graph_id != request.auxiliary_graph_id
        or receipt.superseded_goal_id != request.goal_id
        or receipt.reason is not request.reason
        or receipt.superseded_auxiliary_graph_revision
        != request.expected_current_auxiliary_graph_revision
        or receipt.previous_base_task_graph_revision
        != request.expected_base_task_graph_revision
        or receipt.observed_task_graph_revision
        != request.observed_task_graph_revision
    ):
        raise AuxiliaryGoalSupersedeControllerError(
            "Store receipt crossed the supersede request authority"
        )
    status = (
        AuxiliaryGoalSupersedeControllerStatus.SUPERSEDED
        if result.status == "applied"
        else AuxiliaryGoalSupersedeControllerStatus.ALREADY_SUPERSEDED
    )
    return AuxiliaryGoalSupersedeControllerResult(
        status=status,
        reason_code=(
            "planning_goal_superseded"
            if status is AuxiliaryGoalSupersedeControllerStatus.SUPERSEDED
            else "planning_goal_supersede_replayed"
        ),
        apply_id=apply_id,
        receipt_sha256=receipt.receipt_sha256,
        store_result=result,
    )


def _store_command(
    request: AuxiliaryGoalSupersedeRequest,
    *,
    apply_id: str,
) -> planning_store.SupersedeAuxiliaryPlanningGoalCommand:
    target_change = (
        request.reason
        is planning_store.PlanningGoalSupersedeReason.USER_TARGET_CHANGED
    )
    return planning_store.SupersedeAuxiliaryPlanningGoalCommand(
        apply_id=apply_id,
        session_id=request.session_id,
        invocation_turn_id=request.turn_id,
        task_id=request.task_id,
        auxiliary_graph_id=request.auxiliary_graph_id,
        goal_id=request.goal_id,
        reason=request.reason,
        expected_task_state_version=request.expected_task_state_version,
        expected_control_state_version=request.expected_control_state_version,
        expected_goal_state_version=request.expected_goal_state_version,
        expected_revision_state_version=request.expected_revision_state_version,
        expected_budget_state_version=request.expected_budget_state_version,
        expected_current_auxiliary_graph_revision=(
            request.expected_current_auxiliary_graph_revision
        ),
        expected_base_task_graph_revision=(
            request.expected_base_task_graph_revision
        ),
        observed_task_graph_revision=request.observed_task_graph_revision,
        replacement_objective=request.replacement_objective,
        source_turn_id=request.turn_id if target_change else None,
        source_start=request.source_start if target_change else None,
        source_end=request.source_end if target_change else None,
        source_sha256=request.source_sha256 if target_change else None,
    )


__all__ = [
    "AuxiliaryGoalSupersedeControllerError",
    "AuxiliaryGoalSupersedeControllerResult",
    "AuxiliaryGoalSupersedeControllerStatus",
    "AuxiliaryGoalSupersedeRequest",
    "derive_auxiliary_goal_supersede_apply_id",
    "run_auxiliary_goal_supersede",
]
