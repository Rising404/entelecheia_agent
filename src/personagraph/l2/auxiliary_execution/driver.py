"""确定性的一步归约器用于 AuxiliaryGraph 驱动器。

Store 投影拥有真相，而本模块负责选择。它不进行 I/O 操作，也从不询问模型哪个节点应该运行。每次调用都返回一个类型化的动作：恢复唯一的持久化游标、在显式门处等待、调度第一个源序就绪节点、请求修订或达成共识。
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

from personagraph.l2.auxiliary_graph import (
    AuxiliaryNodeExecutorKind,
    AuxiliaryNodeReference,
    PlanningEpisodeBudgetDisposition,
)
from personagraph.l2.auxiliary_graph.execution_frontier_contracts import (
    AuxiliaryGraphExecutionFrontier,
    RecoverableAuxiliaryNodeExecutionCandidate,
)
from personagraph.l2.work_run import AuxiliaryNodeSubject, WorkRunStatus


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AuxiliaryGraphDriverAction(StrEnum):
    RUN_HOST_PRIMITIVE = "run_host_primitive"
    RUN_MODEL_WORK_RUN = "run_model_work_run"
    OPEN_USER_GATE = "open_user_gate"
    RUN_TERMINAL_PLANNER = "run_terminal_planner"
    RESUME_HOST_PRIMITIVE = "resume_host_primitive"
    RESUME_WORK_RUN = "resume_work_run"
    WAIT_USER = "wait_user"
    WAIT_AUTHORIZATION = "wait_authorization"
    WAIT_EXTERNAL = "wait_external"
    STOP_TURN = "stop_turn"
    REQUEST_REVISION = "request_revision"
    SEAL_REVISION = "seal_revision"
    COMMIT_READY_PROPOSAL = "commit_ready_proposal"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TERMINAL_STOP = "terminal_stop"
    FAILED_CLOSED = "failed_closed"


class AuxiliaryGraphDriverDecision(_Contract):
    action: AuxiliaryGraphDriverAction
    subject: AuxiliaryNodeSubject | None = None
    work_run_id: str | None = None
    primitive_call_id: str | None = None
    reason_code: str
    ready_node_refs: tuple[AuxiliaryNodeReference, ...] = ()

    @model_validator(mode="after")
    def _validate_decision(self) -> "AuxiliaryGraphDriverDecision":
        fresh_dispatch = self.action in {
            AuxiliaryGraphDriverAction.RUN_HOST_PRIMITIVE,
            AuxiliaryGraphDriverAction.RUN_MODEL_WORK_RUN,
            AuxiliaryGraphDriverAction.OPEN_USER_GATE,
            AuxiliaryGraphDriverAction.RUN_TERMINAL_PLANNER,
        }
        if fresh_dispatch:
            if (
                self.subject is None
                or self.work_run_id is not None
                or self.primitive_call_id is not None
            ):
                raise ValueError("fresh dispatch carries exactly one node subject")
        elif self.action is AuxiliaryGraphDriverAction.RESUME_WORK_RUN:
            if (
                self.subject is None
                or self.work_run_id is None
                or self.primitive_call_id is not None
            ):
                raise ValueError(
                    "WorkRun recovery requires its exact subject and run ID"
                )
        elif self.action is AuxiliaryGraphDriverAction.RESUME_HOST_PRIMITIVE:
            if (
                self.subject is None
                or self.primitive_call_id is None
                or self.work_run_id is not None
            ):
                raise ValueError(
                    "primitive recovery requires its exact subject and call ID"
                )
        elif self.action in {
            AuxiliaryGraphDriverAction.WAIT_USER,
            AuxiliaryGraphDriverAction.WAIT_AUTHORIZATION,
            AuxiliaryGraphDriverAction.WAIT_EXTERNAL,
            AuxiliaryGraphDriverAction.STOP_TURN,
        }:
            if self.primitive_call_id is not None or (
                (self.subject is None) != (self.work_run_id is None)
            ):
                raise ValueError("waiting WorkRun cursor identity is malformed")
        elif any(
            value is not None
            for value in (
                self.subject,
                self.work_run_id,
                self.primitive_call_id,
            )
        ):
            raise ValueError("non-dispatch decisions cannot carry cursor identity")
        if not self.reason_code or len(self.reason_code) > 160:
            raise ValueError("driver decisions require a bounded reason code")
        return self


def decide_auxiliary_graph_driver_step(
    frontier: AuxiliaryGraphExecutionFrontier,
) -> AuxiliaryGraphDriverDecision:
    """从 Store 认证的前沿中推导出恰好一个后续动作。"""

    if not isinstance(frontier, AuxiliaryGraphExecutionFrontier):
        raise TypeError("frontier must be an AuxiliaryGraphExecutionFrontier")
    ready_refs = tuple(
        AuxiliaryNodeReference(
            node_id=item.subject.node_id,
            node_revision=item.subject.node_revision,
        )
        for item in frontier.ready_fresh
    )

    if frontier.recoverable_primitive:
        candidate = frontier.recoverable_primitive[0]
        if (
            frontier.budget_disposition
            is PlanningEpisodeBudgetDisposition.HARD_LIMIT_REACHED
        ):
            return _decision(
                AuxiliaryGraphDriverAction.BUDGET_EXHAUSTED,
                "episode_hard_limit_blocks_primitive_recovery",
            )
        return AuxiliaryGraphDriverDecision(
            action=AuxiliaryGraphDriverAction.RESUME_HOST_PRIMITIVE,
            subject=candidate.subject,
            primitive_call_id=candidate.primitive_call_id,
            reason_code="reserved_host_primitive_requires_settlement",
            ready_node_refs=ready_refs,
        )

    if frontier.recoverable:
        candidate = frontier.recoverable[0]
        if (
            frontier.budget_disposition
            is PlanningEpisodeBudgetDisposition.HARD_LIMIT_REACHED
            and candidate.work_run_status
            in {WorkRunStatus.ACTIVE, WorkRunStatus.INTERRUPTED}
        ):
            return _decision(
                AuxiliaryGraphDriverAction.BUDGET_EXHAUSTED,
                "episode_hard_limit_blocks_work_run_dispatch",
            )
        return _recoverable_decision(candidate, ready_refs=ready_refs)

    goal_status = frontier.goal_status.value
    if goal_status == "waiting_user":
        return _decision(AuxiliaryGraphDriverAction.WAIT_USER, "goal_waiting_user")
    if goal_status == "waiting_authorization":
        return _decision(
            AuxiliaryGraphDriverAction.WAIT_AUTHORIZATION,
            "goal_waiting_authorization",
        )
    if goal_status == "waiting_external":
        return _decision(
            AuxiliaryGraphDriverAction.WAIT_EXTERNAL,
            "goal_waiting_external",
        )
    if goal_status in {"proposal_ready", "gapped_ready"}:
        return _decision(
            AuxiliaryGraphDriverAction.COMMIT_READY_PROPOSAL,
            f"goal_{goal_status}",
        )
    if goal_status == "budget_exhausted":
        return _decision(
            AuxiliaryGraphDriverAction.BUDGET_EXHAUSTED,
            "goal_budget_exhausted",
        )
    if goal_status in {"committed", "failed", "cancelled", "superseded"}:
        return _decision(
            AuxiliaryGraphDriverAction.TERMINAL_STOP,
            f"goal_{goal_status}",
        )
    if goal_status == "interrupted":
        return _decision(
            AuxiliaryGraphDriverAction.STOP_TURN,
            "goal_interrupted",
        )

    revision_status = frontier.revision_status
    if revision_status in {"waiting_user", "waiting_authorization", "waiting_external"}:
        action = {
            "waiting_user": AuxiliaryGraphDriverAction.WAIT_USER,
            "waiting_authorization": (
                AuxiliaryGraphDriverAction.WAIT_AUTHORIZATION
            ),
            "waiting_external": AuxiliaryGraphDriverAction.WAIT_EXTERNAL,
        }[revision_status]
        return _decision(action, f"revision_{revision_status}")
    if revision_status in {"proposal_ready", "gapped_ready"}:
        return _decision(
            AuxiliaryGraphDriverAction.COMMIT_READY_PROPOSAL,
            f"revision_{revision_status}",
        )
    if revision_status == "budget_exhausted":
        return _decision(
            AuxiliaryGraphDriverAction.BUDGET_EXHAUSTED,
            "revision_budget_exhausted",
        )
    if revision_status in {"committed", "failed", "cancelled", "superseded"}:
        return _decision(
            AuxiliaryGraphDriverAction.TERMINAL_STOP,
            f"revision_{revision_status}",
        )
    if revision_status == "interrupted":
        return _decision(
            AuxiliaryGraphDriverAction.STOP_TURN,
            "revision_interrupted",
        )

    if (
        frontier.budget_disposition
        is PlanningEpisodeBudgetDisposition.HARD_LIMIT_REACHED
    ):
        return _decision(
            AuxiliaryGraphDriverAction.BUDGET_EXHAUSTED,
            "episode_hard_limit_reached",
        )
    if _is_initial_planning_bootstrap(frontier):
        # 修订 1 的存在只为让 Architect 获得持久化的目标、权威状态和预算绑定。
        # 它不是真正的终端规划任务，绝不能跨越 WorkRun 分发边界；
        # 即使 Entry 或并发 Driver 在两次提交之间观察到该图，
        # 这一约束也不例外。
        return AuxiliaryGraphDriverDecision(
            action=AuxiliaryGraphDriverAction.REQUEST_REVISION,
            reason_code="initial_planning_bootstrap_requires_architect",
            ready_node_refs=ready_refs,
        )
    if frontier.blocking_node_refs:
        return AuxiliaryGraphDriverDecision(
            action=AuxiliaryGraphDriverAction.REQUEST_REVISION,
            reason_code="current_revision_has_blocking_nodes",
            ready_node_refs=ready_refs,
        )
    if any(
        item.node_id == frontier.terminal_node_id
        for item in frontier.completed_node_refs
    ):
        return _decision(
            AuxiliaryGraphDriverAction.SEAL_REVISION,
            "terminal_node_completed",
        )

    selected = frontier.ready_fresh[0] if frontier.ready_fresh else None
    if (
        frontier.budget_disposition
        is PlanningEpisodeBudgetDisposition.SOFT_LIMIT_REACHED
    ):
        selected = next(
            (
                item
                for item in frontier.ready_fresh
                if item.executor_kind
                in {
                    AuxiliaryNodeExecutorKind.USER_GATE,
                    AuxiliaryNodeExecutorKind.TERMINAL_PLANNER,
                }
            ),
            None,
        )
        if selected is None:
            return AuxiliaryGraphDriverDecision(
                action=AuxiliaryGraphDriverAction.REQUEST_REVISION,
                reason_code="episode_soft_limit_requires_convergence",
                ready_node_refs=ready_refs,
            )
    if selected is None:
        return _decision(
            AuxiliaryGraphDriverAction.FAILED_CLOSED,
            "active_revision_has_no_ready_or_waiting_authority",
        )
    action = {
        AuxiliaryNodeExecutorKind.HOST_PRIMITIVE: (
            AuxiliaryGraphDriverAction.RUN_HOST_PRIMITIVE
        ),
        AuxiliaryNodeExecutorKind.MODEL_WORK_RUN: (
            AuxiliaryGraphDriverAction.RUN_MODEL_WORK_RUN
        ),
        AuxiliaryNodeExecutorKind.USER_GATE: (
            AuxiliaryGraphDriverAction.OPEN_USER_GATE
        ),
        AuxiliaryNodeExecutorKind.TERMINAL_PLANNER: (
            AuxiliaryGraphDriverAction.RUN_TERMINAL_PLANNER
        ),
    }[selected.executor_kind]
    return AuxiliaryGraphDriverDecision(
        action=action,
        subject=selected.subject,
        reason_code=f"ready_{selected.executor_kind.value}",
        ready_node_refs=ready_refs,
    )


def _is_initial_planning_bootstrap(
    frontier: AuxiliaryGraphExecutionFrontier,
) -> bool:
    if frontier.auxiliary_graph_revision != 1 or len(frontier.ready_fresh) != 1:
        return False
    candidate = frontier.ready_fresh[0]
    return (
        candidate.local_node_key == "bootstrap_terminal"
        and candidate.subject.node_id == frontier.terminal_node_id
        and candidate.executor_kind
        is AuxiliaryNodeExecutorKind.TERMINAL_PLANNER
        and candidate.capability_profile_id is None
    )


def canonical_auxiliary_graph_driver_state_guard(
    frontier: AuxiliaryGraphExecutionFrontier,
) -> str:
    """将外部动作绑定到完整的 Store-认证边界."""

    if not isinstance(frontier, AuxiliaryGraphExecutionFrontier):
        raise TypeError("frontier must be an AuxiliaryGraphExecutionFrontier")
    payload = {
        "contract": "auxiliary-graph-driver-state-guard-v1",
        "frontier": frontier.model_dump(mode="json"),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _recoverable_decision(
    candidate: RecoverableAuxiliaryNodeExecutionCandidate,
    *,
    ready_refs: tuple[AuxiliaryNodeReference, ...],
) -> AuxiliaryGraphDriverDecision:
    status = candidate.work_run_status
    if status in {WorkRunStatus.ACTIVE, WorkRunStatus.INTERRUPTED}:
        action = AuxiliaryGraphDriverAction.RESUME_WORK_RUN
    elif status is WorkRunStatus.WAITING_USER:
        action = AuxiliaryGraphDriverAction.WAIT_USER
    elif status is WorkRunStatus.WAITING_AUTHORIZATION:
        action = AuxiliaryGraphDriverAction.WAIT_AUTHORIZATION
    elif status is WorkRunStatus.WAITING_EXTERNAL:
        action = AuxiliaryGraphDriverAction.WAIT_EXTERNAL
    elif status in {WorkRunStatus.PAUSED, WorkRunStatus.TURN_LIMIT_REACHED}:
        action = AuxiliaryGraphDriverAction.STOP_TURN
    else:
        return _decision(
            AuxiliaryGraphDriverAction.FAILED_CLOSED,
            "frontier_contains_terminal_work_run",
        )
    return AuxiliaryGraphDriverDecision(
        action=action,
        subject=candidate.subject,
        work_run_id=candidate.work_run_id,
        reason_code=f"work_run_{status.value}",
        ready_node_refs=ready_refs,
    )


def _decision(
    action: AuxiliaryGraphDriverAction,
    reason_code: str,
) -> AuxiliaryGraphDriverDecision:
    return AuxiliaryGraphDriverDecision(action=action, reason_code=reason_code)


__all__ = [
    "AuxiliaryGraphDriverAction",
    "AuxiliaryGraphDriverDecision",
    "canonical_auxiliary_graph_driver_state_guard",
    "decide_auxiliary_graph_driver_step",
]
