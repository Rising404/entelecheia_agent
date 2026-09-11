"""组成基于 AuxiliaryGraph 的规划/执行，带有验证过的任务交付。

两个较低层次的组合故意拥有不同的持久化状态机：Auxiliary application
通过一个提交的 TaskGraph 修订推进 AuxiliaryGraph，而 task delivery composition
执行这个精确的修订并重新加载其验证过的根 ``NodeDelivery``。产品调用者需要一个
边界函数，可以跨过两个接缝而不将图提交误认为是用户可见的答案。

这个模块是内部生产边界。它不最终确定一个 Entry Turn，发布转录消息，或回答一个持久化的 UserGate。这些效果仍然由 Entry 所拥有。重新进入是安全的，因为两个委派的组合在选择另一个物理效果之前会恢复它们现有的持久化回执。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from personagraph.l2.auxiliary_execution.application import (
    AuxiliaryApplicationPorts,
    AuxiliaryApplicationRequest,
    AuxiliaryApplicationResult,
    AuxiliaryApplicationStatus,
    run_auxiliary_application_to_boundary,
)
from personagraph.l2.auxiliary_execution.delivery.composition import (
    AuxiliaryTaskDeliveryPorts,
    AuxiliaryTaskDeliveryRequest,
    AuxiliaryTaskDeliveryResult,
    AuxiliaryTaskDeliveryStatus,
    run_auxiliary_committed_task_to_delivery,
)
from personagraph.runtime.turn_deadline import TurnDeadline


class AuxiliaryProductionChainStatus(StrEnum):
    """内部端到端链的公共有意义的终点。"""

    DELIVERY_READY = "delivery_ready"
    WAITING_USER = "waiting_user"
    WAITING_AUTHORIZATION = "waiting_authorization"
    WAITING_EXTERNAL = "waiting_external"
    TURN_LIMIT_REACHED = "turn_limit_reached"
    STEP_LIMIT_REACHED = "step_limit_reached"
    REVISION_REQUIRED = "revision_required"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AuxiliaryProductionChainRequest:
    session_id: str
    turn_id: str
    task_id: str
    desired_output: str = "完整、可执行、可验证的最终回答"
    max_auxiliary_effect_steps: int = 64
    max_task_graph_revision_cycles: int = 3
    deadline: TurnDeadline | None = None

    def __post_init__(self) -> None:
        for name in ("session_id", "turn_id", "task_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or len(value) > 200:
                raise ValueError(f"{name} must be a 1..200 character identity")
        if (
            not isinstance(self.desired_output, str)
            or not self.desired_output.strip()
        ):
            raise ValueError("desired_output must be non-empty")
        if (
            isinstance(self.max_auxiliary_effect_steps, bool)
            or not isinstance(self.max_auxiliary_effect_steps, int)
            or not 1 <= self.max_auxiliary_effect_steps <= 128
        ):
            raise ValueError(
                "max_auxiliary_effect_steps must be within 1..128"
            )
        if (
            isinstance(self.max_task_graph_revision_cycles, bool)
            or not isinstance(self.max_task_graph_revision_cycles, int)
            or not 1 <= self.max_task_graph_revision_cycles <= 12
        ):
            raise ValueError(
                "max_task_graph_revision_cycles must be within 1..12"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class AuxiliaryProductionChainPorts:
    """两个委派运行时状态机的显式物理端口。"""

    auxiliary: AuxiliaryApplicationPorts
    delivery: AuxiliaryTaskDeliveryPorts

    def __post_init__(self) -> None:
        if not isinstance(self.auxiliary, AuxiliaryApplicationPorts):
            raise TypeError("auxiliary must be AuxiliaryApplicationPorts")
        if not isinstance(self.delivery, AuxiliaryTaskDeliveryPorts):
            raise TypeError("delivery must be AuxiliaryTaskDeliveryPorts")


@dataclass(frozen=True, slots=True)
class AuxiliaryProductionChainResult:
    status: AuxiliaryProductionChainStatus
    reason_code: str
    auxiliary_result: AuxiliaryApplicationResult
    delivery_result: AuxiliaryTaskDeliveryResult | None = None
    final_delivery_id: str | None = None
    publication_body: str | None = None
    publication_format: str | None = None
    replayed_delivery: bool = False
    requested_user_questions: tuple[str, ...] = ()
    task_graph_revision_cycles: int = 0

    def __post_init__(self) -> None:
        if not self.reason_code or len(self.reason_code) > 240:
            raise ValueError("production chain requires a bounded reason code")
        if (
            isinstance(self.task_graph_revision_cycles, bool)
            or not isinstance(self.task_graph_revision_cycles, int)
            or not 0 <= self.task_graph_revision_cycles <= 12
        ):
            raise ValueError("TaskGraph revision cycle count is invalid")
        publication = (
            self.final_delivery_id,
            self.publication_body,
            self.publication_format,
        )
        if self.status is AuxiliaryProductionChainStatus.DELIVERY_READY:
            if self.delivery_result is None or any(
                value is None for value in publication
            ):
                raise ValueError(
                    "DELIVERY_READY requires the verified Delivery projection"
                )
            if not self.publication_body or not self.publication_body.strip():
                raise ValueError("publishable Delivery body must be non-empty")
        elif any(value is not None for value in publication):
            raise ValueError("only DELIVERY_READY may expose publication material")
        if self.replayed_delivery and (
            self.status is not AuxiliaryProductionChainStatus.DELIVERY_READY
        ):
            raise ValueError("only a ready Delivery can be replayed")
        if self.requested_user_questions and (
            self.status is not AuxiliaryProductionChainStatus.WAITING_USER
        ):
            raise ValueError("only WAITING_USER may expose validation questions")


def run_auxiliary_to_verified_delivery(
    request: AuxiliaryProductionChainRequest,
    *,
    ports: AuxiliaryProductionChainPorts,
) -> AuxiliaryProductionChainResult:
    """运行一个 Auxiliary 任务到经过验证的根 Delivery 或命名安全边界。"""

    if not isinstance(request, AuxiliaryProductionChainRequest):
        raise TypeError("request must be AuxiliaryProductionChainRequest")
    if not isinstance(ports, AuxiliaryProductionChainPorts):
        raise TypeError("ports must be AuxiliaryProductionChainPorts")

    remaining_auxiliary_effect_steps = request.max_auxiliary_effect_steps
    revision_cycles = 0

    while True:
        auxiliary_result = run_auxiliary_application_to_boundary(
            AuxiliaryApplicationRequest(
                session_id=request.session_id,
                turn_id=request.turn_id,
                task_id=request.task_id,
                desired_output=request.desired_output,
                max_effect_steps=remaining_auxiliary_effect_steps,
                deadline=request.deadline,
            ),
            ports=ports.auxiliary,
        )
        remaining_auxiliary_effect_steps -= auxiliary_result.effect_steps
        if remaining_auxiliary_effect_steps < 0:
            raise RuntimeError(
                "Auxiliary application exceeded its admitted effect budget"
            )
        if (
            auxiliary_result.status
            is not AuxiliaryApplicationStatus.COMMITTED
        ):
            return AuxiliaryProductionChainResult(
                status=_map_auxiliary_stop(auxiliary_result.status),
                reason_code=auxiliary_result.reason_code,
                auxiliary_result=auxiliary_result,
                requested_user_questions=(
                    (auxiliary_result.requested_user_question,)
                    if auxiliary_result.requested_user_question is not None
                    else ()
                ),
                task_graph_revision_cycles=revision_cycles,
            )

        delivery_result = run_auxiliary_committed_task_to_delivery(
            AuxiliaryTaskDeliveryRequest(
                session_id=request.session_id,
                turn_id=request.turn_id,
                task_id=request.task_id,
                deadline=request.deadline,
            ),
            ports=ports.delivery,
        )
        if (
            delivery_result.status
            is AuxiliaryTaskDeliveryStatus.REVISION_REQUIRED
        ):
            if revision_cycles >= request.max_task_graph_revision_cycles:
                return AuxiliaryProductionChainResult(
                    status=AuxiliaryProductionChainStatus.REVISION_REQUIRED,
                    reason_code="task_graph_revision_cycle_limit_reached",
                    auxiliary_result=auxiliary_result,
                    delivery_result=delivery_result,
                    task_graph_revision_cycles=revision_cycles,
                )
            if remaining_auxiliary_effect_steps == 0:
                return AuxiliaryProductionChainResult(
                    status=AuxiliaryProductionChainStatus.STEP_LIMIT_REACHED,
                    reason_code="task_graph_revision_auxiliary_budget_exhausted",
                    auxiliary_result=auxiliary_result,
                    delivery_result=delivery_result,
                    task_graph_revision_cycles=revision_cycles,
                )
            revision_cycles += 1
            continue

        status = _map_delivery_stop(delivery_result.status)
        return AuxiliaryProductionChainResult(
            status=status,
            reason_code=delivery_result.reason_code,
            auxiliary_result=auxiliary_result,
            delivery_result=delivery_result,
            final_delivery_id=delivery_result.final_delivery_id,
            publication_body=delivery_result.publication_body,
            publication_format=delivery_result.publication_format,
            replayed_delivery=delivery_result.replayed_delivery,
            requested_user_questions=delivery_result.requested_user_questions,
            task_graph_revision_cycles=revision_cycles,
        )


def _map_auxiliary_stop(
    status: AuxiliaryApplicationStatus,
) -> AuxiliaryProductionChainStatus:
    return {
        AuxiliaryApplicationStatus.WAITING_USER: (
            AuxiliaryProductionChainStatus.WAITING_USER
        ),
        AuxiliaryApplicationStatus.WAITING_AUTHORIZATION: (
            AuxiliaryProductionChainStatus.WAITING_AUTHORIZATION
        ),
        AuxiliaryApplicationStatus.WAITING_EXTERNAL: (
            AuxiliaryProductionChainStatus.WAITING_EXTERNAL
        ),
        AuxiliaryApplicationStatus.TURN_LIMIT_REACHED: (
            AuxiliaryProductionChainStatus.TURN_LIMIT_REACHED
        ),
        AuxiliaryApplicationStatus.STEP_LIMIT_REACHED: (
            AuxiliaryProductionChainStatus.STEP_LIMIT_REACHED
        ),
        AuxiliaryApplicationStatus.REVISION_REQUIRED: (
            AuxiliaryProductionChainStatus.REVISION_REQUIRED
        ),
        AuxiliaryApplicationStatus.FAILED: (
            AuxiliaryProductionChainStatus.FAILED
        ),
        AuxiliaryApplicationStatus.COMMITTED: (
            AuxiliaryProductionChainStatus.FAILED
        ),
    }[status]


def _map_delivery_stop(
    status: AuxiliaryTaskDeliveryStatus,
) -> AuxiliaryProductionChainStatus:
    return {
        AuxiliaryTaskDeliveryStatus.DELIVERY_READY: (
            AuxiliaryProductionChainStatus.DELIVERY_READY
        ),
        AuxiliaryTaskDeliveryStatus.WAITING_USER: (
            AuxiliaryProductionChainStatus.WAITING_USER
        ),
        AuxiliaryTaskDeliveryStatus.WAITING_EXTERNAL: (
            AuxiliaryProductionChainStatus.WAITING_EXTERNAL
        ),
        AuxiliaryTaskDeliveryStatus.TURN_LIMIT: (
            AuxiliaryProductionChainStatus.TURN_LIMIT_REACHED
        ),
        AuxiliaryTaskDeliveryStatus.REVISION_REQUIRED: (
            AuxiliaryProductionChainStatus.REVISION_REQUIRED
        ),
        AuxiliaryTaskDeliveryStatus.BLOCKED: (
            AuxiliaryProductionChainStatus.BLOCKED
        ),
        AuxiliaryTaskDeliveryStatus.FAILED: (
            AuxiliaryProductionChainStatus.FAILED
        ),
    }[status]


__all__ = [
    "AuxiliaryProductionChainPorts",
    "AuxiliaryProductionChainRequest",
    "AuxiliaryProductionChainResult",
    "AuxiliaryProductionChainStatus",
    "run_auxiliary_to_verified_delivery",
]
