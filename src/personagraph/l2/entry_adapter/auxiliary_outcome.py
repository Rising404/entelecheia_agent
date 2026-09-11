"""权威 Entry 之下的纯 Auxiliary 结果解释。

调用此 policy 时，生产链已执行其私有 Task/WorkRun effect。它只返回随后可走的现有
Entry 权威路径：发布冻结 Delivery、提出一个持久化 UserGate 问题，或中断 Turn。
Entry 仍负责活动 Window 重读、租约检查、事件发射、Store 变更及所有 finalizer。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

from personagraph.l2.auxiliary_execution.production_chain import (
    AuxiliaryProductionChainStatus,
)
from personagraph.runtime.turn_events import RuntimeErrorCode, RuntimeStage


class AuxiliaryProductionChainOutcomePort(Protocol):
    """从一次已完成生产链调用中读取的私有事实。"""

    @property
    def status(self) -> AuxiliaryProductionChainStatus: ...

    @property
    def final_delivery_id(self) -> str | None: ...

    @property
    def requested_user_questions(self) -> tuple[str, ...]: ...


class AuxiliaryEntryOutcomeDecisionKind(StrEnum):
    """从私有 Auxiliary 结果中选择的现有 Entry 权威路径。"""

    FINALIZE_VERIFIED_DELIVERY = "finalize_verified_delivery"
    FINALIZE_FORMAL_QUESTION = "finalize_formal_question"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class AuxiliaryEntryOutcomeDecision:
    """绝不包含 Delivery 正文的无副作用投影。"""

    kind: AuxiliaryEntryOutcomeDecisionKind
    delivery_id: str | None = None
    reply: str | None = None
    end_reason: (
        Literal["host_stopped", "module_error", "persistence_error"] | None
    ) = None
    error_code: RuntimeErrorCode | None = None
    stage: RuntimeStage | None = None

    def __post_init__(self) -> None:
        if (
            self.kind
            is AuxiliaryEntryOutcomeDecisionKind.FINALIZE_VERIFIED_DELIVERY
        ):
            if self.delivery_id is None:
                raise ValueError("verified delivery decision requires delivery_id")
            return
        if (
            self.kind
            is AuxiliaryEntryOutcomeDecisionKind.FINALIZE_FORMAL_QUESTION
        ):
            if self.reply is None:
                raise ValueError("formal question decision requires reply")
            return
        if (
            self.end_reason is None
            or self.error_code is None
            or self.stage is None
        ):
            raise ValueError("incomplete decision requires a stop projection")


def interpret_auxiliary_production_chain_outcome(
    result: AuxiliaryProductionChainOutcomePort,
    *,
    authoritative_window_available: bool,
    allow_user_input: bool = True,
) -> AuxiliaryEntryOutcomeDecision:
    """将 Auxiliary 结果无副作用地映射到现有 Entry 权威路径。

    ``authoritative_window_available`` 是 executor 返回后由 Entry 提供的读取事实。
    将其保留为显式标量，可维持现有 ready-Delivery 短路：缺失 Window 属于持久化停止，
    不得强制读取格式错误的 Delivery 标识。
    """

    if type(allow_user_input) is not bool:
        raise TypeError("allow_user_input must be a boolean")

    if result.status is AuxiliaryProductionChainStatus.DELIVERY_READY:
        if not authoritative_window_available or result.final_delivery_id is None:
            return _incomplete_decision(
                end_reason="persistence_error",
                error_code=RuntimeErrorCode.PERSIST_FAILED,
                stage=RuntimeStage.PERSIST,
            )
        return AuxiliaryEntryOutcomeDecision(
            kind=(
                AuxiliaryEntryOutcomeDecisionKind.FINALIZE_VERIFIED_DELIVERY
            ),
            delivery_id=result.final_delivery_id,
        )

    if (
        allow_user_input
        and result.status is AuxiliaryProductionChainStatus.WAITING_USER
    ):
        questions = tuple(result.requested_user_questions)
        if len(questions) == 1 and questions[0].strip():
            return AuxiliaryEntryOutcomeDecision(
                kind=(
                    AuxiliaryEntryOutcomeDecisionKind.FINALIZE_FORMAL_QUESTION
                ),
                reply=questions[0],
            )

    end_reason, error_code, stage = _no_public_stop_projection(result.status)
    return _incomplete_decision(
        end_reason=end_reason,
        error_code=error_code,
        stage=stage,
    )


def _no_public_stop_projection(
    status: object,
) -> tuple[
    Literal["host_stopped", "module_error"], RuntimeErrorCode, RuntimeStage
]:
    return {
        AuxiliaryProductionChainStatus.WAITING_USER: (
            "host_stopped",
            RuntimeErrorCode.TRANSITION_DENIED,
            RuntimeStage.L2_PLAN,
        ),
        AuxiliaryProductionChainStatus.WAITING_AUTHORIZATION: (
            "host_stopped",
            RuntimeErrorCode.TRANSITION_DENIED,
            RuntimeStage.TRANSITION_GUARD,
        ),
        AuxiliaryProductionChainStatus.WAITING_EXTERNAL: (
            "host_stopped",
            RuntimeErrorCode.TOOL_COMPLETION_UNCONFIRMED,
            RuntimeStage.TOOL,
        ),
        AuxiliaryProductionChainStatus.TURN_LIMIT_REACHED: (
            "host_stopped",
            RuntimeErrorCode.TURN_DEADLINE_EXCEEDED,
            RuntimeStage.RESPONSE,
        ),
        AuxiliaryProductionChainStatus.STEP_LIMIT_REACHED: (
            "host_stopped",
            RuntimeErrorCode.TRANSITION_DENIED,
            RuntimeStage.L2_PLAN,
        ),
        AuxiliaryProductionChainStatus.REVISION_REQUIRED: (
            "module_error",
            RuntimeErrorCode.TRANSITION_DENIED,
            RuntimeStage.TRANSITION_GUARD,
        ),
        AuxiliaryProductionChainStatus.BLOCKED: (
            "host_stopped",
            RuntimeErrorCode.TRANSITION_DENIED,
            RuntimeStage.L2_PLAN,
        ),
        AuxiliaryProductionChainStatus.FAILED: (
            "module_error",
            RuntimeErrorCode.INTERNAL_FAILURE,
            RuntimeStage.RESPONSE,
        ),
    }.get(
        status,
        (
            "module_error",
            RuntimeErrorCode.INTERNAL_FAILURE,
            RuntimeStage.RESPONSE,
        ),
    )


def _incomplete_decision(
    *,
    end_reason: Literal["host_stopped", "module_error", "persistence_error"],
    error_code: RuntimeErrorCode,
    stage: RuntimeStage,
) -> AuxiliaryEntryOutcomeDecision:
    return AuxiliaryEntryOutcomeDecision(
        kind=AuxiliaryEntryOutcomeDecisionKind.INCOMPLETE,
        end_reason=end_reason,
        error_code=error_code,
        stage=stage,
    )


__all__ = [
    "AuxiliaryEntryOutcomeDecision",
    "AuxiliaryEntryOutcomeDecisionKind",
    "AuxiliaryProductionChainOutcomePort",
    "interpret_auxiliary_production_chain_outcome",
]
