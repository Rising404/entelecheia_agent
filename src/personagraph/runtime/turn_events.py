"""新运行时的轮次生命周期事件。

本模块刻意独立于旧版 ``runtime.events`` 日志。它定义新的单一路径运行时将
发出的封闭、无内容生命周期事实，不依赖流、数据库、模型、图或功能开关。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeStage(StrEnum):
    """新轮次生命周期中由宿主持有的阶段。"""

    INGRESS = "INGRESS"
    CLASSIFY = "CLASSIFY"
    SUPERVISOR = "SUPERVISOR"
    L0_GENERATE = "L0_GENERATE"
    L1_BOOTSTRAP = "L1_BOOTSTRAP"
    L2_UNDERSTAND = "L2_UNDERSTAND"
    L2_PLAN = "L2_PLAN"
    TRANSITION_GUARD = "TRANSITION_GUARD"
    TOOL = "TOOL"
    OBSERVATION = "OBSERVATION"
    VERIFICATION = "VERIFICATION"
    PERSIST = "PERSIST"
    RESPONSE = "RESPONSE"


class TurnEventStatus(StrEnum):
    """表示阶段已开始、已完成或已到达宿主控制的停止点。"""

    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    PAUSED = "paused"


class RuntimeErrorCode(StrEnum):
    """用于聚合与恢复的稳定、无内容失败类别。"""

    INGRESS_REJECTED = "INGRESS_REJECTED"
    INPUT_INVALID = "INPUT_INVALID"
    TURN_DEADLINE_EXCEEDED = "TURN_DEADLINE_EXCEEDED"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_TRANSPORT_FAILURE = "MODEL_TRANSPORT_FAILURE"
    MODEL_COMPLETION_UNCONFIRMED = "MODEL_COMPLETION_UNCONFIRMED"
    MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"
    CONTEXT_BUDGET_EXCEEDED = "CONTEXT_BUDGET_EXCEEDED"
    MODEL_CONFIGURATION_FAILURE = "MODEL_CONFIGURATION_FAILURE"
    L1_RUNTIME_NOT_READY = "L1_RUNTIME_NOT_READY"
    TRANSITION_DENIED = "TRANSITION_DENIED"
    TOOL_FAILED = "TOOL_FAILED"
    TOOL_COMPLETION_UNCONFIRMED = "TOOL_COMPLETION_UNCONFIRMED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    PERSIST_FAILED = "PERSIST_FAILED"
    RESPONSE_FAILED = "RESPONSE_FAILED"
    INTERNAL_FAILURE = "INTERNAL_FAILURE"


class TurnEvent(BaseModel):
    """新运行时轮次中的一项私有、无内容事实。

    ID 是不透明的关联值。该事件刻意不接受自由文本诊断或任意元数据：提示词、
    工具载荷、文档文本、模型推理、异常字符串及秘密均应存入单独保护的诊断
    存储；此处只能通过不透明的 ``diagnostic_ref`` 寻址。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, frozen=True)
    event_id: str = Field(min_length=1, max_length=128)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    turn_id: str = Field(min_length=1, max_length=128)
    parent_event_id: str | None = Field(default=None, min_length=1, max_length=128)
    insession_task_id: str | None = Field(default=None, min_length=1, max_length=128)
    insession_task_node_id: str | None = Field(default=None, min_length=1, max_length=128)
    work_run_id: str | None = Field(default=None, min_length=1, max_length=128)
    attempt_id: str | None = Field(default=None, min_length=1, max_length=128)
    stage: RuntimeStage
    status: TurnEventStatus
    occurred_at: datetime
    duration_ms: int | None = Field(default=None, ge=0)
    model_call_id: str | None = Field(default=None, min_length=1, max_length=128)
    model_attempt: int | None = Field(default=None, ge=1, le=6)
    operation_id: str | None = Field(default=None, min_length=1, max_length=128)
    error_code: RuntimeErrorCode | None = None
    retryable: bool = False
    diagnostic_ref: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def _validate_lifecycle_shape(self) -> 'TurnEvent':
        if self.parent_event_id == self.event_id:
            raise ValueError("parent_event_id must not equal event_id")
        if self.model_attempt is not None and self.model_call_id is None:
            raise ValueError("model_attempt requires model_call_id")
        if self.error_code is not None and self.status not in {
            TurnEventStatus.FAILED,
            TurnEventStatus.BLOCKED,
        }:
            raise ValueError("error_code requires failed or blocked status")
        if self.status in {TurnEventStatus.FAILED, TurnEventStatus.BLOCKED} and self.error_code is None:
            raise ValueError("failed or blocked status requires error_code")
        if self.retryable and self.status not in {
            TurnEventStatus.FAILED,
            TurnEventStatus.BLOCKED,
        }:
            raise ValueError("retryable is only meaningful for failed or blocked status")
        if self.status is TurnEventStatus.STARTED and self.duration_ms is not None:
            raise ValueError("started events cannot have duration_ms")
        return self


# Runtime lanes emit the same persisted lifecycle event shape.  Keeping this
# callable contract beside the event prevents L1 from importing Entry's
# TaskGraph-aware orchestration contracts merely for a type annotation.
EntryEventEmitter = Callable[[TurnEvent], int | None]


class TurnPublicEvent(BaseModel):
    """一项已持久化运行时生命周期事实的严格公开投影。

    公开事件仅包含宿主持有的生命周期数据与不透明关联 ID。它可以安全地流式
    传输和渲染，但绝不会成为提示词材料：``prompt_replay`` 固定为 ``false``，
    而非由调用方控制。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = Field(default=1, frozen=True)
    event_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    sequence: int = Field(ge=1, strict=True)
    session_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    turn_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    stage: RuntimeStage
    status: TurnEventStatus
    occurred_at: datetime
    error_code: RuntimeErrorCode | None = None
    retryable: bool = False
    insession_task_id: str | None = Field(
        default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$"
    )
    work_run_id: str | None = Field(
        default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$"
    )
    attempt_id: str | None = Field(
        default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$"
    )
    operation_id: str | None = Field(
        default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$"
    )
    prompt_replay: Literal[False] = False


def new_turn_event(
    *,
    turn_id: str,
    stage: RuntimeStage,
    status: TurnEventStatus,
    **kwargs: object,
) -> TurnEvent:
    """构建一项由宿主持有的事件，包含不透明 ID 与带时区的 UTC 时间戳。

    构造事件不等于已存储；调用方先 append_runtime_turn_event 取得 sequence，
    再向客户端投影。它只描述生命周期事实，不收录模型消息或工具正文。
    """

    return TurnEvent(
        event_id=f"turnevt_{uuid4().hex}",
        turn_id=turn_id,
        stage=stage,
        status=status,
        occurred_at=datetime.now(timezone.utc),
        **kwargs,
    )


def project_turn_event(event: TurnEvent, *, sequence: int) -> TurnPublicEvent:
    """投影一项已持久化事件，且不暴露私有诊断或载荷。

    sequence 由持久事件存储提供，SSE 和事件追赶 API 共用这份公开字段投影。
    只投影列出的身份、阶段和安全错误码，不透传 diagnostic_ref、模型用量或正文；
    排查消息/工具内容应另查 trajectory，而不是扩大此公开事件合同。
    """

    if event.session_id is None:
        raise ValueError("public runtime event requires session_id")

    return TurnPublicEvent(
        event_id=event.event_id,
        sequence=sequence,
        session_id=event.session_id,
        turn_id=event.turn_id,
        stage=event.stage,
        status=event.status,
        occurred_at=event.occurred_at,
        error_code=event.error_code,
        retryable=event.retryable,
        insession_task_id=event.insession_task_id,
        work_run_id=event.work_run_id,
        attempt_id=event.attempt_id,
        operation_id=event.operation_id,
    )
