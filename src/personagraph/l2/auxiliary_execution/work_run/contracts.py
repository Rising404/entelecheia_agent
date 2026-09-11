"""延迟不可变合约用于 AuxiliaryGraph WorkRuns。

这些值冻结了 WorkRun 的身份、控制器输入和可观测结果。它们不会检查前沿、访问 Store 行、选择能力、调用模型、分发 Tool 或结算节点。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personagraph.l2.auxiliary_graph import (
    AuxiliaryNodeDefinition,
    AuxiliaryNodeExecutorKind,
    TaskGraphSemanticBaseSnapshot,
)
from personagraph.l2.task_graph.contracts import (
    InSessionTaskGraphRevisionValidationContext,
)
from personagraph.l2.work_run import AuxiliaryNodeSubject
from personagraph.l2.auxiliary_execution.adapters.model_binding_contracts import _canonical_sha256
from personagraph.tools.catalog import CatalogSnapshot
from personagraph.l2.task_execution.tool_bridge.contracts import AttemptToolBridge


KNOWLEDGE_COGNITION_CAPABILITY = "knowledge_cognition"
KNOWLEDGE_INDEXING_CAPABILITY = "knowledge_indexing"


@dataclass(frozen=True, slots=True)
class AuxiliaryNodeToolRuntime:
    """An exact node capability catalog and its authorized dispatch port."""

    catalog_snapshot: CatalogSnapshot
    tool_bridge: AttemptToolBridge


AuxiliaryNodeToolRuntimeFactory = Callable[
    [AuxiliaryNodeSubject, AuxiliaryNodeDefinition], AuxiliaryNodeToolRuntime | None
]


_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )


class AuxiliaryWorkRunStatus(StrEnum):
    COMPLETED = "completed"
    WAITING_USER = "waiting_user"
    WAITING_AUTHORIZATION = "waiting_authorization"
    WAITING_EXTERNAL = "waiting_external"
    TURN_LIMIT_REACHED = "turn_limit_reached"
    MODEL_INTERRUPTED = "model_interrupted"
    VERIFICATION_INTERRUPTED = "verification_interrupted"
    STEP_LIMIT_REACHED = "step_limit_reached"
    DEPENDENCY_PROJECTION_UNAVAILABLE = "dependency_projection_unavailable"
    CAPABILITY_CATALOG_UNAVAILABLE = "capability_catalog_unavailable"
    CAPABILITY_TOOL_BRIDGE_UNAVAILABLE = "capability_tool_bridge_unavailable"
    CROSS_TURN_RECOVERY_UNAVAILABLE = "cross_turn_recovery_unavailable"
    FAILED = "failed"
    FAILED_CLOSED = "failed_closed"


class AuxiliaryWorkRunIdPlan(_Contract):
    """由调用组件提供的基础标识；每次重试的身份均从中派生。"""

    work_run_id: str = Field(pattern=_ID_PATTERN)
    create_work_run_apply_id: str = Field(pattern=_ID_PATTERN)
    attempt_id: str = Field(pattern=_ID_PATTERN)
    start_attempt_apply_id: str = Field(pattern=_ID_PATTERN)
    attempt_decision_apply_id: str = Field(pattern=_ID_PATTERN)
    attempt_model_call_id: str = Field(pattern=_ID_PATTERN)
    prepare_verification_apply_id: str = Field(pattern=_ID_PATTERN)
    verification_request_id: str = Field(pattern=_ID_PATTERN)
    verification_model_call_id: str = Field(pattern=_ID_PATTERN)
    commit_verification_apply_id: str = Field(pattern=_ID_PATTERN)
    completion_id: str = Field(pattern=_ID_PATTERN)

    def for_attempt(self, base: str, ordinal: int) -> str:
        return _ordinal_stable_id(base, ordinal=ordinal)

    def tool_call_id(self, attempt_ordinal: int, call_ordinal: int) -> str:
        return _bounded_stable_id(
            self.attempt_id,
            f":a{attempt_ordinal}:tool-{call_ordinal}",
        )


def derive_auxiliary_work_run_ids(
    *,
    session_id: str,
    subject: AuxiliaryNodeSubject,
) -> AuxiliaryWorkRunIdPlan:
    """为一个确切的 节点修订版本推导重启稳定的标识符。

    刻意省略 ``turn_id``：在后续 Turn 中恢复等待中或被中断的节点时，仍须指向同一个逻辑 WorkRun 和模型调用。Attempt 序号由 :meth:`AuxiliaryWorkRunIdPlan.for_attempt` 添加。
    """

    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a non-empty durable identity")
    if len(session_id) > 200:
        raise ValueError("session_id must be at most 200 characters")
    if not isinstance(subject, AuxiliaryNodeSubject):
        raise TypeError("subject must be an AuxiliaryNodeSubject")
    digest = _canonical_sha256(
        {
            "schema_version": "auxiliary-v2-work-run-stable-ids-v1",
            "session_id": session_id,
            "task_id": subject.task_id,
            "auxiliary_graph_id": subject.auxiliary_graph_id,
            "auxiliary_graph_revision": subject.auxiliary_graph_revision,
            "node_id": subject.node_id,
            "node_revision": subject.node_revision,
        }
    )[:32]
    namespace = f"auxv2wr-{digest}"
    return AuxiliaryWorkRunIdPlan(
        work_run_id=f"{namespace}:run",
        create_work_run_apply_id=f"{namespace}:create",
        attempt_id=f"{namespace}:attempt",
        start_attempt_apply_id=f"{namespace}:start",
        attempt_decision_apply_id=f"{namespace}:decide",
        attempt_model_call_id=f"{namespace}:attempt-model",
        prepare_verification_apply_id=f"{namespace}:prepare",
        verification_request_id=f"{namespace}:verification",
        verification_model_call_id=f"{namespace}:verification-model",
        commit_verification_apply_id=f"{namespace}:settle",
        completion_id=f"{namespace}:completion",
    )


class AuxiliaryWorkRunRequest(_Contract):
    session_id: str = Field(pattern=_ID_PATTERN)
    turn_id: str = Field(pattern=_ID_PATTERN)
    subject: AuxiliaryNodeSubject
    executor_kind: Literal[
        AuxiliaryNodeExecutorKind.MODEL_WORK_RUN,
        AuxiliaryNodeExecutorKind.USER_GATE,
        AuxiliaryNodeExecutorKind.TERMINAL_PLANNER,
    ]
    initial_driver_state_guard_sha256: str = Field(pattern=_SHA256_PATTERN)
    id_plan: AuxiliaryWorkRunIdPlan
    allow_user_input: bool = True
    task_graph_validation_context: (
        InSessionTaskGraphRevisionValidationContext | None
    ) = None
    task_graph_semantic_base_snapshot: TaskGraphSemanticBaseSnapshot | None = None

    @model_validator(mode="after")
    def _validate_executor_context(self) -> "AuxiliaryWorkRunRequest":
        terminal = self.executor_kind is AuxiliaryNodeExecutorKind.TERMINAL_PLANNER
        if terminal != (self.task_graph_validation_context is not None):
            raise ValueError(
                "terminal planner alone requires TaskGraph validation context"
            )
        context = self.task_graph_validation_context
        if not terminal and self.task_graph_semantic_base_snapshot is not None:
            raise ValueError("only a terminal planner may receive a base snapshot")
        if context is not None and (
            context.session_id != self.session_id
            or context.source_turn_id != self.turn_id
            or context.target_insession_task_id != self.subject.task_id
        ):
            raise ValueError("TaskGraph context must bind this Session, Turn and Task")
        if context is not None:
            positive_base = context.expected_current_graph_revision is not None
            if positive_base != (self.task_graph_semantic_base_snapshot is not None):
                raise ValueError(
                    "positive-base terminal planning requires exactly one semantic "
                    "base snapshot"
                )
            if (
                self.task_graph_semantic_base_snapshot is not None
                and self.task_graph_semantic_base_snapshot.base_task_graph_revision
                != context.expected_current_graph_revision
            ):
                raise ValueError(
                    "semantic base snapshot revision differs from the expected "
                    "current TaskGraph"
                )
        return self


class AuxiliaryWorkRunResult(_Contract):
    status: AuxiliaryWorkRunStatus
    reason_code: str = Field(min_length=1, max_length=200)
    subject: AuxiliaryNodeSubject
    executor_kind: AuxiliaryNodeExecutorKind
    work_run_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    attempt_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    verification_request_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    completion_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    output_revision: int | None = Field(default=None, ge=1)
    window_state_version: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_completion(self) -> "AuxiliaryWorkRunResult":
        completed = self.status is AuxiliaryWorkRunStatus.COMPLETED
        if completed != (
            self.work_run_id is not None
            and self.completion_id is not None
            and self.output_revision is not None
        ):
            raise ValueError("only a completed result carries completion authority")
        return self


def _ordinal_stable_id(base: str, *, ordinal: int) -> str:
    if ordinal < 1:
        raise ValueError("Attempt ordinal must be positive")
    return base if ordinal == 1 else _bounded_stable_id(base, f":attempt-{ordinal}")


def _bounded_stable_id(base: str, suffix: str) -> str:
    if len(base) + len(suffix) <= 200:
        return f"{base}{suffix}"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
    prefix = base[: 200 - len(suffix) - len(digest) - 1]
    return f"{prefix}:{digest}{suffix}"


__all__ = [
    "AuxiliaryNodeToolRuntime",
    "AuxiliaryNodeToolRuntimeFactory",
    "KNOWLEDGE_COGNITION_CAPABILITY",
    "KNOWLEDGE_INDEXING_CAPABILITY",
    "AuxiliaryWorkRunStatus",
    "AuxiliaryWorkRunRequest",
    "AuxiliaryWorkRunResult",
    "AuxiliaryWorkRunIdPlan",
    "derive_auxiliary_work_run_ids",
]
