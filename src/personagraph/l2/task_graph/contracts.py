"""L2 TaskGraph 领域的类型化、供应商中立契约。

本模块契约只描述任务语义。它们不调用模型、不查询 retrieval、不决定路由、不执行 WorkRun，
也不写 SQLite。保持边界显式，可防止已退役的 Session 级 WorkState 和长期 memory 任务领域
泄漏到新 Runtime 任务图。

本模块刻意位于 ``runtime`` 之外，使 Runtime 编排和 Session 持久化都可依赖同一纯领域
契约，而不会形成 store 到 Runtime 的反向依赖。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_LOCAL_KEY_PATTERN = r"^[a-z][a-z0-9_-]{0,63}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _require_canonical_contract_text(value: str) -> str:
    if value != value.strip() or not value:
        raise ValueError("TaskGraph contract text must be non-empty canonical text")
    if "\x00" in value:
        raise ValueError("TaskGraph contract text cannot contain NUL")
    return value


class InSessionTaskStatus(StrEnum):
    """根节点与子任务节点共享的仅有生命周期状态。"""

    PROPOSED = "proposed"
    ACTIVE = "active"
    AWAITING_USER = "awaiting_user"
    WAITING_EXTERNAL = "waiting_external"
    INTERRUPTED = "interrupted"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class InSessionTaskNodeKind(StrEnum):
    ROOT = "root"
    SUBTASK = "subtask"


class InSessionTaskGraphValidationCode(StrEnum):
    """提议图无法成为 authority 的确定性原因。"""

    EMPTY_BATCH = "empty_batch"
    ROOT_LIMIT_EXCEEDED = "root_limit_exceeded"
    NODE_LIMIT_EXCEEDED = "node_limit_exceeded"
    DEPTH_LIMIT_EXCEEDED = "depth_limit_exceeded"
    DUPLICATE_NODE_KEY = "duplicate_node_key"
    ROOT_NODE_MISSING = "root_node_missing"
    ROOT_NODE_INVALID = "root_node_invalid"
    INVALID_PARENT_REFERENCE = "invalid_parent_reference"
    INVALID_PARENT_KIND = "invalid_parent_kind"
    CYCLE_DETECTED = "cycle_detected"
    ORPHAN_NODE = "orphan_node"
    DUPLICATE_ACCEPTANCE_ID = "duplicate_acceptance_id"
    MISSING_AUTHORIZATION_ANCHOR = "missing_authorization_anchor"
    UNKNOWN_AUTHORIZATION_ANCHOR = "unknown_authorization_anchor"
    UNAUTHORIZED_AUTHORIZATION_SOURCE = "unauthorized_authorization_source"
    NODE_WITHOUT_AUTHORIZED_SOURCE = "node_without_authorized_source"
    ACCEPTANCE_WITHOUT_AUTHORIZED_SOURCE = "acceptance_without_authorized_source"
    UNSOURCED_CONSTRAINT = "unsourced_constraint"
    UNKNOWN_SOURCE_ANCHOR = "unknown_source_anchor"
    MISSING_SOURCE_ANCHOR = "missing_source_anchor"
    UNMAPPED_REQUIRED_ANCHOR = "unmapped_required_anchor"
    SOURCE_TURN_MISMATCH = "source_turn_mismatch"
    CURRENT_USER_ANCHOR_MISMATCH = "current_user_anchor_mismatch"


class InSessionTaskGraphLimits(_Contract):
    """一个图提案的 Host 所有结构资源边界。

    这些限制保护存储和模型输出验证。它们不是判断两个用户请求是否应拥有不同根节点的语义
    规则；不确定语义分组仍属于模型/prompt 质量问题。
    """

    max_root_tasks: int = Field(default=3, ge=1, le=24)
    max_nodes_per_task: int = Field(default=64, ge=1, le=512)
    max_depth: int = Field(default=12, ge=1, le=64)


class InSessionTaskSourceAnchor(_Contract):
    """由 Host 提供、可供提案使用的不可变源 span。"""

    anchor_id: str = Field(pattern=_LOCAL_KEY_PATTERN)
    source_turn_id: str = Field(min_length=1, max_length=128)
    source_kind: Literal[
        "current_user_instruction",
        "current_user_context",
        "previously_authorized_task_state",
        "quoted_external",
        "attachment",
        "retrieved_document",
        "tool_observation",
        "memory",
        "gap",
    ]
    gap_blocking: bool | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    excerpt: str = Field(min_length=1, max_length=4_000)

    @model_validator(mode="after")
    def _validate_range(self) -> 'InSessionTaskSourceAnchor':
        if self.end <= self.start:
            raise ValueError("source anchor end must exceed start")
        if (self.source_kind == "gap") != (self.gap_blocking is not None):
            raise ValueError(
                "gap source anchors alone require an explicit blocking disposition"
            )
        return self


class InSessionTaskAcceptanceProposal(_Contract):
    """由包含它的节点所有、绑定到源的一项完成条件。"""

    acceptance_id: str = Field(pattern=_LOCAL_KEY_PATTERN)
    criterion: str = Field(min_length=1, max_length=1_000)
    source_anchor_ids: tuple[str, ...] = Field(min_length=1, max_length=16)

    @field_validator("criterion")
    @classmethod
    def _require_canonical_criterion(cls, value: str) -> str:
        return _require_canonical_contract_text(value)

    @field_validator("source_anchor_ids")
    @classmethod
    def _require_unique_source_anchors(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("acceptance source anchors must be unique")
        return values


class InSessionTaskNodeProposal(_Contract):
    """不可信图提案中的一个节点，通过局部键寻址。"""

    node_key: str = Field(pattern=_LOCAL_KEY_PATTERN)
    node_kind: InSessionTaskNodeKind
    parent_node_key: str | None = Field(default=None, pattern=_LOCAL_KEY_PATTERN)
    title: str = Field(min_length=1, max_length=240)
    objective: str = Field(min_length=1, max_length=2_000)
    source_anchor_ids: tuple[str, ...] = Field(min_length=1, max_length=32)
    acceptance_criteria: tuple[InSessionTaskAcceptanceProposal, ...] = Field(
        min_length=1, max_length=64
    )
    constraints: tuple[str, ...] = Field(default=(), max_length=32)

    @field_validator("title", "objective")
    @classmethod
    def _require_canonical_node_text(cls, value: str) -> str:
        return _require_canonical_contract_text(value)

    @field_validator("constraints")
    @classmethod
    def _require_canonical_constraints(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("TaskGraph constraints must be unique")
        for value in values:
            _require_canonical_contract_text(value)
        return values

    @field_validator("source_anchor_ids")
    @classmethod
    def _require_unique_node_anchors(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("node source anchors must be unique")
        return values

    @model_validator(mode="after")
    def _validate_parent_shape(self) -> 'InSessionTaskNodeProposal':
        if self.node_kind is InSessionTaskNodeKind.ROOT and self.parent_node_key is not None:
            raise ValueError("root node cannot declare a parent")
        if self.node_kind is InSessionTaskNodeKind.SUBTASK and self.parent_node_key is None:
            raise ValueError("subtask node requires a parent")
        return self


class InSessionTaskRootGraphProposal(_Contract):
    """完整快照提案中一个根任务的完整树。"""

    root_key: str = Field(pattern=_LOCAL_KEY_PATTERN)
    nodes: tuple[InSessionTaskNodeProposal, ...] = Field(min_length=1, max_length=512)


class NewInSessionTaskGraphsProposal(_Contract):
    """创建一棵或多棵独立所有根任务树的 candidate。"""

    schema_version: Literal["insession-task-graph-create-v2"] = "insession-task-graph-create-v2"
    source_turn_id: str = Field(min_length=1, max_length=128)
    roots: tuple[InSessionTaskRootGraphProposal, ...] = Field(min_length=1, max_length=24)


class InSessionTaskGraphRevisionProposal(_Contract):
    """为一个现有根任务提议的不可信完整快照。

    目标 Task、源 Turn 和预期当前图 revision 是 Host 所有提交参数。将这些标识符排除在模型
    提案外，可防止模型输出选择自己的 mutation authority。
    """

    schema_version: Literal["insession-task-graph-revision-v2"] = (
        "insession-task-graph-revision-v2"
    )
    root: InSessionTaskRootGraphProposal


class InSessionTaskGraphValidationContext(_Contract):
    """Host 提供给提案验证的可信事实。"""

    session_id: str = Field(min_length=1, max_length=128)
    source_turn_id: str = Field(min_length=1, max_length=128)
    source_anchors: tuple[InSessionTaskSourceAnchor, ...] = Field(max_length=256)
    authorization_anchor_ids: tuple[str, ...] = Field(min_length=1, max_length=256)
    required_anchor_ids: tuple[str, ...] = Field(default=(), max_length=256)
    limits: InSessionTaskGraphLimits = InSessionTaskGraphLimits()

    @field_validator("source_anchors")
    @classmethod
    def _require_unique_anchor_ids(
        cls, values: tuple[InSessionTaskSourceAnchor, ...]
    ) -> tuple[InSessionTaskSourceAnchor, ...]:
        ids = [item.anchor_id for item in values]
        if len(ids) != len(set(ids)):
            raise ValueError("source anchor ids must be unique")
        return values

    @field_validator("authorization_anchor_ids", "required_anchor_ids")
    @classmethod
    def _require_unique_anchor_references(
        cls, values: tuple[str, ...]
    ) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("anchor references must be unique")
        return values


class InSessionTaskGraphRevisionValidationContext(
    InSessionTaskGraphValidationContext
):
    """一个图 revision 提案的 Host 所有 authority 事实。

    ``None`` 表示目标是等待 revision 1 的现有 shell；整数表示在该当前 revision 之上提议
    完整快照。由于此纯上下文由 Host 而非模型构建，它可以携带
    ``previously_authorized_task_state`` anchor。Store 在 mutation 前仍必须重新加载并验证
    每个 anchor、目标和 revision。
    """

    target_insession_task_id: str = Field(min_length=1, max_length=128)
    expected_current_graph_revision: int | None = Field(ge=1)


class InSessionTaskGraphValidationResult(_Contract):
    """诊断结果；Store 会重新验证提案及可信上下文。"""

    status: Literal["accepted", "rejected"]
    proposal: NewInSessionTaskGraphsProposal | None = None
    trusted_context: InSessionTaskGraphValidationContext | None = None
    error_codes: tuple[InSessionTaskGraphValidationCode, ...] = ()

    @model_validator(mode="after")
    def _validate_status_shape(self) -> 'InSessionTaskGraphValidationResult':
        if self.status == "accepted" and (
            self.proposal is None or self.trusted_context is None
        ):
            raise ValueError("accepted validation result requires proposal and trusted context")
        if self.status == "rejected" and (
            self.proposal is not None or self.trusted_context is not None
        ):
            raise ValueError("rejected validation result cannot carry proposal or trusted context")
        return self


class InSessionTaskGraphRevisionValidationResult(_Contract):
    """绑定到 Host 所提供 revision authority 的纯验证结果。

    接受只证明提案形状和源引用覆盖，并不证明已存 authority、图 CAS 或节点连续性；Store
    必须在最终 mutation 事务中重新验证这些事实。
    """

    status: Literal["accepted", "rejected"]
    proposal: InSessionTaskGraphRevisionProposal | None = None
    trusted_context: InSessionTaskGraphRevisionValidationContext | None = None
    error_codes: tuple[InSessionTaskGraphValidationCode, ...] = ()

    @model_validator(mode="after")
    def _validate_status_shape(self) -> 'InSessionTaskGraphRevisionValidationResult':
        if self.status == "accepted" and (
            self.proposal is None or self.trusted_context is None
        ):
            raise ValueError(
                "accepted revision validation requires proposal and trusted context"
            )
        if self.status == "rejected" and (
            self.proposal is not None or self.trusted_context is not None
        ):
            raise ValueError(
                "rejected revision validation cannot carry proposal or trusted context"
            )
        return self


class InSessionTaskCatalogItem(_Contract):
    """可安全放入 Entry 分类器 payload 的小型可信根任务事实。"""

    insession_task_id: str = Field(min_length=1, max_length=128)
    goal_summary: str = Field(min_length=1, max_length=2_241)
    status: InSessionTaskStatus
    current_graph_revision: int | None = Field(default=None, ge=1)
# 仅供 Entry 使用的有界上下文。它仍是 Task 匹配输入，绝不是模型编写的恢复标识符或执行授权。
    pending_user_question: str | None = Field(
        default=None,
        min_length=1,
        max_length=2_000,
    )


class InSessionTaskCatalog(_Contract):
    """供 Supervisor 使用的有界 catalog 及显式截断事实。"""

    items: tuple[InSessionTaskCatalogItem, ...] = ()
    truncated: bool = False


class InSessionTaskDetails(_Contract):
    """任务范围、树形读取投影；绝不包含原始数据库记录。"""

    insession_task_id: str = Field(min_length=1, max_length=128)
    session_id: str = Field(min_length=1, max_length=128)
    task_state_version: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=240)
    objective: str = Field(min_length=1, max_length=2_000)
    current_graph_revision: int | None = Field(default=None, ge=1)
    # 已投影 TaskGraph revision 的不可变 provenance。执行组合使用该 Turn——而非调用方当前
    # Turn——在崩溃或后续继续后恢复 revision 范围附件 authority。
    source_turn_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )
    status: InSessionTaskStatus
    nodes: tuple[dict[str, object], ...] = ()
    source_anchors: tuple[InSessionTaskSourceAnchor, ...] = ()
    authorization_anchor_ids: tuple[str, ...] = ()
    required_anchor_ids: tuple[str, ...] = ()
    related_turn_count: int = Field(default=0, ge=0)
    work_run_summaries: tuple[dict[str, object], ...] = ()


class InSessionTaskGraphCommitResult(_Contract):
    """原子新图提交或其精确重放的安全结果。"""

    status: Literal["applied", "replayed"]
    created_insession_task_ids: tuple[str, ...] = ()
    turn_task_link_revision: int = Field(ge=0)
    window_state_version: int | None = Field(default=None, ge=0)


class InSessionTaskGraphRevisionCommitResult(_Contract):
    """一个现有 Task 原子图 revision 提交的安全结果。"""

    status: Literal["applied", "replayed"]
    insession_task_id: str = Field(min_length=1, max_length=128)
    previous_graph_revision: int | None = Field(ge=1)
    committed_graph_revision: int = Field(ge=1)
    task_state_version: int = Field(ge=1)
    turn_task_link_revision: int = Field(ge=0)
    window_state_version: int = Field(ge=0)


class InSessionTaskTurnLinkResult(_Contract):
    """为一个权威 Turn 记录根级相关性的结果。"""

    linked_insession_task_ids: tuple[str, ...] = ()
    turn_task_link_revision: int = Field(ge=0)
    window_state_version: int | None = Field(default=None, ge=0)
