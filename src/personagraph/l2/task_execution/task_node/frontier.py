"""纯 TaskGraph 前沿投影与 Task 状态归约。

本模块消费可信任务详情投影，以及 Store 准备的少量事实。它不查询持久层、不验证 Delivery 链、
不认领 WorkRun，也不完成 Task。Store 侧 CAS 仍是执行权威。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Iterable, Never

from pydantic import BaseModel, ConfigDict, Field

from personagraph.l2.task_graph.contracts import (
    InSessionTaskDetails,
    InSessionTaskNodeKind,
    InSessionTaskStatus,
)
from personagraph.l2.work_run import TaskNodeSubject


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskNodeFrontierErrorCode(StrEnum):
    GRAPH_MISSING = "graph_missing"
    INVALID_NODE = "invalid_node"
    DUPLICATE_NODE_ID = "duplicate_node_id"
    INVALID_ORDINALS = "invalid_ordinals"
    INVALID_ROOT = "invalid_root"
    INVALID_PARENT = "invalid_parent"
    CYCLE_OR_ORPHAN = "cycle_or_orphan"
    FACT_COVERAGE_MISMATCH = "fact_coverage_mismatch"
    INVALID_RUNTIME_FACT = "invalid_runtime_fact"


class TaskNodeFrontierError(ValueError):
    """前沿输入投影中的确定性失败关闭错误。"""

    def __init__(self, code: TaskNodeFrontierErrorCode, message: str) -> None:
        self.code = code
        super().__init__(message)


class TaskNodeTreeItem(_Contract):
    subject: TaskNodeSubject
    node_kind: InSessionTaskNodeKind
    ordinal: int = Field(ge=0)
    parent_node_id: str | None = None
    child_node_ids: tuple[str, ...] = ()
    status: InSessionTaskStatus
    state_version: int = Field(ge=1)


class TaskNodeTree(_Contract):
    task_id: str = Field(min_length=1)
    graph_revision: int = Field(ge=1)
    task_status: InSessionTaskStatus
    task_state_version: int = Field(ge=1)
    root_node_id: str = Field(min_length=1)
    nodes: tuple[TaskNodeTreeItem, ...] = Field(min_length=1)


class TaskNodeRuntimeFact(_Contract):
    """由 Store 建立的事实；其存在即具权威性，此处不再验证。

    ``recoverable_work_run_id`` 表示已有一个精确的非终态 WorkRun 持有此节点，因此不得创建新的
    WorkRun。``current_delivery_id`` 表示 Store 已为此当前节点修订版本完整解析出 PASS/冻结
    Delivery。
    """

    node_id: str = Field(min_length=1)
    recoverable_work_run_id: str | None = Field(default=None, min_length=1)
    current_delivery_id: str | None = Field(default=None, min_length=1)


class ReadyFreshTaskNode(_Contract):
    subject: TaskNodeSubject
    node_status: InSessionTaskStatus
    node_state_version: int = Field(ge=1)
    ordinal: int = Field(ge=0)
    dependency_node_ids: tuple[str, ...] = ()
    dependency_delivery_ids: tuple[str, ...] = ()


class TaskAggregateReason(StrEnum):
    TERMINAL_PRESERVED = "terminal_preserved"
    FINISH_GATE_REQUIRED = "finish_gate_required"
    PRISTINE = "pristine"
    RUNNABLE = "runnable"
    AWAITING_USER = "awaiting_user"
    WAITING_EXTERNAL = "waiting_external"
    BLOCKED = "blocked"


class TaskAggregateProjection(_Contract):
    status: InSessionTaskStatus
    reason: TaskAggregateReason
    ready_fresh_nodes: tuple[ReadyFreshTaskNode, ...] = ()
    recoverable_node_ids: tuple[str, ...] = ()
    awaiting_user_node_ids: tuple[str, ...] = ()
    waiting_external_node_ids: tuple[str, ...] = ()
    finish_gate_candidate: bool = False


def build_task_node_tree(details: InSessionTaskDetails) -> TaskNodeTree:
    """将当前详情投影解析并验证为一棵规范树。"""

    if details.current_graph_revision is None or not details.nodes:
        _fail(TaskNodeFrontierErrorCode.GRAPH_MISSING, "Task has no current graph")

    parsed: list[dict[str, object]] = []
    for raw in details.nodes:
        try:
            node_id = _nonempty_string(raw["insession_task_node_id"])
            revision = _positive_int(raw["node_revision"])
            kind = InSessionTaskNodeKind(raw["node_kind"])
            ordinal = _nonnegative_int(raw["ordinal"])
            parent_raw = raw["parent_insession_task_node_id"]
            parent_id = None if parent_raw is None else _nonempty_string(parent_raw)
            status = InSessionTaskStatus(raw["status"])
            state_version = _positive_int(raw["state_version"])
        except (KeyError, TypeError, ValueError):
            _fail(TaskNodeFrontierErrorCode.INVALID_NODE, "invalid TaskNode detail")
        parsed.append(
            {
                "node_id": node_id,
                "revision": revision,
                "kind": kind,
                "ordinal": ordinal,
                "parent_id": parent_id,
                "status": status,
                "state_version": state_version,
            }
        )

    node_ids = [str(item["node_id"]) for item in parsed]
    if len(node_ids) != len(set(node_ids)):
        _fail(TaskNodeFrontierErrorCode.DUPLICATE_NODE_ID, "duplicate TaskNode ID")
    ordinals = [int(item["ordinal"]) for item in parsed]
    if len(ordinals) != len(set(ordinals)) or set(ordinals) != set(range(len(parsed))):
        _fail(TaskNodeFrontierErrorCode.INVALID_ORDINALS, "ordinals must be unique and contiguous")

    roots = [item for item in parsed if item["kind"] is InSessionTaskNodeKind.ROOT]
    if (
        len(roots) != 1
        or roots[0]["node_id"] != details.insession_task_id
        or roots[0]["parent_id"] is not None
    ):
        _fail(TaskNodeFrontierErrorCode.INVALID_ROOT, "TaskGraph requires one canonical root")

    by_id = {str(item["node_id"]): item for item in parsed}
    children: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for item in parsed:
        if item["kind"] is InSessionTaskNodeKind.ROOT:
            continue
        parent_id = item["parent_id"]
        if parent_id is None or parent_id not in by_id or parent_id == item["node_id"]:
            _fail(TaskNodeFrontierErrorCode.INVALID_PARENT, "subtask parent is invalid")
        children[str(parent_id)].append(str(item["node_id"]))

    for child_ids in children.values():
        child_ids.sort(key=lambda node_id: (int(by_id[node_id]["ordinal"]), node_id))
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visited:
            _fail(TaskNodeFrontierErrorCode.CYCLE_OR_ORPHAN, "TaskGraph is cyclic")
        visited.add(node_id)
        for child_id in children[node_id]:
            visit(child_id)

    visit(details.insession_task_id)
    if visited != set(node_ids):
        _fail(TaskNodeFrontierErrorCode.CYCLE_OR_ORPHAN, "TaskGraph contains an orphan")

    ordered = sorted(parsed, key=lambda item: (int(item["ordinal"]), str(item["node_id"])))
    return TaskNodeTree(
        task_id=details.insession_task_id,
        graph_revision=details.current_graph_revision,
        task_status=details.status,
        task_state_version=details.task_state_version,
        root_node_id=details.insession_task_id,
        nodes=tuple(
            TaskNodeTreeItem(
                subject=TaskNodeSubject(
                    task_id=details.insession_task_id,
                    graph_revision=details.current_graph_revision,
                    node_id=str(item["node_id"]),
                    node_revision=int(item["revision"]),
                ),
                node_kind=item["kind"],
                ordinal=int(item["ordinal"]),
                parent_node_id=item["parent_id"],
                child_node_ids=tuple(children[str(item["node_id"])]),
                status=item["status"],
                state_version=int(item["state_version"]),
            )
            for item in ordered
        ),
    )


def project_ready_fresh_nodes(
    tree: TaskNodeTree,
    facts: Iterable[TaskNodeRuntimeFact],
) -> tuple[ReadyFreshTaskNode, ...]:
    """按稳定的 ``(ordinal, node_id)`` 顺序返回全新启动候选。"""

    fact_by_node = _validated_facts(tree, facts)
    node_by_id = {item.subject.node_id: item for item in tree.nodes}
    ready: list[ReadyFreshTaskNode] = []
    for node in tree.nodes:
        fact = fact_by_node[node.subject.node_id]
        dependency_facts = [fact_by_node[node_id] for node_id in node.child_node_ids]
        dependencies_ready = all(
            node_by_id[node_id].status is InSessionTaskStatus.COMPLETED
            and dependency.current_delivery_id is not None
            for node_id, dependency in zip(node.child_node_ids, dependency_facts)
        )
        if (
            node.status
            in {
                InSessionTaskStatus.PROPOSED,
                InSessionTaskStatus.ACTIVE,
                InSessionTaskStatus.INTERRUPTED,
            }
            and fact.recoverable_work_run_id is None
            and dependencies_ready
        ):
            ready.append(
                ReadyFreshTaskNode(
                    subject=node.subject,
                    node_status=node.status,
                    node_state_version=node.state_version,
                    ordinal=node.ordinal,
                    dependency_node_ids=node.child_node_ids,
                    dependency_delivery_ids=tuple(
                        dependency.current_delivery_id
                        for dependency in dependency_facts
                        if dependency.current_delivery_id is not None
                    ),
                )
            )
    return tuple(ready)


def reduce_task_aggregate(
    tree: TaskNodeTree,
    facts: Iterable[TaskNodeRuntimeFact],
) -> TaskAggregateProjection:
    """推导 Task 状态，但绝不宣告 FinishGate 成功。"""

    fact_tuple = tuple(facts)
    fact_by_node = _validated_facts(tree, fact_tuple)
    ready = project_ready_fresh_nodes(tree, fact_tuple)
    ordered = tree.nodes
    recoverable = tuple(
        node.subject.node_id
        for node in ordered
        if fact_by_node[node.subject.node_id].recoverable_work_run_id is not None
    )
    awaiting = tuple(
        node.subject.node_id
        for node in ordered
        if node.status is InSessionTaskStatus.AWAITING_USER
    )
    waiting = tuple(
        node.subject.node_id
        for node in ordered
        if node.status is InSessionTaskStatus.WAITING_EXTERNAL
    )
    common = {
        "ready_fresh_nodes": ready,
        "recoverable_node_ids": recoverable,
        "awaiting_user_node_ids": awaiting,
        "waiting_external_node_ids": waiting,
    }

    if tree.task_status in {InSessionTaskStatus.COMPLETED, InSessionTaskStatus.CANCELLED}:
        return TaskAggregateProjection(
            status=tree.task_status,
            reason=TaskAggregateReason.TERMINAL_PRESERVED,
            **common,
        )
    if all(node.status is InSessionTaskStatus.COMPLETED for node in ordered):
        return TaskAggregateProjection(
            status=InSessionTaskStatus.ACTIVE,
            reason=TaskAggregateReason.FINISH_GATE_REQUIRED,
            finish_gate_candidate=True,
            **common,
        )

    immediately_recoverable = any(
        fact_by_node[node.subject.node_id].recoverable_work_run_id is not None
        and node.status in {InSessionTaskStatus.ACTIVE, InSessionTaskStatus.INTERRUPTED}
        for node in ordered
    )
    if ready or immediately_recoverable:
        pristine = (
            tree.task_status is InSessionTaskStatus.PROPOSED
            and all(node.status is InSessionTaskStatus.PROPOSED for node in ordered)
            and not recoverable
        )
        return TaskAggregateProjection(
            status=(InSessionTaskStatus.PROPOSED if pristine else InSessionTaskStatus.ACTIVE),
            reason=(TaskAggregateReason.PRISTINE if pristine else TaskAggregateReason.RUNNABLE),
            **common,
        )
    if awaiting:
        return TaskAggregateProjection(
            status=InSessionTaskStatus.AWAITING_USER,
            reason=TaskAggregateReason.AWAITING_USER,
            **common,
        )
    if waiting:
        return TaskAggregateProjection(
            status=InSessionTaskStatus.WAITING_EXTERNAL,
            reason=TaskAggregateReason.WAITING_EXTERNAL,
            **common,
        )
    return TaskAggregateProjection(
        status=InSessionTaskStatus.BLOCKED,
        reason=TaskAggregateReason.BLOCKED,
        **common,
    )


def _validated_facts(
    tree: TaskNodeTree,
    facts: Iterable[TaskNodeRuntimeFact],
) -> dict[str, TaskNodeRuntimeFact]:
    fact_tuple = tuple(facts)
    fact_by_node = {fact.node_id: fact for fact in fact_tuple}
    node_by_id = {node.subject.node_id: node for node in tree.nodes}
    node_ids = set(node_by_id)
    if len(fact_by_node) != len(fact_tuple) or set(fact_by_node) != node_ids:
        _fail(TaskNodeFrontierErrorCode.FACT_COVERAGE_MISMATCH, "runtime facts must exactly cover current nodes")
    delivery_ids = [fact.current_delivery_id for fact in fact_tuple if fact.current_delivery_id]
    run_ids = [fact.recoverable_work_run_id for fact in fact_tuple if fact.recoverable_work_run_id]
    if len(delivery_ids) != len(set(delivery_ids)) or len(run_ids) != len(set(run_ids)):
        _fail(TaskNodeFrontierErrorCode.INVALID_RUNTIME_FACT, "runtime fact IDs must be unique")
    for node in tree.nodes:
        fact = fact_by_node[node.subject.node_id]
        completed = node.status is InSessionTaskStatus.COMPLETED
        if completed != (fact.current_delivery_id is not None):
            _fail(TaskNodeFrontierErrorCode.INVALID_RUNTIME_FACT, "only completed nodes may own a current Delivery, and each completed node requires one")
        if fact.recoverable_work_run_id is not None and node.status not in {
            InSessionTaskStatus.ACTIVE,
            InSessionTaskStatus.AWAITING_USER,
            InSessionTaskStatus.WAITING_EXTERNAL,
            InSessionTaskStatus.INTERRUPTED,
        }:
            _fail(TaskNodeFrontierErrorCode.INVALID_RUNTIME_FACT, "recoverable WorkRun conflicts with node status")
        if node.status in {InSessionTaskStatus.AWAITING_USER, InSessionTaskStatus.WAITING_EXTERNAL} and fact.recoverable_work_run_id is None:
            _fail(TaskNodeFrontierErrorCode.INVALID_RUNTIME_FACT, "wait state requires a recoverable WorkRun")
        if fact.recoverable_work_run_id is not None and any(
            node_by_id[child_id].status is not InSessionTaskStatus.COMPLETED
            or fact_by_node[child_id].current_delivery_id is None
            for child_id in node.child_node_ids
        ):
            _fail(
                TaskNodeFrontierErrorCode.INVALID_RUNTIME_FACT,
                "recoverable WorkRun has incomplete direct dependencies",
            )
    return fact_by_node


def _nonempty_string(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError
    return value


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError
    return value


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError
    return value


def _fail(code: TaskNodeFrontierErrorCode, message: str) -> Never:
    raise TaskNodeFrontierError(code, message)


__all__ = [
    'ReadyFreshTaskNode',
    'TaskAggregateProjection',
    'TaskAggregateReason',
    "TaskNodeFrontierError",
    "TaskNodeFrontierErrorCode",
    'TaskNodeRuntimeFact',
    'TaskNodeTreeItem',
    'TaskNodeTree',
    "build_task_node_tree",
    "project_ready_fresh_nodes",
    "reduce_task_aggregate",
]
