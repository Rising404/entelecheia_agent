"""对已验证直接子 TaskNode 交付的纯投影。

调用方提供经 Store 认证的当前目标投影。其来源 Delivery 可以直接持有该目标，也可以依据不可变
沿用收据保留在紧邻的上一图修订版本。本模块绝不会重写历史 Delivery 主体。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Iterable, Never

from personagraph.l2.task_graph.contracts import InSessionTaskStatus
from personagraph.l2.work_run import (
    ResolvedCurrentTaskNodeDelivery,
    ResolvedTaskNodeDelivery,
    TaskNodeSubject,
)
from .dependency_delivery_contracts import (
    TaskNodeDependencyDelivery,
    TaskNodeDependencyDeliveries,
    TaskNodeDependencyInputTooLarge,
    TaskNodeDependencyInputUnsupported,
    TaskNodeDependencyProjection,
    build_task_node_dependency_model_payload,
    serialize_task_node_dependency_model_payload,
    task_node_dependency_serialized_utf8_bytes,
)
from .frontier import TaskNodeTree
from .input_limits import TaskNodeDependencyInputLimits


class TaskNodeDependencyProjectionErrorCode(StrEnum):
    PARENT_MISMATCH = "parent_mismatch"
    CHILD_NOT_COMPLETED = "child_not_completed"
    MISSING_DELIVERY = "missing_delivery"
    EXTRA_DELIVERY = "extra_delivery"
    WRONG_SUBJECT = "wrong_subject"
    WRONG_NODE_REVISION = "wrong_node_revision"
    DUPLICATE_DELIVERY = "duplicate_delivery"


class TaskNodeDependencyProjectionError(ValueError):
    """直接子交付绑定中的确定性失败关闭错误。"""

    def __init__(
        self,
        code: TaskNodeDependencyProjectionErrorCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(message)


def project_task_node_dependencies(
    tree: TaskNodeTree,
    *,
    parent_subject: TaskNodeSubject,
    resolved_deliveries: Iterable[
        ResolvedCurrentTaskNodeDelivery | ResolvedTaskNodeDelivery
    ],
) -> TaskNodeDependencyProjection:
    """精确绑定父节点的当前直接子 Delivery。"""

    node_by_id = {item.subject.node_id: item for item in tree.nodes}
    parent = node_by_id.get(parent_subject.node_id)
    if parent is None or parent.subject != parent_subject:
        _fail(
            TaskNodeDependencyProjectionErrorCode.PARENT_MISMATCH,
            "parent subject is not a current TaskGraph member",
        )

    child_ids = tuple(
        sorted(
            parent.child_node_ids,
            key=lambda node_id: (node_by_id[node_id].ordinal, node_id),
        )
    )
    if any(
        node_by_id[node_id].status is not InSessionTaskStatus.COMPLETED
        for node_id in child_ids
    ):
        _fail(
            TaskNodeDependencyProjectionErrorCode.CHILD_NOT_COMPLETED,
            "all direct children must be completed before dependency projection",
        )

    deliveries = tuple(
        item
        if isinstance(item, ResolvedCurrentTaskNodeDelivery)
        else ResolvedCurrentTaskNodeDelivery.direct(item)
        for item in resolved_deliveries
    )
    delivery_ids = [item.delivery_id for item in deliveries]
    delivery_node_ids = [item.target_subject.node_id for item in deliveries]
    if (
        len(delivery_ids) != len(set(delivery_ids))
        or len(delivery_node_ids) != len(set(delivery_node_ids))
    ):
        _fail(
            TaskNodeDependencyProjectionErrorCode.DUPLICATE_DELIVERY,
            "dependency Delivery IDs and child bindings must be unique",
        )

    expected = set(child_ids)
    by_child: dict[str, ResolvedCurrentTaskNodeDelivery] = {}
    for resolved in deliveries:
        subject = resolved.target_subject
        if (
            subject.task_id != tree.task_id
            or subject.graph_revision != tree.graph_revision
        ):
            _fail(
                TaskNodeDependencyProjectionErrorCode.WRONG_SUBJECT,
                "dependency Delivery belongs to another Task or graph revision",
            )
        if subject.node_id not in expected:
            _fail(
                TaskNodeDependencyProjectionErrorCode.EXTRA_DELIVERY,
                "dependency Delivery is not owned by a direct child",
            )
        expected_subject = node_by_id[subject.node_id].subject
        if subject.node_revision != expected_subject.node_revision:
            _fail(
                TaskNodeDependencyProjectionErrorCode.WRONG_NODE_REVISION,
                "dependency Delivery has a stale child node revision",
            )
        if subject != expected_subject:
            _fail(
                TaskNodeDependencyProjectionErrorCode.WRONG_SUBJECT,
                "dependency Delivery subject does not match the current child",
            )
        by_child[subject.node_id] = resolved

    missing = expected.difference(by_child)
    if missing:
        _fail(
            TaskNodeDependencyProjectionErrorCode.MISSING_DELIVERY,
            "one or more direct-child Deliveries are missing",
        )

    return TaskNodeDependencyProjection(
        parent_subject=parent_subject,
        items=tuple(
            TaskNodeDependencyDelivery(
                child_ordinal=node_by_id[node_id].ordinal,
                resolved_current_delivery=by_child[node_id],
            )
            for node_id in child_ids
        ),
    )


def _fail(
    code: TaskNodeDependencyProjectionErrorCode,
    message: str,
) -> Never:
    raise TaskNodeDependencyProjectionError(code, message)


__all__ = [
    'TaskNodeDependencyDelivery',
    'TaskNodeDependencyDeliveries',
    'TaskNodeDependencyInputLimits',
    "TaskNodeDependencyInputTooLarge",
    "TaskNodeDependencyInputUnsupported",
    "TaskNodeDependencyProjectionError",
    "TaskNodeDependencyProjectionErrorCode",
    'TaskNodeDependencyProjection',
    "build_task_node_dependency_model_payload",
    "project_task_node_dependencies",
    "serialize_task_node_dependency_model_payload",
    "task_node_dependency_serialized_utf8_bytes",
]
