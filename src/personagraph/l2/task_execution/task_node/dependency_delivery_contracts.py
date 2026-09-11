"""面向模型输入的冻结 TaskNode 子 Delivery 契约。

本模块只持有已经 Store 认证的交付正文、其规范模型载荷和精确输入大小守卫。它刻意不重建
TaskGraph 前沿、不决定哪些子节点为当前版本、不读取 Store，也不调用模型。
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personagraph.l2.work_run.contracts import TaskNodeSubject
from personagraph.l2.work_run.verification import (
    CurrentTaskNodeDeliveryResolutionKind,
    ResolvedCurrentTaskNodeDelivery,
    ResolvedTaskNodeDelivery,
)
from .input_limits import TaskNodeDependencyInputLimits


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskNodeDependencyInputTooLarge(RuntimeError):
    code = "task_node_dependency_input_too_large"

    def __init__(
        self,
        *,
        item_count: int,
        serialized_utf8_bytes: int,
        limits: TaskNodeDependencyInputLimits,
    ) -> None:
        self.item_count = item_count
        self.serialized_utf8_bytes = serialized_utf8_bytes
        self.limits = limits
        super().__init__(
            "TaskNode dependency input exceeds its configured Host limits: "
            f"items={item_count}/{limits.max_items}, "
            "serialized_utf8_bytes="
            f"{serialized_utf8_bytes}/{limits.max_serialized_utf8_bytes}, "
            f"profile_id={limits.profile_id}"
        )


class TaskNodeDependencyInputUnsupported(RuntimeError):
    code = "task_node_dependency_input_unsupported"

    def __init__(self, *, limits: TaskNodeDependencyInputLimits) -> None:
        self.limits = limits
        super().__init__(
            "TaskNode dependency input is not canonical JSON/UTF-8: "
            f"profile_id={limits.profile_id}"
        )


class TaskNodeDependencyDelivery(_Contract):
    """对一项由 Store 解析的不可变子 Delivery 的有序引用。"""

    child_ordinal: int = Field(ge=0)
    resolved_current_delivery: ResolvedCurrentTaskNodeDelivery

    @property
    def child_subject(self) -> TaskNodeSubject:
        return self.resolved_current_delivery.target_subject

    @property
    def delivery_id(self) -> str:
        return self.resolved_current_delivery.delivery_id

    @property
    def resolved_delivery(self) -> ResolvedTaskNodeDelivery:
        """返回不可变来源正文，而不改变其主体。"""

        return self.resolved_current_delivery.source_delivery


class TaskNodeDependencyDeliveries(_Contract):
    """独立提供给一个模型边界的冻结瞬态正文。"""

    items: tuple[TaskNodeDependencyDelivery, ...] = ()

    @model_validator(mode="after")
    def _require_unique_delivery_and_child_bindings(
        self,
    ) -> 'TaskNodeDependencyDeliveries':
        delivery_ids = [item.delivery_id for item in self.items]
        child_subjects = [item.child_subject for item in self.items]
        if len(delivery_ids) != len(set(delivery_ids)):
            raise ValueError("dependency Delivery IDs must be unique")
        if len(child_subjects) != len(set(child_subjects)):
            raise ValueError("dependency child subjects must be unique")
        return self

    @property
    def delivery_ids(self) -> tuple[str, ...]:
        return tuple(item.delivery_id for item in self.items)


class TaskNodeDependencyProjection(_Contract):
    """可由 Attempt 与 Verifier 复用的精确直接子依赖集。"""

    parent_subject: TaskNodeSubject
    items: tuple[TaskNodeDependencyDelivery, ...] = ()

    @property
    def delivery_ids(self) -> tuple[str, ...]:
        return tuple(item.delivery_id for item in self.items)

    def to_model_deliveries(self) -> TaskNodeDependencyDeliveries:
        return TaskNodeDependencyDeliveries(items=self.items)


def build_task_node_dependency_model_payload(
    projection: TaskNodeDependencyProjection | TaskNodeDependencyDeliveries,
) -> dict[str, Any]:
    """构建两个模型调用点嵌入的精确瞬态载荷。"""

    return {
        "dependency_deliveries": [
            {
                "child_ordinal": item.child_ordinal,
                "delivery_id": item.delivery_id,
                "subject": item.child_subject.model_dump(mode="json"),
                "delivery_resolution": _delivery_resolution_payload(item),
                "output_window": {
                    "output_revision": (
                        item.resolved_delivery.output_window.output_revision
                    ),
                    "format": item.resolved_delivery.output_window.format.value,
                    "content": item.resolved_delivery.output_window.content,
                },
            }
            for item in projection.items
        ]
    }


def _delivery_resolution_payload(
    item: TaskNodeDependencyDelivery,
) -> dict[str, Any]:
    current = item.resolved_current_delivery
    payload: dict[str, Any] = {
        "kind": current.resolution_kind.value,
        "target_subject": current.target_subject.model_dump(mode="json"),
    }
    if current.resolution_kind is CurrentTaskNodeDeliveryResolutionKind.CARRIED:
        authority = current.carry_authority
        assert authority is not None
        payload.update(
            {
                "source_subject": current.source_delivery.delivery.subject.model_dump(
                    mode="json"
                ),
                "carry_receipt_id": authority.carry_receipt_id,
                "carry_receipt_sha256": authority.carry_receipt_sha256,
                "carry_apply_id": authority.apply_id,
            }
        )
    return payload


def serialize_task_node_dependency_model_payload(
    projection: TaskNodeDependencyProjection | TaskNodeDependencyDeliveries,
    *,
    limits: TaskNodeDependencyInputLimits,
) -> str:
    """序列化并守卫完整载荷，不做截断或选择。"""

    try:
        serialized = json.dumps(
            build_task_node_dependency_model_payload(projection),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        serialized_utf8_bytes = len(serialized.encode("utf-8"))
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError) as exc:
        raise TaskNodeDependencyInputUnsupported(limits=limits) from exc

    item_count = len(projection.items)
    if (
        item_count > limits.max_items
        or serialized_utf8_bytes > limits.max_serialized_utf8_bytes
    ):
        raise TaskNodeDependencyInputTooLarge(
            item_count=item_count,
            serialized_utf8_bytes=serialized_utf8_bytes,
            limits=limits,
        )
    return serialized


def task_node_dependency_serialized_utf8_bytes(
    projection: TaskNodeDependencyProjection | TaskNodeDependencyDeliveries,
    *,
    limits: TaskNodeDependencyInputLimits,
) -> int:
    """在强制执行所提供配置后返回精确字节数。"""

    return len(
        serialize_task_node_dependency_model_payload(
            projection,
            limits=limits,
        ).encode("utf-8")
    )


__all__ = [
    'TaskNodeDependencyDelivery',
    'TaskNodeDependencyDeliveries',
    "TaskNodeDependencyInputTooLarge",
    "TaskNodeDependencyInputUnsupported",
    'TaskNodeDependencyProjection',
    "build_task_node_dependency_model_payload",
    "serialize_task_node_dependency_model_payload",
    "task_node_dependency_serialized_utf8_bytes",
]
