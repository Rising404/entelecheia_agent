"""供权威 Runtime Entry 使用的 Supervisor 任务准入组合。

本模块负责分类完成与 Supervisor 持久化 task-match receipt 之间的狭窄区间。它有意
不持有 entry 生命周期权威：Entry 发出 Supervisor 事件、选择已认证的持久化后
executor 路由，并最终化已接受 Turn。
"""

from __future__ import annotations

from dataclasses import dataclass
from ..context.contracts import EntryContext
from ..ingress.model_contracts import EntryClassification
from .contracts import EntryTaskMatchApplyResult
from ...turn.contracts import AcceptedEntryTurn
from .selection import (
    classification_requires_l2_task_processing,
)
from ..ingress.contracts import IngressDecision
from .policy import ProcessingLevel
from .ports import EntryTaskAdmissionStorePort


@dataclass(frozen=True, slots=True)
class EntryTaskAdmission:
    """由 Entry controller 消费的持久化任务准入事实。"""

    processing_level: ProcessingLevel
    applied: EntryTaskMatchApplyResult | None
    related_insession_task_ids: tuple[str, ...]
    window_revision: int


def admit_entry_task_matches(
    *,
    accepted: AcceptedEntryTurn,
    classification: EntryClassification,
    context: EntryContext,
    ingress: IngressDecision,
    expected_window_revision: int,
    store: EntryTaskAdmissionStorePort | None = None,
) -> EntryTaskAdmission:
    """验证并持久应用一个 classifier task-match 批次（如有）。

    Entry 调用方保留 executor 路由和所有生命周期 effect。稳定 apply id 与完整载荷
    可在事务提交后首个存储响应丢失时保证一次重试安全。

    L1 不接受任何 task-match，也不读取目录或收集长期任务引用。
    processing level 沿用已校验的分类结果；ingress / catalog 截断事实不是另一套
    语义路由规则。返回 window_revision 供 Entry 下一阶段继续做版本校验。
    """

    if classification.processing_level == "L1":
        if classification.task_matches:
            raise RuntimeError("L1 classification cannot reference durable Tasks")
        return EntryTaskAdmission(
            processing_level="L1",
            applied=None,
            related_insession_task_ids=(),
            window_revision=expected_window_revision,
        )

    processing_level = _merge_processing_level(
        ingress,
        classification,
        task_catalog_truncated=context.task_catalog.truncated,
    )
    if processing_level == "L0" and classification_requires_l2_task_processing(
        classification
    ):
        raise RuntimeError("L0 classification cannot request durable Task mutation")

    applied: EntryTaskMatchApplyResult | None = None
    related_insession_task_ids: tuple[str, ...] = ()
    window_revision = expected_window_revision
    if classification.task_matches:
        from ....l2.entry_adapter.task_admission import (
            apply_persisted_task_matches,
        )

        applied = apply_persisted_task_matches(
            session_id=accepted.session_id,
            source_turn_id=accepted.turn_id,
            proposal=classification.task_matches_proposal(),
            exposed_catalog_ids=tuple(
                item.insession_task_id for item in context.task_catalog.items
            ),
            expected_window_revision=window_revision,
            store=store,
        )
        related_insession_task_ids = applied.related_insession_task_ids
        if applied.window_state_version is not None:
            window_revision = applied.window_state_version
    return EntryTaskAdmission(
        processing_level=processing_level,
        applied=applied,
        related_insession_task_ids=related_insession_task_ids,
        window_revision=window_revision,
    )


def _merge_processing_level(
    ingress: IngressDecision,
    classification: EntryClassification,
    *,
    task_catalog_truncated: bool = False,
) -> ProcessingLevel:
    # Ingress 与 catalog 形态约束安全性和上下文真实性，但不解释任务语义，也不覆盖
    # 模型已准入级别。
    del ingress, task_catalog_truncated
    return classification.processing_level


__all__ = [
    "EntryTaskAdmission",
    "admit_entry_task_matches",
]
