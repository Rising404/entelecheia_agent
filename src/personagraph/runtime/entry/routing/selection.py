"""Choose the public L0/L1/L2 processing route for an accepted Turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...turn.contracts import AcceptedEntryTurn
from ..ingress.model_contracts import EntryClassification
from .contracts import EntryTaskMatchApplyResult
from .policy import ProcessingLevel
from .ports import EntryTaskRoutingAuthorityStorePort


@dataclass(frozen=True, slots=True)
class EntryResponseProcessingRoute:
    """A response lane that does not require a level-specific executor."""

    processing_level: Literal["L0", "L2"]
    kind: Literal["generate_response"] = "generate_response"


@dataclass(frozen=True, slots=True)
class EntryL1ProcessingRoute:
    """The L1 TurnRun lane; its handler freezes all L1-private inputs."""

    kind: Literal["l1"] = "l1"
    processing_level: Literal["L1"] = "L1"


@dataclass(frozen=True, slots=True)
class EntryL2TaskProcessingRoute:
    """An L2 Task lane authenticated by its L2 adapter."""

    task_id: str
    kind: Literal["l2_task"] = "l2_task"
    processing_level: Literal["L2"] = "L2"


EntryProcessingRoute = (
    EntryResponseProcessingRoute | EntryL1ProcessingRoute | EntryL2TaskProcessingRoute
)


def classification_requires_l2_task_processing(
    classification: EntryClassification,
) -> bool:
    """Return whether a classifier proposal cannot safely remain in L0/L1.

    This is a defensive floor for adapters that could bypass the normal model
    validator. A new root, a branch, or execution of an existing Task requires
    L2 processing.
    """

    return any(
        match.match_type in {"new_root", "existing_root_branch"}
        or bool(getattr(match, "execute_current", False))
        for match in classification.task_matches
    )


def select_entry_processing_route(
    *,
    accepted: AcceptedEntryTurn,
    classification: EntryClassification,
    applied: EntryTaskMatchApplyResult | None,
    processing_level: ProcessingLevel,
    store: EntryTaskRoutingAuthorityStorePort | None = None,
) -> EntryProcessingRoute:
    """把已准入 processing_level 转成执行 route DTO，不在此重做语义分类。

    L1 直接选择 EntryL1ProcessingRoute，L0 选择直接回复；本函数不调用模型、
    不执行工具或写状态。后续由 Entry application 解释 DTO 并推进生命周期。
    其他 lane 的专属状态读取仅在该分支被选中后惰性进入对应 adapter。
    """

    if processing_level == "L1":
        return EntryL1ProcessingRoute()
    if processing_level == "L2":
        from personagraph.l2.entry_adapter.task_routing import (
            select_l2_task_processing_target,
        )

        task_id = select_l2_task_processing_target(
            accepted=accepted,
            classification=classification,
            applied=applied,
            store=store,
        )
        if task_id is not None:
            return EntryL2TaskProcessingRoute(task_id=task_id)
        return EntryResponseProcessingRoute(processing_level="L2")
    if processing_level == "L0":
        return EntryResponseProcessingRoute(processing_level="L0")
    raise ValueError(f"unsupported processing level: {processing_level}")


__all__ = [
    "EntryL1ProcessingRoute",
    "EntryL2TaskProcessingRoute",
    "EntryProcessingRoute",
    "EntryResponseProcessingRoute",
    "classification_requires_l2_task_processing",
    "select_entry_processing_route",
]
