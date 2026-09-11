"""任务已挂载文档和 Turn 授权关系的冻结。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from personagraph.l2.task_graph.lane_manifest import (
    InSessionTaskExecutionLaneMatch,
    InSessionTaskExecutionLane,
)
from personagraph.session import store as session_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import task_graph as task_graph_store
from .mounted_document_authority import (
    FrozenMountedDocumentPlanningAuthority,
    freeze_mounted_document_planning_authority,
    recover_task_scoped_managed_document_ids,
)


class AuxiliaryTaskDocumentScopeError(RuntimeError):
    """一个被接受的执行通道不能提供一个精确的任务文件范围。"""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class AuxiliaryTaskDocumentScope:
    session_id: str
    turn_id: str
    task_id: str
    allowed_managed_document_ids: tuple[str, ...]
    mounted_authority: FrozenMountedDocumentPlanningAuthority
    current_lane: InSessionTaskExecutionLane
    target_change_intent: InSessionTaskExecutionLaneMatch | None
    authority_turn_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.mounted_authority.session_id != self.session_id
            or self.mounted_authority.task_id != self.task_id
        ):
            raise ValueError("mounted authority crossed its Task document scope")
        if self.allowed_managed_document_ids != tuple(
            sorted(set(self.allowed_managed_document_ids))
        ):
            raise ValueError("Task managed document ids must be sorted and unique")
        if (
            not self.authority_turn_ids
            or len(self.authority_turn_ids) != len(set(self.authority_turn_ids))
        ):
            raise ValueError("Task attachment authority Turns must be non-empty and unique")


AuxiliaryGraphLoader = Callable[..., object | None]


def prepare_auxiliary_task_document_scope(
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
    store: object = session_store,
    task_store: object | None = None,
    load_auxiliary_graph: AuxiliaryGraphLoader = (
        auxiliary_graph_store.get_auxiliary_graph_for_task
    ),
) -> AuxiliaryTaskDocumentScope:
    """冻结任务已挂载文档与创建/当前 Turn 的授权关系。"""

    if task_store is None:
        task_store = task_graph_store if store is session_store else store
    current_manifest = getattr(
        task_store, "get_insession_task_execution_lane_manifest"
    )(
        session_id=session_id,
        turn_id=turn_id,
    )
    current_lane = next(
        (
            lane
            for lane in current_manifest.lanes
            if lane.insession_task_id == task_id
        ),
        None,
    )
    if current_lane is None or not current_lane.execution_requested:
        raise AuxiliaryTaskDocumentScopeError(
            "attachment_authority_turn_not_authorized_for_task"
        )
    target_change_matches = tuple(
        item
        for item in current_lane.matches
        if item.match_type == "existing_root_target_change"
    )
    if len(target_change_matches) > 1:
        raise AuxiliaryTaskDocumentScopeError(
            "multiple_target_change_lane_intents"
        )
    target_change_intent = (
        target_change_matches[0] if target_change_matches else None
    )

    authority_task = getattr(task_store, "get_insession_task_details")(
        session_id,
        task_id,
    )
    if authority_task is None:
        raise AuxiliaryTaskDocumentScopeError(
            "attachment_authority_task_missing"
        )
    existing_auxiliary = load_auxiliary_graph(
        session_id=session_id,
        insession_task_id=task_id,
    )
    task_managed_document_ids = set(
        recover_task_scoped_managed_document_ids(
            session_id=session_id,
            task_id=task_id,
            authority_snapshot=(
                existing_auxiliary.authority_snapshot
                if existing_auxiliary is not None
                else None
            ),
        )
    )

    creation_source = getattr(task_store, "get_insession_task_creation_source")(
        session_id=session_id,
        insession_task_id=task_id,
    )
    authority_turn_ids = [turn_id]
    if creation_source.source_turn_id != turn_id:
        creation_manifest = getattr(
            task_store, "get_insession_task_execution_lane_manifest"
        )(
            session_id=session_id,
            turn_id=creation_source.source_turn_id,
        )
        creation_lane = next(
            (
                lane
                for lane in creation_manifest.lanes
                if lane.insession_task_id == task_id
            ),
            None,
        )
        if creation_lane is None:
            raise AuxiliaryTaskDocumentScopeError(
                "attachment_authority_creation_turn_not_authorized_for_task"
            )
        authority_turn_ids.insert(0, creation_source.source_turn_id)

    allowed_ids = tuple(sorted(task_managed_document_ids))
    mounted_authority = freeze_mounted_document_planning_authority(
        session_id=session_id,
        task_id=task_id,
        allowed_managed_document_ids=allowed_ids,
    )
    return AuxiliaryTaskDocumentScope(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        allowed_managed_document_ids=allowed_ids,
        mounted_authority=mounted_authority,
        current_lane=current_lane,
        target_change_intent=target_change_intent,
        authority_turn_ids=tuple(authority_turn_ids),
    )


__all__ = (
    "AuxiliaryTaskDocumentScopeError",
    "AuxiliaryTaskDocumentScope",
    "prepare_auxiliary_task_document_scope",
)
