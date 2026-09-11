"""供运行时单元测试共享的小型权威 TaskNode 来源夹具。"""

from __future__ import annotations

from personagraph.l2.task_graph.contracts import (
    InSessionTaskAcceptanceProposal,
    InSessionTaskDetails,
    InSessionTaskSourceAnchor,
    InSessionTaskStatus,
)
from personagraph.l2.task_execution.task_node.source_context import (
    TaskNodeSourceContext,
    build_task_node_source_context,
)
from personagraph.l2.work_run import TaskNodeSubject


def task_node_source_context(
    *,
    session_id: str,
    subject: TaskNodeSubject,
    acceptances: tuple[InSessionTaskAcceptanceProposal, ...],
) -> TaskNodeSourceContext:
    anchor_ids = tuple(
        dict.fromkeys(
            source_id
            for acceptance in acceptances
            for source_id in acceptance.source_anchor_ids
        )
    )
    anchors = tuple(
        InSessionTaskSourceAnchor(
            anchor_id=anchor_id,
            source_turn_id="turn-source",
            source_kind="current_user_context",
            start=0,
            end=len(f"source excerpt for {anchor_id}"),
            excerpt=f"source excerpt for {anchor_id}",
        )
        for anchor_id in anchor_ids
    )
    return build_task_node_source_context(
        details=InSessionTaskDetails(
            insession_task_id=subject.task_id,
            session_id=session_id,
            task_state_version=1,
            title="test task",
            objective="test objective",
            current_graph_revision=subject.graph_revision,
            status=InSessionTaskStatus.ACTIVE,
            nodes=(
                {
                    "insession_task_node_id": subject.node_id,
                    "node_revision": subject.node_revision,
                    "source_anchor_ids": list(anchor_ids),
                    "acceptance_criteria": [
                        item.model_dump(mode="json") for item in acceptances
                    ],
                },
            ),
            source_anchors=anchors,
            authorization_anchor_ids=anchor_ids,
            required_anchor_ids=anchor_ids,
        ),
        session_id=session_id,
        subject=subject,
    )
