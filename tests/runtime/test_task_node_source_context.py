from __future__ import annotations

import hashlib

import pytest

from personagraph.l2.task_graph.contracts import (
    InSessionTaskAcceptanceProposal,
    InSessionTaskDetails,
    InSessionTaskSourceAnchor,
    InSessionTaskStatus,
)
from personagraph.l2.task_execution.task_node.source_context import (
    TaskNodeSourceContextAuthorityError,
    TaskNodeSourceContext,
    build_task_node_source_context,
)
from personagraph.l2.work_run import TaskNodeSubject


def _subject(*, graph_revision: int = 3, node_revision: int = 2) -> TaskNodeSubject:
    return TaskNodeSubject(
        task_id="task-1",
        graph_revision=graph_revision,
        node_id="node-1",
        node_revision=node_revision,
    )


def _details(
    *,
    node_source_anchor_ids: tuple[str, ...] = ("request", "document_fact"),
    acceptance_source_anchor_ids: tuple[str, ...] = ("document_fact",),
) -> InSessionTaskDetails:
    return InSessionTaskDetails(
        insession_task_id="task-1",
        session_id="session-1",
        task_state_version=7,
        title="分析材料",
        objective="基于材料给出答案",
        current_graph_revision=3,
        status=InSessionTaskStatus.ACTIVE,
        nodes=(
            {
                "insession_task_node_id": "node-1",
                "node_revision": 2,
                "node_kind": "root",
                "ordinal": 0,
                "title": "回答问题",
                "objective": "只使用授权材料回答",
                "source_anchor_ids": list(node_source_anchor_ids),
                "acceptance_criteria": [
                    {
                        "acceptance_id": "grounded_answer",
                        "criterion": "答案与材料一致",
                        "source_anchor_ids": list(acceptance_source_anchor_ids),
                    }
                ],
                "constraints": [],
                "parent_insession_task_node_id": None,
                "status": "active",
                "state_version": 4,
            },
        ),
        source_anchors=(
            InSessionTaskSourceAnchor(
                anchor_id="request",
                source_turn_id="turn-1",
                source_kind="previously_authorized_task_state",
                start=0,
                end=5,
                excerpt="分析材料。",
            ),
            InSessionTaskSourceAnchor(
                anchor_id="document_fact",
                source_turn_id="turn-2",
                source_kind="retrieved_document",
                start=0,
                end=18,
                excerpt="Birch accuracy is 84%.",
            ),
            InSessionTaskSourceAnchor(
                anchor_id="unrelated_secret",
                source_turn_id="turn-2",
                source_kind="retrieved_document",
                start=0,
                end=22,
                excerpt="This must not be exposed.",
            ),
        ),
        authorization_anchor_ids=("request",),
        required_anchor_ids=("request",),
    )


def test_projects_only_node_scoped_exact_source_excerpts_and_seals_authority() -> None:
    first = build_task_node_source_context(
        session_id="session-1",
        details=_details(),
        subject=_subject(),
    )
    replay = build_task_node_source_context(
        session_id="session-1",
        details=_details(),
        subject=_subject(),
    )

    assert first == replay
    assert first.authority_sha256 == replay.authority_sha256
    assert tuple(item.anchor_id for item in first.anchors) == (
        "request",
        "document_fact",
    )
    assert first.anchors[1].excerpt == "Birch accuracy is 84%."
    assert first.anchors[1].source_kind == "retrieved_document"
    assert first.anchors[1].excerpt_sha256 == hashlib.sha256(
        b"Birch accuracy is 84%."
    ).hexdigest()
    assert first.anchors[0].authorization is True
    assert first.anchors[0].required is True
    assert first.anchors[1].authorization is False
    assert "unrelated_secret" not in first.model_dump_json()
    assert "This must not be exposed" not in first.model_dump_json()


@pytest.mark.parametrize(
    ("details", "subject", "reason"),
    (
        (
            _details(node_source_anchor_ids=("request", "unknown")),
            _subject(),
            "unknown_source_anchor",
        ),
        (
            _details(node_source_anchor_ids=()),
            _subject(),
            "missing_source_anchor",
        ),
        (
            _details(acceptance_source_anchor_ids=("unrelated_secret",)),
            _subject(),
            "acceptance_source_overreach",
        ),
        (
            _details(),
            _subject(graph_revision=2),
            "stale_graph_authority",
        ),
        (
            _details(),
            _subject(node_revision=1),
            "stale_node_authority",
        ),
    ),
)
def test_unknown_overreaching_or_stale_source_authority_fails_closed(
    details: InSessionTaskDetails,
    subject: TaskNodeSubject,
    reason: str,
) -> None:
    with pytest.raises(TaskNodeSourceContextAuthorityError) as error:
        build_task_node_source_context(
            session_id="session-1",
            details=details,
            subject=subject,
        )

    assert error.value.code == "task_node_source_context_unavailable"
    assert error.value.reason == reason


def test_context_rejects_a_forged_authority_hash() -> None:
    valid = build_task_node_source_context(
        session_id="session-1",
        details=_details(),
        subject=_subject(),
    )
    payload = valid.model_dump(mode="python")
    payload["authority_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="authority hash"):
        TaskNodeSourceContext.model_validate(payload)

    forged_copy = valid.model_copy(update={"authority_sha256": "0" * 64})
    with pytest.raises(ValueError, match="authority hash"):
        forged_copy.require_exact_binding(
            session_id="session-1",
            subject=_subject(),
            acceptances=tuple(
                InSessionTaskAcceptanceProposal.model_validate(item)
                for item in _details().nodes[0]["acceptance_criteria"]
            ),
        )


def test_context_cannot_be_reused_with_different_acceptance_source_authority() -> None:
    valid = build_task_node_source_context(
        session_id="session-1",
        details=_details(),
        subject=_subject(),
    )
    retargeted_acceptance = InSessionTaskAcceptanceProposal(
        acceptance_id="grounded_answer",
        criterion="答案与材料一致",
        source_anchor_ids=("request",),
    )

    with pytest.raises(ValueError, match="Acceptance authority"):
        valid.require_exact_binding(
            session_id="session-1",
            subject=_subject(),
            acceptances=(retargeted_acceptance,),
        )


def test_preserves_a_typed_blocking_gap_in_the_exact_projection() -> None:
    base = _details()
    gap = InSessionTaskSourceAnchor(
        anchor_id="document_gap",
        source_turn_id="turn-2",
        source_kind="gap",
        gap_blocking=True,
        start=0,
        end=27,
        excerpt="Page 4 could not be parsed.",
    )
    node = dict(base.nodes[0])
    node["source_anchor_ids"] = ["request", "document_gap"]
    node["acceptance_criteria"] = [
        {
            "acceptance_id": "grounded_answer",
            "criterion": "答案披露材料缺口",
            "source_anchor_ids": ["request", "document_gap"],
        }
    ]
    details = base.model_copy(
        update={
            "nodes": (node,),
            "source_anchors": (*base.source_anchors, gap),
        }
    )

    projected = build_task_node_source_context(
        session_id="session-1",
        details=details,
        subject=_subject(),
    )

    assert projected.anchors[-1].source_kind == "gap"
    assert projected.anchors[-1].gap_blocking is True
    assert projected.model_dump(mode="json")["anchors"][-1]["gap_blocking"] is True
