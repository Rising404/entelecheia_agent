from __future__ import annotations

import hashlib
import json

import pytest

from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from tests.session.test_work_execution_persistence import (
    _commit_passing_verification,
    _seed_task_node,
)


def _freeze_revision_hash(task_id: str, value: str = "base source") -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_task_graph_revisions SET proposal_hash=? "
            "WHERE insession_task_id=? AND graph_revision=1",
            (digest, task_id),
        )
    return digest


def test_projection_is_deterministic_prompt_safe_and_binds_exact_durable_nodes() -> None:
    session_id, _turn_id, root = _seed_task_node(
        request_id="semantic-base-proposed-turn",
        task_id="semantic-base-proposed-task",
        node_id="private-root-id",
    )
    source_hash = _freeze_revision_hash(root.task_id)
    child_id = "private-child-id"
    now = "2026-08-23T00:00:00+00:00"
    acceptances = json.dumps(
        [
            {
                "acceptance_id": "child_ready",
                "criterion": "子节点可核对",
                "source_anchor_ids": ["request"],
            }
        ],
        ensure_ascii=False,
    )
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_graph_nodes "
            "(insession_task_id, graph_revision, insession_task_node_id, "
            "node_revision, node_kind, ordinal, title, objective, "
            "source_anchor_ids_json, acceptance_criteria_json, "
            "constraints_json, created_at) VALUES (?, 1, ?, 1, 'subtask', 1, "
            "'私有子节点', '完成子节点', '[\"request\"]', ?, '[]', ?)",
            (root.task_id, child_id, acceptances, now),
        )
        conn.execute(
            "INSERT INTO insession_task_node_states "
            "(insession_task_id, insession_task_node_id, node_revision, status, "
            "state_version, updated_at) VALUES (?, ?, 1, 'proposed', 1, ?)",
            (root.task_id, child_id, now),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_edges "
            "(insession_task_id, graph_revision, child_insession_task_node_id, "
            "parent_insession_task_node_id, ordinal) VALUES (?, 1, ?, ?, 1)",
            (root.task_id, child_id, root.node_id),
        )

    first = task_graph_store.project_task_graph_semantic_base(
        session_id=session_id,
        task_id=root.task_id,
        graph_revision=1,
    )
    second = task_graph_store.project_task_graph_semantic_base(
        session_id=session_id,
        task_id=root.task_id,
        graph_revision=1,
    )

    assert second == first
    assert first.snapshot.source_snapshot_sha256 == source_hash
    assert first.snapshot.root_node_alias == "base_node_000"
    assert tuple(node.node_alias for node in first.snapshot.nodes) == (
        "base_node_000",
        "base_node_001",
    )
    assert first.snapshot.nodes[1].parent_node_alias == "base_node_000"
    assert "private-root-id" not in first.snapshot.model_dump_json()
    assert "private-child-id" not in first.snapshot.model_dump_json()
    assert tuple(
        (item.node_alias, item.insession_task_node_id)
        for item in first.base_node_alias_bindings
    ) == (
        ("base_node_000", "private-root-id"),
        ("base_node_001", "private-child-id"),
    )


def test_projection_canonicalizes_multiple_source_authority_aliases() -> None:
    session_id, _turn_id, root = _seed_task_node(
        request_id="semantic-base-multiple-sources-turn",
        task_id="semantic-base-multiple-sources-task",
        node_id="semantic-base-multiple-sources-root",
    )
    _freeze_revision_hash(root.task_id, "multiple source aliases")
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_task_graph_nodes SET source_anchor_ids_json=? "
            "WHERE insession_task_id=? AND graph_revision=1 "
            "AND insession_task_node_id=?",
            (
                json.dumps(
                    ["task_creation_source", "document_01_obs_001"]
                ),
                root.task_id,
                root.node_id,
            ),
        )

    projected = task_graph_store.project_task_graph_semantic_base(
        session_id=session_id,
        task_id=root.task_id,
        graph_revision=1,
    )

    assert projected.snapshot.nodes[0].source_anchor_aliases == (
        "document_01_obs_001",
        "task_creation_source",
    )


def test_projection_includes_only_an_authenticated_completed_delivery() -> None:
    session_id, turn_id, subject = _seed_task_node(
        request_id="semantic-base-delivery-turn",
        task_id="semantic-base-delivery-task",
        node_id="semantic-base-delivery-root",
    )
    source_hash = _freeze_revision_hash(subject.task_id, "completed base source")
    _prepared, committed = _commit_passing_verification(
        session_id,
        turn_id,
        subject,
        request_id="semantic-base-verification",
        delivery_id="semantic-base-delivery",
        content="经过验证的完整交付。",
    )
    assert committed.delivery_id == "semantic-base-delivery"

    projected = task_graph_store.project_task_graph_semantic_base(
        session_id=session_id,
        task_id=subject.task_id,
        graph_revision=1,
    )

    node = projected.snapshot.nodes[0]
    assert projected.snapshot.source_snapshot_sha256 == source_hash
    assert node.status.value == "completed"
    assert node.completed_delivery_summary == "经过验证的完整交付。"
    assert node.completed_delivery_coverage.value == "full"
    assert node.completed_delivery_gap_aliases == ()
    assert node.delivery_authority_aliases == ("request",)


def test_projection_fails_closed_when_completed_delivery_is_missing_or_corrupt() -> None:
    session_id, _turn_id, missing = _seed_task_node(
        request_id="semantic-base-missing-turn",
        task_id="semantic-base-missing-task",
        node_id="semantic-base-missing-root",
    )
    _freeze_revision_hash(missing.task_id, "missing base source")
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_task_node_states SET status='completed' "
            "WHERE insession_task_id=? AND insession_task_node_id=?",
            (missing.task_id, missing.node_id),
        )
    with pytest.raises(task_graph_store.TaskGraphSemanticBaseProjectionError) as absent:
        task_graph_store.project_task_graph_semantic_base(
            session_id=session_id,
            task_id=missing.task_id,
            graph_revision=1,
        )
    assert absent.value.code == "delivery_authority_missing"

    session_id, turn_id, corrupt = _seed_task_node(
        request_id="semantic-base-corrupt-turn",
        task_id="semantic-base-corrupt-task",
        node_id="semantic-base-corrupt-root",
    )
    _freeze_revision_hash(corrupt.task_id, "corrupt base source")
    _commit_passing_verification(
        session_id,
        turn_id,
        corrupt,
        request_id="semantic-base-corrupt-verification",
        delivery_id="semantic-base-corrupt-delivery",
        content="原始交付。",
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_run_output_windows SET snapshot_hash=? "
            "WHERE work_run_id='workrun-one'",
            ("f" * 64,),
        )
    with pytest.raises(task_graph_store.TaskGraphSemanticBaseProjectionError) as drifted:
        task_graph_store.project_task_graph_semantic_base(
            session_id=session_id,
            task_id=corrupt.task_id,
            graph_revision=1,
        )
    assert drifted.value.code == "delivery_authority_corrupt"


def test_projection_requires_the_exact_current_revision_and_proposal_hash() -> None:
    session_id, _turn_id, subject = _seed_task_node(
        request_id="semantic-base-revision-turn",
        task_id="semantic-base-revision-task",
        node_id="semantic-base-revision-root",
    )
    with pytest.raises(task_graph_store.TaskGraphSemanticBaseProjectionError) as corrupt:
        task_graph_store.project_task_graph_semantic_base(
            session_id=session_id,
            task_id=subject.task_id,
            graph_revision=1,
        )
    assert corrupt.value.code == "source_snapshot_corrupt"

    _freeze_revision_hash(subject.task_id, "revision base source")
    with pytest.raises(task_graph_store.TaskGraphSemanticBaseProjectionError) as stale:
        task_graph_store.project_task_graph_semantic_base(
            session_id=session_id,
            task_id=subject.task_id,
            graph_revision=2,
        )
    assert stale.value.code == "revision_not_current"


def test_projection_rejects_an_older_delivery_without_current_carry_authority() -> None:
    session_id, turn_id, subject = _seed_task_node(
        request_id="semantic-base-carry-turn",
        task_id="semantic-base-carry-task",
        node_id="semantic-base-carry-root",
    )
    source_hash = _freeze_revision_hash(subject.task_id, "carry base source")
    _commit_passing_verification(
        session_id,
        turn_id,
        subject,
        request_id="semantic-base-carry-verification",
        delivery_id="semantic-base-carry-delivery",
        content="第一版交付。",
    )
    now = "2026-08-23T00:01:00+00:00"
    with store._connect() as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(
            "INSERT INTO insession_task_graph_revisions "
            "(insession_task_id, graph_revision, source_turn_id, proposal_hash, "
            "source_anchors_json, authorization_anchor_ids_json, "
            "required_anchor_ids_json, created_at) "
            "SELECT insession_task_id, 2, source_turn_id, ?, source_anchors_json, "
            "authorization_anchor_ids_json, required_anchor_ids_json, ? "
            "FROM insession_task_graph_revisions WHERE insession_task_id=? "
            "AND graph_revision=1",
            (source_hash, now, subject.task_id),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_nodes "
            "(insession_task_id, graph_revision, insession_task_node_id, "
            "node_revision, node_kind, ordinal, title, objective, "
            "source_anchor_ids_json, acceptance_criteria_json, constraints_json, "
            "created_at) SELECT insession_task_id, 2, insession_task_node_id, "
            "node_revision, node_kind, ordinal, title, objective, "
            "source_anchor_ids_json, acceptance_criteria_json, constraints_json, ? "
            "FROM insession_task_graph_nodes WHERE insession_task_id=? "
            "AND graph_revision=1",
            (now, subject.task_id),
        )
        conn.execute(
            "UPDATE insession_tasks SET current_graph_revision=2 "
            "WHERE insession_task_id=?",
            (subject.task_id,),
        )
        conn.execute("PRAGMA foreign_keys=ON")

    with pytest.raises(task_graph_store.TaskGraphSemanticBaseProjectionError) as missing_carry:
        task_graph_store.project_task_graph_semantic_base(
            session_id=session_id,
            task_id=subject.task_id,
            graph_revision=2,
        )
    assert missing_carry.value.code == "carry_authority_corrupt"
