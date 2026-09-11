from __future__ import annotations

from unittest.mock import ANY

import pytest

from personagraph.api import router, service
from personagraph.api.service import sessions as session_service
from personagraph.l2.task_graph.contracts import (
    InSessionTaskDetails,
    InSessionTaskGraphValidationContext,
    InSessionTaskSourceAnchor,
    InSessionTaskStatus,
    NewInSessionTaskGraphsProposal,
)
from personagraph.l2.task_graph.validation import validate_new_insession_task_graphs
from personagraph.session import store as session_store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.persistence.l2.task_graph import insession_tasks as insession_task_records


def _route_payload(method: str, target: str, body: dict) -> dict:
    return router.dispatch_response(method, target, body).payload


def _create_insession_task(session_id: str) -> tuple[str, str]:
    """通过公开 Store 门面创建一个小型可信图。"""

    user_text = "请分析这份材料并说明结论"
    accepted = session_store.accept_turn_execution(
        session_id=session_id,
        client_request_id="task-detail-request",
        source="test",
        user_text=user_text,
        lease_owner="test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    proposal = NewInSessionTaskGraphsProposal.model_validate({
        "source_turn_id": turn_id,
        "roots": [{
            "root_key": "analysis",
            "nodes": [
                {
                    "node_key": "analysis",
                    "node_kind": "root",
                    "title": "分析材料",
                    "objective": "分析这份材料并说明结论",
                    "source_anchor_ids": ["request"],
                    "acceptance_criteria": [{
                        "acceptance_id": "analysis_complete",
                        "criterion": "结论覆盖材料分析",
                        "source_anchor_ids": ["request"],
                    }],
                },
                {
                    "node_key": "summary",
                    "node_kind": "subtask",
                    "parent_node_key": "analysis",
                    "title": "整理结论",
                    "objective": "整理可读结论",
                    "source_anchor_ids": ["request"],
                    "acceptance_criteria": [{
                        "acceptance_id": "summary_complete",
                        "criterion": "结论可读",
                        "source_anchor_ids": ["request"],
                    }],
                },
            ],
        }],
    })
    context = InSessionTaskGraphValidationContext(
        session_id=session_id,
        source_turn_id=turn_id,
        source_anchors=(InSessionTaskSourceAnchor(
            anchor_id="request",
            source_turn_id=turn_id,
            source_kind="current_user_instruction",
            start=0,
            end=len(user_text),
            excerpt=user_text,
        ),),
        authorization_anchor_ids=("request",),
        required_anchor_ids=("request",),
    )
    validation = validate_new_insession_task_graphs(proposal, context=context)
    assert validation.status == "accepted"
    assert validation.proposal is not None
    assert validation.trusted_context is not None
    window = session_store.get_turn_execution_window(session_id)
    assert window is not None
    committed = insession_task_records.commit_new_insession_task_graphs(
        session_store._deps(),
        session_id=session_id,
        source_turn_id=turn_id,
        apply_id="task-detail-create",
        proposal=validation.proposal,
        trusted_context=validation.trusted_context,
        expected_window_revision=int(window["state_version"]),
    )
    return committed.created_insession_task_ids[0], turn_id


def test_session_scoped_insession_task_detail_route_returns_only_safe_task_card_fields():
    session_id = session_store.create_session("Entelecheia", title="task detail")
    task_id, _turn_id = _create_insession_task(session_id)

    payload = _route_payload(
        "GET",
        f"/api/sessions/{session_id}/insession-tasks/{task_id}",
        {},
    )

    task = payload["task"]
    assert task == {
        "insession_task_id": task_id,
        "title": "分析材料",
        "status": "proposed",
        "current_graph_revision": 1,
        "nodes": [
            {
                "insession_task_node_id": task_id,
                "parent_insession_task_node_id": None,
                "node_kind": "root",
                "ordinal": 0,
                "title": "分析材料",
                "status": "proposed",
            },
            {
                "insession_task_node_id": ANY,
                "parent_insession_task_node_id": task_id,
                "node_kind": "subtask",
                "ordinal": 1,
                "title": "整理结论",
                "status": "proposed",
            },
        ],
        "related_turn_count": 1,
    }
    assert {"objective", "constraints", "acceptance_criteria", "source_anchors"}.isdisjoint(
        task
    )
    assert all(
        {
            "objective",
            "constraints",
            "acceptance_criteria",
            "acceptance_id",
            "source_anchor_ids",
        }.isdisjoint(node)
        for node in task["nodes"]
    )


def test_session_scoped_insession_task_detail_route_projects_a_graphless_task_shell(monkeypatch):
    session_id = session_store.create_session("Entelecheia", title="task shell")
    task_id = "insession_task_shell"
    shell = InSessionTaskDetails(
        insession_task_id=task_id,
        session_id=session_id,
        task_state_version=1,
        title="整理调研方向",
        objective="整理用户提出的调研方向",
        current_graph_revision=None,
        status=InSessionTaskStatus.PROPOSED,
        nodes=(),
        related_turn_count=1,
    )
    monkeypatch.setattr(
        task_graph_store,
        "get_insession_task_details",
        lambda _session_id, _task_id: shell,
    )

    payload = _route_payload(
        "GET",
        f"/api/sessions/{session_id}/insession-tasks/{task_id}",
        {},
    )

    assert payload["task"] == {
        "insession_task_id": task_id,
        "title": "整理调研方向",
        "status": "proposed",
        "current_graph_revision": None,
        "nodes": [],
        "related_turn_count": 1,
    }
    assert "task_state_version" not in payload["task"]
    assert "objective" not in payload["task"]


def test_session_scoped_insession_task_detail_route_hides_cross_session_task_identity():
    owner_session_id = session_store.create_session("Entelecheia", title="owner")
    other_session_id = session_store.create_session("Entelecheia", title="other")
    task_id, _turn_id = _create_insession_task(owner_session_id)

    with pytest.raises(service.ApiError) as exc:
        _route_payload(
            "GET",
            f"/api/sessions/{other_session_id}/insession-tasks/{task_id}",
            {},
        )

    assert exc.value.status == 404
    assert exc.value.code == "INSESSION_TASK_NOT_FOUND"


def test_task_detail_counts_only_same_session_turn_links_even_if_a_bad_row_exists():
    owner_session_id = session_store.create_session("Entelecheia", title="owner")
    other_session_id = session_store.create_session("Entelecheia", title="other")
    task_id, owner_turn_id = _create_insession_task(owner_session_id)
    details = task_graph_store.get_insession_task_details(owner_session_id, task_id)
    assert details is not None
    child_node_id = next(
        node["insession_task_node_id"]
        for node in details.nodes
        if node["node_kind"] == "subtask"
    )

    # 常规写入器会拒绝此类不匹配。这里模拟格式错误的导入，以证明读取路径不会
    # 泄露或聚合另一会话的链接。同一个所属轮次之后可以合法获得节点级链接，
    # 但在公开计数中仍必须只算作一个相关轮次。
    with session_store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, relation, created_at) "
            "VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (other_session_id, "foreign-turn", task_id, "2026-08-11T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, relation, created_at) "
            "VALUES (?, ?, ?, ?, 'referenced', ?)",
            (owner_session_id, owner_turn_id, task_id, child_node_id, "2026-08-11T00:00:00+00:00"),
        )

    payload = _route_payload(
        "GET",
        f"/api/sessions/{owner_session_id}/insession-tasks/{task_id}",
        {},
    )

    assert payload["task"]["related_turn_count"] == 1
    assert "foreign-turn" not in str(payload)


def test_task_detail_projection_fails_closed_when_the_store_manifest_is_unreadable(monkeypatch):
    session_id = session_store.create_session("Entelecheia", title="unreadable")

    def _raise_unreadable(_session_id: str, _task_id: str):
        raise task_graph_store.InSessionTaskPersistenceError(
            "source manifest is corrupt"
        )

    monkeypatch.setattr(
        task_graph_store,
        "get_insession_task_details",
        _raise_unreadable,
    )

    with pytest.raises(service.ApiError) as exc:
        session_service.get_insession_task_details(session_id, "insession-task-1")

    assert exc.value.status == 409
    assert exc.value.code == "INSESSION_TASK_DETAILS_UNAVAILABLE"
    assert "corrupt" not in exc.value.message


def test_task_detail_projection_fails_closed_when_the_store_tree_is_malformed(monkeypatch):
    session_id = session_store.create_session("Entelecheia", title="malformed tree")
    malformed = InSessionTaskDetails(
        insession_task_id="insession_task_expected",
        session_id=session_id,
        task_state_version=1,
        title="错误根节点",
        objective="验证错误根节点不会被投影",
        current_graph_revision=1,
        status=InSessionTaskStatus.PROPOSED,
        nodes=({
            "insession_task_node_id": "wrong-root-id",
            "node_kind": "root",
            "ordinal": 0,
            "parent_insession_task_node_id": None,
            "title": "错误根节点",
            "status": "proposed",
        },),
    )
    monkeypatch.setattr(
        task_graph_store,
        "get_insession_task_details",
        lambda _session_id, _task_id: malformed,
    )

    with pytest.raises(service.ApiError) as exc:
        session_service.get_insession_task_details(session_id, "insession_task_expected")

    assert exc.value.status == 409
    assert exc.value.code == "INSESSION_TASK_DETAILS_UNAVAILABLE"


@pytest.mark.parametrize(
    ("current_graph_revision", "nodes"),
    [
        (
            None,
            ({
                "insession_task_node_id": "insession_task_invalid_shell",
                "node_kind": "root",
                "ordinal": 0,
                "parent_insession_task_node_id": None,
                "title": "不应存在的根节点",
                "status": "proposed",
            },),
        ),
        (1, ()),
    ],
)
def test_task_detail_projection_rejects_inconsistent_shell_and_graph_shapes(
    monkeypatch,
    current_graph_revision,
    nodes,
):
    session_id = session_store.create_session("Entelecheia", title="invalid shell shape")
    malformed = InSessionTaskDetails(
        insession_task_id="insession_task_invalid_shell",
        session_id=session_id,
        task_state_version=1,
        title="无效任务",
        objective="验证 shell 与任务图必须保持一致",
        current_graph_revision=current_graph_revision,
        status=InSessionTaskStatus.PROPOSED,
        nodes=nodes,
    )
    monkeypatch.setattr(
        task_graph_store,
        "get_insession_task_details",
        lambda _session_id, _task_id: malformed,
    )

    with pytest.raises(service.ApiError) as exc:
        session_service.get_insession_task_details(
            session_id,
            "insession_task_invalid_shell",
        )

    assert exc.value.status == 409
    assert exc.value.code == "INSESSION_TASK_DETAILS_UNAVAILABLE"
