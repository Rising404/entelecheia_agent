from __future__ import annotations

import hashlib
import sqlite3

import pytest

from personagraph.l2.task_graph.task_matching import InSessionTaskMatchesProposal
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.persistence.l2.task_graph import insession_tasks as task_records


def _accept(session_id: str, request_id: str, user_text: str) -> str:
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=request_id,
        source="task_match_test",
        user_text=user_text,
        lease_owner="task-match-test",
    )
    return str(accepted["turn"]["turn_id"])  # type: ignore[index]


def _window_revision(session_id: str) -> int:
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    return int(window["state_version"])


def _release_completed_turn(session_id: str, turn_id: str) -> None:
    finalized = store.finalize_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=_window_revision(session_id),
        processing_level="L2",
        assistant_content="任务匹配已记录。",
        post_commit_job_kinds=(),
    )
    store.release_turn_execution_window(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(finalized["window"]["state_version"]),  # type: ignore[index]
    )


def _proposal(*matches: dict[str, object]) -> InSessionTaskMatchesProposal:
    return InSessionTaskMatchesProposal.model_validate({"task_matches": matches})


def _apply(
    *,
    session_id: str,
    turn_id: str,
    apply_id: str,
    proposal: InSessionTaskMatchesProposal,
    exposed_catalog_ids: tuple[str, ...] = (),
    expected_window_revision: int | None = None,
):
    return task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=turn_id,
        apply_id=apply_id,
        proposal=proposal,
        exposed_catalog_ids=exposed_catalog_ids,
        expected_window_revision=(
            _window_revision(session_id)
            if expected_window_revision is None
            else expected_window_revision
        ),
    )


def _seed_shell(
    session_id: str,
    request_id: str,
    *,
    title: str = "材料分析",
    objective: str = "分析材料并形成结论",
) -> str:
    user_text = "请建立材料分析任务"
    turn_id = _accept(session_id, request_id, user_text)
    applied = _apply(
        session_id=session_id,
        turn_id=turn_id,
        apply_id=f"{request_id}-apply",
        proposal=_proposal(
            {
                "match_type": "new_root",
                "local_key": "material",
                "title": title,
                "objective": objective,
                "source_excerpt": user_text,
            }
        ),
    )
    task_id = applied.created_insession_task_ids_by_local_key["material"]
    _release_completed_turn(session_id, turn_id)
    return task_id


def _count(table: str, *, where: str = "", params: tuple[object, ...] = ()) -> int:
    with store._connect() as conn:
        row = conn.execute(
            f"SELECT COUNT(*) AS count FROM {table} {where}", params
        ).fetchone()
    assert row is not None
    return int(row["count"])


def test_target_change_intent_is_durable_and_exactly_replayable():
    session_id = store.create_session("Entelecheia")
    task_id = _seed_shell(session_id, "seed-target-change")
    user_text = "请把材料任务的目标改成只输出实验设计比较"
    excerpt = "目标改成只输出实验设计比较"
    turn_id = _accept(session_id, "target-change", user_text)
    proposal = _proposal(
        {
            "match_type": "existing_root_target_change",
            "insession_task_id": task_id,
            "replacement_objective": "只输出实验设计比较",
            "source_excerpt": excerpt,
            "execute_current": True,
        }
    )

    applied = _apply(
        session_id=session_id,
        turn_id=turn_id,
        apply_id="target-change-apply",
        proposal=proposal,
        exposed_catalog_ids=(task_id,),
    )
    replayed = _apply(
        session_id=session_id,
        turn_id=turn_id,
        apply_id="target-change-apply",
        proposal=proposal,
        exposed_catalog_ids=(task_id,),
    )
    manifest = task_graph_store.get_insession_task_execution_lane_manifest(
        session_id=session_id,
        turn_id=turn_id,
    )

    assert applied.status == "applied"
    assert replayed.status == "replayed"
    assert replayed.related_insession_task_ids == (task_id,)
    lane_match = manifest.lanes[0].matches[0]
    assert lane_match.match_type == "existing_root_target_change"
    assert lane_match.replacement_objective == "只输出实验设计比较"
    assert user_text[lane_match.source_span.start : lane_match.source_span.end] == excerpt


def test_mixed_match_apply_creates_shell_links_and_branch_intent_atomically():
    session_id = store.create_session("Entelecheia")
    existing_task_id = _seed_shell(session_id, "seed-existing")
    user_text = "请新建上海旅行计划，同时继续材料分析，并新增局限分析分支。"
    turn_id = _accept(session_id, "mixed-task-match", user_text)
    before_revision = _window_revision(session_id)
    proposal = _proposal(
        {
            "match_type": "new_root",
            "local_key": "trip",
            "title": "上海旅行",
            "objective": "制定上海旅行计划",
            "source_excerpt": "请新建上海旅行计划",
        },
        {
            "match_type": "existing_root",
            "insession_task_id": existing_task_id,
            "source_excerpt": "继续材料分析",
        },
        {
            "match_type": "existing_root_branch",
            "insession_task_id": existing_task_id,
            "branch_key": "limitations",
            "branch_summary": "增加局限分析分支",
            "source_excerpt": "新增局限分析分支",
        },
    )

    result = _apply(
        session_id=session_id,
        turn_id=turn_id,
        apply_id="mixed-task-match-apply",
        proposal=proposal,
        exposed_catalog_ids=(existing_task_id,),
        expected_window_revision=before_revision,
    )

    new_task_id = result.created_insession_task_ids_by_local_key["trip"]
    assert result.status == "applied"
    assert result.related_insession_task_ids == (new_task_id, existing_task_id)
    assert len(result.branch_intent_ids) == 1
    assert result.window_state_version == before_revision + 1
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert window["state_version"] == before_revision + 1
    assert window["turn_task_link_revision"] == result.turn_task_link_revision

    catalog = {
        item.insession_task_id: item
        for item in store.list_insession_task_catalog(session_id)
    }
    assert set(catalog) == {existing_task_id, new_task_id}
    assert catalog[new_task_id].current_graph_revision is None
    details = task_graph_store.get_insession_task_details(session_id, new_task_id)
    assert details is not None
    assert details.title == "上海旅行"
    assert details.objective == "制定上海旅行计划"
    assert details.current_graph_revision is None
    assert details.nodes == ()
    assert details.source_anchors == ()
    assert details.related_turn_count == 1

    with store._connect() as conn:
        shell = conn.execute(
            "SELECT current_graph_revision, created_turn_id, creation_source_start, "
            "creation_source_end, creation_source_sha256 FROM insession_tasks "
            "WHERE insession_task_id=?",
            (new_task_id,),
        ).fetchone()
        assert shell is not None
        excerpt = "请新建上海旅行计划"
        start = user_text.index(excerpt)
        assert dict(shell) == {
            "current_graph_revision": None,
            "created_turn_id": turn_id,
            "creation_source_start": start,
            "creation_source_end": start + len(excerpt),
            "creation_source_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
        }
        branch = conn.execute(
            "SELECT branch_intent_id, insession_task_id, branch_key, branch_summary, "
            "source_start, source_end, source_sha256 FROM insession_task_branch_intents "
            "WHERE source_turn_id=?",
            (turn_id,),
        ).fetchone()
        assert branch is not None
        branch_excerpt = "新增局限分析分支"
        branch_start = user_text.index(branch_excerpt)
        assert dict(branch) == {
            "branch_intent_id": result.branch_intent_ids[0],
            "insession_task_id": existing_task_id,
            "branch_key": "limitations",
            "branch_summary": "增加局限分析分支",
            "source_start": branch_start,
            "source_end": branch_start + len(branch_excerpt),
            "source_sha256": hashlib.sha256(branch_excerpt.encode()).hexdigest(),
        }
        links = conn.execute(
            "SELECT insession_task_id, insession_task_node_id, relation "
            "FROM insession_task_turn_links WHERE turn_id=? ORDER BY link_id",
            (turn_id,),
        ).fetchall()
        assert [dict(row) for row in links] == [
            {
                "insession_task_id": new_task_id,
                "insession_task_node_id": None,
                "relation": "created",
            },
            {
                "insession_task_id": existing_task_id,
                "insession_task_node_id": None,
                "relation": "referenced",
            },
        ]
    # 分支匹配是根节点下的持久意图；它绝不会假装已经创建 TaskGraph 节点。
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_task_graph_nodes "
            "WHERE insession_task_id=?",
            (existing_task_id,),
        ).fetchone()[0] == 0


def test_exact_replay_returns_original_ids_and_collision_fails_closed():
    session_id = store.create_session("Entelecheia")
    user_text = "请新建论文解读任务"
    turn_id = _accept(session_id, "task-match-replay", user_text)
    original_revision = _window_revision(session_id)
    proposal = _proposal(
        {
            "match_type": "new_root",
            "local_key": "paper",
            "title": "论文解读",
            "objective": "解读论文",
            "source_excerpt": user_text,
        }
    )
    first = _apply(
        session_id=session_id,
        turn_id=turn_id,
        apply_id="task-match-replay-apply",
        proposal=proposal,
        expected_window_revision=original_revision,
    )
    replay = _apply(
        session_id=session_id,
        turn_id=turn_id,
        apply_id="task-match-replay-apply",
        proposal=proposal,
        expected_window_revision=original_revision,
    )

    assert replay.status == "replayed"
    assert replay.created_insession_task_ids_by_local_key == (
        first.created_insession_task_ids_by_local_key
    )
    assert replay.related_insession_task_ids == first.related_insession_task_ids
    assert replay.turn_task_link_revision == first.turn_task_link_revision
    assert replay.window_state_version == first.window_state_version
    assert _count("insession_tasks") == 1
    assert _count("insession_task_turn_links", where="WHERE turn_id=?", params=(turn_id,)) == 1

    changed = _proposal(
        {
            "match_type": "new_root",
            "local_key": "paper",
            "title": "论文解读",
            "objective": "换一个目标",
            "source_excerpt": user_text,
        }
    )
    with pytest.raises(task_graph_store.InSessionTaskApplyIdCollision):
        _apply(
            session_id=session_id,
            turn_id=turn_id,
            apply_id="task-match-replay-apply",
            proposal=changed,
            expected_window_revision=original_revision,
        )
    assert _count("insession_tasks") == 1
    assert _window_revision(session_id) == first.window_state_version


def test_lane_manifest_is_source_ordered_durable_and_exactly_replayed():
    session_id = store.create_session("Entelecheia")
    user_text = "先完成任务甲，再完成任务乙"
    turn_id = _accept(session_id, "lane-manifest-replay", user_text)
    proposal = _proposal(
        {
            "match_type": "new_root",
            "local_key": "beta",
            "title": "任务乙",
            "objective": "完成任务乙",
            "source_excerpt": "任务乙",
        },
        {
            "match_type": "new_root",
            "local_key": "alpha",
            "title": "任务甲",
            "objective": "完成任务甲",
            "source_excerpt": "任务甲",
        },
    )
    revision = _window_revision(session_id)

    applied = _apply(
        session_id=session_id,
        turn_id=turn_id,
        apply_id="lane-manifest-replay-apply",
        proposal=proposal,
        expected_window_revision=revision,
    )
    first = task_graph_store.get_insession_task_execution_lane_manifest(
        session_id=session_id,
        turn_id=turn_id,
    )

    assert [lane.insession_task_id for lane in first.lanes] == [
        applied.created_insession_task_ids_by_local_key["alpha"],
        applied.created_insession_task_ids_by_local_key["beta"],
    ]
    assert [lane.ordinal for lane in first.lanes] == [0, 1]
    assert all(lane.execution_requested for lane in first.lanes)
    assert applied.related_insession_task_ids == tuple(
        lane.insession_task_id for lane in first.lanes
    )
    alpha_source = task_graph_store.get_insession_task_creation_source(
        session_id=session_id,
        insession_task_id=applied.created_insession_task_ids_by_local_key["alpha"],
    )
    assert alpha_source.source_turn_id == turn_id
    assert alpha_source.excerpt == "任务甲"
    assert alpha_source.anchor_id == "task_creation_source"

    replay = _apply(
        session_id=session_id,
        turn_id=turn_id,
        apply_id="lane-manifest-replay-apply",
        proposal=proposal,
        expected_window_revision=revision,
    )
    store._INITIALIZED_PATHS.clear()
    reopened = task_graph_store.get_insession_task_execution_lane_manifest(
        session_id=session_id,
        turn_id=turn_id,
    )

    assert replay.status == "replayed"
    assert reopened == first
    assert _count("insession_tasks") == 2


def test_lane_manifest_tamper_and_source_drift_fail_closed():
    session_id = store.create_session("Entelecheia")
    user_text = "请建立任务甲"
    turn_id = _accept(session_id, "lane-manifest-tamper", user_text)
    _apply(
        session_id=session_id,
        turn_id=turn_id,
        apply_id="lane-manifest-tamper-apply",
        proposal=_proposal(
            {
                "match_type": "new_root",
                "local_key": "alpha",
                "title": "任务甲",
                "objective": "完成任务甲",
                "source_excerpt": "任务甲",
            }
        ),
    )

    with store._connect() as conn:
        original = conn.execute(
            "SELECT execution_lane_manifest_json FROM "
            "insession_task_match_apply_receipts WHERE source_turn_id=?",
            (turn_id,),
        ).fetchone()
        assert original is not None
        conn.execute(
            "UPDATE insession_task_match_apply_receipts "
            "SET execution_lane_manifest_json='{}' WHERE source_turn_id=?",
            (turn_id,),
        )
    with pytest.raises(
        task_graph_store.InSessionTaskPersistenceError,
        match="manifest is corrupt",
    ):
        task_graph_store.get_insession_task_execution_lane_manifest(
            session_id=session_id,
            turn_id=turn_id,
        )

    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_task_match_apply_receipts "
            "SET execution_lane_manifest_json=? WHERE source_turn_id=?",
            (str(original["execution_lane_manifest_json"]), turn_id),
        )
        input_row = conn.execute(
            "SELECT input.turn_idx FROM runtime_turn_inputs AS input "
            "WHERE input.session_id=? AND input.turn_id=?",
            (session_id, turn_id),
        ).fetchone()
        assert input_row is not None
        conn.execute(
            "UPDATE session_turns SET content='已被篡改的输入' "
            "WHERE session_id=? AND turn_idx=?",
            (session_id, int(input_row["turn_idx"])),
        )
    with pytest.raises(
        task_graph_store.InSessionTaskPersistenceError,
        match="source binding has drifted",
    ):
        task_graph_store.get_insession_task_execution_lane_manifest(
            session_id=session_id,
            turn_id=turn_id,
        )


def test_existing_match_requires_exposed_same_session_catalog_membership():
    owner_session = store.create_session("Entelecheia")
    owner_task_id = _seed_shell(owner_session, "owner-task")
    foreign_session = store.create_session("Entelecheia")
    foreign_task_id = _seed_shell(foreign_session, "foreign-task")
    user_text = "继续材料分析"
    turn_id = _accept(owner_session, "catalog-boundary", user_text)
    revision = _window_revision(owner_session)

    omitted = _proposal(
        {
            "match_type": "existing_root",
            "insession_task_id": owner_task_id,
            "source_excerpt": user_text,
        }
    )
    with pytest.raises(task_graph_store.InSessionTaskPersistenceError, match="unknown_insession_task_id"):
        _apply(
            session_id=owner_session,
            turn_id=turn_id,
            apply_id="catalog-omitted",
            proposal=omitted,
            exposed_catalog_ids=(),
            expected_window_revision=revision,
        )

    foreign = _proposal(
        {
            "match_type": "existing_root",
            "insession_task_id": foreign_task_id,
            "source_excerpt": user_text,
        }
    )
    with pytest.raises(
        task_graph_store.InSessionTaskPersistenceError,
        match="unknown or cross-Session",
    ):
        _apply(
            session_id=owner_session,
            turn_id=turn_id,
            apply_id="catalog-foreign",
            proposal=foreign,
            exposed_catalog_ids=(foreign_task_id,),
            expected_window_revision=revision,
        )

    assert _count(
        "insession_task_turn_links", where="WHERE turn_id=?", params=(turn_id,)
    ) == 0
    assert _count(
        "insession_task_match_apply_receipts",
        where="WHERE source_turn_id=?",
        params=(turn_id,),
    ) == 0
    assert _window_revision(owner_session) == revision


def test_authoritative_excerpt_recheck_and_post_insert_error_leave_no_half_write(
    monkeypatch,
):
    session_id = store.create_session("Entelecheia")
    existing_task_id = _seed_shell(session_id, "rollback-existing")
    user_text = "请新建旅行任务，并继续材料分析。"
    turn_id = _accept(session_id, "task-match-rollback", user_text)
    revision = _window_revision(session_id)
    bad_excerpt = _proposal(
        {
            "match_type": "new_root",
            "local_key": "trip",
            "title": "旅行",
            "objective": "制定旅行计划",
            "source_excerpt": "请新建旅行任务",
        },
        {
            "match_type": "existing_root",
            "insession_task_id": existing_task_id,
            "source_excerpt": "不存在于权威输入的片段",
        },
    )
    with pytest.raises(task_graph_store.InSessionTaskPersistenceError, match="source_excerpt_not_found"):
        _apply(
            session_id=session_id,
            turn_id=turn_id,
            apply_id="bad-excerpt",
            proposal=bad_excerpt,
            exposed_catalog_ids=(existing_task_id,),
            expected_window_revision=revision,
        )
    assert _count("insession_tasks") == 1
    assert _count(
        "insession_task_turn_links", where="WHERE turn_id=?", params=(turn_id,)
    ) == 0

    valid = _proposal(
        {
            "match_type": "new_root",
            "local_key": "trip",
            "title": "旅行",
            "objective": "制定旅行计划",
            "source_excerpt": "请新建旅行任务",
        }
    )

    def fail_after_shell_insert(*_args, **_kwargs):
        raise sqlite3.OperationalError("injected link failure")

    monkeypatch.setattr(
        task_records,
        "_link_task_match_roots_in_transaction",
        fail_after_shell_insert,
    )
    with pytest.raises(sqlite3.OperationalError, match="injected link failure"):
        _apply(
            session_id=session_id,
            turn_id=turn_id,
            apply_id="post-insert-failure",
            proposal=valid,
            expected_window_revision=revision,
        )
    assert _count("insession_tasks") == 1
    assert _count(
        "insession_task_match_apply_receipts",
        where="WHERE source_turn_id=?",
        params=(turn_id,),
    ) == 0
    assert _window_revision(session_id) == revision


def test_stale_window_revision_rejects_match_apply_without_writes():
    session_id = store.create_session("Entelecheia")
    user_text = "请新建本地目录分析任务"
    turn_id = _accept(session_id, "task-match-cas", user_text)
    revision = _window_revision(session_id)
    proposal = _proposal(
        {
            "match_type": "new_root",
            "local_key": "directory",
            "title": "目录分析",
            "objective": "分析本地目录",
            "source_excerpt": user_text,
        }
    )

    with pytest.raises(store.TurnExecutionWindowRevisionConflict):
        _apply(
            session_id=session_id,
            turn_id=turn_id,
            apply_id="task-match-cas-apply",
            proposal=proposal,
            expected_window_revision=revision + 1,
        )

    assert _count("insession_tasks") == 0
    assert _count("insession_task_match_apply_receipts") == 0
    assert _count(
        "insession_task_turn_links", where="WHERE turn_id=?", params=(turn_id,)
    ) == 0
    assert _window_revision(session_id) == revision
