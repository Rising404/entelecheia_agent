"""Session 标识、列表/搜索、生命周期与 SQLite 清除持久化。"""

from __future__ import annotations

from typing import Any

from ..history import turns as transcript_turns
from ..deps import StoreDeps
from . import queries


def create_session(
    deps: StoreDeps,
    persona_id: str,
    title: str | None = None,
    folder_id: str | None = None,
    working_dir: str | None = None,
    session_id: str | None = None,
) -> str:
    deps.init_db()
    session_id = session_id or deps.new_id()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, persona_id, title, folder_id, working_dir, status, created_at, last_active_at)"
            " VALUES (?, ?, ?, ?, ?, 'active', ?, ?)",
            (session_id, persona_id, title, folder_id, working_dir, now, now),
        )
    return session_id


def get_session(deps: StoreDeps, session_id: str) -> dict[str, Any] | None:
    deps.init_db()
    with deps.connect() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def list_sessions(
    deps: StoreDeps,
    include_archived: bool = False,
    include_trashed: bool = False,
    *,
    status: str = "active",
    folder_id: str | None = None,
    persona_id: str | None = None,
    query: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    deps.init_db()
    with deps.connect() as conn:
        rows = conn.execute("SELECT * FROM sessions ORDER BY last_active_at DESC")
        return queries.list_sessions(
            (dict(row) for row in rows),
            lambda session_id: transcript_turns._get_turns_in_transaction(
                conn, session_id=session_id
            ),
            include_archived=include_archived,
            include_trashed=include_trashed,
            status=status,
            folder_id=folder_id,
            persona_id=persona_id,
            query=query,
            limit=limit,
        )


def search_sessions(
    deps: StoreDeps,
    query: str,
    status: str = "active",
    folder_id: str | None = None,
    persona_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    deps.init_db()
    with deps.connect() as conn:
        rows = conn.execute("SELECT * FROM sessions ORDER BY last_active_at DESC")
        return queries.search_sessions(
            (dict(row) for row in rows),
            lambda session_id: transcript_turns._get_turns_in_transaction(
                conn, session_id=session_id
            ),
            query=query,
            status=status,
            folder_id=folder_id,
            persona_id=persona_id,
            limit=limit,
        )


def rename_session(deps: StoreDeps, session_id: str, title: str) -> bool:
    deps.init_db()
    with deps.connect() as conn:
        return conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id)).rowcount > 0


def replace_session_title(deps: StoreDeps, session_id: str, *, expected: str | None, title: str) -> bool:
    """仅替换生成开始时的占位值，避免覆盖其后的人工作业。"""
    deps.init_db()
    with deps.connect() as conn:
        return conn.execute(
            "UPDATE sessions SET title=? WHERE id=? AND title IS ?",
            (title, session_id, expected),
        ).rowcount > 0


def archive_session(deps: StoreDeps, session_id: str) -> bool:
    deps.init_db()
    with deps.connect() as conn:
        return conn.execute(
            "UPDATE sessions SET status='archived', archived_at=? WHERE id=? AND status='active'",
            (deps.now(), session_id),
        ).rowcount > 0


def unarchive_session(deps: StoreDeps, session_id: str) -> bool:
    deps.init_db()
    with deps.connect() as conn:
        return conn.execute(
            "UPDATE sessions SET status='active', archived_at=NULL WHERE id=? AND status='archived'", (session_id,)
        ).rowcount > 0


def trash_session(deps: StoreDeps, session_id: str) -> bool:
    deps.init_db()
    with deps.connect() as conn:
        now = deps.now()
        updated = conn.execute(
            "UPDATE sessions SET previous_status=status, status='trashed', deleted_at=?"
            " WHERE id=? AND status IN ('active','archived')",
            (now, session_id),
        ).rowcount > 0
        return updated


def restore_session(deps: StoreDeps, session_id: str) -> bool:
    deps.init_db()
    with deps.connect() as conn:
        updated = conn.execute(
            "UPDATE sessions SET status=COALESCE(previous_status, 'active'), "
            "previous_status=NULL, deleted_at=NULL"
            " WHERE id=? AND status='trashed'",
            (session_id,),
        ).rowcount > 0
        return updated


def purge_session(deps: StoreDeps, session_id: str) -> bool:
    deps.init_db()
    with deps.connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE id=?",
            (session_id,),
        ).fetchone()
        found = row is not None
        # Runtime 权威状态通过 RESTRICT 链接将活动 Window 指向其 input 与 Turn，
        # 且 P2 WorkRun 会保留其创建/更新 Turn。在同一清除事务内按依赖顺序删除这些
        # 由 Session 持有的聚合；之后其子记录可正常级联。已验证交付有意通过
        # RESTRICT 阻止普通 WorkRun 删除；显式的整 Session 清除是唯一获授权先释放
        # 这些引用的生命周期路径。
        conn.execute(
            "DELETE FROM session_turn_commits WHERE session_id=?",
            (session_id,),
        )
        # TaskGraph revision application 会保留 commit receipt，而已完成节点的
        # carry receipt 会保留来源 Delivery。删除任一父聚合前，先释放完整的
        # TaskGraph 提交链。
        conn.execute(
            "DELETE FROM "
            "insession_active_task_graph_execution_replan_requests "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_task_graph_execution_replan_applications "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_task_graph_execution_replan_requests "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM "
            "insession_auxiliary_v2_task_graph_node_carry_receipts "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_auxiliary_v2_task_graph_commit_receipts "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_task_node_deliveries WHERE session_id=?",
            (session_id,),
        )
        # Settlement Attempt 通过 RESTRICT 链接保留其恰好一次的 budget charge。
        # 整 Session 清除持有该审计关系两端，因此先释放链接，再级联 WorkRun。
        conn.execute(
            "UPDATE insession_work_run_attempts SET budget_charge_id=NULL "
            "WHERE work_run_id IN ("
            "SELECT work_run_id FROM insession_work_runs WHERE session_id=?"
            ")",
            (session_id,),
        )
        # 节点 completion 有意通过 RESTRICT 阻止普通 WorkRun 删除。整 Session
        # 清除持有两个不可变端，并在释放精确 execution subject 前移除下游 carry
        # 权威状态。membership-to-carry 链接使用 SQLite RESTRICT（即使声明为
        # deferrable 也会立即执行），因此清除必须先在同一获授权事务内清空这一可变
        # 生命周期指针。
        conn.execute(
            "UPDATE insession_auxiliary_graph_revision_nodes_v2 SET "
            "carried_completion_id=NULL WHERE carried_completion_id IN ("
            "SELECT carry_receipt_id FROM "
            "insession_auxiliary_node_completion_carries_v2 "
            "WHERE session_id=?"
            ")",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_auxiliary_node_completion_carries_v2 "
            "WHERE session_id=?",
            (session_id,),
        )
        # 已封存的终结 completion 由三层不可变 receipt 保留，而这些 receipt 又会
        # 保留语义 quorum 权威状态。删除 completion 本身前，按叶到根的顺序移除每个
        # 由 Session 持有的依赖项；普通记录删除仍保持失败关闭。
        conn.execute(
            "DELETE FROM insession_auxiliary_v2_terminal_proposal_receipts "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_auxiliary_v2_finish_gate_receipts "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_auxiliary_v2_terminal_seal_apply_receipts "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_auxiliary_active_replan_triggers "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_auxiliary_replan_trigger_applications "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_auxiliary_replan_triggers "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_auxiliary_semantic_quorum_reviewers "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_auxiliary_semantic_quorum_settlements "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM "
            "insession_auxiliary_semantic_request_context_artifacts "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_auxiliary_semantic_verification_results "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_auxiliary_semantic_verification_requests "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_auxiliary_node_completions_v2 "
            "WHERE session_id=?",
            (session_id,),
        )
        # 当前 Host 规划原语通过 RESTRICT 链接保留其 observation、已验证 context
        # artifact、graph snapshot、可选 WorkRun 和来源 Turn。释放任一执行 lane 前，
        # 从 invocation 向下到 evidence 记录删除这一由 Session 持有的聚合。
        # artifact 的工具结果子项随 artifact 级联；observation item 与 gap 随
        # observation 级联。
        conn.execute(
            "DELETE FROM insession_auxiliary_planning_primitive_invocations "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_auxiliary_planning_context_artifacts "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_auxiliary_context_verification_receipts "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_auxiliary_observations WHERE session_id=?",
            (session_id,),
        )
        # 当前 WorkRun 聚合在同级表间包含有意设置的 RESTRICT 与延迟 NO ACTION 链接。
        # SQLite 处理父级联的顺序可能过早触及这些守卫，因此整 Session 清除会显式
        # 释放其可变头指针和聚合叶节点。每项变更仍限定于此 Session 的 WorkRun，
        # 并在外围清除事务中执行。
        conn.execute(
            "UPDATE insession_work_runs SET current_attempt_id=NULL, "
            "current_verification_request_id=NULL WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_work_run_output_windows WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_work_run_acceptance_progress "
            "WHERE work_run_id IN ("
            "SELECT work_run_id FROM insession_work_runs WHERE session_id=?"
            ")",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_auxiliary_v2_waiting_user_answer_bindings "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_work_run_attempts WHERE work_run_id IN ("
            "SELECT work_run_id FROM insession_work_runs WHERE session_id=?"
            ")",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_work_run_verification_requests "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_work_run_budget_charges WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_work_run_turn_links WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_work_runs WHERE session_id=?",
            (session_id,),
        )
        # AuxiliaryGraph 在不可变 revision、authority snapshot 和 observation
        # anchor 间有意设置 RESTRICT 链接。整 Session 清除持有所有端，因此只断开
        # 可变 current 指针，并在其精确来源 Turn 消失前按依赖顺序删除该聚合。
        conn.execute(
            "UPDATE insession_auxiliary_graph_v2_containers SET "
            "current_goal_id=NULL, current_auxiliary_graph_revision=NULL "
            "WHERE session_id=?",
            (session_id,),
        )
        # Graph revision 会追加一条以哈希链接的 goal-budget charge 链。第二个
        # revision 会使较旧 charge 成为即时 RESTRICT 父项，因此 SQLite 无法按任意
        # 记录顺序级联整条链。整 Session 清除会以确定性的最新优先顺序释放它。
        charge_rows = conn.execute(
            "SELECT budget_charge_id FROM "
            "insession_auxiliary_goal_budget_charges WHERE session_id=? "
            "ORDER BY goal_id, budget_state_version_after DESC",
            (session_id,),
        ).fetchall()
        for charge_row in charge_rows:
            conn.execute(
                "DELETE FROM insession_auxiliary_goal_budget_charges "
                "WHERE budget_charge_id=?",
                (str(charge_row["budget_charge_id"]),),
            )
        conn.execute(
            "DELETE FROM insession_auxiliary_authority_anchors WHERE "
            "authority_snapshot_id IN (SELECT authority_snapshot_id FROM "
            "insession_auxiliary_authority_snapshots WHERE session_id=?)",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_auxiliary_graph_revision_snapshots "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_auxiliary_authority_snapshots "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM insession_auxiliary_graph_goals WHERE session_id=?",
            (session_id,),
        )
        # 修订后的节点定义构成立即生效的 RESTRICT 谱系链。此时 revision membership
        # 已被移除，因此删除其辅助图容器前，按最新优先顺序释放每个持久化节点。
        definition_rows = conn.execute(
            "SELECT definition.auxiliary_graph_id, "
            "definition.auxiliary_node_id, definition.node_revision FROM "
            "insession_auxiliary_node_definitions_v2 AS definition "
            "JOIN insession_auxiliary_graph_v2_containers AS container "
            "ON container.auxiliary_graph_id=definition.auxiliary_graph_id "
            "WHERE container.session_id=? ORDER BY "
            "definition.auxiliary_graph_id, definition.auxiliary_node_id, "
            "definition.node_revision DESC",
            (session_id,),
        ).fetchall()
        for definition_row in definition_rows:
            conn.execute(
                "DELETE FROM insession_auxiliary_node_definitions_v2 WHERE "
                "auxiliary_graph_id=? AND auxiliary_node_id=? "
                "AND node_revision=?",
                (
                    str(definition_row["auxiliary_graph_id"]),
                    str(definition_row["auxiliary_node_id"]),
                    int(definition_row["node_revision"]),
                ),
            )
        conn.execute(
            "DELETE FROM insession_auxiliary_graph_v2_containers "
            "WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM turn_execution_windows WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM runtime_turns WHERE session_id=?",
            (session_id,),
        )
        for table in (
            "session_context_repair_applies", "session_context_resets", "session_turn_commits", "session_observation_candidates",
            "session_state_transitions", "session_state_items", "session_evidence_events",
            "session_turns", "session_working_memory", "session_context_revisions",
        ):
            conn.execute(f"DELETE FROM {table} WHERE session_id=?", (session_id,))
        # 诊断轨迹是同一权威状态中由 Session 持有的记录。在 Session 标识前移除它们，
        # 且仅回收不再被任何剩余 step 引用的 blob；全部操作都在清除事务内完成。
        conn.execute(
            "DELETE FROM trajectory_steps WHERE session_id=?",
            (session_id,),
        )
        conn.execute(
            "DELETE FROM trajectory_blobs WHERE sha256 NOT IN "
            "(SELECT blob_sha256 FROM trajectory_parts)"
        )
        conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
    return found


def list_trashed(deps: StoreDeps) -> list[dict[str, Any]]:
    deps.init_db()
    with deps.connect() as conn:
        rows = conn.execute("SELECT * FROM sessions WHERE status='trashed' ORDER BY deleted_at DESC").fetchall()
    return [dict(row) for row in rows]
