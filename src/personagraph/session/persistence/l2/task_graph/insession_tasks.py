"""由 Session 持有的 ``insession_task`` graph 的 SQLite 持久化。

本模块负责不可变 graph 快照、技术状态记录和 Turn 相关性链接。它有意不判定
用户意图、不调用模型/RAG provider、不创建 WorkRun，也不依据 observation
推进任务状态。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from personagraph.l2.task_graph.contracts import (
    InSessionTaskCatalog,
    InSessionTaskCatalogItem,
    InSessionTaskDetails,
    InSessionTaskGraphCommitResult,
    InSessionTaskGraphRevisionCommitResult,
    InSessionTaskGraphRevisionProposal,
    InSessionTaskGraphRevisionValidationContext,
    InSessionTaskGraphValidationContext,
    InSessionTaskRootGraphProposal,
    InSessionTaskSourceAnchor,
    InSessionTaskStatus,
    InSessionTaskTurnLinkResult,
    NewInSessionTaskGraphsProposal,
)
from personagraph.l2.task_graph.task_matching import (
    InSessionTaskMatchApplyResult,
    InSessionTaskMatchesProposal,
    InSessionTaskMatchingLimits,
    guard_insession_task_matches,
)
from personagraph.l2.task_graph.lane_manifest import (
    InSessionTaskExecutionLaneManifest,
    build_insession_task_execution_lane_manifest,
)
from personagraph.l2.task_graph.validation import (
    validate_current_user_anchor_spans,
    validate_insession_task_graph_revision,
    validate_new_insession_task_graphs,
)
from ....insession_task_contracts import (
    InSessionTaskApplyIdCollision,
    InSessionTaskGraphRevisionConflict,
    InSessionTaskGraphRevisionNotSupported,
    InSessionTaskPersistenceError,
    InSessionTaskStateVersionConflict,
)
from ....turn_execution_contracts import TurnExecutionWindowRevisionConflict
from ...deps import StoreDeps


_ROOT_STATUS = InSessionTaskStatus.PROPOSED.value
_TERMINAL_TASK_STATUSES = {
    InSessionTaskStatus.CANCELLED.value,
    InSessionTaskStatus.COMPLETED.value,
}
_A4A_ALLOWED_SOURCE_KINDS = {
    "current_user_instruction",
    "current_user_context",
}


@dataclass(frozen=True)
class _RootSourceManifest:
    """由一个已持久化根 graph 持有、而非其整个批次持有的来源事实。"""

    source_anchors: tuple[InSessionTaskSourceAnchor, ...]
    authorization_anchor_ids: tuple[str, ...]
    required_anchor_ids: tuple[str, ...]


@dataclass(frozen=True)
class _RevisionOneTransactionResult:
    """供共享此事务的另一个 Store 命令使用的私有输出。"""

    commit_result: InSessionTaskGraphRevisionCommitResult
    validated_proposal: InSessionTaskGraphRevisionProposal
    local_node_ids: dict[str, str]


def commit_insession_task_graph_revision(
    deps: StoreDeps,
    *,
    session_id: str,
    source_turn_id: str,
    target_insession_task_id: str,
    expected_current_graph_revision: int | None,
    expected_task_state_version: int,
    expected_window_revision: int,
    apply_id: str,
    proposal: InSessionTaskGraphRevisionProposal,
    trusted_context: InSessionTaskGraphRevisionValidationContext,
) -> InSessionTaskGraphRevisionCommitResult:
    """向一个已存在且绑定来源的根 Task shell 提交 revision 1。

    P1.1 有意只接受 ``None -> 1``。正数 base 需要节点标识/revision 连续性协议，
    因此这里采用失败关闭，而不会静默重建节点并使未来执行历史失效。
    """

    _require_nonempty("session_id", session_id)
    _require_nonempty("source_turn_id", source_turn_id)
    _require_nonempty("target_insession_task_id", target_insession_task_id)
    _require_nonempty("apply_id", apply_id)
    _require_positive_revision("expected_task_state_version", expected_task_state_version)
    _require_window_revision(expected_window_revision)
    if expected_current_graph_revision is not None:
        raise InSessionTaskGraphRevisionNotSupported()
    proposal_hash = _revision_proposal_hash(
        proposal,
        trusted_context=trusted_context,
        target_insession_task_id=target_insession_task_id,
        expected_current_graph_revision=expected_current_graph_revision,
        expected_task_state_version=expected_task_state_version,
        expected_window_revision=expected_window_revision,
    )

    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _require_session_turn(conn, session_id=session_id, turn_id=source_turn_id)
        existing = conn.execute(
            "SELECT session_id, source_turn_id, insession_task_id, "
            "expected_graph_revision, committed_graph_revision, "
            "expected_task_state_version, committed_task_state_version, "
            "expected_window_revision, committed_window_state_version, proposal_hash, "
            "turn_task_link_revision "
            "FROM insession_task_graph_revision_apply_receipts "
            "WHERE apply_id=?",
            (apply_id,),
        ).fetchone()
        if existing is not None:
            _validate_revision_replay(
                existing,
                session_id=session_id,
                source_turn_id=source_turn_id,
                target_insession_task_id=target_insession_task_id,
                proposal_hash=proposal_hash,
            )
            return _revision_commit_result(existing, status="replayed")

        mutation = _commit_graph_revision_one_in_transaction(
            conn,
            deps,
            session_id=session_id,
            source_turn_id=source_turn_id,
            target_insession_task_id=target_insession_task_id,
            expected_current_graph_revision=expected_current_graph_revision,
            expected_task_state_version=expected_task_state_version,
            expected_window_revision=expected_window_revision,
            apply_id=apply_id,
            proposal=proposal,
            trusted_context=trusted_context,
            proposal_hash=proposal_hash,
            now=now,
        )
        return mutation.commit_result


def _commit_graph_revision_one_in_transaction(
    conn: sqlite3.Connection,
    deps: StoreDeps,
    *,
    session_id: str,
    source_turn_id: str,
    target_insession_task_id: str,
    expected_current_graph_revision: int | None,
    expected_task_state_version: int,
    expected_window_revision: int,
    apply_id: str,
    proposal: InSessionTaskGraphRevisionProposal,
    trusted_context: InSessionTaskGraphRevisionValidationContext,
    proposal_hash: str,
    now: str,
) -> _RevisionOneTransactionResult:
    """在调用方持有的事务中执行 revision-one 的全部写入。

    通用 graph 命令与专用 Host recipe 共享这一精确变更核心，从而避免任何 recipe
    路径先在一个事务中提交通用 graph、再在后续事务中附加权威状态。
    """

    if not conn.in_transaction:
        raise InSessionTaskPersistenceError(
            "graph revision mutation requires an active transaction"
        )
    _require_active_window(
        conn,
        session_id=session_id,
        turn_id=source_turn_id,
        expected_window_revision=expected_window_revision,
    )
    task = _load_revision_target(
        conn,
        session_id=session_id,
        target_insession_task_id=target_insession_task_id,
    )
    actual_graph_revision = (
        int(task["current_graph_revision"])
        if task["current_graph_revision"] is not None
        else None
    )
    if actual_graph_revision != expected_current_graph_revision:
        raise InSessionTaskGraphRevisionConflict(
            expected=expected_current_graph_revision,
            actual=actual_graph_revision,
        )
    actual_task_state_version = int(task["state_version"])
    if actual_task_state_version != expected_task_state_version:
        raise InSessionTaskStateVersionConflict(
            expected=expected_task_state_version,
            actual=actual_task_state_version,
        )
    if str(task["current_status"]) in _TERMINAL_TASK_STATUSES:
        raise InSessionTaskPersistenceError(
            "terminal in-session Task cannot receive a graph revision"
        )
    _require_turn_root_task_link(
        conn,
        session_id=session_id,
        turn_id=source_turn_id,
        target_insession_task_id=target_insession_task_id,
    )
    proposal = _revalidate_revision_commit_authority(
        conn=conn,
        proposal=proposal,
        trusted_context=trusted_context,
        task=task,
        session_id=session_id,
        source_turn_id=source_turn_id,
        target_insession_task_id=target_insession_task_id,
        expected_current_graph_revision=expected_current_graph_revision,
    )
    root = proposal.root
    root_node = next(node for node in root.nodes if node.node_key == root.root_key)
    source_manifest = _derive_root_source_manifest(root, trusted_context)
    local_node_ids = _allocate_revision_one_node_ids(
        conn,
        deps,
        target_insession_task_id=target_insession_task_id,
        root=root,
    )
    _insert_graph_revision_one(
        conn,
        task_id=target_insession_task_id,
        source_turn_id=source_turn_id,
        proposal_hash=proposal_hash,
        source_manifest=source_manifest,
        now=now,
    )
    _insert_graph_nodes_and_edges(
        conn,
        task_id=target_insession_task_id,
        root_nodes=root.nodes,
        local_node_ids=local_node_ids,
        now=now,
    )
    committed_task_state_version = expected_task_state_version + 1
    updated = conn.execute(
        "UPDATE insession_tasks SET current_graph_revision=1, state_version=?, "
        "root_title=?, root_objective=?, updated_at=? "
        "WHERE insession_task_id=? AND session_id=? "
        "AND current_graph_revision IS NULL AND state_version=? "
        "AND current_status NOT IN ('cancelled', 'completed')",
        (
            committed_task_state_version,
            root_node.title,
            root_node.objective,
            now,
            target_insession_task_id,
            session_id,
            expected_task_state_version,
        ),
    )
    if updated.rowcount != 1:
        raise InSessionTaskPersistenceError(
            "in-session Task changed during graph revision commit"
        )
    turn_task_link_revision = _current_turn_link_revision(
        conn, session_id, source_turn_id
    )
    committed_window_state_version = _advance_window_task_link_pointer(
        conn,
        session_id=session_id,
        turn_id=source_turn_id,
        link_revision=turn_task_link_revision,
        expected_window_revision=expected_window_revision,
        now=now,
    )
    if committed_window_state_version is None:
        raise InSessionTaskPersistenceError(
            "TaskGraph revision commit lost its Turn execution window"
        )
    conn.execute(
        "INSERT INTO insession_task_graph_revision_apply_receipts "
        "(apply_id, session_id, source_turn_id, insession_task_id, "
        "expected_graph_revision, committed_graph_revision, "
        "expected_task_state_version, committed_task_state_version, "
        "expected_window_revision, committed_window_state_version, proposal_hash, "
        "turn_task_link_revision, created_at) "
        "VALUES (?, ?, ?, ?, NULL, 1, ?, ?, ?, ?, ?, ?, ?)",
        (
            apply_id,
            session_id,
            source_turn_id,
            target_insession_task_id,
            expected_task_state_version,
            committed_task_state_version,
            expected_window_revision,
            committed_window_state_version,
            proposal_hash,
            turn_task_link_revision,
            now,
        ),
    )
    commit_result = InSessionTaskGraphRevisionCommitResult(
        status="applied",
        insession_task_id=target_insession_task_id,
        previous_graph_revision=None,
        committed_graph_revision=1,
        task_state_version=committed_task_state_version,
        turn_task_link_revision=turn_task_link_revision,
        window_state_version=committed_window_state_version,
    )
    return _RevisionOneTransactionResult(
        commit_result=commit_result,
        validated_proposal=proposal,
        local_node_ids=local_node_ids,
    )


def apply_insession_task_matches(
    deps: StoreDeps,
    *,
    session_id: str,
    source_turn_id: str,
    apply_id: str,
    proposal: InSessionTaskMatchesProposal,
    exposed_catalog_ids: tuple[str, ...],
    expected_window_revision: int,
    limits: InSessionTaskMatchingLimits | None = None,
) -> InSessionTaskMatchApplyResult:
    """以原子方式重新验证并持久化一个入口 task-match 提案。

    面向模型的 catalog、调用方 Guard 结果和来源摘录在此边界均不可信。事务会在
    创建任何 root shell、Turn link、branch intent 或 receipt 前，重新加载不可变
    的已接受输入和 Session 持有的 catalog 子集。
    """

    _require_nonempty("session_id", session_id)
    _require_nonempty("source_turn_id", source_turn_id)
    _require_nonempty("apply_id", apply_id)
    _require_window_revision(expected_window_revision)
    if len(exposed_catalog_ids) != len(set(exposed_catalog_ids)):
        raise InSessionTaskPersistenceError("exposed Task catalog contains duplicate ids")
    effective_limits = limits or InSessionTaskMatchingLimits()
    proposal_hash = _task_match_proposal_hash(
        proposal,
        exposed_catalog_ids=exposed_catalog_ids,
        limits=effective_limits,
    )

    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _require_session_turn(conn, session_id=session_id, turn_id=source_turn_id)
        authoritative_user_text = _load_authoritative_user_input(
            conn,
            session_id=session_id,
            turn_id=source_turn_id,
        )
        exposed_catalog = _load_exposed_task_catalog(
            conn,
            session_id=session_id,
            exposed_catalog_ids=exposed_catalog_ids,
        )
        guarded = guard_insession_task_matches(
            proposal,
            authoritative_user_text=authoritative_user_text,
            trusted_root_catalog=exposed_catalog,
            limits=effective_limits,
        )
        if guarded.status != "accepted":
            codes = ",".join(code.value for code in guarded.error_codes)
            raise InSessionTaskPersistenceError(
                f"entry task-match proposal failed deterministic validation: {codes}"
            )

        existing = conn.execute(
            "SELECT session_id, source_turn_id, proposal_hash, "
            "created_task_mapping_json, related_insession_task_ids_json, "
            "branch_intent_ids_json, turn_task_link_revision, window_state_version "
            ", execution_lane_manifest_json, execution_lane_manifest_hash "
            "FROM insession_task_match_apply_receipts WHERE apply_id=?",
            (apply_id,),
        ).fetchone()
        if existing is not None:
            _validate_task_match_replay(
                existing,
                session_id=session_id,
                source_turn_id=source_turn_id,
                proposal_hash=proposal_hash,
            )
            replay_mapping = _decode_string_mapping(
                existing["created_task_mapping_json"]
            )
            expected_manifest = build_insession_task_execution_lane_manifest(
                guarded,
                created_insession_task_ids_by_local_key=replay_mapping,
            )
            _load_task_match_lane_manifest(
                existing,
                expected=expected_manifest,
            )
            return InSessionTaskMatchApplyResult(
                status="replayed",
                created_insession_task_ids_by_local_key=replay_mapping,
                related_insession_task_ids=_decode_ids(
                    existing["related_insession_task_ids_json"]
                ),
                branch_intent_ids=_decode_ids(existing["branch_intent_ids_json"]),
                turn_task_link_revision=int(existing["turn_task_link_revision"]),
                window_state_version=(
                    int(existing["window_state_version"])
                    if existing["window_state_version"] is not None
                    else None
                ),
            )

        _require_active_window(
            conn,
            session_id=session_id,
            turn_id=source_turn_id,
            expected_window_revision=expected_window_revision,
        )
        created_by_local_key: dict[str, str] = {}
        related_task_ids: list[str] = []
        branch_intent_ids: list[str] = []
        created_task_ids: set[str] = set()
        for accepted in guarded.accepted_task_matches:
            match = accepted.proposal
            if match.match_type == "new_root":
                task_id = _allocate_id(conn, deps, "insession_task")
                conn.execute(
                    "INSERT INTO insession_tasks "
                    "(insession_task_id, session_id, current_graph_revision, current_status, "
                    "state_version, root_title, root_objective, created_turn_id, "
                    "creation_source_start, creation_source_end, creation_source_sha256, "
                    "created_at, updated_at) VALUES (?, ?, NULL, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        task_id,
                        session_id,
                        _ROOT_STATUS,
                        match.title.strip(),
                        match.objective.strip(),
                        source_turn_id,
                        accepted.source_span.start,
                        accepted.source_span.end,
                        accepted.source_span.text_sha256,
                        now,
                        now,
                    ),
                )
                created_by_local_key[match.local_key] = task_id
                created_task_ids.add(task_id)
            else:
                task_id = match.insession_task_id
                if match.match_type == "existing_root_branch":
                    intent_id = _allocate_branch_intent_id(conn, deps)
                    conn.execute(
                        "INSERT INTO insession_task_branch_intents "
                        "(branch_intent_id, session_id, source_turn_id, insession_task_id, "
                        "branch_key, branch_summary, source_start, source_end, source_sha256, "
                        "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            intent_id,
                            session_id,
                            source_turn_id,
                            task_id,
                            match.branch_key,
                            match.branch_summary.strip(),
                            accepted.source_span.start,
                            accepted.source_span.end,
                            accepted.source_span.text_sha256,
                            now,
                        ),
                    )
                    branch_intent_ids.append(intent_id)
            if task_id not in related_task_ids:
                related_task_ids.append(task_id)

        lane_manifest = build_insession_task_execution_lane_manifest(
            guarded,
            created_insession_task_ids_by_local_key=created_by_local_key,
        )
        related_task_ids = [
            lane.insession_task_id for lane in lane_manifest.lanes
        ]

        link_revision, window_state_version = _link_task_match_roots_in_transaction(
            conn,
            session_id=session_id,
            turn_id=source_turn_id,
            insession_task_ids=tuple(related_task_ids),
            created_task_ids=created_task_ids,
            expected_window_revision=expected_window_revision,
            now=now,
        )
        conn.execute(
            "INSERT INTO insession_task_match_apply_receipts "
            "(apply_id, session_id, source_turn_id, proposal_hash, "
            "created_task_mapping_json, related_insession_task_ids_json, "
            "branch_intent_ids_json, turn_task_link_revision, window_state_version, "
            "execution_lane_manifest_json, execution_lane_manifest_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                apply_id,
                session_id,
                source_turn_id,
                proposal_hash,
                _canonical_json(created_by_local_key),
                _canonical_json(related_task_ids),
                _canonical_json(branch_intent_ids),
                link_revision,
                window_state_version,
                _canonical_json(
                    lane_manifest.model_dump(mode="json", exclude_none=True)
                ),
                lane_manifest.manifest_sha256,
                now,
            ),
        )
        return InSessionTaskMatchApplyResult(
            status="applied",
            created_insession_task_ids_by_local_key=created_by_local_key,
            related_insession_task_ids=tuple(related_task_ids),
            branch_intent_ids=tuple(branch_intent_ids),
            turn_task_link_revision=link_revision,
            window_state_version=window_state_version,
        )


def commit_new_insession_task_graphs(
    deps: StoreDeps,
    *,
    session_id: str,
    source_turn_id: str,
    apply_id: str,
    proposal: NewInSessionTaskGraphsProposal,
    trusted_context: InSessionTaskGraphValidationContext,
    expected_window_revision: int,
) -> InSessionTaskGraphCommitResult:
    """以原子方式物化一个绑定 host 且已重新验证的创建提案。

    单个 apply receipt 覆盖整个批次。重放会返回完全相同的原始 root ID；若复用
    幂等键但改变内容，则抛出冲突而不是创建第二个 graph。持久化边界绝不信任
    调用方提供的 ``accepted`` 标志：它接收提案和 host 构建的 source manifest，
    再在同一变更事务中运行确定性验证。
    """

    _require_nonempty("session_id", session_id)
    _require_nonempty("source_turn_id", source_turn_id)
    _require_nonempty("apply_id", apply_id)
    _require_window_revision(expected_window_revision)
    proposal_hash = _proposal_hash(proposal, trusted_context)

    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _require_session_turn(conn, session_id=session_id, turn_id=source_turn_id)
        proposal = _revalidate_commit_authority(
            conn=conn,
            proposal=proposal,
            trusted_context=trusted_context,
            session_id=session_id,
            source_turn_id=source_turn_id,
        )
        existing = conn.execute(
            "SELECT session_id, source_turn_id, operation, proposal_hash, "
            "created_insession_task_ids_json, turn_task_link_revision "
            "FROM insession_task_graph_apply_receipts WHERE apply_id=?",
            (apply_id,),
        ).fetchone()
        if existing is not None:
            _validate_create_replay(
                existing,
                session_id=session_id,
                source_turn_id=source_turn_id,
                proposal_hash=proposal_hash,
            )
            task_ids = _decode_ids(existing["created_insession_task_ids_json"])
            return InSessionTaskGraphCommitResult(
                status="replayed",
                created_insession_task_ids=task_ids,
                turn_task_link_revision=int(existing["turn_task_link_revision"]),
                window_state_version=_window_state_version(conn, session_id, source_turn_id),
            )

        _require_active_window(
            conn,
            session_id=session_id,
            turn_id=source_turn_id,
            expected_window_revision=expected_window_revision,
        )

        created_task_ids: list[str] = []
        for root in proposal.roots:
            task_id, local_node_ids = _allocate_graph_ids(conn, deps, root.nodes, root.root_key)
            root_node = next(node for node in root.nodes if node.node_key == root.root_key)
            source_manifest = _derive_root_source_manifest(root, trusted_context)
            _insert_task_root(
                conn,
                task_id=task_id,
                session_id=session_id,
                source_turn_id=source_turn_id,
                title=root_node.title,
                objective=root_node.objective,
                proposal_hash=proposal_hash,
                source_manifest=source_manifest,
                now=now,
            )
            _insert_graph_nodes_and_edges(
                conn,
                task_id=task_id,
                root_nodes=root.nodes,
                local_node_ids=local_node_ids,
                now=now,
            )
            created_task_ids.append(task_id)

        link_revision, window_state_version = _link_turn_to_tasks_in_transaction(
            conn,
            session_id=session_id,
            turn_id=source_turn_id,
            insession_task_ids=tuple(created_task_ids),
            relation="created",
            expected_window_revision=expected_window_revision,
            now=now,
        )
        conn.execute(
            "INSERT INTO insession_task_graph_apply_receipts "
            "(apply_id, session_id, source_turn_id, operation, proposal_hash, "
            "created_insession_task_ids_json, turn_task_link_revision, created_at) "
            "VALUES (?, ?, ?, 'create', ?, ?, ?, ?)",
            (
                apply_id,
                session_id,
                source_turn_id,
                proposal_hash,
                _canonical_json(created_task_ids),
                link_revision,
                now,
            ),
        )
        return InSessionTaskGraphCommitResult(
            status="applied",
            created_insession_task_ids=tuple(created_task_ids),
            turn_task_link_revision=link_revision,
            window_state_version=window_state_version,
        )


def link_turn_to_insession_tasks(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    insession_task_ids: tuple[str, ...],
    expected_window_revision: int,
) -> InSessionTaskTurnLinkResult:
    """完成确定性归属检查后，记录某 Turn 的根级相关性。"""

    _require_nonempty("session_id", session_id)
    _require_nonempty("turn_id", turn_id)
    _require_window_revision(expected_window_revision)
    if len(insession_task_ids) != len(set(insession_task_ids)):
        raise InSessionTaskPersistenceError("duplicate insession task ids in one Turn link")
    if not insession_task_ids:
        return InSessionTaskTurnLinkResult()

    deps.init_db()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _require_session_turn(conn, session_id=session_id, turn_id=turn_id)
        link_revision, window_state_version = _link_turn_to_tasks_in_transaction(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            insession_task_ids=insession_task_ids,
            relation="referenced",
            expected_window_revision=expected_window_revision,
            now=deps.now(),
        )
        return InSessionTaskTurnLinkResult(
            linked_insession_task_ids=insession_task_ids,
            turn_task_link_revision=link_revision,
            window_state_version=window_state_version,
        )


def list_insession_task_catalog(
    deps: StoreDeps,
    session_id: str,
    *,
    limit: int | None = None,
) -> tuple[InSessionTaskCatalogItem, ...]:
    """读取紧凑的根事实；token 打包与截断由组合层负责。"""

    _require_nonempty("session_id", session_id)
    if limit is not None and limit < 1:
        return ()
    deps.init_db()
    with deps.connect() as conn:
        _require_session(conn, session_id)
        query = (
            "SELECT task.insession_task_id, task.session_id, task.root_title, task.root_objective, task.current_status, "
            "task.current_graph_revision, task.created_turn_id, task.creation_source_start, "
            "task.creation_source_end, task.creation_source_sha256, "
            "revision.source_turn_id, revision.source_anchors_json, "
            "revision.authorization_anchor_ids_json FROM insession_tasks AS task "
            "LEFT JOIN insession_task_graph_revisions AS revision "
            "ON revision.insession_task_id=task.insession_task_id "
            "AND revision.graph_revision=task.current_graph_revision "
            "WHERE task.session_id=? "
            "ORDER BY CASE WHEN task.current_status IN ('completed', 'cancelled') THEN 1 ELSE 0 END, "
            "task.updated_at DESC, task.insession_task_id"
        )
        rows = conn.execute(query, (session_id,)).fetchall()
        items: list[InSessionTaskCatalogItem] = []
        for row in rows:
            revision = row["current_graph_revision"]
            readable = (
                _has_readable_shell_provenance(conn, row)
                if revision is None
                else _has_readable_authorization_manifest(
                    conn,
                    row,
                    row["source_anchors_json"],
                    row["authorization_anchor_ids_json"],
                    source_turn_id=str(row["source_turn_id"]),
                )
            )
            if not readable:
                # 缺少 manifest 的 Task 无法验证，不能静默作为可信 catalog 事实
                # 提供给模型。
                continue
            items.append(
                InSessionTaskCatalogItem(
                    insession_task_id=str(row["insession_task_id"]),
                    goal_summary=_goal_summary(row["root_title"], row["root_objective"]),
                    status=InSessionTaskStatus(str(row["current_status"])),
                    current_graph_revision=(int(revision) if revision is not None else None),
                )
            )
    return tuple(items[:limit] if limit is not None else items)


def list_turn_insession_task_ids(
    deps: StoreDeps,
    session_id: str,
    turn_id: str,
) -> tuple[str, ...]:
    """按首次链接顺序返回某 Turn 的根 Task 链接。"""

    _require_nonempty("session_id", session_id)
    _require_nonempty("turn_id", turn_id)
    deps.init_db()
    with deps.connect() as conn:
        _require_session_turn(conn, session_id=session_id, turn_id=turn_id)
        rows = conn.execute(
            "SELECT insession_task_id, MIN(link_id) AS first_link_id "
            "FROM insession_task_turn_links WHERE session_id=? AND turn_id=? "
            "GROUP BY insession_task_id ORDER BY first_link_id, insession_task_id",
            (session_id, turn_id),
        ).fetchall()
    return tuple(str(row["insession_task_id"]) for row in rows)


def get_insession_task_execution_lane_manifest(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
) -> InSessionTaskExecutionLaneManifest:
    """加载 Task matching 所持久化的一份来源绑定 lane manifest。"""

    _require_nonempty("session_id", session_id)
    _require_nonempty("turn_id", turn_id)
    deps.init_db()
    with deps.connect() as conn:
        _require_session_turn(conn, session_id=session_id, turn_id=turn_id)
        rows = conn.execute(
            "SELECT related_insession_task_ids_json, "
            "execution_lane_manifest_json, execution_lane_manifest_hash "
            "FROM insession_task_match_apply_receipts "
            "WHERE session_id=? AND source_turn_id=? ORDER BY created_at, apply_id",
            (session_id, turn_id),
        ).fetchall()
        if len(rows) != 1:
            raise InSessionTaskPersistenceError(
                "Turn has no unique durable Task execution lane manifest"
            )
        manifest = _load_task_match_lane_manifest(rows[0])
        related_task_ids = _decode_ids(rows[0]["related_insession_task_ids_json"])
        if tuple(lane.insession_task_id for lane in manifest.lanes) != related_task_ids:
            raise InSessionTaskPersistenceError(
                "Task execution lane manifest disagrees with Task-match receipt"
            )
        authoritative_user_text = _load_authoritative_user_input(
            conn,
            session_id=session_id,
            turn_id=turn_id,
        )
        for lane in manifest.lanes:
            if conn.execute(
                "SELECT 1 FROM insession_tasks WHERE session_id=? "
                "AND insession_task_id=?",
                (session_id, lane.insession_task_id),
            ).fetchone() is None:
                raise InSessionTaskPersistenceError(
                    "Task execution lane references a missing Task"
                )
            for item in lane.matches:
                span = item.source_span
                excerpt = authoritative_user_text[span.start : span.end]
                if hashlib.sha256(excerpt.encode("utf-8")).hexdigest() != span.text_sha256:
                    raise InSessionTaskPersistenceError(
                        "Task execution lane source binding has drifted"
                    )
        return manifest


def get_insession_task_creation_source(
    deps: StoreDeps,
    *,
    session_id: str,
    insession_task_id: str,
) -> InSessionTaskSourceAnchor:
    """投影创建某个根 Task shell 的不可变来源 span。"""

    _require_nonempty("session_id", session_id)
    _require_nonempty("insession_task_id", insession_task_id)
    deps.init_db()
    with deps.connect() as conn:
        row = conn.execute(
            "SELECT created_turn_id, creation_source_start, creation_source_end, "
            "creation_source_sha256 FROM insession_tasks "
            "WHERE session_id=? AND insession_task_id=?",
            (session_id, insession_task_id),
        ).fetchone()
        if row is None:
            raise InSessionTaskPersistenceError("unknown Task in this Session")
        if any(
            row[field] is None
            for field in (
                "created_turn_id",
                "creation_source_start",
                "creation_source_end",
                "creation_source_sha256",
            )
        ):
            raise InSessionTaskPersistenceError(
                "Task has no immutable creation-source authority"
            )
        turn_id = str(row["created_turn_id"])
        start = int(row["creation_source_start"])
        end = int(row["creation_source_end"])
        authoritative_text = _load_authoritative_user_input(
            conn,
            session_id=session_id,
            turn_id=turn_id,
        )
        if start < 0 or end <= start or end > len(authoritative_text):
            raise InSessionTaskPersistenceError(
                "Task creation-source range is corrupt"
            )
        excerpt = authoritative_text[start:end]
        if hashlib.sha256(excerpt.encode("utf-8")).hexdigest() != str(
            row["creation_source_sha256"]
        ):
            raise InSessionTaskPersistenceError(
                "Task creation-source digest has drifted"
            )
        return InSessionTaskSourceAnchor(
            anchor_id="task_creation_source",
            source_turn_id=turn_id,
            source_kind="previously_authorized_task_state",
            start=start,
            end=end,
            excerpt=excerpt,
        )


def get_insession_task_details(
    deps: StoreDeps,
    session_id: str,
    insession_task_id: str,
) -> InSessionTaskDetails | None:
    """将一个当前 graph 加载为稳定树投影，而不返回原始记录。"""

    _require_nonempty("session_id", session_id)
    _require_nonempty("insession_task_id", insession_task_id)
    deps.init_db()
    with deps.connect() as conn:
        task = conn.execute(
            "SELECT task.insession_task_id, task.session_id, task.current_graph_revision, "
            "task.current_status, task.state_version, task.root_title, task.root_objective, task.created_turn_id, "
            "task.creation_source_start, task.creation_source_end, task.creation_source_sha256, "
            "revision.source_turn_id, revision.source_anchors_json, "
            "revision.authorization_anchor_ids_json, revision.required_anchor_ids_json "
            "FROM insession_tasks AS task "
            "LEFT JOIN insession_task_graph_revisions AS revision "
            "ON revision.insession_task_id=task.insession_task_id "
            "AND revision.graph_revision=task.current_graph_revision "
            "WHERE task.insession_task_id=? AND task.session_id=?",
            (insession_task_id, session_id),
        ).fetchone()
        if task is None:
            return None
        revision_value = task["current_graph_revision"]
        if revision_value is None:
            if not _has_readable_shell_provenance(conn, task):
                raise InSessionTaskPersistenceError(
                    "root Task shell source provenance is missing or corrupt"
                )
            node_rows = ()
        else:
            revision = int(revision_value)
            node_rows = conn.execute(
            "SELECT node.insession_task_node_id, node.node_revision, node.node_kind, node.ordinal, "
            "node.title, node.objective, node.source_anchor_ids_json, "
            "node.acceptance_criteria_json, node.constraints_json, edge.parent_insession_task_node_id, "
            "state.status, state.state_version "
            "FROM insession_task_graph_nodes AS node "
            "LEFT JOIN insession_task_graph_edges AS edge "
            "ON edge.insession_task_id=node.insession_task_id "
            "AND edge.graph_revision=node.graph_revision "
            "AND edge.child_insession_task_node_id=node.insession_task_node_id "
            "LEFT JOIN insession_task_node_states AS state "
            "ON state.insession_task_id=node.insession_task_id "
            "AND state.insession_task_node_id=node.insession_task_node_id "
            "AND state.node_revision=node.node_revision "
            "WHERE node.insession_task_id=? AND node.graph_revision=? "
            "ORDER BY node.ordinal, node.insession_task_node_id",
                (insession_task_id, revision),
            ).fetchall()
            manifest_readable = _has_readable_authorization_manifest(
                conn,
                task,
                task["source_anchors_json"],
                task["authorization_anchor_ids_json"],
                source_turn_id=str(task["source_turn_id"]),
            )
        related_turn_count_row = conn.execute(
            "SELECT COUNT(DISTINCT turn_id) AS related_turn_count FROM insession_task_turn_links "
            "WHERE insession_task_id=? AND session_id=?",
            (insession_task_id, session_id),
        ).fetchone()

    nodes = tuple(_node_projection(row) for row in node_rows)
    if revision_value is None:
        revision = None
        source_anchors = ()
        authorization_anchor_ids = ()
        required_anchor_ids = ()
    else:
        source_anchors = _source_anchors_from_json(task["source_anchors_json"])
        authorization_anchor_ids = _decode_ids(task["authorization_anchor_ids_json"])
        required_anchor_ids = _decode_ids(task["required_anchor_ids_json"])
        if not manifest_readable:
            raise InSessionTaskPersistenceError(
                "current TaskGraph source manifest is missing or corrupt"
            )
    if related_turn_count_row is None:
        raise InSessionTaskPersistenceError("stored insession task link count is missing")
    return InSessionTaskDetails(
        insession_task_id=str(task["insession_task_id"]),
        session_id=str(task["session_id"]),
        task_state_version=int(task["state_version"]),
        title=str(task["root_title"]),
        objective=str(task["root_objective"]),
        current_graph_revision=revision,
        source_turn_id=(
            None
            if revision_value is None
            else str(task["source_turn_id"])
        ),
        status=InSessionTaskStatus(str(task["current_status"])),
        nodes=nodes,
        source_anchors=source_anchors,
        authorization_anchor_ids=authorization_anchor_ids,
        required_anchor_ids=required_anchor_ids,
        related_turn_count=int(related_turn_count_row["related_turn_count"]),
        work_run_summaries=(),
    )


def _revalidate_commit_authority(
    *,
    conn: sqlite3.Connection,
    proposal: NewInSessionTaskGraphsProposal,
    trusted_context: InSessionTaskGraphValidationContext,
    session_id: str,
    source_turn_id: str,
) -> NewInSessionTaskGraphsProposal:
    """拒绝伪造的 'accepted' 标志或不匹配的 host source manifest。"""

    if trusted_context.session_id != session_id:
        raise InSessionTaskPersistenceError("trusted TaskGraph context is outside this Session")
    if trusted_context.source_turn_id != source_turn_id:
        raise InSessionTaskPersistenceError("trusted TaskGraph context is bound to another Turn")
    if proposal.source_turn_id != source_turn_id:
        raise InSessionTaskPersistenceError("proposal source Turn does not match commit Turn")
    _require_a4a_source_kinds(trusted_context)
    authoritative_user_text = _load_authoritative_user_input(
        conn,
        session_id=session_id,
        turn_id=source_turn_id,
    )
    span_errors = validate_current_user_anchor_spans(
        trusted_context,
        user_text=authoritative_user_text,
    )
    if span_errors:
        codes = ",".join(code.value for code in span_errors)
        raise InSessionTaskPersistenceError(
            f"TaskGraph source manifest failed authoritative input validation: {codes}"
        )
    validation = validate_new_insession_task_graphs(proposal, context=trusted_context)
    if validation.status != "accepted" or validation.proposal is None:
        codes = ",".join(code.value for code in validation.error_codes)
        raise InSessionTaskPersistenceError(f"TaskGraph proposal failed deterministic validation: {codes}")
    return validation.proposal


def _revalidate_revision_commit_authority(
    *,
    conn: sqlite3.Connection,
    proposal: InSessionTaskGraphRevisionProposal,
    trusted_context: InSessionTaskGraphRevisionValidationContext,
    task: sqlite3.Row,
    session_id: str,
    source_turn_id: str,
    target_insession_task_id: str,
    expected_current_graph_revision: int | None,
) -> InSessionTaskGraphRevisionProposal:
    """在事务内从不可变 Session 记录重建 P1.1 权威状态。"""

    if trusted_context.session_id != session_id:
        raise InSessionTaskPersistenceError(
            "trusted TaskGraph revision context is outside this Session"
        )
    if trusted_context.source_turn_id != source_turn_id:
        raise InSessionTaskPersistenceError(
            "trusted TaskGraph revision context is bound to another Turn"
        )
    if trusted_context.target_insession_task_id != target_insession_task_id:
        raise InSessionTaskPersistenceError(
            "trusted TaskGraph revision context targets another Task"
        )
    if (
        trusted_context.expected_current_graph_revision
        != expected_current_graph_revision
    ):
        raise InSessionTaskPersistenceError(
            "trusted TaskGraph revision context has another base revision"
        )
    if not _has_readable_shell_provenance(conn, task):
        raise InSessionTaskPersistenceError(
            "root Task shell source provenance is missing or corrupt"
        )

    supported_kinds = {
        "current_user_instruction",
        "current_user_context",
        "previously_authorized_task_state",
    }
    unsupported = sorted(
        {
            anchor.source_kind
            for anchor in trusted_context.source_anchors
            if anchor.source_kind not in supported_kinds
        }
    )
    if unsupported:
        raise InSessionTaskPersistenceError(
            "P1.1 TaskGraph revision source manifest contains unsupported source kinds: "
            + ",".join(unsupported)
        )

    authoritative_user_text = _load_authoritative_user_input(
        conn,
        session_id=session_id,
        turn_id=source_turn_id,
    )
    span_errors = validate_current_user_anchor_spans(
        trusted_context,
        user_text=authoritative_user_text,
    )
    if span_errors:
        codes = ",".join(code.value for code in span_errors)
        raise InSessionTaskPersistenceError(
            "TaskGraph revision source manifest failed authoritative input validation: "
            + codes
        )
    for anchor in trusted_context.source_anchors:
        if (
            anchor.source_kind == "previously_authorized_task_state"
            and not _matches_shell_creation_anchor(conn, task, anchor)
        ):
            raise InSessionTaskPersistenceError(
                "TaskGraph revision prior Task anchor does not match shell authority"
            )

    validation = validate_insession_task_graph_revision(
        proposal,
        context=trusted_context,
    )
    if validation.status != "accepted" or validation.proposal is None:
        codes = ",".join(code.value for code in validation.error_codes)
        raise InSessionTaskPersistenceError(
            f"TaskGraph revision proposal failed deterministic validation: {codes}"
        )
    return validation.proposal


def _load_revision_target(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    target_insession_task_id: str,
) -> sqlite3.Row:
    task = conn.execute(
        "SELECT insession_task_id, session_id, current_graph_revision, current_status, "
        "state_version, root_title, root_objective, created_turn_id, "
        "creation_source_start, creation_source_end, creation_source_sha256 "
        "FROM insession_tasks WHERE insession_task_id=?",
        (target_insession_task_id,),
    ).fetchone()
    if task is None or str(task["session_id"]) != session_id:
        raise InSessionTaskPersistenceError(
            "in-session Task is unknown or outside this Session"
        )
    return task


def _require_turn_root_task_link(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    target_insession_task_id: str,
) -> None:
    linked = conn.execute(
        "SELECT 1 FROM insession_task_turn_links "
        "WHERE session_id=? AND turn_id=? AND insession_task_id=? "
        "AND insession_task_node_id IS NULL",
        (session_id, turn_id, target_insession_task_id),
    ).fetchone()
    if linked is None:
        raise InSessionTaskPersistenceError(
            "TaskGraph revision requires a current Turn root-Task link"
        )


def _matches_shell_creation_anchor(
    conn: sqlite3.Connection,
    task: sqlite3.Row,
    anchor: InSessionTaskSourceAnchor,
) -> bool:
    try:
        created_turn_id = str(task["created_turn_id"])
        start = int(task["creation_source_start"])
        end = int(task["creation_source_end"])
        expected_hash = str(task["creation_source_sha256"])
        session_id = str(task["session_id"])
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    if (
        anchor.source_turn_id != created_turn_id
        or anchor.start != start
        or anchor.end != end
        or hashlib.sha256(anchor.excerpt.encode("utf-8")).hexdigest()
        != expected_hash
    ):
        return False
    try:
        source = _load_authoritative_user_input(
            conn,
            session_id=session_id,
            turn_id=created_turn_id,
        )
    except InSessionTaskPersistenceError:
        return False
    return end <= len(source) and source[start:end] == anchor.excerpt


def _require_a4a_source_kinds(
    trusted_context: InSessionTaskGraphValidationContext,
) -> None:
    """在证据锚点具有稳定资源标识前保持失败关闭。"""

    unsupported = sorted(
        {
            anchor.source_kind
            for anchor in trusted_context.source_anchors
            if anchor.source_kind not in _A4A_ALLOWED_SOURCE_KINDS
        }
    )
    if unsupported:
        raise InSessionTaskPersistenceError(
            "A4a TaskGraph source manifest contains unsupported source kinds: "
            + ",".join(unsupported)
        )


def _load_authoritative_user_input(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
) -> str:
    """读取不可变的已接受输入，绝不读取兼容副本。"""

    row = conn.execute(
        "SELECT turn_input.content FROM runtime_turn_inputs AS input "
        "JOIN session_turns AS turn_input "
        "ON turn_input.session_id=input.session_id AND turn_input.turn_idx=input.turn_idx "
        "WHERE input.session_id=? AND input.turn_id=? AND turn_input.role='user'",
        (session_id, turn_id),
    ).fetchone()
    if row is None:
        raise InSessionTaskPersistenceError(
            "TaskGraph source Turn has no authoritative accepted user input"
        )
    return str(row["content"])


def _derive_root_source_manifest(
    root: InSessionTaskRootGraphProposal,
    trusted_context: InSessionTaskGraphValidationContext,
) -> _RootSourceManifest:
    """将批次 manifest 投影到某根节点实际使用的锚点上。"""

    referenced_anchor_ids: set[str] = set()
    acceptance_anchor_ids: set[str] = set()
    for node in root.nodes:
        referenced_anchor_ids.update(node.source_anchor_ids)
        for acceptance in node.acceptance_criteria:
            referenced_anchor_ids.update(acceptance.source_anchor_ids)
            acceptance_anchor_ids.update(acceptance.source_anchor_ids)

    source_anchors = tuple(
        anchor
        for anchor in trusted_context.source_anchors
        if anchor.anchor_id in referenced_anchor_ids
    )
    authorization_anchor_ids = tuple(
        anchor_id
        for anchor_id in trusted_context.authorization_anchor_ids
        if anchor_id in referenced_anchor_ids
    )
    required_anchor_ids = tuple(
        anchor_id
        for anchor_id in trusted_context.required_anchor_ids
        if anchor_id in acceptance_anchor_ids
    )
    if not source_anchors or not authorization_anchor_ids:
        raise InSessionTaskPersistenceError(
            "accepted TaskGraph root lacks a source-bound authorization manifest"
        )
    return _RootSourceManifest(
        source_anchors=source_anchors,
        authorization_anchor_ids=authorization_anchor_ids,
        required_anchor_ids=required_anchor_ids,
    )


def _proposal_hash(
    proposal: NewInSessionTaskGraphsProposal,
    trusted_context: InSessionTaskGraphValidationContext,
) -> str:
    payload = {
        "proposal": proposal.model_dump(mode="json"),
        "trusted_context": trusted_context.model_dump(mode="json"),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _revision_proposal_hash(
    proposal: InSessionTaskGraphRevisionProposal,
    *,
    trusted_context: InSessionTaskGraphRevisionValidationContext,
    target_insession_task_id: str,
    expected_current_graph_revision: int | None,
    expected_task_state_version: int,
    expected_window_revision: int,
) -> str:
    payload = {
        "proposal": proposal.model_dump(mode="json"),
        "trusted_context": trusted_context.model_dump(mode="json"),
        "target_insession_task_id": target_insession_task_id,
        "expected_current_graph_revision": expected_current_graph_revision,
        "expected_task_state_version": expected_task_state_version,
        "expected_window_revision": expected_window_revision,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _task_match_proposal_hash(
    proposal: InSessionTaskMatchesProposal,
    *,
    exposed_catalog_ids: tuple[str, ...],
    limits: InSessionTaskMatchingLimits,
) -> str:
    proposal_payload = proposal.model_dump(mode="json")
    payload = {
        "proposal": proposal_payload,
        "exposed_catalog_ids": sorted(exposed_catalog_ids),
        "limits": limits.model_dump(mode="json"),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _load_exposed_task_catalog(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    exposed_catalog_ids: tuple[str, ...],
) -> InSessionTaskCatalog:
    if not exposed_catalog_ids:
        return InSessionTaskCatalog()
    placeholders = ",".join("?" for _ in exposed_catalog_ids)
    rows = conn.execute(
        "SELECT task.insession_task_id, task.session_id, task.root_title, "
        "task.root_objective, task.current_status, task.current_graph_revision, "
        "task.created_turn_id, task.creation_source_start, task.creation_source_end, "
        "task.creation_source_sha256, revision.source_turn_id, revision.source_anchors_json, "
        "revision.authorization_anchor_ids_json "
        "FROM insession_tasks AS task "
        "LEFT JOIN insession_task_graph_revisions AS revision "
        "ON revision.insession_task_id=task.insession_task_id "
        "AND revision.graph_revision=task.current_graph_revision "
        f"WHERE task.insession_task_id IN ({placeholders})",
        exposed_catalog_ids,
    ).fetchall()
    by_id = {str(row["insession_task_id"]): row for row in rows}
    if set(by_id) != set(exposed_catalog_ids) or any(
        str(row["session_id"]) != session_id for row in rows
    ):
        raise InSessionTaskPersistenceError(
            "exposed Task catalog contains an unknown or cross-Session id"
        )
    items: list[InSessionTaskCatalogItem] = []
    for task_id in exposed_catalog_ids:
        row = by_id[task_id]
        revision = row["current_graph_revision"]
        readable = (
            _has_readable_shell_provenance(conn, row)
            if revision is None
            else _has_readable_authorization_manifest(
                conn,
                row,
                row["source_anchors_json"],
                row["authorization_anchor_ids_json"],
                source_turn_id=str(row["source_turn_id"]),
            )
        )
        if not readable:
            raise InSessionTaskPersistenceError(
                "exposed Task catalog contains unverifiable source provenance"
            )
        items.append(
            InSessionTaskCatalogItem(
                insession_task_id=task_id,
                goal_summary=_goal_summary(row["root_title"], row["root_objective"]),
                status=InSessionTaskStatus(str(row["current_status"])),
                current_graph_revision=(int(revision) if revision is not None else None),
            )
        )
    return InSessionTaskCatalog(items=tuple(items))


def _validate_task_match_replay(
    row: sqlite3.Row,
    *,
    session_id: str,
    source_turn_id: str,
    proposal_hash: str,
) -> None:
    if (
        str(row["session_id"]) != session_id
        or str(row["source_turn_id"]) != source_turn_id
        or str(row["proposal_hash"]) != proposal_hash
    ):
        raise InSessionTaskApplyIdCollision("insession task-match apply id collision")


def _decode_string_mapping(value: object) -> dict[str, str]:
    decoded = _decode_json(value)
    if not isinstance(decoded, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in decoded.items()
    ):
        raise InSessionTaskPersistenceError("stored task-match apply receipt is corrupt")
    return decoded


def _load_task_match_lane_manifest(
    row: sqlite3.Row,
    *,
    expected: InSessionTaskExecutionLaneManifest | None = None,
) -> InSessionTaskExecutionLaneManifest:
    raw = row["execution_lane_manifest_json"]
    stored_hash = row["execution_lane_manifest_hash"]
    if raw is None or stored_hash is None:
        raise InSessionTaskPersistenceError(
            "stored task-match receipt predates durable lane authority"
        )
    try:
        manifest = InSessionTaskExecutionLaneManifest.model_validate_json(
            str(raw)
        )
    except ValueError as exc:
        raise InSessionTaskPersistenceError(
            "stored Task execution lane manifest is corrupt"
        ) from exc
    if manifest.manifest_sha256 != str(stored_hash):
        raise InSessionTaskPersistenceError(
            "stored Task execution lane manifest hash disagrees"
        )
    if expected is not None and manifest != expected:
        raise InSessionTaskApplyIdCollision(
            "task-match replay lane manifest disagrees with its exact proposal"
        )
    return manifest


def _allocate_branch_intent_id(conn: sqlite3.Connection, deps: StoreDeps) -> str:
    for _ in range(16):
        candidate = f"insession_task_branch_intent_{deps.new_id()}"
        if conn.execute(
            "SELECT 1 FROM insession_task_branch_intents WHERE branch_intent_id=?",
            (candidate,),
        ).fetchone() is None:
            return candidate
    raise InSessionTaskPersistenceError("could not allocate a unique branch intent id")


def _link_task_match_roots_in_transaction(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    insession_task_ids: tuple[str, ...],
    created_task_ids: set[str],
    expected_window_revision: int,
    now: str,
) -> tuple[int, int | None]:
    if not insession_task_ids:
        return 0, _window_state_version(conn, session_id, turn_id)
    _require_task_ownership(conn, session_id, insession_task_ids)
    for task_id in insession_task_ids:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, relation, created_at) "
            "VALUES (?, ?, ?, NULL, ?, ?)",
            (
                session_id,
                turn_id,
                task_id,
                "created" if task_id in created_task_ids else "referenced",
                now,
            ),
        )
    placeholders = ",".join("?" for _ in insession_task_ids)
    conn.execute(
        "UPDATE insession_tasks SET updated_at=? "
        f"WHERE session_id=? AND insession_task_id IN ({placeholders})",
        (now, session_id, *insession_task_ids),
    )
    link_revision = _current_turn_link_revision(conn, session_id, turn_id)
    window_state_version = _advance_window_task_link_pointer(
        conn,
        session_id=session_id,
        turn_id=turn_id,
        link_revision=link_revision,
        expected_window_revision=expected_window_revision,
        now=now,
    )
    return link_revision, window_state_version


def _allocate_graph_ids(
    conn: sqlite3.Connection,
    deps: StoreDeps,
    nodes: Iterable[object],
    root_key: str,
) -> tuple[str, dict[str, str]]:
    # 根节点标识有意采用根 Task 标识。本地提案键绝不会作为跨边界权威 ID 持久化。
    task_id = _allocate_id(conn, deps, "insession_task")
    node_ids = {root_key: task_id}
    for node in nodes:
        node_key = str(getattr(node, "node_key"))
        if node_key == root_key:
            continue
        node_ids[node_key] = _allocate_id(conn, deps, "insession_task_node")
    return task_id, node_ids


def _allocate_revision_one_node_ids(
    conn: sqlite3.Connection,
    deps: StoreDeps,
    *,
    target_insession_task_id: str,
    root: InSessionTaskRootGraphProposal,
) -> dict[str, str]:
    """将根节点绑定到其现有 Task，并且只分配子节点 ID。"""

    node_ids = {root.root_key: target_insession_task_id}
    for node in root.nodes:
        if node.node_key != root.root_key:
            node_ids[node.node_key] = _allocate_id(
                conn, deps, "insession_task_node"
            )
    return node_ids


def _allocate_id(conn: sqlite3.Connection, deps: StoreDeps, prefix: str) -> str:
    for _ in range(16):
        candidate = f"{prefix}_{deps.new_id()}"
        exists = conn.execute(
            "SELECT 1 FROM insession_tasks WHERE insession_task_id=? "
            "UNION ALL SELECT 1 FROM insession_task_graph_nodes "
            "WHERE insession_task_node_id=? LIMIT 1",
            (candidate, candidate),
        ).fetchone()
        if exists is None:
            return candidate
    raise InSessionTaskPersistenceError("could not allocate a unique insession task id")


def _insert_task_root(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    session_id: str,
    source_turn_id: str,
    title: str,
    objective: str,
    proposal_hash: str,
    source_manifest: _RootSourceManifest,
    now: str,
) -> None:
    conn.execute(
        "INSERT INTO insession_tasks "
        "(insession_task_id, session_id, current_graph_revision, current_status, state_version, "
        "root_title, root_objective, created_turn_id, created_at, updated_at) "
        "VALUES (?, ?, 1, ?, 1, ?, ?, ?, ?, ?)",
        (task_id, session_id, _ROOT_STATUS, title, objective, source_turn_id, now, now),
    )
    conn.execute(
        "INSERT INTO insession_task_graph_revisions "
        "(insession_task_id, graph_revision, source_turn_id, proposal_hash, "
        "source_anchors_json, authorization_anchor_ids_json, required_anchor_ids_json, created_at) "
        "VALUES (?, 1, ?, ?, ?, ?, ?, ?)",
        (
            task_id,
            source_turn_id,
            proposal_hash,
            _canonical_json(
                [item.model_dump(mode="json") for item in source_manifest.source_anchors]
            ),
            _canonical_json(source_manifest.authorization_anchor_ids),
            _canonical_json(source_manifest.required_anchor_ids),
            now,
        ),
    )


def _insert_graph_revision_one(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    source_turn_id: str,
    proposal_hash: str,
    source_manifest: _RootSourceManifest,
    now: str,
) -> None:
    conn.execute(
        "INSERT INTO insession_task_graph_revisions "
        "(insession_task_id, graph_revision, source_turn_id, proposal_hash, "
        "source_anchors_json, authorization_anchor_ids_json, required_anchor_ids_json, created_at) "
        "VALUES (?, 1, ?, ?, ?, ?, ?, ?)",
        (
            task_id,
            source_turn_id,
            proposal_hash,
            _canonical_json(
                [item.model_dump(mode="json") for item in source_manifest.source_anchors]
            ),
            _canonical_json(source_manifest.authorization_anchor_ids),
            _canonical_json(source_manifest.required_anchor_ids),
            now,
        ),
    )


def _insert_graph_nodes_and_edges(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    root_nodes: Iterable[object],
    local_node_ids: dict[str, str],
    now: str,
) -> None:
    nodes = tuple(root_nodes)
    for ordinal, node in enumerate(nodes):
        node_key = str(getattr(node, "node_key"))
        node_id = local_node_ids[node_key]
        conn.execute(
            "INSERT INTO insession_task_graph_nodes "
            "(insession_task_id, graph_revision, insession_task_node_id, node_revision, node_kind, ordinal, "
            "title, objective, source_anchor_ids_json, acceptance_criteria_json, "
            "constraints_json, created_at) "
            "VALUES (?, 1, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task_id,
                node_id,
                str(getattr(node, "node_kind")),
                ordinal,
                str(getattr(node, "title")),
                str(getattr(node, "objective")),
                _canonical_json(getattr(node, "source_anchor_ids")),
                _canonical_json(
                    [item.model_dump(mode="json") for item in getattr(node, "acceptance_criteria")]
                ),
                _canonical_json(getattr(node, "constraints")),
                now,
            ),
        )
        conn.execute(
            "INSERT INTO insession_task_node_states "
            "(insession_task_id, insession_task_node_id, node_revision, status, state_version, updated_at) "
            "VALUES (?, ?, 1, ?, 1, ?)",
            (task_id, node_id, _ROOT_STATUS, now),
        )

    for ordinal, node in enumerate(nodes):
        parent_key = getattr(node, "parent_node_key")
        if parent_key is None:
            continue
        conn.execute(
            "INSERT INTO insession_task_graph_edges "
            "(insession_task_id, graph_revision, child_insession_task_node_id, "
            "parent_insession_task_node_id, ordinal) VALUES (?, 1, ?, ?, ?)",
            (task_id, local_node_ids[str(getattr(node, "node_key"))], local_node_ids[str(parent_key)], ordinal),
        )


def _link_turn_to_tasks_in_transaction(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    insession_task_ids: tuple[str, ...],
    relation: Literal["created", "referenced", "revised"],
    expected_window_revision: int,
    now: str,
) -> tuple[int, int | None]:
    _require_task_ownership(conn, session_id, insession_task_ids)
    existing_rows = conn.execute(
        "SELECT insession_task_id FROM insession_task_turn_links "
        f"WHERE session_id=? AND turn_id=? AND insession_task_id IN ({','.join('?' for _ in insession_task_ids)})",
        (session_id, turn_id, *insession_task_ids),
    ).fetchall()
    existing_task_ids = {str(row["insession_task_id"]) for row in existing_rows}
    new_task_ids = tuple(task_id for task_id in insession_task_ids if task_id not in existing_task_ids)
    if not new_task_ids:
        return _current_turn_link_revision(conn, session_id, turn_id), _window_state_version(
            conn, session_id, turn_id
        )

    _require_active_window(
        conn,
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=expected_window_revision,
    )
    for task_id in new_task_ids:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, relation, created_at) "
            "VALUES (?, ?, ?, NULL, ?, ?) "
            "ON CONFLICT DO NOTHING",
            (session_id, turn_id, task_id, relation, now),
        )
    # 紧凑 catalog 有意优先展示近期参与的非终态任务。相关性链接代表活动，但不代表
    # 语义 graph revision 或节点状态转换，因此这里只改变这一排序时间戳。
    placeholders = ",".join("?" for _ in new_task_ids)
    conn.execute(
        "UPDATE insession_tasks SET updated_at=? "
        f"WHERE session_id=? AND insession_task_id IN ({placeholders})",
        (now, session_id, *new_task_ids),
    )
    link_revision = _current_turn_link_revision(conn, session_id, turn_id)
    window_state_version = _advance_window_task_link_pointer(
        conn,
        session_id=session_id,
        turn_id=turn_id,
        link_revision=link_revision,
        expected_window_revision=expected_window_revision,
        now=now,
    )
    return link_revision, window_state_version


def _require_task_ownership(
    conn: sqlite3.Connection,
    session_id: str,
    task_ids: tuple[str, ...],
) -> None:
    if not task_ids:
        return
    placeholders = ",".join("?" for _ in task_ids)
    rows = conn.execute(
        "SELECT insession_task_id, session_id FROM insession_tasks "
        f"WHERE insession_task_id IN ({placeholders})",
        task_ids,
    ).fetchall()
    found = {str(row["insession_task_id"]): str(row["session_id"]) for row in rows}
    if set(task_ids) != set(found) or any(owner != session_id for owner in found.values()):
        raise InSessionTaskPersistenceError("insession task is unknown or outside this Session")


def _advance_window_task_link_pointer(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    link_revision: int,
    expected_window_revision: int,
    now: str,
) -> int | None:
    # Task 变更只能发生在当前活动 Turn 下。Window CAS 与链接写入处于同一事务，
    # 因而过期模型分支无法静默覆盖更新的执行阶段。
    cursor = conn.execute(
        "UPDATE turn_execution_windows SET turn_task_link_revision=?, state_version=state_version+1, "
        "updated_at=? WHERE session_id=? AND turn_id=? AND window_state='active' AND state_version=?",
        (link_revision, now, session_id, turn_id, expected_window_revision),
    )
    if cursor.rowcount == 0:
        window = conn.execute(
            "SELECT state_version FROM turn_execution_windows WHERE session_id=? AND turn_id=?",
            (session_id, turn_id),
        ).fetchone()
        if window is not None:
            raise TurnExecutionWindowRevisionConflict(
                expected=expected_window_revision,
                actual=int(window["state_version"]),
            )
        raise InSessionTaskPersistenceError("active Turn window was lost during Task link")
    return _window_state_version(conn, session_id, turn_id)


def _require_active_window(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    expected_window_revision: int,
) -> None:
    window = conn.execute(
        "SELECT turn_id, window_state, state_version FROM turn_execution_windows WHERE session_id=?",
        (session_id,),
    ).fetchone()
    if window is None or str(window["turn_id"] or "") != turn_id:
        raise InSessionTaskPersistenceError("Task mutation requires the current Turn execution window")
    if str(window["window_state"]) != "active":
        raise InSessionTaskPersistenceError("Task mutation requires an active Turn execution window")
    actual = int(window["state_version"])
    if actual != expected_window_revision:
        raise TurnExecutionWindowRevisionConflict(
            expected=expected_window_revision,
            actual=actual,
        )


def _current_turn_link_revision(
    conn: sqlite3.Connection,
    session_id: str,
    turn_id: str,
) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(link_id), 0) AS link_revision "
        "FROM insession_task_turn_links WHERE session_id=? AND turn_id=?",
        (session_id, turn_id),
    ).fetchone()
    return int(row["link_revision"])


def _window_state_version(
    conn: sqlite3.Connection,
    session_id: str,
    turn_id: str,
) -> int | None:
    row = conn.execute(
        "SELECT state_version FROM turn_execution_windows WHERE session_id=? AND turn_id=?",
        (session_id, turn_id),
    ).fetchone()
    return int(row["state_version"]) if row is not None else None


def _require_window_revision(expected_window_revision: int) -> None:
    if (
        not isinstance(expected_window_revision, int)
        or isinstance(expected_window_revision, bool)
        or expected_window_revision < 1
    ):
        raise InSessionTaskPersistenceError("expected_window_revision must be positive")


def _require_positive_revision(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise InSessionTaskPersistenceError(f"{name} must be positive")


def _require_session_turn(conn: sqlite3.Connection, *, session_id: str, turn_id: str) -> None:
    row = conn.execute(
        "SELECT session_id FROM runtime_turns WHERE turn_id=?", (turn_id,)
    ).fetchone()
    if row is None or str(row["session_id"]) != session_id:
        raise InSessionTaskPersistenceError("source Turn is unknown or outside this Session")


def _require_session(conn: sqlite3.Connection, session_id: str) -> None:
    if conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone() is None:
        raise InSessionTaskPersistenceError("unknown Session")


def _validate_create_replay(
    row: sqlite3.Row,
    *,
    session_id: str,
    source_turn_id: str,
    proposal_hash: str,
) -> None:
    if (
        str(row["session_id"]) != session_id
        or str(row["source_turn_id"]) != source_turn_id
        or str(row["operation"]) != "create"
        or str(row["proposal_hash"]) != proposal_hash
    ):
        raise InSessionTaskApplyIdCollision("insession task apply id collision")


def _validate_revision_replay(
    row: sqlite3.Row,
    *,
    session_id: str,
    source_turn_id: str,
    target_insession_task_id: str,
    proposal_hash: str,
) -> None:
    if (
        str(row["session_id"]) != session_id
        or str(row["insession_task_id"]) != target_insession_task_id
        or str(row["proposal_hash"]) != proposal_hash
    ):
        raise InSessionTaskApplyIdCollision(
            "insession task graph revision apply id collision"
        )


def _revision_commit_result(
    row: sqlite3.Row,
    *,
    status: Literal["applied", "replayed"],
) -> InSessionTaskGraphRevisionCommitResult:
    previous = row["expected_graph_revision"]
    return InSessionTaskGraphRevisionCommitResult(
        status=status,
        insession_task_id=str(row["insession_task_id"]),
        previous_graph_revision=(int(previous) if previous is not None else None),
        committed_graph_revision=int(row["committed_graph_revision"]),
        task_state_version=int(row["committed_task_state_version"]),
        turn_task_link_revision=int(row["turn_task_link_revision"]),
        window_state_version=int(row["committed_window_state_version"]),
    )


def _node_projection(row: sqlite3.Row) -> dict[str, object]:
    return {
        "insession_task_node_id": str(row["insession_task_node_id"]),
        "node_revision": int(row["node_revision"]),
        "node_kind": str(row["node_kind"]),
        "ordinal": int(row["ordinal"]),
        "parent_insession_task_node_id": (
            str(row["parent_insession_task_node_id"])
            if row["parent_insession_task_node_id"] is not None
            else None
        ),
        "title": str(row["title"]),
        "objective": str(row["objective"]),
        "status": str(row["status"]),
        "state_version": int(row["state_version"]),
        "source_anchor_ids": _decode_json(row["source_anchor_ids_json"]),
        "acceptance_criteria": _decode_json(row["acceptance_criteria_json"]),
        "constraints": _decode_json(row["constraints_json"]),
    }


def _goal_summary(title: object, objective: object) -> str:
    return f"{str(title).strip()}：{str(objective).strip()}"


def _decode_ids(value: object) -> tuple[str, ...]:
    decoded = _decode_json(value)
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        raise InSessionTaskPersistenceError("stored task apply receipt is corrupt")
    return tuple(decoded)


def _source_anchors_from_json(value: object) -> tuple[InSessionTaskSourceAnchor, ...]:
    decoded = _decode_json(value)
    if not isinstance(decoded, list):
        raise InSessionTaskPersistenceError("stored TaskGraph source manifest is corrupt")
    try:
        return tuple(InSessionTaskSourceAnchor.model_validate(item) for item in decoded)
    except (TypeError, ValueError) as exc:
        raise InSessionTaskPersistenceError("stored TaskGraph source manifest is corrupt") from exc


def _has_readable_authorization_manifest(
    conn: sqlite3.Connection,
    task: sqlite3.Row,
    source_anchors_json: object,
    authorization_anchor_ids_json: object,
    *,
    source_turn_id: str,
) -> bool:
    """向模型暴露 Task 前，要求其来源权威状态可读取。"""

    try:
        source_anchors = _source_anchors_from_json(source_anchors_json)
        authorization_anchor_ids = _decode_ids(authorization_anchor_ids_json)
    except InSessionTaskPersistenceError:
        return False
    anchors_by_id = {anchor.anchor_id: anchor for anchor in source_anchors}
    if not source_anchors or not authorization_anchor_ids:
        return False
    for anchor_id in authorization_anchor_ids:
        anchor = anchors_by_id.get(anchor_id)
        if anchor is None:
            return False
        if anchor.source_kind in _A4A_ALLOWED_SOURCE_KINDS:
            if anchor.source_turn_id != source_turn_id:
                return False
            try:
                user_text = _load_authoritative_user_input(
                    conn,
                    session_id=str(task["session_id"]),
                    turn_id=anchor.source_turn_id,
                )
            except InSessionTaskPersistenceError:
                return False
            if (
                anchor.end > len(user_text)
                or user_text[anchor.start : anchor.end] != anchor.excerpt
            ):
                return False
            continue
        if anchor.source_kind == "previously_authorized_task_state":
            if not _matches_shell_creation_anchor(conn, task, anchor):
                return False
            continue
        return False
    return True


def _has_readable_shell_provenance(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> bool:
    """根据不可变的已接受 Turn 文本验证一个 shell 的指针/哈希。"""

    try:
        turn_id = str(row["created_turn_id"])
        start = int(row["creation_source_start"])
        end = int(row["creation_source_end"])
        expected_hash = str(row["creation_source_sha256"])
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    if start < 0 or end <= start or len(expected_hash) != 64:
        return False
    source = conn.execute(
        "SELECT turn_input.content FROM runtime_turn_inputs AS input "
        "JOIN session_turns AS turn_input "
        "ON turn_input.session_id=input.session_id AND turn_input.turn_idx=input.turn_idx "
        "WHERE input.session_id=? AND input.turn_id=? AND turn_input.role='user'",
        (str(row["session_id"]), turn_id),
    ).fetchone()
    if source is None:
        return False
    user_text = str(source["content"])
    if end > len(user_text):
        return False
    actual_hash = hashlib.sha256(user_text[start:end].encode("utf-8")).hexdigest()
    return actual_hash == expected_hash


def _decode_json(value: object) -> object:
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise InSessionTaskPersistenceError("stored insession task JSON is corrupt") from exc


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _require_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 200:
        raise InSessionTaskPersistenceError(f"invalid {name}")
