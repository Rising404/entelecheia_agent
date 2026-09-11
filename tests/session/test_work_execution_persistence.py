from __future__ import annotations

import json
import hashlib
import sqlite3
import threading

import pytest

from personagraph.session import store
from personagraph.session.persistence.turns.entry_tasks import list_entry_pending_task_questions
from personagraph.session.l2_store import continuation as continuation_store
from personagraph.session.l2_store import verification as verification_store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.session.persistence.turns import turn_execution as turn_execution_records
from personagraph.session.persistence.l2.work_run import work_execution as work_execution_records
from personagraph.session.persistence.l2.work_run import (
    work_verification as work_verification_records,
)
from personagraph.l2.work_run import (
    AcceptanceVerificationFeedback,
    AttemptDecision,
    AcceptanceUpdate,
    AuxiliaryNodeSubject,
    HostAcceptedAttemptDecision,
    HostMaterializedCallToolsAction,
    HostMaterializedToolCall,
    NodeVerificationResult,
    OutputWindowFormat,
    RequestUserInputAction,
    SubmitOutputWindowAction,
    TaskNodeSubject,
    ToolResultStatus,
    ToolResult,
    VerificationVerdict,
    WriteOutputWindowAction,
)
from personagraph.l2.task_graph.lane_manifest import (
    InSessionTaskExecutionLaneManifest,
    InSessionTaskExecutionLaneMatch,
    InSessionTaskExecutionLane,
    canonical_insession_task_lane_manifest_sha256,
)
from personagraph.l2.task_graph.task_matching import (
    InSessionTaskMatchedSourceSpan,
)
from tests.helpers.current_auxiliary_delivery import (
    settle_current_auxiliary_candidate,
)


def _window_revision(session_id: str) -> int:
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    return int(window["state_version"])


def _seed_task_node(
    *,
    request_id: str = "work-execution-turn",
    task_id: str = "task-one",
    node_id: str = "node-one",
    acceptance_ids: tuple[str, ...] = ("deliverable", "quality"),
) -> tuple[str, str, TaskNodeSubject]:
    session_id = store.create_session("Entelecheia")
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=request_id,
        source="runtime_test",
        user_text="执行当前任务节点",
        lease_owner="work-execution-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    now = "2026-08-14T00:00:00+00:00"
    acceptances = [
        {
            "acceptance_id": acceptance_id,
            "criterion": f"满足 {acceptance_id}",
            "source_anchor_ids": ["request"],
        }
        for acceptance_id in acceptance_ids
    ]
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO insession_tasks "
            "(insession_task_id, session_id, current_graph_revision, current_status, "
            "state_version, root_title, root_objective, created_turn_id, created_at, updated_at) "
            "VALUES (?, ?, 1, 'proposed', 1, '测试任务', '完成测试任务', ?, ?, ?)",
            (task_id, session_id, turn_id, now, now),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_revisions "
            "(insession_task_id, graph_revision, source_turn_id, proposal_hash, "
            "source_anchors_json, authorization_anchor_ids_json, required_anchor_ids_json, created_at) "
            "VALUES (?, 1, ?, ?, '[]', '[]', '[]', ?)",
            (task_id, turn_id, "proposal-hash", now),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_nodes "
            "(insession_task_id, graph_revision, insession_task_node_id, node_revision, "
            "node_kind, ordinal, title, objective, source_anchor_ids_json, "
            "acceptance_criteria_json, constraints_json, created_at) "
            "VALUES (?, 1, ?, 1, 'root', 0, '测试节点', '完成节点', '[\"request\"]', ?, '[]', ?)",
            (task_id, node_id, json.dumps(acceptances, ensure_ascii=False), now),
        )
        conn.execute(
            "INSERT INTO insession_task_node_states "
            "(insession_task_id, insession_task_node_id, node_revision, status, state_version, updated_at) "
            "VALUES (?, ?, 1, 'proposed', 1, ?)",
            (task_id, node_id, now),
        )
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, relation, created_at) "
            "VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (session_id, turn_id, task_id, now),
        )
        user_text = "执行当前任务节点"
        lane = InSessionTaskExecutionLane(
            ordinal=0,
            insession_task_id=task_id,
            matches=(
                InSessionTaskExecutionLaneMatch(
                    match_type="existing_root",
                    source_span=InSessionTaskMatchedSourceSpan(
                        start=0,
                        end=len(user_text),
                        text_sha256=hashlib.sha256(
                            user_text.encode("utf-8")
                        ).hexdigest(),
                    ),
                    execution_requested=True,
                ),
            ),
            execution_requested=True,
        )
        manifest = InSessionTaskExecutionLaneManifest(
            lanes=(lane,),
            manifest_sha256=canonical_insession_task_lane_manifest_sha256(
                (lane,)
            ),
        )
        conn.execute(
            "INSERT INTO insession_task_match_apply_receipts "
            "(apply_id, session_id, source_turn_id, proposal_hash, "
            "created_task_mapping_json, related_insession_task_ids_json, "
            "branch_intent_ids_json, turn_task_link_revision, "
            "window_state_version, execution_lane_manifest_json, "
            "execution_lane_manifest_hash, created_at) "
            "VALUES (?, ?, ?, 'test-lane-manifest', '{}', ?, '[]', 0, ?, ?, ?, ?)",
            (
                f"test-lane-{turn_id}",
                session_id,
                turn_id,
                json.dumps([task_id]),
                int(
                    conn.execute(
                        "SELECT state_version FROM turn_execution_windows "
                        "WHERE session_id=?",
                        (session_id,),
                    ).fetchone()[0]
                ),
                manifest.model_dump_json(),
                manifest.manifest_sha256,
                now,
            ),
        )
    return session_id, turn_id, TaskNodeSubject(
        task_id=task_id,
        graph_revision=1,
        node_id=node_id,
        node_revision=1,
    )


def _add_test_root_task_lane(
    session_id: str,
    turn_id: str,
    *,
    task_id: str,
) -> TaskNodeSubject:
    """添加第二个规范根节点，并重写夹具的来源通道。"""

    now = "2026-08-14T00:00:01+00:00"
    acceptances = [
        {
            "acceptance_id": acceptance_id,
            "criterion": f"满足 {acceptance_id}",
            "source_anchor_ids": ["request"],
        }
        for acceptance_id in ("deliverable", "quality")
    ]
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO insession_tasks "
            "(insession_task_id, session_id, current_graph_revision, current_status, "
            "state_version, root_title, root_objective, created_turn_id, created_at, updated_at) "
            "VALUES (?, ?, 1, 'proposed', 1, ?, ?, ?, ?, ?)",
            (task_id, session_id, task_id, task_id, turn_id, now, now),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_revisions "
            "(insession_task_id, graph_revision, source_turn_id, proposal_hash, "
            "source_anchors_json, authorization_anchor_ids_json, "
            "required_anchor_ids_json, created_at) "
            "VALUES (?, 1, ?, ?, '[]', '[]', '[]', ?)",
            (task_id, turn_id, f"proposal-{task_id}", now),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_nodes "
            "(insession_task_id, graph_revision, insession_task_node_id, "
            "node_revision, node_kind, ordinal, title, objective, "
            "source_anchor_ids_json, acceptance_criteria_json, constraints_json, "
            "created_at) VALUES (?, 1, ?, 1, 'root', 0, ?, ?, '[\"request\"]', ?, '[]', ?)",
            (
                task_id,
                task_id,
                task_id,
                task_id,
                json.dumps(acceptances, ensure_ascii=False),
                now,
            ),
        )
        conn.execute(
            "INSERT INTO insession_task_node_states "
            "(insession_task_id, insession_task_node_id, node_revision, status, "
            "state_version, updated_at) VALUES (?, ?, 1, 'proposed', 1, ?)",
            (task_id, task_id, now),
        )
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (session_id, turn_id, task_id, now),
        )
        first_task_id = str(
            conn.execute(
                "SELECT json_each.value FROM insession_task_match_apply_receipts, "
                "json_each(related_insession_task_ids_json) "
                "WHERE session_id=? AND source_turn_id=? ORDER BY json_each.key LIMIT 1",
                (session_id, turn_id),
            ).fetchone()[0]
        )
        user_text = "执行当前任务节点"
        spans = ((0, 4), (4, len(user_text)))
        lanes = tuple(
            InSessionTaskExecutionLane(
                ordinal=ordinal,
                insession_task_id=lane_task_id,
                matches=(
                    InSessionTaskExecutionLaneMatch(
                        match_type="existing_root",
                        source_span=InSessionTaskMatchedSourceSpan(
                            start=span[0],
                            end=span[1],
                            text_sha256=hashlib.sha256(
                                user_text[span[0] : span[1]].encode("utf-8")
                            ).hexdigest(),
                        ),
                        execution_requested=True,
                    ),
                ),
                execution_requested=True,
            )
            for ordinal, (lane_task_id, span) in enumerate(
                zip((first_task_id, task_id), spans, strict=True)
            )
        )
        manifest = InSessionTaskExecutionLaneManifest(
            lanes=lanes,
            manifest_sha256=canonical_insession_task_lane_manifest_sha256(lanes),
        )
        conn.execute(
            "UPDATE insession_task_match_apply_receipts "
            "SET related_insession_task_ids_json=?, execution_lane_manifest_json=?, "
            "execution_lane_manifest_hash=? WHERE session_id=? AND source_turn_id=?",
            (
                json.dumps([first_task_id, task_id]),
                manifest.model_dump_json(),
                manifest.manifest_sha256,
                session_id,
                turn_id,
            ),
        )
    return TaskNodeSubject(
        task_id=task_id,
        graph_revision=1,
        node_id=task_id,
        node_revision=1,
    )


def _create(
    session_id: str,
    turn_id: str,
    subject: TaskNodeSubject,
    *,
    apply_id: str = "create-run",
    work_run_id: str = "workrun-one",
    expected_window_revision: int | None = None,
    expected_task_state_version: int = 1,
    expected_node_state_version: int = 1,
):
    return work_run_store.create_task_node_work_run(
        session_id=session_id,
        turn_id=turn_id,
        subject=subject,
        expected_task_state_version=expected_task_state_version,
        expected_node_state_version=expected_node_state_version,
        expected_window_revision=(
            expected_window_revision
            if expected_window_revision is not None
            else _window_revision(session_id)
        ),
        apply_id=apply_id,
        work_run_id=work_run_id,
    )


def _start(
    session_id: str,
    turn_id: str,
    work_run_id: str,
    *,
    expected_run_revision: int,
    expected_progress_revision: int = 1,
    apply_id: str,
    attempt_id: str,
    expected_window_revision: int | None = None,
    input_checkpoint_id: str | None = None,
    catalog_snapshot: dict[str, object] | None = None,
):
    return work_run_store.start_work_run_attempt(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        expected_work_run_revision=expected_run_revision,
        expected_progress_revision=expected_progress_revision,
        expected_window_revision=(
            expected_window_revision
            if expected_window_revision is not None
            else _window_revision(session_id)
        ),
        apply_id=apply_id,
        input_checkpoint_id=input_checkpoint_id,
        catalog_snapshot=(
            catalog_snapshot
            if catalog_snapshot is not None
            else {"revision": 1, "tools": ["read_test"]}
        ),
        attempt_id=attempt_id,
    )


def _get(session_id: str, work_run_id: str = "workrun-one"):
    return work_run_store.get_work_run(session_id=session_id, work_run_id=work_run_id)


def _seed_waiting_user_question():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    started = _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=1,
        apply_id="start-question",
        attempt_id="question-attempt",
    )
    waiting = work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="question-attempt",
        decision=HostAcceptedAttemptDecision(
            acceptance_updates=(),
            action=RequestUserInputAction(question="请提供旅行日期。"),
        ),
        expected_work_run_revision=started.work_run_revision,
        expected_progress_revision=started.acceptance_progress_revision,
        expected_window_revision=started.window_state_version,
        apply_id="commit-question",
        active_seconds_delta=1,
    )
    return session_id, turn_id, subject, waiting


def _accept_answer_turn(
    session_id: str,
    waiting_turn_id: str,
    subject: TaskNodeSubject,
    *,
    client_request_id: str = "answer-question",
):
    marked = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=waiting_turn_id,
        expected_window_revision=_window_revision(session_id),
        stage="WORKRUN",
        interruption_reason="waiting_user",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=waiting_turn_id,
        expected_window_revision=int(marked["state_version"]),
        end_reason="waiting_user",
        error_code=None,
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=client_request_id,
        source="runtime_test",
        user_text="旅行日期是九月十日至十二日。",
        lease_owner="work-execution-test",
    )
    answer_turn_id = str(accepted["turn"]["turn_id"])
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (
                session_id,
                answer_turn_id,
                subject.task_id,
                "2026-08-14T00:40:00+00:00",
            ),
        )
    return answer_turn_id


def test_waiting_user_lane_detach_preserves_question_and_replays():
    session_id, turn_id, _subject, waiting = _seed_waiting_user_question()
    detached = work_run_store.detach_safe_work_run_lane(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_window_revision=waiting.window_state_version,
        apply_id="detach-waiting-lane",
    )
    assert detached.status == "applied"
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert window["current_work_run_id"] is None
    pending = continuation_store.list_pending_user_questions(session_id=session_id)
    assert len(pending) == 1
    assert pending[0].work_run_id == "workrun-one"
    assert list_entry_pending_task_questions(
        store._deps(),
        session_id=session_id,
    )[0].question == "请提供旅行日期。"

    replayed = work_run_store.detach_safe_work_run_lane(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_window_revision=waiting.window_state_version,
        apply_id="detach-waiting-lane",
    )
    assert replayed.status == "replayed"
    assert replayed.window_state_version == detached.window_state_version
    assert len(continuation_store.list_pending_user_questions(session_id=session_id)) == 1


def test_waiting_branch_keeps_task_active_and_does_not_hide_ready_sibling():
    session_id, turn_id, root = _seed_task_node(
        request_id="waiting-sibling-turn",
        task_id="waiting-sibling-task",
        node_id="waiting-sibling-task",
        acceptance_ids=("root",),
    )
    now = "2026-08-14T00:41:00+00:00"
    child_acceptance = json.dumps(
        [
            {
                "acceptance_id": "child",
                "criterion": "完成子节点",
                "source_anchor_ids": ["request"],
            }
        ],
        ensure_ascii=False,
    )
    with store._connect() as conn:
        for ordinal, child_id in enumerate(("waiting-child", "ready-child"), start=1):
            conn.execute(
                "INSERT INTO insession_task_graph_nodes "
                "(insession_task_id, graph_revision, insession_task_node_id, "
                "node_revision, node_kind, ordinal, title, objective, "
                "source_anchor_ids_json, acceptance_criteria_json, "
                "constraints_json, created_at) VALUES (?, 1, ?, 1, 'subtask', "
                "?, ?, ?, '[\"request\"]', ?, '[]', ?)",
                (
                    root.task_id,
                    child_id,
                    ordinal,
                    child_id,
                    f"完成 {child_id}",
                    child_acceptance,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO insession_task_node_states "
                "(insession_task_id, insession_task_node_id, node_revision, "
                "status, state_version, updated_at) "
                "VALUES (?, ?, 1, 'proposed', 1, ?)",
                (root.task_id, child_id, now),
            )
            conn.execute(
                "INSERT INTO insession_task_graph_edges "
                "(insession_task_id, graph_revision, child_insession_task_node_id, "
                "parent_insession_task_node_id, ordinal) VALUES (?, 1, ?, ?, ?)",
                (root.task_id, child_id, root.node_id, ordinal),
            )

    waiting_subject = TaskNodeSubject(
        task_id=root.task_id,
        graph_revision=1,
        node_id="waiting-child",
        node_revision=1,
    )
    created = work_run_store.create_task_node_work_run(
        session_id=session_id,
        turn_id=turn_id,
        subject=waiting_subject,
        expected_task_state_version=1,
        expected_node_state_version=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="create-waiting-child",
        work_run_id="waiting-child-run",
    )
    started = _start(
        session_id,
        turn_id,
        "waiting-child-run",
        expected_run_revision=created.work_run_revision,
        apply_id="start-waiting-child",
        attempt_id="waiting-child-attempt",
    )
    waiting = work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="waiting-child-run",
        attempt_id="waiting-child-attempt",
        decision=HostAcceptedAttemptDecision(
            action=RequestUserInputAction(question="请补充等待分支输入。")
        ),
        expected_work_run_revision=started.work_run_revision,
        expected_progress_revision=started.acceptance_progress_revision,
        expected_window_revision=started.window_state_version,
        apply_id="wait-waiting-child",
        active_seconds_delta=1,
    )
    with store._connect() as conn:
        task = conn.execute(
            "SELECT current_status, state_version FROM insession_tasks "
            "WHERE insession_task_id=?",
            (root.task_id,),
        ).fetchone()
        assert tuple(task) == ("active", 3)
    pending = continuation_store.list_pending_user_questions(session_id=session_id)
    assert [item.work_run_id for item in pending] == ["waiting-child-run"]
    frontier = work_run_store.project_task_node_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        task_id=root.task_id,
    )
    assert [item.subject.node_id for item in frontier.ready_fresh] == [
        "ready-child"
    ]
    assert [item.work_run_id for item in frontier.recoverable] == [
        "waiting-child-run"
    ]

    detached = work_run_store.detach_safe_work_run_lane(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="waiting-child-run",
        expected_window_revision=waiting.window_state_version,
        apply_id="detach-waiting-child",
    )
    ready_subject = TaskNodeSubject(
        task_id=root.task_id,
        graph_revision=1,
        node_id="ready-child",
        node_revision=1,
    )
    sibling = work_run_store.create_task_node_work_run(
        session_id=session_id,
        turn_id=turn_id,
        subject=ready_subject,
        expected_task_state_version=3,
        expected_node_state_version=1,
        expected_window_revision=detached.window_state_version,
        apply_id="create-ready-sibling",
        work_run_id="ready-sibling-run",
    )
    assert sibling.work_run_status.value == "active"
    assert len(continuation_store.list_pending_user_questions(session_id=session_id)) == 1


def _charge_active_time(
    session_id: str,
    turn_id: str,
    *,
    expected_run_revision: int,
    checkpoint_id: str,
    active_seconds_delta: float,
    apply_id: str,
    expected_window_revision: int | None = None,
):
    return work_execution_records.charge_work_run_active_time(
        store._deps(),
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        checkpoint_id=checkpoint_id,
        active_seconds_delta=active_seconds_delta,
        expected_work_run_revision=expected_run_revision,
        expected_window_revision=(
            expected_window_revision
            if expected_window_revision is not None
            else _window_revision(session_id)
        ),
        apply_id=apply_id,
    )


def _call_decision(
    *,
    call_id: str = "call-one",
    updates: tuple[AcceptanceUpdate, ...] = (),
    modifies_environment: bool = False,
) -> HostAcceptedAttemptDecision:
    return HostAcceptedAttemptDecision(
        acceptance_updates=updates,
        action=HostMaterializedCallToolsAction(
            calls=(
                HostMaterializedToolCall(
                    tool_call_id=call_id,
                    tool_id="read_test",
                    tool_version="1.0.0",
                    arguments={"value": "x"},
                    modifies_environment=modifies_environment,
                ),
            )
        ),
    )


def _decided_tool_recovery_catalog(
    *,
    action: str = "read",
) -> dict[str, object]:
    return {
        "revision": 1,
        "entries": [
            {
                "tool_id": "read_test",
                "contract_version": "contract-1",
                "status": "active",
                "created_revision": 1,
                "updated_revision": 1,
                "registration": {
                    "spec": {
                        "tool_id": "read_test",
                        "contract_version": "contract-1",
                        "name": "Read test",
                        "description": "Controlled persistence test tool.",
                        "input_schema": {"type": "object"},
                        "output_schema": {"type": "object"},
                        "catalog_tags": ["read"],
                    },
                    "implementation_version": "1.0.0",
                    "source": {
                        "kind": "local",
                        "source_id": "work-execution-test",
                        "fingerprint": None,
                        "display_name": None,
                    },
                    "effect_count": 1,
                    "effects": [
                        {
                            "resource": "memory",
                            "action": action,
                            "scope_kind": "local",
                            "default_scope": "*",
                            "data_egress": "none",
                            "idempotency": "unknown",
                            "reversibility": "unknown",
                            "resource_argument": None,
                            "scope_argument": None,
                            "egress_arguments": [],
                        }
                    ],
                    "execution": {
                        "default_timeout_s": None,
                        "hard_timeout_s": None,
                        "max_output_bytes": 1_000_000,
                        "max_transparent_retries": 3,
                        "execution_mode": "sync",
                        "cancellation_mode": "none",
                        "isolation_requirement": "in_process",
                        "concurrency_class": "default",
                    },
                },
            }
        ],
    }


def _seed_decided_tool_recovery(
    *,
    catalog_action: str = "read",
    modifies_environment: bool = False,
    result_count: int = 1,
    result_status: ToolResultStatus = ToolResultStatus.SUCCEEDED,
    cross_task_link: bool = False,
    close_batch: bool = False,
):
    session_id, first_turn_id, subject = _seed_task_node(
        request_id=f"decided-recovery-{catalog_action}-{result_count}-{cross_task_link}"
    )
    other_subject = (
        _add_test_root_task_lane(
            session_id,
            first_turn_id,
            task_id="task-decided-recovery-other",
        )
        if cross_task_link
        else None
    )
    catalog = _decided_tool_recovery_catalog(action=catalog_action)
    _create(session_id, first_turn_id, subject)
    started = _start(
        session_id,
        first_turn_id,
        "workrun-one",
        expected_run_revision=1,
        apply_id="start-decided-tool-recovery",
        attempt_id="attempt-decided-tool-recovery",
        catalog_snapshot=catalog,
    )
    call_count = 1 if modifies_environment else 2
    decision = HostAcceptedAttemptDecision(
        action=HostMaterializedCallToolsAction(
            calls=tuple(
                HostMaterializedToolCall(
                    tool_call_id=f"recovery-call-{ordinal}",
                    tool_id="read_test",
                    tool_version="1.0.0",
                    arguments={"value": [ordinal, {"nested": ["x", "y"]}]},
                    modifies_environment=modifies_environment,
                )
                for ordinal in range(1, call_count + 1)
            )
        )
    )
    cursor = work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=first_turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-decided-tool-recovery",
        decision=decision,
        expected_work_run_revision=started.work_run_revision,
        expected_progress_revision=started.acceptance_progress_revision,
        expected_window_revision=started.window_state_version,
        apply_id="commit-decided-tool-recovery",
    )
    for ordinal in range(1, result_count + 1):
        result = (
            ToolResult(
                status=ToolResultStatus.SUCCEEDED,
                tool_result_id=f"recovery-result-{ordinal}",
                tool_call_id=f"recovery-call-{ordinal}",
                attempt_id="attempt-decided-tool-recovery",
                ordinal=ordinal,
                output={"value": ordinal},
            )
            if result_status is ToolResultStatus.SUCCEEDED
            else ToolResult(
                status=result_status,
                tool_result_id=f"recovery-result-{ordinal}",
                tool_call_id=f"recovery-call-{ordinal}",
                attempt_id="attempt-decided-tool-recovery",
                ordinal=ordinal,
                output=None,
                error_code="controlled_recovery_status",
                error_message="Controlled recovery test result.",
            )
        )
        cursor = work_run_store.append_work_run_tool_result(
            session_id=session_id,
            turn_id=first_turn_id,
            work_run_id="workrun-one",
            result=result,
            expected_work_run_revision=cursor.work_run_revision,
            expected_progress_revision=cursor.acceptance_progress_revision,
            expected_window_revision=cursor.window_state_version,
            apply_id=f"append-decided-tool-recovery-{ordinal}",
        )
    if close_batch:
        cursor = work_run_store.close_work_run_attempt(
            session_id=session_id,
            turn_id=first_turn_id,
            work_run_id="workrun-one",
            attempt_id="attempt-decided-tool-recovery",
            expected_work_run_revision=cursor.work_run_revision,
            expected_progress_revision=cursor.acceptance_progress_revision,
            expected_window_revision=cursor.window_state_version,
            apply_id="close-decided-tool-recovery",
            active_seconds_delta=1,
        )
    marked = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=cursor.window_state_version,
        stage="TOOL",
        interruption_reason="process_lost_mid_tool_batch",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=int(marked["state_version"]),
        end_reason="process_lost",
        error_code="PROCESS_LOST",
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=f"resume-decided-tool-{catalog_action}-{result_count}",
        source="runtime_test",
        user_text="继续",
        lease_owner="work-execution-test",
    )
    second_turn_id = str(accepted["turn"]["turn_id"])
    linked_task_id = (
        other_subject.task_id if other_subject is not None else subject.task_id
    )
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (
                session_id,
                second_turn_id,
                linked_task_id,
                "2026-08-14T00:31:00+00:00",
            ),
        )
    return session_id, first_turn_id, second_turn_id, subject, catalog, cursor


def _output_decision(
    *,
    content: str,
    submit: bool = False,
    updates: tuple[AcceptanceUpdate, ...] = (),
) -> AttemptDecision:
    action_type = SubmitOutputWindowAction if submit else WriteOutputWindowAction
    return AttemptDecision(
        acceptance_updates=updates,
        action=action_type(
            content=content,
            format=OutputWindowFormat.MARKDOWN,
        ),
    )


def _submit_for_verification(
    session_id: str,
    turn_id: str,
    subject: TaskNodeSubject,
    *,
    work_run_id: str = "workrun-one",
    content: str = "verified candidate",
    expected_task_state_version: int = 1,
    expected_node_state_version: int = 1,
):
    _create(
        session_id,
        turn_id,
        subject,
        apply_id=f"create-{work_run_id}",
        work_run_id=work_run_id,
        expected_task_state_version=expected_task_state_version,
        expected_node_state_version=expected_node_state_version,
    )
    _start(
        session_id,
        turn_id,
        work_run_id,
        expected_run_revision=1,
        apply_id=f"start-{work_run_id}",
        attempt_id=f"submit-attempt-{work_run_id}",
    )
    return work_run_store.commit_work_run_output_action(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        attempt_id=f"submit-attempt-{work_run_id}",
        decision=_output_decision(
            content=content,
            submit=True,
            updates=tuple(
                AcceptanceUpdate(
                    acceptance_id=acceptance_id,
                    model_claimed_satisfied=True,
                )
                for acceptance_id in ("deliverable", "quality")
            ),
        ),
        expected_work_run_revision=2,
        expected_progress_revision=1,
        expected_output_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id=f"submit-{work_run_id}",
    )


def _verification_result(prepared, *, all_pass: bool) -> NodeVerificationResult:
    request = prepared.record.request
    results = []
    for index, acceptance_id in enumerate(request.acceptance_ids):
        passed = all_pass or index > 0
        results.append(
            AcceptanceVerificationFeedback(
                acceptance_id=acceptance_id,
                verdict=(
                    VerificationVerdict.PASSED
                    if passed
                    else VerificationVerdict.NOT_SATISFIED
                ),
                finding=(
                    "criterion satisfied" if passed else "deliverable is incomplete"
                ),
                missing_requirements=(
                    () if passed else ("complete the missing section",)
                ),
            )
        )
    return NodeVerificationResult(
        verification_request_id=request.verification_request_id,
        verification_request_revision=request.revision,
        work_run_id=request.work_run_id,
        locked_work_run_revision=request.locked_work_run_revision,
        submitted_attempt_id=request.submitted_attempt_id,
        acceptance_progress_revision=request.acceptance_progress_revision,
        subject=request.subject,
        output_revision=request.output_revision,
        acceptance_results=tuple(results),
        all_pass=all_pass,
    )


def _seed_budgeted_attempt(
    *,
    active_seconds_before_attempt: float,
    attempt_id: str,
):
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    charged = _charge_active_time(
        session_id,
        turn_id,
        expected_run_revision=1,
        checkpoint_id=f"budget-before-{attempt_id}",
        active_seconds_delta=active_seconds_before_attempt,
        apply_id=f"charge-before-{attempt_id}",
    )
    started = _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=charged.work_run_revision,
        apply_id=f"start-{attempt_id}",
        attempt_id=attempt_id,
    )
    return session_id, turn_id, subject, started


def _seed_budgeted_verification(*, active_seconds_before_attempt: float):
    session_id, turn_id, subject, started = _seed_budgeted_attempt(
        active_seconds_before_attempt=active_seconds_before_attempt,
        attempt_id="budget-submit-attempt",
    )
    submitted = work_run_store.commit_work_run_output_action(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="budget-submit-attempt",
        decision=_output_decision(
            content="budgeted candidate",
            submit=True,
            updates=tuple(
                AcceptanceUpdate(
                    acceptance_id=acceptance_id,
                    model_claimed_satisfied=True,
                )
                for acceptance_id in ("deliverable", "quality")
            ),
        ),
        expected_work_run_revision=started.work_run_revision,
        expected_progress_revision=started.acceptance_progress_revision,
        expected_output_revision=started.output_window_revision,
        expected_window_revision=started.window_state_version,
        apply_id="budget-submit",
        active_seconds_delta=1,
    )
    prepared_mutation = verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_work_run_revision=submitted.work_run_revision,
        expected_progress_revision=submitted.acceptance_progress_revision,
        expected_output_revision=submitted.output_window_revision,
        expected_window_revision=submitted.window_state_version,
        apply_id="budget-prepare-verification",
        verification_request_id="budget-verification",
    )
    prepared = verification_store.get_prepared_task_node_verification(
        session_id=session_id,
        invocation_turn_id=turn_id,
        verification_request_id="budget-verification",
    )
    return session_id, turn_id, subject, prepared_mutation, prepared


def _commit_passing_verification(
    session_id: str,
    turn_id: str,
    subject: TaskNodeSubject,
    *,
    request_id: str = "verification-pass",
    delivery_id: str = "delivery-pass",
    work_run_id: str = "workrun-one",
    content: str = "verified candidate",
    expected_task_state_version: int = 1,
    expected_node_state_version: int = 1,
):
    _submit_for_verification(
        session_id,
        turn_id,
        subject,
        work_run_id=work_run_id,
        content=content,
        expected_task_state_version=expected_task_state_version,
        expected_node_state_version=expected_node_state_version,
    )
    verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        expected_work_run_revision=3,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=_window_revision(session_id),
        apply_id=f"prepare-{request_id}",
        verification_request_id=request_id,
    )
    prepared = verification_store.get_prepared_task_node_verification(
        session_id=session_id,
        invocation_turn_id=turn_id,
        verification_request_id=request_id,
    )
    committed = verification_store.commit_task_node_verification_result(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=work_run_id,
        verification_request_id=request_id,
        result=_verification_result(prepared, all_pass=True),
        expected_work_run_revision=4,
        expected_verification_request_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id=f"commit-{request_id}",
        delivery_id=delivery_id,
    )
    return prepared, committed


def test_create_and_get_work_run_are_typed_atomic_and_exactly_replayable():
    session_id, turn_id, subject = _seed_task_node()
    original_window_revision = _window_revision(session_id)

    created = _create(
        session_id,
        turn_id,
        subject,
        expected_window_revision=original_window_revision,
    )

    assert created.status == "applied"
    assert created.work_run_id == "workrun-one"
    assert created.work_run_revision == 1
    assert created.acceptance_progress_revision == 1
    findings = store.get_execution_findings_ledger_for_owner(
        owner_kind="work_run",
        execution_owner_id=created.work_run_id,
    )
    assert findings is not None
    assert findings.ledger.originating_turn_id == turn_id
    assert findings.ledger.revision == 0
    loaded = _get(session_id)
    assert loaded.work_run.budget.attempts_started == 0
    assert [item.acceptance_id for item in loaded.acceptance_progress.items] == [
        "deliverable",
        "quality",
    ]
    assert all(
        not item.model_claimed_satisfied
        for item in loaded.acceptance_progress.items
    )
    assert loaded.output_window.output_revision == 1
    assert loaded.output_window.content == ""
    assert loaded.output_window.updated_attempt_id is None
    assert loaded.acceptance_progress.evaluated_output_revision == 1
    assert loaded.current_verification_request_id is None
    assert loaded.node_delivery_id is None
    window_after_create = _window_revision(session_id)

    replayed = _create(
        session_id,
        turn_id,
        subject,
        expected_window_revision=original_window_revision,
    )
    assert replayed.status == "replayed"
    assert replayed.model_copy(update={"status": "applied"}) == created
    assert _window_revision(session_id) == window_after_create
    with store._connect() as conn:
        receipt_json = str(
            conn.execute(
                "SELECT result_json FROM insession_work_run_apply_receipts "
                "WHERE apply_id='create-run'"
            ).fetchone()[0]
        )
    assert "record" not in receipt_json
    assert "arguments" not in receipt_json
    assert '"content"' not in receipt_json

    other_session_id = store.create_session("Entelecheia")
    with pytest.raises(work_run_store.WorkExecutionPersistenceError, match="this Session"):
        work_run_store.get_work_run(
            session_id=other_session_id,
            work_run_id="workrun-one",
        )

    with pytest.raises(work_run_store.WorkExecutionApplyIdCollision):
        _create(
            session_id,
            turn_id,
            subject,
            apply_id="create-run",
            work_run_id="different-run",
        )


def test_work_run_creation_rolls_back_when_findings_companion_fails():
    session_id, turn_id, subject = _seed_task_node(
        request_id="work-run-findings-rollback"
    )
    with store._connect() as conn:
        conn.executescript(
            """
            CREATE TRIGGER reject_test_findings_companion
            BEFORE INSERT ON execution_findings_ledgers
            BEGIN
                SELECT RAISE(ABORT, 'injected findings companion failure');
            END;
            """
        )

    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="findings companion",
    ):
        _create(session_id, turn_id, subject)

    with store._connect() as conn:
        work_run_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_work_runs WHERE work_run_id=?",
                ("workrun-one",),
            ).fetchone()[0]
        )
        ledger_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM execution_findings_ledgers "
                "WHERE execution_owner_id=?",
                ("workrun-one",),
            ).fetchone()[0]
        )
    assert work_run_count == 0
    assert ledger_count == 0


def test_output_action_whole_replaces_and_exactly_replays_without_copying_body():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=1,
        apply_id="start-output-one",
        attempt_id="attempt-output-one",
    )
    original_window_revision = _window_revision(session_id)
    decision = _output_decision(
        content="# first candidate",
        updates=(
            AcceptanceUpdate(
                acceptance_id="deliverable",
                model_claimed_satisfied=True,
            ),
        ),
    )
    committed = work_run_store.commit_work_run_output_action(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-output-one",
        decision=decision,
        expected_work_run_revision=2,
        expected_progress_revision=1,
        expected_output_revision=1,
        expected_window_revision=original_window_revision,
        apply_id="write-output-one",
    )

    assert committed.work_run_revision == 3
    assert committed.output_window_revision == 2
    assert committed.acceptance_progress_revision == 2
    assert committed.current_attempt_id is None
    assert committed.attempt is not None
    assert committed.attempt.status.value == "closed"
    record = _get(session_id)
    assert record.output_window.content == "# first candidate"
    assert record.output_window.output_revision == 2
    assert record.output_window.updated_attempt_id == "attempt-output-one"
    assert record.acceptance_progress.evaluated_output_revision == 2
    assert record.acceptance_progress.items[0].model_claimed_satisfied is True
    assert record.acceptance_progress.items[1].model_claimed_satisfied is False
    assert record.attempts[-1].committed_output_revision == 2
    assert record.attempts[-1].attempt.submitted_output_revision is None

    after_window_revision = _window_revision(session_id)
    replayed = work_run_store.commit_work_run_output_action(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-output-one",
        decision=decision,
        expected_work_run_revision=2,
        expected_progress_revision=1,
        expected_output_revision=1,
        expected_window_revision=original_window_revision,
        apply_id="write-output-one",
    )
    assert replayed.status == "replayed"
    assert replayed.model_copy(update={"status": "applied"}) == committed
    assert _window_revision(session_id) == after_window_revision

    with pytest.raises(work_run_store.WorkExecutionApplyIdCollision):
        work_run_store.commit_work_run_output_action(
            active_seconds_delta=2,
            session_id=session_id,
            turn_id=turn_id,
            work_run_id="workrun-one",
            attempt_id="attempt-output-one",
            decision=decision,
            expected_work_run_revision=2,
            expected_progress_revision=1,
            expected_output_revision=1,
            expected_window_revision=original_window_revision,
            apply_id="write-output-one",
        )

    with pytest.raises(work_run_store.WorkExecutionApplyIdCollision):
        work_run_store.commit_work_run_output_action(
            active_seconds_delta=1,
            session_id=session_id,
            turn_id=turn_id,
            work_run_id="workrun-one",
            attempt_id="attempt-output-one",
            decision=_output_decision(content="# changed collision"),
            expected_work_run_revision=2,
            expected_progress_revision=1,
            expected_output_revision=1,
            expected_window_revision=original_window_revision,
            apply_id="write-output-one",
        )

    with store._connect() as conn:
        attempt_json = str(
            conn.execute(
                "SELECT decision_json FROM insession_work_run_attempts "
                "WHERE attempt_id='attempt-output-one'"
            ).fetchone()[0]
        )
        receipt_json = str(
            conn.execute(
                "SELECT result_json FROM insession_work_run_apply_receipts "
                "WHERE apply_id='write-output-one'"
            ).fetchone()[0]
        )
        transcript = conn.execute(
            "SELECT role, content FROM session_turns WHERE session_id=? ORDER BY turn_idx",
            (session_id,),
        ).fetchall()
    assert "# first candidate" not in attempt_json
    assert '"content"' not in attempt_json
    assert "# first candidate" not in receipt_json
    assert [tuple(row) for row in transcript] == [("user", "执行当前任务节点")]


def test_output_action_has_independent_output_cas_and_late_failure_is_atomic(monkeypatch):
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=1,
        apply_id="start-output",
        attempt_id="attempt-output",
    )
    before_window = _window_revision(session_id)
    with pytest.raises(work_run_store.WorkExecutionOutputRevisionConflict):
        work_run_store.commit_work_run_output_action(
            active_seconds_delta=1,
            session_id=session_id,
            turn_id=turn_id,
            work_run_id="workrun-one",
            attempt_id="attempt-output",
            decision=_output_decision(content="stale"),
            expected_work_run_revision=2,
            expected_progress_revision=1,
            expected_output_revision=2,
            expected_window_revision=before_window,
            apply_id="stale-output",
        )
    assert _window_revision(session_id) == before_window
    assert _get(session_id).output_window.output_revision == 1

    original_insert_receipt = work_execution_records._insert_receipt

    def _fail_receipt(*_args, **_kwargs):
        raise sqlite3.OperationalError("injected output receipt failure")

    monkeypatch.setattr(work_execution_records, "_insert_receipt", _fail_receipt)
    with pytest.raises(sqlite3.OperationalError, match="output receipt"):
        work_run_store.commit_work_run_output_action(
            active_seconds_delta=1,
            session_id=session_id,
            turn_id=turn_id,
            work_run_id="workrun-one",
            attempt_id="attempt-output",
            decision=_output_decision(content="candidate"),
            expected_work_run_revision=2,
            expected_progress_revision=1,
            expected_output_revision=1,
            expected_window_revision=before_window,
            apply_id="late-output",
        )
    record = _get(session_id)
    assert record.output_window.output_revision == 1
    assert record.output_window.content == ""
    assert record.acceptance_progress.revision == 1
    assert record.work_run.revision == 2
    assert record.current_attempt_id == "attempt-output"
    assert record.attempts[-1].attempt.status.value == "active"
    assert record.attempts[-1].decision is None
    assert _window_revision(session_id) == before_window

    monkeypatch.setattr(
        work_execution_records,
        "_insert_receipt",
        original_insert_receipt,
    )
    retried = work_run_store.commit_work_run_output_action(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-output",
        decision=_output_decision(content="candidate"),
        expected_work_run_revision=2,
        expected_progress_revision=1,
        expected_output_revision=1,
        expected_window_revision=before_window,
        apply_id="late-output",
    )
    assert retried.status == "applied"


def test_output_window_survives_turn_settlement_and_next_turn_whole_replace_resets_progress():
    session_id, first_turn_id, subject = _seed_task_node()
    _create(session_id, first_turn_id, subject)
    _start(
        session_id,
        first_turn_id,
        "workrun-one",
        expected_run_revision=1,
        apply_id="start-first-output",
        attempt_id="attempt-first-output",
    )
    work_run_store.commit_work_run_output_action(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=first_turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-first-output",
        decision=_output_decision(
            content="first draft",
            updates=(
                AcceptanceUpdate(
                    acceptance_id="deliverable",
                    model_claimed_satisfied=True,
                ),
            ),
        ),
        expected_work_run_revision=2,
        expected_progress_revision=1,
        expected_output_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="write-first-output",
    )
    interrupted = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=_window_revision(session_id),
        stage="WORKRUN",
        interruption_reason="turn_boundary",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=int(interrupted["state_version"]),
        end_reason="turn_boundary",
        error_code=None,
    )
    assert _get(session_id).output_window.content == "first draft"

    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="continue-work-execution",
        source="runtime_test",
        user_text="继续当前任务",
        lease_owner="work-execution-test",
    )
    second_turn_id = str(accepted["turn"]["turn_id"])
    now = "2026-08-14T00:10:00+00:00"
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, relation, created_at) "
            "VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (session_id, second_turn_id, subject.task_id, now),
        )
        conn.execute(
            "INSERT INTO insession_work_run_turn_links "
            "(session_id, turn_id, work_run_id, link_revision, relation, created_at) "
            "VALUES (?, ?, 'workrun-one', 1, 'continued', ?)",
            (session_id, second_turn_id, now),
        )
        current_window_revision = int(
            conn.execute(
                "SELECT state_version FROM turn_execution_windows WHERE session_id=?",
                (session_id,),
            ).fetchone()[0]
        )
        assert conn.execute(
            "UPDATE turn_execution_windows SET current_work_run_id='workrun-one', "
            "turn_workrun_link_revision=1, state_version=state_version+1, updated_at=? "
            "WHERE session_id=? AND turn_id=? AND state_version=?",
            (now, session_id, second_turn_id, current_window_revision),
        ).rowcount == 1

    _start(
        session_id,
        second_turn_id,
        "workrun-one",
        expected_run_revision=3,
        expected_progress_revision=2,
        apply_id="start-second-output",
        attempt_id="attempt-second-output",
    )
    work_run_store.commit_work_run_output_action(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=second_turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-second-output",
        decision=_output_decision(
            content="second draft",
            updates=(
                AcceptanceUpdate(
                    acceptance_id="quality",
                    model_claimed_satisfied=True,
                ),
            ),
        ),
        expected_work_run_revision=4,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=_window_revision(session_id),
        apply_id="write-second-output",
    )
    record = _get(session_id)
    assert record.output_window.content == "second draft"
    assert record.output_window.output_revision == 3
    assert record.output_window.updated_turn_id == second_turn_id
    assert record.related_turn_ids == (first_turn_id, second_turn_id)
    assert record.acceptance_progress.evaluated_output_revision == 3
    assert record.acceptance_progress.items[0].model_claimed_satisfied is False
    assert record.acceptance_progress.items[1].model_claimed_satisfied is True


def test_loader_rejects_materialized_output_reference_drift():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=1,
        apply_id="start-output",
        attempt_id="attempt-output",
    )
    work_run_store.commit_work_run_output_action(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-output",
        decision=_output_decision(content="candidate"),
        expected_work_run_revision=2,
        expected_progress_revision=1,
        expected_output_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="write-output",
    )
    with store._connect() as conn:
        decision_json = str(
            conn.execute(
                "SELECT decision_json FROM insession_work_run_attempts "
                "WHERE attempt_id='attempt-output'"
            ).fetchone()[0]
        )
        payload = json.loads(decision_json)
        payload["action"]["work_run_id"] = "different-run"
        conn.execute(
            "UPDATE insession_work_run_attempts SET decision_json=? "
            "WHERE attempt_id='attempt-output'",
            (
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        )

    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="materialized OutputWindow decision binding",
    ):
        _get(session_id)


def test_get_work_run_reads_one_coherent_sqlite_snapshot_during_concurrent_commit(
    monkeypatch,
):
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=1,
        apply_id="start-concurrent-output",
        attempt_id="attempt-concurrent-output",
    )
    with store._connect() as conn:
        assert str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower() == "wal"

    reader_after_run_row = threading.Event()
    writer_committed = threading.Event()
    reader_thread_id: list[int] = []
    reader_result: list[object] = []
    reader_errors: list[BaseException] = []
    original_load_progress = work_execution_records._load_progress

    def _gated_load_progress(conn, work_run_id):
        if reader_thread_id and threading.get_ident() == reader_thread_id[0]:
            reader_after_run_row.set()
            if not writer_committed.wait(timeout=3):
                raise AssertionError("writer did not commit while reader was gated")
        return original_load_progress(conn, work_run_id)

    monkeypatch.setattr(
        work_execution_records,
        "_load_progress",
        _gated_load_progress,
    )

    def _read() -> None:
        reader_thread_id.append(threading.get_ident())
        try:
            reader_result.append(_get(session_id))
        except BaseException as exc:  # pragma: no cover - 错误会在下方显式暴露
            reader_errors.append(exc)

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()
    assert reader_after_run_row.wait(timeout=3)

    work_run_store.commit_work_run_output_action(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-concurrent-output",
        decision=_output_decision(content="concurrently committed"),
        expected_work_run_revision=2,
        expected_progress_revision=1,
        expected_output_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="concurrent-output",
    )
    writer_committed.set()
    reader.join(timeout=3)
    assert not reader.is_alive()
    assert reader_errors == []
    assert len(reader_result) == 1

    record = reader_result[0]
    pre_commit = (
        record.work_run.revision == 2
        and record.current_attempt_id == "attempt-concurrent-output"
        and record.output_window.output_revision == 1
        and record.attempts[-1].attempt.status.value == "active"
    )
    post_commit = (
        record.work_run.revision == 3
        and record.current_attempt_id is None
        and record.output_window.output_revision == 2
        and record.attempts[-1].attempt.status.value == "closed"
    )
    assert pre_commit or post_commit


def test_create_revalidates_subject_ownership_revisions_and_rolls_back_late_failure(
    monkeypatch,
):
    session_id, turn_id, subject = _seed_task_node()
    initial_window = _window_revision(session_id)

    with pytest.raises(work_run_store.WorkExecutionRevisionConflict):
        work_run_store.create_task_node_work_run(
            session_id=session_id,
            turn_id=turn_id,
            subject=subject,
            expected_task_state_version=1,
            expected_node_state_version=2,
            expected_window_revision=initial_window,
            apply_id="stale-node-create",
            work_run_id="never-created",
        )

    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM insession_work_runs").fetchone()[0] == 0
        assert conn.execute(
            "SELECT state_version FROM insession_task_node_states WHERE insession_task_id=?",
            (subject.task_id,),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_apply_receipts"
        ).fetchone()[0] == 0
    assert _window_revision(session_id) == initial_window

    with pytest.raises(TypeError, match="TaskNode execution subject"):
        work_run_store.create_task_node_work_run(
            session_id=session_id,
            turn_id=turn_id,
            subject=AuxiliaryNodeSubject(
                task_id=subject.task_id,
                auxiliary_graph_id="aux-one",
                auxiliary_graph_revision=1,
                node_id="aux-node",
                node_revision=1,
            ),
            expected_task_state_version=1,
            expected_node_state_version=1,
            expected_window_revision=initial_window,
            apply_id="aux-create",
        )

    original_insert_receipt = work_execution_records._insert_receipt

    def _fail_receipt(*_args, **_kwargs):
        raise sqlite3.OperationalError("injected late receipt failure")

    monkeypatch.setattr(work_execution_records, "_insert_receipt", _fail_receipt)
    with pytest.raises(sqlite3.OperationalError, match="late receipt"):
        _create(
            session_id,
            turn_id,
            subject,
            apply_id="late-failure-create",
            work_run_id="late-failure-run",
            expected_window_revision=initial_window,
        )
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_runs"
        ).fetchone()[0] == 0
        assert tuple(
            conn.execute(
                "SELECT current_status, state_version FROM insession_tasks "
                "WHERE insession_task_id=?",
                (subject.task_id,),
            ).fetchone()
        ) == ("proposed", 1)
        assert tuple(
            conn.execute(
                "SELECT status, state_version FROM insession_task_node_states "
                "WHERE insession_task_id=? AND insession_task_node_id=?",
                (subject.task_id, subject.node_id),
            ).fetchone()
        ) == ("proposed", 1)
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_turn_links"
        ).fetchone()[0] == 0
    assert _window_revision(session_id) == initial_window

    monkeypatch.setattr(
        work_execution_records,
        "_insert_receipt",
        original_insert_receipt,
    )
    retried = _create(
        session_id,
        turn_id,
        subject,
        apply_id="late-failure-create",
        work_run_id="late-failure-run",
        expected_window_revision=initial_window,
    )
    assert retried.status == "applied"


def test_turn_limit_lane_projects_authoritative_no_public_stop():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    limited = _charge_active_time(
        session_id,
        turn_id,
        expected_run_revision=1,
        checkpoint_id="no-public-turn-limit",
        active_seconds_delta=720,
        apply_id="charge-no-public-turn-limit",
    )
    assert limited.work_run_status.value == "turn_limit_reached"
    detached = work_run_store.detach_safe_work_run_lane(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_window_revision=limited.window_state_version,
        apply_id="detach-no-public-turn-limit",
    )
    settled = store.advance_turn_execution_window(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=detached.window_state_version,
        stage="PERSIST",
        lease_owner="work-execution-test",
    )

    marked = store.mark_authoritative_no_public_turn_stop(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(settled["state_version"]),
    )

    assert marked["stop_kind"] == "turn_limit_reached"
    assert marked["end_reason"] == "host_stopped"
    assert marked["error_code"] == "TURN_DEADLINE_EXCEEDED"
    assert marked["stage"] == "RESPONSE"
    assert marked["window"]["window_state"] == "interrupted"


def test_active_time_charge_is_exactly_once_and_enforces_soft_and_hard_limits():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    first_window_revision = _window_revision(session_id)

    within = _charge_active_time(
        session_id,
        turn_id,
        expected_run_revision=1,
        checkpoint_id="budget-checkpoint-1",
        active_seconds_delta=719,
        apply_id="charge-budget-1",
        expected_window_revision=first_window_revision,
    )
    assert within.work_run_revision == 2
    assert within.work_run_status.value == "active"
    assert within.work_run_reason is None
    assert within.transition.budget_after.active_seconds_consumed == 719
    assert within.transition.disposition.value == "within_limit"
    charged_window_revision = within.window_state_version

    replayed = _charge_active_time(
        session_id,
        turn_id,
        expected_run_revision=1,
        checkpoint_id="budget-checkpoint-1",
        active_seconds_delta=719,
        apply_id="charge-budget-1",
        expected_window_revision=first_window_revision,
    )
    assert replayed.status == "replayed"
    assert replayed.work_run_revision == 2
    assert replayed.window_state_version == charged_window_revision
    record = _get(session_id)
    assert record.work_run.budget.active_seconds_consumed == 719
    assert len(record.budget_charges) == 1
    assert record.budget_charges[0].checkpoint_id == "budget-checkpoint-1"

    with pytest.raises(
        work_execution_records.WorkExecutionPersistenceError,
        match="checkpoint already",
    ):
        _charge_active_time(
            session_id,
            turn_id,
            expected_run_revision=2,
            checkpoint_id="budget-checkpoint-1",
            active_seconds_delta=1,
            apply_id="charge-same-checkpoint-new-apply",
        )

    with pytest.raises(work_execution_records.WorkExecutionApplyIdCollision):
        _charge_active_time(
            session_id,
            turn_id,
            expected_run_revision=2,
            checkpoint_id="budget-checkpoint-collision",
            active_seconds_delta=1,
            apply_id="charge-budget-1",
        )

    soft = _charge_active_time(
        session_id,
        turn_id,
        expected_run_revision=2,
        checkpoint_id="budget-checkpoint-2",
        active_seconds_delta=1,
        apply_id="charge-budget-2",
    )
    assert soft.work_run_revision == 3
    assert soft.work_run_status.value == "turn_limit_reached"
    assert soft.work_run_reason == "turn_limit_reached"
    assert soft.transition.budget_after.active_seconds_consumed == 720
    assert soft.transition.disposition.value == "soft_limit_reached"

    closing = _charge_active_time(
        session_id,
        turn_id,
        expected_run_revision=3,
        checkpoint_id="budget-checkpoint-3",
        active_seconds_delta=179,
        apply_id="charge-budget-3",
    )
    assert closing.work_run_status.value == "turn_limit_reached"
    assert closing.transition.budget_after.active_seconds_consumed == 899

    hard = _charge_active_time(
        session_id,
        turn_id,
        expected_run_revision=4,
        checkpoint_id="budget-checkpoint-4",
        active_seconds_delta=1,
        apply_id="charge-budget-4",
    )
    assert hard.work_run_revision == 5
    assert hard.work_run_status.value == "failed"
    assert hard.work_run_reason == "work_run_limit_reached"
    assert hard.transition.budget_after.active_seconds_consumed == 900
    assert hard.transition.disposition.value == "hard_limit_reached"
    assert _get(session_id).work_run.budget.active_seconds_consumed == 900
    with store._connect() as conn:
        node = conn.execute(
            "SELECT status FROM insession_task_node_states "
            "WHERE insession_task_id=? AND insession_task_node_id=?",
            (subject.task_id, subject.node_id),
        ).fetchone()
        task = conn.execute(
            "SELECT current_status FROM insession_tasks WHERE insession_task_id=?",
            (subject.task_id,),
        ).fetchone()
        assert node is not None and node["status"] == "interrupted"
        assert task is not None and task["current_status"] == "active"

    # 终态 WorkRun 仍绑定当前窗口，因此所属轮次仍可使用常规中断/稳定路径。
    interrupted = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=hard.window_state_version,
        stage="WORKRUN",
        interruption_reason="work_run_limit_reached",
    )
    assert interrupted["current_work_run_id"] == "workrun-one"


def test_waiting_idle_and_active_attempts_cannot_be_charged():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=1,
        apply_id="start-active-budget-guard",
        attempt_id="attempt-active-budget-guard",
    )
    with pytest.raises(
        work_execution_records.WorkExecutionPersistenceError,
        match="safe WorkRun checkpoint",
    ):
        _charge_active_time(
            session_id,
            turn_id,
            expected_run_revision=2,
            checkpoint_id="unsafe-active-attempt",
            active_seconds_delta=30,
            apply_id="charge-unsafe-active-attempt",
        )
    assert _get(session_id).work_run.budget.active_seconds_consumed == 0

    waiting = work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-active-budget-guard",
        decision=HostAcceptedAttemptDecision(
            action=RequestUserInputAction(question="Which source should I use?")
        ),
        expected_work_run_revision=2,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="wait-before-idle",
        active_seconds_delta=1,
    )
    assert waiting.work_run_status.value == "waiting_user"
    with pytest.raises(
        work_execution_records.WorkExecutionPersistenceError,
        match="does not permit",
    ):
        _charge_active_time(
            session_id,
            turn_id,
            expected_run_revision=3,
            checkpoint_id="idle-wait-must-not-charge",
            active_seconds_delta=600,
            apply_id="charge-idle-wait",
        )
    record = _get(session_id)
    assert record.work_run.status.value == "waiting_user"
    assert record.work_run.budget.active_seconds_consumed == 1
    assert len(record.budget_charges) == 1
    assert record.budget_charges[0].operation == "commit_attempt_decision"


@pytest.mark.parametrize(
    (
        "action_kind",
        "active_seconds_before_attempt",
        "expected_status",
        "expected_reason",
        "expected_disposition",
        "settlement_delta",
    ),
    (
        ("request_user", 719, "waiting_user", "needs_input", "soft_limit_reached", 1),
        ("request_user", 719, "failed", "work_run_limit_reached", "hard_limit_reached", 181),
        ("write", 719, "turn_limit_reached", "turn_limit_reached", "soft_limit_reached", 1),
        ("write", 719, "failed", "work_run_limit_reached", "hard_limit_reached", 181),
        ("submit", 719, "active", "verification_pending", "soft_limit_reached", 1),
        ("submit", 719, "failed", "work_run_limit_reached", "hard_limit_reached", 181),
    ),
)
def test_attempt_terminal_settlement_charges_soft_and_hard_boundaries_atomically(
    action_kind,
    active_seconds_before_attempt,
    expected_status,
    expected_reason,
    expected_disposition,
    settlement_delta,
):
    session_id, turn_id, _subject, started = _seed_budgeted_attempt(
        active_seconds_before_attempt=active_seconds_before_attempt,
        attempt_id="budget-action-attempt",
    )

    if action_kind == "request_user":
        committed = work_run_store.commit_work_run_attempt_decision(
            session_id=session_id,
            turn_id=turn_id,
            work_run_id="workrun-one",
            attempt_id="budget-action-attempt",
            decision=HostAcceptedAttemptDecision(
                action=RequestUserInputAction(question="Need one detail")
            ),
            expected_work_run_revision=started.work_run_revision,
            expected_progress_revision=started.acceptance_progress_revision,
            expected_window_revision=started.window_state_version,
            apply_id="budget-terminal-action",
            active_seconds_delta=settlement_delta,
        )
    else:
        updates = (
            tuple(
                AcceptanceUpdate(
                    acceptance_id=acceptance_id,
                    model_claimed_satisfied=True,
                )
                for acceptance_id in ("deliverable", "quality")
            )
            if action_kind == "submit"
            else ()
        )
        committed = work_run_store.commit_work_run_output_action(
            session_id=session_id,
            turn_id=turn_id,
            work_run_id="workrun-one",
            attempt_id="budget-action-attempt",
            decision=_output_decision(
                content="budget boundary candidate",
                submit=action_kind == "submit",
                updates=updates,
            ),
            expected_work_run_revision=started.work_run_revision,
            expected_progress_revision=started.acceptance_progress_revision,
            expected_output_revision=started.output_window_revision,
            expected_window_revision=started.window_state_version,
            apply_id="budget-terminal-action",
            active_seconds_delta=settlement_delta,
        )

    assert committed.work_run_status.value == expected_status
    assert committed.work_run_reason == expected_reason
    assert committed.budget_transition is not None
    assert committed.budget_transition.disposition.value == expected_disposition
    assert (
        committed.budget_transition.budget_after.active_seconds_consumed
        == active_seconds_before_attempt + settlement_delta
    )
    record = _get(session_id)
    assert record.work_run.budget.active_seconds_consumed == (
        active_seconds_before_attempt + settlement_delta
    )
    assert record.budget_charges[-1].operation in {
        "commit_attempt_decision",
        "commit_output_action",
    }
    assert record.attempts[-1].budget_charge_id == "budget-terminal-action"


@pytest.mark.parametrize(
    (
        "result_status",
        "active_seconds_before_attempt",
        "expected_status",
        "expected_reason",
        "expected_disposition",
        "settlement_delta",
    ),
    (
        ("succeeded", 719, "turn_limit_reached", "turn_limit_reached", "soft_limit_reached", 1),
        ("succeeded", 719, "failed", "work_run_limit_reached", "hard_limit_reached", 181),
        (
            "completion_unconfirmed",
            719,
            "waiting_external",
            "operation_completion_unconfirmed",
            "hard_limit_reached",
            181,
        ),
    ),
)
def test_tool_attempt_close_charges_once_and_preserves_unconfirmed_completion_authority(
    result_status,
    active_seconds_before_attempt,
    expected_status,
    expected_reason,
    expected_disposition,
    settlement_delta,
):
    session_id, turn_id, _subject, started = _seed_budgeted_attempt(
        active_seconds_before_attempt=active_seconds_before_attempt,
        attempt_id="budget-tool-attempt",
    )
    decided = work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="budget-tool-attempt",
        decision=_call_decision(
            call_id="budget-tool-call",
            modifies_environment=(result_status == "completion_unconfirmed"),
        ),
        expected_work_run_revision=started.work_run_revision,
        expected_progress_revision=started.acceptance_progress_revision,
        expected_window_revision=started.window_state_version,
        apply_id="budget-tool-decision",
    )
    status = ToolResultStatus(result_status)
    appended = work_run_store.append_work_run_tool_result(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        result=ToolResult(
            status=status,
            tool_result_id="budget-tool-result",
            tool_call_id="budget-tool-call",
            attempt_id="budget-tool-attempt",
            ordinal=1,
            output={"ok": True} if status is ToolResultStatus.SUCCEEDED else None,
            error_code=(
                None
                if status is ToolResultStatus.SUCCEEDED
                else "transport_completion_unconfirmed"
            ),
            error_message=(
                None
                if status is ToolResultStatus.SUCCEEDED
                else "dispatch result is not yet known"
            ),
        ),
        expected_work_run_revision=decided.work_run_revision,
        expected_progress_revision=decided.acceptance_progress_revision,
        expected_window_revision=decided.window_state_version,
        apply_id="budget-tool-result-apply",
    )
    closed = work_run_store.close_work_run_attempt(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="budget-tool-attempt",
        expected_work_run_revision=appended.work_run_revision,
        expected_progress_revision=appended.acceptance_progress_revision,
        expected_window_revision=appended.window_state_version,
        apply_id="budget-tool-close",
        active_seconds_delta=settlement_delta,
    )

    assert closed.work_run_status.value == expected_status
    assert closed.work_run_reason == expected_reason
    assert closed.budget_transition is not None
    assert closed.budget_transition.disposition.value == expected_disposition
    assert _get(session_id).work_run.budget.active_seconds_consumed == (
        active_seconds_before_attempt + settlement_delta
    )


@pytest.mark.parametrize(
    (
        "all_pass",
        "active_seconds_before_attempt",
        "expected_status",
        "expected_reason",
        "expected_request_status",
        "expect_result",
        "expect_delivery",
        "verification_delta",
    ),
    (
        (True, 718, "completed", "verification_passed", "completed", True, True, 1),
        (
            False,
            718,
            "turn_limit_reached",
            "turn_limit_reached",
            "completed",
            True,
            False,
            1,
        ),
        (
            True,
            718,
            "failed",
            "work_run_limit_reached",
            "interrupted",
            False,
            False,
            181,
        ),
    ),
)
def test_verification_result_settlement_applies_soft_grace_and_hard_cutoff(
    all_pass,
    active_seconds_before_attempt,
    expected_status,
    expected_reason,
    expected_request_status,
    expect_result,
    expect_delivery,
    verification_delta,
):
    session_id, turn_id, _subject, prepared_mutation, prepared = (
        _seed_budgeted_verification(
            active_seconds_before_attempt=active_seconds_before_attempt
        )
    )
    committed = verification_store.commit_task_node_verification_result(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        verification_request_id="budget-verification",
        result=_verification_result(prepared, all_pass=all_pass),
        expected_work_run_revision=prepared_mutation.work_run_revision,
        expected_verification_request_revision=1,
        expected_window_revision=prepared_mutation.window_state_version,
        apply_id="budget-verification-result",
        delivery_id="budget-delivery",
        active_seconds_delta=verification_delta,
    )

    assert committed.work_run_status.value == expected_status
    assert committed.work_run_reason == expected_reason
    assert committed.verification_request_status.value == expected_request_status
    assert (committed.delivery_id is not None) is expect_delivery
    assert committed.budget_transition is not None
    assert committed.budget_transition.budget_after.active_seconds_consumed == (
        active_seconds_before_attempt + 1 + verification_delta
    )
    record = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id="budget-verification",
    )
    assert (record.result is not None) is expect_result
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_task_node_deliveries "
            "WHERE work_run_id='workrun-one'"
        ).fetchone()[0] == int(expect_delivery)


def test_verification_technical_interruption_hard_limit_is_terminal_without_result():
    session_id, turn_id, _subject, prepared_mutation, _prepared = (
        _seed_budgeted_verification(active_seconds_before_attempt=718)
    )
    interrupted = verification_store.interrupt_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        verification_request_id="budget-verification",
        technical_error_code="provider_transport_failed",
        expected_work_run_revision=prepared_mutation.work_run_revision,
        expected_verification_request_revision=1,
        expected_window_revision=prepared_mutation.window_state_version,
        apply_id="budget-verification-interrupt",
        active_seconds_delta=181,
    )

    assert interrupted.work_run_status.value == "failed"
    assert interrupted.work_run_reason == "work_run_limit_reached"
    assert interrupted.verification_request_status.value == "interrupted"
    assert interrupted.budget_transition is not None
    assert interrupted.budget_transition.disposition.value == "hard_limit_reached"
    record = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id="budget-verification",
    )
    assert record.result is None
    assert record.request.technical_error_code == "work_run_limit_reached"


def test_active_time_charge_rejects_a_pending_window_operation_without_writes():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    before_window_revision = _window_revision(session_id)
    with store._connect() as conn:
        conn.execute(
            "UPDATE turn_execution_windows SET pending_operation_id=? "
            "WHERE session_id=? AND turn_id=?",
            ("modifying-operation-in-flight", session_id, turn_id),
        )

    with pytest.raises(
        work_execution_records.WorkExecutionPersistenceError,
        match="safe WorkRun checkpoint",
    ):
        _charge_active_time(
            session_id,
            turn_id,
            expected_run_revision=1,
            checkpoint_id="unsafe-pending-operation",
            active_seconds_delta=30,
            apply_id="charge-unsafe-pending-operation",
            expected_window_revision=before_window_revision,
        )

    with store._connect() as conn:
        assert tuple(
            conn.execute(
                "SELECT revision, active_seconds_consumed FROM insession_work_runs "
                "WHERE work_run_id='workrun-one'"
            ).fetchone()
        ) == (1, 0.0)
        assert tuple(
            conn.execute(
                "SELECT state_version, latest_checkpoint_id, pending_operation_id "
                "FROM turn_execution_windows WHERE session_id=?",
                (session_id,),
            ).fetchone()
        ) == (
            before_window_revision,
            None,
            "modifying-operation-in-flight",
        )
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_budget_charges"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_apply_receipts "
            "WHERE apply_id='charge-unsafe-pending-operation'"
        ).fetchone()[0] == 0


def test_hard_budget_interrupts_node_but_keeps_retry_frontier_active():
    session_id, turn_id, subject = _seed_task_node(
        task_id="canonical-budget-task",
        node_id="canonical-budget-task",
    )
    _create(session_id, turn_id, subject)
    hard = _charge_active_time(
        session_id,
        turn_id,
        expected_run_revision=1,
        checkpoint_id="canonical-hard-budget",
        active_seconds_delta=900,
        apply_id="charge-canonical-hard-budget",
    )
    assert hard.work_run_status.value == "failed"
    with store._connect() as conn:
        node = conn.execute(
            "SELECT status FROM insession_task_node_states "
            "WHERE insession_task_id=? AND insession_task_node_id=?",
            (subject.task_id, subject.node_id),
        ).fetchone()
        task = conn.execute(
            "SELECT current_status FROM insession_tasks WHERE insession_task_id=?",
            (subject.task_id,),
        ).fetchone()
        assert node is not None and node["status"] == "interrupted"
        assert task is not None and task["current_status"] == "active"
    frontier = work_run_store.project_task_node_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        task_id=subject.task_id,
    )
    assert [item.subject for item in frontier.ready_fresh] == [subject]
    assert frontier.ready_fresh[0].node_status == "interrupted"


def test_work_run_getter_fails_closed_when_budget_aggregate_loses_its_ledger():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    _charge_active_time(
        session_id,
        turn_id,
        expected_run_revision=1,
        checkpoint_id="discoverable-charge",
        active_seconds_delta=10,
        apply_id="charge-for-getter-audit",
    )
    assert _get(session_id).work_run.budget.active_seconds_consumed == 10
    with store._connect() as conn:
        conn.execute(
            "DELETE FROM insession_work_run_budget_charges "
            "WHERE budget_charge_id='charge-for-getter-audit'"
        )
    with pytest.raises(
        work_execution_records.WorkExecutionPersistenceError,
        match="aggregate active time disagrees",
    ):
        _get(session_id)


def test_budget_receipt_tamper_cannot_forge_replay_transition_or_cursor():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    initial_window_revision = _window_revision(session_id)
    applied = _charge_active_time(
        session_id,
        turn_id,
        expected_run_revision=1,
        checkpoint_id="tamper-replay-checkpoint",
        active_seconds_delta=10,
        apply_id="tamper-replay-charge",
        expected_window_revision=initial_window_revision,
    )
    with store._connect() as conn:
        row = conn.execute(
            "SELECT result_json FROM insession_work_run_apply_receipts "
            "WHERE apply_id='tamper-replay-charge'"
        ).fetchone()
        payload = json.loads(str(row["result_json"]))
        payload["transition"]["budget_before"]["attempts_started"] = 1
        payload["transition"]["budget_after"]["attempts_started"] = 1
        payload["window_state_version"] = applied.window_state_version + 1
        conn.execute(
            "UPDATE insession_work_run_apply_receipts SET result_json=? "
            "WHERE apply_id='tamper-replay-charge'",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")),),
        )

    with pytest.raises(
        work_execution_records.WorkExecutionPersistenceError,
        match="ledger is not contiguous",
    ):
        _charge_active_time(
            session_id,
            turn_id,
            expected_run_revision=1,
            checkpoint_id="tamper-replay-checkpoint",
            active_seconds_delta=10,
            apply_id="tamper-replay-charge",
            expected_window_revision=initial_window_revision,
        )
    with store._connect() as conn:
        assert tuple(
            conn.execute(
                "SELECT revision, active_seconds_consumed FROM insession_work_runs "
                "WHERE work_run_id='workrun-one'"
            ).fetchone()
        ) == (2, 10.0)
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_budget_charges "
            "WHERE work_run_id='workrun-one'"
        ).fetchone()[0] == 1


def test_work_run_getter_rejects_budget_revision_and_envelope_tamper():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    _charge_active_time(
        session_id,
        turn_id,
        expected_run_revision=1,
        checkpoint_id="tamper-ledger-checkpoint",
        active_seconds_delta=10,
        apply_id="tamper-ledger-charge",
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_run_budget_charges SET "
            "work_run_revision_before=50, work_run_revision_after=51 "
            "WHERE budget_charge_id='tamper-ledger-charge'"
        )
    with pytest.raises(
        work_execution_records.WorkExecutionPersistenceError,
        match="ledger is not contiguous",
    ):
        _get(session_id)

    with store._connect() as conn:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute(
            "UPDATE insession_work_runs SET soft_active_seconds=700 "
            "WHERE work_run_id='workrun-one'"
        )
    with pytest.raises(
        work_execution_records.WorkExecutionPersistenceError,
        match="frozen authority",
    ):
        _get(session_id)


def test_work_run_getter_rejects_attempt_aggregate_ordinal_and_snapshot_drift():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_runs SET attempts_started=1 "
            "WHERE work_run_id='workrun-one'"
        )
    with pytest.raises(
        work_execution_records.WorkExecutionPersistenceError,
        match="Attempt aggregate",
    ):
        _charge_active_time(
            session_id,
            turn_id,
            expected_run_revision=1,
            checkpoint_id="aggregate-drift-must-not-charge",
            active_seconds_delta=10,
            apply_id="aggregate-drift-charge",
        )
    with pytest.raises(
        work_execution_records.WorkExecutionPersistenceError,
        match="Attempt aggregate",
    ):
        _get(session_id)
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_budget_charges"
        ).fetchone()[0] == 0

    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_runs SET attempts_started=0 "
            "WHERE work_run_id='workrun-one'"
        )
    _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=1,
        apply_id="start-attempt-integrity",
        attempt_id="attempt-integrity",
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_run_attempts SET ordinal=2 "
            "WHERE attempt_id='attempt-integrity'"
        )
    with pytest.raises(
        work_execution_records.WorkExecutionPersistenceError,
        match="ordinals are not contiguous",
    ):
        _get(session_id)

    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_run_attempts SET ordinal=1, "
            "budget_before_json=? WHERE attempt_id='attempt-integrity'",
            (
                json.dumps(
                    {
                        "max_attempts": 32,
                        "soft_active_seconds": 720,
                        "hard_active_seconds": 900,
                        "attempts_started": 1,
                        "active_seconds_consumed": 0,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        )
    with pytest.raises(
        work_execution_records.WorkExecutionPersistenceError,
        match="budget-before snapshot",
    ):
        _get(session_id)


@pytest.mark.parametrize(
    ("consumed", "expected_message"),
    [(720.0, "soft active-time"), (900.0, "hard active-time")],
)
def test_attempt_start_fails_closed_if_active_time_aggregate_is_already_exhausted(
    consumed,
    expected_message,
):
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_runs SET active_seconds_consumed=? "
            "WHERE work_run_id='workrun-one'",
            (consumed,),
        )
    with pytest.raises(
        work_execution_records.WorkExecutionPersistenceError,
        match=expected_message,
    ):
        _start(
            session_id,
            turn_id,
            "workrun-one",
            expected_run_revision=1,
            apply_id=f"start-exhausted-{int(consumed)}",
            attempt_id=f"attempt-exhausted-{int(consumed)}",
        )
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_attempts "
            "WHERE work_run_id='workrun-one'"
        ).fetchone()[0] == 0


def test_attempt_start_fails_closed_at_the_frozen_32_attempt_limit():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_runs SET attempts_started=32 "
            "WHERE work_run_id='workrun-one'"
        )
    with pytest.raises(
        work_execution_records.WorkExecutionPersistenceError,
        match="Attempt budget is exhausted",
    ):
        _start(
            session_id,
            turn_id,
            "workrun-one",
            expected_run_revision=1,
            apply_id="start-attempt-33",
            attempt_id="attempt-33",
        )
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_attempts "
            "WHERE work_run_id='workrun-one'"
        ).fetchone()[0] == 0


def test_active_time_budget_survives_cross_turn_rebind_without_reset():
    session_id, first_turn_id, subject = _seed_task_node()
    _create(session_id, first_turn_id, subject)
    first = _charge_active_time(
        session_id,
        first_turn_id,
        expected_run_revision=1,
        checkpoint_id="first-turn-charge",
        active_seconds_delta=100,
        apply_id="charge-first-turn",
    )
    interrupted = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=first.window_state_version,
        stage="WORKRUN",
        interruption_reason="turn_boundary",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=int(interrupted["state_version"]),
        end_reason="turn_boundary",
        error_code=None,
    )

    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="budget-second-turn",
        source="runtime_test",
        user_text="继续当前任务",
        lease_owner="work-execution-test",
    )
    second_turn_id = str(accepted["turn"]["turn_id"])
    now = "2026-08-14T00:20:00+00:00"
    with store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (session_id, second_turn_id, subject.task_id, now),
        )
        conn.execute(
            "INSERT INTO insession_work_run_turn_links "
            "(session_id, turn_id, work_run_id, link_revision, relation, created_at) "
            "VALUES (?, ?, 'workrun-one', 1, 'continued', ?)",
            (session_id, second_turn_id, now),
        )
        window_revision = int(
            conn.execute(
                "SELECT state_version FROM turn_execution_windows WHERE session_id=?",
                (session_id,),
            ).fetchone()[0]
        )
        assert conn.execute(
            "UPDATE turn_execution_windows SET current_work_run_id='workrun-one', "
            "turn_workrun_link_revision=1, state_version=state_version+1, updated_at=? "
            "WHERE session_id=? AND turn_id=? AND state_version=?",
            (now, session_id, second_turn_id, window_revision),
        ).rowcount == 1

    second = _charge_active_time(
        session_id,
        second_turn_id,
        expected_run_revision=2,
        checkpoint_id="second-turn-charge",
        active_seconds_delta=50,
        apply_id="charge-second-turn",
    )
    assert second.transition.budget_before.active_seconds_consumed == 100
    assert second.transition.budget_after.active_seconds_consumed == 150
    record = _get(session_id)
    assert record.work_run.budget.active_seconds_consumed == 150
    assert tuple(charge.turn_id for charge in record.budget_charges) == (
        first_turn_id,
        second_turn_id,
    )


def test_attempt_start_charges_budget_once_and_decision_cannot_use_current_attempt_result():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    start_window_revision = _window_revision(session_id)
    started = _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=1,
        apply_id="start-one",
        attempt_id="attempt-one",
        expected_window_revision=start_window_revision,
    )
    assert started.work_run_revision == 2
    assert _get(session_id).work_run.budget.attempts_started == 1
    window_after_start = _window_revision(session_id)

    replayed = _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=1,
        apply_id="start-one",
        attempt_id="attempt-one",
        expected_window_revision=start_window_revision,
    )
    assert replayed.status == "replayed"
    assert _get(session_id).work_run.budget.attempts_started == 1
    assert _window_revision(session_id) == window_after_start
    with store._connect() as conn:
        checkpoint = conn.execute(
            "SELECT input_checkpoint_id FROM insession_work_run_attempts "
            "WHERE attempt_id='attempt-one'"
        ).fetchone()[0]
    assert checkpoint is None

    # 已损坏或带外的同次尝试结果仍不能进入归约器允许列表。
    now = "2026-08-14T00:01:00+00:00"
    fake_result = ToolResult(
        status=ToolResultStatus.SUCCEEDED,
        tool_result_id="same-attempt-result",
        tool_call_id="fake-call",
        attempt_id="attempt-one",
        ordinal=1,
        output={"value": "untrusted"},
    )
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_work_run_tool_calls "
            "(tool_call_id, work_run_id, attempt_id, ordinal, provider_call_id, tool_id, "
            "tool_version, modifies_environment, arguments_hash, arguments_json, created_at) "
            "VALUES ('fake-call', 'workrun-one', 'attempt-one', 1, NULL, 'read_test', "
            "'1.0.0', 0, ?, '{}', ?)",
            ("0" * 64, now),
        )
        conn.execute(
            "INSERT INTO insession_work_run_tool_results "
            "(tool_result_id, work_run_id, attempt_id, tool_call_id, ordinal, status, result_json, created_at) "
            "VALUES (?, 'workrun-one', 'attempt-one', 'fake-call', 1, 'succeeded', ?, ?)",
            (fake_result.tool_result_id, fake_result.model_dump_json(), now),
        )

    decision = HostAcceptedAttemptDecision(
        acceptance_updates=(
            AcceptanceUpdate(
                acceptance_id="deliverable",
                model_claimed_satisfied=True,
                supporting_tool_result_ids=("same-attempt-result",),
            ),
        ),
        action=RequestUserInputAction(question="继续吗？"),
    )
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="undecided Attempt already contains execution authority",
    ):
        work_run_store.commit_work_run_attempt_decision(
            session_id=session_id,
            turn_id=turn_id,
            work_run_id="workrun-one",
            attempt_id="attempt-one",
            decision=decision,
            expected_work_run_revision=2,
            expected_progress_revision=1,
            expected_window_revision=_window_revision(session_id),
            apply_id="invalid-current-result-decision",
            active_seconds_delta=1,
        )


def test_undecided_active_attempt_rebinds_after_turn_settlement_and_restarts_timing():
    session_id, first_turn_id, subject = _seed_task_node()
    _create(session_id, first_turn_id, subject)
    started = _start(
        session_id,
        first_turn_id,
        "workrun-one",
        expected_run_revision=1,
        apply_id="start-rebind",
        attempt_id="attempt-rebind",
    )
    marked = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=started.window_state_version,
        stage="L2_PLAN",
        interruption_reason="process_lost",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=int(marked["state_version"]),
        end_reason="process_lost",
        error_code="PROCESS_LOST",
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="resume-active-attempt",
        source="runtime_test",
        user_text="继续",
        lease_owner="work-execution-test",
    )
    second_turn_id = str(accepted["turn"]["turn_id"])
    now = "2026-08-14T00:30:00+00:00"
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (session_id, second_turn_id, subject.task_id, now),
        )

    resume_window_revision = _window_revision(session_id)
    resumed = work_run_store.resume_active_work_run_attempt(
        session_id=session_id,
        turn_id=second_turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-rebind",
        expected_work_run_revision=started.work_run_revision,
        expected_window_revision=resume_window_revision,
        apply_id="resume-active-attempt",
    )
    assert resumed.status == "applied"
    assert resumed.work_run_revision == started.work_run_revision + 1
    assert resumed.current_attempt_id == "attempt-rebind"
    assert resumed.attempt is not None and resumed.attempt.ordinal == 1
    record = _get(session_id)
    assert record.work_run.budget.attempts_started == 1
    assert record.work_run.budget.active_seconds_consumed == 0
    assert record.related_turn_ids == (first_turn_id, second_turn_id)
    assert record.attempts[0].turn_id == second_turn_id
    assert record.attempts[0].input_turn_id == first_turn_id
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert window["current_work_run_id"] == "workrun-one"
    assert window["current_attempt_id"] == "attempt-rebind"
    assert window["stage"] == "L2_PLAN"
    with store._connect() as conn:
        assert conn.execute(
            "SELECT relation FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id='workrun-one'",
            (second_turn_id,),
        ).fetchone()[0] == "continued"

    replayed = work_run_store.resume_active_work_run_attempt(
        session_id=session_id,
        turn_id=second_turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-rebind",
        expected_work_run_revision=started.work_run_revision,
        expected_window_revision=resume_window_revision,
        apply_id="resume-active-attempt",
    )
    assert replayed.status == "replayed"
    assert replayed.work_run_revision == resumed.work_run_revision
    assert _window_revision(session_id) == resumed.window_state_version

    marked_again = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=second_turn_id,
        expected_window_revision=resumed.window_state_version,
        stage="L2_PLAN",
        interruption_reason="process_lost_again",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=second_turn_id,
        expected_window_revision=int(marked_again["state_version"]),
        end_reason="process_lost_again",
        error_code="PROCESS_LOST",
    )
    accepted_again = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="resume-active-attempt-again",
        source="runtime_test",
        user_text="再次继续",
        lease_owner="work-execution-test",
    )
    third_turn_id = str(accepted_again["turn"]["turn_id"])
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (session_id, third_turn_id, subject.task_id, "2026-08-14T00:32:00+00:00"),
        )
    resumed_again = work_run_store.resume_active_work_run_attempt(
        session_id=session_id,
        turn_id=third_turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-rebind",
        expected_work_run_revision=resumed.work_run_revision,
        expected_window_revision=_window_revision(session_id),
        apply_id="resume-active-attempt-again",
    )
    after_second_rebind = _get(session_id)
    assert resumed_again.work_run_revision == resumed.work_run_revision + 1
    assert after_second_rebind.work_run.budget.attempts_started == 1
    assert after_second_rebind.work_run.budget.active_seconds_consumed == 0
    assert after_second_rebind.related_turn_ids == (
        first_turn_id,
        second_turn_id,
        third_turn_id,
    )
    assert after_second_rebind.attempts[0].turn_id == third_turn_id
    assert after_second_rebind.attempts[0].input_turn_id == first_turn_id

    settled = work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=third_turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-rebind",
        decision=HostAcceptedAttemptDecision(
            acceptance_updates=(),
            action=RequestUserInputAction(question="请补充信息"),
        ),
        expected_work_run_revision=resumed_again.work_run_revision,
        expected_progress_revision=resumed_again.acceptance_progress_revision,
        expected_window_revision=resumed_again.window_state_version,
        apply_id="settle-resumed-attempt",
        active_seconds_delta=2.5,
    )
    assert settled.work_run_status.value == "waiting_user"
    # 两段丢失进程的尾部都已丢弃；只有最终恢复后的区间进入持久聚合。
    assert _get(session_id).work_run.budget.active_seconds_consumed == 2.5


def test_active_attempt_rebind_rejects_a_decided_attempt_without_writes():
    session_id, first_turn_id, subject = _seed_task_node()
    _create(session_id, first_turn_id, subject)
    started = _start(
        session_id,
        first_turn_id,
        "workrun-one",
        expected_run_revision=1,
        apply_id="start-decided-rebind",
        attempt_id="attempt-decided-rebind",
    )
    decided = work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=first_turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-decided-rebind",
        decision=_call_decision(call_id="decided-rebind-call"),
        expected_work_run_revision=started.work_run_revision,
        expected_progress_revision=started.acceptance_progress_revision,
        expected_window_revision=started.window_state_version,
        apply_id="decide-before-rebind",
    )
    marked = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=decided.window_state_version,
        stage="TOOL_EXECUTION",
        interruption_reason="process_lost",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=int(marked["state_version"]),
        end_reason="process_lost",
        error_code="PROCESS_LOST",
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="reject-decided-attempt-resume",
        source="runtime_test",
        user_text="继续",
        lease_owner="work-execution-test",
    )
    second_turn_id = str(accepted["turn"]["turn_id"])
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (session_id, second_turn_id, subject.task_id, "2026-08-14T00:31:00+00:00"),
        )

    before_window = _window_revision(session_id)
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="undecided active Attempt",
    ):
        work_run_store.resume_active_work_run_attempt(
            session_id=session_id,
            turn_id=second_turn_id,
            work_run_id="workrun-one",
            attempt_id="attempt-decided-rebind",
            expected_work_run_revision=decided.work_run_revision,
            expected_window_revision=before_window,
            apply_id="must-not-resume-decided",
        )
    assert _window_revision(session_id) == before_window
    assert store.get_turn_execution_window(session_id)["current_work_run_id"] is None
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id='workrun-one'",
            (second_turn_id,),
        ).fetchone()[0] == 0


def test_decided_readonly_tool_attempt_rebind_is_exact_replayable_and_preserves_input():
    (
        session_id,
        first_turn_id,
        second_turn_id,
        _subject,
        catalog,
        cursor,
    ) = _seed_decided_tool_recovery(result_count=1)
    before_window = _window_revision(session_id)

    resumed = work_run_store.resume_decided_readonly_tool_attempt(
        session_id=session_id,
        turn_id=second_turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-decided-tool-recovery",
        expected_work_run_revision=cursor.work_run_revision,
        expected_window_revision=before_window,
        catalog_snapshot=catalog,
        apply_id="resume-decided-tool-recovery",
    )

    assert resumed.status == "applied"
    assert resumed.work_run_revision == cursor.work_run_revision + 1
    assert resumed.current_attempt_id == "attempt-decided-tool-recovery"
    assert resumed.acceptance_progress_revision == cursor.acceptance_progress_revision
    record = _get(session_id)
    assert len(record.attempts) == 1
    assert record.attempts[0].turn_id == second_turn_id
    assert record.attempts[0].input_turn_id == first_turn_id
    assert record.attempts[0].attempt.status.value == "active"
    assert len(record.tool_calls) == 2
    assert len(record.tool_results) == 1
    assert record.work_run.budget.active_seconds_consumed == 0
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert window["stage"] == "TOOL"
    assert window["current_work_run_id"] == "workrun-one"
    assert window["current_attempt_id"] == "attempt-decided-tool-recovery"

    replayed = work_run_store.resume_decided_readonly_tool_attempt(
        session_id=session_id,
        turn_id=second_turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-decided-tool-recovery",
        expected_work_run_revision=cursor.work_run_revision,
        expected_window_revision=before_window,
        catalog_snapshot=catalog,
        apply_id="resume-decided-tool-recovery",
    )
    assert replayed.status == "replayed"
    assert replayed.work_run_revision == resumed.work_run_revision
    assert _window_revision(session_id) == resumed.window_state_version

    with pytest.raises(work_run_store.WorkExecutionApplyIdCollision):
        work_run_store.resume_decided_readonly_tool_attempt(
            session_id=session_id,
            turn_id=second_turn_id,
            work_run_id="workrun-one",
            attempt_id="attempt-decided-tool-recovery",
            expected_work_run_revision=cursor.work_run_revision + 1,
            expected_window_revision=before_window,
            catalog_snapshot=catalog,
            apply_id="resume-decided-tool-recovery",
        )
    assert _window_revision(session_id) == resumed.window_state_version


@pytest.mark.parametrize(
    ("case", "seed_kwargs", "catalog_mutation", "revision_delta", "message"),
    [
        (
            "catalog_descriptor_drift",
            {},
            lambda catalog: {
                **catalog,
                "revision": int(catalog["revision"]) + 1,
            },
            0,
            "catalog snapshot drifted",
        ),
        (
            "effectful_catalog",
            {"catalog_action": "update", "modifies_environment": True, "result_count": 0},
            lambda catalog: catalog,
            0,
            "non-modifying ToolCalls",
        ),
        (
            "completion_unconfirmed",
            {
                "catalog_action": "update",
                "modifies_environment": True,
                "result_count": 1,
                "result_status": ToolResultStatus.COMPLETION_UNCONFIRMED,
            },
            lambda catalog: catalog,
            0,
            "non-modifying ToolCalls",
        ),
        (
            "stale_work_run_revision",
            {},
            lambda catalog: catalog,
            -1,
            "revision conflict",
        ),
    ],
)
def test_decided_tool_rebind_rejects_drift_effects_uncertainty_and_stale_revision(
    case: str,
    seed_kwargs: dict[str, object],
    catalog_mutation,
    revision_delta: int,
    message: str,
) -> None:
    session_id, _first_turn_id, second_turn_id, _subject, catalog, cursor = (
        _seed_decided_tool_recovery(**seed_kwargs)
    )
    before_window = _window_revision(session_id)
    before = _get(session_id)

    with pytest.raises(work_run_store.WorkExecutionPersistenceError, match=message):
        work_run_store.resume_decided_readonly_tool_attempt(
            session_id=session_id,
            turn_id=second_turn_id,
            work_run_id="workrun-one",
            attempt_id="attempt-decided-tool-recovery",
            expected_work_run_revision=cursor.work_run_revision + revision_delta,
            expected_window_revision=before_window,
            catalog_snapshot=catalog_mutation(catalog),
            apply_id=f"reject-decided-tool-{case}",
        )

    assert _get(session_id) == before
    assert _window_revision(session_id) == before_window
    assert store.get_turn_execution_window(session_id)["current_work_run_id"] is None


@pytest.mark.parametrize(
    ("result_count", "result_status"),
    [
        (0, ToolResultStatus.SUCCEEDED),
        (1, ToolResultStatus.COMPLETION_UNCONFIRMED),
    ],
)
def test_decided_protected_tool_rebind_requires_explicit_recovery_authority(
    result_count: int,
    result_status: ToolResultStatus,
) -> None:
    session_id, first_turn_id, second_turn_id, _subject, catalog, cursor = (
        _seed_decided_tool_recovery(
            catalog_action="update",
            modifies_environment=True,
            result_count=result_count,
            result_status=result_status,
        )
    )
    before_window = _window_revision(session_id)

    resumed = work_run_store.resume_decided_readonly_tool_attempt(
        session_id=session_id,
        turn_id=second_turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-decided-tool-recovery",
        expected_work_run_revision=cursor.work_run_revision,
        expected_window_revision=before_window,
        catalog_snapshot=catalog,
        apply_id=f"resume-protected-{result_count}",
        allow_protected_recovery=True,
    )

    assert resumed.status == "applied"
    record = _get(session_id)
    assert record.attempts[0].turn_id == second_turn_id
    assert record.attempts[0].input_turn_id == first_turn_id
    assert record.current_attempt_id == "attempt-decided-tool-recovery"
    assert len(record.tool_results) == result_count


def test_decided_readonly_tool_rebind_rejects_a_turn_linked_only_to_another_task():
    session_id, _first_turn_id, second_turn_id, _subject, catalog, cursor = (
        _seed_decided_tool_recovery(result_count=0, cross_task_link=True)
    )
    before_window = _window_revision(session_id)
    before = _get(session_id)

    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="not linked to this Task",
    ):
        work_run_store.resume_decided_readonly_tool_attempt(
            session_id=session_id,
            turn_id=second_turn_id,
            work_run_id="workrun-one",
            attempt_id="attempt-decided-tool-recovery",
            expected_work_run_revision=cursor.work_run_revision,
            expected_window_revision=before_window,
            catalog_snapshot=catalog,
            apply_id="reject-cross-task-decided-tool-resume",
        )

    assert _get(session_id) == before
    assert _window_revision(session_id) == before_window


@pytest.mark.parametrize(
    "tamper",
    ("attempt_budget_after", "close_charge_operation", "close_receipt_revision"),
)
def test_idle_readonly_resume_revalidates_close_budget_authority_before_writes(
    tamper: str,
) -> None:
    session_id, _first_turn_id, second_turn_id, _subject, catalog, cursor = (
        _seed_decided_tool_recovery(result_count=2, close_batch=True)
    )
    before_window = _window_revision(session_id)
    with store._connect() as conn:
        if tamper == "attempt_budget_after":
            row = conn.execute(
                "SELECT budget_after_json FROM insession_work_run_attempts "
                "WHERE attempt_id='attempt-decided-tool-recovery'"
            ).fetchone()
            payload = json.loads(str(row[0]))
            payload["active_seconds_consumed"] = 2.0
            conn.execute(
                "UPDATE insession_work_run_attempts SET budget_after_json=? "
                "WHERE attempt_id='attempt-decided-tool-recovery'",
                (json.dumps(payload, separators=(",", ":"), sort_keys=True),),
            )
        elif tamper == "close_charge_operation":
            conn.execute(
                "UPDATE insession_work_run_budget_charges "
                "SET operation='commit_output_action' "
                "WHERE budget_charge_id='close-decided-tool-recovery'"
            )
        else:
            row = conn.execute(
                "SELECT result_json FROM insession_work_run_apply_receipts "
                "WHERE apply_id='close-decided-tool-recovery'"
            ).fetchone()
            payload = json.loads(str(row[0]))
            payload["work_run_revision"] = int(payload["work_run_revision"]) + 1
            conn.execute(
                "UPDATE insession_work_run_apply_receipts SET result_json=? "
                "WHERE apply_id='close-decided-tool-recovery'",
                (json.dumps(payload, separators=(",", ":"), sort_keys=True),),
            )
        run_before = conn.execute(
            "SELECT revision, attempts_started, current_attempt_id "
            "FROM insession_work_runs WHERE work_run_id='workrun-one'"
        ).fetchone()
        attempts_before = conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_attempts "
            "WHERE work_run_id='workrun-one'"
        ).fetchone()[0]

    with pytest.raises(work_run_store.WorkExecutionPersistenceError):
        work_run_store.resume_idle_readonly_work_run_and_start_attempt(
            session_id=session_id,
            turn_id=second_turn_id,
            work_run_id="workrun-one",
            closed_attempt_id="attempt-decided-tool-recovery",
            next_attempt_id="attempt-after-idle-close",
            expected_work_run_revision=cursor.work_run_revision,
            expected_window_revision=before_window,
            catalog_snapshot=catalog,
            apply_id=f"reject-idle-close-tamper-{tamper}",
        )

    assert _window_revision(session_id) == before_window
    with store._connect() as conn:
        run_after = conn.execute(
            "SELECT revision, attempts_started, current_attempt_id "
            "FROM insession_work_runs WHERE work_run_id='workrun-one'"
        ).fetchone()
        assert tuple(run_after) == tuple(run_before)
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_attempts "
            "WHERE work_run_id='workrun-one'"
        ).fetchone()[0] == attempts_before
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id='workrun-one'",
            (second_turn_id,),
        ).fetchone()[0] == 0


def test_tool_result_chain_supports_next_attempt_progress_and_preserves_ordinals():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    _start(session_id, turn_id, "workrun-one", expected_run_revision=1, apply_id="start-one", attempt_id="attempt-one")
    decision_one = work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-one",
        decision=_call_decision(),
        expected_work_run_revision=2,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="decision-one",
    )
    assert decision_one.work_run_revision == 3
    assert decision_one.acceptance_progress_revision == 1
    assert decision_one.new_tool_call_ids == ("call-one",)

    tool_result = ToolResult(
        status=ToolResultStatus.SUCCEEDED,
        tool_result_id="result-one",
        tool_call_id="call-one",
        attempt_id="attempt-one",
        ordinal=1,
        output={"value": "observed"},
    )
    appended = work_run_store.append_work_run_tool_result(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        result=tool_result,
        expected_work_run_revision=3,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="append-one",
    )
    assert appended.tool_result_id == "result-one"
    assert _get(session_id).tool_results == (tool_result,)
    appended_window = _window_revision(session_id)
    replayed = work_run_store.append_work_run_tool_result(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        result=tool_result,
        expected_work_run_revision=3,
        expected_progress_revision=1,
        expected_window_revision=appended_window - 1,
        apply_id="append-one",
    )
    assert replayed.status == "replayed"
    assert _window_revision(session_id) == appended_window

    closed = work_run_store.close_work_run_attempt(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-one",
        expected_work_run_revision=4,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="close-one",
    )
    assert closed.attempt is not None and closed.attempt.status.value == "closed"
    assert closed.current_attempt_id is None

    _start(session_id, turn_id, "workrun-one", expected_run_revision=5, apply_id="start-two", attempt_id="attempt-two")
    next_decision = HostAcceptedAttemptDecision(
        acceptance_updates=(
            AcceptanceUpdate(
                acceptance_id="deliverable",
                model_claimed_satisfied=True,
                supporting_tool_result_ids=("result-one",),
            ),
        ),
        action=RequestUserInputAction(question="请补充质量要求"),
    )
    committed = work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-two",
        decision=next_decision,
        expected_work_run_revision=6,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="decision-two",
        active_seconds_delta=1,
    )
    assert committed.work_run_status.value == "waiting_user"
    assert committed.work_run_reason == "needs_input"
    record = _get(session_id)
    assert record.work_run.reason == "needs_input"
    assert record.pending_user_question == "请补充质量要求"
    assert isinstance(record.attempts[-1].decision.action, RequestUserInputAction)
    assert record.acceptance_progress.revision == 2
    assert record.acceptance_progress.items[0].supporting_tool_result_ids == (
        "result-one",
    )
    assert record.tool_results[0].ordinal == 1
    with store._connect() as conn:
        task_state = conn.execute(
            "SELECT current_status FROM insession_tasks WHERE insession_task_id=?",
            (subject.task_id,),
        ).fetchone()[0]
        node_state = conn.execute(
            "SELECT status FROM insession_task_node_states "
            "WHERE insession_task_id=? AND insession_task_node_id=?",
            (subject.task_id, subject.node_id),
        ).fetchone()[0]
    assert (task_state, node_state) == ("awaiting_user", "awaiting_user")


def _seed_current_attempt_with_historical_support_candidate(
    *,
    request_id: str,
):
    session_id, turn_id, subject = _seed_task_node(request_id=request_id)
    _create(session_id, turn_id, subject)
    started = _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=1,
        apply_id=f"{request_id}-start-one",
        attempt_id="attempt-one",
    )
    decided = work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-one",
        decision=_call_decision(modifies_environment=True),
        expected_work_run_revision=started.work_run_revision,
        expected_progress_revision=started.acceptance_progress_revision,
        expected_window_revision=started.window_state_version,
        apply_id=f"{request_id}-decision-one",
    )
    appended = work_run_store.append_work_run_tool_result(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        result=ToolResult(
            status=ToolResultStatus.SUCCEEDED,
            tool_result_id="result-one",
            tool_call_id="call-one",
            attempt_id="attempt-one",
            ordinal=1,
            output={"value": "observed"},
        ),
        expected_work_run_revision=decided.work_run_revision,
        expected_progress_revision=decided.acceptance_progress_revision,
        expected_window_revision=decided.window_state_version,
        apply_id=f"{request_id}-append-one",
    )
    closed = work_run_store.close_work_run_attempt(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-one",
        expected_work_run_revision=appended.work_run_revision,
        expected_progress_revision=appended.acceptance_progress_revision,
        expected_window_revision=appended.window_state_version,
        apply_id=f"{request_id}-close-one",
    )
    current = _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=closed.work_run_revision,
        expected_progress_revision=closed.acceptance_progress_revision,
        apply_id=f"{request_id}-start-two",
        attempt_id="attempt-two",
    )
    return session_id, turn_id, subject, current


def _replace_historical_result_status(
    session_id: str,
    *,
    status: ToolResultStatus,
) -> None:
    replacement = ToolResult(
        status=status,
        tool_result_id="result-one",
        tool_call_id="call-one",
        attempt_id="attempt-one",
        ordinal=1,
        output=None,
        error_code="tool_did_not_succeed",
        error_message="The tool did not produce successful evidence.",
    )
    with store._connect() as conn:
        changed = conn.execute(
            "UPDATE insession_work_run_tool_results SET status=?, result_json=? "
            "WHERE work_run_id='workrun-one' AND tool_result_id='result-one' "
            "AND EXISTS (SELECT 1 FROM insession_work_runs AS run "
            "WHERE run.work_run_id='workrun-one' AND run.session_id=?)",
            (status.value, replacement.model_dump_json(), session_id),
        )
        assert changed.rowcount == 1


@pytest.mark.parametrize(
    "status",
    (
        ToolResultStatus.REJECTED,
        ToolResultStatus.FAILED,
        ToolResultStatus.TIMED_OUT,
        ToolResultStatus.CANCELLED,
        ToolResultStatus.COMPLETION_UNCONFIRMED,
    ),
)
def test_store_rejects_non_succeeded_historical_result_as_acceptance_support(
    status: ToolResultStatus,
) -> None:
    session_id, turn_id, _subject, current = (
        _seed_current_attempt_with_historical_support_candidate(
            request_id=f"reject-{status.value}-support",
        )
    )
    _replace_historical_result_status(session_id, status=status)
    before = _get(session_id)
    before_window = _window_revision(session_id)

    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="unknown_supporting_tool_result_id",
    ):
        work_run_store.commit_work_run_attempt_decision(
            session_id=session_id,
            turn_id=turn_id,
            work_run_id="workrun-one",
            attempt_id="attempt-two",
            decision=HostAcceptedAttemptDecision(
                acceptance_updates=(
                    AcceptanceUpdate(
                        acceptance_id="deliverable",
                        model_claimed_satisfied=True,
                        supporting_tool_result_ids=("result-one",),
                    ),
                ),
                action=RequestUserInputAction(question="请补充质量要求"),
            ),
            expected_work_run_revision=current.work_run_revision,
            expected_progress_revision=current.acceptance_progress_revision,
            expected_window_revision=current.window_state_version,
            apply_id=f"reject-{status.value}-support-decision",
            active_seconds_delta=1,
        )

    assert _get(session_id) == before
    assert _window_revision(session_id) == before_window


@pytest.mark.parametrize(
    "status",
    (
        ToolResultStatus.REJECTED,
        ToolResultStatus.FAILED,
        ToolResultStatus.TIMED_OUT,
        ToolResultStatus.CANCELLED,
        ToolResultStatus.COMPLETION_UNCONFIRMED,
    ),
)
def test_verification_prepare_rejects_non_succeeded_supporting_result(
    status: ToolResultStatus,
) -> None:
    session_id, turn_id, _subject, current = (
        _seed_current_attempt_with_historical_support_candidate(
            request_id=f"verify-{status.value}-support",
        )
    )
    submitted = work_run_store.commit_work_run_output_action(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-two",
        decision=_output_decision(
            content="candidate supported only by a successful result",
            submit=True,
            updates=(
                AcceptanceUpdate(
                    acceptance_id="deliverable",
                    model_claimed_satisfied=True,
                    supporting_tool_result_ids=("result-one",),
                ),
                AcceptanceUpdate(
                    acceptance_id="quality",
                    model_claimed_satisfied=True,
                ),
            ),
        ),
        expected_work_run_revision=current.work_run_revision,
        expected_progress_revision=current.acceptance_progress_revision,
        expected_output_revision=current.output_window_revision,
        expected_window_revision=current.window_state_version,
        apply_id=f"verify-{status.value}-support-submit",
    )
    _replace_historical_result_status(session_id, status=status)
    before_window = _window_revision(session_id)

    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="supporting ToolResult authority is corrupt",
    ):
        verification_store.prepare_task_node_verification(
            session_id=session_id,
            turn_id=turn_id,
            work_run_id="workrun-one",
            expected_work_run_revision=submitted.work_run_revision,
            expected_progress_revision=submitted.acceptance_progress_revision,
            expected_output_revision=submitted.output_window_revision,
            expected_window_revision=submitted.window_state_version,
            apply_id=f"verify-{status.value}-support-prepare",
            verification_request_id=f"verify-{status.value}-support-request",
        )

    assert _window_revision(session_id) == before_window
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_verification_requests "
            "WHERE verification_request_id=?",
            (f"verify-{status.value}-support-request",),
        ).fetchone()[0] == 0


def test_pending_question_projection_and_atomic_continuation_are_exactly_replayable():
    session_id, question_turn_id, subject, waiting = _seed_waiting_user_question()

    pending = continuation_store.list_pending_user_questions(session_id=session_id)
    assert len(pending) == 1
    question = pending[0]
    assert question.work_run_id == "workrun-one"
    assert question.work_run_revision == waiting.work_run_revision
    assert question.subject == subject
    assert question.question_attempt_id == "question-attempt"
    assert question.question_attempt_ordinal == 1
    assert question.question_turn_id == question_turn_id
    assert question.question == "请提供旅行日期。"
    assert question.task_state_version == 3
    assert question.node_state_version == 3
    assert question.acceptance_progress_revision == 1
    assert question.output_window_revision == 1

    answer_turn_id = _accept_answer_turn(
        session_id,
        question_turn_id,
        subject,
    )
    before_window_revision = _window_revision(session_id)
    continued = continuation_store.continue_waiting_user_work_run_and_start_attempt(
        session_id=session_id,
        turn_id=answer_turn_id,
        work_run_id="workrun-one",
        subject=subject,
        question_attempt_id=question.question_attempt_id,
        expected_work_run_revision=question.work_run_revision,
        expected_task_state_version=question.task_state_version,
        expected_node_state_version=question.node_state_version,
        expected_progress_revision=question.acceptance_progress_revision,
        expected_window_revision=before_window_revision,
        apply_id="continue-question",
        catalog_snapshot={"revision": 2, "tools": ["read_test"]},
        attempt_id="answer-attempt",
    )

    assert continued.status == "applied"
    assert continued.work_run_revision == question.work_run_revision + 1
    assert continued.current_attempt_id == "answer-attempt"
    assert continued.attempt is not None
    assert continued.attempt.ordinal == 2
    assert continued.attempt.status.value == "active"
    assert continued.turn_work_run_link_revision == 1
    record = _get(session_id)
    assert record.work_run.status.value == "active"
    assert record.work_run.reason is None
    assert record.work_run.budget.attempts_started == 2
    assert record.work_run.budget.active_seconds_consumed == 1
    assert record.attempts[0].input_turn_id == question_turn_id
    assert record.attempts[0].predecessor_question_attempt_id is None
    assert record.attempts[1].turn_id == answer_turn_id
    assert record.attempts[1].input_turn_id == answer_turn_id
    assert (
        record.attempts[1].predecessor_question_attempt_id
        == "question-attempt"
    )
    assert record.attempts[1].input_checkpoint_id == "question-attempt"
    assert continuation_store.list_pending_user_questions(session_id=session_id) == ()
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert window["current_work_run_id"] == "workrun-one"
    assert window["current_attempt_id"] == "answer-attempt"
    assert window["latest_checkpoint_id"] == "question-attempt"
    assert window["stage"] == "L2_PLAN"
    with store._connect() as conn:
        task_state = conn.execute(
            "SELECT current_status, state_version FROM insession_tasks "
            "WHERE insession_task_id=?",
            (subject.task_id,),
        ).fetchone()
        node_state = conn.execute(
            "SELECT status, state_version FROM insession_task_node_states "
            "WHERE insession_task_id=? AND insession_task_node_id=?",
            (subject.task_id, subject.node_id),
        ).fetchone()
        link = conn.execute(
            "SELECT relation FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id='workrun-one'",
            (answer_turn_id,),
        ).fetchone()
        receipt_json = str(
            conn.execute(
                "SELECT result_json FROM insession_work_run_apply_receipts "
                "WHERE apply_id='continue-question'"
            ).fetchone()[0]
        )
    assert tuple(task_state) == ("active", 4)
    assert tuple(node_state) == ("active", 4)
    assert tuple(link) == ("continued",)
    assert "请提供旅行日期" not in receipt_json
    assert "九月十日" not in receipt_json

    replayed = continuation_store.continue_waiting_user_work_run_and_start_attempt(
        session_id=session_id,
        turn_id=answer_turn_id,
        work_run_id="workrun-one",
        subject=subject,
        question_attempt_id=question.question_attempt_id,
        expected_work_run_revision=question.work_run_revision,
        expected_task_state_version=question.task_state_version,
        expected_node_state_version=question.node_state_version,
        expected_progress_revision=question.acceptance_progress_revision,
        expected_window_revision=before_window_revision,
        apply_id="continue-question",
        catalog_snapshot={"revision": 2, "tools": ["read_test"]},
        attempt_id="answer-attempt",
    )
    assert replayed.status == "replayed"
    assert replayed.model_copy(update={"status": "applied"}) == continued
    assert _get(session_id).work_run.budget.attempts_started == 2

    with pytest.raises(work_run_store.WorkExecutionApplyIdCollision):
        continuation_store.continue_waiting_user_work_run_and_start_attempt(
            session_id=session_id,
            turn_id=answer_turn_id,
            work_run_id="workrun-one",
            subject=subject,
            question_attempt_id=question.question_attempt_id,
            expected_work_run_revision=question.work_run_revision,
            expected_task_state_version=question.task_state_version,
            expected_node_state_version=question.node_state_version,
            expected_progress_revision=question.acceptance_progress_revision,
            expected_window_revision=before_window_revision,
            apply_id="continue-question",
            catalog_snapshot={"revision": 3, "tools": []},
            attempt_id="answer-attempt",
        )


def test_waiting_user_continuation_late_failure_rolls_back_consumption_and_bindings(
    monkeypatch,
):
    session_id, question_turn_id, subject, _waiting = _seed_waiting_user_question()
    question = continuation_store.list_pending_user_questions(session_id=session_id)[0]
    answer_turn_id = _accept_answer_turn(
        session_id,
        question_turn_id,
        subject,
        client_request_id="answer-question-rollback",
    )
    before_window_revision = _window_revision(session_id)

    def _fail_receipt(*_args, **_kwargs):
        raise sqlite3.OperationalError("injected continuation receipt failure")

    monkeypatch.setattr(work_execution_records, "_insert_receipt", _fail_receipt)
    with pytest.raises(sqlite3.OperationalError, match="continuation receipt"):
        continuation_store.continue_waiting_user_work_run_and_start_attempt(
            session_id=session_id,
            turn_id=answer_turn_id,
            work_run_id="workrun-one",
            subject=subject,
            question_attempt_id=question.question_attempt_id,
            expected_work_run_revision=question.work_run_revision,
            expected_task_state_version=question.task_state_version,
            expected_node_state_version=question.node_state_version,
            expected_progress_revision=question.acceptance_progress_revision,
            expected_window_revision=before_window_revision,
            apply_id="continue-question-rollback",
            catalog_snapshot={"revision": 2, "tools": []},
            attempt_id="answer-attempt-rollback",
        )

    record = _get(session_id)
    assert record.work_run.status.value == "waiting_user"
    assert record.work_run.revision == question.work_run_revision
    assert record.work_run.budget.attempts_started == 1
    assert len(record.attempts) == 1
    assert continuation_store.list_pending_user_questions(session_id=session_id) == (question,)
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert window["state_version"] == before_window_revision
    assert window["current_work_run_id"] is None
    assert window["current_attempt_id"] is None
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id='workrun-one'",
            (answer_turn_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT current_status FROM insession_tasks "
            "WHERE insession_task_id=?",
            (subject.task_id,),
        ).fetchone()[0] == "awaiting_user"
        assert conn.execute(
            "SELECT status FROM insession_task_node_states "
            "WHERE insession_task_id=? AND insession_task_node_id=?",
            (subject.task_id, subject.node_id),
        ).fetchone()[0] == "awaiting_user"


def test_waiting_user_continuation_requires_task_link_exact_question_and_budget():
    session_id, question_turn_id, subject, _waiting = _seed_waiting_user_question()
    question = continuation_store.list_pending_user_questions(session_id=session_id)[0]
    answer_turn_id = _accept_answer_turn(
        session_id,
        question_turn_id,
        subject,
        client_request_id="answer-question-guards",
    )
    with store._connect() as conn:
        conn.execute(
            "DELETE FROM insession_task_turn_links WHERE session_id=? AND turn_id=?",
            (session_id, answer_turn_id),
        )
    before_window_revision = _window_revision(session_id)
    common = {
        "session_id": session_id,
        "turn_id": answer_turn_id,
        "work_run_id": "workrun-one",
        "subject": subject,
        "expected_work_run_revision": question.work_run_revision,
        "expected_task_state_version": question.task_state_version,
        "expected_node_state_version": question.node_state_version,
        "expected_progress_revision": question.acceptance_progress_revision,
        "expected_window_revision": before_window_revision,
        "catalog_snapshot": {"revision": 2, "tools": []},
        "attempt_id": "guard-answer-attempt",
    }
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="not authoritatively linked",
    ):
        continuation_store.continue_waiting_user_work_run_and_start_attempt(
            **common,
            question_attempt_id=question.question_attempt_id,
            apply_id="missing-task-link",
        )
    assert len(_get(session_id).attempts) == 1
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (
                session_id,
                answer_turn_id,
                subject.task_id,
                "2026-08-14T00:41:00+00:00",
            ),
        )
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="exact current pending interaction",
    ):
        continuation_store.continue_waiting_user_work_run_and_start_attempt(
            **common,
            question_attempt_id="stale-question-attempt",
            apply_id="stale-question",
        )
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_runs SET active_seconds_consumed=720 "
            "WHERE work_run_id='workrun-one'"
        )
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="soft active-time budget",
    ):
        continuation_store.continue_waiting_user_work_run_and_start_attempt(
            **common,
            question_attempt_id=question.question_attempt_id,
            apply_id="exhausted-question",
        )
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_attempts "
            "WHERE work_run_id='workrun-one'"
        ).fetchone()[0] == 1
    assert _window_revision(session_id) == before_window_revision


def test_waiting_user_continuation_records_are_removed_by_session_purge():
    session_id, question_turn_id, subject, _waiting = _seed_waiting_user_question()
    question = continuation_store.list_pending_user_questions(session_id=session_id)[0]
    answer_turn_id = _accept_answer_turn(
        session_id,
        question_turn_id,
        subject,
        client_request_id="answer-question-purge",
    )
    continuation_store.continue_waiting_user_work_run_and_start_attempt(
        session_id=session_id,
        turn_id=answer_turn_id,
        work_run_id="workrun-one",
        subject=subject,
        question_attempt_id=question.question_attempt_id,
        expected_work_run_revision=question.work_run_revision,
        expected_task_state_version=question.task_state_version,
        expected_node_state_version=question.node_state_version,
        expected_progress_revision=question.acceptance_progress_revision,
        expected_window_revision=_window_revision(session_id),
        apply_id="continue-question-purge",
        catalog_snapshot={"revision": 2, "tools": []},
        attempt_id="answer-attempt-purge",
    )

    assert store.purge_session(session_id) is True
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_attempts "
            "WHERE attempt_id IN ('question-attempt', 'answer-attempt-purge')"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_apply_receipts "
            "WHERE apply_id='continue-question-purge'"
        ).fetchone()[0] == 0


def test_close_requires_every_call_result_and_rolls_back_without_partial_close():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    _start(session_id, turn_id, "workrun-one", expected_run_revision=1, apply_id="start-one", attempt_id="attempt-one")
    decision = HostAcceptedAttemptDecision(
        action=HostMaterializedCallToolsAction(
            calls=(
                HostMaterializedToolCall(
                    tool_call_id="call-one",
                    tool_id="read_test",
                    tool_version="1.0.0",
                    arguments={"value": 1},
                    modifies_environment=False,
                ),
                HostMaterializedToolCall(
                    tool_call_id="call-two",
                    tool_id="read_test",
                    tool_version="1.0.0",
                    arguments={"value": 2},
                    modifies_environment=False,
                ),
            )
        )
    )
    work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-one",
        decision=decision,
        expected_work_run_revision=2,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="decision-two-calls",
    )
    work_run_store.append_work_run_tool_result(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        result=ToolResult(
            status=ToolResultStatus.SUCCEEDED,
            tool_result_id="result-one",
            tool_call_id="call-one",
            attempt_id="attempt-one",
            ordinal=1,
            output={"value": 1},
        ),
        expected_work_run_revision=3,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="append-one",
    )
    before = _get(session_id)
    before_window = _window_revision(session_id)
    with pytest.raises(work_run_store.WorkExecutionPersistenceError, match="all materialized"):
        work_run_store.close_work_run_attempt(
            active_seconds_delta=1,
            session_id=session_id,
            turn_id=turn_id,
            work_run_id="workrun-one",
            attempt_id="attempt-one",
            expected_work_run_revision=4,
            expected_progress_revision=1,
            expected_window_revision=before_window,
            apply_id="premature-close",
        )
    assert _get(session_id) == before
    assert _window_revision(session_id) == before_window
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_apply_receipts WHERE apply_id='premature-close'"
        ).fetchone()[0] == 0


def test_append_rejects_unknown_call_without_writes():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    _start(session_id, turn_id, "workrun-one", expected_run_revision=1, apply_id="start-one", attempt_id="attempt-one")
    work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-one",
        decision=_call_decision(),
        expected_work_run_revision=2,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="decision-one",
    )
    before_window = _window_revision(session_id)
    with pytest.raises(work_run_store.WorkExecutionPersistenceError, match="unknown materialized call"):
        work_run_store.append_work_run_tool_result(
            session_id=session_id,
            turn_id=turn_id,
            work_run_id="workrun-one",
            result=ToolResult(
                status=ToolResultStatus.FAILED,
                tool_result_id="bad-result",
                tool_call_id="other-call",
                attempt_id="attempt-one",
                ordinal=1,
                output=None,
                error_code="not_called",
                error_message="No such accepted call",
            ),
            expected_work_run_revision=3,
            expected_progress_revision=1,
            expected_window_revision=before_window,
            apply_id="bad-append",
        )
    assert _window_revision(session_id) == before_window
    assert _get(session_id).tool_results == ()


def test_submit_output_window_locks_the_run_until_a_verifier_resolves_it():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=1,
        apply_id="start-one",
        attempt_id="attempt-one",
    )
    committed = work_run_store.commit_work_run_output_action(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-one",
        decision=AttemptDecision(
            acceptance_updates=(
                AcceptanceUpdate(
                    acceptance_id="deliverable",
                    model_claimed_satisfied=True,
                ),
                AcceptanceUpdate(
                    acceptance_id="quality",
                    model_claimed_satisfied=True,
                ),
            ),
            action=SubmitOutputWindowAction(
                content="verified candidate",
                format=OutputWindowFormat.PLAIN_TEXT,
            ),
        ),
        expected_work_run_revision=2,
        expected_progress_revision=1,
        expected_output_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="submit-one",
    )

    assert committed.work_run_status.value == "active"
    assert committed.work_run_reason == "verification_pending"
    assert committed.current_attempt_id is None
    assert committed.attempt is not None
    assert committed.attempt.status.value == "closed"
    assert committed.attempt.submitted_output_revision == 2
    record = _get(session_id)
    assert record.work_run.reason == "verification_pending"
    assert record.output_window.content == "verified candidate"
    assert record.pending_user_question is None

    before_window = _window_revision(session_id)
    with pytest.raises(work_run_store.WorkExecutionPersistenceError, match="phase"):
        _start(
            session_id,
            turn_id,
            "workrun-one",
            expected_run_revision=3,
            expected_progress_revision=2,
            apply_id="bypass-verifier",
            attempt_id="attempt-two",
        )
    assert _window_revision(session_id) == before_window
    assert _get(session_id).work_run.budget.attempts_started == 1


def test_prepare_and_get_task_node_verification_are_exact_and_body_single_owned():
    session_id, turn_id, subject = _seed_task_node()
    submitted = _submit_for_verification(session_id, turn_id, subject)
    assert submitted.work_run_revision == 3
    original_window_revision = _window_revision(session_id)

    prepared_mutation = verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_work_run_revision=3,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=original_window_revision,
        apply_id="prepare-verification-one",
        verification_request_id="verification-one",
    )

    assert prepared_mutation.status == "applied"
    assert prepared_mutation.verification_request_id == "verification-one"
    assert prepared_mutation.verification_request_revision == 1
    assert prepared_mutation.verification_request_status.value == "pending"
    assert prepared_mutation.work_run_revision == 4
    assert prepared_mutation.work_run_reason == "verification_pending"
    assert prepared_mutation.window_state_version == original_window_revision + 1
    prepared_window = store.get_turn_execution_window(session_id)
    assert prepared_window is not None
    assert prepared_window["stage"] == "VERIFICATION"
    assert prepared_window["latest_checkpoint_id"] == "verification-one"

    record = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id="verification-one",
    )
    assert record.result is None
    assert record.request.request_turn_id == turn_id
    assert record.request.work_run_id == "workrun-one"
    assert record.request.subject == subject
    assert record.request.submitted_attempt_id == "submit-attempt-workrun-one"
    assert record.request.output_revision == 2
    assert record.request.acceptance_progress_revision == 2
    assert record.request.acceptance_ids == ("deliverable", "quality")
    assert record.request.locked_work_run_revision == 3

    prepared = verification_store.get_prepared_task_node_verification(
        session_id=session_id,
        invocation_turn_id=turn_id,
        verification_request_id="verification-one",
    )
    assert prepared.record == record
    assert prepared.work_run.revision == 4
    assert prepared.work_run.reason == "verification_pending"
    assert prepared.submitted_attempt.attempt_id == "submit-attempt-workrun-one"
    assert prepared.acceptance_progress.revision == 2
    assert prepared.locked_output_window.content == "verified candidate"
    assert [item.acceptance_id for item in prepared.acceptances] == [
        "deliverable",
        "quality",
    ]

    replay = verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_work_run_revision=3,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=original_window_revision,
        apply_id="prepare-verification-one",
        verification_request_id="verification-one",
    )
    assert replay.status == "replayed"
    assert replay.model_copy(update={"status": "applied"}) == prepared_mutation

    with pytest.raises(work_run_store.WorkExecutionApplyIdCollision):
        verification_store.prepare_task_node_verification(
            session_id=session_id,
            turn_id=turn_id,
            work_run_id="workrun-one",
            expected_work_run_revision=3,
            expected_progress_revision=2,
            expected_output_revision=2,
            expected_window_revision=original_window_revision,
            apply_id="prepare-verification-one",
            verification_request_id="different-verification-request",
        )

    with store._connect() as conn:
        request_row = conn.execute(
            "SELECT * FROM insession_work_run_verification_requests "
            "WHERE verification_request_id='verification-one'"
        ).fetchone()
        receipt_json = str(
            conn.execute(
                "SELECT result_json FROM insession_work_run_apply_receipts "
                "WHERE apply_id='prepare-verification-one'"
            ).fetchone()[0]
        )
    assert request_row is not None
    assert "verified candidate" not in "|".join(str(value) for value in request_row)
    assert "verified candidate" not in receipt_json
    assert "acceptance_results" not in receipt_json


def test_verification_replay_fails_closed_on_inconsistent_small_receipt():
    session_id, turn_id, subject = _seed_task_node()
    _submit_for_verification(session_id, turn_id, subject)
    original_window_revision = _window_revision(session_id)
    verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_work_run_revision=3,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=original_window_revision,
        apply_id="prepare-corrupt-receipt",
        verification_request_id="verification-corrupt-receipt",
    )
    with store._connect() as conn:
        row = conn.execute(
            "SELECT result_json FROM insession_work_run_apply_receipts "
            "WHERE apply_id='prepare-corrupt-receipt'"
        ).fetchone()
        payload = json.loads(str(row["result_json"]))
        payload["all_pass"] = True
        conn.execute(
            "UPDATE insession_work_run_apply_receipts SET result_json=? "
            "WHERE apply_id='prepare-corrupt-receipt'",
            (json.dumps(payload),),
        )

    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="stored verification apply receipt is corrupt",
    ):
        verification_store.prepare_task_node_verification(
            session_id=session_id,
            turn_id=turn_id,
            work_run_id="workrun-one",
            expected_work_run_revision=3,
            expected_progress_revision=2,
            expected_output_revision=2,
            expected_window_revision=original_window_revision,
            apply_id="prepare-corrupt-receipt",
            verification_request_id="verification-corrupt-receipt",
        )


def test_pending_verification_rejects_prompt_node_and_window_checkpoint_drift():
    session_id, turn_id, subject = _seed_task_node()
    _submit_for_verification(session_id, turn_id, subject)
    verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_work_run_revision=3,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=_window_revision(session_id),
        apply_id="prepare-drift-guards",
        verification_request_id="verification-drift-guards",
    )
    prepared = verification_store.get_prepared_task_node_verification(
        session_id=session_id,
        invocation_turn_id=turn_id,
        verification_request_id="verification-drift-guards",
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_task_graph_nodes SET title='tampered title' "
            "WHERE insession_task_id=? AND graph_revision=1 "
            "AND insession_task_node_id=?",
            (subject.task_id, subject.node_id),
        )
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="immutable binding has drifted",
    ):
        verification_store.get_task_node_verification_record(
            session_id=session_id,
            verification_request_id="verification-drift-guards",
        )

    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_task_graph_nodes SET title='测试节点' "
            "WHERE insession_task_id=? AND graph_revision=1 "
            "AND insession_task_node_id=?",
            (subject.task_id, subject.node_id),
        )
        conn.execute(
            "UPDATE turn_execution_windows SET latest_checkpoint_id='wrong-checkpoint' "
            "WHERE session_id=?",
            (session_id,),
        )
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="does not own the pending verification request",
    ):
        verification_store.get_prepared_task_node_verification(
            session_id=session_id,
            invocation_turn_id=turn_id,
            verification_request_id="verification-drift-guards",
        )
    before_window = _window_revision(session_id)
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="current verification checkpoint",
    ):
        verification_store.commit_task_node_verification_result(
            active_seconds_delta=1,
            session_id=session_id,
            turn_id=turn_id,
            work_run_id="workrun-one",
            verification_request_id="verification-drift-guards",
            result=_verification_result(prepared, all_pass=True),
            expected_work_run_revision=4,
            expected_verification_request_revision=1,
            expected_window_revision=before_window,
            apply_id="commit-after-checkpoint-drift",
            delivery_id="drift-delivery",
        )
    assert _get(session_id).work_run.revision == 4


def test_pending_verification_requires_the_task_node_to_remain_active():
    session_id, turn_id, subject = _seed_task_node()
    _submit_for_verification(session_id, turn_id, subject)
    verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_work_run_revision=3,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=_window_revision(session_id),
        apply_id="prepare-node-status",
        verification_request_id="verification-node-status",
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_task_node_states SET status='completed' "
            "WHERE insession_task_id=? AND insession_task_node_id=?",
            (subject.task_id, subject.node_id),
        )
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="TaskNode must be active",
    ):
        verification_store.get_prepared_task_node_verification(
            session_id=session_id,
            invocation_turn_id=turn_id,
            verification_request_id="verification-node-status",
        )
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="TaskNode must be active",
    ):
        verification_store.get_task_node_verification_record(
            session_id=session_id,
            verification_request_id="verification-node-status",
        )


@pytest.mark.parametrize(
    "corruption",
    ("binding_hash", "submit_revision", "node_status"),
)
def test_get_work_run_fails_closed_when_current_verification_authority_drifts(
    corruption: str,
):
    session_id, turn_id, subject = _seed_task_node()
    _submit_for_verification(session_id, turn_id, subject)
    verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_work_run_revision=3,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=_window_revision(session_id),
        apply_id="prepare-workrun-recovery-drift",
        verification_request_id="verification-workrun-recovery-drift",
    )

    with store._connect() as conn:
        if corruption == "binding_hash":
            conn.execute(
                "UPDATE insession_work_run_verification_requests "
                "SET request_binding_hash=? WHERE verification_request_id=?",
                ("b" * 64, "verification-workrun-recovery-drift"),
            )
        elif corruption == "submit_revision":
            conn.execute(
                "UPDATE insession_work_run_attempts SET committed_output_revision=1, "
                "submitted_output_revision=1 WHERE attempt_id='submit-attempt-workrun-one'"
            )
        else:
            conn.execute(
                "UPDATE insession_task_node_states SET status='interrupted' "
                "WHERE insession_task_id=? AND insession_task_node_id=?",
                (subject.task_id, subject.node_id),
            )

    with pytest.raises(work_run_store.WorkExecutionPersistenceError):
        _get(session_id)


def test_nonpass_verification_unlocks_run_and_next_attempt_consumes_exact_feedback():
    session_id, turn_id, subject = _seed_task_node()
    _submit_for_verification(session_id, turn_id, subject)
    verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_work_run_revision=3,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=_window_revision(session_id),
        apply_id="prepare-nonpass",
        verification_request_id="verification-nonpass",
    )
    prepared = verification_store.get_prepared_task_node_verification(
        session_id=session_id,
        invocation_turn_id=turn_id,
        verification_request_id="verification-nonpass",
    )
    result = _verification_result(prepared, all_pass=False)
    before_commit_window = _window_revision(session_id)

    committed = verification_store.commit_task_node_verification_result(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        verification_request_id="verification-nonpass",
        result=result,
        expected_work_run_revision=4,
        expected_verification_request_revision=1,
        expected_window_revision=before_commit_window,
        apply_id="commit-nonpass",
    )

    assert committed.verification_request_revision == 2
    assert committed.verification_request_status.value == "completed"
    assert committed.work_run_revision == 5
    assert committed.work_run_status.value == "active"
    assert committed.work_run_reason is None
    assert committed.all_pass is False
    assert committed.delivery_id is None
    nonpass_window = store.get_turn_execution_window(session_id)
    assert nonpass_window is not None
    assert nonpass_window["stage"] == "L2_PLAN"
    assert nonpass_window["current_work_run_id"] == "workrun-one"
    assert nonpass_window["latest_checkpoint_id"] is None
    settled = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id="verification-nonpass",
    )
    assert settled.result == result

    started = _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=5,
        expected_progress_revision=2,
        apply_id="start-after-nonpass",
        attempt_id="attempt-after-nonpass",
    )
    assert started.work_run_revision == 6
    loaded = _get(session_id)
    latest = loaded.attempts[-1]
    assert latest.attempt.attempt_id == "attempt-after-nonpass"
    assert latest.input_verification_request_id == "verification-nonpass"
    assert latest.input_checkpoint_id == "verification-nonpass"
    assert latest.input_verification_result == result
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert window["stage"] == "L2_PLAN"
    assert window["latest_checkpoint_id"] == "verification-nonpass"


def test_nonpass_feedback_remains_current_when_progress_changes_without_output_change():
    session_id, turn_id, subject = _seed_task_node()
    _submit_for_verification(session_id, turn_id, subject)
    verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_work_run_revision=3,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=_window_revision(session_id),
        apply_id="prepare-progress-feedback",
        verification_request_id="verification-progress-feedback",
    )
    prepared = verification_store.get_prepared_task_node_verification(
        session_id=session_id,
        invocation_turn_id=turn_id,
        verification_request_id="verification-progress-feedback",
    )
    verification_store.commit_task_node_verification_result(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        verification_request_id="verification-progress-feedback",
        result=_verification_result(prepared, all_pass=False),
        expected_work_run_revision=4,
        expected_verification_request_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="commit-progress-feedback",
    )
    _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=5,
        expected_progress_revision=2,
        apply_id="start-progress-change",
        attempt_id="attempt-progress-change",
    )
    decision = work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-progress-change",
        decision=_call_decision(
            call_id="call-progress-change",
            updates=(
                AcceptanceUpdate(
                    acceptance_id="deliverable",
                    model_claimed_satisfied=False,
                ),
            ),
        ),
        expected_work_run_revision=6,
        expected_progress_revision=2,
        expected_window_revision=_window_revision(session_id),
        apply_id="decision-progress-change",
    )
    assert decision.acceptance_progress_revision == 3
    work_run_store.append_work_run_tool_result(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        result=ToolResult(
            status=ToolResultStatus.SUCCEEDED,
            tool_result_id="result-progress-change",
            tool_call_id="call-progress-change",
            attempt_id="attempt-progress-change",
            ordinal=1,
            output={"checked": True},
        ),
        expected_work_run_revision=7,
        expected_progress_revision=3,
        expected_window_revision=_window_revision(session_id),
        apply_id="result-progress-change",
    )
    work_run_store.close_work_run_attempt(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-progress-change",
        expected_work_run_revision=8,
        expected_progress_revision=3,
        expected_window_revision=_window_revision(session_id),
        apply_id="close-progress-change",
    )
    _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=9,
        expected_progress_revision=3,
        apply_id="start-after-progress-change",
        attempt_id="attempt-after-progress-change",
    )

    latest = _get(session_id).attempts[-1]
    assert latest.input_output_revision == 2
    assert latest.input_verification_request_id == "verification-progress-feedback"
    assert latest.input_verification_result is not None
    assert latest.input_verification_result.acceptance_progress_revision == 2


def test_start_attempt_fails_closed_when_nonpass_submit_attempt_triad_drifts():
    session_id, turn_id, subject = _seed_task_node()
    _submit_for_verification(session_id, turn_id, subject)
    verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_work_run_revision=3,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=_window_revision(session_id),
        apply_id="prepare-feedback-triad",
        verification_request_id="verification-feedback-triad",
    )
    prepared = verification_store.get_prepared_task_node_verification(
        session_id=session_id,
        invocation_turn_id=turn_id,
        verification_request_id="verification-feedback-triad",
    )
    verification_store.commit_task_node_verification_result(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        verification_request_id="verification-feedback-triad",
        result=_verification_result(prepared, all_pass=False),
        expected_work_run_revision=4,
        expected_verification_request_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="commit-feedback-triad",
    )
    before_run = _get(session_id).work_run
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_run_attempts SET committed_output_revision=1, "
            "submitted_output_revision=1 "
            "WHERE attempt_id='submit-attempt-workrun-one'"
        )
    before_window = _window_revision(session_id)
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="verification feedback binding",
    ):
        _start(
            session_id,
            turn_id,
            "workrun-one",
            expected_run_revision=5,
            expected_progress_revision=2,
            expected_window_revision=before_window,
            apply_id="start-corrupt-feedback-triad",
            attempt_id="must-not-start",
        )
    assert _window_revision(session_id) == before_window
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_attempts "
            "WHERE attempt_id='must-not-start'"
        ).fetchone()[0] == 0
        run_row = conn.execute(
            "SELECT revision, attempts_started FROM insession_work_runs "
            "WHERE work_run_id='workrun-one'"
        ).fetchone()
        assert tuple(run_row) == (
            before_run.revision,
            before_run.budget.attempts_started,
        )


def test_pass_verification_freezes_single_output_and_completes_only_node_and_run():
    session_id, turn_id, subject = _seed_task_node()
    _submit_for_verification(session_id, turn_id, subject)
    verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_work_run_revision=3,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=_window_revision(session_id),
        apply_id="prepare-pass",
        verification_request_id="verification-pass",
    )
    prepared = verification_store.get_prepared_task_node_verification(
        session_id=session_id,
        invocation_turn_id=turn_id,
        verification_request_id="verification-pass",
    )
    result = _verification_result(prepared, all_pass=True)
    original_window_revision = _window_revision(session_id)
    with store._connect() as conn:
        task_version_before = int(
            conn.execute(
                "SELECT state_version FROM insession_tasks WHERE insession_task_id=?",
                (subject.task_id,),
            ).fetchone()[0]
        )

    committed = verification_store.commit_task_node_verification_result(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        verification_request_id="verification-pass",
        result=result,
        expected_work_run_revision=4,
        expected_verification_request_revision=1,
        expected_window_revision=original_window_revision,
        apply_id="commit-pass",
        delivery_id="delivery-one",
    )

    assert committed.work_run_revision == 5
    assert committed.work_run_status.value == "completed"
    assert committed.work_run_reason == "verification_passed"
    assert committed.all_pass is True
    assert committed.delivery_id == "delivery-one"
    delivery_projection = verification_store.get_task_node_delivery(
        session_id=session_id,
        delivery_id="delivery-one",
    )
    assert delivery_projection.delivery.work_run_id == "workrun-one"
    assert delivery_projection.delivery.subject == subject
    assert delivery_projection.delivery.verification_request_id == "verification-pass"
    assert delivery_projection.delivery.output_revision == 2
    assert delivery_projection.output_window.content == "verified candidate"
    assert delivery_projection.output_window.format.value == "markdown"
    passed_window = store.get_turn_execution_window(session_id)
    assert passed_window is not None
    assert passed_window["stage"] == "PERSIST"
    assert passed_window["current_work_run_id"] is None
    assert passed_window["latest_checkpoint_id"] is None
    loaded = _get(session_id)
    assert loaded.work_run.status.value == "completed"
    assert loaded.output_window.content == "verified candidate"
    assert loaded.current_verification_request_id is None
    assert loaded.node_delivery_id == "delivery-one"
    with store._connect() as conn:
        delivery = conn.execute(
            "SELECT * FROM insession_task_node_deliveries WHERE delivery_id='delivery-one'"
        ).fetchone()
        output = conn.execute(
            "SELECT output_revision, snapshot_json, frozen_at "
            "FROM insession_work_run_output_windows WHERE work_run_id='workrun-one'"
        ).fetchone()
        node_status = conn.execute(
            "SELECT status FROM insession_task_node_states "
            "WHERE insession_task_id=? AND insession_task_node_id=?",
            (subject.task_id, subject.node_id),
        ).fetchone()[0]
        task_row = conn.execute(
            "SELECT current_status, state_version FROM insession_tasks "
            "WHERE insession_task_id=?",
            (subject.task_id,),
        ).fetchone()
        request_json = str(
            conn.execute(
                "SELECT result_json FROM insession_work_run_verification_requests "
                "WHERE verification_request_id='verification-pass'"
            ).fetchone()[0]
        )
        receipt_json = str(
            conn.execute(
                "SELECT result_json FROM insession_work_run_apply_receipts "
                "WHERE apply_id='commit-pass'"
            ).fetchone()[0]
        )
    assert delivery is not None
    assert int(delivery["output_revision"]) == 2
    assert output["frozen_at"] is not None
    assert "verified candidate" in str(output["snapshot_json"])
    assert node_status == "completed"
    assert task_row["current_status"] == "active"
    assert int(task_row["state_version"]) == task_version_before + 1
    assert "acceptance_results" in request_json
    assert "verified candidate" not in request_json
    assert "acceptance_results" not in receipt_json
    assert "verified candidate" not in receipt_json

    replay = verification_store.commit_task_node_verification_result(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        verification_request_id="verification-pass",
        result=result,
        expected_work_run_revision=4,
        expected_verification_request_revision=1,
        expected_window_revision=original_window_revision,
        apply_id="commit-pass",
        delivery_id="delivery-one",
    )
    assert replay.status == "replayed"
    assert replay.model_copy(update={"status": "applied"}) == committed

    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_tasks SET current_status='completed', "
            "state_version=state_version+1 WHERE insession_task_id=?",
            (subject.task_id,),
        )
    assert verification_store.get_task_node_delivery(
        session_id=session_id,
        delivery_id="delivery-one",
    ).output_window.content == "verified candidate"
    completed_task_record = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id="verification-pass",
    )
    assert completed_task_record.result is not None
    assert completed_task_record.result.all_pass is True

    now = "2026-08-14T00:40:00+00:00"
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_graph_revisions "
            "(insession_task_id, graph_revision, source_turn_id, proposal_hash, "
            "source_anchors_json, authorization_anchor_ids_json, "
            "required_anchor_ids_json, created_at) "
            "VALUES (?, 2, ?, 'replacement-graph', '[]', '[]', '[]', ?)",
            (subject.task_id, turn_id, now),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_nodes "
            "(insession_task_id, graph_revision, insession_task_node_id, "
            "node_revision, node_kind, ordinal, title, objective, "
            "source_anchor_ids_json, acceptance_criteria_json, constraints_json, "
            "created_at) VALUES (?, 2, 'replacement-node', 1, 'root', 0, "
            "'replacement', 'replacement', '[\"request\"]', "
            "'[{\"acceptance_id\":\"replacement\",\"criterion\":\"done\","
            "\"source_anchor_ids\":[\"request\"]}]', '[]', ?)",
            (subject.task_id, now),
        )
        conn.execute(
            "INSERT INTO insession_task_node_states "
            "(insession_task_id, insession_task_node_id, node_revision, status, "
            "state_version, updated_at) VALUES (?, 'replacement-node', 1, "
            "'proposed', 1, ?)",
            (subject.task_id, now),
        )
        conn.execute(
            "UPDATE insession_tasks SET current_graph_revision=2, "
            "current_status='active', state_version=state_version+1 "
            "WHERE insession_task_id=?",
            (subject.task_id,),
        )
    evolved_delivery = verification_store.get_task_node_delivery(
        session_id=session_id,
        delivery_id="delivery-one",
    )
    assert evolved_delivery.delivery.subject == subject
    assert evolved_delivery.output_window.content == "verified candidate"
    evolved_record = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id="verification-pass",
    )
    assert evolved_record.request.subject == subject
    assert evolved_record.result is not None
    assert evolved_record.result.all_pass is True


def test_pass_verification_completes_canonical_single_node_task_once():
    session_id, turn_id, subject = _seed_task_node(
        request_id="canonical-single-node-turn",
        task_id="canonical-single-node",
        node_id="canonical-single-node",
    )
    _submit_for_verification(session_id, turn_id, subject)
    verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_work_run_revision=3,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=_window_revision(session_id),
        apply_id="prepare-canonical-single-node",
        verification_request_id="verification-canonical-single-node",
    )
    prepared = verification_store.get_prepared_task_node_verification(
        session_id=session_id,
        invocation_turn_id=turn_id,
        verification_request_id="verification-canonical-single-node",
    )
    result = _verification_result(prepared, all_pass=True)
    expected_window_revision = _window_revision(session_id)
    with store._connect() as conn:
        task_version_before = int(
            conn.execute(
                "SELECT state_version FROM insession_tasks "
                "WHERE insession_task_id=?",
                (subject.task_id,),
            ).fetchone()[0]
        )

    committed = verification_store.commit_task_node_verification_result(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        verification_request_id="verification-canonical-single-node",
        result=result,
        expected_work_run_revision=4,
        expected_verification_request_revision=1,
        expected_window_revision=expected_window_revision,
        apply_id="commit-canonical-single-node",
        delivery_id="delivery-canonical-single-node",
    )

    with store._connect() as conn:
        assert conn.execute(
            "SELECT current_status FROM insession_tasks "
            "WHERE insession_task_id=?",
            (subject.task_id,),
        ).fetchone()[0] == "completed"
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert window["stage"] == "PERSIST"
    assert window["current_work_run_id"] is None
    assert window["current_attempt_id"] is None
    assert window["latest_checkpoint_id"] is None
    with store._connect() as conn:
        completed_version = int(
            conn.execute(
                "SELECT state_version FROM insession_tasks "
                "WHERE insession_task_id=?",
                (subject.task_id,),
            ).fetchone()[0]
        )
    assert completed_version == task_version_before + 1

    replay = verification_store.commit_task_node_verification_result(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        verification_request_id="verification-canonical-single-node",
        result=result,
        expected_work_run_revision=4,
        expected_verification_request_revision=1,
        expected_window_revision=expected_window_revision,
        apply_id="commit-canonical-single-node",
        delivery_id="delivery-canonical-single-node",
    )
    assert replay.status == "replayed"
    assert replay.model_copy(update={"status": "applied"}) == committed
    with store._connect() as conn:
        assert conn.execute(
            "SELECT state_version FROM insession_tasks "
            "WHERE insession_task_id=?",
            (subject.task_id,),
        ).fetchone()[0] == completed_version


def test_verified_turn_publication_is_reference_only_and_all_readers_resolve_it(
    monkeypatch: pytest.MonkeyPatch,
):
    session_id, turn_id, task_id, candidate = (
        settle_current_auxiliary_candidate(monkeypatch, route="pass")
    )
    delivery_id = candidate.final_delivery_id
    publication_body = candidate.publication_body
    assert delivery_id is not None
    assert publication_body is not None
    expected_window_revision = _window_revision(session_id)

    finalized = store.finalize_verified_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        delivery_id=delivery_id,
        expected_window_revision=expected_window_revision,
        post_commit_job_kinds=("session_rag",),
    )

    assert finalized["replayed"] is False
    assert finalized["node_delivery_ids"] == (delivery_id,)
    assert finalized["turn"]["status"] == "completed"
    assert finalized["turn"]["processing_level"] == "L2"
    assert finalized["window"]["window_state"] == "post_commit_pending"
    assert finalized["window"]["stage"] == "PERSIST"
    assert [job["job_kind"] for job in finalized["post_commit_jobs"]] == [
        "session_rag"
    ]
    with store._connect() as conn:
        raw = conn.execute(
            "SELECT a.content, reference.node_delivery_id, delivery.work_run_id, "
            "output.snapshot_json "
            "FROM session_turn_commits AS c "
            "JOIN session_turns AS a "
            "ON a.session_id=c.session_id AND a.turn_idx=c.assistant_turn_idx "
            "JOIN session_turn_commit_references AS reference "
            "ON reference.run_id=c.run_id AND reference.ordinal=1 "
            "AND reference.reference_kind='node_delivery' "
            "JOIN insession_task_node_deliveries AS delivery "
            "ON delivery.delivery_id=reference.node_delivery_id "
            "JOIN insession_work_run_output_windows AS output "
            "ON output.work_run_id=delivery.work_run_id "
            "AND output.output_revision=delivery.output_revision "
            "WHERE c.turn_id=?",
            (turn_id,),
        ).fetchone()
        assert tuple(raw)[:2] == ("", delivery_id)
        delivery_work_run_id = str(raw["work_run_id"])
        assert json.loads(str(raw["snapshot_json"]))["content"] == publication_body
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO session_turn_commit_references "
                "(run_id, ordinal, reference_kind, node_delivery_id, "
                "question_attempt_id) VALUES (?, 2, 'node_delivery', ?, NULL)",
                (f"commit_{turn_id}", delivery_id),
            )

    assert store.get_turns(session_id)[1]["content"] == publication_body
    assert store.get_turn(session_id, 1)["content"] == publication_body
    assert store.get_history_messages(session_id)[1] == {
        "role": "assistant",
        "content": publication_body,
    }
    run_id = f"commit_{turn_id}"
    assert store.get_committed_turn_pair(session_id, run_id)[
        "assistant_content"
    ] == publication_body
    assert store.list_committed_turn_pairs(session_id)[0][
        "assistant_content"
    ] == publication_body
    search = store.search_sessions(publication_body)
    assert len(search) == 1
    assert search[0]["id"] == session_id
    assert search[0]["match_role"] == "assistant"

    replayed = store.finalize_verified_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        delivery_id=delivery_id,
        expected_window_revision=expected_window_revision,
        post_commit_job_kinds=("session_rag",),
    )
    assert replayed["replayed"] is True
    assert replayed["delivery"]["created"] is False
    with pytest.raises(store.TurnExecutionFinalizationConflict):
        store.finalize_verified_turn_execution(
            session_id=session_id,
            turn_id=turn_id,
            delivery_id=delivery_id,
            expected_window_revision=expected_window_revision,
            post_commit_job_kinds=(),
        )
    with pytest.raises(store.TurnExecutionFinalizationConflict):
        store.finalize_verified_turn_execution(
            session_id=session_id,
            turn_id=turn_id,
            delivery_id="different-delivery",
            expected_window_revision=expected_window_revision,
            post_commit_job_kinds=("session_rag",),
        )
    with pytest.raises(store.TurnExecutionFinalizationConflict):
        store.finalize_turn_execution(
            session_id=session_id,
            turn_id=turn_id,
            expected_window_revision=expected_window_revision,
            processing_level="L2",
            assistant_content=publication_body,
            post_commit_job_kinds=("session_rag",),
        )

    claimed = store.claim_due_turn_post_commit_jobs(
        session_id=session_id,
        worker_id="verified-publication-worker",
        lease_seconds=60,
    )
    store.mark_turn_post_commit_job_applied(
        job_id=str(claimed[0]["job_id"]),
        worker_id="verified-publication-worker",
    )
    store.release_turn_execution_window(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(finalized["window"]["state_version"]),
    )

    evolved_at = "2026-08-14T00:05:00+00:00"
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_graph_revisions "
            "(insession_task_id, graph_revision, source_turn_id, proposal_hash, "
            "source_anchors_json, authorization_anchor_ids_json, "
            "required_anchor_ids_json, created_at) "
            "VALUES (?, 2, ?, 'evolved-graph', '[]', '[]', '[]', ?)",
            (task_id, turn_id, evolved_at),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_nodes "
            "(insession_task_id, graph_revision, insession_task_node_id, "
            "node_revision, node_kind, ordinal, title, objective, "
            "source_anchor_ids_json, acceptance_criteria_json, constraints_json, "
            "created_at) VALUES (?, 2, 'evolved-node', 1, 'root', 0, "
            "'evolved', 'evolved', '[\"request\"]', "
            "'[{\"acceptance_id\":\"evolved\",\"criterion\":\"done\","
            "\"source_anchor_ids\":[\"request\"]}]', '[]', ?)",
            (task_id, evolved_at),
        )
        conn.execute(
            "INSERT INTO insession_task_node_states "
            "(insession_task_id, insession_task_node_id, node_revision, status, "
            "state_version, updated_at) VALUES (?, 'evolved-node', 1, "
            "'proposed', 1, ?)",
            (task_id, evolved_at),
        )
        conn.execute(
            "UPDATE insession_tasks SET current_graph_revision=2, "
            "current_status='active', state_version=state_version+1 "
            "WHERE insession_task_id=?",
            (task_id,),
        )

    drift_replay = store.finalize_verified_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        delivery_id=delivery_id,
        expected_window_revision=expected_window_revision,
        post_commit_job_kinds=("session_rag",),
    )
    assert drift_replay["replayed"] is True
    assert store.get_turn(session_id, 1)["content"] == publication_body

    for index in range(5):
        accepted = store.accept_turn_execution(
            session_id=session_id,
            client_request_id=f"summary-hot-{index}",
            source="runtime_test",
            user_text=f"hot user {index}",
        )
        hot_turn_id = str(accepted["turn"]["turn_id"])
        hot_finalized = store.finalize_turn_execution(
            session_id=session_id,
            turn_id=hot_turn_id,
            expected_window_revision=int(accepted["window"]["state_version"]),
            processing_level="L0",
            assistant_content=f"hot assistant {index}",
            post_commit_job_kinds=(),
        )
        store.release_turn_execution_window(
            session_id=session_id,
            turn_id=hot_turn_id,
            expected_window_revision=int(
                hot_finalized["window"]["state_version"]
            ),
        )
    summary_pairs = store.list_committed_turn_pairs_for_summary(
        session_id,
        after_turn_id=None,
        retain_recent_pairs=5,
        limit=8,
    )
    assert len(summary_pairs) == 1
    assert summary_pairs[0].assistant_content == publication_body

    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_run_output_windows SET frozen_at=NULL "
            "WHERE work_run_id=?",
            (delivery_work_run_id,),
        )
    with pytest.raises(store.TranscriptDeliveryReferenceError):
        store.get_turns(session_id)
    with pytest.raises(store.TranscriptDeliveryReferenceError):
        store.get_committed_turn_pair(session_id, run_id)
    with pytest.raises(store.TranscriptDeliveryReferenceError):
        store.search_sessions(publication_body)
    with pytest.raises(store.TranscriptDeliveryReferenceError):
        store.list_committed_turn_pairs_for_summary(
            session_id,
            after_turn_id=None,
            retain_recent_pairs=5,
            limit=8,
        )

    assert store.purge_session(session_id) is True
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM session_turn_commits WHERE session_id=?",
            (session_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM session_turn_commit_references WHERE run_id=?",
            (run_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_task_node_deliveries "
            "WHERE session_id=?",
            (session_id,),
        ).fetchone()[0] == 0


def test_verified_turn_publication_requires_current_finish_gate_and_persist_window():
    session_id, turn_id, subject = _seed_task_node(
        request_id="verified-authority-turn",
        task_id="verified-authority-task",
        node_id="noncanonical-node",
    )
    _commit_passing_verification(
        session_id,
        turn_id,
        subject,
        request_id="verified-authority-request",
        delivery_id="verified-authority-delivery",
    )
    before_revision = _window_revision(session_id)
    with pytest.raises(
        store.TurnExecutionFinalizationConflict,
        match="Task is not completed",
    ):
        store.finalize_verified_turn_execution(
            session_id=session_id,
            turn_id=turn_id,
            delivery_id="verified-authority-delivery",
            expected_window_revision=before_revision,
            post_commit_job_kinds=(),
        )
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM session_turn_commits WHERE turn_id=?",
            (turn_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT status FROM runtime_turns WHERE turn_id=?",
            (turn_id,),
        ).fetchone()[0] == "running"

        conn.execute(
            "UPDATE insession_tasks SET current_status='completed' "
            "WHERE insession_task_id=?",
            (subject.task_id,),
        )
        conn.execute(
            "UPDATE turn_execution_windows SET stage='RESPONSE' "
            "WHERE session_id=?",
            (session_id,),
        )
    with pytest.raises(
        store.TurnExecutionFinalizationConflict,
        match="active PERSIST",
    ):
        store.finalize_verified_turn_execution(
            session_id=session_id,
            turn_id=turn_id,
            delivery_id="verified-authority-delivery",
            expected_window_revision=before_revision,
            post_commit_job_kinds=(),
        )
    with pytest.raises(store.TurnExecutionWindowRevisionConflict):
        store.finalize_verified_turn_execution(
            session_id=session_id,
            turn_id=turn_id,
            delivery_id="verified-authority-delivery",
            expected_window_revision=before_revision - 1,
            post_commit_job_kinds=(),
        )


def test_verified_turn_publication_late_failure_rolls_back_and_retries(monkeypatch):
    session_id, turn_id, _task_id, candidate = (
        settle_current_auxiliary_candidate(monkeypatch, route="pass")
    )
    delivery_id = candidate.final_delivery_id
    publication_body = candidate.publication_body
    assert delivery_id is not None
    assert publication_body is not None
    before_revision = _window_revision(session_id)
    original_insert_jobs = turn_execution_records._insert_post_commit_jobs

    def fail_after_transcript(*_args, **_kwargs):
        raise sqlite3.OperationalError("injected verified publication failure")

    monkeypatch.setattr(
        turn_execution_records,
        "_insert_post_commit_jobs",
        fail_after_transcript,
    )
    with pytest.raises(sqlite3.OperationalError, match="publication failure"):
        store.finalize_verified_turn_execution(
            session_id=session_id,
            turn_id=turn_id,
            delivery_id=delivery_id,
            expected_window_revision=before_revision,
            post_commit_job_kinds=("session_rag",),
        )
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM session_turn_commits WHERE turn_id=?",
            (turn_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM session_turns "
            "WHERE session_id=? AND role='assistant'",
            (session_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT status FROM runtime_turns WHERE turn_id=?",
            (turn_id,),
        ).fetchone()[0] == "running"
    assert _window_revision(session_id) == before_revision
    assert store.get_turn_execution_window(session_id)["window_state"] == "active"

    monkeypatch.setattr(
        turn_execution_records,
        "_insert_post_commit_jobs",
        original_insert_jobs,
    )
    retried = store.finalize_verified_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        delivery_id=delivery_id,
        expected_window_revision=before_revision,
        post_commit_job_kinds=("session_rag",),
    )
    assert retried["replayed"] is False
    assert store.get_turns(session_id)[1]["content"] == publication_body


def test_pass_verification_does_not_complete_a_multi_node_task():
    session_id, turn_id, root = _seed_task_node(
        request_id="multi-node-finish-gate-turn",
        task_id="multi-node-task",
        node_id="multi-node-task",
    )
    child_id = "multi-node-child"
    now = "2026-08-14T00:45:00+00:00"
    child_acceptances = json.dumps(
        [
            {
                "acceptance_id": acceptance_id,
                "criterion": f"满足 {acceptance_id}",
                "source_anchor_ids": ["request"],
            }
            for acceptance_id in ("deliverable", "quality")
        ],
        ensure_ascii=False,
    )
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_graph_nodes "
            "(insession_task_id, graph_revision, insession_task_node_id, "
            "node_revision, node_kind, ordinal, title, objective, "
            "source_anchor_ids_json, acceptance_criteria_json, constraints_json, "
            "created_at) VALUES (?, 1, ?, 1, 'subtask', 1, '子节点', "
            "'完成子节点', '[\"request\"]', ?, '[]', ?)",
            (root.task_id, child_id, child_acceptances, now),
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
    child = TaskNodeSubject(
        task_id=root.task_id,
        graph_revision=1,
        node_id=child_id,
        node_revision=1,
    )
    _submit_for_verification(session_id, turn_id, child)
    # 即使根节点由外部完成，也不能让该 P2 切片定义未来 P5 多节点任务聚合规则。
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_task_node_states SET status='completed', "
            "state_version=state_version+1 WHERE insession_task_id=? "
            "AND insession_task_node_id=?",
            (root.task_id, root.node_id),
        )
        task_version_before = int(
            conn.execute(
                "SELECT state_version FROM insession_tasks "
                "WHERE insession_task_id=?",
                (root.task_id,),
            ).fetchone()[0]
        )
    verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_work_run_revision=3,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=_window_revision(session_id),
        apply_id="prepare-multi-node-gate",
        verification_request_id="verification-multi-node-gate",
    )
    prepared = verification_store.get_prepared_task_node_verification(
        session_id=session_id,
        invocation_turn_id=turn_id,
        verification_request_id="verification-multi-node-gate",
    )
    verification_store.commit_task_node_verification_result(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        verification_request_id="verification-multi-node-gate",
        result=_verification_result(prepared, all_pass=True),
        expected_work_run_revision=4,
        expected_verification_request_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="commit-multi-node-gate",
        delivery_id="delivery-multi-node-gate",
    )

    with store._connect() as conn:
        assert conn.execute(
            "SELECT current_status FROM insession_tasks "
            "WHERE insession_task_id=?",
            (root.task_id,),
        ).fetchone()[0] == "active"
        assert conn.execute(
            "SELECT state_version FROM insession_tasks "
            "WHERE insession_task_id=?",
            (root.task_id,),
        ).fetchone()[0] == task_version_before + 1


def test_child_then_root_pass_completes_multi_node_task_with_exact_dependencies():
    session_id, turn_id, root = _seed_task_node(
        request_id="full-multi-node-finish-gate-turn",
        task_id="full-multi-node-task",
        node_id="full-multi-node-task",
    )
    child_id = "full-multi-node-child"
    now = "2026-08-14T00:46:00+00:00"
    child_acceptances = json.dumps(
        [
            {
                "acceptance_id": acceptance_id,
                "criterion": f"满足 {acceptance_id}",
                "source_anchor_ids": ["request"],
            }
            for acceptance_id in ("deliverable", "quality")
        ],
        ensure_ascii=False,
    )
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_graph_nodes "
            "(insession_task_id, graph_revision, insession_task_node_id, "
            "node_revision, node_kind, ordinal, title, objective, "
            "source_anchor_ids_json, acceptance_criteria_json, constraints_json, "
            "created_at) VALUES (?, 1, ?, 1, 'subtask', 1, '子节点', "
            "'完成子节点', '[\"request\"]', ?, '[]', ?)",
            (root.task_id, child_id, child_acceptances, now),
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
    child = TaskNodeSubject(
        task_id=root.task_id,
        graph_revision=1,
        node_id=child_id,
        node_revision=1,
    )

    child_created = work_run_store.create_task_node_work_run(
        session_id=session_id,
        turn_id=turn_id,
        subject=child,
        expected_task_state_version=1,
        expected_node_state_version=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="create-full-child-run",
        work_run_id="full-child-run",
    )
    child_started = _start(
        session_id,
        turn_id,
        "full-child-run",
        expected_run_revision=child_created.work_run_revision,
        apply_id="start-full-child-run",
        attempt_id="full-child-attempt",
    )
    child_submitted = work_run_store.commit_work_run_output_action(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="full-child-run",
        attempt_id="full-child-attempt",
        decision=_output_decision(
            content="verified child",
            submit=True,
            updates=tuple(
                AcceptanceUpdate(
                    acceptance_id=acceptance_id,
                    model_claimed_satisfied=True,
                )
                for acceptance_id in ("deliverable", "quality")
            ),
        ),
        expected_work_run_revision=child_started.work_run_revision,
        expected_progress_revision=child_started.acceptance_progress_revision,
        expected_output_revision=child_started.output_window_revision,
        expected_window_revision=child_started.window_state_version,
        apply_id="submit-full-child-run",
    )
    child_prepared_mutation = verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="full-child-run",
        expected_work_run_revision=child_submitted.work_run_revision,
        expected_progress_revision=child_submitted.acceptance_progress_revision,
        expected_output_revision=child_submitted.output_window_revision,
        expected_window_revision=child_submitted.window_state_version,
        apply_id="prepare-full-child",
        verification_request_id="verification-full-child",
    )
    child_prepared = verification_store.get_prepared_task_node_verification(
        session_id=session_id,
        invocation_turn_id=turn_id,
        verification_request_id="verification-full-child",
    )
    assert child_prepared.record.request.dependency_delivery_ids == ()
    child_request = child_prepared.record.request
    with store._connect() as conn:
        child_hash_row = conn.execute(
            "SELECT request_binding_hash FROM "
            "insession_work_run_verification_requests "
            "WHERE verification_request_id='verification-full-child'"
        ).fetchone()
        child_progress_hash = str(
            conn.execute(
                "SELECT snapshot_hash FROM "
                "insession_work_run_acceptance_progress "
                "WHERE work_run_id='full-child-run'"
            ).fetchone()[0]
        )
        child_output_hash = str(
            conn.execute(
                "SELECT snapshot_hash FROM insession_work_run_output_windows "
                "WHERE work_run_id='full-child-run'"
            ).fetchone()[0]
        )
    assert child_hash_row is not None
    current_child_hash = work_verification_records._verification_binding_hash(
        verification_request_id=child_request.verification_request_id,
        session_id=child_request.session_id,
        request_turn_id=child_request.request_turn_id,
        work_run_id=child_request.work_run_id,
        subject=child_request.subject,
        node_title=child_prepared.node_title,
        node_objective=child_prepared.node_objective,
        submitted_attempt=child_prepared.submitted_attempt,
        locked_work_run_revision=child_request.locked_work_run_revision,
        progress=child_prepared.acceptance_progress,
        progress_hash=child_progress_hash,
        output_window=child_prepared.locked_output_window,
        output_hash=child_output_hash,
        acceptances=child_prepared.acceptances,
        supporting_results=child_prepared.supporting_tool_results,
        dependency_delivery_ids=(),
        prepared_budget=child_request.prepared_budget,
    )
    retired_leaf_hash = work_verification_records._payload_hash(
        {
            "verification_request_id": child_request.verification_request_id,
            "session_id": child_request.session_id,
            "request_turn_id": child_request.request_turn_id,
            "work_run_id": child_request.work_run_id,
            "subject": child_request.subject.model_dump(mode="json"),
            "node_title": child_prepared.node_title,
            "node_objective": child_prepared.node_objective,
            "submitted_attempt": child_prepared.submitted_attempt.model_dump(
                mode="json"
            ),
            "locked_work_run_revision": child_request.locked_work_run_revision,
            "acceptance_progress": child_prepared.acceptance_progress.model_dump(
                mode="json"
            ),
            "acceptance_progress_snapshot_hash": child_progress_hash,
            "output_window": child_prepared.locked_output_window.model_dump(
                mode="json"
            ),
            "output_window_snapshot_hash": child_output_hash,
            "acceptances": [
                item.model_dump(mode="json")
                for item in child_prepared.acceptances
            ],
            "supporting_tool_results": [
                item.model_dump(mode="json")
                for item in child_prepared.supporting_tool_results
            ],
            "prepared_budget": child_request.prepared_budget.model_dump(
                mode="json"
            ),
        }
    )
    assert current_child_hash == str(child_hash_row[0])
    assert retired_leaf_hash != current_child_hash
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_run_verification_requests "
            "SET request_binding_hash=? "
            "WHERE verification_request_id='verification-full-child'",
            (retired_leaf_hash,),
        )
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="immutable binding has drifted",
    ):
        verification_store.get_prepared_task_node_verification(
            session_id=session_id,
            invocation_turn_id=turn_id,
            verification_request_id="verification-full-child",
        )
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_run_verification_requests "
            "SET request_binding_hash=? "
            "WHERE verification_request_id='verification-full-child'",
            (current_child_hash,),
        )
    child_settled = verification_store.commit_task_node_verification_result(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="full-child-run",
        verification_request_id="verification-full-child",
        result=_verification_result(child_prepared, all_pass=True),
        expected_work_run_revision=child_prepared_mutation.work_run_revision,
        expected_verification_request_revision=1,
        expected_window_revision=child_prepared_mutation.window_state_version,
        apply_id="commit-full-child",
        delivery_id="delivery-full-child",
    )
    assert child_settled.work_run_status.value == "completed"
    with store._connect() as conn:
        task = conn.execute(
            "SELECT current_status, state_version FROM insession_tasks "
            "WHERE insession_task_id=?",
            (root.task_id,),
        ).fetchone()
        assert tuple(task) == ("active", 3)

    frontier = work_run_store.project_task_node_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        task_id=root.task_id,
    )
    assert [item.subject.node_id for item in frontier.ready_fresh] == [
        root.node_id
    ]
    assert frontier.ready_fresh[0].dependency_delivery_ids == (
        "delivery-full-child",
    )
    direct_dependencies = work_run_store.get_current_task_node_dependency_deliveries(
        session_id=session_id,
        subject=root,
    )
    assert len(direct_dependencies) == 1
    assert direct_dependencies[0].resolution_kind.value == "direct"
    assert (
        direct_dependencies[0].source_delivery.delivery.subject
        == direct_dependencies[0].target_subject
    )

    root_created = work_run_store.create_task_node_work_run(
        session_id=session_id,
        turn_id=turn_id,
        subject=root,
        expected_task_state_version=3,
        expected_node_state_version=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="create-full-root-run",
        work_run_id="full-root-run",
    )
    root_started = _start(
        session_id,
        turn_id,
        "full-root-run",
        expected_run_revision=root_created.work_run_revision,
        apply_id="start-full-root-run",
        attempt_id="full-root-attempt",
    )
    root_submitted = work_run_store.commit_work_run_output_action(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="full-root-run",
        attempt_id="full-root-attempt",
        decision=_output_decision(
            content="verified root from child",
            submit=True,
            updates=tuple(
                AcceptanceUpdate(
                    acceptance_id=acceptance_id,
                    model_claimed_satisfied=True,
                )
                for acceptance_id in ("deliverable", "quality")
            ),
        ),
        expected_work_run_revision=root_started.work_run_revision,
        expected_progress_revision=root_started.acceptance_progress_revision,
        expected_output_revision=root_started.output_window_revision,
        expected_window_revision=root_started.window_state_version,
        apply_id="submit-full-root-run",
    )
    root_prepared_mutation = verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="full-root-run",
        expected_work_run_revision=root_submitted.work_run_revision,
        expected_progress_revision=root_submitted.acceptance_progress_revision,
        expected_output_revision=root_submitted.output_window_revision,
        expected_window_revision=root_submitted.window_state_version,
        apply_id="prepare-full-root",
        verification_request_id="verification-full-root",
    )
    root_prepared = verification_store.get_prepared_task_node_verification(
        session_id=session_id,
        invocation_turn_id=turn_id,
        verification_request_id="verification-full-root",
    )
    assert root_prepared.record.request.dependency_delivery_ids == (
        "delivery-full-child",
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_run_verification_requests "
            "SET dependency_delivery_ids_json='[]' "
            "WHERE verification_request_id='verification-full-root'"
        )
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="dependency Delivery binding has drifted",
    ):
        verification_store.get_prepared_task_node_verification(
            session_id=session_id,
            invocation_turn_id=turn_id,
            verification_request_id="verification-full-root",
        )
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_run_verification_requests "
            "SET dependency_delivery_ids_json=? "
            "WHERE verification_request_id='verification-full-root'",
            (json.dumps(["delivery-full-child"]),),
        )
    root_settled = verification_store.commit_task_node_verification_result(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="full-root-run",
        verification_request_id="verification-full-root",
        result=_verification_result(root_prepared, all_pass=True),
        expected_work_run_revision=root_prepared_mutation.work_run_revision,
        expected_verification_request_revision=1,
        expected_window_revision=root_prepared_mutation.window_state_version,
        apply_id="commit-full-root",
        delivery_id="delivery-full-root",
    )
    assert root_settled.work_run_status.value == "completed"
    with store._connect() as conn:
        task = conn.execute(
            "SELECT current_status FROM insession_tasks "
            "WHERE insession_task_id=?",
            (root.task_id,),
        ).fetchone()
        assert task["current_status"] == "completed"
    assert work_run_store.get_completed_task_final_delivery_id(
        session_id=session_id,
        task_id=root.task_id,
    ) == "delivery-full-root"


def test_current_delivery_resolver_rejects_ambiguous_carry_receipts():
    class _Rows:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class _AmbiguousCarryConnection:
        def execute(self, sql, _params):
            if "insession_task_node_deliveries" in sql:
                return _Rows([])
            if "insession_auxiliary_v2_task_graph_node_carry_receipts" in sql:
                return _Rows([object(), object()])
            raise AssertionError(f"unexpected resolver query: {sql}")

    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="no unique Delivery or carry receipt",
    ):
        work_execution_records._load_current_task_node_delivery_projection(
            _AmbiguousCarryConnection(),
            session_id="session-ambiguous",
            task_id="task-ambiguous",
            graph_revision=2,
            node_id="node-ambiguous",
            node_revision=1,
        )


def test_single_node_finish_gate_rejects_completion_unconfirmed_authority():
    session_id, turn_id, subject = _seed_task_node(
        request_id="outcome-unknown-gate-turn",
        task_id="outcome-unknown-gate",
        node_id="outcome-unknown-gate",
    )
    _submit_for_verification(session_id, turn_id, subject)
    verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_work_run_revision=3,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=_window_revision(session_id),
        apply_id="prepare-outcome-unknown-gate",
        verification_request_id="verification-outcome-unknown-gate",
    )
    prepared = verification_store.get_prepared_task_node_verification(
        session_id=session_id,
        invocation_turn_id=turn_id,
        verification_request_id="verification-outcome-unknown-gate",
    )
    arguments_json = work_execution_records._canonical_json({})
    unknown = ToolResult(
        status=ToolResultStatus.COMPLETION_UNCONFIRMED,
        tool_result_id="outcome-unknown-result",
        tool_call_id="outcome-unknown-call",
        attempt_id="submit-attempt-workrun-one",
        ordinal=1,
        output=None,
        error_code="completion_unconfirmed",
        error_message="the external outcome cannot be confirmed",
    )
    now = "2026-08-14T00:46:00+00:00"
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_work_run_tool_calls "
            "(tool_call_id, work_run_id, attempt_id, ordinal, provider_call_id, "
            "tool_id, tool_version, modifies_environment, arguments_hash, "
            "arguments_json, created_at) VALUES (?, 'workrun-one', "
            "'submit-attempt-workrun-one', 1, NULL, 'drifted_tool', '1', 1, ?, ?, ?)",
            (
                unknown.tool_call_id,
                work_execution_records._text_hash(arguments_json),
                arguments_json,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO insession_work_run_tool_results "
            "(tool_result_id, work_run_id, attempt_id, tool_call_id, ordinal, "
            "status, result_json, created_at) VALUES (?, 'workrun-one', "
            "'submit-attempt-workrun-one', ?, 1, 'completion_unconfirmed', ?, ?)",
            (
                unknown.tool_result_id,
                unknown.tool_call_id,
                work_execution_records._model_json(unknown),
                now,
            ),
        )

    expected_window_revision = _window_revision(session_id)
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="completion-unconfirmed",
    ):
        verification_store.commit_task_node_verification_result(
            active_seconds_delta=1,
            session_id=session_id,
            turn_id=turn_id,
            work_run_id="workrun-one",
            verification_request_id="verification-outcome-unknown-gate",
            result=_verification_result(prepared, all_pass=True),
            expected_work_run_revision=4,
            expected_verification_request_revision=1,
            expected_window_revision=expected_window_revision,
            apply_id="commit-outcome-unknown-gate",
            delivery_id="delivery-outcome-unknown-gate",
        )

    with store._connect() as conn:
        task = conn.execute(
            "SELECT current_status FROM insession_tasks WHERE insession_task_id=?",
            (subject.task_id,),
        ).fetchone()
        node = conn.execute(
            "SELECT status FROM insession_task_node_states "
            "WHERE insession_task_id=? AND insession_task_node_id=?",
            (subject.task_id, subject.node_id),
        ).fetchone()
        run = conn.execute(
            "SELECT status, reason FROM insession_work_runs "
            "WHERE work_run_id='workrun-one'",
        ).fetchone()
        request = conn.execute(
            "SELECT status, result_json FROM "
            "insession_work_run_verification_requests "
            "WHERE verification_request_id='verification-outcome-unknown-gate'",
        ).fetchone()
        delivery_count = conn.execute(
            "SELECT COUNT(*) FROM insession_task_node_deliveries "
            "WHERE delivery_id='delivery-outcome-unknown-gate'",
        ).fetchone()[0]
    assert task["current_status"] == "active"
    assert node["status"] == "active"
    assert tuple(run) == ("active", "verification_pending")
    assert tuple(request) == ("pending", None)
    assert delivery_count == 0
    assert _window_revision(session_id) == expected_window_revision


def test_verification_pass_late_receipt_failure_rolls_back_and_exact_retry_succeeds(
    monkeypatch,
):
    session_id, turn_id, subject = _seed_task_node(
        request_id="late-receipt-canonical-turn",
        task_id="late-receipt-canonical",
        node_id="late-receipt-canonical",
    )
    _submit_for_verification(session_id, turn_id, subject)
    verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_work_run_revision=3,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=_window_revision(session_id),
        apply_id="prepare-late-pass",
        verification_request_id="verification-late-pass",
    )
    prepared = verification_store.get_prepared_task_node_verification(
        session_id=session_id,
        invocation_turn_id=turn_id,
        verification_request_id="verification-late-pass",
    )
    result = _verification_result(prepared, all_pass=True)
    expected_window_revision = _window_revision(session_id)
    with store._connect() as conn:
        task_version_before = int(
            conn.execute(
                "SELECT state_version FROM insession_tasks "
                "WHERE insession_task_id=?",
                (subject.task_id,),
            ).fetchone()[0]
        )
    original_insert_receipt = work_verification_records._insert_receipt

    def fail_receipt(*args, **kwargs):
        raise sqlite3.OperationalError("injected verification receipt failure")

    monkeypatch.setattr(work_verification_records, "_insert_receipt", fail_receipt)
    with pytest.raises(sqlite3.OperationalError, match="verification receipt"):
        verification_store.commit_task_node_verification_result(
            active_seconds_delta=1,
            session_id=session_id,
            turn_id=turn_id,
            work_run_id="workrun-one",
            verification_request_id="verification-late-pass",
            result=result,
            expected_work_run_revision=4,
            expected_verification_request_revision=1,
            expected_window_revision=expected_window_revision,
            apply_id="commit-late-pass",
            delivery_id="delivery-late-pass",
        )

    rolled_back = _get(session_id)
    assert rolled_back.work_run.revision == 4
    assert rolled_back.work_run.status.value == "active"
    assert rolled_back.work_run.reason == "verification_pending"
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert int(window["state_version"]) == expected_window_revision
    assert window["stage"] == "VERIFICATION"
    assert window["latest_checkpoint_id"] == "verification-late-pass"
    with store._connect() as conn:
        request = conn.execute(
            "SELECT status, request_revision, result_json, all_pass "
            "FROM insession_work_run_verification_requests "
            "WHERE verification_request_id='verification-late-pass'"
        ).fetchone()
        assert tuple(request) == ("pending", 1, None, None)
        assert conn.execute(
            "SELECT frozen_at FROM insession_work_run_output_windows "
            "WHERE work_run_id='workrun-one'"
        ).fetchone()[0] is None
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_task_node_deliveries "
            "WHERE delivery_id='delivery-late-pass'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT status FROM insession_task_node_states "
            "WHERE insession_task_id=? AND insession_task_node_id=?",
            (subject.task_id, subject.node_id),
        ).fetchone()[0] == "active"
        task = conn.execute(
            "SELECT current_status, state_version FROM insession_tasks "
            "WHERE insession_task_id=?",
            (subject.task_id,),
        ).fetchone()
        assert tuple(task) == ("active", task_version_before)
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_apply_receipts "
            "WHERE apply_id='commit-late-pass'"
        ).fetchone()[0] == 0

    monkeypatch.setattr(
        work_verification_records,
        "_insert_receipt",
        original_insert_receipt,
    )
    retried = verification_store.commit_task_node_verification_result(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        verification_request_id="verification-late-pass",
        result=result,
        expected_work_run_revision=4,
        expected_verification_request_revision=1,
        expected_window_revision=expected_window_revision,
        apply_id="commit-late-pass",
        delivery_id="delivery-late-pass",
    )
    assert retried.status == "applied"
    assert retried.all_pass is True
    assert retried.delivery_id == "delivery-late-pass"
    with store._connect() as conn:
        task = conn.execute(
            "SELECT current_status, state_version FROM insession_tasks "
            "WHERE insession_task_id=?",
            (subject.task_id,),
        ).fetchone()
    assert tuple(task) == ("completed", task_version_before + 1)


@pytest.mark.parametrize(
    "corruption",
    (
        "unfreeze_output",
        "run_reason",
        "node_status",
        "request_aggregate",
        "submit_revision",
        "output_body",
    ),
)
def test_delivery_loader_fails_closed_on_authority_triad_drift(corruption: str):
    session_id, turn_id, subject = _seed_task_node()
    _commit_passing_verification(
        session_id,
        turn_id,
        subject,
        request_id="verification-delivery-drift",
        delivery_id="delivery-drift",
    )

    with store._connect() as conn:
        if corruption == "unfreeze_output":
            conn.execute(
                "UPDATE insession_work_run_output_windows SET frozen_at=NULL "
                "WHERE work_run_id='workrun-one'"
            )
        elif corruption == "run_reason":
            conn.execute(
                "UPDATE insession_work_runs SET reason='needs_input' "
                "WHERE work_run_id='workrun-one'"
            )
        elif corruption == "node_status":
            conn.execute(
                "UPDATE insession_task_node_states SET status='interrupted' "
                "WHERE insession_task_id=? AND insession_task_node_id=?",
                (subject.task_id, subject.node_id),
            )
        elif corruption == "request_aggregate":
            conn.execute(
                "UPDATE insession_work_run_verification_requests SET all_pass=0 "
                "WHERE verification_request_id='verification-delivery-drift'"
            )
        elif corruption == "submit_revision":
            conn.execute(
                "UPDATE insession_work_run_attempts SET committed_output_revision=1, "
                "submitted_output_revision=1 WHERE attempt_id='submit-attempt-workrun-one'"
            )
        else:
            conn.execute(
                "UPDATE insession_work_run_output_windows SET "
                "snapshot_json=replace(snapshot_json, 'verified candidate', 'tampered') "
                "WHERE work_run_id='workrun-one'"
            )

    with pytest.raises(work_run_store.WorkExecutionPersistenceError):
        verification_store.get_task_node_delivery(
            session_id=session_id,
            delivery_id="delivery-drift",
        )
    with pytest.raises(work_run_store.WorkExecutionPersistenceError):
        _get(session_id)


def test_delivery_created_turn_must_match_terminal_work_run_update_turn():
    session_id, turn_id, subject = _seed_task_node()
    _commit_passing_verification(
        session_id,
        turn_id,
        subject,
        request_id="verification-delivery-turn",
        delivery_id="delivery-turn",
    )
    now = "2026-08-14T00:50:00+00:00"
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO runtime_turns "
            "(turn_id, session_id, source, user_text, status, received_at, completed_at) "
            "VALUES ('other-delivery-turn', ?, 'runtime_test', 'later', "
            "'completed', ?, ?)",
            (session_id, now, now),
        )
        conn.execute(
            "UPDATE insession_task_node_deliveries "
            "SET created_turn_id='other-delivery-turn' WHERE delivery_id='delivery-turn'"
        )

    with pytest.raises(work_run_store.WorkExecutionPersistenceError):
        verification_store.get_task_node_delivery(
            session_id=session_id,
            delivery_id="delivery-turn",
        )
    with pytest.raises(work_run_store.WorkExecutionPersistenceError):
        _get(session_id)


def _seed_pending_verification_rebind():
    session_id, first_turn_id, subject = _seed_task_node()
    _submit_for_verification(session_id, first_turn_id, subject)
    verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=first_turn_id,
        work_run_id="workrun-one",
        expected_work_run_revision=3,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=_window_revision(session_id),
        apply_id="prepare-pending-rebind",
        verification_request_id="verification-pending-rebind",
    )
    original_prepared = verification_store.get_prepared_task_node_verification(
        session_id=session_id,
        invocation_turn_id=first_turn_id,
        verification_request_id="verification-pending-rebind",
    )
    marked = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=_window_revision(session_id),
        stage="VERIFICATION",
        interruption_reason="process_lost_after_verification_prepare",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=int(marked["state_version"]),
        end_reason="process_lost_after_verification_prepare",
        error_code="PROCESS_LOST",
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="pending-verification-rebind-turn",
        source="runtime_test",
        user_text="继续验证",
        lease_owner="work-execution-test",
    )
    second_turn_id = str(accepted["turn"]["turn_id"])
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (
                session_id,
                second_turn_id,
                subject.task_id,
                "2026-08-14T00:30:00+00:00",
            ),
        )
    return session_id, first_turn_id, second_turn_id, original_prepared


def test_pending_verification_rebinds_same_request_and_fences_old_provider_result():
    (
        session_id,
        first_turn_id,
        second_turn_id,
        original_prepared,
    ) = _seed_pending_verification_rebind()
    original_window_revision = _window_revision(session_id)

    resumed = verification_store.resume_task_node_verification(
        session_id=session_id,
        turn_id=second_turn_id,
        work_run_id="workrun-one",
        verification_request_id="verification-pending-rebind",
        expected_work_run_revision=4,
        expected_verification_request_revision=1,
        expected_window_revision=original_window_revision,
        apply_id="rebind-pending-verification",
    )
    assert resumed.status == "applied"
    assert resumed.verification_request_id == "verification-pending-rebind"
    assert resumed.verification_request_revision == 2
    assert resumed.verification_request_status.value == "pending"
    assert resumed.work_run_revision == 5
    assert resumed.work_run_status.value == "active"
    assert resumed.work_run_reason == "verification_pending"
    assert resumed.window_state_version == original_window_revision + 1

    record = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id="verification-pending-rebind",
    )
    assert record.request.request_turn_id == first_turn_id
    assert record.request.revision == 2
    prepared = verification_store.get_prepared_task_node_verification(
        session_id=session_id,
        invocation_turn_id=second_turn_id,
        verification_request_id="verification-pending-rebind",
    )
    assert prepared.invocation_turn_id == second_turn_id
    assert prepared.record.request == record.request
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    assert window["current_work_run_id"] == "workrun-one"
    assert window["current_attempt_id"] is None
    assert window["latest_checkpoint_id"] == "verification-pending-rebind"
    assert window["stage"] == "VERIFICATION"
    with store._connect() as conn:
        assert tuple(
            conn.execute(
                "SELECT session_id, relation FROM insession_work_run_turn_links "
                "WHERE turn_id=? AND work_run_id='workrun-one'",
                (second_turn_id,),
            ).fetchone()
        ) == (session_id, "continued")

    replayed = verification_store.resume_task_node_verification(
        session_id=session_id,
        turn_id=second_turn_id,
        work_run_id="workrun-one",
        verification_request_id="verification-pending-rebind",
        expected_work_run_revision=4,
        expected_verification_request_revision=1,
        expected_window_revision=original_window_revision,
        apply_id="rebind-pending-verification",
    )
    assert replayed.status == "replayed"
    assert replayed.model_copy(update={"status": "applied"}) == resumed
    assert _window_revision(session_id) == resumed.window_state_version

    with pytest.raises(work_run_store.WorkExecutionApplyIdCollision):
        verification_store.resume_task_node_verification(
            session_id=session_id,
            turn_id=second_turn_id,
            work_run_id="workrun-one",
            verification_request_id="verification-pending-rebind",
            expected_work_run_revision=5,
            expected_verification_request_revision=2,
            expected_window_revision=resumed.window_state_version,
            apply_id="rebind-pending-verification",
        )

    before_late_result_window = _window_revision(session_id)
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="semantic result does not match",
    ):
        verification_store.commit_task_node_verification_result(
            active_seconds_delta=1,
            session_id=session_id,
            turn_id=second_turn_id,
            work_run_id="workrun-one",
            verification_request_id="verification-pending-rebind",
            result=_verification_result(original_prepared, all_pass=True),
            expected_work_run_revision=5,
            expected_verification_request_revision=2,
            expected_window_revision=before_late_result_window,
            apply_id="late-pre-rebind-verification-result",
            delivery_id="late-pre-rebind-delivery",
        )
    assert _window_revision(session_id) == before_late_result_window
    assert _get(session_id).work_run.revision == 5
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_task_node_deliveries "
            "WHERE delivery_id='late-pre-rebind-delivery'"
        ).fetchone()[0] == 0


def test_pending_verification_rebind_rejects_stale_revision_and_binding_drift_without_writes():
    session_id, _, second_turn_id, _ = _seed_pending_verification_rebind()
    original_window_revision = _window_revision(session_id)

    with pytest.raises(verification_store.TaskNodeVerificationRequestRevisionConflict):
        verification_store.resume_task_node_verification(
            session_id=session_id,
            turn_id=second_turn_id,
            work_run_id="workrun-one",
            verification_request_id="verification-pending-rebind",
            expected_work_run_revision=4,
            expected_verification_request_revision=2,
            expected_window_revision=original_window_revision,
            apply_id="stale-pending-verification-rebind",
        )
    assert _window_revision(session_id) == original_window_revision
    assert _get(session_id).work_run.revision == 4
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id='workrun-one'",
            (second_turn_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_apply_receipts "
            "WHERE apply_id='stale-pending-verification-rebind'"
        ).fetchone()[0] == 0
        conn.execute(
            "UPDATE insession_work_run_verification_requests "
            "SET request_binding_hash=? WHERE verification_request_id=?",
            ("0" * 64, "verification-pending-rebind"),
        )

    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="immutable binding has drifted",
    ):
        verification_store.resume_task_node_verification(
            session_id=session_id,
            turn_id=second_turn_id,
            work_run_id="workrun-one",
            verification_request_id="verification-pending-rebind",
            expected_work_run_revision=4,
            expected_verification_request_revision=1,
            expected_window_revision=original_window_revision,
            apply_id="drifted-pending-verification-rebind",
        )
    assert _window_revision(session_id) == original_window_revision
    with store._connect() as conn:
        assert conn.execute(
            "SELECT revision FROM insession_work_runs "
            "WHERE work_run_id='workrun-one'"
        ).fetchone()[0] == 4
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id='workrun-one'",
            (second_turn_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_apply_receipts "
            "WHERE apply_id='drifted-pending-verification-rebind'"
        ).fetchone()[0] == 0


def test_technical_verification_interrupt_resumes_same_request_and_fences_late_result():
    session_id, first_turn_id, subject = _seed_task_node()
    _submit_for_verification(session_id, first_turn_id, subject)
    verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=first_turn_id,
        work_run_id="workrun-one",
        expected_work_run_revision=3,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=_window_revision(session_id),
        apply_id="prepare-interrupt",
        verification_request_id="verification-resumable",
    )
    original_prepared = verification_store.get_prepared_task_node_verification(
        session_id=session_id,
        invocation_turn_id=first_turn_id,
        verification_request_id="verification-resumable",
    )
    with store._connect() as conn:
        task_version_before = int(
            conn.execute(
                "SELECT state_version FROM insession_tasks WHERE insession_task_id=?",
                (subject.task_id,),
            ).fetchone()[0]
        )
    interrupted = verification_store.interrupt_task_node_verification(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=first_turn_id,
        work_run_id="workrun-one",
        verification_request_id="verification-resumable",
        technical_error_code="provider_timeout",
        expected_work_run_revision=4,
        expected_verification_request_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="interrupt-verification",
    )
    assert interrupted.verification_request_revision == 2
    assert interrupted.verification_request_status.value == "interrupted"
    assert interrupted.work_run_revision == 5
    assert interrupted.work_run_status.value == "interrupted"
    interrupted_window = store.get_turn_execution_window(session_id)
    assert interrupted_window is not None
    assert interrupted_window["stage"] == "VERIFICATION"
    assert interrupted_window["current_work_run_id"] == "workrun-one"
    assert interrupted_window["latest_checkpoint_id"] == "verification-resumable"
    with store._connect() as conn:
        assert conn.execute(
            "SELECT status FROM insession_task_node_states "
            "WHERE insession_task_id=? AND insession_task_node_id=?",
            (subject.task_id, subject.node_id),
        ).fetchone()[0] == "interrupted"
        task = conn.execute(
            "SELECT current_status, state_version FROM insession_tasks "
            "WHERE insession_task_id=?",
            (subject.task_id,),
        ).fetchone()
    assert task["current_status"] == "active"
    assert int(task["state_version"]) == task_version_before + 1

    marked = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=int(interrupted_window["state_version"]),
        stage="VERIFICATION",
        interruption_reason="provider_timeout",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=int(marked["state_version"]),
        end_reason="provider_timeout",
        error_code="MODEL_TRANSPORT_FAILURE",
    )
    settled_work_run = _get(session_id)
    assert (
        settled_work_run.current_verification_request_id
        == "verification-resumable"
    )
    assert settled_work_run.node_delivery_id is None
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="resume-verification-turn",
        source="runtime_test",
        user_text="继续",
        lease_owner="work-execution-test",
    )
    second_turn_id = str(accepted["turn"]["turn_id"])
    now = "2026-08-14T00:30:00+00:00"
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (session_id, second_turn_id, subject.task_id, now),
        )
        task_version_interrupted = int(
            conn.execute(
                "SELECT state_version FROM insession_tasks WHERE insession_task_id=?",
                (subject.task_id,),
            ).fetchone()[0]
        )

    resumed = verification_store.resume_task_node_verification(
        session_id=session_id,
        turn_id=second_turn_id,
        work_run_id="workrun-one",
        verification_request_id="verification-resumable",
        expected_work_run_revision=5,
        expected_verification_request_revision=2,
        expected_window_revision=_window_revision(session_id),
        apply_id="resume-verification",
    )
    assert resumed.verification_request_revision == 3
    assert resumed.verification_request_status.value == "pending"
    assert resumed.work_run_revision == 6
    resumed_record = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id="verification-resumable",
    )
    assert resumed_record.request.request_turn_id == first_turn_id
    assert resumed_record.request.revision == 3
    resumed_prepared = verification_store.get_prepared_task_node_verification(
        session_id=session_id,
        invocation_turn_id=second_turn_id,
        verification_request_id="verification-resumable",
    )
    assert resumed_prepared.invocation_turn_id == second_turn_id
    assert resumed_prepared.record.request.request_turn_id == first_turn_id
    resumed_window = store.get_turn_execution_window(session_id)
    assert resumed_window is not None
    assert resumed_window["stage"] == "VERIFICATION"
    assert resumed_window["latest_checkpoint_id"] == "verification-resumable"
    with store._connect() as conn:
        assert conn.execute(
            "SELECT relation FROM insession_work_run_turn_links "
            "WHERE turn_id=? AND work_run_id='workrun-one'",
            (second_turn_id,),
        ).fetchone()[0] == "continued"
        task_version_resumed = int(
            conn.execute(
                "SELECT state_version FROM insession_tasks WHERE insession_task_id=?",
                (subject.task_id,),
            ).fetchone()[0]
        )
    assert task_version_resumed == task_version_interrupted + 1

    before_stale_window = _window_revision(session_id)
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="semantic result does not match",
    ):
        verification_store.commit_task_node_verification_result(
            active_seconds_delta=1,
            session_id=session_id,
            turn_id=second_turn_id,
            work_run_id="workrun-one",
            verification_request_id="verification-resumable",
            result=_verification_result(original_prepared, all_pass=True),
            expected_work_run_revision=6,
            expected_verification_request_revision=3,
            expected_window_revision=before_stale_window,
            apply_id="late-old-generation-result",
            delivery_id="must-not-exist",
        )
    assert _window_revision(session_id) == before_stale_window
    assert _get(session_id).work_run.revision == 6
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_task_node_deliveries "
            "WHERE delivery_id='must-not-exist'"
        ).fetchone()[0] == 0


def test_verification_technical_error_code_boundary_is_closed_before_writes():
    session_id, turn_id, subject = _seed_task_node()
    _submit_for_verification(session_id, turn_id, subject)
    verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_work_run_revision=3,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=_window_revision(session_id),
        apply_id="prepare-error-code-boundary",
        verification_request_id="verification-error-code-boundary",
    )
    before_window = _window_revision(session_id)
    with pytest.raises(ValueError, match="at most 160"):
        verification_store.interrupt_task_node_verification(
            active_seconds_delta=1,
            session_id=session_id,
            turn_id=turn_id,
            work_run_id="workrun-one",
            verification_request_id="verification-error-code-boundary",
            technical_error_code="x" * 161,
            expected_work_run_revision=4,
            expected_verification_request_revision=1,
            expected_window_revision=before_window,
            apply_id="reject-long-error-code",
        )
    assert _window_revision(session_id) == before_window
    assert _get(session_id).work_run.revision == 4

    applied = verification_store.interrupt_task_node_verification(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        verification_request_id="verification-error-code-boundary",
        technical_error_code="x" * 160,
        expected_work_run_revision=4,
        expected_verification_request_revision=1,
        expected_window_revision=before_window,
        apply_id="accept-max-error-code",
    )
    assert applied.verification_request_status.value == "interrupted"
    record = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id="verification-error-code-boundary",
    )
    assert record.request.technical_error_code == "x" * 160


def test_session_purge_explicitly_releases_verified_delivery_aggregate():
    session_id, turn_id, subject = _seed_task_node()
    _submit_for_verification(session_id, turn_id, subject)
    verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_work_run_revision=3,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=_window_revision(session_id),
        apply_id="prepare-purge-delivery",
        verification_request_id="verification-purge-delivery",
    )
    prepared = verification_store.get_prepared_task_node_verification(
        session_id=session_id,
        invocation_turn_id=turn_id,
        verification_request_id="verification-purge-delivery",
    )
    verification_store.commit_task_node_verification_result(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        verification_request_id="verification-purge-delivery",
        result=_verification_result(prepared, all_pass=True),
        expected_work_run_revision=4,
        expected_verification_request_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="commit-purge-delivery",
        delivery_id="delivery-purge",
    )

    assert store.purge_session(session_id) is True
    assert store.get_session(session_id) is None
    with store._connect() as conn:
        for table in (
            "insession_task_node_deliveries",
            "insession_work_run_verification_requests",
            "insession_work_runs",
        ):
            assert conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE session_id=?",
                (session_id,),
            ).fetchone()[0] == 0
        for table in (
            "insession_work_run_output_windows",
            "insession_work_run_acceptance_progress",
            "insession_work_run_attempts",
        ):
            assert conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE work_run_id='workrun-one'"
            ).fetchone()[0] == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize("request_state", ["pending", "interrupted"])
def test_session_purge_releases_resumable_verification_checkpoint(request_state):
    session_id, turn_id, subject = _seed_task_node()
    _submit_for_verification(session_id, turn_id, subject)
    verification_store.prepare_task_node_verification(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_work_run_revision=3,
        expected_progress_revision=2,
        expected_output_revision=2,
        expected_window_revision=_window_revision(session_id),
        apply_id=f"prepare-purge-{request_state}",
        verification_request_id=f"verification-purge-{request_state}",
    )
    if request_state == "interrupted":
        verification_store.interrupt_task_node_verification(
            active_seconds_delta=1,
            session_id=session_id,
            turn_id=turn_id,
            work_run_id="workrun-one",
            verification_request_id="verification-purge-interrupted",
            technical_error_code="provider_timeout",
            expected_work_run_revision=4,
            expected_verification_request_revision=1,
            expected_window_revision=_window_revision(session_id),
            apply_id="interrupt-before-purge",
        )

    assert store.purge_session(session_id) is True
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_runs WHERE session_id=?",
            (session_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_verification_requests "
            "WHERE session_id=?",
            (session_id,),
        ).fetchone()[0] == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_loader_rejects_verification_pending_submit_triad_drift():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=1,
        apply_id="start-submit-triad",
        attempt_id="attempt-submit-triad",
    )
    work_run_store.commit_work_run_output_action(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-submit-triad",
        decision=_output_decision(
            content="final candidate",
            submit=True,
            updates=tuple(
                AcceptanceUpdate(
                    acceptance_id=acceptance_id,
                    model_claimed_satisfied=True,
                )
                for acceptance_id in ("deliverable", "quality")
            ),
        ),
        expected_work_run_revision=2,
        expected_progress_revision=1,
        expected_output_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="submit-triad",
    )

    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_runs SET reason=NULL "
            "WHERE work_run_id='workrun-one'"
        )
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="verification_pending WorkRun/Attempt/OutputWindow binding",
    ):
        _get(session_id)

    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_runs SET reason='verification_pending' "
            "WHERE work_run_id='workrun-one'"
        )
        decision_json = str(
            conn.execute(
                "SELECT decision_json FROM insession_work_run_attempts "
                "WHERE attempt_id='attempt-submit-triad'"
            ).fetchone()[0]
        )
        payload = json.loads(decision_json)
        payload["action"]["output_revision"] = 1
        conn.execute(
            "UPDATE insession_work_run_attempts SET committed_output_revision=1, "
            "submitted_output_revision=1, decision_json=? "
            "WHERE attempt_id='attempt-submit-triad'",
            (
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        )
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="verification_pending WorkRun/Attempt/OutputWindow binding",
    ):
        _get(session_id)


def test_completion_unconfirmed_closes_attempt_and_projects_waiting_external():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=1,
        apply_id="start-one",
        attempt_id="attempt-one",
    )
    work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-one",
        decision=_call_decision(modifies_environment=True),
        expected_work_run_revision=2,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="decision-one",
    )
    work_run_store.append_work_run_tool_result(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        result=ToolResult(
            status=ToolResultStatus.COMPLETION_UNCONFIRMED,
            tool_result_id="unknown-result",
            tool_call_id="call-one",
            attempt_id="attempt-one",
            ordinal=1,
            output=None,
            error_code="transport_lost",
            error_message="connection dropped after dispatch",
        ),
        expected_work_run_revision=3,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="append-unknown",
    )
    closed = work_run_store.close_work_run_attempt(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-one",
        expected_work_run_revision=4,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="close-unknown",
    )

    assert closed.work_run_status.value == "waiting_external"
    assert closed.work_run_reason == "operation_completion_unconfirmed"
    assert closed.attempt is not None and closed.attempt.status.value == "closed"
    record = _get(session_id)
    assert record.work_run.status.value == "waiting_external"
    assert record.work_run.reason == "operation_completion_unconfirmed"
    with store._connect() as conn:
        states = conn.execute(
            "SELECT task.current_status, node.status "
            "FROM insession_tasks AS task "
            "JOIN insession_task_node_states AS node "
            "ON node.insession_task_id=task.insession_task_id "
            "WHERE task.insession_task_id=? AND node.insession_task_node_id=?",
            (subject.task_id, subject.node_id),
        ).fetchone()
    assert tuple(states) == ("waiting_external", "waiting_external")
    with pytest.raises(work_run_store.WorkExecutionPersistenceError, match="active WorkRun"):
        _start(
            session_id,
            turn_id,
            "workrun-one",
            expected_run_revision=5,
            apply_id="unsafe-retry",
            attempt_id="attempt-two",
        )

    detached = work_run_store.detach_safe_work_run_lane(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        expected_window_revision=closed.window_state_version,
        apply_id="detach-completion-unconfirmed",
    )
    settled = store.advance_turn_execution_window(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=detached.window_state_version,
        stage="PERSIST",
        lease_owner="work-execution-test",
    )
    marked = store.mark_authoritative_no_public_turn_stop(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(settled["state_version"]),
    )
    assert marked["stop_kind"] == "waiting_external"
    assert marked["end_reason"] == "host_stopped"
    assert marked["error_code"] == "TOOL_COMPLETION_UNCONFIRMED"
    assert marked["stage"] == "TOOL"
    assert marked["window"]["window_state"] == "interrupted"
    replayed = store.mark_authoritative_no_public_turn_stop(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(marked["window"]["state_version"]),
    )
    assert replayed["replayed"] is True
    assert replayed["window"] == marked["window"]


def test_completion_unconfirmed_rejects_readonly_call_without_partial_write():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=1,
        apply_id="start-one",
        attempt_id="attempt-one",
    )
    decided = work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-one",
        decision=_call_decision(),
        expected_work_run_revision=2,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="decision-one",
    )
    before_window_revision = _window_revision(session_id)

    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="requires an environment-modifying ToolCall",
    ):
        work_run_store.append_work_run_tool_result(
            session_id=session_id,
            turn_id=turn_id,
            work_run_id="workrun-one",
            result=ToolResult(
                status=ToolResultStatus.COMPLETION_UNCONFIRMED,
                tool_result_id="unconfirmed-read-result",
                tool_call_id="call-one",
                attempt_id="attempt-one",
                ordinal=1,
                output=None,
                error_code="transport_lost",
                error_message="response missing",
            ),
            expected_work_run_revision=decided.work_run_revision,
            expected_progress_revision=decided.acceptance_progress_revision,
            expected_window_revision=before_window_revision,
            apply_id="append-unconfirmed-read",
        )

    record = _get(session_id)
    assert record.work_run.revision == decided.work_run_revision
    assert record.tool_results == ()
    assert _window_revision(session_id) == before_window_revision


def test_create_requires_completed_child_delivery_not_only_completed_state():
    session_id, turn_id, subject = _seed_task_node(acceptance_ids=("parent",))
    now = "2026-08-14T00:00:00+00:00"
    child_acceptance = json.dumps(
        [
            {
                "acceptance_id": "child",
                "criterion": "完成子节点",
                "source_anchor_ids": ["request"],
            }
        ],
        ensure_ascii=False,
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_tasks SET current_status='active' "
            "WHERE insession_task_id=?",
            (subject.task_id,),
        )
        conn.execute(
            "UPDATE insession_task_node_states SET status='active' "
            "WHERE insession_task_id=? AND insession_task_node_id=?",
            (subject.task_id, subject.node_id),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_nodes "
            "(insession_task_id, graph_revision, insession_task_node_id, node_revision, "
            "node_kind, ordinal, title, objective, source_anchor_ids_json, "
            "acceptance_criteria_json, constraints_json, created_at) "
            "VALUES (?, 1, 'child-one', 1, 'subtask', 1, '子节点', '完成子节点', "
            "'[\"request\"]', ?, '[]', ?)",
            (subject.task_id, child_acceptance, now),
        )
        conn.execute(
            "INSERT INTO insession_task_node_states "
            "(insession_task_id, insession_task_node_id, node_revision, status, "
            "state_version, updated_at) VALUES (?, 'child-one', 1, 'active', 1, ?)",
            (subject.task_id, now),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_edges "
            "(insession_task_id, graph_revision, child_insession_task_node_id, "
            "parent_insession_task_node_id, ordinal) VALUES (?, 1, 'child-one', ?, 1)",
            (subject.task_id, subject.node_id),
        )

    with pytest.raises(work_run_store.WorkExecutionPersistenceError, match="direct child"):
        _create(session_id, turn_id, subject, apply_id="not-ready")
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_task_node_states SET status='completed', "
            "state_version=state_version+1 WHERE insession_task_id=? "
            "AND insession_task_node_id='child-one'",
            (subject.task_id,),
        )

    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="completed current TaskNode has no unique Delivery",
    ):
        _create(session_id, turn_id, subject, apply_id="completed-without-delivery")


def test_session_purge_removes_new_work_execution_authority():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    _charge_active_time(
        session_id,
        turn_id,
        expected_run_revision=1,
        checkpoint_id="purged-budget-charge",
        active_seconds_delta=10,
        apply_id="charge-before-session-purge",
    )

    assert store.purge_session(session_id) is True
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_runs WHERE session_id=?",
            (session_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_output_windows WHERE session_id=?",
            (session_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_budget_charges "
            "WHERE session_id=?",
            (session_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_apply_receipts "
            "WHERE session_id=?",
            (session_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM runtime_turns WHERE session_id=?",
            (session_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM runtime_turn_inputs WHERE session_id=?",
            (session_id,),
        ).fetchone()[0] == 0


def test_loader_and_close_fail_closed_when_tool_authority_columns_drift_from_payloads():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=1,
        apply_id="start-one",
        attempt_id="attempt-one",
    )
    work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-one",
        decision=_call_decision(),
        expected_work_run_revision=2,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="decision-one",
    )
    work_run_store.append_work_run_tool_result(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        result=ToolResult(
            status=ToolResultStatus.SUCCEEDED,
            tool_result_id="result-one",
            tool_call_id="call-one",
            attempt_id="attempt-one",
            ordinal=1,
            output={"value": "observed"},
        ),
        expected_work_run_revision=3,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="append-one",
    )

    with store._connect() as conn:
        original_arguments = str(
            conn.execute(
                "SELECT arguments_json FROM insession_work_run_tool_calls "
                "WHERE tool_call_id='call-one'"
            ).fetchone()[0]
        )
        conn.execute(
            "UPDATE insession_work_run_tool_calls SET arguments_json='{}' "
            "WHERE tool_call_id='call-one'"
        )
    with pytest.raises(work_run_store.WorkExecutionPersistenceError, match="arguments are corrupt"):
        _get(session_id)

    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_run_tool_calls SET arguments_json=? "
            "WHERE tool_call_id='call-one'",
            (original_arguments,),
        )
        conn.execute(
            "UPDATE insession_work_run_tool_results SET status='completion_unconfirmed' "
            "WHERE tool_result_id='result-one'"
        )
    with pytest.raises(work_run_store.WorkExecutionPersistenceError, match="typed payload disagree"):
        _get(session_id)

    unconfirmed_payload = ToolResult(
        status=ToolResultStatus.COMPLETION_UNCONFIRMED,
        tool_result_id="result-one",
        tool_call_id="call-one",
        attempt_id="attempt-one",
        ordinal=1,
        output=None,
        error_code="transport_lost",
        error_message="response missing",
    ).model_dump_json()
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_run_tool_results SET result_json=? "
            "WHERE tool_result_id='result-one'",
            (unconfirmed_payload,),
        )
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="requires an environment-modifying ToolCall",
    ):
        _get(session_id)

    before_window = _window_revision(session_id)
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="requires an environment-modifying ToolCall",
    ):
        work_run_store.close_work_run_attempt(
            active_seconds_delta=1,
            session_id=session_id,
            turn_id=turn_id,
            work_run_id="workrun-one",
            attempt_id="attempt-one",
            expected_work_run_revision=4,
            expected_progress_revision=1,
            expected_window_revision=before_window,
            apply_id="corrupt-close",
        )
    assert _window_revision(session_id) == before_window
    with store._connect() as conn:
        run = conn.execute(
            "SELECT status, revision, current_attempt_id FROM insession_work_runs "
            "WHERE work_run_id='workrun-one'"
        ).fetchone()
        assert tuple(run) == ("active", 4, "attempt-one")


def test_tampered_prior_result_cannot_enter_acceptance_progress_allow_list():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=1,
        apply_id="start-one",
        attempt_id="attempt-one",
    )
    work_run_store.commit_work_run_attempt_decision(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-one",
        decision=_call_decision(),
        expected_work_run_revision=2,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="decision-one",
    )
    work_run_store.append_work_run_tool_result(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        result=ToolResult(
            status=ToolResultStatus.SUCCEEDED,
            tool_result_id="result-one",
            tool_call_id="call-one",
            attempt_id="attempt-one",
            ordinal=1,
            output={"value": "observed"},
        ),
        expected_work_run_revision=3,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="append-one",
    )
    work_run_store.close_work_run_attempt(
        active_seconds_delta=1,
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="workrun-one",
        attempt_id="attempt-one",
        expected_work_run_revision=4,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_id),
        apply_id="close-one",
    )
    _start(
        session_id,
        turn_id,
        "workrun-one",
        expected_run_revision=5,
        apply_id="start-two",
        attempt_id="attempt-two",
    )

    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_run_tool_results SET status='completion_unconfirmed' "
            "WHERE tool_result_id='result-one'"
        )

    before_window = _window_revision(session_id)
    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="typed payload disagree",
    ):
        work_run_store.commit_work_run_attempt_decision(
            session_id=session_id,
            turn_id=turn_id,
            work_run_id="workrun-one",
            attempt_id="attempt-two",
            decision=HostAcceptedAttemptDecision(
                acceptance_updates=(
                    AcceptanceUpdate(
                        acceptance_id="deliverable",
                        model_claimed_satisfied=True,
                        supporting_tool_result_ids=("result-one",),
                    ),
                ),
                action=RequestUserInputAction(question="continue"),
            ),
            expected_work_run_revision=6,
            expected_progress_revision=1,
            expected_window_revision=before_window,
            apply_id="tampered-history-decision",
            active_seconds_delta=1,
        )
    assert _window_revision(session_id) == before_window
    with store._connect() as conn:
        assert tuple(
            conn.execute(
                "SELECT revision, current_attempt_id FROM insession_work_runs "
                "WHERE work_run_id='workrun-one'"
            ).fetchone()
        ) == (6, "attempt-two")
        assert conn.execute(
            "SELECT progress_revision FROM insession_work_run_acceptance_progress "
            "WHERE work_run_id='workrun-one'"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT decision_json FROM insession_work_run_attempts "
            "WHERE attempt_id='attempt-two'"
        ).fetchone()[0] is None


def test_database_rejects_cross_session_chains_and_second_environment_change():
    session_one, turn_one, subject_one = _seed_task_node(
        request_id="request-one",
        task_id="task-one",
        node_id="node-one",
    )
    session_two, turn_two, subject_two = _seed_task_node(
        request_id="request-two",
        task_id="task-two",
        node_id="node-two",
    )
    _create(
        session_one,
        turn_one,
        subject_one,
        apply_id="create-one",
        work_run_id="workrun-one",
    )
    _create(
        session_two,
        turn_two,
        subject_two,
        apply_id="create-two",
        work_run_id="workrun-two",
    )
    _start(
        session_one,
        turn_one,
        "workrun-one",
        expected_run_revision=1,
        apply_id="start-one",
        attempt_id="attempt-one",
    )
    _start(
        session_two,
        turn_two,
        "workrun-two",
        expected_run_revision=1,
        apply_id="start-two",
        attempt_id="attempt-two",
    )

    with store._connect() as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO insession_work_run_turn_links "
            "(session_id, turn_id, work_run_id, link_revision, relation, created_at) "
            "VALUES (?, ?, 'workrun-one', 99, 'continued', ?)",
            (session_one, turn_two, "2026-08-14T00:02:00+00:00"),
        )
    with store._connect() as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO insession_work_run_tool_calls "
            "(tool_call_id, work_run_id, attempt_id, ordinal, provider_call_id, "
            "tool_id, tool_version, modifies_environment, arguments_hash, "
            "arguments_json, created_at) VALUES "
            "('cross-call', 'workrun-one', 'attempt-two', 1, NULL, 'read_test', "
            "'1.0.0', 0, ?, '{}', ?)",
            ("0" * 64, "2026-08-14T00:02:00+00:00"),
        )

    work_run_store.commit_work_run_attempt_decision(
        session_id=session_one,
        turn_id=turn_one,
        work_run_id="workrun-one",
        attempt_id="attempt-one",
        decision=HostAcceptedAttemptDecision(
            action=HostMaterializedCallToolsAction(
                calls=(
                    HostMaterializedToolCall(
                        tool_call_id="environment-call-one",
                        tool_id="write_test",
                        tool_version="1.0.0",
                        arguments={"value": 1},
                        modifies_environment=True,
                    ),
                )
            )
        ),
        expected_work_run_revision=2,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_one),
        apply_id="environment-decision",
    )
    with store._connect() as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO insession_work_run_tool_calls "
            "(tool_call_id, work_run_id, attempt_id, ordinal, provider_call_id, "
            "tool_id, tool_version, modifies_environment, arguments_hash, "
            "arguments_json, created_at) VALUES "
            "('environment-call-two', 'workrun-one', 'attempt-one', 2, NULL, "
            "'write_test', '1.0.0', 1, ?, '{}', ?)",
            ("0" * 64, "2026-08-14T00:03:00+00:00"),
        )


def test_database_binds_run_ownership_attempt_shape_and_result_ordinal():
    session_one, turn_one, subject_one = _seed_task_node(
        request_id="ownership-one",
        task_id="ownership-task-one",
        node_id="ownership-node-one",
        acceptance_ids=("deliverable",),
    )
    session_two, turn_two, _ = _seed_task_node(
        request_id="ownership-two",
        task_id="ownership-task-two",
        node_id="ownership-node-two",
        acceptance_ids=("deliverable",),
    )
    now = "2026-08-14T00:05:00+00:00"
    run_values = (
        "bad-owner-run",
        session_two,
        subject_one.task_id,
        subject_one.graph_revision,
        subject_one.node_id,
        subject_one.node_revision,
        turn_two,
        turn_two,
        now,
        now,
    )
    with store._connect() as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO insession_work_runs "
            "(work_run_id, session_id, subject_kind, insession_task_id, graph_revision, "
            "insession_task_node_id, node_revision, status, reason, revision, "
            "max_attempts, soft_active_seconds, hard_active_seconds, attempts_started, "
            "active_seconds_consumed, current_attempt_id, created_turn_id, "
            "updated_turn_id, created_at, updated_at) "
            "VALUES (?, ?, 'task_node', ?, ?, ?, ?, 'waiting_user', 'needs_input', "
            "2, 32, 720, 900, 0, 0, NULL, ?, ?, ?, ?)",
            run_values,
        )

    _create(
        session_one,
        turn_one,
        subject_one,
        apply_id="ownership-create",
        work_run_id="ownership-run",
    )
    with store._connect() as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO insession_work_run_attempts "
            "(attempt_id, work_run_id, turn_id, ordinal, status, action, "
            "decision_json, progress_revision_before, progress_revision_after, "
            "input_checkpoint_id, catalog_snapshot_json, catalog_snapshot_hash, "
            "budget_before_json, budget_after_json, close_reason, created_at, closed_at) "
            "VALUES ('invalid-closed', 'ownership-run', ?, 1, 'closed', "
            "'ready_for_verification', NULL, 1, NULL, NULL, '{}', ?, '{}', "
            "NULL, NULL, ?, ?)",
            (turn_one, "0" * 64, now, now),
        )
    with pytest.raises(sqlite3.IntegrityError):
        with store._connect() as conn:
            conn.execute(
                "UPDATE insession_work_runs SET current_attempt_id='missing-attempt' "
                "WHERE work_run_id='ownership-run'"
            )

    _start(
        session_one,
        turn_one,
        "ownership-run",
        expected_run_revision=1,
        apply_id="ownership-start",
        attempt_id="ownership-attempt",
    )
    work_run_store.commit_work_run_attempt_decision(
        session_id=session_one,
        turn_id=turn_one,
        work_run_id="ownership-run",
        attempt_id="ownership-attempt",
        decision=_call_decision(call_id="ownership-call"),
        expected_work_run_revision=2,
        expected_progress_revision=1,
        expected_window_revision=_window_revision(session_one),
        apply_id="ownership-decision",
    )
    wrong_ordinal = ToolResult(
        status=ToolResultStatus.SUCCEEDED,
        tool_result_id="wrong-ordinal-result",
        tool_call_id="ownership-call",
        attempt_id="ownership-attempt",
        ordinal=2,
        output={"value": "wrong ordinal"},
    )
    with store._connect() as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO insession_work_run_tool_results "
            "(tool_result_id, work_run_id, attempt_id, tool_call_id, ordinal, "
            "status, result_json, created_at) VALUES (?, 'ownership-run', "
            "'ownership-attempt', 'ownership-call', 2, 'succeeded', ?, ?)",
            (wrong_ordinal.tool_result_id, wrong_ordinal.model_dump_json(), now),
        )


def test_progress_subject_drift_is_not_projected_as_current_authority():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    with store._connect() as conn:
        row = conn.execute(
            "SELECT snapshot_json FROM insession_work_run_acceptance_progress "
            "WHERE work_run_id='workrun-one'"
        ).fetchone()
        payload = json.loads(str(row["snapshot_json"]))
        payload["subject"]["node_id"] = "different-node"
        snapshot_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        conn.execute(
            "UPDATE insession_work_run_acceptance_progress "
            "SET snapshot_json=?, snapshot_hash=? WHERE work_run_id='workrun-one'",
            (snapshot_json, work_execution_records._text_hash(snapshot_json)),
        )

    with pytest.raises(work_run_store.WorkExecutionPersistenceError, match="subject binding"):
        _get(session_id)


def test_nonterminal_subject_uniqueness_ignores_graph_revision_audit_field():
    session_id, turn_id, subject = _seed_task_node()
    _create(session_id, turn_id, subject)
    now = "2026-08-14T00:04:00+00:00"
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_graph_revisions "
            "(insession_task_id, graph_revision, source_turn_id, proposal_hash, "
            "source_anchors_json, authorization_anchor_ids_json, "
            "required_anchor_ids_json, created_at) "
            "VALUES (?, 2, ?, 'second-graph', '[]', '[]', '[]', ?)",
            (subject.task_id, turn_id, now),
        )
        conn.execute(
            "INSERT INTO insession_task_graph_nodes "
            "(insession_task_id, graph_revision, insession_task_node_id, node_revision, "
            "node_kind, ordinal, title, objective, source_anchor_ids_json, "
            "acceptance_criteria_json, constraints_json, created_at) "
            "SELECT insession_task_id, 2, insession_task_node_id, node_revision, "
            "node_kind, ordinal, title, objective, source_anchor_ids_json, "
            "acceptance_criteria_json, constraints_json, ? "
            "FROM insession_task_graph_nodes WHERE insession_task_id=? "
            "AND graph_revision=1 AND insession_task_node_id=?",
            (now, subject.task_id, subject.node_id),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO insession_work_runs "
                "(work_run_id, session_id, subject_kind, insession_task_id, "
                "graph_revision, insession_task_node_id, node_revision, status, reason, "
                "revision, max_attempts, soft_active_seconds, hard_active_seconds, "
                "attempts_started, active_seconds_consumed, current_attempt_id, "
                "created_turn_id, updated_turn_id, created_at, updated_at) "
                "VALUES ('duplicate-subject', ?, 'task_node', ?, 2, ?, 1, "
                "'waiting_user', 'needs_input', 2, 32, 720, 900, 0, 0, NULL, "
                "?, ?, ?, ?)",
                (
                    session_id,
                    subject.task_id,
                    subject.node_id,
                    turn_id,
                    turn_id,
                    now,
                    now,
                ),
            )


    NodeVerificationResult,
