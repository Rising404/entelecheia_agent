from __future__ import annotations

import hashlib
import json

import pytest

from personagraph.l2.task_graph.contracts import InSessionTaskAcceptanceProposal
from personagraph.l2.task_graph.task_matching import InSessionTaskMatchesProposal
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import continuation as continuation_store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.persistence.l2.auxiliary_graph import auxiliary_graphs
from personagraph.session.persistence import schema
from personagraph.session.persistence.turns.entry_tasks import list_entry_pending_task_questions
from personagraph.session.persistence.l2.task_graph import insession_tasks as insession_task_records
from personagraph.l2.work_run import (
    AcceptanceUpdate,
    AttemptDecision,
    AuxiliaryNodeSubject,
    HostAcceptedAttemptDecision,
    OutputWindowFormat,
    RequestUserInputAction,
    SubmitOutputWindowAction,
)


_QUESTION = "需要优先按方法还是实验组织计划？"
_ANSWER = "优先按实验组织，同时保留方法映射。"
_USER_TEXT = "请分析材料并形成一个可执行计划"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _window_revision(session_id: str) -> int:
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    return int(window["state_version"])


def _seed_active_attempt(
    prefix: str,
) -> tuple[str, str, str, AuxiliaryNodeSubject, object]:
    session_id = store.create_session("Entelecheia")
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=f"{prefix}-turn",
        source="auxiliary_v2_continuation_test",
        user_text=_USER_TEXT,
        lease_owner="auxiliary-v2-continuation-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    applied = task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=turn_id,
        apply_id=f"{prefix}-task",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "new_root",
                        "local_key": "root",
                        "title": "分析材料",
                        "objective": "分析材料并形成执行计划",
                        "source_excerpt": _USER_TEXT,
                    }
                ]
            }
        ),
        exposed_catalog_ids=(),
        expected_window_revision=_window_revision(session_id),
    )
    task_id = applied.created_insession_task_ids_by_local_key["root"]
    acceptance = InSessionTaskAcceptanceProposal(
        acceptance_id="grounded",
        criterion="输出必须受任务创建来源约束",
        source_anchor_ids=("task_creation_source",),
    )
    auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=1,
        expected_base_task_graph_revision=None,
        expected_control_state_version=None,
        expected_current_auxiliary_graph_revision=None,
        apply_id=f"{prefix}-graph",
        goal_objective="形成受来源约束的可执行计划",
        proposal=auxiliary_graphs.AuxiliaryGraphRevisionProposalRecord(
            revision_reason="initial",
            terminal_node_key="synthesize",
            nodes=(
                auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
                    local_node_key="investigate",
                    node_kind="analyze",
                    executor_kind="model_work_run",
                    title="调查材料",
                    objective="形成材料调查结论",
                    source_anchor_ids=("task_creation_source",),
                    acceptance_criteria=(acceptance,),
                    output_contract="planning_context_v1",
                    capability_profile_id="readonly_documents_v1",
                ),
                auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
                    local_node_key="synthesize",
                    node_kind="synthesize",
                    executor_kind="terminal_planner",
                    title="形成任务图",
                    objective="形成受约束的 TaskGraph 提案",
                    source_anchor_ids=("task_creation_source",),
                    acceptance_criteria=(acceptance,),
                    output_contract="task_graph_revision_proposal_v2",
                ),
            ),
            edges=(
                auxiliary_graphs.AuxiliaryGraphEdgeProposalRecord(
                    dependency_node_key="investigate",
                    consumer_node_key="synthesize",
                ),
            ),
        ),
        authority_context={"anchors": []},
        budget_profile={"profile_id": "planning-test-v1"},
        auxiliary_graph_id=f"{prefix}-graph",
        goal_id=f"{prefix}-goal",
    )
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    assert len(frontier.ready_fresh) == 1
    subject = frontier.ready_fresh[0].subject
    created = work_run_store.create_auxiliary_node_work_run(
        session_id=session_id,
        turn_id=turn_id,
        subject=subject,
        expected_task_state_version=frontier.task_state_version,
        expected_node_state_version=frontier.ready_fresh[0].node_state_version,
        expected_window_revision=_window_revision(session_id),
        apply_id=f"{prefix}-create-run",
        work_run_id=f"{prefix}-run",
    )
    started = work_run_store.start_work_run_attempt(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id=f"{prefix}-run",
        expected_work_run_revision=created.work_run_revision,
        expected_progress_revision=created.acceptance_progress_revision,
        expected_window_revision=created.window_state_version,
        apply_id=f"{prefix}-start-attempt",
        catalog_snapshot={"revision": 1, "tools": []},
        attempt_id=f"{prefix}-attempt-1",
    )
    return session_id, turn_id, task_id, subject, started


def _recoverable(session_id: str, turn_id: str, task_id: str):
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    assert len(frontier.recoverable) == 1
    return frontier, frontier.recoverable[0]


def test_current_schema_has_only_continuation_foreign_keys() -> None:
    store.init_db()
    with store._connect() as conn:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == (
            schema.SCHEMA_VERSION
        )
        tables = {
            str(row["name"])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {
            "insession_auxiliary_v2_execution_apply_receipts",
            "insession_auxiliary_v2_waiting_user_answer_bindings",
        } <= tables
        {
            str(row["table"])
            for table in (
                "insession_auxiliary_v2_execution_apply_receipts",
                "insession_auxiliary_v2_waiting_user_answer_bindings",
            )
            for row in conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        }
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_question_answer_continuation_is_atomic_replayable_and_source_bound() -> None:
    session_id, question_turn_id, task_id, subject, started = _seed_active_attempt(
        "wait"
    )
    frontier, candidate = _recoverable(session_id, question_turn_id, task_id)
    decision = HostAcceptedAttemptDecision(
        acceptance_updates=(),
        action=RequestUserInputAction(question=_QUESTION),
    )
    command = continuation_store.CommitAuxiliaryWaitingUserAttemptCommand(
        session_id=session_id,
        turn_id=question_turn_id,
        work_run_id="wait-run",
        attempt_id="wait-attempt-1",
        subject=subject,
        decision=decision,
        expected_work_run_revision=started.work_run_revision,
        expected_progress_revision=started.acceptance_progress_revision,
        expected_window_revision=started.window_state_version,
        expected_task_state_version=frontier.task_state_version,
        expected_node_state_version=candidate.node_state_version,
        expected_control_state_version=frontier.control_state_version,
        expected_goal_state_version=frontier.goal_state_version,
        expected_revision_state_version=frontier.revision_state_version,
        apply_id="wait-question",
        active_seconds_delta=1,
    )
    with pytest.raises(work_run_store.WorkExecutionRevisionConflict):
        continuation_store.commit_auxiliary_waiting_user_attempt(
            command=command.model_copy(
                update={
                    "expected_goal_state_version": (
                        command.expected_goal_state_version + 1
                    ),
                    "apply_id": "wait-stale-goal",
                }
            )
        )
    before_wait = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id="wait-run",
    )
    assert before_wait.work_run.status.value == "active"
    assert before_wait.current_attempt_id == "wait-attempt-1"

    waiting = continuation_store.commit_auxiliary_waiting_user_attempt(command=command)
    assert waiting.status == "applied"
    assert waiting.work_run_status.value == "waiting_user"
    assert waiting.work_run_reason == "needs_input"
    assert waiting.execution_subject_contract_version == "auxiliary_node_v2"
    assert waiting.budget_transition is not None
    assert waiting.budget_transition.budget_after.active_seconds_consumed == 1

    replayed = continuation_store.commit_auxiliary_waiting_user_attempt(command=command)
    assert replayed.status == "replayed"
    assert replayed.model_copy(update={"status": "applied"}) == waiting
    with pytest.raises(continuation_store.AuxiliaryContinuationApplyIdCollision):
        continuation_store.commit_auxiliary_waiting_user_attempt(
            command=command.model_copy(update={"active_seconds_delta": 2})
        )

    pending = continuation_store.get_auxiliary_pending_user_question(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert pending is not None
    assert pending.question == _QUESTION
    assert pending.question_sha256 == _sha256(_QUESTION)
    assert pending.work_run_revision == waiting.work_run_revision
    assert pending.task_state_version == waiting.task_state_version
    assert pending.node_state_version == waiting.node_state_version
    assert pending.goal_state_version == waiting.goal_state_version
    assert pending.revision_state_version == waiting.revision_state_version
    assert list_entry_pending_task_questions(
        store._deps(),
        session_id=session_id,
    )[0].question == _QUESTION

    with store._connect() as conn:
        aggregate = conn.execute(
            "SELECT task.current_status, node.status, goal.status, rev.status "
            "FROM insession_tasks AS task "
            "JOIN insession_auxiliary_graph_v2_containers AS control "
            "ON control.session_id=task.session_id "
            "AND control.insession_task_id=task.insession_task_id "
            "JOIN insession_auxiliary_graph_goals AS goal "
            "ON goal.goal_id=control.current_goal_id "
            "JOIN insession_auxiliary_graph_revision_states_v2 AS rev "
            "ON rev.auxiliary_graph_id=control.auxiliary_graph_id "
            "AND rev.auxiliary_graph_revision="
            "control.current_auxiliary_graph_revision "
            "JOIN insession_auxiliary_node_states_v2 AS node "
            "ON node.auxiliary_graph_id=control.auxiliary_graph_id "
            "AND node.auxiliary_graph_revision="
            "control.current_auxiliary_graph_revision "
            "WHERE task.insession_task_id=? AND node.auxiliary_node_id=?",
            (task_id, subject.node_id),
        ).fetchone()
        assert tuple(aggregate) == (
            "awaiting_user",
            "waiting_user",
            "waiting_user",
            "waiting_user",
        )
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_budget_charges "
            "WHERE work_run_id='wait-run'"
        ).fetchone()[0] == 1

    detached = work_run_store.detach_safe_work_run_lane(
        session_id=session_id,
        turn_id=question_turn_id,
        work_run_id="wait-run",
        expected_window_revision=waiting.window_state_version,
        apply_id="wait-detach-question-lane",
    )
    finalized = store.finalize_authoritative_referenced_turn_execution(
        session_id=session_id,
        turn_id=question_turn_id,
        expected_window_revision=detached.window_state_version,
        post_commit_job_kinds=(),
    )
    assert finalized["pending_question_attempt_ids"] == ("wait-attempt-1",)
    assert finalized["formal_reference_items"] == (
        store.FormalTurnReference(
            "pending_question_attempt",
            "wait-attempt-1",
        ),
    )
    assert store.get_turn(session_id, 1)["content"] == _QUESTION
    store.release_turn_execution_window(
        session_id=session_id,
        turn_id=question_turn_id,
        expected_window_revision=int(finalized["window"]["state_version"]),
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="wait-answer-turn",
        source="auxiliary_v2_continuation_test",
        user_text=_ANSWER,
        lease_owner="auxiliary-v2-continuation-test",
    )
    answer_turn_id = str(accepted["turn"]["turn_id"])
    insession_task_records.link_turn_to_insession_tasks(
        store._deps(),
        session_id=session_id,
        turn_id=answer_turn_id,
        insession_task_ids=(task_id,),
        expected_window_revision=_window_revision(session_id),
    )
    answer = store.get_turn_execution_input(
        session_id=session_id,
        turn_id=answer_turn_id,
    )
    continue_command = continuation_store.ContinueAuxiliaryWaitingUserCommand(
        session_id=session_id,
        turn_id=answer_turn_id,
        work_run_id="wait-run",
        subject=subject,
        question_attempt_id=pending.question_attempt_id,
        expected_question_sha256=pending.question_sha256,
        expected_answer_source_sha256=_sha256(str(answer["content"])),
        expected_work_run_revision=pending.work_run_revision,
        expected_progress_revision=pending.acceptance_progress_revision,
        expected_window_revision=_window_revision(session_id),
        expected_task_state_version=pending.task_state_version,
        expected_node_state_version=pending.node_state_version,
        expected_control_state_version=pending.control_state_version,
        expected_goal_state_version=pending.goal_state_version,
        expected_revision_state_version=pending.revision_state_version,
        apply_id="wait-consume-answer",
        catalog_snapshot={"revision": 2, "tools": []},
        attempt_id="wait-attempt-2",
    )
    with pytest.raises(
        continuation_store.AuxiliaryContinuationPersistenceError,
        match="pending question changed",
    ):
        continuation_store.continue_auxiliary_waiting_user_and_start_attempt(
            command=continue_command.model_copy(
                update={
                    "expected_question_sha256": "0" * 64,
                    "apply_id": "wait-wrong-question-hash",
                }
            )
        )
    with pytest.raises(
        continuation_store.AuxiliaryContinuationPersistenceError,
        match="answer source changed",
    ):
        continuation_store.continue_auxiliary_waiting_user_and_start_attempt(
            command=continue_command.model_copy(
                update={
                    "expected_answer_source_sha256": "0" * 64,
                    "apply_id": "wait-wrong-answer-hash",
                }
            )
        )
    before_continue = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id="wait-run",
    )
    assert before_continue.work_run.status.value == "waiting_user"
    assert len(before_continue.attempts) == 1

    continued = continuation_store.continue_auxiliary_waiting_user_and_start_attempt(
        command=continue_command
    )
    assert continued.status == "applied"
    assert continued.work_run_status.value == "active"
    assert continued.answer_source_binding is not None
    assert continued.answer_source_binding.answer_source_turn_id == answer_turn_id
    assert continued.answer_source_binding.answer_source_message_id == answer["message_id"]
    assert (
        continued.answer_source_binding.answer_source_content_sha256
        == _sha256(_ANSWER)
    )
    record = work_run_store.get_work_run(session_id=session_id, work_run_id="wait-run")
    assert record.attempts[-1].attempt.ordinal == 2
    assert record.attempts[-1].turn_id == answer_turn_id
    assert record.attempts[-1].input_turn_id == answer_turn_id
    assert record.attempts[-1].predecessor_question_attempt_id == "wait-attempt-1"

    replayed_continue = continuation_store.continue_auxiliary_waiting_user_and_start_attempt(
        command=continue_command
    )
    assert replayed_continue.status == "replayed"
    assert (
        replayed_continue.model_copy(update={"status": "applied"}) == continued
    )
    with pytest.raises(continuation_store.AuxiliaryContinuationApplyIdCollision):
        continuation_store.continue_auxiliary_waiting_user_and_start_attempt(
            command=continue_command.model_copy(
                update={"catalog_snapshot": {"revision": 999, "tools": []}}
            )
        )

    with store._connect() as conn:
        receipt = conn.execute(
            "SELECT result_json FROM "
            "insession_auxiliary_v2_execution_apply_receipts "
            "WHERE apply_id='wait-consume-answer'"
        ).fetchone()[0]
        assert _ANSWER not in receipt
        assert _QUESTION not in receipt
        conn.execute(
            "UPDATE session_turns SET content='篡改后的回答' "
            "WHERE session_id=? AND turn_idx=? AND role='user'",
            (session_id, int(answer["turn_idx"])),
        )
    with pytest.raises(
        continuation_store.AuxiliaryContinuationPersistenceError,
        match="binding|source",
    ):
        work_run_store.get_work_run(session_id=session_id, work_run_id="wait-run")
    with pytest.raises(
        continuation_store.AuxiliaryContinuationPersistenceError,
        match="binding|source",
    ):
        continuation_store.continue_auxiliary_waiting_user_and_start_attempt(
            command=continue_command
        )
    other_session_id = store.create_session("Entelecheia")
    assert store.purge_session(session_id) is True
    assert store.get_session(other_session_id) is not None
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM "
            "insession_auxiliary_v2_waiting_user_answer_bindings "
            "WHERE session_id=?",
            (session_id,),
        ).fetchone()[0] == 0


def test_undecided_attempt_rebinds_once_to_an_exact_new_turn() -> None:
    session_id, first_turn_id, task_id, subject, started = _seed_active_attempt(
        "resume"
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
        client_request_id="resume-next-turn",
        source="auxiliary_v2_continuation_test",
        user_text="继续执行",
        lease_owner="auxiliary-v2-continuation-test",
    )
    second_turn_id = str(accepted["turn"]["turn_id"])
    insession_task_records.link_turn_to_insession_tasks(
        store._deps(),
        session_id=session_id,
        turn_id=second_turn_id,
        insession_task_ids=(task_id,),
        expected_window_revision=_window_revision(session_id),
    )
    frontier, candidate = _recoverable(session_id, second_turn_id, task_id)
    command = continuation_store.ResumeAuxiliaryActiveAttemptCommand(
        session_id=session_id,
        turn_id=second_turn_id,
        work_run_id="resume-run",
        attempt_id="resume-attempt-1",
        subject=subject,
        expected_work_run_revision=started.work_run_revision,
        expected_progress_revision=started.acceptance_progress_revision,
        expected_window_revision=_window_revision(session_id),
        expected_task_state_version=frontier.task_state_version,
        expected_node_state_version=candidate.node_state_version,
        expected_control_state_version=frontier.control_state_version,
        expected_goal_state_version=frontier.goal_state_version,
        expected_revision_state_version=frontier.revision_state_version,
        apply_id="resume-rebind",
    )
    current_payload = command.model_dump(mode="json")
    assert current_payload["catalog_snapshot"] is None
    assert current_payload["allow_protected_recovery"] is False
    current_hash = _canonical_sha256(current_payload)
    old_payload = dict(current_payload)
    old_payload.pop("catalog_snapshot")
    old_payload.pop("allow_protected_recovery")
    old_hash = _canonical_sha256(old_payload)
    assert old_hash != current_hash

    resumed = continuation_store.resume_auxiliary_active_attempt(command=command)
    assert resumed.status == "applied"
    assert resumed.work_run_revision == started.work_run_revision + 1
    assert resumed.task_state_version == frontier.task_state_version
    assert resumed.node_state_version == candidate.node_state_version
    record = work_run_store.get_work_run(session_id=session_id, work_run_id="resume-run")
    assert record.attempts[-1].turn_id == second_turn_id
    assert record.attempts[-1].input_turn_id == first_turn_id

    replayed = continuation_store.resume_auxiliary_active_attempt(command=command)
    assert replayed.status == "replayed"
    assert replayed.model_copy(update={"status": "applied"}) == resumed
    with pytest.raises(continuation_store.AuxiliaryContinuationApplyIdCollision):
        continuation_store.resume_auxiliary_active_attempt(
            command=command.model_copy(update={"attempt_id": "different-attempt"})
        )
    with store._connect() as conn:
        assert conn.execute(
            "SELECT payload_sha256 FROM "
            "insession_auxiliary_v2_execution_apply_receipts "
            "WHERE apply_id=?",
            (command.apply_id,),
        ).fetchone()[0] == current_hash
        conn.execute(
            "UPDATE insession_auxiliary_v2_execution_apply_receipts "
            "SET payload_sha256=? WHERE apply_id=?",
            (old_hash, command.apply_id),
        )
    with pytest.raises(continuation_store.AuxiliaryContinuationApplyIdCollision):
        continuation_store.resume_auxiliary_active_attempt(command=command)


def test_question_hard_budget_is_charged_once_and_terminates_atomically() -> None:
    session_id, turn_id, task_id, subject, started = _seed_active_attempt("hard")
    frontier, candidate = _recoverable(session_id, turn_id, task_id)
    command = continuation_store.CommitAuxiliaryWaitingUserAttemptCommand(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="hard-run",
        attempt_id="hard-attempt-1",
        subject=subject,
        decision=HostAcceptedAttemptDecision(
            acceptance_updates=(),
            action=RequestUserInputAction(question=_QUESTION),
        ),
        expected_work_run_revision=started.work_run_revision,
        expected_progress_revision=started.acceptance_progress_revision,
        expected_window_revision=started.window_state_version,
        expected_task_state_version=frontier.task_state_version,
        expected_node_state_version=candidate.node_state_version,
        expected_control_state_version=frontier.control_state_version,
        expected_goal_state_version=frontier.goal_state_version,
        expected_revision_state_version=frontier.revision_state_version,
        apply_id="hard-question",
        active_seconds_delta=900,
    )

    failed = continuation_store.commit_auxiliary_waiting_user_attempt(command=command)
    assert failed.status == "applied"
    assert failed.work_run_status.value == "failed"
    assert failed.work_run_reason == "work_run_limit_reached"
    assert failed.current_attempt_id is None
    assert failed.budget_transition is not None
    assert failed.budget_transition.disposition.value == "hard_limit_reached"
    assert failed.budget_transition.budget_after.active_seconds_consumed == 900
    assert failed.window_state_version == started.window_state_version + 1
    assert failed.task_state_version == frontier.task_state_version + 1
    assert failed.node_state_version == candidate.node_state_version + 1
    assert failed.control_state_version == frontier.control_state_version
    assert failed.goal_state_version == frontier.goal_state_version + 1
    assert failed.revision_state_version == frontier.revision_state_version + 1

    record = work_run_store.get_work_run(session_id=session_id, work_run_id="hard-run")
    assert record.work_run.status.value == "failed"
    assert record.work_run.reason == "work_run_limit_reached"
    assert record.work_run.budget.active_seconds_consumed == 900
    assert record.attempts[-1].attempt.status.value == "closed"
    assert record.attempts[-1].action == "request_user_input"
    assert record.attempts[-1].budget_charge_id == "hard-question"
    assert record.pending_user_question is None
    assert continuation_store.get_auxiliary_pending_user_question(
        session_id=session_id,
        insession_task_id=task_id,
    ) is None

    replayed = continuation_store.commit_auxiliary_waiting_user_attempt(command=command)
    assert replayed.status == "replayed"
    assert replayed.model_copy(update={"status": "applied"}) == failed
    with pytest.raises(continuation_store.AuxiliaryContinuationApplyIdCollision):
        continuation_store.commit_auxiliary_waiting_user_attempt(
            command=command.model_copy(update={"active_seconds_delta": 899})
        )

    with store._connect() as conn:
        window = conn.execute(
            "SELECT current_work_run_id, current_attempt_id, state_version "
            "FROM turn_execution_windows WHERE session_id=? AND turn_id=?",
            (session_id, turn_id),
        ).fetchone()
        assert tuple(window) == (
            "hard-run",
            None,
            started.window_state_version + 1,
        )
        aggregate = conn.execute(
            "SELECT task.current_status, node.status, goal.status, rev.status "
            "FROM insession_tasks AS task "
            "JOIN insession_auxiliary_graph_v2_containers AS control "
            "ON control.session_id=task.session_id "
            "AND control.insession_task_id=task.insession_task_id "
            "JOIN insession_auxiliary_graph_goals AS goal "
            "ON goal.goal_id=control.current_goal_id "
            "JOIN insession_auxiliary_graph_revision_states_v2 AS rev "
            "ON rev.auxiliary_graph_id=control.auxiliary_graph_id "
            "AND rev.auxiliary_graph_revision="
            "control.current_auxiliary_graph_revision "
            "JOIN insession_auxiliary_node_states_v2 AS node "
            "ON node.auxiliary_graph_id=control.auxiliary_graph_id "
            "AND node.auxiliary_graph_revision="
            "control.current_auxiliary_graph_revision "
            "WHERE task.insession_task_id=? AND node.auxiliary_node_id=?",
            (task_id, subject.node_id),
        ).fetchone()
        assert tuple(aggregate) == (
            "interrupted",
            "interrupted",
            "interrupted",
            "interrupted",
        )
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_budget_charges "
            "WHERE budget_charge_id='hard-question'"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_run_apply_receipts "
            "WHERE apply_id='hard-question'"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM "
            "insession_auxiliary_v2_execution_apply_receipts "
            "WHERE apply_id='hard-question'"
        ).fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_output_hard_budget_projects_only_the_current_aggregate() -> None:
    session_id, turn_id, task_id, subject, started = _seed_active_attempt(
        "output-hard"
    )

    failed = work_run_store.commit_work_run_output_action(
        session_id=session_id,
        turn_id=turn_id,
        work_run_id="output-hard-run",
        attempt_id="output-hard-attempt-1",
        decision=AttemptDecision(
            acceptance_updates=(
                AcceptanceUpdate(
                    acceptance_id="grounded",
                    model_claimed_satisfied=True,
                ),
            ),
            action=SubmitOutputWindowAction(
                content="输出仍受任务创建来源约束。",
                format=OutputWindowFormat.PLAIN_TEXT,
            ),
        ),
        expected_work_run_revision=started.work_run_revision,
        expected_progress_revision=started.acceptance_progress_revision,
        expected_output_revision=started.output_window_revision,
        expected_window_revision=started.window_state_version,
        apply_id="output-hard-submit",
        active_seconds_delta=900,
    )

    assert failed.work_run_status.value == "failed"
    assert failed.work_run_reason == "work_run_limit_reached"
    with store._connect() as conn:
        aggregate = conn.execute(
            "SELECT task.current_status, node.status, goal.status, rev.status "
            "FROM insession_tasks AS task "
            "JOIN insession_auxiliary_graph_v2_containers AS control "
            "ON control.session_id=task.session_id "
            "AND control.insession_task_id=task.insession_task_id "
            "JOIN insession_auxiliary_graph_goals AS goal "
            "ON goal.goal_id=control.current_goal_id "
            "JOIN insession_auxiliary_graph_revision_states_v2 AS rev "
            "ON rev.auxiliary_graph_id=control.auxiliary_graph_id "
            "AND rev.auxiliary_graph_revision="
            "control.current_auxiliary_graph_revision "
            "JOIN insession_auxiliary_node_states_v2 AS node "
            "ON node.auxiliary_graph_id=control.auxiliary_graph_id "
            "AND node.auxiliary_graph_revision="
            "control.current_auxiliary_graph_revision "
            "WHERE task.insession_task_id=? AND node.auxiliary_node_id=?",
            (task_id, subject.node_id),
        ).fetchone()
        assert tuple(aggregate) == (
            "interrupted",
            "interrupted",
            "interrupted",
            "interrupted",
        )
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
