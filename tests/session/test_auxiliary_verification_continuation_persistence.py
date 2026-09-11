from __future__ import annotations

import json
import hashlib

import pytest

from personagraph.session import store
from personagraph.session.l2_store import continuation as continuation_store
from personagraph.session.l2_store import verification as verification_store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.session.persistence.l2.auxiliary_graph import auxiliary_graphs
from personagraph.session.persistence import schema
from personagraph.session.persistence.l2.task_graph import insession_tasks as insession_task_records
from personagraph.l2.work_run import (
    AcceptanceVerificationFeedback,
    AcceptanceUpdate,
    AttemptDecision,
    NodeVerificationResult,
    OutputWindowFormat,
    SubmitOutputWindowAction,
    VerificationVerdict,
)
from tests.session.test_auxiliary_continuation_persistence import (
    _recoverable,
    _seed_active_attempt,
    _window_revision,
)


def _seed_pending_verification(prefix: str):
    session_id, first_turn_id, task_id, subject, started = _seed_active_attempt(
        prefix
    )
    submitted = work_run_store.commit_work_run_output_action(
        session_id=session_id,
        turn_id=first_turn_id,
        work_run_id=f"{prefix}-run",
        attempt_id=f"{prefix}-attempt-1",
        decision=AttemptDecision(
            acceptance_updates=(
                AcceptanceUpdate(
                    acceptance_id="grounded",
                    model_claimed_satisfied=True,
                ),
            ),
            action=SubmitOutputWindowAction(
                content="材料已经按来源约束完成分析。",
                format=OutputWindowFormat.PLAIN_TEXT,
            ),
        ),
        expected_work_run_revision=started.work_run_revision,
        expected_progress_revision=started.acceptance_progress_revision,
        expected_output_revision=started.output_window_revision,
        expected_window_revision=started.window_state_version,
        apply_id=f"{prefix}-submit",
        active_seconds_delta=2,
    )
    prepared_mutation = verification_store.prepare_auxiliary_node_verification(
        session_id=session_id,
        turn_id=first_turn_id,
        work_run_id=f"{prefix}-run",
        expected_work_run_revision=submitted.work_run_revision,
        expected_progress_revision=submitted.acceptance_progress_revision,
        expected_output_revision=submitted.output_window_revision,
        expected_window_revision=submitted.window_state_version,
        apply_id=f"{prefix}-prepare",
        verification_request_id=f"{prefix}-verification",
    )
    original_prepared = verification_store.get_prepared_auxiliary_node_verification(
        session_id=session_id,
        invocation_turn_id=first_turn_id,
        verification_request_id=f"{prefix}-verification",
    )
    with store._connect() as conn:
        request_before = dict(
            conn.execute(
                "SELECT * FROM insession_work_run_verification_requests "
                "WHERE verification_request_id=?",
                (f"{prefix}-verification",),
            ).fetchone()
        )
        budget_before = tuple(
            conn.execute(
                "SELECT attempts_started, active_seconds_consumed "
                "FROM insession_work_runs WHERE work_run_id=?",
                (f"{prefix}-run",),
            ).fetchone()
        )
        charge_count_before = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_work_run_budget_charges "
                "WHERE work_run_id=?",
                (f"{prefix}-run",),
            ).fetchone()[0]
        )
    marked = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=prepared_mutation.window_state_version,
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
        client_request_id=f"{prefix}-resume-turn",
        source="auxiliary_v2_verification_continuation_test",
        user_text="继续完成同一个验证请求",
        lease_owner="auxiliary-v2-verification-continuation-test",
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
    details = auxiliary_graphs.get_auxiliary_graph_for_task(
        store._deps(),
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert details is not None
    command = continuation_store.ResumeAuxiliaryVerificationCommand(
        session_id=session_id,
        turn_id=second_turn_id,
        work_run_id=f"{prefix}-run",
        verification_request_id=f"{prefix}-verification",
        subject=subject,
        expected_structure_sha256=details.structure_sha256,
        expected_work_run_revision=prepared_mutation.work_run_revision,
        expected_verification_request_revision=1,
        expected_window_revision=_window_revision(session_id),
        expected_task_state_version=frontier.task_state_version,
        expected_node_state_version=candidate.node_state_version,
        expected_control_state_version=frontier.control_state_version,
        expected_goal_state_version=frontier.goal_state_version,
        expected_revision_state_version=frontier.revision_state_version,
        apply_id=f"{prefix}-resume-verification",
    )
    return {
        "session_id": session_id,
        "first_turn_id": first_turn_id,
        "second_turn_id": second_turn_id,
        "task_id": task_id,
        "subject": subject,
        "prepared_mutation": prepared_mutation,
        "original_prepared": original_prepared,
        "request_before": request_before,
        "budget_before": budget_before,
        "charge_count_before": charge_count_before,
        "command": command,
    }


def _resume_state(seeded: dict[str, object]) -> dict[str, object]:
    command = seeded["command"]
    assert isinstance(command, continuation_store.ResumeAuxiliaryVerificationCommand)
    with store._connect() as conn:
        return {
            "run": tuple(
                conn.execute(
                    "SELECT revision, status, reason, updated_turn_id, "
                    "attempts_started, active_seconds_consumed, "
                    "current_attempt_id, current_verification_request_id "
                    "FROM insession_work_runs WHERE work_run_id=?",
                    (command.work_run_id,),
                ).fetchone()
            ),
            "request": tuple(
                conn.execute(
                    "SELECT request_revision, status, request_turn_id, "
                    "submitted_attempt_id, output_revision, request_binding_hash "
                    "FROM insession_work_run_verification_requests "
                    "WHERE verification_request_id=?",
                    (command.verification_request_id,),
                ).fetchone()
            ),
            "window": tuple(
                conn.execute(
                    "SELECT turn_id, current_work_run_id, current_attempt_id, "
                    "latest_checkpoint_id, stage, state_version "
                    "FROM turn_execution_windows WHERE session_id=?",
                    (command.session_id,),
                ).fetchone()
            ),
            "new_link_count": int(
                conn.execute(
                    "SELECT COUNT(*) FROM insession_work_run_turn_links "
                    "WHERE turn_id=? AND work_run_id=?",
                    (command.turn_id, command.work_run_id),
                ).fetchone()[0]
            ),
            "resume_receipt_count": int(
                conn.execute(
                    "SELECT COUNT(*) FROM "
                    "insession_auxiliary_v2_execution_apply_receipts "
                    "WHERE operation='resume_verification' AND work_run_id=?",
                    (command.work_run_id,),
                ).fetchone()[0]
            ),
            "budget_charge_count": int(
                conn.execute(
                    "SELECT COUNT(*) FROM insession_work_run_budget_charges "
                    "WHERE work_run_id=?",
                    (command.work_run_id,),
                ).fetchone()[0]
            ),
        }


def _insert_answer_message_marker(
    seeded: dict[str, object],
    *,
    marker: str,
) -> None:
    command = seeded["command"]
    assert isinstance(command, continuation_store.ResumeAuxiliaryVerificationCommand)
    answer_input = store.get_turn_execution_input(
        session_id=command.session_id,
        turn_id=command.turn_id,
    )
    result_json = "{}"
    result_sha256 = hashlib.sha256(result_json.encode()).hexdigest()
    with store._connect() as conn:
        authority = conn.execute(
            "SELECT run.execution_subject_id, control.current_goal_id "
            "FROM insession_work_runs AS run "
            "JOIN insession_auxiliary_graph_v2_containers AS control "
            "ON control.session_id=run.session_id "
            "AND control.insession_task_id=run.insession_task_id "
            "AND control.auxiliary_graph_id=run.auxiliary_graph_id "
            "WHERE run.work_run_id=?",
            (command.work_run_id,),
        ).fetchone()
        conn.execute(
            "INSERT INTO insession_auxiliary_v2_execution_apply_receipts "
            "(apply_id, operation, session_id, insession_task_id, work_run_id, "
            "execution_subject_id, auxiliary_graph_id, goal_id, "
            "auxiliary_graph_revision, auxiliary_node_id, node_revision, "
            "invocation_turn_id, payload_sha256, result_json, result_sha256, "
            "created_at) VALUES (?, 'resume_active_attempt', ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"{marker}-receipt",
                command.session_id,
                command.subject.task_id,
                command.work_run_id,
                str(authority["execution_subject_id"]),
                command.subject.auxiliary_graph_id,
                str(authority["current_goal_id"]),
                command.subject.auxiliary_graph_revision,
                command.subject.node_id,
                command.subject.node_revision,
                command.turn_id,
                "a" * 64,
                result_json,
                result_sha256,
                "2026-08-22T00:00:00+00:00",
            ),
        )
        conn.execute(
            "INSERT INTO insession_auxiliary_v2_waiting_user_answer_bindings "
            "(answer_binding_id, apply_id, session_id, insession_task_id, "
            "work_run_id, execution_subject_id, auxiliary_graph_id, goal_id, "
            "auxiliary_graph_revision, auxiliary_node_id, node_revision, "
            "question_attempt_id, question_sha256, answer_attempt_id, "
            "answer_source_turn_id, answer_source_message_id, "
            "answer_source_turn_idx, answer_source_content_sha256, "
            "answer_source_utf8_bytes, binding_sha256, created_at) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                marker,
                f"{marker}-receipt",
                command.session_id,
                command.subject.task_id,
                command.work_run_id,
                str(authority["execution_subject_id"]),
                command.subject.auxiliary_graph_id,
                str(authority["current_goal_id"]),
                command.subject.auxiliary_graph_revision,
                command.subject.node_id,
                command.subject.node_revision,
                f"{marker.removesuffix('-marker')}-attempt-1",
                "b" * 64,
                f"{marker.removesuffix('-marker')}-attempt-1",
                command.turn_id,
                str(answer_input["message_id"]),
                int(answer_input["turn_idx"]),
                hashlib.sha256(
                    str(answer_input["content"]).encode()
                ).hexdigest(),
                len(str(answer_input["content"]).encode()),
                "c" * 64,
                "2026-08-22T00:00:00+00:00",
            ),
        )


def _execution_ledger_rows() -> tuple[tuple[object, ...], tuple[object, ...]]:
    with store._connect() as conn:
        receipts = tuple(
            tuple(row)
            for row in conn.execute(
                "SELECT * FROM insession_auxiliary_v2_execution_apply_receipts "
                "ORDER BY apply_id"
            ).fetchall()
        )
        answers = tuple(
            tuple(row)
            for row in conn.execute(
                "SELECT * FROM "
                "insession_auxiliary_v2_waiting_user_answer_bindings "
                "ORDER BY answer_binding_id"
            ).fetchall()
        )
    return receipts, answers


def test_pending_verification_rebinds_same_request_without_budget_charge() -> None:
    seeded = _seed_pending_verification("verification-resume")
    command = seeded["command"]

    resumed = continuation_store.resume_auxiliary_verification(command=command)

    assert resumed.status == "applied"
    assert resumed.operation == "resume_verification"
    assert resumed.verification_request_id == command.verification_request_id
    assert resumed.verification_request_revision == 2
    assert resumed.work_run_revision == command.expected_work_run_revision + 1
    assert resumed.window_state_version == command.expected_window_revision + 1
    assert resumed.subject == command.subject
    assert resumed.structure_sha256 == command.expected_structure_sha256
    assert resumed.request_source_turn_id == seeded["first_turn_id"]
    assert resumed.request_binding_sha256 == seeded["request_before"][
        "request_binding_hash"
    ]
    assert resumed.active_seconds_consumed == 2

    prepared = verification_store.get_prepared_auxiliary_node_verification(
        session_id=seeded["session_id"],
        invocation_turn_id=seeded["second_turn_id"],
        verification_request_id=command.verification_request_id,
    )
    assert prepared.invocation_turn_id == seeded["second_turn_id"]
    assert prepared.record.request.request_turn_id == seeded["first_turn_id"]
    assert prepared.record.request.submitted_attempt_id == (
        seeded["original_prepared"].record.request.submitted_attempt_id
    )
    assert prepared.record.request.output_revision == (
        seeded["original_prepared"].record.request.output_revision
    )

    with store._connect() as conn:
        request_after = dict(
            conn.execute(
                "SELECT * FROM insession_work_run_verification_requests "
                "WHERE verification_request_id=?",
                (command.verification_request_id,),
            ).fetchone()
        )
        for field in (
            "execution_subject_id",
            "request_turn_id",
            "work_run_id",
            "subject_kind",
            "insession_task_id",
            "auxiliary_graph_id",
            "auxiliary_graph_revision",
            "auxiliary_node_id",
            "node_revision",
            "submitted_attempt_id",
            "output_revision",
            "acceptance_progress_revision",
            "acceptance_ids_json",
            "supporting_tool_result_ids_json",
            "locked_work_run_revision",
            "request_binding_hash",
            "prepared_budget_json",
            "created_at",
        ):
            assert request_after[field] == seeded["request_before"][field]
        assert request_after["request_revision"] == 2
        assert request_after["status"] == "pending"
        assert tuple(
            conn.execute(
                "SELECT attempts_started, active_seconds_consumed "
                "FROM insession_work_runs WHERE work_run_id=?",
                (command.work_run_id,),
            ).fetchone()
        ) == seeded["budget_before"]
        assert int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_work_run_budget_charges "
                "WHERE work_run_id=?",
                (command.work_run_id,),
            ).fetchone()[0]
        ) == seeded["charge_count_before"]
        receipt = conn.execute(
            "SELECT operation, invocation_turn_id, result_json "
            "FROM insession_auxiliary_v2_execution_apply_receipts "
            "WHERE apply_id=?",
            (command.apply_id,),
        ).fetchone()
        assert tuple(receipt[:2]) == (
            "resume_verification",
            seeded["second_turn_id"],
        )
        assert json.loads(str(receipt["result_json"]))["request_source_turn_id"] == (
            seeded["first_turn_id"]
        )

    replayed = continuation_store.resume_auxiliary_verification(command=command)
    assert replayed.status == "replayed"
    assert replayed.model_copy(update={"status": "applied"}) == resumed
    with pytest.raises(continuation_store.AuxiliaryContinuationApplyIdCollision):
        continuation_store.resume_auxiliary_verification(
            command=command.model_copy(
                update={"expected_structure_sha256": "0" * 64}
            )
        )

    request = prepared.record.request
    settled = verification_store.commit_auxiliary_node_verification_result(
        session_id=command.session_id,
        turn_id=command.turn_id,
        work_run_id=command.work_run_id,
        verification_request_id=command.verification_request_id,
        result=NodeVerificationResult(
            verification_request_id=request.verification_request_id,
            verification_request_revision=request.revision,
            work_run_id=request.work_run_id,
            locked_work_run_revision=request.locked_work_run_revision,
            submitted_attempt_id=request.submitted_attempt_id,
            acceptance_progress_revision=request.acceptance_progress_revision,
            subject=request.subject,
            output_revision=request.output_revision,
            acceptance_results=(
                AcceptanceVerificationFeedback(
                    acceptance_id="grounded",
                    verdict=VerificationVerdict.PASSED,
                    finding="来源绑定保持完整。",
                ),
            ),
            all_pass=True,
        ),
        expected_work_run_revision=resumed.work_run_revision,
        expected_verification_request_revision=(
            resumed.verification_request_revision
        ),
        expected_window_revision=resumed.window_state_version,
        apply_id="verification-resume-settle",
        active_seconds_delta=3,
        completion_id="verification-resume-completion",
    )
    assert settled.all_pass is True
    assert settled.budget_transition is not None
    assert settled.budget_transition.budget_before.active_seconds_consumed == 2
    assert settled.budget_transition.budget_after.active_seconds_consumed == 5
    with store._connect() as conn:
        output = conn.execute(
            "SELECT frozen_at FROM insession_work_run_output_windows "
            "WHERE work_run_id=? AND output_revision=?",
            (command.work_run_id, request.output_revision),
        ).fetchone()
        assert output["frozen_at"] is not None
        assert conn.execute(
            "SELECT request_turn_id FROM "
            "insession_work_run_verification_requests "
            "WHERE verification_request_id=?",
            (command.verification_request_id,),
        ).fetchone()[0] == seeded["first_turn_id"]

    replayed_after_settlement = continuation_store.resume_auxiliary_verification(
        command=command
    )
    assert replayed_after_settlement.status == "replayed"
    assert replayed_after_settlement.model_copy(
        update={"status": "applied"}
    ) == resumed
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_run_output_windows SET snapshot_hash=? "
            "WHERE work_run_id=? AND output_revision=?",
            ("0" * 64, command.work_run_id, request.output_revision),
        )
    with pytest.raises(
        continuation_store.AuxiliaryContinuationPersistenceError,
        match="OutputWindow|snapshot",
    ):
        continuation_store.resume_auxiliary_verification(command=command)


def test_repeated_verification_resume_rejects_tampered_predecessor_receipt() -> None:
    seeded = _seed_pending_verification("verification-resume-chain-tamper")
    command = seeded["command"]
    assert isinstance(command, continuation_store.ResumeAuxiliaryVerificationCommand)
    resumed = continuation_store.resume_auxiliary_verification(command=command)
    marked = store.mark_turn_execution_interrupted(
        session_id=command.session_id,
        turn_id=command.turn_id,
        expected_window_revision=resumed.window_state_version,
        stage="VERIFICATION",
        interruption_reason="process_lost_after_first_resume",
    )
    store.settle_interrupted_turn_execution(
        session_id=command.session_id,
        turn_id=command.turn_id,
        expected_window_revision=int(marked["state_version"]),
        end_reason="process_lost_after_first_resume",
        error_code="PROCESS_LOST",
    )
    accepted = store.accept_turn_execution(
        session_id=command.session_id,
        client_request_id="verification-resume-chain-tamper-next",
        source="auxiliary_v2_verification_continuation_test",
        user_text="再次继续同一个验证请求",
        lease_owner="auxiliary-v2-verification-continuation-test",
    )
    next_turn_id = str(accepted["turn"]["turn_id"])
    insession_task_records.link_turn_to_insession_tasks(
        store._deps(),
        session_id=command.session_id,
        turn_id=next_turn_id,
        insession_task_ids=(command.subject.task_id,),
        expected_window_revision=_window_revision(command.session_id),
    )
    frontier, candidate = _recoverable(
        command.session_id,
        next_turn_id,
        command.subject.task_id,
    )
    repeated = command.model_copy(
        update={
            "turn_id": next_turn_id,
            "expected_work_run_revision": resumed.work_run_revision,
            "expected_verification_request_revision": (
                resumed.verification_request_revision
            ),
            "expected_window_revision": _window_revision(command.session_id),
            "expected_task_state_version": frontier.task_state_version,
            "expected_node_state_version": candidate.node_state_version,
            "expected_control_state_version": frontier.control_state_version,
            "expected_goal_state_version": frontier.goal_state_version,
            "expected_revision_state_version": frontier.revision_state_version,
            "apply_id": "verification-resume-chain-tamper-second-resume",
        }
    )
    before = work_run_store.get_work_run(
        session_id=command.session_id,
        work_run_id=command.work_run_id,
    )
    before_window_revision = _window_revision(command.session_id)
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_auxiliary_v2_execution_apply_receipts "
            "SET result_sha256=? WHERE apply_id=?",
            ("0" * 64, command.apply_id),
        )

    with pytest.raises(
        continuation_store.AuxiliaryContinuationPersistenceError,
        match="predecessor resume receipt is corrupt",
    ):
        continuation_store.resume_auxiliary_verification(command=repeated)

    after = work_run_store.get_work_run(
        session_id=command.session_id,
        work_run_id=command.work_run_id,
    )
    assert after == before
    assert _window_revision(command.session_id) == before_window_revision
    with store._connect() as conn:
        assert conn.execute(
            "SELECT 1 FROM insession_auxiliary_v2_execution_apply_receipts "
            "WHERE apply_id=?",
            (repeated.apply_id,),
        ).fetchone() is None


def test_verification_resume_rejects_stale_structure_subject_and_request() -> None:
    seeded = _seed_pending_verification("verification-stale")
    command = seeded["command"]
    assert isinstance(command, continuation_store.ResumeAuxiliaryVerificationCommand)
    before = _resume_state(seeded)

    with pytest.raises(verification_store.TaskNodeVerificationRequestRevisionConflict):
        continuation_store.resume_auxiliary_verification(
            command=command.model_copy(
                update={
                    "expected_verification_request_revision": 2,
                    "apply_id": "verification-stale-request-revision",
                }
            )
        )
    assert _resume_state(seeded) == before

    with pytest.raises(
        continuation_store.AuxiliaryContinuationPersistenceError,
        match="structure is stale",
    ):
        continuation_store.resume_auxiliary_verification(
            command=command.model_copy(
                update={
                    "expected_structure_sha256": "0" * 64,
                    "apply_id": "verification-stale-structure",
                }
            )
        )
    assert _resume_state(seeded) == before

    with pytest.raises(
        continuation_store.AuxiliaryContinuationPersistenceError,
        match="subject is stale",
    ):
        continuation_store.resume_auxiliary_verification(
            command=command.model_copy(
                update={
                    "subject": command.subject.model_copy(
                        update={"node_id": "forged-node"}
                    ),
                    "apply_id": "verification-stale-subject",
                }
            )
        )
    assert _resume_state(seeded) == before

    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_run_verification_requests "
            "SET request_binding_hash=? WHERE verification_request_id=?",
            ("f" * 64, command.verification_request_id),
        )
    tampered = _resume_state(seeded)
    with pytest.raises(
        continuation_store.AuxiliaryContinuationPersistenceError,
        match="binding has drifted",
    ):
        continuation_store.resume_auxiliary_verification(
            command=command.model_copy(
                update={"apply_id": "verification-tampered-request"}
            )
        )
    assert _resume_state(seeded) == tampered


def test_verification_resume_rejects_old_and_unrelated_turns() -> None:
    seeded = _seed_pending_verification("verification-turn-fence")
    command = seeded["command"]
    assert isinstance(command, continuation_store.ResumeAuxiliaryVerificationCommand)
    before = _resume_state(seeded)

    with pytest.raises(
        work_run_store.WorkExecutionPersistenceError,
        match="active Window",
    ):
        continuation_store.resume_auxiliary_verification(
            command=command.model_copy(
                update={
                    "turn_id": seeded["first_turn_id"],
                    "apply_id": "verification-old-turn",
                }
            )
        )
    assert _resume_state(seeded) == before

    with store._connect() as conn:
        conn.execute(
            "DELETE FROM insession_task_turn_links WHERE session_id=? "
            "AND turn_id=? AND insession_task_id=?",
            (
                command.session_id,
                command.turn_id,
                command.subject.task_id,
            ),
        )
    unrelated = _resume_state(seeded)
    with pytest.raises(
        continuation_store.AuxiliaryContinuationPersistenceError,
        match="not linked",
    ):
        continuation_store.resume_auxiliary_verification(
            command=command.model_copy(
                update={"apply_id": "verification-unrelated-turn"}
            )
        )
    assert _resume_state(seeded) == unrelated


def test_verification_resume_rejects_turn_already_bound_as_answer_message() -> None:
    seeded = _seed_pending_verification("verification-answer-turn")
    command = seeded["command"]
    assert isinstance(command, continuation_store.ResumeAuxiliaryVerificationCommand)
    _insert_answer_message_marker(
        seeded,
        marker="verification-answer-turn-marker",
    )
    marked = _resume_state(seeded)
    with pytest.raises(
        continuation_store.AuxiliaryContinuationPersistenceError,
        match="answer-message",
    ):
        continuation_store.resume_auxiliary_verification(command=command)
    assert _resume_state(seeded) == marked


def test_current_schema_exposes_only_the_verification_resume_operation() -> None:
    store.init_db()
    with store._connect() as conn:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == (
            schema.SCHEMA_VERSION
        )
        sql = str(
            conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND "
                "name='insession_auxiliary_v2_execution_apply_receipts'"
            ).fetchone()[0]
        )
        assert "'resume_verification'" in sql
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
