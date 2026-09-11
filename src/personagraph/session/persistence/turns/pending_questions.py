"""L2-free proof of the pending questions that Entry may expose."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from ...entry_task_contracts import parse_entry_pending_question_decision_json
from ...insession_task_contracts import InSessionTaskPersistenceError


_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NONTERMINAL_WORK_RUN_STATUSES = (
    "active",
    "paused",
    "waiting_user",
    "waiting_authorization",
    "waiting_external",
    "turn_limit_reached",
    "interrupted",
)


@dataclass(frozen=True)
class PendingQuestionProof:
    """A current pending question and the ordering facts proved with it."""

    insession_task_id: str
    question: str
    question_attempt_id: str
    question_turn_id: str
    node_ordinal: int
    attempt_ordinal: int


@dataclass(frozen=True)
class _QuestionAttempt:
    attempt_id: str
    turn_id: str
    input_turn_id: str
    ordinal: int
    decision_json: str
    progress_revision_before: int
    apply_id: str


@dataclass(frozen=True)
class _SettlementReceipt:
    payload_sha256: str
    created_at: str


@dataclass(frozen=True)
class _AuxiliaryAuthority:
    goal_id: str
    node_ordinal: int


def list_pending_question_proofs(
    conn: sqlite3.Connection,
    *,
    session_id: str,
) -> tuple[PendingQuestionProof, ...]:
    """Prove current question visibility inside the caller's SQLite snapshot."""

    rows = conn.execute(
        "SELECT run.work_run_id, run.execution_subject_id, run.session_id, "
        "run.subject_kind, run.insession_task_id, run.graph_revision, "
        "run.insession_task_node_id, run.auxiliary_graph_id, "
        "run.auxiliary_graph_revision, run.auxiliary_node_id, "
        "run.node_revision, run.status, run.reason, run.revision, "
        "run.attempts_started, run.current_attempt_id, "
        "run.current_verification_request_id, run.updated_at, "
        "registry.subject_contract_version AS registry_contract_version "
        "FROM insession_work_runs AS run "
        "LEFT JOIN insession_execution_subjects AS registry "
        "ON registry.execution_subject_id=run.execution_subject_id "
        "AND registry.session_id=run.session_id "
        "AND registry.insession_task_id=run.insession_task_id "
        "AND registry.subject_kind=run.subject_kind "
        "WHERE run.session_id=? AND run.status='waiting_user' "
        "ORDER BY CASE registry.subject_contract_version "
        "WHEN 'task_node_v1' THEN 0 WHEN 'auxiliary_node_v2' THEN 1 ELSE 2 END, "
        "CASE WHEN registry.subject_contract_version='task_node_v1' "
        "THEN run.updated_at END, "
        "CASE WHEN registry.subject_contract_version='auxiliary_node_v2' "
        "THEN run.insession_task_id END, run.work_run_id",
        (session_id,),
    ).fetchall()
    return tuple(_prove_question(conn, row) for row in rows)


def require_pending_question_proof(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    question_attempt_id: str,
) -> PendingQuestionProof:
    """Return one exact proof without introducing a second authority reader."""

    row = conn.execute(
        "SELECT run.work_run_id, run.execution_subject_id, run.session_id, "
        "run.subject_kind, run.insession_task_id, run.graph_revision, "
        "run.insession_task_node_id, run.auxiliary_graph_id, "
        "run.auxiliary_graph_revision, run.auxiliary_node_id, "
        "run.node_revision, run.status, run.reason, run.revision, "
        "run.attempts_started, run.current_attempt_id, "
        "run.current_verification_request_id, run.updated_at, "
        "registry.subject_contract_version AS registry_contract_version "
        "FROM insession_work_run_attempts AS requested "
        "JOIN insession_work_runs AS run "
        "ON run.work_run_id=requested.work_run_id "
        "LEFT JOIN insession_execution_subjects AS registry "
        "ON registry.execution_subject_id=run.execution_subject_id "
        "AND registry.session_id=run.session_id "
        "AND registry.insession_task_id=run.insession_task_id "
        "AND registry.subject_kind=run.subject_kind "
        "WHERE requested.attempt_id=? AND run.session_id=? "
        "AND run.status='waiting_user'",
        (question_attempt_id, session_id),
    ).fetchone()
    if row is None:
        raise InSessionTaskPersistenceError(
            "pending-question Attempt is not the Session's current question"
        )
    proof = _prove_question(conn, row)
    if proof.question_attempt_id != question_attempt_id:
        raise InSessionTaskPersistenceError(
            "pending-question Attempt is not the WorkRun's current question"
        )
    return proof


def _prove_question(
    conn: sqlite3.Connection,
    run: sqlite3.Row,
) -> PendingQuestionProof:
    _required_text(run["work_run_id"], max_length=200)
    task_id = _required_text(run["insession_task_id"], max_length=128)
    if (
        str(run["status"]) != "waiting_user"
        or str(run["reason"] or "") != "needs_input"
        or run["current_attempt_id"] is not None
        or run["current_verification_request_id"] is not None
    ):
        raise InSessionTaskPersistenceError(
            "waiting-user WorkRun projection is inconsistent"
        )

    contract_version = run["registry_contract_version"]
    if contract_version == "task_node_v1":
        node_ordinal = _require_task_node_authority(conn, run)
        auxiliary_authority = None
    elif contract_version == "auxiliary_node_v2":
        auxiliary_authority = _require_auxiliary_authority(conn, run)
        node_ordinal = auxiliary_authority.node_ordinal
        _require_single_auxiliary_cursor(conn, run)
    else:
        raise InSessionTaskPersistenceError(
            "pending question has no supported execution-subject contract"
        )

    attempt, question = _load_final_question_attempt(conn, run)
    settlement = _require_settlement_receipt(
        conn,
        run=run,
        attempt=attempt,
        bind_decision=auxiliary_authority is None,
    )
    if auxiliary_authority is not None:
        _require_auxiliary_receipt(
            conn,
            run=run,
            authority=auxiliary_authority,
            attempt=attempt,
            question=question,
            generic_settlement=settlement,
        )
    return PendingQuestionProof(
        insession_task_id=task_id,
        question=question,
        question_attempt_id=attempt.attempt_id,
        question_turn_id=attempt.turn_id,
        node_ordinal=node_ordinal,
        attempt_ordinal=attempt.ordinal,
    )


def _require_task_node_authority(
    conn: sqlite3.Connection,
    run: sqlite3.Row,
) -> int:
    graph_revision = _positive_int(run["graph_revision"])
    node_id = _required_text(run["insession_task_node_id"], max_length=200)
    node_revision = _positive_int(run["node_revision"])
    authority = conn.execute(
        "SELECT task.current_graph_revision, task.current_status, node.ordinal, "
        "state.status AS node_status "
        "FROM insession_execution_subjects AS registry "
        "JOIN insession_task_node_execution_subject_bindings AS binding "
        "ON binding.binding_id=registry.task_node_binding_id "
        "AND binding.session_id=registry.session_id "
        "AND binding.insession_task_id=registry.insession_task_id "
        "JOIN insession_tasks AS task "
        "ON task.session_id=binding.session_id "
        "AND task.insession_task_id=binding.insession_task_id "
        "JOIN insession_task_graph_nodes AS node "
        "ON node.insession_task_id=binding.insession_task_id "
        "AND node.graph_revision=binding.graph_revision "
        "AND node.insession_task_node_id=binding.insession_task_node_id "
        "AND node.node_revision=binding.node_revision "
        "JOIN insession_task_node_states AS state "
        "ON state.insession_task_id=node.insession_task_id "
        "AND state.insession_task_node_id=node.insession_task_node_id "
        "AND state.node_revision=node.node_revision "
        "WHERE registry.execution_subject_id=? AND registry.session_id=? "
        "AND registry.insession_task_id=? "
        "AND registry.subject_kind='task_node' "
        "AND registry.subject_contract_version='task_node_v1' "
        "AND registry.auxiliary_v2_binding_id IS NULL "
        "AND binding.graph_revision=? "
        "AND binding.insession_task_node_id=? AND binding.node_revision=?",
        (
            run["execution_subject_id"],
            run["session_id"],
            run["insession_task_id"],
            graph_revision,
            node_id,
            node_revision,
        ),
    ).fetchone()
    if (
        str(run["subject_kind"]) != "task_node"
        or authority is None
        or _positive_int(authority["current_graph_revision"]) != graph_revision
        or str(authority["current_status"]) not in {"active", "awaiting_user"}
        or str(authority["node_status"]) != "awaiting_user"
    ):
        raise InSessionTaskPersistenceError(
            "pending question is detached from current execution authority"
        )
    if conn.execute(
        "SELECT 1 FROM insession_task_graph_edges AS edge "
        "JOIN insession_task_graph_nodes AS child "
        "ON child.insession_task_id=edge.insession_task_id "
        "AND child.graph_revision=edge.graph_revision "
        "AND child.insession_task_node_id=edge.child_insession_task_node_id "
        "LEFT JOIN insession_task_node_states AS child_state "
        "ON child_state.insession_task_id=child.insession_task_id "
        "AND child_state.insession_task_node_id=child.insession_task_node_id "
        "AND child_state.node_revision=child.node_revision "
        "WHERE edge.insession_task_id=? AND edge.graph_revision=? "
        "AND edge.parent_insession_task_node_id=? "
        "AND (child_state.status IS NULL OR child_state.status!='completed') "
        "LIMIT 1",
        (str(run["insession_task_id"]), graph_revision, node_id),
    ).fetchone() is not None:
        raise InSessionTaskPersistenceError(
            "pending TaskNode question is no longer execution-ready"
        )
    return _nonnegative_int(authority["ordinal"])


def _require_auxiliary_authority(
    conn: sqlite3.Connection,
    run: sqlite3.Row,
) -> _AuxiliaryAuthority:
    authority = conn.execute(
        "SELECT binding.goal_id AS bound_goal_id, "
        "binding.executor_kind AS bound_executor_kind, "
        "binding.definition_sha256 AS bound_definition_sha256, "
        "control.current_goal_id, control.current_auxiliary_graph_revision, "
        "goal.goal_id, goal.status AS goal_status, "
        "goal.base_task_graph_revision, "
        "revision.structure_contract_version, "
        "revision_state.status AS revision_status, membership.ordinal, "
        "node_state.status AS node_status, "
        "definition.executor_kind, definition.definition_sha256, "
        "task.current_graph_revision AS task_graph_revision, "
        "task.current_status AS task_status "
        "FROM insession_execution_subjects AS registry "
        "JOIN insession_auxiliary_v2_node_execution_subject_bindings AS binding "
        "ON binding.binding_id=registry.auxiliary_v2_binding_id "
        "AND binding.session_id=registry.session_id "
        "AND binding.insession_task_id=registry.insession_task_id "
        "JOIN insession_auxiliary_graph_v2_containers AS control "
        "ON control.session_id=binding.session_id "
        "AND control.insession_task_id=binding.insession_task_id "
        "AND control.auxiliary_graph_id=binding.auxiliary_graph_id "
        "JOIN insession_auxiliary_graph_revision_snapshots AS revision "
        "ON revision.auxiliary_graph_id=binding.auxiliary_graph_id "
        "AND revision.auxiliary_graph_revision=binding.auxiliary_graph_revision "
        "AND revision.insession_task_id=binding.insession_task_id "
        "AND revision.goal_id=binding.goal_id "
        "JOIN insession_auxiliary_graph_goals AS goal "
        "ON goal.goal_id=revision.goal_id "
        "AND goal.session_id=binding.session_id "
        "AND goal.insession_task_id=binding.insession_task_id "
        "AND goal.auxiliary_graph_id=binding.auxiliary_graph_id "
        "JOIN insession_auxiliary_graph_revision_states_v2 AS revision_state "
        "ON revision_state.auxiliary_graph_id=revision.auxiliary_graph_id "
        "AND revision_state.auxiliary_graph_revision="
        "revision.auxiliary_graph_revision "
        "JOIN insession_auxiliary_graph_revision_nodes_v2 AS membership "
        "ON membership.auxiliary_graph_id=revision.auxiliary_graph_id "
        "AND membership.auxiliary_graph_revision="
        "revision.auxiliary_graph_revision "
        "AND membership.auxiliary_node_id=binding.auxiliary_node_id "
        "AND membership.node_revision=binding.node_revision "
        "JOIN insession_auxiliary_node_definitions_v2 AS definition "
        "ON definition.auxiliary_graph_id=membership.auxiliary_graph_id "
        "AND definition.auxiliary_node_id=membership.auxiliary_node_id "
        "AND definition.node_revision=membership.node_revision "
        "JOIN insession_auxiliary_node_states_v2 AS node_state "
        "ON node_state.auxiliary_graph_id=membership.auxiliary_graph_id "
        "AND node_state.auxiliary_graph_revision="
        "membership.auxiliary_graph_revision "
        "AND node_state.auxiliary_node_id=membership.auxiliary_node_id "
        "AND node_state.node_revision=membership.node_revision "
        "JOIN insession_tasks AS task ON task.session_id=binding.session_id "
        "AND task.insession_task_id=binding.insession_task_id "
        "WHERE registry.execution_subject_id=? AND registry.session_id=? "
        "AND registry.insession_task_id=? "
        "AND registry.subject_kind='auxiliary_node' "
        "AND registry.subject_contract_version='auxiliary_node_v2' "
        "AND registry.task_node_binding_id IS NULL "
        "AND binding.auxiliary_graph_id=? "
        "AND binding.auxiliary_graph_revision=? "
        "AND binding.auxiliary_node_id=? AND binding.node_revision=?",
        (
            run["execution_subject_id"],
            run["session_id"],
            run["insession_task_id"],
            run["auxiliary_graph_id"],
            run["auxiliary_graph_revision"],
            run["auxiliary_node_id"],
            run["node_revision"],
        ),
    ).fetchone()
    if authority is None:
        raise InSessionTaskPersistenceError(
            "pending Auxiliary question lost its execution authority"
        )
    task_revision = _optional_positive_int(authority["task_graph_revision"])
    base_revision = _optional_positive_int(
        authority["base_task_graph_revision"]
    )
    if (
        str(run["subject_kind"]) != "auxiliary_node"
        or str(authority["structure_contract_version"])
        != "auxiliary-graph-revision-v2"
        or str(authority["bound_goal_id"]) != str(authority["goal_id"])
        or str(authority["current_goal_id"]) != str(authority["goal_id"])
        or _positive_int(authority["current_auxiliary_graph_revision"])
        != _positive_int(run["auxiliary_graph_revision"])
        or str(authority["bound_executor_kind"])
        != str(authority["executor_kind"])
        or str(authority["bound_definition_sha256"])
        != str(authority["definition_sha256"])
        or _HEX_SHA256.fullmatch(str(authority["definition_sha256"])) is None
        or task_revision != base_revision
        or str(authority["task_status"]) != "awaiting_user"
        or str(authority["node_status"]) != "waiting_user"
        or str(authority["goal_status"]) != "waiting_user"
        or str(authority["revision_status"]) != "waiting_user"
    ):
        raise InSessionTaskPersistenceError(
            "pending Auxiliary question authority is stale or spliced"
        )
    return _AuxiliaryAuthority(
        goal_id=_required_text(authority["goal_id"], max_length=200),
        node_ordinal=_nonnegative_int(authority["ordinal"]),
    )


def _require_single_auxiliary_cursor(
    conn: sqlite3.Connection,
    run: sqlite3.Row,
) -> None:
    placeholders = ",".join("?" for _ in _NONTERMINAL_WORK_RUN_STATUSES)
    rows = conn.execute(
        "SELECT candidate.work_run_id FROM insession_work_runs AS candidate "
        "JOIN insession_execution_subjects AS registry "
        "ON registry.execution_subject_id=candidate.execution_subject_id "
        "AND registry.session_id=candidate.session_id "
        "AND registry.insession_task_id=candidate.insession_task_id "
        "AND registry.subject_kind='auxiliary_node' "
        "AND registry.subject_contract_version='auxiliary_node_v2' "
        "JOIN insession_auxiliary_v2_node_execution_subject_bindings AS binding "
        "ON binding.binding_id=registry.auxiliary_v2_binding_id "
        "AND binding.session_id=candidate.session_id "
        "AND binding.insession_task_id=candidate.insession_task_id "
        "AND binding.auxiliary_graph_id=candidate.auxiliary_graph_id "
        "AND binding.auxiliary_graph_revision="
        "candidate.auxiliary_graph_revision "
        "AND binding.auxiliary_node_id=candidate.auxiliary_node_id "
        "AND binding.node_revision=candidate.node_revision "
        "JOIN insession_auxiliary_graph_v2_containers AS control "
        "ON control.session_id=candidate.session_id "
        "AND control.insession_task_id=candidate.insession_task_id "
        "AND control.auxiliary_graph_id=candidate.auxiliary_graph_id "
        "WHERE candidate.session_id=? AND candidate.insession_task_id=? "
        "AND candidate.auxiliary_graph_revision="
        "control.current_auxiliary_graph_revision "
        "AND binding.goal_id=control.current_goal_id "
        f"AND candidate.status IN ({placeholders})",
        (
            run["session_id"],
            run["insession_task_id"],
            *_NONTERMINAL_WORK_RUN_STATUSES,
        ),
    ).fetchall()
    if len(rows) != 1 or str(rows[0]["work_run_id"]) != str(run["work_run_id"]):
        raise InSessionTaskPersistenceError(
            "Auxiliary pending question is not its Task's single cursor"
        )


def _load_final_question_attempt(
    conn: sqlite3.Connection,
    run: sqlite3.Row,
) -> tuple[_QuestionAttempt, str]:
    work_run_id = _required_text(run["work_run_id"], max_length=200)
    attempts_started = _positive_int(run["attempts_started"])
    rows = conn.execute(
        "SELECT attempt_id, turn_id, input_turn_id, ordinal, status, action, "
        "decision_json, progress_revision_before, close_reason, closed_at, "
        "budget_charge_id FROM insession_work_run_attempts "
        "WHERE work_run_id=? AND ordinal=? ORDER BY attempt_id LIMIT 2",
        (work_run_id, attempts_started),
    ).fetchall()
    if (
        len(rows) != 1
        or _positive_int(rows[0]["ordinal"]) != attempts_started
    ):
        raise InSessionTaskPersistenceError(
            "waiting-user WorkRun Attempt count or final ordinal is corrupt"
        )
    row = rows[0]
    if (
        str(row["status"]) != "closed"
        or str(row["action"] or "") != "request_user_input"
        or str(row["close_reason"] or "") != "request_user_input"
        or not row["closed_at"]
    ):
        raise InSessionTaskPersistenceError(
            "waiting-user WorkRun does not end at its question Attempt"
        )
    decision_json = _required_text(row["decision_json"])
    question = parse_entry_pending_question_decision_json(decision_json)
    turn_id = _required_text(row["turn_id"], max_length=200)
    input_turn_id = _required_text(row["input_turn_id"], max_length=200)
    linked_turns = conn.execute(
        "SELECT COUNT(DISTINCT turn_id) AS count "
        "FROM insession_work_run_turn_links "
        "WHERE session_id=? AND work_run_id=? AND turn_id IN (?, ?)",
        (run["session_id"], work_run_id, turn_id, input_turn_id),
    ).fetchone()
    expected_turns = len({turn_id, input_turn_id})
    if linked_turns is None or int(linked_turns["count"]) != expected_turns:
        raise InSessionTaskPersistenceError(
            "waiting-user question Attempt lost its owning Turn link"
        )
    attempt_id = _required_text(row["attempt_id"], max_length=200)
    if conn.execute(
        "SELECT 1 FROM insession_work_run_attempts "
        "WHERE work_run_id=? AND predecessor_question_attempt_id=? LIMIT 1",
        (work_run_id, attempt_id),
    ).fetchone() is not None:
        raise InSessionTaskPersistenceError(
            "waiting-user question was already consumed"
        )
    return (
        _QuestionAttempt(
            attempt_id=attempt_id,
            turn_id=turn_id,
            input_turn_id=input_turn_id,
            ordinal=attempts_started,
            decision_json=decision_json,
            progress_revision_before=_positive_int(
                row["progress_revision_before"]
            ),
            apply_id=_required_text(row["budget_charge_id"], max_length=200),
        ),
        question,
    )


def _require_settlement_receipt(
    conn: sqlite3.Connection,
    *,
    run: sqlite3.Row,
    attempt: _QuestionAttempt,
    bind_decision: bool,
) -> _SettlementReceipt:
    charge = conn.execute(
        "SELECT operation, session_id, work_run_id, turn_id, checkpoint_id, "
        "work_run_revision_before, work_run_revision_after, "
        "window_state_version_before, active_seconds_delta, created_at "
        "FROM insession_work_run_budget_charges WHERE budget_charge_id=?",
        (attempt.apply_id,),
    ).fetchone()
    if charge is None:
        raise InSessionTaskPersistenceError(
            "pending question lost its settlement charge"
        )
    revision_before = _positive_int(charge["work_run_revision_before"])
    window_before = _positive_int(charge["window_state_version_before"])
    active_seconds_delta = _finite_positive(charge["active_seconds_delta"])
    if (
        str(charge["operation"]) != "commit_attempt_decision"
        or str(charge["session_id"]) != str(run["session_id"])
        or str(charge["work_run_id"]) != str(run["work_run_id"])
        or str(charge["turn_id"]) != attempt.turn_id
        or str(charge["checkpoint_id"])
        != f"commit_attempt_decision:{attempt.apply_id}"
        or _positive_int(charge["work_run_revision_after"])
        != revision_before + 1
        or _positive_int(charge["work_run_revision_after"])
        != _positive_int(run["revision"])
    ):
        raise InSessionTaskPersistenceError(
            "pending-question settlement charge owner is corrupt"
        )
    receipt = conn.execute(
        "SELECT operation, session_id, work_run_id, payload_hash, created_at "
        "FROM insession_work_run_apply_receipts WHERE apply_id=?",
        (attempt.apply_id,),
    ).fetchone()
    if (
        receipt is None
        or str(receipt["operation"]) != "commit_attempt_decision"
        or str(receipt["session_id"]) != str(run["session_id"])
        or str(receipt["work_run_id"]) != str(run["work_run_id"])
        or str(receipt["created_at"]) != str(charge["created_at"])
    ):
        raise InSessionTaskPersistenceError(
            "pending question lost its exact generic settlement receipt"
        )
    payload_sha256 = str(receipt["payload_hash"])
    if _HEX_SHA256.fullmatch(payload_sha256) is None:
        raise InSessionTaskPersistenceError(
            "pending-question settlement receipt hash is invalid"
        )
    if bind_decision:
        expected_payload = {
            "session_id": str(run["session_id"]),
            "turn_id": attempt.turn_id,
            "work_run_id": str(run["work_run_id"]),
            "attempt_id": attempt.attempt_id,
            "decision": _decode_integrity_json(attempt.decision_json),
            "expected_work_run_revision": revision_before,
            "expected_progress_revision": attempt.progress_revision_before,
            "expected_window_revision": window_before,
            "active_seconds_delta": active_seconds_delta,
        }
        if _sha256(_canonical_json(expected_payload)) != payload_sha256:
            raise InSessionTaskPersistenceError(
                "pending-question decision is detached from its commit receipt"
            )
    return _SettlementReceipt(
        payload_sha256=payload_sha256,
        created_at=str(receipt["created_at"]),
    )


def _require_auxiliary_receipt(
    conn: sqlite3.Connection,
    *,
    run: sqlite3.Row,
    authority: _AuxiliaryAuthority,
    attempt: _QuestionAttempt,
    question: str,
    generic_settlement: _SettlementReceipt,
) -> None:
    receipt = conn.execute(
        "SELECT operation, session_id, insession_task_id, work_run_id, "
        "execution_subject_id, auxiliary_graph_id, goal_id, "
        "auxiliary_graph_revision, auxiliary_node_id, node_revision, "
        "invocation_turn_id, payload_sha256, result_json, result_sha256, "
        "created_at FROM insession_auxiliary_v2_execution_apply_receipts "
        "WHERE apply_id=?",
        (attempt.apply_id,),
    ).fetchone()
    expected_owner = (
        ("operation", "commit_waiting_user_attempt"),
        ("session_id", run["session_id"]),
        ("insession_task_id", run["insession_task_id"]),
        ("work_run_id", run["work_run_id"]),
        ("execution_subject_id", run["execution_subject_id"]),
        ("auxiliary_graph_id", run["auxiliary_graph_id"]),
        ("goal_id", authority.goal_id),
        ("auxiliary_graph_revision", run["auxiliary_graph_revision"]),
        ("auxiliary_node_id", run["auxiliary_node_id"]),
        ("node_revision", run["node_revision"]),
        ("invocation_turn_id", attempt.turn_id),
    )
    if receipt is None or any(
        str(receipt[key]) != str(expected) for key, expected in expected_owner
    ):
        raise InSessionTaskPersistenceError(
            "Auxiliary pending question lost its exact settlement receipt"
        )
    payload_sha256 = str(receipt["payload_sha256"])
    result_json = str(receipt["result_json"])
    result_sha256 = str(receipt["result_sha256"])
    if (
        _HEX_SHA256.fullmatch(payload_sha256) is None
        or _HEX_SHA256.fullmatch(result_sha256) is None
        or payload_sha256 != generic_settlement.payload_sha256
        or str(receipt["created_at"]) != generic_settlement.created_at
        or _sha256(result_json) != result_sha256
    ):
        raise InSessionTaskPersistenceError(
            "Auxiliary question settlement receipt hash is corrupt"
        )
    result = _decode_integrity_json(result_json)
    attempt_result = result.get("attempt") if isinstance(result, dict) else None
    subject = result.get("subject") if isinstance(result, dict) else None
    expected_subject = {
        "kind": "auxiliary_node",
        "task_id": str(run["insession_task_id"]),
        "auxiliary_graph_id": str(run["auxiliary_graph_id"]),
        "auxiliary_graph_revision": _positive_int(
            run["auxiliary_graph_revision"]
        ),
        "node_id": str(run["auxiliary_node_id"]),
        "node_revision": _positive_int(run["node_revision"]),
    }
    if (
        not isinstance(result, dict)
        or result.get("operation") != "commit_waiting_user_attempt"
        or result.get("work_run_id") != str(run["work_run_id"])
        or result.get("execution_subject_id")
        != str(run["execution_subject_id"])
        or result.get("execution_subject_contract_version")
        != "auxiliary_node_v2"
        or result.get("goal_id") != authority.goal_id
        or result.get("question_sha256") != _sha256(question)
        or not isinstance(attempt_result, dict)
        or attempt_result.get("attempt_id") != attempt.attempt_id
        or attempt_result.get("work_run_id") != str(run["work_run_id"])
        or attempt_result.get("ordinal") != attempt.ordinal
        or attempt_result.get("status") != "closed"
        or subject != expected_subject
    ):
        raise InSessionTaskPersistenceError(
            "Auxiliary question settlement receipt owner is corrupt"
        )


def _decode_integrity_json(value: object) -> Any:
    if not isinstance(value, str) or not value:
        raise ValueError("stored integrity JSON is missing")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    def reject_constant(constant: str) -> object:
        raise ValueError(f"non-standard JSON constant: {constant}")

    try:
        return json.loads(
            value,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("stored integrity JSON is corrupt") from exc


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _required_text(
    value: object,
    *,
    max_length: int | None = None,
) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("stored text is missing")
    if max_length is not None and len(value) > max_length:
        raise ValueError("stored text exceeds its bound")
    return value


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("stored integer must be positive")
    return value


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("stored integer must be nonnegative")
    return value


def _optional_positive_int(value: object) -> int | None:
    return None if value is None else _positive_int(value)


def _finite_positive(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError("stored number must be finite and positive")
    return float(value)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "PendingQuestionProof",
    "list_pending_question_proofs",
    "require_pending_question_proof",
]
