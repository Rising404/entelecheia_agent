"""精确 AuxiliaryNode 执行主体与验证器绑定权威源。

调用方保留事务生命周期、WorkRun/Window 状态转换、预算结算和完成写入；每个辅助函数
都通过调用方提供的同一 SQLite 连接工作。
"""

from __future__ import annotations

import json
import sqlite3

from personagraph.l2.auxiliary_graph import (
    AuxiliaryNodeDefinition,
    AuxiliaryNodeExecutorKind,
    AuxiliaryNodeKind,
    AuxiliaryNodeReference,
)
from personagraph.l2.task_graph.contracts import InSessionTaskAcceptanceProposal
from personagraph.l2.work_run import (
    AcceptanceProgressSnapshot,
    Attempt,
    AuxiliaryNodeSubject,
    NodeVerificationResult,
    OutputWindow,
    SupportingToolResult,
    TaskNodeVerificationRecord,
    TaskNodeVerificationRequestStatus,
    TaskNodeVerificationRequest,
    WorkRunBudget,
)
from .auxiliary_graph_errors import AuxiliaryGraphPersistenceError
from ..work_run.work_execution import (
    _load_output_window,
    _load_progress,
    _payload_hash,
    _require_run_row,
    _stable_execution_subject_id,
    _subject_from_run_row,
)
from ..work_run.work_verification import _load_current_submit_attempt, _load_supporting_tool_results


def _require_auxiliary_subject(row: sqlite3.Row) -> AuxiliaryNodeSubject:
    subject = _subject_from_run_row(row)
    if not isinstance(subject, AuxiliaryNodeSubject):
        raise AuxiliaryGraphPersistenceError(
            "operation requires an AuxiliaryNode WorkRun"
        )
    return subject


def _ensure_auxiliary_execution_subject(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    subject: AuxiliaryNodeSubject,
    goal_id: str,
    executor_kind: str,
    definition_sha256: str,
    created_at: str,
) -> str:
    binding_id = _stable_execution_subject_id(
        "aux2bind",
        session_id,
        subject.task_id,
        subject.auxiliary_graph_id,
        goal_id,
        subject.auxiliary_graph_revision,
        subject.node_id,
        subject.node_revision,
    )
    execution_subject_id = _stable_execution_subject_id(
        "execsubject",
        "auxiliary_node_v2",
        session_id,
        subject.task_id,
        subject.auxiliary_graph_id,
        goal_id,
        subject.auxiliary_graph_revision,
        subject.node_id,
        subject.node_revision,
    )
    conn.execute(
        "INSERT OR IGNORE INTO "
        "insession_auxiliary_v2_node_execution_subject_bindings "
        "(binding_id, session_id, insession_task_id, auxiliary_graph_id, "
        "goal_id, auxiliary_graph_revision, auxiliary_node_id, node_revision, "
        "executor_kind, definition_sha256, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            binding_id,
            session_id,
            subject.task_id,
            subject.auxiliary_graph_id,
            goal_id,
            subject.auxiliary_graph_revision,
            subject.node_id,
            subject.node_revision,
            executor_kind,
            definition_sha256,
            created_at,
        ),
    )
    conn.execute(
        "INSERT OR IGNORE INTO insession_execution_subjects "
        "(execution_subject_id, session_id, insession_task_id, subject_kind, "
        "subject_contract_version, task_node_binding_id, "
        "auxiliary_v2_binding_id, created_at) "
        "VALUES (?, ?, ?, 'auxiliary_node', 'auxiliary_node_v2', "
        "NULL, ?, ?)",
        (
            execution_subject_id,
            session_id,
            subject.task_id,
            binding_id,
            created_at,
        ),
    )
    row = conn.execute(
        "SELECT subject_contract_version, auxiliary_v2_binding_id "
        "FROM insession_execution_subjects WHERE execution_subject_id=? "
        "AND session_id=? AND insession_task_id=?",
        (execution_subject_id, session_id, subject.task_id),
    ).fetchone()
    if (
        row is None
        or str(row["subject_contract_version"]) != "auxiliary_node_v2"
        or str(row["auxiliary_v2_binding_id"]) != binding_id
    ):
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryNode execution-subject authority collision"
        )
    return execution_subject_id


def _require_auxiliary_subject_contract(
    conn: sqlite3.Connection,
    run_row: sqlite3.Row,
) -> AuxiliaryNodeSubject:
    subject = _require_auxiliary_subject(run_row)
    row = conn.execute(
        "SELECT subject.subject_contract_version, "
        "aux2.goal_id, aux2.executor_kind, aux2.definition_sha256 "
        "FROM insession_execution_subjects AS subject "
        "LEFT JOIN insession_auxiliary_v2_node_execution_subject_bindings AS aux2 "
        "ON aux2.binding_id=subject.auxiliary_v2_binding_id "
        "WHERE subject.execution_subject_id=? AND subject.session_id=? "
        "AND subject.insession_task_id=?",
        (
            str(run_row["execution_subject_id"]),
            str(run_row["session_id"]),
            subject.task_id,
        ),
    ).fetchone()
    if row is None or str(row["subject_contract_version"]) != "auxiliary_node_v2":
        raise AuxiliaryGraphPersistenceError(
            "WorkRun is not bound to current AuxiliaryNode authority"
        )
    return subject


def _load_exact_auxiliary_node(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    subject: AuxiliaryNodeSubject,
    expected_status: str | None = None,
) -> tuple[sqlite3.Row, tuple[InSessionTaskAcceptanceProposal, ...]]:
    row = conn.execute(
        "SELECT control.current_goal_id, "
        "control.current_auxiliary_graph_revision, goal.status AS goal_status, "
        "goal.base_task_graph_revision, revision.goal_id, "
        "revision.structure_contract_version, "
        "revision_state.status AS revision_status, "
        "membership.ordinal, membership.required AS membership_required, "
        "membership.carried_completion_id, state.status, state.state_version, "
        "definition.node_kind, definition.executor_kind, definition.title, "
        "definition.objective, definition.source_anchor_ids_json, "
        "definition.acceptance_criteria_json, definition.output_contract, "
        "definition.capability_profile_id, "
        "definition.input_resource_aliases_json, "
        "definition.required AS definition_required, "
        "definition.semantic_fingerprint, "
        "definition.origin_auxiliary_node_id, definition.origin_node_revision, "
        "definition.definition_sha256 "
        "FROM insession_auxiliary_graph_v2_containers AS control "
        "JOIN insession_auxiliary_graph_revision_snapshots AS revision "
        "ON revision.auxiliary_graph_id=control.auxiliary_graph_id "
        "AND revision.auxiliary_graph_revision=? "
        "JOIN insession_auxiliary_graph_goals AS goal "
        "ON goal.goal_id=revision.goal_id "
        "JOIN insession_auxiliary_graph_revision_states_v2 AS revision_state "
        "ON revision_state.auxiliary_graph_id=revision.auxiliary_graph_id "
        "AND revision_state.auxiliary_graph_revision="
        "revision.auxiliary_graph_revision "
        "JOIN insession_auxiliary_graph_revision_nodes_v2 AS membership "
        "ON membership.auxiliary_graph_id=revision.auxiliary_graph_id "
        "AND membership.auxiliary_graph_revision="
        "revision.auxiliary_graph_revision "
        "AND membership.auxiliary_node_id=? AND membership.node_revision=? "
        "JOIN insession_auxiliary_node_definitions_v2 AS definition "
        "ON definition.auxiliary_graph_id=membership.auxiliary_graph_id "
        "AND definition.auxiliary_node_id=membership.auxiliary_node_id "
        "AND definition.node_revision=membership.node_revision "
        "JOIN insession_auxiliary_node_states_v2 AS state "
        "ON state.auxiliary_graph_id=membership.auxiliary_graph_id "
        "AND state.auxiliary_graph_revision=membership.auxiliary_graph_revision "
        "AND state.auxiliary_node_id=membership.auxiliary_node_id "
        "AND state.node_revision=membership.node_revision "
        "WHERE control.session_id=? AND control.insession_task_id=? "
        "AND control.auxiliary_graph_id=?",
        (
            subject.auxiliary_graph_revision,
            subject.node_id,
            subject.node_revision,
            session_id,
            subject.task_id,
            subject.auxiliary_graph_id,
        ),
    ).fetchone()
    if row is None:
        raise AuxiliaryGraphPersistenceError(
            "unknown exact AuxiliaryGraph node version"
        )
    try:
        acceptances = tuple(
            InSessionTaskAcceptanceProposal.model_validate(value)
            for value in json.loads(str(row["acceptance_criteria_json"]))
        )
        origin = (
            AuxiliaryNodeReference(
                node_id=str(row["origin_auxiliary_node_id"]),
                node_revision=int(row["origin_node_revision"]),
            )
            if row["origin_auxiliary_node_id"] is not None
            and row["origin_node_revision"] is not None
            else None
        )
        materialized = AuxiliaryNodeDefinition(
            node_id=subject.node_id,
            node_revision=subject.node_revision,
            ordinal=int(row["ordinal"]),
            node_kind=AuxiliaryNodeKind(str(row["node_kind"])),
            executor_kind=AuxiliaryNodeExecutorKind(
                str(row["executor_kind"])
            ),
            title=str(row["title"]),
            objective=str(row["objective"]),
            acceptance_criteria=acceptances,
            capability_profile_id=(
                str(row["capability_profile_id"])
                if row["capability_profile_id"] is not None
                else None
            ),
            input_resource_aliases=tuple(
                json.loads(str(row["input_resource_aliases_json"]))
            ),
            source_anchor_ids=tuple(
                json.loads(str(row["source_anchor_ids_json"]))
            ),
            output_contract=str(row["output_contract"]),
            required=bool(row["definition_required"]),
            semantic_fingerprint=str(row["semantic_fingerprint"]),
            origin_node_ref=origin,
        )
    except (TypeError, ValueError) as exc:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph node definition or Acceptance is corrupt"
        ) from exc
    if (
        bool(row["membership_required"]) != bool(row["definition_required"])
        or materialized.semantic_fingerprint != str(row["definition_sha256"])
    ):
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph exact node definition hash is corrupt"
        )
    if expected_status is not None and str(row["status"]) != expected_status:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph node state is not at the required checkpoint"
        )
    return row, acceptances


def _load_auxiliary_request_row(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    verification_request_id: str,
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT request.* FROM insession_work_run_verification_requests AS request "
        "JOIN insession_execution_subjects AS subject "
        "ON subject.execution_subject_id=request.execution_subject_id "
        "WHERE request.session_id=? AND request.verification_request_id=? "
        "AND request.subject_kind='auxiliary_node' "
        "AND subject.subject_contract_version='auxiliary_node_v2'",
        (session_id, verification_request_id),
    ).fetchone()
    if row is None:
        raise AuxiliaryGraphPersistenceError(
            "unknown AuxiliaryGraph verification request"
        )
    return row


def _revalidate_auxiliary_request_binding(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    expected_node_status: str = "active",
) -> None:
    record = _auxiliary_record_from_row(row)
    request = record.request
    run_row = _require_run_row(conn, request.work_run_id, request.session_id)
    subject = _require_auxiliary_subject_contract(
        conn,
        run_row,
    )
    if subject != request.subject:
        raise AuxiliaryGraphPersistenceError(
            "Auxiliary verification subject has drifted"
        )
    node, acceptances = _load_exact_auxiliary_node(
        conn,
        session_id=request.session_id,
        subject=subject,
        expected_status=expected_node_status,
    )
    progress = _load_progress(conn, request.work_run_id)
    output, output_hash = _load_output_window(conn, request.work_run_id)
    submitted_attempt = _load_current_submit_attempt(
        conn,
        work_run_id=request.work_run_id,
        output_revision=request.output_revision,
    )
    supporting_results = _load_supporting_tool_results(
        conn,
        work_run_id=request.work_run_id,
        result_ids=request.supporting_tool_result_ids,
        before_attempt_ordinal=submitted_attempt.ordinal,
    )
    binding_hash = _auxiliary_verification_binding_hash(
        verification_request_id=request.verification_request_id,
        session_id=request.session_id,
        request_turn_id=request.request_turn_id,
        work_run_id=request.work_run_id,
        subject=subject,
        node_title=str(node["title"]),
        node_objective=str(node["objective"]),
        submitted_attempt=submitted_attempt,
        locked_work_run_revision=request.locked_work_run_revision,
        progress=progress,
        output=output,
        output_hash=output_hash,
        acceptances=acceptances,
        supporting_results=supporting_results,
        prepared_budget=request.prepared_budget,
    )
    if binding_hash != str(row["request_binding_hash"]):
        raise AuxiliaryGraphPersistenceError(
            "Auxiliary verification immutable binding has drifted"
        )


def _auxiliary_record_from_row(row: sqlite3.Row) -> TaskNodeVerificationRecord:
    try:
        dependency_delivery_ids = tuple(
            json.loads(str(row["dependency_delivery_ids_json"]))
        )
        if dependency_delivery_ids:
            raise ValueError(
                "Auxiliary planning verification cannot consume TaskNode deliveries"
            )
        subject = AuxiliaryNodeSubject(
            task_id=str(row["insession_task_id"]),
            auxiliary_graph_id=str(row["auxiliary_graph_id"]),
            auxiliary_graph_revision=int(row["auxiliary_graph_revision"]),
            node_id=str(row["auxiliary_node_id"]),
            node_revision=int(row["node_revision"]),
        )
        request = TaskNodeVerificationRequest(
            verification_request_id=str(row["verification_request_id"]),
            session_id=str(row["session_id"]),
            request_turn_id=str(row["request_turn_id"]),
            work_run_id=str(row["work_run_id"]),
            subject=subject,
            submitted_attempt_id=str(row["submitted_attempt_id"]),
            output_revision=int(row["output_revision"]),
            acceptance_progress_revision=int(row["acceptance_progress_revision"]),
            acceptance_ids=tuple(json.loads(str(row["acceptance_ids_json"]))),
            supporting_tool_result_ids=tuple(
                json.loads(str(row["supporting_tool_result_ids_json"]))
            ),
            locked_work_run_revision=int(row["locked_work_run_revision"]),
            prepared_budget=_budget_from_json(row["prepared_budget_json"]),
            revision=int(row["request_revision"]),
            status=TaskNodeVerificationRequestStatus(str(row["status"])),
            technical_error_code=(
                str(row["technical_error_code"])
                if row["technical_error_code"] is not None
                else None
            ),
        )
        semantic_result = (
            NodeVerificationResult.model_validate_json(str(row["result_json"]))
            if row["result_json"] is not None
            else None
        )
        return TaskNodeVerificationRecord(
            request=request,
            result=semantic_result,
        )
    except (TypeError, ValueError) as exc:
        raise AuxiliaryGraphPersistenceError(
            "stored Auxiliary verification request is corrupt"
        ) from exc


def _auxiliary_verification_binding_hash(
    *,
    verification_request_id: str,
    session_id: str,
    request_turn_id: str,
    work_run_id: str,
    subject: AuxiliaryNodeSubject,
    node_title: str,
    node_objective: str,
    submitted_attempt: Attempt,
    locked_work_run_revision: int,
    progress: AcceptanceProgressSnapshot,
    output: OutputWindow,
    output_hash: str,
    acceptances: tuple[InSessionTaskAcceptanceProposal, ...],
    supporting_results: tuple[SupportingToolResult, ...],
    prepared_budget: WorkRunBudget,
) -> str:
    return _payload_hash(
        {
            "verification_request_id": verification_request_id,
            "session_id": session_id,
            "request_turn_id": request_turn_id,
            "work_run_id": work_run_id,
            "subject": subject.model_dump(mode="json"),
            "node_title": node_title,
            "node_objective": node_objective,
            "submitted_attempt": submitted_attempt.model_dump(mode="json"),
            "locked_work_run_revision": locked_work_run_revision,
            "acceptance_progress": progress.model_dump(mode="json"),
            "output_window": output.model_dump(mode="json"),
            "output_window_snapshot_hash": output_hash,
            "acceptances": [item.model_dump(mode="json") for item in acceptances],
            "supporting_tool_results": [
                item.model_dump(mode="json") for item in supporting_results
            ],
            "prepared_budget": prepared_budget.model_dump(mode="json"),
        }
    )


def _budget_from_json(raw: object) -> WorkRunBudget:
    return WorkRunBudget.model_validate_json(str(raw))


__all__ = [
    "_auxiliary_record_from_row",
    "_auxiliary_verification_binding_hash",
    "_budget_from_json",
    "_ensure_auxiliary_execution_subject",
    "_load_auxiliary_request_row",
    "_load_exact_auxiliary_node",
    "_require_auxiliary_subject",
    "_require_auxiliary_subject_contract",
    "_revalidate_auxiliary_request_binding",
]
