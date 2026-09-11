"""Store 所有的当前 AuxiliaryGraph 节点依赖解析。

依赖边绑定持久完成身份。读取侧从精确当前 revision 和刻意收窄的纯模型结转权威中解析
本地完成项，只有重新认证其 WorkRun 或 Host 原语结算后才返回 Prompt 安全正文。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from typing import Any

from personagraph.l2.auxiliary_graph import (
    PlanningContextArtifactProjection,
    PlanningContextArtifact,
)
from personagraph.l2.auxiliary_graph.dependency_projection import (
    AuxiliaryDependencyBundle,
    AuxiliaryHostContextDependency,
    AuxiliaryModelOutputDependency,
)
from personagraph.l2.work_run import (
    AuxiliaryNodeSubject,
    NodeVerificationResult,
    OutputWindow,
)
from .auxiliary_graph_errors import AuxiliaryGraphPersistenceError
from .auxiliary_graphs import (
    StoredAuxiliaryGraphDetails,
    StoredAuxiliaryGraphNode,
    _load_pure_model_carry_receipt,
    _load_auxiliary_graph,
)
from ...deps import StoreDeps
from ..planning.primitive_invocations import (
    PlanningPrimitiveInvocationPersistenceError,
    _load_by_call_id as _load_primitive_invocation,
)


class AuxiliaryDependencyPersistenceError(AuxiliaryGraphPersistenceError):
    """依赖正文失去当前 完成权威。"""


def resolve_auxiliary_dependencies(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    consumer_subject: AuxiliaryNodeSubject,
) -> AuxiliaryDependencyBundle:
    """在一个读取事务内解析精确传入依赖正文。

    调用方不提供完成 ID 或生产者正文，二者都从当前图派生。因此即使依赖包为空，仍与
    已认证消费者和当前结构哈希绑定。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("turn_id", turn_id)
    if not isinstance(consumer_subject, AuxiliaryNodeSubject):
        raise TypeError("consumer_subject must be AuxiliaryNodeSubject")
    if consumer_subject.task_id.strip() != consumer_subject.task_id:
        raise ValueError("consumer Task ID must be canonical")

    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        current_task_graph_revision = _require_current_turn_task(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            task_id=consumer_subject.task_id,
        )
        try:
            details = _load_auxiliary_graph(
                conn,
                session_id,
                consumer_subject.task_id,
            )
        except AuxiliaryGraphPersistenceError as exc:
            raise AuxiliaryDependencyPersistenceError(
                "current AuxiliaryGraph authority is invalid"
            ) from exc
        if details.base_task_graph_revision != current_task_graph_revision:
            raise AuxiliaryDependencyPersistenceError(
                "dependency graph base differs from the current TaskGraph"
            )
        consumer = _require_current_consumer(
            details,
            session_id=session_id,
            subject=consumer_subject,
        )
        incoming = _incoming_producers(details, consumer=consumer)
        carried_by_node = _load_carried_dependency_ids(
            conn,
            details=details,
            producers=incoming,
        )

        items: list[
            AuxiliaryModelOutputDependency
            | AuxiliaryHostContextDependency
        ] = []
        for producer in incoming:
            if producer.status != "completed":
                raise AuxiliaryDependencyPersistenceError(
                    "incoming AuxiliaryGraph producer is not completed"
                )
            subject = AuxiliaryNodeSubject(
                task_id=details.task_id,
                auxiliary_graph_id=details.auxiliary_graph_id,
                auxiliary_graph_revision=details.auxiliary_graph_revision,
                node_id=producer.auxiliary_node_id,
                node_revision=producer.node_revision,
            )
            carry_receipt_id = carried_by_node[producer.auxiliary_node_id]
            if producer.executor_kind == "host_primitive":
                if carry_receipt_id is not None:
                    raise AuxiliaryDependencyPersistenceError(
                        "Host primitive dependency cannot use model carry authority"
                    )
                items.append(
                    _resolve_host_dependency(
                        conn,
                        details=details,
                        producer=producer,
                        subject=subject,
                    )
                )
            else:
                source_subject: AuxiliaryNodeSubject | None = None
                expected_completion_id: str | None = None
                if carry_receipt_id is not None:
                    receipt = _load_pure_model_carry_receipt(
                        conn,
                        details=details,
                        target_node=producer,
                        carry_receipt_id=carry_receipt_id,
                    )
                    source_subject = receipt.source_subject
                    expected_completion_id = receipt.source_completion_id
                items.append(
                    _resolve_model_dependency(
                        conn,
                        details=details,
                        producer=producer,
                        subject=subject,
                        source_subject=source_subject,
                        expected_completion_id=expected_completion_id,
                    )
                )

        try:
            bundle = AuxiliaryDependencyBundle.create(
                session_id=session_id,
                task_id=details.task_id,
                auxiliary_graph_id=details.auxiliary_graph_id,
                auxiliary_graph_revision=details.auxiliary_graph_revision,
                consumer_subject=consumer_subject,
                consumer_node_alias=consumer.local_node_key,
                structure_sha256=details.structure_sha256,
                items=tuple(items),
            )
        except (TypeError, ValueError) as exc:
            raise AuxiliaryDependencyPersistenceError(
                "resolved dependency bundle violates its prompt contract"
            ) from exc
        conn.commit()
        return bundle
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def _require_current_turn_task(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
) -> int | None:
    turn = conn.execute(
        "SELECT session_id FROM runtime_turns WHERE turn_id=?",
        (turn_id,),
    ).fetchone()
    if turn is None or str(turn["session_id"]) != session_id:
        raise AuxiliaryDependencyPersistenceError(
            "dependency resolver Turn is outside this Session"
        )
    if conn.execute(
        "SELECT 1 FROM insession_task_turn_links WHERE session_id=? "
        "AND turn_id=? AND insession_task_id=?",
        (session_id, turn_id, task_id),
    ).fetchone() is None:
        raise AuxiliaryDependencyPersistenceError(
            "dependency resolver Turn is not linked to its Task"
        )
    task = conn.execute(
        "SELECT current_status, current_graph_revision FROM insession_tasks "
        "WHERE session_id=? "
        "AND insession_task_id=?",
        (session_id, task_id),
    ).fetchone()
    if task is None or str(task["current_status"]) in {"completed", "cancelled"}:
        raise AuxiliaryDependencyPersistenceError(
            "dependency resolver Task is terminal or missing"
        )
    return (
        int(task["current_graph_revision"])
        if task["current_graph_revision"] is not None
        else None
    )


def _require_current_consumer(
    details: StoredAuxiliaryGraphDetails,
    *,
    session_id: str,
    subject: AuxiliaryNodeSubject,
) -> StoredAuxiliaryGraphNode:
    if (
        details.session_id != session_id
        or details.task_id != subject.task_id
        or details.auxiliary_graph_id != subject.auxiliary_graph_id
        or details.auxiliary_graph_revision
        != subject.auxiliary_graph_revision
        or details.revision is None
        or details.goal_status != "active"
        or details.revision_status != "active"
    ):
        raise AuxiliaryDependencyPersistenceError(
            "dependency consumer is outside the active current graph"
        )
    matches = tuple(
        node
        for node in details.nodes
        if node.auxiliary_node_id == subject.node_id
        and node.node_revision == subject.node_revision
    )
    if len(matches) != 1:
        raise AuxiliaryDependencyPersistenceError(
            "dependency consumer is absent or ambiguous in the current revision"
        )
    consumer = matches[0]
    if consumer.status in {"completed", "failed", "cancelled", "superseded"}:
        raise AuxiliaryDependencyPersistenceError(
            "terminal AuxiliaryGraph node cannot consume dependency input"
        )
    return consumer


def _incoming_producers(
    details: StoredAuxiliaryGraphDetails,
    *,
    consumer: StoredAuxiliaryGraphNode,
) -> tuple[StoredAuxiliaryGraphNode, ...]:
    nodes = {node.auxiliary_node_id: node for node in details.nodes}
    incoming_ids: list[str] = []
    for edge in details.edges:
        if edge.consumer_auxiliary_node_id != consumer.auxiliary_node_id:
            continue
        if not edge.required or edge.dependency_auxiliary_node_id in incoming_ids:
            raise AuxiliaryDependencyPersistenceError(
                "current dependency edge authority is invalid"
            )
        incoming_ids.append(edge.dependency_auxiliary_node_id)
    try:
        producers = tuple(nodes[node_id] for node_id in incoming_ids)
    except KeyError as exc:
        raise AuxiliaryDependencyPersistenceError(
            "current dependency edge references an unknown producer"
        ) from exc
    return tuple(
        sorted(
            producers,
            key=lambda node: (node.ordinal, node.auxiliary_node_id),
        )
    )


def _load_carried_dependency_ids(
    conn: sqlite3.Connection,
    *,
    details: StoredAuxiliaryGraphDetails,
    producers: tuple[StoredAuxiliaryGraphNode, ...],
) -> dict[str, str | None]:
    if not producers:
        return {}
    placeholders = ",".join("?" for _ in producers)
    rows = conn.execute(
        "SELECT auxiliary_node_id, carried_completion_id FROM "
        "insession_auxiliary_graph_revision_nodes_v2 WHERE "
        "auxiliary_graph_id=? AND auxiliary_graph_revision=? AND "
        f"auxiliary_node_id IN ({placeholders})",
        (
            details.auxiliary_graph_id,
            details.auxiliary_graph_revision,
            *(node.auxiliary_node_id for node in producers),
        ),
    ).fetchall()
    if len(rows) != len(producers):
        raise AuxiliaryDependencyPersistenceError(
            "dependency membership changed during resolution"
        )
    return {
        str(row["auxiliary_node_id"]): (
            str(row["carried_completion_id"])
            if row["carried_completion_id"] is not None
            else None
        )
        for row in rows
    }


def _resolve_model_dependency(
    conn: sqlite3.Connection,
    *,
    details: StoredAuxiliaryGraphDetails,
    producer: StoredAuxiliaryGraphNode,
    subject: AuxiliaryNodeSubject,
    source_subject: AuxiliaryNodeSubject | None = None,
    expected_completion_id: str | None = None,
) -> AuxiliaryModelOutputDependency:
    completion_subject = source_subject or subject
    rows = conn.execute(
        "SELECT completion.*, run.execution_subject_id, "
        "run.status AS work_run_status, run.reason AS work_run_reason, "
        "run.current_attempt_id, run.current_verification_request_id, "
        "subject.subject_contract_version, binding.goal_id AS bound_goal_id, "
        "binding.executor_kind AS bound_executor_kind, "
        "binding.definition_sha256 AS bound_definition_sha256, "
        "definition.definition_sha256, definition.output_contract, "
        "verification.status AS verification_status, "
        "verification.result_json, verification.all_pass, "
        "output.snapshot_json AS output_json, "
        "output.snapshot_hash AS output_snapshot_sha256, output.frozen_at "
        "FROM insession_auxiliary_node_completions_v2 AS completion "
        "JOIN insession_work_runs AS run "
        "ON run.work_run_id=completion.work_run_id "
        "JOIN insession_execution_subjects AS subject "
        "ON subject.execution_subject_id=run.execution_subject_id "
        "JOIN insession_auxiliary_v2_node_execution_subject_bindings AS binding "
        "ON binding.binding_id=subject.auxiliary_v2_binding_id "
        "JOIN insession_auxiliary_node_definitions_v2 AS definition "
        "ON definition.auxiliary_graph_id=completion.auxiliary_graph_id "
        "AND definition.auxiliary_node_id=completion.auxiliary_node_id "
        "AND definition.node_revision=completion.node_revision "
        "JOIN insession_work_run_verification_requests AS verification "
        "ON verification.work_run_id=completion.work_run_id "
        "AND verification.verification_request_id="
        "completion.verification_request_id "
        "JOIN insession_work_run_output_windows AS output "
        "ON output.work_run_id=completion.work_run_id "
        "AND output.output_revision=completion.output_revision "
        "WHERE completion.session_id=? AND completion.insession_task_id=? "
        "AND completion.auxiliary_graph_id=? "
        "AND completion.auxiliary_graph_revision=? "
        "AND completion.auxiliary_node_id=? AND completion.node_revision=?",
        (
            details.session_id,
            details.task_id,
            details.auxiliary_graph_id,
            completion_subject.auxiliary_graph_revision,
            completion_subject.node_id,
            completion_subject.node_revision,
        ),
    ).fetchall()
    if len(rows) != 1:
        raise AuxiliaryDependencyPersistenceError(
            "model dependency has no unique local completion"
        )
    row = rows[0]
    if expected_completion_id is not None and (
        str(row["completion_id"]) != expected_completion_id
    ):
        raise AuxiliaryDependencyPersistenceError(
            "carried model dependency completion identity changed"
        )
    try:
        output_json = str(row["output_json"])
        output = OutputWindow.model_validate_json(output_json)
        result_json = str(row["result_json"])
        verification = NodeVerificationResult.model_validate_json(result_json)
    except (TypeError, ValueError, RecursionError) as exc:
        raise AuxiliaryDependencyPersistenceError(
            "model dependency contains invalid output or verification JSON"
        ) from exc

    completion_payload = {
        "schema_version": "auxiliary-node-completion-v2",
        "completion_id": str(row["completion_id"]),
        "execution_subject_id": str(row["execution_subject_id"]),
        "subject": completion_subject.model_dump(mode="json"),
        "goal_id": details.goal_id,
        "definition_sha256": producer.semantic_fingerprint,
        "verification_request_id": str(row["verification_request_id"]),
        "verification_result_sha256": _sha256_text(result_json),
        "submitted_attempt_id": str(row["submitted_attempt_id"]),
        "output_revision": int(row["output_revision"]),
        "output_snapshot_sha256": str(row["output_snapshot_sha256"]),
    }
    completion_json = str(row["completion_json"])
    valid = (
        producer.executor_kind
        in {"model_work_run", "user_gate", "terminal_planner"}
        and str(row["subject_contract_version"]) == "auxiliary_node_v2"
        and str(row["bound_goal_id"]) == details.goal_id
        and str(row["bound_executor_kind"]) == producer.executor_kind
        and str(row["bound_definition_sha256"]) == producer.semantic_fingerprint
        and str(row["definition_sha256"]) == producer.semantic_fingerprint
        and str(row["output_contract"]) == producer.output_contract
        and str(row["work_run_status"]) == "completed"
        and str(row["work_run_reason"]) == "verification_passed"
        and row["current_attempt_id"] is None
        and row["current_verification_request_id"] is None
        and str(row["verification_status"]) == "completed"
        and row["all_pass"] == 1
        and row["frozen_at"] is not None
        and output.work_run_id == str(row["work_run_id"])
        and output.output_revision == int(row["output_revision"])
        and output.updated_attempt_id is not None
        and bool(output.content.strip())
        and "\x00" not in output.content
        and output_json == _model_json(output)
        and _sha256_text(output_json) == str(row["output_snapshot_sha256"])
        and verification.all_pass
        and verification.subject == completion_subject
        and verification.work_run_id == str(row["work_run_id"])
        and verification.verification_request_id
        == str(row["verification_request_id"])
        and verification.submitted_attempt_id == str(row["submitted_attempt_id"])
        and verification.output_revision == int(row["output_revision"])
        and result_json == _model_json(verification)
        and completion_json == _canonical_json(completion_payload)
        and _sha256_text(completion_json) == str(row["completion_sha256"])
    )
    if not valid:
        raise AuxiliaryDependencyPersistenceError(
            "model dependency completion binding is corrupt"
        )
    try:
        return AuxiliaryModelOutputDependency.create(
            completion_id=str(row["completion_id"]),
            producer_subject=subject,
            producer_node_alias=producer.local_node_key,
            producer_ordinal=producer.ordinal,
            output_contract=producer.output_contract,
            output_revision=output.output_revision,
            output_format=output.format,
            content=output.content,
            output_snapshot_sha256=str(row["output_snapshot_sha256"]),
            verification_result_sha256=_sha256_text(result_json),
            completion_sha256=str(row["completion_sha256"]),
        )
    except (TypeError, ValueError) as exc:
        raise AuxiliaryDependencyPersistenceError(
            "model dependency cannot be projected into the prompt contract"
        ) from exc


def _resolve_host_dependency(
    conn: sqlite3.Connection,
    *,
    details: StoredAuxiliaryGraphDetails,
    producer: StoredAuxiliaryGraphNode,
    subject: AuxiliaryNodeSubject,
) -> AuxiliaryHostContextDependency:
    rows = conn.execute(
        "SELECT artifact.*, observation.observation_kind, "
        "observation.outcome, observation.request_fingerprint, "
        "observation.data_version, "
        "observation.snapshot_json AS observation_json, "
        "observation.snapshot_sha256 AS observation_sha256, "
        "receipt.disposition, receipt.receipt_json, "
        "receipt.receipt_sha256 AS stored_receipt_sha256 "
        "FROM insession_auxiliary_planning_context_artifacts AS artifact "
        "JOIN insession_auxiliary_observations AS observation "
        "ON observation.observation_id=artifact.producer_primitive_call_id "
        "JOIN insession_auxiliary_context_verification_receipts AS receipt "
        "ON receipt.verification_receipt_id=artifact.verification_receipt_id "
        "AND receipt.source_observation_id=observation.observation_id "
        "WHERE artifact.session_id=? AND artifact.insession_task_id=? "
        "AND artifact.auxiliary_graph_id=? AND artifact.goal_id=? "
        "AND artifact.producer_auxiliary_graph_revision=? "
        "AND artifact.producer_auxiliary_node_id=? "
        "AND artifact.producer_node_revision=? "
        "AND artifact.producer_primitive_call_id IS NOT NULL",
        (
            details.session_id,
            details.task_id,
            details.auxiliary_graph_id,
            details.goal_id,
            details.auxiliary_graph_revision,
            producer.auxiliary_node_id,
            producer.node_revision,
        ),
    ).fetchall()
    if len(rows) != 1:
        raise AuxiliaryDependencyPersistenceError(
            "Host dependency has no unique local artifact"
        )
    row = rows[0]
    primitive_call_id = str(row["producer_primitive_call_id"])
    try:
        reservation = _load_primitive_invocation(conn, primitive_call_id)
    except PlanningPrimitiveInvocationPersistenceError as exc:
        raise AuxiliaryDependencyPersistenceError(
            "Host dependency primitive reservation is corrupt"
        ) from exc
    if reservation is None:
        raise AuxiliaryDependencyPersistenceError(
            "Host dependency primitive reservation is missing"
        )
    try:
        artifact_json = str(row["artifact_json"])
        artifact = PlanningContextArtifact.model_validate_json(artifact_json)
        observation_json = str(row["observation_json"])
        observation = _require_mapping(json.loads(observation_json))
        result = _require_mapping(observation.get("result"))
        prompt_inputs = _require_mapping(result.get("prompt_inputs"))
        projection = PlanningContextArtifactProjection.model_validate(
            prompt_inputs.get("context_artifact")
        )
        receipt_json = str(row["receipt_json"])
        receipt = _require_mapping(json.loads(receipt_json))
    except (TypeError, ValueError, RecursionError) as exc:
        raise AuxiliaryDependencyPersistenceError(
            "Host dependency contains invalid sealed JSON"
        ) from exc

    binding = reservation.invocation.binding
    valid = (
        producer.executor_kind == "host_primitive"
        and reservation.status == "settled"
        and reservation.settled_observation_id == primitive_call_id
        and reservation.settled_artifact_id == artifact.artifact_id
        and reservation.settlement_sha256 == result.get("settlement_sha256")
        and reservation.invocation.structure_sha256 == details.structure_sha256
        and binding.session_id == details.session_id
        and binding.task_id == details.task_id
        and binding.auxiliary_graph_id == details.auxiliary_graph_id
        and binding.goal_id == details.goal_id
        and binding.producer_auxiliary_node == subject
        and binding.artifact_id == artifact.artifact_id
        and binding.verification_receipt_id == artifact.verification_receipt_id
        and binding.producer_node_alias == producer.local_node_key
        and artifact.session_id == details.session_id
        and artifact.task_id == details.task_id
        and artifact.auxiliary_graph_id == details.auxiliary_graph_id
        and artifact.goal_id == details.goal_id
        and artifact.producer_auxiliary_node == subject
        and artifact.producer_primitive_call_id == primitive_call_id
        and artifact.artifact_sha256 == str(row["artifact_sha256"])
        and artifact.verification_receipt_sha256
        == str(row["verification_receipt_sha256"])
        and artifact.verification_receipt_sha256
        == str(row["stored_receipt_sha256"])
        and artifact.scope_snapshot_sha256
        == str(row["scope_snapshot_sha256"])
        and artifact.freshness_manifest_sha256
        == str(row["freshness_manifest_sha256"])
        and artifact_json == _model_json(artifact)
        and str(row["facts_json"])
        == _canonical_json(
            [item.model_dump(mode="json") for item in artifact.facts]
        )
        and str(row["constraints_json"])
        == _canonical_json(
            [item.model_dump(mode="json") for item in artifact.constraints]
        )
        and str(row["conflicts_json"])
        == _canonical_json(
            [item.model_dump(mode="json") for item in artifact.conflicts]
        )
        and str(row["gaps_json"])
        == _canonical_json(
            [item.model_dump(mode="json") for item in artifact.gaps]
        )
        and str(row["evidence_refs_json"])
        == _canonical_json(
            [item.model_dump(mode="json") for item in artifact.evidence_refs]
        )
        and observation.get("schema_version")
        == "sealed-planning-host-primitive-observation-v1"
        and observation.get("primitive_kind")
        == reservation.invocation.primitive_kind.value
        and str(row["outcome"]) == result.get("observation_status")
        and observation_json == _canonical_json(observation)
        and _sha256_text(observation_json) == str(row["observation_sha256"])
        and result.get("logical_request_sha256")
        == reservation.invocation.logical_request_sha256
        and result.get("logical_request_json")
        == reservation.invocation.logical_request_json
        and _canonical_json_hash_is_valid(
            result.get("raw_observation_json"),
            result.get("raw_observation_sha256"),
        )
        and str(row["request_fingerprint"])
        == reservation.invocation.logical_request_sha256
        and result.get("freshness_manifest_sha256")
        == artifact.freshness_manifest_sha256
        and str(row["data_version"])
        == artifact.freshness_manifest_sha256
        and result.get("artifact") == artifact.model_dump(mode="json")
        and result.get("verification_receipt_sha256")
        == artifact.verification_receipt_sha256
        and projection.artifact_alias == binding.artifact_alias
        and projection.artifact_id == artifact.artifact_id
        and projection.artifact_sha256 == artifact.artifact_sha256
        and projection.producer_node_alias == producer.local_node_key
        and receipt_json == _canonical_json(receipt)
        and _sha256_text(receipt_json) == str(row["stored_receipt_sha256"])
        and _receipt_binds_result(
            receipt,
            result=result,
            artifact=artifact,
            primitive_call_id=primitive_call_id,
            primitive_kind=reservation.invocation.primitive_kind.value,
        )
        and _projection_binds_artifact(projection, artifact=artifact)
    )
    if not valid:
        raise AuxiliaryDependencyPersistenceError(
            "Host dependency settlement binding is corrupt"
        )
    expected_disposition = {
        "success": "pass",
        "partial": "partial",
    }.get(str(row["outcome"]), "fail")
    if str(row["disposition"]) != expected_disposition:
        raise AuxiliaryDependencyPersistenceError(
            "Host dependency verification disposition is corrupt"
        )
    try:
        return AuxiliaryHostContextDependency.create(
            completion_id=artifact.artifact_id,
            producer_subject=subject,
            producer_node_alias=producer.local_node_key,
            producer_ordinal=producer.ordinal,
            output_contract=producer.output_contract,
            artifact=projection,
            observation_settlement_sha256=str(reservation.settlement_sha256),
            verification_receipt_sha256=artifact.verification_receipt_sha256,
        )
    except (TypeError, ValueError) as exc:
        raise AuxiliaryDependencyPersistenceError(
            "Host dependency cannot be projected into the prompt contract"
        ) from exc


def _receipt_binds_result(
    receipt: Mapping[str, Any],
    *,
    result: Mapping[str, Any],
    artifact: PlanningContextArtifact,
    primitive_call_id: str,
    primitive_kind: str,
) -> bool:
    schema = receipt.get("schema_version")
    if schema == "planning-resource-verification-receipt-v1":
        kind_valid = primitive_kind == "resource_perception"
    else:
        kind_valid = (
            schema == "planning-context-primitive-verification-receipt-v1"
            and receipt.get("primitive_kind") == primitive_kind
        )
    return bool(
        kind_valid
        and receipt.get("primitive_call_id") == primitive_call_id
        and receipt.get("observation_status") == result.get("observation_status")
        and receipt.get("logical_request_sha256")
        == result.get("logical_request_sha256")
        and receipt.get("raw_observation_sha256")
        == result.get("raw_observation_sha256")
        and receipt.get("freshness_manifest_sha256")
        == result.get("freshness_manifest_sha256")
        and receipt.get("authority_anchors") == result.get("authority_anchors")
        and receipt.get("facts")
        == [item.model_dump(mode="json") for item in artifact.facts]
        and receipt.get("gaps")
        == [item.model_dump(mode="json") for item in artifact.gaps]
    )


def _projection_binds_artifact(
    projection: PlanningContextArtifactProjection,
    *,
    artifact: PlanningContextArtifact,
) -> bool:
    aliases_by_anchor = {
        item.evidence_anchor_id: item.source_alias
        for item in artifact.evidence_refs
    }

    def aliases(anchor_ids: tuple[str, ...]) -> tuple[str, ...] | None:
        try:
            return tuple(aliases_by_anchor[item] for item in anchor_ids)
        except KeyError:
            return None

    return bool(
        tuple(
            (item.fact_alias, item.statement, item.evidence_aliases)
            for item in projection.facts
        )
        == tuple(
            (item.fact_id, item.statement, aliases(item.evidence_anchor_ids))
            for item in artifact.facts
        )
        and tuple(
            (
                item.constraint_alias,
                item.statement,
                item.authorization_aliases,
            )
            for item in projection.constraints
        )
        == tuple(
            (
                item.constraint_id,
                item.statement,
                aliases(item.authorization_anchor_ids),
            )
            for item in artifact.constraints
        )
        and tuple(
            (
                item.conflict_alias,
                item.statement,
                item.evidence_aliases,
                item.blocking,
            )
            for item in projection.conflicts
        )
        == tuple(
            (
                item.conflict_id,
                item.statement,
                aliases(item.evidence_anchor_ids),
                item.blocking,
            )
            for item in artifact.conflicts
        )
        and tuple(
            (
                item.gap_alias,
                item.observation_status,
                item.description,
                item.blocking,
                item.affected_obligations,
                item.evidence_aliases,
                item.resolution_hint,
            )
            for item in projection.gaps
        )
        == tuple(
            (
                item.gap_id,
                item.observation_status,
                item.description,
                item.blocking,
                item.affected_obligations,
                aliases(item.evidence_anchor_ids) or (),
                item.resolution_hint,
            )
            for item in artifact.gaps
        )
    )


def _require_mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("stored dependency payload must be an object")
    return value


def _canonical_json_hash_is_valid(value: object, digest: object) -> bool:
    if not isinstance(value, str) or not isinstance(digest, str):
        return False
    try:
        payload = json.loads(value)
    except (TypeError, ValueError, RecursionError):
        return False
    return value == _canonical_json(payload) and _sha256_text(value) == digest


def _require_identifier(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 200
    ):
        raise ValueError(f"{name} must be a canonical identifier")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _model_json(value: object) -> str:
    return _canonical_json(value.model_dump(mode="json"))  # type: ignore[attr-defined]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "AuxiliaryDependencyPersistenceError",
    "resolve_auxiliary_dependencies",
]
