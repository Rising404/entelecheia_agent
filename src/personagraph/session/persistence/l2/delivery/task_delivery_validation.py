"""持久化的整 Task 验证、发布结算与 revision trigger。

节点 FinishGate 仍负责证明完整的当前 Delivery tree。本模块负责下一道边界：在模型
调用前冻结精确的已完成 TaskGraph 与规范根 Delivery，要求结果必须是通用模型调用
账本中存储的有类型成功结果，并且只结算一次 PASS/REVISE/BLOCKED。

只有 graph-revision 路由会变更 Task 状态。在存储结算的同一事务中，它会追加一个
Prompt 安全的 ``TaskGraphRevisionTrigger``，并将精确的已完成 Task 从
``completed`` 移回 ``active``。缺失信息 BLOCKED 也走同一路径，使真正的
``user_gate`` 能够持有该问题；它绝不会让 Task 陷入仅含文本的 awaiting-user
死胡同。任何通用 reopen 路径都无法制造这一权威状态。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personagraph.l2.task_graph import (
    InSessionTaskAcceptanceProposal,
    InSessionTaskNodeKind,
    TaskDeliveryValidationChildDeliveryMaterial,
    TaskDeliveryValidationChildDeliveryProjection,
    TaskDeliveryValidationDisposition,
    TaskDeliveryValidationFaultDomain,
    TaskDeliveryValidationFinding,
    TaskDeliveryValidationNode,
    TaskDeliveryValidationNodeVerificationAttestation,
    TaskDeliveryValidationPrompt,
    TaskDeliveryValidationRequest,
    TaskDeliveryValidationResult,
    TaskDeliveryValidationVerdict,
    TaskGraphRevisionTriggerApplication,
    TaskGraphRevisionTrigger,
    build_task_delivery_validation_child_delivery_projection,
    validate_task_delivery_validation_result,
)
from .....runtime.model_calls.contracts import RuntimeModelPhysicalOutcome
from personagraph.l2.work_run import NodeVerificationResult, TaskNodeSubject
from ..task_graph import insession_tasks as task_records
from ...calls import runtime_model_calls as model_call_records
from ..work_run import work_execution as execution_records
from ...deps import StoreDeps


_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_CANDIDATE_CALL_KIND = "task_delivery_candidate_validation"
_CANDIDATE_PURPOSE = "runtime_task_delivery_validation_v2"
_CANDIDATE_REQUEST_CONTRACT = "task-delivery-validation-model-call-v2"
_CANDIDATE_RESULT_CONTRACT = "task-delivery-validation-model-envelope-v2"


class TaskDeliveryValidationPersistenceError(RuntimeError):
    """某项整 Task 验证权威不变量以失败关闭方式触发。"""


class TaskDeliveryValidationIdentityCollision(
    TaskDeliveryValidationPersistenceError
):
    """某个 request、result、settlement、trigger 或 application ID 被重复使用。"""


class TaskDeliveryValidationStaleAuthority(
    TaskDeliveryValidationPersistenceError
):
    """精确的已完成 TaskGraph/root Delivery CAS 已不再是当前值。"""


class TaskDeliveryValidationStoredAuthorityCorrupt(
    TaskDeliveryValidationPersistenceError
):
    """已存储的不可变验证记录已无法通过自身认证。"""


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskDeliveryCandidateSettlementIntent(_Record):
    """FinishGate 前的精确 候选项权威状态及其有类型结果。

    ``task_state_version`` 有意记录 reviewer 所观察到的活动 Task 版本。原子结算
    分别记录 completed 版本与 reopened/waiting 版本；它绝不会将此 Prompt 重新
    标记为 FinishGate 后的 whole-Task review。
    """

    schema_version: Literal["task-delivery-candidate-settlement-intent-v1"] = (
        "task-delivery-candidate-settlement-intent-v1"
    )
    session_id: str = Field(pattern=_ID_PATTERN)
    invocation_turn_id: str = Field(pattern=_ID_PATTERN)
    subject: TaskNodeSubject
    task_state_version: int = Field(ge=1)
    work_run_id: str = Field(pattern=_ID_PATTERN)
    submitted_attempt_id: str = Field(pattern=_ID_PATTERN)
    verification_request_id: str = Field(pattern=_ID_PATTERN)
    verification_request_revision: int = Field(ge=1)
    output_revision: int = Field(ge=1)
    output_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_delivery_id: str = Field(pattern=_ID_PATTERN)
    review_request: TaskDeliveryValidationRequest
    authority_sha256: str = Field(pattern=_SHA256_PATTERN)
    result: TaskDeliveryValidationResult

    @model_validator(mode="after")
    def _validate_intent(self) -> "TaskDeliveryCandidateSettlementIntent":
        if self.subject.node_id != self.subject.task_id:
            raise ValueError("candidate settlement requires the canonical root")
        prompt = self.review_request.prompt
        if (
            self.session_id != prompt.session_id
            or self.invocation_turn_id != self.review_request.invocation_turn_id
            or self.subject.task_id != prompt.task_id
            or self.subject.graph_revision != prompt.graph_revision
            or self.task_state_version != prompt.task_state_version
            or self.candidate_delivery_id != prompt.root_delivery_id
            or self.output_sha256
            != hashlib.sha256(prompt.root_output_body.encode("utf-8")).hexdigest()
        ):
            raise ValueError("candidate settlement intent crossed frozen authority")
        validate_task_delivery_validation_result(
            request=self.review_request,
            result=self.result,
        )
        if self.result.disposition not in {
            TaskDeliveryValidationDisposition.PASS,
            TaskDeliveryValidationDisposition.REPLAN_TASK_GRAPH,
            TaskDeliveryValidationDisposition.BLOCKED,
        }:
            raise ValueError("candidate settlement cannot own execution retry")
        authority_payload = _candidate_authority_payload(self)
        authority_payload.pop("authority_sha256")
        if self.authority_sha256 != _sha256_value(authority_payload):
            raise ValueError("candidate settlement authority hash is invalid")
        return self


class TaskDeliveryCandidateSettlement(_Record):
    """跨越 FinishGate 前后版本的持久化路由结算。"""

    schema_version: Literal["task-delivery-candidate-settlement-v1"] = (
        "task-delivery-candidate-settlement-v1"
    )
    settlement_id: str = Field(pattern=_ID_PATTERN)
    apply_id: str = Field(pattern=_ID_PATTERN)
    node_verification_commit_apply_id: str = Field(pattern=_ID_PATTERN)
    session_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    graph_revision: int = Field(ge=1)
    candidate_task_state_version: int = Field(ge=1)
    completed_task_state_version: int = Field(ge=1)
    settled_task_state_version: int = Field(ge=1)
    root_delivery_id: str = Field(pattern=_ID_PATTERN)
    node_verification_request_id: str = Field(pattern=_ID_PATTERN)
    node_verification_request_revision: int = Field(ge=1)
    candidate_authority_sha256: str = Field(pattern=_SHA256_PATTERN)
    verification_request_id: str = Field(pattern=_ID_PATTERN)
    request_binding_sha256: str = Field(pattern=_SHA256_PATTERN)
    verification_result_id: str = Field(pattern=_ID_PATTERN)
    result_sha256: str = Field(pattern=_SHA256_PATTERN)
    disposition: TaskDeliveryValidationDisposition
    created_turn_id: str = Field(pattern=_ID_PATTERN)
    settlement_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_settlement(self) -> "TaskDeliveryCandidateSettlement":
        if self.completed_task_state_version != self.candidate_task_state_version + 1:
            raise ValueError("candidate FinishGate version must be exact +1")
        expected_settled_version = (
            self.completed_task_state_version
            if self.disposition is TaskDeliveryValidationDisposition.PASS
            else self.completed_task_state_version + 1
        )
        if self.settled_task_state_version != expected_settled_version:
            raise ValueError("candidate settlement Task version is inconsistent")
        if self.disposition not in {
            TaskDeliveryValidationDisposition.PASS,
            TaskDeliveryValidationDisposition.REPLAN_TASK_GRAPH,
            TaskDeliveryValidationDisposition.BLOCKED,
        }:
            raise ValueError("candidate settlement cannot own execution retry")
        expected = _sha256_value(
            self.model_dump(mode="json", exclude={"settlement_sha256"})
        )
        if self.settlement_sha256 != expected:
            raise ValueError("candidate settlement hash is invalid")
        return self

    @classmethod
    def create(cls, **values: object) -> "TaskDeliveryCandidateSettlement":
        values = dict(values)
        values["settlement_sha256"] = "0" * 64
        provisional = cls.model_construct(**values)
        values["settlement_sha256"] = _sha256_value(
            provisional.model_dump(mode="json", exclude={"settlement_sha256"})
        )
        return cls.model_validate(values)


class TaskDeliveryCandidateSettlementMutationResult(_Record):
    status: Literal["applied", "replayed"]
    settlement: TaskDeliveryCandidateSettlement
    trigger: TaskGraphRevisionTrigger | None = None


class StoredTaskDeliveryCandidateSettlement(_Record):
    intent: TaskDeliveryCandidateSettlementIntent
    settlement: TaskDeliveryCandidateSettlement
    trigger: TaskGraphRevisionTrigger | None = None


class ConsumeTaskGraphRevisionTriggerCommand(_Record):
    schema_version: Literal["consume-task-graph-revision-trigger-command-v1"] = (
        "consume-task-graph-revision-trigger-command-v1"
    )
    apply_id: str = Field(pattern=_ID_PATTERN)
    session_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    trigger_id: str = Field(pattern=_ID_PATTERN)
    expected_trigger_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_graph_commit_apply_id: str = Field(pattern=_ID_PATTERN)
    consumed_turn_id: str = Field(pattern=_ID_PATTERN)


class TaskGraphRevisionTriggerApplicationMutationResult(_Record):
    status: Literal["applied", "replayed"]
    application: TaskGraphRevisionTriggerApplication


def project_task_delivery_validation_child_deliveries(
    deps: StoreDeps,
    *,
    session_id: str,
    task_id: str,
    graph_revision: int,
) -> TaskDeliveryValidationChildDeliveryProjection:
    """在一个数据库快照中冻结所有当前已验证的非根 Delivery。"""

    _require_id("session_id", session_id)
    _require_id("task_id", task_id)
    if (
        isinstance(graph_revision, bool)
        or not isinstance(graph_revision, int)
        or graph_revision < 1
    ):
        raise ValueError("graph_revision must be a positive integer")
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        task = conn.execute(
            "SELECT current_graph_revision, current_status FROM insession_tasks "
            "WHERE session_id=? AND insession_task_id=?",
            (session_id, task_id),
        ).fetchone()
        if (
            task is None
            or int(task["current_graph_revision"] or 0) != graph_revision
            or str(task["current_status"]) not in {"active", "completed"}
        ):
            raise TaskDeliveryValidationStaleAuthority(
                "child Delivery projection requires the exact current TaskGraph"
            )
        rows = _load_current_validation_node_rows(
            conn,
            task_id=task_id,
            graph_revision=graph_revision,
        )
        roots = tuple(row for row in rows if str(row["node_kind"]) == "root")
        if (
            len(roots) != 1
            or str(roots[0]["insession_task_node_id"]) != task_id
            or str(roots[0]["status"]) != str(task["current_status"])
            or any(
                str(row["status"]) != "completed"
                for row in rows
                if str(row["node_kind"]) != "root"
            )
        ):
            raise TaskDeliveryValidationStaleAuthority(
                "child Delivery projection requires every non-root node completed"
            )
        projection = _project_child_delivery_projection(
            conn,
            session_id=session_id,
            task_id=task_id,
            graph_revision=graph_revision,
            rows=rows,
        )
        conn.commit()
        return projection
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def settle_task_delivery_candidate_route_in_transaction(
    conn: sqlite3.Connection,
    *,
    intent: TaskDeliveryCandidateSettlementIntent,
    node_verification_commit_apply_id: str,
    now: str,
) -> TaskDeliveryCandidateSettlementMutationResult:
    """在同一事务的 FinishGate 后以原子方式结算 根路由。

    调用方已在此事务中插入根 Delivery 并完成精确 Task revision。REPLAN 与有类型的
    missing-information BLOCKED 都会追加普通的不可变
    ``TaskGraphRevisionTrigger`` 并重新打开 N。后者只能由包含真正持久化
    ``user_gate`` 的正数 base graph 消费。两者都会使本地根 Delivery 保持冻结且
    未发布。
    """

    if not conn.in_transaction:
        raise TaskDeliveryValidationPersistenceError(
            "candidate route settlement requires an active transaction"
        )
    intent = _validate_model(
        TaskDeliveryCandidateSettlementIntent,
        intent,
        "Task delivery candidate settlement intent",
    )
    _require_id("node_verification_commit_apply_id", node_verification_commit_apply_id)
    identity = _candidate_settlement_identity(intent)
    settlement_id = f"tdcv2_settlement_{identity}"
    apply_id = f"tdcv2_apply_{identity}"
    _require_candidate_settlement_route(intent.result)
    trigger_id = (
        None
        if intent.result.disposition is TaskDeliveryValidationDisposition.PASS
        else f"tdcv2_trigger_{identity}"
    )

    replay_rows = conn.execute(
        "SELECT * FROM insession_task_delivery_validation_settlements "
        "WHERE apply_id=? OR settlement_id=?",
        (apply_id, settlement_id),
    ).fetchall()
    if replay_rows:
        if len(replay_rows) != 1:
            raise TaskDeliveryValidationIdentityCollision(
                "candidate settlement identities point to multiple rows"
            )
        stored = _stored_candidate_from_settlement_row(conn, replay_rows[0])
        if (
            stored.intent != intent
            or stored.settlement.node_verification_commit_apply_id
            != node_verification_commit_apply_id
        ):
            raise TaskDeliveryValidationIdentityCollision(
                "candidate settlement identity crossed immutable authority"
            )
        return TaskDeliveryCandidateSettlementMutationResult(
            status="replayed",
            settlement=stored.settlement,
            trigger=stored.trigger,
        )

    _require_running_turn(
        conn,
        session_id=intent.session_id,
        turn_id=intent.invocation_turn_id,
    )
    task = conn.execute(
        "SELECT current_graph_revision, current_status, state_version "
        "FROM insession_tasks WHERE session_id=? AND insession_task_id=?",
        (intent.session_id, intent.subject.task_id),
    ).fetchone()
    expected_completed_version = intent.task_state_version + 1
    if (
        task is None
        or int(task["current_graph_revision"] or 0)
        != intent.subject.graph_revision
        or str(task["current_status"]) != "completed"
        or int(task["state_version"]) != expected_completed_version
    ):
        raise TaskDeliveryValidationStaleAuthority(
            "candidate settlement did not follow its exact FinishGate"
        )
    _require_candidate_node_verification(
        conn,
        intent=intent,
    )
    current_prompt = _project_prompt(
        conn,
        session_id=intent.session_id,
        task_id=intent.subject.task_id,
    )
    _require_candidate_prompt_matches_completed_delivery(
        intent=intent,
        current=current_prompt,
    )
    _require_exact_candidate_model_result(conn, intent=intent)

    request = intent.review_request
    request_rows = conn.execute(
        "SELECT * FROM insession_task_delivery_validation_requests "
        "WHERE verification_request_id=? OR logical_call_id=? OR "
        "root_delivery_id=?",
        (
            request.verification_request_id,
            request.logical_call_id,
            request.prompt.root_delivery_id,
        ),
    ).fetchall()
    if request_rows:
        raise TaskDeliveryValidationIdentityCollision(
            "candidate request identity is already settled elsewhere"
        )
    conn.execute(
        "INSERT INTO insession_task_delivery_validation_requests "
        "(verification_request_id, logical_call_id, session_id, "
        "insession_task_id, graph_revision, completed_task_state_version, "
        "root_delivery_id, invocation_turn_id, prompt_payload_sha256, "
        "request_binding_sha256, request_json, created_at) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            request.verification_request_id,
            request.logical_call_id,
            intent.session_id,
            intent.subject.task_id,
            intent.subject.graph_revision,
            # 对 候选记录而言，此列是冻结请求携带并如实记录的 FinishGate 前
            # Task 版本。
            intent.task_state_version,
            intent.candidate_delivery_id,
            intent.invocation_turn_id,
            request.prompt.payload_sha256,
            request.binding_sha256,
            _model_json(request),
            now,
        ),
    )

    mapped_disposition = _candidate_sql_disposition(intent.result.disposition)
    conn.execute(
        "INSERT INTO insession_task_delivery_validation_results "
        "(verification_result_id, verification_request_id, logical_call_id, "
        "session_id, insession_task_id, disposition, result_sha256, "
        "result_json, created_turn_id, created_at) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            intent.result.verification_result_id,
            intent.result.verification_request_id,
            intent.result.logical_call_id,
            intent.session_id,
            intent.subject.task_id,
            mapped_disposition,
            intent.result.result_sha256,
            _model_json(intent.result),
            intent.invocation_turn_id,
            now,
        ),
    )

    settled_task_version = (
        expected_completed_version
        if intent.result.disposition is TaskDeliveryValidationDisposition.PASS
        else expected_completed_version + 1
    )
    settlement = TaskDeliveryCandidateSettlement.create(
        settlement_id=settlement_id,
        apply_id=apply_id,
        node_verification_commit_apply_id=node_verification_commit_apply_id,
        session_id=intent.session_id,
        task_id=intent.subject.task_id,
        graph_revision=intent.subject.graph_revision,
        candidate_task_state_version=intent.task_state_version,
        completed_task_state_version=expected_completed_version,
        settled_task_state_version=settled_task_version,
        root_delivery_id=intent.candidate_delivery_id,
        node_verification_request_id=intent.verification_request_id,
        node_verification_request_revision=intent.verification_request_revision,
        candidate_authority_sha256=intent.authority_sha256,
        verification_request_id=request.verification_request_id,
        request_binding_sha256=request.binding_sha256,
        verification_result_id=intent.result.verification_result_id,
        result_sha256=intent.result.result_sha256,
        disposition=intent.result.disposition,
        created_turn_id=intent.invocation_turn_id,
    )
    command_sha256 = _sha256_value(
        {
            "contract": "task-delivery-candidate-settlement-command-v1",
            "intent": intent.model_dump(mode="json"),
            "node_verification_commit_apply_id": (
                node_verification_commit_apply_id
            ),
        }
    )
    conn.execute(
        "INSERT INTO insession_task_delivery_validation_settlements "
        "(settlement_id, apply_id, verification_request_id, "
        "verification_result_id, session_id, insession_task_id, "
        "graph_revision, completed_task_state_version, root_delivery_id, "
        "disposition, command_sha256, settlement_sha256, settlement_json, "
        "created_turn_id, created_at) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            settlement.settlement_id,
            settlement.apply_id,
            settlement.verification_request_id,
            settlement.verification_result_id,
            settlement.session_id,
            settlement.task_id,
            settlement.graph_revision,
            settlement.completed_task_state_version,
            settlement.root_delivery_id,
            mapped_disposition,
            command_sha256,
            settlement.settlement_sha256,
            _model_json(settlement),
            settlement.created_turn_id,
            now,
        ),
    )

    if intent.result.disposition is not TaskDeliveryValidationDisposition.PASS:
        if conn.execute(
            "UPDATE insession_tasks SET current_status=?, state_version=?, "
            "updated_at=? WHERE session_id=? AND insession_task_id=? "
            "AND current_graph_revision=? AND current_status='completed' "
            "AND state_version=?",
            (
                "active",
                settled_task_version,
                now,
                intent.session_id,
                intent.subject.task_id,
                intent.subject.graph_revision,
                expected_completed_version,
            ),
        ).rowcount != 1:
            raise TaskDeliveryValidationStaleAuthority(
                "Task changed during candidate route settlement"
            )

    trigger: TaskGraphRevisionTrigger | None = None
    if intent.result.disposition is not TaskDeliveryValidationDisposition.PASS:
        assert trigger_id is not None
        revision_objective = (
            intent.result.task_graph_revision_objective
            if intent.result.disposition
            is TaskDeliveryValidationDisposition.REPLAN_TASK_GRAPH
            else _candidate_blocked_revision_objective(intent.result)
        )
        assert revision_objective is not None
        trigger = TaskGraphRevisionTrigger.create(
            trigger_id=trigger_id,
            create_apply_id=apply_id,
            session_id=intent.session_id,
            task_id=intent.subject.task_id,
            base_graph_revision=intent.subject.graph_revision,
            target_graph_revision=intent.subject.graph_revision + 1,
            root_delivery_id=intent.candidate_delivery_id,
            verification_request_id=request.verification_request_id,
            request_binding_sha256=request.binding_sha256,
            verification_result_id=intent.result.verification_result_id,
            result_sha256=intent.result.result_sha256,
            settlement_id=settlement.settlement_id,
            settlement_sha256=settlement.settlement_sha256,
            revision_objective=revision_objective,
            gap_diagnosis=_candidate_gap_diagnosis(intent.result),
            reopened_task_state_version=settled_task_version,
            created_turn_id=intent.invocation_turn_id,
        )
        conn.execute(
            "INSERT INTO insession_task_graph_revision_triggers "
            "(trigger_id, create_apply_id, settlement_id, session_id, "
            "insession_task_id, base_graph_revision, target_graph_revision, "
            "root_delivery_id, reopened_task_state_version, trigger_sha256, "
            "trigger_json, created_turn_id, created_at) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                trigger.trigger_id,
                trigger.create_apply_id,
                trigger.settlement_id,
                trigger.session_id,
                trigger.task_id,
                trigger.base_graph_revision,
                trigger.target_graph_revision,
                trigger.root_delivery_id,
                trigger.reopened_task_state_version,
                trigger.trigger_sha256,
                _model_json(trigger),
                trigger.created_turn_id,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO insession_active_task_graph_revision_triggers "
            "(trigger_id, session_id, insession_task_id, "
            "base_graph_revision, target_graph_revision, activated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                trigger.trigger_id,
                trigger.session_id,
                trigger.task_id,
                trigger.base_graph_revision,
                trigger.target_graph_revision,
                now,
            ),
        )
    return TaskDeliveryCandidateSettlementMutationResult(
        status="applied",
        settlement=settlement,
        trigger=trigger,
    )


def replay_task_delivery_candidate_settlement_in_transaction(
    conn: sqlite3.Connection,
    *,
    intent: TaskDeliveryCandidateSettlementIntent,
    node_verification_commit_apply_id: str,
    expected_root_delivery_id: str,
) -> TaskDeliveryCandidateSettlementMutationResult:
    """认证精确 node-commit 重放的候选项一侧。"""

    if not conn.in_transaction:
        raise TaskDeliveryValidationPersistenceError(
            "candidate settlement replay requires an active transaction"
        )
    intent = _validate_model(
        TaskDeliveryCandidateSettlementIntent,
        intent,
        "Task delivery candidate settlement intent",
    )
    identity = _candidate_settlement_identity(intent)
    rows = conn.execute(
        "SELECT * FROM insession_task_delivery_validation_settlements "
        "WHERE settlement_id=? OR apply_id=?",
        (f"tdcv2_settlement_{identity}", f"tdcv2_apply_{identity}"),
    ).fetchall()
    if len(rows) != 1:
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "node verification replay lost its candidate settlement"
        )
    stored = _stored_candidate_from_settlement_row(conn, rows[0])
    if (
        stored.intent != intent
        or stored.settlement.node_verification_commit_apply_id
        != node_verification_commit_apply_id
        or stored.settlement.root_delivery_id != expected_root_delivery_id
    ):
        raise TaskDeliveryValidationIdentityCollision(
            "candidate replay crossed immutable node settlement authority"
        )
    return TaskDeliveryCandidateSettlementMutationResult(
        status="replayed",
        settlement=stored.settlement,
        trigger=stored.trigger,
    )


def get_task_delivery_candidate_settlement(
    deps: StoreDeps,
    *,
    session_id: str,
    task_id: str,
    graph_revision: int,
) -> StoredTaskDeliveryCandidateSettlement | None:
    """加载精确 graph revision N 唯一已结算的 根路由。"""

    _require_id("session_id", session_id)
    _require_id("task_id", task_id)
    if isinstance(graph_revision, bool) or graph_revision < 1:
        raise ValueError("graph_revision must be positive")
    deps.init_db()
    with deps.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM insession_task_delivery_validation_settlements "
            "WHERE session_id=? AND insession_task_id=? AND graph_revision=?",
            (session_id, task_id, graph_revision),
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise TaskDeliveryValidationStoredAuthorityCorrupt(
                "TaskGraph revision owns multiple candidate settlements"
            )
        return _stored_candidate_from_settlement_row(conn, rows[0])


def require_pass_settlement_for_publication_in_transaction(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
    graph_revision: int,
    task_state_version: int,
    root_delivery_id: str,
) -> None:
    """要求当前 Auxiliary 与整 Task PASS 发布权威状态。"""

    for name, value in (
        ("session_id", session_id),
        ("turn_id", turn_id),
        ("task_id", task_id),
        ("root_delivery_id", root_delivery_id),
    ):
        _require_id(name, value)
    if not conn.in_transaction:
        raise TaskDeliveryValidationPersistenceError(
            "publication validation requires an active transaction"
        )
    if isinstance(graph_revision, bool) or graph_revision < 1:
        raise ValueError("graph_revision must be positive")
    if isinstance(task_state_version, bool) or task_state_version < 1:
        raise ValueError("task_state_version must be positive")

    commit_rows = conn.execute(
        "SELECT apply_id FROM "
        "insession_auxiliary_v2_task_graph_commit_receipts "
        "WHERE session_id=? AND insession_task_id=? "
        "AND committed_task_graph_revision=?",
        (session_id, task_id, graph_revision),
    ).fetchall()
    if not commit_rows:
        raise TaskDeliveryValidationStaleAuthority(
            "root Delivery lacks the current Auxiliary TaskGraph commit receipt"
        )
    if len(commit_rows) != 1:
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "TaskGraph revision owns multiple commit receipts"
        )

    settlement_rows = conn.execute(
        "SELECT * FROM insession_task_delivery_validation_settlements "
        "WHERE session_id=? AND insession_task_id=? AND graph_revision=? "
        "AND root_delivery_id=?",
        (session_id, task_id, graph_revision, root_delivery_id),
    ).fetchall()
    if len(settlement_rows) != 1:
        raise TaskDeliveryValidationStaleAuthority(
            "root Delivery lacks one exact whole-Task settlement"
        )
    candidate = _stored_candidate_from_settlement_row(
        conn,
        settlement_rows[0],
    )
    settlement = candidate.settlement
    valid_pass = (
        candidate.intent.result.disposition
        is TaskDeliveryValidationDisposition.PASS
        and settlement.disposition
        is TaskDeliveryValidationDisposition.PASS
        and candidate.trigger is None
        and settlement.session_id == session_id
        and settlement.task_id == task_id
        and settlement.graph_revision == graph_revision
        and settlement.completed_task_state_version == task_state_version
        and settlement.settled_task_state_version == task_state_version
        and settlement.root_delivery_id == root_delivery_id
        and settlement.created_turn_id == turn_id
    )
    if not valid_pass:
        raise TaskDeliveryValidationStaleAuthority(
            "root Delivery is not backed by the exact current PASS"
        )
    active = conn.execute(
        "SELECT trigger_id FROM insession_active_task_graph_revision_triggers "
        "WHERE session_id=? AND insession_task_id=?",
        (session_id, task_id),
    ).fetchall()
    if active:
        raise TaskDeliveryValidationStaleAuthority(
            "Task still owns an active revision trigger at publication"
        )


def get_active_task_graph_revision_trigger(
    deps: StoreDeps,
    *,
    session_id: str,
    task_id: str,
) -> TaskGraphRevisionTrigger | None:
    _require_id("session_id", session_id)
    _require_id("task_id", task_id)
    deps.init_db()
    with deps.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM insession_active_task_graph_revision_triggers "
            "WHERE session_id=? AND insession_task_id=?",
            (session_id, task_id),
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise TaskDeliveryValidationStoredAuthorityCorrupt(
                "Task owns multiple active graph revision triggers"
            )
        trigger = _load_authenticated_trigger_authority(
            conn,
            str(rows[0]["trigger_id"]),
        )
        row = rows[0]
        if (
            trigger.session_id != session_id
            or trigger.task_id != task_id
            or int(row["base_graph_revision"]) != trigger.base_graph_revision
            or int(row["target_graph_revision"]) != trigger.target_graph_revision
        ):
            raise TaskDeliveryValidationStoredAuthorityCorrupt(
                "active TaskGraph trigger pointer crossed immutable authority"
            )
        task = conn.execute(
            "SELECT current_graph_revision, current_status, state_version "
            "FROM insession_tasks WHERE session_id=? AND insession_task_id=?",
            (session_id, task_id),
        ).fetchone()
        task_status = None if task is None else str(task["current_status"])
        waiting_on_trigger_gate = bool(
            task is not None
            and task_status == "awaiting_user"
            and _has_exact_pending_trigger_user_gate(
                conn,
                trigger=trigger,
            )
        )
        if (
            task is None
            or int(task["current_graph_revision"] or 0)
            != trigger.base_graph_revision
            or (
                task_status != "active"
                and not waiting_on_trigger_gate
            )
            or int(task["state_version"]) < trigger.reopened_task_state_version
        ):
            raise TaskDeliveryValidationStaleAuthority(
                "active TaskGraph trigger no longer owns its reopened base"
            )
        return trigger


def _has_exact_pending_trigger_user_gate(
    conn: sqlite3.Connection,
    *,
    trigger: TaskGraphRevisionTrigger,
) -> bool:
    """认证由此 trigger goal 持有的唯一 user-gate 游标。"""

    rows = conn.execute(
        "SELECT run.work_run_id FROM insession_work_runs AS run "
        "JOIN insession_execution_subjects AS registry "
        "ON registry.execution_subject_id=run.execution_subject_id "
        "AND registry.session_id=run.session_id "
        "AND registry.insession_task_id=run.insession_task_id "
        "AND registry.subject_kind='auxiliary_node' "
        "AND registry.subject_contract_version='auxiliary_node_v2' "
        "JOIN insession_auxiliary_v2_node_execution_subject_bindings AS binding "
        "ON binding.binding_id=registry.auxiliary_v2_binding_id "
        "AND binding.session_id=run.session_id "
        "AND binding.insession_task_id=run.insession_task_id "
        "AND binding.auxiliary_graph_id=run.auxiliary_graph_id "
        "AND binding.auxiliary_graph_revision=run.auxiliary_graph_revision "
        "AND binding.auxiliary_node_id=run.auxiliary_node_id "
        "AND binding.node_revision=run.node_revision "
        "JOIN insession_auxiliary_graph_v2_containers AS control "
        "ON control.session_id=run.session_id "
        "AND control.insession_task_id=run.insession_task_id "
        "AND control.auxiliary_graph_id=run.auxiliary_graph_id "
        "JOIN insession_auxiliary_graph_goals AS goal "
        "ON goal.goal_id=binding.goal_id "
        "AND goal.session_id=run.session_id "
        "AND goal.insession_task_id=run.insession_task_id "
        "AND goal.auxiliary_graph_id=run.auxiliary_graph_id "
        "JOIN insession_auxiliary_graph_revision_states_v2 AS revision_state "
        "ON revision_state.auxiliary_graph_id=run.auxiliary_graph_id "
        "AND revision_state.auxiliary_graph_revision="
        "run.auxiliary_graph_revision "
        "JOIN insession_auxiliary_node_states_v2 AS node_state "
        "ON node_state.auxiliary_graph_id=run.auxiliary_graph_id "
        "AND node_state.auxiliary_graph_revision=run.auxiliary_graph_revision "
        "AND node_state.auxiliary_node_id=run.auxiliary_node_id "
        "AND node_state.node_revision=run.node_revision "
        "JOIN insession_auxiliary_node_definitions_v2 AS definition "
        "ON definition.auxiliary_graph_id=run.auxiliary_graph_id "
        "AND definition.auxiliary_node_id=run.auxiliary_node_id "
        "AND definition.node_revision=run.node_revision "
        "WHERE run.session_id=? AND run.insession_task_id=? "
        "AND run.status='waiting_user' AND run.reason='needs_input' "
        "AND run.auxiliary_graph_revision="
        "control.current_auxiliary_graph_revision "
        "AND binding.goal_id=control.current_goal_id "
        "AND goal.base_task_graph_revision=? "
        "AND goal.target_task_graph_revision=? "
        "AND goal.status='waiting_user' "
        "AND revision_state.status='waiting_user' "
        "AND node_state.status='waiting_user' "
        "AND definition.executor_kind='user_gate'",
        (
            trigger.session_id,
            trigger.task_id,
            trigger.base_graph_revision,
            trigger.target_graph_revision,
        ),
    ).fetchall()
    return len(rows) == 1


def consume_task_graph_revision_trigger(
    deps: StoreDeps,
    *,
    command: ConsumeTaskGraphRevisionTriggerCommand,
) -> TaskGraphRevisionTriggerApplicationMutationResult:
    """将活动 trigger 绑定到一个精确且已提交的 TaskGraph N+1 receipt。"""

    command = _validate_model(
        ConsumeTaskGraphRevisionTriggerCommand,
        command,
        "TaskGraph revision trigger consume command",
    )
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        replay = conn.execute(
            "SELECT * FROM insession_task_graph_revision_trigger_applications "
            "WHERE apply_id=? OR trigger_id=? OR task_graph_commit_apply_id=?",
            (
                command.apply_id,
                command.trigger_id,
                command.task_graph_commit_apply_id,
            ),
        ).fetchall()
        if replay:
            if len(replay) != 1:
                raise TaskDeliveryValidationIdentityCollision(
                    "trigger application identities point to multiple receipts"
                )
            application = _application_from_row(replay[0])
            if (
                application.apply_id != command.apply_id
                or application.trigger_id != command.trigger_id
                or application.trigger_sha256
                != command.expected_trigger_sha256
                or application.task_graph_commit_apply_id
                != command.task_graph_commit_apply_id
                or application.consumed_turn_id != command.consumed_turn_id
            ):
                raise TaskDeliveryValidationIdentityCollision(
                    "trigger application identity crossed immutable content"
                )
            conn.commit()
            return TaskGraphRevisionTriggerApplicationMutationResult(
                status="replayed",
                application=application,
            )
        _require_running_turn(
            conn,
            session_id=command.session_id,
            turn_id=command.consumed_turn_id,
        )
        active = conn.execute(
            "SELECT * FROM insession_active_task_graph_revision_triggers "
            "WHERE trigger_id=?",
            (command.trigger_id,),
        ).fetchone()
        if active is None:
            raise TaskDeliveryValidationStaleAuthority(
                "TaskGraph revision trigger is not active"
            )
        trigger = _load_authenticated_trigger_authority(
            conn,
            command.trigger_id,
        )
        if (
            trigger.session_id != command.session_id
            or trigger.task_id != command.task_id
            or trigger.trigger_sha256 != command.expected_trigger_sha256
        ):
            raise TaskDeliveryValidationStaleAuthority(
                "TaskGraph revision trigger consume CAS is stale"
            )
        commit = conn.execute(
            "SELECT session_id, insession_task_id, base_task_graph_revision, "
            "committed_task_graph_revision FROM "
            "insession_auxiliary_v2_task_graph_commit_receipts WHERE apply_id=?",
            (command.task_graph_commit_apply_id,),
        ).fetchone()
        task = conn.execute(
            "SELECT current_graph_revision, current_status FROM insession_tasks "
            "WHERE session_id=? AND insession_task_id=?",
            (command.session_id, command.task_id),
        ).fetchone()
        if (
            commit is None
            or str(commit["session_id"]) != trigger.session_id
            or str(commit["insession_task_id"]) != trigger.task_id
            or int(commit["base_task_graph_revision"] or 0)
            != trigger.base_graph_revision
            or int(commit["committed_task_graph_revision"])
            != trigger.target_graph_revision
            or task is None
            or int(task["current_graph_revision"] or 0)
            != trigger.target_graph_revision
            or str(task["current_status"]) != "active"
        ):
            raise TaskDeliveryValidationStaleAuthority(
                "trigger is not followed by its exact committed TaskGraph N+1"
            )
        application = TaskGraphRevisionTriggerApplication.create(
            apply_id=command.apply_id,
            trigger_id=trigger.trigger_id,
            trigger_sha256=trigger.trigger_sha256,
            session_id=trigger.session_id,
            task_id=trigger.task_id,
            base_graph_revision=trigger.base_graph_revision,
            committed_graph_revision=trigger.target_graph_revision,
            task_graph_commit_apply_id=command.task_graph_commit_apply_id,
            consumed_turn_id=command.consumed_turn_id,
        )
        now = deps.now()
        conn.execute(
            "INSERT INTO insession_task_graph_revision_trigger_applications "
            "(apply_id, trigger_id, trigger_sha256, session_id, "
            "insession_task_id, base_graph_revision, committed_graph_revision, "
            "task_graph_commit_apply_id, consumed_turn_id, receipt_sha256, "
            "receipt_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                application.apply_id,
                application.trigger_id,
                application.trigger_sha256,
                application.session_id,
                application.task_id,
                application.base_graph_revision,
                application.committed_graph_revision,
                application.task_graph_commit_apply_id,
                application.consumed_turn_id,
                application.receipt_sha256,
                _model_json(application),
                now,
            ),
        )
        if conn.execute(
            "DELETE FROM insession_active_task_graph_revision_triggers "
            "WHERE trigger_id=?",
            (trigger.trigger_id,),
        ).rowcount != 1:
            raise TaskDeliveryValidationStaleAuthority(
                "active trigger changed during consumption"
            )
        conn.commit()
        return TaskGraphRevisionTriggerApplicationMutationResult(
            status="applied",
            application=application,
        )
    except sqlite3.IntegrityError as exc:
        if conn.in_transaction:
            conn.rollback()
        raise TaskDeliveryValidationPersistenceError(
            "trigger application violated v72 durable authority"
        ) from exc
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def consume_task_graph_revision_trigger_in_transaction(
    conn: sqlite3.Connection,
    *,
    application_apply_id: str,
    session_id: str,
    task_id: str,
    trigger_id: str,
    expected_trigger_sha256: str,
    task_graph_commit_apply_id: str,
    consumed_turn_id: str,
    now: str,
) -> TaskGraphRevisionTriggerApplication:
    """在其 TaskGraph N+1 提交事务内以原子方式消费 trigger。"""

    if not conn.in_transaction:
        raise TaskDeliveryValidationPersistenceError(
            "atomic trigger consumption requires an active transaction"
        )
    active = conn.execute(
        "SELECT * FROM insession_active_task_graph_revision_triggers "
        "WHERE trigger_id=?",
        (trigger_id,),
    ).fetchone()
    if active is None:
        raise TaskDeliveryValidationStaleAuthority(
            "TaskGraph revision trigger is not active at N+1 commit"
        )
    trigger = _load_authenticated_trigger_authority(conn, trigger_id)
    if (
        trigger.session_id != session_id
        or trigger.task_id != task_id
        or trigger.trigger_sha256 != expected_trigger_sha256
        or str(active["session_id"]) != session_id
        or str(active["insession_task_id"]) != task_id
        or int(active["base_graph_revision"]) != trigger.base_graph_revision
        or int(active["target_graph_revision"]) != trigger.target_graph_revision
    ):
        raise TaskDeliveryValidationStaleAuthority(
            "TaskGraph revision trigger CAS changed before N+1 commit"
        )
    commit = conn.execute(
        "SELECT session_id, insession_task_id, base_task_graph_revision, "
        "committed_task_graph_revision FROM "
        "insession_auxiliary_v2_task_graph_commit_receipts WHERE apply_id=?",
        (task_graph_commit_apply_id,),
    ).fetchone()
    task = conn.execute(
        "SELECT current_graph_revision, current_status FROM insession_tasks "
        "WHERE session_id=? AND insession_task_id=?",
        (session_id, task_id),
    ).fetchone()
    if (
        commit is None
        or str(commit["session_id"]) != session_id
        or str(commit["insession_task_id"]) != task_id
        or int(commit["base_task_graph_revision"] or 0)
        != trigger.base_graph_revision
        or int(commit["committed_task_graph_revision"])
        != trigger.target_graph_revision
        or task is None
        or int(task["current_graph_revision"] or 0)
        != trigger.target_graph_revision
        or str(task["current_status"]) != "active"
    ):
        raise TaskDeliveryValidationStaleAuthority(
            "atomic trigger consumption is not bound to exact TaskGraph N+1"
        )
    application = TaskGraphRevisionTriggerApplication.create(
        apply_id=application_apply_id,
        trigger_id=trigger.trigger_id,
        trigger_sha256=trigger.trigger_sha256,
        session_id=trigger.session_id,
        task_id=trigger.task_id,
        base_graph_revision=trigger.base_graph_revision,
        committed_graph_revision=trigger.target_graph_revision,
        task_graph_commit_apply_id=task_graph_commit_apply_id,
        consumed_turn_id=consumed_turn_id,
    )
    conn.execute(
        "INSERT INTO insession_task_graph_revision_trigger_applications "
        "(apply_id, trigger_id, trigger_sha256, session_id, "
        "insession_task_id, base_graph_revision, committed_graph_revision, "
        "task_graph_commit_apply_id, consumed_turn_id, receipt_sha256, "
        "receipt_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            application.apply_id,
            application.trigger_id,
            application.trigger_sha256,
            application.session_id,
            application.task_id,
            application.base_graph_revision,
            application.committed_graph_revision,
            application.task_graph_commit_apply_id,
            application.consumed_turn_id,
            application.receipt_sha256,
            _model_json(application),
            now,
        ),
    )
    if conn.execute(
        "DELETE FROM insession_active_task_graph_revision_triggers "
        "WHERE trigger_id=?",
        (trigger.trigger_id,),
    ).rowcount != 1:
        raise TaskDeliveryValidationStaleAuthority(
            "active trigger changed during atomic N+1 commit"
        )
    return application


def _project_prompt(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    task_id: str,
) -> TaskDeliveryValidationPrompt:
    task = conn.execute(
        "SELECT task.current_graph_revision, task.current_status, "
        "task.state_version, task.root_title, task.root_objective, "
        "revision.source_anchors_json FROM insession_tasks AS task "
        "JOIN insession_task_graph_revisions AS revision "
        "ON revision.insession_task_id=task.insession_task_id "
        "AND revision.graph_revision=task.current_graph_revision "
        "WHERE task.session_id=? AND task.insession_task_id=?",
        (session_id, task_id),
    ).fetchone()
    if (
        task is None
        or task["current_graph_revision"] is None
        or str(task["current_status"]) != "completed"
    ):
        raise TaskDeliveryValidationStaleAuthority(
            "whole-Task validation requires one completed current graph"
        )
    graph_revision = int(task["current_graph_revision"])
    rows = _load_current_validation_node_rows(
        conn,
        task_id=task_id,
        graph_revision=graph_revision,
    )
    if not rows or any(str(row["status"]) != "completed" for row in rows):
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "completed Task lost its complete current node set"
        )
    roots = tuple(row for row in rows if str(row["node_kind"]) == "root")
    if len(roots) != 1 or str(roots[0]["insession_task_node_id"]) != task_id:
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "completed Task has no unique canonical root"
        )
    root = roots[0]
    resolved = execution_records._load_current_task_node_delivery_projection(
        conn,
        session_id=session_id,
        task_id=task_id,
        graph_revision=graph_revision,
        node_id=task_id,
        node_revision=int(root["node_revision"]),
    )
    nodes = tuple(_node_from_row(row) for row in rows)
    anchors = task_records._source_anchors_from_json(task["source_anchors_json"])
    child_projection = _project_child_delivery_projection(
        conn,
        session_id=session_id,
        task_id=task_id,
        graph_revision=graph_revision,
        rows=rows,
    )
    node_verification_attestations = _project_node_verification_attestations(
        conn,
        session_id=session_id,
        task_id=task_id,
        graph_revision=graph_revision,
        rows=rows,
    )
    return TaskDeliveryValidationPrompt.create(
        session_id=session_id,
        task_id=task_id,
        graph_revision=graph_revision,
        task_state_version=int(task["state_version"]),
        title=str(task["root_title"]),
        objective=str(task["root_objective"]),
        source_anchors=anchors,
        nodes=nodes,
        root_delivery_id=resolved.delivery_id,
        root_output_format=resolved.source_delivery.output_window.format.value,
        root_output_body=resolved.source_delivery.output_window.content,
        child_delivery_projection=child_projection,
        node_verification_attestations=node_verification_attestations,
    )


def _load_current_validation_node_rows(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    graph_revision: int,
) -> tuple[sqlite3.Row, ...]:
    return tuple(
        conn.execute(
            "SELECT node.insession_task_node_id, node.node_revision, "
            "node.node_kind, node.ordinal, node.title, node.objective, "
            "node.source_anchor_ids_json, node.acceptance_criteria_json, "
            "node.constraints_json, edge.parent_insession_task_node_id, "
            "state.status FROM insession_task_graph_nodes AS node "
            "LEFT JOIN insession_task_graph_edges AS edge "
            "ON edge.insession_task_id=node.insession_task_id "
            "AND edge.graph_revision=node.graph_revision "
            "AND edge.child_insession_task_node_id=node.insession_task_node_id "
            "JOIN insession_task_node_states AS state "
            "ON state.insession_task_id=node.insession_task_id "
            "AND state.insession_task_node_id=node.insession_task_node_id "
            "AND state.node_revision=node.node_revision "
            "WHERE node.insession_task_id=? AND node.graph_revision=? "
            "ORDER BY node.ordinal, node.insession_task_node_id",
            (task_id, graph_revision),
        ).fetchall()
    )


def _project_child_delivery_projection(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    task_id: str,
    graph_revision: int,
    rows: tuple[sqlite3.Row, ...],
) -> TaskDeliveryValidationChildDeliveryProjection:
    materials: list[TaskDeliveryValidationChildDeliveryMaterial] = []
    for row in rows:
        if str(row["node_kind"]) == "root":
            continue
        resolved = execution_records._load_current_task_node_delivery_projection(
            conn,
            session_id=session_id,
            task_id=task_id,
            graph_revision=graph_revision,
            node_id=str(row["insession_task_node_id"]),
            node_revision=int(row["node_revision"]),
        )
        source = resolved.source_delivery
        materials.append(
            TaskDeliveryValidationChildDeliveryMaterial(
                node_id=str(row["insession_task_node_id"]),
                node_revision=int(row["node_revision"]),
                ordinal=int(row["ordinal"]),
                delivery_id=source.delivery.delivery_id,
                source_graph_revision=source.delivery.subject.graph_revision,
                resolution_kind=resolved.resolution_kind.value,
                output_format=source.output_window.format.value,
                output_body=source.output_window.content,
            )
        )
    return build_task_delivery_validation_child_delivery_projection(
        tuple(materials)
    )


def _project_node_verification_attestations(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    task_id: str,
    graph_revision: int,
    rows: tuple[sqlite3.Row, ...],
) -> tuple[TaskDeliveryValidationNodeVerificationAttestation, ...]:
    """为每个当前已验证 Delivery 投影精确的 PASS 标识。"""

    return tuple(
        _project_verified_delivery_attestation(
            conn,
            session_id=session_id,
            task_id=task_id,
            graph_revision=graph_revision,
            row=row,
        )
        for row in rows
    )


def _project_verified_delivery_attestation(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    task_id: str,
    graph_revision: int,
    row: sqlite3.Row,
) -> TaskDeliveryValidationNodeVerificationAttestation:
    """重新认证一个 Delivery，且不暴露 verifier/工具结果正文。"""

    resolved = execution_records._load_current_task_node_delivery_projection(
        conn,
        session_id=session_id,
        task_id=task_id,
        graph_revision=graph_revision,
        node_id=str(row["insession_task_node_id"]),
        node_revision=int(row["node_revision"]),
    )
    source = resolved.source_delivery.delivery
    # 采用延迟导入，因为 work_verification 负责候选项结算，因而会在初始化时导入
    # 本模块。
    from ..work_run.work_verification import _load_request_row, _record_from_row

    request_row = _load_request_row(
        conn,
        session_id=session_id,
        verification_request_id=source.verification_request_id,
    )
    record = _record_from_row(request_row)
    result = record.result
    if (
        result is None
        or not result.all_pass
        or result.subject != source.subject
        or result.work_run_id != source.work_run_id
        or result.submitted_attempt_id != source.submitted_attempt_id
        or result.output_revision != source.output_revision
    ):
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "node verification attestation lost its exact PASS binding"
        )
    return TaskDeliveryValidationNodeVerificationAttestation.create(
        node_id=str(row["insession_task_node_id"]),
        node_revision=int(row["node_revision"]),
        verification_source=(
            "direct_delivery"
            if resolved.resolution_kind.value == "direct"
            else "carried_delivery"
        ),
        delivery_id=source.delivery_id,
        verified_subject_graph_revision=source.subject.graph_revision,
        verification_request_id=result.verification_request_id,
        verification_request_revision=result.verification_request_revision,
        work_run_id=result.work_run_id,
        submitted_attempt_id=result.submitted_attempt_id,
        output_revision=result.output_revision,
        acceptance_ids=record.request.acceptance_ids,
        supporting_tool_result_ids=(
            record.request.supporting_tool_result_ids
        ),
        all_pass=True,
        verification_result_sha256=_sha256_value(
            result.model_dump(mode="json")
        ),
    )


def _node_from_row(row: sqlite3.Row) -> TaskDeliveryValidationNode:
    try:
        acceptance = tuple(
            InSessionTaskAcceptanceProposal.model_validate(item)
            for item in json.loads(str(row["acceptance_criteria_json"]))
        )
        source_anchor_ids = tuple(json.loads(str(row["source_anchor_ids_json"])))
        constraints = tuple(json.loads(str(row["constraints_json"])))
        return TaskDeliveryValidationNode(
            node_id=str(row["insession_task_node_id"]),
            node_revision=int(row["node_revision"]),
            node_kind=InSessionTaskNodeKind(str(row["node_kind"])),
            parent_node_id=(
                str(row["parent_insession_task_node_id"])
                if row["parent_insession_task_node_id"] is not None
                else None
            ),
            ordinal=int(row["ordinal"]),
            title=str(row["title"]),
            objective=str(row["objective"]),
            source_anchor_ids=source_anchor_ids,
            acceptance_criteria=acceptance,
            constraints=constraints,
        )
    except Exception as exc:
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "current TaskGraph node projection is corrupt"
        ) from exc


def _load_request(
    conn: sqlite3.Connection,
    verification_request_id: str,
) -> TaskDeliveryValidationRequest:
    row = conn.execute(
        "SELECT * FROM insession_task_delivery_validation_requests "
        "WHERE verification_request_id=?",
        (verification_request_id,),
    ).fetchone()
    if row is None:
        raise TaskDeliveryValidationStaleAuthority(
            "Task delivery validation request is missing"
        )
    return _request_from_row(row)


def _request_from_row(row: sqlite3.Row) -> TaskDeliveryValidationRequest:
    try:
        request = TaskDeliveryValidationRequest.model_validate_json(
            str(row["request_json"])
        )
    except Exception as exc:
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "stored validation request is invalid"
        ) from exc
    if (
        request.verification_request_id != str(row["verification_request_id"])
        or request.logical_call_id != str(row["logical_call_id"])
        or request.prompt.session_id != str(row["session_id"])
        or request.prompt.task_id != str(row["insession_task_id"])
        or request.prompt.graph_revision != int(row["graph_revision"])
        or request.prompt.task_state_version
        != int(row["completed_task_state_version"])
        or request.prompt.root_delivery_id != str(row["root_delivery_id"])
        or request.invocation_turn_id != str(row["invocation_turn_id"])
        or request.prompt.payload_sha256 != str(row["prompt_payload_sha256"])
        or request.binding_sha256 != str(row["request_binding_sha256"])
        or _model_json(request) != str(row["request_json"])
    ):
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "stored validation request row projection is corrupt"
        )
    return request


def _candidate_authority_payload(
    intent: TaskDeliveryCandidateSettlementIntent,
) -> dict[str, object]:
    return {
        "schema_version": "task-delivery-candidate-authority-v1",
        "session_id": intent.session_id,
        "invocation_turn_id": intent.invocation_turn_id,
        "subject": intent.subject.model_dump(mode="json"),
        "task_state_version": intent.task_state_version,
        "work_run_id": intent.work_run_id,
        "submitted_attempt_id": intent.submitted_attempt_id,
        "verification_request_id": intent.verification_request_id,
        "verification_request_revision": intent.verification_request_revision,
        "output_revision": intent.output_revision,
        "output_sha256": intent.output_sha256,
        "candidate_delivery_id": intent.candidate_delivery_id,
        "review_request": intent.review_request.model_dump(mode="json"),
        "authority_sha256": intent.authority_sha256,
    }


def _candidate_settlement_identity(
    intent: TaskDeliveryCandidateSettlementIntent,
) -> str:
    return _sha256_value(
        {
            "contract": "task-delivery-candidate-settlement-id-v1",
            "authority_sha256": intent.authority_sha256,
            "result_id": intent.result.verification_result_id,
            "result_sha256": intent.result.result_sha256,
            "candidate_delivery_id": intent.candidate_delivery_id,
        }
    )[:40]


def _candidate_sql_disposition(
    disposition: TaskDeliveryValidationDisposition,
) -> str:
    if disposition is TaskDeliveryValidationDisposition.PASS:
        return "pass"
    if disposition is TaskDeliveryValidationDisposition.REPLAN_TASK_GRAPH:
        return "revise"
    if disposition is TaskDeliveryValidationDisposition.BLOCKED:
        return "blocked"
    raise TaskDeliveryValidationPersistenceError(
        "candidate SQL settlement supports only replan or blocked"
    )


def _require_candidate_settlement_route(
    result: TaskDeliveryValidationResult,
) -> None:
    """只准入此持久化切片能够安全授权的路由。

    普通用户回答可以补充缺失信息，但并非 Authorization/Approval/Receipt。在具有
    副作用的权威契约建立前，分类为 ``missing_authority`` 的候选项必须回滚外围的
    节点验证事务，而不能创建可能意外扩大权限的 user gate。
    """

    if result.disposition is not TaskDeliveryValidationDisposition.BLOCKED:
        return
    non_pass_domains = {
        item.fault_domain
        for item in result.findings
        if item.verdict is not TaskDeliveryValidationVerdict.PASS
    }
    if (
        TaskDeliveryValidationFaultDomain.MISSING_AUTHORITY
        in non_pass_domains
    ):
        raise TaskDeliveryValidationPersistenceError(
            "candidate BLOCKED requires formal authority; ordinary user "
            "clarification cannot manufacture Authorization/Receipt"
        )
    if (
        TaskDeliveryValidationFaultDomain.MISSING_INFORMATION
        not in non_pass_domains
    ):
        raise TaskDeliveryValidationPersistenceError(
            "candidate BLOCKED has no typed missing-information authority"
        )


def _candidate_blocked_revision_objective(
    result: TaskDeliveryValidationResult,
) -> str:
    """从有类型问题构建有界、便于人类阅读的目标。

    路由归属绝不依赖这段文字。完整问题和故障域仍保留在已认证的 结算及有类型
    Architect 路由投影中。
    """

    _require_candidate_settlement_route(result)
    prefix = (
        "Insert a required clarification user_gate, obtain its authenticated "
        "user response, then revise TaskGraph N+1 using the missing "
        "information: "
    )
    parts: list[str] = []
    for ordinal, question in enumerate(result.blocking_questions, start=1):
        candidate = f"Q{ordinal}. {question.strip()}"
        joined = " ".join((*parts, candidate))
        if len(prefix) + len(joined) <= 3_850:
            parts.append(candidate)
            continue
        remaining = len(result.blocking_questions) - len(parts)
        parts.append(
            f"[{remaining} additional question(s) remain sealed in result "
            f"{result.result_sha256[:16]}]"
        )
        break
    objective = prefix + " ".join(parts)
    if not objective.strip() or len(objective) > 4_000:
        raise TaskDeliveryValidationPersistenceError(
            "candidate BLOCKED objective cannot be projected within bounds"
        )
    return objective


def _candidate_gap_diagnosis(
    result: TaskDeliveryValidationResult,
) -> tuple[TaskDeliveryValidationFinding, ...]:
    gaps = tuple(
        item for item in result.findings
        if item.verdict is not TaskDeliveryValidationVerdict.PASS
    )
    if not gaps:
        raise TaskDeliveryValidationPersistenceError(
            "candidate replan has no graph-design gap diagnosis"
        )
    return gaps


def _require_candidate_node_verification(
    conn: sqlite3.Connection,
    *,
    intent: TaskDeliveryCandidateSettlementIntent,
) -> None:
    row = conn.execute(
        "SELECT request_revision, status, result_json, all_pass, work_run_id, "
        "submitted_attempt_id, output_revision, insession_task_id, "
        "graph_revision, insession_task_node_id, node_revision "
        "FROM insession_work_run_verification_requests "
        "WHERE verification_request_id=?",
        (intent.verification_request_id,),
    ).fetchone()
    if row is None or row["result_json"] is None:
        raise TaskDeliveryValidationStaleAuthority(
            "candidate settlement lost its node verification PASS"
        )
    try:
        result = NodeVerificationResult.model_validate_json(
            str(row["result_json"])
        )
    except Exception as exc:
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "candidate node verification result is corrupt"
        ) from exc
    if (
        str(row["status"]) != "completed"
        or int(row["all_pass"] or 0) != 1
        or int(row["request_revision"])
        != intent.verification_request_revision + 1
        or str(row["work_run_id"]) != intent.work_run_id
        or str(row["submitted_attempt_id"]) != intent.submitted_attempt_id
        or int(row["output_revision"]) != intent.output_revision
        or str(row["insession_task_id"]) != intent.subject.task_id
        or int(row["graph_revision"]) != intent.subject.graph_revision
        or str(row["insession_task_node_id"]) != intent.subject.node_id
        or int(row["node_revision"]) != intent.subject.node_revision
        or not result.all_pass
        or result.downstream_results
        or result.verification_request_id != intent.verification_request_id
        or result.verification_request_revision
        != intent.verification_request_revision
        or result.work_run_id != intent.work_run_id
        or result.submitted_attempt_id != intent.submitted_attempt_id
        or result.output_revision != intent.output_revision
        or result.subject != intent.subject
    ):
        raise TaskDeliveryValidationStaleAuthority(
            "candidate route is not bound to the exact node Acceptance PASS"
        )
    delivery = conn.execute(
        "SELECT delivery_id, verification_request_id, work_run_id, "
        "submitted_attempt_id, output_revision FROM "
        "insession_task_node_deliveries WHERE delivery_id=?",
        (intent.candidate_delivery_id,),
    ).fetchone()
    if (
        delivery is None
        or str(delivery["verification_request_id"])
        != intent.verification_request_id
        or str(delivery["work_run_id"]) != intent.work_run_id
        or str(delivery["submitted_attempt_id"])
        != intent.submitted_attempt_id
        or int(delivery["output_revision"]) != intent.output_revision
    ):
        raise TaskDeliveryValidationStaleAuthority(
            "candidate settlement lost its exact frozen root Delivery"
        )


def _require_candidate_prompt_matches_completed_delivery(
    *,
    intent: TaskDeliveryCandidateSettlementIntent,
    current: TaskDeliveryValidationPrompt,
) -> None:
    candidate = intent.review_request.prompt
    excluded = {
        "task_state_version",
        "node_verification_attestations",
        "payload_sha256",
    }
    candidate_attestations = []
    for item in candidate.node_verification_attestations:
        payload = item.model_dump(mode="json", exclude={"binding_sha256"})
        if (
            item.node_id == candidate.task_id
            and item.verification_source == "root_candidate"
        ):
            payload["verification_source"] = "direct_delivery"
        candidate_attestations.append(payload)
    current_attestations = [
        item.model_dump(mode="json", exclude={"binding_sha256"})
        for item in current.node_verification_attestations
    ]
    if (
        current.task_state_version != intent.task_state_version + 1
        or candidate.task_state_version != intent.task_state_version
        or candidate.model_dump(mode="json", exclude=excluded)
        != current.model_dump(mode="json", exclude=excluded)
        or candidate_attestations != current_attestations
    ):
        raise TaskDeliveryValidationStaleAuthority(
            "candidate prompt differs from its exact completed root Delivery"
        )


def _require_exact_candidate_model_result(
    conn: sqlite3.Connection,
    *,
    intent: TaskDeliveryCandidateSettlementIntent,
) -> None:
    try:
        logical = model_call_records._require_logical_call(
            conn,
            intent.review_request.logical_call_id,
        )
    except Exception as exc:
        raise TaskDeliveryValidationPersistenceError(
            "candidate result has no durable model-call authority"
        ) from exc
    frozen = logical.request
    if (
        frozen.session_id != intent.session_id
        or frozen.task_id != intent.subject.task_id
        or frozen.auxiliary_graph_id is not None
        or frozen.goal_id is not None
        or frozen.execution_subject_id is not None
        or frozen.invocation_turn_id != intent.invocation_turn_id
        or frozen.call_kind != _CANDIDATE_CALL_KIND
        or frozen.purpose != _CANDIDATE_PURPOSE
        or frozen.request_contract != _CANDIDATE_REQUEST_CONTRACT
        or frozen.request_json != _canonical_json(_candidate_authority_payload(intent))
        or frozen.typed_result_contract != _CANDIDATE_RESULT_CONTRACT
        or frozen.state_guard_sha256 != intent.authority_sha256
    ):
        raise TaskDeliveryValidationPersistenceError(
            "durable candidate request differs from frozen authority"
        )
    if not logical.physical_attempts:
        raise TaskDeliveryValidationPersistenceError(
            "candidate model call has no settled physical attempt"
        )
    last = logical.physical_attempts[-1].settlement
    if (
        last is None
        or last.outcome is not RuntimeModelPhysicalOutcome.SUCCEEDED
        or last.typed_result is None
        or last.typed_result.result_contract != _CANDIDATE_RESULT_CONTRACT
        or last.typed_result.parsed() != _candidate_model_result_payload(intent.result)
    ):
        raise TaskDeliveryValidationPersistenceError(
            "candidate result is not the exact durable typed model success"
        )


def _candidate_model_result_payload(
    result: TaskDeliveryValidationResult,
) -> dict[str, object]:
    return {
        "findings": [item.model_dump(mode="json") for item in result.findings],
        "summary": result.summary,
        "execution_repair_objective": result.execution_repair_objective,
        "task_graph_revision_objective": result.task_graph_revision_objective,
        "blocking_questions": list(result.blocking_questions),
    }


def _stored_candidate_from_settlement_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
) -> StoredTaskDeliveryCandidateSettlement:
    request = _load_request(conn, str(row["verification_request_id"]))
    result_row = conn.execute(
        "SELECT * FROM insession_task_delivery_validation_results "
        "WHERE verification_result_id=?",
        (str(row["verification_result_id"]),),
    ).fetchone()
    if result_row is None:
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "candidate settlement lost its result"
        )
    try:
        result = TaskDeliveryValidationResult.model_validate_json(
            str(result_row["result_json"])
        )
        settlement = TaskDeliveryCandidateSettlement.model_validate_json(
            str(row["settlement_json"])
        )
        validate_task_delivery_validation_result(
            request=request,
            result=result,
        )
        logical = model_call_records._require_logical_call(
            conn,
            request.logical_call_id,
        )
        authority_payload = json.loads(logical.request.request_json)
        if not isinstance(authority_payload, dict):
            raise ValueError("candidate authority payload is not an object")
        authority_payload = dict(authority_payload)
        authority_payload["schema_version"] = (
            "task-delivery-candidate-settlement-intent-v1"
        )
        authority_payload["result"] = result.model_dump(mode="json")
        intent = TaskDeliveryCandidateSettlementIntent.model_validate(
            authority_payload
        )
    except Exception as exc:
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "stored candidate result or settlement is invalid"
        ) from exc
    mapped = _candidate_sql_disposition(result.disposition)
    if (
        result.verification_result_id != str(result_row["verification_result_id"])
        or result.verification_request_id
        != str(result_row["verification_request_id"])
        or result.logical_call_id != str(result_row["logical_call_id"])
        or str(result_row["session_id"]) != intent.session_id
        or str(result_row["insession_task_id"]) != intent.subject.task_id
        or str(result_row["disposition"]) != mapped
        or result.result_sha256 != str(result_row["result_sha256"])
        or _model_json(result) != str(result_row["result_json"])
        or settlement.settlement_id != str(row["settlement_id"])
        or settlement.apply_id != str(row["apply_id"])
        or settlement.verification_request_id
        != str(row["verification_request_id"])
        or settlement.verification_result_id
        != str(row["verification_result_id"])
        or settlement.session_id != str(row["session_id"])
        or settlement.task_id != str(row["insession_task_id"])
        or settlement.graph_revision != int(row["graph_revision"])
        or settlement.completed_task_state_version
        != int(row["completed_task_state_version"])
        or settlement.root_delivery_id != str(row["root_delivery_id"])
        or str(row["disposition"]) != mapped
        or settlement.created_turn_id != str(row["created_turn_id"])
        or settlement.settlement_sha256 != str(row["settlement_sha256"])
        or _model_json(settlement) != str(row["settlement_json"])
        or settlement.session_id != intent.session_id
        or settlement.task_id != intent.subject.task_id
        or settlement.graph_revision != intent.subject.graph_revision
        or settlement.candidate_task_state_version != intent.task_state_version
        or settlement.root_delivery_id != intent.candidate_delivery_id
        or settlement.node_verification_request_id
        != intent.verification_request_id
        or settlement.node_verification_request_revision
        != intent.verification_request_revision
        or settlement.candidate_authority_sha256 != intent.authority_sha256
        or settlement.verification_request_id
        != request.verification_request_id
        or settlement.request_binding_sha256 != request.binding_sha256
        or settlement.verification_result_id != result.verification_result_id
        or settlement.result_sha256 != result.result_sha256
        or settlement.disposition is not result.disposition
        or settlement.created_turn_id != intent.invocation_turn_id
    ):
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "stored candidate settlement row projection is corrupt"
        )
    trigger_rows = conn.execute(
        "SELECT trigger_id FROM insession_task_graph_revision_triggers "
        "WHERE settlement_id=?",
        (settlement.settlement_id,),
    ).fetchall()
    trigger = None
    if result.disposition is not TaskDeliveryValidationDisposition.PASS:
        if len(trigger_rows) != 1:
            raise TaskDeliveryValidationStoredAuthorityCorrupt(
                "candidate non-PASS settlement lost its unique trigger"
            )
        trigger = _load_trigger(conn, str(trigger_rows[0]["trigger_id"]))
        if (
            trigger.create_apply_id != settlement.apply_id
            or trigger.session_id != settlement.session_id
            or trigger.task_id != settlement.task_id
            or trigger.base_graph_revision != settlement.graph_revision
            or trigger.target_graph_revision != settlement.graph_revision + 1
            or trigger.root_delivery_id != settlement.root_delivery_id
            or trigger.verification_request_id
            != settlement.verification_request_id
            or trigger.request_binding_sha256
            != settlement.request_binding_sha256
            or trigger.verification_result_id
            != settlement.verification_result_id
            or trigger.result_sha256 != settlement.result_sha256
            or trigger.settlement_id != settlement.settlement_id
            or trigger.settlement_sha256 != settlement.settlement_sha256
            or trigger.revision_objective
            != (
                result.task_graph_revision_objective
                if result.disposition
                is TaskDeliveryValidationDisposition.REPLAN_TASK_GRAPH
                else _candidate_blocked_revision_objective(result)
            )
            or trigger.gap_diagnosis != _candidate_gap_diagnosis(result)
            or trigger.reopened_task_state_version
            != settlement.settled_task_state_version
            or trigger.created_turn_id != settlement.created_turn_id
        ):
            raise TaskDeliveryValidationStoredAuthorityCorrupt(
                "candidate TaskGraph trigger crossed authority"
            )
    elif trigger_rows:
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "candidate non-REPLAN settlement unexpectedly owns a trigger"
        )
    try:
        _require_exact_candidate_model_result(conn, intent=intent)
        _require_validation_root_delivery(conn, request=request)
    except TaskDeliveryValidationStoredAuthorityCorrupt:
        raise
    except Exception as exc:
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "stored candidate lost model or root Delivery authority"
        ) from exc
    return StoredTaskDeliveryCandidateSettlement(
        intent=intent,
        settlement=settlement,
        trigger=trigger,
    )


def _load_trigger(
    conn: sqlite3.Connection,
    trigger_id: str,
) -> TaskGraphRevisionTrigger:
    row = conn.execute(
        "SELECT * FROM insession_task_graph_revision_triggers WHERE trigger_id=?",
        (trigger_id,),
    ).fetchone()
    if row is None:
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "TaskGraph revision trigger is missing"
        )
    try:
        trigger = TaskGraphRevisionTrigger.model_validate_json(
            str(row["trigger_json"])
        )
    except Exception as exc:
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "stored TaskGraph revision trigger is invalid"
        ) from exc
    if (
        trigger.trigger_id != str(row["trigger_id"])
        or trigger.create_apply_id != str(row["create_apply_id"])
        or trigger.settlement_id != str(row["settlement_id"])
        or trigger.session_id != str(row["session_id"])
        or trigger.task_id != str(row["insession_task_id"])
        or trigger.base_graph_revision != int(row["base_graph_revision"])
        or trigger.target_graph_revision != int(row["target_graph_revision"])
        or trigger.root_delivery_id != str(row["root_delivery_id"])
        or trigger.reopened_task_state_version
        != int(row["reopened_task_state_version"])
        or trigger.trigger_sha256 != str(row["trigger_sha256"])
        or _model_json(trigger) != str(row["trigger_json"])
    ):
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "stored TaskGraph revision trigger row projection is corrupt"
        )
    return trigger


def _load_authenticated_trigger_authority(
    conn: sqlite3.Connection,
    trigger_id: str,
) -> TaskGraphRevisionTrigger:
    """重新加载一个不可变 trigger 背后的完整 REVISE 链。"""

    trigger = _load_trigger(conn, trigger_id)
    rows = conn.execute(
        "SELECT * FROM insession_task_delivery_validation_settlements "
        "WHERE settlement_id=?",
        (trigger.settlement_id,),
    ).fetchall()
    if len(rows) != 1:
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "TaskGraph revision trigger lost its unique settlement"
        )
    stored_candidate = _stored_candidate_from_settlement_row(conn, rows[0])
    if (
        stored_candidate.settlement.disposition
        not in {
            TaskDeliveryValidationDisposition.REPLAN_TASK_GRAPH,
            TaskDeliveryValidationDisposition.BLOCKED,
        }
        or stored_candidate.trigger != trigger
    ):
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "TaskGraph trigger is not backed by exact candidate authority"
        )
    return trigger


def _require_validation_root_delivery(
    conn: sqlite3.Connection,
    *,
    request: TaskDeliveryValidationRequest,
) -> None:
    root = next(
        (
            item
            for item in request.prompt.nodes
            if item.node_kind is InSessionTaskNodeKind.ROOT
        ),
        None,
    )
    if root is None or root.node_id != request.prompt.task_id:
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "stored validation prompt lost its canonical root"
        )
    try:
        resolved = execution_records._load_current_task_node_delivery_projection(
            conn,
            session_id=request.prompt.session_id,
            task_id=request.prompt.task_id,
            graph_revision=request.prompt.graph_revision,
            node_id=root.node_id,
            node_revision=root.node_revision,
        )
    except Exception as exc:
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "stored validation root Delivery is no longer authentic"
        ) from exc
    output = resolved.source_delivery.output_window
    if (
        resolved.delivery_id != request.prompt.root_delivery_id
        or output.format.value != request.prompt.root_output_format
        or output.content != request.prompt.root_output_body
    ):
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "stored validation resolves another root Delivery projection"
        )


def _application_from_row(
    row: sqlite3.Row,
) -> TaskGraphRevisionTriggerApplication:
    try:
        application = TaskGraphRevisionTriggerApplication.model_validate_json(
            str(row["receipt_json"])
        )
    except Exception as exc:
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "stored trigger application is invalid"
        ) from exc
    if (
        application.apply_id != str(row["apply_id"])
        or application.trigger_id != str(row["trigger_id"])
        or application.trigger_sha256 != str(row["trigger_sha256"])
        or application.session_id != str(row["session_id"])
        or application.task_id != str(row["insession_task_id"])
        or application.base_graph_revision != int(row["base_graph_revision"])
        or application.committed_graph_revision
        != int(row["committed_graph_revision"])
        or application.task_graph_commit_apply_id
        != str(row["task_graph_commit_apply_id"])
        or application.consumed_turn_id != str(row["consumed_turn_id"])
        or application.receipt_sha256 != str(row["receipt_sha256"])
        or _model_json(application) != str(row["receipt_json"])
    ):
        raise TaskDeliveryValidationStoredAuthorityCorrupt(
            "stored trigger application row projection is corrupt"
        )
    return application


def _require_running_turn(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
) -> None:
    row = conn.execute(
        "SELECT session_id, status FROM runtime_turns WHERE turn_id=?",
        (turn_id,),
    ).fetchone()
    if (
        row is None
        or str(row["session_id"]) != session_id
        or str(row["status"]) != "running"
    ):
        raise TaskDeliveryValidationStaleAuthority(
            "Task delivery validation requires its exact running Turn"
        )


def _validate_model(model_type: type[_Record], value: object, label: str):
    if not isinstance(value, model_type):
        raise TypeError(f"{label} must be {model_type.__name__}")
    try:
        return model_type.model_validate(value.model_dump(mode="json"))
    except Exception as exc:
        raise TaskDeliveryValidationPersistenceError(
            f"{label} is not self-authenticating"
        ) from exc


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _model_json(value: BaseModel) -> str:
    return _canonical_json(value.model_dump(mode="json"))


def _sha256_value(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_id(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 200
    ):
        raise ValueError(f"{name} must be a bounded canonical identity")


__all__ = [
    "ConsumeTaskGraphRevisionTriggerCommand",
    "StoredTaskDeliveryCandidateSettlement",
    "TaskDeliveryCandidateSettlementIntent",
    "TaskDeliveryCandidateSettlementMutationResult",
    "TaskDeliveryCandidateSettlement",
    "TaskDeliveryValidationIdentityCollision",
    "TaskDeliveryValidationPersistenceError",
    "TaskDeliveryValidationStaleAuthority",
    "TaskDeliveryValidationStoredAuthorityCorrupt",
    "TaskGraphRevisionTriggerApplicationMutationResult",
    "consume_task_graph_revision_trigger",
    "consume_task_graph_revision_trigger_in_transaction",
    "get_active_task_graph_revision_trigger",
    "get_task_delivery_candidate_settlement",
    "project_task_delivery_validation_child_deliveries",
    "replay_task_delivery_candidate_settlement_in_transaction",
    "require_pass_settlement_for_publication_in_transaction",
    "settle_task_delivery_candidate_route_in_transaction",
]
