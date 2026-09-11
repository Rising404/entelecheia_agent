"""持久且绑定原因的 AuxiliaryGraph 自主重规划触发器。

语义审查器绝不写入此权威。Store 根据已认证非 PASS 法定人数和精确当前图、目标与预算
CAS 派生一个不可变触发器。revision 应用是第二份不可变回执，因此进程在图 revision
N+1 后崩溃时，可通过精确重放协调，再进行消费而无需重复语义审查。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from personagraph.l2.auxiliary_graph import (
    AuxiliaryGraphRevisionReason,
    AuxiliaryReplanTriggerApplicationReceipt,
    AuxiliaryReplanTriggerReceipt,
    PlanningEpisodeBudget,
    TaskGraphSemanticVerificationDisposition,
)
from ..auxiliary_graph.auxiliary_graph_errors import AuxiliaryGraphPersistenceError
from ..auxiliary_graph.auxiliary_graphs import StoredAuxiliaryGraphDetails, _load_auxiliary_graph
from ..delivery.auxiliary_semantic_verification import (
    AuxiliarySemanticVerificationPersistenceError,
    StoredAuxiliarySemanticQuorumSettlement,
    _load_settlement,
)
from ...deps import StoreDeps


_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class AuxiliaryReplanTriggerPersistenceError(AuxiliaryGraphPersistenceError):
    """重规划触发器变更丢失权威不变量。"""


class AuxiliaryReplanTriggerIdentityCollision(
    AuxiliaryReplanTriggerPersistenceError
):
    """应用、触发器、结算或应用记录身份被复用。"""


class AuxiliaryReplanTriggerStaleAuthority(
    AuxiliaryReplanTriggerPersistenceError
):
    """当前图、目标或预算权威不再匹配 CAS。"""


class AuxiliaryReplanTriggerStoredAuthorityCorrupt(
    AuxiliaryReplanTriggerPersistenceError
):
    """不可变触发器或应用记录行不再能自认证。"""


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CreateAuxiliaryReplanTriggerCommand(_Record):
    schema_version: Literal["create-auxiliary-replan-trigger-command-v1"] = (
        "create-auxiliary-replan-trigger-command-v1"
    )
    apply_id: str = Field(pattern=_ID_PATTERN)
    trigger_id: str = Field(pattern=_ID_PATTERN)
    session_id: str = Field(pattern=_ID_PATTERN)
    created_turn_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_id: str = Field(pattern=_ID_PATTERN)
    goal_id: str = Field(pattern=_ID_PATTERN)
    expected_control_state_version: int = Field(ge=1)
    expected_goal_state_version: int = Field(ge=1)
    expected_revision_state_version: int = Field(ge=1)
    expected_budget_state_version: int = Field(ge=1)
    expected_current_auxiliary_graph_revision: int = Field(ge=1)
    expected_current_structure_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_current_authority_snapshot_id: str = Field(pattern=_ID_PATTERN)
    expected_current_authority_snapshot_sha256: str = Field(
        pattern=_SHA256_PATTERN
    )
    semantic_settlement_id: str = Field(pattern=_ID_PATTERN)
    expected_semantic_settlement_sha256: str = Field(pattern=_SHA256_PATTERN)


class AuxiliaryReplanTriggerMutationResult(_Record):
    status: Literal["applied", "replayed"]
    trigger: AuxiliaryReplanTriggerReceipt


class ConsumeAuxiliaryReplanTriggerCommand(_Record):
    schema_version: Literal["consume-auxiliary-replan-trigger-command-v1"] = (
        "consume-auxiliary-replan-trigger-command-v1"
    )
    apply_id: str = Field(pattern=_ID_PATTERN)
    session_id: str = Field(pattern=_ID_PATTERN)
    consumed_turn_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    trigger_id: str = Field(pattern=_ID_PATTERN)
    expected_goal_id: str = Field(pattern=_ID_PATTERN)
    expected_applied_auxiliary_graph_revision: int = Field(ge=2)
    expected_applied_structure_sha256: str = Field(pattern=_SHA256_PATTERN)


class AuxiliaryReplanTriggerApplicationMutationResult(_Record):
    status: Literal["applied", "replayed"]
    application: AuxiliaryReplanTriggerApplicationReceipt


def create_auxiliary_replan_trigger(
    deps: StoreDeps,
    *,
    command: CreateAuxiliaryReplanTriggerCommand,
) -> AuxiliaryReplanTriggerMutationResult:
    """根据当前非 PASS 结算派生并追加一个触发器。"""

    command = _validate_command(
        CreateAuxiliaryReplanTriggerCommand,
        command,
        "replan trigger create command",
    )
    command_json = _model_json(command)
    command_sha256 = _sha256_text(command_json)
    deps.init_db()
    try:
        with deps.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            replay = conn.execute(
                "SELECT trigger_id, command_json, command_sha256 FROM "
                "insession_auxiliary_replan_triggers WHERE create_apply_id=?",
                (command.apply_id,),
            ).fetchone()
            if replay is not None:
                if (
                    str(replay["trigger_id"]) != command.trigger_id
                    or str(replay["command_json"]) != command_json
                    or str(replay["command_sha256"]) != command_sha256
                ):
                    raise AuxiliaryReplanTriggerIdentityCollision(
                        "replan trigger apply id crossed immutable content"
                    )
                return AuxiliaryReplanTriggerMutationResult(
                    status="replayed",
                    trigger=_load_trigger(conn, command.trigger_id),
                )

            identity = conn.execute(
                "SELECT trigger_id FROM insession_auxiliary_replan_triggers "
                "WHERE trigger_id=? OR semantic_settlement_id=?",
                (command.trigger_id, command.semantic_settlement_id),
            ).fetchone()
            if identity is not None:
                raise AuxiliaryReplanTriggerIdentityCollision(
                    "trigger or semantic settlement already owns another apply"
                )
            active = conn.execute(
                "SELECT trigger_id FROM "
                "insession_auxiliary_active_replan_triggers "
                "WHERE session_id=? AND insession_task_id=?",
                (command.session_id, command.task_id),
            ).fetchone()
            if active is not None:
                raise AuxiliaryReplanTriggerIdentityCollision(
                    "Task already has an unconsumed replan trigger"
                )
            _require_running_turn(
                conn,
                session_id=command.session_id,
                turn_id=command.created_turn_id,
            )
            details = _load_current_details(conn, command.session_id, command.task_id)
            _require_create_cas(details=details, command=command)
            settlement = _load_authenticated_settlement(
                conn,
                command.semantic_settlement_id,
            )
            trigger = _derive_trigger(
                command=command,
                details=details,
                settlement=settlement,
            )
            now = deps.now()
            _insert_trigger(
                conn,
                trigger=trigger,
                command_json=command_json,
                command_sha256=command_sha256,
                now=now,
            )
            conn.execute(
                "INSERT INTO insession_auxiliary_active_replan_triggers "
                "(trigger_id, semantic_settlement_id, session_id, "
                "insession_task_id, auxiliary_graph_id, goal_id, "
                "source_auxiliary_graph_revision, activated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trigger.trigger_id,
                    trigger.semantic_settlement_id,
                    trigger.session_id,
                    trigger.task_id,
                    trigger.auxiliary_graph_id,
                    trigger.goal_id,
                    trigger.source_auxiliary_graph_revision,
                    now,
                ),
            )
            return AuxiliaryReplanTriggerMutationResult(
                status="applied",
                trigger=_load_trigger(conn, trigger.trigger_id),
            )
    except sqlite3.IntegrityError as exc:
        raise AuxiliaryReplanTriggerPersistenceError(
            "replan trigger violated v71 durable authority"
        ) from exc


def consume_auxiliary_replan_trigger(
    deps: StoreDeps,
    *,
    command: ConsumeAuxiliaryReplanTriggerCommand,
) -> AuxiliaryReplanTriggerApplicationMutationResult:
    """记录触发器 N 生成了当前图 revision N+1。

    它刻意跟随现有图提交，而不嵌入其中。崩溃后活动触发器仍会保留；Runtime 可以精确
    重放图提交，再安全调用此命令。
    """

    command = _validate_command(
        ConsumeAuxiliaryReplanTriggerCommand,
        command,
        "replan trigger consume command",
    )
    command_json = _model_json(command)
    command_sha256 = _sha256_text(command_json)
    deps.init_db()
    try:
        with deps.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            replay = conn.execute(
                "SELECT trigger_id, command_json, command_sha256 FROM "
                "insession_auxiliary_replan_trigger_applications "
                "WHERE apply_id=?",
                (command.apply_id,),
            ).fetchone()
            if replay is not None:
                if (
                    str(replay["trigger_id"]) != command.trigger_id
                    or str(replay["command_json"]) != command_json
                    or str(replay["command_sha256"]) != command_sha256
                ):
                    raise AuxiliaryReplanTriggerIdentityCollision(
                        "replan application apply id crossed immutable content"
                    )
                return AuxiliaryReplanTriggerApplicationMutationResult(
                    status="replayed",
                    application=_load_application(conn, command.trigger_id),
                )
            if conn.execute(
                "SELECT 1 FROM insession_auxiliary_replan_trigger_applications "
                "WHERE trigger_id=?",
                (command.trigger_id,),
            ).fetchone() is not None:
                raise AuxiliaryReplanTriggerIdentityCollision(
                    "replan trigger already owns another application receipt"
                )
            _require_running_turn(
                conn,
                session_id=command.session_id,
                turn_id=command.consumed_turn_id,
            )
            active = conn.execute(
                "SELECT * FROM insession_auxiliary_active_replan_triggers "
                "WHERE trigger_id=?",
                (command.trigger_id,),
            ).fetchone()
            if active is None:
                raise AuxiliaryReplanTriggerStaleAuthority(
                    "replan trigger is not the active durable trigger"
                )
            trigger = _load_trigger(conn, command.trigger_id)
            if (
                command.session_id != trigger.session_id
                or command.task_id != trigger.task_id
                or command.expected_goal_id != trigger.goal_id
                or str(active["session_id"]) != trigger.session_id
                or str(active["insession_task_id"]) != trigger.task_id
                or str(active["auxiliary_graph_id"]) != trigger.auxiliary_graph_id
                or str(active["goal_id"]) != trigger.goal_id
                or str(active["semantic_settlement_id"])
                != trigger.semantic_settlement_id
                or int(active["source_auxiliary_graph_revision"])
                != trigger.source_auxiliary_graph_revision
            ):
                raise AuxiliaryReplanTriggerStoredAuthorityCorrupt(
                    "active replan pointer crossed trigger authority"
                )
            details = _load_current_details(conn, command.session_id, command.task_id)
            expected_revision = trigger.source_auxiliary_graph_revision + 1
            if (
                command.expected_applied_auxiliary_graph_revision
                != expected_revision
                or details.auxiliary_graph_id != trigger.auxiliary_graph_id
                or details.goal_id != trigger.goal_id
                or details.auxiliary_graph_revision != expected_revision
                or details.parent_auxiliary_graph_revision
                != trigger.source_auxiliary_graph_revision
                or details.structure_sha256
                != command.expected_applied_structure_sha256
                or details.reason
                != AuxiliaryGraphRevisionReason.VERIFICATION_FAILED.value
            ):
                raise AuxiliaryReplanTriggerStaleAuthority(
                    "current graph is not the reason-bound N+1 replan revision"
                )
            application = AuxiliaryReplanTriggerApplicationReceipt.create(
                apply_id=command.apply_id,
                trigger_id=trigger.trigger_id,
                trigger_receipt_sha256=trigger.receipt_sha256,
                session_id=trigger.session_id,
                task_id=trigger.task_id,
                auxiliary_graph_id=trigger.auxiliary_graph_id,
                goal_id=trigger.goal_id,
                source_auxiliary_graph_revision=(
                    trigger.source_auxiliary_graph_revision
                ),
                applied_auxiliary_graph_revision=expected_revision,
                applied_structure_sha256=details.structure_sha256,
                consumed_turn_id=command.consumed_turn_id,
            )
            conn.execute(
                "INSERT INTO insession_auxiliary_replan_trigger_applications "
                "(apply_id, trigger_id, trigger_receipt_sha256, session_id, "
                "insession_task_id, auxiliary_graph_id, goal_id, "
                "source_auxiliary_graph_revision, "
                "applied_auxiliary_graph_revision, applied_structure_sha256, "
                "revision_reason, consumed_turn_id, command_json, "
                "command_sha256, receipt_json, receipt_sha256, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    application.apply_id,
                    application.trigger_id,
                    application.trigger_receipt_sha256,
                    application.session_id,
                    application.task_id,
                    application.auxiliary_graph_id,
                    application.goal_id,
                    application.source_auxiliary_graph_revision,
                    application.applied_auxiliary_graph_revision,
                    application.applied_structure_sha256,
                    application.revision_reason.value,
                    application.consumed_turn_id,
                    command_json,
                    command_sha256,
                    _model_json(application),
                    application.receipt_sha256,
                    deps.now(),
                ),
            )
            if conn.execute(
                "DELETE FROM insession_auxiliary_active_replan_triggers "
                "WHERE trigger_id=? AND semantic_settlement_id=?",
                (trigger.trigger_id, trigger.semantic_settlement_id),
            ).rowcount != 1:
                raise AuxiliaryReplanTriggerStaleAuthority(
                    "active replan trigger changed during application"
                )
            return AuxiliaryReplanTriggerApplicationMutationResult(
                status="applied",
                application=_load_application(conn, trigger.trigger_id),
            )
    except sqlite3.IntegrityError as exc:
        raise AuxiliaryReplanTriggerPersistenceError(
            "replan trigger application violated v71 durable authority"
        ) from exc


def get_auxiliary_replan_trigger(
    deps: StoreDeps,
    *,
    session_id: str,
    trigger_id: str,
) -> AuxiliaryReplanTriggerReceipt | None:
    _require_id("session_id", session_id)
    _require_id("trigger_id", trigger_id)
    deps.init_db()
    with deps.connect() as conn:
        if conn.execute(
            "SELECT 1 FROM insession_auxiliary_replan_triggers "
            "WHERE trigger_id=?",
            (trigger_id,),
        ).fetchone() is None:
            return None
        trigger = _load_trigger(conn, trigger_id)
        if trigger.session_id != session_id:
            raise AuxiliaryReplanTriggerStoredAuthorityCorrupt(
                "replan trigger crossed Session authority"
            )
        return trigger


def get_active_auxiliary_replan_trigger(
    deps: StoreDeps,
    *,
    session_id: str,
    task_id: str,
) -> AuxiliaryReplanTriggerReceipt | None:
    _require_id("session_id", session_id)
    _require_id("task_id", task_id)
    deps.init_db()
    with deps.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM insession_auxiliary_active_replan_triggers "
            "WHERE session_id=? AND insession_task_id=?",
            (session_id, task_id),
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise AuxiliaryReplanTriggerStoredAuthorityCorrupt(
                "Task owns multiple active replan triggers"
            )
        row = rows[0]
        trigger = _load_trigger(conn, str(row["trigger_id"]))
        if (
            trigger.session_id != session_id
            or trigger.task_id != task_id
            or str(row["semantic_settlement_id"])
            != trigger.semantic_settlement_id
            or str(row["auxiliary_graph_id"]) != trigger.auxiliary_graph_id
            or str(row["goal_id"]) != trigger.goal_id
            or int(row["source_auxiliary_graph_revision"])
            != trigger.source_auxiliary_graph_revision
        ):
            raise AuxiliaryReplanTriggerStoredAuthorityCorrupt(
                "active replan trigger pointer is corrupt"
            )
        if conn.execute(
            "SELECT 1 FROM insession_auxiliary_replan_trigger_applications "
            "WHERE trigger_id=?",
            (trigger.trigger_id,),
        ).fetchone() is not None:
            raise AuxiliaryReplanTriggerStoredAuthorityCorrupt(
                "applied replan trigger remained active"
            )
        return trigger


def get_auxiliary_replan_trigger_application(
    deps: StoreDeps,
    *,
    session_id: str,
    trigger_id: str,
) -> AuxiliaryReplanTriggerApplicationReceipt | None:
    _require_id("session_id", session_id)
    _require_id("trigger_id", trigger_id)
    deps.init_db()
    with deps.connect() as conn:
        if conn.execute(
            "SELECT 1 FROM insession_auxiliary_replan_trigger_applications "
            "WHERE trigger_id=?",
            (trigger_id,),
        ).fetchone() is None:
            return None
        application = _load_application(conn, trigger_id)
        if application.session_id != session_id:
            raise AuxiliaryReplanTriggerStoredAuthorityCorrupt(
                "replan application crossed Session authority"
            )
        return application


def count_auxiliary_replan_triggers(
    deps: StoreDeps,
    *,
    session_id: str,
    task_id: str,
    goal_id: str,
) -> int:
    """统计一个目标的已认证触发器回执，包括活动和已应用状态。"""

    _require_id("session_id", session_id)
    _require_id("task_id", task_id)
    _require_id("goal_id", goal_id)
    deps.init_db()
    with deps.connect() as conn:
        rows = conn.execute(
            "SELECT trigger_id FROM insession_auxiliary_replan_triggers "
            "WHERE session_id=? AND insession_task_id=? AND goal_id=? "
            "ORDER BY source_auxiliary_graph_revision, trigger_id",
            (session_id, task_id, goal_id),
        ).fetchall()
        for row in rows:
            trigger = _load_trigger(conn, str(row["trigger_id"]))
            if (
                trigger.session_id != session_id
                or trigger.task_id != task_id
                or trigger.goal_id != goal_id
            ):
                raise AuxiliaryReplanTriggerStoredAuthorityCorrupt(
                    "replan trigger count crossed goal authority"
                )
        return len(rows)


def _derive_trigger(
    *,
    command: CreateAuxiliaryReplanTriggerCommand,
    details: StoredAuxiliaryGraphDetails,
    settlement: StoredAuxiliarySemanticQuorumSettlement,
) -> AuxiliaryReplanTriggerReceipt:
    if settlement.settlement_id != command.semantic_settlement_id or (
        settlement.settlement_sha256
        != command.expected_semantic_settlement_sha256
    ):
        raise AuxiliaryReplanTriggerStaleAuthority(
            "semantic settlement CAS no longer matches"
        )
    if settlement.host_disposition not in {
        TaskGraphSemanticVerificationDisposition.REVISE,
        TaskGraphSemanticVerificationDisposition.BLOCKED,
    }:
        raise AuxiliaryReplanTriggerPersistenceError(
            "only a revise or blocked semantic settlement can trigger replanning"
        )
    if not settlement.requests or not settlement.results:
        raise AuxiliaryReplanTriggerStoredAuthorityCorrupt(
            "semantic settlement lost reviewer authority"
        )
    request = settlement.requests[0]
    budget = details.budget
    if budget is None:
        raise AuxiliaryReplanTriggerStoredAuthorityCorrupt(
            "current graph lost its budget snapshot"
        )
    if (
        settlement.session_id != details.session_id
        or settlement.task_id != details.task_id
        or settlement.auxiliary_graph_id != details.auxiliary_graph_id
        or settlement.goal_id != details.goal_id
        or settlement.auxiliary_graph_revision
        != details.auxiliary_graph_revision
        or request.auxiliary_graph_structure_sha256 != details.structure_sha256
        or request.goal != details.goal
        or request.budget != budget
        or request.authority_projection.authority_snapshot_id
        != details.authority_snapshot_id
        or request.authority_projection.authority_snapshot_sha256
        != details.authority_snapshot_sha256
        or settlement.frozen_prompt_payload_sha256
        != request.prompt_payload.payload_sha256
    ):
        raise AuxiliaryReplanTriggerStaleAuthority(
            "semantic settlement does not bind the exact current graph authority"
        )
    return AuxiliaryReplanTriggerReceipt.create(
        trigger_id=command.trigger_id,
        create_apply_id=command.apply_id,
        session_id=details.session_id,
        task_id=details.task_id,
        auxiliary_graph_id=details.auxiliary_graph_id,
        goal_id=details.goal_id,
        source_auxiliary_graph_revision=details.auxiliary_graph_revision,
        source_structure_sha256=details.structure_sha256,
        source_authority_snapshot_id=details.authority_snapshot_id,
        source_authority_snapshot_sha256=details.authority_snapshot_sha256,
        semantic_authority_projection_sha256=(
            request.authority_projection.projection_sha256
        ),
        semantic_prompt_payload_sha256=request.prompt_payload.payload_sha256,
        task_graph_proposal_sha256=request.task_graph_proposal_sha256,
        semantic_settlement_id=settlement.settlement_id,
        semantic_settlement_sha256=settlement.settlement_sha256,
        semantic_result_sha256s=tuple(
            result.result_sha256 for result in settlement.results
        ),
        budget_ledger_id=budget.budget_ledger_id,
        budget_state_version=budget.state_version,
        budget_snapshot_sha256=budget.snapshot_sha256,
        semantic_disposition=settlement.host_disposition,
        created_turn_id=command.created_turn_id,
    )


def _require_create_cas(
    *,
    details: StoredAuxiliaryGraphDetails,
    command: CreateAuxiliaryReplanTriggerCommand,
) -> None:
    if (
        details.auxiliary_graph_id != command.auxiliary_graph_id
        or details.goal_id != command.goal_id
        or details.control_state_version != command.expected_control_state_version
        or details.goal_state_version != command.expected_goal_state_version
        or details.revision_state_version
        != command.expected_revision_state_version
        or details.budget_state_version != command.expected_budget_state_version
        or details.auxiliary_graph_revision
        != command.expected_current_auxiliary_graph_revision
        or details.structure_sha256
        != command.expected_current_structure_sha256
        or details.authority_snapshot_id
        != command.expected_current_authority_snapshot_id
        or details.authority_snapshot_sha256
        != command.expected_current_authority_snapshot_sha256
        or details.revision is None
        or details.authority_snapshot is None
        or details.budget is None
    ):
        raise AuxiliaryReplanTriggerStaleAuthority(
            "replan trigger CAS no longer matches current graph/goal/budget"
        )
    if details.goal_status not in {"active", "proposal_ready", "gapped_ready"}:
        raise AuxiliaryReplanTriggerStaleAuthority(
            "current goal cannot enter autonomous replanning"
        )
    if details.revision_status not in {
        "active",
        "proposal_ready",
        "gapped_ready",
    }:
        raise AuxiliaryReplanTriggerStaleAuthority(
            "current graph revision cannot enter autonomous replanning"
        )


def _insert_trigger(
    conn: sqlite3.Connection,
    *,
    trigger: AuxiliaryReplanTriggerReceipt,
    command_json: str,
    command_sha256: str,
    now: str,
) -> None:
    conn.execute(
        "INSERT INTO insession_auxiliary_replan_triggers "
        "(trigger_id, create_apply_id, session_id, insession_task_id, "
        "auxiliary_graph_id, goal_id, source_auxiliary_graph_revision, "
        "source_structure_sha256, source_authority_snapshot_id, "
        "source_authority_snapshot_sha256, "
        "semantic_authority_projection_sha256, "
        "semantic_prompt_payload_sha256, "
        "task_graph_proposal_sha256, semantic_settlement_id, "
        "semantic_settlement_sha256, semantic_result_sha256s_json, "
        "budget_ledger_id, budget_state_version, budget_snapshot_sha256, "
        "evidence_epoch_sha256, semantic_disposition, trigger_reason, "
        "revision_reason, created_turn_id, command_json, command_sha256, "
        "receipt_json, receipt_sha256, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            trigger.trigger_id,
            trigger.create_apply_id,
            trigger.session_id,
            trigger.task_id,
            trigger.auxiliary_graph_id,
            trigger.goal_id,
            trigger.source_auxiliary_graph_revision,
            trigger.source_structure_sha256,
            trigger.source_authority_snapshot_id,
            trigger.source_authority_snapshot_sha256,
            trigger.semantic_authority_projection_sha256,
            trigger.semantic_prompt_payload_sha256,
            trigger.task_graph_proposal_sha256,
            trigger.semantic_settlement_id,
            trigger.semantic_settlement_sha256,
            _canonical_json(trigger.semantic_result_sha256s),
            trigger.budget_ledger_id,
            trigger.budget_state_version,
            trigger.budget_snapshot_sha256,
            trigger.evidence_epoch_sha256,
            trigger.semantic_disposition.value,
            trigger.trigger_reason.value,
            trigger.revision_reason.value,
            trigger.created_turn_id,
            command_json,
            command_sha256,
            _model_json(trigger),
            trigger.receipt_sha256,
            now,
        ),
    )


def _load_trigger(
    conn: sqlite3.Connection,
    trigger_id: str,
) -> AuxiliaryReplanTriggerReceipt:
    row = conn.execute(
        "SELECT * FROM insession_auxiliary_replan_triggers WHERE trigger_id=?",
        (trigger_id,),
    ).fetchone()
    if row is None:
        raise AuxiliaryReplanTriggerStoredAuthorityCorrupt(
            "replan trigger is missing"
        )
    try:
        trigger = AuxiliaryReplanTriggerReceipt.model_validate_json(
            str(row["receipt_json"])
        )
        command = CreateAuxiliaryReplanTriggerCommand.model_validate_json(
            str(row["command_json"])
        )
        result_hashes = tuple(json.loads(str(row["semantic_result_sha256s_json"])))
    except (TypeError, ValueError) as exc:
        raise AuxiliaryReplanTriggerStoredAuthorityCorrupt(
            "stored replan trigger JSON is invalid"
        ) from exc
    mirrored = (
        trigger.trigger_id == trigger_id
        and trigger.create_apply_id == str(row["create_apply_id"])
        and trigger.session_id == str(row["session_id"])
        and trigger.task_id == str(row["insession_task_id"])
        and trigger.auxiliary_graph_id == str(row["auxiliary_graph_id"])
        and trigger.goal_id == str(row["goal_id"])
        and trigger.source_auxiliary_graph_revision
        == int(row["source_auxiliary_graph_revision"])
        and trigger.source_structure_sha256
        == str(row["source_structure_sha256"])
        and trigger.source_authority_snapshot_id
        == str(row["source_authority_snapshot_id"])
        and trigger.source_authority_snapshot_sha256
        == str(row["source_authority_snapshot_sha256"])
        and trigger.semantic_authority_projection_sha256
        == str(row["semantic_authority_projection_sha256"])
        and trigger.semantic_prompt_payload_sha256
        == str(row["semantic_prompt_payload_sha256"])
        and trigger.task_graph_proposal_sha256
        == str(row["task_graph_proposal_sha256"])
        and trigger.semantic_settlement_id == str(row["semantic_settlement_id"])
        and trigger.semantic_settlement_sha256
        == str(row["semantic_settlement_sha256"])
        and trigger.semantic_result_sha256s == result_hashes
        and _canonical_json(trigger.semantic_result_sha256s)
        == str(row["semantic_result_sha256s_json"])
        and trigger.budget_ledger_id == str(row["budget_ledger_id"])
        and trigger.budget_state_version == int(row["budget_state_version"])
        and trigger.budget_snapshot_sha256
        == str(row["budget_snapshot_sha256"])
        and trigger.evidence_epoch_sha256 == str(row["evidence_epoch_sha256"])
        and trigger.semantic_disposition.value == str(row["semantic_disposition"])
        and trigger.trigger_reason.value == str(row["trigger_reason"])
        and trigger.revision_reason.value == str(row["revision_reason"])
        and trigger.created_turn_id == str(row["created_turn_id"])
        and _model_json(trigger) == str(row["receipt_json"])
        and trigger.receipt_sha256 == str(row["receipt_sha256"])
        and command.apply_id == trigger.create_apply_id
        and command.trigger_id == trigger.trigger_id
        and command.session_id == trigger.session_id
        and command.task_id == trigger.task_id
        and command.auxiliary_graph_id == trigger.auxiliary_graph_id
        and command.goal_id == trigger.goal_id
        and command.created_turn_id == trigger.created_turn_id
        and command.semantic_settlement_id == trigger.semantic_settlement_id
        and command.expected_semantic_settlement_sha256
        == trigger.semantic_settlement_sha256
        and _model_json(command) == str(row["command_json"])
        and _sha256_text(str(row["command_json"]))
        == str(row["command_sha256"])
    )
    if not mirrored:
        raise AuxiliaryReplanTriggerStoredAuthorityCorrupt(
            "stored replan trigger row and receipt differ"
        )
    revision = conn.execute(
        "SELECT * FROM insession_auxiliary_graph_revision_snapshots "
        "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=?",
        (trigger.auxiliary_graph_id, trigger.source_auxiliary_graph_revision),
    ).fetchone()
    budget_row = conn.execute(
        "SELECT snapshot_json, snapshot_sha256, session_id, "
        "insession_task_id, auxiliary_graph_id FROM "
        "insession_auxiliary_goal_budget_snapshots WHERE goal_id=? "
        "AND budget_ledger_id=? AND state_version=?",
        (
            trigger.goal_id,
            trigger.budget_ledger_id,
            trigger.budget_state_version,
        ),
    ).fetchone()
    try:
        budget = PlanningEpisodeBudget.model_validate_json(
            str(budget_row["snapshot_json"]) if budget_row is not None else "{}"
        )
    except (TypeError, ValueError) as exc:
        raise AuxiliaryReplanTriggerStoredAuthorityCorrupt(
            "replan trigger budget snapshot JSON is invalid"
        ) from exc
    if (
        revision is None
        or budget_row is None
        or str(revision["session_id"]) != trigger.session_id
        or str(revision["insession_task_id"]) != trigger.task_id
        or str(revision["goal_id"]) != trigger.goal_id
        or str(revision["structure_sha256"]) != trigger.source_structure_sha256
        or str(revision["authority_snapshot_id"])
        != trigger.source_authority_snapshot_id
        or str(revision["authority_snapshot_sha256"])
        != trigger.source_authority_snapshot_sha256
        or str(budget_row["session_id"]) != trigger.session_id
        or str(budget_row["insession_task_id"]) != trigger.task_id
        or str(budget_row["auxiliary_graph_id"]) != trigger.auxiliary_graph_id
        or str(budget_row["snapshot_sha256"]) != trigger.budget_snapshot_sha256
        or budget.snapshot_sha256 != trigger.budget_snapshot_sha256
        or _model_json(budget) != str(budget_row["snapshot_json"])
    ):
        raise AuxiliaryReplanTriggerStoredAuthorityCorrupt(
            "replan trigger source graph or budget authority is corrupt"
        )
    settlement = _load_authenticated_settlement(
        conn,
        trigger.semantic_settlement_id,
    )
    request = settlement.requests[0]
    if (
        settlement.session_id != trigger.session_id
        or settlement.task_id != trigger.task_id
        or settlement.auxiliary_graph_id != trigger.auxiliary_graph_id
        or settlement.goal_id != trigger.goal_id
        or settlement.auxiliary_graph_revision
        != trigger.source_auxiliary_graph_revision
        or settlement.settlement_sha256 != trigger.semantic_settlement_sha256
        or settlement.frozen_prompt_payload_sha256
        != trigger.semantic_prompt_payload_sha256
        or request.task_graph_proposal_sha256
        != trigger.task_graph_proposal_sha256
        or request.auxiliary_graph_structure_sha256
        != trigger.source_structure_sha256
        or request.authority_projection.authority_snapshot_id
        != trigger.source_authority_snapshot_id
        or request.authority_projection.authority_snapshot_sha256
        != trigger.source_authority_snapshot_sha256
        or request.authority_projection.projection_sha256
        != trigger.semantic_authority_projection_sha256
        or request.budget.budget_ledger_id != trigger.budget_ledger_id
        or request.budget.state_version != trigger.budget_state_version
        or request.budget.snapshot_sha256 != trigger.budget_snapshot_sha256
        or tuple(item.result_sha256 for item in settlement.results)
        != trigger.semantic_result_sha256s
        or settlement.host_disposition != trigger.semantic_disposition
    ):
        raise AuxiliaryReplanTriggerStoredAuthorityCorrupt(
            "replan trigger crossed semantic settlement authority"
        )
    return trigger


def _load_application(
    conn: sqlite3.Connection,
    trigger_id: str,
) -> AuxiliaryReplanTriggerApplicationReceipt:
    row = conn.execute(
        "SELECT * FROM insession_auxiliary_replan_trigger_applications "
        "WHERE trigger_id=?",
        (trigger_id,),
    ).fetchone()
    if row is None:
        raise AuxiliaryReplanTriggerStoredAuthorityCorrupt(
            "replan trigger application is missing"
        )
    try:
        application = (
            AuxiliaryReplanTriggerApplicationReceipt.model_validate_json(
                str(row["receipt_json"])
            )
        )
        command = ConsumeAuxiliaryReplanTriggerCommand.model_validate_json(
            str(row["command_json"])
        )
    except (TypeError, ValueError) as exc:
        raise AuxiliaryReplanTriggerStoredAuthorityCorrupt(
            "stored replan application JSON is invalid"
        ) from exc
    trigger = _load_trigger(conn, trigger_id)
    revision = conn.execute(
        "SELECT * FROM insession_auxiliary_graph_revision_snapshots "
        "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=?",
        (
            application.auxiliary_graph_id,
            application.applied_auxiliary_graph_revision,
        ),
    ).fetchone()
    mirrored = (
        application.apply_id == str(row["apply_id"])
        and application.trigger_id == trigger_id
        and application.trigger_receipt_sha256 == trigger.receipt_sha256
        and application.trigger_receipt_sha256
        == str(row["trigger_receipt_sha256"])
        and application.session_id == str(row["session_id"])
        and application.task_id == str(row["insession_task_id"])
        and application.auxiliary_graph_id == str(row["auxiliary_graph_id"])
        and application.goal_id == str(row["goal_id"])
        and application.source_auxiliary_graph_revision
        == int(row["source_auxiliary_graph_revision"])
        and application.applied_auxiliary_graph_revision
        == int(row["applied_auxiliary_graph_revision"])
        and application.applied_structure_sha256
        == str(row["applied_structure_sha256"])
        and application.revision_reason.value == str(row["revision_reason"])
        and application.consumed_turn_id == str(row["consumed_turn_id"])
        and _model_json(application) == str(row["receipt_json"])
        and application.receipt_sha256 == str(row["receipt_sha256"])
        and _model_json(command) == str(row["command_json"])
        and _sha256_text(str(row["command_json"]))
        == str(row["command_sha256"])
        and command.apply_id == application.apply_id
        and command.session_id == application.session_id
        and command.task_id == application.task_id
        and command.trigger_id == application.trigger_id
        and command.expected_goal_id == application.goal_id
        and command.expected_applied_auxiliary_graph_revision
        == application.applied_auxiliary_graph_revision
        and command.expected_applied_structure_sha256
        == application.applied_structure_sha256
        and command.consumed_turn_id == application.consumed_turn_id
        and revision is not None
        and str(revision["session_id"]) == application.session_id
        and str(revision["insession_task_id"]) == application.task_id
        and str(revision["goal_id"]) == application.goal_id
        and int(revision["parent_auxiliary_graph_revision"])
        == application.source_auxiliary_graph_revision
        and str(revision["structure_sha256"])
        == application.applied_structure_sha256
        and str(revision["reason"])
        == AuxiliaryGraphRevisionReason.VERIFICATION_FAILED.value
        and conn.execute(
            "SELECT 1 FROM insession_auxiliary_active_replan_triggers "
            "WHERE trigger_id=?",
            (trigger_id,),
        ).fetchone()
        is None
    )
    if not mirrored:
        raise AuxiliaryReplanTriggerStoredAuthorityCorrupt(
            "stored replan application row and authority differ"
        )
    return application


def _load_authenticated_settlement(
    conn: sqlite3.Connection,
    settlement_id: str,
) -> StoredAuxiliarySemanticQuorumSettlement:
    try:
        return _load_settlement(conn, settlement_id)
    except AuxiliarySemanticVerificationPersistenceError as exc:
        raise AuxiliaryReplanTriggerStoredAuthorityCorrupt(
            "semantic settlement authority is corrupt"
        ) from exc


def _load_current_details(
    conn: sqlite3.Connection,
    session_id: str,
    task_id: str,
) -> StoredAuxiliaryGraphDetails:
    try:
        return _load_auxiliary_graph(conn, session_id, task_id)
    except AuxiliaryGraphPersistenceError as exc:
        raise AuxiliaryReplanTriggerStoredAuthorityCorrupt(
            "current AuxiliaryGraph authority is corrupt"
        ) from exc


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
        raise AuxiliaryReplanTriggerStaleAuthority(
            "replan trigger mutation requires its exact running Turn"
        )


def _validate_command(model_type: type[_Record], value: object, label: str):
    if not isinstance(value, model_type):
        raise TypeError(f"{label} must be {model_type.__name__}")
    try:
        return model_type.model_validate(value.model_dump(mode="json"))
    except (TypeError, ValueError) as exc:
        raise AuxiliaryReplanTriggerPersistenceError(
            f"{label} is not self-authenticating"
        ) from exc


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _model_json(value: BaseModel) -> str:
    return _canonical_json(value.model_dump(mode="json"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_id(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 200
    ):
        raise ValueError(f"{name} must be a canonical identifier")


__all__ = [
    "AuxiliaryReplanTriggerApplicationMutationResult",
    "AuxiliaryReplanTriggerIdentityCollision",
    "AuxiliaryReplanTriggerMutationResult",
    "AuxiliaryReplanTriggerPersistenceError",
    "AuxiliaryReplanTriggerStaleAuthority",
    "AuxiliaryReplanTriggerStoredAuthorityCorrupt",
    "ConsumeAuxiliaryReplanTriggerCommand",
    "count_auxiliary_replan_triggers",
    "CreateAuxiliaryReplanTriggerCommand",
    "consume_auxiliary_replan_trigger",
    "create_auxiliary_replan_trigger",
    "get_active_auxiliary_replan_trigger",
    "get_auxiliary_replan_trigger",
    "get_auxiliary_replan_trigger_application",
]
