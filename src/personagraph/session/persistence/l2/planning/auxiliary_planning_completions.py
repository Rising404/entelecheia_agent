"""一次性 AuxiliaryGraph 初始与正向计划的不可变完成权威。

完成回执嵌入现有 revision 应用的 ``result_json``。该行已经不可变、受重放保护，并与
revision 快照和当前指针 CAS 写入同一事务。本模块认证额外 Architect 与模型权威，不增加
第二个事务或另一可变投影。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Literal, Mapping, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personagraph.l2.task_graph.lane_manifest import (
    InSessionTaskExecutionLaneManifest,
)
from personagraph.l2.task_graph import TaskGraphRevisionTrigger
from personagraph.l2.auxiliary_execution.planning.architect import (
    AuxiliaryGraphArchitectAction,
    AuxiliaryGraphArchitectDecision,
    AuxiliaryGraphArchitectRequest,
)
from .....runtime.model_calls.contracts import RuntimeModelPhysicalOutcome
from personagraph.l2.work_run.contracts import TaskGraphExecutionReplanRequest
from ...deps import StoreDeps
from ...calls.runtime_model_calls import _load_logical_call


_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class AuxiliaryInitialPlanningPersistenceError(RuntimeError):
    """初始规划完成权威以关闭方式失败。"""

    code = "auxiliary_v2_initial_planning_persistence_rejected"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AuxiliaryInitialPlanningCompletionBinding(_Contract):
    """附加到 Architect revision 提交的精确 Host 命令。"""

    schema_version: Literal[
        "auxiliary-v2-initial-planning-completion-binding-v1"
    ] = "auxiliary-v2-initial-planning-completion-binding-v1"
    completion_receipt_id: str = Field(pattern=_ID_PATTERN)
    session_id: str = Field(pattern=_ID_PATTERN)
    turn_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_id: str = Field(pattern=_ID_PATTERN)
    goal_id: str = Field(pattern=_ID_PATTERN)
    bootstrap_auxiliary_graph_revision: Literal[1] = 1
    bootstrap_structure_sha256: str = Field(pattern=_SHA256_PATTERN)
    revision_apply_id: str = Field(pattern=_ID_PATTERN)
    architect_request: AuxiliaryGraphArchitectRequest
    architect_decision: AuxiliaryGraphArchitectDecision
    runtime_logical_request_binding_sha256: str = Field(
        pattern=_SHA256_PATTERN
    )
    command_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_binding(self) -> Self:
        request = self.architect_request
        decision = self.architect_decision
        current = request.prompt_payload.current_revision
        if (
            request.goal.session_id != self.session_id
            or request.goal.task_id != self.task_id
            or request.goal.auxiliary_graph_id != self.auxiliary_graph_id
            or request.goal.goal_id != self.goal_id
        ):
            raise ValueError("Architect request crossed planning-goal ownership")
        if (
            current is None
            or current.auxiliary_graph_revision != 1
            or current.source_structure_sha256
            != self.bootstrap_structure_sha256
        ):
            raise ValueError("Architect request is not bound to the bootstrap")
        if (
            decision.architect_request_id != request.architect_request_id
            or decision.request_binding_sha256 != request.binding_sha256
            or decision.logical_call_id != request.logical_call_id
            or decision.action is not AuxiliaryGraphArchitectAction.REVISE_REVISION
            or decision.proposal.expected_current_auxiliary_graph_revision != 1
            or decision.proposal.structure is None
            or decision.proposal.requested_user_question is not None
        ):
            raise ValueError("Architect decision is not an exact bootstrap revision")
        expected_receipt_id = _completion_receipt_id(
            self.model_dump(
                mode="json",
                exclude={"completion_receipt_id", "command_sha256"},
            )
        )
        if self.completion_receipt_id != expected_receipt_id:
            raise ValueError("initial-planning completion receipt ID is invalid")
        expected_hash = _sha256_value(
            self.model_dump(mode="json", exclude={"command_sha256"})
        )
        if self.command_sha256 != expected_hash:
            raise ValueError("initial-planning completion command hash is invalid")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        normalized = dict(values)
        request = normalized.get("architect_request")
        decision = normalized.get("architect_decision")
        if not isinstance(request, AuxiliaryGraphArchitectRequest):
            request = AuxiliaryGraphArchitectRequest.model_validate(request)
        if not isinstance(decision, AuxiliaryGraphArchitectDecision):
            decision = AuxiliaryGraphArchitectDecision.model_validate(decision)
        normalized["architect_request"] = request
        normalized["architect_decision"] = decision
        normalized.setdefault("bootstrap_auxiliary_graph_revision", 1)
        provisional = cls.model_construct(
            schema_version=(
                "auxiliary-v2-initial-planning-completion-binding-v1"
            ),
            completion_receipt_id="placeholder",
            command_sha256="0" * 64,
            **normalized,
        )
        identity_payload = provisional.model_dump(
            mode="json",
            exclude={"completion_receipt_id", "command_sha256"},
        )
        normalized["completion_receipt_id"] = _completion_receipt_id(
            identity_payload
        )
        provisional = cls.model_construct(
            schema_version=(
                "auxiliary-v2-initial-planning-completion-binding-v1"
            ),
            command_sha256="0" * 64,
            **normalized,
        )
        normalized["command_sha256"] = _sha256_value(
            provisional.model_dump(mode="json", exclude={"command_sha256"})
        )
        return cls.model_validate(normalized)


class StoredAuxiliaryInitialPlanningCompletion(_Contract):
    """嵌套在一个 revision 应用回执中的自认证完成项。"""

    schema_version: Literal[
        "stored-auxiliary-v2-initial-planning-completion-v1"
    ] = "stored-auxiliary-v2-initial-planning-completion-v1"
    outcome: Literal["planned"] = "planned"
    binding: AuxiliaryInitialPlanningCompletionBinding
    bootstrap_revision_apply_id: str = Field(pattern=_ID_PATTERN)
    model_attempt_count: int = Field(ge=1, le=32)
    physical_attempt_id: str = Field(pattern=_ID_PATTERN)
    physical_ordinal: int = Field(ge=1, le=32)
    model_settlement_id: str = Field(pattern=_ID_PATTERN)
    model_settlement_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    committed_auxiliary_graph_revision: Literal[2] = 2
    committed_control_state_version: int = Field(ge=2)
    committed_goal_state_version: int = Field(ge=1)
    committed_revision_state_version: int = Field(ge=1)
    committed_budget_state_version: int = Field(ge=2)
    committed_budget_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    committed_authority_snapshot_id: str = Field(pattern=_ID_PATTERN)
    committed_authority_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    committed_structure_sha256: str = Field(pattern=_SHA256_PATTERN)
    created_at: str = Field(min_length=1)
    receipt_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_receipt(self) -> Self:
        if self.physical_ordinal > self.model_attempt_count:
            raise ValueError("settled model ordinal exceeds the attempt count")
        expected = _sha256_value(
            self.model_dump(mode="json", exclude={"receipt_sha256"})
        )
        if self.receipt_sha256 != expected:
            raise ValueError("initial-planning completion receipt hash is invalid")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        normalized = dict(values)
        binding = normalized.get("binding")
        if not isinstance(binding, AuxiliaryInitialPlanningCompletionBinding):
            binding = AuxiliaryInitialPlanningCompletionBinding.model_validate(
                binding
            )
        normalized["binding"] = binding
        normalized.setdefault("outcome", "planned")
        normalized.setdefault("committed_auxiliary_graph_revision", 2)
        provisional = cls.model_construct(
            schema_version=(
                "stored-auxiliary-v2-initial-planning-completion-v1"
            ),
            receipt_sha256="0" * 64,
            **normalized,
        )
        normalized["receipt_sha256"] = _sha256_value(
            provisional.model_dump(mode="json", exclude={"receipt_sha256"})
        )
        return cls.model_validate(normalized)

    @property
    def completion_receipt_id(self) -> str:
        return self.binding.completion_receipt_id

    @property
    def architect_decision(self) -> AuxiliaryGraphArchitectDecision:
        return self.binding.architect_decision


def seal_auxiliary_initial_planning_completion(
    conn: sqlite3.Connection,
    *,
    binding: AuxiliaryInitialPlanningCompletionBinding,
    session_id: str,
    turn_id: str,
    task_id: str,
    auxiliary_graph_id: str,
    goal_id: str,
    expected_current_auxiliary_graph_revision: int | None,
    revision_apply_id: str,
    proposal_payload: Mapping[str, Any],
    committed_result: Mapping[str, Any],
    committed_budget_snapshot_sha256: str,
    created_at: str,
) -> StoredAuxiliaryInitialPlanningCompletion:
    """在 revision 事务内认证并物化一个回执。"""

    if not isinstance(binding, AuxiliaryInitialPlanningCompletionBinding):
        raise TypeError(
            "binding must be AuxiliaryInitialPlanningCompletionBinding"
        )
    admitted = _fresh_binding(binding)
    if (
        admitted.session_id != session_id
        or admitted.turn_id != turn_id
        or admitted.task_id != task_id
        or admitted.auxiliary_graph_id != auxiliary_graph_id
        or admitted.goal_id != goal_id
        or admitted.revision_apply_id != revision_apply_id
        or expected_current_auxiliary_graph_revision != 1
    ):
        raise AuxiliaryInitialPlanningPersistenceError(
            "completion binding crossed the revision commit authority"
        )
    _require_lossless_revision_proposal(admitted, proposal_payload)
    bootstrap_apply_id = _require_bootstrap_revision(conn, admitted)
    model = _require_succeeded_architect_model_call(conn, admitted)
    result = dict(committed_result)
    if (
        result.get("committed_auxiliary_graph_revision") != 2
        or result.get("auxiliary_graph_id") != auxiliary_graph_id
        or result.get("goal_id") != goal_id
    ):
        raise AuxiliaryInitialPlanningPersistenceError(
            "completion may only settle the exact bootstrap-to-revision-two commit"
        )
    receipt = StoredAuxiliaryInitialPlanningCompletion.create(
        binding=admitted,
        bootstrap_revision_apply_id=bootstrap_apply_id,
        model_attempt_count=model["attempt_count"],
        physical_attempt_id=model["physical_attempt_id"],
        physical_ordinal=model["physical_ordinal"],
        model_settlement_id=model["settlement_id"],
        model_settlement_receipt_sha256=model["settlement_receipt_sha256"],
        committed_control_state_version=result["control_state_version"],
        committed_goal_state_version=result["goal_state_version"],
        committed_revision_state_version=result["revision_state_version"],
        committed_budget_state_version=result["budget_state_version"],
        committed_budget_snapshot_sha256=committed_budget_snapshot_sha256,
        committed_authority_snapshot_id=result["authority_snapshot_id"],
        committed_authority_snapshot_sha256=result[
            "authority_snapshot_sha256"
        ],
        committed_structure_sha256=result["structure_sha256"],
        created_at=created_at,
    )
    _require_revision_snapshots(conn, receipt)
    return receipt


TaskGraphPositivePlanningAuthority = (
    TaskGraphRevisionTrigger | TaskGraphExecutionReplanRequest
)


class AuxiliaryPositivePlanningCompletionBinding(_Contract):
    """一个目标作用域 R -> R+1 的精确正 base Architect 权威。"""

    schema_version: Literal[
        "auxiliary-v2-positive-planning-completion-binding-v1"
    ] = "auxiliary-v2-positive-planning-completion-binding-v1"
    completion_receipt_id: str = Field(pattern=_ID_PATTERN)
    session_id: str = Field(pattern=_ID_PATTERN)
    turn_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_id: str = Field(pattern=_ID_PATTERN)
    goal_id: str = Field(pattern=_ID_PATTERN)
    bootstrap_auxiliary_graph_revision: int = Field(ge=1)
    bootstrap_structure_sha256: str = Field(pattern=_SHA256_PATTERN)
    revision_apply_id: str = Field(pattern=_ID_PATTERN)
    revision_authority: TaskGraphPositivePlanningAuthority
    architect_request: AuxiliaryGraphArchitectRequest
    architect_decision: AuxiliaryGraphArchitectDecision
    runtime_logical_request_binding_sha256: str = Field(pattern=_SHA256_PATTERN)
    command_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_binding(self) -> Self:
        request = self.architect_request
        decision = self.architect_decision
        authority = self.revision_authority
        current = request.prompt_payload.current_revision
        if isinstance(authority, TaskGraphRevisionTrigger):
            authority_id = authority.trigger_id
            authority_sha256 = authority.trigger_sha256
        else:
            authority_id = authority.request_id
            authority_sha256 = authority.request_sha256
        identity = _positive_identity(authority_id, authority_sha256)
        if (
            self.goal_id != f"auxgoalv2_positive_{identity}"
            or self.revision_apply_id != f"auxv2positive_{identity}:architect"
            or request.logical_call_id != f"auxv2positive_{identity}:model"
            or request.architect_request_id
            != f"auxv2positive_{identity}:request"
        ):
            raise ValueError("positive completion identity crossed authority")
        if (
            authority.session_id != self.session_id
            or authority.task_id != self.task_id
            or request.goal.session_id != self.session_id
            or request.goal.task_id != self.task_id
            or request.goal.auxiliary_graph_id != self.auxiliary_graph_id
            or request.goal.goal_id != self.goal_id
            or request.goal.base_task_graph_revision
            != authority.base_graph_revision
            or request.goal.target_task_graph_revision
            != authority.target_graph_revision
            or request.prompt_payload.goal.objective
            != authority.revision_objective
            or request.prompt_payload.task_graph_revision_trigger != authority
        ):
            raise ValueError("positive Architect request crossed revision authority")
        if (
            current is None
            or current.auxiliary_graph_revision
            != self.bootstrap_auxiliary_graph_revision
            or current.source_structure_sha256
            != self.bootstrap_structure_sha256
            or current.base_task_graph_revision != authority.base_graph_revision
        ):
            raise ValueError("positive Architect request crossed its bootstrap")
        if (
            decision.architect_request_id != request.architect_request_id
            or decision.request_binding_sha256 != request.binding_sha256
            or decision.logical_call_id != request.logical_call_id
            or decision.action is not AuxiliaryGraphArchitectAction.REVISE_REVISION
            or decision.proposal.expected_current_auxiliary_graph_revision
            != self.bootstrap_auxiliary_graph_revision
            or decision.proposal.revision_reason is None
            or decision.proposal.revision_reason.value != "verification_failed"
            or decision.proposal.structure is None
            or decision.proposal.requested_user_question is not None
        ):
            raise ValueError("positive Architect decision is not an exact revision")
        expected_receipt_id = _positive_completion_receipt_id(
            self.model_dump(
                mode="json",
                exclude={"completion_receipt_id", "command_sha256"},
            )
        )
        if self.completion_receipt_id != expected_receipt_id:
            raise ValueError("positive completion receipt ID is invalid")
        expected_hash = _sha256_value(
            self.model_dump(mode="json", exclude={"command_sha256"})
        )
        if self.command_sha256 != expected_hash:
            raise ValueError("positive completion command hash is invalid")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        normalized = dict(values)
        authority = normalized.get("revision_authority")
        if isinstance(authority, TaskGraphRevisionTrigger):
            authority = TaskGraphRevisionTrigger.model_validate_json(
                authority.model_dump_json()
            )
        elif isinstance(authority, TaskGraphExecutionReplanRequest):
            authority = TaskGraphExecutionReplanRequest.model_validate_json(
                authority.model_dump_json()
            )
        else:
            raise TypeError("revision_authority has an unsupported type")
        normalized["revision_authority"] = authority
        request = normalized.get("architect_request")
        decision = normalized.get("architect_decision")
        if not isinstance(request, AuxiliaryGraphArchitectRequest):
            request = AuxiliaryGraphArchitectRequest.model_validate(request)
        if not isinstance(decision, AuxiliaryGraphArchitectDecision):
            decision = AuxiliaryGraphArchitectDecision.model_validate(decision)
        normalized["architect_request"] = request
        normalized["architect_decision"] = decision
        provisional = cls.model_construct(
            schema_version=(
                "auxiliary-v2-positive-planning-completion-binding-v1"
            ),
            completion_receipt_id="placeholder",
            command_sha256="0" * 64,
            **normalized,
        )
        normalized["completion_receipt_id"] = _positive_completion_receipt_id(
            provisional.model_dump(
                mode="json",
                exclude={"completion_receipt_id", "command_sha256"},
            )
        )
        provisional = cls.model_construct(
            schema_version=(
                "auxiliary-v2-positive-planning-completion-binding-v1"
            ),
            command_sha256="0" * 64,
            **normalized,
        )
        normalized["command_sha256"] = _sha256_value(
            provisional.model_dump(mode="json", exclude={"command_sha256"})
        )
        return cls.model_validate(normalized)


class StoredAuxiliaryPositivePlanningCompletion(_Contract):
    schema_version: Literal[
        "stored-auxiliary-v2-positive-planning-completion-v1"
    ] = "stored-auxiliary-v2-positive-planning-completion-v1"
    outcome: Literal["planned"] = "planned"
    binding: AuxiliaryPositivePlanningCompletionBinding
    bootstrap_revision_apply_id: str = Field(pattern=_ID_PATTERN)
    model_attempt_count: int = Field(ge=1, le=32)
    physical_attempt_id: str = Field(pattern=_ID_PATTERN)
    physical_ordinal: int = Field(ge=1, le=32)
    model_settlement_id: str = Field(pattern=_ID_PATTERN)
    model_settlement_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    committed_auxiliary_graph_revision: int = Field(ge=2)
    committed_control_state_version: int = Field(ge=2)
    committed_goal_state_version: int = Field(ge=1)
    committed_revision_state_version: int = Field(ge=1)
    committed_budget_state_version: int = Field(ge=2)
    committed_budget_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    committed_authority_snapshot_id: str = Field(pattern=_ID_PATTERN)
    committed_authority_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    committed_structure_sha256: str = Field(pattern=_SHA256_PATTERN)
    created_at: str = Field(min_length=1)
    receipt_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_receipt(self) -> Self:
        if (
            self.physical_ordinal > self.model_attempt_count
            or self.committed_auxiliary_graph_revision
            != self.binding.bootstrap_auxiliary_graph_revision + 1
        ):
            raise ValueError("positive completion revision/attempt is invalid")
        expected = _sha256_value(
            self.model_dump(mode="json", exclude={"receipt_sha256"})
        )
        if self.receipt_sha256 != expected:
            raise ValueError("positive completion receipt hash is invalid")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        normalized = dict(values)
        binding = normalized.get("binding")
        if not isinstance(binding, AuxiliaryPositivePlanningCompletionBinding):
            binding = AuxiliaryPositivePlanningCompletionBinding.model_validate(
                binding
            )
        normalized["binding"] = binding
        normalized.setdefault("outcome", "planned")
        provisional = cls.model_construct(
            schema_version=(
                "stored-auxiliary-v2-positive-planning-completion-v1"
            ),
            receipt_sha256="0" * 64,
            **normalized,
        )
        normalized["receipt_sha256"] = _sha256_value(
            provisional.model_dump(mode="json", exclude={"receipt_sha256"})
        )
        return cls.model_validate(normalized)


def seal_auxiliary_positive_planning_completion(
    conn: sqlite3.Connection,
    *,
    binding: AuxiliaryPositivePlanningCompletionBinding,
    session_id: str,
    turn_id: str,
    task_id: str,
    auxiliary_graph_id: str,
    goal_id: str,
    expected_current_auxiliary_graph_revision: int | None,
    revision_apply_id: str,
    proposal_payload: Mapping[str, Any],
    committed_result: Mapping[str, Any],
    committed_budget_snapshot_sha256: str,
    created_at: str,
) -> StoredAuxiliaryPositivePlanningCompletion:
    admitted = _fresh_positive_binding(binding)
    if (
        admitted.session_id != session_id
        or admitted.turn_id != turn_id
        or admitted.task_id != task_id
        or admitted.auxiliary_graph_id != auxiliary_graph_id
        or admitted.goal_id != goal_id
        or admitted.revision_apply_id != revision_apply_id
        or expected_current_auxiliary_graph_revision
        != admitted.bootstrap_auxiliary_graph_revision
    ):
        raise AuxiliaryInitialPlanningPersistenceError(
            "positive completion crossed revision commit authority"
        )
    _require_lossless_revision_proposal(admitted, proposal_payload)
    bootstrap_apply_id = _require_positive_bootstrap_revision(conn, admitted)
    model = _require_succeeded_architect_model_call(conn, admitted)
    result = dict(committed_result)
    if (
        result.get("committed_auxiliary_graph_revision")
        != admitted.bootstrap_auxiliary_graph_revision + 1
        or result.get("auxiliary_graph_id") != auxiliary_graph_id
        or result.get("goal_id") != goal_id
    ):
        raise AuxiliaryInitialPlanningPersistenceError(
            "positive completion settled another graph revision"
        )
    receipt = StoredAuxiliaryPositivePlanningCompletion.create(
        binding=admitted,
        bootstrap_revision_apply_id=bootstrap_apply_id,
        model_attempt_count=model["attempt_count"],
        physical_attempt_id=model["physical_attempt_id"],
        physical_ordinal=model["physical_ordinal"],
        model_settlement_id=model["settlement_id"],
        model_settlement_receipt_sha256=model["settlement_receipt_sha256"],
        committed_auxiliary_graph_revision=result[
            "committed_auxiliary_graph_revision"
        ],
        committed_control_state_version=result["control_state_version"],
        committed_goal_state_version=result["goal_state_version"],
        committed_revision_state_version=result["revision_state_version"],
        committed_budget_state_version=result["budget_state_version"],
        committed_budget_snapshot_sha256=committed_budget_snapshot_sha256,
        committed_authority_snapshot_id=result["authority_snapshot_id"],
        committed_authority_snapshot_sha256=result[
            "authority_snapshot_sha256"
        ],
        committed_structure_sha256=result["structure_sha256"],
        created_at=created_at,
    )
    _require_positive_revision_snapshots(conn, receipt)
    return receipt


def get_auxiliary_initial_planning_completion(
    deps: StoreDeps,
    *,
    session_id: str,
    task_id: str,
) -> StoredAuxiliaryInitialPlanningCompletion | None:
    """加载唯一嵌入完成项，并重新校验每条权威链接。"""

    _require_identifier("session_id", session_id)
    _require_identifier("task_id", task_id)
    deps.init_db()
    with deps.connect() as conn:
        return load_authenticated_auxiliary_initial_planning_completion(
            conn,
            session_id=session_id,
            task_id=task_id,
        )


def load_authenticated_auxiliary_initial_planning_completion(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    task_id: str,
) -> StoredAuxiliaryInitialPlanningCompletion | None:
    """精确密封 Architect 完成项的连接作用域加载器。"""

    _require_identifier("session_id", session_id)
    _require_identifier("task_id", task_id)
    rows = conn.execute(
        "SELECT * FROM insession_auxiliary_graph_revision_apply_receipts_v2 "
        "WHERE session_id=? AND insession_task_id=? "
        "AND committed_auxiliary_graph_revision=2 "
        "ORDER BY created_at, apply_id",
        (session_id, task_id),
    ).fetchall()
    found: list[
        tuple[StoredAuxiliaryInitialPlanningCompletion, sqlite3.Row]
    ] = []
    for row in rows:
        result_json = str(row["result_json"])
        if _sha256_text(result_json) != str(row["result_sha256"]):
            raise AuxiliaryInitialPlanningPersistenceError(
                "revision-apply result receipt is corrupt"
            )
        try:
            raw_result = json.loads(result_json)
        except (TypeError, ValueError) as exc:
            raise AuxiliaryInitialPlanningPersistenceError(
                "revision-apply result is not valid JSON"
            ) from exc
        if not isinstance(raw_result, dict):
            raise AuxiliaryInitialPlanningPersistenceError(
                "revision-apply result is not an object"
            )
        raw_completion = raw_result.get("initial_planning_completion")
        if raw_completion is None:
            continue
        try:
            receipt = StoredAuxiliaryInitialPlanningCompletion.model_validate(
                raw_completion
            )
        except Exception as exc:
            raise AuxiliaryInitialPlanningPersistenceError(
                "stored initial-planning completion is corrupt"
            ) from exc
        _validate_stored_completion(
            conn,
            receipt=receipt,
            apply_row=row,
            raw_result=raw_result,
        )
        found.append((receipt, row))
    if len(found) > 1:
        raise AuxiliaryInitialPlanningPersistenceError(
            "Task owns multiple initial-planning completion receipts"
        )
    if not found:
        return None
    receipt = found[0][0]
    control = conn.execute(
        "SELECT auxiliary_graph_id, current_auxiliary_graph_revision "
        "FROM insession_auxiliary_graph_v2_containers "
        "WHERE session_id=? AND insession_task_id=?",
        (session_id, task_id),
    ).fetchone()
    if (
        control is None
        or str(control["auxiliary_graph_id"])
        != receipt.binding.auxiliary_graph_id
        or int(control["current_auxiliary_graph_revision"])
        < receipt.committed_auxiliary_graph_revision
    ):
        raise AuxiliaryInitialPlanningPersistenceError(
            "completion receipt is detached from the current graph lineage"
        )
    return receipt


def load_authenticated_auxiliary_positive_planning_completion(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    task_id: str,
    auxiliary_graph_id: str,
    goal_id: str,
    revision_apply_id: str,
) -> StoredAuxiliaryPositivePlanningCompletion | None:
    """加载一个精确目标作用域正向 Architect 完成项。"""

    for name, value in (
        ("session_id", session_id),
        ("task_id", task_id),
        ("auxiliary_graph_id", auxiliary_graph_id),
        ("goal_id", goal_id),
        ("revision_apply_id", revision_apply_id),
    ):
        _require_identifier(name, value)
    row = conn.execute(
        "SELECT * FROM insession_auxiliary_graph_revision_apply_receipts_v2 "
        "WHERE apply_id=?",
        (revision_apply_id,),
    ).fetchone()
    if row is None:
        return None
    if (
        str(row["operation"]) != "append_goal_revision"
        or str(row["session_id"]) != session_id
        or str(row["insession_task_id"]) != task_id
        or str(row["auxiliary_graph_id"]) != auxiliary_graph_id
        or str(row["goal_id"]) != goal_id
    ):
        raise AuxiliaryInitialPlanningPersistenceError(
            "positive revision apply crossed exact goal authority"
        )
    result_json = str(row["result_json"])
    if _sha256_text(result_json) != str(row["result_sha256"]):
        raise AuxiliaryInitialPlanningPersistenceError(
            "positive revision-apply result receipt is corrupt"
        )
    try:
        raw_result = json.loads(result_json)
    except (TypeError, ValueError) as exc:
        raise AuxiliaryInitialPlanningPersistenceError(
            "positive revision-apply result is not valid JSON"
        ) from exc
    if not isinstance(raw_result, dict):
        raise AuxiliaryInitialPlanningPersistenceError(
            "positive revision-apply result is not an object"
        )
    if "positive_planning_completion" not in raw_result:
        return None
    raw_completion = raw_result["positive_planning_completion"]
    if raw_completion is None:
        raise AuxiliaryInitialPlanningPersistenceError(
            "stored positive-planning completion is explicitly null"
        )
    try:
        receipt = StoredAuxiliaryPositivePlanningCompletion.model_validate(
            raw_completion
        )
    except Exception as exc:
        raise AuxiliaryInitialPlanningPersistenceError(
            "stored positive-planning completion is corrupt"
        ) from exc
    _validate_stored_positive_completion(
        conn,
        receipt=receipt,
        apply_row=row,
        raw_result=raw_result,
    )
    control = conn.execute(
        "SELECT auxiliary_graph_id, current_goal_id, "
        "current_auxiliary_graph_revision FROM "
        "insession_auxiliary_graph_v2_containers "
        "WHERE session_id=? AND insession_task_id=?",
        (session_id, task_id),
    ).fetchone()
    if (
        control is None
        or str(control["auxiliary_graph_id"]) != auxiliary_graph_id
        or str(control["current_goal_id"]) != goal_id
        or int(control["current_auxiliary_graph_revision"])
        < receipt.committed_auxiliary_graph_revision
    ):
        raise AuxiliaryInitialPlanningPersistenceError(
            "positive completion detached from current goal lineage"
        )
    _require_authenticated_positive_descendant_chain(
        conn,
        receipt=receipt,
        current_auxiliary_graph_revision=int(
            control["current_auxiliary_graph_revision"]
        ),
    )
    return receipt


def require_matching_auxiliary_positive_planning_completion(
    conn: sqlite3.Connection,
    *,
    receipt: StoredAuxiliaryPositivePlanningCompletion,
    binding: AuxiliaryPositivePlanningCompletionBinding,
    apply_row: sqlite3.Row,
    raw_result: Mapping[str, Any],
) -> None:
    if receipt.binding != _fresh_positive_binding(binding):
        raise AuxiliaryInitialPlanningPersistenceError(
            "replayed positive completion command changed"
        )
    _validate_stored_positive_completion(
        conn,
        receipt=receipt,
        apply_row=apply_row,
        raw_result=raw_result,
    )


def _validate_stored_positive_completion(
    conn: sqlite3.Connection,
    *,
    receipt: StoredAuxiliaryPositivePlanningCompletion,
    apply_row: sqlite3.Row,
    raw_result: Mapping[str, Any],
) -> None:
    binding = receipt.binding
    expected_result = {
        "auxiliary_graph_id": binding.auxiliary_graph_id,
        "goal_id": binding.goal_id,
        "committed_auxiliary_graph_revision": (
            receipt.committed_auxiliary_graph_revision
        ),
        "control_state_version": receipt.committed_control_state_version,
        "goal_state_version": receipt.committed_goal_state_version,
        "revision_state_version": receipt.committed_revision_state_version,
        "budget_state_version": receipt.committed_budget_state_version,
        "authority_snapshot_id": receipt.committed_authority_snapshot_id,
        "authority_snapshot_sha256": (
            receipt.committed_authority_snapshot_sha256
        ),
        "structure_sha256": receipt.committed_structure_sha256,
    }
    if (
        str(apply_row["operation"]) != "append_goal_revision"
        or str(apply_row["apply_id"]) != binding.revision_apply_id
        or str(apply_row["session_id"]) != binding.session_id
        or str(apply_row["insession_task_id"]) != binding.task_id
        or str(apply_row["auxiliary_graph_id"]) != binding.auxiliary_graph_id
        or str(apply_row["goal_id"]) != binding.goal_id
        or str(apply_row["invocation_turn_id"]) != binding.turn_id
        or int(apply_row["expected_current_auxiliary_graph_revision"])
        != binding.bootstrap_auxiliary_graph_revision
        or int(apply_row["committed_auxiliary_graph_revision"])
        != receipt.committed_auxiliary_graph_revision
        or int(apply_row["committed_control_state_version"])
        != receipt.committed_control_state_version
        or int(apply_row["committed_goal_state_version"])
        != receipt.committed_goal_state_version
        or int(apply_row["committed_budget_state_version"])
        != receipt.committed_budget_state_version
        or str(apply_row["committed_budget_snapshot_sha256"])
        != receipt.committed_budget_snapshot_sha256
        or any(raw_result.get(key) != value for key, value in expected_result.items())
    ):
        raise AuxiliaryInitialPlanningPersistenceError(
            "positive completion differs from revision-apply projection"
        )
    _require_positive_bootstrap_revision(
        conn,
        binding,
        expected_apply_id=receipt.bootstrap_revision_apply_id,
    )
    model = _require_succeeded_architect_model_call(conn, binding)
    if (
        model["attempt_count"] != receipt.model_attempt_count
        or model["physical_attempt_id"] != receipt.physical_attempt_id
        or model["physical_ordinal"] != receipt.physical_ordinal
        or model["settlement_id"] != receipt.model_settlement_id
        or model["settlement_receipt_sha256"]
        != receipt.model_settlement_receipt_sha256
    ):
        raise AuxiliaryInitialPlanningPersistenceError(
            "positive completion differs from durable model settlement"
        )
    _require_positive_revision_snapshots(conn, receipt)


def require_matching_auxiliary_initial_planning_completion(
    conn: sqlite3.Connection,
    *,
    receipt: StoredAuxiliaryInitialPlanningCompletion,
    binding: AuxiliaryInitialPlanningCompletionBinding,
    apply_row: sqlite3.Row,
    raw_result: Mapping[str, Any],
) -> None:
    """校验提供完成命令的精确应用重放。"""

    if receipt.binding != _fresh_binding(binding):
        raise AuxiliaryInitialPlanningPersistenceError(
            "replayed completion command differs from its immutable receipt"
        )
    _validate_stored_completion(
        conn,
        receipt=receipt,
        apply_row=apply_row,
        raw_result=raw_result,
    )


def _validate_stored_completion(
    conn: sqlite3.Connection,
    *,
    receipt: StoredAuxiliaryInitialPlanningCompletion,
    apply_row: sqlite3.Row,
    raw_result: Mapping[str, Any],
) -> None:
    binding = receipt.binding
    expected_result = {
        "auxiliary_graph_id": binding.auxiliary_graph_id,
        "goal_id": binding.goal_id,
        "committed_auxiliary_graph_revision": 2,
        "control_state_version": receipt.committed_control_state_version,
        "goal_state_version": receipt.committed_goal_state_version,
        "revision_state_version": receipt.committed_revision_state_version,
        "budget_state_version": receipt.committed_budget_state_version,
        "authority_snapshot_id": receipt.committed_authority_snapshot_id,
        "authority_snapshot_sha256": (
            receipt.committed_authority_snapshot_sha256
        ),
        "structure_sha256": receipt.committed_structure_sha256,
    }
    if (
        str(apply_row["apply_id"]) != binding.revision_apply_id
        or str(apply_row["session_id"]) != binding.session_id
        or str(apply_row["insession_task_id"]) != binding.task_id
        or str(apply_row["auxiliary_graph_id"]) != binding.auxiliary_graph_id
        or str(apply_row["goal_id"]) != binding.goal_id
        or str(apply_row["invocation_turn_id"]) != binding.turn_id
        or int(apply_row["committed_auxiliary_graph_revision"]) != 2
        or int(apply_row["committed_control_state_version"])
        != receipt.committed_control_state_version
        or int(apply_row["committed_goal_state_version"])
        != receipt.committed_goal_state_version
        or int(apply_row["committed_budget_state_version"])
        != receipt.committed_budget_state_version
        or str(apply_row["committed_budget_snapshot_sha256"])
        != receipt.committed_budget_snapshot_sha256
        or any(raw_result.get(key) != value for key, value in expected_result.items())
    ):
        raise AuxiliaryInitialPlanningPersistenceError(
            "completion receipt differs from its revision-apply projection"
        )
    _require_bootstrap_revision(conn, binding, expected_apply_id=receipt.bootstrap_revision_apply_id)
    model = _require_succeeded_architect_model_call(conn, binding)
    if (
        model["attempt_count"] != receipt.model_attempt_count
        or model["physical_attempt_id"] != receipt.physical_attempt_id
        or model["physical_ordinal"] != receipt.physical_ordinal
        or model["settlement_id"] != receipt.model_settlement_id
        or model["settlement_receipt_sha256"]
        != receipt.model_settlement_receipt_sha256
    ):
        raise AuxiliaryInitialPlanningPersistenceError(
            "completion receipt differs from the durable model settlement"
        )
    _require_revision_snapshots(conn, receipt)


def _require_succeeded_architect_model_call(
    conn: sqlite3.Connection,
    binding: AuxiliaryInitialPlanningCompletionBinding,
) -> dict[str, Any]:
    logical = _load_logical_call(conn, binding.architect_request.logical_call_id)
    if logical is None:
        raise AuxiliaryInitialPlanningPersistenceError(
            "Architect logical model call is missing"
        )
    request = logical.request
    expected_request = binding.architect_request
    try:
        stored_architect_request = AuxiliaryGraphArchitectRequest.model_validate_json(
            request.request_json
        )
    except Exception as exc:
        raise AuxiliaryInitialPlanningPersistenceError(
            "Architect logical request payload is corrupt"
        ) from exc
    if (
        stored_architect_request != expected_request
        or request.logical_call_id != expected_request.logical_call_id
        or request.session_id != binding.session_id
        or request.task_id != binding.task_id
        or request.auxiliary_graph_id != binding.auxiliary_graph_id
        or request.goal_id != binding.goal_id
        or request.request_contract != "auxiliary-graph-architect-request-v1"
        or request.typed_result_contract != "auxiliary-graph-revision-proposal-v2"
        or request.state_guard_sha256 != expected_request.binding_sha256
        or request.binding_sha256
        != binding.runtime_logical_request_binding_sha256
    ):
        raise AuxiliaryInitialPlanningPersistenceError(
            "Architect logical call crossed immutable request authority"
        )
    _require_authenticated_model_execution_history(
        conn,
        session_id=binding.session_id,
        task_id=binding.task_id,
        commit_turn_id=binding.turn_id,
        logical=logical,
    )
    if not logical.physical_attempts:
        raise AuxiliaryInitialPlanningPersistenceError(
            "Architect logical call has no physical attempt"
        )
    final = logical.physical_attempts[-1]
    settlement = final.settlement
    if (
        settlement is None
        or settlement.outcome is not RuntimeModelPhysicalOutcome.SUCCEEDED
        or settlement.typed_result is None
        or settlement.typed_result.result_contract
        != "auxiliary-graph-revision-proposal-v2"
        or settlement.typed_result.parsed()
        != binding.architect_decision.proposal.model_dump(mode="json")
    ):
        raise AuxiliaryInitialPlanningPersistenceError(
            "Architect completion lacks the exact succeeded typed settlement"
        )
    return {
        "attempt_count": len(logical.physical_attempts),
        "physical_attempt_id": final.request.physical_attempt_id,
        "physical_ordinal": final.request.physical_ordinal,
        "settlement_id": settlement.settlement_id,
        "settlement_receipt_sha256": settlement.receipt_sha256,
    }


def _require_authenticated_model_execution_history(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    task_id: str,
    commit_turn_id: str,
    logical: object,
) -> None:
    """认证来源、分派、结算和提交 Turn 租约。"""

    request = getattr(logical, "request", None)
    attempts = getattr(logical, "physical_attempts", None)
    if request is None or not isinstance(attempts, tuple):
        raise AuxiliaryInitialPlanningPersistenceError(
            "Architect logical execution history has the wrong contract"
        )
    turn_ids = [str(request.invocation_turn_id), commit_turn_id]
    for physical in attempts:
        physical_request = getattr(physical, "request", None)
        if physical_request is None:
            raise AuxiliaryInitialPlanningPersistenceError(
                "Architect physical execution history is corrupt"
            )
        turn_ids.append(str(physical_request.started_turn_id))
        settlement = getattr(physical, "settlement", None)
        if settlement is not None:
            turn_ids.append(str(settlement.settled_turn_id))
    for turn_id in dict.fromkeys(turn_ids):
        _require_task_execution_lane(
            conn,
            session_id=session_id,
            task_id=task_id,
            turn_id=turn_id,
        )


def _require_task_execution_lane(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    task_id: str,
    turn_id: str,
) -> None:
    rows = conn.execute(
        "SELECT related_insession_task_ids_json, "
        "execution_lane_manifest_json, execution_lane_manifest_hash "
        "FROM insession_task_match_apply_receipts "
        "WHERE session_id=? AND source_turn_id=? ORDER BY created_at, apply_id",
        (session_id, turn_id),
    ).fetchall()
    if len(rows) != 1:
        raise AuxiliaryInitialPlanningPersistenceError(
            "Architect execution Turn has no unique durable lane manifest"
        )
    row = rows[0]
    raw_manifest = row["execution_lane_manifest_json"]
    stored_hash = row["execution_lane_manifest_hash"]
    try:
        manifest = InSessionTaskExecutionLaneManifest.model_validate_json(
            str(raw_manifest)
        )
        related = json.loads(str(row["related_insession_task_ids_json"]))
    except Exception as exc:
        raise AuxiliaryInitialPlanningPersistenceError(
            "Architect execution lane manifest is corrupt"
        ) from exc
    if (
        manifest.manifest_sha256 != stored_hash
        or not isinstance(related, list)
        or tuple(lane.insession_task_id for lane in manifest.lanes)
        != tuple(related)
    ):
        raise AuxiliaryInitialPlanningPersistenceError(
            "Architect execution lane manifest crossed Task-match authority"
        )
    lanes = tuple(lane for lane in manifest.lanes if lane.insession_task_id == task_id)
    if len(lanes) != 1 or lanes[0].execution_requested is not True:
        raise AuxiliaryInitialPlanningPersistenceError(
            "Architect execution history lacks an executable Task lane"
        )
    input_row = conn.execute(
        "SELECT turn_input.content FROM runtime_turn_inputs AS input "
        "JOIN session_turns AS turn_input "
        "ON turn_input.session_id=input.session_id "
        "AND turn_input.turn_idx=input.turn_idx "
        "WHERE input.session_id=? AND input.turn_id=? "
        "AND turn_input.role='user'",
        (session_id, turn_id),
    ).fetchone()
    if input_row is None:
        raise AuxiliaryInitialPlanningPersistenceError(
            "Architect execution Turn has no authoritative user input"
        )
    user_text = str(input_row["content"])
    for lane in manifest.lanes:
        if conn.execute(
            "SELECT 1 FROM insession_tasks WHERE session_id=? "
            "AND insession_task_id=?",
            (session_id, lane.insession_task_id),
        ).fetchone() is None:
            raise AuxiliaryInitialPlanningPersistenceError(
                "Architect execution lane references a missing Task"
            )
        for item in lane.matches:
            span = item.source_span
            excerpt = user_text[span.start : span.end]
            if (
                hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
                != span.text_sha256
            ):
                raise AuxiliaryInitialPlanningPersistenceError(
                    "Architect execution lane source binding has drifted"
                )


def _require_bootstrap_revision(
    conn: sqlite3.Connection,
    binding: AuxiliaryInitialPlanningCompletionBinding,
    *,
    expected_apply_id: str | None = None,
) -> str:
    snapshot = conn.execute(
        "SELECT structure_sha256 FROM "
        "insession_auxiliary_graph_revision_snapshots "
        "WHERE session_id=? AND insession_task_id=? AND auxiliary_graph_id=? "
        "AND goal_id=? AND auxiliary_graph_revision=1",
        (
            binding.session_id,
            binding.task_id,
            binding.auxiliary_graph_id,
            binding.goal_id,
        ),
    ).fetchone()
    receipts = conn.execute(
        "SELECT apply_id FROM "
        "insession_auxiliary_graph_revision_apply_receipts_v2 "
        "WHERE session_id=? AND insession_task_id=? AND auxiliary_graph_id=? "
        "AND goal_id=? AND committed_auxiliary_graph_revision=1 "
        "AND operation='initialize_goal_revision' ORDER BY apply_id",
        (
            binding.session_id,
            binding.task_id,
            binding.auxiliary_graph_id,
            binding.goal_id,
        ),
    ).fetchall()
    if (
        snapshot is None
        or str(snapshot["structure_sha256"])
        != binding.bootstrap_structure_sha256
        or len(receipts) != 1
    ):
        raise AuxiliaryInitialPlanningPersistenceError(
            "initial-planning bootstrap authority is missing or corrupt"
        )
    apply_id = str(receipts[0]["apply_id"])
    if expected_apply_id is not None and apply_id != expected_apply_id:
        raise AuxiliaryInitialPlanningPersistenceError(
            "completion receipt crossed its bootstrap apply authority"
        )
    return apply_id


def _require_positive_bootstrap_revision(
    conn: sqlite3.Connection,
    binding: AuxiliaryPositivePlanningCompletionBinding,
    *,
    expected_apply_id: str | None = None,
) -> str:
    revision = binding.bootstrap_auxiliary_graph_revision
    snapshot = conn.execute(
        "SELECT structure_sha256 FROM "
        "insession_auxiliary_graph_revision_snapshots "
        "WHERE session_id=? AND insession_task_id=? AND auxiliary_graph_id=? "
        "AND goal_id=? AND auxiliary_graph_revision=?",
        (
            binding.session_id,
            binding.task_id,
            binding.auxiliary_graph_id,
            binding.goal_id,
            revision,
        ),
    ).fetchone()
    receipts = conn.execute(
        "SELECT apply_id FROM "
        "insession_auxiliary_graph_revision_apply_receipts_v2 "
        "WHERE session_id=? AND insession_task_id=? AND auxiliary_graph_id=? "
        "AND goal_id=? AND committed_auxiliary_graph_revision=? "
        "AND operation='initialize_goal_revision' ORDER BY apply_id",
        (
            binding.session_id,
            binding.task_id,
            binding.auxiliary_graph_id,
            binding.goal_id,
            revision,
        ),
    ).fetchall()
    if (
        snapshot is None
        or str(snapshot["structure_sha256"])
        != binding.bootstrap_structure_sha256
        or len(receipts) != 1
    ):
        raise AuxiliaryInitialPlanningPersistenceError(
            "positive-planning bootstrap authority is missing or corrupt"
        )
    apply_id = str(receipts[0]["apply_id"])
    if expected_apply_id is not None and apply_id != expected_apply_id:
        raise AuxiliaryInitialPlanningPersistenceError(
            "positive completion crossed bootstrap apply authority"
        )
    return apply_id


def _require_revision_snapshots(
    conn: sqlite3.Connection,
    receipt: StoredAuxiliaryInitialPlanningCompletion,
) -> None:
    binding = receipt.binding
    committed = conn.execute(
        "SELECT structure_sha256, authority_snapshot_id, "
        "authority_snapshot_sha256, parent_auxiliary_graph_revision "
        "FROM insession_auxiliary_graph_revision_snapshots "
        "WHERE session_id=? AND insession_task_id=? AND auxiliary_graph_id=? "
        "AND goal_id=? AND auxiliary_graph_revision=2",
        (
            binding.session_id,
            binding.task_id,
            binding.auxiliary_graph_id,
            binding.goal_id,
        ),
    ).fetchone()
    if (
        committed is None
        or int(committed["parent_auxiliary_graph_revision"]) != 1
        or str(committed["structure_sha256"])
        != receipt.committed_structure_sha256
        or str(committed["authority_snapshot_id"])
        != receipt.committed_authority_snapshot_id
        or str(committed["authority_snapshot_sha256"])
        != receipt.committed_authority_snapshot_sha256
    ):
        raise AuxiliaryInitialPlanningPersistenceError(
            "completion receipt lost its exact committed revision snapshot"
        )


def _require_positive_revision_snapshots(
    conn: sqlite3.Connection,
    receipt: StoredAuxiliaryPositivePlanningCompletion,
) -> None:
    binding = receipt.binding
    committed = conn.execute(
        "SELECT structure_sha256, authority_snapshot_id, "
        "authority_snapshot_sha256, parent_auxiliary_graph_revision "
        "FROM insession_auxiliary_graph_revision_snapshots "
        "WHERE session_id=? AND insession_task_id=? AND auxiliary_graph_id=? "
        "AND goal_id=? AND auxiliary_graph_revision=?",
        (
            binding.session_id,
            binding.task_id,
            binding.auxiliary_graph_id,
            binding.goal_id,
            receipt.committed_auxiliary_graph_revision,
        ),
    ).fetchone()
    if (
        committed is None
        or int(committed["parent_auxiliary_graph_revision"])
        != binding.bootstrap_auxiliary_graph_revision
        or str(committed["structure_sha256"])
        != receipt.committed_structure_sha256
        or str(committed["authority_snapshot_id"])
        != receipt.committed_authority_snapshot_id
        or str(committed["authority_snapshot_sha256"])
        != receipt.committed_authority_snapshot_sha256
    ):
        raise AuxiliaryInitialPlanningPersistenceError(
            "positive completion lost exact committed revision snapshot"
        )


def _require_authenticated_positive_descendant_chain(
    conn: sqlite3.Connection,
    *,
    receipt: StoredAuxiliaryPositivePlanningCompletion,
    current_auxiliary_graph_revision: int,
) -> None:
    """认证 Architect 种子后的每个同目标 revision。"""

    seed_revision = receipt.committed_auxiliary_graph_revision
    if current_auxiliary_graph_revision == seed_revision:
        return
    binding = receipt.binding
    rows = conn.execute(
        "SELECT trigger_id, source_auxiliary_graph_revision, "
        "applied_auxiliary_graph_revision FROM "
        "insession_auxiliary_replan_trigger_applications "
        "WHERE session_id=? AND insession_task_id=? AND auxiliary_graph_id=? "
        "AND goal_id=? AND applied_auxiliary_graph_revision>? "
        "AND applied_auxiliary_graph_revision<=? "
        "ORDER BY applied_auxiliary_graph_revision, trigger_id",
        (
            binding.session_id,
            binding.task_id,
            binding.auxiliary_graph_id,
            binding.goal_id,
            seed_revision,
            current_auxiliary_graph_revision,
        ),
    ).fetchall()
    if len(rows) != current_auxiliary_graph_revision - seed_revision:
        raise AuxiliaryInitialPlanningPersistenceError(
            "positive completion has an unauthenticated descendant revision"
        )
# 本地导入避免持久化模块循环：重规划账本已经导入 AuxiliaryGraph 存储，而后者导入本模块。
    from . import auxiliary_replan_triggers as replan_records

    for expected_revision, row in enumerate(rows, start=seed_revision + 1):
        if (
            int(row["source_auxiliary_graph_revision"])
            != expected_revision - 1
            or int(row["applied_auxiliary_graph_revision"])
            != expected_revision
        ):
            raise AuxiliaryInitialPlanningPersistenceError(
                "positive descendant revision chain is not contiguous"
            )
        try:
            application = replan_records._load_application(
                conn,
                str(row["trigger_id"]),
            )
        except Exception as exc:
            raise AuxiliaryInitialPlanningPersistenceError(
                "positive descendant replan application is corrupt"
            ) from exc
        if (
            application.session_id != binding.session_id
            or application.task_id != binding.task_id
            or application.auxiliary_graph_id != binding.auxiliary_graph_id
            or application.goal_id != binding.goal_id
            or application.source_auxiliary_graph_revision
            != expected_revision - 1
            or application.applied_auxiliary_graph_revision
            != expected_revision
        ):
            raise AuxiliaryInitialPlanningPersistenceError(
                "positive descendant replan crossed goal authority"
            )


def _require_lossless_revision_proposal(
    binding: AuxiliaryInitialPlanningCompletionBinding,
    proposal_payload: Mapping[str, Any],
) -> None:
    proposal = binding.architect_decision.proposal
    structure = proposal.structure
    if structure is None or proposal.revision_reason is None:
        raise AuxiliaryInitialPlanningPersistenceError(
            "Architect completion lacks a committable structure"
        )
    expected = {
        "revision_reason": proposal.revision_reason.value,
        "terminal_node_key": structure.terminal_node_key,
        "nodes": [
            {
                "local_node_key": node.node_key,
                "node_kind": node.node_kind.value,
                "executor_kind": node.executor_kind.value,
                "title": node.title,
                "objective": node.objective,
                "source_anchor_ids": list(node.source_anchor_ids),
                "acceptance_criteria": [
                    item.model_dump(mode="json")
                    for item in node.acceptance_criteria
                ],
                "output_contract": node.output_contract,
                "capability_profile_id": node.capability_profile_id,
                "input_resource_aliases": list(node.input_resource_aliases),
                "required": node.required,
                "origin_node_alias": node.origin_node_alias,
            }
            for node in structure.nodes
        ],
        "edges": [
            {
                "dependency_node_key": edge.source_node_key,
                "consumer_node_key": edge.target_node_key,
                "required": edge.required,
            }
            for edge in structure.edges
        ],
    }
    if dict(proposal_payload) != expected:
        raise AuxiliaryInitialPlanningPersistenceError(
            "persisted revision proposal is not a lossless Architect decision"
        )


def _fresh_binding(
    binding: AuxiliaryInitialPlanningCompletionBinding,
) -> AuxiliaryInitialPlanningCompletionBinding:
    try:
        return AuxiliaryInitialPlanningCompletionBinding.model_validate_json(
            binding.model_dump_json()
        )
    except Exception as exc:
        raise AuxiliaryInitialPlanningPersistenceError(
            "initial-planning completion binding is invalid"
        ) from exc


def _fresh_positive_binding(
    binding: AuxiliaryPositivePlanningCompletionBinding,
) -> AuxiliaryPositivePlanningCompletionBinding:
    try:
        return AuxiliaryPositivePlanningCompletionBinding.model_validate_json(
            binding.model_dump_json()
        )
    except Exception as exc:
        raise AuxiliaryInitialPlanningPersistenceError(
            "positive-planning completion binding is invalid"
        ) from exc


def _require_identifier(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 200
        or value[0] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-"
            for character in value
        )
    ):
        raise ValueError(f"{name} is not a valid durable identifier")


def _completion_receipt_id(payload: object) -> str:
    return "auxv2initialplanreceipt_v1_" + _sha256_value(payload)[:32]


def _positive_identity(authority_id: str, authority_sha256: str) -> str:
    return _sha256_value(
        {
            "schema_version": "auxiliary-v2-positive-planning-identity-v1",
            "trigger_id": authority_id,
            "trigger_sha256": authority_sha256,
        }
    )[:32]


def _positive_completion_receipt_id(payload: object) -> str:
    return "auxv2positiveplanreceipt_v1_" + _sha256_value(payload)[:32]


def _sha256_value(value: object) -> str:
    return _sha256_text(_canonical_json(value))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "AuxiliaryInitialPlanningCompletionBinding",
    "AuxiliaryInitialPlanningPersistenceError",
    "AuxiliaryPositivePlanningCompletionBinding",
    "StoredAuxiliaryPositivePlanningCompletion",
    "StoredAuxiliaryInitialPlanningCompletion",
    "get_auxiliary_initial_planning_completion",
    "load_authenticated_auxiliary_initial_planning_completion",
    "load_authenticated_auxiliary_positive_planning_completion",
    "require_matching_auxiliary_initial_planning_completion",
    "require_matching_auxiliary_positive_planning_completion",
    "seal_auxiliary_initial_planning_completion",
    "seal_auxiliary_positive_planning_completion",
]
