"""AuxiliaryGraph 资源感知原语的持久结算。

Runtime 在跨越有限资源读取边界前冻结调用。本模块在单一 SQLite 事务中消费
该冻结权威和一个自认证资源感知结果。观察、其带类型证据与缺口、验证回执、上下文制品、目标
预算扣费、节点完成投影、Task 状态转换和精确重放回执要么全部提交，要么全部不提交。

制品 schema 刻意让 ``producer_primitive_call_id`` 引用观察 ID，因此原语调用 ID 也是
观察 ID。I/O 前预留是强制的：本事务只接受其精确 ``reserved`` 行，并在插入两个被
引用对象后将其改为 ``settled``。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
import sqlite3
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personagraph.l2.auxiliary_graph import (
    PlanningAuthorityAnchor,
    PlanningAuthorityClass,
    PlanningAuthorityOriginKind,
    PlanningAuthoritySnapshot,
    PlanningContextArtifact,
    PlanningEpisodeBudgetUsage,
    PlanningEpisodeBudget,
    PlanningObservationStatus,
)
from personagraph.l2.planning.invocation_contracts import (
    PlanningContextPrimitiveKind,
)
from ..auxiliary_graph.auxiliary_graph_errors import AuxiliaryGraphPersistenceError
from ..auxiliary_graph.auxiliary_graphs import _formal_budget_from_row
from ...deps import StoreDeps


_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"


class PlanningArtifactSealPersistenceError(AuxiliaryGraphPersistenceError):
    """Host 原语结算丢失冻结规划权威。"""


class PlanningArtifactSealApplyIdCollision(PlanningArtifactSealPersistenceError):
    """密封应用 ID 或原语身份跨越了不可变载荷。"""


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SealAuxiliaryHostPrimitiveResultCommand(_Record):
    """一个冻结 I/O 前原语调用的 Store 所有副本。"""

    schema_version: Literal["seal-auxiliary-host-primitive-result-command-v1"] = (
        "seal-auxiliary-host-primitive-result-command-v1"
    )
    apply_id: str = Field(pattern=_ID_PATTERN)
    session_id: str = Field(pattern=_ID_PATTERN)
    invocation_turn_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_id: str = Field(pattern=_ID_PATTERN)
    goal_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_revision: int = Field(ge=1)
    auxiliary_node_id: str = Field(pattern=_ID_PATTERN)
    node_revision: int = Field(ge=1)
    primitive_kind: PlanningContextPrimitiveKind
    primitive_call_id: str = Field(pattern=_ID_PATTERN)
    expected_artifact_id: str = Field(pattern=_ID_PATTERN)
    expected_verification_receipt_id: str = Field(pattern=_ID_PATTERN)
    expected_scope_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_base_task_graph_revision: int | None = Field(default=None, ge=1)
    expected_task_state_version: int = Field(ge=1)
    expected_node_state_version: int = Field(ge=1)
    expected_control_state_version: int = Field(ge=1)
    expected_goal_state_version: int = Field(ge=1)
    expected_revision_state_version: int = Field(ge=1)
    expected_budget_state_version: int = Field(ge=1)
    expected_authority_snapshot_id: str = Field(pattern=_ID_PATTERN)
    expected_authority_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_structure_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_budget_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    logical_request_json: str = Field(min_length=2)
    logical_request_sha256: str = Field(pattern=_SHA256_PATTERN)
    state_guard_sha256: str = Field(pattern=_SHA256_PATTERN)
    active_seconds_delta: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _validate_frozen_invocation(
        self,
    ) -> 'SealAuxiliaryHostPrimitiveResultCommand':
        if not math.isfinite(self.active_seconds_delta):
            raise ValueError("active_seconds_delta must be finite")
        logical = _require_canonical_json_hash(
            "logical request",
            self.logical_request_json,
            self.logical_request_sha256,
        )
        if logical.get("schema_version") != (
            "planning-resource-perception-request-v1"
        ):
            raise ValueError("primitive kind differs from logical request schema")
        binding = logical.get("binding")
        if not isinstance(binding, dict):
            raise ValueError("logical request has no frozen artifact binding")
        producer = binding.get("producer_auxiliary_node")
        if not isinstance(producer, dict):
            raise ValueError("logical request has no producer AuxiliaryNode")
        expected_binding = {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "auxiliary_graph_id": self.auxiliary_graph_id,
            "goal_id": self.goal_id,
            "primitive_call_id": self.primitive_call_id,
            "artifact_id": self.expected_artifact_id,
            "verification_receipt_id": self.expected_verification_receipt_id,
            "authority_snapshot_id": self.expected_authority_snapshot_id,
            "scope_snapshot_sha256": self.expected_scope_snapshot_sha256,
        }
        if any(binding.get(name) != value for name, value in expected_binding.items()):
            raise ValueError("logical request binding differs from seal authority")
        expected_subject = {
            "task_id": self.task_id,
            "auxiliary_graph_id": self.auxiliary_graph_id,
            "auxiliary_graph_revision": self.auxiliary_graph_revision,
            "node_id": self.auxiliary_node_id,
            "node_revision": self.node_revision,
        }
        if any(producer.get(name) != value for name, value in expected_subject.items()):
            raise ValueError("logical request producer differs from seal subject")
        expected_guard = _sha256_value(
            {
                "schema_version": (
                    "frozen-planning-context-primitive-invocation-v1"
                ),
                "primitive_kind": self.primitive_kind.value,
                "binding": binding,
                "invocation_turn_id": self.invocation_turn_id,
                "expected_state_versions": {
                    "task": self.expected_task_state_version,
                    "node": self.expected_node_state_version,
                    "control": self.expected_control_state_version,
                    "goal": self.expected_goal_state_version,
                    "revision": self.expected_revision_state_version,
                    "budget": self.expected_budget_state_version,
                },
                "authority_snapshot_sha256": (
                    self.expected_authority_snapshot_sha256
                ),
                "structure_sha256": self.expected_structure_sha256,
                "budget_snapshot_sha256": self.expected_budget_snapshot_sha256,
                "logical_request_sha256": self.logical_request_sha256,
            }
        )
        if self.state_guard_sha256 != expected_guard:
            raise ValueError("planning primitive state guard is invalid")
        return self


class PlanningHostPrimitiveSealResult(_Record):
    status: Literal["applied", "replayed"]
    apply_id: str
    primitive_call_id: str
    observation_id: str
    artifact_id: str
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    verification_receipt_id: str
    verification_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    completion_id: str
    task_state_version: int = Field(ge=1)
    control_state_version: int = Field(ge=1)
    goal_state_version: int = Field(ge=1)
    revision_state_version: int = Field(ge=1)
    node_state_version: int = Field(ge=1)
    budget_state_version: int = Field(ge=1)
    budget_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    budget_disposition: Literal[
        "within_limit", "soft_limit_reached", "hard_limit_reached"
    ]


@runtime_checkable
class PlanningResourcePerceptionResultLike(Protocol):
    observation_status: PlanningObservationStatus
    logical_request_json: str
    logical_request_sha256: str
    raw_observation_json: str
    raw_observation_sha256: str
    freshness_manifest_sha256: str
    authority_anchors: tuple[PlanningAuthorityAnchor, ...]
    artifact: PlanningContextArtifact
    prompt_inputs: object
    verification_receipt_sha256: str
    settlement_sha256: str
    read_call_count: int


@dataclass(frozen=True, slots=True)
class _NormalizedPrimitiveResult:
    result: PlanningResourcePerceptionResultLike
    artifact: PlanningContextArtifact
    anchors: tuple[PlanningAuthorityAnchor, ...]
    logical_request: dict[str, Any]
    raw_observation: dict[str, Any]
    observation_snapshot_json: str
    verification_receipt_json: str
    observation_kind: str
    evidence_count: int
    visual_count: int


def seal_auxiliary_host_primitive_result(
    deps: StoreDeps,
    *,
    command: SealAuxiliaryHostPrimitiveResultCommand,
    result: PlanningResourcePerceptionResultLike,
) -> PlanningHostPrimitiveSealResult:
    """原子密封一个当前 ACTIVE 目标的 Host 原语结果。"""

    if not isinstance(command, SealAuxiliaryHostPrimitiveResultCommand):
        raise TypeError(
            "command must be SealAuxiliaryHostPrimitiveResultCommand"
        )
    try:
        command = SealAuxiliaryHostPrimitiveResultCommand.model_validate(
            command.model_dump(mode="json")
        )
    except (TypeError, ValueError) as exc:
        raise PlanningArtifactSealPersistenceError(
            "primitive seal command lost its frozen state guard"
        ) from exc
    normalized = _normalize_result(command, result)
    payload = {
        "command": command.model_dump(mode="json"),
        "primitive_result": json.loads(normalized.observation_snapshot_json),
    }
    payload_sha256 = _sha256_value(payload)
    deps.init_db()
    now = deps.now()
    try:
        with deps.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            replay = _load_apply_receipt(conn, command.apply_id)
            if replay is not None:
                if (
                    str(replay["operation"]) != "seal_observation"
                    or str(replay["session_id"]) != command.session_id
                    or str(replay["insession_task_id"]) != command.task_id
                    or str(replay["auxiliary_graph_id"])
                    != command.auxiliary_graph_id
                    or str(replay["goal_id"]) != command.goal_id
                    or str(replay["payload_sha256"]) != payload_sha256
                ):
                    raise PlanningArtifactSealApplyIdCollision(
                        "primitive seal apply ID crossed immutable authority"
                    )
                stored = PlanningHostPrimitiveSealResult.model_validate_json(
                    str(replay["result_json"])
                )
                _validate_exact_replay(
                    conn,
                    command=command,
                    normalized=normalized,
                    receipt=replay,
                    stored=stored,
                )
                return stored.model_copy(update={"status": "replayed"})

            reservation = _require_reserved_invocation(conn, command=command)
            _require_current_authority(conn, command=command)
            if conn.execute(
                "SELECT 1 FROM insession_auxiliary_observations "
                "WHERE observation_id=?",
                (command.primitive_call_id,),
            ).fetchone() is not None:
                raise PlanningArtifactSealApplyIdCollision(
                    "primitive call ID is already bound to another seal"
                )
            if conn.execute(
                "SELECT 1 FROM insession_auxiliary_planning_context_artifacts "
                "WHERE artifact_id=?",
                (command.expected_artifact_id,),
            ).fetchone() is not None:
                raise PlanningArtifactSealApplyIdCollision(
                    "planning artifact ID is already bound to another seal"
                )

            budget_before, budget_row = _load_current_budget(
                conn,
                command=command,
            )
            if str(reservation["budget_ledger_id"]) != budget_before.budget_ledger_id:
                raise PlanningArtifactSealPersistenceError(
                    "primitive reservation crossed its budget ledger"
                )
            budget_after, delta = _build_budget_after(
                budget_before,
                evidence_count=normalized.evidence_count,
                visual_count=normalized.visual_count,
                active_seconds_delta=command.active_seconds_delta,
            )
            _write_budget_transition(
                conn,
                command=command,
                budget_row=budget_row,
                budget_before=budget_before,
                budget_after=budget_after,
                delta=delta,
                now=now,
            )
            _insert_observation(
                conn,
                command=command,
                normalized=normalized,
                now=now,
            )
            _insert_verification_receipt(
                conn,
                command=command,
                normalized=normalized,
                now=now,
            )
            _insert_artifact(
                conn,
                command=command,
                artifact=normalized.artifact,
                now=now,
            )
            if conn.execute(
                "UPDATE insession_auxiliary_planning_primitive_invocations "
                "SET status='settled', settled_observation_id=?, "
                "settled_artifact_id=?, settlement_sha256=?, settled_at=? "
                "WHERE primitive_call_id=? AND status='reserved' "
                "AND settled_observation_id IS NULL "
                "AND settled_artifact_id IS NULL AND settlement_sha256 IS NULL",
                (
                    command.primitive_call_id,
                    command.expected_artifact_id,
                    normalized.result.settlement_sha256,
                    now,
                    command.primitive_call_id,
                ),
            ).rowcount != 1:
                raise PlanningArtifactSealPersistenceError(
                    "primitive reservation changed during settlement"
                )

            if conn.execute(
                "UPDATE insession_auxiliary_node_states_v2 "
                "SET status='completed', state_version=state_version+1, "
                "updated_at=? WHERE auxiliary_graph_id=? "
                "AND auxiliary_graph_revision=? AND auxiliary_node_id=? "
                "AND node_revision=? AND state_version=? "
                "AND status IN ('proposed', 'interrupted')",
                (
                    now,
                    command.auxiliary_graph_id,
                    command.auxiliary_graph_revision,
                    command.auxiliary_node_id,
                    command.node_revision,
                    command.expected_node_state_version,
                ),
            ).rowcount != 1:
                raise PlanningArtifactSealPersistenceError(
                    "Host primitive node changed during settlement"
                )
            if conn.execute(
                "UPDATE insession_tasks SET state_version=state_version+1, "
                "current_status='active', updated_at=? WHERE session_id=? "
                "AND insession_task_id=? AND current_status NOT IN "
                "('completed', 'cancelled') AND state_version=?",
                (
                    now,
                    command.session_id,
                    command.task_id,
                    command.expected_task_state_version,
                ),
            ).rowcount != 1:
                raise PlanningArtifactSealPersistenceError(
                    "Task changed during Host primitive settlement"
                )
            if conn.execute(
                "UPDATE insession_auxiliary_graph_goals SET "
                "state_version=state_version+1, updated_at=? "
                "WHERE session_id=? AND insession_task_id=? "
                "AND auxiliary_graph_id=? AND goal_id=? AND status='active' "
                "AND state_version=?",
                (
                    now,
                    command.session_id,
                    command.task_id,
                    command.auxiliary_graph_id,
                    command.goal_id,
                    command.expected_goal_state_version,
                ),
            ).rowcount != 1:
                raise PlanningArtifactSealPersistenceError(
                    "planning goal changed during Host primitive settlement"
                )

            sealed = PlanningHostPrimitiveSealResult(
                status="applied",
                apply_id=command.apply_id,
                primitive_call_id=command.primitive_call_id,
                observation_id=command.primitive_call_id,
                artifact_id=normalized.artifact.artifact_id,
                artifact_sha256=normalized.artifact.artifact_sha256,
                verification_receipt_id=(
                    normalized.artifact.verification_receipt_id
                ),
                verification_receipt_sha256=(
                    normalized.artifact.verification_receipt_sha256
                ),
                completion_id=normalized.artifact.artifact_id,
                task_state_version=command.expected_task_state_version + 1,
                control_state_version=command.expected_control_state_version,
                goal_state_version=command.expected_goal_state_version + 1,
                revision_state_version=command.expected_revision_state_version,
                node_state_version=command.expected_node_state_version + 1,
                budget_state_version=command.expected_budget_state_version + 1,
                budget_snapshot_sha256=budget_after.snapshot_sha256,
                budget_disposition=budget_after.assessment.disposition.value,
            )
            result_json = _model_json(sealed)
            conn.execute(
                "INSERT INTO insession_auxiliary_graph_revision_apply_receipts_v2 "
                "(apply_id, operation, session_id, insession_task_id, "
                "auxiliary_graph_id, goal_id, invocation_turn_id, "
                "expected_control_state_version, committed_control_state_version, "
                "expected_current_auxiliary_graph_revision, "
                "committed_auxiliary_graph_revision, expected_goal_state_version, "
                "committed_goal_state_version, expected_budget_state_version, "
                "committed_budget_state_version, budget_ledger_id, "
                "committed_budget_snapshot_json, "
                "committed_budget_snapshot_sha256, payload_sha256, result_json, "
                "result_sha256, created_at) VALUES "
                "(?, 'seal_observation', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    command.apply_id,
                    command.session_id,
                    command.task_id,
                    command.auxiliary_graph_id,
                    command.goal_id,
                    command.invocation_turn_id,
                    command.expected_control_state_version,
                    command.expected_control_state_version,
                    command.auxiliary_graph_revision,
                    command.auxiliary_graph_revision,
                    command.expected_goal_state_version,
                    command.expected_goal_state_version + 1,
                    command.expected_budget_state_version,
                    command.expected_budget_state_version + 1,
                    budget_after.budget_ledger_id,
                    _model_json(budget_after),
                    budget_after.snapshot_sha256,
                    payload_sha256,
                    result_json,
                    _sha256_text(result_json),
                    now,
                ),
            )
            return sealed
    except sqlite3.IntegrityError as exc:
        raise PlanningArtifactSealPersistenceError(
            "Host primitive settlement violated durable authority"
        ) from exc


def _normalize_result(
    command: SealAuxiliaryHostPrimitiveResultCommand,
    result: PlanningResourcePerceptionResultLike,
) -> _NormalizedPrimitiveResult:
    required = (
        "observation_status",
        "logical_request_json",
        "logical_request_sha256",
        "raw_observation_json",
        "raw_observation_sha256",
        "freshness_manifest_sha256",
        "authority_anchors",
        "artifact",
        "prompt_inputs",
        "verification_receipt_sha256",
        "settlement_sha256",
        "read_call_count",
    )
    if any(not hasattr(result, name) for name in required):
        raise TypeError("result is not a planning Host primitive result")
    if (
        result.logical_request_json != command.logical_request_json
        or result.logical_request_sha256 != command.logical_request_sha256
    ):
        raise PlanningArtifactSealPersistenceError(
            "primitive result differs from the frozen logical request"
        )
    logical = _require_canonical_json_hash(
        "logical request",
        result.logical_request_json,
        result.logical_request_sha256,
    )
    raw = _require_canonical_json_hash(
        "raw observation",
        result.raw_observation_json,
        result.raw_observation_sha256,
    )
    try:
        PlanningObservationStatus(result.observation_status)
        artifact = PlanningContextArtifact.model_validate(
            result.artifact.model_dump(mode="json")
        )
        anchors = tuple(
            PlanningAuthorityAnchor.model_validate(
                item.model_dump(mode="json")
            )
            for item in result.authority_anchors
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise PlanningArtifactSealPersistenceError(
            "primitive result carries invalid typed authority"
        ) from exc
    subject = artifact.producer_auxiliary_node
    expected_subject = (
        subject.task_id == command.task_id
        and subject.auxiliary_graph_id == command.auxiliary_graph_id
        and subject.auxiliary_graph_revision == command.auxiliary_graph_revision
        and subject.node_id == command.auxiliary_node_id
        and subject.node_revision == command.node_revision
    )
    if (
        artifact.session_id != command.session_id
        or artifact.task_id != command.task_id
        or artifact.auxiliary_graph_id != command.auxiliary_graph_id
        or artifact.goal_id != command.goal_id
        or not expected_subject
        or artifact.producer_primitive_call_id != command.primitive_call_id
        or artifact.artifact_id != command.expected_artifact_id
        or artifact.verification_receipt_id
        != command.expected_verification_receipt_id
        or artifact.scope_snapshot_sha256
        != command.expected_scope_snapshot_sha256
    ):
        raise PlanningArtifactSealPersistenceError(
            "context artifact crossed frozen primitive authority"
        )
    if (
        artifact.freshness_manifest_sha256
        != result.freshness_manifest_sha256
        or artifact.verification_receipt_sha256
        != result.verification_receipt_sha256
    ):
        raise PlanningArtifactSealPersistenceError(
            "artifact digest bindings differ from primitive settlement"
        )
    if any(
        anchor.authority_snapshot_id != command.expected_authority_snapshot_id
        for anchor in anchors
    ):
        raise PlanningArtifactSealPersistenceError(
            "observation anchor crossed its frozen authority snapshot"
        )
    if any(
        anchor.authority_class is PlanningAuthorityClass.AUTHORIZATION
        for anchor in anchors
    ) or artifact.constraints:
        raise PlanningArtifactSealPersistenceError(
            "Host observations cannot create authorization authority"
        )
    evidence_anchor_ids = {
        item.anchor_id
        for item in anchors
        if item.authority_class is PlanningAuthorityClass.EVIDENCE
    }
    gap_aliases = {
        item.projection_alias
        for item in anchors
        if item.authority_class is PlanningAuthorityClass.GAP
    }
    if evidence_anchor_ids != {
        item.evidence_anchor_id for item in artifact.evidence_refs
    }:
        raise PlanningArtifactSealPersistenceError(
            "artifact evidence differs from observation authority anchors"
        )
    prompt = result.prompt_inputs
    try:
        prompt_projection = prompt.context_artifact
        prompt_cards = tuple(prompt.source_cards)
    except AttributeError as exc:
        raise PlanningArtifactSealPersistenceError(
            "primitive prompt projection is malformed"
        ) from exc
    if (
        prompt_projection.artifact_id != artifact.artifact_id
        or prompt_projection.artifact_sha256 != artifact.artifact_sha256
        or {item.alias for item in prompt_cards}
        != {item.projection_alias for item in anchors}
        or {item.gap_alias for item in prompt_projection.gaps} != gap_aliases
    ):
        raise PlanningArtifactSealPersistenceError(
            "prompt projection differs from private primitive authority"
        )
    expected_freshness = _recompute_freshness_sha256(
        logical=logical,
        raw=raw,
    )
    if result.freshness_manifest_sha256 != expected_freshness:
        raise PlanningArtifactSealPersistenceError(
            "primitive freshness manifest hash is invalid"
        )
    verification_receipt_json = _verification_receipt_json(
        command=command,
        result=result,
        logical=logical,
        artifact=artifact,
        anchors=anchors,
    )
    if _sha256_text(verification_receipt_json) != result.verification_receipt_sha256:
        raise PlanningArtifactSealPersistenceError(
            "primitive verification receipt hash is invalid"
        )
    expected_settlement = _settlement_sha256(
        result=result,
        artifact=artifact,
        anchors=anchors,
    )
    if result.settlement_sha256 != expected_settlement:
        raise PlanningArtifactSealPersistenceError(
            "primitive settlement hash is invalid"
        )
    observation_kind = _observation_kind(logical)
    snapshot_json = _canonical_json(
        {
            "schema_version": "sealed-planning-host-primitive-observation-v1",
            "primitive_kind": command.primitive_kind.value,
            "result": _jsonable(result),
        }
    )
    return _NormalizedPrimitiveResult(
        result=result,
        artifact=artifact,
        anchors=anchors,
        logical_request=logical,
        raw_observation=raw,
        observation_snapshot_json=snapshot_json,
        verification_receipt_json=verification_receipt_json,
        observation_kind=observation_kind,
        evidence_count=len(artifact.evidence_refs),
        visual_count=sum(
            anchor.origin_kind is PlanningAuthorityOriginKind.VISUAL_UNIT
            for anchor in anchors
        ),
    )


def _require_reserved_invocation(
    conn: sqlite3.Connection,
    *,
    command: SealAuxiliaryHostPrimitiveResultCommand,
) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM insession_auxiliary_planning_primitive_invocations "
        "WHERE primitive_call_id=?",
        (command.primitive_call_id,),
    ).fetchone()
    if row is None:
        raise PlanningArtifactSealPersistenceError(
            "Host primitive result has no pre-I/O reservation"
        )
    expected_columns: dict[str, object] = {
        "session_id": command.session_id,
        "insession_task_id": command.task_id,
        "auxiliary_graph_id": command.auxiliary_graph_id,
        "goal_id": command.goal_id,
        "auxiliary_graph_revision": command.auxiliary_graph_revision,
        "auxiliary_node_id": command.auxiliary_node_id,
        "node_revision": command.node_revision,
        "invocation_turn_id": command.invocation_turn_id,
        "primitive_kind": command.primitive_kind.value,
        "planned_artifact_id": command.expected_artifact_id,
        "planned_verification_receipt_id": (
            command.expected_verification_receipt_id
        ),
        "authority_snapshot_id": command.expected_authority_snapshot_id,
        "scope_snapshot_sha256": command.expected_scope_snapshot_sha256,
        "expected_task_state_version": command.expected_task_state_version,
        "expected_node_state_version": command.expected_node_state_version,
        "expected_control_state_version": command.expected_control_state_version,
        "expected_goal_state_version": command.expected_goal_state_version,
        "expected_revision_state_version": (
            command.expected_revision_state_version
        ),
        "expected_budget_state_version": command.expected_budget_state_version,
        "authority_snapshot_sha256": (
            command.expected_authority_snapshot_sha256
        ),
        "structure_sha256": command.expected_structure_sha256,
        "budget_snapshot_sha256": command.expected_budget_snapshot_sha256,
        "logical_request_json": command.logical_request_json,
        "logical_request_sha256": command.logical_request_sha256,
        "state_guard_sha256": command.state_guard_sha256,
        "status": "reserved",
    }
    if any(row[name] != value for name, value in expected_columns.items()):
        raise PlanningArtifactSealPersistenceError(
            "Host primitive reservation differs from seal authority"
        )
    if any(
        row[name] is not None
        for name in (
            "settled_observation_id",
            "settled_artifact_id",
            "settlement_sha256",
            "settled_at",
        )
    ):
        raise PlanningArtifactSealPersistenceError(
            "reserved primitive already carries settlement authority"
        )
    logical = json.loads(command.logical_request_json)
    expected_invocation = {
        "schema_version": "frozen-planning-context-primitive-invocation-v1",
        "primitive_kind": command.primitive_kind.value,
        "binding": logical["binding"],
        "invocation_turn_id": command.invocation_turn_id,
        "expected_task_state_version": command.expected_task_state_version,
        "expected_node_state_version": command.expected_node_state_version,
        "expected_control_state_version": command.expected_control_state_version,
        "expected_goal_state_version": command.expected_goal_state_version,
        "expected_revision_state_version": command.expected_revision_state_version,
        "expected_budget_state_version": command.expected_budget_state_version,
        "authority_snapshot_sha256": command.expected_authority_snapshot_sha256,
        "structure_sha256": command.expected_structure_sha256,
        "budget_snapshot_sha256": command.expected_budget_snapshot_sha256,
        "logical_request_json": command.logical_request_json,
        "logical_request_sha256": command.logical_request_sha256,
        "state_guard_sha256": command.state_guard_sha256,
    }
    invocation_json = str(row["invocation_json"])
    if (
        invocation_json != _canonical_json(expected_invocation)
        or str(row["invocation_sha256"]) != _sha256_text(invocation_json)
    ):
        raise PlanningArtifactSealPersistenceError(
            "Host primitive reservation JSON/hash binding is corrupt"
        )
    return row


def _require_current_authority(
    conn: sqlite3.Connection,
    *,
    command: SealAuxiliaryHostPrimitiveResultCommand,
) -> None:
    turn = conn.execute(
        "SELECT session_id, status FROM runtime_turns WHERE turn_id=?",
        (command.invocation_turn_id,),
    ).fetchone()
    if (
        turn is None
        or str(turn["session_id"]) != command.session_id
        or str(turn["status"]) != "running"
    ):
        raise PlanningArtifactSealPersistenceError(
            "primitive invocation Turn is outside this Session"
        )
    if conn.execute(
        "SELECT 1 FROM insession_task_turn_links WHERE session_id=? "
        "AND turn_id=? AND insession_task_id=?",
        (command.session_id, command.invocation_turn_id, command.task_id),
    ).fetchone() is None:
        raise PlanningArtifactSealPersistenceError(
            "primitive invocation Turn is not linked to its Task"
        )
    task = conn.execute(
        "SELECT current_graph_revision, current_status, state_version "
        "FROM insession_tasks WHERE session_id=? AND insession_task_id=?",
        (command.session_id, command.task_id),
    ).fetchone()
    if (
        task is None
        or str(task["current_status"]) in {"completed", "cancelled"}
        or int(task["state_version"]) != command.expected_task_state_version
        or _optional_int(task["current_graph_revision"])
        != command.expected_base_task_graph_revision
    ):
        raise PlanningArtifactSealPersistenceError(
            "Task authority changed after primitive dispatch"
        )
    control = conn.execute(
        "SELECT current_goal_id, current_auxiliary_graph_revision, state_version "
        "FROM insession_auxiliary_graph_v2_containers "
        "WHERE session_id=? AND insession_task_id=? AND auxiliary_graph_id=?",
        (command.session_id, command.task_id, command.auxiliary_graph_id),
    ).fetchone()
    if (
        control is None
        or str(control["current_goal_id"]) != command.goal_id
        or int(control["current_auxiliary_graph_revision"])
        != command.auxiliary_graph_revision
        or int(control["state_version"])
        != command.expected_control_state_version
    ):
        raise PlanningArtifactSealPersistenceError(
            "AuxiliaryGraph control pointer changed after primitive dispatch"
        )
    goal = conn.execute(
        "SELECT status, state_version, base_task_graph_revision "
        "FROM insession_auxiliary_graph_goals "
        "WHERE session_id=? AND insession_task_id=? AND auxiliary_graph_id=? "
        "AND goal_id=?",
        (
            command.session_id,
            command.task_id,
            command.auxiliary_graph_id,
            command.goal_id,
        ),
    ).fetchone()
    if (
        goal is None
        or str(goal["status"]) != "active"
        or int(goal["state_version"]) != command.expected_goal_state_version
        or _optional_int(goal["base_task_graph_revision"])
        != command.expected_base_task_graph_revision
    ):
        raise PlanningArtifactSealPersistenceError(
            "planning goal changed after primitive dispatch"
        )
    revision = conn.execute(
        "SELECT snapshot.structure_sha256, snapshot.authority_snapshot_id, "
        "snapshot.authority_snapshot_sha256, state.status, state.state_version "
        "FROM insession_auxiliary_graph_revision_snapshots AS snapshot "
        "JOIN insession_auxiliary_graph_revision_states_v2 AS state "
        "ON state.auxiliary_graph_id=snapshot.auxiliary_graph_id "
        "AND state.auxiliary_graph_revision=snapshot.auxiliary_graph_revision "
        "WHERE snapshot.session_id=? AND snapshot.insession_task_id=? "
        "AND snapshot.auxiliary_graph_id=? AND snapshot.goal_id=? "
        "AND snapshot.auxiliary_graph_revision=?",
        (
            command.session_id,
            command.task_id,
            command.auxiliary_graph_id,
            command.goal_id,
            command.auxiliary_graph_revision,
        ),
    ).fetchone()
    if (
        revision is None
        or str(revision["status"]) != "active"
        or int(revision["state_version"])
        != command.expected_revision_state_version
        or str(revision["structure_sha256"])
        != command.expected_structure_sha256
        or str(revision["authority_snapshot_id"])
        != command.expected_authority_snapshot_id
        or str(revision["authority_snapshot_sha256"])
        != command.expected_authority_snapshot_sha256
    ):
        raise PlanningArtifactSealPersistenceError(
            "AuxiliaryGraph revision changed after primitive dispatch"
        )
    authority = conn.execute(
        "SELECT contract_version, snapshot_json, snapshot_sha256 "
        "FROM insession_auxiliary_authority_snapshots WHERE session_id=? "
        "AND insession_task_id=? AND auxiliary_graph_id=? AND goal_id=? "
        "AND authority_snapshot_id=?",
        (
            command.session_id,
            command.task_id,
            command.auxiliary_graph_id,
            command.goal_id,
            command.expected_authority_snapshot_id,
        ),
    ).fetchone()
    try:
        authority_snapshot = (
            None
            if authority is None
            else PlanningAuthoritySnapshot.model_validate_json(
                str(authority["snapshot_json"])
            )
        )
    except (TypeError, ValueError) as exc:
        raise PlanningArtifactSealPersistenceError(
            "frozen planning authority snapshot is corrupt"
        ) from exc
    if (
        authority is None
        or authority_snapshot is None
        or str(authority["contract_version"])
        != "planning-authority-snapshot-v1"
        or str(authority["snapshot_sha256"])
        != command.expected_authority_snapshot_sha256
        or authority_snapshot.snapshot_sha256
        != command.expected_authority_snapshot_sha256
        or _model_json(authority_snapshot) != str(authority["snapshot_json"])
    ):
        raise PlanningArtifactSealPersistenceError(
            "frozen planning authority snapshot is corrupt or stale"
        )
    node = conn.execute(
        "SELECT definition.executor_kind, node_state.status, "
        "node_state.state_version FROM "
        "insession_auxiliary_graph_revision_nodes_v2 AS membership "
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
        "WHERE membership.auxiliary_graph_id=? "
        "AND membership.auxiliary_graph_revision=? "
        "AND membership.auxiliary_node_id=? AND membership.node_revision=?",
        (
            command.auxiliary_graph_id,
            command.auxiliary_graph_revision,
            command.auxiliary_node_id,
            command.node_revision,
        ),
    ).fetchone()
    if (
        node is None
        or str(node["executor_kind"]) != "host_primitive"
        or str(node["status"]) not in {"proposed", "interrupted"}
        or int(node["state_version"]) != command.expected_node_state_version
    ):
        raise PlanningArtifactSealPersistenceError(
            "primitive result is not bound to a ready Host node"
        )
    missing_dependency = conn.execute(
        "SELECT 1 FROM insession_auxiliary_graph_edges AS edge "
        "JOIN insession_auxiliary_node_states_v2 AS dependency "
        "ON dependency.auxiliary_graph_id=edge.auxiliary_graph_id "
        "AND dependency.auxiliary_graph_revision=edge.auxiliary_graph_revision "
        "AND dependency.auxiliary_node_id=edge.dependency_auxiliary_node_id "
        "AND dependency.node_revision=edge.dependency_node_revision "
        "WHERE edge.auxiliary_graph_id=? AND edge.auxiliary_graph_revision=? "
        "AND edge.consumer_auxiliary_node_id=? "
        "AND dependency.status<>'completed' LIMIT 1",
        (
            command.auxiliary_graph_id,
            command.auxiliary_graph_revision,
            command.auxiliary_node_id,
        ),
    ).fetchone()
    if missing_dependency is not None:
        raise PlanningArtifactSealPersistenceError(
            "Host primitive dependencies are not completed"
        )


def _load_current_budget(
    conn: sqlite3.Connection,
    *,
    command: SealAuxiliaryHostPrimitiveResultCommand,
) -> tuple[PlanningEpisodeBudget, sqlite3.Row]:
    row = conn.execute(
        "SELECT budget_ledger_id, contract_version, profile_json, "
        "profile_sha256, usage_json, usage_sha256, extensions_json, "
        "extensions_sha256, snapshot_json, snapshot_sha256, state_version "
        "FROM insession_auxiliary_goal_budgets WHERE session_id=? "
        "AND insession_task_id=? AND auxiliary_graph_id=? AND goal_id=?",
        (
            command.session_id,
            command.task_id,
            command.auxiliary_graph_id,
            command.goal_id,
        ),
    ).fetchone()
    if row is None:
        raise PlanningArtifactSealPersistenceError("planning goal budget is missing")
    try:
        budget = _formal_budget_from_row(
            row,
            expected_goal_id=command.goal_id,
        )
    except AuxiliaryGraphPersistenceError as exc:
        raise PlanningArtifactSealPersistenceError(str(exc)) from exc
    if (
        budget.state_version != command.expected_budget_state_version
        or budget.snapshot_sha256 != command.expected_budget_snapshot_sha256
    ):
        raise PlanningArtifactSealPersistenceError(
            "planning budget changed after primitive dispatch"
        )
    return budget, row


def _build_budget_after(
    before: PlanningEpisodeBudget,
    *,
    evidence_count: int,
    visual_count: int,
    active_seconds_delta: float,
) -> tuple[PlanningEpisodeBudget, dict[str, int | float]]:
    delta: dict[str, int | float] = {
        "logical_tool_calls": 1,
        "evidence_units": evidence_count,
        "visual_units": visual_count,
        "active_seconds": active_seconds_delta,
    }
    usage = before.usage.model_dump(mode="python")
    for name, value in delta.items():
        usage[name] = usage[name] + value
    try:
        after_usage = PlanningEpisodeBudgetUsage.model_validate(usage)
        after = PlanningEpisodeBudget.create(
            budget_ledger_id=before.budget_ledger_id,
            goal_id=before.goal_id,
            base_profile=before.base_profile,
            usage=after_usage,
            extensions=before.extensions,
            state_version=before.state_version + 1,
        )
    except (TypeError, ValueError) as exc:
        raise PlanningArtifactSealPersistenceError(
            "Host primitive budget transition is invalid"
        ) from exc
    return after, delta


def _write_budget_transition(
    conn: sqlite3.Connection,
    *,
    command: SealAuxiliaryHostPrimitiveResultCommand,
    budget_row: sqlite3.Row,
    budget_before: PlanningEpisodeBudget,
    budget_after: PlanningEpisodeBudget,
    delta: dict[str, int | float],
    now: str,
) -> None:
    before_usage_json = _model_json(budget_before.usage)
    after_usage_json = _model_json(budget_after.usage)
    after_json = _model_json(budget_after)
    if conn.execute(
        "UPDATE insession_auxiliary_goal_budgets SET usage_json=?, "
        "usage_sha256=?, snapshot_json=?, snapshot_sha256=?, "
        "state_version=state_version+1, updated_at=? WHERE goal_id=? "
        "AND state_version=? AND snapshot_sha256=?",
        (
            after_usage_json,
            _sha256_text(after_usage_json),
            after_json,
            budget_after.snapshot_sha256,
            now,
            command.goal_id,
            command.expected_budget_state_version,
            command.expected_budget_snapshot_sha256,
        ),
    ).rowcount != 1:
        raise PlanningArtifactSealPersistenceError(
            "planning budget changed during primitive settlement"
        )
    conn.execute(
        "INSERT INTO insession_auxiliary_goal_budget_snapshots "
        "(goal_id, budget_ledger_id, state_version, session_id, "
        "insession_task_id, auxiliary_graph_id, snapshot_json, "
        "snapshot_sha256, created_turn_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            command.goal_id,
            budget_after.budget_ledger_id,
            budget_after.state_version,
            command.session_id,
            command.task_id,
            command.auxiliary_graph_id,
            after_json,
            budget_after.snapshot_sha256,
            command.invocation_turn_id,
            now,
        ),
    )
    previous = conn.execute(
        "SELECT charge_sha256, budget_state_version_after, "
        "budget_snapshot_after_sha256 FROM "
        "insession_auxiliary_goal_budget_charges WHERE goal_id=? "
        "ORDER BY budget_state_version_after DESC LIMIT 1",
        (command.goal_id,),
    ).fetchone()
    if (
        previous is None
        or int(previous["budget_state_version_after"])
        != command.expected_budget_state_version
        or str(previous["budget_snapshot_after_sha256"])
        != command.expected_budget_snapshot_sha256
    ):
        raise PlanningArtifactSealPersistenceError(
            "planning budget charge chain is incomplete"
        )
    charge_payload = {
        "goal_id": command.goal_id,
        "charge_kind": "tool_call",
        "charge_key": command.apply_id,
        "primitive_call_id": command.primitive_call_id,
        "state_guard_sha256": command.state_guard_sha256,
        "usage_before": budget_before.usage.model_dump(mode="json"),
        "usage_after": budget_after.usage.model_dump(mode="json"),
        "delta": delta,
    }
    payload_sha256 = _sha256_value(charge_payload)
    budget_charge_id = "auxbudget_" + command.apply_id
    charge_sha256 = _sha256_value(
        {
            **charge_payload,
            "budget_charge_id": budget_charge_id,
            "budget_ledger_id": budget_after.budget_ledger_id,
            "budget_state_version_before": command.expected_budget_state_version,
            "budget_state_version_after": command.expected_budget_state_version + 1,
            "previous_charge_sha256": str(previous["charge_sha256"]),
        }
    )
    conn.execute(
        "INSERT INTO insession_auxiliary_goal_budget_charges "
        "(budget_charge_id, session_id, insession_task_id, auxiliary_graph_id, "
        "goal_id, budget_ledger_id, charge_kind, charge_key, "
        "budget_state_version_before, budget_state_version_after, "
        "usage_before_json, usage_before_sha256, usage_after_json, "
        "usage_after_sha256, budget_snapshot_after_json, "
        "budget_snapshot_after_sha256, delta_json, payload_sha256, "
        "previous_charge_sha256, charge_sha256, created_turn_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?)",
        (
            budget_charge_id,
            command.session_id,
            command.task_id,
            command.auxiliary_graph_id,
            command.goal_id,
            budget_after.budget_ledger_id,
            "tool_call",
            command.apply_id,
            command.expected_budget_state_version,
            command.expected_budget_state_version + 1,
            before_usage_json,
            _sha256_text(before_usage_json),
            after_usage_json,
            _sha256_text(after_usage_json),
            after_json,
            budget_after.snapshot_sha256,
            _canonical_json(delta),
            payload_sha256,
            str(previous["charge_sha256"]),
            charge_sha256,
            command.invocation_turn_id,
            now,
        ),
    )


def _insert_observation(
    conn: sqlite3.Connection,
    *,
    command: SealAuxiliaryHostPrimitiveResultCommand,
    normalized: _NormalizedPrimitiveResult,
    now: str,
) -> None:
    result = normalized.result
    snapshot_sha256 = _sha256_text(normalized.observation_snapshot_json)
    conn.execute(
        "INSERT INTO insession_auxiliary_observations "
        "(observation_id, session_id, insession_task_id, auxiliary_graph_id, "
        "goal_id, auxiliary_graph_revision, auxiliary_node_id, node_revision, "
        "work_run_id, attempt_id, tool_result_id, observation_kind, outcome, "
        "request_fingerprint, data_version, snapshot_json, snapshot_sha256, "
        "created_turn_id, created_at) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            command.primitive_call_id,
            command.session_id,
            command.task_id,
            command.auxiliary_graph_id,
            command.goal_id,
            command.auxiliary_graph_revision,
            command.auxiliary_node_id,
            command.node_revision,
            normalized.observation_kind,
            PlanningObservationStatus(result.observation_status).value,
            result.logical_request_sha256,
            result.freshness_manifest_sha256,
            normalized.observation_snapshot_json,
            snapshot_sha256,
            command.invocation_turn_id,
            now,
        ),
    )
    anchors_by_id = {item.anchor_id: item for item in normalized.anchors}
    facts_by_anchor: dict[str, str] = {}
    for fact in normalized.artifact.facts:
        for anchor_id in fact.evidence_anchor_ids:
            facts_by_anchor.setdefault(anchor_id, fact.statement)
    for ordinal, evidence in enumerate(normalized.artifact.evidence_refs, start=1):
        anchor = anchors_by_id[evidence.evidence_anchor_id]
        excerpt = facts_by_anchor.get(anchor.anchor_id, evidence.locator)
        citation = {
            "evidence_ref": evidence.model_dump(mode="json"),
            "authority_anchor": anchor.model_dump(mode="json"),
        }
        conn.execute(
            "INSERT INTO insession_auxiliary_observation_items "
            "(observation_item_id, observation_id, ordinal, source_type, "
            "source_unit_id, source_revision, indexed_content_hash, excerpt, "
            "excerpt_sha256, citation_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _derived_id("auxobsitem", command.primitive_call_id, str(ordinal)),
                command.primitive_call_id,
                ordinal,
                anchor.origin_kind.value,
                anchor.anchor_id,
                (
                    str(evidence.source_revision)
                    if evidence.source_revision is not None
                    else "freshness:" + evidence.freshness_binding_sha256[:24]
                ),
                evidence.content_sha256,
                excerpt,
                _sha256_text(excerpt),
                _canonical_json(citation),
                now,
            ),
        )
    for ordinal, gap in enumerate(normalized.artifact.gaps, start=1):
        conn.execute(
            "INSERT INTO insession_auxiliary_observation_gaps "
            "(observation_gap_id, observation_id, ordinal, source_type, "
            "reason_code, blocking, known_count, created_at) "
            "VALUES (?, ?, ?, 'host_primitive', ?, ?, ?, ?)",
            (
                _derived_id("auxobsgap", command.primitive_call_id, str(ordinal)),
                command.primitive_call_id,
                ordinal,
                gap.gap_id,
                int(gap.blocking),
                normalized.evidence_count,
                now,
            ),
        )


def _insert_verification_receipt(
    conn: sqlite3.Connection,
    *,
    command: SealAuxiliaryHostPrimitiveResultCommand,
    normalized: _NormalizedPrimitiveResult,
    now: str,
) -> None:
    status = PlanningObservationStatus(normalized.result.observation_status)
    disposition = (
        "pass"
        if status is PlanningObservationStatus.SUCCESS
        else "partial"
        if status is PlanningObservationStatus.PARTIAL
        else "fail"
    )
    conn.execute(
        "INSERT INTO insession_auxiliary_context_verification_receipts "
        "(verification_receipt_id, session_id, insession_task_id, "
        "auxiliary_graph_id, goal_id, auxiliary_graph_revision, "
        "auxiliary_node_id, node_revision, source_observation_id, "
        "source_work_run_id, source_attempt_id, disposition, receipt_json, "
        "receipt_sha256, created_turn_id, created_at) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?)",
        (
            command.expected_verification_receipt_id,
            command.session_id,
            command.task_id,
            command.auxiliary_graph_id,
            command.goal_id,
            command.auxiliary_graph_revision,
            command.auxiliary_node_id,
            command.node_revision,
            command.primitive_call_id,
            disposition,
            normalized.verification_receipt_json,
            normalized.result.verification_receipt_sha256,
            command.invocation_turn_id,
            now,
        ),
    )


def _insert_artifact(
    conn: sqlite3.Connection,
    *,
    command: SealAuxiliaryHostPrimitiveResultCommand,
    artifact: PlanningContextArtifact,
    now: str,
) -> None:
    artifact_json = _model_json(artifact)
    conn.execute(
        "INSERT INTO insession_auxiliary_planning_context_artifacts "
        "(artifact_id, session_id, insession_task_id, auxiliary_graph_id, "
        "goal_id, producer_auxiliary_graph_revision, "
        "producer_auxiliary_node_id, producer_node_revision, "
        "producer_work_run_id, producer_attempt_id, "
        "producer_tool_result_ids_json, producer_primitive_call_id, "
        "scope_snapshot_sha256, facts_json, constraints_json, conflicts_json, "
        "gaps_json, evidence_refs_json, freshness_manifest_sha256, "
        "verification_receipt_id, verification_receipt_sha256, artifact_json, "
        "artifact_sha256, created_turn_id, created_at) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, '[]', ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?, ?)",
        (
            artifact.artifact_id,
            command.session_id,
            command.task_id,
            command.auxiliary_graph_id,
            command.goal_id,
            command.auxiliary_graph_revision,
            command.auxiliary_node_id,
            command.node_revision,
            command.primitive_call_id,
            artifact.scope_snapshot_sha256,
            _canonical_json(
                [item.model_dump(mode="json") for item in artifact.facts]
            ),
            _canonical_json(
                [item.model_dump(mode="json") for item in artifact.constraints]
            ),
            _canonical_json(
                [item.model_dump(mode="json") for item in artifact.conflicts]
            ),
            _canonical_json(
                [item.model_dump(mode="json") for item in artifact.gaps]
            ),
            _canonical_json(
                [item.model_dump(mode="json") for item in artifact.evidence_refs]
            ),
            artifact.freshness_manifest_sha256,
            artifact.verification_receipt_id,
            artifact.verification_receipt_sha256,
            artifact_json,
            artifact.artifact_sha256,
            command.invocation_turn_id,
            now,
        ),
    )


def _recompute_freshness_sha256(
    *,
    logical: dict[str, Any],
    raw: dict[str, Any],
) -> str:
    resource = _require_mapping(_nested_get(logical, "read_request", "resource"))
    outcome = _require_mapping(raw.get("outcome"))
    observed_version = outcome.get("observed_resource_version")
    return _sha256_value(
        {
            "schema_version": "planning-resource-freshness-manifest-v1",
            "expected": {
                "resource_alias": resource.get("resource_alias"),
                "resource_id": resource.get("resource_id"),
                "resource_version": resource.get("resource_version"),
                "content_sha256": resource.get("content_sha256"),
                "coverage": resource.get("coverage"),
            },
            "observed": (
                None
                if observed_version is None
                else {
                    "resource_version": observed_version,
                    "content_sha256": outcome.get("observed_content_sha256"),
                    "coverage": outcome.get("observed_coverage"),
                }
            ),
        }
    )


def _verification_receipt_json(
    *,
    command: SealAuxiliaryHostPrimitiveResultCommand,
    result: PlanningResourcePerceptionResultLike,
    logical: dict[str, Any],
    artifact: PlanningContextArtifact,
    anchors: tuple[PlanningAuthorityAnchor, ...],
) -> str:
    resource = _require_mapping(_nested_get(logical, "read_request", "resource"))
    return _canonical_json(
        {
            "schema_version": "planning-resource-verification-receipt-v1",
            "primitive_call_id": command.primitive_call_id,
            "resource_alias": resource.get("resource_alias"),
            "resource_format": resource.get("resource_format"),
            "observation_status": PlanningObservationStatus(
                result.observation_status
            ).value,
            "logical_request_sha256": result.logical_request_sha256,
            "raw_observation_sha256": result.raw_observation_sha256,
            "freshness_manifest_sha256": result.freshness_manifest_sha256,
            "authority_anchors": [
                item.model_dump(mode="json") for item in anchors
            ],
            "facts": [item.model_dump(mode="json") for item in artifact.facts],
            "gaps": [item.model_dump(mode="json") for item in artifact.gaps],
        }
    )


def _settlement_sha256(
    *,
    result: PlanningResourcePerceptionResultLike,
    artifact: PlanningContextArtifact,
    anchors: tuple[PlanningAuthorityAnchor, ...],
) -> str:
    return _sha256_value(
        {
            "schema_version": "planning-resource-perception-settlement-v1",
            "observation_status": PlanningObservationStatus(
                result.observation_status
            ).value,
            "logical_request_sha256": result.logical_request_sha256,
            "raw_observation_sha256": result.raw_observation_sha256,
            "freshness_manifest_sha256": result.freshness_manifest_sha256,
            "authority_anchors": [
                item.model_dump(mode="json") for item in anchors
            ],
            "artifact": artifact.model_dump(mode="json"),
            "prompt_inputs": _jsonable(result.prompt_inputs),
            "verification_receipt_sha256": result.verification_receipt_sha256,
            "read_call_count": result.read_call_count,
        }
    )


def _observation_kind(logical: dict[str, Any]) -> str:
    resource = _require_mapping(_nested_get(logical, "read_request", "resource"))
    return (
        "visual"
        if resource.get("resource_format") in {"jpg", "jpeg", "png"}
        else "document"
    )


def _load_apply_receipt(
    conn: sqlite3.Connection,
    apply_id: str,
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM insession_auxiliary_graph_revision_apply_receipts_v2 "
        "WHERE apply_id=?",
        (apply_id,),
    ).fetchone()


def _validate_exact_replay(
    conn: sqlite3.Connection,
    *,
    command: SealAuxiliaryHostPrimitiveResultCommand,
    normalized: _NormalizedPrimitiveResult,
    receipt: sqlite3.Row,
    stored: PlanningHostPrimitiveSealResult,
) -> None:
    result_json = str(receipt["result_json"])
    budget_json = str(receipt["committed_budget_snapshot_json"])
    try:
        budget = PlanningEpisodeBudget.model_validate_json(budget_json)
    except (TypeError, ValueError) as exc:
        raise PlanningArtifactSealPersistenceError(
            "primitive seal replay budget is corrupt"
        ) from exc
    observation = conn.execute(
        "SELECT snapshot_json, snapshot_sha256, request_fingerprint, "
        "data_version FROM insession_auxiliary_observations "
        "WHERE observation_id=?",
        (command.primitive_call_id,),
    ).fetchone()
    verification = conn.execute(
        "SELECT receipt_json, receipt_sha256 FROM "
        "insession_auxiliary_context_verification_receipts "
        "WHERE verification_receipt_id=?",
        (command.expected_verification_receipt_id,),
    ).fetchone()
    artifact_row = conn.execute(
        "SELECT artifact_json, artifact_sha256, producer_primitive_call_id, "
        "verification_receipt_id, verification_receipt_sha256, "
        "scope_snapshot_sha256, facts_json, constraints_json, conflicts_json, "
        "gaps_json, evidence_refs_json, freshness_manifest_sha256 "
        "FROM insession_auxiliary_planning_context_artifacts "
        "WHERE artifact_id=?",
        (command.expected_artifact_id,),
    ).fetchone()
    charge = conn.execute(
        "SELECT charge_sha256, payload_sha256, budget_state_version_after, "
        "budget_snapshot_after_json, budget_snapshot_after_sha256 "
        "FROM insession_auxiliary_goal_budget_charges "
        "WHERE goal_id=? AND charge_key=?",
        (command.goal_id, command.apply_id),
    ).fetchone()
    node = conn.execute(
        "SELECT status, state_version FROM insession_auxiliary_node_states_v2 "
        "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=? "
        "AND auxiliary_node_id=? AND node_revision=?",
        (
            command.auxiliary_graph_id,
            command.auxiliary_graph_revision,
            command.auxiliary_node_id,
            command.node_revision,
        ),
    ).fetchone()
    reservation = conn.execute(
        "SELECT status, settled_observation_id, settled_artifact_id, "
        "settlement_sha256, invocation_json, invocation_sha256, "
        "logical_request_sha256, state_guard_sha256 "
        "FROM insession_auxiliary_planning_primitive_invocations "
        "WHERE primitive_call_id=?",
        (command.primitive_call_id,),
    ).fetchone()
    try:
        loaded_artifact = (
            None
            if artifact_row is None
            else PlanningContextArtifact.model_validate_json(
                str(artifact_row["artifact_json"])
            )
        )
    except (TypeError, ValueError) as exc:
        raise PlanningArtifactSealPersistenceError(
            "primitive seal replay artifact is corrupt"
        ) from exc
    valid = (
        _sha256_text(result_json) == str(receipt["result_sha256"])
        and stored.apply_id == command.apply_id
        and stored.primitive_call_id == command.primitive_call_id
        and stored.artifact_id == command.expected_artifact_id
        and stored.artifact_sha256 == normalized.artifact.artifact_sha256
        and stored.verification_receipt_id
        == command.expected_verification_receipt_id
        and stored.verification_receipt_sha256
        == normalized.result.verification_receipt_sha256
        and int(receipt["committed_control_state_version"])
        == stored.control_state_version
        and int(receipt["committed_goal_state_version"])
        == stored.goal_state_version
        and int(receipt["committed_budget_state_version"])
        == stored.budget_state_version
        and int(receipt["committed_auxiliary_graph_revision"])
        == command.auxiliary_graph_revision
        and budget.goal_id == command.goal_id
        and budget.state_version == stored.budget_state_version
        and budget.snapshot_sha256 == stored.budget_snapshot_sha256
        and _model_json(budget) == budget_json
        and observation is not None
        and str(observation["snapshot_json"])
        == normalized.observation_snapshot_json
        and _sha256_text(str(observation["snapshot_json"]))
        == str(observation["snapshot_sha256"])
        and str(observation["request_fingerprint"])
        == command.logical_request_sha256
        and str(observation["data_version"])
        == normalized.result.freshness_manifest_sha256
        and verification is not None
        and str(verification["receipt_json"])
        == normalized.verification_receipt_json
        and _sha256_text(str(verification["receipt_json"]))
        == str(verification["receipt_sha256"])
        and artifact_row is not None
        and loaded_artifact == normalized.artifact
        and str(artifact_row["artifact_sha256"])
        == normalized.artifact.artifact_sha256
        and str(artifact_row["producer_primitive_call_id"])
        == command.primitive_call_id
        and str(artifact_row["verification_receipt_id"])
        == command.expected_verification_receipt_id
        and str(artifact_row["verification_receipt_sha256"])
        == normalized.result.verification_receipt_sha256
        and str(artifact_row["scope_snapshot_sha256"])
        == normalized.artifact.scope_snapshot_sha256
        and str(artifact_row["facts_json"])
        == _canonical_json(
            [item.model_dump(mode="json") for item in normalized.artifact.facts]
        )
        and str(artifact_row["constraints_json"])
        == _canonical_json(
            [
                item.model_dump(mode="json")
                for item in normalized.artifact.constraints
            ]
        )
        and str(artifact_row["conflicts_json"])
        == _canonical_json(
            [item.model_dump(mode="json") for item in normalized.artifact.conflicts]
        )
        and str(artifact_row["gaps_json"])
        == _canonical_json(
            [item.model_dump(mode="json") for item in normalized.artifact.gaps]
        )
        and str(artifact_row["evidence_refs_json"])
        == _canonical_json(
            [
                item.model_dump(mode="json")
                for item in normalized.artifact.evidence_refs
            ]
        )
        and str(artifact_row["freshness_manifest_sha256"])
        == normalized.result.freshness_manifest_sha256
        and charge is not None
        and int(charge["budget_state_version_after"])
        == stored.budget_state_version
        and str(charge["budget_snapshot_after_json"]) == budget_json
        and str(charge["budget_snapshot_after_sha256"])
        == budget.snapshot_sha256
        and node is not None
        and str(node["status"]) == "completed"
        and int(node["state_version"]) >= stored.node_state_version
        and reservation is not None
        and str(reservation["status"]) == "settled"
        and str(reservation["settled_observation_id"])
        == command.primitive_call_id
        and str(reservation["settled_artifact_id"])
        == command.expected_artifact_id
        and str(reservation["settlement_sha256"])
        == normalized.result.settlement_sha256
        and str(reservation["logical_request_sha256"])
        == command.logical_request_sha256
        and str(reservation["state_guard_sha256"])
        == command.state_guard_sha256
        and _sha256_text(str(reservation["invocation_json"]))
        == str(reservation["invocation_sha256"])
    )
    if not valid:
        raise PlanningArtifactSealPersistenceError(
            "primitive seal exact replay lost immutable authority"
        )


def _require_canonical_json_hash(
    name: str,
    payload: str,
    digest: str,
) -> dict[str, Any]:
    try:
        decoded = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must be a JSON object")
    if _canonical_json(decoded) != payload:
        raise ValueError(f"{name} must use canonical JSON")
    if _sha256_text(payload) != digest:
        raise ValueError(f"{name} hash does not match its JSON")
    return decoded


def _require_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlanningArtifactSealPersistenceError(
            "primitive replay payload has an invalid object boundary"
        )
    return value


def _nested_get(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for name in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(name)
    return current


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _jsonable(value: object) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _jsonable(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key.value if isinstance(key, Enum) else key): _jsonable(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(
                    pair[0].value if isinstance(pair[0], Enum) else pair[0]
                ),
            )
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=_canonical_json)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not canonically serializable: {type(value).__name__}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _model_json(value: BaseModel) -> str:
    return _canonical_json(value.model_dump(mode="json"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_value(value: object) -> str:
    return _sha256_text(_canonical_json(value))


def _derived_id(prefix: str, *parts: str) -> str:
    digest = _sha256_text("\x1f".join((prefix, *parts)))
    return f"{prefix}_{digest[:40]}"


__all__ = [
    "PlanningArtifactSealApplyIdCollision",
    "PlanningArtifactSealPersistenceError",
    'PlanningHostPrimitiveSealResult',
    'SealAuxiliaryHostPrimitiveResultCommand',
    "seal_auxiliary_host_primitive_result",
]
