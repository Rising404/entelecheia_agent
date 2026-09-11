"""AuxiliaryGraph 资源感知调用的持久 I/O 前预留。

一项预留是恰好一个当前 资源感知节点的逻辑调用。Runtime 必须在跨越有限资源读取
边界前预留它并重新检查其状态权威。之后，成功观察会在同一结算事务中把
此行绑定到不可变观察和 PlanningContextArtifact。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import sqlite3
from typing import Literal

from personagraph.l2.planning.invocation_contracts import (
    FrozenPlanningContextArtifactBinding,
    FrozenPlanningContextPrimitiveInvocation,
    PlanningContextPrimitiveKind,
)
from personagraph.l2.work_run import AuxiliaryNodeSubject
from ...deps import StoreDeps


class PlanningPrimitiveInvocationPersistenceError(RuntimeError):
    """原语预留不变量以关闭方式失败。"""


class PlanningPrimitiveInvocationIdentityCollision(
    PlanningPrimitiveInvocationPersistenceError
):
    """调用或节点身份被复用于另一不可变调用。"""


class PlanningPrimitiveInvocationStateGuardRejected(
    PlanningPrimitiveInvocationPersistenceError
):
    """图权威不再匹配 I/O 前调用密封。"""


@dataclass(frozen=True, slots=True)
class StoredPlanningPrimitiveInvocation:
    invocation: FrozenPlanningContextPrimitiveInvocation
    status: Literal["reserved", "settled"]
    invocation_sha256: str
    settled_observation_id: str | None = None
    settled_artifact_id: str | None = None
    settlement_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.invocation, FrozenPlanningContextPrimitiveInvocation
        ):
            raise TypeError(
                "invocation must be a FrozenPlanningContextPrimitiveInvocation"
            )
        if self.invocation_sha256 != planning_primitive_invocation_sha256(
            self.invocation
        ):
            raise ValueError("stored planning primitive invocation hash is invalid")
        if self.status not in {"reserved", "settled"}:
            raise ValueError("stored primitive status is invalid")
        settled = self.status == "settled"
        if settled != all(
            value is not None
            for value in (
                self.settled_observation_id,
                self.settled_artifact_id,
                self.settlement_sha256,
            )
        ):
            raise ValueError("settled primitive projection is incomplete")
        if not settled and any(
            value is not None
            for value in (
                self.settled_observation_id,
                self.settled_artifact_id,
                self.settlement_sha256,
            )
        ):
            raise ValueError("reserved primitive cannot carry settlement fields")
        if settled and (
            self.settled_observation_id
            != self.invocation.binding.primitive_call_id
            or self.settled_artifact_id != self.invocation.binding.artifact_id
        ):
            raise ValueError("primitive settlement differs from its planned identities")
        if self.settlement_sha256 is not None:
            _require_sha256("settlement_sha256", self.settlement_sha256)


@dataclass(frozen=True, slots=True)
class PlanningPrimitiveInvocationMutationResult:
    status: Literal["applied", "replayed"]
    record: StoredPlanningPrimitiveInvocation


def planning_primitive_invocation_sha256(
    invocation: FrozenPlanningContextPrimitiveInvocation,
) -> str:
    if not isinstance(invocation, FrozenPlanningContextPrimitiveInvocation):
        raise TypeError(
            "invocation must be a FrozenPlanningContextPrimitiveInvocation"
        )
    return _sha256_text(_canonical_json(_invocation_payload(invocation)))


def reserve_planning_primitive_invocation(
    deps: StoreDeps,
    *,
    invocation: FrozenPlanningContextPrimitiveInvocation,
) -> PlanningPrimitiveInvocationMutationResult:
    """在任何外部读取前持久化一个精确逻辑调用。"""

    if not isinstance(invocation, FrozenPlanningContextPrimitiveInvocation):
        raise TypeError(
            "invocation must be a FrozenPlanningContextPrimitiveInvocation"
        )
    deps.init_db()
    invocation_payload = _invocation_payload(invocation)
    invocation_json = _canonical_json(invocation_payload)
    invocation_sha256 = _sha256_text(invocation_json)
    now = deps.now()
    binding = invocation.binding
    subject = binding.producer_auxiliary_node
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        existing = _load_by_call_id(conn, binding.primitive_call_id)
        if existing is not None:
            if existing.invocation != invocation:
                raise PlanningPrimitiveInvocationIdentityCollision(
                    "primitive call ID crossed immutable invocation authority"
                )
            return PlanningPrimitiveInvocationMutationResult(
                status="replayed",
                record=existing,
            )
        node_existing = conn.execute(
            "SELECT primitive_call_id FROM "
            "insession_auxiliary_planning_primitive_invocations "
            "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=? "
            "AND auxiliary_node_id=? AND node_revision=?",
            (
                subject.auxiliary_graph_id,
                subject.auxiliary_graph_revision,
                subject.node_id,
                subject.node_revision,
            ),
        ).fetchone()
        if node_existing is not None:
            raise PlanningPrimitiveInvocationIdentityCollision(
                "one Host primitive node version already owns another call"
            )
        budget_ledger_id = _require_current_authority(conn, invocation)
        try:
            conn.execute(
                "INSERT INTO insession_auxiliary_planning_primitive_invocations "
                "(primitive_call_id, session_id, insession_task_id, "
                "auxiliary_graph_id, goal_id, auxiliary_graph_revision, "
                "auxiliary_node_id, node_revision, invocation_turn_id, "
                "primitive_kind, planned_artifact_id, "
                "planned_verification_receipt_id, authority_snapshot_id, "
                "scope_snapshot_sha256, expected_task_state_version, "
                "expected_node_state_version, expected_control_state_version, "
                "expected_goal_state_version, expected_revision_state_version, "
                "expected_budget_state_version, budget_ledger_id, "
                "authority_snapshot_sha256, structure_sha256, "
                "budget_snapshot_sha256, logical_request_json, "
                "logical_request_sha256, state_guard_sha256, invocation_json, "
                "invocation_sha256, status, settled_observation_id, "
                "settled_artifact_id, settlement_sha256, reserved_at, settled_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'reserved', NULL, NULL, "
                "NULL, ?, NULL)",
                (
                    binding.primitive_call_id,
                    binding.session_id,
                    binding.task_id,
                    binding.auxiliary_graph_id,
                    binding.goal_id,
                    subject.auxiliary_graph_revision,
                    subject.node_id,
                    subject.node_revision,
                    invocation.invocation_turn_id,
                    invocation.primitive_kind.value,
                    binding.artifact_id,
                    binding.verification_receipt_id,
                    binding.authority_snapshot_id,
                    binding.scope_snapshot_sha256,
                    invocation.expected_task_state_version,
                    invocation.expected_node_state_version,
                    invocation.expected_control_state_version,
                    invocation.expected_goal_state_version,
                    invocation.expected_revision_state_version,
                    invocation.expected_budget_state_version,
                    budget_ledger_id,
                    invocation.authority_snapshot_sha256,
                    invocation.structure_sha256,
                    invocation.budget_snapshot_sha256,
                    invocation.logical_request_json,
                    invocation.logical_request_sha256,
                    invocation.state_guard_sha256,
                    invocation_json,
                    invocation_sha256,
                    now,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise PlanningPrimitiveInvocationIdentityCollision(
                "primitive reservation collided with current durable authority"
            ) from exc
        record = _load_by_call_id(conn, binding.primitive_call_id)
        if record is None:
            raise PlanningPrimitiveInvocationPersistenceError(
                "primitive reservation disappeared before commit"
            )
        return PlanningPrimitiveInvocationMutationResult(
            status="applied",
            record=record,
        )


def require_planning_primitive_invocation_current(
    deps: StoreDeps,
    *,
    invocation: FrozenPlanningContextPrimitiveInvocation,
) -> None:
    """在外部读取前立即重新检查预留调用。"""

    if not isinstance(invocation, FrozenPlanningContextPrimitiveInvocation):
        raise TypeError(
            "invocation must be a FrozenPlanningContextPrimitiveInvocation"
        )
    deps.init_db()
    with deps.connect() as conn:
        record = _load_by_call_id(conn, invocation.binding.primitive_call_id)
        if record is None or record.invocation != invocation:
            raise PlanningPrimitiveInvocationStateGuardRejected(
                "primitive invocation is not reserved under this authority"
            )
        if record.status != "reserved":
            raise PlanningPrimitiveInvocationStateGuardRejected(
                "settled primitive invocation cannot cross I/O again"
            )
        _require_current_authority(conn, invocation)


def get_planning_primitive_invocation(
    deps: StoreDeps,
    *,
    session_id: str,
    primitive_call_id: str,
) -> StoredPlanningPrimitiveInvocation | None:
    _require_identifier("session_id", session_id)
    _require_identifier("primitive_call_id", primitive_call_id)
    deps.init_db()
    with deps.connect() as conn:
        record = _load_by_call_id(conn, primitive_call_id)
        if record is None:
            return None
        if record.invocation.binding.session_id != session_id:
            return None
        return record


def get_planning_primitive_invocation_for_node(
    deps: StoreDeps,
    *,
    session_id: str,
    subject: AuxiliaryNodeSubject,
) -> StoredPlanningPrimitiveInvocation | None:
    """发现绑定到节点版本的唯一持久逻辑原语。"""

    _require_identifier("session_id", session_id)
    if not isinstance(subject, AuxiliaryNodeSubject):
        raise TypeError("subject must be an AuxiliaryNodeSubject")
    deps.init_db()
    with deps.connect() as conn:
        row = conn.execute(
            "SELECT primitive_call_id FROM "
            "insession_auxiliary_planning_primitive_invocations "
            "WHERE session_id=? AND insession_task_id=? "
            "AND auxiliary_graph_id=? AND auxiliary_graph_revision=? "
            "AND auxiliary_node_id=? AND node_revision=?",
            (
                session_id,
                subject.task_id,
                subject.auxiliary_graph_id,
                subject.auxiliary_graph_revision,
                subject.node_id,
                subject.node_revision,
            ),
        ).fetchone()
        if row is None:
            return None
        return _load_by_call_id(conn, str(row["primitive_call_id"]))


def _require_current_authority(
    conn: sqlite3.Connection,
    invocation: FrozenPlanningContextPrimitiveInvocation,
) -> str:
    binding = invocation.binding
    subject = binding.producer_auxiliary_node
    turn = conn.execute(
        "SELECT turn.status FROM runtime_turns AS turn "
        "JOIN insession_task_turn_links AS link "
        "ON link.session_id=turn.session_id AND link.turn_id=turn.turn_id "
        "WHERE turn.session_id=? AND turn.turn_id=? "
        "AND link.insession_task_id=? LIMIT 1",
        (binding.session_id, invocation.invocation_turn_id, binding.task_id),
    ).fetchone()
    if turn is None or str(turn["status"]) != "running":
        _stale("invocation Turn is not running and linked to the Task")

    task = conn.execute(
        "SELECT current_graph_revision, current_status, state_version "
        "FROM insession_tasks WHERE session_id=? AND insession_task_id=?",
        (binding.session_id, binding.task_id),
    ).fetchone()
    if (
        task is None
        or str(task["current_status"]) in {"completed", "cancelled"}
        or int(task["state_version"]) != invocation.expected_task_state_version
    ):
        _stale("Task state changed before primitive dispatch")

    control = conn.execute(
        "SELECT current_goal_id, current_auxiliary_graph_revision, state_version "
        "FROM insession_auxiliary_graph_v2_containers "
        "WHERE session_id=? AND insession_task_id=? AND auxiliary_graph_id=?",
        (binding.session_id, binding.task_id, binding.auxiliary_graph_id),
    ).fetchone()
    if (
        control is None
        or str(control["current_goal_id"] or "") != binding.goal_id
        or int(control["current_auxiliary_graph_revision"] or 0)
        != subject.auxiliary_graph_revision
        or int(control["state_version"])
        != invocation.expected_control_state_version
    ):
        _stale("AuxiliaryGraph control state changed before primitive dispatch")

    goal = conn.execute(
        "SELECT base_task_graph_revision, budget_ledger_id, status, state_version "
        "FROM insession_auxiliary_graph_goals WHERE session_id=? "
        "AND insession_task_id=? AND auxiliary_graph_id=? AND goal_id=?",
        (
            binding.session_id,
            binding.task_id,
            binding.auxiliary_graph_id,
            binding.goal_id,
        ),
    ).fetchone()
    if (
        goal is None
        or str(goal["status"]) != "active"
        or int(goal["state_version"]) != invocation.expected_goal_state_version
    ):
        _stale("planning goal changed before primitive dispatch")
    current_task_graph_revision = (
        None
        if task["current_graph_revision"] is None
        else int(task["current_graph_revision"])
    )
    base_task_graph_revision = (
        None
        if goal["base_task_graph_revision"] is None
        else int(goal["base_task_graph_revision"])
    )
    if current_task_graph_revision != base_task_graph_revision:
        _stale("planning goal base no longer matches the current TaskGraph")

    revision = conn.execute(
        "SELECT snapshot.authority_snapshot_id, "
        "snapshot.authority_snapshot_sha256, snapshot.structure_sha256, "
        "snapshot.structure_contract_version, state.status, state.state_version "
        "FROM insession_auxiliary_graph_revision_snapshots AS snapshot "
        "JOIN insession_auxiliary_graph_revision_states_v2 AS state "
        "ON state.auxiliary_graph_id=snapshot.auxiliary_graph_id "
        "AND state.auxiliary_graph_revision=snapshot.auxiliary_graph_revision "
        "WHERE snapshot.session_id=? AND snapshot.insession_task_id=? "
        "AND snapshot.auxiliary_graph_id=? AND snapshot.goal_id=? "
        "AND snapshot.auxiliary_graph_revision=?",
        (
            binding.session_id,
            binding.task_id,
            binding.auxiliary_graph_id,
            binding.goal_id,
            subject.auxiliary_graph_revision,
        ),
    ).fetchone()
    if (
        revision is None
        or str(revision["structure_contract_version"])
        != "auxiliary-graph-revision-v2"
        or str(revision["status"]) != "active"
        or int(revision["state_version"])
        != invocation.expected_revision_state_version
        or str(revision["authority_snapshot_id"])
        != binding.authority_snapshot_id
        or str(revision["authority_snapshot_sha256"])
        != invocation.authority_snapshot_sha256
        or str(revision["structure_sha256"]) != invocation.structure_sha256
    ):
        _stale("AuxiliaryGraph revision changed before primitive dispatch")

    node = conn.execute(
        "SELECT definition.executor_kind, definition.output_contract, "
        "state.status, state.state_version FROM "
        "insession_auxiliary_graph_revision_nodes_v2 AS membership "
        "JOIN insession_auxiliary_node_definitions_v2 AS definition "
        "ON definition.auxiliary_graph_id=membership.auxiliary_graph_id "
        "AND definition.auxiliary_node_id=membership.auxiliary_node_id "
        "AND definition.node_revision=membership.node_revision "
        "JOIN insession_auxiliary_node_states_v2 AS state "
        "ON state.auxiliary_graph_id=membership.auxiliary_graph_id "
        "AND state.auxiliary_graph_revision=membership.auxiliary_graph_revision "
        "AND state.auxiliary_node_id=membership.auxiliary_node_id "
        "AND state.node_revision=membership.node_revision "
        "WHERE membership.auxiliary_graph_id=? "
        "AND membership.auxiliary_graph_revision=? "
        "AND membership.auxiliary_node_id=? AND membership.node_revision=?",
        (
            subject.auxiliary_graph_id,
            subject.auxiliary_graph_revision,
            subject.node_id,
            subject.node_revision,
        ),
    ).fetchone()
    if (
        node is None
        or str(node["executor_kind"]) != "host_primitive"
        or not str(node["output_contract"]).startswith("planning_context")
        or str(node["status"]) not in {"proposed", "interrupted"}
        or int(node["state_version"]) != invocation.expected_node_state_version
    ):
        _stale("Host primitive node changed before dispatch")

    budget_ledger_id = str(goal["budget_ledger_id"])
    budget = conn.execute(
        "SELECT current.state_version, current.snapshot_sha256 "
        "FROM insession_auxiliary_goal_budgets AS current "
        "JOIN insession_auxiliary_goal_budget_snapshots AS snapshot "
        "ON snapshot.goal_id=current.goal_id "
        "AND snapshot.budget_ledger_id=current.budget_ledger_id "
        "AND snapshot.state_version=current.state_version "
        "AND snapshot.snapshot_sha256=current.snapshot_sha256 "
        "WHERE current.goal_id=? AND current.budget_ledger_id=?",
        (binding.goal_id, budget_ledger_id),
    ).fetchone()
    if (
        budget is None
        or int(budget["state_version"])
        != invocation.expected_budget_state_version
        or str(budget["snapshot_sha256"]) != invocation.budget_snapshot_sha256
    ):
        _stale("planning budget changed before primitive dispatch")
    return budget_ledger_id


def _load_by_call_id(
    conn: sqlite3.Connection,
    primitive_call_id: str,
) -> StoredPlanningPrimitiveInvocation | None:
    row = conn.execute(
        "SELECT * FROM insession_auxiliary_planning_primitive_invocations "
        "WHERE primitive_call_id=?",
        (primitive_call_id,),
    ).fetchone()
    if row is None:
        return None
    invocation_json = str(row["invocation_json"])
    invocation_sha256 = str(row["invocation_sha256"])
    try:
        payload = json.loads(invocation_json)
    except (TypeError, ValueError) as exc:
        raise PlanningPrimitiveInvocationPersistenceError(
            "stored primitive invocation JSON is invalid"
        ) from exc
    if (
        invocation_json != _canonical_json(payload)
        or invocation_sha256 != _sha256_text(invocation_json)
    ):
        raise PlanningPrimitiveInvocationPersistenceError(
            "stored primitive invocation JSON/hash binding is invalid"
        )
    invocation = _invocation_from_payload(payload)
    if _canonical_json(_invocation_payload(invocation)) != invocation_json:
        raise PlanningPrimitiveInvocationPersistenceError(
            "stored primitive invocation payload is noncanonical"
        )
    expected_columns = {
        "primitive_call_id": invocation.binding.primitive_call_id,
        "session_id": invocation.binding.session_id,
        "insession_task_id": invocation.binding.task_id,
        "auxiliary_graph_id": invocation.binding.auxiliary_graph_id,
        "goal_id": invocation.binding.goal_id,
        "auxiliary_graph_revision": (
            invocation.binding.producer_auxiliary_node.auxiliary_graph_revision
        ),
        "auxiliary_node_id": invocation.binding.producer_auxiliary_node.node_id,
        "node_revision": invocation.binding.producer_auxiliary_node.node_revision,
        "invocation_turn_id": invocation.invocation_turn_id,
        "primitive_kind": invocation.primitive_kind.value,
        "planned_artifact_id": invocation.binding.artifact_id,
        "planned_verification_receipt_id": (
            invocation.binding.verification_receipt_id
        ),
        "authority_snapshot_id": invocation.binding.authority_snapshot_id,
        "scope_snapshot_sha256": invocation.binding.scope_snapshot_sha256,
        "expected_task_state_version": invocation.expected_task_state_version,
        "expected_node_state_version": invocation.expected_node_state_version,
        "expected_control_state_version": invocation.expected_control_state_version,
        "expected_goal_state_version": invocation.expected_goal_state_version,
        "expected_revision_state_version": invocation.expected_revision_state_version,
        "expected_budget_state_version": invocation.expected_budget_state_version,
        "authority_snapshot_sha256": invocation.authority_snapshot_sha256,
        "structure_sha256": invocation.structure_sha256,
        "budget_snapshot_sha256": invocation.budget_snapshot_sha256,
        "logical_request_json": invocation.logical_request_json,
        "logical_request_sha256": invocation.logical_request_sha256,
        "state_guard_sha256": invocation.state_guard_sha256,
    }
    if any(str(row[name]) != str(value) for name, value in expected_columns.items()):
        raise PlanningPrimitiveInvocationPersistenceError(
            "stored primitive columns differ from their sealed invocation"
        )
    return StoredPlanningPrimitiveInvocation(
        invocation=invocation,
        status=str(row["status"]),  # type: ignore[arg-type]
        invocation_sha256=invocation_sha256,
        settled_observation_id=(
            None
            if row["settled_observation_id"] is None
            else str(row["settled_observation_id"])
        ),
        settled_artifact_id=(
            None
            if row["settled_artifact_id"] is None
            else str(row["settled_artifact_id"])
        ),
        settlement_sha256=(
            None
            if row["settlement_sha256"] is None
            else str(row["settlement_sha256"])
        ),
    )


def _invocation_payload(
    invocation: FrozenPlanningContextPrimitiveInvocation,
) -> dict[str, object]:
    logical = json.loads(invocation.logical_request_json)
    return {
        "schema_version": "frozen-planning-context-primitive-invocation-v1",
        "primitive_kind": invocation.primitive_kind.value,
        "binding": logical["binding"],
        "invocation_turn_id": invocation.invocation_turn_id,
        "expected_task_state_version": invocation.expected_task_state_version,
        "expected_node_state_version": invocation.expected_node_state_version,
        "expected_control_state_version": invocation.expected_control_state_version,
        "expected_goal_state_version": invocation.expected_goal_state_version,
        "expected_revision_state_version": invocation.expected_revision_state_version,
        "expected_budget_state_version": invocation.expected_budget_state_version,
        "authority_snapshot_sha256": invocation.authority_snapshot_sha256,
        "structure_sha256": invocation.structure_sha256,
        "budget_snapshot_sha256": invocation.budget_snapshot_sha256,
        "logical_request_json": invocation.logical_request_json,
        "logical_request_sha256": invocation.logical_request_sha256,
        "state_guard_sha256": invocation.state_guard_sha256,
    }


def _invocation_from_payload(
    payload: object,
) -> FrozenPlanningContextPrimitiveInvocation:
    if not isinstance(payload, dict):
        raise PlanningPrimitiveInvocationPersistenceError(
            "stored primitive invocation payload must be an object"
        )
    try:
        binding_payload = payload["binding"]
        if not isinstance(binding_payload, dict):
            raise TypeError("binding is not an object")
        producer = AuxiliaryNodeSubject.model_validate(
            binding_payload["producer_auxiliary_node"]
        )
        binding = FrozenPlanningContextArtifactBinding(
            session_id=str(binding_payload["session_id"]),
            task_id=str(binding_payload["task_id"]),
            auxiliary_graph_id=str(binding_payload["auxiliary_graph_id"]),
            goal_id=str(binding_payload["goal_id"]),
            producer_auxiliary_node=producer,
            primitive_call_id=str(binding_payload["primitive_call_id"]),
            artifact_id=str(binding_payload["artifact_id"]),
            verification_receipt_id=str(
                binding_payload["verification_receipt_id"]
            ),
            authority_snapshot_id=str(binding_payload["authority_snapshot_id"]),
            scope_snapshot_sha256=str(
                binding_payload["scope_snapshot_sha256"]
            ),
            alias_prefix=str(binding_payload["alias_prefix"]),
            artifact_alias=str(binding_payload["artifact_alias"]),
            producer_node_alias=str(binding_payload["producer_node_alias"]),
            affected_obligations=tuple(binding_payload["affected_obligations"]),
        )
        return FrozenPlanningContextPrimitiveInvocation(
            primitive_kind=PlanningContextPrimitiveKind(
                str(payload["primitive_kind"])
            ),
            binding=binding,
            invocation_turn_id=str(payload["invocation_turn_id"]),
            expected_task_state_version=int(
                payload["expected_task_state_version"]
            ),
            expected_node_state_version=int(
                payload["expected_node_state_version"]
            ),
            expected_control_state_version=int(
                payload["expected_control_state_version"]
            ),
            expected_goal_state_version=int(
                payload["expected_goal_state_version"]
            ),
            expected_revision_state_version=int(
                payload["expected_revision_state_version"]
            ),
            expected_budget_state_version=int(
                payload["expected_budget_state_version"]
            ),
            authority_snapshot_sha256=str(
                payload["authority_snapshot_sha256"]
            ),
            structure_sha256=str(payload["structure_sha256"]),
            budget_snapshot_sha256=str(payload["budget_snapshot_sha256"]),
            logical_request_json=str(payload["logical_request_json"]),
            logical_request_sha256=str(payload["logical_request_sha256"]),
            state_guard_sha256=str(payload["state_guard_sha256"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PlanningPrimitiveInvocationPersistenceError(
            "stored primitive invocation cannot be revalidated"
        ) from exc


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_identifier(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 200
    ):
        raise ValueError(f"{name} must be a canonical identifier")


def _require_sha256(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _stale(message: str) -> None:
    raise PlanningPrimitiveInvocationStateGuardRejected(message)


__all__ = [
    "PlanningPrimitiveInvocationIdentityCollision",
    'PlanningPrimitiveInvocationMutationResult',
    "PlanningPrimitiveInvocationPersistenceError",
    "PlanningPrimitiveInvocationStateGuardRejected",
    'StoredPlanningPrimitiveInvocation',
    "get_planning_primitive_invocation",
    "get_planning_primitive_invocation_for_node",
    "planning_primitive_invocation_sha256",
    "require_planning_primitive_invocation_current",
    "reserve_planning_primitive_invocation",
]
