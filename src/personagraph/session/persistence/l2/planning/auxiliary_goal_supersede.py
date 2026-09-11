"""对一个 AuxiliaryGraph 规划目标执行原子且可安全重放的取代。

此事务刻意在创建替代目标前停止。其回执认证旧权威、取代原因，以及之后新目标引导可消费
的 base 与 objective。
"""

from __future__ import annotations

import sqlite3
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personagraph.l2.auxiliary_graph import PlanningEpisodeBudget
from ...deps import StoreDeps
from ..task_graph.insession_tasks import _load_authoritative_user_input
from ..work_run.work_execution import (
    _model_json,
    _payload_hash,
    _require_identifier,
    _require_positive,
    _text_hash,
)


_NONTERMINAL_GOAL_STATUSES = {
    "active",
    "waiting_user",
    "waiting_authorization",
    "waiting_external",
    "interrupted",
    "proposal_ready",
    "gapped_ready",
}
_NONTERMINAL_REVISION_STATUSES = _NONTERMINAL_GOAL_STATUSES


class AuxiliaryGoalSupersedePersistenceError(RuntimeError):
    """无法认证并提交目标取代。"""


class AuxiliaryGoalSupersedeIdentityCollision(
    AuxiliaryGoalSupersedePersistenceError
):
    """应用身份已绑定到另一不可变载荷。"""


class AuxiliaryGoalSupersedeStaleAuthority(
    AuxiliaryGoalSupersedePersistenceError
):
    """调用方 CAS 不再匹配当前目标权威。"""


class AuxiliaryGoalSupersedeUnsafeWorkRun(
    AuxiliaryGoalSupersedePersistenceError
):
    """待处理或不确定的 WorkRun 阻止安全目标取代。"""


class AuxiliaryGoalSupersedeStoredAuthorityCorrupt(
    AuxiliaryGoalSupersedePersistenceError
):
    """已存储取代回执不再能认证其权威。"""


class PlanningGoalSupersedeReason(StrEnum):
    BASE_DRIFT = "base_drift"
    USER_TARGET_CHANGED = "user_target_changed"


class _GoalSupersedeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PlanningGoalSupersedeSourceBinding(_GoalSupersedeRecord):
    source_turn_id: str = Field(min_length=1, max_length=200)
    source_start: int = Field(ge=0)
    source_end: int = Field(ge=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_excerpt: str = Field(min_length=1, max_length=20_000)
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        source_turn_id: str,
        source_start: int,
        source_end: int,
        source_sha256: str,
        source_excerpt: str,
    ) -> "PlanningGoalSupersedeSourceBinding":
        payload = {
            "source_turn_id": source_turn_id,
            "source_start": source_start,
            "source_end": source_end,
            "source_sha256": source_sha256,
            "source_excerpt": source_excerpt,
        }
        return cls(**payload, binding_sha256=_payload_hash(payload))

    @model_validator(mode="after")
    def _validate_binding(self) -> "PlanningGoalSupersedeSourceBinding":
        if self.source_end - self.source_start != len(self.source_excerpt):
            raise ValueError("supersede source span length is inconsistent")
        if _text_hash(self.source_excerpt) != self.source_sha256:
            raise ValueError("supersede source excerpt hash is inconsistent")
        payload = self.model_dump(mode="json", exclude={"binding_sha256"})
        if _payload_hash(payload) != self.binding_sha256:
            raise ValueError("supersede source binding hash is inconsistent")
        return self


class SupersedeAuxiliaryPlanningGoalCommand(_GoalSupersedeRecord):
    contract_version: Literal["auxiliary-v2-goal-supersede-command-v1"] = (
        "auxiliary-v2-goal-supersede-command-v1"
    )
    apply_id: str = Field(min_length=1, max_length=200)
    session_id: str = Field(min_length=1, max_length=200)
    invocation_turn_id: str = Field(min_length=1, max_length=200)
    task_id: str = Field(min_length=1, max_length=200)
    auxiliary_graph_id: str = Field(min_length=1, max_length=200)
    goal_id: str = Field(min_length=1, max_length=200)
    reason: PlanningGoalSupersedeReason
    expected_task_state_version: int = Field(ge=1)
    expected_control_state_version: int = Field(ge=1)
    expected_goal_state_version: int = Field(ge=1)
    expected_revision_state_version: int = Field(ge=1)
    expected_budget_state_version: int = Field(ge=1)
    expected_current_auxiliary_graph_revision: int = Field(ge=1)
    expected_base_task_graph_revision: int | None = Field(default=None, ge=1)
    observed_task_graph_revision: int | None = Field(default=None, ge=1)
    replacement_objective: str | None = Field(
        default=None,
        min_length=1,
        max_length=2_000,
    )
    source_turn_id: str | None = Field(default=None, min_length=1, max_length=200)
    source_start: int | None = Field(default=None, ge=0)
    source_end: int | None = Field(default=None, ge=1)
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_reason_authority(
        self,
    ) -> "SupersedeAuxiliaryPlanningGoalCommand":
        source_values = (
            self.source_turn_id,
            self.source_start,
            self.source_end,
            self.source_sha256,
        )
        if self.reason is PlanningGoalSupersedeReason.BASE_DRIFT:
            if self.observed_task_graph_revision is None:
                raise ValueError("BASE_DRIFT requires the observed TaskGraph revision")
            if self.replacement_objective is not None or any(
                value is not None for value in source_values
            ):
                raise ValueError("BASE_DRIFT cannot carry user target authority")
        else:
            if self.replacement_objective is None or any(
                value is None for value in source_values
            ):
                raise ValueError(
                    "USER_TARGET_CHANGED requires objective and exact source binding"
                )
            if self.source_turn_id != self.invocation_turn_id:
                raise ValueError(
                    "target-change authority must come from the invocation Turn"
                )
            assert self.source_start is not None and self.source_end is not None
            if self.source_start >= self.source_end:
                raise ValueError("target-change source span must be non-empty")
        return self


class PlanningGoalSupersedeReceipt(_GoalSupersedeRecord):
    contract_version: Literal["auxiliary-v2-goal-supersede-receipt-v1"] = (
        "auxiliary-v2-goal-supersede-receipt-v1"
    )
    apply_id: str
    session_id: str
    invocation_turn_id: str
    task_id: str
    auxiliary_graph_id: str
    superseded_goal_id: str
    reason: PlanningGoalSupersedeReason
    source_binding: PlanningGoalSupersedeSourceBinding | None = None
    superseded_auxiliary_graph_revision: int = Field(ge=1)
    previous_base_task_graph_revision: int | None = Field(default=None, ge=1)
    observed_task_graph_revision: int | None = Field(default=None, ge=1)
    next_base_task_graph_revision: int | None = Field(default=None, ge=1)
    next_target_task_graph_revision: int = Field(ge=1)
    next_goal_objective: str = Field(min_length=1, max_length=2_000)
    source_structure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_authority_snapshot_id: str
    source_authority_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    budget_ledger_id: str
    budget_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_state_version_before: int = Field(ge=1)
    task_state_version_after: int = Field(ge=1)
    control_state_version_before: int = Field(ge=1)
    control_state_version_after: int = Field(ge=2)
    goal_state_version_before: int = Field(ge=1)
    goal_state_version_after: int = Field(ge=2)
    revision_state_version_before: int = Field(ge=1)
    revision_state_version_after: int = Field(ge=2)
    budget_state_version_before: int = Field(ge=1)
    budget_state_version_after: int = Field(ge=1)
    receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, **values: object) -> "PlanningGoalSupersedeReceipt":
        payload = {
            "contract_version": "auxiliary-v2-goal-supersede-receipt-v1",
            **values,
        }
        hash_payload = {
            key: (
                value.model_dump(mode="json")
                if isinstance(value, BaseModel)
                else value
            )
            for key, value in payload.items()
        }
        return cls(**payload, receipt_sha256=_payload_hash(hash_payload))

    @model_validator(mode="after")
    def _validate_receipt(self) -> "PlanningGoalSupersedeReceipt":
        if self.task_state_version_after != self.task_state_version_before:
            raise ValueError("goal supersede must not mutate Task authority")
        if self.control_state_version_after != self.control_state_version_before + 1:
            raise ValueError("goal supersede control version transition is invalid")
        if self.goal_state_version_after != self.goal_state_version_before + 1:
            raise ValueError("goal supersede goal version transition is invalid")
        if (
            self.revision_state_version_after
            != self.revision_state_version_before + 1
        ):
            raise ValueError("goal supersede revision version transition is invalid")
        if self.budget_state_version_after != self.budget_state_version_before:
            raise ValueError("goal supersede must not charge or rewrite budget")
        expected_target = (
            1
            if self.next_base_task_graph_revision is None
            else self.next_base_task_graph_revision + 1
        )
        if self.next_target_task_graph_revision != expected_target:
            raise ValueError("goal supersede next TaskGraph target is inconsistent")
        if (
            self.reason is PlanningGoalSupersedeReason.BASE_DRIFT
            and self.source_binding is not None
        ):
            raise ValueError("BASE_DRIFT receipt cannot carry a user source")
        if (
            self.reason is PlanningGoalSupersedeReason.USER_TARGET_CHANGED
            and self.source_binding is None
        ):
            raise ValueError("target-change receipt requires a user source")
        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if _payload_hash(payload) != self.receipt_sha256:
            raise ValueError("goal supersede receipt hash is inconsistent")
        return self


class SupersedeAuxiliaryPlanningGoalResult(_GoalSupersedeRecord):
    status: Literal["applied", "replayed"]
    receipt: PlanningGoalSupersedeReceipt


def require_authenticated_planning_goal_supersede_receipt(
    deps: StoreDeps,
    *,
    receipt: PlanningGoalSupersedeReceipt,
) -> PlanningGoalSupersedeReceipt:
    """根据不可变 Store 权威重新校验一个回执。

    自洽 Pydantic 值不足以充当后继权威：调用方无需真正取代引用目标，就能构造并哈希此值。
    本次读取要求精确密封应用回执，以及该事务生成的旧目标、revision 和预算终态。
    """

    if not isinstance(receipt, PlanningGoalSupersedeReceipt):
        raise TypeError("receipt must be PlanningGoalSupersedeReceipt")
    try:
        admitted = PlanningGoalSupersedeReceipt.model_validate_json(
            receipt.model_dump_json()
        )
    except (TypeError, ValueError) as exc:
        raise AuxiliaryGoalSupersedeStoredAuthorityCorrupt(
            "supersede receipt contract is invalid"
        ) from exc
    deps.init_db()
    with deps.connect() as conn:
        row = conn.execute(
            "SELECT * FROM insession_auxiliary_graph_revision_apply_receipts_v2 "
            "WHERE apply_id=?",
            (admitted.apply_id,),
        ).fetchone()
        if row is None:
            raise AuxiliaryGoalSupersedeStoredAuthorityCorrupt(
                "supersede receipt has no sealed Store apply authority"
            )
        return _authenticate_stored_supersede_receipt(
            conn,
            row=row,
            expected=admitted,
        )


def get_pending_auxiliary_goal_supersede_receipt(
    deps: StoreDeps,
    *,
    session_id: str,
    task_id: str,
) -> PlanningGoalSupersedeReceipt | None:
    """返回等待后继目标的已认证取代回执。

    pending 表示容器仍指向精确的已取代目标和 revision，并带有回执提交的控制版本。一旦
    新目标推进该指针，此查询会刻意返回 ``None``。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("task_id", task_id)
    deps.init_db()
    with deps.connect() as conn:
        rows = conn.execute(
            "SELECT receipt.* FROM "
            "insession_auxiliary_graph_revision_apply_receipts_v2 AS receipt "
            "JOIN insession_auxiliary_graph_v2_containers AS control "
            "ON control.auxiliary_graph_id=receipt.auxiliary_graph_id "
            "AND control.session_id=receipt.session_id "
            "AND control.insession_task_id=receipt.insession_task_id "
            "AND control.current_goal_id=receipt.goal_id "
            "AND control.current_auxiliary_graph_revision="
            "receipt.committed_auxiliary_graph_revision "
            "AND control.state_version=receipt.committed_control_state_version "
            "WHERE receipt.operation='supersede_goal' "
            "AND receipt.session_id=? AND receipt.insession_task_id=? "
            "ORDER BY receipt.created_at DESC, receipt.apply_id DESC LIMIT 2",
            (session_id, task_id),
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise AuxiliaryGoalSupersedeStoredAuthorityCorrupt(
                "multiple supersede receipts claim the current control pointer"
            )
        return _authenticate_stored_supersede_receipt(
            conn,
            row=rows[0],
            expected=None,
        )


def get_authenticated_user_target_change_receipt(
    deps: StoreDeps,
    *,
    session_id: str,
    task_id: str,
    invocation_turn_id: str,
    replacement_objective: str,
    source_start: int,
    source_end: int,
    source_sha256: str,
) -> PlanningGoalSupersedeReceipt | None:
    """重放绑定到已接受目标变更通道意图的唯一回执。

    与待处理指针查询不同，后继目标推进容器后此回执仍可读取。这一区别可防止重放同一已
    接受 Turn 时取代其自己的后继目标。
    """

    _require_identifier("session_id", session_id)
    _require_identifier("task_id", task_id)
    _require_identifier("invocation_turn_id", invocation_turn_id)
    if (
        not isinstance(replacement_objective, str)
        or not replacement_objective.strip()
        or replacement_objective != replacement_objective.strip()
        or len(replacement_objective) > 2_000
    ):
        raise ValueError("replacement_objective must be canonical bounded text")
    if (
        isinstance(source_start, bool)
        or not isinstance(source_start, int)
        or isinstance(source_end, bool)
        or not isinstance(source_end, int)
        or not 0 <= source_start < source_end
    ):
        raise ValueError("target-change source span must be non-empty")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in source_sha256)
    ):
        raise ValueError("target-change source hash must be lowercase sha256")
    deps.init_db()
    with deps.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM "
            "insession_auxiliary_graph_revision_apply_receipts_v2 "
            "WHERE operation='supersede_goal' AND session_id=? "
            "AND insession_task_id=? AND invocation_turn_id=? "
            "ORDER BY created_at DESC, apply_id DESC LIMIT 2",
            (session_id, task_id, invocation_turn_id),
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise AuxiliaryGoalSupersedeStoredAuthorityCorrupt(
                "multiple supersede receipts claim one target-change Turn"
            )
        receipt = _authenticate_stored_supersede_receipt(
            conn,
            row=rows[0],
            expected=None,
        )
        source = receipt.source_binding
        if (
            receipt.reason is not PlanningGoalSupersedeReason.USER_TARGET_CHANGED
            or receipt.next_goal_objective != replacement_objective
            or source is None
            or source.source_turn_id != invocation_turn_id
            or source.source_start != source_start
            or source.source_end != source_end
            or source.source_sha256 != source_sha256
        ):
            raise AuxiliaryGoalSupersedeStoredAuthorityCorrupt(
                "stored supersede receipt disagrees with target-change lane authority"
            )
        return receipt


def supersede_auxiliary_planning_goal(
    deps: StoreDeps,
    *,
    command: SupersedeAuxiliaryPlanningGoalCommand,
) -> SupersedeAuxiliaryPlanningGoalResult:
    """原子取代当前目标与 revision，并密封其回执。"""

    if not isinstance(command, SupersedeAuxiliaryPlanningGoalCommand):
        raise TypeError(
            "command must be SupersedeAuxiliaryPlanningGoalCommand"
        )
    for name in (
        "apply_id",
        "session_id",
        "invocation_turn_id",
        "task_id",
        "auxiliary_graph_id",
        "goal_id",
    ):
        _require_identifier(name, str(getattr(command, name)))
    for name in (
        "expected_task_state_version",
        "expected_control_state_version",
        "expected_goal_state_version",
        "expected_revision_state_version",
        "expected_budget_state_version",
        "expected_current_auxiliary_graph_revision",
    ):
        _require_positive(name, int(getattr(command, name)))

    payload_hash = _payload_hash(command.model_dump(mode="json", exclude={"apply_id"}))
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = conn.execute(
            "SELECT * FROM insession_auxiliary_graph_revision_apply_receipts_v2 "
            "WHERE apply_id=?",
            (command.apply_id,),
        ).fetchone()
        if replay is not None:
            return _load_exact_replay(
                conn,
                command=command,
                payload_hash=payload_hash,
                row=replay,
            )

        _require_no_active_task_graph_revision_authority(
            conn,
            command=command,
        )

        task = conn.execute(
            "SELECT current_graph_revision, current_status, state_version "
            "FROM insession_tasks WHERE session_id=? AND insession_task_id=?",
            (command.session_id, command.task_id),
        ).fetchone()
        if task is None:
            raise AuxiliaryGoalSupersedeStaleAuthority("unknown Task authority")
        actual_task_version = int(task["state_version"])
        actual_task_graph_revision = (
            int(task["current_graph_revision"])
            if task["current_graph_revision"] is not None
            else None
        )
        if actual_task_version != command.expected_task_state_version:
            raise AuxiliaryGoalSupersedeStaleAuthority(
                "Task state version changed before goal supersede"
            )
        if actual_task_graph_revision != command.observed_task_graph_revision:
            raise AuxiliaryGoalSupersedeStaleAuthority(
                "observed TaskGraph revision is stale"
            )
        if str(task["current_status"]) in {"completed", "cancelled"}:
            raise AuxiliaryGoalSupersedeStaleAuthority(
                "terminal Task cannot authorize a fresh planning goal"
            )
        _require_linked_invocation_turn(conn, command=command)

        control = conn.execute(
            "SELECT current_goal_id, current_auxiliary_graph_revision, "
            "state_version FROM insession_auxiliary_graph_v2_containers "
            "WHERE session_id=? AND insession_task_id=? AND auxiliary_graph_id=?",
            (command.session_id, command.task_id, command.auxiliary_graph_id),
        ).fetchone()
        if control is None:
            raise AuxiliaryGoalSupersedeStaleAuthority(
                "AuxiliaryGraph control authority is missing"
            )
        if (
            str(control["current_goal_id"]) != command.goal_id
            or int(control["current_auxiliary_graph_revision"])
            != command.expected_current_auxiliary_graph_revision
            or int(control["state_version"])
            != command.expected_control_state_version
        ):
            raise AuxiliaryGoalSupersedeStaleAuthority(
                "AuxiliaryGraph control CAS changed before goal supersede"
            )

        goal = conn.execute(
            "SELECT objective, base_task_graph_revision, target_task_graph_revision, "
            "authorization_manifest_id, authorization_manifest_sha256, "
            "budget_ledger_id, status, state_version "
            "FROM insession_auxiliary_graph_goals WHERE session_id=? "
            "AND insession_task_id=? AND auxiliary_graph_id=? AND goal_id=?",
            (
                command.session_id,
                command.task_id,
                command.auxiliary_graph_id,
                command.goal_id,
            ),
        ).fetchone()
        if goal is None:
            raise AuxiliaryGoalSupersedeStaleAuthority(
                "current planning goal authority is missing"
            )
        stored_base = (
            int(goal["base_task_graph_revision"])
            if goal["base_task_graph_revision"] is not None
            else None
        )
        goal_status = str(goal["status"])
        if (
            int(goal["state_version"]) != command.expected_goal_state_version
            or stored_base != command.expected_base_task_graph_revision
            or goal_status not in _NONTERMINAL_GOAL_STATUSES
        ):
            raise AuxiliaryGoalSupersedeStaleAuthority(
                "planning goal CAS or nonterminal status changed"
            )

        revision = conn.execute(
            "SELECT snapshot.structure_sha256, snapshot.authority_snapshot_id, "
            "snapshot.authority_snapshot_sha256, state.status, state.state_version "
            "FROM insession_auxiliary_graph_revision_snapshots AS snapshot "
            "JOIN insession_auxiliary_graph_revision_states_v2 AS state "
            "ON state.auxiliary_graph_id=snapshot.auxiliary_graph_id "
            "AND state.auxiliary_graph_revision=snapshot.auxiliary_graph_revision "
            "WHERE snapshot.auxiliary_graph_id=? "
            "AND snapshot.auxiliary_graph_revision=? "
            "AND snapshot.insession_task_id=? AND snapshot.goal_id=?",
            (
                command.auxiliary_graph_id,
                command.expected_current_auxiliary_graph_revision,
                command.task_id,
                command.goal_id,
            ),
        ).fetchone()
        if revision is None:
            raise AuxiliaryGoalSupersedeStaleAuthority(
                "current revision authority is missing"
            )
        revision_status = str(revision["status"])
        if (
            int(revision["state_version"])
            != command.expected_revision_state_version
            or revision_status not in _NONTERMINAL_REVISION_STATUSES
        ):
            raise AuxiliaryGoalSupersedeStaleAuthority(
                "planning revision CAS or nonterminal status changed"
            )

        budget_row = conn.execute(
            "SELECT budget_ledger_id, snapshot_json, snapshot_sha256, "
            "state_version FROM insession_auxiliary_goal_budgets "
            "WHERE session_id=? AND insession_task_id=? "
            "AND auxiliary_graph_id=? AND goal_id=?",
            (
                command.session_id,
                command.task_id,
                command.auxiliary_graph_id,
                command.goal_id,
            ),
        ).fetchone()
        if budget_row is None:
            raise AuxiliaryGoalSupersedeStaleAuthority(
                "planning goal budget authority is missing"
            )
        if (
            int(budget_row["state_version"])
            != command.expected_budget_state_version
            or str(budget_row["budget_ledger_id"]) != str(goal["budget_ledger_id"])
        ):
            raise AuxiliaryGoalSupersedeStaleAuthority(
                "planning budget CAS changed before goal supersede"
            )
        budget = _validate_budget_row(budget_row, expected_goal_id=command.goal_id)

        source_binding = _derive_reason_authority(
            conn,
            command=command,
            stored_base=stored_base,
            stored_objective=str(goal["objective"]),
        )
        _require_safe_work_run_checkpoint(conn, command=command)

        next_objective = (
            str(goal["objective"])
            if command.reason is PlanningGoalSupersedeReason.BASE_DRIFT
            else str(command.replacement_objective).strip()
        )
        next_base = actual_task_graph_revision
        receipt = PlanningGoalSupersedeReceipt.create(
            apply_id=command.apply_id,
            session_id=command.session_id,
            invocation_turn_id=command.invocation_turn_id,
            task_id=command.task_id,
            auxiliary_graph_id=command.auxiliary_graph_id,
            superseded_goal_id=command.goal_id,
            reason=command.reason,
            source_binding=source_binding,
            superseded_auxiliary_graph_revision=(
                command.expected_current_auxiliary_graph_revision
            ),
            previous_base_task_graph_revision=stored_base,
            observed_task_graph_revision=actual_task_graph_revision,
            next_base_task_graph_revision=next_base,
            next_target_task_graph_revision=(
                1 if next_base is None else next_base + 1
            ),
            next_goal_objective=next_objective,
            source_structure_sha256=str(revision["structure_sha256"]),
            source_authority_snapshot_id=str(revision["authority_snapshot_id"]),
            source_authority_snapshot_sha256=str(
                revision["authority_snapshot_sha256"]
            ),
            budget_ledger_id=str(budget_row["budget_ledger_id"]),
            budget_snapshot_sha256=budget.snapshot_sha256,
            task_state_version_before=actual_task_version,
            task_state_version_after=actual_task_version,
            control_state_version_before=command.expected_control_state_version,
            control_state_version_after=command.expected_control_state_version + 1,
            goal_state_version_before=command.expected_goal_state_version,
            goal_state_version_after=command.expected_goal_state_version + 1,
            revision_state_version_before=command.expected_revision_state_version,
            revision_state_version_after=(
                command.expected_revision_state_version + 1
            ),
            budget_state_version_before=command.expected_budget_state_version,
            budget_state_version_after=command.expected_budget_state_version,
        )

        if conn.execute(
            "UPDATE insession_auxiliary_graph_revision_states_v2 "
            "SET status='superseded', state_version=state_version+1, updated_at=? "
            "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=? "
            "AND status=? AND state_version=?",
            (
                now,
                command.auxiliary_graph_id,
                command.expected_current_auxiliary_graph_revision,
                revision_status,
                command.expected_revision_state_version,
            ),
        ).rowcount != 1:
            raise AuxiliaryGoalSupersedeStaleAuthority(
                "revision changed during goal supersede"
            )
        if conn.execute(
            "UPDATE insession_auxiliary_graph_goals SET status='superseded', "
            "state_version=state_version+1, updated_at=? WHERE goal_id=? "
            "AND auxiliary_graph_id=? AND status=? AND state_version=?",
            (
                now,
                command.goal_id,
                command.auxiliary_graph_id,
                goal_status,
                command.expected_goal_state_version,
            ),
        ).rowcount != 1:
            raise AuxiliaryGoalSupersedeStaleAuthority(
                "goal changed during supersede"
            )
        if conn.execute(
            "UPDATE insession_auxiliary_graph_v2_containers "
            "SET state_version=state_version+1, updated_at=? "
            "WHERE auxiliary_graph_id=? AND session_id=? "
            "AND insession_task_id=? AND current_goal_id=? "
            "AND current_auxiliary_graph_revision=? AND state_version=?",
            (
                now,
                command.auxiliary_graph_id,
                command.session_id,
                command.task_id,
                command.goal_id,
                command.expected_current_auxiliary_graph_revision,
                command.expected_control_state_version,
            ),
        ).rowcount != 1:
            raise AuxiliaryGoalSupersedeStaleAuthority(
                "control changed during goal supersede"
            )

        result_json = _model_json(receipt)
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
            "(?, 'supersede_goal', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?)",
            (
                command.apply_id,
                command.session_id,
                command.task_id,
                command.auxiliary_graph_id,
                command.goal_id,
                command.invocation_turn_id,
                command.expected_control_state_version,
                receipt.control_state_version_after,
                command.expected_current_auxiliary_graph_revision,
                command.expected_current_auxiliary_graph_revision,
                command.expected_goal_state_version,
                receipt.goal_state_version_after,
                command.expected_budget_state_version,
                receipt.budget_state_version_after,
                str(budget_row["budget_ledger_id"]),
                str(budget_row["snapshot_json"]),
                budget.snapshot_sha256,
                payload_hash,
                result_json,
                _text_hash(result_json),
                now,
            ),
        )
        return SupersedeAuxiliaryPlanningGoalResult(
            status="applied",
            receipt=receipt,
        )


def _require_no_active_task_graph_revision_authority(
    conn: sqlite3.Connection,
    *,
    command: SupersedeAuxiliaryPlanningGoalCommand,
) -> None:
    """相对于 TaskGraph 重规划，保持目标取代单向。"""

    whole_task = conn.execute(
        "SELECT trigger_id FROM insession_active_task_graph_revision_triggers "
        "WHERE session_id=? AND insession_task_id=?",
        (command.session_id, command.task_id),
    ).fetchone()
    execution = conn.execute(
        "SELECT request_id FROM "
        "insession_active_task_graph_execution_replan_requests "
        "WHERE session_id=? AND insession_task_id=?",
        (command.session_id, command.task_id),
    ).fetchone()
    if whole_task is not None or execution is not None:
        raise AuxiliaryGoalSupersedeStaleAuthority(
            "Task owns an active TaskGraph revision authority"
        )


def _require_linked_invocation_turn(
    conn: sqlite3.Connection,
    *,
    command: SupersedeAuxiliaryPlanningGoalCommand,
) -> None:
    if conn.execute(
        "SELECT 1 FROM runtime_turns WHERE session_id=? AND turn_id=?",
        (command.session_id, command.invocation_turn_id),
    ).fetchone() is None:
        raise AuxiliaryGoalSupersedeStaleAuthority("unknown invocation Turn")
    if conn.execute(
        "SELECT 1 FROM insession_task_turn_links WHERE session_id=? "
        "AND turn_id=? AND insession_task_id=?",
        (command.session_id, command.invocation_turn_id, command.task_id),
    ).fetchone() is None:
        raise AuxiliaryGoalSupersedeStaleAuthority(
            "invocation Turn is not linked to the Task"
        )


def _derive_reason_authority(
    conn: sqlite3.Connection,
    *,
    command: SupersedeAuxiliaryPlanningGoalCommand,
    stored_base: int | None,
    stored_objective: str,
) -> PlanningGoalSupersedeSourceBinding | None:
    if command.reason is PlanningGoalSupersedeReason.BASE_DRIFT:
        if command.observed_task_graph_revision == stored_base:
            raise AuxiliaryGoalSupersedeStaleAuthority(
                "BASE_DRIFT requires a TaskGraph revision different from goal base"
            )
        if (
            stored_base is not None
            and command.observed_task_graph_revision is not None
            and command.observed_task_graph_revision <= stored_base
        ):
            raise AuxiliaryGoalSupersedeStaleAuthority(
                "BASE_DRIFT must advance beyond the frozen goal base"
            )
        return None
    if command.observed_task_graph_revision != stored_base:
        raise AuxiliaryGoalSupersedeStaleAuthority(
            "target change cannot hide a simultaneous TaskGraph base drift"
        )
    objective = str(command.replacement_objective).strip()
    if not objective or objective == stored_objective.strip():
        raise AuxiliaryGoalSupersedeStaleAuthority(
            "target change must provide a materially different objective"
        )
    assert command.source_turn_id is not None
    assert command.source_start is not None
    assert command.source_end is not None
    assert command.source_sha256 is not None
    try:
        source_text = _load_authoritative_user_input(
            conn,
            session_id=command.session_id,
            turn_id=command.source_turn_id,
        )
    except Exception as exc:
        raise AuxiliaryGoalSupersedeStaleAuthority(
            "target-change user source authority is missing"
        ) from exc
    if not 0 <= command.source_start < command.source_end <= len(source_text):
        raise AuxiliaryGoalSupersedeStaleAuthority(
            "target-change source span is outside accepted input"
        )
    excerpt = source_text[command.source_start : command.source_end]
    if _text_hash(excerpt) != command.source_sha256:
        raise AuxiliaryGoalSupersedeStaleAuthority(
            "target-change source hash differs from accepted input"
        )
    return PlanningGoalSupersedeSourceBinding.create(
        source_turn_id=command.source_turn_id,
        source_start=command.source_start,
        source_end=command.source_end,
        source_sha256=command.source_sha256,
        source_excerpt=excerpt,
    )


def _require_safe_work_run_checkpoint(
    conn: sqlite3.Connection,
    *,
    command: SupersedeAuxiliaryPlanningGoalCommand,
) -> None:
    uncertain = conn.execute(
        "SELECT run.work_run_id FROM insession_work_run_tool_results AS result "
        "JOIN insession_work_runs AS run ON run.work_run_id=result.work_run_id "
        "WHERE run.session_id=? AND run.insession_task_id=? "
        "AND run.subject_kind='auxiliary_node' "
        "AND run.auxiliary_graph_id=? AND run.auxiliary_graph_revision=? "
        "AND result.status='completion_unconfirmed' LIMIT 1",
        (
            command.session_id,
            command.task_id,
            command.auxiliary_graph_id,
            command.expected_current_auxiliary_graph_revision,
        ),
    ).fetchone()
    if uncertain is not None:
        raise AuxiliaryGoalSupersedeUnsafeWorkRun(
            "completion-unconfirmed WorkRun requires external reconciliation"
        )
    pending = conn.execute(
        "SELECT work_run_id, status FROM insession_work_runs WHERE session_id=? "
        "AND insession_task_id=? AND subject_kind='auxiliary_node' "
        "AND auxiliary_graph_id=? AND auxiliary_graph_revision=? "
        "AND status NOT IN ('completed', 'failed', 'cancelled') LIMIT 1",
        (
            command.session_id,
            command.task_id,
            command.auxiliary_graph_id,
            command.expected_current_auxiliary_graph_revision,
        ),
    ).fetchone()
    if pending is not None:
        raise AuxiliaryGoalSupersedeUnsafeWorkRun(
            f"pending WorkRun {pending['work_run_id']} ({pending['status']}) "
            "must reach a safe checkpoint"
        )


def _validate_budget_row(
    row: sqlite3.Row,
    *,
    expected_goal_id: str,
) -> PlanningEpisodeBudget:
    snapshot_json = str(row["snapshot_json"])
    try:
        budget = PlanningEpisodeBudget.model_validate_json(snapshot_json)
    except (TypeError, ValueError) as exc:
        raise AuxiliaryGoalSupersedeStoredAuthorityCorrupt(
            "planning budget snapshot is invalid"
        ) from exc
    if (
        budget.goal_id != expected_goal_id
        or budget.budget_ledger_id != str(row["budget_ledger_id"])
        or budget.state_version != int(row["state_version"])
        or budget.snapshot_sha256 != str(row["snapshot_sha256"])
        or _model_json(budget) != snapshot_json
    ):
        raise AuxiliaryGoalSupersedeStoredAuthorityCorrupt(
            "planning budget snapshot binding is corrupt"
        )
    return budget


def _authenticate_stored_supersede_receipt(
    conn: sqlite3.Connection,
    *,
    row: sqlite3.Row,
    expected: PlanningGoalSupersedeReceipt | None,
) -> PlanningGoalSupersedeReceipt:
    result_json = str(row["result_json"])
    try:
        stored = PlanningGoalSupersedeReceipt.model_validate_json(result_json)
        budget = PlanningEpisodeBudget.model_validate_json(
            str(row["committed_budget_snapshot_json"])
        )
    except (TypeError, ValueError) as exc:
        raise AuxiliaryGoalSupersedeStoredAuthorityCorrupt(
            "stored supersede authority is invalid"
        ) from exc
    if expected is not None and stored != expected:
        raise AuxiliaryGoalSupersedeStoredAuthorityCorrupt(
            "provided receipt is not the exact sealed supersede receipt"
        )
    if (
        str(row["operation"]) != "supersede_goal"
        or str(row["apply_id"]) != stored.apply_id
        or str(row["session_id"]) != stored.session_id
        or str(row["insession_task_id"]) != stored.task_id
        or str(row["auxiliary_graph_id"]) != stored.auxiliary_graph_id
        or str(row["goal_id"]) != stored.superseded_goal_id
        or str(row["invocation_turn_id"]) != stored.invocation_turn_id
        or int(row["expected_control_state_version"])
        != stored.control_state_version_before
        or int(row["committed_control_state_version"])
        != stored.control_state_version_after
        or int(row["expected_current_auxiliary_graph_revision"])
        != stored.superseded_auxiliary_graph_revision
        or int(row["committed_auxiliary_graph_revision"])
        != stored.superseded_auxiliary_graph_revision
        or int(row["expected_goal_state_version"])
        != stored.goal_state_version_before
        or int(row["committed_goal_state_version"])
        != stored.goal_state_version_after
        or int(row["expected_budget_state_version"])
        != stored.budget_state_version_before
        or int(row["committed_budget_state_version"])
        != stored.budget_state_version_after
        or str(row["budget_ledger_id"]) != stored.budget_ledger_id
        or str(row["committed_budget_snapshot_sha256"])
        != stored.budget_snapshot_sha256
        or _text_hash(result_json) != str(row["result_sha256"])
        or _model_json(stored) != result_json
        or budget.goal_id != stored.superseded_goal_id
        or budget.budget_ledger_id != stored.budget_ledger_id
        or budget.state_version != stored.budget_state_version_after
        or budget.snapshot_sha256 != stored.budget_snapshot_sha256
        or _model_json(budget) != str(row["committed_budget_snapshot_json"])
    ):
        raise AuxiliaryGoalSupersedeStoredAuthorityCorrupt(
            "stored supersede receipt crossed its mirrored authority"
        )
    goal = conn.execute(
        "SELECT status, state_version FROM insession_auxiliary_graph_goals "
        "WHERE goal_id=? AND auxiliary_graph_id=?",
        (stored.superseded_goal_id, stored.auxiliary_graph_id),
    ).fetchone()
    revision = conn.execute(
        "SELECT status, state_version FROM "
        "insession_auxiliary_graph_revision_states_v2 "
        "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=?",
        (
            stored.auxiliary_graph_id,
            stored.superseded_auxiliary_graph_revision,
        ),
    ).fetchone()
    current_budget = conn.execute(
        "SELECT budget_ledger_id, snapshot_json, snapshot_sha256, state_version "
        "FROM insession_auxiliary_goal_budgets WHERE goal_id=? "
        "AND auxiliary_graph_id=?",
        (stored.superseded_goal_id, stored.auxiliary_graph_id),
    ).fetchone()
    if (
        goal is None
        or revision is None
        or current_budget is None
        or str(goal["status"]) != "superseded"
        or int(goal["state_version"]) != stored.goal_state_version_after
        or str(revision["status"]) != "superseded"
        or int(revision["state_version"])
        != stored.revision_state_version_after
        or str(current_budget["budget_ledger_id"]) != stored.budget_ledger_id
        or int(current_budget["state_version"])
        != stored.budget_state_version_after
        or str(current_budget["snapshot_sha256"])
        != stored.budget_snapshot_sha256
        or str(current_budget["snapshot_json"])
        != str(row["committed_budget_snapshot_json"])
    ):
        raise AuxiliaryGoalSupersedeStoredAuthorityCorrupt(
            "superseded goal state no longer authenticates its receipt"
        )
    return stored


def _load_exact_replay(
    conn: sqlite3.Connection,
    *,
    command: SupersedeAuxiliaryPlanningGoalCommand,
    payload_hash: str,
    row: sqlite3.Row,
) -> SupersedeAuxiliaryPlanningGoalResult:
    if (
        str(row["operation"]) != "supersede_goal"
        or str(row["session_id"]) != command.session_id
        or str(row["insession_task_id"]) != command.task_id
        or str(row["auxiliary_graph_id"]) != command.auxiliary_graph_id
        or str(row["goal_id"]) != command.goal_id
        or str(row["payload_sha256"]) != payload_hash
    ):
        raise AuxiliaryGoalSupersedeIdentityCollision(
            "goal supersede apply_id was reused for another payload"
        )
    result_json = str(row["result_json"])
    try:
        receipt = PlanningGoalSupersedeReceipt.model_validate_json(result_json)
        budget = PlanningEpisodeBudget.model_validate_json(
            str(row["committed_budget_snapshot_json"])
        )
    except (TypeError, ValueError) as exc:
        raise AuxiliaryGoalSupersedeStoredAuthorityCorrupt(
            "stored goal supersede receipt is invalid"
        ) from exc
    if (
        _text_hash(result_json) != str(row["result_sha256"])
        or receipt.apply_id != command.apply_id
        or receipt.invocation_turn_id != str(row["invocation_turn_id"])
        or receipt.control_state_version_before
        != int(row["expected_control_state_version"])
        or receipt.control_state_version_after
        != int(row["committed_control_state_version"])
        or receipt.superseded_auxiliary_graph_revision
        != int(row["committed_auxiliary_graph_revision"])
        or receipt.superseded_auxiliary_graph_revision
        != int(row["expected_current_auxiliary_graph_revision"])
        or receipt.goal_state_version_before
        != int(row["expected_goal_state_version"])
        or receipt.goal_state_version_after
        != int(row["committed_goal_state_version"])
        or receipt.budget_state_version_before
        != int(row["expected_budget_state_version"])
        or receipt.budget_state_version_after
        != int(row["committed_budget_state_version"])
        or receipt.budget_ledger_id != str(row["budget_ledger_id"])
        or receipt.budget_snapshot_sha256
        != str(row["committed_budget_snapshot_sha256"])
        or budget.goal_id != receipt.superseded_goal_id
        or budget.budget_ledger_id != receipt.budget_ledger_id
        or budget.state_version != receipt.budget_state_version_after
        or budget.snapshot_sha256 != receipt.budget_snapshot_sha256
        or _model_json(budget) != str(row["committed_budget_snapshot_json"])
    ):
        raise AuxiliaryGoalSupersedeStoredAuthorityCorrupt(
            "stored goal supersede receipt binding is corrupt"
        )
    goal = conn.execute(
        "SELECT status, state_version FROM insession_auxiliary_graph_goals "
        "WHERE goal_id=? AND auxiliary_graph_id=?",
        (receipt.superseded_goal_id, receipt.auxiliary_graph_id),
    ).fetchone()
    revision = conn.execute(
        "SELECT status, state_version FROM "
        "insession_auxiliary_graph_revision_states_v2 "
        "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=?",
        (
            receipt.auxiliary_graph_id,
            receipt.superseded_auxiliary_graph_revision,
        ),
    ).fetchone()
    current_budget = conn.execute(
        "SELECT budget_ledger_id, snapshot_json, snapshot_sha256, state_version "
        "FROM insession_auxiliary_goal_budgets WHERE goal_id=? "
        "AND auxiliary_graph_id=?",
        (receipt.superseded_goal_id, receipt.auxiliary_graph_id),
    ).fetchone()
    if (
        goal is None
        or revision is None
        or current_budget is None
        or str(goal["status"]) != "superseded"
        or int(goal["state_version"]) != receipt.goal_state_version_after
        or str(revision["status"]) != "superseded"
        or int(revision["state_version"]) != receipt.revision_state_version_after
        or str(current_budget["budget_ledger_id"]) != receipt.budget_ledger_id
        or int(current_budget["state_version"])
        != receipt.budget_state_version_after
        or str(current_budget["snapshot_sha256"])
        != receipt.budget_snapshot_sha256
        or str(current_budget["snapshot_json"])
        != str(row["committed_budget_snapshot_json"])
    ):
        raise AuxiliaryGoalSupersedeStoredAuthorityCorrupt(
            "superseded goal/revision state no longer authenticates receipt"
        )
    return SupersedeAuxiliaryPlanningGoalResult(
        status="replayed",
        receipt=receipt,
    )


__all__ = [
    "AuxiliaryGoalSupersedeIdentityCollision",
    "AuxiliaryGoalSupersedePersistenceError",
    "AuxiliaryGoalSupersedeStaleAuthority",
    "AuxiliaryGoalSupersedeStoredAuthorityCorrupt",
    "AuxiliaryGoalSupersedeUnsafeWorkRun",
    "PlanningGoalSupersedeReason",
    "PlanningGoalSupersedeReceipt",
    "PlanningGoalSupersedeSourceBinding",
    "SupersedeAuxiliaryPlanningGoalCommand",
    "SupersedeAuxiliaryPlanningGoalResult",
    "get_authenticated_user_target_change_receipt",
    "get_pending_auxiliary_goal_supersede_receipt",
    "require_authenticated_planning_goal_supersede_receipt",
    "supersede_auxiliary_planning_goal",
]
