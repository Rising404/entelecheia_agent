"""从 AuxiliaryGraph 终态提案到 TaskGraph 的原子提交。

此命令消费不可变 v68 终态与完成门禁回执，绝不接受调用方编写的提案、语义裁决、生产裁决
或持久节点身份。``None -> 1`` 与 ``N -> N+1`` 都是完整快照提交；正 base 连续性根据
冻结语义谱系规划，完成状态只在 Store 所有结转证明后复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from personagraph.l2.auxiliary_graph import (
    PlanningEpisodeBudgetDisposition,
    TaskGraphSemanticDeliveryCoverage,
    TaskGraphSemanticVerificationDisposition,
)
from personagraph.l2.task_graph import (
    InSessionTaskDetails,
    InSessionTaskGraphRevisionLineageHints,
    InSessionTaskGraphRevisionProposal,
    InSessionTaskGraphRevisionTransition,
    InSessionTaskNodeLineageHint,
    InSessionTaskStatus,
    TaskGraphNodeLineageDisposition,
    TaskGraphRevisionTrigger,
    TaskGraphRevisionTransitionError,
    plan_insession_task_graph_revision_transition,
    validate_insession_task_graph_revision,
)
from personagraph.l2.task_graph.production_gate import (
    TaskGraphProductionEvaluation,
    evaluate_task_graph_production,
)
from personagraph.l2.work_run import OutputWindow, TaskNodeSubject
from personagraph.l2.work_run.contracts import (
    TaskGraphExecutionReplanApplication,
    TaskGraphExecutionReplanRequest,
)
from . import auxiliary_graphs as auxiliary_graph_records
from ..delivery import auxiliary_semantic_verification as semantic_records
from . import auxiliary_terminal_seal as terminal_records
from ..task_graph import insession_tasks as task_records
from ..work_run import work_execution as work_execution_records
from ..work_run import work_verification as verification_records
from ..delivery import task_delivery_validation as task_delivery_validation_records
from ..planning import auxiliary_planning_completions as initial_planning_records
from ...deps import StoreDeps


_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"
_ALIAS_PATTERN = r"^[a-z][a-z0-9_-]{0,63}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_NONTERMINAL_RUN_STATUSES = {
    "active",
    "paused",
    "waiting_user",
    "waiting_authorization",
    "waiting_external",
    "turn_limit_reached",
    "interrupted",
}


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AuxiliaryTaskGraphCommitFailureCode(StrEnum):
    APPLY_ID_COLLISION = "apply_id_collision"
    AUTHORITY_NOT_CURRENT = "authority_not_current"
    STATE_VERSION_CONFLICT = "state_version_conflict"
    BASE_TASK_GRAPH_DRIFT = "base_task_graph_drift"
    TERMINAL_RECEIPT_CORRUPT = "terminal_receipt_corrupt"
    SEMANTIC_BINDING_MISMATCH = "semantic_binding_mismatch"
    PRODUCTION_EVALUATION_FAILED = "production_evaluation_failed"
    BASE_ALIAS_BINDING_INVALID = "base_alias_binding_invalid"
    LINEAGE_INVALID = "lineage_invalid"
    CARRY_UNSAFE = "carry_unsafe"
    ACTIVE_EXECUTION_PRESENT = "active_execution_present"
    STORED_AUTHORITY_CORRUPT = "stored_authority_corrupt"


class AuxiliaryTaskGraphCommitPersistenceError(RuntimeError):
    def __init__(
        self,
        code: AuxiliaryTaskGraphCommitFailureCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(message)


class AuxiliaryTaskGraphCommitIdentityCollision(
    AuxiliaryTaskGraphCommitPersistenceError
):
    """不可变命令身份被复用于另一载荷。"""


class AuxiliaryBaseNodeAliasBinding(_Record):
    node_alias: str = Field(pattern=_ALIAS_PATTERN)
    insession_task_node_id: str = Field(pattern=_ID_PATTERN)


class CommitAuxiliaryTaskGraphProposalCommand(_Record):
    schema_version: Literal[
        "commit-auxiliary-v2-task-graph-proposal-command-v1"
    ] = "commit-auxiliary-v2-task-graph-proposal-command-v1"
    apply_id: str = Field(pattern=_ID_PATTERN)
    session_id: str = Field(pattern=_ID_PATTERN)
    source_turn_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    terminal_proposal_receipt_id: str = Field(pattern=_ID_PATTERN)
    expected_base_task_graph_revision: int | None = Field(default=None, ge=1)
    expected_task_state_version: int = Field(ge=1)
    expected_window_revision: int = Field(ge=1)
    base_node_alias_bindings: tuple[AuxiliaryBaseNodeAliasBinding, ...] = Field(
        default=(), max_length=512
    )
    task_graph_revision_trigger_id: str | None = Field(
        default=None,
        pattern=_ID_PATTERN,
    )
    expected_task_graph_revision_trigger_sha256: str | None = Field(
        default=None,
        pattern=_SHA256_PATTERN,
    )
    task_graph_execution_replan_request_id: str | None = Field(
        default=None,
        pattern=_ID_PATTERN,
    )
    expected_task_graph_execution_replan_request_sha256: str | None = Field(
        default=None,
        pattern=_SHA256_PATTERN,
    )

    @field_validator("base_node_alias_bindings")
    @classmethod
    def _require_unique_base_bindings(
        cls,
        values: tuple[AuxiliaryBaseNodeAliasBinding, ...],
    ) -> tuple[AuxiliaryBaseNodeAliasBinding, ...]:
        aliases = tuple(item.node_alias for item in values)
        node_ids = tuple(item.insession_task_node_id for item in values)
        if len(aliases) != len(set(aliases)) or len(node_ids) != len(set(node_ids)):
            raise ValueError("base aliases and durable node IDs must be unique")
        if aliases != tuple(sorted(aliases)):
            raise ValueError("base node alias bindings must use canonical alias order")
        return values

    @model_validator(mode="after")
    def _validate_revision_trigger_pair(
        self,
    ) -> "CommitAuxiliaryTaskGraphProposalCommand":
        if (self.task_graph_revision_trigger_id is None) != (
            self.expected_task_graph_revision_trigger_sha256 is None
        ):
            raise ValueError("TaskGraph revision trigger ID/hash must be paired")
        if (self.task_graph_execution_replan_request_id is None) != (
            self.expected_task_graph_execution_replan_request_sha256 is None
        ):
            raise ValueError(
                "TaskGraph execution replan request ID/hash must be paired"
            )
        if (
            self.task_graph_revision_trigger_id is not None
            and self.task_graph_execution_replan_request_id is not None
        ):
            raise ValueError(
                "TaskGraph commit cannot consume two revision authorities"
            )
        if (
            self.expected_base_task_graph_revision is None
            and (
                self.task_graph_revision_trigger_id is not None
                or self.task_graph_execution_replan_request_id is not None
            )
        ):
            raise ValueError(
                "base-null TaskGraph commit cannot consume revision authority"
            )
        return self


class AuxiliaryTaskGraphNodeCarryReceipt(_Record):
    schema_version: Literal["auxiliary-v2-task-graph-node-carry-receipt-v1"] = (
        "auxiliary-v2-task-graph-node-carry-receipt-v1"
    )
    carry_receipt_id: str = Field(pattern=_ID_PATTERN)
    apply_id: str = Field(pattern=_ID_PATTERN)
    session_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    base_task_graph_revision: int = Field(ge=1)
    target_task_graph_revision: int = Field(ge=2)
    node_id: str = Field(pattern=_ID_PATTERN)
    node_revision: int = Field(ge=1)
    source_delivery_id: str = Field(pattern=_ID_PATTERN)
    definition_sha256: str = Field(pattern=_SHA256_PATTERN)
    dependency_delivery_ids: tuple[str, ...]
    dependency_closure_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_authority_sha256: str = Field(pattern=_SHA256_PATTERN)
    capability_catalog_sha256: str = Field(pattern=_SHA256_PATTERN)
    freshness_authority_sha256: str = Field(pattern=_SHA256_PATTERN)


class AuxiliaryTaskGraphCommitResult(_Record):
    schema_version: Literal["auxiliary-v2-task-graph-commit-result-v1"] = (
        "auxiliary-v2-task-graph-commit-result-v1"
    )
    status: Literal["applied", "replayed"]
    apply_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    terminal_proposal_receipt_id: str = Field(pattern=_ID_PATTERN)
    finish_gate_receipt_id: str = Field(pattern=_ID_PATTERN)
    previous_graph_revision: int | None = Field(default=None, ge=1)
    committed_graph_revision: int = Field(ge=1)
    proposal_sha256: str = Field(pattern=_SHA256_PATTERN)
    transition_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    carry_receipt_ids: tuple[str, ...] = ()
    task_state_version: int = Field(ge=1)
    goal_state_version: int = Field(ge=1)
    revision_state_version: int = Field(ge=1)
    turn_task_link_revision: int = Field(ge=0)
    window_state_version: int = Field(ge=1)


@dataclass(frozen=True)
class _TerminalAuthority:
    terminal: terminal_records.AuxiliaryTerminalProposalReceipt
    finish: terminal_records.AuxiliaryFinishGateReceipt
    proposal: InSessionTaskGraphRevisionProposal
    validation_context: object
    evaluation: TaskGraphProductionEvaluation
    settlement: object
    semantic_request: object


def commit_auxiliary_task_graph_proposal(
    deps: StoreDeps,
    *,
    command: CommitAuxiliaryTaskGraphProposalCommand,
) -> AuxiliaryTaskGraphCommitResult:
    """把一个精确密封 提案提交为 Task 的下一完整快照。"""

    if not isinstance(command, CommitAuxiliaryTaskGraphProposalCommand):
        raise TypeError(
            "command must be a CommitAuxiliaryTaskGraphProposalCommand"
        )
    command_sha256 = _sha256_value(command.model_dump(mode="json"))
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        replay_row = conn.execute(
            "SELECT * FROM insession_auxiliary_v2_task_graph_commit_receipts "
            "WHERE apply_id=?",
            (command.apply_id,),
        ).fetchone()
        if replay_row is not None:
            result = _load_replay(
                conn,
                command=command,
                command_sha256=command_sha256,
                row=replay_row,
            )
            conn.commit()
            return result

        consumed = conn.execute(
            "SELECT apply_id FROM "
            "insession_auxiliary_v2_task_graph_commit_receipts "
            "WHERE terminal_proposal_receipt_id=?",
            (command.terminal_proposal_receipt_id,),
        ).fetchone()
        if consumed is not None:
            _fail_collision("terminal proposal receipt was already consumed")

        task_records._require_session_turn(
            conn,
            session_id=command.session_id,
            turn_id=command.source_turn_id,
        )
        task_records._require_active_window(
            conn,
            session_id=command.session_id,
            turn_id=command.source_turn_id,
            expected_window_revision=command.expected_window_revision,
        )
        task = task_records._load_revision_target(
            conn,
            session_id=command.session_id,
            target_insession_task_id=command.task_id,
        )
        _require_task_cas(command, task)
        revision_authority = _require_task_graph_revision_authority_cas(
            conn,
            command,
        )
        task_records._require_turn_root_task_link(
            conn,
            session_id=command.session_id,
            turn_id=command.source_turn_id,
            target_insession_task_id=command.task_id,
        )
        _require_no_active_task_execution(conn, command)

        details = auxiliary_graph_records._load_auxiliary_graph(
            conn,
            command.session_id,
            command.task_id,
        )
        _require_revision_authority_goal_binding(
            conn=conn,
            revision_authority=revision_authority,
            details=details,
            failure_code=(
                AuxiliaryTaskGraphCommitFailureCode.AUTHORITY_NOT_CURRENT
            ),
        )
        authority = _load_terminal_authority(
            conn,
            command=command,
            task=task,
            details=details,
        )
        proposal = authority.proposal
        validation_context = authority.validation_context
        target_revision = (
            1
            if command.expected_base_task_graph_revision is None
            else command.expected_base_task_graph_revision + 1
        )
        now = deps.now()
        transition: InSessionTaskGraphRevisionTransition | None = None
        carry_receipts: tuple[AuxiliaryTaskGraphNodeCarryReceipt, ...] = ()

        source_manifest = task_records._derive_root_source_manifest(
            proposal.root,
            validation_context,
        )
        graph_proposal_hash = task_records._revision_proposal_hash(
            proposal,
            trusted_context=validation_context,
            target_insession_task_id=command.task_id,
            expected_current_graph_revision=(
                command.expected_base_task_graph_revision
            ),
            expected_task_state_version=command.expected_task_state_version,
            expected_window_revision=command.expected_window_revision,
        )

        if command.expected_base_task_graph_revision is None:
            if command.base_node_alias_bindings:
                _fail(
                    AuxiliaryTaskGraphCommitFailureCode.BASE_ALIAS_BINDING_INVALID,
                    "base-null commit cannot carry base-node bindings",
                )
            if authority.semantic_request.base_task_graph is not None or (
                authority.semantic_request.prompt_payload.lineage
            ) or authority.terminal.lineage:
                _fail(
                    AuxiliaryTaskGraphCommitFailureCode.SEMANTIC_BINDING_MISMATCH,
                    "base-null semantic authority unexpectedly carries lineage",
                )
            local_node_ids = task_records._allocate_revision_one_node_ids(
                conn,
                deps,
                target_insession_task_id=command.task_id,
                root=proposal.root,
            )
            task_records._insert_graph_revision_one(
                conn,
                task_id=command.task_id,
                source_turn_id=command.source_turn_id,
                proposal_hash=graph_proposal_hash,
                source_manifest=source_manifest,
                now=now,
            )
            task_records._insert_graph_nodes_and_edges(
                conn,
                task_id=command.task_id,
                root_nodes=proposal.root.nodes,
                local_node_ids=local_node_ids,
                now=now,
            )
        else:
            base = _load_task_details_in_transaction(conn, task=task)
            base_ids_by_alias = _authenticate_base_alias_bindings(
                command,
                request=authority.semantic_request,
                base=base,
            )
            if (
                isinstance(
                    revision_authority,
                    TaskGraphExecutionReplanRequest,
                )
                and base_ids_by_alias.get(revision_authority.source_node_alias)
                != revision_authority.requesting_subject.node_id
            ):
                _fail(
                    AuxiliaryTaskGraphCommitFailureCode.LINEAGE_INVALID,
                    "execution request source alias crossed its TaskNode",
                )
            lineage = _derive_lineage_authority(
                command,
                authority=authority,
            )
            allocated = _allocate_new_transition_node_ids(
                conn,
                deps,
                proposal=proposal,
                lineage=lineage,
            )
            try:
                transition = plan_insession_task_graph_revision_transition(
                    base,
                    proposal,
                    lineage=lineage,
                    base_node_ids_by_alias=base_ids_by_alias,
                    allocated_node_ids_by_key=allocated,
                    force_reexecution_base_node_ids=(
                        _forced_reexecution_base_node_ids(
                            revision_authority,
                            root_node_id=command.task_id,
                        )
                    ),
                )
            except TaskGraphRevisionTransitionError as exc:
                _fail(
                    AuxiliaryTaskGraphCommitFailureCode.LINEAGE_INVALID,
                    f"TaskGraph lineage is invalid: {exc.code.value}",
                )
            _require_triggered_revision(revision_authority, transition)
            _require_transition_states_safe(
                conn,
                command=command,
                authority=authority,
                transition=transition,
                base=base,
                base_ids_by_alias=base_ids_by_alias,
            )
            _insert_positive_revision(
                conn,
                command=command,
                proposal=proposal,
                transition=transition,
                proposal_hash=graph_proposal_hash,
                source_manifest=source_manifest,
                now=now,
            )
            carry_receipts = _derive_completed_carry_receipts(
                conn,
                command=command,
                authority=authority,
                transition=transition,
                base=base,
                base_ids_by_alias=base_ids_by_alias,
            )

        root_node = next(
            node
            for node in proposal.root.nodes
            if node.node_key == proposal.root.root_key
        )
        next_task_version = command.expected_task_state_version + 1
        if conn.execute(
            "UPDATE insession_tasks SET current_graph_revision=?, "
            "state_version=state_version+1, root_title=?, root_objective=?, "
            "updated_at=? WHERE session_id=? AND insession_task_id=? "
            "AND current_graph_revision IS ? AND state_version=? "
            "AND current_status NOT IN ('cancelled','completed')",
            (
                target_revision,
                root_node.title,
                root_node.objective,
                now,
                command.session_id,
                command.task_id,
                command.expected_base_task_graph_revision,
                command.expected_task_state_version,
            ),
        ).rowcount != 1:
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.STATE_VERSION_CONFLICT,
                "Task authority changed during TaskGraph commit",
            )
        turn_link_revision = task_records._current_turn_link_revision(
            conn,
            command.session_id,
            command.source_turn_id,
        )
        window_version = task_records._advance_window_task_link_pointer(
            conn,
            session_id=command.session_id,
            turn_id=command.source_turn_id,
            link_revision=turn_link_revision,
            expected_window_revision=command.expected_window_revision,
            now=now,
        )
        if window_version is None:
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.STATE_VERSION_CONFLICT,
                "TaskGraph commit lost its active execution window",
            )

        next_goal_version = details.goal_state_version + 1
        next_revision_version = details.revision_state_version + 1
        _commit_auxiliary_projection(
            conn,
            details=details,
            now=now,
        )
        target_snapshot_json = _target_snapshot_json(
            conn,
            task_id=command.task_id,
            graph_revision=target_revision,
        )
        result = AuxiliaryTaskGraphCommitResult(
            status="applied",
            apply_id=command.apply_id,
            task_id=command.task_id,
            terminal_proposal_receipt_id=command.terminal_proposal_receipt_id,
            finish_gate_receipt_id=authority.finish.finish_gate_receipt_id,
            previous_graph_revision=command.expected_base_task_graph_revision,
            committed_graph_revision=target_revision,
            proposal_sha256=authority.terminal.proposal_sha256,
            transition_sha256=(
                transition.transition_sha256 if transition is not None else None
            ),
            carry_receipt_ids=tuple(
                item.carry_receipt_id for item in carry_receipts
            ),
            task_state_version=next_task_version,
            goal_state_version=next_goal_version,
            revision_state_version=next_revision_version,
            turn_task_link_revision=turn_link_revision,
            window_state_version=window_version,
        )
        _insert_commit_receipt(
            conn,
            command=command,
            command_sha256=command_sha256,
            authority=authority,
            transition=transition,
            target_snapshot_json=target_snapshot_json,
            result=result,
            now=now,
        )
        if command.task_graph_revision_trigger_id is not None:
            trigger_application_hash = _sha256_value(
                {
                    "contract": "atomic-task-graph-revision-trigger-application-id-v1",
                    "task_graph_commit_apply_id": command.apply_id,
                    "trigger_id": command.task_graph_revision_trigger_id,
                }
            )
            try:
                consume_trigger = (
                    task_delivery_validation_records
                    .consume_task_graph_revision_trigger_in_transaction
                )
                consume_trigger(
                    conn,
                    application_apply_id=f"tgr_apply_{trigger_application_hash[:40]}",
                    session_id=command.session_id,
                    task_id=command.task_id,
                    trigger_id=command.task_graph_revision_trigger_id,
                    expected_trigger_sha256=(
                        command.expected_task_graph_revision_trigger_sha256
                    ),
                    task_graph_commit_apply_id=command.apply_id,
                    consumed_turn_id=command.source_turn_id,
                    now=now,
                )
            except (
                task_delivery_validation_records.TaskDeliveryValidationPersistenceError
            ) as exc:
                _fail_corrupt(
                    "TaskGraph revision-trigger application lost exact authority",
                    exc,
                )
        elif isinstance(
            revision_authority,
            TaskGraphExecutionReplanRequest,
        ):
            _consume_execution_replan_request_in_transaction(
                conn,
                command=command,
                request=revision_authority,
                result=result,
                now=now,
            )
        for carry in carry_receipts:
            _insert_carry_receipt(conn, receipt=carry, now=now)
        conn.commit()
        return result
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def _consume_execution_replan_request_in_transaction(
    conn,
    *,
    command,
    request: TaskGraphExecutionReplanRequest,
    result: AuxiliaryTaskGraphCommitResult,
    now: str,
) -> None:
    application_hash = _sha256_value(
        {
            "contract": (
                "atomic-task-graph-execution-replan-application-id-v1"
            ),
            "task_graph_commit_apply_id": command.apply_id,
            "request_id": request.request_id,
        }
    )
    application = TaskGraphExecutionReplanApplication.create(
        apply_id=f"tger_apply_{application_hash[:40]}",
        request_id=request.request_id,
        request_sha256=request.request_sha256,
        session_id=command.session_id,
        task_id=command.task_id,
        base_graph_revision=request.base_graph_revision,
        committed_graph_revision=result.committed_graph_revision,
        task_graph_commit_apply_id=command.apply_id,
        consumed_turn_id=command.source_turn_id,
    )
    receipt_json = _model_json(application)
    try:
        conn.execute(
            "INSERT INTO "
            "insession_task_graph_execution_replan_applications "
            "(apply_id, request_id, request_sha256, session_id, "
            "insession_task_id, base_graph_revision, "
            "committed_graph_revision, task_graph_commit_apply_id, "
            "consumed_turn_id, receipt_sha256, receipt_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                application.apply_id,
                application.request_id,
                application.request_sha256,
                application.session_id,
                application.task_id,
                application.base_graph_revision,
                application.committed_graph_revision,
                application.task_graph_commit_apply_id,
                application.consumed_turn_id,
                application.receipt_sha256,
                receipt_json,
                now,
            ),
        )
    except Exception as exc:
        _fail_corrupt(
            "TaskGraph execution-request application conflicts with history",
            exc,
        )
    if conn.execute(
        "DELETE FROM "
        "insession_active_task_graph_execution_replan_requests "
        "WHERE request_id=? AND request_sha256=? AND session_id=? "
        "AND insession_task_id=? AND base_graph_revision=? "
        "AND target_graph_revision=?",
        (
            request.request_id,
            request.request_sha256,
            request.session_id,
            request.task_id,
            request.base_graph_revision,
            request.target_graph_revision,
        ),
    ).rowcount != 1:
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.AUTHORITY_NOT_CURRENT,
            "TaskGraph execution request lost its active pointer",
        )


def _require_task_cas(command, task) -> None:
    current = (
        int(task["current_graph_revision"])
        if task["current_graph_revision"] is not None
        else None
    )
    if current != command.expected_base_task_graph_revision:
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.BASE_TASK_GRAPH_DRIFT,
            "TaskGraph base changed after terminal proposal seal",
        )
    if int(task["state_version"]) != command.expected_task_state_version:
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.STATE_VERSION_CONFLICT,
            "Task state version is stale",
        )
    if str(task["current_status"]) != "active":
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.AUTHORITY_NOT_CURRENT,
            "TaskGraph commit requires an active Task",
        )


def _require_task_graph_revision_authority_cas(conn, command):
    trigger_rows = conn.execute(
        "SELECT * FROM insession_active_task_graph_revision_triggers "
        "WHERE session_id=? AND insession_task_id=?",
        (command.session_id, command.task_id),
    ).fetchall()
    request_rows = conn.execute(
        "SELECT * FROM "
        "insession_active_task_graph_execution_replan_requests "
        "WHERE session_id=? AND insession_task_id=?",
        (command.session_id, command.task_id),
    ).fetchall()
    if (
        len(trigger_rows) > 1
        or len(request_rows) > 1
        or (trigger_rows and request_rows)
    ):
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
            "Task owns conflicting active TaskGraph revision authorities",
        )
    if command.expected_base_task_graph_revision is None:
        if (
            trigger_rows
            or request_rows
            or command.task_graph_revision_trigger_id is not None
            or command.task_graph_execution_replan_request_id is not None
        ):
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.AUTHORITY_NOT_CURRENT,
                "base-null TaskGraph commit cannot cross revision authority",
            )
        return None
    if not trigger_rows and not request_rows:
        if (
            command.task_graph_revision_trigger_id is not None
            or command.task_graph_execution_replan_request_id is not None
        ):
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.AUTHORITY_NOT_CURRENT,
                "TaskGraph revision authority is no longer active",
            )
        orphaned_trigger = conn.execute(
            "SELECT trigger.trigger_id FROM insession_task_graph_revision_triggers "
            "AS trigger LEFT JOIN "
            "insession_task_graph_revision_trigger_applications AS application "
            "ON application.trigger_id=trigger.trigger_id "
            "WHERE trigger.session_id=? AND trigger.insession_task_id=? "
            "AND trigger.base_graph_revision=? AND application.trigger_id IS NULL "
            "LIMIT 1",
            (
                command.session_id,
                command.task_id,
                command.expected_base_task_graph_revision,
            ),
        ).fetchone()
        orphaned_request = conn.execute(
            "SELECT request.request_id FROM "
            "insession_task_graph_execution_replan_requests AS request "
            "LEFT JOIN insession_task_graph_execution_replan_applications "
            "AS application ON application.request_id=request.request_id "
            "WHERE request.session_id=? AND request.insession_task_id=? "
            "AND request.base_graph_revision=? AND application.request_id IS NULL "
            "LIMIT 1",
            (
                command.session_id,
                command.task_id,
                command.expected_base_task_graph_revision,
            ),
        ).fetchone()
        if orphaned_trigger is not None or orphaned_request is not None:
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.AUTHORITY_NOT_CURRENT,
                "TaskGraph revision authority lost its active pointer",
            )
# 手动构造的正向 base 夹具可能没有发布门禁权威；生产正向规划始终从活动触发器开始，
# 因此不会进入这个夹具专用分支。
        return None
    if request_rows:
        request_id = str(request_rows[0]["request_id"])
        if (
            command.task_graph_revision_trigger_id is not None
            or command.task_graph_execution_replan_request_id != request_id
            or command.expected_task_graph_execution_replan_request_sha256
            is None
        ):
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.AUTHORITY_NOT_CURRENT,
                "positive TaskGraph commit omitted its active execution request",
            )
        try:
            request = (
                work_execution_records
                ._load_authenticated_execution_replan_request(
                    conn,
                    request_id=request_id,
                    session_id=command.session_id,
                    task_id=command.task_id,
                    require_active=True,
                )
            )
        except Exception as exc:
            _fail_corrupt("active TaskGraph execution request is corrupt", exc)
        if request is None:
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.AUTHORITY_NOT_CURRENT,
                "TaskGraph execution request is no longer active",
            )
        if (
            request.base_graph_revision
            != command.expected_base_task_graph_revision
            or request.target_graph_revision
            != command.expected_base_task_graph_revision + 1
            or request.request_sha256
            != command.expected_task_graph_execution_replan_request_sha256
        ):
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.AUTHORITY_NOT_CURRENT,
                "positive TaskGraph commit execution-request CAS is stale",
            )
        return request

    trigger_id = str(trigger_rows[0]["trigger_id"])
    if (
        command.task_graph_execution_replan_request_id is not None
        or
        command.task_graph_revision_trigger_id != trigger_id
        or command.expected_task_graph_revision_trigger_sha256 is None
    ):
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.AUTHORITY_NOT_CURRENT,
            "positive TaskGraph commit omitted its active revision trigger",
        )
    try:
        trigger = (
            task_delivery_validation_records._load_authenticated_trigger_authority(
                conn,
                trigger_id,
            )
        )
    except Exception as exc:
        _fail_corrupt("active TaskGraph revision trigger is corrupt", exc)
    active = trigger_rows[0]
    task = conn.execute(
        "SELECT current_graph_revision, current_status, state_version "
        "FROM insession_tasks WHERE session_id=? AND insession_task_id=?",
        (command.session_id, command.task_id),
    ).fetchone()
    if (
        trigger.session_id != command.session_id
        or trigger.task_id != command.task_id
        or str(active["session_id"]) != trigger.session_id
        or str(active["insession_task_id"]) != trigger.task_id
        or int(active["base_graph_revision"]) != trigger.base_graph_revision
        or int(active["target_graph_revision"])
        != trigger.target_graph_revision
        or task is None
        or int(task["current_graph_revision"] or 0)
        != trigger.base_graph_revision
        or str(task["current_status"]) != "active"
        or int(task["state_version"]) != command.expected_task_state_version
        or int(task["state_version"]) < trigger.reopened_task_state_version
    ):
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
            "active TaskGraph revision-trigger projection is corrupt",
        )
    if (
        trigger.base_graph_revision
        != command.expected_base_task_graph_revision
        or trigger.target_graph_revision
        != command.expected_base_task_graph_revision + 1
        or trigger.trigger_sha256
        != command.expected_task_graph_revision_trigger_sha256
    ):
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.AUTHORITY_NOT_CURRENT,
            "positive TaskGraph commit revision-trigger CAS is stale",
        )
    return trigger


def _require_triggered_revision(authority, transition) -> None:
    """请求 revision 的精确节点不能原样结转。"""

    if authority is None:
        return
    if isinstance(authority, TaskGraphExecutionReplanRequest):
        source = tuple(
            item
            for item in transition.nodes
            if item.base_node_id == authority.requesting_subject.node_id
        )
        if source and source[0].disposition is TaskGraphNodeLineageDisposition.REUSE:
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.LINEAGE_INVALID,
                "execution-requesting TaskNode cannot be reused unchanged",
            )
        return
    root = tuple(
        item
        for item in transition.nodes
        if item.target_node_id == transition.root_node_id
    )
    if len(root) != 1:
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.LINEAGE_INVALID,
            "triggered TaskGraph revision has no unique root transition",
        )
    if root[0].disposition is TaskGraphNodeLineageDisposition.REUSE:
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.LINEAGE_INVALID,
            "whole-Task REVISE authority forbids reuse of the failed root Delivery",
        )


def _forced_reexecution_base_node_ids(
    authority: TaskGraphRevisionTrigger | TaskGraphExecutionReplanRequest | None,
    *,
    root_node_id: str,
) -> frozenset[str]:
    """投影被 revision 权威判定失效的精确已完成执行。"""

    if authority is None:
        return frozenset()
    if isinstance(authority, TaskGraphExecutionReplanRequest):
        return frozenset({authority.requesting_subject.node_id})
    return frozenset({root_node_id})


def _require_no_active_task_execution(conn, command) -> None:
    rows = conn.execute(
        "SELECT work_run_id, status FROM insession_work_runs "
        "WHERE session_id=? AND insession_task_id=? AND subject_kind='task_node'",
        (command.session_id, command.task_id),
    ).fetchall()
    if any(str(row["status"]) in _NONTERMINAL_RUN_STATUSES for row in rows):
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.ACTIVE_EXECUTION_PRESENT,
            "TaskGraph revision cannot race a nonterminal TaskNode WorkRun",
        )


def _require_revision_authority_goal_binding(
    *,
    conn,
    revision_authority: (
        TaskGraphRevisionTrigger
        | TaskGraphExecutionReplanRequest
        | None
    ),
    details,
    failure_code: AuxiliaryTaskGraphCommitFailureCode,
) -> None:
    """将 N -> N+1 提交绑定到精确正向 Architect 权威。"""

    if revision_authority is None:
        return
    if not isinstance(
        revision_authority,
        (TaskGraphRevisionTrigger, TaskGraphExecutionReplanRequest),
    ):
        _fail(
            failure_code,
            "TaskGraph revision authority has an unsupported durable type",
        )
# 全 Task 验证器触发器早于目标作用域 Architect 回执。其触发器与应用、终态、语义和转换
# 账本仍是重放权威边界。执行期重规划请求始终要求下方更严格的精确正向规划完成项。
    if isinstance(revision_authority, TaskGraphRevisionTrigger):
        return
    authority_id = revision_authority.request_id
    authority_sha256 = revision_authority.request_sha256
    identity = _sha256_value(
        {
            "schema_version": "auxiliary-v2-positive-planning-identity-v1",
            "trigger_id": authority_id,
            "trigger_sha256": authority_sha256,
        }
    )[:32]
    expected_goal_id = f"auxgoalv2_positive_{identity}"
    if (
        details.goal_id != expected_goal_id
        or details.goal_objective != revision_authority.revision_objective
        or details.base_task_graph_revision
        != revision_authority.base_graph_revision
        or details.target_task_graph_revision
        != revision_authority.target_graph_revision
        or details.auxiliary_graph_revision < 2
    ):
        _fail(
            failure_code,
            "current Auxiliary goal is not the exact positive revision goal",
        )
    try:
        completion = initial_planning_records.load_authenticated_auxiliary_positive_planning_completion(
            conn,
            session_id=revision_authority.session_id,
            task_id=revision_authority.task_id,
            auxiliary_graph_id=details.auxiliary_graph_id,
            goal_id=expected_goal_id,
            revision_apply_id=f"auxv2positive_{identity}:architect",
        )
    except Exception as exc:
        _fail_corrupt("positive Architect completion authority is corrupt", exc)
    if completion is None:
        _fail(
            failure_code,
            "positive TaskGraph revision goal has no Architect completion",
        )
    binding = completion.binding
    architect_request = binding.architect_request
    current = architect_request.prompt_payload.current_revision
    semantic_base = architect_request.prompt_payload.task_graph_semantic_base
    if (
        binding.session_id != revision_authority.session_id
        or binding.task_id != revision_authority.task_id
        or binding.auxiliary_graph_id != details.auxiliary_graph_id
        or binding.goal_id != expected_goal_id
        or binding.revision_authority != revision_authority
        or architect_request.goal.session_id != revision_authority.session_id
        or architect_request.goal.task_id != revision_authority.task_id
        or architect_request.goal.auxiliary_graph_id
        != details.auxiliary_graph_id
        or architect_request.goal.goal_id != expected_goal_id
        or architect_request.goal.base_task_graph_revision
        != revision_authority.base_graph_revision
        or architect_request.goal.target_task_graph_revision
        != revision_authority.target_graph_revision
        or architect_request.prompt_payload.goal.objective
        != revision_authority.revision_objective
        or architect_request.prompt_payload.task_graph_revision_trigger
        != revision_authority
        or semantic_base is None
        or semantic_base.base_task_graph_revision
        != revision_authority.base_graph_revision
        or current is None
        or current.auxiliary_graph_revision
        != binding.bootstrap_auxiliary_graph_revision
        or current.base_task_graph_revision
        != revision_authority.base_graph_revision
        or completion.committed_auxiliary_graph_revision
        != binding.bootstrap_auxiliary_graph_revision + 1
        or completion.committed_auxiliary_graph_revision
        > details.auxiliary_graph_revision
    ):
        _fail(
            failure_code,
            "positive Architect request crossed exact revision authority",
        )


def _load_terminal_authority(
    conn,
    *,
    command,
    task,
    details,
    allow_committed: bool = False,
) -> _TerminalAuthority:
    terminal_row = conn.execute(
        "SELECT * FROM insession_auxiliary_v2_terminal_proposal_receipts "
        "WHERE terminal_proposal_receipt_id=?",
        (command.terminal_proposal_receipt_id,),
    ).fetchone()
    if terminal_row is None:
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.TERMINAL_RECEIPT_CORRUPT,
            "unknown terminal proposal receipt",
        )
    finish_row = conn.execute(
        "SELECT * FROM insession_auxiliary_v2_finish_gate_receipts "
        "WHERE finish_gate_receipt_id=?",
        (str(terminal_row["finish_gate_receipt_id"]),),
    ).fetchone()
    if finish_row is None:
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.TERMINAL_RECEIPT_CORRUPT,
            "terminal proposal lost its finish-gate receipt",
        )
    try:
        proposal = InSessionTaskGraphRevisionProposal.model_validate_json(
            str(terminal_row["proposal_json"])
        )
        output = OutputWindow.model_validate_json(
            str(terminal_row["output_window_json"])
        )
        finish = terminal_records.AuxiliaryFinishGateReceipt.model_validate_json(
            str(finish_row["receipt_json"])
        )
        output_proposal, lineage = terminal_records._parse_terminal_output_material(
            output.content,
            expected_base_task_graph_revision=(
                finish.base_task_graph_revision
            ),
        )
        terminal = terminal_records.AuxiliaryTerminalProposalReceipt(
            terminal_proposal_receipt_id=str(
                terminal_row["terminal_proposal_receipt_id"]
            ),
            finish_gate_receipt_id=str(terminal_row["finish_gate_receipt_id"]),
            session_id=str(terminal_row["session_id"]),
            task_id=str(terminal_row["insession_task_id"]),
            auxiliary_graph_id=str(terminal_row["auxiliary_graph_id"]),
            goal_id=str(terminal_row["goal_id"]),
            auxiliary_graph_revision=int(
                terminal_row["auxiliary_graph_revision"]
            ),
            terminal_completion_id=str(terminal_row["terminal_completion_id"]),
            proposal=proposal,
            lineage=lineage,
            proposal_sha256=str(terminal_row["proposal_sha256"]),
            output_window=output,
            output_window_sha256=str(terminal_row["output_window_sha256"]),
            created_turn_id=str(terminal_row["created_turn_id"]),
        )
        stored_evaluation = TaskGraphProductionEvaluation.model_validate_json(
            str(finish_row["production_evaluation_json"])
        )
    except (TypeError, ValueError, ValidationError) as exc:
        _fail_corrupt("terminal authority is not typed", exc)
    if (
        _model_json(proposal) != str(terminal_row["proposal_json"])
        or terminal.proposal_sha256
        != _sha256_value(proposal.model_dump(mode="json"))
        or _model_json(output) != str(terminal_row["output_window_json"])
        or terminal.output_window_sha256
        != _sha256_text(str(terminal_row["output_window_json"]))
        or output_proposal != proposal
        or _model_json(finish) != str(finish_row["receipt_json"])
        or _sha256_text(str(finish_row["receipt_json"]))
        != str(finish_row["receipt_sha256"])
        or _model_json(stored_evaluation)
        != str(finish_row["production_evaluation_json"])
        or stored_evaluation.evaluation_sha256
        != str(finish_row["production_evaluation_sha256"])
        or finish.production_evaluation_sha256
        != stored_evaluation.evaluation_sha256
        or finish.validation_context_sha256
        != str(finish_row["validation_context_sha256"])
    ):
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.TERMINAL_RECEIPT_CORRUPT,
            "terminal receipt hashes or typed bodies are corrupt",
        )
    current_base = command.expected_base_task_graph_revision
    target_revision = 1 if current_base is None else current_base + 1
    expected_ready_statuses = (
        {"committed"} if allow_committed else {"proposal_ready", "gapped_ready"}
    )
    if (
        terminal.session_id != command.session_id
        or terminal.task_id != command.task_id
        or terminal.created_turn_id != command.source_turn_id
        or finish.session_id != command.session_id
        or finish.task_id != command.task_id
        or finish.finish_gate_receipt_id != terminal.finish_gate_receipt_id
        or finish.terminal_completion_id != terminal.terminal_completion_id
        or finish.proposal_sha256 != terminal.proposal_sha256
        or finish.base_task_graph_revision != current_base
        or finish.target_task_graph_revision != target_revision
        or details.session_id != command.session_id
        or details.task_id != command.task_id
        or details.auxiliary_graph_id != terminal.auxiliary_graph_id
        or details.goal_id != terminal.goal_id
        or details.auxiliary_graph_revision != terminal.auxiliary_graph_revision
        or details.structure_sha256 != finish.structure_sha256
        or details.base_task_graph_revision != current_base
        or details.target_task_graph_revision != target_revision
        or details.goal_status not in expected_ready_statuses
        or details.revision_status not in expected_ready_statuses
        or (
            not allow_committed
            and details.goal_status != finish.readiness_status
        )
        or (
            not allow_committed
            and details.revision_status != finish.readiness_status
        )
        or details.budget is None
        or details.budget.snapshot_sha256 != finish.budget_snapshot_sha256
        or details.budget.assessment.disposition
        is PlanningEpisodeBudgetDisposition.HARD_LIMIT_REACHED
    ):
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.AUTHORITY_NOT_CURRENT,
            "terminal/finish authority is not the current ready revision",
        )

    membership_rows = conn.execute(
        "SELECT auxiliary_node_id, node_revision, ordinal, required, "
        "carried_completion_id FROM insession_auxiliary_graph_revision_nodes_v2 "
        "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=? "
        "ORDER BY ordinal",
        (details.auxiliary_graph_id, details.auxiliary_graph_revision),
    ).fetchall()
    membership = {str(row["auxiliary_node_id"]): row for row in membership_rows}
    try:
        completion_ids = (
            auxiliary_graph_records._load_auxiliary_frontier_completion_ids(
                conn,
                details=details,
                membership_by_node=membership,
            )
        )
        terminal_node, required_node_ids = (
            terminal_records._require_terminal_and_predecessors(
                finish,
                details=details,
                completion_ids=completion_ids,
            )
        )
        source_output, source_proposal, source_lineage, source_output_hash = (
            terminal_records._load_terminal_output(
                conn,
                command=finish,
                details=details,
                terminal_node=terminal_node,
                completion_ids=completion_ids,
            )
        )
    except terminal_records.AuxiliaryTerminalSealPersistenceError as exc:
        _fail_corrupt("terminal completion authority is corrupt", exc)
    expected_required = tuple(
        completion_ids[node_id]
        for node_id in required_node_ids
        if node_id != finish.terminal_auxiliary_node_id
    )
    if (
        source_output != output
        or source_proposal != proposal
        or source_lineage != terminal.lineage
        or source_output_hash != terminal.output_window_sha256
        or expected_required != finish.required_completion_ids
    ):
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.TERMINAL_RECEIPT_CORRUPT,
            "finish-gate completion closure has drifted",
        )
    try:
        settlement = semantic_records._load_auxiliary_semantic_quorum_settlement(
            conn,
            session_id=command.session_id,
            task_id=command.task_id,
            auxiliary_graph_id=terminal.auxiliary_graph_id,
            goal_id=terminal.goal_id,
            auxiliary_graph_revision=terminal.auxiliary_graph_revision,
            frozen_prompt_payload_sha256=finish.semantic_prompt_payload_sha256,
        )
    except semantic_records.AuxiliarySemanticVerificationPersistenceError as exc:
        _fail_corrupt("semantic settlement authority is corrupt", exc)
    if (
        settlement is None
        or settlement.host_disposition
        is not TaskGraphSemanticVerificationDisposition.PASS
        or settlement.settlement_id != finish.semantic_settlement_id
        or settlement.settlement_sha256 != finish.semantic_settlement_sha256
        or settlement.review_policy.policy_sha256
        != finish.semantic_review_policy_sha256
        or not settlement.requests
    ):
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.SEMANTIC_BINDING_MISMATCH,
            "finish-gate semantic settlement is missing, stale, or not PASS",
        )
    request = settlement.requests[0]
    frozen_goal = request.goal.model_dump(mode="json")
    current_goal = details.goal.model_dump(mode="json")
    frozen_goal.pop("status", None)
    frozen_goal.pop("state_version", None)
    current_goal.pop("status", None)
    current_goal.pop("state_version", None)
    if (
        not terminal_records._semantic_request_matches_terminal_material(
            request,
            proposal=proposal,
            lineage=terminal.lineage,
            expected_base_task_graph_revision=current_base,
        )
        or request.task_graph_proposal_sha256 != terminal.proposal_sha256
        or request.auxiliary_graph_structure_sha256 != details.structure_sha256
        or frozen_goal != current_goal
        or request.budget != details.budget
        or request.authority_projection.authority_snapshot_id
        != details.authority_snapshot_id
        or request.authority_projection.authority_snapshot_sha256
        != details.authority_snapshot_sha256
        or request.prompt_payload.payload_sha256
        != finish.semantic_prompt_payload_sha256
        or request.review_policy != settlement.review_policy
    ):
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.SEMANTIC_BINDING_MISMATCH,
            "semantic terminal authority drifted after its ready-state transition",
        )
    validation_context = terminal_records._derive_validation_context(
        conn,
        session_id=command.session_id,
        invocation_turn_id=command.source_turn_id,
        task_id=command.task_id,
        task=task,
        request=request,
        details=details,
        allowed_graph_statuses=frozenset(
            {"committed" if allow_committed else finish.readiness_status}
        ),
        require_current_task_graph_revision=not allow_committed,
    )
    validation = validate_insession_task_graph_revision(
        proposal,
        context=validation_context,
    )
    if validation.status != "accepted":
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.SEMANTIC_BINDING_MISMATCH,
            "sealed proposal no longer passes canonical validation",
        )
    production_context = terminal_records._derive_production_context(
        request=request,
        validation_context=validation_context,
        task_current_graph_revision=current_base,
    )
    evaluation = evaluate_task_graph_production(
        proposal,
        context=production_context,
    )
    if (
        not evaluation.passed
        or evaluation != stored_evaluation
        or evaluation.evaluation_sha256
        != finish.production_evaluation_sha256
        or validation_context != finish.validation_context
        or _sha256_value(validation_context.model_dump(mode="json"))
        != finish.validation_context_sha256
    ):
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.PRODUCTION_EVALUATION_FAILED,
            "Store re-evaluation does not match the sealed PASS authority",
        )
    return _TerminalAuthority(
        terminal=terminal,
        finish=finish,
        proposal=proposal,
        validation_context=validation_context,
        evaluation=evaluation,
        settlement=settlement,
        semantic_request=request,
    )


def _load_task_details_in_transaction(conn, *, task) -> InSessionTaskDetails:
    revision = int(task["current_graph_revision"])
    revision_row = conn.execute(
        "SELECT source_anchors_json, authorization_anchor_ids_json, "
        "required_anchor_ids_json FROM insession_task_graph_revisions "
        "WHERE insession_task_id=? AND graph_revision=?",
        (str(task["insession_task_id"]), revision),
    ).fetchone()
    if revision_row is None:
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
            "current base TaskGraph revision is missing",
        )
    rows = conn.execute(
        "SELECT node.insession_task_node_id, node.node_revision, "
        "node.node_kind, node.ordinal, node.title, node.objective, "
        "node.source_anchor_ids_json, node.acceptance_criteria_json, "
        "node.constraints_json, edge.parent_insession_task_node_id, "
        "state.status, state.state_version "
        "FROM insession_task_graph_nodes AS node "
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
        (str(task["insession_task_id"]), revision),
    ).fetchall()
    if not rows:
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
            "current base TaskGraph has no readable nodes",
        )
    try:
        return InSessionTaskDetails(
            insession_task_id=str(task["insession_task_id"]),
            session_id=str(task["session_id"]),
            task_state_version=int(task["state_version"]),
            title=str(task["root_title"]),
            objective=str(task["root_objective"]),
            current_graph_revision=revision,
            status=InSessionTaskStatus(str(task["current_status"])),
            nodes=tuple(task_records._node_projection(row) for row in rows),
            source_anchors=task_records._source_anchors_from_json(
                revision_row["source_anchors_json"]
            ),
            authorization_anchor_ids=task_records._decode_ids(
                revision_row["authorization_anchor_ids_json"]
            ),
            required_anchor_ids=task_records._decode_ids(
                revision_row["required_anchor_ids_json"]
            ),
        )
    except (TypeError, ValueError) as exc:
        _fail_corrupt("current base TaskGraph projection is corrupt", exc)


def _authenticate_base_alias_bindings(command, *, request, base) -> dict[str, str]:
    snapshot = request.base_task_graph
    if (
        snapshot is None
        or snapshot.base_task_graph_revision
        != command.expected_base_task_graph_revision
    ):
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.SEMANTIC_BINDING_MISMATCH,
            "positive-base commit lacks the exact semantic base snapshot",
        )
    bindings = {
        item.node_alias: item.insession_task_node_id
        for item in command.base_node_alias_bindings
    }
    aliases = {item.node_alias for item in snapshot.nodes}
    base_by_id = {
        str(item["insession_task_node_id"]): item for item in base.nodes
    }
    if (
        set(bindings) != aliases
        or set(bindings.values()) != set(base_by_id)
        or len(bindings) != len(base_by_id)
    ):
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.BASE_ALIAS_BINDING_INVALID,
            "base-node aliases do not exactly cover the frozen base graph",
        )
    alias_by_id = {node_id: alias for alias, node_id in bindings.items()}
    for projected in snapshot.nodes:
        actual = base_by_id[bindings[projected.node_alias]]
        actual_parent = actual["parent_insession_task_node_id"]
        actual_parent_alias = (
            alias_by_id[str(actual_parent)] if actual_parent is not None else None
        )
        if (
            int(actual["node_revision"]) != projected.node_revision
            or str(actual["node_kind"]) != projected.node_kind.value
            or actual_parent_alias != projected.parent_node_alias
            or str(actual["title"]) != projected.title
            or str(actual["objective"]) != projected.objective
            or tuple(
                sorted(str(item) for item in actual["source_anchor_ids"])
            )
            != projected.source_anchor_aliases
            or tuple(actual["acceptance_criteria"])
            != tuple(
                item.model_dump(mode="json")
                for item in projected.acceptance_criteria
            )
            or tuple(actual["constraints"]) != projected.constraints
            or str(actual["status"]) != projected.status.value
        ):
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.BASE_ALIAS_BINDING_INVALID,
                "semantic base alias does not reproduce its durable node",
            )
    root_actual_id = bindings[snapshot.root_node_alias]
    root_ids = {
        str(item["insession_task_node_id"])
        for item in base.nodes
        if item["node_kind"] == "root"
        and item["parent_insession_task_node_id"] is None
    }
    if root_ids != {root_actual_id}:
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.BASE_ALIAS_BINDING_INVALID,
            "semantic base root alias is not the durable root",
        )
    return bindings


def _derive_lineage_authority(command, *, authority):
    output_lineage = authority.terminal.lineage
    if (
        not output_lineage
        or output_lineage
        != authority.semantic_request.prompt_payload.lineage
    ):
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.SEMANTIC_BINDING_MISMATCH,
            "positive-base terminal lineage is missing or was not reviewed",
        )
    try:
        return InSessionTaskGraphRevisionLineageHints.create(
            task_id=command.task_id,
            expected_base_graph_revision=(
                command.expected_base_task_graph_revision
            ),
            task_graph_proposal_sha256=authority.terminal.proposal_sha256,
            production_evaluation_sha256=(
                authority.evaluation.evaluation_sha256
            ),
            semantic_verification_result_sha256s=tuple(
                result.result_sha256 for result in authority.settlement.results
            ),
            hints=tuple(
                InSessionTaskNodeLineageHint(
                    node_key=item.proposal_node_key,
                    disposition=TaskGraphNodeLineageDisposition(
                        item.disposition.value
                    ),
                    base_node_alias=item.base_node_alias,
                )
                for item in output_lineage
            ),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        _fail_corrupt("semantic lineage authority is not canonical", exc)


def _allocate_new_transition_node_ids(conn, deps, *, proposal, lineage):
    by_key = {item.node_key: item for item in lineage.hints}
    return {
        node.node_key: task_records._allocate_id(
            conn,
            deps,
            "insession_task_node",
        )
        for node in proposal.root.nodes
        if by_key[node.node_key].disposition
        is TaskGraphNodeLineageDisposition.NEW
    }


def _require_transition_states_safe(
    conn,
    *,
    command,
    authority,
    transition,
    base,
    base_ids_by_alias,
) -> None:
    del conn
    base_by_id = {
        str(item["insession_task_node_id"]): item for item in base.nodes
    }
    projected_by_alias = {
        item.node_alias: item
        for item in authority.semantic_request.base_task_graph.nodes
    }
    alias_by_id = {value: key for key, value in base_ids_by_alias.items()}
    for item in transition.nodes:
        if item.disposition is not TaskGraphNodeLineageDisposition.REUSE:
            continue
        assert item.base_node_id is not None
        status = str(base_by_id[item.base_node_id]["status"])
        if status == "proposed":
            continue
        if status != "completed" or not item.definition_carry_candidate:
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.CARRY_UNSAFE,
                "only proposed state or authenticated completed state may be reused",
            )
        projected = projected_by_alias[alias_by_id[item.base_node_id]]
        if (
            projected.status is not InSessionTaskStatus.COMPLETED
            or projected.completed_delivery_coverage
            is not TaskGraphSemanticDeliveryCoverage.FULL
            or projected.completed_delivery_gap_aliases
            or not projected.delivery_authority_aliases
        ):
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.CARRY_UNSAFE,
                "completed reuse lacks full, gap-free Delivery authority",
            )


def _insert_positive_revision(
    conn,
    *,
    command,
    proposal,
    transition,
    proposal_hash,
    source_manifest,
    now,
) -> None:
    revision = transition.target_graph_revision
    conn.execute(
        "INSERT INTO insession_task_graph_revisions "
        "(insession_task_id, graph_revision, source_turn_id, proposal_hash, "
        "source_anchors_json, authorization_anchor_ids_json, "
        "required_anchor_ids_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            command.task_id,
            revision,
            command.source_turn_id,
            proposal_hash,
            _canonical_json(
                [
                    item.model_dump(mode="json")
                    for item in source_manifest.source_anchors
                ]
            ),
            _canonical_json(list(source_manifest.authorization_anchor_ids)),
            _canonical_json(list(source_manifest.required_anchor_ids)),
            now,
        ),
    )
    by_key = {item.node_key: item for item in transition.nodes}
    for ordinal, node in enumerate(proposal.root.nodes):
        item = by_key[node.node_key]
        conn.execute(
            "INSERT INTO insession_task_graph_nodes "
            "(insession_task_id, graph_revision, insession_task_node_id, "
            "node_revision, node_kind, ordinal, title, objective, "
            "source_anchor_ids_json, acceptance_criteria_json, "
            "constraints_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                command.task_id,
                revision,
                item.target_node_id,
                item.target_node_revision,
                node.node_kind.value,
                ordinal,
                node.title,
                node.objective,
                _canonical_json(list(node.source_anchor_ids)),
                _canonical_json(
                    [
                        acceptance.model_dump(mode="json")
                        for acceptance in node.acceptance_criteria
                    ]
                ),
                _canonical_json(list(node.constraints)),
                now,
            ),
        )
        if item.disposition is not TaskGraphNodeLineageDisposition.REUSE:
            conn.execute(
                "INSERT INTO insession_task_node_states "
                "(insession_task_id, insession_task_node_id, node_revision, "
                "status, state_version, updated_at) VALUES (?, ?, ?, 'proposed', 1, ?)",
                (
                    command.task_id,
                    item.target_node_id,
                    item.target_node_revision,
                    now,
                ),
            )
    for ordinal, node in enumerate(proposal.root.nodes):
        if node.parent_node_key is None:
            continue
        item = by_key[node.node_key]
        parent = by_key[node.parent_node_key]
        conn.execute(
            "INSERT INTO insession_task_graph_edges "
            "(insession_task_id, graph_revision, child_insession_task_node_id, "
            "parent_insession_task_node_id, ordinal) VALUES (?, ?, ?, ?, ?)",
            (
                command.task_id,
                revision,
                item.target_node_id,
                parent.target_node_id,
                ordinal,
            ),
        )


def _derive_completed_carry_receipts(
    conn,
    *,
    command,
    authority,
    transition,
    base,
    base_ids_by_alias,
) -> tuple[AuxiliaryTaskGraphNodeCarryReceipt, ...]:
    base_by_id = {
        str(item["insession_task_node_id"]): item for item in base.nodes
    }
    alias_by_id = {value: key for key, value in base_ids_by_alias.items()}
    semantic_by_alias = {
        item.node_alias: item
        for item in authority.semantic_request.base_task_graph.nodes
    }
    transition_by_id = {item.target_node_id: item for item in transition.nodes}
    target_children: dict[str, list[object]] = {
        item.target_node_id: [] for item in transition.nodes
    }
    for item in transition.nodes:
        if item.target_parent_node_id is not None:
            target_children[item.target_parent_node_id].append(item)
    cards = {
        item.alias: item
        for item in authority.semantic_request.authority_projection.cards
    }
    derived: dict[str, AuxiliaryTaskGraphNodeCarryReceipt] = {}
    visiting: set[str] = set()

    def prove(node_id: str) -> AuxiliaryTaskGraphNodeCarryReceipt:
        existing = derived.get(node_id)
        if existing is not None:
            return existing
        if node_id in visiting:
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.CARRY_UNSAFE,
                "completed carry dependency closure contains a cycle",
            )
        visiting.add(node_id)
        item = transition_by_id[node_id]
        base_node = base_by_id[node_id]
        if (
            item.disposition is not TaskGraphNodeLineageDisposition.REUSE
            or str(base_node["status"]) != "completed"
            or not item.definition_carry_candidate
        ):
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.CARRY_UNSAFE,
                "completed carry closure contains a non-reusable child",
            )
        semantic_node = semantic_by_alias[alias_by_id[node_id]]
        if (
            semantic_node.completed_delivery_coverage
            is not TaskGraphSemanticDeliveryCoverage.FULL
            or semantic_node.completed_delivery_gap_aliases
            or not semantic_node.delivery_authority_aliases
        ):
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.CARRY_UNSAFE,
                "completed node has no full semantic Delivery projection",
            )
        expected_subject = TaskNodeSubject(
            task_id=command.task_id,
            graph_revision=transition.base_graph_revision,
            node_id=node_id,
            node_revision=item.target_node_revision,
        )
        try:
            source_projection = (
                work_execution_records._load_current_task_node_delivery_projection(
                    conn,
                    session_id=command.session_id,
                    task_id=command.task_id,
                    graph_revision=transition.base_graph_revision,
                    node_id=node_id,
                    node_revision=item.target_node_revision,
                )
            )
            resolved = source_projection.source_delivery
            delivery_id = source_projection.delivery_id
        except work_execution_records.WorkExecutionPersistenceError as exc:
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.CARRY_UNSAFE,
                f"completed reuse source chain is not authentic: {exc}",
            )
        except (
            TypeError,
            ValueError,
            ValidationError,
        ) as exc:
            _fail_corrupt("completed base Delivery chain is not authentic", exc)
        if source_projection.target_subject != expected_subject:
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.CARRY_UNSAFE,
                "completed reuse source chain targets another base node",
            )
        try:
            reloaded_source = verification_records._load_task_node_delivery(
                conn,
                session_id=command.session_id,
                delivery_id=delivery_id,
            )
        except (
            verification_records.WorkExecutionPersistenceError,
            TypeError,
            ValueError,
            ValidationError,
        ) as exc:
            _fail_corrupt("completed base Delivery is not authentic", exc)
        if (
            reloaded_source != resolved
            or resolved.delivery.subject.task_id != expected_subject.task_id
            or resolved.delivery.subject.node_id != expected_subject.node_id
            or resolved.delivery.subject.node_revision
            != expected_subject.node_revision
            or resolved.delivery.subject.graph_revision
            > expected_subject.graph_revision
            or resolved.output_window.content
            != semantic_node.completed_delivery_summary
        ):
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.CARRY_UNSAFE,
                "completed Delivery body or subject differs from semantic authority",
            )
        try:
            request_row = verification_records._load_request_row(
                conn,
                session_id=command.session_id,
                verification_request_id=(
                    resolved.delivery.verification_request_id
                ),
            )
            verification = verification_records._record_from_row(request_row)
        except (
            verification_records.WorkExecutionPersistenceError,
            TypeError,
            ValueError,
            ValidationError,
        ) as exc:
            _fail_corrupt("completed Delivery verification is corrupt", exc)
        target_child_items = target_children[node_id]
        base_child_rows = conn.execute(
            "SELECT child.insession_task_node_id, child.node_revision "
            "FROM insession_task_graph_edges AS edge "
            "JOIN insession_task_graph_nodes AS child "
            "ON child.insession_task_id=edge.insession_task_id "
            "AND child.graph_revision=edge.graph_revision "
            "AND child.insession_task_node_id=edge.child_insession_task_node_id "
            "WHERE edge.insession_task_id=? AND edge.graph_revision=? "
            "AND edge.parent_insession_task_node_id=? "
            "ORDER BY child.ordinal, child.insession_task_node_id",
            (command.task_id, transition.base_graph_revision, node_id),
        ).fetchall()
        base_child_pairs = tuple(
            (str(row["insession_task_node_id"]), int(row["node_revision"]))
            for row in base_child_rows
        )
        target_child_pairs = tuple(
            (child.target_node_id, child.target_node_revision)
            for child in target_child_items
        )
        if base_child_pairs != target_child_pairs:
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.CARRY_UNSAFE,
                "completed reuse changed its direct dependency closure",
            )
        child_receipts = tuple(prove(child_id) for child_id, _ in target_child_pairs)
        dependency_delivery_ids = tuple(
            child.source_delivery_id for child in child_receipts
        )
        if verification.request.dependency_delivery_ids != dependency_delivery_ids:
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.CARRY_UNSAFE,
                "completed reuse verification names another dependency closure",
            )
        attempt_row = conn.execute(
            "SELECT catalog_snapshot_json, catalog_snapshot_hash FROM "
            "insession_work_run_attempts WHERE work_run_id=? AND attempt_id=?",
            (
                resolved.delivery.work_run_id,
                resolved.delivery.submitted_attempt_id,
            ),
        ).fetchone()
        if (
            attempt_row is None
            or _sha256_text(str(attempt_row["catalog_snapshot_json"]))
            != str(attempt_row["catalog_snapshot_hash"])
        ):
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.CARRY_UNSAFE,
                "completed reuse capability catalog is missing or corrupt",
            )
        try:
            authority_cards = tuple(
                cards[alias]
                for alias in semantic_node.delivery_authority_aliases
            )
        except KeyError as exc:
            _fail_corrupt("Delivery authority alias is unknown", exc)
        source_cards = tuple(
            cards[alias] for alias in semantic_node.source_anchor_aliases
        )
        dependency_closure_sha256 = _sha256_value(
            {
                "node_id": node_id,
                "base_children": base_child_pairs,
                "target_children": target_child_pairs,
                "dependency_delivery_ids": dependency_delivery_ids,
            }
        )
        source_authority_sha256 = _sha256_value(
            {
                "source_anchor_aliases": semantic_node.source_anchor_aliases,
                "source_cards": [
                    card.model_dump(mode="json") for card in source_cards
                ],
            }
        )
        freshness_authority_sha256 = _sha256_value(
            {
                "semantic_request_binding_sha256": (
                    authority.semantic_request.binding_sha256
                ),
                "semantic_settlement_sha256": (
                    authority.settlement.settlement_sha256
                ),
                "semantic_base_projection_sha256": (
                    authority.semantic_request.base_task_graph.projection_sha256
                ),
                "delivery_id": delivery_id,
                "source_resolution": (
                    source_projection.model_dump(mode="json")
                ),
                "delivery_output_sha256": _sha256_text(
                    resolved.output_window.content
                ),
                "delivery_authority_cards": [
                    card.model_dump(mode="json") for card in authority_cards
                ],
            }
        )
        receipt_seed = _sha256_value(
            {
                "apply_id": command.apply_id,
                "node_id": node_id,
                "node_revision": item.target_node_revision,
                "delivery_id": delivery_id,
                "definition_sha256": item.definition_sha256,
                "dependency_closure_sha256": dependency_closure_sha256,
                "source_authority_sha256": source_authority_sha256,
                "capability_catalog_sha256": str(
                    attempt_row["catalog_snapshot_hash"]
                ),
                "freshness_authority_sha256": freshness_authority_sha256,
            }
        )
        receipt = AuxiliaryTaskGraphNodeCarryReceipt(
            carry_receipt_id=f"aux-v2-carry-{receipt_seed[:40]}",
            apply_id=command.apply_id,
            session_id=command.session_id,
            task_id=command.task_id,
            base_task_graph_revision=transition.base_graph_revision,
            target_task_graph_revision=transition.target_graph_revision,
            node_id=node_id,
            node_revision=item.target_node_revision,
            source_delivery_id=delivery_id,
            definition_sha256=item.definition_sha256,
            dependency_delivery_ids=dependency_delivery_ids,
            dependency_closure_sha256=dependency_closure_sha256,
            source_authority_sha256=source_authority_sha256,
            capability_catalog_sha256=str(attempt_row["catalog_snapshot_hash"]),
            freshness_authority_sha256=freshness_authority_sha256,
        )
        visiting.remove(node_id)
        derived[node_id] = receipt
        return receipt

    for item in transition.nodes:
        if (
            item.disposition is TaskGraphNodeLineageDisposition.REUSE
            and str(base_by_id[item.target_node_id]["status"]) == "completed"
        ):
            prove(item.target_node_id)
    return tuple(
        derived[node_id]
        for node_id in sorted(
            derived,
            key=lambda value: (
                next(
                    index
                    for index, item in enumerate(transition.nodes)
                    if item.target_node_id == value
                ),
                value,
            ),
        )
    )


def _commit_auxiliary_projection(conn, *, details, now) -> None:
    if conn.execute(
        "UPDATE insession_auxiliary_graph_revision_states_v2 "
        "SET status='committed', state_version=state_version+1, updated_at=? "
        "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=? "
        "AND status IN ('proposal_ready','gapped_ready') AND state_version=?",
        (
            now,
            details.auxiliary_graph_id,
            details.auxiliary_graph_revision,
            details.revision_state_version,
        ),
    ).rowcount != 1:
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.STATE_VERSION_CONFLICT,
            "AuxiliaryGraph revision changed during TaskGraph commit",
        )
    if conn.execute(
        "UPDATE insession_auxiliary_graph_goals "
        "SET status='committed', state_version=state_version+1, updated_at=? "
        "WHERE session_id=? AND insession_task_id=? AND auxiliary_graph_id=? "
        "AND goal_id=? AND status IN ('proposal_ready','gapped_ready') "
        "AND state_version=?",
        (
            now,
            details.session_id,
            details.task_id,
            details.auxiliary_graph_id,
            details.goal_id,
            details.goal_state_version,
        ),
    ).rowcount != 1:
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.STATE_VERSION_CONFLICT,
            "AuxiliaryGraph goal changed during TaskGraph commit",
        )


def _target_snapshot_json(conn, *, task_id: str, graph_revision: int) -> str:
    revision = conn.execute(
        "SELECT * FROM insession_task_graph_revisions "
        "WHERE insession_task_id=? AND graph_revision=?",
        (task_id, graph_revision),
    ).fetchone()
    nodes = conn.execute(
        "SELECT node.* FROM insession_task_graph_nodes AS node "
        "WHERE node.insession_task_id=? AND node.graph_revision=? "
        "ORDER BY node.ordinal, node.insession_task_node_id",
        (task_id, graph_revision),
    ).fetchall()
    edges = conn.execute(
        "SELECT * FROM insession_task_graph_edges "
        "WHERE insession_task_id=? AND graph_revision=? "
        "ORDER BY ordinal, child_insession_task_node_id",
        (task_id, graph_revision),
    ).fetchall()
    if revision is None or not nodes:
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
            "committed TaskGraph snapshot is incomplete",
        )
    return _canonical_json(
        {
            "schema_version": "auxiliary-v2-committed-task-graph-snapshot-v1",
            "task_id": task_id,
            "graph_revision": graph_revision,
            "revision": dict(revision),
            "nodes": [dict(row) for row in nodes],
            "edges": [dict(row) for row in edges],
        }
    )


def _insert_commit_receipt(
    conn,
    *,
    command,
    command_sha256,
    authority,
    transition,
    target_snapshot_json,
    result,
    now,
) -> None:
    transition_json = _model_json(transition) if transition is not None else None
    result_json = _model_json(result)
    conn.execute(
        "INSERT INTO insession_auxiliary_v2_task_graph_commit_receipts "
        "(apply_id, operation, session_id, insession_task_id, source_turn_id, "
        "terminal_proposal_receipt_id, finish_gate_receipt_id, "
        "auxiliary_graph_id, goal_id, auxiliary_graph_revision, "
        "base_task_graph_revision, committed_task_graph_revision, "
        "expected_task_state_version, committed_task_state_version, "
        "expected_window_revision, committed_window_state_version, "
        "turn_task_link_revision, command_sha256, proposal_sha256, "
        "validation_context_sha256, production_evaluation_sha256, "
        "semantic_settlement_id, semantic_settlement_sha256, "
        "transition_json, transition_sha256, target_snapshot_json, "
        "target_snapshot_sha256, result_json, result_sha256, created_at) "
        "VALUES (?, 'commit_auxiliary_v2_task_graph_proposal', ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            command.apply_id,
            command.session_id,
            command.task_id,
            command.source_turn_id,
            authority.terminal.terminal_proposal_receipt_id,
            authority.finish.finish_gate_receipt_id,
            authority.finish.auxiliary_graph_id,
            authority.finish.goal_id,
            authority.finish.auxiliary_graph_revision,
            command.expected_base_task_graph_revision,
            result.committed_graph_revision,
            command.expected_task_state_version,
            result.task_state_version,
            command.expected_window_revision,
            result.window_state_version,
            result.turn_task_link_revision,
            command_sha256,
            authority.terminal.proposal_sha256,
            authority.finish.validation_context_sha256,
            authority.evaluation.evaluation_sha256,
            authority.settlement.settlement_id,
            authority.settlement.settlement_sha256,
            transition_json,
            transition.transition_sha256 if transition is not None else None,
            target_snapshot_json,
            _sha256_text(target_snapshot_json),
            result_json,
            _sha256_text(result_json),
            now,
        ),
    )


def _insert_carry_receipt(conn, *, receipt, now) -> None:
    receipt_json = _model_json(receipt)
    conn.execute(
        "INSERT INTO insession_auxiliary_v2_task_graph_node_carry_receipts "
        "(carry_receipt_id, apply_id, session_id, insession_task_id, "
        "base_task_graph_revision, target_task_graph_revision, "
        "insession_task_node_id, node_revision, source_delivery_id, "
        "definition_sha256, dependency_delivery_ids_json, "
        "dependency_closure_sha256, source_authority_sha256, "
        "capability_catalog_sha256, freshness_authority_sha256, "
        "receipt_json, receipt_sha256, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            receipt.carry_receipt_id,
            receipt.apply_id,
            receipt.session_id,
            receipt.task_id,
            receipt.base_task_graph_revision,
            receipt.target_task_graph_revision,
            receipt.node_id,
            receipt.node_revision,
            receipt.source_delivery_id,
            receipt.definition_sha256,
            _canonical_json(list(receipt.dependency_delivery_ids)),
            receipt.dependency_closure_sha256,
            receipt.source_authority_sha256,
            receipt.capability_catalog_sha256,
            receipt.freshness_authority_sha256,
            receipt_json,
            _sha256_text(receipt_json),
            now,
        ),
    )


def _load_replay(
    conn,
    *,
    command,
    command_sha256,
    row,
) -> AuxiliaryTaskGraphCommitResult:
    result_json = str(row["result_json"])
    if str(row["command_sha256"]) != command_sha256:
        _fail_collision("TaskGraph commit apply ID has another command")
    try:
        result = AuxiliaryTaskGraphCommitResult.model_validate_json(
            result_json
        )
    except (TypeError, ValueError, ValidationError) as exc:
        _fail_corrupt("TaskGraph commit replay result is not typed", exc)
    if (
        str(row["operation"])
        != "commit_auxiliary_v2_task_graph_proposal"
        or str(row["session_id"]) != command.session_id
        or str(row["insession_task_id"]) != command.task_id
        or str(row["source_turn_id"]) != command.source_turn_id
        or str(row["terminal_proposal_receipt_id"])
        != command.terminal_proposal_receipt_id
        or (
            int(row["base_task_graph_revision"])
            if row["base_task_graph_revision"] is not None
            else None
        )
        != command.expected_base_task_graph_revision
        or int(row["expected_task_state_version"])
        != command.expected_task_state_version
        or int(row["expected_window_revision"])
        != command.expected_window_revision
        or _model_json(result) != result_json
        or _sha256_text(result_json) != str(row["result_sha256"])
        or result.status != "applied"
        or result.apply_id != command.apply_id
        or result.terminal_proposal_receipt_id
        != command.terminal_proposal_receipt_id
        or result.previous_graph_revision
        != command.expected_base_task_graph_revision
        or result.committed_graph_revision
        != int(row["committed_task_graph_revision"])
        or result.task_state_version
        != int(row["committed_task_state_version"])
        or result.window_state_version
        != int(row["committed_window_state_version"])
        or result.turn_task_link_revision
        != int(row["turn_task_link_revision"])
    ):
        _fail_collision("TaskGraph commit receipt has another identity")
    snapshot_json = _target_snapshot_json(
        conn,
        task_id=command.task_id,
        graph_revision=result.committed_graph_revision,
    )
    if (
        snapshot_json != str(row["target_snapshot_json"])
        or _sha256_text(snapshot_json)
        != str(row["target_snapshot_sha256"])
    ):
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
            "committed TaskGraph immutable snapshot is corrupt",
        )
    task = task_records._load_revision_target(
        conn,
        session_id=command.session_id,
        target_insession_task_id=command.task_id,
    )
    details = auxiliary_graph_records._load_auxiliary_graph(
        conn,
        command.session_id,
        command.task_id,
    )
    authority = _load_terminal_authority(
        conn,
        command=command,
        task=task,
        details=details,
        allow_committed=True,
    )
    if (
        result.finish_gate_receipt_id != authority.finish.finish_gate_receipt_id
        or result.proposal_sha256 != authority.terminal.proposal_sha256
        or str(row["proposal_sha256"])
        != authority.terminal.proposal_sha256
        or str(row["validation_context_sha256"])
        != authority.finish.validation_context_sha256
        or str(row["production_evaluation_sha256"])
        != authority.evaluation.evaluation_sha256
        or str(row["semantic_settlement_id"])
        != authority.settlement.settlement_id
        or str(row["semantic_settlement_sha256"])
        != authority.settlement.settlement_sha256
    ):
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
            "TaskGraph replay lost terminal/semantic/production binding",
        )
    revision_authority = _require_replay_revision_application(
        conn,
        command=command,
        result=result,
    )
    _require_revision_authority_goal_binding(
        conn=conn,
        revision_authority=revision_authority,
        details=details,
        failure_code=(
            AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT
        ),
    )
    transition_json = row["transition_json"]
    if command.expected_base_task_graph_revision is None:
        if transition_json is not None or result.transition_sha256 is not None:
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
                "base-null replay unexpectedly carries a transition",
            )
    else:
        try:
            transition = InSessionTaskGraphRevisionTransition.model_validate_json(
                str(transition_json)
            )
        except (TypeError, ValueError, ValidationError) as exc:
            _fail_corrupt("stored TaskGraph transition is not typed", exc)
        if (
            _model_json(transition) != str(transition_json)
            or transition.transition_sha256 != str(row["transition_sha256"])
            or transition.transition_sha256 != result.transition_sha256
        ):
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
                "stored TaskGraph transition hash is corrupt",
            )
        if (
            isinstance(
                revision_authority,
                TaskGraphExecutionReplanRequest,
            )
            and {
                item.node_alias: item.insession_task_node_id
                for item in command.base_node_alias_bindings
            }.get(revision_authority.source_node_alias)
            != revision_authority.requesting_subject.node_id
        ):
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
                "execution request replay crossed its source-node alias",
            )
        _require_triggered_revision(revision_authority, transition)
    carry_rows = conn.execute(
        "SELECT * FROM "
        "insession_auxiliary_v2_task_graph_node_carry_receipts "
        "WHERE apply_id=? ORDER BY carry_receipt_id",
        (command.apply_id,),
    ).fetchall()
    carry_ids: list[str] = []
    for carry_row in carry_rows:
        try:
            carry = AuxiliaryTaskGraphNodeCarryReceipt.model_validate_json(
                str(carry_row["receipt_json"])
            )
        except (TypeError, ValueError, ValidationError) as exc:
            _fail_corrupt("stored completed-node carry is not typed", exc)
        if (
            _model_json(carry) != str(carry_row["receipt_json"])
            or _sha256_text(str(carry_row["receipt_json"]))
            != str(carry_row["receipt_sha256"])
            or carry.carry_receipt_id != str(carry_row["carry_receipt_id"])
            or carry.apply_id != command.apply_id
            or carry.source_delivery_id != str(carry_row["source_delivery_id"])
            or _canonical_json(list(carry.dependency_delivery_ids))
            != str(carry_row["dependency_delivery_ids_json"])
        ):
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
                "stored completed-node carry receipt is corrupt",
            )
        try:
            projection = (
                work_execution_records._load_current_task_node_delivery_projection(
                    conn,
                    session_id=command.session_id,
                    task_id=command.task_id,
                    graph_revision=carry.target_task_graph_revision,
                    node_id=carry.node_id,
                    node_revision=carry.node_revision,
                )
            )
        except (
            work_execution_records.WorkExecutionPersistenceError,
            TypeError,
            ValueError,
            ValidationError,
        ) as exc:
            _fail_corrupt(
                "carry source receipt/Delivery chain is no longer authentic",
                exc,
            )
        if projection.delivery_id != carry.source_delivery_id:
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
                "carry replay resolves another historical Delivery",
            )
        carry_ids.append(carry.carry_receipt_id)
    if tuple(sorted(result.carry_receipt_ids)) != tuple(sorted(carry_ids)):
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
            "TaskGraph result/carry receipt set is inconsistent",
        )
    return result.model_copy(update={"status": "replayed"})


def _load_stored_execution_replan_request(
    conn,
    *,
    request_id: str,
    session_id: str,
    task_id: str,
) -> TaskGraphExecutionReplanRequest:
    row = conn.execute(
        "SELECT * FROM insession_task_graph_execution_replan_requests "
        "WHERE request_id=?",
        (request_id,),
    ).fetchone()
    if row is None:
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
            "TaskGraph execution-request history is missing",
        )
    try:
        request = TaskGraphExecutionReplanRequest.model_validate_json(
            str(row["request_json"])
        )
    except (TypeError, ValueError, ValidationError) as exc:
        _fail_corrupt("TaskGraph execution-request history is not typed", exc)
    support_json = _canonical_json(list(request.supporting_tool_result_ids))
    if (
        _model_json(request) != str(row["request_json"])
        or request.request_id != str(row["request_id"])
        or request.create_apply_id != str(row["create_apply_id"])
        or request.request_sha256 != str(row["request_sha256"])
        or request.session_id != str(row["session_id"])
        or request.task_id != str(row["insession_task_id"])
        or request.base_graph_revision != int(row["base_graph_revision"])
        or request.target_graph_revision != int(row["target_graph_revision"])
        or request.work_run_id != str(row["work_run_id"])
        or request.attempt_id != str(row["attempt_id"])
        or request.requesting_subject.node_id
        != str(row["insession_task_node_id"])
        or request.requesting_subject.node_revision != int(row["node_revision"])
        or request.source_node_alias != str(row["source_node_alias"])
        or request.reason.value != str(row["reason"])
        or request.diagnosis != str(row["diagnosis"])
        or request.revision_objective != str(row["revision_objective"])
        or support_json != str(row["supporting_tool_result_ids_json"])
        or _sha256_text(support_json)
        != str(row["supporting_tool_result_ids_sha256"])
        or request.task_state_version != int(row["task_state_version"])
        or request.created_turn_id != str(row["created_turn_id"])
        or request.session_id != session_id
        or request.task_id != task_id
    ):
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
            "TaskGraph execution-request history is corrupt",
        )
    return request


def _require_replay_revision_application(conn, *, command, result):
    trigger_id = command.task_graph_revision_trigger_id
    request_id = command.task_graph_execution_replan_request_id
    if trigger_id is None and request_id is None:
        trigger_rows = conn.execute(
            "SELECT apply_id FROM "
            "insession_task_graph_revision_trigger_applications "
            "WHERE task_graph_commit_apply_id=?",
            (command.apply_id,),
        ).fetchall()
        request_rows = conn.execute(
            "SELECT apply_id FROM "
            "insession_task_graph_execution_replan_applications "
            "WHERE task_graph_commit_apply_id=?",
            (command.apply_id,),
        ).fetchall()
        if trigger_rows or request_rows:
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
                "commit unexpectedly consumed TaskGraph revision authority",
            )
        return None

    if request_id is not None:
        trigger_rows = conn.execute(
            "SELECT apply_id FROM "
            "insession_task_graph_revision_trigger_applications "
            "WHERE task_graph_commit_apply_id=?",
            (command.apply_id,),
        ).fetchall()
        if trigger_rows:
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
                "execution-request commit also consumed a whole-Task trigger",
            )
        request = _load_stored_execution_replan_request(
            conn,
            request_id=request_id,
            session_id=command.session_id,
            task_id=command.task_id,
        )
        rows = conn.execute(
            "SELECT * FROM "
            "insession_task_graph_execution_replan_applications "
            "WHERE task_graph_commit_apply_id=? OR request_id=?",
            (command.apply_id, request_id),
        ).fetchall()
        if len(rows) != 1:
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
                "execution-request commit lost its unique application receipt",
            )
        try:
            application = TaskGraphExecutionReplanApplication.model_validate_json(
                str(rows[0]["receipt_json"])
            )
        except (TypeError, ValueError, ValidationError) as exc:
            _fail_corrupt(
                "TaskGraph execution-request application is not typed",
                exc,
            )
        active = conn.execute(
            "SELECT request_id FROM "
            "insession_active_task_graph_execution_replan_requests "
            "WHERE request_id=?",
            (request_id,),
        ).fetchall()
        expected_application_hash = _sha256_value(
            {
                "contract": (
                    "atomic-task-graph-execution-replan-application-id-v1"
                ),
                "task_graph_commit_apply_id": command.apply_id,
                "request_id": request_id,
            }
        )
        if (
            active
            or _model_json(application) != str(rows[0]["receipt_json"])
            or application.request_id != str(rows[0]["request_id"])
            or application.request_sha256 != str(rows[0]["request_sha256"])
            or application.session_id != str(rows[0]["session_id"])
            or application.task_id != str(rows[0]["insession_task_id"])
            or application.base_graph_revision
            != int(rows[0]["base_graph_revision"])
            or application.committed_graph_revision
            != int(rows[0]["committed_graph_revision"])
            or application.task_graph_commit_apply_id
            != str(rows[0]["task_graph_commit_apply_id"])
            or application.consumed_turn_id
            != str(rows[0]["consumed_turn_id"])
            or application.receipt_sha256 != str(rows[0]["receipt_sha256"])
            or application.apply_id != str(rows[0]["apply_id"])
            or application.apply_id
            != f"tger_apply_{expected_application_hash[:40]}"
            or application.request_id != request.request_id
            or application.request_sha256 != request.request_sha256
            or application.request_sha256
            != command.expected_task_graph_execution_replan_request_sha256
            or application.session_id != command.session_id
            or application.task_id != command.task_id
            or application.base_graph_revision
            != command.expected_base_task_graph_revision
            or application.committed_graph_revision
            != result.committed_graph_revision
            or application.committed_graph_revision
            != request.target_graph_revision
            or application.task_graph_commit_apply_id != command.apply_id
            or application.consumed_turn_id != command.source_turn_id
        ):
            _fail(
                AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
                "execution-request application crossed commit authority",
            )
        return request

    rows = conn.execute(
        "SELECT * FROM insession_task_graph_revision_trigger_applications "
        "WHERE task_graph_commit_apply_id=? OR trigger_id=?",
        (command.apply_id, trigger_id),
    ).fetchall()
    if len(rows) != 1:
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
            "triggered TaskGraph commit lost its unique application receipt",
        )
    try:
        application = task_delivery_validation_records._application_from_row(
            rows[0]
        )
        trigger = (
            task_delivery_validation_records._load_authenticated_trigger_authority(
                conn,
                trigger_id,
            )
        )
    except Exception as exc:
        _fail_corrupt(
            "TaskGraph trigger application chain is corrupt",
            exc,
        )
    active = conn.execute(
        "SELECT trigger_id FROM insession_active_task_graph_revision_triggers "
        "WHERE trigger_id=?",
        (trigger_id,),
    ).fetchall()
    expected_application_hash = _sha256_value(
        {
            "contract": "atomic-task-graph-revision-trigger-application-id-v1",
            "task_graph_commit_apply_id": command.apply_id,
            "trigger_id": trigger_id,
        }
    )
    if (
        active
        or application.apply_id
        != f"tgr_apply_{expected_application_hash[:40]}"
        or application.trigger_id != trigger.trigger_id
        or application.trigger_sha256 != trigger.trigger_sha256
        or application.trigger_sha256
        != command.expected_task_graph_revision_trigger_sha256
        or application.session_id != command.session_id
        or application.task_id != command.task_id
        or application.base_graph_revision
        != command.expected_base_task_graph_revision
        or application.committed_graph_revision
        != result.committed_graph_revision
        or application.committed_graph_revision
        != trigger.target_graph_revision
        or application.task_graph_commit_apply_id != command.apply_id
        or application.consumed_turn_id != command.source_turn_id
    ):
        _fail(
            AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
            "TaskGraph trigger application receipt crossed commit authority",
        )
    return trigger


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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_value(value: object) -> str:
    return _sha256_text(_canonical_json(value))


def _fail(code, message):
    raise AuxiliaryTaskGraphCommitPersistenceError(code, message)


def _fail_collision(message: str) -> None:
    raise AuxiliaryTaskGraphCommitIdentityCollision(
        AuxiliaryTaskGraphCommitFailureCode.APPLY_ID_COLLISION,
        message,
    )


def _fail_corrupt(message: str, cause: BaseException) -> None:
    raise AuxiliaryTaskGraphCommitPersistenceError(
        AuxiliaryTaskGraphCommitFailureCode.STORED_AUTHORITY_CORRUPT,
        message,
    ) from cause


__all__ = [
    "AuxiliaryBaseNodeAliasBinding",
    "AuxiliaryTaskGraphCommitFailureCode",
    "AuxiliaryTaskGraphCommitIdentityCollision",
    "AuxiliaryTaskGraphCommitPersistenceError",
    "AuxiliaryTaskGraphCommitResult",
    "AuxiliaryTaskGraphNodeCarryReceipt",
    "CommitAuxiliaryTaskGraphProposalCommand",
    "commit_auxiliary_task_graph_proposal",
]
