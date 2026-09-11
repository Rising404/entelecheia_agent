"""AuxiliaryGraph 终态提案的原子密封。

本模块是唯一可以把已完成 终态规划器投影为 ``proposal_ready`` 或 ``gapped_ready``
的持久化边界。它刻意在 TaskGraph 提交前停止，不读写已退役终态回执接缝。
"""

from __future__ import annotations

from enum import StrEnum
import hashlib
import json
import sqlite3
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from personagraph.l2.auxiliary_graph import (
    AuxiliaryNodeExecutorKind,
    PlanningEpisodeBudgetDisposition,
    TaskGraphRevisionCandidate,
    TaskGraphSemanticLineageProjection,
    TaskGraphSemanticVerificationDisposition,
)
from personagraph.l2.task_graph import (
    InSessionTaskGraphLimits,
    InSessionTaskGraphRevisionProposal,
    InSessionTaskGraphRevisionValidationContext,
    bind_used_evidence_to_required_acceptance_coverage,
    validate_insession_task_graph_revision,
)
from personagraph.l2.task_graph.production_gate import (
    TaskGraphAcceptanceRef,
    TaskGraphNodeExecutionRequirement,
    TaskGraphPlanningGap,
    TaskGraphProductionCapabilityContext,
    TaskGraphProductionEvaluationContext,
    TaskGraphProductionEvaluation,
    TaskGraphProductionFreshness,
    evaluate_task_graph_production,
)
from personagraph.l2.work_run import OutputWindow
from . import auxiliary_graphs as auxiliary_graph_records
from ..delivery import auxiliary_semantic_verification as semantic_records
from . import auxiliary_terminal_validation as terminal_validation_records
from ...deps import StoreDeps


_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AuxiliaryTerminalSealFailureCode(StrEnum):
    """终态密封被拒绝时的稳定关闭式失败原因。"""

    APPLY_ID_COLLISION = "apply_id_collision"
    AUTHORITY_NOT_CURRENT = "authority_not_current"
    STATE_VERSION_CONFLICT = "state_version_conflict"
    BASE_TASK_GRAPH_DRIFT = "base_task_graph_drift"
    TERMINAL_NODE_INVALID = "terminal_node_invalid"
    REQUIRED_COMPLETION_MISSING = "required_completion_missing"
    TERMINAL_COMPLETION_INVALID = "terminal_completion_invalid"
    OUTPUT_WINDOW_TAMPERED = "output_window_tampered"
    PROPOSAL_INVALID = "proposal_invalid"
    BUDGET_HARD_LIMIT = "budget_hard_limit"
    BLOCKING_GAP = "blocking_gap"
    SEMANTIC_SETTLEMENT_MISSING = "semantic_settlement_missing"
    SEMANTIC_SETTLEMENT_NOT_PASS = "semantic_settlement_not_pass"
    SEMANTIC_BINDING_MISMATCH = "semantic_binding_mismatch"
    PRODUCTION_EVALUATION_FAILED = "production_evaluation_failed"
    STORED_AUTHORITY_CORRUPT = "stored_authority_corrupt"


class AuxiliaryTerminalSealPersistenceError(RuntimeError):
    """终态密封不变量失败，且未修改权威数据。"""

    def __init__(
        self,
        code: AuxiliaryTerminalSealFailureCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(message)


class AuxiliaryTerminalSealIdentityCollision(
    AuxiliaryTerminalSealPersistenceError
):
    """不可变密封或应用身份被复用于另一载荷。"""


class SealAuxiliaryTerminalProposalCommand(_Record):
    """密封一个当前终态完成项所需的精确 Host 权威。"""

    schema_version: Literal["seal-auxiliary-v2-terminal-proposal-command-v1"] = (
        "seal-auxiliary-v2-terminal-proposal-command-v1"
    )
    apply_id: str = Field(pattern=_ID_PATTERN)
    finish_gate_receipt_id: str = Field(pattern=_ID_PATTERN)
    terminal_proposal_receipt_id: str = Field(pattern=_ID_PATTERN)
    session_id: str = Field(pattern=_ID_PATTERN)
    invocation_turn_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_id: str = Field(pattern=_ID_PATTERN)
    goal_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_revision: int = Field(ge=1)
    terminal_auxiliary_node_id: str = Field(pattern=_ID_PATTERN)
    terminal_node_revision: int = Field(ge=1)
    terminal_completion_id: str = Field(pattern=_ID_PATTERN)
    semantic_settlement_id: str = Field(pattern=_ID_PATTERN)
    semantic_prompt_payload_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_base_task_graph_revision: int | None = Field(default=None, ge=1)
    expected_task_state_version: int = Field(ge=1)
    expected_control_state_version: int = Field(ge=1)
    expected_goal_state_version: int = Field(ge=1)
    expected_revision_state_version: int = Field(ge=1)
    expected_budget_state_version: int = Field(ge=1)
    expected_structure_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_budget_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)


class AuxiliaryFinishGateReceipt(_Record):
    schema_version: Literal["auxiliary-v2-finish-gate-receipt-v1"] = (
        "auxiliary-v2-finish-gate-receipt-v1"
    )
    finish_gate_receipt_id: str = Field(pattern=_ID_PATTERN)
    apply_id: str = Field(pattern=_ID_PATTERN)
    session_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_id: str = Field(pattern=_ID_PATTERN)
    goal_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_revision: int = Field(ge=1)
    structure_sha256: str = Field(pattern=_SHA256_PATTERN)
    base_task_graph_revision: int | None = Field(default=None, ge=1)
    target_task_graph_revision: int = Field(ge=1)
    terminal_auxiliary_node_id: str = Field(pattern=_ID_PATTERN)
    terminal_node_revision: int = Field(ge=1)
    terminal_completion_id: str = Field(pattern=_ID_PATTERN)
    terminal_work_run_id: str = Field(pattern=_ID_PATTERN)
    terminal_output_revision: int = Field(ge=1)
    required_completion_ids: tuple[str, ...]
    proposal_sha256: str = Field(pattern=_SHA256_PATTERN)
    validation_context: InSessionTaskGraphRevisionValidationContext
    validation_context_sha256: str = Field(pattern=_SHA256_PATTERN)
    production_evaluation_sha256: str = Field(pattern=_SHA256_PATTERN)
    semantic_settlement_id: str = Field(pattern=_ID_PATTERN)
    semantic_prompt_payload_sha256: str = Field(pattern=_SHA256_PATTERN)
    semantic_review_policy_sha256: str = Field(pattern=_SHA256_PATTERN)
    semantic_settlement_sha256: str = Field(pattern=_SHA256_PATTERN)
    budget_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    budget_disposition: Literal["within_limit", "soft_limit_reached"]
    readiness_status: Literal["proposal_ready", "gapped_ready"]
    non_blocking_gap_ids: tuple[str, ...]
    created_turn_id: str = Field(pattern=_ID_PATTERN)

    @model_validator(mode="after")
    def _require_validation_context_hash(self) -> "AuxiliaryFinishGateReceipt":
        expected = _sha256_value(self.validation_context.model_dump(mode="json"))
        if self.validation_context_sha256 != expected:
            raise ValueError("validation context hash does not match its body")
        return self


class AuxiliaryTerminalProposalReceipt(_Record):
    schema_version: Literal["auxiliary-v2-terminal-proposal-receipt-v1"] = (
        "auxiliary-v2-terminal-proposal-receipt-v1"
    )
    terminal_proposal_receipt_id: str = Field(pattern=_ID_PATTERN)
    finish_gate_receipt_id: str = Field(pattern=_ID_PATTERN)
    session_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_id: str = Field(pattern=_ID_PATTERN)
    goal_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_revision: int = Field(ge=1)
    terminal_completion_id: str = Field(pattern=_ID_PATTERN)
    proposal: InSessionTaskGraphRevisionProposal
    lineage: tuple[TaskGraphSemanticLineageProjection, ...] = ()
    proposal_sha256: str = Field(pattern=_SHA256_PATTERN)
    output_window: OutputWindow
    output_window_sha256: str = Field(pattern=_SHA256_PATTERN)
    created_turn_id: str = Field(pattern=_ID_PATTERN)

    @model_validator(mode="after")
    def _require_output_material_binding(
        self,
    ) -> "AuxiliaryTerminalProposalReceipt":
        if self.proposal_sha256 != _sha256_value(
            self.proposal.model_dump(mode="json")
        ):
            raise ValueError("proposal hash does not match its body")
        material: BaseModel
        if self.lineage:
            material = TaskGraphRevisionCandidate(
                proposal=self.proposal,
                lineage=self.lineage,
            )
        else:
            material = self.proposal
        if self.output_window.content != material.model_dump_json():
            raise ValueError(
                "terminal OutputWindow does not contain its exact typed material"
            )
        if self.output_window_sha256 != _sha256_value(
            self.output_window.model_dump(mode="json")
        ):
            raise ValueError("OutputWindow hash does not match its body")
        return self


class AuxiliaryTerminalSealResult(_Record):
    schema_version: Literal["auxiliary-v2-terminal-seal-result-v1"] = (
        "auxiliary-v2-terminal-seal-result-v1"
    )
    status: Literal["applied", "replayed"]
    apply_id: str = Field(pattern=_ID_PATTERN)
    finish_gate_receipt_id: str = Field(pattern=_ID_PATTERN)
    terminal_proposal_receipt_id: str = Field(pattern=_ID_PATTERN)
    terminal_completion_id: str = Field(pattern=_ID_PATTERN)
    proposal_sha256: str = Field(pattern=_SHA256_PATTERN)
    production_evaluation_sha256: str = Field(pattern=_SHA256_PATTERN)
    semantic_settlement_id: str = Field(pattern=_ID_PATTERN)
    semantic_settlement_sha256: str = Field(pattern=_SHA256_PATTERN)
    readiness_status: Literal["proposal_ready", "gapped_ready"]
    task_state_version: int = Field(ge=1)
    control_state_version: int = Field(ge=1)
    goal_state_version: int = Field(ge=1)
    revision_state_version: int = Field(ge=1)
    budget_state_version: int = Field(ge=1)


def seal_auxiliary_terminal_proposal(
    deps: StoreDeps,
    *,
    command: SealAuxiliaryTerminalProposalCommand,
) -> AuxiliaryTerminalSealResult:
    """在单一 SQLite 事务中密封一个精确 终态提案。"""

    if not isinstance(command, SealAuxiliaryTerminalProposalCommand):
        raise TypeError(
            "command must be a SealAuxiliaryTerminalProposalCommand"
        )
    payload_sha256 = _sha256_value(command.model_dump(mode="json"))
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        replay = conn.execute(
            "SELECT * FROM "
            "insession_auxiliary_v2_terminal_seal_apply_receipts "
            "WHERE apply_id=?",
            (command.apply_id,),
        ).fetchone()
        if replay is not None:
            result = _load_replay(
                conn,
                command=command,
                payload_sha256=payload_sha256,
                row=replay,
            )
            conn.commit()
            return result

        duplicate = conn.execute(
            "SELECT finish_gate_receipt_id FROM "
            "insession_auxiliary_v2_finish_gate_receipts "
            "WHERE terminal_completion_id=?",
            (command.terminal_completion_id,),
        ).fetchone()
        if duplicate is not None:
            _fail_collision(
                "terminal completion already belongs to another immutable seal"
            )

        task = _load_current_task(conn, command)
        details = auxiliary_graph_records._load_auxiliary_graph(
            conn,
            command.session_id,
            command.task_id,
        )
        _require_current_authority(command, task=task, details=details)
        if details.budget is None or details.revision is None:
            _fail(
                AuxiliaryTerminalSealFailureCode.STORED_AUTHORITY_CORRUPT,
                "terminal seal requires a formal revision and budget",
            )
        if (
            details.budget.assessment.disposition
            is PlanningEpisodeBudgetDisposition.HARD_LIMIT_REACHED
        ):
            _fail(
                AuxiliaryTerminalSealFailureCode.BUDGET_HARD_LIMIT,
                "terminal seal is forbidden after a hard planning-budget limit",
            )

        membership_rows = conn.execute(
            "SELECT auxiliary_node_id, node_revision, ordinal, required, "
            "carried_completion_id FROM "
            "insession_auxiliary_graph_revision_nodes_v2 "
            "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=? "
            "ORDER BY ordinal",
            (command.auxiliary_graph_id, command.auxiliary_graph_revision),
        ).fetchall()
        membership = {
            str(row["auxiliary_node_id"]): row for row in membership_rows
        }
        completion_ids = (
            auxiliary_graph_records._load_auxiliary_frontier_completion_ids(
                conn,
                details=details,
                membership_by_node=membership,
            )
        )
        terminal_node, required_completion_node_ids = (
            _require_terminal_and_predecessors(
                command,
                details=details,
                completion_ids=completion_ids,
            )
        )
        output, proposal, lineage, output_sha256 = _load_terminal_output(
            conn,
            command=command,
            details=details,
            terminal_node=terminal_node,
            completion_ids=completion_ids,
        )
        proposal_sha256 = _sha256_value(proposal.model_dump(mode="json"))

        _require_no_persisted_semantic_blocker(
            conn,
            command=command,
            details=details,
            proposal=proposal,
            lineage=lineage,
            proposal_sha256=proposal_sha256,
        )

        try:
            settlement = (
                semantic_records._load_auxiliary_semantic_quorum_settlement(
                    conn,
                    session_id=command.session_id,
                    task_id=command.task_id,
                    auxiliary_graph_id=command.auxiliary_graph_id,
                    goal_id=command.goal_id,
                    auxiliary_graph_revision=command.auxiliary_graph_revision,
                    frozen_prompt_payload_sha256=(
                        command.semantic_prompt_payload_sha256
                    ),
                )
            )
        except semantic_records.AuxiliarySemanticVerificationPersistenceError as exc:
            _fail_corrupt("semantic quorum authority is corrupt", exc)
        if settlement is None:
            _fail(
                AuxiliaryTerminalSealFailureCode.SEMANTIC_SETTLEMENT_MISSING,
                "no exact semantic quorum settlement exists for the proposal",
            )
        _require_semantic_binding(
            command,
            details=details,
            settlement=settlement,
            proposal=proposal,
            lineage=lineage,
            proposal_sha256=proposal_sha256,
        )
        reference_request = settlement.requests[0]
        if reference_request.blocking_gap_aliases:
            _fail(
                AuxiliaryTerminalSealFailureCode.BLOCKING_GAP,
                "a terminal proposal with blocking gaps cannot be sealed",
            )

        validation_context = _derive_validation_context(
            conn,
            session_id=command.session_id,
            invocation_turn_id=command.invocation_turn_id,
            task_id=command.task_id,
            task=task,
            request=reference_request,
            details=details,
            allowed_graph_statuses=frozenset({"active"}),
        )
        validation = validate_insession_task_graph_revision(
            proposal,
            context=validation_context,
        )
        if validation.status != "accepted":
            _fail(
                AuxiliaryTerminalSealFailureCode.PROPOSAL_INVALID,
                "terminal proposal failed canonical Store validation: "
                + ",".join(code.value for code in validation.error_codes),
            )
        production_context = _derive_production_context(
            request=reference_request,
            validation_context=validation_context,
            task_current_graph_revision=(
                int(task["current_graph_revision"])
                if task["current_graph_revision"] is not None
                else None
            ),
        )
        evaluation = evaluate_task_graph_production(
            proposal,
            context=production_context,
        )
        if not evaluation.passed:
            _fail(
                AuxiliaryTerminalSealFailureCode.PRODUCTION_EVALUATION_FAILED,
                "terminal proposal failed Store production evaluation: "
                + ",".join(code.value for code in evaluation.failure_codes),
            )

        non_blocking_gap_ids = reference_request.non_blocking_gap_aliases
        readiness_status: Literal["proposal_ready", "gapped_ready"] = (
            "gapped_ready" if non_blocking_gap_ids else "proposal_ready"
        )
        now = deps.now()
        next_goal_version = details.goal_state_version + 1
        next_revision_version = details.revision_state_version + 1
        result = AuxiliaryTerminalSealResult(
            status="applied",
            apply_id=command.apply_id,
            finish_gate_receipt_id=command.finish_gate_receipt_id,
            terminal_proposal_receipt_id=(
                command.terminal_proposal_receipt_id
            ),
            terminal_completion_id=command.terminal_completion_id,
            proposal_sha256=proposal_sha256,
            production_evaluation_sha256=evaluation.evaluation_sha256,
            semantic_settlement_id=settlement.settlement_id,
            semantic_settlement_sha256=settlement.settlement_sha256,
            readiness_status=readiness_status,
            task_state_version=int(task["state_version"]),
            control_state_version=details.control_state_version,
            goal_state_version=next_goal_version,
            revision_state_version=next_revision_version,
            budget_state_version=details.budget_state_version,
        )
        finish_receipt = AuxiliaryFinishGateReceipt(
            finish_gate_receipt_id=command.finish_gate_receipt_id,
            apply_id=command.apply_id,
            session_id=command.session_id,
            task_id=command.task_id,
            auxiliary_graph_id=command.auxiliary_graph_id,
            goal_id=command.goal_id,
            auxiliary_graph_revision=command.auxiliary_graph_revision,
            structure_sha256=details.structure_sha256,
            base_task_graph_revision=details.base_task_graph_revision,
            target_task_graph_revision=details.target_task_graph_revision,
            terminal_auxiliary_node_id=command.terminal_auxiliary_node_id,
            terminal_node_revision=command.terminal_node_revision,
            terminal_completion_id=command.terminal_completion_id,
            terminal_work_run_id=output.work_run_id,
            terminal_output_revision=output.output_revision,
            required_completion_ids=tuple(
                completion_ids[node_id]
                for node_id in required_completion_node_ids
                if node_id != command.terminal_auxiliary_node_id
            ),
            proposal_sha256=proposal_sha256,
            validation_context=validation_context,
            validation_context_sha256=_sha256_value(
                validation_context.model_dump(mode="json")
            ),
            production_evaluation_sha256=evaluation.evaluation_sha256,
            semantic_settlement_id=settlement.settlement_id,
            semantic_prompt_payload_sha256=(
                settlement.frozen_prompt_payload_sha256
            ),
            semantic_review_policy_sha256=(
                settlement.review_policy.policy_sha256
            ),
            semantic_settlement_sha256=settlement.settlement_sha256,
            budget_snapshot_sha256=details.budget.snapshot_sha256,
            budget_disposition=details.budget.assessment.disposition.value,
            readiness_status=readiness_status,
            non_blocking_gap_ids=non_blocking_gap_ids,
            created_turn_id=command.invocation_turn_id,
        )
        terminal_receipt = AuxiliaryTerminalProposalReceipt(
            terminal_proposal_receipt_id=(
                command.terminal_proposal_receipt_id
            ),
            finish_gate_receipt_id=command.finish_gate_receipt_id,
            session_id=command.session_id,
            task_id=command.task_id,
            auxiliary_graph_id=command.auxiliary_graph_id,
            goal_id=command.goal_id,
            auxiliary_graph_revision=command.auxiliary_graph_revision,
            terminal_completion_id=command.terminal_completion_id,
            proposal=proposal,
            lineage=lineage,
            proposal_sha256=proposal_sha256,
            output_window=output,
            output_window_sha256=output_sha256,
            created_turn_id=command.invocation_turn_id,
        )
        _insert_seal_rows(
            conn,
            command=command,
            payload_sha256=payload_sha256,
            result=result,
            finish_receipt=finish_receipt,
            terminal_receipt=terminal_receipt,
            evaluation=evaluation,
            now=now,
        )
        if conn.execute(
            "UPDATE insession_auxiliary_graph_revision_states_v2 "
            "SET status=?, state_version=state_version+1, updated_at=? "
            "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=? "
            "AND status='active' AND state_version=?",
            (
                readiness_status,
                now,
                command.auxiliary_graph_id,
                command.auxiliary_graph_revision,
                command.expected_revision_state_version,
            ),
        ).rowcount != 1:
            _fail(
                AuxiliaryTerminalSealFailureCode.STATE_VERSION_CONFLICT,
                "AuxiliaryGraph revision changed during terminal seal",
            )
        if conn.execute(
            "UPDATE insession_auxiliary_graph_goals "
            "SET status=?, state_version=state_version+1, updated_at=? "
            "WHERE session_id=? AND insession_task_id=? "
            "AND auxiliary_graph_id=? AND goal_id=? "
            "AND status='active' AND state_version=?",
            (
                readiness_status,
                now,
                command.session_id,
                command.task_id,
                command.auxiliary_graph_id,
                command.goal_id,
                command.expected_goal_state_version,
            ),
        ).rowcount != 1:
            _fail(
                AuxiliaryTerminalSealFailureCode.STATE_VERSION_CONFLICT,
                "AuxiliaryGraph goal changed during terminal seal",
            )
        conn.commit()
        return result
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def get_auxiliary_terminal_proposal_receipt(
    deps: StoreDeps,
    *,
    session_id: str,
    terminal_proposal_receipt_id: str,
) -> AuxiliaryTerminalProposalReceipt:
    """加载并哈希校验一个不可变 提案回执。"""

    deps.init_db()
    with deps.connect() as conn:
        row = conn.execute(
            "SELECT proposal_json, proposal_sha256, output_window_json, "
            "output_window_sha256, finish_gate_receipt_id, session_id, "
            "insession_task_id, auxiliary_graph_id, goal_id, "
            "auxiliary_graph_revision, terminal_completion_id, "
            "created_turn_id FROM "
            "insession_auxiliary_v2_terminal_proposal_receipts "
            "WHERE terminal_proposal_receipt_id=? AND session_id=?",
            (terminal_proposal_receipt_id, session_id),
        ).fetchone()
        if row is None:
            _fail(
                AuxiliaryTerminalSealFailureCode.STORED_AUTHORITY_CORRUPT,
                "unknown terminal proposal receipt",
            )
        finish_row = conn.execute(
            "SELECT receipt_json, receipt_sha256 FROM "
            "insession_auxiliary_v2_finish_gate_receipts "
            "WHERE finish_gate_receipt_id=?",
            (str(row["finish_gate_receipt_id"]),),
        ).fetchone()
        if finish_row is None:
            _fail(
                AuxiliaryTerminalSealFailureCode.STORED_AUTHORITY_CORRUPT,
                "terminal proposal lost its finish-gate receipt",
            )
        try:
            proposal = InSessionTaskGraphRevisionProposal.model_validate_json(
                str(row["proposal_json"])
            )
            output = OutputWindow.model_validate_json(
                str(row["output_window_json"])
            )
            finish = AuxiliaryFinishGateReceipt.model_validate_json(
                str(finish_row["receipt_json"])
            )
            output_proposal, lineage = _parse_terminal_output_material(
                output.content,
                expected_base_task_graph_revision=(
                    finish.base_task_graph_revision
                ),
            )
            receipt = AuxiliaryTerminalProposalReceipt(
                terminal_proposal_receipt_id=terminal_proposal_receipt_id,
                finish_gate_receipt_id=str(row["finish_gate_receipt_id"]),
                session_id=str(row["session_id"]),
                task_id=str(row["insession_task_id"]),
                auxiliary_graph_id=str(row["auxiliary_graph_id"]),
                goal_id=str(row["goal_id"]),
                auxiliary_graph_revision=int(row["auxiliary_graph_revision"]),
                terminal_completion_id=str(row["terminal_completion_id"]),
                proposal=proposal,
                lineage=lineage,
                proposal_sha256=str(row["proposal_sha256"]),
                output_window=output,
                output_window_sha256=str(row["output_window_sha256"]),
                created_turn_id=str(row["created_turn_id"]),
            )
        except (TypeError, ValueError, ValidationError) as exc:
            _fail_corrupt("terminal proposal receipt is not typed", exc)
        source_output = conn.execute(
            "SELECT snapshot_json, snapshot_hash, frozen_at FROM "
            "insession_work_run_output_windows WHERE work_run_id=? "
            "AND output_revision=?",
            (output.work_run_id, output.output_revision),
        ).fetchone()
        try:
            settlement = (
                semantic_records._load_auxiliary_semantic_quorum_settlement(
                    conn,
                    session_id=receipt.session_id,
                    task_id=receipt.task_id,
                    auxiliary_graph_id=receipt.auxiliary_graph_id,
                    goal_id=receipt.goal_id,
                    auxiliary_graph_revision=receipt.auxiliary_graph_revision,
                    frozen_prompt_payload_sha256=(
                        finish.semantic_prompt_payload_sha256
                    ),
                )
            )
        except semantic_records.AuxiliarySemanticVerificationPersistenceError as exc:
            _fail_corrupt("terminal semantic settlement is corrupt", exc)
        reference_request = (
            settlement.requests[0]
            if settlement is not None and settlement.requests
            else None
        )
        if (
            _model_json(proposal) != str(row["proposal_json"])
            or _model_json(output) != str(row["output_window_json"])
            or _sha256_text(str(row["output_window_json"]))
            != receipt.output_window_sha256
            or _model_json(finish) != str(finish_row["receipt_json"])
            or _sha256_text(str(finish_row["receipt_json"]))
            != str(finish_row["receipt_sha256"])
            or finish.finish_gate_receipt_id != receipt.finish_gate_receipt_id
            or finish.session_id != receipt.session_id
            or finish.task_id != receipt.task_id
            or finish.auxiliary_graph_id != receipt.auxiliary_graph_id
            or finish.goal_id != receipt.goal_id
            or finish.auxiliary_graph_revision
            != receipt.auxiliary_graph_revision
            or finish.terminal_completion_id != receipt.terminal_completion_id
            or finish.proposal_sha256 != receipt.proposal_sha256
            or output_proposal != proposal
            or source_output is None
            or str(source_output["snapshot_json"])
            != str(row["output_window_json"])
            or str(source_output["snapshot_hash"])
            != receipt.output_window_sha256
            or source_output["frozen_at"] is None
            or settlement is None
            or settlement.host_disposition
            is not TaskGraphSemanticVerificationDisposition.PASS
            or settlement.settlement_id != finish.semantic_settlement_id
            or settlement.settlement_sha256
            != finish.semantic_settlement_sha256
            or reference_request is None
            or reference_request.task_graph_proposal != proposal
            or reference_request.task_graph_proposal_sha256
            != receipt.proposal_sha256
            or not _semantic_request_matches_terminal_material(
                reference_request,
                proposal=proposal,
                lineage=lineage,
                expected_base_task_graph_revision=(
                    finish.base_task_graph_revision
                ),
            )
        ):
            _fail(
                AuxiliaryTerminalSealFailureCode.STORED_AUTHORITY_CORRUPT,
                "terminal proposal receipt hash binding is corrupt",
            )
        return receipt


def _load_current_task(
    conn: sqlite3.Connection,
    command: SealAuxiliaryTerminalProposalCommand,
) -> sqlite3.Row:
    if conn.execute(
        "SELECT 1 FROM insession_task_turn_links WHERE session_id=? "
        "AND turn_id=? AND insession_task_id=?",
        (command.session_id, command.invocation_turn_id, command.task_id),
    ).fetchone() is None:
        _fail(
            AuxiliaryTerminalSealFailureCode.AUTHORITY_NOT_CURRENT,
            "terminal seal invocation Turn is not linked to the Task",
        )
    row = conn.execute(
        "SELECT session_id, current_graph_revision, current_status, "
        "state_version, created_turn_id, creation_source_start, "
        "creation_source_end, creation_source_sha256 FROM insession_tasks "
        "WHERE session_id=? "
        "AND insession_task_id=?",
        (command.session_id, command.task_id),
    ).fetchone()
    if row is None:
        _fail(
            AuxiliaryTerminalSealFailureCode.AUTHORITY_NOT_CURRENT,
            "terminal seal targets an unknown Task",
        )
    return row


def _require_current_authority(command, *, task, details) -> None:
    exact_ids = (
        details.session_id == command.session_id
        and details.task_id == command.task_id
        and details.auxiliary_graph_id == command.auxiliary_graph_id
        and details.goal_id == command.goal_id
        and details.auxiliary_graph_revision
        == command.auxiliary_graph_revision
        and details.terminal_auxiliary_node_id
        == command.terminal_auxiliary_node_id
    )
    if not exact_ids or details.goal_status != "active" or details.revision_status != "active":
        _fail(
            AuxiliaryTerminalSealFailureCode.AUTHORITY_NOT_CURRENT,
            "terminal seal does not target the current active revision",
        )
    if str(task["current_status"]) != "active":
        _fail(
            AuxiliaryTerminalSealFailureCode.AUTHORITY_NOT_CURRENT,
            "terminal seal owner Task is not active",
        )
    current_base = (
        int(task["current_graph_revision"])
        if task["current_graph_revision"] is not None
        else None
    )
    if (
        current_base != command.expected_base_task_graph_revision
        or current_base != details.base_task_graph_revision
    ):
        _fail(
            AuxiliaryTerminalSealFailureCode.BASE_TASK_GRAPH_DRIFT,
            "TaskGraph base changed after the planning goal was frozen",
        )
    versions = (
        int(task["state_version"]) == command.expected_task_state_version,
        details.control_state_version == command.expected_control_state_version,
        details.goal_state_version == command.expected_goal_state_version,
        details.revision_state_version
        == command.expected_revision_state_version,
        details.budget_state_version == command.expected_budget_state_version,
        details.structure_sha256 == command.expected_structure_sha256,
        details.budget is not None
        and details.budget.snapshot_sha256
        == command.expected_budget_snapshot_sha256,
    )
    if not all(versions):
        _fail(
            AuxiliaryTerminalSealFailureCode.STATE_VERSION_CONFLICT,
            "terminal seal expected state versions are stale",
        )


def _require_terminal_and_predecessors(command, *, details, completion_ids):
    terminals = tuple(
        node
        for node in details.nodes
        if node.executor_kind == AuxiliaryNodeExecutorKind.TERMINAL_PLANNER.value
    )
    if (
        len(terminals) != 1
        or terminals[0].auxiliary_node_id
        != command.terminal_auxiliary_node_id
        or terminals[0].node_revision != command.terminal_node_revision
        or terminals[0].output_contract != "task_graph_revision_proposal_v2"
        or not terminals[0].required
    ):
        _fail(
            AuxiliaryTerminalSealFailureCode.TERMINAL_NODE_INVALID,
            "current revision has no unique exact terminal planner",
        )
    dependency_closure = {command.terminal_auxiliary_node_id}
    changed = True
    while changed:
        changed = False
        for edge in details.edges:
            if (
                edge.required
                and edge.consumer_auxiliary_node_id in dependency_closure
                and edge.dependency_auxiliary_node_id not in dependency_closure
            ):
                dependency_closure.add(edge.dependency_auxiliary_node_id)
                changed = True
    required_completion_node_ids = tuple(
        node.auxiliary_node_id
        for node in details.nodes
        if node.required or node.auxiliary_node_id in dependency_closure
    )
    missing = tuple(
        node_id
        for node_id in required_completion_node_ids
        if node_id not in completion_ids
    )
    if missing:
        _fail(
            AuxiliaryTerminalSealFailureCode.REQUIRED_COMPLETION_MISSING,
            "required nodes lack verified completion authority: "
            + ",".join(missing),
        )
    return terminals[0], required_completion_node_ids


def _load_terminal_output(
    conn,
    *,
    command,
    details,
    terminal_node,
    completion_ids,
):
    if completion_ids.get(command.terminal_auxiliary_node_id) != command.terminal_completion_id:
        _fail(
            AuxiliaryTerminalSealFailureCode.TERMINAL_COMPLETION_INVALID,
            "terminal completion ID is not the current verified completion",
        )
    row = conn.execute(
        "SELECT completion.work_run_id, completion.output_revision, "
        "completion.verification_request_id, completion.submitted_attempt_id, "
        "completion.completion_json, completion.completion_sha256, "
        "output.snapshot_json, output.snapshot_hash, output.frozen_at, "
        "output.updated_turn_id AS output_updated_turn_id, "
        "output.updated_attempt_id AS output_updated_attempt_id, "
        "run.status AS run_status, run.reason AS run_reason, "
        "run.current_attempt_id, run.current_verification_request_id, "
        "request.status AS request_status, request.all_pass, "
        "request.submitted_attempt_id AS request_submitted_attempt_id, "
        "request.output_revision AS request_output_revision, "
        "submitted_attempt.status AS submitted_attempt_status, "
        "submitted_attempt.committed_output_revision AS "
        "submitted_committed_output_revision, "
        "submitted_attempt.submitted_output_revision AS "
        "submitted_output_revision, "
        "output_author.status AS output_author_status, "
        "output_author.turn_id AS output_author_turn_id, "
        "output_author.input_output_revision AS "
        "output_author_input_output_revision, "
        "output_author.committed_output_revision AS "
        "output_author_committed_output_revision, "
        "node_state.status AS node_status "
        "FROM insession_auxiliary_node_completions_v2 AS completion "
        "JOIN insession_work_runs AS run "
        "ON run.work_run_id=completion.work_run_id "
        "JOIN insession_work_run_output_windows AS output "
        "ON output.work_run_id=completion.work_run_id "
        "AND output.output_revision=completion.output_revision "
        "JOIN insession_work_run_verification_requests AS request "
        "ON request.work_run_id=completion.work_run_id "
        "AND request.verification_request_id=completion.verification_request_id "
        "JOIN insession_work_run_attempts AS submitted_attempt "
        "ON submitted_attempt.work_run_id=completion.work_run_id "
        "AND submitted_attempt.attempt_id=completion.submitted_attempt_id "
        "JOIN insession_work_run_attempts AS output_author "
        "ON output_author.work_run_id=completion.work_run_id "
        "AND output_author.attempt_id=output.updated_attempt_id "
        "JOIN insession_auxiliary_node_states_v2 AS node_state "
        "ON node_state.auxiliary_graph_id=completion.auxiliary_graph_id "
        "AND node_state.auxiliary_graph_revision="
        "completion.auxiliary_graph_revision "
        "AND node_state.auxiliary_node_id=completion.auxiliary_node_id "
        "AND node_state.node_revision=completion.node_revision "
        "WHERE completion.completion_id=? AND completion.session_id=? "
        "AND completion.insession_task_id=? "
        "AND completion.auxiliary_graph_id=? "
        "AND completion.auxiliary_graph_revision=? "
        "AND completion.auxiliary_node_id=? "
        "AND completion.node_revision=?",
        (
            command.terminal_completion_id,
            command.session_id,
            command.task_id,
            command.auxiliary_graph_id,
            command.auxiliary_graph_revision,
            command.terminal_auxiliary_node_id,
            command.terminal_node_revision,
        ),
    ).fetchone()
    if (
        row is None
        or str(row["run_status"]) != "completed"
        or str(row["run_reason"]) != "verification_passed"
        or row["current_attempt_id"] is not None
        or row["current_verification_request_id"] is not None
        or str(row["request_status"]) != "completed"
        or int(row["all_pass"] or 0) != 1
        or str(row["request_submitted_attempt_id"])
        != str(row["submitted_attempt_id"])
        or int(row["request_output_revision"]) != int(row["output_revision"])
        or str(row["submitted_attempt_status"]) != "closed"
        or int(row["submitted_committed_output_revision"] or 0)
        != int(row["output_revision"])
        or int(row["submitted_output_revision"] or 0)
        != int(row["output_revision"])
        or str(row["output_author_status"]) != "closed"
        or str(row["output_author_turn_id"])
        != str(row["output_updated_turn_id"])
        or int(row["output_author_input_output_revision"])
        != int(row["output_revision"]) - 1
        or int(row["output_author_committed_output_revision"] or 0)
        != int(row["output_revision"])
        or str(row["node_status"]) != "completed"
        or row["frozen_at"] is None
        or terminal_node.status != "completed"
    ):
        _fail(
            AuxiliaryTerminalSealFailureCode.TERMINAL_COMPLETION_INVALID,
            "terminal completion lost frozen passed-verification authority",
        )
    snapshot_json = str(row["snapshot_json"])
    if _sha256_text(snapshot_json) != str(row["snapshot_hash"]):
        _fail(
            AuxiliaryTerminalSealFailureCode.OUTPUT_WINDOW_TAMPERED,
            "terminal OutputWindow snapshot hash is corrupt",
        )
    try:
        output = OutputWindow.model_validate_json(snapshot_json)
        proposal, lineage = _parse_terminal_output_material(
            output.content,
            expected_base_task_graph_revision=details.base_task_graph_revision,
        )
    except (TypeError, ValueError, ValidationError) as exc:
        _fail_corrupt(
            "terminal OutputWindow does not contain canonical typed material",
            exc,
        )
    if (
        _model_json(output) != snapshot_json
        or output.work_run_id != str(row["work_run_id"])
        or output.output_revision != int(row["output_revision"])
        or output.updated_turn_id != str(row["output_updated_turn_id"])
        or output.updated_attempt_id != str(row["output_updated_attempt_id"])
    ):
        _fail(
            AuxiliaryTerminalSealFailureCode.OUTPUT_WINDOW_TAMPERED,
            "terminal OutputWindow provenance is corrupt",
        )
    return output, proposal, lineage, str(row["snapshot_hash"])


def _parse_terminal_output_material(
    content: str,
    *,
    expected_base_task_graph_revision: int | None,
) -> tuple[
    InSessionTaskGraphRevisionProposal,
    tuple[TaskGraphSemanticLineageProjection, ...],
]:
    """解析由冻结 base 权威选择的精确终态信封。

    base 为 null 的规划 episode 直接存储规范提案。正 base episode 存储规范 revision
    候选，使谱系成为不可变终态输出，而非之后由调用方编写的提示。
    """

    if expected_base_task_graph_revision is None:
        proposal = InSessionTaskGraphRevisionProposal.model_validate_json(content)
        if content != proposal.model_dump_json():
            raise ValueError("base-null terminal proposal is not canonical")
        return proposal, ()
    candidate = TaskGraphRevisionCandidate.model_validate_json(content)
    if content != candidate.model_dump_json():
        raise ValueError("positive-base revision candidate is not canonical")
    return candidate.proposal, candidate.lineage


def _semantic_request_matches_terminal_material(
    request,
    *,
    proposal: InSessionTaskGraphRevisionProposal,
    lineage: tuple[TaskGraphSemanticLineageProjection, ...],
    expected_base_task_graph_revision: int | None,
) -> bool:
    if (
        request.task_graph_proposal != proposal
        or request.prompt_payload.lineage != lineage
    ):
        return False
    base = request.prompt_payload.base_task_graph
    if expected_base_task_graph_revision is None:
        return base is None and not lineage
    return (
        base is not None
        and base.base_task_graph_revision
        == expected_base_task_graph_revision
        and bool(lineage)
    )


def _require_semantic_binding(
    command,
    *,
    details,
    settlement,
    proposal,
    lineage,
    proposal_sha256,
) -> None:
    if (
        settlement.settlement_id != command.semantic_settlement_id
        or settlement.session_id != command.session_id
        or settlement.task_id != command.task_id
        or settlement.auxiliary_graph_id != command.auxiliary_graph_id
        or settlement.goal_id != command.goal_id
        or settlement.auxiliary_graph_revision
        != command.auxiliary_graph_revision
        or settlement.frozen_prompt_payload_sha256
        != command.semantic_prompt_payload_sha256
        or settlement.host_disposition
        is not TaskGraphSemanticVerificationDisposition.PASS
    ):
        code = (
            AuxiliaryTerminalSealFailureCode.SEMANTIC_SETTLEMENT_NOT_PASS
            if settlement.host_disposition
            is not TaskGraphSemanticVerificationDisposition.PASS
            else AuxiliaryTerminalSealFailureCode.SEMANTIC_BINDING_MISMATCH
        )
        _fail(code, "semantic quorum does not grant this terminal seal")
    if not settlement.requests or len(settlement.requests) != len(settlement.results):
        _fail(
            AuxiliaryTerminalSealFailureCode.SEMANTIC_BINDING_MISMATCH,
            "semantic quorum has no exact request/result pairs",
        )
    reference = settlement.requests[0]
    _require_semantic_request_binding(
        command,
        details=details,
        request=reference,
        proposal=proposal,
        lineage=lineage,
        proposal_sha256=proposal_sha256,
        expected_review_policy=settlement.review_policy,
    )


def _require_no_persisted_semantic_blocker(
    conn,
    *,
    command,
    details,
    proposal,
    lineage,
    proposal_sha256,
) -> None:
    """在要求通过的法定人数回执前拒绝已知阻碍。

    在此认证精确持久请求和结果权威，使已知阻塞缺口或未通过审查结果与尚未产生法定人数
    结算的调用保持可区分。
    """

    rows = conn.execute(
        "SELECT verification_request_id FROM "
        "insession_auxiliary_semantic_verification_requests "
        "WHERE session_id=? AND insession_task_id=? "
        "AND auxiliary_graph_id=? AND goal_id=? "
        "AND auxiliary_graph_revision=? AND prompt_payload_sha256=? "
        "ORDER BY reviewer_ordinal, verification_request_id",
        (
            command.session_id,
            command.task_id,
            command.auxiliary_graph_id,
            command.goal_id,
            command.auxiliary_graph_revision,
            command.semantic_prompt_payload_sha256,
        ),
    ).fetchall()
    for row in rows:
        request_id = str(row["verification_request_id"])
        try:
            request = semantic_records._load_request(
                conn,
                request_id,
            ).request
        except semantic_records.AuxiliarySemanticVerificationPersistenceError as exc:
            _fail_corrupt("semantic verification request is corrupt", exc)
        _require_semantic_request_binding(
            command,
            details=details,
            request=request,
            proposal=proposal,
            lineage=lineage,
            proposal_sha256=proposal_sha256,
        )
        if request.blocking_gap_aliases:
            _fail(
                AuxiliaryTerminalSealFailureCode.BLOCKING_GAP,
                "a terminal proposal with blocking gaps cannot be sealed",
            )
        result_row = conn.execute(
            "SELECT verification_result_id FROM "
            "insession_auxiliary_semantic_verification_results "
            "WHERE verification_request_id=?",
            (request_id,),
        ).fetchone()
        if result_row is None:
            continue
        try:
            result_record = semantic_records._load_result(
                conn,
                str(result_row["verification_result_id"]),
            )
        except semantic_records.AuxiliarySemanticVerificationPersistenceError as exc:
            _fail_corrupt("semantic verification result is corrupt", exc)
        if (
            result_record.host_disposition
            is not TaskGraphSemanticVerificationDisposition.PASS
        ):
            _fail(
                AuxiliaryTerminalSealFailureCode.SEMANTIC_SETTLEMENT_NOT_PASS,
                "semantic verification rejected the terminal proposal",
            )


def _require_semantic_request_binding(
    command,
    *,
    details,
    request,
    proposal,
    lineage,
    proposal_sha256,
    expected_review_policy=None,
) -> None:
    if (
        not _semantic_request_matches_terminal_material(
            request,
            proposal=proposal,
            lineage=lineage,
            expected_base_task_graph_revision=(
                details.base_task_graph_revision
            ),
        )
        or request.task_graph_proposal_sha256 != proposal_sha256
        or request.auxiliary_graph_structure_sha256
        != details.structure_sha256
        or request.goal != details.goal
        or request.budget != details.budget
        or request.authority_projection.authority_snapshot_id
        != details.authority_snapshot_id
        or request.authority_projection.authority_snapshot_sha256
        != details.authority_snapshot_sha256
        or request.prompt_payload.payload_sha256
        != command.semantic_prompt_payload_sha256
        or (
            expected_review_policy is not None
            and request.review_policy != expected_review_policy
        )
    ):
        _fail(
            AuxiliaryTerminalSealFailureCode.SEMANTIC_BINDING_MISMATCH,
            "semantic quorum proposal/prompt/policy authority is stale",
        )


def _derive_validation_context(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    invocation_turn_id: str,
    task_id: str,
    task: sqlite3.Row,
    request,
    details,
    allowed_graph_statuses: frozenset[str],
    require_current_task_graph_revision: bool = True,
) -> InSessionTaskGraphRevisionValidationContext:
    try:
        context = (
            terminal_validation_records._build_auxiliary_terminal_validation_context(
                conn,
                session_id=session_id,
                invocation_turn_id=invocation_turn_id,
                task_id=task_id,
                task_authority=task,
                allowed_graph_statuses=allowed_graph_statuses,
                expected_authority_cards=request.authority_projection.cards,
                require_current_task_graph_revision=(
                    require_current_task_graph_revision
                ),
            )
        )
    except terminal_validation_records.AuxiliaryTerminalSemanticAuthorityMismatch:
        _fail(
            AuxiliaryTerminalSealFailureCode.SEMANTIC_BINDING_MISMATCH,
            "semantic source cards differ from canonical Store authority",
        )
    except terminal_validation_records.AuxiliaryTerminalValidationContextError as exc:
        _fail_corrupt(
            "canonical terminal TaskGraph source authority is corrupt",
            exc,
        )
    profile = details.budget.effective_profile
    bounded = context.model_copy(
        update={
            "limits": InSessionTaskGraphLimits(
                max_root_tasks=1,
                max_nodes_per_task=min(512, profile.hard_current_graph_nodes),
                max_depth=min(64, profile.hard_current_graph_depth),
            )
        }
    )
    return bind_used_evidence_to_required_acceptance_coverage(
        request.task_graph_proposal,
        context=bounded,
    )


def _derive_production_context(
    *,
    request,
    validation_context,
    task_current_graph_revision,
) -> TaskGraphProductionEvaluationContext:
    proposal = request.task_graph_proposal
    capabilities = request.prompt_payload.capabilities.capabilities
    available_ids = tuple(
        sorted(item.capability_alias for item in capabilities if item.available)
    )
    gaps: list[TaskGraphPlanningGap] = []
    for artifact in request.prompt_payload.context_artifacts:
        for gap in artifact.gaps:
            mapped_nodes = tuple(
                node.node_key
                for node in proposal.root.nodes
                if gap.gap_alias in node.source_anchor_ids
            )
            mapped_acceptances = tuple(
                TaskGraphAcceptanceRef(
                    node_key=node.node_key,
                    acceptance_id=acceptance.acceptance_id,
                )
                for node in proposal.root.nodes
                for acceptance in node.acceptance_criteria
                if gap.gap_alias in acceptance.source_anchor_ids
            )
            gaps.append(
                TaskGraphPlanningGap(
                    gap_id=gap.gap_alias,
                    blocking=gap.blocking,
                    affected_required_anchor_ids=(gap.gap_alias,),
                    mapped_node_keys=mapped_nodes,
                    mapped_acceptances=mapped_acceptances,
                )
            )
    authority_hash = request.authority_projection.projection_sha256
    catalog_hash = (
        request.prompt_payload.capabilities.capability_catalog_snapshot_sha256
    )
    return TaskGraphProductionEvaluationContext(
        validation_context=validation_context,
        freshness=TaskGraphProductionFreshness(
            observed_current_graph_revision=task_current_graph_revision,
            expected_authority_snapshot_sha256=(
                request.authority_projection.authority_snapshot_sha256
            ),
            observed_authority_snapshot_sha256=(
                request.authority_projection.authority_snapshot_sha256
            ),
            expected_source_manifest_sha256=authority_hash,
            observed_source_manifest_sha256=authority_hash,
            expected_capability_catalog_sha256=catalog_hash,
            observed_capability_catalog_sha256=catalog_hash,
        ),
        capabilities=TaskGraphProductionCapabilityContext(
            available_capability_ids=available_ids,
            satisfiable_effect_ids=available_ids,
            node_requirements=tuple(
                TaskGraphNodeExecutionRequirement(node_key=node.node_key)
                for node in proposal.root.nodes
            ),
        ),
        gaps=tuple(gaps),
    )


def _insert_seal_rows(
    conn,
    *,
    command,
    payload_sha256,
    result,
    finish_receipt,
    terminal_receipt,
    evaluation,
    now,
) -> None:
    result_json = _model_json(result)
    conn.execute(
        "INSERT INTO insession_auxiliary_v2_terminal_seal_apply_receipts "
        "(apply_id, operation, session_id, insession_task_id, "
        "auxiliary_graph_id, goal_id, auxiliary_graph_revision, "
        "terminal_auxiliary_node_id, terminal_node_revision, "
        "terminal_completion_id, semantic_settlement_id, "
        "invocation_turn_id, payload_sha256, result_json, result_sha256, "
        "created_at) VALUES (?, 'seal_terminal_proposal', ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?, ?, ?, ?)",
        (
            command.apply_id,
            command.session_id,
            command.task_id,
            command.auxiliary_graph_id,
            command.goal_id,
            command.auxiliary_graph_revision,
            command.terminal_auxiliary_node_id,
            command.terminal_node_revision,
            command.terminal_completion_id,
            command.semantic_settlement_id,
            command.invocation_turn_id,
            payload_sha256,
            result_json,
            _sha256_text(result_json),
            now,
        ),
    )
    required_json = _canonical_json(list(finish_receipt.required_completion_ids))
    gaps_json = _canonical_json(list(finish_receipt.non_blocking_gap_ids))
    finish_json = _model_json(finish_receipt)
    evaluation_json = _model_json(evaluation)
    conn.execute(
        "INSERT INTO insession_auxiliary_v2_finish_gate_receipts "
        "(finish_gate_receipt_id, apply_id, session_id, insession_task_id, "
        "auxiliary_graph_id, goal_id, auxiliary_graph_revision, "
        "structure_sha256, base_task_graph_revision, "
        "target_task_graph_revision, terminal_auxiliary_node_id, "
        "terminal_node_revision, terminal_completion_id, "
        "terminal_work_run_id, terminal_output_revision, "
        "required_completion_ids_json, required_completion_ids_sha256, "
        "proposal_sha256, validation_context_sha256, "
        "production_evaluation_json, production_evaluation_sha256, "
        "semantic_settlement_id, semantic_prompt_payload_sha256, "
        "semantic_review_policy_sha256, semantic_settlement_sha256, "
        "budget_snapshot_sha256, budget_disposition, readiness_status, "
        "non_blocking_gap_ids_json, non_blocking_gap_ids_sha256, "
        "receipt_json, receipt_sha256, created_turn_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            finish_receipt.finish_gate_receipt_id,
            command.apply_id,
            finish_receipt.session_id,
            finish_receipt.task_id,
            finish_receipt.auxiliary_graph_id,
            finish_receipt.goal_id,
            finish_receipt.auxiliary_graph_revision,
            finish_receipt.structure_sha256,
            finish_receipt.base_task_graph_revision,
            finish_receipt.target_task_graph_revision,
            finish_receipt.terminal_auxiliary_node_id,
            finish_receipt.terminal_node_revision,
            finish_receipt.terminal_completion_id,
            finish_receipt.terminal_work_run_id,
            finish_receipt.terminal_output_revision,
            required_json,
            _sha256_text(required_json),
            finish_receipt.proposal_sha256,
            finish_receipt.validation_context_sha256,
            evaluation_json,
            evaluation.evaluation_sha256,
            finish_receipt.semantic_settlement_id,
            finish_receipt.semantic_prompt_payload_sha256,
            finish_receipt.semantic_review_policy_sha256,
            finish_receipt.semantic_settlement_sha256,
            finish_receipt.budget_snapshot_sha256,
            finish_receipt.budget_disposition,
            finish_receipt.readiness_status,
            gaps_json,
            _sha256_text(gaps_json),
            finish_json,
            _sha256_text(finish_json),
            finish_receipt.created_turn_id,
            now,
        ),
    )
    proposal_json = _model_json(terminal_receipt.proposal)
    output_json = _model_json(terminal_receipt.output_window)
    conn.execute(
        "INSERT INTO insession_auxiliary_v2_terminal_proposal_receipts "
        "(terminal_proposal_receipt_id, finish_gate_receipt_id, session_id, "
        "insession_task_id, auxiliary_graph_id, goal_id, "
        "auxiliary_graph_revision, terminal_completion_id, "
        "proposal_schema_version, proposal_json, proposal_sha256, "
        "output_window_json, output_window_sha256, created_turn_id, "
        "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            terminal_receipt.terminal_proposal_receipt_id,
            terminal_receipt.finish_gate_receipt_id,
            terminal_receipt.session_id,
            terminal_receipt.task_id,
            terminal_receipt.auxiliary_graph_id,
            terminal_receipt.goal_id,
            terminal_receipt.auxiliary_graph_revision,
            terminal_receipt.terminal_completion_id,
            terminal_receipt.proposal.schema_version,
            proposal_json,
            terminal_receipt.proposal_sha256,
            output_json,
            terminal_receipt.output_window_sha256,
            terminal_receipt.created_turn_id,
            now,
        ),
    )


def _load_replay(conn, *, command, payload_sha256, row):
    result_json = str(row["result_json"])
    try:
        result = AuxiliaryTerminalSealResult.model_validate_json(result_json)
    except (TypeError, ValueError, ValidationError) as exc:
        _fail_corrupt("terminal seal replay result is not typed", exc)
    if (
        str(row["operation"]) != "seal_terminal_proposal"
        or str(row["session_id"]) != command.session_id
        or str(row["insession_task_id"]) != command.task_id
        or str(row["payload_sha256"]) != payload_sha256
        or _model_json(result) != result_json
        or _sha256_text(result_json) != str(row["result_sha256"])
        or result.apply_id != command.apply_id
        or result.finish_gate_receipt_id != command.finish_gate_receipt_id
        or result.terminal_proposal_receipt_id
        != command.terminal_proposal_receipt_id
    ):
        _fail_collision("terminal seal apply ID was reused with another payload")
    terminal = conn.execute(
        "SELECT proposal_json, proposal_sha256, output_window_json, "
        "output_window_sha256 FROM "
        "insession_auxiliary_v2_terminal_proposal_receipts "
        "WHERE terminal_proposal_receipt_id=? AND finish_gate_receipt_id=? "
        "AND terminal_completion_id=?",
        (
            result.terminal_proposal_receipt_id,
            result.finish_gate_receipt_id,
            result.terminal_completion_id,
        ),
    ).fetchone()
    finish = conn.execute(
        "SELECT receipt_json, receipt_sha256, production_evaluation_json, "
        "production_evaluation_sha256, proposal_sha256, "
        "semantic_settlement_id, semantic_settlement_sha256, "
        "readiness_status FROM insession_auxiliary_v2_finish_gate_receipts "
        "WHERE finish_gate_receipt_id=? AND apply_id=?",
        (result.finish_gate_receipt_id, command.apply_id),
    ).fetchone()
    if terminal is None or finish is None:
        _fail(
            AuxiliaryTerminalSealFailureCode.STORED_AUTHORITY_CORRUPT,
            "terminal seal replay lost immutable receipt rows",
        )
    try:
        settlement = (
            semantic_records._load_auxiliary_semantic_quorum_settlement(
                conn,
                session_id=command.session_id,
                task_id=command.task_id,
                auxiliary_graph_id=command.auxiliary_graph_id,
                goal_id=command.goal_id,
                auxiliary_graph_revision=command.auxiliary_graph_revision,
                frozen_prompt_payload_sha256=(
                    command.semantic_prompt_payload_sha256
                ),
            )
        )
    except semantic_records.AuxiliarySemanticVerificationPersistenceError as exc:
        _fail_corrupt("terminal seal replay semantic authority is corrupt", exc)
    if settlement is None:
        _fail(
            AuxiliaryTerminalSealFailureCode.STORED_AUTHORITY_CORRUPT,
            "terminal seal replay lost its semantic settlement",
        )
    try:
        finish_receipt = AuxiliaryFinishGateReceipt.model_validate_json(
            str(finish["receipt_json"])
        )
        evaluation = TaskGraphProductionEvaluation.model_validate_json(
            str(finish["production_evaluation_json"])
        )
        proposal = InSessionTaskGraphRevisionProposal.model_validate_json(
            str(terminal["proposal_json"])
        )
        output = OutputWindow.model_validate_json(
            str(terminal["output_window_json"])
        )
        output_proposal, lineage = _parse_terminal_output_material(
            output.content,
            expected_base_task_graph_revision=(
                finish_receipt.base_task_graph_revision
            ),
        )
    except (TypeError, ValueError, ValidationError) as exc:
        _fail_corrupt("terminal seal replay receipt is not typed", exc)
    source_output = conn.execute(
        "SELECT snapshot_json, snapshot_hash, frozen_at FROM "
        "insession_work_run_output_windows WHERE work_run_id=? "
        "AND output_revision=?",
        (output.work_run_id, output.output_revision),
    ).fetchone()
    source_completion = conn.execute(
        "SELECT completion_json, completion_sha256 FROM "
        "insession_auxiliary_node_completions_v2 WHERE completion_id=?",
        (result.terminal_completion_id,),
    ).fetchone()
    reference_request = settlement.requests[0] if settlement.requests else None
    if (
        _model_json(finish_receipt) != str(finish["receipt_json"])
        or _sha256_text(str(finish["receipt_json"]))
        != str(finish["receipt_sha256"])
        or _model_json(evaluation) != str(finish["production_evaluation_json"])
        or evaluation.evaluation_sha256
        != str(finish["production_evaluation_sha256"])
        or evaluation.evaluation_sha256
        != result.production_evaluation_sha256
        or _model_json(proposal) != str(terminal["proposal_json"])
        or _sha256_value(proposal.model_dump(mode="json"))
        != str(terminal["proposal_sha256"])
        or _model_json(output) != str(terminal["output_window_json"])
        or _sha256_text(str(terminal["output_window_json"]))
        != str(terminal["output_window_sha256"])
        or output_proposal != proposal
        or finish_receipt.base_task_graph_revision
        != command.expected_base_task_graph_revision
        or finish_receipt.target_task_graph_revision
        != (
            1
            if command.expected_base_task_graph_revision is None
            else command.expected_base_task_graph_revision + 1
        )
        or str(finish["proposal_sha256"]) != result.proposal_sha256
        or str(finish["semantic_settlement_id"])
        != result.semantic_settlement_id
        or str(finish["semantic_settlement_sha256"])
        != result.semantic_settlement_sha256
        or str(finish["readiness_status"]) != result.readiness_status
        or settlement.settlement_id != result.semantic_settlement_id
        or settlement.settlement_sha256 != result.semantic_settlement_sha256
        or settlement.host_disposition
        is not TaskGraphSemanticVerificationDisposition.PASS
        or reference_request is None
        or reference_request.task_graph_proposal_sha256
        != result.proposal_sha256
        or not _semantic_request_matches_terminal_material(
            reference_request,
            proposal=proposal,
            lineage=lineage,
            expected_base_task_graph_revision=(
                command.expected_base_task_graph_revision
            ),
        )
        or source_output is None
        or str(source_output["snapshot_json"])
        != str(terminal["output_window_json"])
        or str(source_output["snapshot_hash"])
        != str(terminal["output_window_sha256"])
        or source_output["frozen_at"] is None
        or source_completion is None
        or _sha256_text(str(source_completion["completion_json"]))
        != str(source_completion["completion_sha256"])
    ):
        _fail(
            AuxiliaryTerminalSealFailureCode.STORED_AUTHORITY_CORRUPT,
            "terminal seal replay authority is corrupt",
        )
    return result.model_copy(update={"status": "replayed"})


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
    raise AuxiliaryTerminalSealPersistenceError(code, message)


def _fail_collision(message: str) -> None:
    raise AuxiliaryTerminalSealIdentityCollision(
        AuxiliaryTerminalSealFailureCode.APPLY_ID_COLLISION,
        message,
    )


def _fail_corrupt(message: str, cause: BaseException) -> None:
    raise AuxiliaryTerminalSealPersistenceError(
        AuxiliaryTerminalSealFailureCode.STORED_AUTHORITY_CORRUPT,
        message,
    ) from cause


__all__ = [
    "AuxiliaryFinishGateReceipt",
    "AuxiliaryTerminalProposalReceipt",
    "AuxiliaryTerminalSealFailureCode",
    "AuxiliaryTerminalSealIdentityCollision",
    "AuxiliaryTerminalSealPersistenceError",
    "AuxiliaryTerminalSealResult",
    "SealAuxiliaryTerminalProposalCommand",
    "get_auxiliary_terminal_proposal_receipt",
    "seal_auxiliary_terminal_proposal",
]
