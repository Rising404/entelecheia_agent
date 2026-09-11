"""AuxiliaryGraph 规划及节点执行所用 SQLite 权威源。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from personagraph.l2.auxiliary_graph import (
    AuxiliaryGraphEdge,
    AuxiliaryGraphAggregate,
    AuxiliaryGraphRevisionReason,
    AuxiliaryGraphRevision,
    AuxiliaryNodeDefinition,
    AuxiliaryNodeExecutorKind,
    AuxiliaryNodeKind,
    AuxiliaryNodeReference,
    AuxiliaryPlanningGoal,
    AuxiliaryPlanningGoalStatus,
    PlanningAuthorityAnchor,
    PlanningAuthorityClass,
    PlanningAuthorityOriginKind,
    PlanningAuthoritySnapshot,
    PlanningContextArtifact,
    PlanningEpisodeBudgetProfile,
    PlanningEpisodeBudgetUsage,
    PlanningEpisodeBudget,
    TaskGraphSemanticTerminalRoute,
    TaskGraphSemanticVerificationDisposition,
    derive_task_graph_semantic_terminal_route,
)
from personagraph.l2.auxiliary_graph.contracts import (
    is_task_graph_semantic_user_information_block,
)
from personagraph.l2.auxiliary_graph.execution_frontier_contracts import (
    AuxiliaryGraphExecutionFrontier,
    ReadyAuxiliaryNodeExecutionCandidate,
    RecoverableAuxiliaryNodeExecutionCandidate,
    RecoverableAuxiliaryPrimitiveInvocationCandidate,
)
from personagraph.l2.task_graph.contracts import (
    InSessionTaskAcceptanceProposal,
    InSessionTaskGraphRevisionValidationContext,
)
from personagraph.l2.work_run import (
    AuxiliaryNodeSubject,
    NodeVerificationResult,
    OutputWindow,
    PreparedTaskNodeVerification,
    TaskNodeVerificationMutationResult,
    TaskNodeVerificationRequestStatus,
    WorkRunBudgetDisposition,
    WorkRunStatus,
    WorkRun,
    create_work_run,
    initialize_acceptance_progress,
    initialize_output_window,
)
from ...deps import StoreDeps
from ...execution_findings import (
    ExecutionFindingsPersistenceError,
    create_execution_findings_owner_companion_in_transaction,
)
from .auxiliary_graph_errors import AuxiliaryGraphApplyIdCollision, AuxiliaryGraphPersistenceError
from .auxiliary_node_execution_bindings import (
    _auxiliary_record_from_row,
    _auxiliary_verification_binding_hash,
    _ensure_auxiliary_execution_subject,
    _load_auxiliary_request_row,
    _load_exact_auxiliary_node,
    _require_auxiliary_subject,
    _require_auxiliary_subject_contract,
    _revalidate_auxiliary_request_binding,
)
from ..task_graph.insession_tasks import _load_authoritative_user_input
from ..planning.primitive_invocations import (
    _load_by_call_id as _load_planning_primitive_invocation_by_call_id,
)
from ..planning.auxiliary_planning_completions import (
    AuxiliaryInitialPlanningCompletionBinding,
    AuxiliaryPositivePlanningCompletionBinding,
    StoredAuxiliaryInitialPlanningCompletion,
    StoredAuxiliaryPositivePlanningCompletion,
    require_matching_auxiliary_initial_planning_completion,
    require_matching_auxiliary_positive_planning_completion,
    seal_auxiliary_initial_planning_completion,
    seal_auxiliary_positive_planning_completion,
)
from ..work_run.work_execution import (
    WorkExecutionApplyIdCollision,
    WorkExecutionMutationResult,
    WorkExecutionRevisionConflict,
    _allocate_id,
    _budget_from_row,
    _build_mutation_result,
    _canonical_json,
    _insert_budget_charge,
    _insert_receipt,
    _load_record,
    _load_output_window,
    _load_progress,
    _load_replay,
    _model_json,
    _payload_hash,
    _project_auxiliary_node_budget_failure,
    _require_active_window,
    _require_identifier,
    _require_owned_work_run_window,
    _require_positive,
    _require_progress_revision,
    _require_run_revision,
    _require_run_row,
    _require_settlement_budget_transition,
    _settlement_budget_checkpoint_id,
    _text_hash,
)
from ..work_run.work_verification import _load_current_submit_attempt, _load_supporting_tool_results


class _AuxiliaryGraphRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AuxiliaryGraphNodeProposalRecord(_AuxiliaryGraphRecord):
    """一个 revision 节点的 Host 准入定义输入。

    ``local_node_key`` 只在提案内有效，绝不会成为持久权威。Store 在提交事务中分配实际
    节点身份。
    """

    local_node_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    node_kind: AuxiliaryNodeKind
    executor_kind: AuxiliaryNodeExecutorKind
    title: str = Field(min_length=1, max_length=240)
    objective: str = Field(min_length=1, max_length=2_000)
    source_anchor_ids: tuple[str, ...] = Field(min_length=1, max_length=64)
    acceptance_criteria: tuple[InSessionTaskAcceptanceProposal, ...] = Field(
        min_length=1,
        max_length=64,
    )
    output_contract: str = Field(min_length=1, max_length=200)
    capability_profile_id: str | None = Field(default=None, max_length=200)
    input_resource_aliases: tuple[str, ...] = Field(default=(), max_length=64)
    required: bool = True
    origin_node_alias: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    )

    @field_validator("source_anchor_ids")
    @classmethod
    def _unique_source_anchors(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("node source anchors must be unique")
        if value != tuple(sorted(value)):
            raise ValueError("node source anchors must use canonical order")
        return value

    @field_validator("input_resource_aliases")
    @classmethod
    def _unique_input_aliases(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)) or value != tuple(sorted(value)):
            raise ValueError("input resource aliases must be canonical")
        return value


class AuxiliaryGraphEdgeProposalRecord(_AuxiliaryGraphRecord):
    dependency_node_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    consumer_node_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    required: Literal[True] = True

    @model_validator(mode="after")
    def _reject_self_edge(self) -> "AuxiliaryGraphEdgeProposalRecord":
        if self.dependency_node_key == self.consumer_node_key:
            raise ValueError("AuxiliaryGraph self edges are forbidden")
        return self


class AuxiliaryGraphRevisionProposalRecord(_AuxiliaryGraphRecord):
    revision_reason: AuxiliaryGraphRevisionReason
    terminal_node_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    nodes: tuple[AuxiliaryGraphNodeProposalRecord, ...] = Field(
        min_length=1,
        max_length=64,
    )
    edges: tuple[AuxiliaryGraphEdgeProposalRecord, ...] = Field(
        default=(),
        max_length=512,
    )


class StoredAuxiliaryGraphNode(_AuxiliaryGraphRecord):
    auxiliary_node_id: str
    node_revision: int = Field(ge=1)
    ordinal: int = Field(ge=0)
    local_node_key: str
    node_kind: str
    executor_kind: str
    title: str
    objective: str
    source_anchor_ids: tuple[str, ...]
    acceptance_criteria: tuple[InSessionTaskAcceptanceProposal, ...]
    output_contract: str
    capability_profile_id: str | None
    input_resource_aliases: tuple[str, ...]
    required: bool
    semantic_fingerprint: str
    origin_node_ref: AuxiliaryNodeReference | None
    status: str
    state_version: int = Field(ge=1)


class StoredAuxiliaryGraphEdge(_AuxiliaryGraphRecord):
    dependency_auxiliary_node_id: str
    consumer_auxiliary_node_id: str
    ordinal: int = Field(ge=0)
    required: bool


class StoredAuxiliaryGraphDetails(_AuxiliaryGraphRecord):
    session_id: str
    task_id: str
    auxiliary_graph_id: str
    control_state_version: int = Field(ge=1)
    goal_id: str
    goal_objective: str
    goal_status: str
    goal_state_version: int = Field(ge=1)
    base_task_graph_revision: int | None = Field(default=None, ge=1)
    target_task_graph_revision: int = Field(ge=1)
    auxiliary_graph_revision: int = Field(ge=1)
    parent_auxiliary_graph_revision: int | None = Field(default=None, ge=1)
    revision_status: str
    revision_state_version: int = Field(ge=1)
    source_turn_id: str
    reason: str
    authority_snapshot_id: str
    authority_snapshot_sha256: str
    structure_sha256: str
    terminal_auxiliary_node_id: str
    budget_profile: dict[str, Any]
    budget_usage: dict[str, Any]
    budget_state_version: int = Field(ge=1)
    nodes: tuple[StoredAuxiliaryGraphNode, ...]
    edges: tuple[StoredAuxiliaryGraphEdge, ...]
    aggregate: AuxiliaryGraphAggregate
    goal: AuxiliaryPlanningGoal
    revision: AuxiliaryGraphRevision
    authority_snapshot: PlanningAuthoritySnapshot
    budget: PlanningEpisodeBudget


class AuxiliaryGraphRevisionCommitResult(_AuxiliaryGraphRecord):
    status: Literal["applied", "replayed"]
    auxiliary_graph_id: str
    goal_id: str
    committed_auxiliary_graph_revision: int = Field(ge=1)
    control_state_version: int = Field(ge=1)
    goal_state_version: int = Field(ge=1)
    revision_state_version: int = Field(ge=1)
    budget_state_version: int = Field(ge=1)
    authority_snapshot_id: str
    authority_snapshot_sha256: str
    structure_sha256: str
    carried_completion_receipt_ids: tuple[str, ...] = Field(
        default=(),
        exclude_if=lambda value: not value,
    )
    initial_planning_completion: (
        StoredAuxiliaryInitialPlanningCompletion | None
    ) = None
    positive_planning_completion: (
        StoredAuxiliaryPositivePlanningCompletion | None
    ) = Field(default=None, exclude_if=lambda value: value is None)


class AuxiliaryNodePureModelCompletionCarryReceipt(_AuxiliaryGraphRecord):
    """一个无依赖纯模型节点的关闭式失败结转权威。

    当前窄契约只把模型完成结转表用于紧邻上一 revision、
    同 Turn 且不含工具、输入资源、ContextArtifact 或图依赖的 ``model_analysis`` 完成项。
    Host 原语和所有有副作用或携带资源的完成项不在本契约内，因此会在目标 revision 重跑。
    """

    schema_version: Literal[
        "auxiliary-node-pure-model-completion-carry-receipt-v1"
    ] = "auxiliary-node-pure-model-completion-carry-receipt-v1"
    carry_receipt_id: str
    apply_id: str
    session_id: str
    task_id: str
    auxiliary_graph_id: str
    goal_id: str
    source_subject: AuxiliaryNodeSubject
    source_completion_id: str
    source_completion_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_subject: AuxiliaryNodeSubject
    definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependency_completion_ids: tuple[str, ...] = ()
    dependency_closure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_artifact_ids: tuple[str, ...] = ()
    context_artifact_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_authority_projection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    authority_snapshot_id: str
    authority_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    capability_catalog_snapshot_id: str
    capability_catalog_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    freshness_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_turn_id: str

    @model_validator(mode="after")
    def _validate_narrow_carry(
        self,
    ) -> "AuxiliaryNodePureModelCompletionCarryReceipt":
        if self.source_subject.task_id != self.task_id or (
            self.target_subject.task_id != self.task_id
        ):
            raise ValueError("carry subjects must belong to the carried Task")
        if (
            self.source_subject.auxiliary_graph_id != self.auxiliary_graph_id
            or self.target_subject.auxiliary_graph_id != self.auxiliary_graph_id
            or self.target_subject.auxiliary_graph_revision
            != self.source_subject.auxiliary_graph_revision + 1
            or self.target_subject.node_id != self.source_subject.node_id
            or self.target_subject.node_revision
            != self.source_subject.node_revision + 1
        ):
            raise ValueError("carry subjects must be one contiguous node lineage step")
        if self.dependency_completion_ids or self.context_artifact_ids:
            raise ValueError("pure model carry cannot retain dependencies or artifacts")
        expected_dependency = _payload_hash(
            {
                "schema_version": "auxiliary-pure-model-dependency-closure-v1",
                "completion_ids": [],
            }
        )
        expected_artifacts = _payload_hash(
            {
                "schema_version": "auxiliary-pure-model-context-artifact-manifest-v1",
                "artifact_ids": [],
            }
        )
        if self.dependency_closure_sha256 != expected_dependency:
            raise ValueError("pure model carry dependency closure is not empty")
        if self.context_artifact_manifest_sha256 != expected_artifacts:
            raise ValueError("pure model carry artifact manifest is not empty")
        return self


class _PureModelCompletionSource(_AuxiliaryGraphRecord):
    subject: AuxiliaryNodeSubject
    completion_id: str
    completion_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_snapshot_id: str
    catalog_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def commit_auxiliary_graph_revision(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    insession_task_id: str,
    expected_task_state_version: int,
    expected_base_task_graph_revision: int | None,
    expected_control_state_version: int | None,
    expected_current_auxiliary_graph_revision: int | None,
    apply_id: str,
    goal_objective: str,
    proposal: AuxiliaryGraphRevisionProposalRecord,
    authority_context: Mapping[str, Any],
    budget_profile: Mapping[str, Any],
    auxiliary_graph_id: str | None = None,
    goal_id: str | None = None,
    initial_planning_completion: (
        AuxiliaryInitialPlanningCompletionBinding | None
    ) = None,
    positive_planning_completion: (
        AuxiliaryPositivePlanningCompletionBinding | None
    ) = None,
    terminal_candidate_semantic_settlement_id: str | None = None,
) -> AuxiliaryGraphRevisionCommitResult:
    """初始化或追加一个不可变且绑定来源的 DAG revision。

    这是首个持久 原语。它刻意不创建 WorkRun 或修改 TaskGraph。全部结构、权威、预算
    扣费、当前指针 CAS 和精确重放回执共享一个事务。
    """

    for name, value in (
        ("session_id", session_id),
        ("turn_id", turn_id),
        ("insession_task_id", insession_task_id),
        ("apply_id", apply_id),
    ):
        _require_identifier(name, value)
    _require_positive("expected_task_state_version", expected_task_state_version)
    if expected_base_task_graph_revision is not None:
        _require_positive(
            "expected_base_task_graph_revision",
            expected_base_task_graph_revision,
        )
    if expected_control_state_version is not None:
        _require_positive(
            "expected_control_state_version",
            expected_control_state_version,
        )
    if expected_current_auxiliary_graph_revision is not None:
        _require_positive(
            "expected_current_auxiliary_graph_revision",
            expected_current_auxiliary_graph_revision,
        )
    if auxiliary_graph_id is not None:
        _require_identifier("auxiliary_graph_id", auxiliary_graph_id)
    if goal_id is not None:
        _require_identifier("goal_id", goal_id)
    if initial_planning_completion is not None and not isinstance(
        initial_planning_completion,
        AuxiliaryInitialPlanningCompletionBinding,
    ):
        raise TypeError(
            "initial_planning_completion must be "
            "AuxiliaryInitialPlanningCompletionBinding"
        )
    if positive_planning_completion is not None and not isinstance(
        positive_planning_completion,
        AuxiliaryPositivePlanningCompletionBinding,
    ):
        raise TypeError(
            "positive_planning_completion must be "
            "AuxiliaryPositivePlanningCompletionBinding"
        )
    if terminal_candidate_semantic_settlement_id is not None:
        _require_identifier(
            "terminal_candidate_semantic_settlement_id",
            terminal_candidate_semantic_settlement_id,
        )
    normalized_objective = goal_objective.strip()
    if not normalized_objective:
        raise ValueError("goal_objective must not be empty")
    if not isinstance(proposal, AuxiliaryGraphRevisionProposalRecord):
        raise TypeError("proposal must be AuxiliaryGraphRevisionProposalRecord")
    authority_context_json = _canonical_json(dict(authority_context))
    try:
        budget_profile_model = PlanningEpisodeBudgetProfile.model_validate(
            dict(budget_profile)
        )
    except (TypeError, ValueError) as exc:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph budget profile is invalid"
        ) from exc
    budget_profile_json = _model_json(budget_profile_model)
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "insession_task_id": insession_task_id,
        "expected_task_state_version": expected_task_state_version,
        "expected_base_task_graph_revision": expected_base_task_graph_revision,
        "expected_control_state_version": expected_control_state_version,
        "expected_current_auxiliary_graph_revision": (
            expected_current_auxiliary_graph_revision
        ),
        "goal_objective": normalized_objective,
        "proposal": proposal.model_dump(mode="json"),
        "authority_context": json.loads(authority_context_json),
        "budget_profile": json.loads(budget_profile_json),
        "requested_auxiliary_graph_id": auxiliary_graph_id,
        "requested_goal_id": goal_id,
        "initial_planning_completion": (
            initial_planning_completion.model_dump(mode="json")
            if initial_planning_completion is not None
            else None
        ),
    }
    if positive_planning_completion is not None:
        payload["positive_planning_completion"] = (
            positive_planning_completion.model_dump(mode="json")
        )
    if terminal_candidate_semantic_settlement_id is not None:
        payload["terminal_candidate_semantic_settlement_id"] = (
            terminal_candidate_semantic_settlement_id
        )
    payload_hash = _payload_hash(payload)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = conn.execute(
            "SELECT apply_id, session_id, insession_task_id, auxiliary_graph_id, goal_id, "
            "invocation_turn_id, expected_control_state_version, "
            "committed_control_state_version, "
            "expected_current_auxiliary_graph_revision, "
            "committed_auxiliary_graph_revision, expected_goal_state_version, "
            "committed_goal_state_version, expected_budget_state_version, "
            "committed_budget_state_version, budget_ledger_id, "
            "committed_budget_snapshot_json, "
            "committed_budget_snapshot_sha256, payload_sha256, result_json, "
            "result_sha256 "
            "FROM insession_auxiliary_graph_revision_apply_receipts_v2 "
            "WHERE apply_id=?",
            (apply_id,),
        ).fetchone()
        if replay is not None:
            if (
                str(replay["session_id"]) != session_id
                or str(replay["insession_task_id"]) != insession_task_id
                or str(replay["payload_sha256"]) != payload_hash
            ):
                raise AuxiliaryGraphApplyIdCollision(
                    "AuxiliaryGraph apply id was reused for another payload"
                )
            stored = AuxiliaryGraphRevisionCommitResult.model_validate_json(
                str(replay["result_json"])
            )
            _validate_revision_replay(conn, stored=stored, receipt=replay)
            if terminal_candidate_semantic_settlement_id is not None:
                _require_terminal_candidate_semantic_replan_authority(
                    conn,
                    session_id=session_id,
                    turn_id=turn_id,
                    task_id=insession_task_id,
                    auxiliary_graph_id=str(replay["auxiliary_graph_id"]),
                    goal_id=str(replay["goal_id"]),
                    auxiliary_graph_revision=int(
                        replay["expected_current_auxiliary_graph_revision"]
                    ),
                    settlement_id=terminal_candidate_semantic_settlement_id,
                    archive=False,
                    now=now,
                )
            if initial_planning_completion is not None:
                completion = stored.initial_planning_completion
                if completion is None:
                    raise AuxiliaryGraphPersistenceError(
                        "replayed revision lost its initial-planning completion"
                    )
                require_matching_auxiliary_initial_planning_completion(
                    conn,
                    receipt=completion,
                    binding=initial_planning_completion,
                    apply_row=replay,
                    raw_result=json.loads(str(replay["result_json"])),
                )
            if positive_planning_completion is not None:
                positive_completion = stored.positive_planning_completion
                if positive_completion is None:
                    raise AuxiliaryGraphPersistenceError(
                        "replayed revision lost its positive-planning completion"
                    )
                require_matching_auxiliary_positive_planning_completion(
                    conn,
                    receipt=positive_completion,
                    binding=positive_planning_completion,
                    apply_row=replay,
                    raw_result=json.loads(str(replay["result_json"])),
                )
            return stored.model_copy(update={"status": "replayed"})

        task = conn.execute(
            "SELECT current_graph_revision, current_status, state_version, "
            "root_objective, created_turn_id, creation_source_start, "
            "creation_source_end, creation_source_sha256 "
            "FROM insession_tasks WHERE session_id=? AND insession_task_id=?",
            (session_id, insession_task_id),
        ).fetchone()
        if task is None:
            raise AuxiliaryGraphPersistenceError("unknown Task in this Session")
        actual_base = (
            int(task["current_graph_revision"])
            if task["current_graph_revision"] is not None
            else None
        )
        if actual_base != expected_base_task_graph_revision:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph base TaskGraph revision is stale"
            )
        if int(task["state_version"]) != expected_task_state_version:
            raise WorkExecutionRevisionConflict(
                expected=expected_task_state_version,
                actual=int(task["state_version"]),
            )
        if str(task["current_status"]) in {"completed", "cancelled"}:
            raise AuxiliaryGraphPersistenceError(
                "terminal Task cannot receive an AuxiliaryGraph revision"
            )
        if conn.execute(
            "SELECT 1 FROM runtime_turns WHERE session_id=? AND turn_id=?",
            (session_id, turn_id),
        ).fetchone() is None:
            raise AuxiliaryGraphPersistenceError("unknown source Turn")
        if conn.execute(
            "SELECT 1 FROM insession_task_turn_links WHERE session_id=? "
            "AND turn_id=? AND insession_task_id=?",
            (session_id, turn_id, insession_task_id),
        ).fetchone() is None:
            raise AuxiliaryGraphPersistenceError(
                "source Turn is not authoritatively linked to the Task"
            )

        authoritative_text = _load_authoritative_user_input(
            conn,
            session_id=session_id,
            turn_id=str(task["created_turn_id"]),
        )
        source_start = int(task["creation_source_start"])
        source_end = int(task["creation_source_end"])
        if not (0 <= source_start < source_end <= len(authoritative_text)):
            raise AuxiliaryGraphPersistenceError(
                "Task creation source span is invalid"
            )
        source_excerpt = authoritative_text[source_start:source_end]
        if _text_hash(source_excerpt) != str(task["creation_source_sha256"]):
            raise AuxiliaryGraphPersistenceError(
                "Task creation source authority is stale or corrupt"
            )

        control = conn.execute(
            "SELECT * FROM insession_auxiliary_graph_v2_containers "
            "WHERE session_id=? AND insession_task_id=?",
            (session_id, insession_task_id),
        ).fetchone()
        creating_container = control is None
        if creating_container:
            if terminal_candidate_semantic_settlement_id is not None:
                raise AuxiliaryGraphPersistenceError(
                    "terminal candidate settlement cannot initialize a graph"
                )
            if (
                expected_control_state_version is not None
                or expected_current_auxiliary_graph_revision is not None
            ):
                raise AuxiliaryGraphPersistenceError(
                    "new AuxiliaryGraph requires null expected control state"
                )
            graph_id = auxiliary_graph_id or _allocate_id(deps, "auxgraphv2")
            selected_goal_id = goal_id or _allocate_id(deps, "auxgoal")
            goal_ordinal = 1
            previous_revision = None
            control_state_before = 0
            new_goal = True
        else:
            graph_id = str(control["auxiliary_graph_id"])
            if auxiliary_graph_id is not None and auxiliary_graph_id != graph_id:
                raise AuxiliaryGraphPersistenceError(
                    "Task already owns another AuxiliaryGraph container"
                )
            control_state_before = int(control["state_version"])
            previous_revision = int(control["current_auxiliary_graph_revision"])
            if (
                expected_control_state_version != control_state_before
                or expected_current_auxiliary_graph_revision != previous_revision
            ):
                raise WorkExecutionRevisionConflict(
                    expected=expected_control_state_version or 0,
                    actual=control_state_before,
                )
            current_goal_id = str(control["current_goal_id"])
            requested_goal_id = goal_id or current_goal_id
            current_goal = conn.execute(
                "SELECT * FROM insession_auxiliary_graph_goals WHERE goal_id=? "
                "AND auxiliary_graph_id=?",
                (current_goal_id, graph_id),
            ).fetchone()
            if current_goal is None:
                raise AuxiliaryGraphPersistenceError(
                    "AuxiliaryGraph current goal binding is corrupt"
                )
            new_goal = requested_goal_id != current_goal_id
            if new_goal:
                if terminal_candidate_semantic_settlement_id is not None:
                    raise AuxiliaryGraphPersistenceError(
                        "terminal candidate settlement cannot create a new goal"
                    )
                if str(current_goal["status"]) not in {
                    "committed", "failed", "cancelled", "superseded",
                    "budget_exhausted",
                }:
                    raise AuxiliaryGraphPersistenceError(
                        "a nonterminal AuxiliaryGraph goal must be superseded first"
                    )
                selected_goal_id = requested_goal_id
                goal_ordinal = int(
                    conn.execute(
                        "SELECT COALESCE(MAX(goal_ordinal), 0) + 1 "
                        "FROM insession_auxiliary_graph_goals "
                        "WHERE auxiliary_graph_id=?",
                        (graph_id,),
                    ).fetchone()[0]
                )
            else:
                selected_goal_id = current_goal_id
                goal_ordinal = int(current_goal["goal_ordinal"])
                stored_base = (
                    int(current_goal["base_task_graph_revision"])
                    if current_goal["base_task_graph_revision"] is not None
                    else None
                )
                if stored_base != expected_base_task_graph_revision:
                    raise AuxiliaryGraphPersistenceError(
                        "AuxiliaryGraph goal cannot change its TaskGraph base"
                    )
                if str(current_goal["status"]) not in {
                    "active", "waiting_user", "waiting_authorization",
                    "waiting_external", "interrupted"
                }:
                    raise AuxiliaryGraphPersistenceError(
                        "current AuxiliaryGraph goal cannot accept another revision"
                    )
                if terminal_candidate_semantic_settlement_id is not None:
                    if (
                        proposal.revision_reason
                        is not AuxiliaryGraphRevisionReason.VERIFICATION_FAILED
                    ):
                        raise AuxiliaryGraphPersistenceError(
                            "terminal candidate semantic settlement only authorizes "
                            "verification_failed revision"
                        )
                    _require_terminal_candidate_semantic_replan_authority(
                        conn,
                        session_id=session_id,
                        turn_id=turn_id,
                        task_id=insession_task_id,
                        auxiliary_graph_id=graph_id,
                        goal_id=current_goal_id,
                        auxiliary_graph_revision=previous_revision,
                        settlement_id=(
                            terminal_candidate_semantic_settlement_id
                        ),
                        archive=True,
                        now=now,
                    )
                _require_revision_safe_point(
                    conn,
                    session_id=session_id,
                    task_id=insession_task_id,
                    auxiliary_graph_id=graph_id,
                    auxiliary_graph_revision=previous_revision,
                )

        auxiliary_revision = (previous_revision or 0) + 1
        authority_snapshot_id = _allocate_id(deps, "auxauth")
        authority_snapshot = _build_authority_snapshot(
            authority_snapshot_id=authority_snapshot_id,
            session_id=session_id,
            task_id=insession_task_id,
            auxiliary_graph_id=graph_id,
            goal_id=selected_goal_id,
            source_turn_id=turn_id,
            authority_context=json.loads(authority_context_json),
            task_creation_turn_id=str(task["created_turn_id"]),
            task_creation_start=source_start,
            task_creation_end=source_end,
            task_creation_sha256=str(task["creation_source_sha256"]),
            task_creation_excerpt=source_excerpt,
        )
        _validate_revision_proposal(
            proposal,
            authority_anchors=authority_snapshot.anchors,
        )
        authority_json = _model_json(authority_snapshot)
        authority_hash = authority_snapshot.snapshot_sha256

        if creating_container:
            conn.execute(
                "INSERT INTO insession_auxiliary_graph_v2_containers "
                "(auxiliary_graph_id, session_id, insession_task_id, "
                "current_goal_id, current_auxiliary_graph_revision, "
                "state_version, created_at, updated_at) "
                "VALUES (?, ?, ?, NULL, NULL, 1, ?, ?)",
                (graph_id, session_id, insession_task_id, now, now),
            )
        if new_goal or creating_container:
            budget_ledger_id = (
                "auxbudgetledger_v1_" + _text_hash(selected_goal_id)
            )
            target_revision = (
                1
                if expected_base_task_graph_revision is None
                else expected_base_task_graph_revision + 1
            )
            authorization_manifest_id = _allocate_id(deps, "auxauthmanifest")
            authorization_manifest = {
                "schema_version": "planning-authorization-manifest-v1",
                "session_id": session_id,
                "task_id": insession_task_id,
                "auxiliary_graph_id": graph_id,
                "goal_id": selected_goal_id,
                "source_turn_id": turn_id,
                "authority_snapshot_id": authority_snapshot_id,
                "authorization_anchor_ids": [
                    anchor.anchor_id
                    for anchor in authority_snapshot.anchors
                    if anchor.authority_class
                    is PlanningAuthorityClass.AUTHORIZATION
                ],
            }
            authorization_manifest_json = _canonical_json(
                authorization_manifest
            )
            authorization_manifest_hash = _text_hash(
                authorization_manifest_json
            )
            conn.execute(
                "INSERT INTO insession_auxiliary_graph_goals "
                "(goal_id, session_id, insession_task_id, auxiliary_graph_id, "
                "goal_ordinal, base_task_graph_revision, "
                "target_task_graph_revision, creation_turn_id, objective, "
                "authorization_manifest_id, authorization_manifest_sha256, "
                "budget_ledger_id, "
                "status, state_version, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', 1, ?, ?)",
                (
                    selected_goal_id,
                    session_id,
                    insession_task_id,
                    graph_id,
                    goal_ordinal,
                    expected_base_task_graph_revision,
                    target_revision,
                    turn_id,
                    normalized_objective,
                    authorization_manifest_id,
                    authorization_manifest_hash,
                    budget_ledger_id,
                    now,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO insession_auxiliary_authorization_manifests "
                "(authorization_manifest_id, session_id, insession_task_id, "
                "auxiliary_graph_id, goal_id, contract_version, manifest_json, "
                "manifest_sha256, created_turn_id, created_at) VALUES "
                "(?, ?, ?, ?, ?, 'planning-authorization-manifest-v1', "
                "?, ?, ?, ?)",
                (
                    authorization_manifest_id,
                    session_id,
                    insession_task_id,
                    graph_id,
                    selected_goal_id,
                    authorization_manifest_json,
                    authorization_manifest_hash,
                    turn_id,
                    now,
                ),
            )
            usage_before = PlanningEpisodeBudgetUsage()
            initial_budget = PlanningEpisodeBudget.create(
                budget_ledger_id=budget_ledger_id,
                goal_id=selected_goal_id,
                base_profile=budget_profile_model,
                usage=usage_before,
                extensions=(),
                state_version=1,
            )
            usage_before_json = _model_json(usage_before)
            extensions_json = _canonical_json([])
            conn.execute(
                "INSERT INTO insession_auxiliary_goal_budgets "
                "(goal_id, budget_ledger_id, session_id, insession_task_id, "
                "auxiliary_graph_id, "
                "contract_version, profile_json, profile_sha256, usage_json, "
                "usage_sha256, extensions_json, extensions_sha256, snapshot_json, "
                "snapshot_sha256, counter_completeness, state_version, created_at, "
                "updated_at) VALUES (?, ?, ?, ?, ?, 'planning-episode-budget-v1', "
                "?, ?, ?, ?, ?, ?, ?, ?, 'complete', 1, ?, ?)",
                (
                    selected_goal_id,
                    budget_ledger_id,
                    session_id,
                    insession_task_id,
                    graph_id,
                    budget_profile_json,
                    _text_hash(budget_profile_json),
                    usage_before_json,
                    _text_hash(usage_before_json),
                    extensions_json,
                    _text_hash(extensions_json),
                    _model_json(initial_budget),
                    initial_budget.snapshot_sha256,
                    now,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO insession_auxiliary_goal_budget_snapshots "
                "(goal_id, budget_ledger_id, state_version, session_id, "
                "insession_task_id, auxiliary_graph_id, snapshot_json, "
                "snapshot_sha256, created_turn_id, created_at) "
                "VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?)",
                (
                    selected_goal_id,
                    budget_ledger_id,
                    session_id,
                    insession_task_id,
                    graph_id,
                    _model_json(initial_budget),
                    initial_budget.snapshot_sha256,
                    turn_id,
                    now,
                ),
            )
            goal_state_before = 0
        else:
            goal_row = conn.execute(
                "SELECT state_version FROM insession_auxiliary_graph_goals "
                "WHERE goal_id=?",
                (selected_goal_id,),
            ).fetchone()
            budget_row = conn.execute(
                "SELECT profile_json, profile_sha256 FROM "
                "insession_auxiliary_goal_budgets WHERE goal_id=?",
                (selected_goal_id,),
            ).fetchone()
            if goal_row is None or budget_row is None:
                raise AuxiliaryGraphPersistenceError(
                    "AuxiliaryGraph goal budget authority is missing"
                )
            if (
                str(budget_row["profile_json"]) != budget_profile_json
                or str(budget_row["profile_sha256"])
                != _text_hash(budget_profile_json)
            ):
                raise AuxiliaryGraphPersistenceError(
                    "AuxiliaryGraph goal budget profile cannot change on revision"
                )
            goal_state_before = int(goal_row["state_version"])

        conn.execute(
            "INSERT INTO insession_auxiliary_authority_snapshots "
            "(authority_snapshot_id, session_id, insession_task_id, "
            "auxiliary_graph_id, goal_id, contract_version, snapshot_json, "
            "snapshot_sha256, created_turn_id, created_at) VALUES "
            "(?, ?, ?, ?, ?, 'planning-authority-snapshot-v1', ?, ?, ?, ?)",
            (
                authority_snapshot_id,
                session_id,
                insession_task_id,
                graph_id,
                selected_goal_id,
                authority_json,
                authority_hash,
                turn_id,
                now,
            ),
        )
        _insert_authority_anchors(
            conn,
            authority_snapshot_id=authority_snapshot_id,
            anchors=authority_snapshot.anchors,
        )

        node_identity_plan = _plan_node_revision_identities(
            conn,
            deps=deps,
            auxiliary_graph_id=graph_id,
            previous_auxiliary_graph_revision=previous_revision,
            proposal=proposal,
        )
        node_ids = {
            local_key: identity[0]
            for local_key, identity in node_identity_plan.items()
        }
        node_revisions = {
            local_key: identity[1]
            for local_key, identity in node_identity_plan.items()
        }
        materialized_nodes: list[AuxiliaryNodeDefinition] = []
        for ordinal, node in enumerate(proposal.nodes):
            node_id, node_revision, origin_node_ref = node_identity_plan[
                node.local_node_key
            ]
            materialized = AuxiliaryNodeDefinition.create(
                node_id=node_id,
                node_revision=node_revision,
                ordinal=ordinal,
                node_kind=node.node_kind,
                executor_kind=node.executor_kind,
                title=node.title,
                objective=node.objective,
                acceptance_criteria=node.acceptance_criteria,
                capability_profile_id=node.capability_profile_id,
                input_resource_aliases=node.input_resource_aliases,
                source_anchor_ids=node.source_anchor_ids,
                output_contract=node.output_contract,
                required=node.required,
                origin_node_ref=origin_node_ref,
            )
            materialized_nodes.append(materialized)
            conn.execute(
                "INSERT INTO insession_auxiliary_node_definitions_v2 "
                "(auxiliary_graph_id, auxiliary_node_id, node_revision, "
                "node_kind, executor_kind, title, objective, "
                "source_anchor_ids_json, "
                "acceptance_criteria_json, output_contract, "
                "capability_profile_id, input_resource_aliases_json, required, "
                "semantic_fingerprint, origin_auxiliary_node_id, "
                "origin_node_revision, definition_sha256, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    graph_id,
                    node_id,
                    materialized.node_revision,
                    materialized.node_kind.value,
                    materialized.executor_kind.value,
                    materialized.title,
                    materialized.objective,
                    _canonical_json(materialized.source_anchor_ids),
                    _canonical_json(
                        [
                            item.model_dump(mode="json")
                            for item in materialized.acceptance_criteria
                        ]
                    ),
                    materialized.output_contract,
                    materialized.capability_profile_id,
                    _canonical_json(materialized.input_resource_aliases),
                    int(materialized.required),
                    materialized.semantic_fingerprint,
                    (
                        materialized.origin_node_ref.node_id
                        if materialized.origin_node_ref is not None
                        else None
                    ),
                    (
                        materialized.origin_node_ref.node_revision
                        if materialized.origin_node_ref is not None
                        else None
                    ),
                    materialized.semantic_fingerprint,
                    now,
                ),
            )

        proposal_node_ordinals = {
            node.local_node_key: ordinal
            for ordinal, node in enumerate(proposal.nodes)
        }
        ordered_edges = tuple(
            sorted(
                proposal.edges,
                key=lambda edge: (
                    proposal_node_ordinals[edge.dependency_node_key],
                    proposal_node_ordinals[edge.consumer_node_key],
                ),
            )
        )
        materialized_edges = tuple(
            AuxiliaryGraphEdge(
                source_node_id=node_ids[edge.dependency_node_key],
                target_node_id=node_ids[edge.consumer_node_key],
                required=edge.required,
            )
            for edge in ordered_edges
        )
        materialized_revision = AuxiliaryGraphRevision.create(
            session_id=session_id,
            task_id=insession_task_id,
            auxiliary_graph_id=graph_id,
            goal_id=selected_goal_id,
            auxiliary_graph_revision=auxiliary_revision,
            parent_auxiliary_graph_revision=previous_revision,
            base_task_graph_revision=expected_base_task_graph_revision,
            source_turn_id=turn_id,
            revision_reason=proposal.revision_reason,
            authority_snapshot_id=authority_snapshot_id,
            authority_snapshot_sha256=authority_hash,
            terminal_node_id=node_ids[proposal.terminal_node_key],
            nodes=tuple(materialized_nodes),
            edges=materialized_edges,
        )
        structure_hash = materialized_revision.structure_sha256
        conn.execute(
            "INSERT INTO insession_auxiliary_graph_revision_snapshots "
            "(auxiliary_graph_id, auxiliary_graph_revision, session_id, "
            "insession_task_id, goal_id, parent_auxiliary_graph_revision, "
            "base_task_graph_revision, source_turn_id, reason, "
            "authority_snapshot_id, authority_snapshot_sha256, "
            "terminal_auxiliary_node_id, structure_contract_version, "
            "structure_sha256, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, 'auxiliary-graph-revision-v2', ?, ?)",
            (
                graph_id,
                auxiliary_revision,
                session_id,
                insession_task_id,
                selected_goal_id,
                previous_revision,
                expected_base_task_graph_revision,
                turn_id,
                materialized_revision.revision_reason.value,
                authority_snapshot_id,
                authority_hash,
                node_ids[proposal.terminal_node_key],
                structure_hash,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO insession_auxiliary_graph_revision_states_v2 "
            "(auxiliary_graph_id, auxiliary_graph_revision, status, "
            "state_version, updated_at) VALUES (?, ?, 'active', 1, ?)",
            (graph_id, auxiliary_revision, now),
        )
        for node, materialized in zip(proposal.nodes, materialized_nodes, strict=True):
            node_id = materialized.node_id
            conn.execute(
                "INSERT INTO insession_auxiliary_graph_revision_nodes_v2 "
                "(auxiliary_graph_id, auxiliary_graph_revision, "
                "auxiliary_node_id, node_revision, ordinal, required, "
                "local_node_key, carried_completion_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)",
                (
                    graph_id,
                    auxiliary_revision,
                    node_id,
                    materialized.node_revision,
                    materialized.ordinal,
                    int(materialized.required),
                    node.local_node_key,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO insession_auxiliary_node_states_v2 "
                "(auxiliary_graph_id, auxiliary_graph_revision, "
                "auxiliary_node_id, node_revision, status, state_version, "
                "updated_at) VALUES (?, ?, ?, ?, 'proposed', 1, ?)",
                (
                    graph_id,
                    auxiliary_revision,
                    node_id,
                    materialized.node_revision,
                    now,
                ),
            )
        consumer_ordinals: dict[str, int] = {}
        for edge in ordered_edges:
            edge_ordinal = consumer_ordinals.get(edge.consumer_node_key, 0)
            consumer_ordinals[edge.consumer_node_key] = edge_ordinal + 1
            conn.execute(
                "INSERT INTO insession_auxiliary_graph_edges "
                "(auxiliary_graph_id, auxiliary_graph_revision, "
                "dependency_auxiliary_node_id, dependency_node_revision, "
                "consumer_auxiliary_node_id, consumer_node_revision, required, "
                "ordinal, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    graph_id,
                    auxiliary_revision,
                    node_ids[edge.dependency_node_key],
                    node_revisions[edge.dependency_node_key],
                    node_ids[edge.consumer_node_key],
                    node_revisions[edge.consumer_node_key],
                    int(edge.required),
                    edge_ordinal,
                    now,
                ),
            )

        carried_completion_receipts = _derive_pure_model_completion_carries(
            conn,
            apply_id=apply_id,
            session_id=session_id,
            turn_id=turn_id,
            task_id=insession_task_id,
            auxiliary_graph_id=graph_id,
            goal_id=selected_goal_id,
            source_auxiliary_graph_revision=previous_revision,
            target_revision=materialized_revision,
            target_authority_snapshot=authority_snapshot,
            now=now,
        )

        if previous_revision is not None:
            previous_state = conn.execute(
                "SELECT status FROM insession_auxiliary_graph_revision_states_v2 "
                "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=?",
                (graph_id, previous_revision),
            ).fetchone()
            if previous_state is None:
                raise AuxiliaryGraphPersistenceError(
                    "previous AuxiliaryGraph revision state is missing"
                )
            if str(previous_state["status"]) not in {
                "committed", "failed", "cancelled", "superseded"
            }:
                if conn.execute(
                    "UPDATE insession_auxiliary_graph_revision_states_v2 "
                    "SET status='superseded', state_version=state_version+1, "
                    "updated_at=? WHERE auxiliary_graph_id=? "
                    "AND auxiliary_graph_revision=? AND status=?",
                    (
                        now,
                        graph_id,
                        previous_revision,
                        str(previous_state["status"]),
                    ),
                ).rowcount != 1:
                    raise AuxiliaryGraphPersistenceError(
                        "previous AuxiliaryGraph revision changed"
                    )

        budget_row = conn.execute(
            "SELECT budget_ledger_id, contract_version, profile_json, "
            "profile_sha256, usage_json, usage_sha256, extensions_json, "
            "extensions_sha256, snapshot_json, snapshot_sha256, state_version "
            "FROM "
            "insession_auxiliary_goal_budgets WHERE goal_id=?",
            (selected_goal_id,),
        ).fetchone()
        if budget_row is None:
            raise AuxiliaryGraphPersistenceError("goal budget disappeared")
        budget_before = _formal_budget_from_row(
            budget_row,
            expected_goal_id=selected_goal_id,
        )
        usage_before = budget_before.usage
        usage_after = PlanningEpisodeBudgetUsage.model_validate(
            usage_before.model_dump(mode="python")
            | {
                "current_graph_nodes": len(proposal.nodes),
                "current_graph_depth": _proposal_depth(proposal),
                "auxiliary_graph_revisions": (
                    usage_before.auxiliary_graph_revisions + 1
                ),
                "distinct_auxiliary_nodes": (
                    usage_before.distinct_auxiliary_nodes
                    + sum(
                        node.origin_node_alias is None
                        for node in proposal.nodes
                    )
                ),
            }
        )
        budget_state_before = int(budget_row["state_version"])
        budget_after = PlanningEpisodeBudget.create(
            budget_ledger_id=str(budget_row["budget_ledger_id"]),
            goal_id=selected_goal_id,
            base_profile=budget_before.base_profile,
            usage=usage_after,
            extensions=budget_before.extensions,
            state_version=budget_state_before + 1,
        )
        if budget_after.assessment.disposition.value == "hard_limit_reached":
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph revision exceeds the frozen hard budget"
            )
        usage_before_json = _model_json(usage_before)
        usage_after_json = _model_json(usage_after)
        if conn.execute(
            "UPDATE insession_auxiliary_goal_budgets SET usage_json=?, "
            "usage_sha256=?, snapshot_json=?, snapshot_sha256=?, "
            "state_version=state_version+1, updated_at=? "
            "WHERE goal_id=? AND state_version=? AND usage_sha256=?",
            (
                usage_after_json,
                _text_hash(usage_after_json),
                _model_json(budget_after),
                budget_after.snapshot_sha256,
                now,
                selected_goal_id,
                budget_state_before,
                str(budget_row["usage_sha256"]),
            ),
        ).rowcount != 1:
            raise AuxiliaryGraphPersistenceError(
                "goal budget changed during revision commit"
            )
        conn.execute(
            "INSERT INTO insession_auxiliary_goal_budget_snapshots "
            "(goal_id, budget_ledger_id, state_version, session_id, "
            "insession_task_id, auxiliary_graph_id, snapshot_json, "
            "snapshot_sha256, created_turn_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                selected_goal_id,
                str(budget_row["budget_ledger_id"]),
                budget_state_before + 1,
                session_id,
                insession_task_id,
                graph_id,
                _model_json(budget_after),
                budget_after.snapshot_sha256,
                turn_id,
                now,
            ),
        )
        budget_charge_payload = {
            "goal_id": selected_goal_id,
            "charge_kind": "graph_revision",
            "charge_key": apply_id,
            "usage_before": usage_before.model_dump(mode="json"),
            "usage_after": usage_after.model_dump(mode="json"),
            "delta": {
                "current_graph_nodes": (
                    usage_after.current_graph_nodes
                    - usage_before.current_graph_nodes
                ),
                "current_graph_depth": (
                    usage_after.current_graph_depth
                    - usage_before.current_graph_depth
                ),
                "auxiliary_graph_revisions": 1,
                "distinct_auxiliary_nodes": sum(
                    node.origin_node_alias is None for node in proposal.nodes
                ),
            },
        }
        previous_charge = conn.execute(
            "SELECT charge_sha256 FROM insession_auxiliary_goal_budget_charges "
            "WHERE goal_id=? ORDER BY budget_state_version_after DESC LIMIT 1",
            (selected_goal_id,),
        ).fetchone()
        previous_charge_sha256 = (
            str(previous_charge["charge_sha256"])
            if previous_charge is not None
            else None
        )
        charge_payload_sha256 = _payload_hash(budget_charge_payload)
        charge_sha256 = _payload_hash(
            {
                **budget_charge_payload,
                "budget_charge_id": "auxbudget_" + apply_id,
                "budget_ledger_id": str(budget_row["budget_ledger_id"]),
                "budget_state_version_before": budget_state_before,
                "budget_state_version_after": budget_state_before + 1,
                "previous_charge_sha256": previous_charge_sha256,
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
            "VALUES (?, ?, ?, ?, ?, ?, 'graph_revision', ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "auxbudget_" + apply_id,
                session_id,
                insession_task_id,
                graph_id,
                selected_goal_id,
                str(budget_row["budget_ledger_id"]),
                apply_id,
                budget_state_before,
                budget_state_before + 1,
                usage_before_json,
                _text_hash(usage_before_json),
                usage_after_json,
                _text_hash(usage_after_json),
                _model_json(budget_after),
                budget_after.snapshot_sha256,
                _canonical_json(budget_charge_payload["delta"]),
                charge_payload_sha256,
                previous_charge_sha256,
                charge_sha256,
                turn_id,
                now,
            ),
        )

        if not (new_goal or creating_container):
            if conn.execute(
                "UPDATE insession_auxiliary_graph_goals SET status='active', "
                "state_version=state_version+1, updated_at=? WHERE goal_id=? "
                "AND state_version=?",
                (now, selected_goal_id, goal_state_before),
            ).rowcount != 1:
                raise AuxiliaryGraphPersistenceError(
                    "AuxiliaryGraph goal changed during revision commit"
                )
            goal_state_after = goal_state_before + 1
        else:
            goal_state_after = 1

        control_state_after = (
            2 if creating_container else control_state_before + 1
        )
        expected_stored_control_state = (
            1 if creating_container else control_state_before
        )
        if conn.execute(
            "UPDATE insession_auxiliary_graph_v2_containers SET "
            "current_goal_id=?, current_auxiliary_graph_revision=?, "
            "state_version=?, updated_at=? WHERE auxiliary_graph_id=? "
            "AND state_version=?",
            (
                selected_goal_id,
                auxiliary_revision,
                control_state_after,
                now,
                graph_id,
                expected_stored_control_state,
            ),
        ).rowcount != 1:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph control pointer changed"
            )
        result = AuxiliaryGraphRevisionCommitResult(
            status="applied",
            auxiliary_graph_id=graph_id,
            goal_id=selected_goal_id,
            committed_auxiliary_graph_revision=auxiliary_revision,
            control_state_version=control_state_after,
            goal_state_version=goal_state_after,
            revision_state_version=1,
            budget_state_version=budget_state_before + 1,
            authority_snapshot_id=authority_snapshot_id,
            authority_snapshot_sha256=authority_hash,
            structure_sha256=structure_hash,
            carried_completion_receipt_ids=tuple(
                receipt.carry_receipt_id
                for receipt in carried_completion_receipts
            ),
        )
        if initial_planning_completion is not None:
            completion = seal_auxiliary_initial_planning_completion(
                conn,
                binding=initial_planning_completion,
                session_id=session_id,
                turn_id=turn_id,
                task_id=insession_task_id,
                auxiliary_graph_id=graph_id,
                goal_id=selected_goal_id,
                expected_current_auxiliary_graph_revision=(
                    expected_current_auxiliary_graph_revision
                ),
                revision_apply_id=apply_id,
                proposal_payload=proposal.model_dump(mode="json"),
                committed_result=result.model_dump(
                    mode="json",
                    exclude={"initial_planning_completion"},
                ),
                committed_budget_snapshot_sha256=budget_after.snapshot_sha256,
                created_at=now,
            )
            result = result.model_copy(
                update={"initial_planning_completion": completion}
            )
        if positive_planning_completion is not None:
            positive_completion = seal_auxiliary_positive_planning_completion(
                conn,
                binding=positive_planning_completion,
                session_id=session_id,
                turn_id=turn_id,
                task_id=insession_task_id,
                auxiliary_graph_id=graph_id,
                goal_id=selected_goal_id,
                expected_current_auxiliary_graph_revision=(
                    expected_current_auxiliary_graph_revision
                ),
                revision_apply_id=apply_id,
                proposal_payload=proposal.model_dump(mode="json"),
                committed_result=result.model_dump(
                    mode="json",
                    exclude={
                        "initial_planning_completion",
                        "positive_planning_completion",
                    },
                ),
                committed_budget_snapshot_sha256=budget_after.snapshot_sha256,
                created_at=now,
            )
            result = result.model_copy(
                update={"positive_planning_completion": positive_completion}
            )
        operation = (
            "initialize_goal_revision"
            if creating_container or new_goal
            else "append_goal_revision"
        )
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
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                apply_id,
                operation,
                session_id,
                insession_task_id,
                graph_id,
                selected_goal_id,
                turn_id,
                expected_control_state_version,
                control_state_after,
                expected_current_auxiliary_graph_revision,
                auxiliary_revision,
                None if new_goal or creating_container else goal_state_before,
                goal_state_after,
                None if new_goal or creating_container else budget_state_before,
                budget_state_before + 1,
                str(budget_row["budget_ledger_id"]),
                _model_json(budget_after),
                budget_after.snapshot_sha256,
                payload_hash,
                _model_json(result),
                _text_hash(_model_json(result)),
                now,
            ),
        )
        return result


def get_auxiliary_graph_for_task(
    deps: StoreDeps,
    *,
    session_id: str,
    insession_task_id: str,
) -> StoredAuxiliaryGraphDetails | None:
    """读取并哈希校验当前 目标与 revision 投影。"""

    _require_identifier("session_id", session_id)
    _require_identifier("insession_task_id", insession_task_id)
    deps.init_db()
    with deps.connect() as conn:
        control = conn.execute(
            "SELECT 1 FROM insession_auxiliary_graph_v2_containers "
            "WHERE session_id=? AND insession_task_id=?",
            (session_id, insession_task_id),
        ).fetchone()
        if control is None:
            return None
        return _load_auxiliary_graph(conn, session_id, insession_task_id)


def get_current_auxiliary_graph_revision(
    deps: StoreDeps,
    *,
    session_id: str,
    insession_task_id: str,
) -> AuxiliaryGraphRevision | None:
    """返回精确冻结领域 revision。"""

    details = get_auxiliary_graph_for_task(
        deps,
        session_id=session_id,
        insession_task_id=insession_task_id,
    )
    if details is None:
        return None
    return details.revision


def get_current_auxiliary_planning_goal(
    deps: StoreDeps,
    *,
    session_id: str,
    insession_task_id: str,
) -> AuxiliaryPlanningGoal | None:
    details = get_auxiliary_graph_for_task(
        deps,
        session_id=session_id,
        insession_task_id=insession_task_id,
    )
    return None if details is None else details.goal


def project_auxiliary_graph_execution_frontier(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    insession_task_id: str,
) -> AuxiliaryGraphExecutionFrontier:
    """投影一个精确当前 revision，不选择节点。

    这是未来驱动器的持久读取侧。它在一个读取事务内重新校验正式图、base TaskGraph、
    节点完成与结转权威，以及每项非终态执行主体绑定。调用方只会收到一个可恢复游标或
    按来源排序的新鲜前沿之一，绝不会同时收到二者。
    """

    for name, value in (
        ("session_id", session_id),
        ("turn_id", turn_id),
        ("insession_task_id", insession_task_id),
    ):
        _require_identifier(name, value)
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        turn = conn.execute(
            "SELECT session_id FROM runtime_turns WHERE turn_id=?",
            (turn_id,),
        ).fetchone()
        if turn is None or str(turn["session_id"]) != session_id:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph frontier Turn is outside this Session"
            )
        if conn.execute(
            "SELECT 1 FROM insession_task_turn_links WHERE session_id=? "
            "AND turn_id=? AND insession_task_id=?",
            (session_id, turn_id, insession_task_id),
        ).fetchone() is None:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph frontier Turn is not linked to its Task"
            )
        task = conn.execute(
            "SELECT current_graph_revision, current_status, state_version "
            "FROM insession_tasks WHERE session_id=? AND insession_task_id=?",
            (session_id, insession_task_id),
        ).fetchone()
        if task is None or str(task["current_status"]) in {
            "completed",
            "cancelled",
        }:
            raise AuxiliaryGraphPersistenceError(
                "terminal or unknown Task has no AuxiliaryGraph frontier"
            )
        details = _load_auxiliary_graph(conn, session_id, insession_task_id)
        current_task_graph_revision = (
            int(task["current_graph_revision"])
            if task["current_graph_revision"] is not None
            else None
        )
        if current_task_graph_revision != details.base_task_graph_revision:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph goal base differs from the current TaskGraph"
            )

        nodes_by_id = {node.auxiliary_node_id: node for node in details.nodes}
        if len(nodes_by_id) != len(details.nodes):
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph frontier contains duplicate node identities"
            )
        membership_rows = conn.execute(
            "SELECT auxiliary_node_id, node_revision, carried_completion_id "
            "FROM insession_auxiliary_graph_revision_nodes_v2 "
            "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=? "
            "ORDER BY ordinal, auxiliary_node_id",
            (details.auxiliary_graph_id, details.auxiliary_graph_revision),
        ).fetchall()
        membership_by_node = {
            str(row["auxiliary_node_id"]): row for row in membership_rows
        }
        if set(membership_by_node) != set(nodes_by_id):
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph membership differs from its frozen revision"
            )

        completion_ids = _load_auxiliary_frontier_completion_ids(
            conn,
            details=details,
            membership_by_node=membership_by_node,
        )
        for node_id, node in nodes_by_id.items():
            completed = node.status == "completed"
            if completed != (node_id in completion_ids):
                raise AuxiliaryGraphPersistenceError(
                    "AuxiliaryGraph completion authority disagrees with node state"
                )

        dependency_ids: dict[str, list[str]] = {
            node_id: [] for node_id in nodes_by_id
        }
        for edge in details.edges:
            completion_id = completion_ids.get(
                edge.dependency_auxiliary_node_id
            )
            if completion_id is not None:
                dependency_ids[edge.consumer_auxiliary_node_id].append(
                    completion_id
                )

        run_rows = conn.execute(
            "SELECT run.*, subject.subject_contract_version, "
            "binding.goal_id AS bound_goal_id, "
            "binding.executor_kind AS bound_executor_kind, "
            "binding.definition_sha256 AS bound_definition_sha256 "
            "FROM insession_work_runs AS run "
            "JOIN insession_execution_subjects AS subject "
            "ON subject.execution_subject_id=run.execution_subject_id "
            "JOIN insession_auxiliary_v2_node_execution_subject_bindings AS binding "
            "ON binding.binding_id=subject.auxiliary_v2_binding_id "
            "WHERE run.session_id=? AND run.insession_task_id=? "
            "AND run.auxiliary_graph_id=? "
            "AND run.auxiliary_graph_revision=? "
            "AND run.status NOT IN ('completed', 'failed', 'cancelled') "
            "ORDER BY run.created_at, run.work_run_id",
            (
                session_id,
                insession_task_id,
                details.auxiliary_graph_id,
                details.auxiliary_graph_revision,
            ),
        ).fetchall()
        run_by_node: dict[str, sqlite3.Row] = {}
        for row in run_rows:
            node_id = str(row["auxiliary_node_id"])
            node = nodes_by_id.get(node_id)
            if (
                node is None
                or str(row["subject_kind"]) != "auxiliary_node"
                or str(row["subject_contract_version"]) != "auxiliary_node_v2"
                or int(row["node_revision"]) != node.node_revision
                or str(row["bound_goal_id"]) != details.goal_id
                or str(row["bound_executor_kind"]) != node.executor_kind
                or str(row["bound_definition_sha256"])
                != node.semantic_fingerprint
                or node_id in run_by_node
            ):
                raise AuxiliaryGraphPersistenceError(
                    "AuxiliaryGraph WorkRun authority is stale or ambiguous"
                )
            if node.status not in {
                "active",
                "waiting_user",
                "waiting_authorization",
                "waiting_external",
                "interrupted",
            }:
                raise AuxiliaryGraphPersistenceError(
                    "AuxiliaryGraph WorkRun disagrees with node state"
                )
            _load_record(conn, str(row["work_run_id"]), session_id=session_id)
            run_by_node[node_id] = row
        if len(run_by_node) > 1:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph has more than one recoverable WorkRun"
            )

        verification_request_revisions: dict[str, int] = {}
        for row in run_by_node.values():
            verification_request_id = row["current_verification_request_id"]
            if verification_request_id is None:
                continue
            request_row = _load_auxiliary_request_row(
                conn,
                session_id=session_id,
                verification_request_id=str(verification_request_id),
            )
            request = _auxiliary_record_from_row(request_row).request
            if (
                request.work_run_id != str(row["work_run_id"])
                or request.status is not TaskNodeVerificationRequestStatus.PENDING
            ):
                raise AuxiliaryGraphPersistenceError(
                    "AuxiliaryGraph verification cursor is stale or detached"
                )
            _revalidate_auxiliary_request_binding(conn, request_row)
            verification_request_revisions[str(verification_request_id)] = (
                request.revision
            )

        recoverable = tuple(
            RecoverableAuxiliaryNodeExecutionCandidate(
                subject=AuxiliaryNodeSubject(
                    task_id=insession_task_id,
                    auxiliary_graph_id=details.auxiliary_graph_id,
                    auxiliary_graph_revision=details.auxiliary_graph_revision,
                    node_id=node_id,
                    node_revision=nodes_by_id[node_id].node_revision,
                ),
                ordinal=nodes_by_id[node_id].ordinal,
                local_node_key=nodes_by_id[node_id].local_node_key,
                executor_kind=AuxiliaryNodeExecutorKind(
                    nodes_by_id[node_id].executor_kind
                ),
                capability_profile_id=nodes_by_id[node_id].capability_profile_id,
                node_status=nodes_by_id[node_id].status,
                node_state_version=nodes_by_id[node_id].state_version,
                work_run_id=str(row["work_run_id"]),
                work_run_status=WorkRunStatus(str(row["status"])),
                work_run_reason=(
                    str(row["reason"]) if row["reason"] is not None else None
                ),
                work_run_revision=int(row["revision"]),
                current_attempt_id=(
                    str(row["current_attempt_id"])
                    if row["current_attempt_id"] is not None
                    else None
                ),
                current_verification_request_id=(
                    str(row["current_verification_request_id"])
                    if row["current_verification_request_id"] is not None
                    else None
                ),
                current_verification_request_revision=(
                    verification_request_revisions[
                        str(row["current_verification_request_id"])
                    ]
                    if row["current_verification_request_id"] is not None
                    else None
                ),
            )
            for node_id, row in run_by_node.items()
        )
        primitive_rows = conn.execute(
            "SELECT primitive_call_id FROM "
            "insession_auxiliary_planning_primitive_invocations "
            "WHERE session_id=? AND insession_task_id=? "
            "AND auxiliary_graph_id=? AND auxiliary_graph_revision=? "
            "AND status='reserved' ORDER BY reserved_at, primitive_call_id",
            (
                session_id,
                insession_task_id,
                details.auxiliary_graph_id,
                details.auxiliary_graph_revision,
            ),
        ).fetchall()
        if len(primitive_rows) > 1:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph has more than one reserved primitive"
            )
        recoverable_primitive_values: list[
            RecoverableAuxiliaryPrimitiveInvocationCandidate
        ] = []
        for row in primitive_rows:
            record = _load_planning_primitive_invocation_by_call_id(
                conn,
                str(row["primitive_call_id"]),
            )
            if record is None or record.status != "reserved":
                raise AuxiliaryGraphPersistenceError(
                    "reserved primitive projection disappeared or settled"
                )
            invocation = record.invocation
            subject = invocation.binding.producer_auxiliary_node
            node = nodes_by_id.get(subject.node_id)
            if (
                invocation.binding.session_id != session_id
                or invocation.binding.task_id != insession_task_id
                or invocation.binding.auxiliary_graph_id
                != details.auxiliary_graph_id
                or invocation.binding.goal_id != details.goal_id
                or subject.auxiliary_graph_revision
                != details.auxiliary_graph_revision
                or node is None
                or subject.node_revision != node.node_revision
                or node.executor_kind != "host_primitive"
                or node.capability_profile_id is None
                or node.status not in {"proposed", "interrupted"}
                or invocation.expected_task_state_version
                != int(task["state_version"])
                or invocation.expected_node_state_version != node.state_version
                or invocation.expected_control_state_version
                != details.control_state_version
                or invocation.expected_goal_state_version
                != details.goal_state_version
                or invocation.expected_revision_state_version
                != details.revision_state_version
                or invocation.expected_budget_state_version
                != details.budget_state_version
                or invocation.binding.authority_snapshot_id
                != details.authority_snapshot_id
                or invocation.authority_snapshot_sha256
                != details.authority_snapshot_sha256
                or invocation.structure_sha256 != details.structure_sha256
                or invocation.budget_snapshot_sha256
                != details.budget.snapshot_sha256
            ):
                raise AuxiliaryGraphPersistenceError(
                    "reserved primitive authority is stale or detached"
                )
            recoverable_primitive_values.append(
                RecoverableAuxiliaryPrimitiveInvocationCandidate(
                    subject=subject,
                    ordinal=node.ordinal,
                    local_node_key=node.local_node_key,
                    capability_profile_id=node.capability_profile_id,
                    node_state_version=node.state_version,
                    primitive_call_id=invocation.binding.primitive_call_id,
                    primitive_kind=invocation.primitive_kind.value,
                    invocation_turn_id=invocation.invocation_turn_id,
                    state_guard_sha256=invocation.state_guard_sha256,
                )
            )
        recoverable_primitive = tuple(recoverable_primitive_values)
        if recoverable and recoverable_primitive:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph WorkRun and primitive cursors coexist"
            )
        ready: tuple[ReadyAuxiliaryNodeExecutionCandidate, ...] = ()
        if (
            not recoverable
            and not recoverable_primitive
            and details.goal_status == "active"
            and details.revision_status == "active"
        ):
            ready = tuple(
                ReadyAuxiliaryNodeExecutionCandidate(
                    subject=AuxiliaryNodeSubject(
                        task_id=insession_task_id,
                        auxiliary_graph_id=details.auxiliary_graph_id,
                        auxiliary_graph_revision=details.auxiliary_graph_revision,
                        node_id=node.auxiliary_node_id,
                        node_revision=node.node_revision,
                    ),
                    ordinal=node.ordinal,
                    local_node_key=node.local_node_key,
                    executor_kind=AuxiliaryNodeExecutorKind(node.executor_kind),
                    capability_profile_id=node.capability_profile_id,
                    node_state_version=node.state_version,
                    dependency_completion_ids=tuple(
                        dependency_ids[node.auxiliary_node_id]
                    ),
                )
                for node in details.nodes
                if node.status in {"proposed", "interrupted"}
                and all(
                    edge.dependency_auxiliary_node_id in completion_ids
                    for edge in details.edges
                    if edge.consumer_auxiliary_node_id
                    == node.auxiliary_node_id
                )
            )
        completed_refs = tuple(
            AuxiliaryNodeReference(
                node_id=node.auxiliary_node_id,
                node_revision=node.node_revision,
            )
            for node in details.nodes
            if node.status == "completed"
        )
        blocking_refs = tuple(
            AuxiliaryNodeReference(
                node_id=node.auxiliary_node_id,
                node_revision=node.node_revision,
            )
            for node in details.nodes
            if node.status in {"failed", "cancelled", "superseded"}
        )
        projection = AuxiliaryGraphExecutionFrontier(
            session_id=session_id,
            turn_id=turn_id,
            task_id=insession_task_id,
            auxiliary_graph_id=details.auxiliary_graph_id,
            goal_id=details.goal_id,
            auxiliary_graph_revision=details.auxiliary_graph_revision,
            terminal_node_id=details.terminal_auxiliary_node_id,
            control_state_version=details.control_state_version,
            goal_state_version=details.goal_state_version,
            revision_state_version=details.revision_state_version,
            budget_state_version=details.budget_state_version,
            base_task_graph_revision=details.base_task_graph_revision,
            target_task_graph_revision=details.target_task_graph_revision,
            task_state_version=int(task["state_version"]),
            authority_snapshot_id=details.authority_snapshot_id,
            authority_snapshot_sha256=details.authority_snapshot_sha256,
            structure_sha256=details.structure_sha256,
            budget_snapshot_sha256=details.budget.snapshot_sha256,
            budget_disposition=details.budget.assessment.disposition,
            goal_status=AuxiliaryPlanningGoalStatus(details.goal_status),
            revision_status=details.revision_status,
            ready_fresh=ready,
            recoverable=recoverable,
            recoverable_primitive=recoverable_primitive,
            completed_node_refs=completed_refs,
            blocking_node_refs=blocking_refs,
        )
        conn.commit()
        return projection
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def _load_auxiliary_frontier_completion_ids(
    conn: sqlite3.Connection,
    *,
    details: StoredAuxiliaryGraphDetails,
    membership_by_node: Mapping[str, sqlite3.Row],
) -> dict[str, str]:
    """重新加载并哈希校验本地完成项及结转完成回执。"""

    nodes_by_id = {node.auxiliary_node_id: node for node in details.nodes}
    completion_rows = conn.execute(
        "SELECT completion.*, run.execution_subject_id, "
        "definition.definition_sha256, request.result_json, "
        "output.snapshot_hash AS output_snapshot_sha256 "
        "FROM insession_auxiliary_node_completions_v2 AS completion "
        "JOIN insession_work_runs AS run "
        "ON run.work_run_id=completion.work_run_id "
        "JOIN insession_auxiliary_node_definitions_v2 AS definition "
        "ON definition.auxiliary_graph_id=completion.auxiliary_graph_id "
        "AND definition.auxiliary_node_id=completion.auxiliary_node_id "
        "AND definition.node_revision=completion.node_revision "
        "JOIN insession_work_run_verification_requests AS request "
        "ON request.work_run_id=completion.work_run_id "
        "AND request.verification_request_id=completion.verification_request_id "
        "JOIN insession_work_run_output_windows AS output "
        "ON output.work_run_id=completion.work_run_id "
        "AND output.output_revision=completion.output_revision "
        "WHERE completion.auxiliary_graph_id=? "
        "AND completion.auxiliary_graph_revision=? "
        "ORDER BY completion.created_at, completion.completion_id",
        (details.auxiliary_graph_id, details.auxiliary_graph_revision),
    ).fetchall()
    completion_ids: dict[str, str] = {}
    for row in completion_rows:
        node_id = str(row["auxiliary_node_id"])
        node = nodes_by_id.get(node_id)
        if node is None or node_id in completion_ids or row["result_json"] is None:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph node completion is stale or ambiguous"
            )
        payload = {
            "schema_version": "auxiliary-node-completion-v2",
            "completion_id": str(row["completion_id"]),
            "execution_subject_id": str(row["execution_subject_id"]),
            "subject": AuxiliaryNodeSubject(
                task_id=details.task_id,
                auxiliary_graph_id=details.auxiliary_graph_id,
                auxiliary_graph_revision=details.auxiliary_graph_revision,
                node_id=node_id,
                node_revision=node.node_revision,
            ).model_dump(mode="json"),
            "goal_id": details.goal_id,
            "definition_sha256": str(row["definition_sha256"]),
            "verification_request_id": str(row["verification_request_id"]),
            "verification_result_sha256": _text_hash(str(row["result_json"])),
            "submitted_attempt_id": str(row["submitted_attempt_id"]),
            "output_revision": int(row["output_revision"]),
            "output_snapshot_sha256": str(row["output_snapshot_sha256"]),
        }
        completion_json = str(row["completion_json"])
        if (
            _canonical_json(payload) != completion_json
            or _text_hash(completion_json) != str(row["completion_sha256"])
        ):
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph node completion binding is corrupt"
            )
        completion_ids[node_id] = str(row["completion_id"])

# Host 原语刻意没有 WorkRun。其本地完成权威是已验证 PlanningContextArtifact，其中原语
# 调用也是当前观察 ID。此读取路径必须严格：已完成 Host 节点必须恰好拥有一个自验证
# 制品、观察与回执三元组；模型所有节点仍要求上方 WorkRun 完成账本。
    primitive_rows = conn.execute(
        "SELECT artifact.*, observation.snapshot_json AS observation_json, "
        "observation.snapshot_sha256 AS observation_sha256, "
        "receipt.receipt_json AS verification_json, "
        "receipt.receipt_sha256 AS stored_verification_sha256, "
        "definition.executor_kind "
        "FROM insession_auxiliary_planning_context_artifacts AS artifact "
        "JOIN insession_auxiliary_observations AS observation "
        "ON observation.observation_id=artifact.producer_primitive_call_id "
        "JOIN insession_auxiliary_context_verification_receipts AS receipt "
        "ON receipt.verification_receipt_id=artifact.verification_receipt_id "
        "JOIN insession_auxiliary_node_definitions_v2 AS definition "
        "ON definition.auxiliary_graph_id=artifact.auxiliary_graph_id "
        "AND definition.auxiliary_node_id=artifact.producer_auxiliary_node_id "
        "AND definition.node_revision=artifact.producer_node_revision "
        "WHERE artifact.session_id=? AND artifact.insession_task_id=? "
        "AND artifact.auxiliary_graph_id=? AND artifact.goal_id=? "
        "AND artifact.producer_auxiliary_graph_revision=? "
        "AND artifact.producer_primitive_call_id IS NOT NULL "
        "ORDER BY artifact.created_at, artifact.artifact_id",
        (
            details.session_id,
            details.task_id,
            details.auxiliary_graph_id,
            details.goal_id,
            details.auxiliary_graph_revision,
        ),
    ).fetchall()
    for row in primitive_rows:
        node_id = str(row["producer_auxiliary_node_id"])
        node = nodes_by_id.get(node_id)
        if (
            node is None
            or node_id in completion_ids
            or str(row["executor_kind"]) != "host_primitive"
        ):
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph Host completion is stale or ambiguous"
            )
        artifact_json = str(row["artifact_json"])
        observation_json = str(row["observation_json"])
        verification_json = str(row["verification_json"])
        try:
            artifact = PlanningContextArtifact.model_validate_json(
                artifact_json
            )
            observation_payload = json.loads(observation_json)
        except (TypeError, ValueError, RecursionError) as exc:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph Host completion payload is invalid"
            ) from exc
        result_payload = (
            observation_payload.get("result")
            if isinstance(observation_payload, dict)
            else None
        )
        result_artifact = (
            result_payload.get("artifact")
            if isinstance(result_payload, dict)
            else None
        )
        if (
            not isinstance(result_payload, dict)
            or not isinstance(result_artifact, dict)
            or artifact.session_id != details.session_id
            or artifact.task_id != details.task_id
            or artifact.auxiliary_graph_id != details.auxiliary_graph_id
            or artifact.goal_id != details.goal_id
            or artifact.producer_auxiliary_node.node_id != node_id
            or artifact.producer_auxiliary_node.auxiliary_graph_revision
            != details.auxiliary_graph_revision
            or artifact.producer_auxiliary_node.node_revision
            != node.node_revision
            or artifact.producer_primitive_call_id
            != str(row["producer_primitive_call_id"])
            or artifact.artifact_id != str(row["artifact_id"])
            or artifact.artifact_sha256 != str(row["artifact_sha256"])
            or _model_json(artifact) != artifact_json
            or result_artifact.get("artifact_sha256")
            != artifact.artifact_sha256
            or result_payload.get("settlement_sha256") is None
            or _canonical_json(observation_payload) != observation_json
            or _text_hash(observation_json) != str(row["observation_sha256"])
            or _text_hash(verification_json)
            != str(row["stored_verification_sha256"])
            or artifact.verification_receipt_sha256
            != str(row["verification_receipt_sha256"])
            or artifact.verification_receipt_sha256
            != str(row["stored_verification_sha256"])
        ):
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph Host completion binding is corrupt"
            )
        completion_ids[node_id] = artifact.artifact_id

    for node_id, membership in membership_by_node.items():
        carry_receipt_id = membership["carried_completion_id"]
        if carry_receipt_id is None:
            continue
        if node_id in completion_ids:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph node has local and carried completions"
            )
        receipt = _load_pure_model_carry_receipt(
            conn,
            details=details,
            target_node=nodes_by_id[node_id],
            carry_receipt_id=str(carry_receipt_id),
        )
        completion_ids[node_id] = receipt.source_completion_id
    return completion_ids


def require_auxiliary_graph_commit_context_valid(
    deps: StoreDeps,
    *,
    session_id: str,
    invocation_turn_id: str,
    task_id: str,
    context: InSessionTaskGraphRevisionValidationContext,
) -> None:
    """规划修改状态前要求精确规范来源上下文。"""

    for name, value in (
        ("session_id", session_id),
        ("invocation_turn_id", invocation_turn_id),
        ("task_id", task_id),
    ):
        _require_identifier(name, value)
    if not isinstance(context, InSessionTaskGraphRevisionValidationContext):
        raise TypeError(
            "context must be an InSessionTaskGraphRevisionValidationContext"
        )

    deps.init_db()
    with deps.connect() as conn:
        task_authority = conn.execute(
            "SELECT current_graph_revision, current_status, state_version, "
            "created_turn_id, creation_source_start, creation_source_end, "
            "creation_source_sha256 FROM insession_tasks "
            "WHERE session_id=? AND insession_task_id=?",
            (session_id, task_id),
        ).fetchone()
        if task_authority is None:
            raise AuxiliaryGraphPersistenceError(
                "Auxiliary graph commit context targets an unknown Task"
            )
        if conn.execute(
            "SELECT 1 FROM insession_task_turn_links WHERE session_id=? "
            "AND turn_id=? AND insession_task_id=?",
            (session_id, invocation_turn_id, task_id),
        ).fetchone() is None:
            raise AuxiliaryGraphPersistenceError(
                "Auxiliary graph invocation Turn is not linked to its Task"
            )
        trusted_context = _terminal_graph_validation_context(
            conn,
            session_id=session_id,
            invocation_turn_id=invocation_turn_id,
            task_id=task_id,
            task_authority=task_authority,
        )
        if context != trusted_context:
            raise AuxiliaryGraphPersistenceError(
                "Auxiliary request does not carry the canonical TaskGraph source context"
            )


def _require_auxiliary_dependencies_completed(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    subject: AuxiliaryNodeSubject,
    expected_structure_contract_version: str,
    expected_definition_sha256: str,
) -> None:
    """创建前重新检查当前 revision 的精确完成权威。

    模型 WorkRun、密封 Host PlanningContextArtifact 和正式结转完成项，均由执行前沿使用的
    同一加载器解释。这样可防止新执行器类型获得完成项时，写入侧就绪后备与读取侧前沿
    发生漂移。
    """

    details = _load_auxiliary_graph(conn, session_id, subject.task_id)
    consumer = tuple(
        node
        for node in details.nodes
        if node.auxiliary_node_id == subject.node_id
        and node.node_revision == subject.node_revision
    )
    if (
        details.auxiliary_graph_id != subject.auxiliary_graph_id
        or details.auxiliary_graph_revision
        != subject.auxiliary_graph_revision
        or details.revision.schema_version
        != expected_structure_contract_version
        or len(consumer) != 1
        or consumer[0].semantic_fingerprint != expected_definition_sha256
    ):
        raise AuxiliaryGraphPersistenceError(
            "dependency readiness lost its exact current structure binding"
        )
    membership_rows = conn.execute(
        "SELECT auxiliary_node_id, node_revision, carried_completion_id "
        "FROM insession_auxiliary_graph_revision_nodes_v2 "
        "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=? "
        "ORDER BY ordinal, auxiliary_node_id",
        (details.auxiliary_graph_id, details.auxiliary_graph_revision),
    ).fetchall()
    membership_by_node = {
        str(row["auxiliary_node_id"]): row for row in membership_rows
    }
    if (
        len(membership_by_node) != len(membership_rows)
        or set(membership_by_node)
        != {node.auxiliary_node_id for node in details.nodes}
    ):
        raise AuxiliaryGraphPersistenceError(
            "dependency readiness membership is corrupt"
        )
    completion_ids = _load_auxiliary_frontier_completion_ids(
        conn,
        details=details,
        membership_by_node=membership_by_node,
    )
    incoming = tuple(
        edge.dependency_auxiliary_node_id
        for edge in details.edges
        if edge.consumer_auxiliary_node_id == subject.node_id
    )
    if any(node_id not in completion_ids for node_id in incoming):
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryNode is not ready because a dependency is incomplete"
        )


def create_auxiliary_node_work_run(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    subject: AuxiliaryNodeSubject,
    expected_task_state_version: int,
    expected_node_state_version: int,
    expected_window_revision: int,
    apply_id: str,
    work_run_id: str | None = None,
) -> WorkExecutionMutationResult:
    """在一个精确当前正式 节点上创建 WorkRun。

    过期 revision 仍可通过已绑定 WorkRun 恢复，但绝不能通过此接缝获得新 WorkRun。
    """

    for name, value in (
        ("session_id", session_id),
        ("turn_id", turn_id),
        ("apply_id", apply_id),
    ):
        _require_identifier(name, value)
    for name, value in (
        ("expected_task_state_version", expected_task_state_version),
        ("expected_node_state_version", expected_node_state_version),
        ("expected_window_revision", expected_window_revision),
    ):
        _require_positive(name, value)
    if not isinstance(subject, AuxiliaryNodeSubject):
        raise TypeError("subject must be an AuxiliaryNodeSubject")
    if work_run_id is not None:
        _require_identifier("work_run_id", work_run_id)
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "subject_contract_version": "auxiliary_node_v2",
        "subject": subject.model_dump(mode="json"),
        "expected_task_state_version": expected_task_state_version,
        "expected_node_state_version": expected_node_state_version,
        "expected_window_revision": expected_window_revision,
        "requested_work_run_id": work_run_id,
    }
    payload_hash = _payload_hash(payload)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_replay(
            conn,
            apply_id=apply_id,
            operation="create_work_run",
            session_id=session_id,
            payload_hash=payload_hash,
        )
        if replay is not None:
            return replay
        window = _require_active_window(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            expected_window_revision=expected_window_revision,
        )
        if window["current_work_run_id"] is not None:
            raise AuxiliaryGraphPersistenceError(
                "the active Turn already points to a WorkRun"
            )
        task = conn.execute(
            "SELECT current_graph_revision, current_status, state_version "
            "FROM insession_tasks WHERE session_id=? AND insession_task_id=?",
            (session_id, subject.task_id),
        ).fetchone()
        if task is None or str(task["current_status"]) in {
            "completed",
            "cancelled",
        }:
            raise AuxiliaryGraphPersistenceError(
                "terminal or unknown Task cannot start a Auxiliary WorkRun"
            )
        if int(task["state_version"]) != expected_task_state_version:
            raise WorkExecutionRevisionConflict(
                expected=expected_task_state_version,
                actual=int(task["state_version"]),
            )
        node, acceptances = _load_exact_auxiliary_node(
            conn,
            session_id=session_id,
            subject=subject,
        )
        if (
            str(node["structure_contract_version"])
            != "auxiliary-graph-revision-v2"
            or str(node["current_goal_id"]) != str(node["goal_id"])
            or int(node["current_auxiliary_graph_revision"])
            != subject.auxiliary_graph_revision
            or str(node["goal_status"]) != "active"
            or str(node["revision_status"]) != "active"
        ):
            raise AuxiliaryGraphPersistenceError(
                "new WorkRun requires the current active formal goal/revision"
            )
        base_revision = (
            int(node["base_task_graph_revision"])
            if node["base_task_graph_revision"] is not None
            else None
        )
        current_revision = (
            int(task["current_graph_revision"])
            if task["current_graph_revision"] is not None
            else None
        )
        if current_revision != base_revision:
            raise AuxiliaryGraphPersistenceError(
                "Auxiliary goal base no longer matches current TaskGraph"
            )
        if str(node["executor_kind"]) not in {
            "model_work_run",
            "user_gate",
            "terminal_planner",
        }:
            raise AuxiliaryGraphPersistenceError(
                "node executor does not permit an Attempt WorkRun"
            )
        if str(node["status"]) not in {"proposed", "interrupted"}:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryNode state does not permit starting a WorkRun"
            )
        if int(node["state_version"]) != expected_node_state_version:
            raise WorkExecutionRevisionConflict(
                expected=expected_node_state_version,
                actual=int(node["state_version"]),
            )
        _require_auxiliary_dependencies_completed(
            conn,
            session_id=session_id,
            subject=subject,
            expected_structure_contract_version=str(
                node["structure_contract_version"]
            ),
            expected_definition_sha256=str(node["definition_sha256"]),
        )
        if conn.execute(
            "SELECT 1 FROM insession_task_turn_links WHERE session_id=? "
            "AND turn_id=? AND insession_task_id=?",
            (session_id, turn_id, subject.task_id),
        ).fetchone() is None:
            raise AuxiliaryGraphPersistenceError(
                "active Turn is not linked to the AuxiliaryGraph owner Task"
            )
        if conn.execute(
            "SELECT 1 FROM insession_work_runs WHERE session_id=? "
            "AND status='active'",
            (session_id,),
        ).fetchone() is not None:
            raise AuxiliaryGraphPersistenceError(
                "the Session already has an active WorkRun"
            )
        execution_subject_id = _ensure_auxiliary_execution_subject(
            conn,
            session_id=session_id,
            subject=subject,
            goal_id=str(node["goal_id"]),
            executor_kind=str(node["executor_kind"]),
            definition_sha256=str(node["definition_sha256"]),
            created_at=now,
        )
        if conn.execute(
            "SELECT 1 FROM insession_work_runs WHERE execution_subject_id=? "
            "AND status NOT IN ('completed', 'failed', 'cancelled')",
            (execution_subject_id,),
        ).fetchone() is not None:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryNode version already has a nonterminal WorkRun"
            )

        run_id = work_run_id or _allocate_id(deps, "workrun")
        run = create_work_run(work_run_id=run_id, subject=subject)
        progress = initialize_acceptance_progress(
            work_run_id=run_id,
            subject=subject,
            acceptance_ids=tuple(item.acceptance_id for item in acceptances),
        )
        output = initialize_output_window(
            work_run_id=run_id,
            updated_turn_id=turn_id,
        )
        conn.execute(
            "INSERT INTO insession_work_runs "
            "(work_run_id, execution_subject_id, session_id, subject_kind, "
            "insession_task_id, graph_revision, insession_task_node_id, "
            "auxiliary_graph_id, auxiliary_graph_revision, auxiliary_node_id, "
            "node_revision, status, reason, revision, max_attempts, "
            "soft_active_seconds, hard_active_seconds, attempts_started, "
            "active_seconds_consumed, current_attempt_id, "
            "current_verification_request_id, created_turn_id, updated_turn_id, "
            "created_at, updated_at) VALUES (?, ?, ?, 'auxiliary_node', ?, NULL, "
            "NULL, ?, ?, ?, ?, 'active', NULL, ?, ?, ?, ?, ?, ?, NULL, NULL, "
            "?, ?, ?, ?)",
            (
                run_id,
                execution_subject_id,
                session_id,
                subject.task_id,
                subject.auxiliary_graph_id,
                subject.auxiliary_graph_revision,
                subject.node_id,
                subject.node_revision,
                run.revision,
                run.budget.max_attempts,
                run.budget.soft_active_seconds,
                run.budget.hard_active_seconds,
                run.budget.attempts_started,
                run.budget.active_seconds_consumed,
                turn_id,
                turn_id,
                now,
                now,
            ),
        )
        try:
            create_execution_findings_owner_companion_in_transaction(
                conn,
                session_id=session_id,
                owner_kind="work_run",
                execution_owner_id=run_id,
                now=now,
            )
        except ExecutionFindingsPersistenceError as exc:
            raise AuxiliaryGraphPersistenceError(
                "Auxiliary WorkRun findings companion could not be created"
            ) from exc
        progress_json = _model_json(progress)
        conn.execute(
            "INSERT INTO insession_work_run_acceptance_progress "
            "(work_run_id, progress_revision, evaluated_output_revision, "
            "snapshot_hash, snapshot_json, updated_attempt_id, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?, NULL, ?, ?)",
            (
                run_id,
                progress.revision,
                progress.evaluated_output_revision,
                _text_hash(progress_json),
                progress_json,
                now,
                now,
            ),
        )
        output_json = _model_json(output)
        conn.execute(
            "INSERT INTO insession_work_run_output_windows "
            "(work_run_id, session_id, output_revision, snapshot_hash, "
            "snapshot_json, updated_turn_id, updated_attempt_id, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)",
            (
                run_id,
                session_id,
                output.output_revision,
                _text_hash(output_json),
                output_json,
                turn_id,
                now,
                now,
            ),
        )
        link_revision = int(window["turn_workrun_link_revision"] or 0) + 1
        conn.execute(
            "INSERT INTO insession_work_run_turn_links "
            "(session_id, turn_id, work_run_id, link_revision, relation, "
            "created_at) VALUES (?, ?, ?, ?, 'started', ?)",
            (session_id, turn_id, run_id, link_revision, now),
        )
        if conn.execute(
            "UPDATE insession_auxiliary_node_states_v2 SET status='active', "
            "state_version=state_version+1, updated_at=? "
            "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=? "
            "AND auxiliary_node_id=? AND node_revision=? AND state_version=? "
            "AND status IN ('proposed', 'interrupted')",
            (
                now,
                subject.auxiliary_graph_id,
                subject.auxiliary_graph_revision,
                subject.node_id,
                subject.node_revision,
                expected_node_state_version,
            ),
        ).rowcount != 1:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryNode changed during WorkRun creation"
            )
        next_task_state_version = expected_task_state_version + 1
        if conn.execute(
            "UPDATE insession_tasks SET current_status='active', state_version=?, "
            "updated_at=? WHERE session_id=? AND insession_task_id=? "
            "AND current_graph_revision IS ? AND state_version=? "
            "AND current_status NOT IN ('completed', 'cancelled')",
            (
                next_task_state_version,
                now,
                session_id,
                subject.task_id,
                base_revision,
                expected_task_state_version,
            ),
        ).rowcount != 1:
            raise AuxiliaryGraphPersistenceError(
                "Task changed during Auxiliary WorkRun creation"
            )
        next_window_revision = expected_window_revision + 1
        if conn.execute(
            "UPDATE turn_execution_windows SET current_work_run_id=?, "
            "current_attempt_id=NULL, turn_workrun_link_revision=?, "
            "state_version=?, updated_at=? WHERE session_id=? AND turn_id=? "
            "AND window_state='active' AND state_version=? "
            "AND current_work_run_id IS NULL",
            (
                run_id,
                link_revision,
                next_window_revision,
                now,
                session_id,
                turn_id,
                expected_window_revision,
            ),
        ).rowcount != 1:
            raise AuxiliaryGraphPersistenceError(
                "Turn Window changed during Auxiliary WorkRun creation"
            )
        result = _build_mutation_result(
            conn,
            status="applied",
            work_run_id=run_id,
            turn_work_run_link_revision=link_revision,
            window_state_version=next_window_revision,
        ).model_copy(
            update={
                "task_state_version": next_task_state_version,
                "node_state_version": expected_node_state_version + 1,
            }
        )
        _insert_receipt(
            conn,
            apply_id=apply_id,
            operation="create_work_run",
            session_id=session_id,
            work_run_id=run_id,
            payload_hash=payload_hash,
            result=result,
            now=now,
        )
        return result


def prepare_auxiliary_node_verification(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    expected_work_run_revision: int,
    expected_progress_revision: int,
    expected_output_revision: int,
    expected_window_revision: int,
    apply_id: str,
    verification_request_id: str | None = None,
) -> TaskNodeVerificationMutationResult:
    """持久化一个带标签 AuxiliaryNode 验证请求。"""

    for name, value in (
        ("session_id", session_id),
        ("turn_id", turn_id),
        ("work_run_id", work_run_id),
        ("apply_id", apply_id),
    ):
        _require_identifier(name, value)
    for name, value in (
        ("expected_work_run_revision", expected_work_run_revision),
        ("expected_progress_revision", expected_progress_revision),
        ("expected_output_revision", expected_output_revision),
        ("expected_window_revision", expected_window_revision),
    ):
        _require_positive(name, value)
    if verification_request_id is not None:
        _require_identifier("verification_request_id", verification_request_id)
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "work_run_id": work_run_id,
        "expected_work_run_revision": expected_work_run_revision,
        "expected_progress_revision": expected_progress_revision,
        "expected_output_revision": expected_output_revision,
        "expected_window_revision": expected_window_revision,
        "requested_verification_request_id": verification_request_id,
    }
    payload["subject_contract_version"] = "auxiliary_node_v2"
    payload_hash = _payload_hash(payload)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_verification_replay(
            conn,
            apply_id=apply_id,
            operation="prepare_verification",
            session_id=session_id,
            work_run_id=work_run_id,
            payload_hash=payload_hash,
        )
        if replay is not None:
            return replay
        window = _require_owned_work_run_window(
            conn,
            session_id,
            turn_id,
            work_run_id,
            expected_window_revision,
        )
        if window["current_attempt_id"] is not None:
            raise AuxiliaryGraphPersistenceError(
                "verification prepare requires no current Attempt"
            )
        run_row = _require_run_row(conn, work_run_id, session_id)
        _require_run_revision(run_row, expected_work_run_revision)
        subject = _require_auxiliary_subject_contract(
            conn,
            run_row,
        )
        task_authority = conn.execute(
            "SELECT current_graph_revision, current_status, state_version, "
            "created_turn_id, creation_source_start, creation_source_end, "
            "creation_source_sha256 "
            "FROM insession_tasks WHERE session_id=? AND insession_task_id=?",
            (session_id, subject.task_id),
        ).fetchone()
        if task_authority is None or str(task_authority["current_status"]) != "active":
            raise AuxiliaryGraphPersistenceError(
                "Auxiliary WorkRun owner Task is no longer active"
            )
        if (
            str(run_row["status"]) != "active"
            or str(run_row["reason"] or "") != "verification_pending"
            or run_row["current_attempt_id"] is not None
            or run_row["current_verification_request_id"] is not None
        ):
            raise AuxiliaryGraphPersistenceError(
                "Auxiliary WorkRun is not ready for verification"
            )
        node_row, acceptances = _load_exact_auxiliary_node(
            conn,
            session_id=session_id,
            subject=subject,
            expected_status="active",
        )
        binding = conn.execute(
            "SELECT aux2.goal_id, aux2.executor_kind, "
            "aux2.definition_sha256 FROM insession_execution_subjects AS es "
            "JOIN insession_auxiliary_v2_node_execution_subject_bindings AS aux2 "
            "ON aux2.binding_id=es.auxiliary_v2_binding_id "
            "WHERE es.execution_subject_id=?",
            (str(run_row["execution_subject_id"]),),
        ).fetchone()
        if (
            binding is None
            or str(binding["goal_id"]) != str(node_row["goal_id"])
            or str(binding["executor_kind"]) != str(node_row["executor_kind"])
            or str(binding["definition_sha256"])
            != str(node_row["definition_sha256"])
        ):
            raise AuxiliaryGraphPersistenceError(
                "WorkRun exact definition binding has drifted"
            )
        node_title = str(node_row["title"])
        node_objective = str(node_row["objective"])
        progress = _load_progress(conn, work_run_id)
        _require_progress_revision(progress, expected_progress_revision)
        output, output_hash = _load_output_window(conn, work_run_id)
        if output.output_revision != expected_output_revision:
            raise AuxiliaryGraphPersistenceError(
                "Auxiliary verification OutputWindow revision is stale"
            )
        if not output.content.strip() or progress.evaluated_output_revision != output.output_revision:
            raise AuxiliaryGraphPersistenceError(
                "Auxiliary verification requires one nonempty submitted OutputWindow"
            )
        if not all(item.model_claimed_satisfied for item in progress.items):
            raise AuxiliaryGraphPersistenceError(
                "Auxiliary verification requires all Acceptance self-claims"
            )
        acceptance_ids = tuple(item.acceptance_id for item in acceptances)
        if set(acceptance_ids) != {item.acceptance_id for item in progress.items}:
            raise AuxiliaryGraphPersistenceError(
                "AcceptanceProgress does not cover the AuxiliaryNode"
            )
        submitted_attempt = _load_current_submit_attempt(
            conn,
            work_run_id=work_run_id,
            output_revision=output.output_revision,
        )
        supporting_ids = tuple(
            sorted(
                {
                    result_id
                    for item in progress.items
                    for result_id in item.supporting_tool_result_ids
                }
            )
        )
        supporting_results = _load_supporting_tool_results(
            conn,
            work_run_id=work_run_id,
            result_ids=supporting_ids,
            before_attempt_ordinal=submitted_attempt.ordinal,
        )
        request_id = verification_request_id or _allocate_id(deps, "verification")
        prepared_budget = _budget_from_row(run_row)
        binding_hash = _auxiliary_verification_binding_hash(
            verification_request_id=request_id,
            session_id=session_id,
            request_turn_id=turn_id,
            work_run_id=work_run_id,
            subject=subject,
            node_title=node_title,
            node_objective=node_objective,
            submitted_attempt=submitted_attempt,
            locked_work_run_revision=expected_work_run_revision,
            progress=progress,
            output=output,
            output_hash=output_hash,
            acceptances=acceptances,
            supporting_results=supporting_results,
            prepared_budget=prepared_budget,
        )
        conn.execute(
            "INSERT INTO insession_work_run_verification_requests "
            "(verification_request_id, execution_subject_id, session_id, "
            "request_turn_id, work_run_id, "
            "subject_kind, insession_task_id, graph_revision, insession_task_node_id, "
            "auxiliary_graph_id, auxiliary_graph_revision, auxiliary_node_id, "
            "node_revision, submitted_attempt_id, output_revision, "
            "acceptance_progress_revision, acceptance_ids_json, "
            "supporting_tool_result_ids_json, locked_work_run_revision, "
            "request_binding_hash, prepared_budget_json, request_revision, status, "
            "technical_error_code, result_json, all_pass, created_at, updated_at, "
            "completed_at) VALUES (?, ?, ?, ?, ?, 'auxiliary_node', ?, NULL, NULL, "
            "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'pending', NULL, NULL, NULL, "
            "?, ?, NULL)",
            (
                request_id,
                str(run_row["execution_subject_id"]),
                session_id,
                turn_id,
                work_run_id,
                subject.task_id,
                subject.auxiliary_graph_id,
                subject.auxiliary_graph_revision,
                subject.node_id,
                subject.node_revision,
                submitted_attempt.attempt_id,
                output.output_revision,
                progress.revision,
                _canonical_json(acceptance_ids),
                _canonical_json(supporting_ids),
                expected_work_run_revision,
                binding_hash,
                _model_json(prepared_budget),
                now,
                now,
            ),
        )
        next_run_revision = expected_work_run_revision + 1
        if conn.execute(
            "UPDATE insession_work_runs SET current_verification_request_id=?, "
            "revision=?, updated_turn_id=?, updated_at=? WHERE work_run_id=? "
            "AND revision=? AND status='active' AND reason='verification_pending' "
            "AND current_attempt_id IS NULL AND current_verification_request_id IS NULL",
            (
                request_id,
                next_run_revision,
                turn_id,
                now,
                work_run_id,
                expected_work_run_revision,
            ),
        ).rowcount != 1:
            raise AuxiliaryGraphPersistenceError(
                "Auxiliary WorkRun changed during verification prepare"
            )
        next_window_revision = expected_window_revision + 1
        if conn.execute(
            "UPDATE turn_execution_windows SET latest_checkpoint_id=?, "
            "stage='VERIFICATION', state_version=?, updated_at=? "
            "WHERE session_id=? AND turn_id=? AND window_state='active' "
            "AND current_work_run_id=? AND current_attempt_id IS NULL "
            "AND state_version=?",
            (
                request_id,
                next_window_revision,
                now,
                session_id,
                turn_id,
                work_run_id,
                expected_window_revision,
            ),
        ).rowcount != 1:
            raise AuxiliaryGraphPersistenceError(
                "Turn Window changed during Auxiliary verification prepare"
            )
        mutation = TaskNodeVerificationMutationResult(
            status="applied",
            verification_request_id=request_id,
            verification_request_revision=1,
            verification_request_status=TaskNodeVerificationRequestStatus.PENDING,
            work_run_id=work_run_id,
            work_run_revision=next_run_revision,
            work_run_status=WorkRunStatus.ACTIVE,
            work_run_reason="verification_pending",
            window_state_version=next_window_revision,
        )
        _insert_receipt(
            conn,
            apply_id=apply_id,
            operation="prepare_verification",
            session_id=session_id,
            work_run_id=work_run_id,
            payload_hash=payload_hash,
            result=mutation,
            now=now,
        )
        return mutation


def get_prepared_auxiliary_node_verification(
    deps: StoreDeps,
    *,
    session_id: str,
    invocation_turn_id: str,
    verification_request_id: str,
) -> PreparedTaskNodeVerification:
    """根据带标签持久引用重建精确验证器输入。"""

    for name, value in (
        ("session_id", session_id),
        ("invocation_turn_id", invocation_turn_id),
        ("verification_request_id", verification_request_id),
    ):
        _require_identifier(name, value)
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        row = _load_auxiliary_request_row(
            conn,
            session_id=session_id,
            verification_request_id=verification_request_id,
        )
        record = _auxiliary_record_from_row(row)
        request = record.request
        if request.status is not TaskNodeVerificationRequestStatus.PENDING:
            raise AuxiliaryGraphPersistenceError(
                "only pending Auxiliary verification is callable"
            )
        run_row = _require_run_row(conn, request.work_run_id, session_id)
        subject = _require_auxiliary_subject_contract(
            conn,
            run_row,
        )
        if subject != request.subject:
            raise AuxiliaryGraphPersistenceError(
                "Auxiliary verification subject has drifted"
            )
        window = conn.execute(
            "SELECT turn_id, window_state, stage, state_version, "
            "current_work_run_id, current_attempt_id, latest_checkpoint_id "
            "FROM turn_execution_windows WHERE session_id=?",
            (session_id,),
        ).fetchone()
        if (
            window is None
            or str(window["turn_id"] or "") != invocation_turn_id
            or str(window["window_state"]) != "active"
            or str(window["stage"]) != "VERIFICATION"
            or str(window["current_work_run_id"] or "") != request.work_run_id
            or window["current_attempt_id"] is not None
            or str(window["latest_checkpoint_id"] or "") != verification_request_id
        ):
            raise AuxiliaryGraphPersistenceError(
                "invocation Turn does not own Auxiliary verification"
            )
        node_row, acceptances = _load_exact_auxiliary_node(
            conn,
            session_id=session_id,
            subject=subject,
            expected_status="active",
        )
        node_title = str(node_row["title"])
        node_objective = str(node_row["objective"])
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
            session_id=session_id,
            request_turn_id=request.request_turn_id,
            work_run_id=request.work_run_id,
            subject=subject,
            node_title=node_title,
            node_objective=node_objective,
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
        prepared = PreparedTaskNodeVerification(
            record=record,
            invocation_turn_id=invocation_turn_id,
            window_state_version=int(window["state_version"]),
            work_run=_work_run_from_row(run_row),
            submitted_attempt=submitted_attempt,
            acceptance_progress=progress,
            node_title=node_title,
            node_objective=node_objective,
            acceptances=acceptances,
            locked_output_window=output,
            supporting_tool_results=supporting_results,
        )
        conn.commit()
        return prepared
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def commit_auxiliary_node_verification_result(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    work_run_id: str,
    verification_request_id: str,
    result: NodeVerificationResult,
    expected_work_run_revision: int,
    expected_verification_request_revision: int,
    expected_window_revision: int,
    apply_id: str,
    active_seconds_delta: float,
    completion_id: str | None = None,
) -> TaskNodeVerificationMutationResult:
    """结算一个精确当前节点。"""

    for name, value in (
        ("session_id", session_id),
        ("turn_id", turn_id),
        ("work_run_id", work_run_id),
        ("verification_request_id", verification_request_id),
        ("apply_id", apply_id),
    ):
        _require_identifier(name, value)
    for name, value in (
        ("expected_work_run_revision", expected_work_run_revision),
        (
            "expected_verification_request_revision",
            expected_verification_request_revision,
        ),
        ("expected_window_revision", expected_window_revision),
    ):
        _require_positive(name, value)
    if completion_id is not None:
        _require_identifier("completion_id", completion_id)
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "subject_contract_version": "auxiliary_node_v2",
        "work_run_id": work_run_id,
        "verification_request_id": verification_request_id,
        "result": result.model_dump(mode="json"),
        "expected_work_run_revision": expected_work_run_revision,
        "expected_verification_request_revision": (
            expected_verification_request_revision
        ),
        "expected_window_revision": expected_window_revision,
        "active_seconds_delta": active_seconds_delta,
        "requested_completion_id": completion_id,
    }
    payload_hash = _payload_hash(payload)
    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        replay = _load_verification_replay(
            conn,
            apply_id=apply_id,
            operation="commit_verification_result",
            session_id=session_id,
            work_run_id=work_run_id,
            payload_hash=payload_hash,
        )
        if replay is not None:
            return replay
        window = _require_owned_work_run_window(
            conn,
            session_id,
            turn_id,
            work_run_id,
            expected_window_revision,
        )
        if (
            str(window["latest_checkpoint_id"] or "")
            != verification_request_id
            or str(window["stage"] or "") != "VERIFICATION"
        ):
            raise AuxiliaryGraphPersistenceError(
                "Turn Window does not own Auxiliary verification"
            )
        run_row = _require_run_row(conn, work_run_id, session_id)
        _require_run_revision(run_row, expected_work_run_revision)
        subject = _require_auxiliary_subject_contract(
            conn,
            run_row,
        )
        node, acceptances = _load_exact_auxiliary_node(
            conn,
            session_id=session_id,
            subject=subject,
            expected_status="active",
        )
        task = conn.execute(
            "SELECT current_status, state_version FROM insession_tasks "
            "WHERE session_id=? AND insession_task_id=?",
            (session_id, subject.task_id),
        ).fetchone()
        if task is None or str(task["current_status"]) != "active":
            raise AuxiliaryGraphPersistenceError(
                "Auxiliary WorkRun owner Task is no longer active"
            )
        current_task_state_version = int(task["state_version"])
        request_row = _load_auxiliary_request_row(
            conn,
            session_id=session_id,
            verification_request_id=verification_request_id,
        )
        record = _auxiliary_record_from_row(request_row)
        request = record.request
        if (
            request.revision != expected_verification_request_revision
            or request.status is not TaskNodeVerificationRequestStatus.PENDING
            or request.work_run_id != work_run_id
            or request.subject != subject
            or result.verification_request_id != verification_request_id
            or result.verification_request_revision
            != expected_verification_request_revision
            or result.work_run_id != work_run_id
            or result.locked_work_run_revision
            != request.locked_work_run_revision
            or result.subject != subject
            or result.submitted_attempt_id != request.submitted_attempt_id
            or result.output_revision != request.output_revision
            or result.acceptance_progress_revision
            != request.acceptance_progress_revision
            or tuple(item.acceptance_id for item in result.acceptance_results)
            != request.acceptance_ids
            or request.acceptance_ids
            != tuple(item.acceptance_id for item in acceptances)
        ):
            raise AuxiliaryGraphPersistenceError(
                "semantic result does not match exact Auxiliary verification"
            )
        if (
            str(run_row["status"]) != "active"
            or str(run_row["reason"] or "") != "verification_pending"
            or str(run_row["current_verification_request_id"] or "")
            != verification_request_id
        ):
            raise AuxiliaryGraphPersistenceError(
                "Auxiliary WorkRun does not own verification"
            )
        _revalidate_auxiliary_request_binding(conn, request_row)
        budget_transition = _require_settlement_budget_transition(
            conn,
            run_row=run_row,
            work_run_id=work_run_id,
            active_seconds_delta=active_seconds_delta,
        )
        next_request_revision = expected_verification_request_revision + 1
        next_run_revision = expected_work_run_revision + 1
        next_window_revision = expected_window_revision + 1
        if (
            budget_transition.disposition
            is WorkRunBudgetDisposition.HARD_LIMIT_REACHED
        ):
            if conn.execute(
                "UPDATE insession_work_run_verification_requests SET "
                "status='interrupted', request_revision=?, "
                "technical_error_code='work_run_limit_reached', "
                "result_json=NULL, all_pass=NULL, updated_at=?, completed_at=NULL "
                "WHERE verification_request_id=? AND work_run_id=? "
                "AND request_revision=? AND status='pending'",
                (
                    next_request_revision,
                    now,
                    verification_request_id,
                    work_run_id,
                    expected_verification_request_revision,
                ),
            ).rowcount != 1:
                raise AuxiliaryGraphPersistenceError(
                    "verification changed during hard-budget settlement"
                )
            _project_auxiliary_node_budget_failure(
                conn,
                run_row=run_row,
                subject=subject,
                now=now,
            )
            next_status = WorkRunStatus.FAILED
            next_reason = "work_run_limit_reached"
            next_task_state_version = current_task_state_version + 1
            all_pass: bool | None = None
            allocated_completion_id = None
        else:
            result_json = _model_json(result)
            if conn.execute(
                "UPDATE insession_work_run_verification_requests SET "
                "status='completed', request_revision=?, technical_error_code=NULL, "
                "result_json=?, all_pass=?, updated_at=?, completed_at=? "
                "WHERE verification_request_id=? AND work_run_id=? "
                "AND request_revision=? AND status='pending'",
                (
                    next_request_revision,
                    result_json,
                    int(result.all_pass),
                    now,
                    now,
                    verification_request_id,
                    work_run_id,
                    expected_verification_request_revision,
                ),
            ).rowcount != 1:
                raise AuxiliaryGraphPersistenceError(
                    "Auxiliary verification changed during settlement"
                )
            allocated_completion_id = None
            next_task_state_version = current_task_state_version
            if result.all_pass:
                output, output_hash = _load_output_window(conn, work_run_id)
                allocated_completion_id = completion_id or _allocate_id(
                    deps, "auxv2completion"
                )
                completion_payload = {
                    "schema_version": "auxiliary-node-completion-v2",
                    "completion_id": allocated_completion_id,
                    "execution_subject_id": str(run_row["execution_subject_id"]),
                    "subject": subject.model_dump(mode="json"),
                    "goal_id": str(node["goal_id"]),
                    "definition_sha256": str(node["definition_sha256"]),
                    "verification_request_id": verification_request_id,
                    "verification_result_sha256": _text_hash(result_json),
                    "submitted_attempt_id": request.submitted_attempt_id,
                    "output_revision": request.output_revision,
                    "output_snapshot_sha256": output_hash,
                }
                completion_json = _canonical_json(completion_payload)
                conn.execute(
                    "INSERT INTO insession_auxiliary_node_completions_v2 "
                    "(completion_id, session_id, insession_task_id, "
                    "auxiliary_graph_id, auxiliary_graph_revision, "
                    "auxiliary_node_id, node_revision, work_run_id, "
                    "verification_request_id, submitted_attempt_id, "
                    "output_revision, completion_json, completion_sha256, "
                    "created_turn_id, created_at) VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        allocated_completion_id,
                        session_id,
                        subject.task_id,
                        subject.auxiliary_graph_id,
                        subject.auxiliary_graph_revision,
                        subject.node_id,
                        subject.node_revision,
                        work_run_id,
                        verification_request_id,
                        request.submitted_attempt_id,
                        request.output_revision,
                        completion_json,
                        _text_hash(completion_json),
                        turn_id,
                        now,
                    ),
                )
                if conn.execute(
                    "UPDATE insession_auxiliary_node_states_v2 "
                    "SET status='completed', state_version=state_version+1, "
                    "updated_at=? WHERE auxiliary_graph_id=? "
                    "AND auxiliary_graph_revision=? AND auxiliary_node_id=? "
                    "AND node_revision=? AND status='active'",
                    (
                        now,
                        subject.auxiliary_graph_id,
                        subject.auxiliary_graph_revision,
                        subject.node_id,
                        subject.node_revision,
                    ),
                ).rowcount != 1:
                    raise AuxiliaryGraphPersistenceError(
                        "AuxiliaryNode changed during verified completion"
                    )
                if conn.execute(
                    "UPDATE insession_work_run_output_windows SET frozen_at=? "
                    "WHERE work_run_id=? AND output_revision=? "
                    "AND snapshot_hash=? AND frozen_at IS NULL",
                    (
                        now,
                        work_run_id,
                        output.output_revision,
                        output_hash,
                    ),
                ).rowcount != 1:
                    raise AuxiliaryGraphPersistenceError(
                        "verified Auxiliary OutputWindow could not be frozen"
                    )
                next_task_state_version = current_task_state_version + 1
                if conn.execute(
                    "UPDATE insession_tasks SET state_version=?, updated_at=? "
                    "WHERE session_id=? AND insession_task_id=? "
                    "AND current_status='active' AND state_version=?",
                    (
                        next_task_state_version,
                        now,
                        session_id,
                        subject.task_id,
                        current_task_state_version,
                    ),
                ).rowcount != 1:
                    raise AuxiliaryGraphPersistenceError(
                        "Task changed during Auxiliary completion"
                    )
                next_status = WorkRunStatus.COMPLETED
                next_reason = "verification_passed"
            elif (
                budget_transition.disposition
                is WorkRunBudgetDisposition.SOFT_LIMIT_REACHED
            ):
                next_status = WorkRunStatus.TURN_LIMIT_REACHED
                next_reason = "turn_limit_reached"
            else:
                next_status = WorkRunStatus.ACTIVE
                next_reason = None
            all_pass = result.all_pass

        if conn.execute(
            "UPDATE insession_work_runs SET status=?, reason=?, revision=?, "
            "active_seconds_consumed=?, current_verification_request_id=NULL, "
            "updated_turn_id=?, updated_at=? WHERE work_run_id=? AND revision=? "
            "AND status='active' AND reason='verification_pending' "
            "AND current_verification_request_id=?",
            (
                next_status.value,
                next_reason,
                next_run_revision,
                budget_transition.budget_after.active_seconds_consumed,
                turn_id,
                now,
                work_run_id,
                expected_work_run_revision,
                verification_request_id,
            ),
        ).rowcount != 1:
            raise AuxiliaryGraphPersistenceError(
                "Auxiliary WorkRun changed during verification settlement"
            )
        if conn.execute(
            "UPDATE turn_execution_windows SET current_work_run_id=?, "
            "current_attempt_id=NULL, latest_checkpoint_id=?, stage=?, "
            "state_version=?, updated_at=? WHERE session_id=? AND turn_id=? "
            "AND window_state='active' AND current_work_run_id=? "
            "AND current_attempt_id IS NULL AND state_version=?",
            (
                None if next_status is WorkRunStatus.COMPLETED else work_run_id,
                (
                    _settlement_budget_checkpoint_id(
                        "commit_verification_result", apply_id
                    )
                    if next_status is WorkRunStatus.FAILED
                    else None
                ),
                "L2_PLAN" if next_status is not WorkRunStatus.FAILED else "VERIFICATION",
                next_window_revision,
                now,
                session_id,
                turn_id,
                work_run_id,
                expected_window_revision,
            ),
        ).rowcount != 1:
            raise AuxiliaryGraphPersistenceError(
                "Turn Window changed during verification settlement"
            )
        _insert_budget_charge(
            conn,
            budget_charge_id=apply_id,
            operation="commit_verification_result",
            session_id=session_id,
            work_run_id=work_run_id,
            turn_id=turn_id,
            checkpoint_id=_settlement_budget_checkpoint_id(
                "commit_verification_result", apply_id
            ),
            work_run_revision_before=expected_work_run_revision,
            work_run_revision_after=next_run_revision,
            window_state_version_before=expected_window_revision,
            window_state_version_after=next_window_revision,
            transition=budget_transition,
            work_run_status_after=next_status,
            work_run_reason_after=next_reason,
            now=now,
        )
        mutation = TaskNodeVerificationMutationResult(
            status="applied",
            verification_request_id=verification_request_id,
            verification_request_revision=next_request_revision,
            verification_request_status=(
                TaskNodeVerificationRequestStatus.INTERRUPTED
                if next_status is WorkRunStatus.FAILED
                else TaskNodeVerificationRequestStatus.COMPLETED
            ),
            work_run_id=work_run_id,
            work_run_revision=next_run_revision,
            work_run_status=next_status,
            work_run_reason=next_reason,
            window_state_version=next_window_revision,
            task_state_version=next_task_state_version,
            node_state_version=_load_auxiliary_node_state_version(
                conn, subject
            ),
            all_pass=all_pass,
            auxiliary_completion_id=allocated_completion_id,
            budget_transition=budget_transition,
        )
        _insert_receipt(
            conn,
            apply_id=apply_id,
            operation="commit_verification_result",
            session_id=session_id,
            work_run_id=work_run_id,
            payload_hash=payload_hash,
            result=mutation,
            now=now,
        )
        return mutation


def _terminal_graph_validation_context(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    invocation_turn_id: str,
    task_id: str,
    task_authority: sqlite3.Row,
) -> InSessionTaskGraphRevisionValidationContext:
    """为此图类型重建规范终态来源权威。"""

# 保持本地导入以维持持久化模块无环导入图：来源卡验证器复用本模块严格图加载器。
    from .auxiliary_terminal_validation import _build_auxiliary_terminal_validation_context

    return _build_auxiliary_terminal_validation_context(
        conn,
        session_id=session_id,
        invocation_turn_id=invocation_turn_id,
        task_id=task_id,
        task_authority=task_authority,
    )


def _load_auxiliary_node_state_version(
    conn: sqlite3.Connection,
    subject: AuxiliaryNodeSubject,
) -> int:
    row = conn.execute(
        "SELECT state_version FROM insession_auxiliary_node_states_v2 "
        "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=? "
        "AND auxiliary_node_id=? AND node_revision=?",
        (
            subject.auxiliary_graph_id,
            subject.auxiliary_graph_revision,
            subject.node_id,
            subject.node_revision,
        ),
    ).fetchone()
    if row is None:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph node state is missing"
        )
    return int(row["state_version"])


def _load_verification_replay(
    conn: sqlite3.Connection,
    *,
    apply_id: str,
    operation: str,
    session_id: str,
    work_run_id: str,
    payload_hash: str,
) -> TaskNodeVerificationMutationResult | None:
    row = conn.execute(
        "SELECT operation, session_id, work_run_id, payload_hash, result_json "
        "FROM insession_work_run_apply_receipts WHERE apply_id=?",
        (apply_id,),
    ).fetchone()
    if row is None:
        return None
    if (
        str(row["operation"]) != operation
        or str(row["session_id"]) != session_id
        or str(row["work_run_id"]) != work_run_id
        or str(row["payload_hash"]) != payload_hash
    ):
        raise WorkExecutionApplyIdCollision(
            "Auxiliary verification apply id collision"
        )
    return TaskNodeVerificationMutationResult.model_validate_json(
        str(row["result_json"])
    ).model_copy(update={"status": "replayed"})


def _work_run_from_row(row: sqlite3.Row) -> WorkRun:
    return WorkRun(
        work_run_id=str(row["work_run_id"]),
        subject=_require_auxiliary_subject(row),
        revision=int(row["revision"]),
        status=WorkRunStatus(str(row["status"])),
        reason=(str(row["reason"]) if row["reason"] is not None else None),
        budget=_budget_from_row(row),
    )


def _load_formal_authority_snapshot_json(
    snapshot_json: str,
    *,
    expected_snapshot_sha256: str,
) -> PlanningAuthoritySnapshot:
    try:
        snapshot = PlanningAuthoritySnapshot.model_validate_json(snapshot_json)
    except (TypeError, ValueError) as exc:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph authority snapshot is corrupt"
        ) from exc
    if (
        snapshot.snapshot_sha256 != expected_snapshot_sha256
        or _model_json(snapshot) != snapshot_json
    ):
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph authority snapshot binding is corrupt"
        )
    return snapshot


def _load_formal_authority_snapshot(
    conn: sqlite3.Connection,
    *,
    authority_snapshot_id: str,
    expected_session_id: str,
    expected_task_id: str,
    expected_auxiliary_graph_id: str,
    expected_goal_id: str,
    expected_source_turn_id: str,
    expected_snapshot_sha256: str,
) -> PlanningAuthoritySnapshot:
    row = conn.execute(
        "SELECT session_id, insession_task_id, auxiliary_graph_id, goal_id, "
        "contract_version, snapshot_json, snapshot_sha256, created_turn_id "
        "FROM insession_auxiliary_authority_snapshots "
        "WHERE authority_snapshot_id=?",
        (authority_snapshot_id,),
    ).fetchone()
    if (
        row is None
        or str(row["contract_version"]) != "planning-authority-snapshot-v1"
        or str(row["session_id"]) != expected_session_id
        or str(row["insession_task_id"]) != expected_task_id
        or str(row["auxiliary_graph_id"]) != expected_auxiliary_graph_id
        or str(row["goal_id"]) != expected_goal_id
        or str(row["created_turn_id"]) != expected_source_turn_id
        or str(row["snapshot_sha256"]) != expected_snapshot_sha256
    ):
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph authority snapshot scope is corrupt"
        )
    snapshot = _load_formal_authority_snapshot_json(
        str(row["snapshot_json"]),
        expected_snapshot_sha256=expected_snapshot_sha256,
    )
    if (
        snapshot.authority_snapshot_id != authority_snapshot_id
        or snapshot.session_id != expected_session_id
        or snapshot.task_id != expected_task_id
        or snapshot.auxiliary_graph_id != expected_auxiliary_graph_id
        or snapshot.goal_id != expected_goal_id
        or snapshot.source_turn_id != expected_source_turn_id
    ):
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph authority snapshot identity is corrupt"
        )
    rows = conn.execute(
        "SELECT * FROM insession_auxiliary_authority_anchors "
        "WHERE authority_snapshot_id=? ORDER BY ordinal",
        (authority_snapshot_id,),
    ).fetchall()
    loaded: list[PlanningAuthorityAnchor] = []
    for ordinal, anchor_row in enumerate(rows):
        anchor_json = str(anchor_row["anchor_json"])
        try:
            anchor = PlanningAuthorityAnchor.model_validate_json(anchor_json)
        except (TypeError, ValueError) as exc:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph authority anchor is corrupt"
            ) from exc
        mirror = (
            int(anchor_row["ordinal"]) == ordinal
            and _model_json(anchor) == anchor_json
            and _text_hash(anchor_json) == str(anchor_row["anchor_sha256"])
            and anchor.authority_snapshot_id
            == str(anchor_row["authority_snapshot_id"])
            and anchor.anchor_id == str(anchor_row["anchor_id"])
            and anchor.projection_alias == str(anchor_row["projection_alias"])
            and anchor.authority_class.value == str(anchor_row["authority_class"])
            and anchor.origin_kind.value == str(anchor_row["origin_kind"])
            and anchor.origin_id == str(anchor_row["origin_id"])
            and anchor.source_revision == anchor_row["source_revision"]
            and anchor.content_sha256 == str(anchor_row["content_sha256"])
            and anchor.item_ordinal == int(anchor_row["item_ordinal"])
            and anchor.span_start == anchor_row["span_start"]
            and anchor.span_end == anchor_row["span_end"]
            and anchor.projection_sha256
            == str(anchor_row["projection_sha256"])
            and anchor.freshness_binding_sha256
            == str(anchor_row["freshness_binding_sha256"])
            and anchor.disclosure_receipt_id
            == anchor_row["disclosure_receipt_id"]
        )
        if not mirror:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph authority anchor mirror is corrupt"
            )
        loaded.append(anchor)
    if tuple(loaded) != snapshot.anchors:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph authority anchor manifest is corrupt"
        )
    return snapshot


def _formal_budget_from_row(
    row: sqlite3.Row,
    *,
    expected_goal_id: str,
    snapshot_json_column: str = "snapshot_json",
    snapshot_sha256_column: str = "snapshot_sha256",
    state_version_column: str = "state_version",
) -> PlanningEpisodeBudget:
    if str(row["contract_version"]) != "planning-episode-budget-v1":
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph goal does not carry a formal planning budget"
        )
    profile_json = str(row["profile_json"])
    usage_json = str(row["usage_json"])
    extensions_json = str(row["extensions_json"])
    snapshot_json = str(row[snapshot_json_column])
    if (
        _text_hash(profile_json) != str(row["profile_sha256"])
        or _text_hash(usage_json) != str(row["usage_sha256"])
        or _text_hash(extensions_json) != str(row["extensions_sha256"])
    ):
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph budget component hash is corrupt"
        )
    try:
        budget = PlanningEpisodeBudget.model_validate_json(snapshot_json)
    except (TypeError, ValueError) as exc:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph budget snapshot is corrupt"
        ) from exc
    if (
        budget.goal_id != expected_goal_id
        or budget.budget_ledger_id != str(row["budget_ledger_id"])
        or budget.state_version != int(row[state_version_column])
        or _model_json(budget.base_profile) != profile_json
        or _model_json(budget.usage) != usage_json
        or _canonical_json(
            [item.model_dump(mode="json") for item in budget.extensions]
        )
        != extensions_json
        or budget.snapshot_sha256 != str(row[snapshot_sha256_column])
        or _model_json(budget) != snapshot_json
    ):
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph budget snapshot binding is corrupt"
        )
    return budget


def _proposal_depth(proposal: AuxiliaryGraphRevisionProposalRecord) -> int:
    parents: dict[str, list[str]] = {
        node.local_node_key: [] for node in proposal.nodes
    }
    for edge in proposal.edges:
        parents[edge.consumer_node_key].append(edge.dependency_node_key)
    memo: dict[str, int] = {}

    def depth(node_key: str) -> int:
        cached = memo.get(node_key)
        if cached is not None:
            return cached
        value = 1 + max((depth(parent) for parent in parents[node_key]), default=0)
        memo[node_key] = value
        return value

    return max(depth(node.local_node_key) for node in proposal.nodes)


def _pure_model_authority_projection_sha256(
    snapshot: PlanningAuthoritySnapshot,
) -> str:
    """对权威语义计算哈希，同时排除 revision 本地快照 ID。"""

    return _payload_hash(
        {
            "schema_version": "auxiliary-pure-model-authority-projection-v1",
            "session_id": snapshot.session_id,
            "task_id": snapshot.task_id,
            "auxiliary_graph_id": snapshot.auxiliary_graph_id,
            "goal_id": snapshot.goal_id,
            "source_turn_id": snapshot.source_turn_id,
            "anchors": [
                anchor.model_dump(
                    mode="json",
                    exclude={"authority_snapshot_id"},
                )
                for anchor in snapshot.anchors
            ],
        }
    )


def _empty_pure_model_dependency_closure_sha256() -> str:
    return _payload_hash(
        {
            "schema_version": "auxiliary-pure-model-dependency-closure-v1",
            "completion_ids": [],
        }
    )


def _empty_pure_model_context_artifact_manifest_sha256() -> str:
    return _payload_hash(
        {
            "schema_version": (
                "auxiliary-pure-model-context-artifact-manifest-v1"
            ),
            "artifact_ids": [],
        }
    )


def _catalog_snapshot_has_no_tools(raw: str, expected_sha256: str) -> bool:
    if _text_hash(raw) != expected_sha256:
        raise AuxiliaryGraphPersistenceError(
            "pure model completion catalog snapshot hash is corrupt"
        )
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, RecursionError) as exc:
        raise AuxiliaryGraphPersistenceError(
            "pure model completion catalog snapshot is invalid"
        ) from exc
    if not isinstance(payload, dict) or _canonical_json(payload) != raw:
        raise AuxiliaryGraphPersistenceError(
            "pure model completion catalog snapshot is not canonical"
        )
    declarations = [payload[key] for key in ("entries", "tools") if key in payload]
    return bool(declarations) and all(
        isinstance(items, list) and not items for items in declarations
    )


def _load_pure_model_completion_source(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    task_id: str,
    auxiliary_graph_id: str,
    goal_id: str,
    source_subject: AuxiliaryNodeSubject,
    expected_turn_id: str,
    expected_definition_sha256: str,
    expected_completion_id: str | None = None,
) -> _PureModelCompletionSource | None:
    """认证刻意极小的可结转完成项类别。

    执行器、能力或历史不匹配属于普通不合格，返回 ``None``。一旦某行声称是合格纯完成项，
    不可变权威格式错误会抛出异常，而非静默重跑。
    """

    definition = conn.execute(
        "SELECT definition.node_kind, definition.executor_kind, "
        "definition.capability_profile_id, "
        "definition.input_resource_aliases_json, "
        "definition.semantic_fingerprint, definition.definition_sha256, "
        "membership.carried_completion_id, state.status "
        "FROM insession_auxiliary_graph_revision_nodes_v2 AS membership "
        "JOIN insession_auxiliary_node_definitions_v2 AS definition "
        "ON definition.auxiliary_graph_id=membership.auxiliary_graph_id "
        "AND definition.auxiliary_node_id=membership.auxiliary_node_id "
        "AND definition.node_revision=membership.node_revision "
        "JOIN insession_auxiliary_node_states_v2 AS state "
        "ON state.auxiliary_graph_id=membership.auxiliary_graph_id "
        "AND state.auxiliary_graph_revision="
        "membership.auxiliary_graph_revision "
        "AND state.auxiliary_node_id=membership.auxiliary_node_id "
        "AND state.node_revision=membership.node_revision "
        "WHERE membership.auxiliary_graph_id=? "
        "AND membership.auxiliary_graph_revision=? "
        "AND membership.auxiliary_node_id=? "
        "AND membership.node_revision=?",
        (
            auxiliary_graph_id,
            source_subject.auxiliary_graph_revision,
            source_subject.node_id,
            source_subject.node_revision,
        ),
    ).fetchone()
    if definition is None:
        raise AuxiliaryGraphPersistenceError(
            "pure model carry source definition is missing"
        )
    try:
        input_aliases = tuple(
            json.loads(str(definition["input_resource_aliases_json"]))
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise AuxiliaryGraphPersistenceError(
            "pure model carry source input manifest is corrupt"
        ) from exc
    if (
        str(definition["node_kind"]) != "analyze"
        or str(definition["executor_kind"]) != "model_work_run"
        or str(definition["capability_profile_id"] or "") != "model_analysis"
        or input_aliases
        or definition["carried_completion_id"] is not None
        or str(definition["status"]) != "completed"
        or str(definition["semantic_fingerprint"])
        != expected_definition_sha256
        or str(definition["definition_sha256"])
        != expected_definition_sha256
    ):
        return None
    if conn.execute(
        "SELECT 1 FROM insession_auxiliary_graph_edges WHERE "
        "auxiliary_graph_id=? AND auxiliary_graph_revision=? "
        "AND consumer_auxiliary_node_id=? AND consumer_node_revision=? LIMIT 1",
        (
            auxiliary_graph_id,
            source_subject.auxiliary_graph_revision,
            source_subject.node_id,
            source_subject.node_revision,
        ),
    ).fetchone() is not None:
        return None

    rows = conn.execute(
        "SELECT completion.*, run.execution_subject_id, "
        "run.status AS work_run_status, run.reason AS work_run_reason, "
        "run.current_attempt_id, run.current_verification_request_id, "
        "run.attempts_started, run.created_turn_id AS run_created_turn_id, "
        "run.updated_turn_id AS run_updated_turn_id, "
        "subject.subject_contract_version, binding.goal_id AS bound_goal_id, "
        "binding.executor_kind AS bound_executor_kind, "
        "binding.definition_sha256 AS bound_definition_sha256, "
        "verification.status AS verification_status, "
        "verification.result_json, verification.all_pass, "
        "verification.supporting_tool_result_ids_json, "
        "verification.dependency_delivery_ids_json, "
        "verification.request_turn_id, "
        "output.snapshot_json AS output_json, "
        "output.snapshot_hash AS output_snapshot_sha256, output.frozen_at, "
        "output.updated_turn_id AS output_updated_turn_id, "
        "attempt.status AS attempt_status, attempt.action AS attempt_action, "
        "attempt.turn_id AS attempt_turn_id, attempt.input_checkpoint_id, "
        "attempt.catalog_snapshot_json, attempt.catalog_snapshot_hash, "
        "(SELECT COUNT(*) FROM insession_work_run_tool_calls AS tool_call "
        " WHERE tool_call.work_run_id=completion.work_run_id) AS tool_call_count, "
        "(SELECT COUNT(*) FROM insession_work_run_tool_results AS tool_result "
        " WHERE tool_result.work_run_id=completion.work_run_id) AS tool_result_count, "
        "(SELECT COUNT(*) FROM "
        " insession_auxiliary_planning_context_artifacts AS artifact "
        " WHERE artifact.producer_work_run_id=completion.work_run_id) "
        "AS context_artifact_count "
        "FROM insession_auxiliary_node_completions_v2 AS completion "
        "JOIN insession_work_runs AS run "
        "ON run.work_run_id=completion.work_run_id "
        "JOIN insession_execution_subjects AS subject "
        "ON subject.execution_subject_id=run.execution_subject_id "
        "JOIN insession_auxiliary_v2_node_execution_subject_bindings AS binding "
        "ON binding.binding_id=subject.auxiliary_v2_binding_id "
        "JOIN insession_work_run_verification_requests AS verification "
        "ON verification.work_run_id=completion.work_run_id "
        "AND verification.verification_request_id="
        "completion.verification_request_id "
        "JOIN insession_work_run_output_windows AS output "
        "ON output.work_run_id=completion.work_run_id "
        "AND output.output_revision=completion.output_revision "
        "JOIN insession_work_run_attempts AS attempt "
        "ON attempt.work_run_id=completion.work_run_id "
        "AND attempt.attempt_id=completion.submitted_attempt_id "
        "WHERE completion.session_id=? "
        "AND completion.insession_task_id=? "
        "AND completion.auxiliary_graph_id=? "
        "AND completion.auxiliary_graph_revision=? "
        "AND completion.auxiliary_node_id=? "
        "AND completion.node_revision=?",
        (
            session_id,
            task_id,
            auxiliary_graph_id,
            source_subject.auxiliary_graph_revision,
            source_subject.node_id,
            source_subject.node_revision,
        ),
    ).fetchall()
    if len(rows) != 1:
        raise AuxiliaryGraphPersistenceError(
            "pure model carry source has no unique local completion"
        )
    row = rows[0]
    if expected_completion_id is not None and (
        str(row["completion_id"]) != expected_completion_id
    ):
        raise AuxiliaryGraphPersistenceError(
            "pure model carry source completion identity changed"
        )
    try:
        output_json = str(row["output_json"])
        output = OutputWindow.model_validate_json(output_json)
        result_json = str(row["result_json"])
        verification = NodeVerificationResult.model_validate_json(result_json)
        supporting_ids = json.loads(
            str(row["supporting_tool_result_ids_json"])
        )
        dependency_delivery_ids = json.loads(
            str(row["dependency_delivery_ids_json"])
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise AuxiliaryGraphPersistenceError(
            "pure model carry completion payload is invalid"
        ) from exc
    completion_payload = {
        "schema_version": "auxiliary-node-completion-v2",
        "completion_id": str(row["completion_id"]),
        "execution_subject_id": str(row["execution_subject_id"]),
        "subject": source_subject.model_dump(mode="json"),
        "goal_id": goal_id,
        "definition_sha256": expected_definition_sha256,
        "verification_request_id": str(row["verification_request_id"]),
        "verification_result_sha256": _text_hash(result_json),
        "submitted_attempt_id": str(row["submitted_attempt_id"]),
        "output_revision": int(row["output_revision"]),
        "output_snapshot_sha256": str(row["output_snapshot_sha256"]),
    }
    completion_json = str(row["completion_json"])
    canonical_catalog = _catalog_snapshot_has_no_tools(
        str(row["catalog_snapshot_json"]),
        str(row["catalog_snapshot_hash"]),
    )
    if (
        not canonical_catalog
        or supporting_ids != []
        or dependency_delivery_ids != []
        or int(row["tool_call_count"]) != 0
        or int(row["tool_result_count"]) != 0
        or int(row["context_artifact_count"]) != 0
        or int(row["attempts_started"]) != 1
        or row["input_checkpoint_id"] is not None
    ):
        return None
    valid = (
        str(row["subject_contract_version"]) == "auxiliary_node_v2"
        and str(row["bound_goal_id"]) == goal_id
        and str(row["bound_executor_kind"]) == "model_work_run"
        and str(row["bound_definition_sha256"])
        == expected_definition_sha256
        and str(row["work_run_status"]) == "completed"
        and str(row["work_run_reason"]) == "verification_passed"
        and row["current_attempt_id"] is None
        and row["current_verification_request_id"] is None
        and str(row["attempt_status"]) == "closed"
        and str(row["attempt_action"]) == "submit_output_window"
        and str(row["verification_status"]) == "completed"
        and int(row["all_pass"] or 0) == 1
        and str(row["created_turn_id"]) == expected_turn_id
        and str(row["run_created_turn_id"]) == expected_turn_id
        and str(row["run_updated_turn_id"]) == expected_turn_id
        and str(row["request_turn_id"]) == expected_turn_id
        and str(row["attempt_turn_id"]) == expected_turn_id
        and str(row["output_updated_turn_id"]) == expected_turn_id
        and row["frozen_at"] is not None
        and output.work_run_id == str(row["work_run_id"])
        and output.output_revision == int(row["output_revision"])
        and output.updated_attempt_id == str(row["submitted_attempt_id"])
        and bool(output.content.strip())
        and "\x00" not in output.content
        and _model_json(output) == output_json
        and _text_hash(output_json) == str(row["output_snapshot_sha256"])
        and verification.all_pass
        and verification.subject == source_subject
        and verification.work_run_id == str(row["work_run_id"])
        and verification.verification_request_id
        == str(row["verification_request_id"])
        and verification.submitted_attempt_id
        == str(row["submitted_attempt_id"])
        and verification.output_revision == int(row["output_revision"])
        and _model_json(verification) == result_json
        and _canonical_json(completion_payload) == completion_json
        and _text_hash(completion_json) == str(row["completion_sha256"])
    )
    if not valid:
        raise AuxiliaryGraphPersistenceError(
            "pure model carry completion binding is corrupt or impure"
        )
    return _PureModelCompletionSource(
        subject=source_subject,
        completion_id=str(row["completion_id"]),
        completion_sha256=str(row["completion_sha256"]),
        output_snapshot_sha256=str(row["output_snapshot_sha256"]),
        catalog_snapshot_id=(
            "workrun-catalog:" + str(row["submitted_attempt_id"])
        ),
        catalog_snapshot_sha256=str(row["catalog_snapshot_hash"]),
    )


def _derive_pure_model_completion_carries(
    conn: sqlite3.Connection,
    *,
    apply_id: str,
    session_id: str,
    turn_id: str,
    task_id: str,
    auxiliary_graph_id: str,
    goal_id: str,
    source_auxiliary_graph_revision: int | None,
    target_revision: AuxiliaryGraphRevision,
    target_authority_snapshot: PlanningAuthoritySnapshot,
    now: str,
) -> tuple[AuxiliaryNodePureModelCompletionCarryReceipt, ...]:
    """只结转无依赖、同 Turn、无工具的模型分析。"""

    if (
        source_auxiliary_graph_revision is None
        or target_revision.revision_reason
        is not AuxiliaryGraphRevisionReason.VERIFICATION_FAILED
        or target_revision.parent_auxiliary_graph_revision
        != source_auxiliary_graph_revision
        or target_revision.auxiliary_graph_revision
        != source_auxiliary_graph_revision + 1
        or target_revision.source_turn_id != turn_id
    ):
        return ()
    source_revision = conn.execute(
        "SELECT goal_id, source_turn_id, authority_snapshot_id, "
        "authority_snapshot_sha256 FROM "
        "insession_auxiliary_graph_revision_snapshots WHERE "
        "auxiliary_graph_id=? AND auxiliary_graph_revision=?",
        (auxiliary_graph_id, source_auxiliary_graph_revision),
    ).fetchone()
    if source_revision is None:
        raise AuxiliaryGraphPersistenceError(
            "pure model carry source revision is missing"
        )
# Attempt Prompt 包含 Turn 作用域用户输入，因此由不可变 revision 行而非调用方断言强制
# 执行同 Turn 边界。
    if (
        str(source_revision["goal_id"]) != goal_id
        or str(source_revision["source_turn_id"]) != turn_id
        or target_authority_snapshot.source_turn_id != turn_id
    ):
        return ()
    source_authority = _load_formal_authority_snapshot(
        conn,
        authority_snapshot_id=str(source_revision["authority_snapshot_id"]),
        expected_session_id=session_id,
        expected_task_id=task_id,
        expected_auxiliary_graph_id=auxiliary_graph_id,
        expected_goal_id=goal_id,
        expected_source_turn_id=turn_id,
        expected_snapshot_sha256=str(
            source_revision["authority_snapshot_sha256"]
        ),
    )
    source_authority_projection = _pure_model_authority_projection_sha256(
        source_authority
    )
    if source_authority_projection != _pure_model_authority_projection_sha256(
        target_authority_snapshot
    ):
        return ()

    carried: list[AuxiliaryNodePureModelCompletionCarryReceipt] = []
    for target in target_revision.nodes:
        origin = target.origin_node_ref
        if (
            target.node_kind is not AuxiliaryNodeKind.ANALYZE
            or target.executor_kind
            is not AuxiliaryNodeExecutorKind.MODEL_WORK_RUN
            or target.capability_profile_id != "model_analysis"
            or target.input_resource_aliases
            or origin is None
            or origin.node_id != target.node_id
            or origin.node_revision != target.node_revision - 1
        ):
            continue
        if conn.execute(
            "SELECT 1 FROM insession_auxiliary_graph_edges WHERE "
            "auxiliary_graph_id=? AND auxiliary_graph_revision=? "
            "AND consumer_auxiliary_node_id=? AND consumer_node_revision=? LIMIT 1",
            (
                auxiliary_graph_id,
                target_revision.auxiliary_graph_revision,
                target.node_id,
                target.node_revision,
            ),
        ).fetchone() is not None:
            continue
        source_subject = AuxiliaryNodeSubject(
            task_id=task_id,
            auxiliary_graph_id=auxiliary_graph_id,
            auxiliary_graph_revision=source_auxiliary_graph_revision,
            node_id=origin.node_id,
            node_revision=origin.node_revision,
        )
        source = _load_pure_model_completion_source(
            conn,
            session_id=session_id,
            task_id=task_id,
            auxiliary_graph_id=auxiliary_graph_id,
            goal_id=goal_id,
            source_subject=source_subject,
            expected_turn_id=turn_id,
            expected_definition_sha256=target.semantic_fingerprint,
        )
        if source is None:
            continue
        target_subject = AuxiliaryNodeSubject(
            task_id=task_id,
            auxiliary_graph_id=auxiliary_graph_id,
            auxiliary_graph_revision=target_revision.auxiliary_graph_revision,
            node_id=target.node_id,
            node_revision=target.node_revision,
        )
        dependency_sha256 = _empty_pure_model_dependency_closure_sha256()
        artifact_sha256 = (
            _empty_pure_model_context_artifact_manifest_sha256()
        )
        freshness_sha256 = _payload_hash(
            {
                "schema_version": "auxiliary-pure-model-carry-freshness-v1",
                "source_completion_id": source.completion_id,
                "source_completion_sha256": source.completion_sha256,
                "output_snapshot_sha256": source.output_snapshot_sha256,
                "definition_sha256": target.semantic_fingerprint,
                "dependency_closure_sha256": dependency_sha256,
                "context_artifact_manifest_sha256": artifact_sha256,
                "authority_projection_sha256": source_authority_projection,
                "capability_catalog_snapshot_sha256": (
                    source.catalog_snapshot_sha256
                ),
            }
        )
        carry_receipt_id = "auxcarry_" + _payload_hash(
            {
                "schema_version": "auxiliary-pure-model-carry-id-v1",
                "apply_id": apply_id,
                "source_completion_id": source.completion_id,
                "target_subject": target_subject.model_dump(mode="json"),
                "freshness_manifest_sha256": freshness_sha256,
            }
        )
        receipt = AuxiliaryNodePureModelCompletionCarryReceipt(
            carry_receipt_id=carry_receipt_id,
            apply_id=apply_id,
            session_id=session_id,
            task_id=task_id,
            auxiliary_graph_id=auxiliary_graph_id,
            goal_id=goal_id,
            source_subject=source.subject,
            source_completion_id=source.completion_id,
            source_completion_sha256=source.completion_sha256,
            target_subject=target_subject,
            definition_sha256=target.semantic_fingerprint,
            dependency_closure_sha256=dependency_sha256,
            context_artifact_manifest_sha256=artifact_sha256,
            source_authority_projection_sha256=source_authority_projection,
            authority_snapshot_id=target_authority_snapshot.authority_snapshot_id,
            authority_snapshot_sha256=target_authority_snapshot.snapshot_sha256,
            capability_catalog_snapshot_id=source.catalog_snapshot_id,
            capability_catalog_snapshot_sha256=source.catalog_snapshot_sha256,
            freshness_manifest_sha256=freshness_sha256,
            created_turn_id=turn_id,
        )
        receipt_json = _model_json(receipt)
        conn.execute(
            "INSERT INTO insession_auxiliary_node_completion_carries_v2 "
            "(carry_receipt_id, session_id, insession_task_id, "
            "auxiliary_graph_id, goal_id, "
            "source_auxiliary_graph_revision, source_auxiliary_node_id, "
            "source_node_revision, source_completion_id, "
            "target_auxiliary_graph_revision, target_auxiliary_node_id, "
            "target_node_revision, definition_sha256, "
            "dependency_closure_sha256, "
            "context_artifact_manifest_sha256, authority_snapshot_id, "
            "authority_snapshot_sha256, capability_catalog_snapshot_id, "
            "capability_catalog_snapshot_sha256, freshness_manifest_sha256, "
            "receipt_json, receipt_sha256, created_turn_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?, ?)",
            (
                receipt.carry_receipt_id,
                session_id,
                task_id,
                auxiliary_graph_id,
                goal_id,
                source_subject.auxiliary_graph_revision,
                source_subject.node_id,
                source_subject.node_revision,
                source.completion_id,
                target_subject.auxiliary_graph_revision,
                target_subject.node_id,
                target_subject.node_revision,
                target.semantic_fingerprint,
                dependency_sha256,
                artifact_sha256,
                target_authority_snapshot.authority_snapshot_id,
                target_authority_snapshot.snapshot_sha256,
                source.catalog_snapshot_id,
                source.catalog_snapshot_sha256,
                freshness_sha256,
                receipt_json,
                _text_hash(receipt_json),
                turn_id,
                now,
            ),
        )
        if conn.execute(
            "UPDATE insession_auxiliary_graph_revision_nodes_v2 SET "
            "carried_completion_id=? WHERE auxiliary_graph_id=? "
            "AND auxiliary_graph_revision=? AND auxiliary_node_id=? "
            "AND node_revision=? AND carried_completion_id IS NULL",
            (
                receipt.carry_receipt_id,
                auxiliary_graph_id,
                target_subject.auxiliary_graph_revision,
                target_subject.node_id,
                target_subject.node_revision,
            ),
        ).rowcount != 1:
            raise AuxiliaryGraphPersistenceError(
                "pure model carry target membership changed"
            )
        if conn.execute(
            "UPDATE insession_auxiliary_node_states_v2 SET "
            "status='completed', state_version=state_version+1, "
            "updated_at=? WHERE auxiliary_graph_id=? "
            "AND auxiliary_graph_revision=? AND auxiliary_node_id=? "
            "AND node_revision=? AND status='proposed' AND state_version=1",
            (
                now,
                auxiliary_graph_id,
                target_subject.auxiliary_graph_revision,
                target_subject.node_id,
                target_subject.node_revision,
            ),
        ).rowcount != 1:
            raise AuxiliaryGraphPersistenceError(
                "pure model carry target node state changed"
            )
        carried.append(receipt)
    return tuple(carried)


def _reauthenticate_pure_model_carry(
    conn: sqlite3.Connection,
    receipt: AuxiliaryNodePureModelCompletionCarryReceipt,
) -> _PureModelCompletionSource:
    """为当前读取或重放读取重新检查不可变来源与目标权威。"""

    revisions = conn.execute(
        "SELECT auxiliary_graph_revision, parent_auxiliary_graph_revision, "
        "goal_id, source_turn_id, reason, authority_snapshot_id, "
        "authority_snapshot_sha256 FROM "
        "insession_auxiliary_graph_revision_snapshots WHERE "
        "auxiliary_graph_id=? AND auxiliary_graph_revision IN (?, ?) "
        "ORDER BY auxiliary_graph_revision",
        (
            receipt.auxiliary_graph_id,
            receipt.source_subject.auxiliary_graph_revision,
            receipt.target_subject.auxiliary_graph_revision,
        ),
    ).fetchall()
    if len(revisions) != 2:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph pure model carry revision authority is missing"
        )
    source_revision, target_revision = revisions
    revision_scope_valid = (
        int(source_revision["auxiliary_graph_revision"])
        == receipt.source_subject.auxiliary_graph_revision
        and int(target_revision["auxiliary_graph_revision"])
        == receipt.target_subject.auxiliary_graph_revision
        and receipt.target_subject.auxiliary_graph_revision
        == receipt.source_subject.auxiliary_graph_revision + 1
        and target_revision["parent_auxiliary_graph_revision"] is not None
        and int(target_revision["parent_auxiliary_graph_revision"])
        == receipt.source_subject.auxiliary_graph_revision
        and str(source_revision["goal_id"]) == receipt.goal_id
        and str(target_revision["goal_id"]) == receipt.goal_id
        and str(source_revision["source_turn_id"]) == receipt.created_turn_id
        and str(target_revision["source_turn_id"]) == receipt.created_turn_id
        and str(target_revision["reason"]) == "verification_failed"
        and str(target_revision["authority_snapshot_id"])
        == receipt.authority_snapshot_id
        and str(target_revision["authority_snapshot_sha256"])
        == receipt.authority_snapshot_sha256
    )
    if not revision_scope_valid:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph pure model carry crossed its revision or Turn boundary"
        )
    target = conn.execute(
        "SELECT definition.node_kind, definition.executor_kind, "
        "definition.capability_profile_id, "
        "definition.input_resource_aliases_json, "
        "definition.semantic_fingerprint, definition.definition_sha256, "
        "definition.origin_auxiliary_node_id, definition.origin_node_revision, "
        "membership.carried_completion_id, state.status FROM "
        "insession_auxiliary_graph_revision_nodes_v2 AS membership "
        "JOIN insession_auxiliary_node_definitions_v2 AS definition "
        "ON definition.auxiliary_graph_id=membership.auxiliary_graph_id "
        "AND definition.auxiliary_node_id=membership.auxiliary_node_id "
        "AND definition.node_revision=membership.node_revision "
        "JOIN insession_auxiliary_node_states_v2 AS state "
        "ON state.auxiliary_graph_id=membership.auxiliary_graph_id "
        "AND state.auxiliary_graph_revision="
        "membership.auxiliary_graph_revision "
        "AND state.auxiliary_node_id=membership.auxiliary_node_id "
        "AND state.node_revision=membership.node_revision WHERE "
        "membership.auxiliary_graph_id=? "
        "AND membership.auxiliary_graph_revision=? "
        "AND membership.auxiliary_node_id=? AND membership.node_revision=?",
        (
            receipt.auxiliary_graph_id,
            receipt.target_subject.auxiliary_graph_revision,
            receipt.target_subject.node_id,
            receipt.target_subject.node_revision,
        ),
    ).fetchone()
    try:
        target_input_aliases = (
            tuple(json.loads(str(target["input_resource_aliases_json"])))
            if target is not None
            else ()
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph pure model carry target input manifest is corrupt"
        ) from exc
    if (
        target is None
        or str(target["node_kind"]) != "analyze"
        or str(target["executor_kind"]) != "model_work_run"
        or str(target["capability_profile_id"] or "") != "model_analysis"
        or target_input_aliases
        or str(target["semantic_fingerprint"]) != receipt.definition_sha256
        or str(target["definition_sha256"]) != receipt.definition_sha256
        or str(target["origin_auxiliary_node_id"])
        != receipt.source_subject.node_id
        or int(target["origin_node_revision"] or 0)
        != receipt.source_subject.node_revision
        or str(target["carried_completion_id"] or "")
        != receipt.carry_receipt_id
        or str(target["status"]) != "completed"
        or conn.execute(
            "SELECT 1 FROM insession_auxiliary_graph_edges WHERE "
            "auxiliary_graph_id=? AND auxiliary_graph_revision=? "
            "AND consumer_auxiliary_node_id=? "
            "AND consumer_node_revision=? LIMIT 1",
            (
                receipt.auxiliary_graph_id,
                receipt.target_subject.auxiliary_graph_revision,
                receipt.target_subject.node_id,
                receipt.target_subject.node_revision,
            ),
        ).fetchone()
        is not None
    ):
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph pure model carry target authority changed"
        )
    source_authority = _load_formal_authority_snapshot(
        conn,
        authority_snapshot_id=str(source_revision["authority_snapshot_id"]),
        expected_session_id=receipt.session_id,
        expected_task_id=receipt.task_id,
        expected_auxiliary_graph_id=receipt.auxiliary_graph_id,
        expected_goal_id=receipt.goal_id,
        expected_source_turn_id=receipt.created_turn_id,
        expected_snapshot_sha256=str(
            source_revision["authority_snapshot_sha256"]
        ),
    )
    target_authority = _load_formal_authority_snapshot(
        conn,
        authority_snapshot_id=receipt.authority_snapshot_id,
        expected_session_id=receipt.session_id,
        expected_task_id=receipt.task_id,
        expected_auxiliary_graph_id=receipt.auxiliary_graph_id,
        expected_goal_id=receipt.goal_id,
        expected_source_turn_id=receipt.created_turn_id,
        expected_snapshot_sha256=receipt.authority_snapshot_sha256,
    )
    source_projection = _pure_model_authority_projection_sha256(
        source_authority
    )
    if (
        source_projection != receipt.source_authority_projection_sha256
        or source_projection
        != _pure_model_authority_projection_sha256(target_authority)
    ):
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph pure model carry authority changed"
        )
    source = _load_pure_model_completion_source(
        conn,
        session_id=receipt.session_id,
        task_id=receipt.task_id,
        auxiliary_graph_id=receipt.auxiliary_graph_id,
        goal_id=receipt.goal_id,
        source_subject=receipt.source_subject,
        expected_turn_id=receipt.created_turn_id,
        expected_definition_sha256=receipt.definition_sha256,
        expected_completion_id=receipt.source_completion_id,
    )
    if source is None:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph pure model carry source became ineligible"
        )
    expected_freshness = _payload_hash(
        {
            "schema_version": "auxiliary-pure-model-carry-freshness-v1",
            "source_completion_id": source.completion_id,
            "source_completion_sha256": source.completion_sha256,
            "output_snapshot_sha256": source.output_snapshot_sha256,
            "definition_sha256": receipt.definition_sha256,
            "dependency_closure_sha256": receipt.dependency_closure_sha256,
            "context_artifact_manifest_sha256": (
                receipt.context_artifact_manifest_sha256
            ),
            "authority_projection_sha256": source_projection,
            "capability_catalog_snapshot_sha256": (
                source.catalog_snapshot_sha256
            ),
        }
    )
    if (
        receipt.source_completion_sha256 != source.completion_sha256
        or receipt.capability_catalog_snapshot_id != source.catalog_snapshot_id
        or receipt.capability_catalog_snapshot_sha256
        != source.catalog_snapshot_sha256
        or receipt.freshness_manifest_sha256 != expected_freshness
    ):
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph pure model carry freshness binding is corrupt"
        )
    return source


def _load_pure_model_carry_receipt(
    conn: sqlite3.Connection,
    *,
    details: StoredAuxiliaryGraphDetails,
    target_node: StoredAuxiliaryGraphNode,
    carry_receipt_id: str,
) -> AuxiliaryNodePureModelCompletionCarryReceipt:
    """重新认证一个当前结转完成项及其来源正文。"""

    row = conn.execute(
        "SELECT * FROM insession_auxiliary_node_completion_carries_v2 "
        "WHERE auxiliary_graph_id=? AND target_auxiliary_graph_revision=? "
        "AND target_auxiliary_node_id=? AND target_node_revision=? "
        "AND carry_receipt_id=?",
        (
            details.auxiliary_graph_id,
            details.auxiliary_graph_revision,
            target_node.auxiliary_node_id,
            target_node.node_revision,
            carry_receipt_id,
        ),
    ).fetchone()
    if row is None:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph carried completion receipt is missing"
        )
    receipt_json = str(row["receipt_json"])
    try:
        receipt = AuxiliaryNodePureModelCompletionCarryReceipt.model_validate_json(
            receipt_json
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph pure model carry receipt is invalid"
        ) from exc
    target_subject = AuxiliaryNodeSubject(
        task_id=details.task_id,
        auxiliary_graph_id=details.auxiliary_graph_id,
        auxiliary_graph_revision=details.auxiliary_graph_revision,
        node_id=target_node.auxiliary_node_id,
        node_revision=target_node.node_revision,
    )
    mirror = (
        _model_json(receipt) == receipt_json
        and _text_hash(receipt_json) == str(row["receipt_sha256"])
        and receipt.carry_receipt_id == str(row["carry_receipt_id"])
        and receipt.session_id == str(row["session_id"])
        and receipt.task_id == str(row["insession_task_id"])
        and receipt.auxiliary_graph_id == str(row["auxiliary_graph_id"])
        and receipt.goal_id == str(row["goal_id"])
        and receipt.source_subject.auxiliary_graph_revision
        == int(row["source_auxiliary_graph_revision"])
        and receipt.source_subject.node_id
        == str(row["source_auxiliary_node_id"])
        and receipt.source_subject.node_revision
        == int(row["source_node_revision"])
        and receipt.source_completion_id == str(row["source_completion_id"])
        and receipt.target_subject == target_subject
        and receipt.definition_sha256 == str(row["definition_sha256"])
        and receipt.dependency_closure_sha256
        == str(row["dependency_closure_sha256"])
        and receipt.context_artifact_manifest_sha256
        == str(row["context_artifact_manifest_sha256"])
        and receipt.authority_snapshot_id
        == str(row["authority_snapshot_id"])
        and receipt.authority_snapshot_sha256
        == str(row["authority_snapshot_sha256"])
        and receipt.capability_catalog_snapshot_id
        == str(row["capability_catalog_snapshot_id"])
        and receipt.capability_catalog_snapshot_sha256
        == str(row["capability_catalog_snapshot_sha256"])
        and receipt.freshness_manifest_sha256
        == str(row["freshness_manifest_sha256"])
        and receipt.created_turn_id == str(row["created_turn_id"])
        and receipt.session_id == details.session_id
        and receipt.task_id == details.task_id
        and receipt.auxiliary_graph_id == details.auxiliary_graph_id
        and receipt.goal_id == details.goal_id
        and receipt.target_subject == target_subject
        and receipt.authority_snapshot_id == details.authority_snapshot_id
        and receipt.authority_snapshot_sha256
        == details.authority_snapshot_sha256
        and receipt.created_turn_id == details.source_turn_id
        and details.reason == "verification_failed"
        and details.parent_auxiliary_graph_revision
        == receipt.source_subject.auxiliary_graph_revision
        and receipt.definition_sha256 == target_node.semantic_fingerprint
        and target_node.node_kind == "analyze"
        and target_node.executor_kind == "model_work_run"
        and target_node.capability_profile_id == "model_analysis"
        and target_node.input_resource_aliases == ()
    )
    if not mirror:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph pure model carry receipt mirror is corrupt"
        )
    apply_row = conn.execute(
        "SELECT session_id, insession_task_id, auxiliary_graph_id, goal_id, "
        "invocation_turn_id, committed_auxiliary_graph_revision "
        "FROM insession_auxiliary_graph_revision_apply_receipts_v2 "
        "WHERE apply_id=?",
        (receipt.apply_id,),
    ).fetchone()
    if (
        apply_row is None
        or str(apply_row["session_id"]) != details.session_id
        or str(apply_row["insession_task_id"]) != details.task_id
        or str(apply_row["auxiliary_graph_id"]) != details.auxiliary_graph_id
        or str(apply_row["goal_id"]) != details.goal_id
        or str(apply_row["invocation_turn_id"]) != details.source_turn_id
        or int(apply_row["committed_auxiliary_graph_revision"])
        != details.auxiliary_graph_revision
    ):
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph pure model carry apply authority is corrupt"
        )
    _reauthenticate_pure_model_carry(conn, receipt)
    return receipt


def _build_authority_snapshot(
    *,
    authority_snapshot_id: str,
    session_id: str,
    task_id: str,
    auxiliary_graph_id: str,
    goal_id: str,
    source_turn_id: str,
    authority_context: dict[str, Any],
    task_creation_turn_id: str,
    task_creation_start: int,
    task_creation_end: int,
    task_creation_sha256: str,
    task_creation_excerpt: str,
) -> PlanningAuthoritySnapshot:
    """为一个 revision 物化精确正式权威快照。"""

    raw_anchors = authority_context.get("anchors", [])
    if not isinstance(raw_anchors, list):
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph authority_context.anchors must be a list"
        )
    task_binding = {
        "source_turn_id": task_creation_turn_id,
        "start": task_creation_start,
        "end": task_creation_end,
        "sha256": task_creation_sha256,
    }
    normalized: list[PlanningAuthorityAnchor] = [
        PlanningAuthorityAnchor(
            anchor_id="auxanchor_task_" + task_creation_sha256[:32],
            authority_snapshot_id=authority_snapshot_id,
            projection_alias="task_creation_source",
            authority_class=PlanningAuthorityClass.AUTHORIZATION,
            origin_kind=PlanningAuthorityOriginKind.USER_INSTRUCTION_SPAN,
            origin_id=task_creation_turn_id,
            source_revision=None,
            content_sha256=task_creation_sha256,
            item_ordinal=0,
            span_start=task_creation_start,
            span_end=task_creation_end,
            projection_sha256=_text_hash(task_creation_excerpt),
            freshness_binding_sha256=_payload_hash(task_binding),
            disclosure_receipt_id=None,
        )
    ]
    for raw in raw_anchors:
        if not isinstance(raw, dict):
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph authority anchor must be an object"
            )
        try:
            anchor = PlanningAuthorityAnchor.model_validate(
                {
                    **raw,
                    "authority_snapshot_id": authority_snapshot_id,
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph authority anchor is not a formal authority anchor"
            ) from exc
        normalized.append(anchor)
    normalized.sort(key=lambda item: item.projection_alias)
    try:
        return PlanningAuthoritySnapshot.create(
            authority_snapshot_id=authority_snapshot_id,
            session_id=session_id,
            task_id=task_id,
            auxiliary_graph_id=auxiliary_graph_id,
            goal_id=goal_id,
            source_turn_id=source_turn_id,
            anchors=tuple(normalized),
        )
    except (TypeError, ValueError) as exc:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph authority snapshot is invalid"
        ) from exc


def _insert_authority_anchors(
    conn: sqlite3.Connection,
    *,
    authority_snapshot_id: str,
    anchors: tuple[PlanningAuthorityAnchor, ...],
) -> None:
    for ordinal, anchor in enumerate(anchors):
        anchor_json = _model_json(anchor)
        conn.execute(
            "INSERT INTO insession_auxiliary_authority_anchors "
            "(authority_snapshot_id, anchor_id, ordinal, projection_alias, "
            "authority_class, origin_kind, origin_id, source_revision, "
            "content_sha256, item_ordinal, span_start, span_end, "
            "projection_sha256, freshness_binding_sha256, "
            "disclosure_receipt_id, anchor_json, anchor_sha256) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                authority_snapshot_id,
                anchor.anchor_id,
                ordinal,
                anchor.projection_alias,
                anchor.authority_class.value,
                anchor.origin_kind.value,
                anchor.origin_id,
                anchor.source_revision,
                anchor.content_sha256,
                anchor.item_ordinal,
                anchor.span_start,
                anchor.span_end,
                anchor.projection_sha256,
                anchor.freshness_binding_sha256,
                anchor.disclosure_receipt_id,
                anchor_json,
                _text_hash(anchor_json),
            ),
        )


def _validate_revision_proposal(
    proposal: AuxiliaryGraphRevisionProposalRecord,
    *,
    authority_anchors: tuple[PlanningAuthorityAnchor, ...],
) -> None:
    nodes_by_key: dict[str, AuxiliaryGraphNodeProposalRecord] = {}
    for node in proposal.nodes:
        if node.local_node_key in nodes_by_key:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph node keys must be unique"
            )
        nodes_by_key[node.local_node_key] = node
    terminal = nodes_by_key.get(proposal.terminal_node_key)
    if terminal is None or terminal.node_kind != "synthesize" or not terminal.required:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph requires one required synthesize terminal node"
        )
    if sum(node.node_kind == "synthesize" for node in proposal.nodes) != 1:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph requires exactly one synthesize node"
        )
    anchor_classes = {
        anchor.projection_alias: anchor.authority_class.value
        for anchor in authority_anchors
    }
    for node in proposal.nodes:
        if not set(node.source_anchor_ids) <= set(anchor_classes):
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph node references unknown source authority"
            )
        if not any(
            anchor_classes[anchor_id] == "authorization"
            for anchor_id in node.source_anchor_ids
        ):
            raise AuxiliaryGraphPersistenceError(
                "every AuxiliaryGraph node requires authorization authority"
            )
        for acceptance in node.acceptance_criteria:
            if not set(acceptance.source_anchor_ids) <= set(anchor_classes):
                raise AuxiliaryGraphPersistenceError(
                    "AuxiliaryGraph Acceptance references unknown authority"
                )
            if not any(
                anchor_classes[anchor_id] == "authorization"
                for anchor_id in acceptance.source_anchor_ids
            ):
                raise AuxiliaryGraphPersistenceError(
                    "every AuxiliaryGraph Acceptance requires authorization"
                )

    edges = {
        (edge.dependency_node_key, edge.consumer_node_key)
        for edge in proposal.edges
    }
    if len(edges) != len(proposal.edges):
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph edges must be unique"
        )
    if any(
        dependency not in nodes_by_key or consumer not in nodes_by_key
        for dependency, consumer in edges
    ):
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph edge references an unknown node"
        )
    if any(dependency == proposal.terminal_node_key for dependency, _ in edges):
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph terminal node cannot feed another node"
        )
    children: dict[str, list[str]] = {key: [] for key in nodes_by_key}
    parents: dict[str, list[str]] = {key: [] for key in nodes_by_key}
    for dependency, consumer in edges:
        children[dependency].append(consumer)
        parents[consumer].append(dependency)
    indegree = {key: len(value) for key, value in parents.items()}
    frontier = sorted(key for key, count in indegree.items() if count == 0)
    visited: list[str] = []
    depth = {key: 1 for key in frontier}
    while frontier:
        key = frontier.pop(0)
        visited.append(key)
        for child in sorted(children[key]):
            depth[child] = max(depth.get(child, 1), depth[key] + 1)
            if depth[child] > 12:
                raise AuxiliaryGraphPersistenceError(
                    "AuxiliaryGraph depth exceeds the Host bound"
                )
            indegree[child] -= 1
            if indegree[child] == 0:
                frontier.append(child)
                frontier.sort()
    if len(visited) != len(nodes_by_key):
        raise AuxiliaryGraphPersistenceError("AuxiliaryGraph must be acyclic")
    terminal_ancestors = {proposal.terminal_node_key}
    stack = [proposal.terminal_node_key]
    while stack:
        current = stack.pop()
        for parent in parents[current]:
            if parent not in terminal_ancestors:
                terminal_ancestors.add(parent)
                stack.append(parent)
    missing_required = {
        node.local_node_key
        for node in proposal.nodes
        if node.required and node.local_node_key not in terminal_ancestors
    }
    if missing_required:
        raise AuxiliaryGraphPersistenceError(
            "every required AuxiliaryGraph node must feed the terminal"
        )


def _plan_node_revision_identities(
    conn: sqlite3.Connection,
    *,
    deps: StoreDeps,
    auxiliary_graph_id: str,
    previous_auxiliary_graph_revision: int | None,
    proposal: AuxiliaryGraphRevisionProposalRecord,
) -> dict[str, tuple[str, int, AuxiliaryNodeReference | None]]:
    """结转派生前解析可选 Architect 谱系提示。

    ``origin_node_alias`` 保留持久节点身份，并将其不可变定义 revision 精确推进一版。新成员
    关系最初以 ``proposed`` 状态创建，不带结转回执。之后事务本地派生只能跨
    ``verification_failed`` R->R+1 边界结转独立认证的纯模型完成项；仅有谱系提示不授予
    完成权威。没有来源的节点始终获得新的持久身份。
    """

    origins = tuple(
        node.origin_node_alias
        for node in proposal.nodes
        if node.origin_node_alias is not None
    )
    if len(origins) != len(set(origins)):
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph origin node aliases must be unique"
        )
    if previous_auxiliary_graph_revision is None:
        if origins:
            raise AuxiliaryGraphPersistenceError(
                "initial AuxiliaryGraph revision cannot reference origin nodes"
            )
        previous_by_alias: dict[str, tuple[str, int]] = {}
    else:
        rows = conn.execute(
            "SELECT membership.local_node_key, membership.auxiliary_node_id, "
            "membership.node_revision FROM "
            "insession_auxiliary_graph_revision_nodes_v2 AS membership "
            "JOIN insession_auxiliary_node_definitions_v2 AS definition "
            "ON definition.auxiliary_graph_id=membership.auxiliary_graph_id "
            "AND definition.auxiliary_node_id=membership.auxiliary_node_id "
            "AND definition.node_revision=membership.node_revision "
            "WHERE membership.auxiliary_graph_id=? "
            "AND membership.auxiliary_graph_revision=? "
            "ORDER BY membership.ordinal, membership.auxiliary_node_id",
            (auxiliary_graph_id, previous_auxiliary_graph_revision),
        ).fetchall()
        previous_by_alias = {
            str(row["local_node_key"]): (
                str(row["auxiliary_node_id"]),
                int(row["node_revision"]),
            )
            for row in rows
        }
        if not previous_by_alias or len(previous_by_alias) != len(rows):
            raise AuxiliaryGraphPersistenceError(
                "current AuxiliaryGraph node lineage is missing or ambiguous"
            )
        unknown = set(origins) - set(previous_by_alias)
        if unknown:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph proposal references an unknown origin node"
            )

    planned: dict[str, tuple[str, int, AuxiliaryNodeReference | None]] = {}
    for node in proposal.nodes:
        origin_alias = node.origin_node_alias
        if origin_alias is None:
            planned[node.local_node_key] = (
                _allocate_id(deps, "auxnodev2"),
                1,
                None,
            )
            continue
        node_id, previous_node_revision = previous_by_alias[origin_alias]
        newer = conn.execute(
            "SELECT 1 FROM insession_auxiliary_node_definitions_v2 "
            "WHERE auxiliary_graph_id=? AND auxiliary_node_id=? "
            "AND node_revision>? LIMIT 1",
            (auxiliary_graph_id, node_id, previous_node_revision),
        ).fetchone()
        if newer is not None:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph origin node is not the latest definition"
            )
        planned[node.local_node_key] = (
            node_id,
            previous_node_revision + 1,
            AuxiliaryNodeReference(
                node_id=node_id,
                node_revision=previous_node_revision,
            ),
        )
    return planned


def _require_terminal_candidate_semantic_replan_authority(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
    auxiliary_graph_id: str,
    goal_id: str,
    auxiliary_graph_revision: int,
    settlement_id: str,
    archive: bool,
    now: str,
) -> None:
    """认证并可选归档一个被拒终态游标。

    ``archive=True`` 在追加 revision ``R+1`` 的同一事务中运行。因此 revision 提交失败时，
    验证中断、WorkRun 与节点取消以及 Turn Window 解除会一同回滚。重放使用
    ``archive=False``，证明精确被拒游标仍归档在已密封 revision 回执之后。
    """

    from ..delivery.auxiliary_semantic_verification import (
        AuxiliarySemanticVerificationPersistenceError,
        _load_settlement,
    )

    try:
        settlement = _load_settlement(conn, settlement_id)
    except AuxiliarySemanticVerificationPersistenceError as exc:
        raise AuxiliaryGraphPersistenceError(
            "terminal candidate semantic settlement is not authenticated"
        ) from exc
    route = derive_task_graph_semantic_terminal_route(settlement.results)
    if (
        settlement.session_id != session_id
        or settlement.task_id != task_id
        or settlement.auxiliary_graph_id != auxiliary_graph_id
        or settlement.goal_id != goal_id
        or settlement.auxiliary_graph_revision != auxiliary_graph_revision
        or settlement.created_turn_id != turn_id
        or not _semantic_settlement_authorizes_candidate_replan(
            settlement,
            route=route,
        )
    ):
        raise AuxiliaryGraphPersistenceError(
            "terminal candidate settlement does not authorize this replan"
        )
    candidate = settlement.requests[0].terminal_candidate_binding
    if candidate is None or any(
        request.terminal_candidate_binding != candidate
        for request in settlement.requests
    ):
        raise AuxiliaryGraphPersistenceError(
            "terminal candidate settlement lost one exact cursor binding"
        )
    row = conn.execute(
        "SELECT request.request_revision, request.status AS request_status, "
        "request.technical_error_code, request.auxiliary_node_id, "
        "request.node_revision, run.revision AS work_run_revision, "
        "run.status AS work_run_status, run.reason AS work_run_reason, "
        "run.current_attempt_id, run.current_verification_request_id, "
        "node.status AS node_status, window.turn_id AS window_turn_id, "
        "window.window_state, window.stage, window.current_work_run_id, "
        "window.current_attempt_id AS window_attempt_id, "
        "window.latest_checkpoint_id, output.frozen_at, "
        "(SELECT COUNT(*) FROM insession_auxiliary_node_completions_v2 AS completion "
        "WHERE completion.work_run_id=run.work_run_id) AS completion_count "
        "FROM insession_work_run_verification_requests AS request "
        "JOIN insession_work_runs AS run ON run.work_run_id=request.work_run_id "
        "JOIN insession_auxiliary_node_states_v2 AS node "
        "ON node.auxiliary_graph_id=request.auxiliary_graph_id "
        "AND node.auxiliary_graph_revision=request.auxiliary_graph_revision "
        "AND node.auxiliary_node_id=request.auxiliary_node_id "
        "AND node.node_revision=request.node_revision "
        "JOIN insession_work_run_output_windows AS output "
        "ON output.work_run_id=run.work_run_id "
        "AND output.output_revision=request.output_revision "
        "LEFT JOIN turn_execution_windows AS window "
        "ON window.session_id=run.session_id "
        "WHERE request.verification_request_id=? AND request.session_id=? "
        "AND request.work_run_id=? AND request.submitted_attempt_id=? "
        "AND request.output_revision=? AND request.insession_task_id=? "
        "AND request.auxiliary_graph_id=? "
        "AND request.auxiliary_graph_revision=?",
        (
            candidate.node_verification_request_id,
            session_id,
            candidate.work_run_id,
            candidate.submitted_attempt_id,
            candidate.output_revision,
            task_id,
            auxiliary_graph_id,
            auxiliary_graph_revision,
        ),
    ).fetchone()
    if row is None or row["frozen_at"] is not None or int(row["completion_count"]) != 0:
        raise AuxiliaryGraphPersistenceError(
            "rejected terminal candidate was completed or frozen"
        )
    if not archive:
        if (
            str(row["request_status"]) != "interrupted"
            or str(row["technical_error_code"] or "")
            != "terminal_candidate_semantic_replan"
            or str(row["work_run_status"]) != "cancelled"
            or str(row["work_run_reason"] or "")
            != "terminal_candidate_semantic_replan"
            or row["current_attempt_id"] is not None
            or row["current_verification_request_id"] is not None
            or str(row["node_status"]) != "cancelled"
            or str(row["current_work_run_id"] or "") == candidate.work_run_id
            or str(row["latest_checkpoint_id"] or "")
            == candidate.node_verification_request_id
        ):
            raise AuxiliaryGraphPersistenceError(
                "replayed terminal candidate replan lost its archived cursor"
            )
        return
    if (
        str(row["request_status"]) != "pending"
        or row["technical_error_code"] is not None
        or str(row["work_run_status"]) != "active"
        or str(row["work_run_reason"] or "") != "verification_pending"
        or row["current_attempt_id"] is not None
        or str(row["current_verification_request_id"] or "")
        != candidate.node_verification_request_id
        or str(row["node_status"]) != "active"
        or str(row["window_turn_id"] or "") != turn_id
        or str(row["window_state"] or "") != "active"
        or str(row["stage"] or "") != "VERIFICATION"
        or str(row["current_work_run_id"] or "") != candidate.work_run_id
        or row["window_attempt_id"] is not None
        or str(row["latest_checkpoint_id"] or "")
        != candidate.node_verification_request_id
    ):
        raise AuxiliaryGraphPersistenceError(
            "terminal candidate is no longer at its semantic replan checkpoint"
        )
    if conn.execute(
        "UPDATE insession_work_run_verification_requests SET "
        "status='interrupted', request_revision=request_revision+1, "
        "technical_error_code='terminal_candidate_semantic_replan', "
        "result_json=NULL, all_pass=NULL, updated_at=?, completed_at=NULL "
        "WHERE verification_request_id=? AND request_revision=? "
        "AND status='pending'",
        (
            now,
            candidate.node_verification_request_id,
            int(row["request_revision"]),
        ),
    ).rowcount != 1:
        raise AuxiliaryGraphPersistenceError(
            "terminal candidate verification changed during replan"
        )
    if conn.execute(
        "UPDATE insession_work_runs SET status='cancelled', "
        "reason='terminal_candidate_semantic_replan', revision=revision+1, "
        "current_verification_request_id=NULL, updated_turn_id=?, updated_at=? "
        "WHERE work_run_id=? AND revision=? AND status='active' "
        "AND reason='verification_pending' "
        "AND current_attempt_id IS NULL "
        "AND current_verification_request_id=?",
        (
            turn_id,
            now,
            candidate.work_run_id,
            int(row["work_run_revision"]),
            candidate.node_verification_request_id,
        ),
    ).rowcount != 1:
        raise AuxiliaryGraphPersistenceError(
            "terminal candidate WorkRun changed during replan"
        )
    if conn.execute(
        "UPDATE insession_auxiliary_node_states_v2 SET status='cancelled', "
        "state_version=state_version+1, updated_at=? "
        "WHERE auxiliary_graph_id=? AND auxiliary_graph_revision=? "
        "AND auxiliary_node_id=? AND node_revision=? AND status='active'",
        (
            now,
            auxiliary_graph_id,
            auxiliary_graph_revision,
            str(row["auxiliary_node_id"]),
            int(row["node_revision"]),
        ),
    ).rowcount != 1:
        raise AuxiliaryGraphPersistenceError(
            "terminal candidate node changed during replan"
        )
    if conn.execute(
        "UPDATE turn_execution_windows SET current_work_run_id=NULL, "
        "current_attempt_id=NULL, latest_checkpoint_id=NULL, stage='L2_PLAN', "
        "state_version=state_version+1, updated_at=? "
        "WHERE session_id=? AND turn_id=? AND window_state='active' "
        "AND current_work_run_id=? AND current_attempt_id IS NULL "
        "AND latest_checkpoint_id=? AND stage='VERIFICATION'",
        (
            now,
            session_id,
            turn_id,
            candidate.work_run_id,
            candidate.node_verification_request_id,
        ),
    ).rowcount != 1:
        raise AuxiliaryGraphPersistenceError(
            "terminal candidate Turn cursor changed during replan"
        )


def _semantic_settlement_authorizes_candidate_replan(
    settlement,
    *,
    route: TaskGraphSemanticTerminalRoute,
) -> bool:
    if route is TaskGraphSemanticTerminalRoute.REPLAN_AUXILIARY:
        return (
            settlement.host_disposition
            is TaskGraphSemanticVerificationDisposition.REVISE
        )
    if (
        route is not TaskGraphSemanticTerminalRoute.BLOCKED
        or settlement.host_disposition
        is not TaskGraphSemanticVerificationDisposition.BLOCKED
    ):
        return False
    return is_task_graph_semantic_user_information_block(
        requests=settlement.requests,
        results=settlement.results,
    )


def _require_revision_safe_point(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    task_id: str,
    auxiliary_graph_id: str,
    auxiliary_graph_revision: int,
) -> None:
    if conn.execute(
        "SELECT 1 FROM insession_work_runs WHERE session_id=? "
        "AND insession_task_id=? AND subject_kind='auxiliary_node' "
        "AND auxiliary_graph_id=? AND auxiliary_graph_revision=? "
        "AND status NOT IN ('completed', 'failed', 'cancelled') LIMIT 1",
        (session_id, task_id, auxiliary_graph_id, auxiliary_graph_revision),
    ).fetchone() is not None:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph revision requires a safe WorkRun checkpoint"
        )
    if conn.execute(
        "SELECT 1 FROM insession_work_run_tool_results AS result "
        "JOIN insession_work_runs AS run ON run.work_run_id=result.work_run_id "
        "WHERE run.session_id=? AND run.insession_task_id=? "
        "AND run.subject_kind='auxiliary_node' "
        "AND run.auxiliary_graph_id=? AND run.auxiliary_graph_revision=? "
        "AND result.status='completion_unconfirmed' LIMIT 1",
        (session_id, task_id, auxiliary_graph_id, auxiliary_graph_revision),
    ).fetchone() is not None:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph revision is blocked by completion-unconfirmed authority"
        )


def _validate_revision_replay(
    conn: sqlite3.Connection,
    *,
    stored: AuxiliaryGraphRevisionCommitResult,
    receipt: sqlite3.Row,
) -> None:
    result_json = str(receipt["result_json"])
    budget_snapshot_json = str(receipt["committed_budget_snapshot_json"])
    try:
        receipt_budget = PlanningEpisodeBudget.model_validate_json(
            budget_snapshot_json
        )
    except (TypeError, ValueError) as exc:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph replay budget receipt is corrupt"
        ) from exc
    expected_control = receipt["expected_control_state_version"]
    expected_revision = receipt["expected_current_auxiliary_graph_revision"]
    expected_goal = receipt["expected_goal_state_version"]
    expected_budget = receipt["expected_budget_state_version"]
    receipt_shape_valid = (
        _text_hash(result_json) == str(receipt["result_sha256"])
        and int(receipt["committed_control_state_version"])
        == stored.control_state_version
        and int(receipt["committed_goal_state_version"])
        == stored.goal_state_version
        and int(receipt["committed_budget_state_version"])
        == stored.budget_state_version
        and receipt_budget.goal_id == stored.goal_id
        and receipt_budget.budget_ledger_id
        == str(receipt["budget_ledger_id"])
        and receipt_budget.state_version == stored.budget_state_version
        and _model_json(receipt_budget) == budget_snapshot_json
        and receipt_budget.snapshot_sha256
        == str(receipt["committed_budget_snapshot_sha256"])
        and (
            (
                expected_control is None
                and expected_revision is None
                and stored.control_state_version == 2
                and stored.committed_auxiliary_graph_revision == 1
            )
            or (
                expected_control is not None
                and expected_revision is not None
                and stored.control_state_version == int(expected_control) + 1
                and stored.committed_auxiliary_graph_revision
                == int(expected_revision) + 1
            )
        )
        and (
            (
                expected_goal is None
                and expected_budget is None
                and stored.goal_state_version == 1
                and stored.budget_state_version == 2
            )
            or (
                expected_goal is not None
                and expected_budget is not None
                and stored.goal_state_version == int(expected_goal) + 1
                and stored.budget_state_version == int(expected_budget) + 1
            )
        )
    )
    row = conn.execute(
        "SELECT revision.goal_id, revision.source_turn_id, revision.structure_sha256, "
        "revision.authority_snapshot_id, revision.authority_snapshot_sha256, "
        "authority.snapshot_json, authority.snapshot_sha256, "
        "budget.state_version AS budget_state_version, "
        "charge.budget_state_version_after AS charge_budget_state_version, "
        "charge.budget_snapshot_after_json, "
        "charge.budget_snapshot_after_sha256 "
        "FROM insession_auxiliary_graph_revision_snapshots AS revision "
        "JOIN insession_auxiliary_authority_snapshots AS authority "
        "ON authority.authority_snapshot_id=revision.authority_snapshot_id "
        "JOIN insession_auxiliary_goal_budgets AS budget "
        "ON budget.goal_id=revision.goal_id "
        "JOIN insession_auxiliary_goal_budget_charges AS charge "
        "ON charge.goal_id=revision.goal_id AND charge.charge_key=? "
        "WHERE revision.auxiliary_graph_id=? "
        "AND revision.auxiliary_graph_revision=?",
        (
            str(receipt["apply_id"]),
            stored.auxiliary_graph_id,
            stored.committed_auxiliary_graph_revision,
        ),
    ).fetchone()
    if (
        not receipt_shape_valid
        or row is None
        or str(receipt["auxiliary_graph_id"]) != stored.auxiliary_graph_id
        or str(receipt["goal_id"]) != stored.goal_id
        or int(receipt["committed_auxiliary_graph_revision"])
        != stored.committed_auxiliary_graph_revision
        or str(row["goal_id"]) != stored.goal_id
        or str(row["structure_sha256"]) != stored.structure_sha256
        or str(row["authority_snapshot_id"]) != stored.authority_snapshot_id
        or str(row["authority_snapshot_sha256"])
        != stored.authority_snapshot_sha256
        or str(row["snapshot_sha256"]) != stored.authority_snapshot_sha256
        or _load_formal_authority_snapshot(
            conn,
            authority_snapshot_id=stored.authority_snapshot_id,
            expected_session_id=str(receipt["session_id"]),
            expected_task_id=str(receipt["insession_task_id"]),
            expected_auxiliary_graph_id=stored.auxiliary_graph_id,
            expected_goal_id=stored.goal_id,
            expected_source_turn_id=str(row["source_turn_id"]),
            expected_snapshot_sha256=stored.authority_snapshot_sha256,
        ).snapshot_sha256
        != stored.authority_snapshot_sha256
        or int(row["budget_state_version"]) < stored.budget_state_version
        or int(row["charge_budget_state_version"])
        != stored.budget_state_version
        or str(row["budget_snapshot_after_json"])
        != budget_snapshot_json
        or str(row["budget_snapshot_after_sha256"])
        != receipt_budget.snapshot_sha256
    ):
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph replay lost immutable revision authority"
        )
    carry_rows = conn.execute(
        "SELECT carry.*, membership.carried_completion_id FROM "
        "insession_auxiliary_node_completion_carries_v2 AS carry "
        "JOIN insession_auxiliary_graph_revision_nodes_v2 AS membership "
        "ON membership.auxiliary_graph_id=carry.auxiliary_graph_id "
        "AND membership.auxiliary_graph_revision="
        "carry.target_auxiliary_graph_revision "
        "AND membership.auxiliary_node_id=carry.target_auxiliary_node_id "
        "AND membership.node_revision=carry.target_node_revision "
        "WHERE carry.auxiliary_graph_id=? "
        "AND carry.target_auxiliary_graph_revision=? "
        "ORDER BY membership.ordinal, carry.target_auxiliary_node_id",
        (
            stored.auxiliary_graph_id,
            stored.committed_auxiliary_graph_revision,
        ),
    ).fetchall()
    carry_ids: list[str] = []
    for carry_row in carry_rows:
        raw = str(carry_row["receipt_json"])
        try:
            carry = (
                AuxiliaryNodePureModelCompletionCarryReceipt.model_validate_json(
                    raw
                )
            )
        except (TypeError, ValueError, RecursionError) as exc:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph replay carry receipt is corrupt"
            ) from exc
        if (
            _model_json(carry) != raw
            or _text_hash(raw) != str(carry_row["receipt_sha256"])
            or carry.carry_receipt_id != str(carry_row["carry_receipt_id"])
            or str(carry_row["carried_completion_id"])
            != carry.carry_receipt_id
            or carry.session_id != str(carry_row["session_id"])
            or carry.task_id != str(carry_row["insession_task_id"])
            or carry.auxiliary_graph_id
            != str(carry_row["auxiliary_graph_id"])
            or carry.goal_id != str(carry_row["goal_id"])
            or carry.source_subject.auxiliary_graph_revision
            != int(carry_row["source_auxiliary_graph_revision"])
            or carry.source_subject.node_id
            != str(carry_row["source_auxiliary_node_id"])
            or carry.source_subject.node_revision
            != int(carry_row["source_node_revision"])
            or carry.target_subject.auxiliary_graph_revision
            != int(carry_row["target_auxiliary_graph_revision"])
            or carry.target_subject.node_id
            != str(carry_row["target_auxiliary_node_id"])
            or carry.target_subject.node_revision
            != int(carry_row["target_node_revision"])
            or carry.apply_id != str(receipt["apply_id"])
            or carry.session_id != str(receipt["session_id"])
            or carry.task_id != str(receipt["insession_task_id"])
            or carry.auxiliary_graph_id != stored.auxiliary_graph_id
            or carry.goal_id != stored.goal_id
            or carry.target_subject.auxiliary_graph_revision
            != stored.committed_auxiliary_graph_revision
            or carry.authority_snapshot_id != stored.authority_snapshot_id
            or carry.authority_snapshot_sha256
            != stored.authority_snapshot_sha256
            or carry.created_turn_id != str(receipt["invocation_turn_id"])
            or carry.source_completion_id
            != str(carry_row["source_completion_id"])
            or carry.definition_sha256
            != str(carry_row["definition_sha256"])
            or carry.dependency_closure_sha256
            != str(carry_row["dependency_closure_sha256"])
            or carry.context_artifact_manifest_sha256
            != str(carry_row["context_artifact_manifest_sha256"])
            or carry.authority_snapshot_id
            != str(carry_row["authority_snapshot_id"])
            or carry.authority_snapshot_sha256
            != str(carry_row["authority_snapshot_sha256"])
            or carry.capability_catalog_snapshot_id
            != str(carry_row["capability_catalog_snapshot_id"])
            or carry.capability_catalog_snapshot_sha256
            != str(carry_row["capability_catalog_snapshot_sha256"])
            or carry.freshness_manifest_sha256
            != str(carry_row["freshness_manifest_sha256"])
            or carry.created_turn_id != str(carry_row["created_turn_id"])
        ):
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph replay carry authority changed"
            )
        _reauthenticate_pure_model_carry(conn, carry)
        carry_ids.append(carry.carry_receipt_id)
    if tuple(carry_ids) != stored.carried_completion_receipt_ids:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph replay carry set changed"
        )


def _load_auxiliary_graph(
    conn: sqlite3.Connection,
    session_id: str,
    task_id: str,
) -> StoredAuxiliaryGraphDetails:
    row = conn.execute(
        "SELECT control.auxiliary_graph_id, control.state_version AS control_version, "
        "goal.goal_id, goal.objective AS goal_objective, "
        "goal.status AS goal_status, "
        "goal.state_version AS goal_version, goal.base_task_graph_revision, "
        "goal.target_task_graph_revision, goal.creation_turn_id, "
        "goal.authorization_manifest_id, goal.authorization_manifest_sha256, "
        "goal.budget_ledger_id AS goal_budget_ledger_id, "
        "revision.auxiliary_graph_revision, "
        "revision.parent_auxiliary_graph_revision, revision.source_turn_id, "
        "revision.reason, revision.authority_snapshot_id, "
        "revision.authority_snapshot_sha256, revision.structure_sha256, "
        "revision.structure_contract_version, "
        "revision.terminal_auxiliary_node_id, state.status AS revision_status, "
        "state.state_version AS revision_state_version, authority.snapshot_json, "
        "authority.snapshot_sha256, budget.budget_ledger_id, "
        "budget.contract_version, budget.profile_json, budget.profile_sha256, "
        "budget.usage_json, budget.usage_sha256, budget.extensions_json, "
        "budget.extensions_sha256, budget.snapshot_json AS budget_snapshot_json, "
        "budget.snapshot_sha256 AS budget_snapshot_sha256, "
        "budget.state_version AS budget_state_version "
        "FROM insession_auxiliary_graph_v2_containers AS control "
        "JOIN insession_auxiliary_graph_goals AS goal "
        "ON goal.auxiliary_graph_id=control.auxiliary_graph_id "
        "AND goal.goal_id=control.current_goal_id "
        "JOIN insession_auxiliary_graph_revision_snapshots AS revision "
        "ON revision.auxiliary_graph_id=control.auxiliary_graph_id "
        "AND revision.auxiliary_graph_revision="
        "control.current_auxiliary_graph_revision "
        "AND revision.goal_id=goal.goal_id "
        "JOIN insession_auxiliary_graph_revision_states_v2 AS state "
        "ON state.auxiliary_graph_id=revision.auxiliary_graph_id "
        "AND state.auxiliary_graph_revision=revision.auxiliary_graph_revision "
        "JOIN insession_auxiliary_authority_snapshots AS authority "
        "ON authority.authority_snapshot_id=revision.authority_snapshot_id "
        "JOIN insession_auxiliary_goal_budgets AS budget "
        "ON budget.goal_id=goal.goal_id "
        "WHERE control.session_id=? AND control.insession_task_id=?",
        (session_id, task_id),
    ).fetchone()
    if row is None:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph current projection is incomplete"
        )
    if str(row["structure_contract_version"]) != "auxiliary-graph-revision-v2":
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph revision uses a retired structure contract"
        )
    try:
        aggregate = AuxiliaryGraphAggregate(
            session_id=session_id,
            task_id=task_id,
            auxiliary_graph_id=str(row["auxiliary_graph_id"]),
            current_goal_id=str(row["goal_id"]),
            current_auxiliary_graph_revision=int(
                row["auxiliary_graph_revision"]
            ),
            state_version=int(row["control_version"]),
        )
        goal = AuxiliaryPlanningGoal(
            session_id=session_id,
            task_id=task_id,
            auxiliary_graph_id=str(row["auxiliary_graph_id"]),
            goal_id=str(row["goal_id"]),
            base_task_graph_revision=(
                int(row["base_task_graph_revision"])
                if row["base_task_graph_revision"] is not None
                else None
            ),
            target_task_graph_revision=int(row["target_task_graph_revision"]),
            creation_turn_id=str(row["creation_turn_id"]),
            authorization_manifest_id=str(row["authorization_manifest_id"]),
            budget_ledger_id=str(row["goal_budget_ledger_id"]),
            status=str(row["goal_status"]),
            state_version=int(row["goal_version"]),
        )
    except (TypeError, ValueError) as exc:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph aggregate or goal projection is corrupt"
        ) from exc
    manifest = conn.execute(
        "SELECT session_id, insession_task_id, auxiliary_graph_id, goal_id, "
        "manifest_json, manifest_sha256, created_turn_id FROM "
        "insession_auxiliary_authorization_manifests "
        "WHERE authorization_manifest_id=?",
        (goal.authorization_manifest_id,),
    ).fetchone()
    if (
        manifest is None
        or str(manifest["session_id"]) != session_id
        or str(manifest["insession_task_id"]) != task_id
        or str(manifest["auxiliary_graph_id"]) != aggregate.auxiliary_graph_id
        or str(manifest["goal_id"]) != goal.goal_id
        or str(manifest["created_turn_id"]) != goal.creation_turn_id
        or str(manifest["manifest_sha256"])
        != str(row["authorization_manifest_sha256"])
        or _text_hash(str(manifest["manifest_json"]))
        != str(manifest["manifest_sha256"])
        or goal.budget_ledger_id != str(row["budget_ledger_id"])
    ):
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph goal manifest binding is corrupt"
        )
    authority_snapshot = _load_formal_authority_snapshot(
        conn,
        authority_snapshot_id=str(row["authority_snapshot_id"]),
        expected_session_id=session_id,
        expected_task_id=task_id,
        expected_auxiliary_graph_id=str(row["auxiliary_graph_id"]),
        expected_goal_id=str(row["goal_id"]),
        expected_source_turn_id=str(row["source_turn_id"]),
        expected_snapshot_sha256=str(row["authority_snapshot_sha256"]),
    )
    budget_snapshot = _formal_budget_from_row(
        row,
        expected_goal_id=str(row["goal_id"]),
        snapshot_json_column="budget_snapshot_json",
        snapshot_sha256_column="budget_snapshot_sha256",
        state_version_column="budget_state_version",
    )
    node_rows = conn.execute(
        "SELECT membership.auxiliary_node_id, membership.node_revision, "
        "membership.ordinal, membership.local_node_key, membership.required, "
        "definition.node_kind, definition.executor_kind, "
        "definition.title, definition.objective, definition.source_anchor_ids_json, "
        "definition.acceptance_criteria_json, definition.output_contract, "
        "definition.capability_profile_id, "
        "definition.input_resource_aliases_json, definition.required AS definition_required, "
        "definition.semantic_fingerprint, definition.origin_auxiliary_node_id, "
        "definition.origin_node_revision, definition.definition_sha256, "
        "node_state.status, "
        "node_state.state_version "
        "FROM insession_auxiliary_graph_revision_nodes_v2 AS membership "
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
        "AND membership.auxiliary_graph_revision=? ORDER BY membership.ordinal",
        (str(row["auxiliary_graph_id"]), int(row["auxiliary_graph_revision"])),
    ).fetchall()
    node_ordinals: dict[str, int] = {}
    materialized_nodes: list[AuxiliaryNodeDefinition] = []
    for item in node_rows:
        node_id = str(item["auxiliary_node_id"])
        source_anchor_ids = tuple(
            json.loads(str(item["source_anchor_ids_json"]))
        )
        acceptance_criteria = json.loads(
            str(item["acceptance_criteria_json"])
        )
        capability_profile_id = (
            str(item["capability_profile_id"])
            if item["capability_profile_id"] is not None
            else None
        )
        origin_node_ref = (
            AuxiliaryNodeReference(
                node_id=str(item["origin_auxiliary_node_id"]),
                node_revision=int(item["origin_node_revision"]),
            )
            if item["origin_auxiliary_node_id"] is not None
            and item["origin_node_revision"] is not None
            else None
        )
        if bool(item["required"]) != bool(item["definition_required"]):
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph membership changed immutable required semantics"
            )
        try:
            materialized = AuxiliaryNodeDefinition(
                node_id=node_id,
                node_revision=int(item["node_revision"]),
                ordinal=int(item["ordinal"]),
                node_kind=AuxiliaryNodeKind(str(item["node_kind"])),
                executor_kind=AuxiliaryNodeExecutorKind(
                    str(item["executor_kind"])
                ),
                title=str(item["title"]),
                objective=str(item["objective"]),
                acceptance_criteria=tuple(
                    InSessionTaskAcceptanceProposal.model_validate(value)
                    for value in acceptance_criteria
                ),
                capability_profile_id=capability_profile_id,
                input_resource_aliases=tuple(
                    json.loads(str(item["input_resource_aliases_json"]))
                ),
                source_anchor_ids=source_anchor_ids,
                output_contract=str(item["output_contract"]),
                required=bool(item["definition_required"]),
                semantic_fingerprint=str(item["semantic_fingerprint"]),
                origin_node_ref=origin_node_ref,
            )
        except (TypeError, ValueError) as exc:
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph node definition is corrupt"
            ) from exc
        if materialized.semantic_fingerprint != str(item["definition_sha256"]):
            raise AuxiliaryGraphPersistenceError(
                "AuxiliaryGraph node definition hash is corrupt"
            )
        ordinal = int(item["ordinal"])
        node_ordinals[node_id] = ordinal
        materialized_nodes.append(materialized)
    nodes = tuple(
        StoredAuxiliaryGraphNode(
            auxiliary_node_id=str(item["auxiliary_node_id"]),
            node_revision=int(item["node_revision"]),
            ordinal=int(item["ordinal"]),
            local_node_key=str(item["local_node_key"]),
            node_kind=str(item["node_kind"]),
            executor_kind=str(item["executor_kind"]),
            title=str(item["title"]),
            objective=str(item["objective"]),
            source_anchor_ids=tuple(json.loads(str(item["source_anchor_ids_json"]))),
            acceptance_criteria=tuple(
                InSessionTaskAcceptanceProposal.model_validate(value)
                for value in json.loads(str(item["acceptance_criteria_json"]))
            ),
            output_contract=str(item["output_contract"]),
            capability_profile_id=(
                str(item["capability_profile_id"])
                if item["capability_profile_id"] is not None
                else None
            ),
            input_resource_aliases=tuple(
                json.loads(str(item["input_resource_aliases_json"]))
            ),
            required=bool(item["required"]),
            semantic_fingerprint=str(item["semantic_fingerprint"]),
            origin_node_ref=(
                AuxiliaryNodeReference(
                    node_id=str(item["origin_auxiliary_node_id"]),
                    node_revision=int(item["origin_node_revision"]),
                )
                if item["origin_auxiliary_node_id"] is not None
                and item["origin_node_revision"] is not None
                else None
            ),
            status=str(item["status"]),
            state_version=int(item["state_version"]),
        )
        for item in node_rows
    )
    if authority_snapshot is not None:
        authority_classes = {
            anchor.projection_alias: anchor.authority_class
            for anchor in authority_snapshot.anchors
        }
        for node in materialized_nodes:
            if not set(node.source_anchor_ids).issubset(authority_classes):
                raise AuxiliaryGraphPersistenceError(
                    "AuxiliaryGraph node authority alias is missing"
                )
            if not any(
                authority_classes[alias]
                is PlanningAuthorityClass.AUTHORIZATION
                for alias in node.source_anchor_ids
            ):
                raise AuxiliaryGraphPersistenceError(
                    "AuxiliaryGraph node lost authorization authority"
                )
    edge_rows = conn.execute(
        "SELECT dependency_auxiliary_node_id, consumer_auxiliary_node_id, "
        "required, ordinal "
        "FROM insession_auxiliary_graph_edges WHERE auxiliary_graph_id=? "
        "AND auxiliary_graph_revision=? "
        "ORDER BY dependency_auxiliary_node_id, consumer_auxiliary_node_id",
        (str(row["auxiliary_graph_id"]), int(row["auxiliary_graph_revision"])),
    ).fetchall()
    ordered_edge_rows = sorted(
        edge_rows,
        key=lambda item: (
            node_ordinals[str(item["dependency_auxiliary_node_id"])],
            node_ordinals[str(item["consumer_auxiliary_node_id"])],
        ),
    )
    edges = tuple(
        StoredAuxiliaryGraphEdge(
            dependency_auxiliary_node_id=str(
                item["dependency_auxiliary_node_id"]
            ),
            consumer_auxiliary_node_id=str(item["consumer_auxiliary_node_id"]),
            ordinal=int(item["ordinal"]),
            required=bool(item["required"]),
        )
        for item in ordered_edge_rows
    )
    try:
        materialized_revision = AuxiliaryGraphRevision(
            session_id=session_id,
            task_id=task_id,
            auxiliary_graph_id=str(row["auxiliary_graph_id"]),
            goal_id=str(row["goal_id"]),
            auxiliary_graph_revision=int(row["auxiliary_graph_revision"]),
            parent_auxiliary_graph_revision=(
                int(row["parent_auxiliary_graph_revision"])
                if row["parent_auxiliary_graph_revision"] is not None
                else None
            ),
            base_task_graph_revision=(
                int(row["base_task_graph_revision"])
                if row["base_task_graph_revision"] is not None
                else None
            ),
            source_turn_id=str(row["source_turn_id"]),
            revision_reason=AuxiliaryGraphRevisionReason(str(row["reason"])),
            authority_snapshot_id=str(row["authority_snapshot_id"]),
            authority_snapshot_sha256=str(row["authority_snapshot_sha256"]),
            terminal_node_id=str(row["terminal_auxiliary_node_id"]),
            nodes=tuple(materialized_nodes),
            edges=tuple(
                AuxiliaryGraphEdge(
                    source_node_id=str(item["dependency_auxiliary_node_id"]),
                    target_node_id=str(item["consumer_auxiliary_node_id"]),
                    required=bool(item["required"]),
                )
                for item in ordered_edge_rows
            ),
            structure_sha256=str(row["structure_sha256"]),
        )
    except (TypeError, ValueError) as exc:
        raise AuxiliaryGraphPersistenceError(
            "AuxiliaryGraph structure hash is corrupt"
        ) from exc
    return StoredAuxiliaryGraphDetails(
        session_id=session_id,
        task_id=task_id,
        auxiliary_graph_id=str(row["auxiliary_graph_id"]),
        control_state_version=int(row["control_version"]),
        goal_id=str(row["goal_id"]),
        goal_objective=str(row["goal_objective"]),
        goal_status=str(row["goal_status"]),
        goal_state_version=int(row["goal_version"]),
        base_task_graph_revision=(
            int(row["base_task_graph_revision"])
            if row["base_task_graph_revision"] is not None
            else None
        ),
        target_task_graph_revision=int(row["target_task_graph_revision"]),
        auxiliary_graph_revision=int(row["auxiliary_graph_revision"]),
        parent_auxiliary_graph_revision=(
            int(row["parent_auxiliary_graph_revision"])
            if row["parent_auxiliary_graph_revision"] is not None
            else None
        ),
        revision_status=str(row["revision_status"]),
        revision_state_version=int(row["revision_state_version"]),
        source_turn_id=str(row["source_turn_id"]),
        reason=str(row["reason"]),
        authority_snapshot_id=str(row["authority_snapshot_id"]),
        authority_snapshot_sha256=str(row["authority_snapshot_sha256"]),
        structure_sha256=str(row["structure_sha256"]),
        terminal_auxiliary_node_id=str(row["terminal_auxiliary_node_id"]),
        budget_profile=json.loads(str(row["profile_json"])),
        budget_usage=json.loads(str(row["usage_json"])),
        budget_state_version=int(row["budget_state_version"]),
        nodes=nodes,
        edges=edges,
        revision=materialized_revision,
        aggregate=aggregate,
        goal=goal,
        authority_snapshot=authority_snapshot,
        budget=budget_snapshot,
    )


__all__ = [
    "AuxiliaryGraphRevisionCommitResult",
    "AuxiliaryGraphApplyIdCollision",
    "AuxiliaryGraphPersistenceError",
    "commit_auxiliary_graph_revision",
    "commit_auxiliary_node_verification_result",
    "create_auxiliary_node_work_run",
    "get_current_auxiliary_graph_revision",
    "get_current_auxiliary_planning_goal",
    "get_auxiliary_graph_for_task",
    "get_prepared_auxiliary_node_verification",
    "prepare_auxiliary_node_verification",
    "project_auxiliary_graph_execution_frontier",
    "require_auxiliary_graph_commit_context_valid",
]
