"""持久化 AuxiliaryGraph TaskGraph 语义验证权威状态。

v62 schema 已负责不可变的能力投影、审查请求/结果、精确的 ContextArtifact
绑定以及 quorum 结算表。本模块是这些表的唯一写入方。它有意接收完整的公开
语义验证契约，并在一个 SQLite 事务内重新验证其 graph、goal、revision、
authority、budget、capability、prompt、policy 与 terminal proposal 绑定。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personagraph.l2.auxiliary_graph import (
    PlanningAuthorityAnchor,
    PlanningAuthorityClass,
    PlanningAuthorityOriginKind,
    PlanningAuthoritySnapshot,
    PlanningAuthoritySourceCard,
    PlanningAuthoritySourceKind,
    PlanningCapabilityCatalogProjection,
    PlanningCapabilityEffect,
    PlanningContextArtifactProjection,
    PlanningContextArtifact,
    PlanningEpisodeBudget,
    TaskGraphRevisionCandidate,
    TaskGraphSemanticLineageDisposition,
    TaskGraphSemanticLineageProjection,
    TaskGraphSemanticReviewPolicy,
    TaskGraphSemanticVerificationDisposition,
    TaskGraphSemanticVerificationPromptPayload,
    TaskGraphSemanticVerificationRequest,
    TaskGraphSemanticVerificationResult,
    validate_task_graph_semantic_verification_quorum,
    validate_task_graph_semantic_verification_result,
)
from personagraph.l2.task_graph import InSessionTaskGraphRevisionProposal, InSessionTaskStatus
from personagraph.l2.work_run import (
    AuxiliaryNodeSubject,
    NodeVerificationResult,
    OutputWindow,
)
from ..auxiliary_graph.auxiliary_graph_errors import AuxiliaryGraphPersistenceError
from ..auxiliary_graph.auxiliary_graphs import (
    _load_formal_authority_snapshot,
    _load_auxiliary_graph,
)
from ..auxiliary_graph.auxiliary_node_execution_bindings import (
    _auxiliary_record_from_row,
    _load_auxiliary_request_row,
    _revalidate_auxiliary_request_binding,
)
from ..auxiliary_graph.auxiliary_dependencies import (
    _projection_binds_artifact,
    _resolve_model_dependency,
)
from ..auxiliary_graph.auxiliary_terminal_validation import (
    AuxiliaryTerminalValidationContextError,
    _build_auxiliary_terminal_semantic_support,
    _planning_resource_source_cards,
)
from ...deps import StoreDeps


_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"


class AuxiliarySemanticVerificationPersistenceError(
    AuxiliaryGraphPersistenceError
):
    """语义验证丢失了一项精确的持久化权威绑定。"""


class AuxiliarySemanticVerificationIdentityCollision(
    AuxiliarySemanticVerificationPersistenceError
):
    """同一个自然不可变标识被复用于不同内容。"""


class AuxiliarySemanticVerificationStaleAuthority(
    AuxiliarySemanticVerificationPersistenceError
):
    """当前辅助图聚合已不再匹配给定的 CAS 守卫。"""


class AuxiliarySemanticVerificationStoredAuthorityCorrupt(
    AuxiliarySemanticVerificationPersistenceError
):
    """已存储的语义权威记录已无法通过自身认证。"""


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FreezeAuxiliarySemanticCapabilityCatalogCommand(_Record):
    schema_version: Literal[
        "freeze-auxiliary-semantic-capability-catalog-command-v1"
    ] = "freeze-auxiliary-semantic-capability-catalog-command-v1"
    session_id: str = Field(pattern=_ID_PATTERN)
    created_turn_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_id: str = Field(pattern=_ID_PATTERN)
    goal_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_revision: int = Field(ge=1)
    expected_control_state_version: int = Field(ge=1)
    expected_goal_state_version: int = Field(ge=1)
    expected_revision_state_version: int = Field(ge=1)
    expected_budget_state_version: int = Field(ge=1)
    expected_authority_snapshot_id: str = Field(pattern=_ID_PATTERN)
    expected_authority_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_structure_sha256: str = Field(pattern=_SHA256_PATTERN)
    catalog: PlanningCapabilityCatalogProjection


class CommitAuxiliarySemanticVerificationRequestCommand(_Record):
    schema_version: Literal[
        "commit-auxiliary-semantic-verification-request-command-v1"
    ] = "commit-auxiliary-semantic-verification-request-command-v1"
    session_id: str = Field(pattern=_ID_PATTERN)
    created_turn_id: str = Field(pattern=_ID_PATTERN)
    expected_control_state_version: int = Field(ge=1)
    expected_goal_state_version: int = Field(ge=1)
    expected_revision_state_version: int = Field(ge=1)
    expected_budget_state_version: int = Field(ge=1)
    request: TaskGraphSemanticVerificationRequest

    @model_validator(mode="after")
    def _validate_scope(self) -> "CommitAuxiliarySemanticVerificationRequestCommand":
        if self.request.goal.session_id != self.session_id:
            raise ValueError("semantic request belongs to another Session")
        if self.request.goal.state_version != self.expected_goal_state_version:
            raise ValueError("semantic request goal version differs from its CAS")
        if self.request.budget.state_version != self.expected_budget_state_version:
            raise ValueError("semantic request budget version differs from its CAS")
        return self


class CommitAuxiliarySemanticVerificationResultCommand(_Record):
    schema_version: Literal[
        "commit-auxiliary-semantic-verification-result-command-v1"
    ] = "commit-auxiliary-semantic-verification-result-command-v1"
    session_id: str = Field(pattern=_ID_PATTERN)
    created_turn_id: str = Field(pattern=_ID_PATTERN)
    expected_control_state_version: int = Field(ge=1)
    expected_goal_state_version: int = Field(ge=1)
    expected_revision_state_version: int = Field(ge=1)
    expected_budget_state_version: int = Field(ge=1)
    result: TaskGraphSemanticVerificationResult


class SettleAuxiliarySemanticVerificationQuorumCommand(_Record):
    schema_version: Literal[
        "settle-auxiliary-semantic-verification-quorum-command-v1"
    ] = "settle-auxiliary-semantic-verification-quorum-command-v1"
    settlement_id: str = Field(pattern=_ID_PATTERN)
    session_id: str = Field(pattern=_ID_PATTERN)
    created_turn_id: str = Field(pattern=_ID_PATTERN)
    expected_control_state_version: int = Field(ge=1)
    expected_goal_state_version: int = Field(ge=1)
    expected_revision_state_version: int = Field(ge=1)
    expected_budget_state_version: int = Field(ge=1)
    requests: tuple[TaskGraphSemanticVerificationRequest, ...] = Field(
        min_length=1, max_length=2
    )
    results: tuple[TaskGraphSemanticVerificationResult, ...] = Field(
        min_length=1, max_length=2
    )


class StoredAuxiliarySemanticCapabilityCatalog(_Record):
    schema_version: Literal[
        "stored-auxiliary-semantic-capability-catalog-v1"
    ] = "stored-auxiliary-semantic-capability-catalog-v1"
    session_id: str
    task_id: str
    auxiliary_graph_id: str
    goal_id: str
    auxiliary_graph_revision: int = Field(ge=1)
    projection: PlanningCapabilityCatalogProjection
    created_turn_id: str


class AuxiliarySemanticCapabilityCatalogMutationResult(_Record):
    status: Literal["applied", "replayed"]
    catalog: StoredAuxiliarySemanticCapabilityCatalog


class StoredAuxiliarySemanticVerificationRequest(_Record):
    schema_version: Literal[
        "stored-auxiliary-semantic-verification-request-v1"
    ] = "stored-auxiliary-semantic-verification-request-v1"
    session_id: str
    task_id: str
    auxiliary_graph_id: str
    goal_id: str
    auxiliary_graph_revision: int = Field(ge=1)
    request: TaskGraphSemanticVerificationRequest
    created_turn_id: str


class AuxiliarySemanticVerificationRequestMutationResult(_Record):
    status: Literal["applied", "replayed"]
    record: StoredAuxiliarySemanticVerificationRequest


class StoredAuxiliarySemanticVerificationResult(_Record):
    schema_version: Literal[
        "stored-auxiliary-semantic-verification-result-v1"
    ] = "stored-auxiliary-semantic-verification-result-v1"
    session_id: str
    task_id: str
    auxiliary_graph_id: str
    goal_id: str
    auxiliary_graph_revision: int = Field(ge=1)
    result: TaskGraphSemanticVerificationResult
    host_disposition: TaskGraphSemanticVerificationDisposition
    created_turn_id: str


class AuxiliarySemanticVerificationResultMutationResult(_Record):
    status: Literal["applied", "replayed"]
    record: StoredAuxiliarySemanticVerificationResult


class StoredAuxiliarySemanticQuorumSettlement(_Record):
    schema_version: Literal[
        "stored-auxiliary-semantic-quorum-settlement-v1"
    ] = "stored-auxiliary-semantic-quorum-settlement-v1"
    settlement_id: str = Field(pattern=_ID_PATTERN)
    session_id: str
    task_id: str
    auxiliary_graph_id: str
    goal_id: str
    auxiliary_graph_revision: int = Field(ge=1)
    required_reviewer_count: int = Field(ge=1, le=2)
    frozen_prompt_payload_sha256: str = Field(pattern=_SHA256_PATTERN)
    review_policy: TaskGraphSemanticReviewPolicy
    requests: tuple[TaskGraphSemanticVerificationRequest, ...] = Field(
        min_length=1, max_length=2
    )
    results: tuple[TaskGraphSemanticVerificationResult, ...] = Field(
        min_length=1, max_length=2
    )
    host_disposition: TaskGraphSemanticVerificationDisposition
    settlement_sha256: str = Field(pattern=_SHA256_PATTERN)
    created_turn_id: str


class AuxiliarySemanticQuorumSettlementMutationResult(_Record):
    status: Literal["applied", "replayed"]
    settlement: StoredAuxiliarySemanticQuorumSettlement


class AuxiliarySemanticMaterialProjection(_Record):
    """构造终结语义 Prompt 所需的已认证输入。

    此投影有意排除该 revision 的基础 authority cards 和 capability catalog。
    它们属于应用侧的 Prompt 投影；Store 读取侧负责那些在进程重启后无法安全
    重建的值：已验证的终结提案、其正向基础谱系（如有），以及当前 revision 中
    每一对已封存的 Host ContextArtifact/source-card。
    """

    schema_version: Literal["auxiliary-v2-semantic-material-projection-v1"] = (
        "auxiliary-v2-semantic-material-projection-v1"
    )
    session_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_id: str = Field(pattern=_ID_PATTERN)
    goal_id: str = Field(pattern=_ID_PATTERN)
    auxiliary_graph_revision: int = Field(ge=1)
    structure_sha256: str = Field(pattern=_SHA256_PATTERN)
    terminal_subject: AuxiliaryNodeSubject
    terminal_completion_id: str = Field(pattern=_ID_PATTERN)
    terminal_work_run_id: str = Field(pattern=_ID_PATTERN)
    terminal_submitted_attempt_id: str = Field(pattern=_ID_PATTERN)
    terminal_verification_request_id: str = Field(pattern=_ID_PATTERN)
    terminal_output_revision: int = Field(ge=1)
    terminal_output_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    task_graph_proposal: InSessionTaskGraphRevisionProposal
    task_graph_proposal_sha256: str = Field(pattern=_SHA256_PATTERN)
    lineage: tuple[TaskGraphSemanticLineageProjection, ...] = Field(
        default=(), max_length=512
    )
    context_artifacts: tuple[PlanningContextArtifactProjection, ...] = Field(
        default=(), max_length=64
    )
    observation_source_cards: tuple[PlanningAuthoritySourceCard, ...] = Field(
        default=(), max_length=1_024
    )
    projection_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_projection(self) -> "AuxiliarySemanticMaterialProjection":
        if (
            self.terminal_subject.task_id != self.task_id
            or self.terminal_subject.auxiliary_graph_id != self.auxiliary_graph_id
            or self.terminal_subject.auxiliary_graph_revision
            != self.auxiliary_graph_revision
        ):
            raise ValueError("semantic terminal subject differs from its graph")
        artifact_ids = tuple(item.artifact_id for item in self.context_artifacts)
        artifact_aliases = tuple(
            item.artifact_alias for item in self.context_artifacts
        )
        card_aliases = tuple(item.alias for item in self.observation_source_cards)
        if (
            len(artifact_ids) != len(set(artifact_ids))
            or len(artifact_aliases) != len(set(artifact_aliases))
            or len(card_aliases) != len(set(card_aliases))
            or card_aliases != tuple(sorted(card_aliases))
        ):
            raise ValueError("semantic material aliases must be unique and canonical")
        referenced_aliases = {
            alias
            for artifact in self.context_artifacts
            for item in (*artifact.facts, *artifact.conflicts, *artifact.gaps)
            for alias in item.evidence_aliases
        } | {
            alias
            for artifact in self.context_artifacts
            for item in artifact.constraints
            for alias in item.authorization_aliases
        }
        if not referenced_aliases.issubset(card_aliases):
            raise ValueError("semantic artifacts reference an unprojected source card")
        if self.task_graph_proposal_sha256 != _sha256_value(
            self.task_graph_proposal.model_dump(mode="json")
        ):
            raise ValueError("semantic terminal proposal hash is invalid")
        if self.lineage:
            TaskGraphRevisionCandidate(
                proposal=self.task_graph_proposal,
                lineage=self.lineage,
            )
        expected = _sha256_value(
            self.model_dump(mode="json", exclude={"projection_sha256"})
        )
        if self.projection_sha256 != expected:
            raise ValueError("semantic material projection hash is invalid")
        return self

    @classmethod
    def create(cls, **values: object) -> "AuxiliarySemanticMaterialProjection":
        values = dict(values)
        proposal = values.get("task_graph_proposal")
        if not isinstance(proposal, InSessionTaskGraphRevisionProposal):
            proposal = InSessionTaskGraphRevisionProposal.model_validate(proposal)
            values["task_graph_proposal"] = proposal
        values["lineage"] = tuple(
            item
            if isinstance(item, TaskGraphSemanticLineageProjection)
            else TaskGraphSemanticLineageProjection.model_validate(item)
            for item in values.get("lineage", ())  # type: ignore[union-attr]
        )
        values["context_artifacts"] = tuple(values.get("context_artifacts", ()))
        values["observation_source_cards"] = tuple(
            sorted(
                values.get("observation_source_cards", ()),
                key=lambda item: item.alias,  # type: ignore[union-attr]
            )
        )
        values["task_graph_proposal_sha256"] = _sha256_value(
            proposal.model_dump(mode="json")
        )
        values["projection_sha256"] = "0" * 64
        provisional = cls.model_construct(**values)
        values["projection_sha256"] = _sha256_value(
            provisional.model_dump(mode="json", exclude={"projection_sha256"})
        )
        return cls.model_validate(values)


def project_auxiliary_semantic_material(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
) -> AuxiliarySemanticMaterialProjection:
    """从当前 graph 投影已验证的终结材料与证据材料。

    此读取不执行模型调用，也不接受调用方构造的 completion 或 artifact 标识。
    它从当前 graph 推导每一项，再复用依赖交付所用的同一组
    completion/artifact 验证器。
    """

    for name, value in (
        ("session_id", session_id),
        ("turn_id", turn_id),
        ("task_id", task_id),
    ):
        if not isinstance(value, str) or not value or len(value) > 200:
            raise ValueError(f"{name} must be a bounded durable identity")
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        _require_turn(conn, session_id=session_id, turn_id=turn_id)
        if conn.execute(
            "SELECT 1 FROM insession_task_turn_links WHERE session_id=? "
            "AND turn_id=? AND insession_task_id=?",
            (session_id, turn_id, task_id),
        ).fetchone() is None:
            raise AuxiliarySemanticVerificationPersistenceError(
                "semantic material Turn is not linked to its Task"
            )
        task = conn.execute(
            "SELECT current_graph_revision, current_status, state_version, "
            "created_turn_id, creation_source_start, creation_source_end, "
            "creation_source_sha256 FROM insession_tasks "
            "WHERE session_id=? AND insession_task_id=?",
            (session_id, task_id),
        ).fetchone()
        if task is None or str(task["current_status"]) in {"completed", "cancelled"}:
            raise AuxiliarySemanticVerificationPersistenceError(
                "semantic material Task is terminal or missing"
            )
        details = _load_auxiliary_graph(conn, session_id, task_id)
        current_task_graph_revision = (
            int(task["current_graph_revision"])
            if task["current_graph_revision"] is not None
            else None
        )
        if current_task_graph_revision != details.base_task_graph_revision:
            raise AuxiliarySemanticVerificationStaleAuthority(
                "semantic material graph base differs from the current TaskGraph"
            )
        if (
            details.goal_status != "active"
            or details.revision_status != "active"
            or details.goal is None
            or details.budget is None
            or details.authority_snapshot is None
        ):
            raise AuxiliarySemanticVerificationPersistenceError(
                "semantic material requires current active formal authority"
            )

        nodes_by_id = {
            node.auxiliary_node_id: node for node in details.nodes
        }
        terminal = nodes_by_id.get(details.terminal_auxiliary_node_id)
        if (
            terminal is None
            or terminal.executor_kind != "terminal_planner"
            or terminal.status != "completed"
        ):
            raise AuxiliarySemanticVerificationPersistenceError(
                "semantic material has no completed current terminal planner"
            )
        terminal_subject = AuxiliaryNodeSubject(
            task_id=task_id,
            auxiliary_graph_id=details.auxiliary_graph_id,
            auxiliary_graph_revision=details.auxiliary_graph_revision,
            node_id=terminal.auxiliary_node_id,
            node_revision=terminal.node_revision,
        )
        terminal_output = _resolve_model_dependency(
            conn,
            details=details,
            producer=terminal,
            subject=terminal_subject,
        )
        terminal_binding = conn.execute(
            "SELECT work_run_id, submitted_attempt_id, "
            "verification_request_id, output_revision "
            "FROM insession_auxiliary_node_completions_v2 "
            "WHERE completion_id=? AND session_id=? "
            "AND insession_task_id=? AND auxiliary_graph_id=? "
            "AND auxiliary_graph_revision=? AND auxiliary_node_id=?",
            (
                terminal_output.completion_id,
                session_id,
                task_id,
                details.auxiliary_graph_id,
                details.auxiliary_graph_revision,
                terminal.auxiliary_node_id,
            ),
        ).fetchone()
        if (
            terminal_binding is None
            or int(terminal_binding["output_revision"])
            != terminal_output.output_revision
        ):
            raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
                "semantic terminal completion lost its candidate binding"
            )
        try:
            if details.base_task_graph_revision is None:
                proposal = InSessionTaskGraphRevisionProposal.model_validate_json(
                    terminal_output.content
                )
                lineage: tuple[TaskGraphSemanticLineageProjection, ...] = ()
                expected_terminal_content = proposal.model_dump_json()
            else:
                candidate = TaskGraphRevisionCandidate.model_validate_json(
                    terminal_output.content
                )
                proposal = candidate.proposal
                lineage = candidate.lineage
                expected_terminal_content = candidate.model_dump_json()
        except (TypeError, ValueError, RecursionError) as exc:
            raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
                "semantic terminal OutputWindow has the wrong typed TaskGraph envelope"
            ) from exc
        if terminal_output.content != expected_terminal_content:
            raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
                "semantic terminal OutputWindow is not canonical"
            )

        try:
            terminal_support = _build_auxiliary_terminal_semantic_support(
                conn,
                session_id=session_id,
                invocation_turn_id=turn_id,
                task_id=task_id,
                task_authority=task,
            )
        except AuxiliaryTerminalValidationContextError as exc:
            raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
                "semantic material no longer matches terminal source authority"
            ) from exc
        projection = AuxiliarySemanticMaterialProjection.create(
            session_id=session_id,
            task_id=task_id,
            auxiliary_graph_id=details.auxiliary_graph_id,
            goal_id=details.goal_id,
            auxiliary_graph_revision=details.auxiliary_graph_revision,
            structure_sha256=details.structure_sha256,
            terminal_subject=terminal_subject,
            terminal_completion_id=terminal_output.completion_id,
            terminal_work_run_id=str(terminal_binding["work_run_id"]),
            terminal_submitted_attempt_id=str(
                terminal_binding["submitted_attempt_id"]
            ),
            terminal_verification_request_id=str(
                terminal_binding["verification_request_id"]
            ),
            terminal_output_revision=terminal_output.output_revision,
            terminal_output_snapshot_sha256=(
                terminal_output.output_snapshot_sha256
            ),
            task_graph_proposal=proposal,
            lineage=lineage,
            context_artifacts=terminal_support.context_artifacts,
            observation_source_cards=terminal_support.observation_source_cards,
        )
        conn.commit()
        return projection
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def _load_host_observation_source_cards(
    conn: sqlite3.Connection,
    *,
    artifact_id: str,
    expected_projection: PlanningContextArtifactProjection,
) -> tuple[PlanningAuthoritySourceCard, ...]:
    row = conn.execute(
        "SELECT observation.snapshot_json, observation.snapshot_sha256 "
        "FROM insession_auxiliary_planning_context_artifacts AS artifact "
        "JOIN insession_auxiliary_observations AS observation "
        "ON observation.observation_id=artifact.producer_primitive_call_id "
        "WHERE artifact.artifact_id=?",
        (artifact_id,),
    ).fetchone()
    if row is None:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic Host artifact lost its sealed observation"
        )
    observation_json = str(row["snapshot_json"])
    try:
        observation = json.loads(observation_json)
        prompt_inputs = observation["result"]["prompt_inputs"]
        stored_projection = PlanningContextArtifactProjection.model_validate(
            prompt_inputs["context_artifact"]
        )
        cards = tuple(
            PlanningAuthoritySourceCard.model_validate(item)
            for item in prompt_inputs["source_cards"]
        )
    except (KeyError, TypeError, ValueError, RecursionError) as exc:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic Host observation prompt projection is invalid"
        ) from exc
    if (
        observation_json != _canonical_json(observation)
        or _sha256_text(observation_json) != str(row["snapshot_sha256"])
        or stored_projection != expected_projection
        or len({item.alias for item in cards}) != len(cards)
    ):
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic Host observation prompt projection is corrupt"
        )
    return cards


def derive_auxiliary_semantic_review_policy(
    *,
    prompt_payload: TaskGraphSemanticVerificationPromptPayload,
) -> TaskGraphSemanticReviewPolicy:
    """从一份精确冻结的 Prompt 载荷推导保守的审查事实。"""

    if not isinstance(prompt_payload, TaskGraphSemanticVerificationPromptPayload):
        raise TypeError(
            "prompt_payload must be TaskGraphSemanticVerificationPromptPayload"
        )
    prompt_payload = TaskGraphSemanticVerificationPromptPayload.model_validate(
        prompt_payload.model_dump(mode="json")
    )
    base = prompt_payload.base_task_graph
    modifies_executed = False
    if base is not None:
        base_by_alias = {item.node_alias: item for item in base.nodes}
        if any(item.status is not InSessionTaskStatus.PROPOSED for item in base.nodes):
            reused = {
                item.base_node_alias
                for item in prompt_payload.lineage
                if item.disposition is TaskGraphSemanticLineageDisposition.REUSE
                and item.base_node_alias is not None
            }
            modifies_executed = reused != set(base_by_alias) or any(
                item.disposition is not TaskGraphSemanticLineageDisposition.REUSE
                for item in prompt_payload.lineage
            )
    policy_source_sha256 = _sha256_value(
        {
            "schema_version": "auxiliary-semantic-review-policy-source-v1",
            "prompt_payload_sha256": prompt_payload.payload_sha256,
            "authority_projection_sha256": (
                prompt_payload.authority.projection_sha256
            ),
            "context_artifact_projection_sha256s": [
                item.projection_sha256 for item in prompt_payload.context_artifacts
            ],
            "capability_projection_sha256": (
                prompt_payload.capabilities.projection_sha256
            ),
            "base_projection_sha256": (
                None if base is None else base.projection_sha256
            ),
        }
    )
    return TaskGraphSemanticReviewPolicy.create(
        policy_source_sha256=policy_source_sha256,
        distinct_document_count=len(
            {
                # 新卡片按冻结的 Document 标识对所有分块/读取路径分组。
                # 旧卡片没有分组，并通过回退逻辑精确保留其保守的逐卡重放行为。
                item.document_group_alias or item.alias
                for item in prompt_payload.authority.cards
                if item.source_kind is PlanningAuthoritySourceKind.DOCUMENT
            }
        ),
        has_visual_input=any(
            item.source_kind is PlanningAuthoritySourceKind.VISUAL
            for item in prompt_payload.authority.cards
        ),
        requires_protected_effect=any(
            item.available and item.effect is PlanningCapabilityEffect.PROTECTED
            for item in prompt_payload.capabilities.capabilities
        ),
        modifies_executed_task_graph=modifies_executed,
    )


def freeze_auxiliary_semantic_capability_catalog(
    deps: StoreDeps,
    *,
    command: FreezeAuxiliarySemanticCapabilityCatalogCommand,
) -> AuxiliarySemanticCapabilityCatalogMutationResult:
    command = _validated_command(
        FreezeAuxiliarySemanticCapabilityCatalogCommand,
        command,
        "capability catalog command",
    )
    deps.init_db()
    try:
        with deps.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing_rows = conn.execute(
                "SELECT projection_sha256 FROM "
                "insession_auxiliary_capability_catalog_projections "
                "WHERE capability_catalog_snapshot_id=? ORDER BY projection_sha256",
                (command.catalog.capability_catalog_snapshot_id,),
            ).fetchall()
            if existing_rows:
                if len(existing_rows) != 1 or str(
                    existing_rows[0]["projection_sha256"]
                ) != command.catalog.projection_sha256:
                    raise AuxiliarySemanticVerificationIdentityCollision(
                        "capability catalog snapshot ID crossed immutable content"
                    )
                stored = _load_catalog(
                    conn,
                    snapshot_id=command.catalog.capability_catalog_snapshot_id,
                    projection_sha256=command.catalog.projection_sha256,
                )
                expected = _catalog_record(command)
                if stored != expected:
                    raise AuxiliarySemanticVerificationIdentityCollision(
                        "capability catalog replay crossed its frozen graph authority"
                    )
                return AuxiliarySemanticCapabilityCatalogMutationResult(
                    status="replayed", catalog=stored
                )
            _require_turn(
                conn,
                session_id=command.session_id,
                turn_id=command.created_turn_id,
            )
            details = _require_current_guard(
                conn,
                session_id=command.session_id,
                task_id=command.task_id,
                graph_id=command.auxiliary_graph_id,
                goal_id=command.goal_id,
                revision=command.auxiliary_graph_revision,
                expected_control=command.expected_control_state_version,
                expected_goal=command.expected_goal_state_version,
                expected_revision=command.expected_revision_state_version,
                expected_budget=command.expected_budget_state_version,
            )
            if (
                details.authority_snapshot_id
                != command.expected_authority_snapshot_id
                or details.authority_snapshot_sha256
                != command.expected_authority_snapshot_sha256
                or details.structure_sha256 != command.expected_structure_sha256
            ):
                raise AuxiliarySemanticVerificationStaleAuthority(
                    "capability catalog graph authority changed"
                )
            now = deps.now()
            projection_json = _model_json(command.catalog)
            conn.execute(
                "INSERT INTO insession_auxiliary_capability_catalog_projections "
                "(capability_catalog_snapshot_id, "
                "capability_catalog_snapshot_sha256, projection_sha256, "
                "session_id, insession_task_id, auxiliary_graph_id, goal_id, "
                "auxiliary_graph_revision, projection_json, created_turn_id, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    command.catalog.capability_catalog_snapshot_id,
                    command.catalog.capability_catalog_snapshot_sha256,
                    command.catalog.projection_sha256,
                    command.session_id,
                    command.task_id,
                    command.auxiliary_graph_id,
                    command.goal_id,
                    command.auxiliary_graph_revision,
                    projection_json,
                    command.created_turn_id,
                    now,
                ),
            )
            for ordinal, descriptor in enumerate(command.catalog.capabilities):
                descriptor_json = _model_json(descriptor)
                conn.execute(
                    "INSERT INTO insession_auxiliary_capability_catalog_items "
                    "(capability_catalog_snapshot_id, projection_sha256, "
                    "capability_alias, ordinal, descriptor_json, "
                    "descriptor_sha256) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        command.catalog.capability_catalog_snapshot_id,
                        command.catalog.projection_sha256,
                        descriptor.capability_alias,
                        ordinal,
                        descriptor_json,
                        _sha256_text(descriptor_json),
                    ),
                )
            stored = _load_catalog(
                conn,
                snapshot_id=command.catalog.capability_catalog_snapshot_id,
                projection_sha256=command.catalog.projection_sha256,
            )
            return AuxiliarySemanticCapabilityCatalogMutationResult(
                status="applied", catalog=stored
            )
    except sqlite3.IntegrityError as exc:
        raise AuxiliarySemanticVerificationPersistenceError(
            "capability catalog violated v62 durable authority"
        ) from exc


def commit_auxiliary_semantic_verification_request(
    deps: StoreDeps,
    *,
    command: CommitAuxiliarySemanticVerificationRequestCommand,
) -> AuxiliarySemanticVerificationRequestMutationResult:
    command = _validated_command(
        CommitAuxiliarySemanticVerificationRequestCommand,
        command,
        "semantic request command",
    )
    deps.init_db()
    request = command.request
    try:
        with deps.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT 1 FROM insession_auxiliary_semantic_verification_requests "
                "WHERE verification_request_id=?",
                (request.verification_request_id,),
            ).fetchone()
            if existing is not None:
                stored = _load_request(conn, request.verification_request_id)
                expected = _request_record(command)
                if stored != expected:
                    raise AuxiliarySemanticVerificationIdentityCollision(
                        "semantic request ID crossed immutable content"
                    )
                _validate_frozen_request_authority(conn, request=stored.request)
                return AuxiliarySemanticVerificationRequestMutationResult(
                    status="replayed", record=stored
                )
            _require_turn(
                conn,
                session_id=command.session_id,
                turn_id=command.created_turn_id,
            )
            _require_current_request_guard(conn, command=command)
            _validate_frozen_request_authority(conn, request=request)
            _insert_request(
                conn,
                request=request,
                created_turn_id=command.created_turn_id,
                now=deps.now(),
            )
            stored = _load_request(conn, request.verification_request_id)
            return AuxiliarySemanticVerificationRequestMutationResult(
                status="applied", record=stored
            )
    except sqlite3.IntegrityError as exc:
        raise AuxiliarySemanticVerificationPersistenceError(
            "semantic request violated v62 durable authority"
        ) from exc


def commit_auxiliary_semantic_verification_result(
    deps: StoreDeps,
    *,
    command: CommitAuxiliarySemanticVerificationResultCommand,
) -> AuxiliarySemanticVerificationResultMutationResult:
    command = _validated_command(
        CommitAuxiliarySemanticVerificationResultCommand,
        command,
        "semantic result command",
    )
    deps.init_db()
    result = command.result
    try:
        with deps.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            by_id = conn.execute(
                "SELECT verification_result_id FROM "
                "insession_auxiliary_semantic_verification_results "
                "WHERE verification_result_id=? OR verification_request_id=?",
                (result.verification_result_id, result.verification_request_id),
            ).fetchall()
            if by_id:
                if len(by_id) != 1 or str(by_id[0]["verification_result_id"]) != (
                    result.verification_result_id
                ):
                    raise AuxiliarySemanticVerificationIdentityCollision(
                        "semantic result/request identity crossed immutable content"
                    )
                stored = _load_result(conn, result.verification_result_id)
                if (
                    stored.result != result
                    or stored.created_turn_id != command.created_turn_id
                    or stored.session_id != command.session_id
                ):
                    raise AuxiliarySemanticVerificationIdentityCollision(
                        "semantic result ID crossed immutable content"
                    )
                return AuxiliarySemanticVerificationResultMutationResult(
                    status="replayed", record=stored
                )
            request_record = _load_request(conn, result.verification_request_id)
            _require_turn(
                conn,
                session_id=command.session_id,
                turn_id=command.created_turn_id,
            )
            _require_current_result_guard(
                conn,
                command=command,
                request=request_record.request,
            )
            try:
                validate_task_graph_semantic_verification_result(
                    request=request_record.request,
                    result=result,
                )
            except ValueError as exc:
                raise AuxiliarySemanticVerificationPersistenceError(
                    "semantic result lost its exact request binding"
                ) from exc
            now = deps.now()
            request = request_record.request
            result_json = _model_json(result)
            conn.execute(
                "INSERT INTO insession_auxiliary_semantic_verification_results "
                "(verification_result_id, session_id, insession_task_id, "
                "auxiliary_graph_id, goal_id, auxiliary_graph_revision, "
                "verification_request_id, request_binding_sha256, "
                "prompt_payload_sha256, review_policy_sha256, logical_call_id, "
                "verification_profile_id, reviewer_ordinal, "
                "required_reviewer_count, items_json, host_disposition, "
                "result_json, result_sha256, created_turn_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.verification_result_id,
                    request.goal.session_id,
                    request.goal.task_id,
                    request.goal.auxiliary_graph_id,
                    request.goal.goal_id,
                    request.auxiliary_graph_revision,
                    result.verification_request_id,
                    result.request_binding_sha256,
                    request.prompt_payload.payload_sha256,
                    request.review_policy.policy_sha256,
                    result.logical_call_id,
                    result.verification_profile_id,
                    result.reviewer_ordinal,
                    result.required_reviewer_count,
                    _canonical_json(
                        [item.model_dump(mode="json") for item in result.items]
                    ),
                    result.host_disposition.value,
                    result_json,
                    result.result_sha256,
                    command.created_turn_id,
                    now,
                ),
            )
            stored = _load_result(conn, result.verification_result_id)
            return AuxiliarySemanticVerificationResultMutationResult(
                status="applied", record=stored
            )
    except sqlite3.IntegrityError as exc:
        raise AuxiliarySemanticVerificationPersistenceError(
            "semantic result violated v62 durable authority"
        ) from exc


def settle_auxiliary_semantic_verification_quorum(
    deps: StoreDeps,
    *,
    command: SettleAuxiliarySemanticVerificationQuorumCommand,
) -> AuxiliarySemanticQuorumSettlementMutationResult:
    command = _validated_command(
        SettleAuxiliarySemanticVerificationQuorumCommand,
        command,
        "semantic quorum command",
    )
    try:
        ordered_results = validate_task_graph_semantic_verification_quorum(
            requests=command.requests,
            results=command.results,
        )
    except ValueError as exc:
        raise AuxiliarySemanticVerificationPersistenceError(
            "semantic quorum is incomplete or not independently bound"
        ) from exc
    ordered_requests = tuple(
        sorted(command.requests, key=lambda item: item.reviewer_ordinal)
    )
    if tuple(item.reviewer_ordinal for item in ordered_results) != tuple(
        item.reviewer_ordinal for item in ordered_requests
    ):
        raise AuxiliarySemanticVerificationPersistenceError(
            "semantic quorum request/result ordinals differ"
        )
    expected = _build_settlement(
        command=command,
        requests=ordered_requests,
        results=ordered_results,
    )
    deps.init_db()
    try:
        with deps.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing_by_id = conn.execute(
                "SELECT settlement_id FROM "
                "insession_auxiliary_semantic_quorum_settlements "
                "WHERE settlement_id=?",
                (command.settlement_id,),
            ).fetchone()
            if existing_by_id is not None:
                stored = _load_settlement(conn, command.settlement_id)
                if stored != expected:
                    raise AuxiliarySemanticVerificationIdentityCollision(
                        "semantic settlement ID crossed immutable content"
                    )
                return AuxiliarySemanticQuorumSettlementMutationResult(
                    status="replayed", settlement=stored
                )
            first = ordered_requests[0]
            existing_scope = conn.execute(
                "SELECT settlement_id FROM "
                "insession_auxiliary_semantic_quorum_settlements "
                "WHERE session_id=? AND insession_task_id=? "
                "AND auxiliary_graph_id=? AND goal_id=? "
                "AND auxiliary_graph_revision=? "
                "AND frozen_prompt_payload_sha256=?",
                (
                    first.goal.session_id,
                    first.goal.task_id,
                    first.goal.auxiliary_graph_id,
                    first.goal.goal_id,
                    first.auxiliary_graph_revision,
                    first.prompt_payload.payload_sha256,
                ),
            ).fetchone()
            if existing_scope is not None:
                raise AuxiliarySemanticVerificationIdentityCollision(
                    "frozen semantic prompt already owns another settlement"
                )
            for request, result in zip(
                ordered_requests, ordered_results, strict=True
            ):
                if _load_request(conn, request.verification_request_id).request != request:
                    raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
                        "settlement request differs from its stored contract"
                    )
                if _load_result(conn, result.verification_result_id).result != result:
                    raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
                        "settlement result differs from its stored contract"
                    )
            _require_turn(
                conn,
                session_id=command.session_id,
                turn_id=command.created_turn_id,
            )
            _require_current_settlement_guard(
                conn,
                command=command,
                request=ordered_requests[0],
            )
            _insert_settlement(conn, settlement=expected, now=deps.now())
            stored = _load_settlement(conn, command.settlement_id)
            return AuxiliarySemanticQuorumSettlementMutationResult(
                status="applied", settlement=stored
            )
    except sqlite3.IntegrityError as exc:
        raise AuxiliarySemanticVerificationPersistenceError(
            "semantic quorum violated v62 durable authority"
        ) from exc


def get_auxiliary_semantic_verification_request(
    deps: StoreDeps,
    *,
    session_id: str,
    verification_request_id: str,
) -> StoredAuxiliarySemanticVerificationRequest | None:
    """加载并重新认证一个不可变的语义审查请求。"""

    _require_id("session_id", session_id)
    _require_id("verification_request_id", verification_request_id)
    deps.init_db()
    with deps.connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM insession_auxiliary_semantic_verification_requests "
            "WHERE verification_request_id=?",
            (verification_request_id,),
        ).fetchone()
        if exists is None:
            return None
        record = _load_request(conn, verification_request_id)
        if record.session_id != session_id:
            raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
                "semantic verification request crossed Session authority"
            )
        _validate_frozen_request_authority(conn, request=record.request)
        return record


def get_auxiliary_semantic_capability_catalog(
    deps: StoreDeps,
    *,
    session_id: str,
    capability_catalog_snapshot_id: str,
    projection_sha256: str,
) -> StoredAuxiliarySemanticCapabilityCatalog | None:
    """加载一个不可变 catalog 及其原始 Turn 标识。"""

    _require_id("session_id", session_id)
    _require_id(
        "capability_catalog_snapshot_id",
        capability_catalog_snapshot_id,
    )
    _require_sha("projection_sha256", projection_sha256)
    deps.init_db()
    with deps.connect() as conn:
        rows = conn.execute(
            "SELECT projection_sha256 FROM "
            "insession_auxiliary_capability_catalog_projections "
            "WHERE capability_catalog_snapshot_id=? "
            "ORDER BY projection_sha256",
            (capability_catalog_snapshot_id,),
        ).fetchall()
        if not rows:
            return None
        if (
            len(rows) != 1
            or str(rows[0]["projection_sha256"]) != projection_sha256
        ):
            raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
                "semantic capability catalog identity is ambiguous"
            )
        record = _load_catalog(
            conn,
            snapshot_id=capability_catalog_snapshot_id,
            projection_sha256=projection_sha256,
        )
        if record.session_id != session_id:
            raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
                "semantic capability catalog crossed Session authority"
            )
        return record


def get_auxiliary_semantic_verification_result(
    deps: StoreDeps,
    *,
    session_id: str,
    verification_result_id: str,
) -> StoredAuxiliarySemanticVerificationResult | None:
    """加载一个结果及其完整的请求/冻结权威链。"""

    _require_id("session_id", session_id)
    _require_id("verification_result_id", verification_result_id)
    deps.init_db()
    with deps.connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM insession_auxiliary_semantic_verification_results "
            "WHERE verification_result_id=?",
            (verification_result_id,),
        ).fetchone()
        if exists is None:
            return None
        record = _load_result(conn, verification_result_id)
        if record.session_id != session_id:
            raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
                "semantic verification result crossed Session authority"
            )
        return record


def get_auxiliary_semantic_quorum_settlement(
    deps: StoreDeps,
    *,
    session_id: str,
    task_id: str,
    auxiliary_graph_id: str,
    goal_id: str,
    auxiliary_graph_revision: int,
    frozen_prompt_payload_sha256: str,
) -> StoredAuxiliarySemanticQuorumSettlement | None:
    _require_id("session_id", session_id)
    _require_id("task_id", task_id)
    _require_id("auxiliary_graph_id", auxiliary_graph_id)
    _require_id("goal_id", goal_id)
    if auxiliary_graph_revision < 1:
        raise ValueError("auxiliary_graph_revision must be positive")
    _require_sha("frozen_prompt_payload_sha256", frozen_prompt_payload_sha256)
    deps.init_db()
    with deps.connect() as conn:
        return _load_auxiliary_semantic_quorum_settlement(
            conn,
            session_id=session_id,
            task_id=task_id,
            auxiliary_graph_id=auxiliary_graph_id,
            goal_id=goal_id,
            auxiliary_graph_revision=auxiliary_graph_revision,
            frozen_prompt_payload_sha256=frozen_prompt_payload_sha256,
        )


def _load_auxiliary_semantic_quorum_settlement(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    task_id: str,
    auxiliary_graph_id: str,
    goal_id: str,
    auxiliary_graph_revision: int,
    frozen_prompt_payload_sha256: str,
) -> StoredAuxiliarySemanticQuorumSettlement | None:
    """在调用方持有的 SQLite 事务中加载一项精确结算。"""

    rows = conn.execute(
        "SELECT settlement_id FROM "
        "insession_auxiliary_semantic_quorum_settlements "
        "WHERE session_id=? AND insession_task_id=? "
        "AND auxiliary_graph_id=? AND goal_id=? "
        "AND auxiliary_graph_revision=? "
        "AND frozen_prompt_payload_sha256=?",
        (
            session_id,
            task_id,
            auxiliary_graph_id,
            goal_id,
            auxiliary_graph_revision,
            frozen_prompt_payload_sha256,
        ),
    ).fetchall()
    if not rows:
        return None
    if len(rows) != 1:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic prompt owns multiple quorum settlements"
        )
    return _load_settlement(conn, str(rows[0]["settlement_id"]))


def _validated_command(model_type: type[_Record], value: object, label: str):
    if not isinstance(value, model_type):
        raise TypeError(f"{label} must be {model_type.__name__}")
    try:
        return model_type.model_validate(value.model_dump(mode="json"))
    except (TypeError, ValueError) as exc:
        raise AuxiliarySemanticVerificationPersistenceError(
            f"{label} is not self-authenticating"
        ) from exc


def _catalog_record(
    command: FreezeAuxiliarySemanticCapabilityCatalogCommand,
) -> StoredAuxiliarySemanticCapabilityCatalog:
    return StoredAuxiliarySemanticCapabilityCatalog(
        session_id=command.session_id,
        task_id=command.task_id,
        auxiliary_graph_id=command.auxiliary_graph_id,
        goal_id=command.goal_id,
        auxiliary_graph_revision=command.auxiliary_graph_revision,
        projection=command.catalog,
        created_turn_id=command.created_turn_id,
    )


def _request_record(
    command: CommitAuxiliarySemanticVerificationRequestCommand,
) -> StoredAuxiliarySemanticVerificationRequest:
    request = command.request
    return StoredAuxiliarySemanticVerificationRequest(
        session_id=command.session_id,
        task_id=request.goal.task_id,
        auxiliary_graph_id=request.goal.auxiliary_graph_id,
        goal_id=request.goal.goal_id,
        auxiliary_graph_revision=request.auxiliary_graph_revision,
        request=request,
        created_turn_id=command.created_turn_id,
    )


def _require_turn(
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
        raise AuxiliarySemanticVerificationStaleAuthority(
            "semantic mutation requires its exact running Turn"
        )


def _require_current_guard(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    task_id: str,
    graph_id: str,
    goal_id: str,
    revision: int,
    expected_control: int,
    expected_goal: int,
    expected_revision: int,
    expected_budget: int,
):
    try:
        details = _load_auxiliary_graph(conn, session_id, task_id)
    except AuxiliaryGraphPersistenceError as exc:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "current AuxiliaryGraph authority is corrupt"
        ) from exc
    if (
        details.auxiliary_graph_id != graph_id
        or details.goal_id != goal_id
        or details.auxiliary_graph_revision != revision
        or details.control_state_version != expected_control
        or details.goal_state_version != expected_goal
        or details.revision_state_version != expected_revision
        or details.budget_state_version != expected_budget
        or details.revision is None
        or details.authority_snapshot is None
        or details.budget is None
    ):
        raise AuxiliarySemanticVerificationStaleAuthority(
            "semantic authority CAS no longer matches the current graph"
        )
    if details.goal_status not in {"active", "proposal_ready", "gapped_ready"}:
        raise AuxiliarySemanticVerificationStaleAuthority(
            "current goal cannot enter semantic verification"
        )
    if details.revision_status not in {"active", "proposal_ready", "gapped_ready"}:
        raise AuxiliarySemanticVerificationStaleAuthority(
            "current revision cannot enter semantic verification"
        )
    return details


def _require_current_request_guard(
    conn: sqlite3.Connection,
    *,
    command: CommitAuxiliarySemanticVerificationRequestCommand,
) -> None:
    request = command.request
    details = _require_current_guard(
        conn,
        session_id=command.session_id,
        task_id=request.goal.task_id,
        graph_id=request.goal.auxiliary_graph_id,
        goal_id=request.goal.goal_id,
        revision=request.auxiliary_graph_revision,
        expected_control=command.expected_control_state_version,
        expected_goal=command.expected_goal_state_version,
        expected_revision=command.expected_revision_state_version,
        expected_budget=command.expected_budget_state_version,
    )
    if details.goal != request.goal or details.budget != request.budget:
        raise AuxiliarySemanticVerificationStaleAuthority(
            "semantic request goal or budget changed"
        )


def _require_current_result_guard(
    conn: sqlite3.Connection,
    *,
    command: CommitAuxiliarySemanticVerificationResultCommand,
    request: TaskGraphSemanticVerificationRequest,
) -> None:
    details = _require_current_guard(
        conn,
        session_id=command.session_id,
        task_id=request.goal.task_id,
        graph_id=request.goal.auxiliary_graph_id,
        goal_id=request.goal.goal_id,
        revision=request.auxiliary_graph_revision,
        expected_control=command.expected_control_state_version,
        expected_goal=command.expected_goal_state_version,
        expected_revision=command.expected_revision_state_version,
        expected_budget=command.expected_budget_state_version,
    )
    if details.goal != request.goal or details.budget != request.budget:
        raise AuxiliarySemanticVerificationStaleAuthority(
            "semantic result request authority is stale"
        )
    _validate_frozen_request_authority(conn, request=request)


def _require_current_settlement_guard(
    conn: sqlite3.Connection,
    *,
    command: SettleAuxiliarySemanticVerificationQuorumCommand,
    request: TaskGraphSemanticVerificationRequest,
) -> None:
    details = _require_current_guard(
        conn,
        session_id=command.session_id,
        task_id=request.goal.task_id,
        graph_id=request.goal.auxiliary_graph_id,
        goal_id=request.goal.goal_id,
        revision=request.auxiliary_graph_revision,
        expected_control=command.expected_control_state_version,
        expected_goal=command.expected_goal_state_version,
        expected_revision=command.expected_revision_state_version,
        expected_budget=command.expected_budget_state_version,
    )
    if details.goal != request.goal or details.budget != request.budget:
        raise AuxiliarySemanticVerificationStaleAuthority(
            "semantic settlement request authority is stale"
        )
    _validate_frozen_request_authority(conn, request=request)


def _load_catalog(
    conn: sqlite3.Connection,
    *,
    snapshot_id: str,
    projection_sha256: str,
) -> StoredAuxiliarySemanticCapabilityCatalog:
    row = conn.execute(
        "SELECT * FROM insession_auxiliary_capability_catalog_projections "
        "WHERE capability_catalog_snapshot_id=? AND projection_sha256=?",
        (snapshot_id, projection_sha256),
    ).fetchone()
    if row is None:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic capability catalog is missing"
        )
    projection_json = str(row["projection_json"])
    try:
        projection = PlanningCapabilityCatalogProjection.model_validate_json(
            projection_json
        )
    except (TypeError, ValueError) as exc:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic capability projection JSON is invalid"
        ) from exc
    item_rows = conn.execute(
        "SELECT * FROM insession_auxiliary_capability_catalog_items "
        "WHERE capability_catalog_snapshot_id=? AND projection_sha256=? "
        "ORDER BY ordinal",
        (snapshot_id, projection_sha256),
    ).fetchall()
    descriptors = []
    for ordinal, item_row in enumerate(item_rows):
        descriptor_json = str(item_row["descriptor_json"])
        try:
            descriptor = type(projection.capabilities[0]).model_validate_json(
                descriptor_json
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
                "semantic capability descriptor JSON is invalid"
            ) from exc
        if (
            int(item_row["ordinal"]) != ordinal
            or str(item_row["capability_alias"]) != descriptor.capability_alias
            or _model_json(descriptor) != descriptor_json
            or _sha256_text(descriptor_json) != str(item_row["descriptor_sha256"])
        ):
            raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
                "semantic capability descriptor mirror is corrupt"
            )
        descriptors.append(descriptor)
    if (
        projection.capability_catalog_snapshot_id != snapshot_id
        or projection.capability_catalog_snapshot_sha256
        != str(row["capability_catalog_snapshot_sha256"])
        or projection.projection_sha256 != projection_sha256
        or _model_json(projection) != projection_json
        or tuple(descriptors) != projection.capabilities
    ):
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic capability catalog row and projection differ"
        )
    return StoredAuxiliarySemanticCapabilityCatalog(
        session_id=str(row["session_id"]),
        task_id=str(row["insession_task_id"]),
        auxiliary_graph_id=str(row["auxiliary_graph_id"]),
        goal_id=str(row["goal_id"]),
        auxiliary_graph_revision=int(row["auxiliary_graph_revision"]),
        projection=projection,
        created_turn_id=str(row["created_turn_id"]),
    )


def _validate_frozen_request_authority(
    conn: sqlite3.Connection,
    *,
    request: TaskGraphSemanticVerificationRequest,
) -> None:
    goal = request.goal
    revision = conn.execute(
        "SELECT * FROM insession_auxiliary_graph_revision_snapshots "
        "WHERE session_id=? AND insession_task_id=? AND auxiliary_graph_id=? "
        "AND auxiliary_graph_revision=? AND goal_id=?",
        (
            goal.session_id,
            goal.task_id,
            goal.auxiliary_graph_id,
            request.auxiliary_graph_revision,
            goal.goal_id,
        ),
    ).fetchone()
    goal_row = conn.execute(
        "SELECT * FROM insession_auxiliary_graph_goals WHERE session_id=? "
        "AND insession_task_id=? AND auxiliary_graph_id=? AND goal_id=?",
        (goal.session_id, goal.task_id, goal.auxiliary_graph_id, goal.goal_id),
    ).fetchone()
    if (
        revision is None
        or goal_row is None
        or str(revision["structure_contract_version"])
        != "auxiliary-graph-revision-v2"
        or str(revision["structure_sha256"])
        != request.auxiliary_graph_structure_sha256
        or str(revision["authority_snapshot_id"])
        != request.authority_projection.authority_snapshot_id
        or str(revision["authority_snapshot_sha256"])
        != request.authority_projection.authority_snapshot_sha256
    ):
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic request revision/authority snapshot binding is missing"
        )
    immutable_goal = (
        str(goal_row["session_id"]) == goal.session_id
        and str(goal_row["insession_task_id"]) == goal.task_id
        and str(goal_row["auxiliary_graph_id"]) == goal.auxiliary_graph_id
        and str(goal_row["goal_id"]) == goal.goal_id
        and goal_row["base_task_graph_revision"] == goal.base_task_graph_revision
        and int(goal_row["target_task_graph_revision"])
        == goal.target_task_graph_revision
        and str(goal_row["creation_turn_id"]) == goal.creation_turn_id
        and str(goal_row["authorization_manifest_id"])
        == goal.authorization_manifest_id
        and str(goal_row["budget_ledger_id"]) == goal.budget_ledger_id
    )
    if not immutable_goal:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic request durable goal identity is corrupt"
        )
    try:
        authority = _load_formal_authority_snapshot(
            conn,
            authority_snapshot_id=str(revision["authority_snapshot_id"]),
            expected_session_id=goal.session_id,
            expected_task_id=goal.task_id,
            expected_auxiliary_graph_id=goal.auxiliary_graph_id,
            expected_goal_id=goal.goal_id,
            expected_source_turn_id=str(revision["source_turn_id"]),
            expected_snapshot_sha256=str(revision["authority_snapshot_sha256"]),
        )
    except AuxiliaryGraphPersistenceError as exc:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic request private authority snapshot is corrupt"
        ) from exc
    budget_row = conn.execute(
        "SELECT snapshot_json, snapshot_sha256 FROM "
        "insession_auxiliary_goal_budget_snapshots WHERE goal_id=? "
        "AND budget_ledger_id=? AND state_version=?",
        (
            goal.goal_id,
            goal.budget_ledger_id,
            request.budget.state_version,
        ),
    ).fetchone()
    if budget_row is None:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic request budget snapshot is missing"
        )
    try:
        budget = PlanningEpisodeBudget.model_validate_json(
            str(budget_row["snapshot_json"])
        )
    except (TypeError, ValueError) as exc:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic request budget snapshot is invalid"
        ) from exc
    if (
        budget != request.budget
        or _model_json(budget) != str(budget_row["snapshot_json"])
        or budget.snapshot_sha256 != str(budget_row["snapshot_sha256"])
    ):
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic request budget differs from its immutable snapshot"
        )
    catalog = _load_catalog(
        conn,
        snapshot_id=request.prompt_payload.capabilities.capability_catalog_snapshot_id,
        projection_sha256=request.prompt_payload.capabilities.projection_sha256,
    )
    if (
        catalog.session_id != goal.session_id
        or catalog.task_id != goal.task_id
        or catalog.auxiliary_graph_id != goal.auxiliary_graph_id
        or catalog.goal_id != goal.goal_id
        or catalog.auxiliary_graph_revision != request.auxiliary_graph_revision
        or catalog.projection != request.prompt_payload.capabilities
    ):
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic request capability projection crossed graph authority"
        )
    if str(goal_row["objective"]) != request.prompt_payload.goal.objective:
        raise AuxiliarySemanticVerificationPersistenceError(
            "semantic prompt goal objective differs from durable goal"
        )
    _validate_authorization_manifest(
        conn,
        request=request,
        authority=authority,
        expected_manifest_sha256=str(goal_row["authorization_manifest_sha256"]),
    )
    _validate_base_snapshot(conn, request=request)
    if request.terminal_candidate_binding is None:
        _validate_terminal_proposal(conn, request=request, revision=revision)
    else:
        _validate_terminal_candidate_proposal(
            conn,
            request=request,
            revision=revision,
        )
    _validate_prompt_authority_and_artifacts(
        conn,
        request=request,
        base_authority=authority,
    )
    expected_policy = derive_auxiliary_semantic_review_policy(
        prompt_payload=request.prompt_payload
    )
    if request.review_policy != expected_policy:
        raise AuxiliarySemanticVerificationPersistenceError(
            "semantic review policy was not derived from the frozen prompt authority"
        )


def _validate_authorization_manifest(
    conn: sqlite3.Connection,
    *,
    request: TaskGraphSemanticVerificationRequest,
    authority,
    expected_manifest_sha256: str,
) -> None:
    manifest_row = conn.execute(
        "SELECT * FROM insession_auxiliary_authorization_manifests "
        "WHERE authorization_manifest_id=?",
        (request.goal.authorization_manifest_id,),
    ).fetchone()
    if manifest_row is None:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic request authorization manifest is missing"
        )
    manifest_json = str(manifest_row["manifest_json"])
    try:
        manifest = json.loads(manifest_json)
    except (TypeError, ValueError) as exc:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic authorization manifest JSON is invalid"
        ) from exc
    if (
        manifest_json != _canonical_json(manifest)
        or _sha256_text(manifest_json) != str(manifest_row["manifest_sha256"])
        or str(manifest_row["manifest_sha256"]) != expected_manifest_sha256
    ):
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic authorization manifest hash binding is corrupt"
        )
    source_snapshot_id = manifest.get("authority_snapshot_id")
    source_turn_id = manifest.get("source_turn_id")
    if not isinstance(source_snapshot_id, str) or not isinstance(source_turn_id, str):
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic authorization manifest source binding is invalid"
        )
    source_snapshot_row = conn.execute(
        "SELECT snapshot_json, snapshot_sha256 FROM "
        "insession_auxiliary_authority_snapshots WHERE authority_snapshot_id=? "
        "AND session_id=? AND insession_task_id=? AND auxiliary_graph_id=? "
        "AND goal_id=? AND created_turn_id=?",
        (
            source_snapshot_id,
            request.goal.session_id,
            request.goal.task_id,
            request.goal.auxiliary_graph_id,
            request.goal.goal_id,
            source_turn_id,
        ),
    ).fetchone()
    if source_snapshot_row is None:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic authorization manifest source snapshot is missing"
        )
    try:
        source_snapshot = PlanningAuthoritySnapshot.model_validate_json(
            str(source_snapshot_row["snapshot_json"])
        )
    except (TypeError, ValueError) as exc:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic authorization manifest source snapshot is invalid"
        ) from exc
    if (
        source_snapshot.authority_snapshot_id != source_snapshot_id
        or source_snapshot.snapshot_sha256
        != str(source_snapshot_row["snapshot_sha256"])
        or _model_json(source_snapshot) != str(source_snapshot_row["snapshot_json"])
    ):
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic authorization manifest source snapshot is corrupt"
        )
    anchors_by_id = {item.anchor_id: item for item in authority.anchors}
    source_anchors_by_id = {
        item.anchor_id: item for item in source_snapshot.anchors
    }
    try:
        authorization_anchor_ids = tuple(
            str(anchor_id) for anchor_id in manifest["authorization_anchor_ids"]
        )
        expected_aliases = tuple(
            sorted(anchors_by_id[anchor_id].projection_alias for anchor_id in authorization_anchor_ids)
        )
    except (KeyError, TypeError) as exc:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic authorization manifest references unknown anchors"
        ) from exc
    if (
        len(authorization_anchor_ids) != len(set(authorization_anchor_ids))
        or any(anchor_id not in source_anchors_by_id for anchor_id in authorization_anchor_ids)
        or any(
            source_anchors_by_id[anchor_id].model_copy(
                update={
                    "authority_snapshot_id": authority.authority_snapshot_id,
                }
            )
            != anchors_by_id[anchor_id]
            for anchor_id in authorization_anchor_ids
        )
        or any(
            anchors_by_id[anchor_id].authority_class
            is not PlanningAuthorityClass.AUTHORIZATION
            for anchor_id in authorization_anchor_ids
        )
    ):
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic authorization authority drifted across graph revisions"
        )
    if request.prompt_payload.goal.authorization_aliases != expected_aliases:
        raise AuxiliarySemanticVerificationPersistenceError(
            "semantic prompt goal does not exactly cover authorization authority"
        )


def _validate_base_snapshot(
    conn: sqlite3.Connection,
    *,
    request: TaskGraphSemanticVerificationRequest,
) -> None:
    base_revision = request.goal.base_task_graph_revision
    base = request.prompt_payload.base_task_graph
    if base_revision is None:
        if base is not None:
            raise AuxiliarySemanticVerificationPersistenceError(
                "base-null semantic goal carried a TaskGraph base projection"
            )
        return
    if base is None or base.base_task_graph_revision != base_revision:
        raise AuxiliarySemanticVerificationPersistenceError(
            "semantic TaskGraph base projection is missing or stale"
        )
    row = conn.execute(
        "SELECT proposal_hash FROM insession_task_graph_revisions "
        "WHERE insession_task_id=? AND graph_revision=?",
        (request.goal.task_id, base_revision),
    ).fetchone()
    if row is None or str(row["proposal_hash"]) != base.source_snapshot_sha256:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic TaskGraph base source snapshot is not durable"
        )


def _validate_terminal_proposal(
    conn: sqlite3.Connection,
    *,
    request: TaskGraphSemanticVerificationRequest,
    revision: sqlite3.Row,
) -> None:
    terminal_node_id = str(revision["terminal_auxiliary_node_id"])
    row = conn.execute(
        "SELECT completion.*, run.execution_subject_id, run.status AS run_status, "
        "run.reason AS run_reason, run.current_attempt_id, "
        "run.current_verification_request_id, subject.subject_contract_version, "
        "binding.session_id AS bound_session_id, "
        "binding.insession_task_id AS bound_task_id, "
        "binding.auxiliary_graph_id AS bound_graph_id, "
        "binding.goal_id AS bound_goal_id, "
        "binding.auxiliary_graph_revision AS bound_graph_revision, "
        "binding.auxiliary_node_id AS bound_node_id, "
        "binding.node_revision AS bound_node_revision, "
        "binding.executor_kind AS bound_executor_kind, "
        "binding.definition_sha256 AS bound_definition_sha256, "
        "member.local_node_key, definition.definition_sha256, "
        "definition.output_contract, node_state.status AS node_status, "
        "verification.status AS verification_status, verification.result_json, "
        "verification.all_pass, output.snapshot_json AS output_json, "
        "output.snapshot_hash AS output_sha256, output.frozen_at "
        "FROM insession_auxiliary_node_completions_v2 AS completion "
        "JOIN insession_work_runs AS run ON run.work_run_id=completion.work_run_id "
        "JOIN insession_execution_subjects AS subject "
        "ON subject.execution_subject_id=run.execution_subject_id "
        "JOIN insession_auxiliary_v2_node_execution_subject_bindings AS binding "
        "ON binding.binding_id=subject.auxiliary_v2_binding_id "
        "JOIN insession_auxiliary_graph_revision_nodes_v2 AS member "
        "ON member.auxiliary_graph_id=completion.auxiliary_graph_id "
        "AND member.auxiliary_graph_revision=completion.auxiliary_graph_revision "
        "AND member.auxiliary_node_id=completion.auxiliary_node_id "
        "AND member.node_revision=completion.node_revision "
        "JOIN insession_auxiliary_node_definitions_v2 AS definition "
        "ON definition.auxiliary_graph_id=completion.auxiliary_graph_id "
        "AND definition.auxiliary_node_id=completion.auxiliary_node_id "
        "AND definition.node_revision=completion.node_revision "
        "JOIN insession_auxiliary_node_states_v2 AS node_state "
        "ON node_state.auxiliary_graph_id=completion.auxiliary_graph_id "
        "AND node_state.auxiliary_graph_revision=completion.auxiliary_graph_revision "
        "AND node_state.auxiliary_node_id=completion.auxiliary_node_id "
        "AND node_state.node_revision=completion.node_revision "
        "JOIN insession_work_run_verification_requests AS verification "
        "ON verification.verification_request_id=completion.verification_request_id "
        "AND verification.work_run_id=completion.work_run_id "
        "JOIN insession_work_run_output_windows AS output "
        "ON output.work_run_id=completion.work_run_id "
        "AND output.output_revision=completion.output_revision "
        "WHERE completion.session_id=? AND completion.insession_task_id=? "
        "AND completion.auxiliary_graph_id=? "
        "AND completion.auxiliary_graph_revision=? "
        "AND completion.auxiliary_node_id=?",
        (
            request.goal.session_id,
            request.goal.task_id,
            request.goal.auxiliary_graph_id,
            request.auxiliary_graph_revision,
            terminal_node_id,
        ),
    ).fetchone()
    if row is None:
        raise AuxiliarySemanticVerificationPersistenceError(
            "semantic request has no verified terminal proposal completion"
        )
    try:
        output_json = str(row["output_json"])
        output = OutputWindow.model_validate_json(output_json)
        result_json = str(row["result_json"])
        verification = NodeVerificationResult.model_validate_json(result_json)
        if request.goal.base_task_graph_revision is None:
            proposal = InSessionTaskGraphRevisionProposal.model_validate_json(
                output.content
            )
            lineage: tuple[TaskGraphSemanticLineageProjection, ...] = ()
            expected_terminal_content = proposal.model_dump_json()
        else:
            candidate = TaskGraphRevisionCandidate.model_validate_json(
                output.content
            )
            proposal = candidate.proposal
            lineage = candidate.lineage
            expected_terminal_content = candidate.model_dump_json()
    except (TypeError, ValueError, RecursionError) as exc:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic terminal completion contains invalid typed JSON"
        ) from exc
    subject_value = AuxiliaryNodeSubject(
        task_id=request.goal.task_id,
        auxiliary_graph_id=request.goal.auxiliary_graph_id,
        auxiliary_graph_revision=request.auxiliary_graph_revision,
        node_id=terminal_node_id,
        node_revision=int(row["node_revision"]),
    )
    completion_payload = {
        "schema_version": "auxiliary-node-completion-v2",
        "completion_id": str(row["completion_id"]),
        "execution_subject_id": str(row["execution_subject_id"]),
        "subject": subject_value.model_dump(mode="json"),
        "goal_id": request.goal.goal_id,
        "definition_sha256": str(row["definition_sha256"]),
        "verification_request_id": str(row["verification_request_id"]),
        "verification_result_sha256": _sha256_text(result_json),
        "submitted_attempt_id": str(row["submitted_attempt_id"]),
        "output_revision": int(row["output_revision"]),
        "output_snapshot_sha256": str(row["output_sha256"]),
    }
    valid = (
        str(row["subject_contract_version"]) == "auxiliary_node_v2"
        and str(row["bound_session_id"]) == request.goal.session_id
        and str(row["bound_task_id"]) == request.goal.task_id
        and str(row["bound_graph_id"]) == request.goal.auxiliary_graph_id
        and str(row["bound_goal_id"]) == request.goal.goal_id
        and int(row["bound_graph_revision"]) == request.auxiliary_graph_revision
        and str(row["bound_node_id"]) == terminal_node_id
        and int(row["bound_node_revision"]) == int(row["node_revision"])
        and str(row["bound_executor_kind"]) == "terminal_planner"
        and str(row["bound_definition_sha256"])
        == str(row["definition_sha256"])
        and str(row["output_contract"]) == "task_graph_revision_proposal_v2"
        and str(row["node_status"]) == "completed"
        and str(row["run_status"]) == "completed"
        and str(row["run_reason"]) == "verification_passed"
        and row["current_attempt_id"] is None
        and row["current_verification_request_id"] is None
        and str(row["verification_status"]) == "completed"
        and int(row["all_pass"]) == 1
        and row["frozen_at"] is not None
        and output.work_run_id == str(row["work_run_id"])
        and output.output_revision == int(row["output_revision"])
        and output.updated_attempt_id is not None
        and output_json == _model_json(output)
        and _sha256_text(output_json) == str(row["output_sha256"])
        and output.content == expected_terminal_content
        and verification.all_pass
        and verification.subject == subject_value
        and verification.work_run_id == str(row["work_run_id"])
        and verification.verification_request_id
        == str(row["verification_request_id"])
        and result_json == _model_json(verification)
        and str(row["completion_json"]) == _canonical_json(completion_payload)
        and _sha256_text(str(row["completion_json"]))
        == str(row["completion_sha256"])
        and proposal == request.task_graph_proposal
        and lineage == request.prompt_payload.lineage
    )
    if not valid:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic terminal proposal completion binding is corrupt"
        )


def _validate_terminal_candidate_proposal(
    conn: sqlite3.Connection,
    *,
    request: TaskGraphSemanticVerificationRequest,
    revision: sqlite3.Row,
) -> None:
    """认证处于 open、rejected 或 completed 状态的终结候选项。

    语义请求/结果在节点 verifier 仍持有其待处理请求时完成结算。成功的 replan
    事务随后会归档同一游标，但不会创建 completion。该 rejected 形态与普通的
    completed 形态都仍可作为持久化重放权威读取。
    """

    candidate = request.terminal_candidate_binding
    if candidate is None:
        raise AuxiliarySemanticVerificationPersistenceError(
            "terminal candidate authority is missing"
        )
    try:
        verification_row = _load_auxiliary_request_row(
            conn,
            session_id=request.goal.session_id,
            verification_request_id=candidate.node_verification_request_id,
        )
        verification_record = _auxiliary_record_from_row(verification_row)
        verification = verification_record.request
    except AuxiliaryGraphPersistenceError as exc:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "terminal candidate verification request is missing or corrupt"
        ) from exc

    terminal_node_id = str(revision["terminal_auxiliary_node_id"])
    binding_row = conn.execute(
        "SELECT run.status AS run_status, run.reason AS run_reason, "
        "run.current_attempt_id, run.current_verification_request_id, "
        "subject.subject_contract_version, binding.goal_id AS bound_goal_id, "
        "binding.executor_kind, binding.definition_sha256 AS bound_definition_sha256, "
        "definition.definition_sha256, definition.output_contract, "
        "node_state.status AS node_status, revision_state.status AS revision_status, "
        "output.snapshot_json AS output_json, output.snapshot_hash AS output_sha256, "
        "output.frozen_at, window.turn_id AS window_turn_id, "
        "window.window_state, window.stage, window.current_work_run_id, "
        "window.current_attempt_id AS window_attempt_id, "
        "window.latest_checkpoint_id, "
        "(SELECT COUNT(*) FROM insession_auxiliary_node_completions_v2 AS completion "
        "WHERE completion.work_run_id=run.work_run_id) AS completion_count "
        "FROM insession_work_runs AS run "
        "JOIN insession_execution_subjects AS subject "
        "ON subject.execution_subject_id=run.execution_subject_id "
        "JOIN insession_auxiliary_v2_node_execution_subject_bindings AS binding "
        "ON binding.binding_id=subject.auxiliary_v2_binding_id "
        "JOIN insession_auxiliary_node_definitions_v2 AS definition "
        "ON definition.auxiliary_graph_id=binding.auxiliary_graph_id "
        "AND definition.auxiliary_node_id=binding.auxiliary_node_id "
        "AND definition.node_revision=binding.node_revision "
        "JOIN insession_auxiliary_node_states_v2 AS node_state "
        "ON node_state.auxiliary_graph_id=binding.auxiliary_graph_id "
        "AND node_state.auxiliary_graph_revision=binding.auxiliary_graph_revision "
        "AND node_state.auxiliary_node_id=binding.auxiliary_node_id "
        "AND node_state.node_revision=binding.node_revision "
        "JOIN insession_auxiliary_graph_revision_states_v2 AS revision_state "
        "ON revision_state.auxiliary_graph_id=binding.auxiliary_graph_id "
        "AND revision_state.auxiliary_graph_revision=binding.auxiliary_graph_revision "
        "JOIN insession_work_run_output_windows AS output "
        "ON output.work_run_id=run.work_run_id "
        "LEFT JOIN turn_execution_windows AS window "
        "ON window.session_id=run.session_id "
        "WHERE run.work_run_id=? AND run.session_id=? "
        "AND binding.insession_task_id=? AND binding.auxiliary_graph_id=? "
        "AND binding.auxiliary_graph_revision=? AND binding.auxiliary_node_id=? "
        "AND output.output_revision=?",
        (
            candidate.work_run_id,
            request.goal.session_id,
            request.goal.task_id,
            request.goal.auxiliary_graph_id,
            request.auxiliary_graph_revision,
            terminal_node_id,
            candidate.output_revision,
        ),
    ).fetchone()
    if binding_row is None:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "terminal candidate WorkRun binding is missing"
        )
    try:
        output_json = str(binding_row["output_json"])
        output = OutputWindow.model_validate_json(output_json)
        if request.goal.base_task_graph_revision is None:
            proposal = InSessionTaskGraphRevisionProposal.model_validate_json(
                output.content
            )
            lineage: tuple[TaskGraphSemanticLineageProjection, ...] = ()
            expected_content = proposal.model_dump_json()
        else:
            proposal_candidate = TaskGraphRevisionCandidate.model_validate_json(
                output.content
            )
            proposal = proposal_candidate.proposal
            lineage = proposal_candidate.lineage
            expected_content = proposal_candidate.model_dump_json()
    except (TypeError, ValueError, RecursionError) as exc:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "terminal candidate contains invalid typed JSON"
        ) from exc

    open_candidate = (
        verification.status.value == "pending"
        and verification.technical_error_code is None
        and str(binding_row["run_status"]) == "active"
        and str(binding_row["run_reason"]) == "verification_pending"
        and str(binding_row["current_verification_request_id"] or "")
        == candidate.node_verification_request_id
        and str(binding_row["node_status"]) == "active"
        and str(binding_row["revision_status"]) == "active"
        and str(binding_row["window_turn_id"] or "")
        == verification.request_turn_id
        and str(binding_row["window_state"] or "") == "active"
        and str(binding_row["stage"] or "") == "VERIFICATION"
        and str(binding_row["current_work_run_id"] or "")
        == candidate.work_run_id
        and str(binding_row["latest_checkpoint_id"] or "")
        == candidate.node_verification_request_id
    )
    rejected_candidate = (
        verification.status.value == "interrupted"
        and verification.technical_error_code
        == "terminal_candidate_semantic_replan"
        and str(binding_row["run_status"]) == "cancelled"
        and str(binding_row["run_reason"])
        == "terminal_candidate_semantic_replan"
        and binding_row["current_verification_request_id"] is None
        and str(binding_row["node_status"]) == "cancelled"
        and str(binding_row["revision_status"]) == "superseded"
        and str(binding_row["current_work_run_id"] or "")
        != candidate.work_run_id
        and str(binding_row["latest_checkpoint_id"] or "")
        != candidate.node_verification_request_id
    )
    completed_candidate = (
        verification.status.value == "completed"
        and verification.technical_error_code is None
        and verification_record.result is not None
        and verification_record.result.all_pass
        and str(binding_row["run_status"]) == "completed"
        and str(binding_row["run_reason"]) == "verification_passed"
        and binding_row["current_verification_request_id"] is None
        and str(binding_row["node_status"]) == "completed"
        and str(binding_row["current_work_run_id"] or "")
        != candidate.work_run_id
        and str(binding_row["latest_checkpoint_id"] or "")
        != candidate.node_verification_request_id
    )
    if completed_candidate:
        _validate_terminal_proposal(conn, request=request, revision=revision)
    try:
        _revalidate_auxiliary_request_binding(
            conn,
            verification_row,
            expected_node_status=(
                "active"
                if open_candidate
                else ("cancelled" if rejected_candidate else "completed")
            ),
        )
    except AuxiliaryGraphPersistenceError as exc:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "terminal candidate immutable verifier binding is corrupt"
        ) from exc
    valid = (
        (open_candidate or rejected_candidate or completed_candidate)
        and verification.work_run_id == candidate.work_run_id
        and verification.submitted_attempt_id
        == candidate.submitted_attempt_id
        and verification.verification_request_id
        == candidate.node_verification_request_id
        and verification.output_revision == candidate.output_revision
        and verification.subject.task_id == request.goal.task_id
        and verification.subject.auxiliary_graph_id
        == request.goal.auxiliary_graph_id
        and verification.subject.auxiliary_graph_revision
        == request.auxiliary_graph_revision
        and verification.subject.node_id == terminal_node_id
        and str(binding_row["subject_contract_version"])
        == "auxiliary_node_v2"
        and str(binding_row["bound_goal_id"]) == request.goal.goal_id
        and str(binding_row["executor_kind"]) == "terminal_planner"
        and str(binding_row["bound_definition_sha256"])
        == str(binding_row["definition_sha256"])
        and str(binding_row["output_contract"])
        == "task_graph_revision_proposal_v2"
        and binding_row["current_attempt_id"] is None
        and binding_row["window_attempt_id"] is None
        and (
            (completed_candidate and binding_row["frozen_at"] is not None)
            or (
                not completed_candidate
                and binding_row["frozen_at"] is None
            )
        )
        and int(binding_row["completion_count"])
        == (1 if completed_candidate else 0)
        and output.work_run_id == candidate.work_run_id
        and output.output_revision == candidate.output_revision
        and output_json == _model_json(output)
        and _sha256_text(output_json) == str(binding_row["output_sha256"])
        and str(binding_row["output_sha256"])
        == candidate.output_snapshot_sha256
        and output.content == expected_content
        and proposal == request.task_graph_proposal
        and lineage == request.prompt_payload.lineage
    )
    if not valid:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "terminal candidate semantic binding is corrupt"
        )


def _validate_prompt_authority_and_artifacts(
    conn: sqlite3.Connection,
    *,
    request: TaskGraphSemanticVerificationRequest,
    base_authority,
) -> None:
    observation_cards: dict[str, PlanningAuthoritySourceCard] = {}
    observation_anchors: dict[str, PlanningAuthorityAnchor] = {}
    for projection in request.prompt_payload.context_artifacts:
        cards, anchors = _load_exact_artifact_projection(
            conn,
            request=request,
            projection=projection,
        )
        for card in cards:
            if card.alias in observation_cards:
                raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
                    "semantic artifacts reused a source-card alias"
                )
            observation_cards[card.alias] = card
        for anchor in anchors:
            if anchor.anchor_id in observation_anchors:
                raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
                    "semantic artifacts reused a private authority anchor"
                )
            observation_anchors[anchor.anchor_id] = anchor
    all_base_by_alias = {
        item.projection_alias: item for item in base_authority.anchors
    }
    if set(all_base_by_alias) & set(observation_cards):
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic artifact authority collided with revision authority"
        )
    # 原始内容证据只能通过已完成且封存的 Host observation 获取。精确的已挂载文档
    # 卡片则不同：它们允许未来的 TaskGraph 节点读取冻结资源，但不对其内容作任何
    # 断言。此处恢复已封存的 initial-planning 卡片，使验证、重新规划和重放都使用
    # 完全相同的持久化值。
    base_by_alias = {
        alias: anchor
        for alias, anchor in all_base_by_alias.items()
        if anchor.authority_class is PlanningAuthorityClass.AUTHORIZATION
    }
    try:
        resource_cards = _planning_resource_source_cards(
            conn,
            session_id=request.goal.session_id,
            task_id=request.goal.task_id,
            authority_snapshot=base_authority,
        )
    except AuxiliaryTerminalValidationContextError as exc:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic resource authority cannot be reconstructed"
        ) from exc
    resource_by_alias = {card.alias: card for card in resource_cards}
    if len(resource_by_alias) != len(resource_cards):
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic resource authority reused a source-card alias"
        )
    expected_aliases = (
        set(base_by_alias) | set(resource_by_alias) | set(observation_cards)
    )
    actual_by_alias = {
        item.alias: item for item in request.authority_projection.cards
    }
    if set(actual_by_alias) != expected_aliases:
        raise AuxiliarySemanticVerificationPersistenceError(
            "semantic authority projection does not exactly cover frozen authority"
        )
    for alias, expected in observation_cards.items():
        if actual_by_alias[alias] != expected:
            raise AuxiliarySemanticVerificationPersistenceError(
                "semantic artifact source card differs from its sealed projection"
            )
    for alias, expected in resource_by_alias.items():
        if actual_by_alias[alias] != expected:
            raise AuxiliarySemanticVerificationPersistenceError(
                "semantic resource card differs from frozen resource authority"
            )
    for alias, anchor in base_by_alias.items():
        card = actual_by_alias[alias]
        if (
            card.authority_class is not anchor.authority_class
            or card.projection_sha256 != anchor.projection_sha256
            or not _origin_matches_source_kind(anchor.origin_kind, card.source_kind)
        ):
            raise AuxiliarySemanticVerificationPersistenceError(
                "semantic source card differs from private revision authority"
            )
        if anchor.origin_kind in {
            PlanningAuthorityOriginKind.USER_INSTRUCTION_SPAN,
            PlanningAuthorityOriginKind.USER_ANSWER_SPAN,
        } and _sha256_text(card.excerpt) != anchor.content_sha256:
            raise AuxiliarySemanticVerificationPersistenceError(
                "semantic user authority excerpt differs from its exact source span"
            )
    known_anchor_ids = {
        item.anchor_id for item in base_authority.anchors
    } | set(observation_anchors)
    for projection in request.prompt_payload.context_artifacts:
        artifact_row = conn.execute(
            "SELECT artifact_json FROM "
            "insession_auxiliary_planning_context_artifacts WHERE artifact_id=?",
            (projection.artifact_id,),
        ).fetchone()
        if artifact_row is None:
            raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
                "semantic ContextArtifact disappeared"
            )
        artifact = PlanningContextArtifact.model_validate_json(
            str(artifact_row["artifact_json"])
        )
        referenced = {
            anchor_id
            for item in (*artifact.facts, *artifact.conflicts, *artifact.gaps)
            for anchor_id in item.evidence_anchor_ids
        } | {
            anchor_id
            for item in artifact.constraints
            for anchor_id in item.authorization_anchor_ids
        }
        if not referenced <= known_anchor_ids:
            raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
                "semantic ContextArtifact references unauthenticated authority"
            )


def _load_exact_artifact_projection(
    conn: sqlite3.Connection,
    *,
    request: TaskGraphSemanticVerificationRequest,
    projection: PlanningContextArtifactProjection,
) -> tuple[
    tuple[PlanningAuthoritySourceCard, ...],
    tuple[PlanningAuthorityAnchor, ...],
]:
    row = conn.execute(
        "SELECT artifact.*, member.local_node_key, "
        "observation.snapshot_json AS observation_json, "
        "observation.snapshot_sha256 AS observation_sha256, "
        "receipt.receipt_json, receipt.receipt_sha256 AS stored_receipt_sha256, "
        "reservation.status AS reservation_status, "
        "reservation.settlement_sha256 AS reserved_settlement_sha256 "
        "FROM insession_auxiliary_planning_context_artifacts AS artifact "
        "JOIN insession_auxiliary_graph_revision_nodes_v2 AS member "
        "ON member.auxiliary_graph_id=artifact.auxiliary_graph_id "
        "AND member.auxiliary_graph_revision="
        "artifact.producer_auxiliary_graph_revision "
        "AND member.auxiliary_node_id=artifact.producer_auxiliary_node_id "
        "AND member.node_revision=artifact.producer_node_revision "
        "LEFT JOIN insession_auxiliary_observations AS observation "
        "ON observation.observation_id=artifact.producer_primitive_call_id "
        "LEFT JOIN insession_auxiliary_context_verification_receipts AS receipt "
        "ON receipt.verification_receipt_id=artifact.verification_receipt_id "
        "LEFT JOIN insession_auxiliary_planning_primitive_invocations AS reservation "
        "ON reservation.primitive_call_id=artifact.producer_primitive_call_id "
        "WHERE artifact.artifact_id=? AND artifact.session_id=? "
        "AND artifact.insession_task_id=? AND artifact.auxiliary_graph_id=? "
        "AND artifact.goal_id=?",
        (
            projection.artifact_id,
            request.goal.session_id,
            request.goal.task_id,
            request.goal.auxiliary_graph_id,
            request.goal.goal_id,
        ),
    ).fetchone()
    if row is None:
        raise AuxiliarySemanticVerificationPersistenceError(
            "semantic prompt references an unknown ContextArtifact"
        )
    artifact_json = str(row["artifact_json"])
    try:
        artifact = PlanningContextArtifact.model_validate_json(artifact_json)
    except (TypeError, ValueError) as exc:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic private ContextArtifact JSON is invalid"
        ) from exc
    if (
        artifact.artifact_id != projection.artifact_id
        or artifact.artifact_sha256 != projection.artifact_sha256
        or artifact.session_id != request.goal.session_id
        or artifact.task_id != request.goal.task_id
        or artifact.auxiliary_graph_id != request.goal.auxiliary_graph_id
        or artifact.goal_id != request.goal.goal_id
        or artifact.producer_auxiliary_node.auxiliary_graph_revision
        != int(row["producer_auxiliary_graph_revision"])
        or artifact.producer_auxiliary_node.node_id
        != str(row["producer_auxiliary_node_id"])
        or artifact.producer_auxiliary_node.node_revision
        != int(row["producer_node_revision"])
        or projection.producer_node_alias != str(row["local_node_key"])
        or _model_json(artifact) != artifact_json
        or artifact.artifact_sha256 != str(row["artifact_sha256"])
        or str(row["facts_json"])
        != _canonical_json(
            [item.model_dump(mode="json") for item in artifact.facts]
        )
        or str(row["constraints_json"])
        != _canonical_json(
            [item.model_dump(mode="json") for item in artifact.constraints]
        )
        or str(row["conflicts_json"])
        != _canonical_json(
            [item.model_dump(mode="json") for item in artifact.conflicts]
        )
        or str(row["gaps_json"])
        != _canonical_json([item.model_dump(mode="json") for item in artifact.gaps])
        or str(row["evidence_refs_json"])
        != _canonical_json(
            [item.model_dump(mode="json") for item in artifact.evidence_refs]
        )
        or not _projection_binds_artifact(projection, artifact=artifact)
    ):
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic ContextArtifact projection is not exact"
        )
    if artifact.producer_primitive_call_id is None:
        return (), ()
    observation_json = row["observation_json"]
    receipt_json = row["receipt_json"]
    if observation_json is None or receipt_json is None:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic primitive ContextArtifact lost its sealed observation"
        )
    try:
        observation = json.loads(str(observation_json))
        result = observation["result"]
        prompt_inputs = result["prompt_inputs"]
        stored_projection = PlanningContextArtifactProjection.model_validate(
            prompt_inputs["context_artifact"]
        )
        cards = tuple(
            PlanningAuthoritySourceCard.model_validate(item)
            for item in prompt_inputs["source_cards"]
        )
        anchors = tuple(
            PlanningAuthorityAnchor.model_validate(item)
            for item in result["authority_anchors"]
        )
        receipt = json.loads(str(receipt_json))
    except (KeyError, TypeError, ValueError, RecursionError) as exc:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic primitive observation projection JSON is invalid"
        ) from exc
    cards_by_alias = {item.alias: item for item in cards}
    valid = (
        len(cards_by_alias) == len(cards)
        and stored_projection == projection
        and result.get("artifact") == artifact.model_dump(mode="json")
        and observation.get("schema_version")
        == "sealed-planning-host-primitive-observation-v1"
        and str(observation_json) == _canonical_json(observation)
        and _sha256_text(str(observation_json)) == str(row["observation_sha256"])
        and str(row["receipt_json"]) == _canonical_json(receipt)
        and _sha256_text(str(row["receipt_json"]))
        == str(row["stored_receipt_sha256"])
        and str(row["stored_receipt_sha256"])
        == artifact.verification_receipt_sha256
        and receipt.get("authority_anchors") == result.get("authority_anchors")
        and str(row["reservation_status"]) == "settled"
        and str(row["reserved_settlement_sha256"])
        == result.get("settlement_sha256")
        and all(
            anchor.authority_snapshot_id
            == request.authority_projection.authority_snapshot_id
            and anchor.projection_alias in cards_by_alias
            and cards_by_alias[anchor.projection_alias].authority_class
            is anchor.authority_class
            and cards_by_alias[anchor.projection_alias].projection_sha256
            == anchor.projection_sha256
            for anchor in anchors
        )
        and {item.projection_alias for item in anchors} == set(cards_by_alias)
    )
    if not valid:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic primitive ContextArtifact settlement is corrupt"
        )
    return cards, anchors


def _origin_matches_source_kind(
    origin: PlanningAuthorityOriginKind,
    source: PlanningAuthoritySourceKind,
) -> bool:
    allowed: dict[
        PlanningAuthorityOriginKind,
        set[PlanningAuthoritySourceKind],
    ] = {
        PlanningAuthorityOriginKind.USER_INSTRUCTION_SPAN: {
            PlanningAuthoritySourceKind.USER_INSTRUCTION
        },
        PlanningAuthorityOriginKind.USER_ANSWER_SPAN: {
            PlanningAuthoritySourceKind.USER_ANSWER
        },
        PlanningAuthorityOriginKind.PRIOR_TASK_STATE: {
            PlanningAuthoritySourceKind.PRIOR_TASK_STATE
        },
        PlanningAuthorityOriginKind.RETRIEVED_SOURCE_UNIT: {
            PlanningAuthoritySourceKind.DOCUMENT
        },
        PlanningAuthorityOriginKind.WORKSPACE_RESOURCE: {
            PlanningAuthoritySourceKind.WORKSPACE,
            PlanningAuthoritySourceKind.DOCUMENT,
        },
        PlanningAuthorityOriginKind.TOOL_RESULT: {
            PlanningAuthoritySourceKind.TOOL_OBSERVATION,
            PlanningAuthoritySourceKind.DOCUMENT,
        },
        PlanningAuthorityOriginKind.VISUAL_UNIT: {
            PlanningAuthoritySourceKind.VISUAL
        },
        PlanningAuthorityOriginKind.MEMORY_RECORD: {
            PlanningAuthoritySourceKind.MEMORY
        },
        PlanningAuthorityOriginKind.ARTIFACT: {
            PlanningAuthoritySourceKind.ARTIFACT
        },
        PlanningAuthorityOriginKind.PRIMITIVE_RESULT: {
            PlanningAuthoritySourceKind.TOOL_OBSERVATION,
            PlanningAuthoritySourceKind.DOCUMENT,
            PlanningAuthoritySourceKind.WORKSPACE,
        },
        PlanningAuthorityOriginKind.GAP_OBSERVATION: {
            PlanningAuthoritySourceKind.GAP
        },
    }
    return source in allowed[origin]


def _insert_request(
    conn: sqlite3.Connection,
    *,
    request: TaskGraphSemanticVerificationRequest,
    created_turn_id: str,
    now: str,
) -> None:
    goal = request.goal
    prompt = request.prompt_payload
    policy = request.review_policy
    conn.execute(
        "INSERT INTO insession_auxiliary_semantic_verification_requests "
        "(verification_request_id, session_id, insession_task_id, "
        "auxiliary_graph_id, goal_id, auxiliary_graph_revision, "
        "auxiliary_graph_structure_sha256, logical_call_id, "
        "verification_profile_id, reviewer_ordinal, required_reviewer_count, "
        "authority_snapshot_id, authority_snapshot_sha256, budget_ledger_id, "
        "budget_state_version, budget_snapshot_sha256, "
        "capability_catalog_snapshot_id, capability_catalog_snapshot_sha256, "
        "capability_projection_sha256, prompt_payload_json, "
        "prompt_payload_sha256, review_policy_json, review_policy_sha256, "
        "policy_source_sha256, distinct_document_count, has_visual_input, "
        "requires_protected_effect, modifies_executed_task_graph, "
        "task_graph_proposal_sha256, blocking_gap_aliases_json, "
        "non_blocking_gap_aliases_json, request_json, binding_sha256, "
        "created_turn_id, created_at) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            request.verification_request_id,
            goal.session_id,
            goal.task_id,
            goal.auxiliary_graph_id,
            goal.goal_id,
            request.auxiliary_graph_revision,
            request.auxiliary_graph_structure_sha256,
            request.logical_call_id,
            request.verification_profile_id,
            request.reviewer_ordinal,
            request.required_reviewer_count,
            prompt.authority.authority_snapshot_id,
            prompt.authority.authority_snapshot_sha256,
            prompt.budget.budget_ledger_id,
            prompt.budget.state_version,
            prompt.budget.snapshot_sha256,
            prompt.capabilities.capability_catalog_snapshot_id,
            prompt.capabilities.capability_catalog_snapshot_sha256,
            prompt.capabilities.projection_sha256,
            _model_json(prompt),
            prompt.payload_sha256,
            _model_json(policy),
            policy.policy_sha256,
            policy.policy_source_sha256,
            policy.distinct_document_count,
            int(policy.has_visual_input),
            int(policy.requires_protected_effect),
            int(policy.modifies_executed_task_graph),
            request.task_graph_proposal_sha256,
            _canonical_json(list(request.blocking_gap_aliases)),
            _canonical_json(list(request.non_blocking_gap_aliases)),
            _model_json(request),
            request.binding_sha256,
            created_turn_id,
            now,
        ),
    )
    for ordinal, artifact in enumerate(prompt.context_artifacts):
        conn.execute(
            "INSERT INTO "
            "insession_auxiliary_semantic_request_context_artifacts "
            "(verification_request_id, session_id, insession_task_id, "
            "auxiliary_graph_id, goal_id, ordinal, artifact_alias, "
            "artifact_id, artifact_sha256, projection_json, projection_sha256) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                request.verification_request_id,
                goal.session_id,
                goal.task_id,
                goal.auxiliary_graph_id,
                goal.goal_id,
                ordinal,
                artifact.artifact_alias,
                artifact.artifact_id,
                artifact.artifact_sha256,
                _model_json(artifact),
                artifact.projection_sha256,
            ),
        )


def _load_request(
    conn: sqlite3.Connection,
    verification_request_id: str,
) -> StoredAuxiliarySemanticVerificationRequest:
    row = conn.execute(
        "SELECT * FROM insession_auxiliary_semantic_verification_requests "
        "WHERE verification_request_id=?",
        (verification_request_id,),
    ).fetchone()
    if row is None:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic verification request is missing"
        )
    request_json = str(row["request_json"])
    prompt_json = str(row["prompt_payload_json"])
    policy_json = str(row["review_policy_json"])
    try:
        request = TaskGraphSemanticVerificationRequest.model_validate_json(
            request_json
        )
        prompt = TaskGraphSemanticVerificationPromptPayload.model_validate_json(
            prompt_json
        )
        policy = TaskGraphSemanticReviewPolicy.model_validate_json(policy_json)
    except (TypeError, ValueError) as exc:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "stored semantic request JSON is invalid"
        ) from exc
    goal = request.goal
    mirrored = (
        request.verification_request_id == verification_request_id
        and request_json == _model_json(request)
        and prompt_json == _model_json(prompt)
        and policy_json == _model_json(policy)
        and prompt == request.prompt_payload
        and policy == request.review_policy
        and str(row["session_id"]) == goal.session_id
        and str(row["insession_task_id"]) == goal.task_id
        and str(row["auxiliary_graph_id"]) == goal.auxiliary_graph_id
        and str(row["goal_id"]) == goal.goal_id
        and int(row["auxiliary_graph_revision"])
        == request.auxiliary_graph_revision
        and str(row["auxiliary_graph_structure_sha256"])
        == request.auxiliary_graph_structure_sha256
        and str(row["logical_call_id"]) == request.logical_call_id
        and str(row["verification_profile_id"])
        == request.verification_profile_id
        and int(row["reviewer_ordinal"]) == request.reviewer_ordinal
        and int(row["required_reviewer_count"])
        == request.required_reviewer_count
        and str(row["authority_snapshot_id"])
        == prompt.authority.authority_snapshot_id
        and str(row["authority_snapshot_sha256"])
        == prompt.authority.authority_snapshot_sha256
        and str(row["budget_ledger_id"]) == prompt.budget.budget_ledger_id
        and int(row["budget_state_version"]) == prompt.budget.state_version
        and str(row["budget_snapshot_sha256"])
        == prompt.budget.snapshot_sha256
        and str(row["capability_catalog_snapshot_id"])
        == prompt.capabilities.capability_catalog_snapshot_id
        and str(row["capability_catalog_snapshot_sha256"])
        == prompt.capabilities.capability_catalog_snapshot_sha256
        and str(row["capability_projection_sha256"])
        == prompt.capabilities.projection_sha256
        and str(row["prompt_payload_sha256"]) == prompt.payload_sha256
        and str(row["review_policy_sha256"]) == policy.policy_sha256
        and str(row["policy_source_sha256"]) == policy.policy_source_sha256
        and int(row["distinct_document_count"])
        == policy.distinct_document_count
        and bool(row["has_visual_input"]) == policy.has_visual_input
        and bool(row["requires_protected_effect"])
        == policy.requires_protected_effect
        and bool(row["modifies_executed_task_graph"])
        == policy.modifies_executed_task_graph
        and str(row["task_graph_proposal_sha256"])
        == request.task_graph_proposal_sha256
        and str(row["blocking_gap_aliases_json"])
        == _canonical_json(list(request.blocking_gap_aliases))
        and str(row["non_blocking_gap_aliases_json"])
        == _canonical_json(list(request.non_blocking_gap_aliases))
        and str(row["binding_sha256"]) == request.binding_sha256
    )
    if not mirrored:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "stored semantic request row and contract differ"
        )
    artifact_rows = conn.execute(
        "SELECT * FROM insession_auxiliary_semantic_request_context_artifacts "
        "WHERE verification_request_id=? ORDER BY ordinal",
        (verification_request_id,),
    ).fetchall()
    if len(artifact_rows) != len(prompt.context_artifacts):
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "stored semantic request ContextArtifact manifest is incomplete"
        )
    for ordinal, (artifact_row, artifact) in enumerate(
        zip(artifact_rows, prompt.context_artifacts, strict=True)
    ):
        projection_json = str(artifact_row["projection_json"])
        try:
            loaded_projection = PlanningContextArtifactProjection.model_validate_json(
                projection_json
            )
        except (TypeError, ValueError) as exc:
            raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
                "stored semantic ContextArtifact projection is invalid"
            ) from exc
        if (
            loaded_projection != artifact
            or int(artifact_row["ordinal"]) != ordinal
            or str(artifact_row["session_id"]) != goal.session_id
            or str(artifact_row["insession_task_id"]) != goal.task_id
            or str(artifact_row["auxiliary_graph_id"])
            != goal.auxiliary_graph_id
            or str(artifact_row["goal_id"]) != goal.goal_id
            or str(artifact_row["artifact_alias"]) != artifact.artifact_alias
            or str(artifact_row["artifact_id"]) != artifact.artifact_id
            or str(artifact_row["artifact_sha256"]) != artifact.artifact_sha256
            or projection_json != _model_json(artifact)
            or str(artifact_row["projection_sha256"])
            != artifact.projection_sha256
        ):
            raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
                "stored semantic ContextArtifact binding is corrupt"
            )
    return StoredAuxiliarySemanticVerificationRequest(
        session_id=goal.session_id,
        task_id=goal.task_id,
        auxiliary_graph_id=goal.auxiliary_graph_id,
        goal_id=goal.goal_id,
        auxiliary_graph_revision=request.auxiliary_graph_revision,
        request=request,
        created_turn_id=str(row["created_turn_id"]),
    )


def _load_result(
    conn: sqlite3.Connection,
    verification_result_id: str,
) -> StoredAuxiliarySemanticVerificationResult:
    row = conn.execute(
        "SELECT * FROM insession_auxiliary_semantic_verification_results "
        "WHERE verification_result_id=?",
        (verification_result_id,),
    ).fetchone()
    if row is None:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic verification result is missing"
        )
    result_json = str(row["result_json"])
    try:
        result = TaskGraphSemanticVerificationResult.model_validate_json(
            result_json
        )
    except (TypeError, ValueError) as exc:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "stored semantic result JSON is invalid"
        ) from exc
    request_record = _load_request(conn, result.verification_request_id)
    request = request_record.request
    _validate_frozen_request_authority(conn, request=request)
    try:
        validate_task_graph_semantic_verification_result(
            request=request,
            result=result,
        )
    except ValueError as exc:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "stored semantic result lost its request binding"
        ) from exc
    mirrored = (
        result_json == _model_json(result)
        and str(row["session_id"]) == request.goal.session_id
        and str(row["insession_task_id"]) == request.goal.task_id
        and str(row["auxiliary_graph_id"])
        == request.goal.auxiliary_graph_id
        and str(row["goal_id"]) == request.goal.goal_id
        and int(row["auxiliary_graph_revision"])
        == request.auxiliary_graph_revision
        and str(row["verification_request_id"])
        == result.verification_request_id
        and str(row["request_binding_sha256"])
        == result.request_binding_sha256
        and str(row["prompt_payload_sha256"])
        == request.prompt_payload.payload_sha256
        and str(row["review_policy_sha256"])
        == request.review_policy.policy_sha256
        and str(row["logical_call_id"]) == result.logical_call_id
        and str(row["verification_profile_id"])
        == result.verification_profile_id
        and int(row["reviewer_ordinal"]) == result.reviewer_ordinal
        and int(row["required_reviewer_count"])
        == result.required_reviewer_count
        and str(row["items_json"])
        == _canonical_json([item.model_dump(mode="json") for item in result.items])
        and str(row["host_disposition"]) == result.host_disposition.value
        and str(row["result_sha256"]) == result.result_sha256
    )
    if not mirrored:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "stored semantic result row and contract differ"
        )
    return StoredAuxiliarySemanticVerificationResult(
        session_id=request.goal.session_id,
        task_id=request.goal.task_id,
        auxiliary_graph_id=request.goal.auxiliary_graph_id,
        goal_id=request.goal.goal_id,
        auxiliary_graph_revision=request.auxiliary_graph_revision,
        result=result,
        host_disposition=result.host_disposition,
        created_turn_id=str(row["created_turn_id"]),
    )


def _build_settlement(
    *,
    command: SettleAuxiliarySemanticVerificationQuorumCommand,
    requests: tuple[TaskGraphSemanticVerificationRequest, ...],
    results: tuple[TaskGraphSemanticVerificationResult, ...],
) -> StoredAuxiliarySemanticQuorumSettlement:
    first = requests[0]
    if command.session_id != first.goal.session_id:
        raise AuxiliarySemanticVerificationPersistenceError(
            "semantic settlement belongs to another Session"
        )
    if any(
        request.goal.session_id != first.goal.session_id
        or request.goal.task_id != first.goal.task_id
        or request.goal.auxiliary_graph_id != first.goal.auxiliary_graph_id
        or request.goal.goal_id != first.goal.goal_id
        or request.auxiliary_graph_revision != first.auxiliary_graph_revision
        or request.prompt_payload.payload_sha256
        != first.prompt_payload.payload_sha256
        or request.review_policy != first.review_policy
        for request in requests
    ):
        raise AuxiliarySemanticVerificationPersistenceError(
            "semantic settlement crossed frozen graph or prompt authority"
        )
    dispositions = {result.host_disposition for result in results}
    if TaskGraphSemanticVerificationDisposition.BLOCKED in dispositions:
        host_disposition = TaskGraphSemanticVerificationDisposition.BLOCKED
    elif TaskGraphSemanticVerificationDisposition.REVISE in dispositions:
        host_disposition = TaskGraphSemanticVerificationDisposition.REVISE
    else:
        host_disposition = TaskGraphSemanticVerificationDisposition.PASS
    payload = {
        "schema_version": "auxiliary-semantic-quorum-settlement-v1",
        "settlement_id": command.settlement_id,
        "session_id": first.goal.session_id,
        "task_id": first.goal.task_id,
        "auxiliary_graph_id": first.goal.auxiliary_graph_id,
        "goal_id": first.goal.goal_id,
        "auxiliary_graph_revision": first.auxiliary_graph_revision,
        "required_reviewer_count": first.required_reviewer_count,
        "frozen_prompt_payload_sha256": first.prompt_payload.payload_sha256,
        "review_policy": first.review_policy.model_dump(mode="json"),
        "requests": [item.model_dump(mode="json") for item in requests],
        "results": [item.model_dump(mode="json") for item in results],
        "host_disposition": host_disposition.value,
        "created_turn_id": command.created_turn_id,
    }
    return StoredAuxiliarySemanticQuorumSettlement(
        settlement_id=command.settlement_id,
        session_id=first.goal.session_id,
        task_id=first.goal.task_id,
        auxiliary_graph_id=first.goal.auxiliary_graph_id,
        goal_id=first.goal.goal_id,
        auxiliary_graph_revision=first.auxiliary_graph_revision,
        required_reviewer_count=first.required_reviewer_count,
        frozen_prompt_payload_sha256=first.prompt_payload.payload_sha256,
        review_policy=first.review_policy,
        requests=requests,
        results=results,
        host_disposition=host_disposition,
        settlement_sha256=_sha256_value(payload),
        created_turn_id=command.created_turn_id,
    )


def _settlement_payload(
    settlement: StoredAuxiliarySemanticQuorumSettlement,
) -> dict[str, Any]:
    return {
        "schema_version": "auxiliary-semantic-quorum-settlement-v1",
        "settlement_id": settlement.settlement_id,
        "session_id": settlement.session_id,
        "task_id": settlement.task_id,
        "auxiliary_graph_id": settlement.auxiliary_graph_id,
        "goal_id": settlement.goal_id,
        "auxiliary_graph_revision": settlement.auxiliary_graph_revision,
        "required_reviewer_count": settlement.required_reviewer_count,
        "frozen_prompt_payload_sha256": (
            settlement.frozen_prompt_payload_sha256
        ),
        "review_policy": settlement.review_policy.model_dump(mode="json"),
        "requests": [item.model_dump(mode="json") for item in settlement.requests],
        "results": [item.model_dump(mode="json") for item in settlement.results],
        "host_disposition": settlement.host_disposition.value,
        "created_turn_id": settlement.created_turn_id,
    }


def _insert_settlement(
    conn: sqlite3.Connection,
    *,
    settlement: StoredAuxiliarySemanticQuorumSettlement,
    now: str,
) -> None:
    policy = settlement.review_policy
    conn.execute(
        "INSERT INTO insession_auxiliary_semantic_quorum_settlements "
        "(settlement_id, session_id, insession_task_id, auxiliary_graph_id, "
        "goal_id, auxiliary_graph_revision, required_reviewer_count, "
        "frozen_prompt_payload_sha256, review_policy_json, "
        "review_policy_sha256, policy_source_sha256, distinct_document_count, "
        "has_visual_input, requires_protected_effect, "
        "modifies_executed_task_graph, request_ids_json, result_ids_json, "
        "host_disposition, settlement_json, settlement_sha256, "
        "created_turn_id, created_at) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            settlement.settlement_id,
            settlement.session_id,
            settlement.task_id,
            settlement.auxiliary_graph_id,
            settlement.goal_id,
            settlement.auxiliary_graph_revision,
            settlement.required_reviewer_count,
            settlement.frozen_prompt_payload_sha256,
            _model_json(policy),
            policy.policy_sha256,
            policy.policy_source_sha256,
            policy.distinct_document_count,
            int(policy.has_visual_input),
            int(policy.requires_protected_effect),
            int(policy.modifies_executed_task_graph),
            _canonical_json(
                [item.verification_request_id for item in settlement.requests]
            ),
            _canonical_json(
                [item.verification_result_id for item in settlement.results]
            ),
            settlement.host_disposition.value,
            _model_json(settlement),
            settlement.settlement_sha256,
            settlement.created_turn_id,
            now,
        ),
    )
    for request, result in zip(
        settlement.requests, settlement.results, strict=True
    ):
        conn.execute(
            "INSERT INTO insession_auxiliary_semantic_quorum_reviewers "
            "(settlement_id, session_id, insession_task_id, "
            "auxiliary_graph_id, goal_id, auxiliary_graph_revision, "
            "frozen_prompt_payload_sha256, review_policy_sha256, "
            "reviewer_ordinal, required_reviewer_count, "
            "verification_request_id, request_binding_sha256, "
            "verification_result_id, result_sha256) VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                settlement.settlement_id,
                settlement.session_id,
                settlement.task_id,
                settlement.auxiliary_graph_id,
                settlement.goal_id,
                settlement.auxiliary_graph_revision,
                settlement.frozen_prompt_payload_sha256,
                settlement.review_policy.policy_sha256,
                request.reviewer_ordinal,
                settlement.required_reviewer_count,
                request.verification_request_id,
                request.binding_sha256,
                result.verification_result_id,
                result.result_sha256,
            ),
        )


def _load_settlement(
    conn: sqlite3.Connection,
    settlement_id: str,
) -> StoredAuxiliarySemanticQuorumSettlement:
    row = conn.execute(
        "SELECT * FROM insession_auxiliary_semantic_quorum_settlements "
        "WHERE settlement_id=?",
        (settlement_id,),
    ).fetchone()
    if row is None:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "semantic quorum settlement is missing"
        )
    settlement_json = str(row["settlement_json"])
    try:
        settlement = StoredAuxiliarySemanticQuorumSettlement.model_validate_json(
            settlement_json
        )
        policy = TaskGraphSemanticReviewPolicy.model_validate_json(
            str(row["review_policy_json"])
        )
    except (TypeError, ValueError) as exc:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "stored semantic settlement JSON is invalid"
        ) from exc
    mirrored = (
        settlement.settlement_id == settlement_id
        and settlement_json == _model_json(settlement)
        and settlement.session_id == str(row["session_id"])
        and settlement.task_id == str(row["insession_task_id"])
        and settlement.auxiliary_graph_id == str(row["auxiliary_graph_id"])
        and settlement.goal_id == str(row["goal_id"])
        and settlement.auxiliary_graph_revision
        == int(row["auxiliary_graph_revision"])
        and settlement.required_reviewer_count
        == int(row["required_reviewer_count"])
        and settlement.frozen_prompt_payload_sha256
        == str(row["frozen_prompt_payload_sha256"])
        and settlement.review_policy == policy
        and _model_json(policy) == str(row["review_policy_json"])
        and policy.policy_sha256 == str(row["review_policy_sha256"])
        and policy.policy_source_sha256 == str(row["policy_source_sha256"])
        and policy.distinct_document_count
        == int(row["distinct_document_count"])
        and policy.has_visual_input == bool(row["has_visual_input"])
        and policy.requires_protected_effect
        == bool(row["requires_protected_effect"])
        and policy.modifies_executed_task_graph
        == bool(row["modifies_executed_task_graph"])
        and _canonical_json(
            [item.verification_request_id for item in settlement.requests]
        )
        == str(row["request_ids_json"])
        and _canonical_json(
            [item.verification_result_id for item in settlement.results]
        )
        == str(row["result_ids_json"])
        and settlement.host_disposition.value == str(row["host_disposition"])
        and settlement.settlement_sha256 == str(row["settlement_sha256"])
        and settlement.settlement_sha256
        == _sha256_value(_settlement_payload(settlement))
        and settlement.created_turn_id == str(row["created_turn_id"])
    )
    if not mirrored:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "stored semantic settlement row and contract differ"
        )
    reviewer_rows = conn.execute(
        "SELECT * FROM insession_auxiliary_semantic_quorum_reviewers "
        "WHERE settlement_id=? ORDER BY reviewer_ordinal",
        (settlement_id,),
    ).fetchall()
    if len(reviewer_rows) != settlement.required_reviewer_count:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "stored semantic settlement reviewer manifest is incomplete"
        )
    for ordinal, (reviewer, request, result) in enumerate(
        zip(
            reviewer_rows,
            settlement.requests,
            settlement.results,
            strict=True,
        ),
        start=1,
    ):
        if (
            int(reviewer["reviewer_ordinal"]) != ordinal
            or int(reviewer["required_reviewer_count"])
            != settlement.required_reviewer_count
            or str(reviewer["session_id"]) != settlement.session_id
            or str(reviewer["insession_task_id"]) != settlement.task_id
            or str(reviewer["auxiliary_graph_id"])
            != settlement.auxiliary_graph_id
            or str(reviewer["goal_id"]) != settlement.goal_id
            or int(reviewer["auxiliary_graph_revision"])
            != settlement.auxiliary_graph_revision
            or str(reviewer["frozen_prompt_payload_sha256"])
            != settlement.frozen_prompt_payload_sha256
            or str(reviewer["review_policy_sha256"])
            != settlement.review_policy.policy_sha256
            or str(reviewer["verification_request_id"])
            != request.verification_request_id
            or str(reviewer["request_binding_sha256"])
            != request.binding_sha256
            or str(reviewer["verification_result_id"])
            != result.verification_result_id
            or str(reviewer["result_sha256"]) != result.result_sha256
            or _load_request(conn, request.verification_request_id).request
            != request
            or _load_result(conn, result.verification_result_id).result
            != result
        ):
            raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
                "stored semantic settlement reviewer binding is corrupt"
            )
    try:
        validate_task_graph_semantic_verification_quorum(
            requests=settlement.requests,
            results=settlement.results,
        )
    except ValueError as exc:
        raise AuxiliarySemanticVerificationStoredAuthorityCorrupt(
            "stored semantic settlement no longer forms a valid quorum"
        ) from exc
    return settlement


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


def _sha256_value(value: object) -> str:
    return _sha256_text(_canonical_json(value))


def _require_id(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 200
    ):
        raise ValueError(f"{name} must be a canonical identifier")


def _require_sha(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase sha256")


__all__ = [
    "AuxiliarySemanticCapabilityCatalogMutationResult",
    "AuxiliarySemanticQuorumSettlementMutationResult",
    "AuxiliarySemanticVerificationIdentityCollision",
    "AuxiliarySemanticVerificationPersistenceError",
    "AuxiliarySemanticVerificationRequestMutationResult",
    "AuxiliarySemanticVerificationResultMutationResult",
    "AuxiliarySemanticVerificationStaleAuthority",
    "AuxiliarySemanticVerificationStoredAuthorityCorrupt",
    "AuxiliarySemanticMaterialProjection",
    "CommitAuxiliarySemanticVerificationRequestCommand",
    "CommitAuxiliarySemanticVerificationResultCommand",
    "FreezeAuxiliarySemanticCapabilityCatalogCommand",
    "SettleAuxiliarySemanticVerificationQuorumCommand",
    "StoredAuxiliarySemanticCapabilityCatalog",
    "StoredAuxiliarySemanticQuorumSettlement",
    "StoredAuxiliarySemanticVerificationRequest",
    "StoredAuxiliarySemanticVerificationResult",
    "_load_auxiliary_semantic_quorum_settlement",
    "commit_auxiliary_semantic_verification_request",
    "commit_auxiliary_semantic_verification_result",
    "derive_auxiliary_semantic_review_policy",
    "freeze_auxiliary_semantic_capability_catalog",
    "get_auxiliary_semantic_capability_catalog",
    "get_auxiliary_semantic_quorum_settlement",
    "get_auxiliary_semantic_verification_request",
    "get_auxiliary_semantic_verification_result",
    "project_auxiliary_semantic_material",
    "settle_auxiliary_semantic_verification_quorum",
]
