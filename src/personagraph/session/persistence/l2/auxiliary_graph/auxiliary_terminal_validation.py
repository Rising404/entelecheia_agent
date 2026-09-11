"""Store 所有的当前 AuxiliaryGraph 终态提案来源权威。

终态规划器可以引用已完成 Host 原语发现的证据，但绝不能伪造证据或选择任意制品。本模块
从当前图派生完整验证上下文，并重新认证每项选中的原语预留、观察、验证回执、私有
制品、Prompt 投影和来源卡。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass

from personagraph.l2.auxiliary_graph import (
    PlanningAuthorityAnchor,
    PlanningAuthorityClass,
    PlanningAuthorityOriginKind,
    PlanningAuthoritySourceCard,
    PlanningAuthoritySourceKind,
    PlanningAuthoritySnapshot,
    PlanningContextArtifactProjection,
    PlanningContextArtifact,
)
from personagraph.l2.task_graph import (
    InSessionTaskGraphRevisionValidationContext,
    InSessionTaskSourceAnchor,
)
from personagraph.l2.work_run import AuxiliaryNodeSubject
from . import auxiliary_graphs
from .auxiliary_graph_errors import AuxiliaryGraphPersistenceError
from .auxiliary_dependencies import AuxiliaryDependencyPersistenceError, _resolve_host_dependency
from ..planning.auxiliary_planning_completions import (
    AuxiliaryInitialPlanningPersistenceError,
    load_authenticated_auxiliary_initial_planning_completion,
)
from ...deps import StoreDeps
from ..task_graph.insession_tasks import _load_authoritative_user_input
from ..planning.primitive_invocations import (
    PlanningPrimitiveInvocationPersistenceError,
    _load_by_call_id as _load_planning_primitive_invocation,
)


class AuxiliaryTerminalValidationContextError(AuxiliaryGraphPersistenceError):
    """当前图无法产生受信终态 TaskGraph 来源。"""


class AuxiliaryTerminalSemanticAuthorityMismatch(
    AuxiliaryTerminalValidationContextError
):
    """冻结语义卡与规范 Store 来源权威不一致。"""


@dataclass(frozen=True, slots=True)
class AuxiliaryTerminalSemanticSupport:
    """终态冻结前可用的 Store 认证 Prompt 支撑信息。"""

    validation_context: InSessionTaskGraphRevisionValidationContext
    context_artifacts: tuple[PlanningContextArtifactProjection, ...]
    observation_source_cards: tuple[PlanningAuthoritySourceCard, ...]


@dataclass(frozen=True, slots=True)
class _CanonicalSourceCardBinding:
    alias: str
    authority_class: PlanningAuthorityClass
    source_kind: PlanningAuthoritySourceKind
    excerpt: str
    projection_sha256: str


def build_auxiliary_terminal_task_graph_validation_context(
    deps: StoreDeps,
    *,
    session_id: str,
    invocation_turn_id: str,
    task_id: str,
) -> InSessionTaskGraphRevisionValidationContext:
    """为当前 构建唯一可接受的 TaskGraph 验证上下文。

    调用方只提供图作用域。制品身份、来源别名、摘录和完成成员关系全部由 Store 在一个
    读取事务内选择并验证。
    """

    for name, value in (
        ("session_id", session_id),
        ("invocation_turn_id", invocation_turn_id),
        ("task_id", task_id),
    ):
        _require_identifier(name, value)
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        task_authority = conn.execute(
            "SELECT current_graph_revision, current_status, state_version, "
            "created_turn_id, creation_source_start, creation_source_end, "
            "creation_source_sha256 FROM insession_tasks "
            "WHERE session_id=? AND insession_task_id=?",
            (session_id, task_id),
        ).fetchone()
        if task_authority is None:
            raise AuxiliaryTerminalValidationContextError(
                "terminal validation targets an unknown Task"
            )
        context = _build_auxiliary_terminal_validation_context(
            conn,
            session_id=session_id,
            invocation_turn_id=invocation_turn_id,
            task_id=task_id,
            task_authority=task_authority,
        )
        conn.commit()
        return context
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def project_auxiliary_terminal_semantic_support(
    deps: StoreDeps,
    *,
    session_id: str,
    invocation_turn_id: str,
    task_id: str,
) -> AuxiliaryTerminalSemanticSupport:
    """投影相同终态来源及带类型 Host 语义制品。"""

    for name, value in (
        ("session_id", session_id),
        ("invocation_turn_id", invocation_turn_id),
        ("task_id", task_id),
    ):
        _require_identifier(name, value)
    deps.init_db()
    conn = deps.connect()
    try:
        conn.execute("BEGIN")
        task_authority = conn.execute(
            "SELECT current_graph_revision, current_status, state_version, "
            "created_turn_id, creation_source_start, creation_source_end, "
            "creation_source_sha256 FROM insession_tasks "
            "WHERE session_id=? AND insession_task_id=?",
            (session_id, task_id),
        ).fetchone()
        if task_authority is None:
            raise AuxiliaryTerminalValidationContextError(
                "terminal semantic support targets an unknown Task"
            )
        support = _build_auxiliary_terminal_semantic_support(
            conn,
            session_id=session_id,
            invocation_turn_id=invocation_turn_id,
            task_id=task_id,
            task_authority=task_authority,
        )
        conn.commit()
        return support
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def _build_auxiliary_terminal_validation_context(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    invocation_turn_id: str,
    task_id: str,
    task_authority: sqlite3.Row,
    allowed_graph_statuses: frozenset[str] = frozenset({"active"}),
    expected_authority_cards: tuple[PlanningAuthoritySourceCard, ...] | None = None,
    require_current_task_graph_revision: bool = True,
) -> InSessionTaskGraphRevisionValidationContext:
    """使用现有 Store 事务派生规范上下文。"""

    return _build_auxiliary_terminal_semantic_support(
        conn,
        session_id=session_id,
        invocation_turn_id=invocation_turn_id,
        task_id=task_id,
        task_authority=task_authority,
        allowed_graph_statuses=allowed_graph_statuses,
        expected_authority_cards=expected_authority_cards,
        require_current_task_graph_revision=require_current_task_graph_revision,
    ).validation_context


def _build_auxiliary_terminal_semantic_support(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    invocation_turn_id: str,
    task_id: str,
    task_authority: sqlite3.Row,
    allowed_graph_statuses: frozenset[str] = frozenset({"active"}),
    expected_authority_cards: tuple[PlanningAuthoritySourceCard, ...] | None = None,
    require_current_task_graph_revision: bool = True,
) -> AuxiliaryTerminalSemanticSupport:
    """在一个读取快照中派生终态验证和语义支撑。"""

    if not allowed_graph_statuses or not allowed_graph_statuses.issubset(
        {"active", "proposal_ready", "gapped_ready", "committed"}
    ):
        raise ValueError("allowed graph statuses must be canonical terminal states")

    _require_turn_task_scope(
        conn,
        session_id=session_id,
        invocation_turn_id=invocation_turn_id,
        task_id=task_id,
    )
    details = auxiliary_graphs._load_auxiliary_graph(conn, session_id, task_id)
    observed_task_graph_revision = (
        int(task_authority["current_graph_revision"])
        if task_authority["current_graph_revision"] is not None
        else None
    )
    context_task_graph_revision = (
        observed_task_graph_revision
        if require_current_task_graph_revision
        else details.base_task_graph_revision
    )
    if (
        str(task_authority["current_status"]) in {"completed", "cancelled"}
        or details.session_id != session_id
        or details.task_id != task_id
        or (
            require_current_task_graph_revision
            and details.base_task_graph_revision != observed_task_graph_revision
        )
        or details.goal_status not in allowed_graph_statuses
        or details.revision_status != details.goal_status
        or details.revision is None
        or details.authority_snapshot is None
    ):
        raise AuxiliaryTerminalValidationContextError(
            "terminal validation requires the current nonterminal formal revision"
        )
    terminal = next(
        (
            node
            for node in details.nodes
            if node.auxiliary_node_id == details.terminal_auxiliary_node_id
        ),
        None,
    )
    if terminal is None or terminal.executor_kind != "terminal_planner":
        raise AuxiliaryTerminalValidationContextError(
            "current terminal planner authority is missing"
        )

    creation_anchor = _task_creation_anchor(
        conn,
        session_id=session_id,
        task_authority=task_authority,
    )
    creation_binding = _task_creation_card_binding(
        details=details,
        creation_anchor=creation_anchor,
    )
    ancestor_ids = _terminal_ancestor_ids(details)
    evidence_anchors: list[InSessionTaskSourceAnchor] = []
    context_artifacts: list[PlanningContextArtifactProjection] = []
    observation_source_cards: list[PlanningAuthoritySourceCard] = []
    card_bindings: list[_CanonicalSourceCardBinding] = [creation_binding]
    seen_aliases = {creation_anchor.anchor_id}
    resource_cards = _planning_resource_source_cards(
        conn,
        session_id=session_id,
        task_id=task_id,
        authority_snapshot=details.authority_snapshot,
    )
    for card in resource_cards:
        if card.alias in seen_aliases:
            raise AuxiliaryTerminalValidationContextError(
                "terminal planning-resource aliases collide"
            )
        seen_aliases.add(card.alias)
        evidence_anchors.append(
            InSessionTaskSourceAnchor(
                anchor_id=card.alias,
                source_turn_id=details.authority_snapshot.source_turn_id,
                source_kind=_task_source_kind(card.source_kind),
                start=0,
                end=len(card.excerpt),
                excerpt=card.excerpt,
            )
        )
        observation_source_cards.append(card)
        card_bindings.append(
            _CanonicalSourceCardBinding(
                alias=card.alias,
                authority_class=card.authority_class,
                source_kind=card.source_kind,
                excerpt=card.excerpt,
                projection_sha256=card.projection_sha256,
            )
        )
    for producer in sorted(details.nodes, key=lambda item: item.ordinal):
        if (
            producer.auxiliary_node_id not in ancestor_ids
            or producer.executor_kind != "host_primitive"
            or producer.status != "completed"
        ):
            continue
        subject = AuxiliaryNodeSubject(
            task_id=task_id,
            auxiliary_graph_id=details.auxiliary_graph_id,
            auxiliary_graph_revision=details.auxiliary_graph_revision,
            node_id=producer.auxiliary_node_id,
            node_revision=producer.node_revision,
        )
        try:
            dependency = _resolve_host_dependency(
                conn,
                details=details,
                producer=producer,
                subject=subject,
            )
            cards, source_turn_id, gap_blocking_by_alias = (
                _load_verified_host_source_cards(
                    conn,
                    details=details,
                    producer=producer,
                    subject=subject,
                    artifact_projection=dependency.artifact,
                    expected_artifact_id=dependency.completion_id,
                )
            )
        except (
            AuxiliaryDependencyPersistenceError,
            PlanningPrimitiveInvocationPersistenceError,
            TypeError,
            ValueError,
        ) as exc:
            raise AuxiliaryTerminalValidationContextError(
                "terminal Host evidence lost its sealed completion authority"
            ) from exc
        context_artifacts.append(dependency.artifact)
        observation_source_cards.extend(cards)
        for card in cards:
            if card.alias in seen_aliases:
                raise AuxiliaryTerminalValidationContextError(
                    "terminal Host evidence aliases collide"
                )
            seen_aliases.add(card.alias)
            evidence_anchors.append(
                InSessionTaskSourceAnchor(
                    anchor_id=card.alias,
                    source_turn_id=source_turn_id,
                    source_kind=_task_source_kind(card.source_kind),
                    gap_blocking=gap_blocking_by_alias.get(card.alias),
                    start=0,
                    end=len(card.excerpt),
                    excerpt=card.excerpt,
                )
            )
            card_bindings.append(
                _CanonicalSourceCardBinding(
                    alias=card.alias,
                    authority_class=card.authority_class,
                    source_kind=card.source_kind,
                    excerpt=card.excerpt,
                    projection_sha256=card.projection_sha256,
                )
            )

    if expected_authority_cards is not None:
        _require_exact_semantic_authority_cards(
            expected=tuple(card_bindings),
            actual=expected_authority_cards,
        )

    try:
        context = InSessionTaskGraphRevisionValidationContext(
            session_id=session_id,
            source_turn_id=invocation_turn_id,
            target_insession_task_id=task_id,
            expected_current_graph_revision=context_task_graph_revision,
            source_anchors=(creation_anchor, *evidence_anchors),
            authorization_anchor_ids=(creation_anchor.anchor_id,),
            required_anchor_ids=(creation_anchor.anchor_id,),
        )
    except (TypeError, ValueError) as exc:
        raise AuxiliaryTerminalValidationContextError(
            "terminal source cards exceed the TaskGraph validation contract"
        ) from exc
    return AuxiliaryTerminalSemanticSupport(
        validation_context=context,
        context_artifacts=tuple(context_artifacts),
        observation_source_cards=tuple(
            sorted(observation_source_cards, key=lambda item: item.alias)
        ),
    )


def _require_turn_task_scope(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    invocation_turn_id: str,
    task_id: str,
) -> None:
    turn = conn.execute(
        "SELECT session_id FROM runtime_turns WHERE turn_id=?",
        (invocation_turn_id,),
    ).fetchone()
    linked = conn.execute(
        "SELECT 1 FROM insession_task_turn_links WHERE session_id=? "
        "AND turn_id=? AND insession_task_id=?",
        (session_id, invocation_turn_id, task_id),
    ).fetchone()
    if turn is None or str(turn["session_id"]) != session_id or linked is None:
        raise AuxiliaryTerminalValidationContextError(
            "terminal validation Turn is outside the current Task scope"
        )


def _task_creation_anchor(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    task_authority: sqlite3.Row,
) -> InSessionTaskSourceAnchor:
    creation_turn_id = str(task_authority["created_turn_id"])
    source_start = int(task_authority["creation_source_start"])
    source_end = int(task_authority["creation_source_end"])
    source_sha256 = str(task_authority["creation_source_sha256"])
    authoritative_text = _load_authoritative_user_input(
        conn,
        session_id=session_id,
        turn_id=creation_turn_id,
    )
    if (
        source_start < 0
        or source_end <= source_start
        or source_end > len(authoritative_text)
    ):
        raise AuxiliaryTerminalValidationContextError(
            "Task creation source span is invalid during terminal validation"
        )
    excerpt = authoritative_text[source_start:source_end]
    if _sha256_text(excerpt) != source_sha256:
        raise AuxiliaryTerminalValidationContextError(
            "Task creation source hash changed during terminal validation"
        )
    return InSessionTaskSourceAnchor(
        anchor_id="task_creation_source",
        source_turn_id=creation_turn_id,
        source_kind="previously_authorized_task_state",
        start=source_start,
        end=source_end,
        excerpt=excerpt,
    )


def _task_creation_card_binding(
    *,
    details: auxiliary_graphs.StoredAuxiliaryGraphDetails,
    creation_anchor: InSessionTaskSourceAnchor,
) -> _CanonicalSourceCardBinding:
    snapshot = details.authority_snapshot
    if snapshot is None:
        raise AuxiliaryTerminalValidationContextError(
            "current authority snapshot is missing"
        )
    matches = tuple(
        anchor
        for anchor in snapshot.anchors
        if anchor.projection_alias == creation_anchor.anchor_id
    )
    if len(matches) != 1:
        raise AuxiliaryTerminalValidationContextError(
            "Task creation source has no unique snapshot authority"
        )
    authority = matches[0]
    source_kinds = {
        PlanningAuthorityOriginKind.USER_INSTRUCTION_SPAN: (
            PlanningAuthoritySourceKind.USER_INSTRUCTION
        ),
        PlanningAuthorityOriginKind.USER_ANSWER_SPAN: (
            PlanningAuthoritySourceKind.USER_ANSWER
        ),
        PlanningAuthorityOriginKind.PRIOR_TASK_STATE: (
            PlanningAuthoritySourceKind.PRIOR_TASK_STATE
        ),
    }
    source_kind = source_kinds.get(authority.origin_kind)
    excerpt_sha256 = _sha256_text(creation_anchor.excerpt)
    if (
        source_kind is None
        or authority.authority_class is not PlanningAuthorityClass.AUTHORIZATION
        or authority.origin_id != creation_anchor.source_turn_id
        or authority.span_start != creation_anchor.start
        or authority.span_end != creation_anchor.end
        or authority.content_sha256 != excerpt_sha256
        or authority.projection_sha256 != excerpt_sha256
    ):
        raise AuxiliaryTerminalValidationContextError(
            "Task creation source differs from its frozen snapshot authority"
        )
    return _CanonicalSourceCardBinding(
        alias=creation_anchor.anchor_id,
        authority_class=authority.authority_class,
        source_kind=source_kind,
        excerpt=creation_anchor.excerpt,
        projection_sha256=authority.projection_sha256,
    )


def _require_exact_semantic_authority_cards(
    *,
    expected: tuple[_CanonicalSourceCardBinding, ...],
    actual: tuple[PlanningAuthoritySourceCard, ...],
) -> None:
    expected_by_alias = {item.alias: item for item in expected}
    actual_by_alias = {item.alias: item for item in actual}
    if (
        len(expected_by_alias) != len(expected)
        or len(actual_by_alias) != len(actual)
        or set(expected_by_alias) != set(actual_by_alias)
    ):
        raise AuxiliaryTerminalSemanticAuthorityMismatch(
            "semantic authority cards do not exactly cover canonical Store sources"
        )
    for alias, binding in expected_by_alias.items():
        card = actual_by_alias[alias]
        if (
            card.authority_class is not binding.authority_class
            or card.source_kind is not binding.source_kind
            or card.excerpt != binding.excerpt
            or card.projection_sha256 != binding.projection_sha256
        ):
            raise AuxiliaryTerminalSemanticAuthorityMismatch(
                "semantic authority card differs from canonical Store source authority"
            )


def _terminal_ancestor_ids(
    details: auxiliary_graphs.StoredAuxiliaryGraphDetails,
) -> frozenset[str]:
    node_ids = {node.auxiliary_node_id for node in details.nodes}
    parents: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for edge in details.edges:
        if (
            edge.dependency_auxiliary_node_id not in node_ids
            or edge.consumer_auxiliary_node_id not in node_ids
        ):
            raise AuxiliaryTerminalValidationContextError(
                "current edge crossed its revision membership"
            )
        parents[edge.consumer_auxiliary_node_id].append(
            edge.dependency_auxiliary_node_id
        )
    ancestors = {details.terminal_auxiliary_node_id}
    stack = [details.terminal_auxiliary_node_id]
    while stack:
        current = stack.pop()
        for parent in parents[current]:
            if parent not in ancestors:
                ancestors.add(parent)
                stack.append(parent)
    return frozenset(ancestors)


def _load_verified_host_source_cards(
    conn: sqlite3.Connection,
    *,
    details: auxiliary_graphs.StoredAuxiliaryGraphDetails,
    producer: auxiliary_graphs.StoredAuxiliaryGraphNode,
    subject: AuxiliaryNodeSubject,
    artifact_projection: PlanningContextArtifactProjection,
    expected_artifact_id: str,
) -> tuple[
    tuple[PlanningAuthoritySourceCard, ...],
    str,
    dict[str, bool],
]:
    rows = conn.execute(
        "SELECT artifact.artifact_json, observation.snapshot_json, "
        "observation.snapshot_sha256, observation.created_turn_id, "
        "receipt.receipt_json, receipt.receipt_sha256, "
        "artifact.producer_primitive_call_id "
        "FROM insession_auxiliary_planning_context_artifacts AS artifact "
        "JOIN insession_auxiliary_observations AS observation "
        "ON observation.observation_id=artifact.producer_primitive_call_id "
        "JOIN insession_auxiliary_context_verification_receipts AS receipt "
        "ON receipt.verification_receipt_id=artifact.verification_receipt_id "
        "AND receipt.source_observation_id=observation.observation_id "
        "WHERE artifact.artifact_id=? AND artifact.session_id=? "
        "AND artifact.insession_task_id=? AND artifact.auxiliary_graph_id=? "
        "AND artifact.goal_id=? "
        "AND artifact.producer_auxiliary_graph_revision=? "
        "AND artifact.producer_auxiliary_node_id=? "
        "AND artifact.producer_node_revision=?",
        (
            expected_artifact_id,
            details.session_id,
            details.task_id,
            details.auxiliary_graph_id,
            details.goal_id,
            details.auxiliary_graph_revision,
            producer.auxiliary_node_id,
            producer.node_revision,
        ),
    ).fetchall()
    if len(rows) != 1:
        raise AuxiliaryTerminalValidationContextError(
            "completed Host node has no unique current-revision artifact"
        )
    row = rows[0]
    primitive_call_id = str(row["producer_primitive_call_id"])
    reservation = _load_planning_primitive_invocation(conn, primitive_call_id)
    if reservation is None:
        raise AuxiliaryTerminalValidationContextError(
            "completed Host node lost its primitive reservation"
        )
    observation_json = str(row["snapshot_json"])
    receipt_json = str(row["receipt_json"])
    artifact_json = str(row["artifact_json"])
    try:
        observation = _mapping(json.loads(observation_json))
        result = _mapping(observation.get("result"))
        prompt_inputs = _mapping(result.get("prompt_inputs"))
        raw_cards = prompt_inputs.get("source_cards")
        if not isinstance(raw_cards, list):
            raise ValueError("source cards are not a list")
        cards = tuple(
            PlanningAuthoritySourceCard.model_validate(item)
            for item in raw_cards
        )
        projection = PlanningContextArtifactProjection.model_validate(
            prompt_inputs.get("context_artifact")
        )
        artifact = PlanningContextArtifact.model_validate_json(artifact_json)
        anchors = tuple(
            PlanningAuthorityAnchor.model_validate(item)
            for item in result.get("authority_anchors", ())
        )
        receipt = _mapping(json.loads(receipt_json))
        receipt_anchors = tuple(
            PlanningAuthorityAnchor.model_validate(item)
            for item in receipt.get("authority_anchors", ())
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise AuxiliaryTerminalValidationContextError(
            "Host source-card settlement contains invalid typed JSON"
        ) from exc

    binding = reservation.invocation.binding
    aliases = tuple(card.alias for card in cards)
    anchors_by_alias = {anchor.projection_alias: anchor for anchor in anchors}
    evidence_by_alias = {
        evidence.source_alias: evidence for evidence in artifact.evidence_refs
    }
    gap_aliases = {gap.gap_id for gap in artifact.gaps}
    valid = (
        observation.get("schema_version")
        == "sealed-planning-host-primitive-observation-v1"
        and observation.get("primitive_kind")
        == reservation.invocation.primitive_kind.value
        and observation_json == _canonical_json(observation)
        and _sha256_text(observation_json) == str(row["snapshot_sha256"])
        and receipt_json == _canonical_json(receipt)
        and _sha256_text(receipt_json) == str(row["receipt_sha256"])
        and reservation.status == "settled"
        and reservation.settled_observation_id == primitive_call_id
        and reservation.settled_artifact_id == expected_artifact_id
        and reservation.settlement_sha256 == result.get("settlement_sha256")
        and reservation.invocation.invocation_turn_id
        == str(row["created_turn_id"])
        and binding.producer_auxiliary_node == subject
        and binding.authority_snapshot_id == details.authority_snapshot_id
        and artifact.producer_auxiliary_node == subject
        and artifact.artifact_id == expected_artifact_id
        and artifact_json == _model_json(artifact)
        and result.get("artifact") == artifact.model_dump(mode="json")
        and projection == artifact_projection
        and projection.artifact_id == expected_artifact_id
        and anchors == receipt_anchors
        and aliases == tuple(sorted(aliases))
        and len(aliases) == len(set(aliases))
        and set(aliases) == set(anchors_by_alias)
        and set(aliases) == set(evidence_by_alias) | gap_aliases
        and all(
            anchor.authority_snapshot_id == details.authority_snapshot_id
            and anchor.authority_class is not PlanningAuthorityClass.AUTHORIZATION
            for anchor in anchors
        )
        and all(
            card.authority_class == anchors_by_alias[card.alias].authority_class
            and card.projection_sha256
            == anchors_by_alias[card.alias].projection_sha256
            and card.projection_sha256
            == _source_card_projection_sha256(
                card,
                resource_perception=(
                    reservation.invocation.primitive_kind.value
                    == "resource_perception"
                ),
            )
            for card in cards
        )
        and all(
            evidence.evidence_anchor_id
            == anchors_by_alias[alias].anchor_id
            and anchors_by_alias[alias].authority_class
            is PlanningAuthorityClass.EVIDENCE
            for alias, evidence in evidence_by_alias.items()
        )
        and all(
            anchors_by_alias[alias].authority_class
            is PlanningAuthorityClass.GAP
            for alias in gap_aliases
        )
    )
    if not valid:
        raise AuxiliaryTerminalValidationContextError(
            "Host source cards differ from their sealed artifact authority"
        )
    return (
        cards,
        str(row["created_turn_id"]),
        {gap.gap_alias: gap.blocking for gap in projection.gaps},
    )


def _task_source_kind(source_kind: PlanningAuthoritySourceKind) -> str:
    mapping = {
        PlanningAuthoritySourceKind.DOCUMENT: "retrieved_document",
        PlanningAuthoritySourceKind.VISUAL: "attachment",
        PlanningAuthoritySourceKind.WORKSPACE: "tool_observation",
        PlanningAuthoritySourceKind.TOOL_OBSERVATION: "tool_observation",
        PlanningAuthoritySourceKind.MEMORY: "memory",
        PlanningAuthoritySourceKind.ARTIFACT: "tool_observation",
        PlanningAuthoritySourceKind.GAP: "gap",
    }
    try:
        return mapping[source_kind]
    except KeyError as exc:
        raise AuxiliaryTerminalValidationContextError(
            "Host source card attempted to create authorization authority"
        ) from exc


def _planning_resource_source_cards(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    task_id: str,
    authority_snapshot: PlanningAuthoritySnapshot,
) -> tuple[PlanningAuthoritySourceCard, ...]:
    """恢复 TaskGraph 可读资源的持久 Prompt 卡。

    初始 Architect 回执已密封 Prompt 安全文档卡。复用该精确值可防止措辞变化使重放失效，
    也可防止语义重规划在同一别名下看到两张不同卡。原始视觉别名会刻意排除，直到普通
    TaskGraph WorkRun 暴露匹配的已挂载视觉工具桥。
    """

    try:
        completion = load_authenticated_auxiliary_initial_planning_completion(
            conn,
            session_id=session_id,
            task_id=task_id,
        )
    except AuxiliaryInitialPlanningPersistenceError as exc:
        raise AuxiliaryTerminalValidationContextError(
            "terminal planning-resource receipt is corrupt"
        ) from exc
    if completion is None:
        return ()
    initial_projection = (
        completion.binding.architect_request.prompt_payload.authority
    )
    initial_cards = {card.alias: card for card in initial_projection.cards}
    if len(initial_cards) != len(initial_projection.cards):
        raise AuxiliaryTerminalValidationContextError(
            "terminal planning-resource cards reuse an alias"
        )
    cards: list[PlanningAuthoritySourceCard] = []
    for anchor in sorted(
        authority_snapshot.anchors,
        key=lambda item: (item.item_ordinal, item.projection_alias),
    ):
        if (
            anchor.authority_class is not PlanningAuthorityClass.EVIDENCE
            or anchor.origin_kind
            is not PlanningAuthorityOriginKind.WORKSPACE_RESOURCE
        ):
            continue
        card = initial_cards.get(anchor.projection_alias)
        if (
            card is None
            or card.authority_class is not PlanningAuthorityClass.EVIDENCE
            or card.source_kind
            not in {
                PlanningAuthoritySourceKind.DOCUMENT,
                PlanningAuthoritySourceKind.WORKSPACE,
            }
            or card.projection_sha256 != anchor.projection_sha256
        ):
            raise AuxiliaryTerminalValidationContextError(
                "terminal planning-resource card differs from frozen authority"
            )
        cards.append(card)
    return tuple(cards)


def _source_card_projection_sha256(
    card: PlanningAuthoritySourceCard,
    *,
    resource_perception: bool,
) -> str:
    payload: dict[str, object] = {
        "alias": card.alias,
        "authority_class": card.authority_class.value,
        "source_kind": card.source_kind.value,
        "source_label": card.source_label,
        "excerpt": card.excerpt,
    }
    if not resource_perception:
        payload = {
            "schema_version": "planning-authority-source-card-v1",
            **payload,
        }
    if card.document_group_alias is not None:
        payload["document_group_alias"] = card.document_group_alias
    return _sha256_text(_canonical_json(payload))


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError("sealed Host value must be an object")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _model_json(value: object) -> str:
    return _canonical_json(
        value.model_dump(mode="json")  # type: ignore[union-attr]
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_identifier(name: str, value: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be one canonical non-empty identifier")


__all__ = [
    "AuxiliaryTerminalSemanticAuthorityMismatch",
    "AuxiliaryTerminalSemanticSupport",
    "AuxiliaryTerminalValidationContextError",
    "build_auxiliary_terminal_task_graph_validation_context",
    "project_auxiliary_terminal_semantic_support",
]
