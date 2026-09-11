"""L2 TaskGraph ``N -> N+1`` 修订的纯节点连续性规划。

模型可以提出谱系建议，但绝不分配持久节点标识，也不决定已完成的权威信息能否复用。
本模块将经过验证的完整快照提案和 Host 持有的谱系提示转换为精确的转换计划。
持久化层仍须在其提交事务中重新加载同一基础修订，并重新验证每个沿用候选项。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import (
    InSessionTaskAcceptanceProposal,
    InSessionTaskDetails,
    InSessionTaskGraphRevisionProposal,
    InSessionTaskNodeKind,
    InSessionTaskNodeProposal,
)


_LOCAL_KEY_PATTERN = r"^[a-z][a-z0-9_-]{0,63}$"
_DURABLE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskGraphNodeLineageDisposition(StrEnum):
    NEW = "new"
    REUSE = "reuse"
    REVISE = "revise"


class InSessionTaskNodeLineageHint(_Contract):
    """针对一个提案局部节点键的不可信谱系提示。"""

    node_key: str = Field(pattern=_LOCAL_KEY_PATTERN)
    disposition: TaskGraphNodeLineageDisposition
    base_node_alias: str | None = Field(default=None, pattern=_LOCAL_KEY_PATTERN)

    @model_validator(mode="after")
    def _validate_shape(self) -> 'InSessionTaskNodeLineageHint':
        if self.disposition is TaskGraphNodeLineageDisposition.NEW:
            if self.base_node_alias is not None:
                raise ValueError("new node lineage cannot name a base node")
        elif self.base_node_alias is None:
            raise ValueError("reuse/revise lineage requires a base node alias")
        return self


class InSessionTaskGraphRevisionLineageHints(_Contract):
    """经 Host 接受、绑定到精确提案和审查回执的谱系。

    此值可自验证但不能自授权：持久化层必须重新加载指定的生产评估和语义结果，
    要求其精确的 PASS 绑定，并在使用转换计划前重建基础别名命名空间。
    """

    schema_version: Literal["insession-task-graph-lineage-hints-v1"] = (
        "insession-task-graph-lineage-hints-v1"
    )
    task_id: str = Field(pattern=_DURABLE_ID_PATTERN)
    expected_base_graph_revision: int = Field(ge=1)
    task_graph_proposal_sha256: str = Field(pattern=_SHA256_PATTERN)
    production_evaluation_sha256: str = Field(pattern=_SHA256_PATTERN)
    semantic_verification_result_sha256s: tuple[str, ...] = Field(
        min_length=1,
        max_length=2,
    )
    hints: tuple[InSessionTaskNodeLineageHint, ...] = Field(
        min_length=1,
        max_length=512,
    )
    binding_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("semantic_verification_result_sha256s")
    @classmethod
    def _require_review_hashes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if (
            any(not _is_sha256(value) for value in values)
            or len(values) != len(set(values))
            or values != tuple(sorted(values))
        ):
            raise ValueError("semantic result hashes must be unique canonical SHA-256s")
        return values

    @field_validator("hints")
    @classmethod
    def _require_unique_bindings(
        cls,
        values: tuple[InSessionTaskNodeLineageHint, ...],
    ) -> tuple[InSessionTaskNodeLineageHint, ...]:
        node_keys = [item.node_key for item in values]
        if len(node_keys) != len(set(node_keys)):
            raise ValueError("lineage hints must bind each proposal node once")
        base_aliases = [
            item.base_node_alias for item in values if item.base_node_alias
        ]
        if len(base_aliases) != len(set(base_aliases)):
            raise ValueError("one base node cannot back multiple proposal nodes")
        return values

    @model_validator(mode="after")
    def _validate_binding(self) -> 'InSessionTaskGraphRevisionLineageHints':
        expected = canonical_insession_task_graph_revision_lineage_sha256(
            task_id=self.task_id,
            expected_base_graph_revision=self.expected_base_graph_revision,
            task_graph_proposal_sha256=self.task_graph_proposal_sha256,
            production_evaluation_sha256=self.production_evaluation_sha256,
            semantic_verification_result_sha256s=(
                self.semantic_verification_result_sha256s
            ),
            hints=self.hints,
        )
        if self.binding_sha256 != expected:
            raise ValueError("lineage authority hash does not match its payload")
        return self

    @classmethod
    def create(cls, **values: object) -> 'InSessionTaskGraphRevisionLineageHints':
        values = dict(values)
        hints = tuple(
            item
            if isinstance(item, InSessionTaskNodeLineageHint)
            else InSessionTaskNodeLineageHint.model_validate(item)
            for item in values["hints"]  # type: ignore[union-attr]
        )
        result_hashes = tuple(
            sorted(str(item) for item in values["semantic_verification_result_sha256s"])  # type: ignore[union-attr]
        )
        values["hints"] = hints
        values["semantic_verification_result_sha256s"] = result_hashes
        values["binding_sha256"] = canonical_insession_task_graph_revision_lineage_sha256(
            task_id=str(values["task_id"]),
            expected_base_graph_revision=int(values["expected_base_graph_revision"]),
            task_graph_proposal_sha256=str(values["task_graph_proposal_sha256"]),
            production_evaluation_sha256=str(values["production_evaluation_sha256"]),
            semantic_verification_result_sha256s=result_hashes,
            hints=hints,
        )
        return cls.model_validate(values)


def canonical_insession_task_graph_revision_lineage_sha256(
    *,
    task_id: str,
    expected_base_graph_revision: int,
    task_graph_proposal_sha256: str,
    production_evaluation_sha256: str,
    semantic_verification_result_sha256s: tuple[str, ...],
    hints: tuple[InSessionTaskNodeLineageHint, ...],
) -> str:
    return _sha256(
        {
            "schema_version": "insession-task-graph-lineage-hints-v1",
            "task_id": task_id,
            "expected_base_graph_revision": expected_base_graph_revision,
            "task_graph_proposal_sha256": task_graph_proposal_sha256,
            "production_evaluation_sha256": production_evaluation_sha256,
            "semantic_verification_result_sha256s": list(
                semantic_verification_result_sha256s
            ),
            "hints": [item.model_dump(mode="json") for item in hints],
        }
    )


class TaskGraphRevisionTransitionCode(StrEnum):
    BASE_GRAPH_REQUIRED = "base_graph_required"
    BASE_REVISION_MISMATCH = "base_revision_mismatch"
    BASE_GRAPH_CORRUPT = "base_graph_corrupt"
    PROPOSAL_GRAPH_INVALID = "proposal_graph_invalid"
    LINEAGE_AUTHORITY_MISMATCH = "lineage_authority_mismatch"
    LINEAGE_COVERAGE_MISMATCH = "lineage_coverage_mismatch"
    UNKNOWN_BASE_NODE = "unknown_base_node"
    ROOT_IDENTITY_MISMATCH = "root_identity_mismatch"
    NEW_NODE_ALLOCATION_MISMATCH = "new_node_allocation_mismatch"
    NODE_ID_COLLISION = "node_id_collision"
    REUSE_DEFINITION_DRIFT = "reuse_definition_drift"
    REVISION_WITHOUT_CHANGE = "revision_without_change"


class TaskGraphRevisionTransitionError(ValueError):
    """提议的谱系无法确定性地形成下一个快照。"""

    def __init__(self, code: TaskGraphRevisionTransitionCode, message: str) -> None:
        self.code = code
        super().__init__(message)


class InSessionTaskNodeRevisionDefinition(_Contract):
    """由一个转换项封存的精确目标节点定义。"""

    node_kind: InSessionTaskNodeKind
    parent_node_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )
    title: str = Field(min_length=1, max_length=240)
    objective: str = Field(min_length=1, max_length=2_000)
    source_anchor_ids: tuple[str, ...] = Field(min_length=1, max_length=32)
    acceptance_criteria: tuple[InSessionTaskAcceptanceProposal, ...] = Field(
        min_length=1,
        max_length=64,
    )
    constraints: tuple[str, ...] = Field(default=(), max_length=32)

    @field_validator("source_anchor_ids")
    @classmethod
    def _require_unique_source_anchors(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("transition definition source anchors must be unique")
        return values

    @field_validator("acceptance_criteria")
    @classmethod
    def _require_unique_acceptances(
        cls,
        values: tuple[InSessionTaskAcceptanceProposal, ...],
    ) -> tuple[InSessionTaskAcceptanceProposal, ...]:
        ids = tuple(item.acceptance_id for item in values)
        if len(ids) != len(set(ids)):
            raise ValueError("transition definition Acceptance IDs must be unique")
        return values

    @model_validator(mode="after")
    def _validate_parent_shape(self) -> 'InSessionTaskNodeRevisionDefinition':
        if self.node_kind is InSessionTaskNodeKind.ROOT:
            if self.parent_node_id is not None:
                raise ValueError("transition root definition cannot declare a parent")
        elif self.parent_node_id is None:
            raise ValueError("transition subtask definition requires a parent")
        return self


class InSessionTaskNodeRevisionTransition(_Contract):
    node_key: str = Field(pattern=_LOCAL_KEY_PATTERN)
    disposition: TaskGraphNodeLineageDisposition
    target_node_id: str = Field(min_length=1, max_length=128)
    target_node_revision: int = Field(ge=1)
    target_parent_node_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )
    base_node_id: str | None = Field(default=None, min_length=1, max_length=128)
    base_node_revision: int | None = Field(default=None, ge=1)
    definition: InSessionTaskNodeRevisionDefinition
    definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    definition_carry_candidate: bool

    @model_validator(mode="after")
    def _validate_lineage_shape(self) -> 'InSessionTaskNodeRevisionTransition':
        if self.definition.parent_node_id != self.target_parent_node_id:
            raise ValueError("transition definition parent does not match target parent")
        if self.definition_sha256 != _sha256(self.definition.model_dump(mode="json")):
            raise ValueError("transition definition hash does not match its payload")
        if self.disposition is TaskGraphNodeLineageDisposition.NEW:
            if self.base_node_id is not None or self.base_node_revision is not None:
                raise ValueError("new transition cannot carry base lineage")
            if self.target_node_revision != 1:
                raise ValueError("new transition must start at node revision one")
        else:
            if self.base_node_id is None or self.base_node_revision is None:
                raise ValueError("reuse/revise transition requires exact base lineage")
            if self.target_node_id != self.base_node_id:
                raise ValueError("reuse/revise transition must preserve node identity")
            expected_revision = (
                self.base_node_revision
                if self.disposition is TaskGraphNodeLineageDisposition.REUSE
                else self.base_node_revision + 1
            )
            if self.target_node_revision != expected_revision:
                raise ValueError("transition target node revision is inconsistent")
        if self.definition_carry_candidate and (
            self.disposition is not TaskGraphNodeLineageDisposition.REUSE
        ):
            raise ValueError("only exact reuse may become a carry candidate")
        return self


class InSessionTaskGraphRevisionTransition(_Contract):
    schema_version: Literal["insession-task-graph-revision-transition-v1"] = (
        "insession-task-graph-revision-transition-v1"
    )
    task_id: str = Field(min_length=1, max_length=128)
    base_graph_revision: int = Field(ge=1)
    target_graph_revision: int = Field(ge=2)
    root_node_id: str = Field(min_length=1, max_length=128)
    nodes: tuple[InSessionTaskNodeRevisionTransition, ...] = Field(
        min_length=1,
        max_length=512,
    )
    removed_base_node_ids: tuple[str, ...] = Field(max_length=512)
    transition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_revision_and_nodes(self) -> 'InSessionTaskGraphRevisionTransition':
        if self.target_graph_revision != self.base_graph_revision + 1:
            raise ValueError("target graph revision must be exactly base + 1")
        node_keys = [item.node_key for item in self.nodes]
        if len(node_keys) != len(set(node_keys)):
            raise ValueError("transition proposal-local node keys must be unique")
        target_ids = [item.target_node_id for item in self.nodes]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("transition target node identities must be unique")
        if self.root_node_id not in target_ids:
            raise ValueError("transition root must be present in target nodes")
        if len(self.removed_base_node_ids) != len(set(self.removed_base_node_ids)):
            raise ValueError("removed base node identities must be unique")
        if set(self.removed_base_node_ids) & set(target_ids):
            raise ValueError("removed base nodes cannot remain in the target graph")
        by_id = {item.target_node_id: item for item in self.nodes}
        root = by_id[self.root_node_id]
        if (
            root.definition.node_kind is not InSessionTaskNodeKind.ROOT
            or root.target_parent_node_id is not None
        ):
            raise ValueError("transition root identity must bind the only root definition")
        for item in self.nodes:
            if item.target_node_id == self.root_node_id:
                continue
            if (
                item.definition.node_kind is not InSessionTaskNodeKind.SUBTASK
                or item.target_parent_node_id not in by_id
            ):
                raise ValueError("every transition subtask requires a target parent")
            seen: set[str] = set()
            cursor = item
            while cursor.target_parent_node_id is not None:
                if cursor.target_node_id in seen:
                    raise ValueError("transition target graph contains a parent cycle")
                seen.add(cursor.target_node_id)
                cursor = by_id[cursor.target_parent_node_id]
            if cursor.target_node_id != self.root_node_id:
                raise ValueError("transition target node is not reachable from its root")
        expected_hash = _sha256(
            {
                "contract_version": self.schema_version,
                "task_id": self.task_id,
                "base_graph_revision": self.base_graph_revision,
                "target_graph_revision": self.target_graph_revision,
                "root_node_id": self.root_node_id,
                "nodes": [item.model_dump(mode="json") for item in self.nodes],
                "removed_base_node_ids": list(self.removed_base_node_ids),
            }
        )
        if self.transition_sha256 != expected_hash:
            raise ValueError("transition hash does not match its payload")
        return self


def plan_insession_task_graph_revision_transition(
    base: InSessionTaskDetails,
    proposal: InSessionTaskGraphRevisionProposal,
    *,
    lineage: InSessionTaskGraphRevisionLineageHints,
    base_node_ids_by_alias: Mapping[str, str],
    allocated_node_ids_by_key: Mapping[str, str],
    force_reexecution_base_node_ids: frozenset[str] = frozenset(),
) -> InSessionTaskGraphRevisionTransition:
    """为一个完整快照规划精确的节点标识及修订连续性。

    两个标识映射均由 Host 持有。模型只能看到基础别名和提案局部键，绝不选择持久节点 ID。
    调用此纯规划器前，Store 必须验证 ``lineage`` 封存的评估与语义结果哈希。
    已完成且精确复用的节点只会标记为沿用“候选项”；Store 仍须证明其 Delivery、
    依赖闭包、来源、能力及新鲜度权威信息。Host 持有的修订原因还可在保留节点精确定义的
    同时使其已完成执行失效。此类节点会推进到新的节点修订并恢复为 proposed；
    模型的 ``reuse`` 提示绝不能覆盖这种更强的执行权威信息。
    """

    if base.current_graph_revision is None:
        raise TaskGraphRevisionTransitionError(
            TaskGraphRevisionTransitionCode.BASE_GRAPH_REQUIRED,
            "TaskGraph revision transition requires a positive base revision",
        )
    if base.current_graph_revision != lineage.expected_base_graph_revision:
        raise TaskGraphRevisionTransitionError(
            TaskGraphRevisionTransitionCode.BASE_REVISION_MISMATCH,
            "lineage hints target another base graph revision",
        )
    if (
        lineage.task_id != base.insession_task_id
        or lineage.task_graph_proposal_sha256
        != _sha256(proposal.model_dump(mode="json"))
    ):
        raise TaskGraphRevisionTransitionError(
            TaskGraphRevisionTransitionCode.LINEAGE_AUTHORITY_MISMATCH,
            "lineage authority does not bind this TaskGraph proposal",
        )

    base_nodes = _parse_base_nodes(base)
    proposal_nodes = _validate_proposal_tree(proposal)
    if (
        not base_node_ids_by_alias
        or any(
            not isinstance(alias, str)
            or not alias
            or not isinstance(node_id, str)
            or not node_id
            for alias, node_id in base_node_ids_by_alias.items()
        )
        or len(set(base_node_ids_by_alias.values()))
        != len(base_node_ids_by_alias)
        or set(base_node_ids_by_alias.values()) != set(base_nodes)
    ):
        raise TaskGraphRevisionTransitionError(
            TaskGraphRevisionTransitionCode.LINEAGE_AUTHORITY_MISMATCH,
            "Host base-node aliases do not exactly cover the frozen base graph",
        )
    if (
        not isinstance(force_reexecution_base_node_ids, frozenset)
        or any(
            not isinstance(node_id, str) or not node_id
            for node_id in force_reexecution_base_node_ids
        )
        or not force_reexecution_base_node_ids.issubset(base_nodes)
    ):
        raise TaskGraphRevisionTransitionError(
            TaskGraphRevisionTransitionCode.LINEAGE_AUTHORITY_MISMATCH,
            "Host forced-reexecution nodes are outside the frozen base graph",
        )
    hints = {item.node_key: item for item in lineage.hints}
    if set(hints) != set(proposal_nodes):
        raise TaskGraphRevisionTransitionError(
            TaskGraphRevisionTransitionCode.LINEAGE_COVERAGE_MISMATCH,
            "lineage hints must cover every proposal node exactly once",
        )

    referenced_base_aliases = {
        hint.base_node_alias
        for hint in hints.values()
        if hint.base_node_alias is not None
    }
    unknown_base_aliases = referenced_base_aliases - set(base_node_ids_by_alias)
    if unknown_base_aliases:
        raise TaskGraphRevisionTransitionError(
            TaskGraphRevisionTransitionCode.UNKNOWN_BASE_NODE,
            "lineage hints reference an unknown base node alias",
        )
    referenced_base_ids = {
        base_node_ids_by_alias[alias] for alias in referenced_base_aliases
    }

    root_key = proposal.root.root_key
    base_root_ids = {
        node_id
        for node_id, node in base_nodes.items()
        if node["node_kind"] == InSessionTaskNodeKind.ROOT.value
        and node["parent_node_id"] is None
    }
    root_hint = hints[root_key]
    if (
        len(base_root_ids) != 1
        or root_hint.disposition is TaskGraphNodeLineageDisposition.NEW
        or (
            root_hint.base_node_alias is None
            or base_node_ids_by_alias[root_hint.base_node_alias] not in base_root_ids
        )
    ):
        raise TaskGraphRevisionTransitionError(
            TaskGraphRevisionTransitionCode.ROOT_IDENTITY_MISMATCH,
            "a revised TaskGraph must preserve the root node identity",
        )

    new_keys = {
        key
        for key, hint in hints.items()
        if hint.disposition is TaskGraphNodeLineageDisposition.NEW
    }
    if set(allocated_node_ids_by_key) != new_keys or any(
        not isinstance(value, str) or not value.strip() or len(value) > 128
        for value in allocated_node_ids_by_key.values()
    ):
        raise TaskGraphRevisionTransitionError(
            TaskGraphRevisionTransitionCode.NEW_NODE_ALLOCATION_MISMATCH,
            "Host node allocations must match exactly the new proposal nodes",
        )

    target_ids_by_key = {
        key: (
            str(base_node_ids_by_alias[hint.base_node_alias])
            if hint.base_node_alias is not None
            else str(allocated_node_ids_by_key[key])
        )
        for key, hint in hints.items()
    }
    target_ids = tuple(target_ids_by_key.values())
    if len(target_ids) != len(set(target_ids)) or set(target_ids) - set(
        allocated_node_ids_by_key.values()
    ) - set(base_nodes):
        raise TaskGraphRevisionTransitionError(
            TaskGraphRevisionTransitionCode.NODE_ID_COLLISION,
            "target node identities collide or are not Host-authorized",
        )
    if set(allocated_node_ids_by_key.values()) & set(base_nodes):
        raise TaskGraphRevisionTransitionError(
            TaskGraphRevisionTransitionCode.NODE_ID_COLLISION,
            "new node identity collides with the base graph",
        )

    transitions: list[InSessionTaskNodeRevisionTransition] = []
    for node in proposal.root.nodes:
        hint = hints[node.node_key]
        parent_id = (
            target_ids_by_key[node.parent_node_key]
            if node.parent_node_key is not None
            else None
        )
        definition = InSessionTaskNodeRevisionDefinition.model_validate(
            _proposal_node_definition(node, parent_node_id=parent_id)
        )
        definition_sha = _sha256(definition.model_dump(mode="json"))
        if hint.disposition is TaskGraphNodeLineageDisposition.NEW:
            target_revision = 1
            base_revision = None
            carry_candidate = False
        else:
            assert hint.base_node_alias is not None
            base_node_id = base_node_ids_by_alias[hint.base_node_alias]
            base_node = base_nodes[base_node_id]
            force_reexecution = (
                base_node_id in force_reexecution_base_node_ids
            )
            base_definition = InSessionTaskNodeRevisionDefinition.model_validate(
                _base_node_definition(base_node)
            )
            base_definition_sha = _sha256(base_definition.model_dump(mode="json"))
            definitions_equal = definition_sha == base_definition_sha
            if (
                hint.disposition is TaskGraphNodeLineageDisposition.REUSE
                and not definitions_equal
            ):
                raise TaskGraphRevisionTransitionError(
                    TaskGraphRevisionTransitionCode.REUSE_DEFINITION_DRIFT,
                    f"reuse lineage changed node definition: {node.node_key}",
                )
            if (
                hint.disposition is TaskGraphNodeLineageDisposition.REVISE
                and definitions_equal
                and not force_reexecution
            ):
                raise TaskGraphRevisionTransitionError(
                    TaskGraphRevisionTransitionCode.REVISION_WITHOUT_CHANGE,
                    f"revise lineage did not change node definition: {node.node_key}",
                )
            base_revision = int(base_node["node_revision"])
            effective_disposition = (
                TaskGraphNodeLineageDisposition.REVISE
                if force_reexecution
                else hint.disposition
            )
            target_revision = (
                base_revision
                if effective_disposition
                is TaskGraphNodeLineageDisposition.REUSE
                else base_revision + 1
            )
            carry_candidate = bool(
                effective_disposition
                is TaskGraphNodeLineageDisposition.REUSE
                and base_node["status"] == "completed"
            )
        transitions.append(
            InSessionTaskNodeRevisionTransition(
                node_key=node.node_key,
                disposition=(
                    hint.disposition
                    if hint.disposition is TaskGraphNodeLineageDisposition.NEW
                    else effective_disposition
                ),
                target_node_id=target_ids_by_key[node.node_key],
                target_node_revision=target_revision,
                target_parent_node_id=parent_id,
                base_node_id=(
                    None
                    if hint.base_node_alias is None
                    else base_node_ids_by_alias[hint.base_node_alias]
                ),
                base_node_revision=base_revision,
                definition=definition,
                definition_sha256=definition_sha,
                definition_carry_candidate=carry_candidate,
            )
        )

    removed = tuple(
        node_id
        for node_id, node in sorted(
            base_nodes.items(),
            key=lambda item: (int(item[1]["ordinal"]), item[0]),
        )
        if node_id not in referenced_base_ids
    )
    transition_payload = {
        "contract_version": "insession-task-graph-revision-transition-v1",
        "task_id": base.insession_task_id,
        "base_graph_revision": base.current_graph_revision,
        "target_graph_revision": base.current_graph_revision + 1,
        "root_node_id": target_ids_by_key[root_key],
        "nodes": [item.model_dump(mode="json") for item in transitions],
        "removed_base_node_ids": list(removed),
    }
    return InSessionTaskGraphRevisionTransition(
        task_id=base.insession_task_id,
        base_graph_revision=base.current_graph_revision,
        target_graph_revision=base.current_graph_revision + 1,
        root_node_id=target_ids_by_key[root_key],
        nodes=tuple(transitions),
        removed_base_node_ids=removed,
        transition_sha256=_sha256(transition_payload),
    )


def _parse_base_nodes(base: InSessionTaskDetails) -> dict[str, dict[str, object]]:
    parsed: dict[str, dict[str, object]] = {}
    try:
        for raw in base.nodes:
            node_id = str(raw["insession_task_node_id"])
            if not node_id or node_id in parsed:
                raise ValueError
            node_revision = int(raw["node_revision"])
            ordinal = int(raw["ordinal"])
            if node_revision < 1 or ordinal < 0:
                raise ValueError
            node_kind = InSessionTaskNodeKind(str(raw["node_kind"]))
            parent = raw.get("parent_insession_task_node_id")
            parent_id = str(parent) if parent is not None else None
            acceptances = tuple(
                InSessionTaskAcceptanceProposal.model_validate(item)
                for item in raw["acceptance_criteria"]
            )
            source_ids = tuple(str(item) for item in raw["source_anchor_ids"])
            constraints = tuple(str(item) for item in raw["constraints"])
            status = str(raw["status"])
            parsed[node_id] = {
                "node_revision": node_revision,
                "node_kind": node_kind.value,
                "ordinal": ordinal,
                "parent_node_id": parent_id,
                "title": str(raw["title"]),
                "objective": str(raw["objective"]),
                "source_anchor_ids": source_ids,
                "acceptance_criteria": acceptances,
                "constraints": constraints,
                "status": status,
            }
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskGraphRevisionTransitionError(
            TaskGraphRevisionTransitionCode.BASE_GRAPH_CORRUPT,
            "base TaskGraph projection is incomplete or corrupt",
        ) from exc
    if not parsed:
        raise TaskGraphRevisionTransitionError(
            TaskGraphRevisionTransitionCode.BASE_GRAPH_CORRUPT,
            "positive base TaskGraph has no nodes",
        )
    if any(
        node["parent_node_id"] is not None
        and node["parent_node_id"] not in parsed
        for node in parsed.values()
    ):
        raise TaskGraphRevisionTransitionError(
            TaskGraphRevisionTransitionCode.BASE_GRAPH_CORRUPT,
            "base TaskGraph contains an unknown parent identity",
        )
    ordinals = tuple(int(node["ordinal"]) for node in parsed.values())
    if len(ordinals) != len(set(ordinals)) or set(ordinals) != set(range(len(parsed))):
        raise TaskGraphRevisionTransitionError(
            TaskGraphRevisionTransitionCode.BASE_GRAPH_CORRUPT,
            "base TaskGraph ordinals are not a contiguous unique sequence",
        )
    allowed_statuses = {
        "proposed",
        "active",
        "awaiting_user",
        "waiting_external",
        "interrupted",
        "blocked",
        "cancelled",
        "completed",
    }
    if any(node["status"] not in allowed_statuses for node in parsed.values()):
        raise TaskGraphRevisionTransitionError(
            TaskGraphRevisionTransitionCode.BASE_GRAPH_CORRUPT,
            "base TaskGraph contains an invalid node status",
        )
    roots = [
        node_id
        for node_id, node in parsed.items()
        if node["node_kind"] == InSessionTaskNodeKind.ROOT.value
        and node["parent_node_id"] is None
    ]
    if len(roots) != 1 or any(
        (
            node["node_kind"] == InSessionTaskNodeKind.ROOT.value
            or node["parent_node_id"] is None
        )
        for node_id, node in parsed.items()
        if node_id not in roots
    ):
        raise TaskGraphRevisionTransitionError(
            TaskGraphRevisionTransitionCode.BASE_GRAPH_CORRUPT,
            "base TaskGraph must contain exactly one root and parented subtasks",
        )
    root_id = roots[0]
    for node_id in parsed:
        seen: set[str] = set()
        cursor_id: str | None = node_id
        while cursor_id is not None:
            if cursor_id in seen:
                raise TaskGraphRevisionTransitionError(
                    TaskGraphRevisionTransitionCode.BASE_GRAPH_CORRUPT,
                    "base TaskGraph contains a parent cycle",
                )
            seen.add(cursor_id)
            parent = parsed[cursor_id]["parent_node_id"]
            cursor_id = str(parent) if parent is not None else None
        if root_id not in seen:
            raise TaskGraphRevisionTransitionError(
                TaskGraphRevisionTransitionCode.BASE_GRAPH_CORRUPT,
                "base TaskGraph contains a node outside the root tree",
            )
    return parsed


def _validate_proposal_tree(
    proposal: InSessionTaskGraphRevisionProposal,
) -> dict[str, InSessionTaskNodeProposal]:
    nodes: dict[str, InSessionTaskNodeProposal] = {}
    for node in proposal.root.nodes:
        if node.node_key in nodes:
            raise TaskGraphRevisionTransitionError(
                TaskGraphRevisionTransitionCode.PROPOSAL_GRAPH_INVALID,
                "proposal contains duplicate node keys",
            )
        nodes[node.node_key] = node
    root = nodes.get(proposal.root.root_key)
    if (
        root is None
        or root.node_kind is not InSessionTaskNodeKind.ROOT
        or root.parent_node_key is not None
    ):
        raise TaskGraphRevisionTransitionError(
            TaskGraphRevisionTransitionCode.PROPOSAL_GRAPH_INVALID,
            "proposal root is missing or invalid",
        )
    for node in nodes.values():
        if node is root:
            continue
        if node.parent_node_key not in nodes:
            raise TaskGraphRevisionTransitionError(
                TaskGraphRevisionTransitionCode.PROPOSAL_GRAPH_INVALID,
                "proposal contains an unknown parent key",
            )
        seen: set[str] = set()
        cursor: InSessionTaskNodeProposal | None = node
        while cursor is not None:
            if cursor.node_key in seen:
                raise TaskGraphRevisionTransitionError(
                    TaskGraphRevisionTransitionCode.PROPOSAL_GRAPH_INVALID,
                    "proposal contains a parent cycle",
                )
            seen.add(cursor.node_key)
            cursor = (
                nodes[cursor.parent_node_key]
                if cursor.parent_node_key is not None
                else None
            )
    return nodes


def _proposal_node_definition(
    node: InSessionTaskNodeProposal,
    *,
    parent_node_id: str | None,
) -> dict[str, object]:
    return {
        "node_kind": node.node_kind.value,
        "parent_node_id": parent_node_id,
        "title": node.title,
        "objective": node.objective,
        "source_anchor_ids": list(node.source_anchor_ids),
        "acceptance_criteria": [
            item.model_dump(mode="json") for item in node.acceptance_criteria
        ],
        "constraints": list(node.constraints),
    }


def _base_node_definition(node: Mapping[str, object]) -> dict[str, object]:
    acceptances = node["acceptance_criteria"]
    assert isinstance(acceptances, tuple)
    return {
        "node_kind": node["node_kind"],
        "parent_node_id": node["parent_node_id"],
        "title": node["title"],
        "objective": node["objective"],
        "source_anchor_ids": list(node["source_anchor_ids"]),
        "acceptance_criteria": [
            item.model_dump(mode="json") for item in acceptances
        ],
        "constraints": list(node["constraints"]),
    }


def _sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = [
    'InSessionTaskGraphRevisionLineageHints',
    'InSessionTaskGraphRevisionTransition',
    'InSessionTaskNodeLineageHint',
    'InSessionTaskNodeRevisionDefinition',
    'InSessionTaskNodeRevisionTransition',
    'TaskGraphNodeLineageDisposition',
    'TaskGraphRevisionTransitionCode',
    "TaskGraphRevisionTransitionError",
    'canonical_insession_task_graph_revision_lineage_sha256',
    "plan_insession_task_graph_revision_transition",
]
