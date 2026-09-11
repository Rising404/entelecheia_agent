"""针对单个 TaskGraph 提案的自适应确定性生产检查。

该评估器既不奖励也不要求任意的节点数量、深度或分支形态。它是在语义审查器和
存储提交事务运行前，针对宿主能够证明的事实执行的纯机械门禁：

* 规范的 TaskGraph 结构与来源权威；
* 必需目标与证据来源的覆盖情况；
* 由宿主解析的能力及效应可满足性；
* 基线、上下文和目录的新鲜度，以及类型化缺口闭合情况；
* 完全相同的规范化契约副本。

结果是确定、不可变的，并携带规范 SHA-256 绑定。它不会修改权威、调用模型，
也不会声称机械上有效的分解在语义上必然正确。
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import (
    InSessionTaskGraphRevisionProposal,
    InSessionTaskGraphRevisionValidationContext,
    InSessionTaskGraphValidationCode,
    InSessionTaskNodeProposal,
)
from .validation import validate_insession_task_graph_revision


_LOCAL_KEY_PATTERN = r"^[a-z][a-z0-9_-]{0,63}$"
_RUNTIME_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskGraphProductionFailureCode(StrEnum):
    """自适应生产门禁发出的稳定硬失败。"""

    CANONICAL_VALIDATION_REJECTED = "canonical_validation_rejected"
    NODE_LIMIT_EXCEEDED = "node_limit_exceeded"
    DEPTH_LIMIT_EXCEEDED = "depth_limit_exceeded"
    DUPLICATE_NODE_KEY = "duplicate_node_key"
    ROOT_NODE_MISSING = "root_node_missing"
    ROOT_NODE_INVALID = "root_node_invalid"
    INVALID_PARENT_REFERENCE = "invalid_parent_reference"
    INVALID_PARENT_KIND = "invalid_parent_kind"
    CYCLE_DETECTED = "cycle_detected"
    ORPHAN_NODE = "orphan_node"
    DUPLICATE_ACCEPTANCE_ID = "duplicate_acceptance_id"
    DUPLICATE_ACCEPTANCE_CRITERION = "duplicate_acceptance_criterion"
    MISSING_AUTHORIZATION_ANCHOR = "missing_authorization_anchor"
    UNKNOWN_AUTHORIZATION_ANCHOR = "unknown_authorization_anchor"
    UNAUTHORIZED_AUTHORIZATION_SOURCE = "unauthorized_authorization_source"
    NODE_WITHOUT_AUTHORIZATION = "node_without_authorization"
    ACCEPTANCE_WITHOUT_AUTHORIZATION = "acceptance_without_authorization"
    UNKNOWN_SOURCE_REFERENCE = "unknown_source_reference"
    MISSING_SOURCE_REFERENCE = "missing_source_reference"
    UNSOURCED_CONSTRAINT = "unsourced_constraint"
    SOURCE_TURN_MISMATCH = "source_turn_mismatch"
    REQUIRED_GOAL_UNCOVERED = "required_goal_uncovered"
    REQUIRED_SOURCE_UNCOVERED = "required_source_uncovered"
    NODE_EXECUTION_REQUIREMENT_MISSING = (
        "node_execution_requirement_missing"
    )
    UNKNOWN_EXECUTION_REQUIREMENT_NODE = (
        "unknown_execution_requirement_node"
    )
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    EFFECT_UNSATISFIABLE = "effect_unsatisfiable"
    BASE_AUTHORITY_STALE = "base_authority_stale"
    CURRENT_AUTHORITY_STALE = "current_authority_stale"
    SOURCE_MANIFEST_STALE = "source_manifest_stale"
    CAPABILITY_CATALOG_STALE = "capability_catalog_stale"
    BLOCKING_GAP = "blocking_gap"
    UNKNOWN_GAP_ANCHOR = "unknown_gap_anchor"
    NON_BLOCKING_GAP_UNMAPPED = "non_blocking_gap_unmapped"
    EXACT_CONTRACT_DUPLICATE = "exact_contract_duplicate"


class TaskGraphAcceptanceRef(_Contract):
    node_key: str = Field(pattern=_LOCAL_KEY_PATTERN)
    acceptance_id: str = Field(pattern=_LOCAL_KEY_PATTERN)


class TaskGraphPlanningGap(_Contract):
    """一项由宿主推导的缺口及其拟议的形式图覆盖。"""

    gap_id: str = Field(pattern=_LOCAL_KEY_PATTERN)
    blocking: bool
    affected_required_anchor_ids: tuple[str, ...] = Field(
        default=(), max_length=256
    )
    mapped_node_keys: tuple[str, ...] = Field(default=(), max_length=128)
    mapped_acceptances: tuple[TaskGraphAcceptanceRef, ...] = Field(
        default=(), max_length=256
    )

    @field_validator(
        "affected_required_anchor_ids",
        "mapped_node_keys",
    )
    @classmethod
    def _require_unique_strings(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("gap string references must be unique")
        return values

    @field_validator("mapped_acceptances")
    @classmethod
    def _require_unique_acceptance_refs(
        cls,
        values: tuple[TaskGraphAcceptanceRef, ...],
    ) -> tuple[TaskGraphAcceptanceRef, ...]:
        keys = [(item.node_key, item.acceptance_id) for item in values]
        if len(keys) != len(set(keys)):
            raise ValueError("gap acceptance references must be unique")
        return values


class TaskGraphNodeExecutionRequirement(_Contract):
    """宿主为一个拟议本地节点键解析出的执行需求。"""

    node_key: str = Field(pattern=_LOCAL_KEY_PATTERN)
    required_capability_ids: tuple[str, ...] = Field(default=(), max_length=64)
    required_effect_ids: tuple[str, ...] = Field(default=(), max_length=64)

    @field_validator("required_capability_ids", "required_effect_ids")
    @classmethod
    def _require_unique_runtime_ids(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("execution requirement IDs must be unique")
        if values != tuple(sorted(values)):
            raise ValueError("execution requirement IDs must use canonical order")
        for value in values:
            if re.fullmatch(_RUNTIME_ID_PATTERN, value) is None:
                raise ValueError("execution requirement IDs must be canonical runtime IDs")
        return values


class TaskGraphProductionCapabilityContext(_Contract):
    """用于可行性检查的冻结能力与效应全集。"""

    available_capability_ids: tuple[str, ...] = Field(default=(), max_length=512)
    satisfiable_effect_ids: tuple[str, ...] = Field(default=(), max_length=512)
    node_requirements: tuple[TaskGraphNodeExecutionRequirement, ...] = Field(
        default=(), max_length=512
    )

    @field_validator("available_capability_ids", "satisfiable_effect_ids")
    @classmethod
    def _validate_runtime_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("capability/effect IDs must be unique")
        if values != tuple(sorted(values)):
            raise ValueError("capability/effect IDs must use canonical order")
        for value in values:
            if re.fullmatch(_RUNTIME_ID_PATTERN, value) is None:
                raise ValueError("capability/effect IDs must be canonical runtime IDs")
        return values

    @field_validator("node_requirements")
    @classmethod
    def _require_unique_node_requirements(
        cls,
        values: tuple[TaskGraphNodeExecutionRequirement, ...],
    ) -> tuple[TaskGraphNodeExecutionRequirement, ...]:
        node_keys = [item.node_key for item in values]
        if len(node_keys) != len(set(node_keys)):
            raise ValueError("node execution requirements must be unique")
        return values


class TaskGraphProductionFreshness(_Contract):
    """预期权威绑定与最新观测绑定的对照。"""

    observed_current_graph_revision: int | None = Field(default=None, ge=1)
    expected_authority_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    observed_authority_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_source_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    observed_source_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    expected_capability_catalog_sha256: str = Field(pattern=_SHA256_PATTERN)
    observed_capability_catalog_sha256: str = Field(pattern=_SHA256_PATTERN)


class TaskGraphProductionEvaluationContext(_Contract):
    """一次纯生产评估所需的完整可信输入。"""

    validation_context: InSessionTaskGraphRevisionValidationContext
    freshness: TaskGraphProductionFreshness
    capabilities: TaskGraphProductionCapabilityContext
    gaps: tuple[TaskGraphPlanningGap, ...] = Field(default=(), max_length=512)

    @field_validator("gaps")
    @classmethod
    def _require_unique_gap_ids(
        cls,
        values: tuple[TaskGraphPlanningGap, ...],
    ) -> tuple[TaskGraphPlanningGap, ...]:
        gap_ids = [item.gap_id for item in values]
        if len(gap_ids) != len(set(gap_ids)):
            raise ValueError("planning gap IDs must be unique")
        return values


class TaskGraphProductionFinding(_Contract):
    """一项稳定且定位范围精确的机械性失败。"""

    code: TaskGraphProductionFailureCode
    node_key: str | None = Field(default=None, pattern=_LOCAL_KEY_PATTERN)
    related_node_key: str | None = Field(
        default=None, pattern=_LOCAL_KEY_PATTERN
    )
    acceptance_id: str | None = Field(
        default=None, pattern=_LOCAL_KEY_PATTERN
    )
    anchor_id: str | None = Field(default=None, min_length=1, max_length=200)
    capability_id: str | None = Field(
        default=None, pattern=_RUNTIME_ID_PATTERN
    )
    effect_id: str | None = Field(default=None, pattern=_RUNTIME_ID_PATTERN)
    gap_id: str | None = Field(default=None, pattern=_LOCAL_KEY_PATTERN)
    canonical_code: InSessionTaskGraphValidationCode | None = None


class TaskGraphProductionTelemetry(_Contract):
    """拓扑与覆盖观测；不设置人为复杂度下限。"""

    total_node_count: int = Field(ge=0)
    unique_node_count: int = Field(ge=0)
    reachable_node_count: int = Field(ge=0)
    max_depth: int = Field(ge=0)
    root_child_count: int = Field(ge=0)
    internal_node_count: int = Field(ge=0)
    unary_internal_node_count: int = Field(ge=0)
    branching_node_count: int = Field(ge=0)
    leaf_node_count: int = Field(ge=0)
    max_branching_factor: int = Field(ge=0)
    acceptance_count: int = Field(ge=0)
    duplicate_acceptance_criterion_count: int = Field(ge=0)
    source_reference_count: int = Field(ge=0)
    unknown_source_reference_count: int = Field(ge=0)
    node_authorization_coverage_count: int = Field(ge=0)
    acceptance_authorization_coverage_count: int = Field(ge=0)
    required_goal_count: int = Field(ge=0)
    required_goal_node_coverage_count: int = Field(ge=0)
    required_goal_acceptance_coverage_count: int = Field(ge=0)
    required_source_count: int = Field(ge=0)
    required_source_node_coverage_count: int = Field(ge=0)
    required_source_acceptance_coverage_count: int = Field(ge=0)
    execution_requirement_count: int = Field(ge=0)
    missing_execution_requirement_count: int = Field(ge=0)
    unknown_execution_requirement_count: int = Field(ge=0)
    capability_requirement_count: int = Field(ge=0)
    unavailable_capability_count: int = Field(ge=0)
    effect_requirement_count: int = Field(ge=0)
    unsatisfiable_effect_count: int = Field(ge=0)
    blocking_gap_count: int = Field(ge=0)
    non_blocking_gap_count: int = Field(ge=0)
    unmapped_non_blocking_gap_count: int = Field(ge=0)
    exact_duplicate_node_count: int = Field(ge=0)
    canonical_error_count: int = Field(ge=0)


class TaskGraphProductionEvaluation(_Contract):
    schema_version: Literal["task-graph-production-eval-v2"] = (
        "task-graph-production-eval-v2"
    )
    profile_id: Literal["task-graph-production-adaptive-v2"] = (
        "task-graph-production-adaptive-v2"
    )
    passed: bool
    telemetry: TaskGraphProductionTelemetry
    failure_codes: tuple[TaskGraphProductionFailureCode, ...] = ()
    findings: tuple[TaskGraphProductionFinding, ...] = ()
    canonical_validation_codes: tuple[InSessionTaskGraphValidationCode, ...] = ()
    semantic_review_required: Literal[True] = True
    proposal_sha256: str = Field(pattern=_SHA256_PATTERN)
    context_sha256: str = Field(pattern=_SHA256_PATTERN)
    evaluation_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_result_binding(self) -> 'TaskGraphProductionEvaluation':
        if self.passed is bool(self.failure_codes):
            raise ValueError("passed must be true exactly when failure_codes is empty")
        if len(self.failure_codes) != len(set(self.failure_codes)):
            raise ValueError("failure_codes must be unique")
        if tuple(sorted(self.failure_codes, key=str)) != self.failure_codes:
            raise ValueError("failure_codes must use stable ordering")
        if len(self.canonical_validation_codes) != len(
            set(self.canonical_validation_codes)
        ):
            raise ValueError("canonical_validation_codes must be unique")
        if tuple(
            sorted(self.canonical_validation_codes, key=str)
        ) != self.canonical_validation_codes:
            raise ValueError(
                "canonical_validation_codes must use stable ordering"
            )
        finding_keys = tuple(_finding_from_model(item) for item in self.findings)
        if len(finding_keys) != len(set(finding_keys)):
            raise ValueError("findings must be unique")
        if tuple(sorted(finding_keys, key=_finding_sort_key)) != finding_keys:
            raise ValueError("findings must use stable ordering")
        expected = _canonical_sha256(
            self.model_dump(mode="json", exclude={"evaluation_sha256"})
        )
        if self.evaluation_sha256 != expected:
            raise ValueError("evaluation_sha256 does not bind the result payload")
        return self


_CANONICAL_FAILURE_MAP: dict[
    InSessionTaskGraphValidationCode,
    TaskGraphProductionFailureCode,
] = {
    InSessionTaskGraphValidationCode.NODE_LIMIT_EXCEEDED: (
        TaskGraphProductionFailureCode.NODE_LIMIT_EXCEEDED
    ),
    InSessionTaskGraphValidationCode.DEPTH_LIMIT_EXCEEDED: (
        TaskGraphProductionFailureCode.DEPTH_LIMIT_EXCEEDED
    ),
    InSessionTaskGraphValidationCode.DUPLICATE_NODE_KEY: (
        TaskGraphProductionFailureCode.DUPLICATE_NODE_KEY
    ),
    InSessionTaskGraphValidationCode.ROOT_NODE_MISSING: (
        TaskGraphProductionFailureCode.ROOT_NODE_MISSING
    ),
    InSessionTaskGraphValidationCode.ROOT_NODE_INVALID: (
        TaskGraphProductionFailureCode.ROOT_NODE_INVALID
    ),
    InSessionTaskGraphValidationCode.INVALID_PARENT_REFERENCE: (
        TaskGraphProductionFailureCode.INVALID_PARENT_REFERENCE
    ),
    InSessionTaskGraphValidationCode.INVALID_PARENT_KIND: (
        TaskGraphProductionFailureCode.INVALID_PARENT_KIND
    ),
    InSessionTaskGraphValidationCode.CYCLE_DETECTED: (
        TaskGraphProductionFailureCode.CYCLE_DETECTED
    ),
    InSessionTaskGraphValidationCode.ORPHAN_NODE: (
        TaskGraphProductionFailureCode.ORPHAN_NODE
    ),
    InSessionTaskGraphValidationCode.DUPLICATE_ACCEPTANCE_ID: (
        TaskGraphProductionFailureCode.DUPLICATE_ACCEPTANCE_ID
    ),
    InSessionTaskGraphValidationCode.MISSING_AUTHORIZATION_ANCHOR: (
        TaskGraphProductionFailureCode.MISSING_AUTHORIZATION_ANCHOR
    ),
    InSessionTaskGraphValidationCode.UNKNOWN_AUTHORIZATION_ANCHOR: (
        TaskGraphProductionFailureCode.UNKNOWN_AUTHORIZATION_ANCHOR
    ),
    InSessionTaskGraphValidationCode.UNAUTHORIZED_AUTHORIZATION_SOURCE: (
        TaskGraphProductionFailureCode.UNAUTHORIZED_AUTHORIZATION_SOURCE
    ),
    InSessionTaskGraphValidationCode.NODE_WITHOUT_AUTHORIZED_SOURCE: (
        TaskGraphProductionFailureCode.NODE_WITHOUT_AUTHORIZATION
    ),
    InSessionTaskGraphValidationCode.ACCEPTANCE_WITHOUT_AUTHORIZED_SOURCE: (
        TaskGraphProductionFailureCode.ACCEPTANCE_WITHOUT_AUTHORIZATION
    ),
    InSessionTaskGraphValidationCode.UNSOURCED_CONSTRAINT: (
        TaskGraphProductionFailureCode.UNSOURCED_CONSTRAINT
    ),
    InSessionTaskGraphValidationCode.UNKNOWN_SOURCE_ANCHOR: (
        TaskGraphProductionFailureCode.UNKNOWN_SOURCE_REFERENCE
    ),
    InSessionTaskGraphValidationCode.MISSING_SOURCE_ANCHOR: (
        TaskGraphProductionFailureCode.MISSING_SOURCE_REFERENCE
    ),
    InSessionTaskGraphValidationCode.SOURCE_TURN_MISMATCH: (
        TaskGraphProductionFailureCode.SOURCE_TURN_MISMATCH
    ),
}


def evaluate_task_graph_production(
    proposal: InSessionTaskGraphRevisionProposal,
    *,
    context: TaskGraphProductionEvaluationContext,
) -> TaskGraphProductionEvaluation:
    """评估一个不可信的完整快照提案，且不填充图形结构。"""

    canonical = validate_insession_task_graph_revision(
        proposal,
        context=context.validation_context,
    )
    canonical_codes = tuple(sorted(canonical.error_codes, key=str))
    failures: set[TaskGraphProductionFailureCode] = set()
    raw_findings: set[tuple[object, ...]] = set()

    for canonical_code in canonical_codes:
        failure = _CANONICAL_FAILURE_MAP.get(
            canonical_code,
            TaskGraphProductionFailureCode.CANONICAL_VALIDATION_REJECTED,
        )
        failures.add(failure)
        raw_findings.add(_finding_key(failure, canonical_code=canonical_code))

    nodes_by_key: dict[str, InSessionTaskNodeProposal] = {}
    ordered_unique_nodes: list[InSessionTaskNodeProposal] = []
    for node in proposal.root.nodes:
        if node.node_key in nodes_by_key:
            failures.add(TaskGraphProductionFailureCode.DUPLICATE_NODE_KEY)
            raw_findings.add(
                _finding_key(
                    TaskGraphProductionFailureCode.DUPLICATE_NODE_KEY,
                    node_key=node.node_key,
                )
            )
            continue
        nodes_by_key[node.node_key] = node
        ordered_unique_nodes.append(node)

    reachable, depth_by_key, children = _tree_projection(
        root_key=proposal.root.root_key,
        nodes_by_key=nodes_by_key,
    )
    reachable_child_counts = {
        node_key: sum(child in reachable for child in children.get(node_key, ()))
        for node_key in reachable
    }

    available_anchor_ids = {
        anchor.anchor_id for anchor in context.validation_context.source_anchors
    }
    authorization_anchor_ids = set(
        context.validation_context.authorization_anchor_ids
    )
    required_anchor_ids = set(context.validation_context.required_anchor_ids)
    required_goal_ids = required_anchor_ids & authorization_anchor_ids
    required_source_ids = required_anchor_ids - authorization_anchor_ids

    node_source_refs = {
        anchor_id
        for node in ordered_unique_nodes
        for anchor_id in node.source_anchor_ids
    }
    acceptance_source_refs = {
        anchor_id
        for node in ordered_unique_nodes
        for acceptance in node.acceptance_criteria
        for anchor_id in acceptance.source_anchor_ids
    }
    all_source_ref_list = [
        anchor_id
        for node in ordered_unique_nodes
        for anchor_id in node.source_anchor_ids
    ] + [
        anchor_id
        for node in ordered_unique_nodes
        for acceptance in node.acceptance_criteria
        for anchor_id in acceptance.source_anchor_ids
    ]
    unknown_source_references = [
        anchor_id
        for anchor_id in all_source_ref_list
        if anchor_id not in available_anchor_ids
    ]

    node_authorization_coverage_count = 0
    acceptance_authorization_coverage_count = 0
    acceptance_count = 0
    duplicate_acceptance_criterion_count = 0
    acceptance_lookup: dict[tuple[str, str], frozenset[str]] = {}
    for node in ordered_unique_nodes:
        if set(node.source_anchor_ids) & authorization_anchor_ids:
            node_authorization_coverage_count += 1
        normalized_criteria = Counter(
            _normalize_text(item.criterion) for item in node.acceptance_criteria
        )
        duplicate_acceptance_criterion_count += sum(
            count - 1 for count in normalized_criteria.values() if count > 1
        )
        if any(count > 1 for count in normalized_criteria.values()):
            failures.add(
                TaskGraphProductionFailureCode.DUPLICATE_ACCEPTANCE_CRITERION
            )
            for _criterion, count in normalized_criteria.items():
                if count > 1:
                    raw_findings.add(
                        _finding_key(
                            TaskGraphProductionFailureCode.DUPLICATE_ACCEPTANCE_CRITERION,
                            node_key=node.node_key,
                        )
                    )
        for acceptance in node.acceptance_criteria:
            acceptance_count += 1
            if set(acceptance.source_anchor_ids) & authorization_anchor_ids:
                acceptance_authorization_coverage_count += 1
            acceptance_lookup[(node.node_key, acceptance.acceptance_id)] = (
                frozenset(acceptance.source_anchor_ids)
            )

    for anchor_id in sorted(required_goal_ids):
        if (
            anchor_id not in node_source_refs
            or anchor_id not in acceptance_source_refs
        ):
            failures.add(TaskGraphProductionFailureCode.REQUIRED_GOAL_UNCOVERED)
            raw_findings.add(
                _finding_key(
                    TaskGraphProductionFailureCode.REQUIRED_GOAL_UNCOVERED,
                    anchor_id=anchor_id,
                )
            )
    for anchor_id in sorted(required_source_ids):
        if (
            anchor_id not in node_source_refs
            or anchor_id not in acceptance_source_refs
        ):
            failures.add(TaskGraphProductionFailureCode.REQUIRED_SOURCE_UNCOVERED)
            raw_findings.add(
                _finding_key(
                    TaskGraphProductionFailureCode.REQUIRED_SOURCE_UNCOVERED,
                    anchor_id=anchor_id,
                )
            )

    duplicate_pairs = _exact_duplicate_nodes(ordered_unique_nodes)
    if duplicate_pairs:
        failures.add(TaskGraphProductionFailureCode.EXACT_CONTRACT_DUPLICATE)
        for node_key, earlier_node_key in duplicate_pairs:
            raw_findings.add(
                _finding_key(
                    TaskGraphProductionFailureCode.EXACT_CONTRACT_DUPLICATE,
                    node_key=node_key,
                    related_node_key=earlier_node_key,
                )
            )

    requirement_by_node = {
        item.node_key: item for item in context.capabilities.node_requirements
    }
    graph_node_keys = set(nodes_by_key)
    missing_requirement_keys = graph_node_keys - set(requirement_by_node)
    unknown_requirement_keys = set(requirement_by_node) - graph_node_keys
    for node_key in sorted(missing_requirement_keys):
        failures.add(
            TaskGraphProductionFailureCode.NODE_EXECUTION_REQUIREMENT_MISSING
        )
        raw_findings.add(
            _finding_key(
                TaskGraphProductionFailureCode.NODE_EXECUTION_REQUIREMENT_MISSING,
                node_key=node_key,
            )
        )
    for node_key in sorted(unknown_requirement_keys):
        failures.add(
            TaskGraphProductionFailureCode.UNKNOWN_EXECUTION_REQUIREMENT_NODE
        )
        raw_findings.add(
            _finding_key(
                TaskGraphProductionFailureCode.UNKNOWN_EXECUTION_REQUIREMENT_NODE,
                node_key=node_key,
            )
        )

    available_capability_ids = set(
        context.capabilities.available_capability_ids
    )
    satisfiable_effect_ids = set(context.capabilities.satisfiable_effect_ids)
    capability_requirement_count = 0
    unavailable_capability_count = 0
    effect_requirement_count = 0
    unsatisfiable_effect_count = 0
    for node_key in sorted(graph_node_keys & set(requirement_by_node)):
        requirement = requirement_by_node[node_key]
        capability_requirement_count += len(requirement.required_capability_ids)
        effect_requirement_count += len(requirement.required_effect_ids)
        for capability_id in sorted(
            set(requirement.required_capability_ids) - available_capability_ids
        ):
            unavailable_capability_count += 1
            failures.add(TaskGraphProductionFailureCode.CAPABILITY_UNAVAILABLE)
            raw_findings.add(
                _finding_key(
                    TaskGraphProductionFailureCode.CAPABILITY_UNAVAILABLE,
                    node_key=node_key,
                    capability_id=capability_id,
                )
            )
        for effect_id in sorted(
            set(requirement.required_effect_ids) - satisfiable_effect_ids
        ):
            unsatisfiable_effect_count += 1
            failures.add(TaskGraphProductionFailureCode.EFFECT_UNSATISFIABLE)
            raw_findings.add(
                _finding_key(
                    TaskGraphProductionFailureCode.EFFECT_UNSATISFIABLE,
                    node_key=node_key,
                    effect_id=effect_id,
                )
            )

    freshness = context.freshness
    if (
        freshness.observed_current_graph_revision
        != context.validation_context.expected_current_graph_revision
    ):
        failures.add(TaskGraphProductionFailureCode.BASE_AUTHORITY_STALE)
        raw_findings.add(
            _finding_key(TaskGraphProductionFailureCode.BASE_AUTHORITY_STALE)
        )
    if (
        freshness.expected_authority_snapshot_sha256
        != freshness.observed_authority_snapshot_sha256
    ):
        failures.add(TaskGraphProductionFailureCode.CURRENT_AUTHORITY_STALE)
        raw_findings.add(
            _finding_key(TaskGraphProductionFailureCode.CURRENT_AUTHORITY_STALE)
        )
    if (
        freshness.expected_source_manifest_sha256
        != freshness.observed_source_manifest_sha256
    ):
        failures.add(TaskGraphProductionFailureCode.SOURCE_MANIFEST_STALE)
        raw_findings.add(
            _finding_key(TaskGraphProductionFailureCode.SOURCE_MANIFEST_STALE)
        )
    if (
        freshness.expected_capability_catalog_sha256
        != freshness.observed_capability_catalog_sha256
    ):
        failures.add(TaskGraphProductionFailureCode.CAPABILITY_CATALOG_STALE)
        raw_findings.add(
            _finding_key(TaskGraphProductionFailureCode.CAPABILITY_CATALOG_STALE)
        )

    blocking_gap_count = 0
    non_blocking_gap_count = 0
    unmapped_non_blocking_gap_count = 0
    for gap in context.gaps:
        unknown_gap_anchors = set(gap.affected_required_anchor_ids) - required_anchor_ids
        for anchor_id in sorted(unknown_gap_anchors):
            failures.add(TaskGraphProductionFailureCode.UNKNOWN_GAP_ANCHOR)
            raw_findings.add(
                _finding_key(
                    TaskGraphProductionFailureCode.UNKNOWN_GAP_ANCHOR,
                    anchor_id=anchor_id,
                    gap_id=gap.gap_id,
                )
            )
        if gap.blocking:
            blocking_gap_count += 1
            failures.add(TaskGraphProductionFailureCode.BLOCKING_GAP)
            raw_findings.add(
                _finding_key(
                    TaskGraphProductionFailureCode.BLOCKING_GAP,
                    gap_id=gap.gap_id,
                )
            )
            continue
        non_blocking_gap_count += 1
        if not _non_blocking_gap_is_mapped(
            gap,
            nodes_by_key=nodes_by_key,
            acceptance_lookup=acceptance_lookup,
        ):
            unmapped_non_blocking_gap_count += 1
            failures.add(
                TaskGraphProductionFailureCode.NON_BLOCKING_GAP_UNMAPPED
            )
            raw_findings.add(
                _finding_key(
                    TaskGraphProductionFailureCode.NON_BLOCKING_GAP_UNMAPPED,
                    gap_id=gap.gap_id,
                )
            )

    max_depth = max(depth_by_key.values(), default=0)
    root_child_count = sum(
        child in reachable
        for child in children.get(proposal.root.root_key, ())
    )
    telemetry = TaskGraphProductionTelemetry(
        total_node_count=len(proposal.root.nodes),
        unique_node_count=len(ordered_unique_nodes),
        reachable_node_count=len(reachable),
        max_depth=max_depth,
        root_child_count=root_child_count,
        internal_node_count=sum(
            count > 0 for count in reachable_child_counts.values()
        ),
        unary_internal_node_count=sum(
            count == 1 for count in reachable_child_counts.values()
        ),
        branching_node_count=sum(
            count >= 2 for count in reachable_child_counts.values()
        ),
        leaf_node_count=sum(
            count == 0 for count in reachable_child_counts.values()
        ),
        max_branching_factor=max(reachable_child_counts.values(), default=0),
        acceptance_count=acceptance_count,
        duplicate_acceptance_criterion_count=(
            duplicate_acceptance_criterion_count
        ),
        source_reference_count=len(all_source_ref_list),
        unknown_source_reference_count=len(unknown_source_references),
        node_authorization_coverage_count=node_authorization_coverage_count,
        acceptance_authorization_coverage_count=(
            acceptance_authorization_coverage_count
        ),
        required_goal_count=len(required_goal_ids),
        required_goal_node_coverage_count=len(required_goal_ids & node_source_refs),
        required_goal_acceptance_coverage_count=len(
            required_goal_ids & acceptance_source_refs
        ),
        required_source_count=len(required_source_ids),
        required_source_node_coverage_count=len(
            required_source_ids & node_source_refs
        ),
        required_source_acceptance_coverage_count=len(
            required_source_ids & acceptance_source_refs
        ),
        execution_requirement_count=len(requirement_by_node),
        missing_execution_requirement_count=len(missing_requirement_keys),
        unknown_execution_requirement_count=len(unknown_requirement_keys),
        capability_requirement_count=capability_requirement_count,
        unavailable_capability_count=unavailable_capability_count,
        effect_requirement_count=effect_requirement_count,
        unsatisfiable_effect_count=unsatisfiable_effect_count,
        blocking_gap_count=blocking_gap_count,
        non_blocking_gap_count=non_blocking_gap_count,
        unmapped_non_blocking_gap_count=unmapped_non_blocking_gap_count,
        exact_duplicate_node_count=len(duplicate_pairs),
        canonical_error_count=len(canonical_codes),
    )
    ordered_failure_codes = tuple(sorted(failures, key=str))
    findings = tuple(
        TaskGraphProductionFinding(
            **{
                field_name: value
                for field_name, value in zip(_FINDING_FIELDS, key, strict=True)
                if value is not None
            }
        )
        for key in sorted(raw_findings, key=_finding_sort_key)
    )
    payload = {
        "schema_version": "task-graph-production-eval-v2",
        "profile_id": "task-graph-production-adaptive-v2",
        "passed": not ordered_failure_codes,
        "telemetry": telemetry.model_dump(mode="json"),
        "failure_codes": [item.value for item in ordered_failure_codes],
        "findings": [item.model_dump(mode="json") for item in findings],
        "canonical_validation_codes": [
            item.value for item in canonical_codes
        ],
        "semantic_review_required": True,
        "proposal_sha256": _canonical_sha256(proposal.model_dump(mode="json")),
        "context_sha256": _canonical_sha256(context.model_dump(mode="json")),
    }
    return TaskGraphProductionEvaluation.model_validate(
        {
            **payload,
            "evaluation_sha256": _canonical_sha256(payload),
        }
    )


def _tree_projection(
    *,
    root_key: str,
    nodes_by_key: dict[str, InSessionTaskNodeProposal],
) -> tuple[set[str], dict[str, int], dict[str, list[str]]]:
    children: dict[str, list[str]] = {node_key: [] for node_key in nodes_by_key}
    for node in nodes_by_key.values():
        if node.parent_node_key in children and node.node_key != root_key:
            children[node.parent_node_key].append(node.node_key)
    if root_key not in nodes_by_key:
        return set(), {}, children
    reachable: set[str] = set()
    depth_by_key: dict[str, int] = {}
    stack: list[tuple[str, int, frozenset[str]]] = [
        (root_key, 1, frozenset())
    ]
    while stack:
        node_key, depth, lineage = stack.pop()
        if node_key in lineage:
            continue
        reachable.add(node_key)
        depth_by_key[node_key] = max(depth, depth_by_key.get(node_key, 0))
        next_lineage = lineage | {node_key}
        for child_key in reversed(children.get(node_key, ())):
            stack.append((child_key, depth + 1, next_lineage))
    return reachable, depth_by_key, children


def _exact_duplicate_nodes(
    nodes: list[InSessionTaskNodeProposal],
) -> tuple[tuple[str, str], ...]:
    first_by_contract: dict[tuple[object, ...], str] = {}
    duplicates: list[tuple[str, str]] = []
    for node in nodes:
        signature = _node_contract_signature(node)
        earlier = first_by_contract.get(signature)
        if earlier is not None:
            duplicates.append((node.node_key, earlier))
        else:
            first_by_contract[signature] = node.node_key
    return tuple(duplicates)


def _node_contract_signature(
    node: InSessionTaskNodeProposal,
) -> tuple[object, ...]:
    acceptance = tuple(
        sorted(
            (
                _normalize_text(item.criterion),
                tuple(sorted(item.source_anchor_ids)),
            )
            for item in node.acceptance_criteria
        )
    )
    return (
        _normalize_text(node.title),
        _normalize_text(node.objective),
        tuple(sorted(node.source_anchor_ids)),
        acceptance,
        tuple(_normalize_text(item) for item in node.constraints),
    )


def _non_blocking_gap_is_mapped(
    gap: TaskGraphPlanningGap,
    *,
    nodes_by_key: dict[str, InSessionTaskNodeProposal],
    acceptance_lookup: dict[tuple[str, str], frozenset[str]],
) -> bool:
    if not gap.mapped_node_keys and not gap.mapped_acceptances:
        return False
    mapped_anchor_ids: set[str] = set()
    for node_key in gap.mapped_node_keys:
        node = nodes_by_key.get(node_key)
        if node is None:
            return False
        mapped_anchor_ids.update(node.source_anchor_ids)
    for acceptance_ref in gap.mapped_acceptances:
        source_anchor_ids = acceptance_lookup.get(
            (acceptance_ref.node_key, acceptance_ref.acceptance_id)
        )
        if source_anchor_ids is None:
            return False
        mapped_anchor_ids.update(source_anchor_ids)
    return set(gap.affected_required_anchor_ids).issubset(mapped_anchor_ids)


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


_FINDING_FIELDS = (
    "code",
    "node_key",
    "related_node_key",
    "acceptance_id",
    "anchor_id",
    "capability_id",
    "effect_id",
    "gap_id",
    "canonical_code",
)


def _finding_key(
    code: TaskGraphProductionFailureCode,
    *,
    node_key: str | None = None,
    related_node_key: str | None = None,
    acceptance_id: str | None = None,
    anchor_id: str | None = None,
    capability_id: str | None = None,
    effect_id: str | None = None,
    gap_id: str | None = None,
    canonical_code: InSessionTaskGraphValidationCode | None = None,
) -> tuple[object, ...]:
    return (
        code,
        node_key,
        related_node_key,
        acceptance_id,
        anchor_id,
        capability_id,
        effect_id,
        gap_id,
        canonical_code,
    )


def _finding_sort_key(value: tuple[object, ...]) -> tuple[str, ...]:
    return tuple("" if item is None else str(item) for item in value)


def _finding_from_model(
    finding: TaskGraphProductionFinding,
) -> tuple[object, ...]:
    return tuple(getattr(finding, field_name) for field_name in _FINDING_FIELDS)


def _canonical_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


__all__ = [
    'TaskGraphAcceptanceRef',
    'TaskGraphNodeExecutionRequirement',
    'TaskGraphPlanningGap',
    'TaskGraphProductionCapabilityContext',
    'TaskGraphProductionEvaluationContext',
    'TaskGraphProductionEvaluation',
    "TaskGraphProductionFailureCode",
    'TaskGraphProductionFinding',
    'TaskGraphProductionFreshness',
    'TaskGraphProductionTelemetry',
    "evaluate_task_graph_production",
]
