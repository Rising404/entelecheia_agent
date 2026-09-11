"""L2 TaskGraph 完整 Task 交付验证与 revision authority 的纯契约。

节点验证根据节点 Acceptance 证明一个 WorkRun 输出。这些契约覆盖独立发布边界：规范根
Delivery 存在后，独立 reviewer 会根据 Task 原始且绑定源的目标，评估完整当前 TaskGraph 与
最终正文。

模型只拥有各维度 verdict 和有界正文。Host 派生聚合 PASS/REVISE/BLOCKED disposition。
REVISE 结果随后可以成为不可变且精确的 ``TaskGraph N -> N+1`` trigger；在 Store 于同一
事务中认证请求、结果、根 Delivery 和已完成 Task 状态前，它自身绝不是 mutation authority。
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import (
    InSessionTaskAcceptanceProposal,
    InSessionTaskNodeKind,
    InSessionTaskSourceAnchor,
)


_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
TASK_DELIVERY_VALIDATION_MAX_CHILD_DELIVERIES = 511
TASK_DELIVERY_VALIDATION_CHILD_OUTPUT_PER_DELIVERY_MAX_UTF8_BYTES = 32_768
TASK_DELIVERY_VALIDATION_CHILD_OUTPUT_MAX_UTF8_BYTES = 262_144


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_value(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskDeliveryValidationDimension(StrEnum):
    GOAL_COMPLETENESS = "goal_completeness"
    FACTUAL_CORRECTNESS = "factual_correctness"
    EVIDENCE_GROUNDING = "evidence_grounding"
    CONSTRAINT_FIDELITY = "constraint_fidelity"
    CROSS_NODE_COHERENCE = "cross_node_coherence"
    FINAL_DELIVERY_QUALITY = "final_delivery_quality"


class TaskDeliveryValidationVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class TaskDeliveryValidationFaultDomain(StrEnum):
    """一项完整 Task 验证 finding 的语义所有权。

    reviewer 对缺陷分类，但不拥有聚合路由。``NONE`` 保留给 PASS finding；每个非 PASS
    finding 都有一个显式修复 authority 边界。
    """

    NONE = "none"
    EXECUTION_OUTPUT = "execution_output"
    TASK_GRAPH_DESIGN = "task_graph_design"
    MISSING_INFORMATION = "missing_information"
    MISSING_AUTHORITY = "missing_authority"


class TaskDeliveryValidationDisposition(StrEnum):
    """完整 Task 验证边界处由 Host 派生的动作。"""

    PASS = "pass"
    RETRY_EXECUTION = "retry_execution"
    REPLAN_TASK_GRAPH = "replan_task_graph"
    BLOCKED = "blocked"


class TaskDeliveryValidationNode(_Contract):
    node_id: str = Field(pattern=_ID_PATTERN)
    node_revision: int = Field(ge=1)
    node_kind: InSessionTaskNodeKind
    parent_node_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    ordinal: int = Field(ge=0)
    title: str = Field(min_length=1, max_length=240)
    objective: str = Field(min_length=1, max_length=2_000)
    source_anchor_ids: tuple[str, ...] = Field(min_length=1, max_length=32)
    acceptance_criteria: tuple[InSessionTaskAcceptanceProposal, ...] = Field(
        min_length=1,
        max_length=64,
    )
    constraints: tuple[str, ...] = Field(default=(), max_length=32)

    @field_validator("source_anchor_ids", "constraints")
    @classmethod
    def _require_unique_values(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("Task delivery validation node values must be unique")
        return values

    @model_validator(mode="after")
    def _validate_parent_shape(self) -> "TaskDeliveryValidationNode":
        if self.node_kind is InSessionTaskNodeKind.ROOT:
            if self.parent_node_id is not None:
                raise ValueError("root validation node cannot have a parent")
        elif self.parent_node_id is None:
            raise ValueError("subtask validation node requires a parent")
        return self


class TaskDeliveryValidationNodeVerificationAttestation(_Contract):
    """一个精确节点输出通过节点验证的 prompt 安全证明。

    此投影刻意只携带不可变身份和由 Host 派生的 PASS 聚合。ToolResult 正文仍由节点验证记录
    所有，绝不会复制进完整 Task prompt。
    """

    schema_version: Literal[
        "task-delivery-validation-node-verification-attestation-v1"
    ] = "task-delivery-validation-node-verification-attestation-v1"
    node_id: str = Field(pattern=_ID_PATTERN)
    node_revision: int = Field(ge=1)
    verification_source: Literal[
        "direct_delivery",
        "carried_delivery",
        "root_candidate",
    ]
    delivery_id: str = Field(pattern=_ID_PATTERN)
    verified_subject_graph_revision: int = Field(ge=1)
    verification_request_id: str = Field(pattern=_ID_PATTERN)
    verification_request_revision: int = Field(ge=1)
    work_run_id: str = Field(pattern=_ID_PATTERN)
    submitted_attempt_id: str = Field(pattern=_ID_PATTERN)
    output_revision: int = Field(ge=1)
    acceptance_ids: tuple[str, ...] = Field(min_length=1, max_length=64)
    supporting_tool_result_ids: tuple[str, ...] = Field(
        default=(),
        max_length=4_096,
    )
    all_pass: Literal[True] = True
    verification_result_sha256: str = Field(pattern=_SHA256_PATTERN)
    binding_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("acceptance_ids", "supporting_tool_result_ids")
    @classmethod
    def _require_unique_nonempty_ids(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("node verification attestation IDs must not be empty")
        if len(values) != len(set(values)):
            raise ValueError("node verification attestation IDs must be unique")
        return values

    @model_validator(mode="after")
    def _validate_binding(self) -> "TaskDeliveryValidationNodeVerificationAttestation":
        expected = _sha256_value(
            self.model_dump(mode="json", exclude={"binding_sha256"})
        )
        if self.binding_sha256 != expected:
            raise ValueError("node verification attestation binding is invalid")
        return self

    @classmethod
    def create(cls, **values: object) -> "TaskDeliveryValidationNodeVerificationAttestation":
        values = dict(values)
        values["binding_sha256"] = "0" * 64
        provisional = cls.model_construct(**values)
        values["binding_sha256"] = _sha256_value(
            provisional.model_dump(mode="json", exclude={"binding_sha256"})
        )
        return cls.model_validate(values)


class TaskDeliveryValidationChildDeliveryMaterial(_Contract):
    """一个当前非根 Delivery 的 Host 私有源材料。"""

    node_id: str = Field(pattern=_ID_PATTERN)
    node_revision: int = Field(ge=1)
# TaskGraph 节点以规范零基序号存储。根节点不要求位于首位，因此非根 Delivery 可以合法地
# 拥有序号零。
    ordinal: int = Field(ge=0)
    delivery_id: str = Field(pattern=_ID_PATTERN)
    source_graph_revision: int = Field(ge=1)
    resolution_kind: Literal["direct", "carried"]
    output_format: str = Field(min_length=1, max_length=64)
    output_body: str = Field(min_length=1)


class TaskDeliveryValidationChildDelivery(_Contract):
    """一个已验证子输出的有界、可自认证投影。"""

    schema_version: Literal["task-delivery-validation-child-delivery-v1"] = (
        "task-delivery-validation-child-delivery-v1"
    )
    node_id: str = Field(pattern=_ID_PATTERN)
    node_revision: int = Field(ge=1)
    ordinal: int = Field(ge=0)
    delivery_id: str = Field(pattern=_ID_PATTERN)
    source_graph_revision: int = Field(ge=1)
    resolution_kind: Literal["direct", "carried"]
    output_format: str = Field(min_length=1, max_length=64)
    output_utf8_bytes: int = Field(ge=1)
    output_sha256: str = Field(pattern=_SHA256_PATTERN)
    truncated: bool
    output_body: str | None = Field(
        default=None,
        max_length=(
            TASK_DELIVERY_VALIDATION_CHILD_OUTPUT_PER_DELIVERY_MAX_UTF8_BYTES
        ),
        exclude_if=lambda value: value is None,
    )
    output_body_prefix: str | None = Field(
        default=None,
        max_length=(
            TASK_DELIVERY_VALIDATION_CHILD_OUTPUT_PER_DELIVERY_MAX_UTF8_BYTES
        ),
        exclude_if=lambda value: value is None,
    )
    output_body_suffix: str | None = Field(
        default=None,
        max_length=(
            TASK_DELIVERY_VALIDATION_CHILD_OUTPUT_PER_DELIVERY_MAX_UTF8_BYTES
        ),
        exclude_if=lambda value: value is None,
    )
    excerpt_utf8_bytes: int = Field(
        ge=1,
        le=TASK_DELIVERY_VALIDATION_CHILD_OUTPUT_PER_DELIVERY_MAX_UTF8_BYTES,
    )
    binding_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_projection(self) -> "TaskDeliveryValidationChildDelivery":
        if self.truncated:
            if (
                self.output_body is not None
                or not self.output_body_prefix
                or not self.output_body_suffix
            ):
                raise ValueError(
                    "truncated child Delivery requires only a body prefix and suffix"
                )
            excerpt = self.output_body_prefix + self.output_body_suffix
            if self.excerpt_utf8_bytes >= self.output_utf8_bytes:
                raise ValueError("truncated child Delivery must omit source bytes")
        else:
            if (
                not self.output_body
                or self.output_body_prefix is not None
                or self.output_body_suffix is not None
            ):
                raise ValueError("complete child Delivery requires only output_body")
            excerpt = self.output_body
            if (
                hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
                != self.output_sha256
            ):
                raise ValueError("complete child Delivery output hash is invalid")
            if self.excerpt_utf8_bytes != self.output_utf8_bytes:
                raise ValueError("complete child Delivery byte length is invalid")
        if len(excerpt.encode("utf-8")) != self.excerpt_utf8_bytes:
            raise ValueError("child Delivery excerpt byte length is invalid")
        expected = _sha256_value(
            self.model_dump(mode="json", exclude={"binding_sha256"})
        )
        if self.binding_sha256 != expected:
            raise ValueError("child Delivery binding hash is invalid")
        return self

    @classmethod
    def create(
        cls,
        *,
        material: TaskDeliveryValidationChildDeliveryMaterial,
        excerpt_utf8_limit: int,
    ) -> "TaskDeliveryValidationChildDelivery":
        if not isinstance(material, TaskDeliveryValidationChildDeliveryMaterial):
            raise TypeError("child Delivery material must be typed")
        if (
            isinstance(excerpt_utf8_limit, bool)
            or not isinstance(excerpt_utf8_limit, int)
            or not 8 <= excerpt_utf8_limit <= (
                TASK_DELIVERY_VALIDATION_CHILD_OUTPUT_PER_DELIVERY_MAX_UTF8_BYTES
            )
        ):
            raise ValueError("child Delivery excerpt limit is invalid")
        encoded = material.output_body.encode("utf-8")
        values: dict[str, object] = {
            **material.model_dump(mode="python", exclude={"output_body"}),
            "output_utf8_bytes": len(encoded),
            "output_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        if len(encoded) <= excerpt_utf8_limit:
            values.update(
                {
                    "truncated": False,
                    "output_body": material.output_body,
                    "excerpt_utf8_bytes": len(encoded),
                }
            )
        else:
            prefix = _utf8_prefix(
                encoded,
                (excerpt_utf8_limit + 1) // 2,
            )
            suffix = _utf8_suffix(encoded, excerpt_utf8_limit // 2)
            values.update(
                {
                    "truncated": True,
                    "output_body_prefix": prefix,
                    "output_body_suffix": suffix,
                    "excerpt_utf8_bytes": len(
                        (prefix + suffix).encode("utf-8")
                    ),
                }
            )
        values["binding_sha256"] = "0" * 64
        provisional = cls.model_construct(**values)
        values["binding_sha256"] = _sha256_value(
            provisional.model_dump(mode="json", exclude={"binding_sha256"})
        )
        return cls.model_validate(values)


class TaskDeliveryValidationChildDeliveryProjection(_Contract):
    """带有界聚合文本信封的完整当前子项集合。"""

    schema_version: Literal[
        "task-delivery-validation-child-delivery-projection-v1"
    ] = "task-delivery-validation-child-delivery-projection-v1"
    deliveries: tuple[TaskDeliveryValidationChildDelivery, ...] = Field(
        default=(),
        max_length=TASK_DELIVERY_VALIDATION_MAX_CHILD_DELIVERIES,
    )
    source_output_utf8_bytes: int = Field(ge=0)
    included_output_utf8_bytes: int = Field(
        ge=0,
        le=TASK_DELIVERY_VALIDATION_CHILD_OUTPUT_MAX_UTF8_BYTES,
    )
    truncated_delivery_count: int = Field(
        ge=0,
        le=TASK_DELIVERY_VALIDATION_MAX_CHILD_DELIVERIES,
    )
    projection_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_projection(
        self,
    ) -> "TaskDeliveryValidationChildDeliveryProjection":
        order = tuple(
            (item.ordinal, item.node_id, item.delivery_id)
            for item in self.deliveries
        )
        if order != tuple(sorted(order)):
            raise ValueError("child Deliveries must use deterministic node order")
        node_ids = tuple(item.node_id for item in self.deliveries)
        delivery_ids = tuple(item.delivery_id for item in self.deliveries)
        if len(node_ids) != len(set(node_ids)) or len(delivery_ids) != len(
            set(delivery_ids)
        ):
            raise ValueError("child Delivery identities must be unique")
        if self.source_output_utf8_bytes != sum(
            item.output_utf8_bytes for item in self.deliveries
        ):
            raise ValueError("child Delivery source byte total is invalid")
        if self.included_output_utf8_bytes != sum(
            item.excerpt_utf8_bytes for item in self.deliveries
        ):
            raise ValueError("child Delivery excerpt byte total is invalid")
        if self.truncated_delivery_count != sum(
            item.truncated for item in self.deliveries
        ):
            raise ValueError("child Delivery truncation count is invalid")
        expected = _sha256_value(
            self.model_dump(mode="json", exclude={"projection_sha256"})
        )
        if self.projection_sha256 != expected:
            raise ValueError("child Delivery projection hash is invalid")
        return self

    @classmethod
    def create(
        cls,
        *,
        deliveries: tuple[TaskDeliveryValidationChildDelivery, ...],
    ) -> "TaskDeliveryValidationChildDeliveryProjection":
        ordered = tuple(
            sorted(
                deliveries,
                key=lambda item: (item.ordinal, item.node_id, item.delivery_id),
            )
        )
        values: dict[str, object] = {
            "deliveries": ordered,
            "source_output_utf8_bytes": sum(
                item.output_utf8_bytes for item in ordered
            ),
            "included_output_utf8_bytes": sum(
                item.excerpt_utf8_bytes for item in ordered
            ),
            "truncated_delivery_count": sum(item.truncated for item in ordered),
            "projection_sha256": "0" * 64,
        }
        provisional = cls.model_construct(**values)
        values["projection_sha256"] = _sha256_value(
            provisional.model_dump(mode="json", exclude={"projection_sha256"})
        )
        return cls.model_validate(values)


def build_task_delivery_validation_child_delivery_projection(
    materials: tuple[TaskDeliveryValidationChildDeliveryMaterial, ...],
) -> TaskDeliveryValidationChildDeliveryProjection:
    """使用确定性共享 UTF-8 文本预算投影每个子项。"""

    if not isinstance(materials, tuple) or any(
        not isinstance(item, TaskDeliveryValidationChildDeliveryMaterial)
        for item in materials
    ):
        raise TypeError("child Delivery materials must be a typed tuple")
    if len(materials) > TASK_DELIVERY_VALIDATION_MAX_CHILD_DELIVERIES:
        raise ValueError("too many child Deliveries for whole-Task validation")
    if not materials:
        return TaskDeliveryValidationChildDeliveryProjection.create(
            deliveries=()
        )
    excerpt_limit = min(
        TASK_DELIVERY_VALIDATION_CHILD_OUTPUT_PER_DELIVERY_MAX_UTF8_BYTES,
        TASK_DELIVERY_VALIDATION_CHILD_OUTPUT_MAX_UTF8_BYTES // len(materials),
    )
    deliveries = tuple(
        TaskDeliveryValidationChildDelivery.create(
            material=material,
            excerpt_utf8_limit=excerpt_limit,
        )
        for material in materials
    )
    return TaskDeliveryValidationChildDeliveryProjection.create(
        deliveries=deliveries
    )


class TaskDeliveryValidationPrompt(_Contract):
    """模型评审前冻结的精确 prompt 安全完整 Task 材料。"""

    schema_version: Literal["task-delivery-validation-prompt-v1"] = (
        "task-delivery-validation-prompt-v1"
    )
    session_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    graph_revision: int = Field(ge=1)
    task_state_version: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=240)
    objective: str = Field(min_length=1, max_length=2_000)
    source_anchors: tuple[InSessionTaskSourceAnchor, ...] = Field(
        max_length=256
    )
    nodes: tuple[TaskDeliveryValidationNode, ...] = Field(
        min_length=1,
        max_length=512,
    )
    root_delivery_id: str = Field(pattern=_ID_PATTERN)
    root_output_format: str = Field(min_length=1, max_length=64)
    root_output_body: str = Field(min_length=1, max_length=1_000_000)
    child_delivery_projection: (
        TaskDeliveryValidationChildDeliveryProjection | None
    ) = Field(default=None, exclude_if=lambda value: value is None)
    node_verification_attestations: tuple[
        TaskDeliveryValidationNodeVerificationAttestation,
        ...,
    ] = Field(
        default=(),
        max_length=512,
        exclude_if=lambda value: not value,
    )
    payload_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_prompt(self) -> "TaskDeliveryValidationPrompt":
        node_ids = tuple(item.node_id for item in self.nodes)
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("validation prompt node IDs must be unique")
        roots = tuple(
            item for item in self.nodes if item.node_kind is InSessionTaskNodeKind.ROOT
        )
        if len(roots) != 1 or roots[0].node_id != self.task_id:
            raise ValueError("validation prompt requires one canonical Task root")
        known_nodes = set(node_ids)
        if any(
            item.parent_node_id is not None
            and item.parent_node_id not in known_nodes
            for item in self.nodes
        ):
            raise ValueError("validation prompt contains an unknown parent node")
        anchor_ids = tuple(item.anchor_id for item in self.source_anchors)
        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError("validation prompt source anchors must be unique")
        attestations = self.node_verification_attestations
        if attestations:
            ordered_nodes = tuple(
                sorted(self.nodes, key=lambda item: (item.ordinal, item.node_id))
            )
            if tuple(
                (item.node_id, item.node_revision) for item in attestations
            ) != tuple(
                (item.node_id, item.node_revision) for item in ordered_nodes
            ):
                raise ValueError(
                    "node verification attestations must cover the exact node set"
                )
            if len({item.delivery_id for item in attestations}) != len(attestations):
                raise ValueError(
                    "node verification attestation Delivery IDs must be unique"
                )
            if len(
                {item.verification_request_id for item in attestations}
            ) != len(attestations):
                raise ValueError(
                    "node verification request IDs must be unique per prompt"
                )
            for attestation, node in zip(attestations, ordered_nodes, strict=True):
                acceptance_ids = tuple(
                    item.acceptance_id for item in node.acceptance_criteria
                )
                if attestation.acceptance_ids != acceptance_ids:
                    raise ValueError(
                        "node verification attestation Acceptance binding is invalid"
                    )
                if attestation.verification_source == "root_candidate":
                    if (
                        node.node_kind is not InSessionTaskNodeKind.ROOT
                        or attestation.delivery_id != self.root_delivery_id
                        or attestation.verified_subject_graph_revision
                        != self.graph_revision
                    ):
                        raise ValueError(
                            "root-candidate attestation is not bound to the root"
                        )
                elif attestation.verification_source == "direct_delivery":
                    if (
                        attestation.verified_subject_graph_revision
                        != self.graph_revision
                    ):
                        raise ValueError(
                            "direct Delivery attestation has a stale graph revision"
                        )
                elif (
                    attestation.verified_subject_graph_revision
                    >= self.graph_revision
                ):
                    raise ValueError(
                        "carried Delivery attestation requires an older source graph"
                    )
            root_attestation = next(
                item for item in attestations if item.node_id == self.task_id
            )
            if root_attestation.delivery_id != self.root_delivery_id:
                raise ValueError(
                    "root verification attestation crossed the root Delivery"
                )
        projection = self.child_delivery_projection
        if projection is not None:
            expected_children = tuple(
                (item.ordinal, item.node_id, item.node_revision)
                for item in sorted(
                    (
                        node
                        for node in self.nodes
                        if node.node_kind is not InSessionTaskNodeKind.ROOT
                    ),
                    key=lambda item: (item.ordinal, item.node_id),
                )
            )
            actual_children = tuple(
                (item.ordinal, item.node_id, item.node_revision)
                for item in projection.deliveries
            )
            if actual_children != expected_children:
                raise ValueError(
                    "child Delivery projection must cover every non-root node"
                )
            if any(
                item.source_graph_revision > self.graph_revision
                or (
                    item.resolution_kind == "direct"
                    and item.source_graph_revision != self.graph_revision
                )
                or (
                    item.resolution_kind == "carried"
                    and item.source_graph_revision >= self.graph_revision
                )
                for item in projection.deliveries
            ):
                raise ValueError(
                    "child Delivery source revision is inconsistent with the graph"
                )
            if attestations:
                child_attestations = {
                    item.node_id: item
                    for item in attestations
                    if item.node_id != self.task_id
                }
                if any(
                    child_attestations[item.node_id].delivery_id
                    != item.delivery_id
                    or child_attestations[item.node_id].verified_subject_graph_revision
                    != item.source_graph_revision
                    or child_attestations[item.node_id].verification_source
                    != f"{item.resolution_kind}_delivery"
                    for item in projection.deliveries
                ):
                    raise ValueError(
                        "child verification attestations crossed Delivery projection"
                    )
        expected = _sha256_value(
            self.model_dump(mode="json", exclude={"payload_sha256"})
        )
        if self.payload_sha256 != expected:
            raise ValueError("validation prompt hash does not match its payload")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        values["payload_sha256"] = "0" * 64
        provisional = cls.model_construct(**values)
        values["payload_sha256"] = _sha256_value(
            provisional.model_dump(mode="json", exclude={"payload_sha256"})
        )
        return cls.model_validate(values)


def _utf8_prefix(encoded: bytes, limit: int) -> str:
    return encoded[:limit].decode("utf-8", errors="ignore")


def _utf8_suffix(encoded: bytes, limit: int) -> str:
    return encoded[-limit:].decode("utf-8", errors="ignore")


class TaskDeliveryValidationRequest(_Contract):
    schema_version: Literal["task-delivery-validation-request-v1"] = (
        "task-delivery-validation-request-v1"
    )
    verification_request_id: str = Field(pattern=_ID_PATTERN)
    verification_result_id: str = Field(pattern=_ID_PATTERN)
    logical_call_id: str = Field(pattern=_ID_PATTERN)
    verification_profile_id: str = Field(pattern=_ID_PATTERN)
    invocation_turn_id: str = Field(pattern=_ID_PATTERN)
    prompt: TaskDeliveryValidationPrompt
    binding_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_binding(self) -> "TaskDeliveryValidationRequest":
        expected = _sha256_value(
            self.model_dump(mode="json", exclude={"binding_sha256"})
        )
        if self.binding_sha256 != expected:
            raise ValueError("Task delivery validation request binding is invalid")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        values["binding_sha256"] = "0" * 64
        provisional = cls.model_construct(**values)
        values["binding_sha256"] = _sha256_value(
            provisional.model_dump(mode="json", exclude={"binding_sha256"})
        )
        return cls.model_validate(values)


class TaskDeliveryValidationFinding(_Contract):
    """一项带显式机械检查 fault domain 的 finding。"""

    dimension: TaskDeliveryValidationDimension
    verdict: TaskDeliveryValidationVerdict
    fault_domain: TaskDeliveryValidationFaultDomain
    finding: str = Field(min_length=1, max_length=2_000)
    affected_node_ids: tuple[str, ...] = Field(default=(), max_length=64)
    evidence_anchor_ids: tuple[str, ...] = Field(default=(), max_length=64)

    @field_validator("affected_node_ids", "evidence_anchor_ids")
    @classmethod
    def _require_unique_refs(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("Task delivery validation references must be unique")
        return values

    @model_validator(mode="after")
    def _bind_verdict_to_fault_domain(self) -> "TaskDeliveryValidationFinding":
        if self.verdict is TaskDeliveryValidationVerdict.PASS:
            if (
                self.fault_domain is not TaskDeliveryValidationFaultDomain.NONE
                or self.affected_node_ids
            ):
                raise ValueError(
                    "PASS findings require the none fault domain and no affected nodes"
                )
            return self
        if self.verdict is TaskDeliveryValidationVerdict.FAIL:
            if self.fault_domain not in {
                TaskDeliveryValidationFaultDomain.EXECUTION_OUTPUT,
                TaskDeliveryValidationFaultDomain.TASK_GRAPH_DESIGN,
            }:
                raise ValueError(
                    "FAIL findings require execution_output or task_graph_design"
                )
            if (
                self.fault_domain
                is TaskDeliveryValidationFaultDomain.EXECUTION_OUTPUT
                and not self.affected_node_ids
            ):
                raise ValueError(
                    "execution_output findings require affected node authority"
                )
            return self
        if self.fault_domain not in {
            TaskDeliveryValidationFaultDomain.MISSING_INFORMATION,
            TaskDeliveryValidationFaultDomain.MISSING_AUTHORITY,
        }:
            raise ValueError(
                "INSUFFICIENT_EVIDENCE findings require missing_information "
                "or missing_authority"
            )
        return self


class TaskDeliveryValidationResult(_Contract):
    """路由由 Host 派生的四分支完整 Task 结果。"""

    schema_version: Literal["task-delivery-validation-result-v2"] = (
        "task-delivery-validation-result-v2"
    )
    verification_result_id: str = Field(pattern=_ID_PATTERN)
    verification_request_id: str = Field(pattern=_ID_PATTERN)
    logical_call_id: str = Field(pattern=_ID_PATTERN)
    request_binding_sha256: str = Field(pattern=_SHA256_PATTERN)
    findings: tuple[TaskDeliveryValidationFinding, ...] = Field(
        min_length=len(TaskDeliveryValidationDimension),
        max_length=len(TaskDeliveryValidationDimension),
    )
    disposition: TaskDeliveryValidationDisposition
    summary: str = Field(min_length=1, max_length=2_000)
    execution_repair_objective: str | None = Field(
        default=None,
        min_length=1,
        max_length=4_000,
    )
    task_graph_revision_objective: str | None = Field(
        default=None,
        min_length=1,
        max_length=4_000,
    )
    blocking_questions: tuple[str, ...] = Field(default=(), max_length=16)
    result_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("blocking_questions")
    @classmethod
    def _validate_questions(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)) or any(
            not value.strip() or len(value) > 1_000 for value in values
        ):
            raise ValueError("blocking questions must be unique bounded text")
        return values

    @model_validator(mode="after")
    def _validate_result(self) -> "TaskDeliveryValidationResult":
        dimensions = tuple(item.dimension for item in self.findings)
        if len(dimensions) != len(set(dimensions)) or set(dimensions) != set(
            TaskDeliveryValidationDimension
        ):
            raise ValueError("delivery validation must cover every dimension once")
        derived = derive_task_delivery_validation_disposition(self.findings)
        if self.disposition is not derived:
            raise ValueError("delivery validation disposition is not Host-derived")
        if derived is TaskDeliveryValidationDisposition.RETRY_EXECUTION:
            if (
                self.execution_repair_objective is None
                or self.task_graph_revision_objective is not None
                or self.blocking_questions
            ):
                raise ValueError(
                    "RETRY_EXECUTION requires only an execution repair objective"
                )
        elif derived is TaskDeliveryValidationDisposition.REPLAN_TASK_GRAPH:
            if (
                self.task_graph_revision_objective is None
                or self.execution_repair_objective is not None
                or self.blocking_questions
            ):
                raise ValueError(
                    "REPLAN_TASK_GRAPH requires only a graph revision objective"
                )
        elif derived is TaskDeliveryValidationDisposition.BLOCKED:
            if (
                self.execution_repair_objective is not None
                or self.task_graph_revision_objective is not None
                or not self.blocking_questions
            ):
                raise ValueError("BLOCKED requires only blocking questions")
        elif (
            self.execution_repair_objective is not None
            or self.task_graph_revision_objective is not None
            or self.blocking_questions
        ):
            raise ValueError("PASS cannot carry repair or blocking work")
        expected = _sha256_value(
            self.model_dump(mode="json", exclude={"result_sha256"})
        )
        if self.result_sha256 != expected:
            raise ValueError("delivery validation result hash is invalid")
        return self

    @classmethod
    def create(
        cls,
        *,
        findings: tuple[TaskDeliveryValidationFinding, ...],
        **values: object,
    ) -> Self:
        values = dict(values)
        values["findings"] = findings
        values["disposition"] = derive_task_delivery_validation_disposition(
            findings
        )
        values["result_sha256"] = "0" * 64
        provisional = cls.model_construct(**values)
        values["result_sha256"] = _sha256_value(
            provisional.model_dump(mode="json", exclude={"result_sha256"})
        )
        return cls.model_validate(values)


def derive_task_delivery_validation_disposition(
    findings: tuple[TaskDeliveryValidationFinding, ...],
) -> TaskDeliveryValidationDisposition:
    """按保守失败优先级派生唯一允许的路由。"""

    if any(
        item.verdict is TaskDeliveryValidationVerdict.INSUFFICIENT_EVIDENCE
        or item.fault_domain in {
            TaskDeliveryValidationFaultDomain.MISSING_INFORMATION,
            TaskDeliveryValidationFaultDomain.MISSING_AUTHORITY,
        }
        for item in findings
    ):
        return TaskDeliveryValidationDisposition.BLOCKED
    if any(
        item.fault_domain is TaskDeliveryValidationFaultDomain.TASK_GRAPH_DESIGN
        for item in findings
    ):
        return TaskDeliveryValidationDisposition.REPLAN_TASK_GRAPH
    if any(
        item.fault_domain is TaskDeliveryValidationFaultDomain.EXECUTION_OUTPUT
        for item in findings
    ):
        return TaskDeliveryValidationDisposition.RETRY_EXECUTION
    return TaskDeliveryValidationDisposition.PASS


def validate_task_delivery_validation_result(
    *,
    request: TaskDeliveryValidationRequest,
    result: TaskDeliveryValidationResult,
) -> TaskDeliveryValidationResult:
    """将结果绑定到现有冻结请求 authority。"""

    if (
        result.verification_result_id != request.verification_result_id
        or result.verification_request_id != request.verification_request_id
        or result.logical_call_id != request.logical_call_id
        or result.request_binding_sha256 != request.binding_sha256
    ):
        raise ValueError("delivery validation result crossed request authority")
    node_ids = {item.node_id for item in request.prompt.nodes}
    anchor_ids = {item.anchor_id for item in request.prompt.source_anchors}
    for finding in result.findings:
        if not set(finding.affected_node_ids) <= node_ids:
            raise ValueError("delivery validation finding cites an unknown node")
        if not set(finding.evidence_anchor_ids) <= anchor_ids:
            raise ValueError("delivery validation finding cites an unknown anchor")
    return result


def require_root_only_task_delivery_execution_retry(
    *,
    result: TaskDeliveryValidationResult,
    root_node_id: str,
) -> TaskDeliveryValidationResult:
    """保护首个执行修复切片，防止重写冻结子项。

    这被刻意设计为非通用路由验证器。调用方只会在应用 RETRY_EXECUTION 结算前使用它；此时
    Phase 1 支持仍可变的规范根 WorkRun，且没有已完成子 generation。
    """

    if not isinstance(result, TaskDeliveryValidationResult):
        raise TypeError("result must be TaskDeliveryValidationResult")
    if (
        not isinstance(root_node_id, str)
        or not root_node_id
        or len(root_node_id) > 200
    ):
        raise ValueError("root_node_id must be a bounded identity")
    if result.disposition is not TaskDeliveryValidationDisposition.RETRY_EXECUTION:
        raise ValueError("root-only guard requires RETRY_EXECUTION")
    execution_findings = tuple(
        item
        for item in result.findings
        if item.fault_domain
        is TaskDeliveryValidationFaultDomain.EXECUTION_OUTPUT
    )
    expected = {root_node_id}
    if not execution_findings or any(
        set(item.affected_node_ids) != expected for item in execution_findings
    ):
        raise ValueError(
            "execution repair must affect only the canonical root node"
        )
    return result


class TaskGraphRevisionTrigger(_Contract):
    """修订精确 TaskGraph revision N 的 prompt 安全不可变 authority。"""

    schema_version: Literal["task-graph-revision-trigger-v1"] = (
        "task-graph-revision-trigger-v1"
    )
    trigger_id: str = Field(pattern=_ID_PATTERN)
    create_apply_id: str = Field(pattern=_ID_PATTERN)
    session_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    base_graph_revision: int = Field(ge=1)
    target_graph_revision: int = Field(ge=2)
    root_delivery_id: str = Field(pattern=_ID_PATTERN)
    verification_request_id: str = Field(pattern=_ID_PATTERN)
    request_binding_sha256: str = Field(pattern=_SHA256_PATTERN)
    verification_result_id: str = Field(pattern=_ID_PATTERN)
    result_sha256: str = Field(pattern=_SHA256_PATTERN)
    settlement_id: str = Field(pattern=_ID_PATTERN)
    settlement_sha256: str = Field(pattern=_SHA256_PATTERN)
    revision_objective: str = Field(min_length=1, max_length=4_000)
    gap_diagnosis: tuple[TaskDeliveryValidationFinding, ...] = Field(
        min_length=1,
        max_length=len(TaskDeliveryValidationDimension),
    )
    reopened_task_state_version: int = Field(ge=1)
    created_turn_id: str = Field(pattern=_ID_PATTERN)
    trigger_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_trigger(self) -> "TaskGraphRevisionTrigger":
        if self.target_graph_revision != self.base_graph_revision + 1:
            raise ValueError("TaskGraph revision trigger must target exact N+1")
        if any(
            item.verdict is TaskDeliveryValidationVerdict.PASS
            for item in self.gap_diagnosis
        ):
            raise ValueError("revision trigger diagnosis may contain only gaps")
        expected = _sha256_value(
            self.model_dump(mode="json", exclude={"trigger_sha256"})
        )
        if self.trigger_sha256 != expected:
            raise ValueError("TaskGraph revision trigger hash is invalid")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        values["trigger_sha256"] = "0" * 64
        provisional = cls.model_construct(**values)
        values["trigger_sha256"] = _sha256_value(
            provisional.model_dump(mode="json", exclude={"trigger_sha256"})
        )
        return cls.model_validate(values)


class TaskGraphRevisionTriggerApplication(_Contract):
    schema_version: Literal["task-graph-revision-trigger-application-v1"] = (
        "task-graph-revision-trigger-application-v1"
    )
    apply_id: str = Field(pattern=_ID_PATTERN)
    trigger_id: str = Field(pattern=_ID_PATTERN)
    trigger_sha256: str = Field(pattern=_SHA256_PATTERN)
    session_id: str = Field(pattern=_ID_PATTERN)
    task_id: str = Field(pattern=_ID_PATTERN)
    base_graph_revision: int = Field(ge=1)
    committed_graph_revision: int = Field(ge=2)
    task_graph_commit_apply_id: str = Field(pattern=_ID_PATTERN)
    consumed_turn_id: str = Field(pattern=_ID_PATTERN)
    receipt_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_application(self) -> "TaskGraphRevisionTriggerApplication":
        if self.committed_graph_revision != self.base_graph_revision + 1:
            raise ValueError("trigger application must bind exact N+1")
        expected = _sha256_value(
            self.model_dump(mode="json", exclude={"receipt_sha256"})
        )
        if self.receipt_sha256 != expected:
            raise ValueError("trigger application hash is invalid")
        return self

    @classmethod
    def create(cls, **values: object) -> Self:
        values = dict(values)
        values["receipt_sha256"] = "0" * 64
        provisional = cls.model_construct(**values)
        values["receipt_sha256"] = _sha256_value(
            provisional.model_dump(mode="json", exclude={"receipt_sha256"})
        )
        return cls.model_validate(values)


__all__ = [
    "TASK_DELIVERY_VALIDATION_CHILD_OUTPUT_MAX_UTF8_BYTES",
    "TASK_DELIVERY_VALIDATION_CHILD_OUTPUT_PER_DELIVERY_MAX_UTF8_BYTES",
    "TASK_DELIVERY_VALIDATION_MAX_CHILD_DELIVERIES",
    "TaskDeliveryValidationChildDeliveryMaterial",
    "TaskDeliveryValidationChildDeliveryProjection",
    "TaskDeliveryValidationChildDelivery",
    "TaskDeliveryValidationDimension",
    "TaskDeliveryValidationDisposition",
    "TaskDeliveryValidationFaultDomain",
    "TaskDeliveryValidationFinding",
    "TaskDeliveryValidationNode",
    "TaskDeliveryValidationNodeVerificationAttestation",
    "TaskDeliveryValidationPrompt",
    "TaskDeliveryValidationRequest",
    "TaskDeliveryValidationResult",
    "TaskDeliveryValidationVerdict",
    "TaskGraphRevisionTriggerApplication",
    "TaskGraphRevisionTrigger",
    "build_task_delivery_validation_child_delivery_projection",
    "derive_task_delivery_validation_disposition",
    "require_root_only_task_delivery_execution_retry",
    "validate_task_delivery_validation_result",
]
