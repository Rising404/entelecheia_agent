"""在 I/O 前封存一个规划上下文原语的冷契约。

该封存是资源感知适配器与 Session 持久化共享的持久边界。它特意不了解文件读取或结算：
适配器提供已经规范化的逻辑请求 JSON，此所有者将其绑定到已授权状态版本。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import re

from ..work_run.contracts import AuxiliaryNodeSubject


MAX_ALIAS_PREFIX_CHARACTERS = 44

_DURABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
_LOCAL_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PlanningContextPrimitiveKind(StrEnum):
    RESOURCE_PERCEPTION = "resource_perception"


@dataclass(frozen=True, slots=True)
class FrozenPlanningContextArtifactBinding:
    """一个观测制品的 Host 冻结标识与别名命名空间。"""

    session_id: str
    task_id: str
    auxiliary_graph_id: str
    goal_id: str
    producer_auxiliary_node: AuxiliaryNodeSubject
    primitive_call_id: str
    artifact_id: str
    verification_receipt_id: str
    authority_snapshot_id: str
    scope_snapshot_sha256: str
    alias_prefix: str
    artifact_alias: str
    producer_node_alias: str
    affected_obligations: tuple[str, ...] = ("task_graph_proposal",)

    def __post_init__(self) -> None:
        for name, value in (
            ("session_id", self.session_id),
            ("task_id", self.task_id),
            ("auxiliary_graph_id", self.auxiliary_graph_id),
            ("goal_id", self.goal_id),
            ("primitive_call_id", self.primitive_call_id),
            ("artifact_id", self.artifact_id),
            ("verification_receipt_id", self.verification_receipt_id),
            ("authority_snapshot_id", self.authority_snapshot_id),
        ):
            _require_durable_id(name, value)
        _require_sha256("scope_snapshot_sha256", self.scope_snapshot_sha256)
        for name, value in (
            ("alias_prefix", self.alias_prefix),
            ("artifact_alias", self.artifact_alias),
            ("producer_node_alias", self.producer_node_alias),
        ):
            _require_local_key(name, value)
        if len(self.alias_prefix) > MAX_ALIAS_PREFIX_CHARACTERS:
            raise ValueError(
                f"alias_prefix must be at most {MAX_ALIAS_PREFIX_CHARACTERS} characters"
            )
        if not isinstance(self.producer_auxiliary_node, AuxiliaryNodeSubject):
            raise ValueError("producer_auxiliary_node must be an AuxiliaryNodeSubject")
        if (
            self.producer_auxiliary_node.task_id != self.task_id
            or self.producer_auxiliary_node.auxiliary_graph_id
            != self.auxiliary_graph_id
        ):
            raise ValueError("producer AuxiliaryNode must belong to the bound Task and graph")
        _require_canonical_values(
            "affected_obligations",
            self.affected_obligations,
            minimum=1,
            maximum=64,
        )


@dataclass(frozen=True, slots=True)
class FrozenPlanningContextPrimitiveInvocation:
    """恰好一个 Host 资源感知调用的 I/O 前权威封存。

    该封存在有限资源读取越过外部边界前构建。它将完整逻辑请求绑定到选定节点，
    以及授权该派发的每个乐观并发版本。因此，持久化适配器可先预留调用，之后拒绝依据已变化的图、
    目标、预算或权威快照生成的结算。
    """

    primitive_kind: PlanningContextPrimitiveKind
    binding: FrozenPlanningContextArtifactBinding
    invocation_turn_id: str
    expected_task_state_version: int
    expected_node_state_version: int
    expected_control_state_version: int
    expected_goal_state_version: int
    expected_revision_state_version: int
    expected_budget_state_version: int
    authority_snapshot_sha256: str
    structure_sha256: str
    budget_snapshot_sha256: str
    logical_request_json: str
    logical_request_sha256: str
    state_guard_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.primitive_kind, PlanningContextPrimitiveKind):
            raise ValueError("primitive_kind must be a PlanningContextPrimitiveKind")
        if not isinstance(self.binding, FrozenPlanningContextArtifactBinding):
            raise ValueError("binding must be a FrozenPlanningContextArtifactBinding")
        _require_durable_id("invocation_turn_id", self.invocation_turn_id)
        for name, value in (
            ("expected_task_state_version", self.expected_task_state_version),
            ("expected_node_state_version", self.expected_node_state_version),
            ("expected_control_state_version", self.expected_control_state_version),
            ("expected_goal_state_version", self.expected_goal_state_version),
            ("expected_revision_state_version", self.expected_revision_state_version),
            ("expected_budget_state_version", self.expected_budget_state_version),
        ):
            _require_positive_int(name, value)
        for name, value in (
            ("authority_snapshot_sha256", self.authority_snapshot_sha256),
            ("structure_sha256", self.structure_sha256),
            ("budget_snapshot_sha256", self.budget_snapshot_sha256),
            ("state_guard_sha256", self.state_guard_sha256),
        ):
            _require_sha256(name, value)
        _require_canonical_json_hash(
            "logical request", self.logical_request_json, self.logical_request_sha256
        )
        logical_request = json.loads(self.logical_request_json)
        if logical_request.get("schema_version") != (
            "planning-resource-perception-request-v1"
        ):
            raise ValueError("primitive kind differs from the sealed logical request")
        if logical_request.get("binding") != _binding_payload(self.binding):
            raise ValueError("sealed logical request differs from its artifact binding")
        expected_guard = _planning_context_primitive_state_guard_sha256(
            primitive_kind=self.primitive_kind,
            binding=self.binding,
            invocation_turn_id=self.invocation_turn_id,
            expected_task_state_version=self.expected_task_state_version,
            expected_node_state_version=self.expected_node_state_version,
            expected_control_state_version=self.expected_control_state_version,
            expected_goal_state_version=self.expected_goal_state_version,
            expected_revision_state_version=self.expected_revision_state_version,
            expected_budget_state_version=self.expected_budget_state_version,
            authority_snapshot_sha256=self.authority_snapshot_sha256,
            structure_sha256=self.structure_sha256,
            budget_snapshot_sha256=self.budget_snapshot_sha256,
            logical_request_sha256=self.logical_request_sha256,
        )
        if self.state_guard_sha256 != expected_guard:
            raise ValueError("planning primitive state guard is invalid")


def freeze_serialized_planning_context_primitive_invocation(
    *,
    primitive_kind: PlanningContextPrimitiveKind,
    binding: FrozenPlanningContextArtifactBinding,
    logical_request_json: str,
    invocation_turn_id: str,
    expected_task_state_version: int,
    expected_node_state_version: int,
    expected_control_state_version: int,
    expected_goal_state_version: int,
    expected_revision_state_version: int,
    expected_budget_state_version: int,
    authority_snapshot_sha256: str,
    structure_sha256: str,
    budget_snapshot_sha256: str,
) -> FrozenPlanningContextPrimitiveInvocation:
    """冻结资源感知适配器发出的规范请求。"""

    logical_request_sha256 = _sha256_text(logical_request_json)
    state_guard_sha256 = _planning_context_primitive_state_guard_sha256(
        primitive_kind=primitive_kind,
        binding=binding,
        invocation_turn_id=invocation_turn_id,
        expected_task_state_version=expected_task_state_version,
        expected_node_state_version=expected_node_state_version,
        expected_control_state_version=expected_control_state_version,
        expected_goal_state_version=expected_goal_state_version,
        expected_revision_state_version=expected_revision_state_version,
        expected_budget_state_version=expected_budget_state_version,
        authority_snapshot_sha256=authority_snapshot_sha256,
        structure_sha256=structure_sha256,
        budget_snapshot_sha256=budget_snapshot_sha256,
        logical_request_sha256=logical_request_sha256,
    )
    return FrozenPlanningContextPrimitiveInvocation(
        primitive_kind=primitive_kind,
        binding=binding,
        invocation_turn_id=invocation_turn_id,
        expected_task_state_version=expected_task_state_version,
        expected_node_state_version=expected_node_state_version,
        expected_control_state_version=expected_control_state_version,
        expected_goal_state_version=expected_goal_state_version,
        expected_revision_state_version=expected_revision_state_version,
        expected_budget_state_version=expected_budget_state_version,
        authority_snapshot_sha256=authority_snapshot_sha256,
        structure_sha256=structure_sha256,
        budget_snapshot_sha256=budget_snapshot_sha256,
        logical_request_json=logical_request_json,
        logical_request_sha256=logical_request_sha256,
        state_guard_sha256=state_guard_sha256,
    )


def _binding_payload(binding: FrozenPlanningContextArtifactBinding) -> dict[str, object]:
    return {
        "session_id": binding.session_id,
        "task_id": binding.task_id,
        "auxiliary_graph_id": binding.auxiliary_graph_id,
        "goal_id": binding.goal_id,
        "producer_auxiliary_node": binding.producer_auxiliary_node.model_dump(mode="json"),
        "primitive_call_id": binding.primitive_call_id,
        "artifact_id": binding.artifact_id,
        "verification_receipt_id": binding.verification_receipt_id,
        "authority_snapshot_id": binding.authority_snapshot_id,
        "scope_snapshot_sha256": binding.scope_snapshot_sha256,
        "alias_prefix": binding.alias_prefix,
        "artifact_alias": binding.artifact_alias,
        "producer_node_alias": binding.producer_node_alias,
        "affected_obligations": list(binding.affected_obligations),
    }


def _planning_context_primitive_state_guard_sha256(
    *,
    primitive_kind: PlanningContextPrimitiveKind,
    binding: FrozenPlanningContextArtifactBinding,
    invocation_turn_id: str,
    expected_task_state_version: int,
    expected_node_state_version: int,
    expected_control_state_version: int,
    expected_goal_state_version: int,
    expected_revision_state_version: int,
    expected_budget_state_version: int,
    authority_snapshot_sha256: str,
    structure_sha256: str,
    budget_snapshot_sha256: str,
    logical_request_sha256: str,
) -> str:
    return _sha256_value(
        {
            "schema_version": "frozen-planning-context-primitive-invocation-v1",
            "primitive_kind": primitive_kind.value,
            "binding": _binding_payload(binding),
            "invocation_turn_id": invocation_turn_id,
            "expected_state_versions": {
                "task": expected_task_state_version,
                "node": expected_node_state_version,
                "control": expected_control_state_version,
                "goal": expected_goal_state_version,
                "revision": expected_revision_state_version,
                "budget": expected_budget_state_version,
            },
            "authority_snapshot_sha256": authority_snapshot_sha256,
            "structure_sha256": structure_sha256,
            "budget_snapshot_sha256": budget_snapshot_sha256,
            "logical_request_sha256": logical_request_sha256,
        }
    )


def _canonical_json(value: object) -> str:
    # 封存只对已解码 JSON 和上方原语绑定载荷执行哈希。让此编解码器保持冷态，
    # 而不导入适配器用于 dataclass、Path 和任意 Pydantic 值的更宽泛重放序列化器。
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_value(value: object) -> str:
    return _sha256_text(_canonical_json(value))


def _require_durable_id(name: str, value: str) -> None:
    if not isinstance(value, str) or _DURABLE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical durable ID")


def _require_local_key(name: str, value: str) -> None:
    if not isinstance(value, str) or _LOCAL_KEY.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical local key")


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_canonical_values(
    name: str,
    values: tuple[str, ...],
    *,
    minimum: int,
    maximum: int,
) -> None:
    if (
        not isinstance(values, tuple)
        or not minimum <= len(values) <= maximum
        or any(
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or "\x00" in value
            or len(value) > 500
            for value in values
        )
        or len(values) != len(set(values))
    ):
        raise ValueError(f"{name} must contain {minimum}..{maximum} unique canonical strings")
    if name == "affected_obligations" and values != tuple(sorted(values)):
        raise ValueError("affected_obligations must use ascending canonical order")


def _require_canonical_json_hash(name: str, payload: str, digest: str) -> None:
    _require_sha256(f"{name}_sha256", digest)
    try:
        decoded = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be valid JSON") from exc
    if payload != _canonical_json(decoded):
        raise ValueError(f"{name} JSON must use canonical encoding")
    if _sha256_text(payload) != digest:
        raise ValueError(f"{name} hash does not match its JSON")


__all__ = [
    'FrozenPlanningContextArtifactBinding',
    'FrozenPlanningContextPrimitiveInvocation',
    'PlanningContextPrimitiveKind',
    "freeze_serialized_planning_context_primitive_invocation",
]
