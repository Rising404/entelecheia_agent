"""一个当前 TaskGraph 节点的失败关闭来源投影。

TaskGraph 持久层保存修订版本范围的来源清单，每个节点则保存获准使用的锚点 ID。面向模型的
Attempt 与验证器端口不得接收完整任务清单：本模块连接这两项权威，只投影当前节点的锚点，
并以确定性的 Host 哈希封存精确瞬态投影。

该哈希是精确模型输入的完整性绑定，不能替代 Store 来源检查。调用方必须根据新的
``InSessionTaskDetails`` 投影构建此值；接受模型或调用方编写的来源卡会跨越 TaskGraph
权威边界。
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Never

from pydantic import BaseModel, ConfigDict, Field, model_validator

from personagraph.l2.task_graph.contracts import (
    InSessionTaskAcceptanceProposal,
    InSessionTaskDetails,
    InSessionTaskSourceAnchor,
)
from personagraph.l2.work_run import TaskNodeSubject


_SHA256_PATTERN = r"^[0-9a-f]{64}$"

TaskNodeSourceContextFailureReason = Literal[
    "task_authority_mismatch",
    "stale_graph_authority",
    "stale_node_authority",
    "ambiguous_node_authority",
    "missing_source_anchor",
    "unknown_source_anchor",
    "acceptance_source_overreach",
    "corrupt_source_manifest",
]


class TaskNodeSourceContextAuthorityError(RuntimeError):
    """Host 无法构造一个精确的节点范围来源投影。"""

    code = "task_node_source_context_unavailable"

    def __init__(self, reason: TaskNodeSourceContextFailureReason) -> None:
        self.reason = reason
        super().__init__(f"TaskNode source context is unavailable: {reason}")


class _SourceContextContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskNodeSourceAnchorProjection(_SourceContextContract):
    """由当前节点锚点 ID 选择的一段精确来源区间。"""

    anchor_id: str = Field(min_length=1, max_length=200)
    source_turn_id: str = Field(min_length=1, max_length=200)
    source_kind: Literal[
        "current_user_instruction",
        "current_user_context",
        "previously_authorized_task_state",
        "quoted_external",
        "attachment",
        "retrieved_document",
        "tool_observation",
        "memory",
        "gap",
    ]
    gap_blocking: bool | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    excerpt: str = Field(min_length=1, max_length=4_000)
    excerpt_sha256: str = Field(pattern=_SHA256_PATTERN)
    authorization: bool
    required: bool

    @model_validator(mode="after")
    def _validate_exact_excerpt_hash(self) -> 'TaskNodeSourceAnchorProjection':
        if self.end <= self.start:
            raise ValueError("source anchor end must exceed start")
        if self.excerpt_sha256 != _sha256_text(self.excerpt):
            raise ValueError("source anchor excerpt hash does not match exact text")
        if (self.source_kind == "gap") != (self.gap_blocking is not None):
            raise ValueError(
                "gap source projections alone require a blocking disposition"
            )
        return self


class TaskNodeAcceptanceSourceBinding(_SourceContextContract):
    """来自不可变节点契约的精确 Acceptance 到锚点映射。"""

    acceptance_id: str = Field(min_length=1, max_length=200)
    source_anchor_ids: tuple[str, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def _require_unique_anchor_ids(self) -> 'TaskNodeAcceptanceSourceBinding':
        if len(self.source_anchor_ids) != len(set(self.source_anchor_ids)):
            raise ValueError("Acceptance source anchor IDs must be unique")
        return self


class TaskNodeSourceContext(_SourceContextContract):
    """绑定到一个不可变 TaskNode 主体的冻结精确来源上下文。"""

    schema_version: Literal["task-node-source-context-v1"] = (
        "task-node-source-context-v1"
    )
    session_id: str = Field(min_length=1, max_length=200)
    task_id: str = Field(min_length=1, max_length=200)
    graph_revision: int = Field(ge=1)
    node_id: str = Field(min_length=1, max_length=200)
    node_revision: int = Field(ge=1)
    node_source_anchor_ids: tuple[str, ...] = Field(min_length=1, max_length=32)
    acceptance_source_bindings: tuple[TaskNodeAcceptanceSourceBinding, ...] = (
        Field(min_length=1, max_length=64)
    )
    anchors: tuple[TaskNodeSourceAnchorProjection, ...] = Field(
        min_length=1,
        max_length=32,
    )
    authority_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_exact_authority_binding(self) -> 'TaskNodeSourceContext':
        self._require_self_integrity()
        return self

    def _require_self_integrity(self) -> None:
        if len(self.node_source_anchor_ids) != len(set(self.node_source_anchor_ids)):
            raise ValueError("node source anchor IDs must be unique")
        projected_ids = tuple(item.anchor_id for item in self.anchors)
        if projected_ids != self.node_source_anchor_ids:
            raise ValueError(
                "source context anchors must exactly follow node source anchor IDs"
            )
        acceptance_ids = tuple(
            item.acceptance_id for item in self.acceptance_source_bindings
        )
        if len(acceptance_ids) != len(set(acceptance_ids)):
            raise ValueError("source context Acceptance IDs must be unique")
        allowed = frozenset(self.node_source_anchor_ids)
        if any(
            not set(item.source_anchor_ids).issubset(allowed)
            for item in self.acceptance_source_bindings
        ):
            raise ValueError("source context Acceptance authority exceeds the node")
        expected = _source_context_sha256(
            task_id=self.task_id,
            session_id=self.session_id,
            graph_revision=self.graph_revision,
            node_id=self.node_id,
            node_revision=self.node_revision,
            node_source_anchor_ids=self.node_source_anchor_ids,
            acceptance_source_bindings=self.acceptance_source_bindings,
            anchors=self.anchors,
        )
        if self.authority_sha256 != expected:
            raise ValueError("source context authority hash does not match exact projection")

    def require_exact_binding(
        self,
        *,
        session_id: str,
        subject: TaskNodeSubject,
        acceptances: tuple[InSessionTaskAcceptanceProposal, ...],
    ) -> None:
        """拒绝重新定向到另一主体或 Acceptance 集合的上下文。"""

        self._require_self_integrity()
        if (
            self.session_id != session_id
            or self.task_id != subject.task_id
            or self.graph_revision != subject.graph_revision
            or self.node_id != subject.node_id
            or self.node_revision != subject.node_revision
        ):
            raise ValueError("source context does not match the current TaskNode subject")
        actual_bindings = tuple(
            TaskNodeAcceptanceSourceBinding(
                acceptance_id=item.acceptance_id,
                source_anchor_ids=item.source_anchor_ids,
            )
            for item in acceptances
        )
        if actual_bindings != self.acceptance_source_bindings:
            raise ValueError(
                "source context does not match the current Acceptance authority"
            )


def build_task_node_source_context(
    *,
    session_id: str,
    details: InSessionTaskDetails,
    subject: TaskNodeSubject,
) -> TaskNodeSourceContext:
    """连接当前 Store 权威，并且只返回此节点的来源区间。

    未知、缺失、越权、有歧义或过期的绑定都会在模型提供商可被调用前失败。锚点顺序遵循
    不可变节点声明，而不是任务范围来源清单。
    """

    if (
        details.session_id != session_id
        or details.insession_task_id != subject.task_id
    ):
        _fail("task_authority_mismatch")
    if details.current_graph_revision != subject.graph_revision:
        _fail("stale_graph_authority")

    same_node = tuple(
        item
        for item in details.nodes
        if item.get("insession_task_node_id") == subject.node_id
    )
    exact_node = tuple(
        item
        for item in same_node
        if item.get("node_revision") == subject.node_revision
    )
    if not exact_node:
        _fail("stale_node_authority")
    if len(exact_node) != 1 or len(same_node) != 1:
        _fail("ambiguous_node_authority")
    node = exact_node[0]

    raw_node_anchor_ids = node.get("source_anchor_ids")
    if not isinstance(raw_node_anchor_ids, (list, tuple)):
        _fail("missing_source_anchor")
    node_anchor_ids = tuple(raw_node_anchor_ids)
    if (
        not node_anchor_ids
        or any(not isinstance(item, str) or not item for item in node_anchor_ids)
        or len(node_anchor_ids) != len(set(node_anchor_ids))
    ):
        _fail("missing_source_anchor")

    try:
        acceptances = tuple(
            InSessionTaskAcceptanceProposal.model_validate(item)
            for item in node.get("acceptance_criteria", ())
        )
    except (TypeError, ValueError):
        _fail("corrupt_source_manifest")
    if not acceptances:
        _fail("corrupt_source_manifest")
    anchors_by_id: dict[str, InSessionTaskSourceAnchor] = {}
    for anchor in details.source_anchors:
        if anchor.anchor_id in anchors_by_id:
            _fail("corrupt_source_manifest")
        anchors_by_id[anchor.anchor_id] = anchor
    authorization_ids = tuple(details.authorization_anchor_ids)
    required_ids = tuple(details.required_anchor_ids)
    if (
        len(authorization_ids) != len(set(authorization_ids))
        or len(required_ids) != len(set(required_ids))
        or not set(authorization_ids).issubset(anchors_by_id)
        or not set(required_ids).issubset(anchors_by_id)
    ):
        _fail("corrupt_source_manifest")
    allowed = frozenset(node_anchor_ids)
    unknown = allowed.difference(anchors_by_id)
    if unknown:
        _fail("unknown_source_anchor")
    if any(
        not set(acceptance.source_anchor_ids).issubset(allowed)
        for acceptance in acceptances
    ):
        _fail("acceptance_source_overreach")

    projected = tuple(
        _project_anchor(
            anchors_by_id[anchor_id],
            authorization=anchor_id in authorization_ids,
            required=anchor_id in required_ids,
        )
        for anchor_id in node_anchor_ids
    )
    acceptance_bindings = tuple(
        TaskNodeAcceptanceSourceBinding(
            acceptance_id=item.acceptance_id,
            source_anchor_ids=item.source_anchor_ids,
        )
        for item in acceptances
    )
    authority_sha256 = _source_context_sha256(
        task_id=subject.task_id,
        session_id=session_id,
        graph_revision=subject.graph_revision,
        node_id=subject.node_id,
        node_revision=subject.node_revision,
        node_source_anchor_ids=node_anchor_ids,
        acceptance_source_bindings=acceptance_bindings,
        anchors=projected,
    )
    return TaskNodeSourceContext(
        session_id=session_id,
        task_id=subject.task_id,
        graph_revision=subject.graph_revision,
        node_id=subject.node_id,
        node_revision=subject.node_revision,
        node_source_anchor_ids=node_anchor_ids,
        acceptance_source_bindings=acceptance_bindings,
        anchors=projected,
        authority_sha256=authority_sha256,
    )


def _project_anchor(
    anchor: InSessionTaskSourceAnchor,
    *,
    authorization: bool,
    required: bool,
) -> TaskNodeSourceAnchorProjection:
    return TaskNodeSourceAnchorProjection(
        anchor_id=anchor.anchor_id,
        source_turn_id=anchor.source_turn_id,
        source_kind=anchor.source_kind,
        gap_blocking=anchor.gap_blocking,
        start=anchor.start,
        end=anchor.end,
        excerpt=anchor.excerpt,
        excerpt_sha256=_sha256_text(anchor.excerpt),
        authorization=authorization,
        required=required,
    )


def _source_context_sha256(
    *,
    session_id: str,
    task_id: str,
    graph_revision: int,
    node_id: str,
    node_revision: int,
    node_source_anchor_ids: tuple[str, ...],
    acceptance_source_bindings: tuple[TaskNodeAcceptanceSourceBinding, ...],
    anchors: tuple[TaskNodeSourceAnchorProjection, ...],
) -> str:
    payload = {
        "schema_version": "task-node-source-context-v1",
        "session_id": session_id,
        "task_id": task_id,
        "graph_revision": graph_revision,
        "node_id": node_id,
        "node_revision": node_revision,
        "node_source_anchor_ids": list(node_source_anchor_ids),
        "acceptance_source_bindings": [
            item.model_dump(mode="json") for item in acceptance_source_bindings
        ],
        "anchors": [item.model_dump(mode="json") for item in anchors],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(serialized)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fail(reason: TaskNodeSourceContextFailureReason) -> Never:
    raise TaskNodeSourceContextAuthorityError(reason)
