"""L2 TaskGraph 中一个已接受 Turn 的纯、按源排序执行 lane authority。

Entry 分类器可以按任意数组顺序返回 Task 匹配。本模块将已接受 Guard 结果与 Store 分配的
新根 Task ID 转换为规范 manifest。它不执行持久化，也不选择 WorkRun、Attempt、TaskNode
或恢复动作。

同一持久根 Task 的多个匹配组成一条 lane。不同根 Task 必须具有不相交源 span，因为不经
另一次语义模型决定，重叠 span 无法证明它们的相对执行顺序。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .task_matching import (
    AcceptedInSessionTaskMatch,
    InSessionTaskMatchGuardResult,
    InSessionTaskMatchedSourceSpan,
)


_MANIFEST_SCHEMA_VERSION = "insession-task-execution-lane-manifest-v1"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InSessionTaskLaneManifestBuildCode(StrEnum):
    """拒绝派生 lane authority 的稳定保守失败原因。"""

    GUARD_NOT_ACCEPTED = "guard_not_accepted"
    INVALID_ACCEPTED_MATCH = "invalid_accepted_match"
    CREATED_TASK_MAPPING_MISMATCH = "created_task_mapping_mismatch"
    CREATED_TASK_ID_COLLISION = "created_task_id_collision"
    CROSS_TASK_SOURCE_OVERLAP = "cross_task_source_overlap"


class InSessionTaskLaneManifestBuildError(ValueError):
    """一批已接受匹配无法形成无歧义 lane manifest。"""

    def __init__(
        self,
        code: InSessionTaskLaneManifestBuildCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(message)


class InSessionTaskExecutionLaneMatch(_Contract):
    """一项已接受匹配中由其 lane 保留的源绑定部分。"""

    match_type: Literal[
        "new_root",
        "existing_root",
        "existing_root_branch",
        "existing_root_target_change",
    ]
    source_span: InSessionTaskMatchedSourceSpan
    execution_requested: bool
    replacement_objective: str | None = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def _new_roots_always_request_execution(
        self,
    ) -> 'InSessionTaskExecutionLaneMatch':
        if self.match_type == "new_root" and not self.execution_requested:
            raise ValueError("a new-root lane match must request execution")
        if self.match_type == "existing_root_target_change":
            if not self.execution_requested:
                raise ValueError("a target-change lane match must request execution")
            if (
                self.replacement_objective is None
                or not self.replacement_objective.strip()
                or self.replacement_objective != self.replacement_objective.strip()
            ):
                raise ValueError(
                    "a target-change lane match requires a canonical replacement objective"
                )
        elif self.replacement_objective is not None:
            raise ValueError(
                "only a target-change lane match may retain a replacement objective"
            )
        return self


class InSessionTaskExecutionLane(_Contract):
    """按规范源顺序排列的一条持久根 Task lane。"""

    ordinal: int = Field(ge=0, le=23)
    insession_task_id: str = Field(min_length=1, max_length=128)
    matches: tuple[InSessionTaskExecutionLaneMatch, ...] = Field(
        min_length=1,
        max_length=24,
    )
    execution_requested: bool

    @model_validator(mode="after")
    def _validate_canonical_lane(self) -> 'InSessionTaskExecutionLane':
        if not self.insession_task_id.strip():
            raise ValueError("lane Task ID must not be blank")
        if self.matches != tuple(sorted(self.matches, key=_lane_match_sort_key)):
            raise ValueError("lane matches must use canonical source order")
        identities = {
            (
                item.match_type,
                item.source_span.start,
                item.source_span.end,
                item.source_span.text_sha256,
            )
            for item in self.matches
        }
        if len(identities) != len(self.matches):
            raise ValueError("lane matches must not contain duplicate source bindings")
        new_root_count = sum(item.match_type == "new_root" for item in self.matches)
        if new_root_count and (new_root_count != 1 or len(self.matches) != 1):
            raise ValueError("a new-root lane cannot merge with another Task match")
        target_change_count = sum(
            item.match_type == "existing_root_target_change"
            for item in self.matches
        )
        if target_change_count > 1:
            raise ValueError(
                "a Task lane cannot contain competing target-change intents"
            )
        expected_execution = any(item.execution_requested for item in self.matches)
        if self.execution_requested != expected_execution:
            raise ValueError("lane execution request must be the OR of its matches")
        return self

    @property
    def source_start(self) -> int:
        return self.matches[0].source_span.start

    @property
    def source_end(self) -> int:
        return max(item.source_span.end for item in self.matches)


class InSessionTaskExecutionLaneManifest(_Contract):
    """从一批 Guard 结果派生的可自认证规范 lane 计划。"""

    schema_version: Literal[
        "insession-task-execution-lane-manifest-v1"
    ] = _MANIFEST_SCHEMA_VERSION
    lanes: tuple[InSessionTaskExecutionLane, ...] = Field(
        default=(),
        max_length=24,
    )
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_manifest(self) -> 'InSessionTaskExecutionLaneManifest':
        if tuple(item.ordinal for item in self.lanes) != tuple(range(len(self.lanes))):
            raise ValueError("lane ordinals must be contiguous from zero")
        task_ids = tuple(item.insession_task_id for item in self.lanes)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("lane Task IDs must be unique")
        if self.lanes != tuple(sorted(self.lanes, key=_lane_sort_key)):
            raise ValueError("lanes must use canonical source order")
        if _first_cross_task_overlap(self.lanes) is not None:
            raise ValueError("different Task lanes cannot have overlapping source spans")
        expected_hash = canonical_insession_task_lane_manifest_sha256(self.lanes)
        if self.manifest_sha256 != expected_hash:
            raise ValueError("lane manifest hash does not match its canonical payload")
        return self


def build_insession_task_execution_lane_manifest(
    guard_result: InSessionTaskMatchGuardResult,
    *,
    created_insession_task_ids_by_local_key: Mapping[str, str],
) -> InSessionTaskExecutionLaneManifest:
    """根据已接受匹配与 Store 分配 ID 构建规范 lane。

    ``execute_current`` 仍只是请求 Task 范围控制的不可信值。在此持久化可实现精确重放；
    它不选择或授权任何更低层执行身份。
    """

    if guard_result.status != "accepted":
        raise InSessionTaskLaneManifestBuildError(
            InSessionTaskLaneManifestBuildCode.GUARD_NOT_ACCEPTED,
            "only an accepted Task-match Guard result can form execution lanes",
        )
    accepted = guard_result.accepted_task_matches
    _validate_created_task_mapping(
        accepted,
        created_insession_task_ids_by_local_key,
    )

    grouped: dict[str, list[InSessionTaskExecutionLaneMatch]] = {}
    new_root_task_ids: set[str] = set()
    existing_root_task_ids: set[str] = set()
    for item in accepted:
        _validate_accepted_match_binding(item)
        proposal = item.proposal
        if proposal.match_type == "new_root":
            task_id = created_insession_task_ids_by_local_key[proposal.local_key]
            new_root_task_ids.add(task_id)
            execution_requested = True
        else:
            task_id = proposal.insession_task_id
            existing_root_task_ids.add(task_id)
            execution_requested = bool(proposal.execute_current)
        if not isinstance(task_id, str) or not task_id.strip() or len(task_id) > 128:
            raise InSessionTaskLaneManifestBuildError(
                InSessionTaskLaneManifestBuildCode.INVALID_ACCEPTED_MATCH,
                "accepted Task match resolved to an invalid durable Task ID",
            )
        grouped.setdefault(task_id, []).append(
            InSessionTaskExecutionLaneMatch(
                match_type=proposal.match_type,
                source_span=item.source_span,
                execution_requested=execution_requested,
                replacement_objective=(
                    proposal.replacement_objective.strip()
                    if proposal.match_type == "existing_root_target_change"
                    else None
                ),
            )
        )

    if new_root_task_ids & existing_root_task_ids:
        raise InSessionTaskLaneManifestBuildError(
            InSessionTaskLaneManifestBuildCode.CREATED_TASK_ID_COLLISION,
            "a newly allocated Task ID collides with an existing-root match",
        )

    unordered_lanes = [
        InSessionTaskExecutionLane(
            ordinal=0,
            insession_task_id=task_id,
            matches=tuple(sorted(matches, key=_lane_match_sort_key)),
            execution_requested=any(item.execution_requested for item in matches),
        )
        for task_id, matches in grouped.items()
    ]
    ordered_lanes = tuple(sorted(unordered_lanes, key=_lane_sort_key))
    overlap = _first_cross_task_overlap(ordered_lanes)
    if overlap is not None:
        left_task_id, right_task_id = overlap
        raise InSessionTaskLaneManifestBuildError(
            InSessionTaskLaneManifestBuildCode.CROSS_TASK_SOURCE_OVERLAP,
            "different Task lanes have overlapping source spans: "
            f"{left_task_id}, {right_task_id}",
        )
    canonical_lanes = tuple(
        lane.model_copy(update={"ordinal": ordinal})
        for ordinal, lane in enumerate(ordered_lanes)
    )
    return InSessionTaskExecutionLaneManifest(
        lanes=canonical_lanes,
        manifest_sha256=canonical_insession_task_lane_manifest_sha256(
            canonical_lanes
        ),
    )


def canonical_insession_task_lane_manifest_sha256(
    lanes: tuple[InSessionTaskExecutionLane, ...],
) -> str:
    """为与 lane manifest receipt 一同存储的精确 JSON payload 计算哈希。"""

    payload = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
    # 新 match discriminator 添加的可选字段在缺失时不能重写历史 lane 身份。目标变更目标仍
    # 包含在内，因为它们是非空 authority。
        "lanes": [
            item.model_dump(mode="json", exclude_none=True) for item in lanes
        ],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_created_task_mapping(
    accepted: tuple[AcceptedInSessionTaskMatch, ...],
    mapping: Mapping[str, str],
) -> None:
    expected_keys = tuple(
        item.proposal.local_key
        for item in accepted
        if item.proposal.match_type == "new_root"
    )
    if len(expected_keys) != len(set(expected_keys)):
        raise InSessionTaskLaneManifestBuildError(
            InSessionTaskLaneManifestBuildCode.INVALID_ACCEPTED_MATCH,
            "accepted new-root matches contain duplicate local keys",
        )
    if not all(isinstance(key, str) for key in mapping) or set(mapping) != set(
        expected_keys
    ):
        raise InSessionTaskLaneManifestBuildError(
            InSessionTaskLaneManifestBuildCode.CREATED_TASK_MAPPING_MISMATCH,
            "created Task mapping must exactly cover accepted new-root local keys",
        )
    values = tuple(mapping[key] for key in expected_keys)
    if any(
        not isinstance(value, str) or not value.strip() or len(value) > 128
        for value in values
    ):
        raise InSessionTaskLaneManifestBuildError(
            InSessionTaskLaneManifestBuildCode.CREATED_TASK_MAPPING_MISMATCH,
            "created Task mapping contains an invalid durable Task ID",
        )
    if len(values) != len(set(values)):
        raise InSessionTaskLaneManifestBuildError(
            InSessionTaskLaneManifestBuildCode.CREATED_TASK_ID_COLLISION,
            "different new-root local keys resolved to the same durable Task ID",
        )


def _validate_accepted_match_binding(item: AcceptedInSessionTaskMatch) -> None:
    excerpt = item.proposal.source_excerpt
    span = item.source_span
    if (
        span.end - span.start != len(excerpt)
        or span.text_sha256
        != hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
    ):
        raise InSessionTaskLaneManifestBuildError(
            InSessionTaskLaneManifestBuildCode.INVALID_ACCEPTED_MATCH,
            "accepted Task match source binding is inconsistent with its excerpt",
        )


def _lane_match_sort_key(
    item: InSessionTaskExecutionLaneMatch,
) -> tuple[int, int, str, str, bool, str]:
    return (
        item.source_span.start,
        item.source_span.end,
        item.match_type,
        item.source_span.text_sha256,
        item.execution_requested,
        item.replacement_objective or "",
    )


def _lane_sort_key(item: InSessionTaskExecutionLane) -> tuple[int, int, str]:
    return (item.source_start, item.source_end, item.insession_task_id)


def _first_cross_task_overlap(
    lanes: tuple[InSessionTaskExecutionLane, ...]
    | list[InSessionTaskExecutionLane],
) -> tuple[str, str] | None:
    for left_index, left in enumerate(lanes):
        for right in lanes[left_index + 1 :]:
            for left_match in left.matches:
                for right_match in right.matches:
                    left_span = left_match.source_span
                    right_span = right_match.source_span
                    if (
                        left_span.start < right_span.end
                        and right_span.start < left_span.end
                    ):
                        return left.insession_task_id, right.insession_task_id
    return None


__all__ = [
    'InSessionTaskExecutionLaneManifest',
    'InSessionTaskExecutionLaneMatch',
    'InSessionTaskExecutionLane',
    "InSessionTaskLaneManifestBuildCode",
    "InSessionTaskLaneManifestBuildError",
    "build_insession_task_execution_lane_manifest",
    "canonical_insession_task_lane_manifest_sha256",
]
