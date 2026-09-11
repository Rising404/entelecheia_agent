from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from personagraph.l2.task_graph.contracts import (
    InSessionTaskCatalogItem,
    InSessionTaskCatalog,
)
from personagraph.l2.task_graph.lane_manifest import (
    InSessionTaskExecutionLaneManifest,
    InSessionTaskLaneManifestBuildCode,
    InSessionTaskLaneManifestBuildError,
    build_insession_task_execution_lane_manifest,
    canonical_insession_task_lane_manifest_sha256,
)
from personagraph.l2.task_graph.task_matching import (
    InSessionTaskMatchesProposal,
    guard_insession_task_matches,
)


def _catalog(*task_ids: str) -> InSessionTaskCatalog:
    return InSessionTaskCatalog(
        items=tuple(
            InSessionTaskCatalogItem(
                insession_task_id=task_id,
                goal_summary=f"{task_id} objective",
                status="active",
                current_graph_revision=1,
            )
            for task_id in task_ids
        )
    )


def _guard(
    user_text: str,
    *matches: dict[str, object],
    catalog: InSessionTaskCatalog | None = None,
):
    result = guard_insession_task_matches(
        InSessionTaskMatchesProposal.model_validate(
            {"task_matches": list(matches)}
        ),
        authoritative_user_text=user_text,
        trusted_root_catalog=catalog or InSessionTaskCatalog(),
    )
    assert result.status == "accepted"
    return result


def test_builds_lanes_in_source_order_not_classifier_array_order() -> None:
    guarded = _guard(
        "先整理产品方案；再整理发布计划",
        {
            "match_type": "new_root",
            "local_key": "release",
            "title": "发布计划",
            "objective": "整理发布计划",
            "source_excerpt": "整理发布计划",
        },
        {
            "match_type": "new_root",
            "local_key": "product",
            "title": "产品方案",
            "objective": "整理产品方案",
            "source_excerpt": "整理产品方案",
        },
    )

    manifest = build_insession_task_execution_lane_manifest(
        guarded,
        created_insession_task_ids_by_local_key={
            "release": "task_release",
            "product": "task_product",
        },
    )

    assert [lane.insession_task_id for lane in manifest.lanes] == [
        "task_product",
        "task_release",
    ]
    assert [lane.ordinal for lane in manifest.lanes] == [0, 1]
    assert all(lane.execution_requested for lane in manifest.lanes)


def test_merges_same_existing_task_and_ors_execute_current() -> None:
    guarded = _guard(
        "补充论文局限，并继续论文摘要",
        {
            "match_type": "existing_root",
            "insession_task_id": "task_paper",
            "source_excerpt": "继续论文摘要",
            "execute_current": True,
        },
        {
            "match_type": "existing_root_branch",
            "insession_task_id": "task_paper",
            "branch_key": "limitations",
            "branch_summary": "补充论文局限",
            "source_excerpt": "补充论文局限",
            "execute_current": False,
        },
        catalog=_catalog("task_paper"),
    )

    manifest = build_insession_task_execution_lane_manifest(
        guarded,
        created_insession_task_ids_by_local_key={},
    )

    assert len(manifest.lanes) == 1
    lane = manifest.lanes[0]
    assert lane.insession_task_id == "task_paper"
    assert lane.execution_requested is True
    assert [item.match_type for item in lane.matches] == [
        "existing_root_branch",
        "existing_root",
    ]
    assert [item.execution_requested for item in lane.matches] == [False, True]


def test_target_change_lane_retains_replacement_and_exact_source_authority() -> None:
    user_text = "请把论文任务的目标改成只比较实验设计与消融结果"
    excerpt = "目标改成只比较实验设计与消融结果"
    guarded = _guard(
        user_text,
        {
            "match_type": "existing_root_target_change",
            "insession_task_id": "task_paper",
            "replacement_objective": "比较论文的实验设计与消融结果",
            "source_excerpt": excerpt,
            "execute_current": True,
        },
        catalog=_catalog("task_paper"),
    )

    manifest = build_insession_task_execution_lane_manifest(
        guarded,
        created_insession_task_ids_by_local_key={},
    )

    match = manifest.lanes[0].matches[0]
    assert match.match_type == "existing_root_target_change"
    assert match.replacement_objective == "比较论文的实验设计与消融结果"
    assert match.execution_requested is True
    assert user_text[match.source_span.start : match.source_span.end] == excerpt


def test_lane_rejects_competing_target_changes_for_one_task() -> None:
    guarded = _guard(
        "先改成目标甲，再改成目标乙",
        {
            "match_type": "existing_root_target_change",
            "insession_task_id": "task_paper",
            "replacement_objective": "目标甲",
            "source_excerpt": "改成目标甲",
            "execute_current": True,
        },
        {
            "match_type": "existing_root_target_change",
            "insession_task_id": "task_paper",
            "replacement_objective": "目标乙",
            "source_excerpt": "改成目标乙",
            "execute_current": True,
        },
        catalog=_catalog("task_paper"),
    )

    with pytest.raises(ValidationError, match="target-change"):
        build_insession_task_execution_lane_manifest(
            guarded,
            created_insession_task_ids_by_local_key={},
        )


def test_canonical_hash_is_independent_of_accepted_match_array_order() -> None:
    user_text = "先任务甲，再任务乙"
    first = {
        "match_type": "new_root",
        "local_key": "alpha",
        "title": "甲",
        "objective": "完成甲",
        "source_excerpt": "任务甲",
    }
    second = {
        "match_type": "new_root",
        "local_key": "beta",
        "title": "乙",
        "objective": "完成乙",
        "source_excerpt": "任务乙",
    }
    mapping = {"alpha": "task_alpha", "beta": "task_beta"}

    forward = build_insession_task_execution_lane_manifest(
        _guard(user_text, first, second),
        created_insession_task_ids_by_local_key=mapping,
    )
    reversed_manifest = build_insession_task_execution_lane_manifest(
        _guard(user_text, second, first),
        created_insession_task_ids_by_local_key=mapping,
    )

    assert forward == reversed_manifest
    assert forward.manifest_sha256 == canonical_insession_task_lane_manifest_sha256(
        forward.lanes
    )


def test_cross_task_source_overlap_fails_closed() -> None:
    guarded = _guard(
        "abcdef",
        {
            "match_type": "new_root",
            "local_key": "left",
            "title": "左",
            "objective": "处理左侧",
            "source_excerpt": "abcde",
        },
        {
            "match_type": "new_root",
            "local_key": "right",
            "title": "右",
            "objective": "处理右侧",
            "source_excerpt": "cdef",
        },
    )

    with pytest.raises(InSessionTaskLaneManifestBuildError) as caught:
        build_insession_task_execution_lane_manifest(
            guarded,
            created_insession_task_ids_by_local_key={
                "left": "task_left",
                "right": "task_right",
            },
        )

    assert caught.value.code is (
        InSessionTaskLaneManifestBuildCode.CROSS_TASK_SOURCE_OVERLAP
    )


@pytest.mark.parametrize(
    "mapping, code",
    [
        ({}, InSessionTaskLaneManifestBuildCode.CREATED_TASK_MAPPING_MISMATCH),
        (
            {"new_task": "task_new", "extra": "task_extra"},
            InSessionTaskLaneManifestBuildCode.CREATED_TASK_MAPPING_MISMATCH,
        ),
        (
            {"new_task": "   "},
            InSessionTaskLaneManifestBuildCode.CREATED_TASK_MAPPING_MISMATCH,
        ),
    ],
)
def test_created_task_mapping_must_exactly_cover_new_roots(
    mapping: dict[str, str],
    code: InSessionTaskLaneManifestBuildCode,
) -> None:
    guarded = _guard(
        "创建任务",
        {
            "match_type": "new_root",
            "local_key": "new_task",
            "title": "任务",
            "objective": "创建任务",
            "source_excerpt": "创建任务",
        },
    )

    with pytest.raises(InSessionTaskLaneManifestBuildError) as caught:
        build_insession_task_execution_lane_manifest(
            guarded,
            created_insession_task_ids_by_local_key=mapping,
        )

    assert caught.value.code is code


def test_created_task_ids_must_be_unique_and_not_collide_with_existing_roots() -> None:
    guarded = _guard(
        "新建甲和新建乙",
        {
            "match_type": "new_root",
            "local_key": "alpha",
            "title": "甲",
            "objective": "新建甲",
            "source_excerpt": "新建甲",
        },
        {
            "match_type": "new_root",
            "local_key": "beta",
            "title": "乙",
            "objective": "新建乙",
            "source_excerpt": "新建乙",
        },
    )
    with pytest.raises(InSessionTaskLaneManifestBuildError) as duplicate:
        build_insession_task_execution_lane_manifest(
            guarded,
            created_insession_task_ids_by_local_key={
                "alpha": "task_same",
                "beta": "task_same",
            },
        )
    assert duplicate.value.code is (
        InSessionTaskLaneManifestBuildCode.CREATED_TASK_ID_COLLISION
    )

    colliding = _guard(
        "新建甲并继续旧任务",
        {
            "match_type": "new_root",
            "local_key": "alpha",
            "title": "甲",
            "objective": "新建甲",
            "source_excerpt": "新建甲",
        },
        {
            "match_type": "existing_root",
            "insession_task_id": "task_existing",
            "source_excerpt": "继续旧任务",
            "execute_current": True,
        },
        catalog=_catalog("task_existing"),
    )
    with pytest.raises(InSessionTaskLaneManifestBuildError) as existing:
        build_insession_task_execution_lane_manifest(
            colliding,
            created_insession_task_ids_by_local_key={
                "alpha": "task_existing",
            },
        )
    assert existing.value.code is (
        InSessionTaskLaneManifestBuildCode.CREATED_TASK_ID_COLLISION
    )


def test_rejected_guard_result_cannot_form_manifest() -> None:
    proposal = InSessionTaskMatchesProposal.model_validate(
        {
            "task_matches": [
                {
                    "match_type": "existing_root",
                    "insession_task_id": "unknown",
                    "source_excerpt": "继续任务",
                }
            ]
        }
    )
    rejected = guard_insession_task_matches(
        proposal,
        authoritative_user_text="继续任务",
        trusted_root_catalog=InSessionTaskCatalog(),
    )
    assert rejected.status == "rejected"

    with pytest.raises(InSessionTaskLaneManifestBuildError) as caught:
        build_insession_task_execution_lane_manifest(
            rejected,
            created_insession_task_ids_by_local_key={},
        )

    assert caught.value.code is InSessionTaskLaneManifestBuildCode.GUARD_NOT_ACCEPTED


def test_manifest_dto_rejects_hash_and_canonical_order_tampering() -> None:
    manifest = build_insession_task_execution_lane_manifest(
        _guard(
            "任务甲；任务乙",
            {
                "match_type": "new_root",
                "local_key": "alpha",
                "title": "甲",
                "objective": "任务甲",
                "source_excerpt": "任务甲",
            },
            {
                "match_type": "new_root",
                "local_key": "beta",
                "title": "乙",
                "objective": "任务乙",
                "source_excerpt": "任务乙",
            },
        ),
        created_insession_task_ids_by_local_key={
            "alpha": "task_alpha",
            "beta": "task_beta",
        },
    )
    payload = manifest.model_dump(mode="json")

    bad_hash = deepcopy(payload)
    bad_hash["manifest_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="hash does not match"):
        InSessionTaskExecutionLaneManifest.model_validate(bad_hash)

    bad_order = deepcopy(payload)
    bad_order["lanes"] = list(reversed(bad_order["lanes"]))
    with pytest.raises(ValidationError, match="ordinals|source order"):
        InSessionTaskExecutionLaneManifest.model_validate(bad_order)


def test_empty_accepted_batch_has_a_stable_manifest() -> None:
    manifest = build_insession_task_execution_lane_manifest(
        _guard("普通闲聊"),
        created_insession_task_ids_by_local_key={},
    )

    assert manifest.lanes == ()
    assert len(manifest.manifest_sha256) == 64
    assert manifest.manifest_sha256 == canonical_insession_task_lane_manifest_sha256(
        ()
    )
