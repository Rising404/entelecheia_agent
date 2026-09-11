from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from personagraph.l2.task_graph import (
    InSessionTaskDetails,
    InSessionTaskGraphRevisionLineageHints,
    InSessionTaskGraphRevisionProposal,
    TaskGraphNodeLineageDisposition,
    TaskGraphRevisionTransitionCode,
    TaskGraphRevisionTransitionError,
    plan_insession_task_graph_revision_transition,
)
from personagraph.l2.task_graph.contracts import InSessionTaskStatus


def _acceptance(acceptance_id: str, criterion: str) -> dict[str, object]:
    return {
        "acceptance_id": acceptance_id,
        "criterion": criterion,
        "source_anchor_ids": ["goal"],
    }


def _base() -> InSessionTaskDetails:
    return InSessionTaskDetails(
        insession_task_id="task_1",
        session_id="session_1",
        task_state_version=4,
        title="交付系统",
        objective="实现并验收系统",
        current_graph_revision=3,
        status=InSessionTaskStatus.ACTIVE,
        nodes=(
            {
                "insession_task_node_id": "node_root",
                "node_revision": 1,
                "node_kind": "root",
                "ordinal": 0,
                "parent_insession_task_node_id": None,
                "title": "交付系统",
                "objective": "实现并验收系统",
                "status": "completed",
                "state_version": 2,
                "source_anchor_ids": ["goal"],
                "acceptance_criteria": [_acceptance("root_done", "系统完成")],
                "constraints": [],
            },
            {
                "insession_task_node_id": "node_build",
                "node_revision": 2,
                "node_kind": "subtask",
                "ordinal": 1,
                "parent_insession_task_node_id": "node_root",
                "title": "实现",
                "objective": "实现核心模块",
                "status": "completed",
                "state_version": 3,
                "source_anchor_ids": ["goal"],
                "acceptance_criteria": [_acceptance("build_done", "核心模块通过测试")],
                "constraints": [],
            },
            {
                "insession_task_node_id": "node_old",
                "node_revision": 1,
                "node_kind": "subtask",
                "ordinal": 2,
                "parent_insession_task_node_id": "node_root",
                "title": "旧调查",
                "objective": "已不再需要",
                "status": "proposed",
                "state_version": 1,
                "source_anchor_ids": ["goal"],
                "acceptance_criteria": [_acceptance("old_done", "旧调查完成")],
                "constraints": [],
            },
        ),
    )


def _proposal(*, revised_build: bool = False) -> InSessionTaskGraphRevisionProposal:
    return InSessionTaskGraphRevisionProposal.model_validate(
        {
            "root": {
                "root_key": "root",
                "nodes": [
                    {
                        "node_key": "root",
                        "node_kind": "root",
                        "title": "交付系统",
                        "objective": "实现并验收系统",
                        "source_anchor_ids": ["goal"],
                        "acceptance_criteria": [
                            _acceptance("root_done", "系统完成")
                        ],
                    },
                    {
                        "node_key": "build",
                        "node_kind": "subtask",
                        "parent_node_key": "root",
                        "title": "实现",
                        "objective": (
                            "实现核心模块并完成性能验收"
                            if revised_build
                            else "实现核心模块"
                        ),
                        "source_anchor_ids": ["goal"],
                        "acceptance_criteria": [
                            _acceptance("build_done", "核心模块通过测试")
                        ],
                    },
                    {
                        "node_key": "release",
                        "node_kind": "subtask",
                        "parent_node_key": "root",
                        "title": "发布",
                        "objective": "完成正式发布",
                        "source_anchor_ids": ["goal"],
                        "acceptance_criteria": [
                            _acceptance("release_done", "发布验收通过")
                        ],
                    },
                ],
            }
        }
    )


def _base_aliases() -> dict[str, str]:
    return {
        "base_root": "node_root",
        "base_build": "node_build",
        "base_old": "node_old",
    }


def _lineage(
    *,
    revise_build: bool = False,
    proposal: InSessionTaskGraphRevisionProposal | None = None,
) -> InSessionTaskGraphRevisionLineageHints:
    bound_proposal = proposal or _proposal(revised_build=revise_build)
    proposal_sha256 = hashlib.sha256(
        json.dumps(
            bound_proposal.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return InSessionTaskGraphRevisionLineageHints.create(
        task_id="task_1",
        expected_base_graph_revision=3,
        task_graph_proposal_sha256=proposal_sha256,
        production_evaluation_sha256="a" * 64,
        semantic_verification_result_sha256s=("b" * 64,),
        hints=(
            {
                "node_key": "root",
                "disposition": "reuse",
                "base_node_alias": "base_root",
            },
            {
                "node_key": "build",
                "disposition": "revise" if revise_build else "reuse",
                "base_node_alias": "base_build",
            },
            {"node_key": "release", "disposition": "new"},
        ),
    )


def _duplicate_node_key_with_valid_hash(raw: dict[str, object]) -> None:
    nodes = raw["nodes"]
    assert isinstance(nodes, list)
    nodes[2]["node_key"] = "build"
    payload = {
        "contract_version": raw["schema_version"],
        "task_id": raw["task_id"],
        "base_graph_revision": raw["base_graph_revision"],
        "target_graph_revision": raw["target_graph_revision"],
        "root_node_id": raw["root_node_id"],
        "nodes": nodes,
        "removed_base_node_ids": raw["removed_base_node_ids"],
    }
    raw["transition_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_transition_reuses_completed_nodes_adds_new_and_removes_old() -> None:
    result = plan_insession_task_graph_revision_transition(
        _base(),
        _proposal(),
        lineage=_lineage(),
        base_node_ids_by_alias=_base_aliases(),
        allocated_node_ids_by_key={"release": "node_release"},
    )

    assert result.base_graph_revision == 3
    assert result.target_graph_revision == 4
    assert result.root_node_id == "node_root"
    assert result.removed_base_node_ids == ("node_old",)
    by_key = {item.node_key: item for item in result.nodes}
    assert by_key["root"].definition_carry_candidate is True
    assert by_key["build"].target_node_revision == 2
    assert by_key["build"].definition_carry_candidate is True
    assert by_key["release"].target_node_revision == 1
    assert by_key["release"].target_parent_node_id == "node_root"
    assert by_key["release"].definition_carry_candidate is False
    assert len(result.transition_sha256) == 64


def test_transition_revises_changed_definition_without_carry() -> None:
    result = plan_insession_task_graph_revision_transition(
        _base(),
        _proposal(revised_build=True),
        lineage=_lineage(revise_build=True),
        base_node_ids_by_alias=_base_aliases(),
        allocated_node_ids_by_key={"release": "node_release"},
    )

    build = next(item for item in result.nodes if item.node_key == "build")
    assert build.disposition is TaskGraphNodeLineageDisposition.REVISE
    assert build.target_node_id == "node_build"
    assert build.base_node_revision == 2
    assert build.target_node_revision == 3
    assert build.definition_carry_candidate is False


def test_transition_host_forces_completed_root_reexecution_without_definition_drift() -> None:
    result = plan_insession_task_graph_revision_transition(
        _base(),
        _proposal(),
        lineage=_lineage(),
        base_node_ids_by_alias=_base_aliases(),
        allocated_node_ids_by_key={"release": "node_release"},
        force_reexecution_base_node_ids=frozenset({"node_root"}),
    )

    root = next(item for item in result.nodes if item.node_key == "root")
    assert root.disposition is TaskGraphNodeLineageDisposition.REVISE
    assert root.target_node_id == "node_root"
    assert root.base_node_revision == 1
    assert root.target_node_revision == 2
    assert root.definition_carry_candidate is False
    assert root.definition.objective == "实现并验收系统"


def test_transition_host_forces_the_requesting_child_without_reexecuting_root() -> None:
    result = plan_insession_task_graph_revision_transition(
        _base(),
        _proposal(),
        lineage=_lineage(),
        base_node_ids_by_alias=_base_aliases(),
        allocated_node_ids_by_key={"release": "node_release"},
        force_reexecution_base_node_ids=frozenset({"node_build"}),
    )

    by_key = {item.node_key: item for item in result.nodes}
    assert by_key["root"].disposition is TaskGraphNodeLineageDisposition.REUSE
    assert by_key["root"].definition_carry_candidate is True
    assert by_key["build"].disposition is TaskGraphNodeLineageDisposition.REVISE
    assert by_key["build"].base_node_revision == 2
    assert by_key["build"].target_node_revision == 3
    assert by_key["build"].definition_carry_candidate is False


def test_transition_rejects_unknown_host_forced_reexecution_node() -> None:
    with pytest.raises(TaskGraphRevisionTransitionError) as captured:
        plan_insession_task_graph_revision_transition(
            _base(),
            _proposal(),
            lineage=_lineage(),
            base_node_ids_by_alias=_base_aliases(),
            allocated_node_ids_by_key={"release": "node_release"},
            force_reexecution_base_node_ids=frozenset({"node_missing"}),
        )

    assert (
        captured.value.code
        is TaskGraphRevisionTransitionCode.LINEAGE_AUTHORITY_MISMATCH
    )


@pytest.mark.parametrize(
    ("proposal", "lineage", "allocations", "expected_code"),
    (
        (
            _proposal(revised_build=True),
            _lineage(
                revise_build=False,
                proposal=_proposal(revised_build=True),
            ),
            {"release": "node_release"},
            TaskGraphRevisionTransitionCode.REUSE_DEFINITION_DRIFT,
        ),
        (
            _proposal(),
            _lineage(revise_build=True, proposal=_proposal()),
            {"release": "node_release"},
            TaskGraphRevisionTransitionCode.REVISION_WITHOUT_CHANGE,
        ),
        (
            _proposal(),
            _lineage(),
            {},
            TaskGraphRevisionTransitionCode.NEW_NODE_ALLOCATION_MISMATCH,
        ),
        (
            _proposal(),
            InSessionTaskGraphRevisionLineageHints.create(
                task_id="task_1",
                expected_base_graph_revision=3,
                task_graph_proposal_sha256=hashlib.sha256(
                    json.dumps(
                        _proposal().model_dump(mode="json"),
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                production_evaluation_sha256="a" * 64,
                semantic_verification_result_sha256s=("b" * 64,),
                hints=(
                    {"node_key": "root", "disposition": "new"},
                    {
                        "node_key": "build",
                        "disposition": "reuse",
                        "base_node_alias": "base_build",
                    },
                    {"node_key": "release", "disposition": "new"},
                ),
            ),
            {"root": "new_root", "release": "node_release"},
            TaskGraphRevisionTransitionCode.ROOT_IDENTITY_MISMATCH,
        ),
    ),
)
def test_transition_rejects_unsafe_lineage(
    proposal: InSessionTaskGraphRevisionProposal,
    lineage: InSessionTaskGraphRevisionLineageHints,
    allocations: dict[str, str],
    expected_code: TaskGraphRevisionTransitionCode,
) -> None:
    with pytest.raises(TaskGraphRevisionTransitionError) as captured:
        plan_insession_task_graph_revision_transition(
            _base(),
            proposal,
            lineage=lineage,
            base_node_ids_by_alias=_base_aliases(),
            allocated_node_ids_by_key=allocations,
        )

    assert captured.value.code is expected_code


def test_transition_hash_is_stable_for_exact_replay() -> None:
    first = plan_insession_task_graph_revision_transition(
        _base(),
        _proposal(),
        lineage=_lineage(),
        base_node_ids_by_alias=_base_aliases(),
        allocated_node_ids_by_key={"release": "node_release"},
    )
    second = plan_insession_task_graph_revision_transition(
        _base(),
        _proposal(),
        lineage=_lineage(),
        base_node_ids_by_alias=_base_aliases(),
        allocated_node_ids_by_key={"release": "node_release"},
    )

    assert first == second
    assert first.transition_sha256 == second.transition_sha256


@pytest.mark.parametrize(
    "mutate",
    (
        lambda raw: raw["nodes"][1].update(
            {"target_node_id": "hijacked_identity", "target_node_revision": 999}
        ),
        lambda raw: raw["removed_base_node_ids"].append("node_root"),
        lambda raw: raw.update({"transition_sha256": "0" * 64}),
        lambda raw: raw["nodes"][1].update({"definition_sha256": "0" * 64}),
        _duplicate_node_key_with_valid_hash,
    ),
)
def test_transition_contract_rejects_tampered_derived_authority(mutate: object) -> None:
    result = plan_insession_task_graph_revision_transition(
        _base(),
        _proposal(),
        lineage=_lineage(),
        base_node_ids_by_alias=_base_aliases(),
        allocated_node_ids_by_key={"release": "node_release"},
    )
    raw = deepcopy(result.model_dump(mode="json"))
    mutate(raw)  # type: ignore[operator]

    with pytest.raises(ValidationError):
        type(result).model_validate(raw)


def test_transition_rejects_corrupt_base_tree_even_when_bad_node_is_removed() -> None:
    raw = _base().model_dump(mode="python")
    nodes = [dict(node) for node in raw["nodes"]]
    nodes[2]["parent_insession_task_node_id"] = "node_old"
    raw["nodes"] = tuple(nodes)
    corrupt = InSessionTaskDetails.model_validate(raw)

    with pytest.raises(TaskGraphRevisionTransitionError) as captured:
        plan_insession_task_graph_revision_transition(
            corrupt,
            _proposal(),
            lineage=_lineage(),
            base_node_ids_by_alias=_base_aliases(),
            allocated_node_ids_by_key={"release": "node_release"},
        )

    assert captured.value.code is TaskGraphRevisionTransitionCode.BASE_GRAPH_CORRUPT


def test_transition_requires_exact_reviewed_proposal_and_host_alias_namespace() -> None:
    with pytest.raises(TaskGraphRevisionTransitionError) as wrong_proposal:
        plan_insession_task_graph_revision_transition(
            _base(),
            _proposal(revised_build=True),
            lineage=_lineage(),
            base_node_ids_by_alias=_base_aliases(),
            allocated_node_ids_by_key={"release": "node_release"},
        )
    assert (
        wrong_proposal.value.code
        is TaskGraphRevisionTransitionCode.LINEAGE_AUTHORITY_MISMATCH
    )

    aliases = _base_aliases()
    aliases["base_build"] = "node_old"
    aliases["base_old"] = "node_build"
    with pytest.raises(TaskGraphRevisionTransitionError) as wrong_aliases:
        plan_insession_task_graph_revision_transition(
            _base(),
            _proposal(),
            lineage=_lineage(),
            base_node_ids_by_alias=aliases,
            allocated_node_ids_by_key={"release": "node_release"},
        )
    assert (
        wrong_aliases.value.code
        is TaskGraphRevisionTransitionCode.REUSE_DEFINITION_DRIFT
    )
