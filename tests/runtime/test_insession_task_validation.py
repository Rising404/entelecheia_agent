from __future__ import annotations

import pytest
from pydantic import ValidationError

from personagraph.l2.task_graph.contracts import (
    InSessionTaskGraphLimits,
    InSessionTaskGraphValidationCode,
    InSessionTaskGraphValidationContext,
    InSessionTaskNodeKind,
    InSessionTaskSourceAnchor,
    NewInSessionTaskGraphsProposal,
)
from personagraph.l2.task_graph.validation import (
    validate_new_insession_task_graphs,
)


def _context(*, required_anchor_ids: tuple[str, ...] = ("request",)) -> InSessionTaskGraphValidationContext:
    return InSessionTaskGraphValidationContext(
        session_id="session_alpha",
        source_turn_id="turn_alpha",
        source_anchors=(
            InSessionTaskSourceAnchor(
                anchor_id="request",
                source_turn_id="turn_alpha",
                source_kind="current_user_instruction",
                start=0,
                end=3,
                excerpt="请分析",
            ),
        ),
        authorization_anchor_ids=("request",),
        required_anchor_ids=required_anchor_ids,
        limits=InSessionTaskGraphLimits(
            max_root_tasks=3,
            max_nodes_per_task=8,
            max_depth=4,
        ),
    )


def _node(
    key: str,
    *,
    parent: str | None = None,
    acceptance: tuple[dict[str, object], ...] | None = None,
) -> dict[str, object]:
    return {
        "node_key": key,
        "node_kind": "root" if parent is None else "subtask",
        "parent_node_key": parent,
        "title": f"{key} 标题",
        "objective": f"完成 {key} 的目标",
        "source_anchor_ids": ["request"],
        "acceptance_criteria": acceptance
        or [
            {
                "acceptance_id": f"{key}_acceptance",
                "criterion": f"核对 {key} 产出",
                "source_anchor_ids": ["request"],
            }
        ],
    }


def _proposal(*roots: dict[str, object]) -> NewInSessionTaskGraphsProposal:
    return NewInSessionTaskGraphsProposal.model_validate(
        {
            "source_turn_id": "turn_alpha",
            "roots": list(roots),
        }
    )


def _root(root_key: str, *nodes: dict[str, object]) -> dict[str, object]:
    return {"root_key": root_key, "nodes": list(nodes)}


def test_validates_a_source_bound_tree_and_acceptance_coverage():
    proposal = _proposal(
        _root(
            "research",
            _node("research"),
            _node("method", parent="research"),
        )
    )

    result = validate_new_insession_task_graphs(proposal, context=_context())

    assert result.status == "accepted"
    assert result.proposal == proposal
    assert result.trusted_context == _context()
    assert result.error_codes == ()


def test_graph_create_wire_contract_is_v2_and_rejects_v1_discriminator():
    payload = {
        "schema_version": "insession-task-graph-create-v1",
        "source_turn_id": "turn_alpha",
        "roots": [_root("research", _node("research"))],
    }

    with pytest.raises(ValidationError):
        NewInSessionTaskGraphsProposal.model_validate(payload)

    payload["schema_version"] = "insession-task-graph-create-v2"
    proposal = NewInSessionTaskGraphsProposal.model_validate(payload)
    assert proposal.schema_version == "insession-task-graph-create-v2"


def test_rejects_an_unmapped_explicit_anchor_even_when_tree_is_valid():
    proposal = _proposal(_root("research", _node("research")))

    result = validate_new_insession_task_graphs(
        proposal,
        context=_context(required_anchor_ids=("request", "missing")),
    )

    assert result.status == "rejected"
    assert InSessionTaskGraphValidationCode.UNMAPPED_REQUIRED_ANCHOR in result.error_codes


def test_rejects_a_proposal_bound_to_a_different_source_turn():
    proposal = _proposal(_root("research", _node("research"))).model_copy(
        update={"source_turn_id": "turn_other"}
    )

    result = validate_new_insession_task_graphs(proposal, context=_context())

    assert result.status == "rejected"
    assert InSessionTaskGraphValidationCode.SOURCE_TURN_MISMATCH in result.error_codes


def test_rejects_unknown_acceptance_source_anchor():
    root = _node(
        "research",
        acceptance=(
            {
                "acceptance_id": "research_acceptance",
                "criterion": "完整",
                "source_anchor_ids": ["unknown"],
            },
        ),
    )

    result = validate_new_insession_task_graphs(
        _proposal(_root("research", root)),
        context=_context(),
    )

    assert result.status == "rejected"
    assert InSessionTaskGraphValidationCode.UNKNOWN_SOURCE_ANCHOR in result.error_codes
    assert (
        InSessionTaskGraphValidationCode.ACCEPTANCE_WITHOUT_AUTHORIZED_SOURCE
        in result.error_codes
    )
    assert InSessionTaskGraphValidationCode.UNMAPPED_REQUIRED_ANCHOR in result.error_codes


def test_rejects_an_acceptance_with_only_retrieved_material_as_its_source():
    context = InSessionTaskGraphValidationContext(
        session_id="session_alpha",
        source_turn_id="turn_alpha",
        source_anchors=(
            InSessionTaskSourceAnchor(
                anchor_id="request",
                source_turn_id="turn_alpha",
                source_kind="current_user_instruction",
                start=0,
                end=3,
                excerpt="请分析",
            ),
            InSessionTaskSourceAnchor(
                anchor_id="retrieved",
                source_turn_id="turn_alpha",
                source_kind="retrieved_document",
                start=0,
                end=4,
                excerpt="论文内容",
            ),
        ),
        authorization_anchor_ids=("request",),
        required_anchor_ids=("request",),
    )
    root = _node(
        "research",
        acceptance=(
            {
                "acceptance_id": "research_acceptance",
                "criterion": "完整",
                "source_anchor_ids": ["retrieved"],
            },
        ),
    )

    result = validate_new_insession_task_graphs(
        _proposal(_root("research", root)), context=context
    )

    assert result.status == "rejected"
    assert (
        InSessionTaskGraphValidationCode.ACCEPTANCE_WITHOUT_AUTHORIZED_SOURCE
        in result.error_codes
    )


def test_rejects_retrieved_material_when_host_marks_it_as_task_authorization():
    context = InSessionTaskGraphValidationContext(
        session_id="session_alpha",
        source_turn_id="turn_alpha",
        source_anchors=(
            InSessionTaskSourceAnchor(
                anchor_id="retrieved",
                source_turn_id="turn_alpha",
                source_kind="retrieved_document",
                start=0,
                end=4,
                excerpt="论文内容",
            ),
        ),
        authorization_anchor_ids=("retrieved",),
    )
    root = _node("research")
    root["source_anchor_ids"] = ["retrieved"]
    root["acceptance_criteria"] = [
        {
            "acceptance_id": "research_acceptance",
            "criterion": "分析论文",
            "source_anchor_ids": ["retrieved"],
        }
    ]

    result = validate_new_insession_task_graphs(
        _proposal(_root("research", root)), context=context
    )

    assert result.status == "rejected"
    assert InSessionTaskGraphValidationCode.UNAUTHORIZED_AUTHORIZATION_SOURCE in result.error_codes


def test_rejects_node_scope_or_acceptance_without_an_authorized_source():
    context = InSessionTaskGraphValidationContext(
        session_id="session_alpha",
        source_turn_id="turn_alpha",
        source_anchors=(
            InSessionTaskSourceAnchor(
                anchor_id="request",
                source_turn_id="turn_alpha",
                source_kind="current_user_instruction",
                start=0,
                end=3,
                excerpt="请分析",
            ),
            InSessionTaskSourceAnchor(
                anchor_id="retrieved",
                source_turn_id="turn_alpha",
                source_kind="retrieved_document",
                start=0,
                end=4,
                excerpt="论文内容",
            ),
        ),
        authorization_anchor_ids=("request",),
        required_anchor_ids=("request",),
    )
    node = _node(
        "research",
        acceptance=(
            {
                "acceptance_id": "research_acceptance",
                "criterion": "额外材料必须全部完成",
                "source_anchor_ids": ["retrieved"],
            },
        ),
    )
    node["source_anchor_ids"] = ["retrieved"]

    result = validate_new_insession_task_graphs(
        _proposal(_root("research", node)), context=context
    )

    assert result.status == "rejected"
    assert InSessionTaskGraphValidationCode.NODE_WITHOUT_AUTHORIZED_SOURCE in result.error_codes
    assert InSessionTaskGraphValidationCode.ACCEPTANCE_WITHOUT_AUTHORIZED_SOURCE in result.error_codes


def test_rejects_untyped_model_authored_constraints_until_they_have_source_anchors():
    node = _node("research")
    node["constraints"] = ["必须额外交付一个模型没有从用户处获得的报告"]

    result = validate_new_insession_task_graphs(
        _proposal(_root("research", node)), context=_context()
    )

    assert result.status == "rejected"
    assert InSessionTaskGraphValidationCode.UNSOURCED_CONSTRAINT in result.error_codes


def test_node_objective_source_does_not_replace_acceptance_coverage():
    context = InSessionTaskGraphValidationContext(
        session_id="session_alpha",
        source_turn_id="turn_alpha",
        source_anchors=(
            InSessionTaskSourceAnchor(
                anchor_id="first_demand",
                source_turn_id="turn_alpha",
                source_kind="current_user_instruction",
                start=0,
                end=3,
                excerpt="请分析",
            ),
            InSessionTaskSourceAnchor(
                anchor_id="second_demand",
                source_turn_id="turn_alpha",
                source_kind="current_user_context",
                start=3,
                end=6,
                excerpt="并回答",
            ),
        ),
        authorization_anchor_ids=("first_demand", "second_demand"),
        required_anchor_ids=("first_demand", "second_demand"),
    )
    node = _node(
        "research",
        acceptance=(
            {
                "acceptance_id": "answer_complete",
                "criterion": "回答完成",
                "source_anchor_ids": ["second_demand"],
            },
        ),
    )
    node["source_anchor_ids"] = ["first_demand"]

    result = validate_new_insession_task_graphs(
        _proposal(_root("research", node)), context=context
    )

    assert result.status == "rejected"
    assert InSessionTaskGraphValidationCode.UNMAPPED_REQUIRED_ANCHOR in result.error_codes


def test_acceptance_sources_cover_all_explicit_user_demands():
    context = InSessionTaskGraphValidationContext(
        session_id="session_alpha",
        source_turn_id="turn_alpha",
        source_anchors=(
            InSessionTaskSourceAnchor(
                anchor_id="first_demand",
                source_turn_id="turn_alpha",
                source_kind="current_user_instruction",
                start=0,
                end=3,
                excerpt="请分析",
            ),
            InSessionTaskSourceAnchor(
                anchor_id="second_demand",
                source_turn_id="turn_alpha",
                source_kind="current_user_context",
                start=3,
                end=6,
                excerpt="并回答",
            ),
        ),
        authorization_anchor_ids=("first_demand", "second_demand"),
        required_anchor_ids=("first_demand", "second_demand"),
    )
    node = _node(
        "research",
        acceptance=(
            {
                "acceptance_id": "both_complete",
                "criterion": "分析与回答均完成",
                "source_anchor_ids": ["first_demand", "second_demand"],
            },
        ),
    )
    node["source_anchor_ids"] = ["first_demand"]

    result = validate_new_insession_task_graphs(
        _proposal(_root("research", node)), context=context
    )

    assert result.status == "accepted"
    assert result.error_codes == ()


def test_rejects_duplicate_acceptance_ids_within_one_node():
    acceptance = {
        "acceptance_id": "complete",
        "criterion": "产出完整",
        "source_anchor_ids": ["request"],
    }
    node = _node("research", acceptance=(acceptance, acceptance))

    result = validate_new_insession_task_graphs(
        _proposal(_root("research", node)), context=_context()
    )

    assert result.status == "rejected"
    assert InSessionTaskGraphValidationCode.DUPLICATE_ACCEPTANCE_ID in result.error_codes


def test_acceptance_ids_are_local_to_their_owning_node():
    acceptance = {
        "acceptance_id": "complete",
        "criterion": "产出完整",
        "source_anchor_ids": ["request"],
    }
    proposal = _proposal(
        _root(
            "research",
            _node("research", acceptance=(acceptance,)),
            _node("method", parent="research", acceptance=(acceptance,)),
        )
    )

    result = validate_new_insession_task_graphs(proposal, context=_context())

    assert result.status == "accepted"


def test_contract_requires_at_least_one_acceptance():
    without_acceptance = _node("research")
    without_acceptance["acceptance_criteria"] = []

    with pytest.raises(ValidationError):
        _proposal(_root("research", without_acceptance))


def test_contract_rejects_retired_obligation_and_acceptance_fields():
    with_retired_fields = _node("research")
    with_retired_fields["obligations"] = [
        {
            "obligation_key": "retired",
            "summary": "旧合同",
            "source_anchor_ids": ["request"],
        }
    ]
    with_retired_fields["acceptance_criteria"] = [
        {
            "acceptance_key": "retired",
            "criterion": "旧合同",
            "obligation_keys": ["retired"],
            "source_anchor_ids": ["request"],
        }
    ]

    with pytest.raises(ValidationError):
        _proposal(_root("research", with_retired_fields))


def test_graph_creation_contract_rejects_runtime_evidence_refs():
    with_runtime_evidence = _node("research")
    with_runtime_evidence["acceptance_criteria"] = [
        {
            "acceptance_id": "complete",
            "criterion": "产出完整",
            "source_anchor_ids": ["request"],
            "evidence_refs": ["runtime_evidence"],
        }
    ]

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _proposal(_root("research", with_runtime_evidence))


def test_rejects_cycles_or_orphaned_nodes_without_creating_an_implicit_dag():
    proposal = _proposal(
        _root(
            "root",
            _node("root"),
            _node("first", parent="second"),
            _node("second", parent="first"),
        )
    )

    result = validate_new_insession_task_graphs(proposal, context=_context())

    assert result.status == "rejected"
    assert InSessionTaskGraphValidationCode.CYCLE_DETECTED in result.error_codes
    assert InSessionTaskGraphValidationCode.ORPHAN_NODE in result.error_codes


def test_rejects_more_roots_than_the_host_limit_without_semantic_reclassification():
    proposal = _proposal(
        _root("one", _node("one")),
        _root("two", _node("two")),
        _root("three", _node("three")),
        _root("four", _node("four")),
    )

    result = validate_new_insession_task_graphs(proposal, context=_context())

    assert result.status == "rejected"
    assert InSessionTaskGraphValidationCode.ROOT_LIMIT_EXCEEDED in result.error_codes


def test_root_and_subtask_contracts_reject_wrong_parent_shape_before_host_validation():
    root_payload = _node("root")
    root_payload["parent_node_key"] = "parent"
    root_payload["node_kind"] = InSessionTaskNodeKind.ROOT.value

    try:
        NewInSessionTaskGraphsProposal.model_validate(
            {"source_turn_id": "turn_alpha", "roots": [_root("root", root_payload)]}
        )
    except ValueError as exc:
        assert "root node cannot declare a parent" in str(exc)
    else:  # pragma: no cover - 保护测试预期的防御分支
        raise AssertionError("invalid root shape unexpectedly parsed")
