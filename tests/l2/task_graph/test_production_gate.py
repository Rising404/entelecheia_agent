from __future__ import annotations

import pytest
from pydantic import ValidationError

from personagraph.l2.task_graph import (
    InSessionTaskGraphRevisionProposal,
    InSessionTaskGraphRevisionValidationContext,
    InSessionTaskSourceAnchor,
)
from personagraph.l2.task_graph.production_gate import (
    TaskGraphAcceptanceRef,
    TaskGraphNodeExecutionRequirement,
    TaskGraphPlanningGap,
    TaskGraphProductionCapabilityContext,
    TaskGraphProductionEvaluationContext,
    TaskGraphProductionFailureCode,
    TaskGraphProductionFreshness,
    evaluate_task_graph_production,
)


_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64
_HASH_D = "d" * 64


def _validation_context(
    *,
    expected_revision: int | None = None,
    authorization_ids: tuple[str, ...] = ("goal",),
    required_ids: tuple[str, ...] = ("goal", "evidence"),
) -> InSessionTaskGraphRevisionValidationContext:
    return InSessionTaskGraphRevisionValidationContext(
        session_id="session_eval",
        source_turn_id="turn_current",
        target_insession_task_id="task_eval",
        expected_current_graph_revision=expected_revision,
        source_anchors=(
            InSessionTaskSourceAnchor(
                anchor_id="goal",
                source_turn_id="turn_current",
                source_kind="current_user_instruction",
                start=0,
                end=12,
                excerpt="deliver work",
            ),
            InSessionTaskSourceAnchor(
                anchor_id="goal_two",
                source_turn_id="turn_current",
                source_kind="current_user_instruction",
                start=13,
                end=24,
                excerpt="second goal",
            ),
            InSessionTaskSourceAnchor(
                anchor_id="evidence",
                source_turn_id="document_v3",
                source_kind="retrieved_document",
                start=0,
                end=15,
                excerpt="document fact",
            ),
        ),
        authorization_anchor_ids=authorization_ids,
        required_anchor_ids=required_ids,
    )


def _node(
    node_key: str,
    *,
    parent: str | None,
    title: str | None = None,
    objective: str | None = None,
    source_ids: tuple[str, ...] = ("goal", "evidence"),
    acceptance_id: str | None = None,
    criterion: str | None = None,
) -> dict[str, object]:
    return {
        "node_key": node_key,
        "node_kind": "root" if parent is None else "subtask",
        "parent_node_key": parent,
        "title": title or f"Node {node_key}",
        "objective": objective or f"Produce the {node_key} deliverable",
        "source_anchor_ids": list(source_ids),
        "acceptance_criteria": [
            {
                "acceptance_id": acceptance_id or f"done_{node_key}",
                "criterion": criterion or f"The {node_key} deliverable is verified",
                "source_anchor_ids": list(source_ids),
            }
        ],
    }


def _proposal(*nodes: dict[str, object]) -> InSessionTaskGraphRevisionProposal:
    return InSessionTaskGraphRevisionProposal.model_validate(
        {
            "root": {
                "root_key": "root",
                "nodes": list(nodes or (_node("root", parent=None),)),
            }
        }
    )


def _context(
    proposal: InSessionTaskGraphRevisionProposal,
    *,
    validation_context: InSessionTaskGraphRevisionValidationContext | None = None,
    observed_revision: int | None = None,
    authority_hashes: tuple[str, str] = (_HASH_A, _HASH_A),
    source_hashes: tuple[str, str] = (_HASH_B, _HASH_B),
    catalog_hashes: tuple[str, str] = (_HASH_C, _HASH_C),
    available_capabilities: tuple[str, ...] = ("model.write",),
    satisfiable_effects: tuple[str, ...] = ("local:read",),
    requirements: tuple[TaskGraphNodeExecutionRequirement, ...] | None = None,
    gaps: tuple[TaskGraphPlanningGap, ...] = (),
) -> TaskGraphProductionEvaluationContext:
    resolved_requirements = requirements
    if resolved_requirements is None:
        resolved_requirements = tuple(
            TaskGraphNodeExecutionRequirement(
                node_key=node.node_key,
                required_capability_ids=("model.write",),
                required_effect_ids=("local:read",),
            )
            for node in proposal.root.nodes
        )
    return TaskGraphProductionEvaluationContext(
        validation_context=validation_context or _validation_context(),
        freshness=TaskGraphProductionFreshness(
            observed_current_graph_revision=observed_revision,
            expected_authority_snapshot_sha256=authority_hashes[0],
            observed_authority_snapshot_sha256=authority_hashes[1],
            expected_source_manifest_sha256=source_hashes[0],
            observed_source_manifest_sha256=source_hashes[1],
            expected_capability_catalog_sha256=catalog_hashes[0],
            observed_capability_catalog_sha256=catalog_hashes[1],
        ),
        capabilities=TaskGraphProductionCapabilityContext(
            available_capability_ids=available_capabilities,
            satisfiable_effect_ids=satisfiable_effects,
            node_requirements=resolved_requirements,
        ),
        gaps=gaps,
    )


def test_adaptive_gate_accepts_a_single_node_without_complexity_padding() -> None:
    proposal = _proposal(_node("root", parent=None))
    context = _context(proposal)

    first = evaluate_task_graph_production(proposal, context=context)
    second = evaluate_task_graph_production(proposal, context=context)

    assert first.passed is True
    assert first.failure_codes == ()
    assert first.semantic_review_required is True
    assert first.telemetry.total_node_count == 1
    assert first.telemetry.max_depth == 1
    assert first.telemetry.root_child_count == 0
    assert first.evaluation_sha256 == second.evaluation_sha256
    assert first == second
    assert hash(first) == hash(second)

    changed_proposal = _proposal(
        _node("root", parent=None, objective="A different valid deliverable")
    )
    changed = evaluate_task_graph_production(
        changed_proposal,
        context=_context(changed_proposal),
    )
    assert changed.passed is True
    assert changed.telemetry == first.telemetry
    assert changed.proposal_sha256 != first.proposal_sha256
    assert changed.evaluation_sha256 != first.evaluation_sha256

    tampered = first.model_dump(mode="json")
    tampered["proposal_sha256"] = _HASH_D
    with pytest.raises(ValidationError, match="does not bind the result payload"):
        type(first).model_validate(tampered)


def test_required_goals_and_sources_need_node_and_acceptance_coverage() -> None:
    proposal = _proposal(
        _node(
            "root",
            parent=None,
            source_ids=("goal",),
        )
    )
    validation_context = _validation_context(
        authorization_ids=("goal", "goal_two"),
        required_ids=("goal", "goal_two", "evidence"),
    )

    result = evaluate_task_graph_production(
        proposal,
        context=_context(proposal, validation_context=validation_context),
    )

    assert result.passed is False
    assert set(result.failure_codes) >= {
        TaskGraphProductionFailureCode.CANONICAL_VALIDATION_REJECTED,
        TaskGraphProductionFailureCode.REQUIRED_GOAL_UNCOVERED,
        TaskGraphProductionFailureCode.REQUIRED_SOURCE_UNCOVERED,
    }
    assert result.telemetry.required_goal_count == 2
    assert result.telemetry.required_goal_node_coverage_count == 1
    assert result.telemetry.required_source_count == 1
    assert result.telemetry.required_source_node_coverage_count == 0


def test_canonical_structure_failures_remain_hard_failures() -> None:
    proposal = _proposal(
        _node("root", parent=None),
        _node("node_a", parent="node_b"),
        _node("node_b", parent="node_a"),
    )

    result = evaluate_task_graph_production(
        proposal,
        context=_context(proposal),
    )

    assert result.passed is False
    assert set(result.failure_codes) >= {
        TaskGraphProductionFailureCode.CYCLE_DETECTED,
        TaskGraphProductionFailureCode.ORPHAN_NODE,
    }
    assert result.telemetry.reachable_node_count == 1
    assert result.telemetry.unique_node_count == 3


def test_capability_effect_and_requirement_coverage_fail_closed() -> None:
    proposal = _proposal(
        _node("root", parent=None),
        _node("child", parent="root"),
    )
    requirements = (
        TaskGraphNodeExecutionRequirement(
            node_key="root",
            required_capability_ids=("missing.capability",),
            required_effect_ids=("network:transmit",),
        ),
        TaskGraphNodeExecutionRequirement(node_key="ghost"),
    )

    result = evaluate_task_graph_production(
        proposal,
        context=_context(
            proposal,
            requirements=requirements,
            available_capabilities=(),
            satisfiable_effects=(),
        ),
    )

    assert set(result.failure_codes) >= {
        TaskGraphProductionFailureCode.NODE_EXECUTION_REQUIREMENT_MISSING,
        TaskGraphProductionFailureCode.UNKNOWN_EXECUTION_REQUIREMENT_NODE,
        TaskGraphProductionFailureCode.CAPABILITY_UNAVAILABLE,
        TaskGraphProductionFailureCode.EFFECT_UNSATISFIABLE,
    }
    assert result.telemetry.missing_execution_requirement_count == 1
    assert result.telemetry.unknown_execution_requirement_count == 1
    assert result.telemetry.unavailable_capability_count == 1
    assert result.telemetry.unsatisfiable_effect_count == 1


@pytest.mark.parametrize(
    ("contract", "kwargs"),
    (
        (
            TaskGraphNodeExecutionRequirement,
            {"node_key": "root", "required_capability_ids": (" bad id ",)},
        ),
        (
            TaskGraphNodeExecutionRequirement,
            {"node_key": "root", "required_effect_ids": ("bad effect",)},
        ),
        (
            TaskGraphProductionCapabilityContext,
            {"available_capability_ids": (" bad id ",)},
        ),
        (
            TaskGraphProductionCapabilityContext,
            {"satisfiable_effect_ids": ("bad effect",)},
        ),
    ),
)
def test_capability_and_effect_ids_must_be_canonical_runtime_ids(
    contract: type[object],
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="canonical runtime IDs"):
        contract(**kwargs)  # type: ignore[operator]

    with pytest.raises(ValidationError, match="canonical order"):
        contract(
            **(
                {"node_key": "root"} if contract is TaskGraphNodeExecutionRequirement else {}
            ),
            **(
                {"required_capability_ids": ("z.cap", "a.cap")}
                if contract is TaskGraphNodeExecutionRequirement
                else {"available_capability_ids": ("z.cap", "a.cap")}
            ),
        )  # type: ignore[operator]


def test_freshness_and_gap_closure_are_bound_into_the_result() -> None:
    proposal = _proposal(_node("root", parent=None))
    validation_context = _validation_context(expected_revision=3)
    gaps = (
        TaskGraphPlanningGap(
            gap_id="blocking_gap",
            blocking=True,
            affected_required_anchor_ids=("evidence",),
        ),
        TaskGraphPlanningGap(
            gap_id="unmapped_gap",
            blocking=False,
            affected_required_anchor_ids=("evidence",),
        ),
        TaskGraphPlanningGap(
            gap_id="mapped_gap",
            blocking=False,
            affected_required_anchor_ids=("evidence",),
            mapped_node_keys=("root",),
            mapped_acceptances=(
                TaskGraphAcceptanceRef(
                    node_key="root",
                    acceptance_id="done_root",
                ),
            ),
        ),
    )

    result = evaluate_task_graph_production(
        proposal,
        context=_context(
            proposal,
            validation_context=validation_context,
            observed_revision=2,
            authority_hashes=(_HASH_A, _HASH_D),
            source_hashes=(_HASH_B, _HASH_D),
            catalog_hashes=(_HASH_C, _HASH_D),
            gaps=gaps,
        ),
    )

    assert set(result.failure_codes) >= {
        TaskGraphProductionFailureCode.BASE_AUTHORITY_STALE,
        TaskGraphProductionFailureCode.CURRENT_AUTHORITY_STALE,
        TaskGraphProductionFailureCode.SOURCE_MANIFEST_STALE,
        TaskGraphProductionFailureCode.CAPABILITY_CATALOG_STALE,
        TaskGraphProductionFailureCode.BLOCKING_GAP,
        TaskGraphProductionFailureCode.NON_BLOCKING_GAP_UNMAPPED,
    }
    assert result.telemetry.blocking_gap_count == 1
    assert result.telemetry.non_blocking_gap_count == 2
    assert result.telemetry.unmapped_non_blocking_gap_count == 1


def test_exact_node_and_acceptance_duplicates_are_hard_failures() -> None:
    duplicate_one = _node(
        "first",
        parent="root",
        title="Same title",
        objective="Same objective",
        acceptance_id="first_done",
        criterion="Same criterion",
    )
    duplicate_one["acceptance_criteria"].append(
        {
            "acceptance_id": "first_done_again",
            "criterion": "SAME criterion",
            "source_anchor_ids": ["goal", "evidence"],
        }
    )
    duplicate_two = _node(
        "second",
        parent="root",
        title="same TITLE",
        objective="same OBJECTIVE",
        acceptance_id="second_done",
        criterion="same criterion",
    )
    # 使用不同的本地验收 ID，同时让第二个节点的有效契约与第一个保持一致。
    duplicate_two["acceptance_criteria"].append(
        {
            "acceptance_id": "second_done_again",
            "criterion": "Same criterion",
            "source_anchor_ids": ["goal", "evidence"],
        }
    )
    proposal = _proposal(
        _node("root", parent=None),
        duplicate_one,
        duplicate_two,
    )

    result = evaluate_task_graph_production(
        proposal,
        context=_context(proposal),
    )

    assert set(result.failure_codes) >= {
        TaskGraphProductionFailureCode.DUPLICATE_ACCEPTANCE_CRITERION,
        TaskGraphProductionFailureCode.EXACT_CONTRACT_DUPLICATE,
    }
    assert result.telemetry.duplicate_acceptance_criterion_count == 2
    assert result.telemetry.exact_duplicate_node_count == 1


@pytest.mark.parametrize("field", ("title", "objective"))
def test_task_graph_node_contract_text_cannot_be_blank_or_padded(field: str) -> None:
    node = _node("root", parent=None)
    node[field] = " "
    with pytest.raises(ValidationError, match="canonical text"):
        _proposal(node)

    node = _node("root", parent=None)
    node["acceptance_criteria"][0]["criterion"] = " padded criterion "
    with pytest.raises(ValidationError, match="canonical text"):
        _proposal(node)
