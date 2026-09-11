from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from personagraph.l2.task_graph import (
    TASK_DELIVERY_VALIDATION_CHILD_OUTPUT_MAX_UTF8_BYTES,
    TaskDeliveryValidationChildDeliveryMaterial,
    TaskDeliveryValidationDimension,
    TaskDeliveryValidationDisposition,
    TaskDeliveryValidationFaultDomain,
    TaskDeliveryValidationFinding,
    TaskDeliveryValidationResult,
    TaskDeliveryValidationVerdict,
    build_task_delivery_validation_child_delivery_projection,
    derive_task_delivery_validation_disposition,
    require_root_only_task_delivery_execution_retry,
)


def test_child_delivery_projection_is_sorted_bounded_and_hash_bound() -> None:
    bodies = {
        "child-b": "乙" * 120_000,
        "child-a": "甲" * 120_000,
    }
    projection = build_task_delivery_validation_child_delivery_projection(
        (
            TaskDeliveryValidationChildDeliveryMaterial(
                node_id="child-b",
                node_revision=1,
                ordinal=1,
                delivery_id="delivery-b",
                source_graph_revision=1,
                resolution_kind="direct",
                output_format="markdown",
                output_body=bodies["child-b"],
            ),
            TaskDeliveryValidationChildDeliveryMaterial(
                node_id="child-a",
                node_revision=1,
                # TaskGraph 节点序号采用规范的零基值。因此，位于根节点之前的叶节点
                # 可以合法地拥有序号零，并且必须在整项任务投影中保留下来。
                ordinal=0,
                delivery_id="delivery-a",
                source_graph_revision=1,
                resolution_kind="direct",
                output_format="markdown",
                output_body=bodies["child-a"],
            ),
        )
    )

    assert tuple(item.node_id for item in projection.deliveries) == (
        "child-a",
        "child-b",
    )
    assert projection.included_output_utf8_bytes <= (
        TASK_DELIVERY_VALIDATION_CHILD_OUTPUT_MAX_UTF8_BYTES
    )
    assert projection.truncated_delivery_count == 2
    for item in projection.deliveries:
        assert item.output_body is None
        assert item.output_body_prefix
        assert item.output_body_suffix
        assert item.output_sha256 == hashlib.sha256(
            bodies[item.node_id].encode("utf-8")
        ).hexdigest()
        assert item.excerpt_utf8_bytes == len(
            (item.output_body_prefix + item.output_body_suffix).encode("utf-8")
        )

    tampered = projection.model_dump(mode="json")
    prefix = tampered["deliveries"][0]["output_body_prefix"]
    tampered["deliveries"][0]["output_body_prefix"] = "篡" + prefix[1:]
    with pytest.raises(ValidationError, match="binding hash"):
        type(projection).model_validate(tampered)


def _finding(
    dimension: TaskDeliveryValidationDimension,
    *,
    verdict: TaskDeliveryValidationVerdict = (
        TaskDeliveryValidationVerdict.PASS
    ),
    fault_domain: TaskDeliveryValidationFaultDomain = (
        TaskDeliveryValidationFaultDomain.NONE
    ),
    affected_node_ids: tuple[str, ...] = (),
) -> TaskDeliveryValidationFinding:
    return TaskDeliveryValidationFinding(
        dimension=dimension,
        verdict=verdict,
        fault_domain=fault_domain,
        finding=f"Review for {dimension.value}.",
        affected_node_ids=affected_node_ids,
        evidence_anchor_ids=(),
    )


def _findings(
    *overrides: TaskDeliveryValidationFinding,
) -> tuple[TaskDeliveryValidationFinding, ...]:
    by_dimension = {item.dimension: item for item in overrides}
    return tuple(
        by_dimension.get(dimension, _finding(dimension))
        for dimension in TaskDeliveryValidationDimension
    )


def _result(
    findings: tuple[TaskDeliveryValidationFinding, ...],
) -> TaskDeliveryValidationResult:
    disposition = derive_task_delivery_validation_disposition(findings)
    return TaskDeliveryValidationResult.create(
        verification_result_id="result-v2",
        verification_request_id="request-v2",
        logical_call_id="call-v2",
        request_binding_sha256="a" * 64,
        findings=findings,
        summary="Independent whole-Task review completed.",
        execution_repair_objective=(
            "Rewrite the root output while preserving verified content."
            if disposition
            is TaskDeliveryValidationDisposition.RETRY_EXECUTION
            else None
        ),
        task_graph_revision_objective=(
            "Add the missing dependency before executing the root."
            if disposition
            is TaskDeliveryValidationDisposition.REPLAN_TASK_GRAPH
            else None
        ),
        blocking_questions=(
            ("Which user-owned value should be used?",)
            if disposition is TaskDeliveryValidationDisposition.BLOCKED
            else ()
        ),
    )


def test_host_derives_all_four_routes_with_safe_precedence() -> None:
    assert derive_task_delivery_validation_disposition(_findings()) is (
        TaskDeliveryValidationDisposition.PASS
    )

    execution = _finding(
        TaskDeliveryValidationDimension.FINAL_DELIVERY_QUALITY,
        verdict=TaskDeliveryValidationVerdict.FAIL,
        fault_domain=TaskDeliveryValidationFaultDomain.EXECUTION_OUTPUT,
        affected_node_ids=("task-root",),
    )
    assert derive_task_delivery_validation_disposition(
        _findings(execution)
    ) is TaskDeliveryValidationDisposition.RETRY_EXECUTION

    graph = _finding(
        TaskDeliveryValidationDimension.GOAL_COMPLETENESS,
        verdict=TaskDeliveryValidationVerdict.FAIL,
        fault_domain=TaskDeliveryValidationFaultDomain.TASK_GRAPH_DESIGN,
        affected_node_ids=("task-root",),
    )
    assert derive_task_delivery_validation_disposition(
        _findings(execution, graph)
    ) is TaskDeliveryValidationDisposition.REPLAN_TASK_GRAPH

    blocked = _finding(
        TaskDeliveryValidationDimension.EVIDENCE_GROUNDING,
        verdict=TaskDeliveryValidationVerdict.INSUFFICIENT_EVIDENCE,
        fault_domain=TaskDeliveryValidationFaultDomain.MISSING_AUTHORITY,
    )
    assert derive_task_delivery_validation_disposition(
        _findings(execution, graph, blocked)
    ) is TaskDeliveryValidationDisposition.BLOCKED


@pytest.mark.parametrize(
    ("verdict", "fault_domain", "affected_node_ids"),
    (
        (
            TaskDeliveryValidationVerdict.PASS,
            TaskDeliveryValidationFaultDomain.EXECUTION_OUTPUT,
            (),
        ),
        (
            TaskDeliveryValidationVerdict.FAIL,
            TaskDeliveryValidationFaultDomain.NONE,
            ("task-root",),
        ),
        (
            TaskDeliveryValidationVerdict.FAIL,
            TaskDeliveryValidationFaultDomain.MISSING_AUTHORITY,
            ("task-root",),
        ),
        (
            TaskDeliveryValidationVerdict.INSUFFICIENT_EVIDENCE,
            TaskDeliveryValidationFaultDomain.TASK_GRAPH_DESIGN,
            (),
        ),
    ),
)
def test_finding_rejects_verdict_fault_domain_mismatches(
    verdict: TaskDeliveryValidationVerdict,
    fault_domain: TaskDeliveryValidationFaultDomain,
    affected_node_ids: tuple[str, ...],
) -> None:
    with pytest.raises(ValidationError):
        _finding(
            TaskDeliveryValidationDimension.FACTUAL_CORRECTNESS,
            verdict=verdict,
            fault_domain=fault_domain,
            affected_node_ids=affected_node_ids,
        )


def test_result_requires_only_the_objective_owned_by_its_route() -> None:
    execution = _finding(
        TaskDeliveryValidationDimension.FINAL_DELIVERY_QUALITY,
        verdict=TaskDeliveryValidationVerdict.FAIL,
        fault_domain=TaskDeliveryValidationFaultDomain.EXECUTION_OUTPUT,
        affected_node_ids=("task-root",),
    )
    with pytest.raises(ValidationError, match="RETRY_EXECUTION"):
        TaskDeliveryValidationResult.create(
            verification_result_id="result-v2",
            verification_request_id="request-v2",
            logical_call_id="call-v2",
            request_binding_sha256="a" * 64,
            findings=_findings(execution),
            summary="The root output needs a local rewrite.",
            execution_repair_objective=None,
            task_graph_revision_objective="Wrong route objective.",
            blocking_questions=(),
        )


def test_root_only_execution_retry_guard_rejects_child_or_mixed_scope() -> None:
    root_retry = _result(
        _findings(
            _finding(
                TaskDeliveryValidationDimension.FINAL_DELIVERY_QUALITY,
                verdict=TaskDeliveryValidationVerdict.FAIL,
                fault_domain=(
                    TaskDeliveryValidationFaultDomain.EXECUTION_OUTPUT
                ),
                affected_node_ids=("task-root",),
            )
        )
    )
    assert require_root_only_task_delivery_execution_retry(
        result=root_retry,
        root_node_id="task-root",
    ) is root_retry

    child_retry = _result(
        _findings(
            _finding(
                TaskDeliveryValidationDimension.FACTUAL_CORRECTNESS,
                verdict=TaskDeliveryValidationVerdict.FAIL,
                fault_domain=(
                    TaskDeliveryValidationFaultDomain.EXECUTION_OUTPUT
                ),
                affected_node_ids=("child-node",),
            )
        )
    )
    with pytest.raises(ValueError, match="canonical root"):
        require_root_only_task_delivery_execution_retry(
            result=child_retry,
            root_node_id="task-root",
        )

    mixed_retry = _result(
        _findings(
            _finding(
                TaskDeliveryValidationDimension.FACTUAL_CORRECTNESS,
                verdict=TaskDeliveryValidationVerdict.FAIL,
                fault_domain=(
                    TaskDeliveryValidationFaultDomain.EXECUTION_OUTPUT
                ),
                affected_node_ids=("task-root", "child-node"),
            )
        )
    )
    with pytest.raises(ValueError, match="canonical root"):
        require_root_only_task_delivery_execution_retry(
            result=mixed_retry,
            root_node_id="task-root",
        )


def test_root_only_guard_is_not_a_generic_route_guard() -> None:
    with pytest.raises(ValueError, match="RETRY_EXECUTION"):
        require_root_only_task_delivery_execution_retry(
            result=_result(_findings()),
            root_node_id="task-root",
        )
