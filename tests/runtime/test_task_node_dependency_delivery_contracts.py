"""冻结 TaskNode 依赖交付契约的边界覆盖。"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from personagraph.l2.task_execution.task_node import (
    dependency_delivery_contracts as contracts,
)
from personagraph.l2.task_execution.attempts import (
    input_projection as attempt_input_projection,
)
from personagraph.l2.task_execution.verification import (
    decision as node_verification,
)
from personagraph.l2.task_execution.task_node.input_limits import TaskNodeDependencyInputLimits
from personagraph.l2.work_run import (
    CurrentTaskNodeDeliveryResolutionKind,
    OutputWindowFormat,
    OutputWindow,
    ResolvedCurrentTaskNodeDelivery,
    ResolvedTaskNodeDelivery,
    TaskNodeDeliveryCarryAuthority,
    TaskNodeDelivery,
    TaskNodeSubject,
)


_PAYLOAD_HELPER_NAMES = {
    'TaskNodeDependencyDeliveries',
    "build_task_node_dependency_model_payload",
    "serialize_task_node_dependency_model_payload",
}
_DIRECT_DELIVERY_CONSUMERS = {
    "node_verification_controller.py": {
        'TaskNodeDependencyDeliveries',
        "TaskNodeDependencyInputTooLarge",
        "TaskNodeDependencyInputUnsupported",
    },
    "work_run_turn_controller.py": {
        'TaskNodeDependencyDeliveries',
        "TaskNodeDependencyInputTooLarge",
        "TaskNodeDependencyInputUnsupported",
    },
}


def _subject(
    node_id: str,
    *,
    graph_revision: int = 1,
) -> TaskNodeSubject:
    return TaskNodeSubject(
        task_id="task-1",
        graph_revision=graph_revision,
        node_id=node_id,
        node_revision=1,
    )


def _resolved(
    node_id: str,
    *,
    graph_revision: int = 1,
    content: str = "complete child delivery",
) -> ResolvedTaskNodeDelivery:
    work_run_id = f"work-run-{node_id}-{graph_revision}"
    return ResolvedTaskNodeDelivery(
        delivery=TaskNodeDelivery(
            delivery_id=f"delivery-{node_id}-{graph_revision}",
            session_id="session-1",
            work_run_id=work_run_id,
            subject=_subject(node_id, graph_revision=graph_revision),
            verification_request_id=f"verification-{node_id}-{graph_revision}",
            submitted_attempt_id=f"attempt-{node_id}-{graph_revision}",
            output_revision=2,
            created_turn_id="turn-1",
        ),
        output_window=OutputWindow(
            work_run_id=work_run_id,
            output_revision=2,
            format=OutputWindowFormat.MARKDOWN,
            content=content,
            updated_turn_id="turn-1",
            updated_attempt_id=f"attempt-{node_id}-{graph_revision}",
        ),
    )


def _limits(*, max_items: int = 1, max_bytes: int = 100_000) -> TaskNodeDependencyInputLimits:
    return TaskNodeDependencyInputLimits(
        profile_id="dependency-delivery-contracts-boundary",
        max_items=max_items,
        max_serialized_utf8_bytes=max_bytes,
    )


def _imported_names(path: Path, *module_names: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module in module_names
        for alias in node.names
    }


def test_delivery_contract_owner_has_cold_import_fence_and_direct_consumers() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(repository_root / "src"), existing_pythonpath)
        if value
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import sys
import personagraph.l2.task_execution.task_node.dependency_delivery_contracts
blocked = {
    "personagraph.session",
    "personagraph.session.store",
    "personagraph.runtime.attempt_input_projection",
    "personagraph.l2.auxiliary_execution.work_run.controller",
    "personagraph.runtime.node_verification",
    "personagraph.runtime.task_node_dependencies",
    "personagraph.runtime.task_node_frontier",
    "personagraph.runtime.work_run_turn_controller",
}
print(json.dumps(sorted(blocked & set(sys.modules))))
""",
        ],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []

    owner_path = Path(contracts.__file__)
    tree = ast.parse(owner_path.read_text(encoding="utf-8"))
    relative_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level > 0
        and node.module is not None
    }
    assert relative_imports == {"input_limits"}
    absolute_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0
    }
    assert {
        "personagraph.l2.work_run.contracts",
        "personagraph.l2.work_run.verification",
    } <= absolute_imports

    for consumer in (attempt_input_projection, node_verification):
        consumer_path = Path(consumer.__file__)
        assert _PAYLOAD_HELPER_NAMES <= _imported_names(
            consumer_path,
            "task_node.dependency_delivery_contracts",
        )
        assert not (
            _PAYLOAD_HELPER_NAMES
            & _imported_names(consumer_path, "task_node.dependencies")
        )

    execution_dir = (
        repository_root / "src" / "personagraph" / "l2" / "task_execution"
    )
    direct_delivery_consumers = {
        execution_dir / "verification" / "controller.py": (
            _DIRECT_DELIVERY_CONSUMERS["node_verification_controller.py"]
        ),
        execution_dir / "work_run" / "turn_controller.py": (
            _DIRECT_DELIVERY_CONSUMERS["work_run_turn_controller.py"]
        ),
    }
    for consumer_path, names in direct_delivery_consumers.items():
        assert names <= _imported_names(
            consumer_path,
            "task_node.dependency_delivery_contracts",
        )
        assert not (
            names & _imported_names(consumer_path, "task_node.dependencies")
        )


def test_direct_payload_preserves_exact_json_utf8_and_guard_errors() -> None:
    resolved = _resolved("child-direct", content="完整正文🙂\n第二行")
    deliveries = contracts.TaskNodeDependencyDeliveries(
        items=(
            contracts.TaskNodeDependencyDelivery(
                child_ordinal=4,
                resolved_current_delivery=ResolvedCurrentTaskNodeDelivery.direct(
                    resolved
                ),
            ),
        )
    )

    payload = contracts.build_task_node_dependency_model_payload(deliveries)
    dependency = payload["dependency_deliveries"][0]
    assert dependency["child_ordinal"] == 4
    assert dependency["subject"] == resolved.delivery.subject.model_dump(mode="json")
    assert dependency["delivery_resolution"] == {
        "kind": "direct",
        "target_subject": resolved.delivery.subject.model_dump(mode="json"),
    }
    assert dependency["output_window"]["content"] == "完整正文🙂\n第二行"

    serialized = contracts.serialize_task_node_dependency_model_payload(
        deliveries,
        limits=_limits(),
    )
    exact_bytes = len(serialized.encode("utf-8"))
    assert json.loads(serialized) == payload
    assert contracts.task_node_dependency_serialized_utf8_bytes(
        deliveries,
        limits=_limits(max_bytes=exact_bytes),
    ) == exact_bytes
    assert contracts.serialize_task_node_dependency_model_payload(
        deliveries,
        limits=_limits(max_bytes=exact_bytes),
    ) == serialized

    with pytest.raises(contracts.TaskNodeDependencyInputTooLarge) as too_large:
        contracts.serialize_task_node_dependency_model_payload(
            deliveries,
            limits=_limits(max_bytes=exact_bytes - 1),
        )
    assert too_large.value.code == "task_node_dependency_input_too_large"
    assert too_large.value.item_count == 1
    assert too_large.value.serialized_utf8_bytes == exact_bytes

    subject = _subject("child-invalid")
    invalid_projection = SimpleNamespace(
        items=(
            SimpleNamespace(
                child_ordinal=0,
                delivery_id="delivery-invalid",
                child_subject=subject,
                resolved_current_delivery=SimpleNamespace(
                    resolution_kind=CurrentTaskNodeDeliveryResolutionKind.DIRECT,
                    target_subject=subject,
                ),
                resolved_delivery=SimpleNamespace(
                    output_window=SimpleNamespace(
                        output_revision=2,
                        format=SimpleNamespace(value="markdown"),
                        content=float("nan"),
                    )
                ),
            ),
        )
    )
    unsupported_limits = _limits()
    with pytest.raises(contracts.TaskNodeDependencyInputUnsupported) as unsupported:
        contracts.serialize_task_node_dependency_model_payload(
            invalid_projection,
            limits=unsupported_limits,
        )
    assert unsupported.value.code == "task_node_dependency_input_unsupported"
    assert unsupported.value.limits is unsupported_limits
    assert isinstance(unsupported.value.__cause__, ValueError)


def test_carried_payload_keeps_source_subject_and_projects_current_target() -> None:
    target = _subject("child-carried", graph_revision=2)
    source = _resolved(
        "child-carried",
        graph_revision=1,
        content="来自历史图版本的完整交付",
    )
    carried = ResolvedCurrentTaskNodeDelivery(
        target_subject=target,
        source_delivery=source,
        resolution_kind=CurrentTaskNodeDeliveryResolutionKind.CARRIED,
        carry_authority=TaskNodeDeliveryCarryAuthority(
            carry_receipt_id="carry-1",
            carry_receipt_sha256="a" * 64,
            apply_id="apply-1",
            source_delivery_id=source.delivery.delivery_id,
            base_task_graph_revision=1,
            target_subject=target,
        ),
    )
    projection = contracts.TaskNodeDependencyProjection(
        parent_subject=_subject("parent", graph_revision=2),
        items=(
            contracts.TaskNodeDependencyDelivery(
                child_ordinal=1,
                resolved_current_delivery=carried,
            ),
        ),
    )

    payload = contracts.build_task_node_dependency_model_payload(projection)
    dependency = payload["dependency_deliveries"][0]
    assert dependency["subject"] == target.model_dump(mode="json")
    assert dependency["output_window"]["content"] == "来自历史图版本的完整交付"
    assert dependency["delivery_resolution"] == {
        "kind": "carried",
        "target_subject": target.model_dump(mode="json"),
        "source_subject": source.delivery.subject.model_dump(mode="json"),
        "carry_receipt_id": "carry-1",
        "carry_receipt_sha256": "a" * 64,
        "carry_apply_id": "apply-1",
    }
    assert source.delivery.subject.graph_revision == 1
