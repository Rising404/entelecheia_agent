from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from personagraph.l2.task_graph.contracts import InSessionTaskDetails, InSessionTaskStatus
from personagraph.l2.task_execution.work_run import (
    turn_controller as work_run_turn_controller,
)
from personagraph.l2.task_execution.verification.controller import (
    NodeVerificationControllerStateConflict,
    SqliteNodeVerificationApplicationStore,
)
from personagraph.l2.task_execution.task_node.dependencies import (
    TaskNodeDependencyInputLimits,
    TaskNodeDependencyInputTooLarge,
    TaskNodeDependencyProjectionError,
    TaskNodeDependencyProjectionErrorCode,
    build_task_node_dependency_model_payload,
    project_task_node_dependencies,
    serialize_task_node_dependency_model_payload,
)
from personagraph.l2.task_execution.task_node.frontier import build_task_node_tree
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


TASK_ID = "task_root"


def _node(
    node_id: str,
    *,
    ordinal: int,
    parent: str | None,
    status: InSessionTaskStatus,
    revision: int = 1,
) -> dict[str, object]:
    return {
        "insession_task_node_id": node_id,
        "node_revision": revision,
        "node_kind": "root" if parent is None else "subtask",
        "ordinal": ordinal,
        "parent_insession_task_node_id": parent,
        "status": status.value,
        "state_version": 1,
    }


def _details(*, graph_revision: int = 1) -> InSessionTaskDetails:
    return InSessionTaskDetails(
        insession_task_id=TASK_ID,
        session_id="session_one",
        task_state_version=2,
        title="root",
        objective="aggregate child results",
        current_graph_revision=graph_revision,
        status=InSessionTaskStatus.ACTIVE,
        nodes=(
            _node(TASK_ID, ordinal=0, parent=None, status=InSessionTaskStatus.PROPOSED),
            _node("child_b", ordinal=2, parent=TASK_ID, status=InSessionTaskStatus.COMPLETED),
            _node("grandchild", ordinal=3, parent="child_a", status=InSessionTaskStatus.COMPLETED),
            _node("child_a", ordinal=1, parent=TASK_ID, status=InSessionTaskStatus.COMPLETED, revision=2),
        ),
    )


def _tree():
    return build_task_node_tree(_details())


def _subject(node_id: str, *, revision: int = 1, task_id: str = TASK_ID, graph_revision: int = 1):
    return TaskNodeSubject(
        task_id=task_id,
        graph_revision=graph_revision,
        node_id=node_id,
        node_revision=revision,
    )


def _resolved(
    node_id: str,
    *,
    content: str | None = None,
    revision: int = 1,
    task_id: str = TASK_ID,
    graph_revision: int = 1,
    delivery_id: str | None = None,
) -> ResolvedTaskNodeDelivery:
    work_run_id = f"run_{node_id}_{revision}"
    output_revision = 2
    return ResolvedTaskNodeDelivery(
        delivery=TaskNodeDelivery(
            delivery_id=delivery_id or f"delivery_{node_id}_{revision}",
            session_id="session_one",
            work_run_id=work_run_id,
            subject=_subject(
                node_id,
                revision=revision,
                task_id=task_id,
                graph_revision=graph_revision,
            ),
            verification_request_id=f"verification_{node_id}_{revision}",
            submitted_attempt_id=f"attempt_{node_id}_{revision}",
            output_revision=output_revision,
            created_turn_id="turn_one",
        ),
        output_window=OutputWindow(
            work_run_id=work_run_id,
            output_revision=output_revision,
            format=OutputWindowFormat.MARKDOWN,
            content=content or f"result for {node_id}",
            updated_turn_id="turn_one",
            updated_attempt_id=f"attempt_{node_id}_{revision}",
        ),
    )


def _project():
    tree = _tree()
    return project_task_node_dependencies(
        tree,
        parent_subject=_subject(TASK_ID),
        resolved_deliveries=(
            _resolved("child_b"),
            _resolved("child_a", revision=2),
        ),
    )


def _direct(
    resolved: ResolvedTaskNodeDelivery,
) -> ResolvedCurrentTaskNodeDelivery:
    return ResolvedCurrentTaskNodeDelivery.direct(resolved)


def test_projection_is_direct_child_only_and_stably_orders_unsorted_facts():
    projection = _project()

    assert [item.child_subject.node_id for item in projection.items] == [
        "child_a",
        "child_b",
    ]
    assert projection.delivery_ids == ("delivery_child_a_2", "delivery_child_b_1")
    assert "grandchild" not in {
        item.child_subject.node_id for item in projection.items
    }


def test_canonical_payload_preserves_exact_unicode_body_without_truncation():
    content = "完整正文🙂\n第二行"
    projection = project_task_node_dependencies(
        _tree(),
        parent_subject=_subject(TASK_ID),
        resolved_deliveries=(
            _resolved("child_a", revision=2, content=content),
            _resolved("child_b"),
        ),
    )
    payload = build_task_node_dependency_model_payload(projection)
    assert payload["dependency_deliveries"][0]["output_window"]["content"] == content

    serialized = serialize_task_node_dependency_model_payload(
        projection,
        limits=TaskNodeDependencyInputLimits(
            profile_id="test",
            max_items=2,
            max_serialized_utf8_bytes=100_000,
        ),
    )
    assert json.loads(serialized) == payload
    assert "🙂" in serialized


def test_payload_exposes_carried_target_and_receipt_without_rewriting_source():
    target = _subject("child_a", revision=2, graph_revision=2)
    source = _resolved(
        "child_a",
        revision=2,
        graph_revision=1,
        content="历史 revision 的可信正文",
    )
    carried = ResolvedCurrentTaskNodeDelivery(
        target_subject=target,
        source_delivery=source,
        resolution_kind=CurrentTaskNodeDeliveryResolutionKind.CARRIED,
        carry_authority=TaskNodeDeliveryCarryAuthority(
            carry_receipt_id="carry-child-a",
            carry_receipt_sha256="a" * 64,
            apply_id="commit-revision-two",
            source_delivery_id=source.delivery.delivery_id,
            base_task_graph_revision=1,
            target_subject=target,
        ),
    )
    projection = project_task_node_dependencies(
        build_task_node_tree(_details(graph_revision=2)),
        parent_subject=_subject(TASK_ID, graph_revision=2),
        resolved_deliveries=(
            carried,
            _resolved("child_b", graph_revision=2),
        ),
    )

    payload = build_task_node_dependency_model_payload(projection)
    first = payload["dependency_deliveries"][0]
    assert first["subject"]["graph_revision"] == 2
    assert first["output_window"]["content"] == "历史 revision 的可信正文"
    assert first["delivery_resolution"] == {
        "kind": "carried",
        "target_subject": target.model_dump(mode="json"),
        "source_subject": source.delivery.subject.model_dump(mode="json"),
        "carry_receipt_id": "carry-child-a",
        "carry_receipt_sha256": "a" * 64,
        "carry_apply_id": "commit-revision-two",
    }
    assert source.delivery.subject.graph_revision == 1


def test_item_and_utf8_byte_limits_accept_equality_and_reject_one_over():
    projection = _project()
    generous = TaskNodeDependencyInputLimits(
        profile_id="measure",
        max_items=2,
        max_serialized_utf8_bytes=100_000,
    )
    serialized = serialize_task_node_dependency_model_payload(
        projection,
        limits=generous,
    )
    exact_bytes = len(serialized.encode("utf-8"))

    assert serialize_task_node_dependency_model_payload(
        projection,
        limits=TaskNodeDependencyInputLimits(
            profile_id="equal",
            max_items=2,
            max_serialized_utf8_bytes=exact_bytes,
        ),
    ) == serialized
    with pytest.raises(TaskNodeDependencyInputTooLarge) as bytes_exc:
        serialize_task_node_dependency_model_payload(
            projection,
            limits=TaskNodeDependencyInputLimits(
                profile_id="bytes-over",
                max_items=2,
                max_serialized_utf8_bytes=exact_bytes - 1,
            ),
        )
    assert bytes_exc.value.serialized_utf8_bytes == exact_bytes

    with pytest.raises(TaskNodeDependencyInputTooLarge) as items_exc:
        serialize_task_node_dependency_model_payload(
            projection,
            limits=TaskNodeDependencyInputLimits(
                profile_id="items-over",
                max_items=1,
                max_serialized_utf8_bytes=exact_bytes,
            ),
        )
    assert items_exc.value.item_count == 2


@pytest.mark.parametrize(
    ("deliveries", "code"),
    (
        (
            (_resolved("child_a", revision=2),),
            TaskNodeDependencyProjectionErrorCode.MISSING_DELIVERY,
        ),
        (
            (
                _resolved("child_a", revision=2),
                _resolved("child_b"),
                _resolved("grandchild"),
            ),
            TaskNodeDependencyProjectionErrorCode.EXTRA_DELIVERY,
        ),
        (
            (
                _resolved("child_a", revision=2),
                _resolved("child_b", task_id="other_task"),
            ),
            TaskNodeDependencyProjectionErrorCode.WRONG_SUBJECT,
        ),
        (
            (_resolved("child_a"), _resolved("child_b")),
            TaskNodeDependencyProjectionErrorCode.WRONG_NODE_REVISION,
        ),
        (
            (
                _resolved("child_a", revision=2),
                _resolved("child_a", revision=2, delivery_id="another_delivery"),
            ),
            TaskNodeDependencyProjectionErrorCode.DUPLICATE_DELIVERY,
        ),
    ),
)
def test_projection_rejects_inexact_delivery_sets(deliveries, code):
    with pytest.raises(TaskNodeDependencyProjectionError) as exc:
        project_task_node_dependencies(
            _tree(),
            parent_subject=_subject(TASK_ID),
            resolved_deliveries=deliveries,
        )
    assert exc.value.code is code


def test_projection_rejects_a_parent_subject_with_stale_revision():
    with pytest.raises(TaskNodeDependencyProjectionError) as exc:
        project_task_node_dependencies(
            _tree(),
            parent_subject=_subject(TASK_ID, revision=2),
            resolved_deliveries=(),
        )
    assert exc.value.code is TaskNodeDependencyProjectionErrorCode.PARENT_MISMATCH


def test_attempt_loader_uses_exact_frontier_refs_and_full_delivery_bodies(
    monkeypatch,
):
    resolved = (
        _direct(_resolved(
            "child_a",
            revision=2,
            content="甲的完整正文🙂",
        )),
        _direct(_resolved(
            "child_b",
            content="乙的完整正文",
        )),
    )
    monkeypatch.setattr(
        work_run_turn_controller.work_run_store,
        "get_current_task_node_dependency_deliveries",
        lambda **_kwargs: resolved,
    )

    dependencies = work_run_turn_controller._load_attempt_dependency_deliveries(
        session_id="session_one",
        subject=_subject(TASK_ID),
        details=_details(),
    )

    assert list(dependencies.delivery_ids) == [
        "delivery_child_a_2",
        "delivery_child_b_1",
    ]
    assert [
        item.resolved_delivery.output_window.content
        for item in dependencies.items
    ] == ["甲的完整正文🙂", "乙的完整正文"]


def test_attempt_loader_rejects_frontier_order_or_body_tampering(monkeypatch):
    monkeypatch.setattr(
        work_run_turn_controller.work_run_store,
        "get_current_task_node_dependency_deliveries",
        lambda **_kwargs: (
            _direct(_resolved("child_b")),
            _direct(_resolved("child_a", revision=2)),
        ),
    )

    with pytest.raises(RuntimeError, match="order crossed"):
        work_run_turn_controller._load_attempt_dependency_deliveries(
            session_id="session_one",
            subject=_subject(TASK_ID),
            details=_details(),
        )


def test_verifier_loader_uses_only_durable_request_refs_and_rejects_tamper():
    class FakeStore:
        def __init__(self, delivery_ids):
            self.delivery_ids = delivery_ids
            self.loaded = []

        def get_insession_task_details(self, session_id, task_id):
            assert (session_id, task_id) == ("session_one", TASK_ID)
            return _details()

        def get_current_task_node_dependency_deliveries(
            self,
            *,
            session_id,
            subject,
        ):
            assert session_id == "session_one"
            assert subject == _subject(TASK_ID)
            resolved_by_id = {
                "delivery_child_a_2": _direct(
                    _resolved("child_a", revision=2)
                ),
                "delivery_child_b_1": _direct(_resolved("child_b")),
            }
            self.loaded.extend(self.delivery_ids)
            return tuple(resolved_by_id[item] for item in self.delivery_ids)

    def prepared(delivery_ids):
        return SimpleNamespace(
            record=SimpleNamespace(
                request=SimpleNamespace(
                    session_id="session_one",
                    subject=_subject(TASK_ID),
                    dependency_delivery_ids=delivery_ids,
                )
            )
        )

    exact_store = FakeStore(
        ("delivery_child_a_2", "delivery_child_b_1")
    )
    exact = SqliteNodeVerificationApplicationStore(
        task_store=exact_store,
        work_run_store=exact_store,
    )._resolve_dependency_deliveries(
        prepared(exact_store.delivery_ids)
    )
    assert exact.delivery_ids == exact_store.delivery_ids
    assert exact_store.loaded == list(exact_store.delivery_ids)

    tampered_store = FakeStore(
        ("delivery_child_b_1", "delivery_child_a_2")
    )
    with pytest.raises(NodeVerificationControllerStateConflict, match="order"):
        SqliteNodeVerificationApplicationStore(
            task_store=tampered_store,
            work_run_store=tampered_store,
        )._resolve_dependency_deliveries(
            prepared(tampered_store.delivery_ids)
        )
