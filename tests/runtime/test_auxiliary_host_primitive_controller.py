from __future__ import annotations

from dataclasses import replace

import pytest

from personagraph.input_processing.documents import (
    ChunkSpan,
    DocumentChunk,
    DocumentLocator,
    ElementKind,
)
from personagraph.workspace.documents import application as docstore
from personagraph.l2.auxiliary_execution.planning import (
    host_primitive_controller as controller,
)
from personagraph.l2.auxiliary_execution.driver import (
    AuxiliaryGraphDriverAction,
    canonical_auxiliary_graph_driver_state_guard,
    decide_auxiliary_graph_driver_step,
)
from personagraph.l2.planning.invocation_contracts import (
    PlanningContextPrimitiveKind,
)
from personagraph.l2.planning.resource_perception import (
    FrozenPlanningResource,
    PlanningResourceCoverage,
    PlanningResourceFormat,
    PlanningResourcePerceptionRequest,
    PlanningResourceReadRequest,
)
from personagraph.session import store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import planning as planning_store
from tests.session.test_planning_primitive_invocation_persistence import (
    _seed_current_host_node,
)
from tests.documents._authority import bound_project_document_authority


def _controller_request(session_id, turn_id, frontier):
    return controller.AuxiliaryHostPrimitiveControllerRequest(
        session_id=session_id,
        turn_id=turn_id,
        subject=frontier.ready_fresh[0].subject,
        initial_driver_state_guard_sha256=(
            canonical_auxiliary_graph_driver_state_guard(frontier)
        ),
    )


def _run(request, factory):
    return controller.run_auxiliary_host_primitive(
        request,
        request_factory=factory,
        primitive_kinds_by_capability_profile={
            "mounted_document_read": PlanningContextPrimitiveKind.RESOURCE_PERCEPTION
        },
        monotonic_clock=lambda: 10.0,
    )


def _resource_factory(
    *,
    document_id: str,
    version_id: str,
    source_sha256: str,
    changing_on_recovery: bool = False,
):
    contexts = []

    def factory(context):
        contexts.append(context)
        return PlanningResourcePerceptionRequest(
            binding=context.artifact_binding(
                scope_snapshot_sha256="f" * 64,
                alias_prefix="paperctx",
                artifact_alias="paper_context",
            ),
            read_request=PlanningResourceReadRequest(
                resource=FrozenPlanningResource(
                    session_id=context.session_id,
                    resource_alias="paper_01",
                    resource_id=document_id,
                    resource_version=version_id,
                    content_sha256=(
                        "b" * 64
                        if changing_on_recovery and context.recovering_reserved_call
                        else source_sha256
                    ),
                    coverage=PlanningResourceCoverage.COMPLETE,
                    resource_format=PlanningResourceFormat.PDF,
                    media_type="application/pdf",
                    file_extension=".pdf",
                )
            ),
        )

    return factory, contexts


def test_fresh_resource_primitive_reserves_seals_and_advances_frontier() -> None:
    session_id, turn_id, task_id, frontier = _seed_current_host_node()
    request = _controller_request(session_id, turn_id, frontier)
    factory, contexts = _resource_factory(
        document_id="missing-document",
        version_id="missing-version",
        source_sha256="a" * 64,
    )

    completed = _run(request, factory)

    assert completed.status == "completed"
    assert completed.primitive_kind is PlanningContextPrimitiveKind.RESOURCE_PERCEPTION
    assert completed.observation_status.value == "blocked"
    assert completed.seal_status == "applied"
    assert len(contexts) == 1
    assert contexts[0].recovering_reserved_call is False
    stored = planning_store.get_planning_primitive_invocation(
        session_id=session_id,
        primitive_call_id=completed.primitive_call_id,
    )
    assert stored is not None
    assert stored.status == "settled"
    assert stored.settled_artifact_id == completed.artifact_id
    next_frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    assert next_frontier.recoverable_primitive == ()
    assert [item.local_node_key for item in next_frontier.ready_fresh] == [
        "synthesize"
    ]
    assert next_frontier.ready_fresh[0].dependency_completion_ids == (
        completed.artifact_id,
    )


def test_resource_primitive_uses_the_mounted_document_port_by_default(
    tmp_path,
) -> None:
    session_id, turn_id, task_id, frontier = _seed_current_host_node()
    chunk = DocumentChunk(
        chunk_id="paper_chunk_01",
        text="The frozen paper reports a 94% accuracy result.",
        span=ChunkSpan(DocumentLocator(page=1), DocumentLocator(page=1)),
        section_path=("Results",),
        element_ids=("paper_element_01",),
        token_count=10,
        kind=ElementKind.PARAGRAPH,
        source_pages=(1,),
    )
    with bound_project_document_authority(tmp_path) as authority:
        with authority.session(session_id):
            stored = authority.ingest(
                "private/paper.pdf",
                "paper",
                "pdf",
                [{"content": chunk.text, "loc": chunk.loc}],
                session_id=session_id,
                processor_fingerprint="reader@test",
                document_chunks=(chunk,),
                chunker_fingerprint="chunker@test",
                processing_status="complete",
                processing_diagnostics=(),
            )
        version = docstore.list_document_versions(stored["doc_id"])[0]
        factory, contexts = _resource_factory(
            document_id=stored["doc_id"],
            version_id=str(version["id"]),
            source_sha256=str(version["source_sha256"]),
        )

        completed = controller.run_auxiliary_host_primitive(
            _controller_request(session_id, turn_id, frontier),
            request_factory=factory,
            primitive_kinds_by_capability_profile={
                "mounted_document_read": (
                    PlanningContextPrimitiveKind.RESOURCE_PERCEPTION
                )
            },
            monotonic_clock=lambda: 10.0,
        )

    assert completed.status == "completed"
    assert completed.primitive_kind is PlanningContextPrimitiveKind.RESOURCE_PERCEPTION
    assert completed.observation_status is not None
    assert completed.observation_status.value == "success"
    assert len(contexts) == 1
    next_frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    assert next_frontier.ready_fresh[0].dependency_completion_ids == (
        completed.artifact_id,
    )


def test_crash_after_reservation_resumes_same_logical_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id, frontier = _seed_current_host_node()
    request = _controller_request(session_id, turn_id, frontier)
    factory, contexts = _resource_factory(
        document_id="missing-document",
        version_id="missing-version",
        source_sha256="a" * 64,
    )
    actual_run = controller.run_planning_resource_perception

    def crash_after_reservation(_request, **_kwargs):
        raise RuntimeError("synthetic process loss")

    monkeypatch.setattr(
        controller,
        "run_planning_resource_perception",
        crash_after_reservation,
    )
    with pytest.raises(RuntimeError, match="synthetic process loss"):
        _run(request, factory)

    pending = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    decision = decide_auxiliary_graph_driver_step(pending)
    assert decision.action is AuxiliaryGraphDriverAction.RESUME_HOST_PRIMITIVE
    stable = controller.derive_auxiliary_host_primitive_ids(
        session_id=session_id,
        subject=request.subject,
    )
    assert decision.primitive_call_id == stable.primitive_call_id

    monkeypatch.setattr(
        controller,
        "run_planning_resource_perception",
        actual_run,
    )
    resumed_request = request.model_copy(
        update={
            "initial_driver_state_guard_sha256": (
                canonical_auxiliary_graph_driver_state_guard(pending)
            )
        }
    )
    completed = _run(resumed_request, factory)

    assert completed.primitive_call_id == stable.primitive_call_id
    assert [item.recovering_reserved_call for item in contexts] == [False, True]
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM "
            "insession_auxiliary_planning_primitive_invocations"
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_auxiliary_observations"
        ).fetchone()[0] == 1


def test_recovery_rejects_changed_request_before_second_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id, frontier = _seed_current_host_node()
    request = _controller_request(session_id, turn_id, frontier)
    factory, _contexts = _resource_factory(
        document_id="missing-document",
        version_id="missing-version",
        source_sha256="a" * 64,
        changing_on_recovery=True,
    )
    calls = 0

    def crash(_request, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic process loss")

    monkeypatch.setattr(controller, "run_planning_resource_perception", crash)
    with pytest.raises(RuntimeError, match="synthetic process loss"):
        _run(request, factory)
    pending = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    resumed = request.model_copy(
        update={
            "initial_driver_state_guard_sha256": (
                canonical_auxiliary_graph_driver_state_guard(pending)
            )
        }
    )

    with pytest.raises(
        controller.AuxiliaryHostPrimitiveStateConflict,
        match="reproduce the reserved invocation",
    ):
        _run(resumed, factory)

    assert calls == 1
    stored = planning_store.get_planning_primitive_invocation(
        session_id=session_id,
        primitive_call_id=(
            controller.derive_auxiliary_host_primitive_ids(
                session_id=session_id,
                subject=request.subject,
            ).primitive_call_id
        ),
    )
    assert stored is not None and stored.status == "reserved"


def test_fresh_reservation_replay_never_grants_a_second_physical_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, _task_id, frontier = _seed_current_host_node()
    request = _controller_request(session_id, turn_id, frontier)
    factory, _contexts = _resource_factory(
        document_id="missing-document",
        version_id="missing-version",
        source_sha256="a" * 64,
    )
    actual_reserve = planning_store.reserve_planning_primitive_invocation
    reads = 0

    def simulate_other_dispatch_won(*, invocation):
        applied = actual_reserve(invocation=invocation)
        return replace(applied, status="replayed")

    def forbidden_read(_request, **_kwargs):
        nonlocal reads
        reads += 1
        raise AssertionError("reservation replay must not authorize I/O")

    monkeypatch.setattr(
        controller.planning_store,
        "reserve_planning_primitive_invocation",
        simulate_other_dispatch_won,
    )
    monkeypatch.setattr(
        controller,
        "run_planning_resource_perception",
        forbidden_read,
    )

    waiting = _run(request, factory)

    assert waiting.status is controller.AuxiliaryHostPrimitiveControllerStatus.WAITING_EXTERNAL
    assert waiting.reason_code == "host_primitive_dispatch_already_reserved"
    assert reads == 0
