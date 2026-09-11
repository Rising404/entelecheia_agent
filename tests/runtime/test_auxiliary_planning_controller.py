from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from personagraph.input_processing.documents import (
    ChunkSpan,
    DocumentChunk,
    DocumentLocator,
    ElementKind,
)
from personagraph.input_processing.files import SourceFingerprint
from personagraph.l2.task_graph.task_matching import (
    InSessionTaskMatchesProposal,
)
from personagraph.workspace.documents import application as docstore
from personagraph.model_io.gateway import ModelResult
from personagraph.l2.auxiliary_execution.planning import controller
from personagraph.l2.auxiliary_execution.planning.controller import (
    AuxiliaryInitialPlanningError,
    AuxiliaryInitialPlanningStatus,
    run_initial_auxiliary_planning,
)
from personagraph.l2.auxiliary_execution.driver import (
    canonical_auxiliary_graph_driver_state_guard,
)
from personagraph.l2.auxiliary_execution.planning.mounted_document_authority import (
    build_mounted_document_perception_request,
    freeze_mounted_document_planning_authority,
)
from personagraph.l2.auxiliary_execution.planning.host_primitive_controller import (
    AuxiliaryHostPrimitiveControllerRequest,
    AuxiliaryHostPrimitiveControllerStatus,
    run_auxiliary_host_primitive,
)
from personagraph.l2.auxiliary_execution.planning.model_provider import (
    build_auxiliary_architect_structured_provider,
)
from personagraph.l2.auxiliary_execution.terminal import composition as terminal
from personagraph.l2.planning.invocation_contracts import (
    PlanningContextPrimitiveKind,
)
from personagraph.runtime.model_calls import (
    DurableModelCallStateGuardRejected,
)
from personagraph.runtime.model_calls import (
    RuntimeModelCallWaitingExternal,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import terminal as terminal_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import planning as planning_store


USER_TEXT = "请理解挂载材料并形成可执行、可验证的任务图"


def _virtual_mounted_freshness(
    session_id: str,
    document_id: str | None = None,
):
    mounted = tuple(docstore.mounted_docs(session_id))
    selected = tuple(
        document
        for document in mounted
        if document_id is None or document.get("id") == document_id
    )
    if document_id is not None and not selected:
        return {
            "ok": False,
            "status": "freshness_blocked",
            "documents": [{"doc_id": document_id, "status": "not_mounted"}],
        }
    return {
        "ok": True,
        "status": "verified_current",
        "documents": [
            {
                "doc_id": str(document["id"]),
                "version_id": str(document["current_version_id"]),
                "status": "verified_current",
            }
            for document in selected
        ],
    }


@pytest.fixture(autouse=True)
def _virtual_planning_sources_are_physically_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """挂载规划器夹具使用虚构的非披露路径。"""

    monkeypatch.setattr(
        docstore,
        "check_mounted_document_freshness",
        _virtual_mounted_freshness,
    )


def _window_revision(session_id: str) -> int:
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    return int(window["state_version"])






def _seed_task() -> tuple[str, str, str]:
    session_id = store.create_session("Entelecheia")
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="aux-v2-initial-planning",
        source="auxiliary_v2_initial_planning_test",
        user_text=USER_TEXT,
        lease_owner="auxiliary-v2-initial-planning-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    applied = task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=turn_id,
        apply_id="aux-v2-initial-planning-task",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "new_root",
                        "local_key": "root",
                        "title": "理解材料",
                        "objective": "理解材料并形成可执行任务图",
                        "source_excerpt": USER_TEXT,
                    }
                ]
            }
        ),
        exposed_catalog_ids=(),
        expected_window_revision=_window_revision(session_id),
    )
    return (
        session_id,
        turn_id,
        applied.created_insession_task_ids_by_local_key["root"],
    )


def _settle_turn(*, session_id: str, turn_id: str) -> None:
    window = store.get_turn_execution_window(session_id)
    assert window is not None and window["turn_id"] == turn_id
    interrupted = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(window["state_version"]),
        stage="RESPONSE",
        interruption_reason="TEST_INITIAL_PLANNING_CONTINUATION",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(interrupted["state_version"]),
        end_reason="host_stopped",
        error_code="TEST_INITIAL_PLANNING_CONTINUATION",
    )


def _accept_continuation_turn(
    *,
    session_id: str,
    prior_turn_id: str,
    task_id: str,
    execute_current: bool = True,
) -> str:
    _settle_turn(session_id=session_id, turn_id=prior_turn_id)
    user_text = "继续执行已有任务；不得把本句重绑定为初始规划目标"
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=(
            "initial-planning-execute-lane"
            if execute_current
            else "initial-planning-readonly-lane"
        ),
        source="runtime_test",
        user_text=user_text,
        lease_owner="initial-planning-continuation-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=turn_id,
        apply_id=f"initial-planning-lane-{turn_id}",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "existing_root",
                        "insession_task_id": task_id,
                        "source_excerpt": user_text,
                        "execute_current": execute_current,
                    }
                ]
            }
        ),
        exposed_catalog_ids=(task_id,),
        expected_window_revision=_window_revision(session_id),
    )
    return turn_id


def _no_mounted_documents(monkeypatch: pytest.MonkeyPatch, session_id: str) -> None:
    monkeypatch.setattr(
        "personagraph.tools.documents.mounted_document_source_authority.docstore.mounted_docs",
        lambda selected_session_id: (
            () if selected_session_id == session_id else None
        ),
    )


def _ingest_planning_document(
    *,
    session_id: str,
    path: str,
    title: str,
    content: str,
) -> dict[str, object]:
    source_sha256 = hashlib.sha256(
        f"{path}:{content}".encode("utf-8")
    ).hexdigest()
    chunk = DocumentChunk(
        chunk_id="planning_chunk_" + source_sha256[:16],
        text=content,
        span=ChunkSpan(
            start=DocumentLocator(page=1, ordinal=0),
            end=DocumentLocator(page=1, ordinal=0),
        ),
        section_path=("Document",),
        element_ids=("planning_element_" + source_sha256[:16],),
        token_count=max(1, len(content.split())),
        kind=ElementKind.PARAGRAPH,
        source_pages=(1,),
    )
    return docstore.ingest(
        path,
        title,
        "application/octet-stream",
        [{"content": content, "loc": chunk.loc}],
        session_id=session_id,
        source_fingerprint=SourceFingerprint(
            sha256=source_sha256,
            size_bytes=len(content.encode("utf-8")),
            mtime_ns=7,
        ),
        processor_fingerprint="initial-planning-reader@test",
        document_chunks=(chunk,),
        chunker_fingerprint="initial-planning-chunker@test",
        processing_status="complete",
        processing_diagnostics=(),
    )


def test_initial_planning_commits_revision_two_and_never_replans_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    physical = build_auxiliary_architect_structured_provider()
    provider_calls = 0

    def provider(*args: object, **kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return physical(*args, **kwargs)  # type: ignore[arg-type]

    first = run_initial_auxiliary_planning(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        provider=provider,
        ledger_store=store,
        emit=lambda _event: None,
    )

    assert first.status is AuxiliaryInitialPlanningStatus.PLANNED
    assert first.bootstrapped is True
    assert first.bootstrap_commit is not None
    assert first.revision_commit is not None
    assert first.architect_decision is not None
    assert first.details.auxiliary_graph_revision == 2
    assert [node.local_node_key for node in first.details.nodes] == [
        "analyze_verified_context",
        "synthesize_task_graph",
    ]
    terminal = first.details.nodes[-1]
    assert terminal.node_revision == 2
    assert terminal.origin_node_ref is not None
    assert terminal.origin_node_ref.node_id == terminal.auxiliary_node_id
    assert terminal.origin_node_ref.node_revision == 1
    assert first.details.budget is not None
    assert first.details.budget.usage.auxiliary_graph_revisions == 2
    assert first.details.budget.usage.distinct_auxiliary_nodes == 2

    second = run_initial_auxiliary_planning(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        provider=lambda *_args, **_kwargs: pytest.fail(
            "completed initial planning reached Provider"
        ),
        ledger_store=store,
        emit=lambda _event: None,
    )

    assert second.status is AuxiliaryInitialPlanningStatus.ALREADY_PLANNED
    assert second.details == first.details
    assert second.model_replayed is True
    assert second.architect_decision == first.architect_decision
    assert second.revision_commit is not None
    assert second.revision_commit.status == "replayed"
    assert provider_calls == 1

    completion = planning_store.get_auxiliary_initial_planning_completion(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert completion is not None
    assert completion.architect_decision == first.architect_decision
    assert completion == first.revision_commit.initial_planning_completion

    logical = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=first.architect_decision.logical_call_id,
    )
    assert logical is not None
    assert len(logical.physical_attempts) == 1
    assert logical.physical_attempts[0].settlement is not None


def test_initial_planning_canonicalizes_model_initial_label_for_bootstrap() -> None:
    session_id, turn_id, task_id = _seed_task()
    physical = build_auxiliary_architect_structured_provider()

    def provider(*args: object, **kwargs: object) -> ModelResult:
        result = physical(*args, **kwargs)  # type: ignore[arg-type]
        proposal = json.loads(result.reply)
        assert proposal["disposition"] == "revise_revision"
        proposal["revision_reason"] = "initial"
        return replace(result, reply=json.dumps(proposal))

    planned = run_initial_auxiliary_planning(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        provider=provider,
        ledger_store=store,
        emit=lambda _event: None,
    )

    assert planned.status is AuxiliaryInitialPlanningStatus.PLANNED
    assert planned.architect_decision is not None
    assert (
        planned.architect_decision.proposal.revision_reason.value
        == "manual_replan"
    )


def test_initial_planning_materializes_one_host_node_per_frozen_document() -> None:
    session_id, turn_id, task_id = _seed_task()
    private_pdf = "/private/customer/secret-paper.pdf"
    private_png = "/private/customer/secret-chart.png"
    pdf = _ingest_planning_document(
        session_id=session_id,
        path=private_pdf,
        title="Secret Paper",
        content="The method improves grounded document analysis.",
    )
    png = _ingest_planning_document(
        session_id=session_id,
        path=private_png,
        title="Secret Chart",
        content="The chart reports a ten point improvement.",
    )

    result = run_initial_auxiliary_planning(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        ledger_store=store,
        emit=lambda _event: None,
    )

    assert result.status is AuxiliaryInitialPlanningStatus.PLANNED
    assert [node.local_node_key for node in result.details.nodes] == [
        "observe_document_01",
        "observe_document_02",
        "analyze_verified_context",
        "synthesize_task_graph",
    ]
    assert [
        node.input_resource_aliases for node in result.details.nodes[:2]
    ] == [("mounted_document_01",), ("mounted_document_02",)]
    assert [node.capability_profile_id for node in result.details.nodes[:2]] == [
        "mounted_document_read",
        "mounted_document_read",
    ]
    assert result.details.authority_snapshot is not None
    assert tuple(
        anchor.projection_alias
        for anchor in result.details.authority_snapshot.anchors
    ) == (
        "mounted_document_01",
        "mounted_document_02",
        "task_creation_source",
    )
    terminal_context = (
        terminal_store.build_auxiliary_terminal_task_graph_validation_context(
            session_id=session_id,
            invocation_turn_id=turn_id,
            task_id=task_id,
        )
    )
    assert tuple(
        anchor.anchor_id for anchor in terminal_context.source_anchors
    ) == (
        "task_creation_source",
        "mounted_document_01",
        "mounted_document_02",
    )
    semantic_support = terminal_store.project_auxiliary_terminal_semantic_support(
        session_id=session_id,
        invocation_turn_id=turn_id,
        task_id=task_id,
    )
    assert tuple(
        card.alias for card in semantic_support.observation_source_cards
    ) == ("mounted_document_01", "mounted_document_02")
    planning_completion = planning_store.get_auxiliary_initial_planning_completion(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert planning_completion is not None
    initial_cards = {
        card.alias: card
        for card in planning_completion.binding.architect_request.prompt_payload.authority.cards
    }
    assert semantic_support.observation_source_cards == (
        initial_cards["mounted_document_01"],
        initial_cards["mounted_document_02"],
    )

    assert result.architect_decision is not None
    logical = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=result.architect_decision.logical_call_id,
    )
    assert logical is not None
    public_request = logical.request.request_json
    private_values = (
        private_pdf,
        private_png,
        "Secret Paper",
        "Secret Chart",
        str(pdf["doc_id"]),
        str(png["doc_id"]),
    )
    assert all(value not in public_request for value in private_values)


def test_terminal_projection_rejects_document_that_became_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    document = _ingest_planning_document(
        session_id=session_id,
        path="/private/customer/replaced-paper.pdf",
        title="Replaceable Paper",
        content="The initially frozen evidence generation.",
    )
    planned = run_initial_auxiliary_planning(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        ledger_store=store,
        emit=lambda _event: None,
    )
    assert planned.status is AuxiliaryInitialPlanningStatus.PLANNED

    monkeypatch.setattr(
        docstore,
        "check_mounted_document_freshness",
        lambda _session_id, document_id=None: {
            "ok": False,
            "status": "freshness_blocked",
            "documents": [
                {
                    "doc_id": document_id,
                    "status": "content_changed",
                }
            ],
        },
    )
    with pytest.raises(
        terminal.AuxiliaryTerminalCompositionError,
        match="stale or unavailable",
    ):
        terminal._rebuild_planning_prompt_projections(
            session_id=session_id,
            task_id=task_id,
            details=planned.details,
            allowed_managed_document_ids=(str(document["doc_id"]),),
        )


def test_planned_document_nodes_execute_into_verified_dependency_artifacts() -> None:
    session_id, turn_id, task_id = _seed_task()
    first_text = "PDF finding: the method improves grounded analysis."
    second_text = "PNG finding: the chart reports a ten point gain."
    _ingest_planning_document(
        session_id=session_id,
        path="/private/evidence/paper.pdf",
        title="Private Paper",
        content=first_text,
    )
    _ingest_planning_document(
        session_id=session_id,
        path="/private/evidence/chart.png",
        title="Private Chart",
        content=second_text,
    )
    planned = run_initial_auxiliary_planning(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        ledger_store=store,
        emit=lambda _event: None,
    )
    mounted = freeze_mounted_document_planning_authority(session_id=session_id)

    for expected_key in ("observe_document_01", "observe_document_02"):
        frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
            session_id=session_id,
            turn_id=turn_id,
            insession_task_id=task_id,
        )
        selected = frontier.ready_fresh[0]
        assert selected.local_node_key == expected_key
        node = next(
            item
            for item in planned.details.nodes
            if item.auxiliary_node_id == selected.subject.node_id
        )
        clock_values = iter((1.0, 1.25))
        result = run_auxiliary_host_primitive(
            AuxiliaryHostPrimitiveControllerRequest(
                session_id=session_id,
                turn_id=turn_id,
                subject=selected.subject,
                initial_driver_state_guard_sha256=(
                    canonical_auxiliary_graph_driver_state_guard(frontier)
                ),
            ),
            request_factory=lambda context, aliases=node.input_resource_aliases: (
                build_mounted_document_perception_request(
                    context=context,
                    input_resource_aliases=aliases,
                    mounted_authority=mounted,
                )
            ),
            primitive_kinds_by_capability_profile={
                "mounted_document_read": (
                    PlanningContextPrimitiveKind.RESOURCE_PERCEPTION
                )
            },
            monotonic_clock=lambda: next(clock_values),
        )
        assert (
            result.status
            is AuxiliaryHostPrimitiveControllerStatus.COMPLETED
        )

    analysis_frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    analysis = analysis_frontier.ready_fresh[0]
    assert analysis.local_node_key == "analyze_verified_context"
    dependencies = auxiliary_graph_store.resolve_auxiliary_dependencies(
        session_id=session_id,
        turn_id=turn_id,
        consumer_subject=analysis.subject,
    )
    assert len(dependencies.items) == 2
    serialized = dependencies.model_dump_json()
    assert first_text in serialized
    assert second_text in serialized
    assert "/private/evidence" not in serialized


def test_revision_commit_response_loss_recovers_without_second_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    physical = build_auxiliary_architect_structured_provider()
    provider_calls = 0

    def provider(*args: object, **kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return physical(*args, **kwargs)  # type: ignore[arg-type]

    real_commit = auxiliary_graph_store.commit_auxiliary_graph_revision
    lost = False
    captured: dict[str, object] | None = None

    def commit_then_lose(**kwargs: object):
        nonlocal captured, lost
        captured = dict(kwargs)
        result = real_commit(**kwargs)  # type: ignore[arg-type]
        if kwargs["expected_current_auxiliary_graph_revision"] == 1 and not lost:
            lost = True
            raise RuntimeError("simulated response loss after revision commit")
        return result

    monkeypatch.setattr(
        auxiliary_graph_store,
        "commit_auxiliary_graph_revision",
        commit_then_lose,
    )
    with pytest.raises(RuntimeError, match="simulated response loss"):
        run_initial_auxiliary_planning(
            session_id=session_id,
            turn_id=turn_id,
            insession_task_id=task_id,
            provider=provider,
            ledger_store=store,
            emit=lambda _event: None,
        )

    assert captured is not None
    replayed_commit = real_commit(**captured)  # type: ignore[arg-type]
    assert replayed_commit.status == "replayed"
    assert replayed_commit.initial_planning_completion is not None
    original_binding = captured["initial_planning_completion"]
    assert isinstance(
        original_binding,
        planning_store.AuxiliaryInitialPlanningCompletionBinding,
    )
    collided = dict(captured)
    collided["initial_planning_completion"] = original_binding.model_copy(
        update={"runtime_logical_request_binding_sha256": "0" * 64}
    )
    with pytest.raises(auxiliary_graph_store.AuxiliaryGraphApplyIdCollision):
        real_commit(**collided)  # type: ignore[arg-type]

    recovered = run_initial_auxiliary_planning(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        provider=lambda *_args, **_kwargs: pytest.fail(
            "response-loss recovery reached Provider"
        ),
        ledger_store=store,
        emit=lambda _event: None,
    )
    assert recovered.status is AuxiliaryInitialPlanningStatus.ALREADY_PLANNED
    assert recovered.details.auxiliary_graph_revision == 2
    assert recovered.model_replayed is True
    assert recovered.revision_commit is not None
    assert recovered.revision_commit.initial_planning_completion is not None
    assert provider_calls == 1
    completion = planning_store.get_auxiliary_initial_planning_completion(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert completion == recovered.revision_commit.initial_planning_completion
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM "
            "insession_auxiliary_graph_revision_apply_receipts_v2 "
            "WHERE apply_id=?",
            (completion.binding.revision_apply_id,),
        ).fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_initial_architect_replays_succeeded_call_across_turn_after_precommit_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, source_turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    physical = build_auxiliary_architect_structured_provider()
    provider_calls = 0

    def count_provider(*args: object, **kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return physical(*args, **kwargs)  # type: ignore[arg-type]

    real_commit = auxiliary_graph_store.commit_auxiliary_graph_revision

    def lose_before_architect_commit(**kwargs: object):
        if kwargs["expected_current_auxiliary_graph_revision"] == 1:
            raise RuntimeError("simulated initial Architect precommit loss")
        return real_commit(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        auxiliary_graph_store,
        "commit_auxiliary_graph_revision",
        lose_before_architect_commit,
    )
    with pytest.raises(RuntimeError, match="initial Architect precommit loss"):
        run_initial_auxiliary_planning(
            session_id=session_id,
            turn_id=source_turn_id,
            insession_task_id=task_id,
            provider=count_provider,
            ledger_store=store,
            emit=lambda _event: None,
        )
    assert provider_calls == 1

    continuation_turn_id = _accept_continuation_turn(
        session_id=session_id,
        prior_turn_id=source_turn_id,
        task_id=task_id,
    )
    monkeypatch.setattr(
        auxiliary_graph_store,
        "commit_auxiliary_graph_revision",
        real_commit,
    )
    recovered = run_initial_auxiliary_planning(
        session_id=session_id,
        turn_id=continuation_turn_id,
        insession_task_id=task_id,
        provider=lambda *_args, **_kwargs: pytest.fail(
            "settled initial Architect call reached Provider after handoff"
        ),
        ledger_store=store,
        emit=lambda _event: None,
    )

    assert recovered.status is AuxiliaryInitialPlanningStatus.PLANNED
    assert recovered.model_replayed is True
    assert recovered.bootstrapped is False
    assert recovered.architect_decision is not None
    logical = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=recovered.architect_decision.logical_call_id,
    )
    assert logical is not None
    assert logical.request.invocation_turn_id == source_turn_id
    assert len(logical.physical_attempts) == 1
    completion = planning_store.get_auxiliary_initial_planning_completion(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert completion is not None
    assert completion.binding.turn_id == continuation_turn_id
    assert (
        completion.binding.runtime_logical_request_binding_sha256
        == logical.request.binding_sha256
    )
    assert provider_calls == 1


def test_initial_architect_retryable_attempt_uses_same_logical_call_on_later_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, source_turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    real_request = controller.request_auxiliary_graph_architect
    logical_call_id: str | None = None

    def leave_retryable(
        architect_request: object,
        *,
        invocation_turn_id: str,
        provider: object,
        emit: object,
        deadline: object,
        durable_call: object,
    ) -> object:
        nonlocal logical_call_id
        del provider, emit, deadline
        assert invocation_turn_id == source_turn_id
        logical_call_id = str(architect_request.logical_call_id)  # type: ignore[attr-defined]
        durable_call.require_current_state()  # type: ignore[attr-defined]
        durable_call.reserve(turn_id=invocation_turn_id)  # type: ignore[attr-defined]
        physical_attempt = durable_call.begin_physical_attempt(  # type: ignore[attr-defined]
            turn_id=invocation_turn_id,
            max_physical_attempts=(
                durable_call.logical_request.max_physical_attempts  # type: ignore[attr-defined]
            ),
            output_repair_enabled=True,
        )
        durable_call.settle_physical_attempt(  # type: ignore[attr-defined]
            turn_id=invocation_turn_id,
            physical=physical_attempt,
            outcome="retryable_failure",
            result_fingerprint="a" * 64,
            error_code="MODEL_RATE_LIMIT",
        )
        raise RuntimeError("simulated initial retryable boundary")

    monkeypatch.setattr(
        controller,
        "request_auxiliary_graph_architect",
        leave_retryable,
    )
    with pytest.raises(RuntimeError, match="initial retryable boundary"):
        run_initial_auxiliary_planning(
            session_id=session_id,
            turn_id=source_turn_id,
            insession_task_id=task_id,
            provider=lambda *_args, **_kwargs: pytest.fail(
                "synthetic initial retry setup reached Provider"
            ),
            ledger_store=store,
            emit=lambda _event: None,
        )
    assert logical_call_id is not None

    continuation_turn_id = _accept_continuation_turn(
        session_id=session_id,
        prior_turn_id=source_turn_id,
        task_id=task_id,
    )
    monkeypatch.setattr(
        controller,
        "request_auxiliary_graph_architect",
        real_request,
    )
    provider_calls = 0
    physical = build_auxiliary_architect_structured_provider()

    def count_provider(*args: object, **kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return physical(*args, **kwargs)  # type: ignore[arg-type]

    recovered = run_initial_auxiliary_planning(
        session_id=session_id,
        turn_id=continuation_turn_id,
        insession_task_id=task_id,
        provider=count_provider,
        ledger_store=store,
        emit=lambda _event: None,
    )

    assert recovered.status is AuxiliaryInitialPlanningStatus.PLANNED
    assert recovered.model_attempts == 2
    assert recovered.model_replayed is False
    logical = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=logical_call_id,
    )
    assert logical is not None
    assert logical.request.invocation_turn_id == source_turn_id
    assert len(logical.physical_attempts) == 2
    assert (
        logical.physical_attempts[1].request.started_turn_id
        == continuation_turn_id
    )
    assert provider_calls == 1


@pytest.mark.parametrize("outcome", ["pending", "uncertain"])
def test_initial_architect_pending_or_uncertain_waits_after_turn_handoff(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
) -> None:
    session_id, source_turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    real_request = controller.request_auxiliary_graph_architect

    def leave_unreconciled(
        _architect_request: object,
        *,
        invocation_turn_id: str,
        provider: object,
        emit: object,
        deadline: object,
        durable_call: object,
    ) -> object:
        del provider, emit, deadline
        durable_call.require_current_state()  # type: ignore[attr-defined]
        durable_call.reserve(turn_id=invocation_turn_id)  # type: ignore[attr-defined]
        physical_attempt = durable_call.begin_physical_attempt(  # type: ignore[attr-defined]
            turn_id=invocation_turn_id,
            max_physical_attempts=(
                durable_call.logical_request.max_physical_attempts  # type: ignore[attr-defined]
            ),
            output_repair_enabled=True,
        )
        if outcome == "uncertain":
            durable_call.settle_physical_attempt(  # type: ignore[attr-defined]
                turn_id=invocation_turn_id,
                physical=physical_attempt,
                outcome="uncertain",
                result_fingerprint="b" * 64,
                error_code="MODEL_RESPONSE_UNCERTAIN",
            )
        raise RuntimeError(f"simulated initial {outcome} boundary")

    monkeypatch.setattr(
        controller,
        "request_auxiliary_graph_architect",
        leave_unreconciled,
    )
    with pytest.raises(RuntimeError, match=outcome):
        run_initial_auxiliary_planning(
            session_id=session_id,
            turn_id=source_turn_id,
            insession_task_id=task_id,
            provider=lambda *_args, **_kwargs: pytest.fail(
                "synthetic initial unresolved setup reached Provider"
            ),
            ledger_store=store,
            emit=lambda _event: None,
        )

    continuation_turn_id = _accept_continuation_turn(
        session_id=session_id,
        prior_turn_id=source_turn_id,
        task_id=task_id,
    )
    monkeypatch.setattr(
        controller,
        "request_auxiliary_graph_architect",
        real_request,
    )
    with pytest.raises(RuntimeModelCallWaitingExternal):
        run_initial_auxiliary_planning(
            session_id=session_id,
            turn_id=continuation_turn_id,
            insession_task_id=task_id,
            provider=lambda *_args, **_kwargs: pytest.fail(
                f"{outcome} initial Architect call was blindly resent"
            ),
            ledger_store=store,
            emit=lambda _event: None,
        )


def test_initial_architect_rejects_read_only_continuation_lane_before_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, source_turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    continuation_turn_id = _accept_continuation_turn(
        session_id=session_id,
        prior_turn_id=source_turn_id,
        task_id=task_id,
        execute_current=False,
    )

    with pytest.raises(AuxiliaryInitialPlanningError, match="executable Task lane"):
        run_initial_auxiliary_planning(
            session_id=session_id,
            turn_id=continuation_turn_id,
            insession_task_id=task_id,
            provider=lambda *_args, **_kwargs: pytest.fail(
                "read-only initial planning lane reached Provider"
            ),
            ledger_store=store,
            emit=lambda _event: None,
        )
    assert (
        auxiliary_graph_store.get_auxiliary_graph_for_task(
            session_id=session_id,
            insession_task_id=task_id,
        )
        is None
    )


def test_foreign_revision_two_without_completion_is_not_already_planned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    real_commit = auxiliary_graph_store.commit_auxiliary_graph_revision

    def omit_completion(**kwargs: object):
        if kwargs["expected_current_auxiliary_graph_revision"] == 1:
            kwargs.pop("initial_planning_completion")
        return real_commit(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        auxiliary_graph_store,
        "commit_auxiliary_graph_revision",
        omit_completion,
    )
    planned_without_receipt = run_initial_auxiliary_planning(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        ledger_store=store,
        emit=lambda _event: None,
    )
    assert planned_without_receipt.details.auxiliary_graph_revision == 2
    assert planning_store.get_auxiliary_initial_planning_completion(
        session_id=session_id,
        insession_task_id=task_id,
    ) is None

    with pytest.raises(
        AuxiliaryInitialPlanningError,
        match="no verifiable initial-planning completion",
    ):
        run_initial_auxiliary_planning(
            session_id=session_id,
            turn_id=turn_id,
            insession_task_id=task_id,
            provider=lambda *_args, **_kwargs: pytest.fail(
                "foreign revision two reached Provider"
            ),
            ledger_store=store,
            emit=lambda _event: None,
        )


def test_tampered_completion_receipt_cannot_authorize_already_planned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    run_initial_auxiliary_planning(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        ledger_store=store,
        emit=lambda _event: None,
    )

    with store._connect() as conn:
        row = conn.execute(
            "SELECT apply_id, result_json FROM "
            "insession_auxiliary_graph_revision_apply_receipts_v2 "
            "WHERE session_id=? AND insession_task_id=? "
            "AND committed_auxiliary_graph_revision=2",
            (session_id, task_id),
        ).fetchone()
        assert row is not None
        tampered = json.loads(str(row["result_json"]))
        tampered["initial_planning_completion"]["binding"][
            "bootstrap_structure_sha256"
        ] = "f" * 64
        tampered_json = json.dumps(
            tampered,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        conn.execute(
            "UPDATE insession_auxiliary_graph_revision_apply_receipts_v2 "
            "SET result_json=?, result_sha256=? WHERE apply_id=?",
            (
                tampered_json,
                hashlib.sha256(tampered_json.encode("utf-8")).hexdigest(),
                str(row["apply_id"]),
            ),
        )

    with pytest.raises(
        AuxiliaryInitialPlanningError,
        match="failed authority validation",
    ):
        run_initial_auxiliary_planning(
            session_id=session_id,
            turn_id=turn_id,
            insession_task_id=task_id,
            provider=lambda *_args, **_kwargs: pytest.fail(
                "tampered completion reached Provider"
            ),
            ledger_store=store,
            emit=lambda _event: None,
        )


def test_initial_planning_returns_terminal_failure_without_dispatchable_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)

    def provider(
        _system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        current = json.loads(user_content)["current_revision"]
        proposal = {
            "schema_version": "auxiliary-graph-revision-proposal-v2",
            "disposition": "terminal_fail",
            "expected_current_auxiliary_graph_revision": current[
                "auxiliary_graph_revision"
            ],
            "revision_reason": None,
            "structure": None,
            "explanation": "The authorized objective cannot be planned safely.",
            "blocking_gap_ids": [],
            "requested_user_question": None,
            "failure_reason": "No safe initial graph can satisfy the objective.",
        }
        return ModelResult(
            reply=json.dumps(proposal, ensure_ascii=False),
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    result = run_initial_auxiliary_planning(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        provider=provider,
        ledger_store=store,
        emit=lambda _event: None,
    )

    assert result.status is AuxiliaryInitialPlanningStatus.TERMINAL_FAILED
    assert result.failure_reason == "No safe initial graph can satisfy the objective."
    assert result.revision_commit is None
    assert result.details.auxiliary_graph_revision == 1
    assert result.details.nodes[0].local_node_key == "bootstrap_terminal"


def test_mounted_authority_drift_rejects_before_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    from personagraph.l2.auxiliary_execution.planning import controller

    real_freeze = controller.freeze_mounted_document_planning_authority
    freeze_calls = 0

    def freeze(**values):
        nonlocal freeze_calls
        freeze_calls += 1
        if freeze_calls > 1:
            raise RuntimeError("mounted scope changed")
        return real_freeze(**values)

    monkeypatch.setattr(
        controller,
        "freeze_mounted_document_planning_authority",
        freeze,
    )
    with pytest.raises(DurableModelCallStateGuardRejected):
        run_initial_auxiliary_planning(
            session_id=session_id,
            turn_id=turn_id,
            insession_task_id=task_id,
            provider=lambda *_args, **_kwargs: pytest.fail(
                "drifted authority reached Provider"
            ),
            ledger_store=store,
            emit=lambda _event: None,
        )

    details = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert details is not None
    assert details.auxiliary_graph_revision == 1
    assert details.nodes[0].local_node_key == "bootstrap_terminal"
