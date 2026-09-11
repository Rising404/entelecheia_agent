"""入口监督器任务准入阶段的单元与边界检查。"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import pytest

from personagraph.l2.task_graph.contracts import InSessionTaskCatalog
from personagraph.l2.task_graph.task_matching import (
    ExistingRootTaskMatchProposal,
    InSessionTaskMatchApplyResult,
    NewRootTaskMatchProposal,
)
from personagraph.runtime.entry.context.contracts import EntryContext
from personagraph.runtime.entry.ingress.model_contracts import EntryClassification
import personagraph.runtime.entry.application as entry_application
import personagraph.runtime.entry.routing.task_admission as admission_module
from personagraph.runtime.entry.ingress.contracts import (
    AuthoritativeRuntimeSnapshot,
    CapabilityCeiling,
    IngressDecision,
    IngressAnalysisFloor,
    TrustedTurnEnvelope,
)
from personagraph.runtime.entry.routing.task_admission import (
    admit_entry_task_matches,
)
from personagraph.runtime.turn.contracts import (
    AcceptedEntryTurn,
    EntryExecutionSnapshot,
)


class _AdmissionStore:
    def __init__(
        self,
        *,
        apply_result: InSessionTaskMatchApplyResult | None = None,
        lose_first_response: bool = False,
    ) -> None:
        self.apply_result = apply_result or InSessionTaskMatchApplyResult(
            status="applied",
            turn_task_link_revision=1,
        )
        self.lose_first_response = lose_first_response
        self.apply_commands: list[dict[str, object]] = []

    def get_insession_task_details(self, *_args: object) -> object:
        return SimpleNamespace(status=SimpleNamespace(value="active"))

    def get_insession_task_execution_lane_manifest(self, **_kwargs: object) -> object:
        return SimpleNamespace(lanes=())

    def list_pending_user_questions(self, **_kwargs: object) -> tuple[object, ...]:
        return ()

    def apply_insession_task_matches(
        self, **kwargs: object
    ) -> InSessionTaskMatchApplyResult:
        self.apply_commands.append(kwargs)
        if self.lose_first_response and len(self.apply_commands) == 1:
            raise OSError("task-match response lost")
        return self.apply_result


def _accepted() -> AcceptedEntryTurn:
    return AcceptedEntryTurn(
        session_id="session-1",
        turn_id="turn-1",
        client_request_id="request-1",
        user_input="Create a task",
        attachment_ids=(),
        window_revision=7,
        replayed=False,
        execution_snapshot=EntryExecutionSnapshot.create(
            features={},
            post_commit_job_kinds=(),
        ),
    )


def _context(*, truncated: bool = False) -> EntryContext:
    return EntryContext(
        envelope=TrustedTurnEnvelope(
            turn_id="turn-1",
            session_id="session-1",
            received_at=datetime.now(timezone.utc),
            input_kind="user_text",
            user_text="Create a task",
        ),
        snapshot=AuthoritativeRuntimeSnapshot(),
        ceiling=CapabilityCeiling(),
        estimated_input_tokens=1,
        history_pairs=(),
        session_summary=None,
        task_catalog=InSessionTaskCatalog(truncated=truncated),
    )


def _classification(
    *,
    processing_level: Literal["L0", "L1", "L2"] = "L0",
    roots: int = 1,
) -> EntryClassification:
    return EntryClassification(
        processing_level=processing_level,
        task_matches=tuple(
            NewRootTaskMatchProposal(
                local_key=f"task-{index}",
                title=f"Task {index}",
                objective="Split the Entry supervisor phase",
                source_excerpt="Create a task",
            )
            for index in range(roots)
        ),
    )


def _ingress(
    floor: IngressAnalysisFloor = IngressAnalysisFloor.L0,
) -> IngressDecision:
    return cast(IngressDecision, SimpleNamespace(analysis_floor=floor))


class _ContextWhoseTaskCatalogMustNotBeRead:
    @property
    def task_catalog(self) -> object:
        raise AssertionError("L1 admission must not read the durable Task catalog")


def _context_with_forbidden_task_catalog_read() -> EntryContext:
    return cast(EntryContext, _ContextWhoseTaskCatalogMustNotBeRead())


def test_admission_fails_closed_instead_of_upgrading_l0_task_mutation() -> None:
    store = _AdmissionStore(
        apply_result=InSessionTaskMatchApplyResult(
            status="applied",
            created_insession_task_ids_by_local_key={"task-0": "task-1"},
            related_insession_task_ids=("task-1",),
            turn_task_link_revision=3,
            window_state_version=12,
        )
    )
    invalid_classification = EntryClassification.model_construct(
        processing_level="L0",
        task_matches=(
            NewRootTaskMatchProposal(
                local_key="task-0",
                title="Task 0",
                objective="Split the Entry supervisor phase",
                source_excerpt="Create a task",
            ),
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="L0 classification cannot request durable Task mutation",
    ):
        admit_entry_task_matches(
            accepted=_accepted(),
            classification=invalid_classification,
            context=_context(),
            ingress=_ingress(),
            expected_window_revision=7,
            store=store,
        )

    assert store.apply_commands == []


def test_admission_keeps_empty_batches_write_free() -> None:
    empty_store = _AdmissionStore()
    empty = admit_entry_task_matches(
        accepted=_accepted(),
        classification=_classification(roots=0),
        context=_context(truncated=True),
        ingress=_ingress(),
        expected_window_revision=7,
        store=empty_store,
    )

    assert empty.processing_level == "L0"
    assert empty.applied is None
    assert empty.related_insession_task_ids == ()
    assert empty.window_revision == 7
    assert empty_store.apply_commands == []


def test_l1_empty_admission_returns_no_association_without_reading_catalog() -> None:
    store = _AdmissionStore()

    outcome = admit_entry_task_matches(
        accepted=_accepted(),
        classification=EntryClassification(
            processing_level="L1",
            task_matches=(),
        ),
        context=_context_with_forbidden_task_catalog_read(),
        ingress=_ingress(),
        expected_window_revision=7,
        store=store,
    )

    assert outcome.processing_level == "L1"
    assert outcome.related_insession_task_ids == ()
    assert outcome.applied is None
    assert outcome.window_revision == 7
    assert store.apply_commands == []


def test_l1_admission_rejects_nonempty_matches_that_bypass_schema() -> None:
    invalid_classification = EntryClassification.model_construct(
        processing_level="L1",
        task_matches=(
            ExistingRootTaskMatchProposal(
                insession_task_id="task-existing",
                source_excerpt="参考已有任务，但不要续接",
                execute_current=False,
            ),
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="^L1 classification cannot reference durable Tasks$",
    ):
        admit_entry_task_matches(
            accepted=_accepted(),
            classification=invalid_classification,
            context=_context_with_forbidden_task_catalog_read(),
            ingress=_ingress(),
            expected_window_revision=7,
            store=_AdmissionStore(),
        )


def test_admission_projects_the_l2_persistence_receipt() -> None:
    store = _AdmissionStore(
        apply_result=InSessionTaskMatchApplyResult(
            status="replayed",
            related_insession_task_ids=("task-1",),
            turn_task_link_revision=3,
        ),
    )

    outcome = admit_entry_task_matches(
        accepted=_accepted(),
        classification=_classification(processing_level="L2"),
        context=_context(),
        ingress=_ingress(),
        expected_window_revision=7,
        store=store,
    )

    assert outcome.applied is store.apply_result
    assert outcome.related_insession_task_ids == ("task-1",)
    assert outcome.window_revision == 7
    assert len(store.apply_commands) == 1


def test_admission_has_no_entry_lifecycle_or_work_run_authority() -> None:
    source = Path(admission_module.__file__).read_text(encoding="utf-8")
    module = ast.parse(source)
    relative_imports = {
        node.module
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.level == 1
    }
    called_names = {
        node.func.id
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    called_attributes = {
        node.func.attr
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert "entry" not in relative_imports
    assert (
        not {
            "new_turn_event",
            "project_turn_event",
            "_finalize_formal_reply",
            "_incomplete_turn",
            "_execute_auxiliary_production_chain",
        }
        & called_names
    )
    assert (
        not {
            "advance_turn_execution_window",
            "append_runtime_turn_event",
            "finalize_turn_execution",
            "finalize_verified_turn_execution",
            "finalize_authoritative_referenced_turn_execution",
            "mark_authoritative_no_public_turn_stop",
        }
        & called_attributes
    )


def test_entry_keeps_admission_between_supervisor_events_and_route_selection() -> None:
    source = Path(entry_application.__file__).read_text(encoding="utf-8")
    module = ast.parse(source)
    execute_new_turn = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_execute_new_turn"
    )
    function_source = ast.get_source_segment(source, execute_new_turn)
    assert function_source is not None

    admission_index = function_source.index("admission = admit_entry_task_matches(")
    route_index = function_source.index(
        "processing_route = select_entry_processing_route(",
        admission_index,
    )
    supervisor_started = function_source.rfind(
        "stage=RuntimeStage.SUPERVISOR",
        0,
        admission_index,
    )
    supervisor_completed = function_source.index(
        "stage=RuntimeStage.SUPERVISOR",
        admission_index,
    )

    assert supervisor_started >= 0
    assert supervisor_started < admission_index < route_index < supervisor_completed
    assert (
        "status=TurnEventStatus.STARTED"
        in function_source[supervisor_started:admission_index]
    )
    assert "status=TurnEventStatus.COMPLETED" in function_source[supervisor_completed:]
