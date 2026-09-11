from __future__ import annotations

from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import terminal as terminal_store


def test_task_graph_facade_delegates_complete_operations(monkeypatch) -> None:
    deps = object()
    calls: list[tuple[str, object, dict[str, object]]] = []
    monkeypatch.setattr(
        task_graph_store.session_store,
        "current_store_deps",
        lambda: deps,
    )

    def record(name: str, result: object):
        def delegated(actual_deps, *args, **kwargs):
            assert actual_deps is deps
            calls.append((name, args, kwargs))
            return result

        return delegated

    monkeypatch.setattr(
        task_graph_store.task_records,
        "apply_insession_task_matches",
        record("apply", "applied"),
    )
    monkeypatch.setattr(
        task_graph_store.task_records,
        "get_insession_task_execution_lane_manifest",
        record("lane", "manifest"),
    )
    monkeypatch.setattr(
        task_graph_store.task_records,
        "get_insession_task_creation_source",
        record("source", "anchor"),
    )
    monkeypatch.setattr(
        task_graph_store.task_records,
        "get_insession_task_details",
        record("details", "task"),
    )
    monkeypatch.setattr(
        task_graph_store.semantic_base_records,
        "project_task_graph_semantic_base",
        record("semantic_base", "base"),
    )

    assert task_graph_store.apply_insession_task_matches(
        session_id="session",
        source_turn_id="turn",
        apply_id="apply",
        proposal=object(),  # type: ignore[arg-type]
        exposed_catalog_ids=("task",),
        expected_window_revision=3,
    ) == "applied"
    assert task_graph_store.get_insession_task_execution_lane_manifest(
        session_id="session",
        turn_id="turn",
    ) == "manifest"
    assert task_graph_store.get_insession_task_creation_source(
        session_id="session",
        insession_task_id="task",
    ) == "anchor"
    assert task_graph_store.get_insession_task_details(
        "session",
        "task",
    ) == "task"
    assert task_graph_store.project_task_graph_semantic_base(
        session_id="session",
        task_id="task",
        graph_revision=2,
    ) == "base"

    assert [name for name, _args, _kwargs in calls] == [
        "apply",
        "lane",
        "source",
        "details",
        "semantic_base",
    ]
    assert calls[3][1] == ("session", "task")
    assert calls[0][2]["expected_window_revision"] == 3


def test_terminal_facade_delegates_complete_operations(monkeypatch) -> None:
    deps = object()
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        terminal_store.session_store,
        "current_store_deps",
        lambda: deps,
    )

    def record(name: str, result: object):
        def delegated(actual_deps, **kwargs):
            assert actual_deps is deps
            calls.append((name, kwargs))
            return result

        return delegated

    monkeypatch.setattr(
        terminal_store.validation_records,
        "build_auxiliary_terminal_task_graph_validation_context",
        record("validation", "context"),
    )
    monkeypatch.setattr(
        terminal_store.validation_records,
        "project_auxiliary_terminal_semantic_support",
        record("support", "support"),
    )
    monkeypatch.setattr(
        terminal_store.seal_records,
        "seal_auxiliary_terminal_proposal",
        record("seal", "sealed"),
    )
    monkeypatch.setattr(
        terminal_store.seal_records,
        "get_auxiliary_terminal_proposal_receipt",
        record("receipt", "receipt"),
    )
    monkeypatch.setattr(
        terminal_store.commit_records,
        "commit_auxiliary_task_graph_proposal",
        record("commit", "committed"),
    )

    assert terminal_store.build_auxiliary_terminal_task_graph_validation_context(
        session_id="session",
        invocation_turn_id="turn",
        task_id="task",
    ) == "context"
    assert terminal_store.project_auxiliary_terminal_semantic_support(
        session_id="session",
        invocation_turn_id="turn",
        task_id="task",
    ) == "support"
    assert terminal_store.seal_auxiliary_terminal_proposal(
        command=object(),  # type: ignore[arg-type]
    ) == "sealed"
    assert terminal_store.get_auxiliary_terminal_proposal_receipt(
        session_id="session",
        terminal_proposal_receipt_id="receipt",
    ) == "receipt"
    assert terminal_store.commit_auxiliary_task_graph_proposal(
        command=object(),  # type: ignore[arg-type]
    ) == "committed"

    assert [name for name, _kwargs in calls] == [
        "validation",
        "support",
        "seal",
        "receipt",
        "commit",
    ]
    assert calls[0][1]["invocation_turn_id"] == "turn"
    assert calls[-1][1]["command"] is not None


def test_facade_aliases_preserve_persistence_contract_identity() -> None:
    assert (
        task_graph_store.TaskGraphSemanticBaseProjectionError
        is task_graph_store.semantic_base_records.TaskGraphSemanticBaseProjectionError
    )
    assert (
        terminal_store.SealAuxiliaryTerminalProposalCommand
        is terminal_store.seal_records.SealAuxiliaryTerminalProposalCommand
    )
    assert (
        terminal_store.CommitAuxiliaryTaskGraphProposalCommand
        is terminal_store.commit_records.CommitAuxiliaryTaskGraphProposalCommand
    )
