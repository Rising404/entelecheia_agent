import ast
import inspect
from pathlib import Path

import pytest

from personagraph.session import history_store_facade, store
from personagraph.session.persistence.history import turns
from personagraph.session.persistence.turns import turn_execution
from tests.helpers.session_records import append_test_turn


def test_turn_records_depend_only_on_narrow_store_support():
    tree = ast.parse(Path(turns.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(
        forbidden in module
        for module in imported
        for forbidden in ("paths", "graph", "memory", "context_store")
    )
    assert "store" not in imported


def test_test_turn_fixture_rejects_unknown_session(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "sessions.sqlite")

    with pytest.raises(ValueError, match="unknown session: missing"):
        append_test_turn("missing", "user", "must not be stored")

    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM session_turns").fetchone()[0] == 0


def test_production_does_not_expose_a_non_atomic_single_turn_writer():
    assert not hasattr(store, "append_turn")
    assert not hasattr(turns, "append_turn")
    assert not hasattr(history_store_facade.HistoryStoreFacade, "append_turn")


def test_turn_finalization_contract_has_no_retired_history_generation_parameter():
    finalizers = (
        store.finalize_turn_execution,
        store.finalize_verified_turn_execution,
        store.finalize_verified_turn_deliveries,
        store.finalize_referenced_turn_execution,
        store.finalize_authoritative_referenced_turn_execution,
        turn_execution.finalize_turn_execution,
        turn_execution.finalize_verified_turn_execution,
        turn_execution.finalize_verified_turn_deliveries,
        turn_execution.finalize_referenced_turn_execution,
        turn_execution.finalize_authoritative_referenced_turn_execution,
        turns.append_assistant_for_accepted_input_in_transaction,
        turns.append_verified_assistant_deliveries_for_accepted_input_in_transaction,
        turns.append_referenced_assistant_for_accepted_input_in_transaction,
    )
    assert all(
        "history_retrieval_data_version"
        not in inspect.signature(finalizer).parameters
        and "retrieval_data_version"
        not in inspect.signature(finalizer).parameters
        for finalizer in finalizers
    )


def test_test_fixture_can_seed_one_deliberately_incomplete_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "sessions.sqlite")
    session_id = store.create_session("Entelecheia")

    turn_idx = append_test_turn(session_id, "user", "fixture only")

    assert turn_idx == 0
    assert store.get_turn(session_id, turn_idx)["content"] == "fixture only"
