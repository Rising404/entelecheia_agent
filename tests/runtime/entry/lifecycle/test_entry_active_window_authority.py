"""入口活跃窗口读取器的行为与所有权覆盖。"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType

import pytest

from personagraph.runtime.entry import application as entry_application
from personagraph.runtime.entry.lifecycle import active_window as authority_module
from personagraph.runtime.entry.lifecycle.active_window import (
    load_authoritative_active_turn_window,
)
from personagraph.runtime.turn.contracts import (
    AcceptedEntryTurn,
    EntryExecutionSnapshot,
)


def _accepted() -> AcceptedEntryTurn:
    return AcceptedEntryTurn(
        session_id="session-1",
        turn_id="turn-1",
        client_request_id="request-1",
        user_input="continue the accepted turn",
        attachment_ids=(),
        window_revision=7,
        replayed=False,
        execution_snapshot=EntryExecutionSnapshot.create(
            features={},
            post_commit_job_kinds=(),
        ),
    )


def _window(**overrides: object) -> dict[str, object]:
    window: dict[str, object] = {
        "turn_id": "turn-1",
        "window_state": "active",
        "lease_owner": "lease-1",
        "state_version": 7,
    }
    window.update(overrides)
    return window


class _WindowStore:
    def __init__(self, window: object, *, fail: bool = False) -> None:
        self.window = window
        self.fail = fail
        self.calls = 0

    def inspect_turn_execution(self, session_id: str) -> dict[str, object]:
        self.calls += 1
        assert session_id == "session-1"
        if self.fail:
            raise OSError("execution window read failed")
        return {"window": self.window}


def test_valid_active_window_is_returned_without_copying_or_a_second_read() -> None:
    window = _window()
    store = _WindowStore(window)

    result = load_authoritative_active_turn_window(
        accepted=_accepted(),
        store=store,
    )

    assert result is window
    assert store.calls == 1


def test_optional_lease_check_is_strict_only_when_requested() -> None:
    window = _window(lease_owner="lease-current")

    assert load_authoritative_active_turn_window(
        accepted=_accepted(),
        store=_WindowStore(window),
    ) is window
    assert load_authoritative_active_turn_window(
        accepted=_accepted(),
        store=_WindowStore(window),
        expected_lease_owner="lease-current",
    ) is window
    assert load_authoritative_active_turn_window(
        accepted=_accepted(),
        store=_WindowStore(window),
        expected_lease_owner="lease-other",
    ) is None

    for empty_lease_owner in ("", None):
        empty_owner_window = _window(lease_owner=empty_lease_owner)
        assert load_authoritative_active_turn_window(
            accepted=_accepted(),
            store=_WindowStore(empty_owner_window),
            expected_lease_owner="",
        ) is empty_owner_window

    missing_owner_window = _window()
    del missing_owner_window["lease_owner"]
    assert load_authoritative_active_turn_window(
        accepted=_accepted(),
        store=_WindowStore(missing_owner_window),
        expected_lease_owner="",
    ) is missing_owner_window


@pytest.mark.parametrize(
    "window",
    (
        None,
        [],
        MappingProxyType(_window()),
        _window(turn_id="other-turn"),
        _window(window_state="interrupted"),
        _window(state_version=0),
        _window(state_version=-1),
        _window(state_version=True),
        _window(state_version="7"),
        _window(state_version=7.0),
    ),
)
def test_malformed_or_nonowning_window_fails_closed(window: object) -> None:
    store = _WindowStore(window)

    assert load_authoritative_active_turn_window(
        accepted=_accepted(),
        store=store,
        expected_lease_owner="lease-1",
    ) is None
    assert store.calls == 1


def test_read_exception_fails_closed_after_the_single_authoritative_read() -> None:
    store = _WindowStore(_window(), fail=True)

    assert load_authoritative_active_turn_window(
        accepted=_accepted(),
        store=store,
    ) is None
    assert store.calls == 1


def test_active_window_owner_cold_imports_only_the_turn_contract() -> None:
    code = """
import json
import sys
import personagraph.runtime.entry.lifecycle.active_window
allowed = {
    'personagraph',
    'personagraph.entry_execution_snapshot',
    'personagraph.runtime',
    'personagraph.runtime.entry',
    'personagraph.runtime.entry.lifecycle',
    'personagraph.runtime.entry.lifecycle.active_window',
    'personagraph.runtime.turn',
    'personagraph.runtime.turn.contracts',
}
blocked = {
    'personagraph.model_io.gateway',
    'personagraph.runtime.entry.application',
    'personagraph.runtime.entry.ports',
    'personagraph.runtime.turn_events',
    'personagraph.session',
    'personagraph.session.store',
    'pydantic',
}
loaded = {
    name
    for name in sys.modules
    if name == 'personagraph' or name.startswith('personagraph.')
}
print(json.dumps({
    'blocked': sorted(blocked & set(sys.modules)),
    'unexpected': sorted(loaded - allowed),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {"blocked": [], "unexpected": []}


def test_owner_has_one_narrow_read_port_and_entry_keeps_the_exact_alias() -> None:
    source = Path(authority_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports |= {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    store_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "store"
    }
    read_port = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EntryActiveWindowReadPort"
    )
    port_methods = {
        node.name for node in read_port.body if isinstance(node, ast.FunctionDef)
    }

    assert imports == {"__future__", "typing", "turn.contracts"}
    assert store_calls == {"inspect_turn_execution"}
    assert port_methods == store_calls
    assert entry_application._authoritative_active_turn_window is (
        authority_module.load_authoritative_active_turn_window
    )

    entry_tree = ast.parse(
        Path(entry_application.__file__).read_text(encoding="utf-8")
    )
    assert not any(
        isinstance(node, ast.FunctionDef)
        and node.name == "_authoritative_active_turn_window"
        for node in entry_tree.body
    )
