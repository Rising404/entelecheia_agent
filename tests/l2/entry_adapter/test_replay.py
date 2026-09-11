"""L2 replay receipt/manifest 读取适配器的单元与边界覆盖。"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from personagraph.l2.entry_adapter import replay as replay_module
from personagraph.l2.entry_adapter.replay import (
    inspect_task_execution_lane_manifest,
    require_task_execution_lane_manifest,
)


class _ReplayStore:
    def __init__(self, deps: object | None = None) -> None:
        self.deps = object() if deps is None else deps
        self.calls: list[str] = []

    def current_store_deps(self) -> object:
        self.calls.append("deps")
        return self.deps


class _LaneStore:
    def __init__(
        self,
        *,
        has_receipt: bool = True,
        manifest: object = SimpleNamespace(lanes=()),
        fail_on: frozenset[str] = frozenset(),
    ) -> None:
        self.has_receipt = has_receipt
        self.manifest = manifest
        self.fail_on = fail_on
        self.calls: list[str] = []

    def has_turn_task_execution_lane_receipt(
        self,
        session_id: str,
        turn_id: str,
    ) -> bool:
        self.calls.append("receipt")
        if "receipt" in self.fail_on:
            raise OSError("lost receipt read")
        assert (session_id, turn_id) == ("session-1", "turn-1")
        return self.has_receipt

    def get_insession_task_execution_lane_manifest(
        self,
        *,
        session_id: str,
        turn_id: str,
    ) -> object:
        self.calls.append("manifest")
        if "manifest" in self.fail_on:
            raise OSError("lost manifest read")
        assert (session_id, turn_id) == ("session-1", "turn-1")
        return self.manifest


def test_require_manifest_stops_at_receipt_gate() -> None:
    lane_store = _LaneStore(has_receipt=False)

    with pytest.raises(LookupError, match="no Task execution lane receipt"):
        require_task_execution_lane_manifest(
            replay_store=object(),
            task_execution_lane_store=lane_store,
            session_id="session-1",
            turn_id="turn-1",
        )

    assert lane_store.calls == ["receipt"]


@pytest.mark.parametrize("fail_on", ("receipt", "manifest"))
def test_inspect_manifest_fails_closed_on_unreadable_authority(
    fail_on: str,
) -> None:
    lane_store = _LaneStore(fail_on=frozenset({fail_on}))

    assert inspect_task_execution_lane_manifest(
        replay_store=object(),
        task_execution_lane_store=lane_store,
        session_id="session-1",
        turn_id="turn-1",
    ) == (False, None)


def test_default_store_stops_before_loading_manifest_without_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from personagraph.session.persistence.turns import entry_replay

    deps = object()
    replay_store = _ReplayStore(deps)
    calls: list[object] = []

    def has_receipt(
        actual_deps: object,
        session_id: str,
        turn_id: str,
    ) -> bool:
        calls.append((actual_deps, session_id, turn_id))
        return False

    def unexpected_manifest(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("L2 manifest owner must stay unloaded")

    monkeypatch.setattr(
        entry_replay,
        "has_turn_task_execution_lane_receipt",
        has_receipt,
    )
    monkeypatch.setattr(
        replay_module._SessionTaskExecutionLaneStore,
        "get_insession_task_execution_lane_manifest",
        unexpected_manifest,
    )

    assert inspect_task_execution_lane_manifest(
        replay_store=replay_store,
        task_execution_lane_store=None,
        session_id="session-1",
        turn_id="turn-1",
    ) == (False, None)
    assert replay_store.calls == ["deps"]
    assert calls == [(deps, "session-1", "turn-1")]


def test_default_store_delegates_a_present_receipt_to_task_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from personagraph.session import l2_store
    from personagraph.session.persistence.turns import entry_replay

    deps = object()
    calls: list[object] = []
    manifest = object()

    def has_receipt(
        actual_deps: object,
        session_id: str,
        turn_id: str,
    ) -> bool:
        calls.append(("receipt", actual_deps, session_id, turn_id))
        return True

    def get_manifest(**kwargs: object) -> object:
        calls.append(("manifest", kwargs))
        return manifest

    monkeypatch.setattr(
        entry_replay,
        "has_turn_task_execution_lane_receipt",
        has_receipt,
    )
    monkeypatch.setattr(
        l2_store,
        "task_graph",
        SimpleNamespace(
            get_insession_task_execution_lane_manifest=get_manifest,
        ),
        raising=False,
    )

    assert inspect_task_execution_lane_manifest(
        replay_store=_ReplayStore(deps),
        task_execution_lane_store=None,
        session_id="session-1",
        turn_id="turn-1",
    ) == (True, manifest)
    assert calls == [
        ("receipt", deps, "session-1", "turn-1"),
        (
            "manifest",
            {"session_id": "session-1", "turn_id": "turn-1"},
        ),
    ]


def test_l2_replay_owner_has_no_runtime_entry_reverse_dependency() -> None:
    source = Path(replay_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    protocol = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "L2ReplayTaskExecutionLaneStorePort"
    )
    session_adapter = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "_SessionTaskExecutionLaneStore"
    )

    assert "personagraph.runtime.entry" not in source
    assert {
        node.name for node in protocol.body if isinstance(node, ast.FunctionDef)
    } == {
        "has_turn_task_execution_lane_receipt",
        "get_insession_task_execution_lane_manifest",
    }
    assert {
        (node.level, node.module)
        for node in ast.walk(session_adapter)
        if isinstance(node, ast.ImportFrom)
    } == {
        (3, "session.l2_store"),
        (3, "session.persistence.turns.entry_replay"),
    }
