from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from personagraph.runtime.l1 import entry_lane
from personagraph.runtime.turn_deadline import TurnDeadline


class _Store:
    def __init__(self) -> None:
        self.created: dict[str, object] | None = None

    def create_l1_turn_run(self, **kwargs: object) -> dict[str, object]:
        self.created = kwargs
        return {
            "window": {"state_version": 8},
            "run": {"l1_turn_run_id": "run-from-store"},
        }


def _accepted() -> SimpleNamespace:
    return SimpleNamespace(
        session_id="session-1",
        turn_id="turn-1",
        execution_snapshot=SimpleNamespace(
            file_retrieval_data_version="files-1",
            session_retrieval_data_version="history-1",
            session_retrieval_assistant_turn_cutoff="assistant-7",
        ),
    )


def test_prepare_l1_entry_lane_freezes_and_persists_one_bootstrap(
    monkeypatch,
) -> None:
    tool_runtime = SimpleNamespace(
        catalog_snapshot_json='{"catalog":1}',
        catalog_snapshot_sha256="catalog-sha",
    )
    execution_config = SimpleNamespace(
        snapshot_json='{"execution":1}',
        snapshot_sha256="execution-sha",
    )
    corpus_manifest = SimpleNamespace(
        manifest_json='{"corpus":1}',
        manifest_sha256="corpus-sha",
    )
    seen: dict[str, object] = {}

    def build_runtime(session_id: str, **kwargs: object) -> object:
        seen["runtime"] = (session_id, kwargs)
        return tool_runtime

    monkeypatch.setattr(entry_lane, "build_l1_tool_runtime", build_runtime)
    monkeypatch.setattr(
        entry_lane,
        "freeze_l1_execution_config",
        lambda features: execution_config,
    )
    monkeypatch.setattr(
        entry_lane,
        "derive_l1_turn_run_id",
        lambda **_kwargs: "derived-run",
    )
    monkeypatch.setattr(
        entry_lane,
        "freeze_l1_corpus_manifest",
        lambda **kwargs: seen.setdefault("corpus", kwargs) and corpus_manifest,
    )
    store = _Store()
    context = SimpleNamespace(attachments=object())

    prepared = entry_lane.prepare_l1_entry_lane(
        accepted=_accepted(),
        context=context,
        features={
            "execution_findings_enabled": False,
            "l1_max_attempts": 17,
            "l1_max_tool_calls_per_attempt": 6,
        },
        initial_turn_window_revision=4,
        routing_policy_snapshot_hash="routing-sha",
        deadline=TurnDeadline.starting_now(60.0),
        store=store,
        lease_owner="lease-1",
    )

    assert prepared.l1_turn_run_id == "run-from-store"
    assert prepared.turn_window_revision == 8
    assert prepared.tool_runtime is tool_runtime
    assert prepared.corpus_manifest is corpus_manifest
    assert seen["runtime"] == (
        "session-1",
        {
            "turn_id": "turn-1",
            # L1 的阶段笔记是固定步骤，旧 feature 值不能关闭伴生台账。
            "execution_findings_enabled": True,
            "execution_features": {
                "execution_findings_enabled": False,
                "l1_max_attempts": 17,
                "l1_max_tool_calls_per_attempt": 6,
            },
            "file_retrieval_data_version": "files-1",
            "session_retrieval_data_version": "history-1",
            "session_retrieval_assistant_turn_cutoff": "assistant-7",
        },
    )
    assert seen["corpus"]["attachments"] is context.attachments
    assert store.created is not None
    assert store.created["l1_turn_run_id"] == "derived-run"
    assert store.created["routing_policy_snapshot_hash"] == "routing-sha"
    assert store.created["expected_window_revision"] == 4
    assert store.created["expected_lease_owner"] == "lease-1"
    assert store.created["max_attempts"] == 17
    assert store.created["max_tool_calls_per_attempt"] == 6
    assert datetime.fromisoformat(str(store.created["deadline_at"])).utcoffset() is not None


def test_run_l1_entry_lane_only_adapts_controller_inputs(monkeypatch) -> None:
    expected = object()
    seen: dict[str, object] = {}

    def run_controller(**kwargs: object) -> object:
        seen.update(kwargs)
        return expected

    monkeypatch.setattr(entry_lane, "run_l1_turn", run_controller)
    accepted = _accepted()
    context = SimpleNamespace()
    deadline = TurnDeadline.starting_now(60.0)
    store = object()
    tool_runtime = object()
    corpus_manifest = object()

    result = entry_lane.run_l1_entry_lane(
        accepted=accepted,
        context=context,
        l1_turn_run_id="run-1",
        initial_turn_window_revision=9,
        deadline=deadline,
        features={
            "l1_max_attempts": 11,
            "l1_max_tool_calls_per_attempt": 5,
        },
        emit=lambda _event: None,
        store=store,
        lease_owner="lease-1",
        frozen_tool_runtime=tool_runtime,
        frozen_corpus_manifest=corpus_manifest,
    )

    assert result is expected
    assert seen["accepted"] is accepted
    assert seen["context"] is context
    assert seen["store"] is store
    assert seen["deadline"] is deadline
    assert seen["max_attempts"] == 11
    assert seen["max_tool_calls_per_attempt"] == 5
    assert seen["frozen_tool_runtime"] is tool_runtime
    assert seen["frozen_corpus_manifest"] is corpus_manifest


def test_l1_entry_lane_has_no_runtime_entry_or_l2_dependency() -> None:
    source = Path(entry_lane.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    imports.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert not any("runtime.entry" in module for module in imports)
    assert not any("personagraph.l2" in module for module in imports)
