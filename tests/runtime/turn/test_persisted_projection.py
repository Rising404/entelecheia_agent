"""Lane-neutral persisted Turn projection tests."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from personagraph.runtime.turn import persisted_projection as projection
from personagraph.runtime.turn.contracts import EntryExecutionSnapshot


def _persisted_turn() -> dict[str, object]:
    snapshot = EntryExecutionSnapshot.create(
        features={"file_retrieval_read_enabled": True},
        post_commit_job_kinds=("session_summary",),
    )
    return {
        "execution_snapshot_json": snapshot.to_json(),
        "execution_snapshot_sha256": snapshot.sha256,
    }


def test_closed_persisted_vocabularies_keep_the_original_behavior() -> None:
    assert projection.require_persisted_mapping({"turn": {}}, "turn") == {}
    assert projection.optional_text(None) is None
    assert projection.optional_text(False) == "False"
    assert projection.require_entry_turn_status("completed") == "completed"
    assert projection.require_processing_level(None) is None
    assert projection.require_processing_level("L0") == "L0"
    assert projection.require_processing_level("L1") == "L1"
    assert projection.require_entry_window_state("post_commit_pending") == (
        "post_commit_pending"
    )

    with pytest.raises(RuntimeError, match="^missing persisted turn$"):
        projection.require_persisted_mapping({"turn": []}, "turn")
    with pytest.raises(RuntimeError, match="^unexpected runtime Turn status: L1$"):
        projection.require_entry_turn_status("L1")
    with pytest.raises(RuntimeError, match="^unexpected processing level: L3$"):
        projection.require_processing_level("L3")
    with pytest.raises(
        RuntimeError, match="^unexpected execution Window state: unknown$"
    ):
        projection.require_entry_window_state("unknown")


def test_execution_snapshot_is_hash_authenticated() -> None:
    turn = _persisted_turn()
    snapshot = projection.execution_snapshot_from_persisted_turn(turn)

    assert snapshot.features["file_retrieval_read_enabled"] is True
    assert snapshot.post_commit_job_kinds == ("session_summary",)

    turn["execution_snapshot_sha256"] = "0" * 64
    with pytest.raises(
        RuntimeError,
        match="accepted Turn execution snapshot authentication failed",
    ):
        projection.execution_snapshot_from_persisted_turn(turn)


@pytest.mark.parametrize(
    ("turn", "error"),
    (
        ({}, "accepted Turn is missing its execution snapshot"),
        (
            {"execution_snapshot_json": "{}"},
            "accepted Turn execution snapshot is incomplete",
        ),
    ),
)
def test_execution_snapshot_rejects_missing_or_incomplete_storage(
    turn: dict[str, object],
    error: str,
) -> None:
    with pytest.raises(RuntimeError, match=f"^{error}$"):
        projection.execution_snapshot_from_persisted_turn(turn)


def test_shared_projection_imports_no_entry_lane_or_storage_runtime() -> None:
    code = """
import json
import sys
import personagraph.runtime.turn.persisted_projection
blocked_prefixes = (
    'personagraph.l2',
    'personagraph.model_io',
    'personagraph.runtime.entry',
    'personagraph.runtime.l1',
    'personagraph.session',
)
print(json.dumps(sorted(
    name
    for name in sys.modules
    if name.startswith(blocked_prefixes)
)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []


def test_shared_projection_has_only_the_cold_turn_contract_dependency() -> None:
    tree = ast.parse(Path(projection.__file__).read_text(encoding="utf-8"))
    imports = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }

    assert imports == {"__future__", "contracts"}
