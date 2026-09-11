"""纯入口任务目录投影策略的边界覆盖。"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from personagraph.l2.task_graph.contracts import InSessionTaskCatalogItem
from personagraph.runtime.entry.context import application as assembly_module
from personagraph.runtime.entry.context import task_catalog_projection as projection_module
from personagraph.runtime.entry.context.task_catalog_projection import (
    project_entry_task_catalog_items,
)
from personagraph.runtime.entry.routing.policy import (
    TurnRoutingPolicy,
    freeze_turn_routing_policy,
)
from personagraph.runtime.turn.contracts import (
    AcceptedEntryTurn,
    EntryExecutionSnapshot,
)
from personagraph.session.entry_task_contracts import (
    EntryPendingTaskQuestion,
    EntryTaskCatalogItem,
)


def _item(
    task_id: str,
    *,
    pending_user_question: str | None = None,
) -> EntryTaskCatalogItem:
    return EntryTaskCatalogItem(
        insession_task_id=task_id,
        goal_summary=f"goal for {task_id}",
        status="active",
        current_graph_revision=1,
        pending_user_question=pending_user_question,
    )


def _pending(task_id: object, question: object) -> object:
    if isinstance(task_id, str) and task_id and isinstance(question, str) and question:
        return EntryPendingTaskQuestion(
            insession_task_id=task_id,
            question=question,
        )
    return SimpleNamespace(insession_task_id=task_id, question=question)


def test_exact_one_question_enriches_and_stably_prioritizes_catalog_items() -> None:
    plain_first = _item("task-plain-first")
    existing_zero = _item("task-existing-zero", pending_user_question="keep zero")
    exact_one = _item("task-exact-one")
    plain_second = _item("task-plain-second")
    existing_multiple = _item(
        "task-existing-multiple",
        pending_user_question="keep multiple",
    )
    existing_last = _item("task-existing-last", pending_user_question="keep last")

    result = project_entry_task_catalog_items(
        catalog_items=(
            plain_first,
            existing_zero,
            exact_one,
            plain_second,
            existing_multiple,
            existing_last,
        ),
        pending_user_questions=(
            _pending("task-exact-one", "the only answerable question"),
            _pending("task-existing-multiple", "same duplicate"),
            _pending("task-existing-multiple", "same duplicate"),
        ),
    )

    assert tuple(item.insession_task_id for item in result) == (
        "task-existing-zero",
        "task-exact-one",
        "task-existing-multiple",
        "task-existing-last",
        "task-plain-first",
        "task-plain-second",
    )
    assert result[0] is existing_zero
    assert result[1] is not exact_one
    assert result[1].pending_user_question == "the only answerable question"
    assert result[2] is existing_multiple
    assert result[2].pending_user_question == "keep multiple"
    assert result[3] is existing_last
    assert result[4] is plain_first
    assert result[5] is plain_second


def test_only_nonempty_string_pending_facts_enrich_items_without_normalizing_text() -> None:
    valid = _item("task-valid")
    whitespace = _item(" ")
    existing = _item("task-existing", pending_user_question="keep existing")

    result = project_entry_task_catalog_items(
        catalog_items=(valid, whitespace, existing),
        pending_user_questions=(
            _pending("task-valid", ""),
            _pending("task-valid", 0),
            _pending(0, "wrong task id type"),
            {"insession_task_id": "task-valid", "question": "mapping ignored"},
            SimpleNamespace(insession_task_id=None, question="missing task id"),
            _pending(" ", " "),
        ),
    )

    assert tuple(item.insession_task_id for item in result) == (
        " ",
        "task-existing",
        "task-valid",
    )
    assert result[0] is not whitespace
    assert result[0].pending_user_question == " "
    assert result[1] is existing
    assert result[1].pending_user_question == "keep existing"
    assert result[2] is valid
    assert result[2].pending_user_question is None


def test_malformed_pending_attribute_failures_keep_existing_getattr_behavior() -> None:
    class ExplosivePending:
        @property
        def insession_task_id(self) -> object:
            raise RuntimeError("pending task identity lookup failed")

    with pytest.raises(RuntimeError, match="^pending task identity lookup failed$"):
        project_entry_task_catalog_items(
            catalog_items=(_item("task-1"),),
            pending_user_questions=(ExplosivePending(),),
        )


def test_projection_owner_has_only_contract_and_typing_dependencies() -> None:
    tree = ast.parse(Path(projection_module.__file__).read_text(encoding="utf-8"))
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

    assert imports == {"__future__", "typing"}


def test_projection_owner_cold_import_does_not_load_entry_or_session_layers() -> None:
    code = """
import json
import sys
import personagraph.runtime.entry.context.task_catalog_projection
allowed = {
    'personagraph',
    'personagraph.runtime',
    'personagraph.runtime.entry',
    'personagraph.runtime.entry.context',
    'personagraph.runtime.entry.context.task_catalog_projection',
}
blocked = {
    'personagraph.model_io.gateway',
    'personagraph.runtime.entry.application',
    'personagraph.runtime.entry.context.application',
    'personagraph.runtime.entry.context.contracts',
    'personagraph.runtime.entry.ingress.model',
    'personagraph.runtime.entry.response.model',
    'personagraph.session',
    'personagraph.session.store',
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


def test_context_assembly_keeps_exactly_two_catalog_reads_and_uses_policy() -> None:
    assert assembly_module.project_entry_task_catalog_items is (
        projection_module.project_entry_task_catalog_items
    )
    tree = ast.parse(Path(assembly_module.__file__).read_text(encoding="utf-8"))
    store_calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "store"
    ]
    policy_calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert store_calls.count("list_insession_task_catalog") == 1
    assert store_calls.count("list_pending_user_questions") == 1
    assert "project_entry_task_catalog_items" in policy_calls


def _accepted_turn() -> AcceptedEntryTurn:
    return AcceptedEntryTurn(
        session_id="session-1",
        turn_id="turn-1",
        client_request_id="request-1",
        user_input="assemble the task catalog",
        attachment_ids=(),
        window_revision=1,
        replayed=False,
        execution_snapshot=EntryExecutionSnapshot.create(
            features={},
            post_commit_job_kinds=(),
        ),
        routing_policy=freeze_turn_routing_policy(
            TurnRoutingPolicy(l1_enabled=False, l2_enabled=True),
            source="request_override",
        ),
    )


class _CatalogReadStore:
    def __init__(self, *, fail_catalog: bool = False, fail_pending: bool = False) -> None:
        self.fail_catalog = fail_catalog
        self.fail_pending = fail_pending
        self.calls: list[str] = []

    def list_turn_attachments(
        self,
        _session_id: str,
        _turn_id: str,
    ) -> list[dict[str, object]]:
        self.calls.append("attachments")
        return []

    def get_session_summary_state(self, _session_id: str) -> None:
        self.calls.append("summary")
        return None

    def list_committed_turn_pairs(
        self,
        _session_id: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, object]]:
        del limit
        self.calls.append("history")
        return []

    def list_insession_task_catalog(
        self,
        _session_id: str,
    ) -> tuple[InSessionTaskCatalogItem, ...]:
        self.calls.append("catalog")
        if self.fail_catalog:
            raise OSError("catalog read failed")
        return (_item("task-1"),)

    def list_pending_user_questions(self, *, session_id: str) -> tuple[object, ...]:
        assert session_id == "session-1"
        self.calls.append("pending")
        if self.fail_pending:
            raise OSError("pending read failed")
        return ()


def test_context_assembly_reads_catalog_before_pending_and_propagates_catalog_error() -> None:
    store = _CatalogReadStore(fail_catalog=True)

    with pytest.raises(OSError, match="^catalog read failed$"):
        assembly_module.build_entry_context(
            accepted=_accepted_turn(),
            features={"context_guard_limit": 24_000},
            store=store,
        )

    assert [call for call in store.calls if call in {"catalog", "pending"}] == [
        "catalog"
    ]


def test_context_assembly_reads_pending_after_catalog_and_propagates_pending_error() -> None:
    store = _CatalogReadStore(fail_pending=True)

    with pytest.raises(OSError, match="^pending read failed$"):
        assembly_module.build_entry_context(
            accepted=_accepted_turn(),
            features={"context_guard_limit": 24_000},
            store=store,
        )

    assert [call for call in store.calls if call in {"catalog", "pending"}] == [
        "catalog",
        "pending",
    ]
