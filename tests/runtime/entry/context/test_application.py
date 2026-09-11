"""可信只读运行时入口上下文组装的边界测试。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from personagraph.context_budget.token_counter import estimate_tokens
from personagraph.runtime.entry.context.attachments import (
    AttachmentAccess,
    attachment_projection_token_cost,
)
from personagraph.runtime.entry.context import application as assembly_module
from personagraph.runtime.entry.context.application import build_turn_attachments
from personagraph.runtime.entry.context import ports as context_ports_module
from personagraph.runtime.entry.routing.policy import (
    TurnRoutingPolicy,
    freeze_turn_routing_policy,
)
from personagraph.runtime.turn.contracts import (
    AcceptedEntryTurn,
    EntryExecutionSnapshot,
)
from personagraph.runtime.turn_deadline import TurnDeadline, TurnDeadlineExceeded
from personagraph.session.entry_task_contracts import (
    EntryPendingTaskQuestion,
    EntryTaskCatalogItem,
)


class _AttachmentStoreMustNotBeRead:
    def list_turn_attachments(self, _session_id: str, _turn_id: str) -> list[dict]:
        raise AssertionError("expired deadline must stop before reading attachments")


def test_expired_deadline_stops_before_attachment_binding_read() -> None:
    with pytest.raises(TurnDeadlineExceeded):
        build_turn_attachments(
            session_id="session-1",
            turn_id="turn-1",
            store=_AttachmentStoreMustNotBeRead(),
            deadline=TurnDeadline.starting_now(0),
        )


def test_context_assembly_has_only_the_five_read_facts_and_no_entry_lifecycle() -> None:
    tree = ast.parse(Path(assembly_module.__file__).read_text(encoding="utf-8"))
    ports_tree = ast.parse(
        Path(context_ports_module.__file__).read_text(encoding="utf-8")
    )
    relative_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1
    }
    store_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "store"
    }
    port = next(
        node
        for node in ports_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EntryContextAssemblyStorePort"
    )
    port_methods = {
        node.name for node in port.body if isinstance(node, ast.FunctionDef)
    }

    assert "entry" not in relative_imports
    assert store_calls == {
        "list_turn_attachments",
        "get_session_summary_state",
        "list_committed_turn_pairs",
        "list_insession_task_catalog",
        "list_pending_user_questions",
    }
    assert port_methods == store_calls
    assert not {
        "accept_turn_execution",
        "advance_turn_execution_window",
        "append_runtime_turn_event",
        "finalize_turn_execution",
        "mark_turn_execution_interrupted",
    } & store_calls


class _OnDemandAttachmentStore:
    def list_turn_attachments(
        self,
        _session_id: str,
        _turn_id: str,
    ) -> list[dict[str, object]]:
        return [{
            "attachment_id": "attachment-1",
            "original_name": "report.pdf",
            "media_type": "application/pdf",
            "size_bytes": 1_024,
            "kind": "document",
            "stored_rel_path": "input/attachment-1/report.pdf",
            "content_hash": "0" * 64,
        }]

    def get_session_summary_state(self, _session_id: str) -> None:
        return None

    def list_committed_turn_pairs(
        self,
        _session_id: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, str]]:
        del limit
        return []

    def list_insession_task_catalog(
        self,
        _session_id: str,
    ) -> tuple[object, ...]:
        return ()

    def list_pending_user_questions(
        self,
        *,
        session_id: str,
    ) -> tuple[object, ...]:
        del session_id
        return ()


class _L1TaskReadsForbiddenStore(_OnDemandAttachmentStore):
    def list_insession_task_catalog(
        self,
        _session_id: str,
    ) -> tuple[object, ...]:
        raise AssertionError("L1-capable context must not read the durable Task catalog")

    def list_pending_user_questions(
        self,
        *,
        session_id: str,
    ) -> tuple[object, ...]:
        del session_id
        raise AssertionError("L1-capable context must not read pending Task questions")


class _TaskCatalogProbeStore(_OnDemandAttachmentStore):
    def __init__(self) -> None:
        self.task_reads: list[str] = []

    def list_insession_task_catalog(
        self,
        _session_id: str,
    ) -> tuple[EntryTaskCatalogItem, ...]:
        self.task_reads.append("catalog")
        return (
            EntryTaskCatalogItem(
                insession_task_id="task-existing",
                goal_summary="保留非 L1 策略的长期任务上下文",
                status="awaiting_user",
                current_graph_revision=2,
            ),
        )

    def list_pending_user_questions(
        self,
        *,
        session_id: str,
    ) -> tuple[EntryPendingTaskQuestion, ...]:
        assert session_id == "session-1"
        self.task_reads.append("pending")
        return (
            EntryPendingTaskQuestion(
                insession_task_id="task-existing",
                question="是否继续长期任务？",
            ),
        )


def _accepted_turn(
    *,
    l1_enabled: bool = True,
    l2_enabled: bool = False,
) -> AcceptedEntryTurn:
    policy = TurnRoutingPolicy(
        l1_enabled=l1_enabled,
        l2_enabled=l2_enabled,
    )
    return AcceptedEntryTurn(
        session_id="session-1",
        turn_id="turn-1",
        client_request_id="request-1",
        user_input="请总结附件",
        attachment_ids=("attachment-1",),
        window_revision=1,
        replayed=False,
        execution_snapshot=EntryExecutionSnapshot.create(
            features={},
            post_commit_job_kinds=(),
        ),
        routing_policy=freeze_turn_routing_policy(
            policy,
            source="request_override",
        ),
    )


def test_context_always_uses_metadata_only_attachment_projection() -> None:
    accepted = _accepted_turn()

    context = assembly_module.build_entry_context(
        accepted=accepted,
        features={
            "context_guard_limit": 24_000,
        },
        store=_OnDemandAttachmentStore(),
    )

    assert len(context.attachments.items) == 1
    item = context.attachments.items[0]
    assert item.access is AttachmentAccess.ON_DEMAND
    assert context.envelope.attachments[0].attachment_id == "attachment-1"
    assert context.estimated_input_tokens == (
        estimate_tokens(accepted.user_input)
        + attachment_projection_token_cost(context.attachments)
    )


def test_l1_capable_context_omits_durable_task_context_without_reading_it() -> None:
    context = assembly_module.build_entry_context(
        accepted=_accepted_turn(l1_enabled=True, l2_enabled=False),
        features={"context_guard_limit": 24_000},
        store=_L1TaskReadsForbiddenStore(),
    )

    assert context.routing_policy.allows("L1") is True
    assert context.task_catalog.items == ()
    assert context.task_catalog.truncated is False


@pytest.mark.parametrize("l2_enabled", (False, True), ids=("l0-only", "l2"))
def test_non_l1_context_keeps_the_durable_task_projection(
    l2_enabled: bool,
) -> None:
    store = _TaskCatalogProbeStore()

    context = assembly_module.build_entry_context(
        accepted=_accepted_turn(l1_enabled=False, l2_enabled=l2_enabled),
        features={"context_guard_limit": 24_000},
        store=store,
    )

    assert context.routing_policy.allows("L1") is False
    assert store.task_reads == ["catalog", "pending"]
    assert tuple(
        item.insession_task_id for item in context.task_catalog.items
    ) == ("task-existing",)
    assert context.task_catalog.items[0].pending_user_question == (
        "是否继续长期任务？"
    )


def test_metadata_only_projection_fails_closed_when_candidate_metadata_is_missing() -> None:
    store = _OnDemandAttachmentStore()
    store.list_turn_attachments = lambda _session_id, _turn_id: [{
        "attachment_id": "attachment-1",
        "original_name": "report.pdf",
        "media_type": "application/pdf",
        "size_bytes": 1_024,
        "kind": "document",
        "stored_rel_path": "input/attachment-1/report.pdf",
    }]

    with pytest.raises(KeyError, match="content_hash"):
        build_turn_attachments(
            session_id="session-1",
            turn_id="turn-1",
            store=store,
        )
