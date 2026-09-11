"""任务范围常规 TaskGraph 文档工具运行时测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from personagraph.session import store
from personagraph.l2.task_execution.task_node import (
    document_tool_runtime as runtime_module,
)
from personagraph.l2.task_execution.tool_bridge.attempt_contracts import (
    AttemptToolBridgePreflightRequest,
)
from personagraph.l2.auxiliary_execution.planning.mounted_visual_resource import (
    MountedVisualPlanningAuthorityError,
)
from personagraph.tools.findings.contracts import EXECUTION_FINDINGS_TOOL_IDS
from personagraph.l2.task_execution.tool_bridge.mounted_visual_adapter import (
    SessionMountedVisualToolRuntime,
)
from personagraph.tools.visual.mounted_visual_tools import (
    MOUNTED_VISUAL_TOOL_ID,
)
from personagraph.tools.policy import (
    ProtectedToolExecutionAuthority,
)
from personagraph.l2.task_execution.tool_bridge.protected_dispatch import (
    RuntimeProtectedToolDispatcher,
)
from personagraph.l2.task_execution.task_graph.controller import (
    TaskNodeToolRuntime,
)
from personagraph.l2.task_execution.task_node.document_tool_runtime import (
    TaskNodeDocumentToolRuntimeError,
    build_task_scoped_document_node_runtime_factory,
    compose_task_node_document_tool_runtime,
)
from personagraph.tools.catalog import CatalogSnapshot, ToolCatalog
from personagraph.tools.contracts import (
    ToolSourceDescriptor,
    ToolSourceKind,
    ToolSpec,
)
from personagraph.tools.effects import (
    DataEgress,
    EffectAction,
    EffectDescriptor,
    EffectResource,
    EffectScopeKind,
    Idempotency,
    Reversibility,
    ToolEffectProfile,
)
from personagraph.tools.policy import AuthorityFacts, ScopeGrant
from personagraph.tools.registration import (
    ToolExecutionProfile,
    ToolRegistration,
)
from personagraph.l2.work_run import (
    AttemptDecision,
    CallToolsAction,
    TaskNodeSubject,
    ToolCallProposal,
)


SESSION_ID = "session-task-node-documents"
TASK_ID = "task-task-node-documents"


def _registration(
    tool_id: str,
    *,
    scope_kind: EffectScopeKind,
    scope: str,
    resource: EffectResource = EffectResource.FILESYSTEM,
    action: EffectAction = EffectAction.READ,
    data_egress: DataEgress = DataEgress.CONTENT,
    idempotency: Idempotency = Idempotency.IDEMPOTENT,
    reversibility: Reversibility = Reversibility.REVERSIBLE,
) -> ToolRegistration:
    object_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
    }
    return ToolRegistration(
        spec=ToolSpec(
            tool_id=tool_id,
            contract_version="test-v1",
            name=tool_id,
            description=f"Read through {tool_id}.",
            input_schema=object_schema,
            output_schema=object_schema,
            catalog_tags=("document", "read"),
        ),
        implementation_version="test-1",
        source=ToolSourceDescriptor(
            kind=ToolSourceKind.LOCAL,
            source_id=f"test.{tool_id}",
        ),
        handler=lambda _payload: {},
        effect_profile=ToolEffectProfile(
            (
                EffectDescriptor(
                    resource=resource,
                    action=action,
                    scope_kind=scope_kind,
                    default_scope=scope,
                    data_egress=data_egress,
                    idempotency=idempotency,
                    reversibility=reversibility,
                ),
            )
        ),
        execution_profile=ToolExecutionProfile(max_transparent_retries=0),
    )


def _snapshot(*registrations: ToolRegistration):
    catalog = ToolCatalog()
    for registration in registrations:
        catalog.register(registration)
    return catalog.snapshot()


def test_combined_runtime_exposes_both_catalogs_through_the_same_bridge() -> None:
    workspace_scope = "/workspace/root"
    workspace_snapshot = _snapshot(
        _registration(
            "list_workspace_fixture",
            scope_kind=EffectScopeKind.WORKSPACE,
            scope=workspace_scope,
        )
    )
    mounted_snapshot = _snapshot(
        _registration(
            "read_mounted_fixture",
            scope_kind=EffectScopeKind.SESSION,
            scope=SESSION_ID,
        )
    )
    workspace = SimpleNamespace(
        session_id=SESSION_ID,
        catalog_snapshot=workspace_snapshot,
        tool_bridge=object(),
        authority=AuthorityFacts(
            grants=(
                ScopeGrant(
                    EffectResource.FILESYSTEM,
                    EffectAction.READ,
                    EffectScopeKind.WORKSPACE,
                    workspace_scope,
                ),
            )
        ),
        protected_authority_by_key={},
    )
    mounted = SimpleNamespace(
        session_id=SESSION_ID,
        catalog_snapshot=mounted_snapshot,
        tool_bridge=object(),
        authority=AuthorityFacts(
            grants=(
                ScopeGrant(
                    EffectResource.FILESYSTEM,
                    EffectAction.READ,
                    EffectScopeKind.SESSION,
                    SESSION_ID,
                ),
            )
        ),
        contextual_bindings=(),
    )

    runtime = compose_task_node_document_tool_runtime(
        session_id=SESSION_ID,
        workspace_runtime=workspace,
        workspace_tool_bridge=workspace.tool_bridge,
        mounted_runtime=mounted,
        ledger_store=store,
    )

    assert {
        entry.registration.tool_id
        for entry in runtime.catalog_snapshot.exposed()
    } == {
        "list_workspace_fixture",
        "read_mounted_fixture",
        *EXECUTION_FINDINGS_TOOL_IDS,
    }
    allowed_tools = tuple(
        entry.registration.spec for entry in runtime.catalog_snapshot.exposed()
    )
    accepted = runtime.tool_bridge.preflight(
        AttemptToolBridgePreflightRequest(
            session_id=SESSION_ID,
            turn_id="turn-combined-documents",
            work_run_id="workrun-combined-documents",
            attempt_id="attempt-combined-documents",
            decision=AttemptDecision(
                action=CallToolsAction(
                    calls=(
                        ToolCallProposal(
                            tool_id="list_workspace_fixture",
                            arguments={},
                        ),
                        ToolCallProposal(
                            tool_id="read_mounted_fixture",
                            arguments={},
                        ),
                    )
                )
            ),
            tool_call_ids=("call-workspace", "call-mounted"),
            allowed_tools=allowed_tools,
            catalog_snapshot=runtime.catalog_snapshot.to_descriptor(),
        )
    )

    assert tuple(call.tool_id for call in accepted.action.calls) == (
        "list_workspace_fixture",
        "read_mounted_fixture",
    )
    assert runtime.tool_bridge._protected_dispatcher is None


def test_combined_runtime_exposes_mounted_visuals_with_their_authority() -> None:
    mounted_registration = _registration(
        "read_mounted_fixture",
        scope_kind=EffectScopeKind.SESSION,
        scope=SESSION_ID,
    )
    visual_registration = _registration(
        MOUNTED_VISUAL_TOOL_ID,
        scope_kind=EffectScopeKind.SESSION,
        scope=SESSION_ID,
        resource=EffectResource.NETWORK,
        action=EffectAction.TRANSMIT,
        idempotency=Idempotency.NOT_IDEMPOTENT,
        reversibility=Reversibility.IRREVERSIBLE,
    )
    mounted_grant = ScopeGrant(
        EffectResource.FILESYSTEM,
        EffectAction.READ,
        EffectScopeKind.SESSION,
        SESSION_ID,
    )
    visual_grant = ScopeGrant(
        EffectResource.NETWORK,
        EffectAction.TRANSMIT,
        EffectScopeKind.SESSION,
        SESSION_ID,
    )
    provider_identity = "a" * 64
    workspace = SimpleNamespace(
        session_id=SESSION_ID,
        catalog_snapshot=_snapshot(),
        tool_bridge=object(),
        authority=AuthorityFacts(),
        protected_authority_by_key={},
    )
    mounted = SimpleNamespace(
        session_id=SESSION_ID,
        catalog_snapshot=_snapshot(mounted_registration),
        tool_bridge=object(),
        authority=AuthorityFacts(grants=(mounted_grant,)),
        contextual_bindings=(),
    )
    visual_runtime = SessionMountedVisualToolRuntime(
        session_id=SESSION_ID,
        registration=visual_registration,
        authority=AuthorityFacts(approval_grants=(visual_grant,)),
        protected_authority_by_key={
            (
                visual_registration.tool_id,
                visual_registration.contract_version,
            ): ProtectedToolExecutionAuthority(
                approval_receipt_ids=("receipt-mounted-visual",),
                execution_backend_identity_sha256=provider_identity,
                revalidate=lambda: True,
            )
        },
    )

    runtime = compose_task_node_document_tool_runtime(
        session_id=SESSION_ID,
        workspace_runtime=workspace,
        workspace_tool_bridge=workspace.tool_bridge,
        mounted_runtime=mounted,
        ledger_store=store,
        mounted_visual_runtime=visual_runtime,
    )

    assert {
        entry.registration.tool_id
        for entry in runtime.catalog_snapshot.exposed()
    } == {
        "read_mounted_fixture",
        MOUNTED_VISUAL_TOOL_ID,
        *EXECUTION_FINDINGS_TOOL_IDS,
    }
    assert runtime.tool_bridge.authority is not None
    assert visual_grant in runtime.tool_bridge.authority.approval_grants
    assert isinstance(
        runtime.tool_bridge._protected_dispatcher,
        RuntimeProtectedToolDispatcher,
    )
    allowed_tools = tuple(
        entry.registration.spec for entry in runtime.catalog_snapshot.exposed()
    )
    accepted = runtime.tool_bridge.preflight(
        AttemptToolBridgePreflightRequest(
            session_id=SESSION_ID,
            turn_id="turn-mounted-visuals",
            work_run_id="workrun-mounted-visuals",
            attempt_id="attempt-mounted-visuals",
            decision=AttemptDecision(
                action=CallToolsAction(
                    calls=(
                        ToolCallProposal(
                            tool_id=MOUNTED_VISUAL_TOOL_ID,
                            arguments={},
                        ),
                    )
                )
            ),
            tool_call_ids=("call-mounted-visual",),
            allowed_tools=allowed_tools,
            catalog_snapshot=runtime.catalog_snapshot.to_descriptor(),
        )
    )

    assert tuple(call.tool_id for call in accepted.action.calls) == (
        MOUNTED_VISUAL_TOOL_ID,
    )


def test_node_runtime_factory_is_lazy_and_reuses_one_revision_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = TaskNodeSubject(
        task_id=TASK_ID,
        graph_revision=3,
        node_id="node-a",
        node_revision=1,
    )
    second = first.model_copy(update={"node_id": "node-b"})
    runtime = TaskNodeToolRuntime(
        catalog_snapshot=_snapshot(
            _registration(
                "read_lazy_fixture",
                scope_kind=EffectScopeKind.SESSION,
                scope=SESSION_ID,
            )
        )
    )
    calls: list[TaskNodeSubject] = []

    def build_revision(**kwargs):
        calls.append(kwargs["subject"])
        return runtime_module._TaskScopedDocumentRuntimeRevision(
            graph_revision=3,
            subjects=frozenset({first, second}),
            runtime=runtime,
        )

    monkeypatch.setattr(
        runtime_module,
        "_build_task_scoped_document_runtime_revision",
        build_revision,
    )

    factory = build_task_scoped_document_node_runtime_factory(
        session_id=SESSION_ID,
        task_id=TASK_ID,
        workspace_runtime=None,
        workspace_tool_bridge=None,
        ledger_store=store,
    )

    assert calls == []
    assert factory(first) is runtime
    assert factory(second) is runtime
    assert calls == [first]
    with pytest.raises(TaskNodeDocumentToolRuntimeError):
        factory(first.model_copy(update={"task_id": "other-task"}))
    assert calls == [first]




def test_node_runtime_factory_refreezes_each_task_graph_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision_three = TaskNodeSubject(
        task_id=TASK_ID,
        graph_revision=3,
        node_id="node-r3",
        node_revision=1,
    )
    revision_four = TaskNodeSubject(
        task_id=TASK_ID,
        graph_revision=4,
        node_id="node-r4",
        node_revision=1,
    )
    runtime_three = TaskNodeToolRuntime(
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
    )
    runtime_four = TaskNodeToolRuntime(
        catalog_snapshot=CatalogSnapshot(revision=1, entries=()),
    )
    calls: list[TaskNodeSubject] = []

    def build_revision(**kwargs):
        subject = kwargs["subject"]
        calls.append(subject)
        selected = (
            runtime_three
            if subject.graph_revision == revision_three.graph_revision
            else runtime_four
        )
        return runtime_module._TaskScopedDocumentRuntimeRevision(
            graph_revision=subject.graph_revision,
            subjects=frozenset({subject}),
            runtime=selected,
        )

    monkeypatch.setattr(
        runtime_module,
        "_build_task_scoped_document_runtime_revision",
        build_revision,
    )
    factory = build_task_scoped_document_node_runtime_factory(
        session_id=SESSION_ID,
        task_id=TASK_ID,
        workspace_runtime=None,
        workspace_tool_bridge=None,
        ledger_store=store,
    )

    assert factory(revision_three) is runtime_three
    assert factory(revision_three) is runtime_three
    assert factory(revision_four) is runtime_four
    assert factory(revision_four) is runtime_four
    assert calls == [revision_three, revision_four]


def test_workspace_runtime_survives_when_no_documents_are_mounted() -> None:
    snapshot = _snapshot(
        _registration(
            "list_workspace_only_fixture",
            scope_kind=EffectScopeKind.WORKSPACE,
            scope="/workspace/only",
        )
    )
    bridge = object()
    workspace = SimpleNamespace(
        session_id=SESSION_ID,
        catalog_snapshot=snapshot,
        tool_bridge=bridge,
    )

    runtime = compose_task_node_document_tool_runtime(
        session_id=SESSION_ID,
        workspace_runtime=workspace,
        workspace_tool_bridge=bridge,
        mounted_runtime=None,
        ledger_store=store,
    )

    assert runtime.catalog_snapshot is snapshot
    assert runtime.tool_bridge is bridge


def test_missing_document_capability_has_no_contextual_binding_candidates() -> None:
    runtime = compose_task_node_document_tool_runtime(
        session_id=SESSION_ID,
        workspace_runtime=None,
        workspace_tool_bridge=None,
        mounted_runtime=None,
        ledger_store=store,
    )

    assert runtime.contextual_bindings == ()


def test_revision_builder_reproduces_exact_task_authority_before_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = TaskNodeSubject(
        task_id=TASK_ID,
        graph_revision=2,
        node_id="node-exact",
        node_revision=4,
    )
    authority_snapshot = SimpleNamespace(anchors=())
    auxiliary = SimpleNamespace(
        authority_snapshot=authority_snapshot,
        goal_status="committed",
        revision_status="committed",
        target_task_graph_revision=2,
    )
    task = SimpleNamespace(
        current_graph_revision=2,
        nodes=(
            {
                "insession_task_node_id": "node-exact",
                "node_revision": 4,
            },
        ),
    )
    creation_source = object()
    mounted_authority = object()
    mounted_runtime = object()
    mounted_visual_runtime = object()
    expected_runtime = TaskNodeToolRuntime(
        catalog_snapshot=_snapshot(
            _registration(
                "read_exact_fixture",
                scope_kind=EffectScopeKind.SESSION,
                scope=SESSION_ID,
            )
        )
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        runtime_module.auxiliary_graph_store,
        "get_auxiliary_graph_for_task",
        lambda **_kwargs: auxiliary,
    )
    monkeypatch.setattr(
        runtime_module.task_graph_store,
        "get_insession_task_details",
        lambda *_args: task,
    )
    monkeypatch.setattr(
        runtime_module.task_graph_store,
        "get_insession_task_creation_source",
        lambda **_kwargs: creation_source,
    )
    monkeypatch.setattr(
        runtime_module,
        "recover_task_scoped_managed_document_ids",
        lambda **_kwargs: ("managed-doc-exact",),
    )

    def freeze(**kwargs):
        observed["freeze"] = kwargs
        return mounted_authority

    def project(**kwargs):
        observed["project"] = kwargs

    monkeypatch.setattr(
        runtime_module,
        "freeze_mounted_document_planning_authority",
        freeze,
    )
    monkeypatch.setattr(
        runtime_module,
        "build_mounted_document_authority_projection",
        project,
    )

    def build_mounted_runtime(authority):
        observed["mounted_runtime_authority"] = authority
        return mounted_runtime

    def build_visual_runtime(authority):
        observed["mounted_visual_runtime_authority"] = authority
        return mounted_visual_runtime

    def compose(**kwargs):
        observed["compose"] = kwargs
        return expected_runtime

    monkeypatch.setattr(
        runtime_module,
        "build_session_mounted_document_cognition_runtime",
        build_mounted_runtime,
    )
    monkeypatch.setattr(
        runtime_module,
        "build_session_mounted_visual_tool_runtime",
        build_visual_runtime,
    )
    monkeypatch.setattr(
        runtime_module,
        "compose_task_node_document_tool_runtime",
        compose,
    )

    factory = build_task_scoped_document_node_runtime_factory(
        session_id=SESSION_ID,
        task_id=TASK_ID,
        workspace_runtime=None,
        workspace_tool_bridge=None,
        ledger_store=store,
    )
    result = factory(subject)

    assert result is expected_runtime
    assert observed["freeze"] == {
        "session_id": SESSION_ID,
        "task_id": TASK_ID,
        "allowed_managed_document_ids": ("managed-doc-exact",),
    }
    assert observed["project"] == {
        "authority_snapshot": authority_snapshot,
        "task_creation_source": creation_source,
        "mounted_authority": mounted_authority,
    }
    assert observed["mounted_runtime_authority"] is mounted_authority
    assert observed["mounted_visual_runtime_authority"] is mounted_authority
    assert observed["compose"] == {
        "session_id": SESSION_ID,
        "workspace_runtime": None,
        "workspace_tool_bridge": None,
        "mounted_runtime": mounted_runtime,
        "ledger_store": store,
        "mounted_visual_runtime": mounted_visual_runtime,
    }


def test_task_node_runtime_projects_visual_authority_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject = TaskNodeSubject(
        task_id=TASK_ID,
        graph_revision=2,
        node_id="node-visual-authority",
        node_revision=1,
    )
    authority_snapshot = SimpleNamespace(anchors=())
    auxiliary = SimpleNamespace(
        authority_snapshot=authority_snapshot,
        goal_status="committed",
        revision_status="committed",
        target_task_graph_revision=2,
    )
    task = SimpleNamespace(
        current_graph_revision=2,
        nodes=(
            {
                "insession_task_node_id": subject.node_id,
                "node_revision": subject.node_revision,
            },
        ),
    )
    monkeypatch.setattr(
        runtime_module.auxiliary_graph_store,
        "get_auxiliary_graph_for_task",
        lambda **_kwargs: auxiliary,
    )
    monkeypatch.setattr(
        runtime_module.task_graph_store,
        "get_insession_task_details",
        lambda *_args: task,
    )
    monkeypatch.setattr(
        runtime_module,
        "recover_task_scoped_managed_document_ids",
        lambda **_kwargs: (),
    )
    monkeypatch.setattr(
        runtime_module,
        "freeze_mounted_document_planning_authority",
        lambda **_kwargs: (_ for _ in ()).throw(
            MountedVisualPlanningAuthorityError("visual authority is stale")
        ),
    )
    factory = build_task_scoped_document_node_runtime_factory(
        session_id=SESSION_ID,
        task_id=TASK_ID,
        workspace_runtime=None,
        workspace_tool_bridge=None,
        ledger_store=store,
    )

    with pytest.raises(
        TaskNodeDocumentToolRuntimeError,
        match="could not be reproduced",
    ) as captured:
        factory(subject)

    assert isinstance(captured.value.__cause__, MountedVisualPlanningAuthorityError)
