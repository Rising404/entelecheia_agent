"""绑定会话的只读工具接口的专项组合测试。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from personagraph.l2.task_execution.attempts.controller import (
    AttemptToolBridgePreflightRequest,
    AttemptToolBridgeRequest,
)
from personagraph.l2.auxiliary_execution.planning.mounted_document_authority import (
    freeze_mounted_document_planning_authority,
)
from personagraph.l2.auxiliary_execution.planning.profiles import (
    MODEL_ANALYSIS_CAPABILITY,
    WORKSPACE_READONLY_CAPABILITY,
    build_auxiliary_execution_capability_catalogs,
    build_auxiliary_planning_capability_catalog,
)
from personagraph.tools.workspace.session_read_source import (
    build_session_workspace_readonly_runtime,
)
from personagraph.session.local_file_authority import (
    SqliteSessionFileAuthority,
)
from personagraph.l2.task_execution.tool_bridge.workspace_readonly_adapter import (
    build_workspace_readonly_work_run_bridge,
)
from personagraph.tools.effects import EffectAction
from personagraph.tools.catalog.snapshots.attempt import (
    AttemptToolBindingOwnerKind,
)
from personagraph.tools.catalog.persistence import ToolCatalogRepository
from personagraph.tools.composition.default_catalog import (
    bootstrap_production_default_catalog,
    build_production_default_factory_registry,
)
from personagraph.tools.execution import ToolBusinessFailure
from personagraph.tools.documents.format_observation_tools import (
    FORMAT_OBSERVATION_LOCAL_TOOL_IDS,
)
from personagraph.tools.catalog.materialization import (
    RuntimeCatalogMaterializer,
)
from personagraph.tools.workspace.workspace_tools import (
    WORKSPACE_DISCOVERY_TOOL_IDS,
)
from personagraph.input_processing.vision.contracts import (
    VisionCapabilitySnapshot,
    VisionPurpose,
)
from personagraph.session import store
from personagraph.model_io.output_validation import ModelOutputValidationError
from personagraph.l2.work_run import (
    AttemptDecision,
    CallToolsAction,
    ToolCallProposal,
)


EXPECTED_WORKSPACE_TOOL_IDS = {
    # 发现。
    "workspace_overview",
    "list_workspace_directory",
    "find_files",
    "search_text_files",
    "inspect_file",
    # 格式专属观察。
    "read_text",
    "read_pdf_text",
    "read_word",
    "read_slides",
    "inspect_image",
}
EXPECTED_EXTERNAL_VISUAL_TOOL_IDS = {
    "analyze_image",
    "analyze_pdf_page",
}


def _catalog_tool_ids(snapshot) -> set[str]:
    return {entry.registration.tool_id for entry in snapshot.exposed()}


def _empty_mounted_authority(monkeypatch, session_id: str):
    monkeypatch.setattr(
        "personagraph.tools.documents.mounted_document_source_authority.docstore.mounted_docs",
        lambda selected_session_id: (
            () if selected_session_id == session_id else (_ for _ in ()).throw(
                AssertionError("unexpected Session authority lookup")
            )
        ),
    )
    return freeze_mounted_document_planning_authority(session_id=session_id)


def _workspace_capability(catalog):
    return next(
        item
        for item in catalog.capabilities
        if item.capability_alias == WORKSPACE_READONLY_CAPABILITY
    )


def test_bound_working_directory_composes_all_three_tool_groups(
    tmp_path: Path,
    bound_partitioned_session,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("bounded content", encoding="utf-8")
    session_id = bound_partitioned_session(working_dir=workspace)

    runtime = build_session_workspace_readonly_runtime(session_id)

    assert runtime is not None
    assert set(runtime.tool_ids) == EXPECTED_WORKSPACE_TOOL_IDS
    assert _catalog_tool_ids(runtime.catalog_snapshot) == EXPECTED_WORKSPACE_TOOL_IDS
    assert tuple(
        binding.identity.tool_id
        for binding in runtime.workspace_discovery_bindings
    ) == (
        "workspace_overview",
        "list_workspace_directory",
        "find_files",
        "search_text_files",
        "inspect_file",
    )
    assert tuple(
        binding.identity.tool_id
        for binding in runtime.format_observation_bindings
    ) == (
        "read_text",
        "read_pdf_text",
        "read_word",
        "read_slides",
        "inspect_image",
    )
    assert runtime.external_visual_analysis_bindings == ()
    serialized_binding_assertions = json.dumps(
        [
            dict(binding.binding_assertion)
            for binding in (
                *runtime.workspace_discovery_bindings,
                *runtime.format_observation_bindings,
            )
        ],
        sort_keys=True,
    )
    assert str(workspace.resolve()) not in serialized_binding_assertions
    assert runtime.local_file_authorization_receipt_id not in (
        serialized_binding_assertions
    )
    assert runtime.boundary.root == workspace.resolve()
    assert runtime.boundary.session_id == session_id
    assert len(runtime.scope_snapshot_sha256) == 64
    assert all(
        effect.action in {EffectAction.READ, EffectAction.SEARCH}
        for entry in runtime.catalog_snapshot.exposed()
        for effect in entry.registration.effect_profile.effects
    )

    execution_catalogs = build_auxiliary_execution_capability_catalogs(
        workspace_runtime=runtime
    )
    assert set(execution_catalogs) == {
        MODEL_ANALYSIS_CAPABILITY,
        WORKSPACE_READONLY_CAPABILITY,
    }
    assert execution_catalogs[WORKSPACE_READONLY_CAPABILITY] is (
        runtime.catalog_snapshot
    )


def test_unbound_session_exposes_no_workspace_catalog(
    partitioned_project_state,
) -> None:
    del partitioned_project_state
    session_id = store.create_session("Entelecheia")

    with store.session_database_scope(session_id):
        runtime = build_session_workspace_readonly_runtime(session_id)

    assert runtime is None
    execution_catalogs = build_auxiliary_execution_capability_catalogs(
        workspace_runtime=None
    )
    assert set(execution_catalogs) == {MODEL_ANALYSIS_CAPABILITY}
    assert _catalog_tool_ids(execution_catalogs[MODEL_ANALYSIS_CAPABILITY]) == set()


def test_workspace_bindings_materialize_from_the_persistent_default_profile(
    tmp_path: Path,
    bound_partitioned_session,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("bounded content", encoding="utf-8")
    session_id = bound_partitioned_session(working_dir=workspace)
    workspace_runtime = build_session_workspace_readonly_runtime(session_id)
    assert workspace_runtime is not None
    repository = ToolCatalogRepository(tmp_path / "tool_catalog.sqlite")
    bootstrap_production_default_catalog(repository)

    materialized = RuntimeCatalogMaterializer(
        repository,
        build_production_default_factory_registry(),
    ).materialize_new(
        attempt_id="attempt-workspace-discovery",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        contextual_bindings=(
            *workspace_runtime.workspace_discovery_bindings,
            *workspace_runtime.format_observation_bindings,
        ),
    )

    exposed_ids = {
        definition.identity.tool_id
        for definition in materialized.exposed_definitions
    }
    expected_contextual_ids = {
        *WORKSPACE_DISCOVERY_TOOL_IDS,
        *FORMAT_OBSERVATION_LOCAL_TOOL_IDS,
    }
    assert expected_contextual_ids <= exposed_ids
    owners = {
        owner.identity.tool_id: owner.kind
        for owner in materialized.frozen_attempt_catalog.binding_owners
    }
    assert {tool_id: owners[tool_id] for tool_id in expected_contextual_ids} == {
        tool_id: AttemptToolBindingOwnerKind.CONTEXTUAL_CANDIDATE
        for tool_id in expected_contextual_ids
    }










def test_external_vision_binds_before_the_model_selects_an_authorized_source(
    tmp_path: Path,
    monkeypatch,
    bound_partitioned_session,
) -> None:
    class ExternalVision:
        transmits_externally = True

        def capabilities(self):
            return VisionCapabilitySnapshot(
                available=True,
                provider="configured",
                model="configured",
                endpoint_identity="configured",
                processor_fingerprint="configured-external@1",
                supported_purposes=tuple(VisionPurpose),
            )

    monkeypatch.setattr(
        "personagraph.tools.workspace.session_read_source.default_vision_adapter",
        lambda: ExternalVision(),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = bound_partitioned_session(working_dir=workspace)

    runtime = build_session_workspace_readonly_runtime(session_id)

    assert runtime is not None
    assert runtime.visual_analysis_available is True
    assert runtime.visual_analysis_reason is None
    assert EXPECTED_EXTERNAL_VISUAL_TOOL_IDS <= set(runtime.tool_ids)
    assert runtime.authority.approval_grants == ()
    assert {key[0] for key in runtime.invocation_authority_resolver_by_key} == EXPECTED_EXTERNAL_VISUAL_TOOL_IDS


def test_one_session_visual_grant_exposes_protected_adjustable_tools(
    tmp_path: Path,
    monkeypatch,
    bound_partitioned_session,
) -> None:
    class ExternalVision:
        transmits_externally = True

        def capabilities(self):
            return VisionCapabilitySnapshot(
                available=True,
                provider="configured",
                model="configured",
                endpoint_identity="configured",
                processor_fingerprint="configured-external@1",
                supported_purposes=tuple(VisionPurpose),
            )

    monkeypatch.setattr(
        "personagraph.tools.workspace.session_read_source.default_vision_adapter",
        lambda: ExternalVision(),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = bound_partitioned_session(working_dir=workspace)

    runtime = build_session_workspace_readonly_runtime(session_id)

    assert runtime is not None
    assert runtime.visual_analysis_available is True
    assert runtime.visual_analysis_reason is None
    assert set(runtime.tool_ids) == (
        EXPECTED_WORKSPACE_TOOL_IDS | EXPECTED_EXTERNAL_VISUAL_TOOL_IDS
    )
    assert runtime.authority.approval_grants == ()
    assert {
        binding.identity.tool_id
        for binding in runtime.external_visual_analysis_bindings
    } == EXPECTED_EXTERNAL_VISUAL_TOOL_IDS
    serialized_assertions = json.dumps(
        [
            dict(binding.binding_assertion)
            for binding in runtime.external_visual_analysis_bindings
        ],
        sort_keys=True,
    )
    assert str(workspace.resolve()) not in serialized_assertions
    assert "auto_visual_egress_" not in serialized_assertions
    assert runtime.local_file_authorization_receipt_id not in serialized_assertions
    by_id = {
        entry.registration.tool_id: entry.registration
        for entry in runtime.catalog_snapshot.exposed()
    }
    for tool_id in EXPECTED_EXTERNAL_VISUAL_TOOL_IDS:
        actions = {
            item.action for item in by_id[tool_id].effect_profile.effects
        }
        assert actions == {EffectAction.READ, EffectAction.TRANSMIT, EffectAction.UPDATE}

    repository = ToolCatalogRepository(tmp_path / "visual-tool-catalog.sqlite")
    bootstrap_production_default_catalog(repository)
    materialized = RuntimeCatalogMaterializer(
        repository,
        build_production_default_factory_registry(),
    ).materialize_new(
        attempt_id="attempt-external-visual",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        contextual_bindings=(
            *runtime.workspace_discovery_bindings,
            *runtime.format_observation_bindings,
            *runtime.external_visual_analysis_bindings,
        ),
    )
    assert EXPECTED_EXTERNAL_VISUAL_TOOL_IDS <= {
        definition.identity.tool_id
        for definition in materialized.exposed_definitions
    }
    owner_by_tool_id = {
        owner.identity.tool_id: owner.kind
        for owner in materialized.frozen_attempt_catalog.binding_owners
    }
    assert all(
        owner_by_tool_id[tool_id]
        is AttemptToolBindingOwnerKind.CONTEXTUAL_CANDIDATE
        for tool_id in EXPECTED_EXTERNAL_VISUAL_TOOL_IDS
    )
    allowed_tools = tuple(
        entry.registration.spec for entry in runtime.catalog_snapshot.exposed()
    )
    bridge = build_workspace_readonly_work_run_bridge(
        runtime,
        ledger_store=store,
    )
    request = AttemptToolBridgePreflightRequest(
            session_id=session_id,
            turn_id="turn-visual",
            work_run_id="workrun-visual",
            attempt_id="attempt-visual",
            decision=AttemptDecision(
                action=CallToolsAction(
                    calls=(
                        ToolCallProposal(
                            tool_id="analyze_image",
                            arguments={
                                "path": "chart.png",
                                "purpose": "chart",
                                "detail": "standard",
                                "region": "page",
                            },
                        ),
                    )
                )
            ),
            tool_call_ids=("call-visual",),
            allowed_tools=allowed_tools,
            catalog_snapshot=runtime.catalog_snapshot.to_descriptor(),
        )
    # L2 尚未接入逐调用授权解析器，必须明确拒绝，不能复用会话 grant 绕开新入口。
    with pytest.raises(ModelOutputValidationError, match="Tool Bridge preflight rejected"):
        bridge.preflight(request)


def test_same_bound_root_reuses_one_local_file_authorization_receipt(
    tmp_path: Path,
    bound_partitioned_session,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = bound_partitioned_session(working_dir=workspace)

    first = build_session_workspace_readonly_runtime(session_id)
    second = build_session_workspace_readonly_runtime(session_id)

    assert first is not None and second is not None
    assert first.local_file_authorization_receipt_id == (
        second.local_file_authorization_receipt_id
    )


def test_already_bound_read_handlers_reject_a_revoked_grant(
    tmp_path: Path,
    bound_partitioned_session,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("private content", encoding="utf-8")
    session_id = bound_partitioned_session(working_dir=workspace)
    runtime = build_session_workspace_readonly_runtime(session_id)
    assert runtime is not None
    registrations = {
        entry.registration.tool_id: entry.registration
        for entry in runtime.catalog_snapshot.exposed()
    }

    revoked = SqliteSessionFileAuthority().revoke(
        session_id=session_id,
        grant_id=runtime.local_file_authorization_receipt_id,
    )

    assert revoked == 1
    for tool_id, payload in (
        ("workspace_overview", {}),
        ("read_text", {"path": "notes.txt"}),
    ):
        with pytest.raises(ToolBusinessFailure) as raised:
            registrations[tool_id].handler(payload)
        assert raised.value.error.code == "workspace_authority_revoked"


def test_replacing_directory_at_same_path_invalidates_old_read_receipt(
    tmp_path: Path,
    bound_partitioned_session,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = bound_partitioned_session(working_dir=workspace)
    assert build_session_workspace_readonly_runtime(session_id) is not None

    workspace.rename(tmp_path / "retired-workspace")
    workspace.mkdir()

    assert build_session_workspace_readonly_runtime(session_id) is None


def test_already_built_workspace_handlers_reject_same_path_root_replacement(
    tmp_path: Path,
    bound_partitioned_session,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("original", encoding="utf-8")
    session_id = bound_partitioned_session(working_dir=workspace)
    runtime = build_session_workspace_readonly_runtime(session_id)
    assert runtime is not None
    tools = {
        entry.registration.tool_id: entry.registration
        for entry in runtime.catalog_snapshot.exposed()
    }

    workspace.rename(tmp_path / "retired-workspace")
    workspace.mkdir()
    (workspace / "notes.txt").write_text("replacement secret", encoding="utf-8")

    for tool_id, payload in (
        ("inspect_file", {"path": "notes.txt"}),
        ("read_text", {"path": "notes.txt"}),
    ):
        with pytest.raises(ToolBusinessFailure) as raised:
            tools[tool_id].handler(payload)
        assert raised.value.error.code == "workspace_authority_changed"


def test_planning_workspace_capability_tracks_the_bound_runtime(
    tmp_path: Path,
    monkeypatch,
    partitioned_project_state,
) -> None:
    del partitioned_project_state
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bound_session_id = store.create_session(
        "Entelecheia", working_dir=str(workspace)
    )
    unbound_session_id = store.create_session("Entelecheia")
    mounted = _empty_mounted_authority(monkeypatch, bound_session_id)
    with store.session_database_scope(bound_session_id):
        runtime = build_session_workspace_readonly_runtime(bound_session_id)
    assert runtime is not None

    available_catalog = build_auxiliary_planning_capability_catalog(
        mounted,
        workspace_runtime=runtime,
    )
    with store.session_database_scope(unbound_session_id):
        unavailable_runtime = build_session_workspace_readonly_runtime(
            unbound_session_id
        )
    unavailable_catalog = build_auxiliary_planning_capability_catalog(
        mounted,
        workspace_runtime=unavailable_runtime,
    )

    available = _workspace_capability(available_catalog)
    unavailable = _workspace_capability(unavailable_catalog)
    assert available.available is True
    assert unavailable.available is False
    assert "iterative_model_tool_calls" in available.supported_operations
    if not runtime.visual_analysis_available:
        assert "visual_semantics_requires_protected_operation" in (
            available.limitations
        )
    assert available_catalog.capability_catalog_snapshot_sha256 != (
        unavailable_catalog.capability_catalog_snapshot_sha256
    )


def test_one_bridge_builds_stable_plans_for_auxiliary_and_task_attempt_ids(
    tmp_path: Path,
    bound_partitioned_session,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = bound_partitioned_session(working_dir=workspace)
    runtime = build_session_workspace_readonly_runtime(session_id)
    assert runtime is not None
    bridge = build_workspace_readonly_work_run_bridge(
        runtime,
        ledger_store=store,
    )
    snapshot = runtime.catalog_snapshot
    allowed_tools = tuple(
        entry.registration.spec for entry in snapshot.exposed()
    )
    proposal = AttemptDecision(
        action=CallToolsAction(
            calls=(
                ToolCallProposal(
                    tool_id="workspace_overview",
                    arguments={},
                ),
            )
        )
    )

    def persistence_plan(
        *,
        turn_id: str,
        work_run_id: str,
        attempt_id: str,
        call_id: str,
        apply_id: str,
    ):
        accepted = bridge.preflight(
            AttemptToolBridgePreflightRequest(
                session_id=session_id,
                turn_id=turn_id,
                work_run_id=work_run_id,
                attempt_id=attempt_id,
                decision=proposal,
                tool_call_ids=(call_id,),
                allowed_tools=allowed_tools,
                catalog_snapshot=snapshot.to_descriptor(),
            )
        )
        request = AttemptToolBridgeRequest(
            session_id=session_id,
            turn_id=turn_id,
            work_run_id=work_run_id,
            attempt_id=attempt_id,
            expected_work_run_revision=1,
            expected_progress_revision=1,
            expected_output_revision=1,
            expected_window_revision=1,
            apply_id=apply_id,
            decision=accepted,
            allowed_tools=allowed_tools,
            catalog_snapshot=snapshot.to_descriptor(),
        )
    # 桥接器有意拥有该计划工厂。在这里验证它，可以避免为任一图家族填充种子，
    # 同时证明它不假设任何一方的持久 ID 语法。
        return bridge._persistence_plan_factory(request)

    auxiliary_values = {
        "turn_id": "turn-auxiliary",
        "work_run_id": "auxv2:graph-7:node-3:run",
        "attempt_id": "auxv2:graph-7:node-3:attempt:2",
        "call_id": "auxv2:graph-7:node-3:attempt:2:tool:1",
        "apply_id": "auxv2:graph-7:node-3:attempt:2:decision",
    }
    task_values = {
        "turn_id": "turn-task",
        "work_run_id": "workrun-task-node-17",
        "attempt_id": "attempt-task-node-17-3",
        "call_id": "tool-call-task-node-17-3-1",
        "apply_id": "apply-task-node-17-3-decision",
    }

    auxiliary_first = persistence_plan(**auxiliary_values)
    auxiliary_replay = persistence_plan(**auxiliary_values)
    task_plan = persistence_plan(**task_values)

    assert auxiliary_first == auxiliary_replay
    assert auxiliary_first.decision_apply_id == auxiliary_values["apply_id"]
    assert task_plan.decision_apply_id == task_values["apply_id"]
    assert auxiliary_first.calls[0].tool_call_id == auxiliary_values["call_id"]
    assert task_plan.calls[0].tool_call_id == task_values["call_id"]
    assert auxiliary_first.close_apply_id != task_plan.close_apply_id
    assert auxiliary_first.calls[0].tool_result_id != (
        task_plan.calls[0].tool_result_id
    )
    assert auxiliary_first.calls[0].result_apply_id != (
        task_plan.calls[0].result_apply_id
    )


def test_directory_changes_do_not_change_file_tool_authority(tmp_path, bound_partitioned_session):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    session_id = bound_partitioned_session(working_dir=workspace)
    first = build_session_workspace_readonly_runtime(session_id)
    (workspace / 'late.md').write_text('not admitted', encoding='utf-8')
    second = build_session_workspace_readonly_runtime(session_id)
    assert first is not None and second is not None
    assert first.scope_snapshot_sha256 == second.scope_snapshot_sha256
    assert first.catalog_snapshot.to_descriptor() == second.catalog_snapshot.to_descriptor()
    assert not hasattr(second, 'document_candidates')


def test_discovery_returns_paths_without_candidate_registration(tmp_path, bound_partitioned_session):
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'brief.md').write_text('renewal deadline', encoding='utf-8')
    session_id = bound_partitioned_session(working_dir=workspace)
    runtime = build_session_workspace_readonly_runtime(session_id)
    assert runtime is not None
    tools = {entry.registration.tool_id: entry.registration for entry in runtime.catalog_snapshot.exposed()}
    result = tools['find_files'].handler({'name': 'brief.md'})
    assert [item['path'] for item in result['items']] == ['brief.md']
    assert all('candidate_id' not in item for item in result['items'])
