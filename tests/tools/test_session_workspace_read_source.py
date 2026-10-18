"""绑定会话的只读工具接口的专项组合测试。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from personagraph.tools.workspace.session_read_source import (
    build_session_workspace_readonly_runtime,
)
from personagraph.session.local_file_authority import (
    SqliteSessionFileAuthority,
)
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
