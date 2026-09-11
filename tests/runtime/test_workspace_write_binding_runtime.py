"""Runtime/L1 wiring for the contextual protected workspace writer."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from personagraph.runtime.l1.protected_tool_dispatch import (
    l1_protected_dispatch_identity_sha256,
    l1_protected_operation_ledger_identity_sha256,
)
from personagraph.runtime.l1.tool_runtime import build_l1_tool_runtime
from personagraph.tools.workspace.workspace_write_adapter import (
    build_session_workspace_action_tool_source,
)
from personagraph.tools.workspace.session_read_source import (
    build_session_workspace_readonly_runtime,
)
from personagraph.session import store as session_store
from personagraph.tools.catalog.snapshots.attempt import (
    AttemptToolBindingOwnerKind,
)
from personagraph.tools.catalog.persistence import (
    DefaultProfileAvailability,
    ToolCatalogRepository,
    ToolCatalogSeed,
)
from personagraph.tools.policy import PolicyDisposition
from personagraph.tools.documents.format_observation_tools import (
    FORMAT_OBSERVATION_LOCAL_TOOL_IDS,
)
from personagraph.tools.catalog.materialization import (
    RuntimeCatalogMaterializer,
)
from personagraph.tools.catalog.trusted_factories import TrustedDefaultFactoryRegistry
from personagraph.tools.workspace.workspace_write_catalog import (
    build_workspace_write_tool_definition_manifest,
)
from personagraph.tools.workspace.workspace_write_tools import (
    WORKSPACE_WRITE_TOOL_ID,
)
from personagraph.tools.workspace.workspace_tools import (
    WORKSPACE_DISCOVERY_TOOL_IDS,
)


def _action_source(session_id: str):
    workspace = build_session_workspace_readonly_runtime(session_id)
    assert workspace is not None
    return build_session_workspace_action_tool_source(
        session_id=session_id,
        boundary=workspace.boundary,
        boundary_sha256=workspace.boundary_fingerprint,
        protected_dispatch_sha256=(
            l1_protected_dispatch_identity_sha256()
        ),
        operation_ledger_sha256=(
            l1_protected_operation_ledger_identity_sha256()
        ),
    )


def _materialize_writer(tmp_path: Path, binding):
    repository = ToolCatalogRepository(tmp_path / "workspace-write-catalog.sqlite")
    (manifest_item,) = build_workspace_write_tool_definition_manifest()
    repository.bootstrap(
        (
            ToolCatalogSeed(
                manifest_item.definition,
                DefaultProfileAvailability.IF_AVAILABLE,
            ),
        )
    )
    return RuntimeCatalogMaterializer(
        repository,
        TrustedDefaultFactoryRegistry(()),
    ).materialize_new(
        attempt_id="attempt-workspace-write-binding",
        created_at=datetime(2026, 9, 3, tzinfo=timezone.utc),
        contextual_bindings=(binding,),
    )


def test_unapproved_writer_materializes_but_l1_policy_fails_closed(
    tmp_path: Path,
    bound_partitioned_session,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = bound_partitioned_session(working_dir=workspace)
    source = _action_source(session_id)

    assert source.write_approval_receipt_id is None
    assert source.protected_authority_by_key == {}
    (binding,) = source.workspace_write_bindings
    materialized = _materialize_writer(tmp_path, binding)
    assert tuple(
        definition.identity.tool_id
        for definition in materialized.exposed_definitions
    ) == (WORKSPACE_WRITE_TOOL_ID,)
    (owner,) = materialized.frozen_attempt_catalog.binding_owners
    assert owner.kind is AttemptToolBindingOwnerKind.CONTEXTUAL_CANDIDATE

    runtime = build_l1_tool_runtime(session_id)
    model_tool_ids = tuple(item["tool_id"] for item in runtime.model_catalog())
    assert {
        *WORKSPACE_DISCOVERY_TOOL_IDS,
        *FORMAT_OBSERVATION_LOCAL_TOOL_IDS,
        WORKSPACE_WRITE_TOOL_ID,
    } <= set(model_tool_ids)
    assert (
        runtime.registrations_by_tool_id[
            WORKSPACE_WRITE_TOOL_ID
        ].binding.descriptor()
        == source.workspace_write_bindings[0].descriptor()
    )
    denied = runtime.prepare(
        tool_id=WORKSPACE_WRITE_TOOL_ID,
        arguments={"path": "result.txt", "content": "must not write"},
        remaining_tool_calls=1,
    )
    assert denied.policy["disposition"] == (
        PolicyDisposition.AUTHORIZATION_REQUIRED.value
    )
    assert denied.rejected_outcome is not None
    assert not (workspace / "result.txt").exists()


def test_approved_writer_keeps_exact_protected_authority_and_revalidation(
    tmp_path: Path,
    bound_partitioned_session,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = bound_partitioned_session(working_dir=workspace)
    unapproved = _action_source(session_id)
    grant = session_store.grant_session_workspace_write_authority(session_id)

    approved = _action_source(session_id)
    assert approved.write_approval_receipt_id == grant.grant_id
    assert approved.workspace_write_bindings[0].digest != (
        unapproved.workspace_write_bindings[0].digest
    )
    assertion_json = json.dumps(
        dict(approved.workspace_write_bindings[0].binding_assertion),
        sort_keys=True,
    )
    assert str(workspace.resolve()) not in assertion_json
    assert grant.grant_id not in assertion_json

    runtime = build_l1_tool_runtime(session_id)
    prepared = runtime.prepare(
        tool_id=WORKSPACE_WRITE_TOOL_ID,
        arguments={"path": "result.txt", "content": "protected"},
        remaining_tool_calls=1,
    )
    assert prepared.rejected_outcome is None
    assert prepared.requires_protected_dispatch is True
    authority = prepared.protected_authority
    assert authority is not None
    assert authority.approval_receipt_ids == (grant.grant_id,)
    assert authority.revalidate() is True

    revoked = session_store.revoke_session_workspace_write_authority(
        session_id,
        grant_id=grant.grant_id,
    )
    assert revoked == 1
    assert authority.revalidate() is False
    direct = runtime.execute_prepared(
        prepared,
        deadline_monotonic=10_000_000.0,
    )
    assert direct.outcome.error is not None
    assert direct.outcome.error.code == "protected_tool_dispatch_required"
    assert not (workspace / "result.txt").exists()


def test_l1_route_and_ledger_identities_are_stable_non_operation_hashes() -> None:
    route = l1_protected_dispatch_identity_sha256()
    ledger = l1_protected_operation_ledger_identity_sha256()

    assert len(route) == 64
    assert len(ledger) == 64
    assert route != ledger
    assert route == l1_protected_dispatch_identity_sha256()
    assert ledger == l1_protected_operation_ledger_identity_sha256()
