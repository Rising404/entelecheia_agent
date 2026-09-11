"""Stable Definition and protected contextual Binding for workspace writes."""

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from personagraph.tools.catalog.binding import BoundToolRegistration, ToolBinding
from personagraph.tools.contracts import ToolSourceKind
from personagraph.tools.effects import (
    DataEgress,
    EffectAction,
    EffectResource,
    EffectScopeKind,
    Idempotency,
    Reversibility,
)
from personagraph.tools.workspace.workspace_tools import FrozenWorkspaceToolBoundary
from personagraph.tools.workspace.workspace_write_catalog import (
    WORKSPACE_WRITE_BINDING_ASSERTION_SCHEMA,
    WorkspaceWriteBindingFacts,
    WorkspaceWriteCatalogError,
    build_workspace_write_tool_bindings,
    build_workspace_write_tool_definition_manifest,
    derive_workspace_write_source_fingerprint,
)
from personagraph.tools.workspace.workspace_write_tools import (
    WORKSPACE_WRITE_SOURCE_DISPLAY_NAME,
    WORKSPACE_WRITE_SOURCE_ID,
    WORKSPACE_WRITE_TOOL_CONTRACT_VERSION,
    WORKSPACE_WRITE_TOOL_ID,
    WORKSPACE_WRITE_TOOL_IMPLEMENTATION_VERSION,
    build_workspace_write_tool_registrations,
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _binding_facts() -> WorkspaceWriteBindingFacts:
    return WorkspaceWriteBindingFacts(
        boundary_sha256=_sha256("exact-workspace-boundary"),
        update_authority_sha256=_sha256("session-and-grant-identity"),
        protected_dispatch_sha256=_sha256("trusted-protected-dispatch-route"),
        operation_ledger_sha256=_sha256("trusted-operation-ledger-authority"),
    )


def _frozen_registration(tmp_path):
    boundary = FrozenWorkspaceToolBoundary(
        session_id="session-workspace-write-catalog",
        root=tmp_path,
    )
    (registration,) = build_workspace_write_tool_registrations(boundary)
    facts = _binding_facts()
    return replace(
        registration,
        source=replace(
            registration.source,
            fingerprint=derive_workspace_write_source_fingerprint(facts),
        ),
    )


def test_definition_is_stable_context_free_and_declares_protected_effects() -> None:
    (first,) = build_workspace_write_tool_definition_manifest()
    (second,) = build_workspace_write_tool_definition_manifest()

    definition = first.definition
    assert definition.identity.tool_id == WORKSPACE_WRITE_TOOL_ID
    assert definition.identity.contract_version == WORKSPACE_WRITE_TOOL_CONTRACT_VERSION
    assert (
        definition.identity.implementation_version
        == WORKSPACE_WRITE_TOOL_IMPLEMENTATION_VERSION
    )
    assert first.implementation_ref == "builtin/write_workspace_file"
    assert first.declared_behavior_revision == "write-workspace-file-handler-1"
    assert definition.digest == second.definition.digest
    assert definition.execution_profile.max_transparent_retries == 0

    read, update = definition.effect_template.effects
    assert (
        read.resource,
        read.action,
        read.scope_kind,
        read.default_scope,
        read.resource_argument,
        read.data_egress,
        read.idempotency,
        read.reversibility,
    ) == (
        EffectResource.FILESYSTEM,
        EffectAction.READ,
        EffectScopeKind.WORKSPACE,
        "*",
        "path",
        DataEgress.METADATA,
        Idempotency.IDEMPOTENT,
        Reversibility.REVERSIBLE,
    )
    assert (
        update.resource,
        update.action,
        update.scope_kind,
        update.default_scope,
        update.resource_argument,
        update.data_egress,
        update.idempotency,
        update.reversibility,
    ) == (
        EffectResource.FILESYSTEM,
        EffectAction.UPDATE,
        EffectScopeKind.WORKSPACE,
        "*",
        "path",
        DataEgress.NONE,
        Idempotency.NOT_IDEMPOTENT,
        Reversibility.IRREVERSIBLE,
    )
    serialized = json.dumps(definition.descriptor(), sort_keys=True)
    assert "exact-workspace-boundary" not in serialized
    assert "session-and-grant-identity" not in serialized
    assert "approval_receipt" not in serialized
    assert "handler" not in serialized
    assert "source" not in definition.descriptor()


def test_frozen_registration_binds_without_changing_legacy_descriptor(
    tmp_path,
) -> None:
    registration = _frozen_registration(tmp_path)
    (manifest_item,) = build_workspace_write_tool_definition_manifest()
    (binding,) = build_workspace_write_tool_bindings(
        (registration,),
        facts=_binding_facts(),
    )

    assert isinstance(binding, ToolBinding)
    bound = BoundToolRegistration(manifest_item.definition, binding)
    assert bound.descriptor() == registration.descriptor()
    assert bound.spec.to_dict() == registration.spec.to_dict()
    assert bound.handler is registration.handler
    assert {
        effect.default_scope for effect in binding.effect_profile.effects
    } == {str(tmp_path.resolve())}


def test_binding_assertion_contains_only_exact_secret_free_identities(
    tmp_path,
) -> None:
    facts = _binding_facts()
    registration = _frozen_registration(tmp_path)
    (binding,) = build_workspace_write_tool_bindings(
        (registration,),
        facts=facts,
    )

    assertion = dict(binding.binding_assertion)
    assert set(assertion) == {
        "schema_version",
        "binding_kind",
        "boundary_sha256",
        "update_authority_sha256",
        "protected_dispatch_sha256",
        "operation_ledger_sha256",
        "source_fingerprint",
    }
    assert assertion["schema_version"] == WORKSPACE_WRITE_BINDING_ASSERTION_SCHEMA
    assert assertion["binding_kind"] == "protected_workspace_write"
    assert assertion["source_fingerprint"] == (
        derive_workspace_write_source_fingerprint(facts)
    )
    encoded = json.dumps(assertion, sort_keys=True)
    for private_value in (
        str(tmp_path),
        "session-and-grant-identity",
        "trusted-protected-dispatch-route",
        "trusted-operation-ledger-authority",
    ):
        assert private_value not in encoded
    for field_name in (
        "boundary_sha256",
        "update_authority_sha256",
        "protected_dispatch_sha256",
        "operation_ledger_sha256",
        "source_fingerprint",
    ):
        assert len(assertion[field_name]) == 64


@pytest.mark.parametrize(
    "field_name",
    (
        "boundary_sha256",
        "update_authority_sha256",
        "protected_dispatch_sha256",
        "operation_ledger_sha256",
    ),
)
def test_source_fingerprint_binds_every_contextual_identity(
    field_name: str,
) -> None:
    facts = _binding_facts()
    changed = replace(facts, **{field_name: _sha256(f"changed:{field_name}")})

    assert derive_workspace_write_source_fingerprint(changed) != (
        derive_workspace_write_source_fingerprint(facts)
    )


def test_binding_rejects_unfrozen_or_mismatched_source_identity(tmp_path) -> None:
    boundary = FrozenWorkspaceToolBoundary(
        session_id="session-workspace-write-catalog",
        root=tmp_path,
    )
    raw = build_workspace_write_tool_registrations(boundary)
    with pytest.raises(
        WorkspaceWriteCatalogError,
        match="source fingerprint drifted",
    ):
        build_workspace_write_tool_bindings(raw, facts=_binding_facts())

    registration = _frozen_registration(tmp_path)
    wrong_fingerprint = replace(
        registration,
        source=replace(registration.source, fingerprint=_sha256("other facts")),
    )
    with pytest.raises(
        WorkspaceWriteCatalogError,
        match="source fingerprint drifted",
    ):
        build_workspace_write_tool_bindings(
            (wrong_fingerprint,),
            facts=_binding_facts(),
        )

    wrong_source = replace(
        registration,
        source=replace(registration.source, kind=ToolSourceKind.PROVIDER),
    )
    with pytest.raises(WorkspaceWriteCatalogError, match="source drifted"):
        build_workspace_write_tool_bindings(
            (wrong_source,),
            facts=_binding_facts(),
        )
    assert registration.source.source_id == WORKSPACE_WRITE_SOURCE_ID
    assert registration.source.display_name == WORKSPACE_WRITE_SOURCE_DISPLAY_NAME


def test_binding_rejects_identity_model_execution_and_effect_drift(
    tmp_path,
) -> None:
    registration = _frozen_registration(tmp_path)

    identity_drift = replace(registration, implementation_version="different")
    with pytest.raises(WorkspaceWriteCatalogError, match="identity drifted"):
        build_workspace_write_tool_bindings(
            (identity_drift,),
            facts=_binding_facts(),
        )

    model_drift = replace(
        registration,
        spec=replace(registration.spec, name="Changed write contract"),
    )
    with pytest.raises(WorkspaceWriteCatalogError, match="model contract drifted"):
        build_workspace_write_tool_bindings(
            (model_drift,),
            facts=_binding_facts(),
        )

    execution_drift = replace(
        registration,
        execution_profile=replace(
            registration.execution_profile,
            max_transparent_retries=1,
        ),
    )
    with pytest.raises(
        WorkspaceWriteCatalogError,
        match="execution contract drifted",
    ):
        build_workspace_write_tool_bindings(
            (execution_drift,),
            facts=_binding_facts(),
        )

    read, update = registration.effect_profile.effects
    effect_drift = replace(
        registration,
        effect_profile=replace(
            registration.effect_profile,
            effects=(read, replace(update, data_egress=DataEgress.CONTENT)),
        ),
    )
    with pytest.raises(WorkspaceWriteCatalogError, match="effects drifted"):
        build_workspace_write_tool_bindings(
            (effect_drift,),
            facts=_binding_facts(),
        )

    mismatched_scopes = replace(
        registration,
        effect_profile=replace(
            registration.effect_profile,
            effects=(read, replace(update, default_scope="/other/workspace")),
        ),
    )
    with pytest.raises(WorkspaceWriteCatalogError, match="effects drifted"):
        build_workspace_write_tool_bindings(
            (mismatched_scopes,),
            facts=_binding_facts(),
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "boundary_sha256",
        "update_authority_sha256",
        "protected_dispatch_sha256",
        "operation_ledger_sha256",
    ),
)
def test_binding_facts_require_canonical_hashes(field_name: str) -> None:
    values = {
        "boundary_sha256": _sha256("boundary"),
        "update_authority_sha256": _sha256("authority"),
        "protected_dispatch_sha256": _sha256("dispatch"),
        "operation_ledger_sha256": _sha256("ledger"),
    }
    values[field_name] = "bad"

    with pytest.raises(WorkspaceWriteCatalogError, match=field_name):
        WorkspaceWriteBindingFacts(**values)


def test_catalog_owner_does_not_read_runtime_or_session_state() -> None:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "src/personagraph/tools/workspace/workspace_write_catalog.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    absolute_imports = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0
    }
    absolute_imports.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any(
        module == forbidden or module.startswith(f"{forbidden}.")
        for module in absolute_imports
        for forbidden in (
            "personagraph.runtime",
            "personagraph.l1",
            "personagraph.l2",
            "personagraph.session",
        )
    )
    assert not any(
        node.level > 0
        and (node.module or "").split(".", 1)[0]
        in {"runtime", "l1", "l2", "session"}
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
