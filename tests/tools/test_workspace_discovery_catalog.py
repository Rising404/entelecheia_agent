from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from personagraph.tools.catalog.binding import (
    BoundToolRegistration,
    ToolBinding,
)
from personagraph.tools.effects import DataEgress
from personagraph.tools.workspace.workspace_discovery_catalog import (
    WORKSPACE_DISCOVERY_BINDING_ASSERTION_SCHEMA,
    WorkspaceDiscoveryBindingFacts,
    WorkspaceDiscoveryCatalogError,
    build_workspace_discovery_tool_bindings,
    build_workspace_discovery_tool_definition_manifest,
)
from personagraph.tools.workspace.workspace_tools import (
    FrozenWorkspaceToolBoundary,
    WORKSPACE_DISCOVERY_TOOL_CONTRACT_VERSION,
    WORKSPACE_DISCOVERY_TOOL_IDS,
    WORKSPACE_DISCOVERY_TOOL_IMPLEMENTATION_VERSION,
    build_workspace_discovery_tool_registrations,
)


_EXPECTED_DECLARATIONS = (
    (
        "workspace_overview",
        "builtin/workspace_overview",
        "workspace-overview-handler-2",
    ),
    (
        "list_workspace_directory",
        "builtin/list_workspace_directory",
        "list-workspace-directory-handler-2",
    ),
    ("find_files", "builtin/find_files", "find-files-handler-2"),
    (
        "search_text_files",
        "builtin/search_text_files",
        "search-text-files-handler-2",
    ),
    ("inspect_file", "builtin/inspect_file", "inspect-file-handler-2"),
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _frozen_registrations(tmp_path):
    boundary = FrozenWorkspaceToolBoundary(
        session_id="session-workspace-catalog",
        root=tmp_path,
    )
    return tuple(
        replace(
            registration,
            source=replace(
                registration.source,
                fingerprint=_sha256(
                    f"workspace-source:{registration.tool_id}"
                ),
            ),
        )
        for registration in build_workspace_discovery_tool_registrations(
            boundary
        )
    )


def _binding_facts() -> WorkspaceDiscoveryBindingFacts:
    return WorkspaceDiscoveryBindingFacts(
        boundary_sha256=_sha256("workspace-boundary"),
        read_authority_sha256=_sha256("grant-receipt-secret"),
    )


def test_definition_manifest_is_stable_context_free_and_ordered() -> None:
    first = build_workspace_discovery_tool_definition_manifest()
    second = build_workspace_discovery_tool_definition_manifest()

    assert tuple(item.definition.identity.tool_id for item in first) == (
        WORKSPACE_DISCOVERY_TOOL_IDS
    )
    assert tuple(
        (
            item.definition.identity.tool_id,
            item.implementation_ref,
            item.declared_behavior_revision,
        )
        for item in first
    ) == _EXPECTED_DECLARATIONS
    assert all(
        item.definition.identity.contract_version
        == WORKSPACE_DISCOVERY_TOOL_CONTRACT_VERSION
        for item in first
    )
    assert all(
        item.definition.identity.implementation_version
        == WORKSPACE_DISCOVERY_TOOL_IMPLEMENTATION_VERSION
        for item in first
    )
    assert [item.definition.digest for item in first] == [
        item.definition.digest for item in second
    ]
    assert all(
        tuple(effect.default_scope for effect in item.definition.effect_template.effects)
        == ("*",)
        for item in first
    )
    effects_by_tool_id = {
        item.definition.identity.tool_id: item.definition.effect_template.effects[0]
        for item in first
    }
    assert (
        effects_by_tool_id["list_workspace_directory"].data_egress
        is DataEgress.METADATA
    )
    assert effects_by_tool_id["find_files"].data_egress is DataEgress.METADATA
    assert (
        effects_by_tool_id["search_text_files"].data_egress
        is DataEgress.CONTENT
    )


def test_frozen_registrations_bind_without_changing_legacy_descriptor(
    tmp_path,
) -> None:
    registrations = _frozen_registrations(tmp_path)
    manifest = build_workspace_discovery_tool_definition_manifest()
    bindings = build_workspace_discovery_tool_bindings(
        registrations,
        facts=_binding_facts(),
    )

    assert len(bindings) == 5
    assert all(isinstance(binding, ToolBinding) for binding in bindings)
    for registration, item, binding in zip(
        registrations,
        manifest,
        bindings,
        strict=True,
    ):
        bound = BoundToolRegistration(item.definition, binding)
        assert bound.descriptor() == registration.descriptor()
        assert bound.spec.to_dict() == registration.spec.to_dict()
        assert bound.handler is registration.handler


def test_binding_assertions_are_fixed_secret_free_fingerprint_records(
    tmp_path,
) -> None:
    bindings = build_workspace_discovery_tool_bindings(
        _frozen_registrations(tmp_path),
        facts=_binding_facts(),
    )

    expected_keys = {
        "schema_version",
        "binding_kind",
        "boundary_sha256",
        "read_authority_sha256",
        "source_fingerprint",
    }
    for binding in bindings:
        assertion = dict(binding.binding_assertion)
        assert set(assertion) == expected_keys
        assert (
            assertion["schema_version"]
            == WORKSPACE_DISCOVERY_BINDING_ASSERTION_SCHEMA
        )
        encoded = json.dumps(assertion, sort_keys=True)
        assert str(tmp_path) not in encoded
        assert "grant-receipt-secret" not in encoded
        for key in (
            "boundary_sha256",
            "read_authority_sha256",
                "source_fingerprint",
        ):
            assert isinstance(assertion[key], str)
            assert len(assertion[key]) == 64


def test_binding_rejects_registration_order_and_identity_drift(tmp_path) -> None:
    registrations = _frozen_registrations(tmp_path)

    with pytest.raises(
        WorkspaceDiscoveryCatalogError,
        match="canonical order",
    ):
        build_workspace_discovery_tool_bindings(
            tuple(reversed(registrations)),
            facts=_binding_facts(),
        )

    drifted = (
        replace(registrations[0], implementation_version="different"),
        *registrations[1:],
    )
    with pytest.raises(WorkspaceDiscoveryCatalogError, match="identity drifted"):
        build_workspace_discovery_tool_bindings(
            drifted,
            facts=_binding_facts(),
        )


def test_binding_rejects_unfrozen_or_noncanonical_source_fingerprint(
    tmp_path,
) -> None:
    boundary = FrozenWorkspaceToolBoundary(
        session_id="session-workspace-catalog",
        root=tmp_path,
    )
    raw = build_workspace_discovery_tool_registrations(boundary)
    with pytest.raises(
        WorkspaceDiscoveryCatalogError,
        match="source fingerprint",
    ):
        build_workspace_discovery_tool_bindings(raw, facts=_binding_facts())

    frozen = _frozen_registrations(tmp_path)
    malformed = (
        replace(
            frozen[0],
            source=replace(frozen[0].source, fingerprint="not-a-sha256"),
        ),
        *frozen[1:],
    )
    with pytest.raises(
        WorkspaceDiscoveryCatalogError,
        match="source fingerprint",
    ):
        build_workspace_discovery_tool_bindings(
            malformed,
            facts=_binding_facts(),
        )


def test_binding_rejects_execution_and_effect_surface_drift(tmp_path) -> None:
    registrations = _frozen_registrations(tmp_path)
    execution_drift = (
        replace(
            registrations[0],
            execution_profile=replace(
                registrations[0].execution_profile,
                hard_timeout_s=99.0,
            ),
        ),
        *registrations[1:],
    )
    with pytest.raises(
        WorkspaceDiscoveryCatalogError,
        match="execution contract drifted",
    ):
        build_workspace_discovery_tool_bindings(
            execution_drift,
            facts=_binding_facts(),
        )

    effect = registrations[0].effect_profile.effects[0]
    effect_drift = (
        replace(
            registrations[0],
            effect_profile=replace(
                registrations[0].effect_profile,
                effects=(replace(effect, action=effect.action.SEARCH),),
            ),
        ),
        *registrations[1:],
    )
    with pytest.raises(WorkspaceDiscoveryCatalogError, match="effects drifted"):
        build_workspace_discovery_tool_bindings(
            effect_drift,
            facts=_binding_facts(),
        )


def test_binding_facts_require_canonical_hashes() -> None:
    with pytest.raises(WorkspaceDiscoveryCatalogError, match="boundary_sha256"):
        WorkspaceDiscoveryBindingFacts(
            boundary_sha256="bad",
            read_authority_sha256=_sha256("read"),
        )
