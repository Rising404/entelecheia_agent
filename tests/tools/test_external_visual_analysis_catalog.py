from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from personagraph.input_processing.vision.contracts import (
    VisionCapabilitySnapshot,
    VisionPurpose,
)
from personagraph.tools.catalog.binding import BoundToolRegistration, ToolBinding
from personagraph.tools.contracts import ToolSourceKind
from personagraph.tools.documents.external_visual_analysis_catalog import (
    EXTERNAL_VISUAL_ANALYSIS_BINDING_ASSERTION_SCHEMA,
    ExternalVisualAnalysisBindingFacts,
    ExternalVisualAnalysisCatalogError,
    build_external_visual_analysis_tool_bindings,
    build_external_visual_analysis_tool_definition_manifest,
)
from personagraph.tools.documents.format_observation_tools import (
    EXTERNAL_VISUAL_ANALYSIS_TOOL_IDS,
    VISUAL_OBSERVATION_TOOL_CONTRACT_VERSION,
    VISUAL_OBSERVATION_TOOL_IMPLEMENTATION_VERSION,
    build_external_visual_analysis_tool_registrations,
)
from personagraph.tools.effects import (
    DataEgress,
    EffectAction,
    EffectResource,
    EffectScopeKind,
    Idempotency,
    Reversibility,
)
from personagraph.tools.workspace.workspace_tools import FrozenWorkspaceToolBoundary


_EXPECTED_DECLARATIONS = (
    ("analyze_image", "builtin/analyze_image", "analyze-image-handler-3"),
    (
        "analyze_pdf_page",
        "builtin/analyze_pdf_page",
        "analyze-pdf-page-batch-handler",
    ),
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _ProviderAdapter:
    transmits_externally = True

    def __init__(self, identity: str) -> None:
        self.identity = identity

    def capabilities(self) -> VisionCapabilitySnapshot:
        return VisionCapabilitySnapshot(
            available=True,
            provider=f"provider-{self.identity}",
            model=f"model-{self.identity}",
            endpoint_identity=f"endpoint-{self.identity}",
            processor_fingerprint=f"processor-{self.identity}",
            supported_purposes=tuple(VisionPurpose),
        )

    def analyze(self, request):  # pragma: no cover - registration-only fixture
        raise AssertionError("catalog construction must not call the provider")


def _boundary(tmp_path, *, timeout_s: float = 10.0):
    return FrozenWorkspaceToolBoundary(
        session_id="session-external-visual-catalog",
        root=tmp_path,
        timeout_s=timeout_s,
    )


def _frozen_registrations(tmp_path, *, provider: str = "alpha"):
    return tuple(
        replace(
            registration,
            source=replace(
                registration.source,
                fingerprint=_sha256(
                    f"external-visual-source:{provider}:{registration.tool_id}"
                ),
            ),
        )
        for registration in build_external_visual_analysis_tool_registrations(
            _boundary(tmp_path),
            vision_adapter=_ProviderAdapter(provider),

        )
    )


def _binding_facts(*, provider: str = "alpha"):
    return ExternalVisualAnalysisBindingFacts(
        boundary_sha256=_sha256("private/session/root"),
        read_authority_sha256=_sha256("private-read-grant"),
        provider_identity_sha256=_sha256(f"provider:{provider}"),
        capability_snapshot_sha256=_sha256(f"capabilities:{provider}"),
        egress_policy_sha256=_sha256("private-disclosure-receipts"),
        physical_call_ledger_sha256=_sha256("private-ledger-path-and-schema"),
    )


def test_definition_manifest_is_stable_provider_neutral_and_ordered() -> None:
    first = build_external_visual_analysis_tool_definition_manifest()
    second = build_external_visual_analysis_tool_definition_manifest()

    assert tuple(item.definition.identity.tool_id for item in first) == (
        EXTERNAL_VISUAL_ANALYSIS_TOOL_IDS
    )
    assert tuple(
        (
            item.definition.identity.tool_id,
            item.implementation_ref,
            item.declared_behavior_revision,
        )
        for item in first
    ) == _EXPECTED_DECLARATIONS
    assert [item.definition.digest for item in first] == [
        item.definition.digest for item in second
    ]

    encoded = json.dumps(
        [item.definition.descriptor() for item in first],
        sort_keys=True,
    )
    for forbidden in (
        "session-external-visual-catalog",
        "provider-alpha",
        "endpoint-alpha",
        "processor-alpha",
        "private-read-grant",
    ):
        assert forbidden not in encoded

    for item in first:
        identity = item.definition.identity
        assert identity.contract_version == VISUAL_OBSERVATION_TOOL_CONTRACT_VERSION
        assert (
            identity.implementation_version
            == VISUAL_OBSERVATION_TOOL_IMPLEMENTATION_VERSION
        )
        filesystem, network, publication = item.definition.effect_template.effects
        assert (publication.resource, publication.action, publication.scope_kind, publication.default_scope) == (
            EffectResource.RUNTIME_STATE, EffectAction.UPDATE, EffectScopeKind.SESSION, "*",
        )
        assert (
            filesystem.resource,
            filesystem.action,
            filesystem.scope_kind,
            filesystem.default_scope,
            filesystem.data_egress,
        ) == (
            EffectResource.FILESYSTEM,
            EffectAction.READ,
            EffectScopeKind.WORKSPACE,
            "*",
            DataEgress.CONTENT,
        )
        assert filesystem.resource_argument == "path"
        assert (
            network.resource,
            network.action,
            network.scope_kind,
            network.default_scope,
            network.data_egress,
            network.idempotency,
            network.reversibility,
        ) == (
            EffectResource.NETWORK,
            EffectAction.TRANSMIT,
            EffectScopeKind.SESSION,
            "*",
            DataEgress.CONTENT,
            Idempotency.NOT_IDEMPOTENT,
            Reversibility.IRREVERSIBLE,
        )


def test_provider_fingerprint_is_binding_identity_not_implementation_version(
    tmp_path,
) -> None:
    first = _frozen_registrations(tmp_path, provider="alpha")
    second = _frozen_registrations(tmp_path, provider="beta")

    assert {
        registration.implementation_version for registration in (*first, *second)
    } == {VISUAL_OBSERVATION_TOOL_IMPLEMENTATION_VERSION}
    assert [registration.spec for registration in first] == [
        registration.spec for registration in second
    ]
    first_bindings = build_external_visual_analysis_tool_bindings(
        first,
        facts=_binding_facts(provider="alpha"),
    )
    second_bindings = build_external_visual_analysis_tool_bindings(
        second,
        facts=_binding_facts(provider="beta"),
    )
    assert [binding.identity for binding in first_bindings] == [
        binding.identity for binding in second_bindings
    ]
    assert [binding.definition_digest for binding in first_bindings] == [
        binding.definition_digest for binding in second_bindings
    ]
    assert [binding.digest for binding in first_bindings] != [
        binding.digest for binding in second_bindings
    ]


def test_frozen_registrations_bind_without_changing_handler_or_descriptor(
    tmp_path,
) -> None:
    registrations = _frozen_registrations(tmp_path)
    manifest = build_external_visual_analysis_tool_definition_manifest()
    bindings = build_external_visual_analysis_tool_bindings(
        registrations,
        facts=_binding_facts(),
    )

    assert len(bindings) == 2
    assert all(isinstance(binding, ToolBinding) for binding in bindings)
    for registration, item, binding in zip(
        registrations,
        manifest,
        bindings,
        strict=True,
    ):
        bound = BoundToolRegistration(item.definition, binding)
        assert bound.descriptor() == registration.descriptor()
        assert bound.handler is registration.handler
        filesystem, network, publication = binding.effect_profile.effects
        assert publication.default_scope == "session-external-visual-catalog"
        assert filesystem.default_scope == str(tmp_path.resolve())
        assert network.default_scope == "session-external-visual-catalog"


def test_binding_assertions_are_fixed_secret_free_identity_records(tmp_path) -> None:
    bindings = build_external_visual_analysis_tool_bindings(
        _frozen_registrations(tmp_path),
        facts=_binding_facts(),
    )

    expected_keys = {
        "schema_version",
        "binding_kind",
        "boundary_sha256",
        "read_authority_sha256",
        "provider_identity_sha256",
        "capability_snapshot_sha256",
        "egress_policy_sha256",
        "physical_call_ledger_sha256",
        "source_fingerprint",
    }
    for binding in bindings:
        assertion = dict(binding.binding_assertion)
        assert set(assertion) == expected_keys
        assert (
            assertion["schema_version"]
            == EXTERNAL_VISUAL_ANALYSIS_BINDING_ASSERTION_SCHEMA
        )
        encoded = json.dumps(assertion, sort_keys=True)
        for forbidden in (
            str(tmp_path),
            "session-external-visual-catalog",
            "private-read-grant",
            "private-disclosure-receipts",
            "provider-alpha",
        ):
            assert forbidden not in encoded
        for key in expected_keys - {"schema_version", "binding_kind"}:
            assert isinstance(assertion[key], str)
            assert len(assertion[key]) == 64


def test_external_execution_envelope_does_not_drift_with_workspace_timeout(
    tmp_path,
) -> None:
    registration = build_external_visual_analysis_tool_registrations(
        _boundary(tmp_path, timeout_s=600.0),
        vision_adapter=_ProviderAdapter("alpha"),

    )[0]

    assert registration.execution_profile.default_timeout_s == 120.0
    assert registration.execution_profile.hard_timeout_s == 120.0


def test_binding_rejects_legacy_provider_suffixed_identity_and_other_drift(
    tmp_path,
) -> None:
    registrations = _frozen_registrations(tmp_path)
    old_identity = (
        replace(
            registrations[0],
            implementation_version=(
                f"{VISUAL_OBSERVATION_TOOL_IMPLEMENTATION_VERSION}+processor-alpha"
            ),
        ),
        registrations[1],
    )
    with pytest.raises(
        ExternalVisualAnalysisCatalogError,
        match="identity drifted",
    ):
        build_external_visual_analysis_tool_bindings(
            old_identity,
            facts=_binding_facts(),
        )

    wrong_source = (
        replace(
            registrations[0],
            source=replace(registrations[0].source, kind=ToolSourceKind.PROVIDER),
        ),
        registrations[1],
    )
    with pytest.raises(
        ExternalVisualAnalysisCatalogError,
        match="source drifted",
    ):
        build_external_visual_analysis_tool_bindings(
            wrong_source,
            facts=_binding_facts(),
        )

    filesystem, network, publication = registrations[0].effect_profile.effects
    wrong_effect = (
        replace(
            registrations[0],
            effect_profile=replace(
                registrations[0].effect_profile,
                effects=(
                    filesystem,
                    replace(network, idempotency=Idempotency.IDEMPOTENT),
                    publication,
                ),
            ),
        ),
        registrations[1],
    )
    with pytest.raises(
        ExternalVisualAnalysisCatalogError,
        match="effects drifted",
    ):
        build_external_visual_analysis_tool_bindings(
            wrong_effect,
            facts=_binding_facts(),
        )


def test_binding_requires_exact_order_source_fingerprint_and_hash_facts(
    tmp_path,
) -> None:
    registrations = _frozen_registrations(tmp_path)
    with pytest.raises(
        ExternalVisualAnalysisCatalogError,
        match="canonical order",
    ):
        build_external_visual_analysis_tool_bindings(
            tuple(reversed(registrations)),
            facts=_binding_facts(),
        )

    raw = build_external_visual_analysis_tool_registrations(
        _boundary(tmp_path),
        vision_adapter=_ProviderAdapter("alpha"),

    )
    with pytest.raises(
        ExternalVisualAnalysisCatalogError,
        match="source fingerprint",
    ):
        build_external_visual_analysis_tool_bindings(
            raw,
            facts=_binding_facts(),
        )

    with pytest.raises(
        ExternalVisualAnalysisCatalogError,
        match="provider_identity_sha256",
    ):
        ExternalVisualAnalysisBindingFacts(
            boundary_sha256=_sha256("boundary"),
            read_authority_sha256=_sha256("read"),
            provider_identity_sha256="bad",
            capability_snapshot_sha256=_sha256("capability"),
            egress_policy_sha256=_sha256("disclosure"),
            physical_call_ledger_sha256=_sha256("physical"),
        )
