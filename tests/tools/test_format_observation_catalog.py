from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from personagraph.input_processing.vision.providers import (
    UnavailableVisionModelAdapter,
)
from personagraph.tools.catalog.binding import (
    BoundToolRegistration,
    ToolBinding,
)
from personagraph.tools.contracts import ToolSourceKind
from personagraph.tools.documents.format_observation_catalog import (
    FORMAT_OBSERVATION_BINDING_ASSERTION_SCHEMA,
    FormatObservationBindingFacts,
    FormatObservationCatalogError,
    build_format_observation_tool_bindings,
    build_format_observation_tool_definition_manifest,
)
from personagraph.tools.documents.format_observation_tools import (
    FORMAT_OBSERVATION_LOCAL_TOOL_IDS,
    FORMAT_OBSERVATION_TOOL_CONTRACT_VERSION,
    FORMAT_OBSERVATION_TOOL_IMPLEMENTATION_VERSION,
    VISUAL_OBSERVATION_TOOL_CONTRACT_VERSION,
    VISUAL_OBSERVATION_TOOL_IMPLEMENTATION_VERSION,
    build_external_visual_analysis_tool_registrations,
    build_format_observation_tool_registrations,
)
from personagraph.tools.effects import DataEgress
from personagraph.tools.workspace.workspace_tools import (
    FrozenWorkspaceToolBoundary,
)


_EXPECTED_DECLARATIONS = (
    ("read_text", "builtin/read_text", "read-text-handler-2"),
    (
        "read_pdf_text",
        "builtin/read_pdf_text",
        "read-pdf-text-handler-2",
    ),
    ("read_word", "builtin/read_word", "read-word-handler-2"),
    ("read_slides", "builtin/read_slides", "read-slides-handler-2"),
    ("inspect_image", "builtin/inspect_image", "inspect-image-handler-3"),
)
_VISUAL_LOCAL_TOOL_IDS = frozenset({"inspect_image"})


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _boundary(tmp_path) -> FrozenWorkspaceToolBoundary:
    return FrozenWorkspaceToolBoundary(
        session_id="session-format-observation-catalog",
        root=tmp_path,
    )


def _frozen_registrations(tmp_path):
    return tuple(
        replace(
            registration,
            source=replace(
                registration.source,
                fingerprint=_sha256(
                    f"format-observation-source:{registration.tool_id}"
                ),
            ),
        )
        for registration in build_format_observation_tool_registrations(
            _boundary(tmp_path)
        )
    )


def _binding_facts() -> FormatObservationBindingFacts:
    return FormatObservationBindingFacts(
        boundary_sha256=_sha256("format-observation-boundary"),
        read_authority_sha256=_sha256("grant-receipt-secret"),
        reader_stack_sha256=_sha256("local-reader-stack"),
    )


def test_definition_manifest_is_stable_local_only_and_ordered() -> None:
    first = build_format_observation_tool_definition_manifest()
    second = build_format_observation_tool_definition_manifest()

    assert tuple(item.definition.identity.tool_id for item in first) == (
        FORMAT_OBSERVATION_LOCAL_TOOL_IDS
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
    assert not {
        "analyze_image",
        "analyze_pdf_page",
    }.intersection(item.definition.identity.tool_id for item in first)

    for item in first:
        identity = item.definition.identity
        if identity.tool_id in _VISUAL_LOCAL_TOOL_IDS:
            assert identity.contract_version == VISUAL_OBSERVATION_TOOL_CONTRACT_VERSION
            assert (
                identity.implementation_version
                == VISUAL_OBSERVATION_TOOL_IMPLEMENTATION_VERSION
            )
        else:
            assert (
                identity.contract_version
                == FORMAT_OBSERVATION_TOOL_CONTRACT_VERSION
            )
            assert (
                identity.implementation_version
                == FORMAT_OBSERVATION_TOOL_IMPLEMENTATION_VERSION
            )
        (effect,) = item.definition.effect_template.effects
        assert effect.default_scope == "*"
        assert effect.resource_argument == "path"
        assert effect.data_egress is DataEgress.CONTENT


def test_frozen_registrations_bind_without_changing_legacy_descriptor(
    tmp_path,
) -> None:
    registrations = _frozen_registrations(tmp_path)
    manifest = build_format_observation_tool_definition_manifest()
    bindings = build_format_observation_tool_bindings(
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
        assert binding.effect_profile.effects[0].default_scope == str(
            tmp_path.resolve()
        )


def test_binding_assertions_are_fixed_secret_free_fingerprint_records(
    tmp_path,
) -> None:
    bindings = build_format_observation_tool_bindings(
        _frozen_registrations(tmp_path),
        facts=_binding_facts(),
    )

    expected_keys = {
        "schema_version",
        "binding_kind",
        "boundary_sha256",
        "read_authority_sha256",
        "reader_stack_sha256",
        "source_fingerprint",
    }
    for binding in bindings:
        assertion = dict(binding.binding_assertion)
        assert set(assertion) == expected_keys
        assert (
            assertion["schema_version"]
            == FORMAT_OBSERVATION_BINDING_ASSERTION_SCHEMA
        )
        encoded = json.dumps(assertion, sort_keys=True)
        assert str(tmp_path) not in encoded
        assert "grant-receipt-secret" not in encoded
        for key in (
            "boundary_sha256",
            "read_authority_sha256",
            "reader_stack_sha256",
            "source_fingerprint",
        ):
            assert isinstance(assertion[key], str)
            assert len(assertion[key]) == 64


def test_binding_rejects_external_analysis_and_noncanonical_order(
    tmp_path,
) -> None:
    registrations = _frozen_registrations(tmp_path)
    external = build_external_visual_analysis_tool_registrations(
        _boundary(tmp_path),
        vision_adapter=UnavailableVisionModelAdapter(),
    )

    with pytest.raises(
        FormatObservationCatalogError,
        match="exactly five",
    ):
        build_format_observation_tool_bindings(
            (*registrations, *external),
            facts=_binding_facts(),
        )
    with pytest.raises(
        FormatObservationCatalogError,
        match="canonical order",
    ):
        build_format_observation_tool_bindings(
            tuple(reversed(registrations)),
            facts=_binding_facts(),
        )


def test_binding_rejects_identity_model_and_source_drift(tmp_path) -> None:
    registrations = _frozen_registrations(tmp_path)

    identity_drift = (
        replace(registrations[0], implementation_version="different"),
        *registrations[1:],
    )
    with pytest.raises(FormatObservationCatalogError, match="identity drifted"):
        build_format_observation_tool_bindings(
            identity_drift,
            facts=_binding_facts(),
        )

    model_drift = (
        replace(
            registrations[0],
            spec=replace(registrations[0].spec, name="Changed model wire"),
        ),
        *registrations[1:],
    )
    with pytest.raises(
        FormatObservationCatalogError,
        match="model contract drifted",
    ):
        build_format_observation_tool_bindings(
            model_drift,
            facts=_binding_facts(),
        )

    source_drift = (
        replace(
            registrations[0],
            source=replace(
                registrations[0].source,
                kind=ToolSourceKind.PROVIDER,
            ),
        ),
        *registrations[1:],
    )
    with pytest.raises(FormatObservationCatalogError, match="source drifted"):
        build_format_observation_tool_bindings(
            source_drift,
            facts=_binding_facts(),
        )


def test_binding_rejects_unfrozen_or_noncanonical_source_fingerprint(
    tmp_path,
) -> None:
    raw = build_format_observation_tool_registrations(_boundary(tmp_path))
    with pytest.raises(
        FormatObservationCatalogError,
        match="source fingerprint",
    ):
        build_format_observation_tool_bindings(raw, facts=_binding_facts())

    registrations = _frozen_registrations(tmp_path)
    malformed = (
        replace(
            registrations[0],
            source=replace(registrations[0].source, fingerprint="not-a-sha256"),
        ),
        *registrations[1:],
    )
    with pytest.raises(
        FormatObservationCatalogError,
        match="source fingerprint",
    ):
        build_format_observation_tool_bindings(
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
                hard_timeout_s=121.0,
            ),
        ),
        *registrations[1:],
    )
    with pytest.raises(
        FormatObservationCatalogError,
        match="execution contract drifted",
    ):
        build_format_observation_tool_bindings(
            execution_drift,
            facts=_binding_facts(),
        )

    effect = registrations[0].effect_profile.effects[0]
    effect_drift = (
        replace(
            registrations[0],
            effect_profile=replace(
                registrations[0].effect_profile,
                effects=(replace(effect, data_egress=DataEgress.METADATA),),
            ),
        ),
        *registrations[1:],
    )
    with pytest.raises(FormatObservationCatalogError, match="effects drifted"):
        build_format_observation_tool_bindings(
            effect_drift,
            facts=_binding_facts(),
        )


def test_binding_facts_require_canonical_hashes() -> None:
    with pytest.raises(FormatObservationCatalogError, match="boundary_sha256"):
        FormatObservationBindingFacts(
            boundary_sha256="bad",
            read_authority_sha256=_sha256("read"),
            reader_stack_sha256=_sha256("readers"),
        )
