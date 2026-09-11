from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from personagraph.tools.catalog.binding import BoundToolRegistration, ToolBinding
from personagraph.tools.effects import (
    DataEgress,
    EffectAction,
    EffectResource,
    Idempotency,
    Reversibility,
)
from personagraph.tools.visual.file_visual_catalog import (
    FILE_VISUAL_BINDING_ASSERTION_SCHEMA,
    ExternalFileVisualReadBindingFacts,
    FileVisualBindingFacts,
    FileVisualCatalogError,
    build_file_visual_tool_bindings,
    build_file_visual_tool_definition_manifest,
    derive_file_visual_source_fingerprint,
)
from personagraph.tools.visual.file_visual_tools import (
    FILE_VISUAL_SOURCE_ID,
    FILE_VISUAL_TOOL_IDS,
    LIST_FILE_VISUALS_TOOL_ID,
    READ_FILE_VISUALS_TOOL_ID,
    build_list_file_visuals_registration,
    build_read_file_visuals_registration,
)


_EFFECT_SCOPE = "file-visuals:" + "e" * 64
_DECLARATIONS = (
    (
        LIST_FILE_VISUALS_TOOL_ID,
        "builtin/list_file_visuals",
        "file-visual-list-handler-4",
    ),
    (
        READ_FILE_VISUALS_TOOL_ID,
        "builtin/read_file_visuals",
        "file-visual-external-read-handler-4",
    ),
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _handler(payload: dict[str, object]) -> dict[str, object]:
    return dict(payload)


def _facts(*, external: bool) -> FileVisualBindingFacts:
    return FileVisualBindingFacts(
        session_scope_sha256=_sha256("private-session-scope"),
        file_access_policy_sha256=_sha256("private-receipt-and-paths"),
        external_read=(
            ExternalFileVisualReadBindingFacts(
                provider_identity_sha256=_sha256("private-provider"),
                capability_snapshot_sha256=_sha256("private-capabilities"),
                egress_policy_sha256=_sha256("private-disclosures"),
                physical_call_ledger_sha256=_sha256("private-ledger-path"),
            )
            if external
            else None
        ),
    )


def _registrations(*, sends_externally: bool, external_facts: bool):
    facts = _facts(external=external_facts)
    raw = (
        build_list_file_visuals_registration(
            handler=_handler,
            effect_scope=_EFFECT_SCOPE,
        ),
        build_read_file_visuals_registration(
            handler=_handler,
            effect_scope=_EFFECT_SCOPE,
            sends_externally=sends_externally,
        ),
    )
    frozen = []
    for registration in raw:
        if (
            registration.tool_id == READ_FILE_VISUALS_TOOL_ID
            and facts.external_read is None
        ):
            frozen.append(registration)
            continue
        frozen.append(
            replace(
                registration,
                source=replace(
                    registration.source,
                    fingerprint=derive_file_visual_source_fingerprint(
                        facts,
                        tool_id=registration.tool_id,
                    ),
                ),
            )
        )
    return facts, tuple(frozen)


def test_definition_manifest_is_stable_and_external_read_is_truthful() -> None:
    first = build_file_visual_tool_definition_manifest()
    second = build_file_visual_tool_definition_manifest()

    assert tuple(item.definition.identity.tool_id for item in first) == (
        FILE_VISUAL_TOOL_IDS
    )
    assert tuple(
        (
            item.definition.identity.tool_id,
            item.implementation_ref,
            item.declared_behavior_revision,
        )
        for item in first
    ) == _DECLARATIONS
    assert tuple(item.definition.digest for item in first) == tuple(
        item.definition.digest for item in second
    )

    listed, read = (item.definition for item in first)
    assert len(listed.effect_template.effects) == 1
    filesystem, network, runtime_state = read.effect_template.effects
    assert (
        filesystem.resource,
        filesystem.action,
        filesystem.default_scope,
        filesystem.data_egress,
    ) == (
        EffectResource.FILESYSTEM,
        EffectAction.READ,
        "*",
        DataEgress.CONTENT,
    )
    assert (
        network.resource,
        network.action,
        network.default_scope,
        network.data_egress,
        network.idempotency,
        network.reversibility,
    ) == (
        EffectResource.NETWORK,
        EffectAction.TRANSMIT,
        "*",
        DataEgress.CONTENT,
        Idempotency.NOT_IDEMPOTENT,
        Reversibility.IRREVERSIBLE,
    )
    assert (
        runtime_state.resource,
        runtime_state.action,
        runtime_state.default_scope,
        runtime_state.data_egress,
        runtime_state.idempotency,
        runtime_state.reversibility,
    ) == (
        EffectResource.RUNTIME_STATE,
        EffectAction.UPDATE,
        "*",
        DataEgress.NONE,
        Idempotency.DEDUPLICATED,
        Reversibility.UNKNOWN,
    )
    assert read.execution_profile.max_transparent_retries == 0

    encoded = json.dumps(
        [item.definition.descriptor() for item in first],
        sort_keys=True,
    )
    for private in (
        "private-session-scope",
        "private-receipt-and-paths",
        "private-provider",
        "private-ledger-path",
    ):
        assert private not in encoded


def test_external_registrations_bind_exactly_without_private_values() -> None:
    facts, registrations = _registrations(
        sends_externally=True,
        external_facts=True,
    )
    bindings = build_file_visual_tool_bindings(registrations, facts=facts)
    definitions = build_file_visual_tool_definition_manifest()

    assert len(bindings) == 2
    assert all(isinstance(binding, ToolBinding) for binding in bindings)
    for registration, item, binding in zip(
        registrations,
        definitions,
        bindings,
        strict=True,
    ):
        bound = BoundToolRegistration(item.definition, binding)
        assert bound.descriptor() == registration.descriptor()
        assert bound.handler is registration.handler
        assert binding.source.source_id == FILE_VISUAL_SOURCE_ID

    list_assertion = dict(bindings[0].binding_assertion)
    read_assertion = dict(bindings[1].binding_assertion)
    assert set(list_assertion) == {
        "schema_version",
        "binding_kind",
        "session_scope_sha256",
        "file_access_policy_sha256",
        "source_fingerprint",
    }
    assert set(read_assertion) == {
        *list_assertion,
        "provider_identity_sha256",
        "capability_snapshot_sha256",
        "egress_policy_sha256",
        "physical_call_ledger_sha256",
    }
    assert (
        read_assertion["schema_version"]
        == FILE_VISUAL_BINDING_ASSERTION_SCHEMA
    )
    encoded = json.dumps(
        [binding.descriptor() for binding in bindings],
        sort_keys=True,
    )
    for private in (
        "private-session-scope",
        "private-receipt-and-paths",
        "private-provider",
        "private-disclosures",
        "private-ledger-path",
    ):
        assert private not in encoded


def test_local_or_unavailable_adapter_binds_list_only() -> None:
    facts, registrations = _registrations(
        sends_externally=False,
        external_facts=False,
    )

    bindings = build_file_visual_tool_bindings(
        registrations[:1],
        facts=facts,
    )

    assert tuple(binding.identity.tool_id for binding in bindings) == (
        LIST_FILE_VISUALS_TOOL_ID,
    )
    assert len(registrations[1].effect_profile.effects) == 1
    assert registrations[1].execution_profile.max_transparent_retries == 1
    with pytest.raises(
        FileVisualCatalogError,
        match="external read registration requires",
    ):
        build_file_visual_tool_bindings(registrations, facts=facts)


def test_local_read_cannot_impersonate_the_external_definition() -> None:
    facts, registrations = _registrations(
        sends_externally=False,
        external_facts=True,
    )

    with pytest.raises(FileVisualCatalogError, match="execution contract drifted"):
        build_file_visual_tool_bindings(registrations, facts=facts)


def test_binding_rejects_source_and_authority_drift() -> None:
    facts, registrations = _registrations(
        sends_externally=True,
        external_facts=True,
    )
    bad_source = replace(
        registrations[0],
        source=replace(registrations[0].source, fingerprint="0" * 64),
    )
    with pytest.raises(FileVisualCatalogError, match="fingerprint drifted"):
        build_file_visual_tool_bindings(
            (bad_source, registrations[1]),
            facts=facts,
        )
    with pytest.raises(FileVisualCatalogError, match="canonical SHA-256"):
        FileVisualBindingFacts(
            session_scope_sha256="not-a-digest",
            file_access_policy_sha256=_sha256("authority"),
        )
