"""Stable Definition and contextual Binding for mounted-document cognition."""

from __future__ import annotations

import ast
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pytest

from personagraph.tools.catalog.binding import BoundToolRegistration, ToolBinding
from personagraph.tools.contracts import ToolSourceKind
from personagraph.tools.documents.frozen_mounted_document_reader import (
    FrozenMountedDocument,
    FrozenMountedDocumentReader,
)
from personagraph.tools.documents.mounted_document_catalog import (
    MOUNTED_DOCUMENT_BINDING_ASSERTION_SCHEMA,
    MountedDocumentCatalogError,
    build_mounted_document_binding_facts,
    build_mounted_document_tool_bindings,
    build_mounted_document_tool_definition_manifest,
)
from personagraph.tools.documents.mounted_document_cognition_tools import (
    MOUNTED_DOCUMENT_COGNITION_CONTRACT_VERSION,
    MOUNTED_DOCUMENT_COGNITION_IMPLEMENTATION_VERSION,
    MOUNTED_DOCUMENT_COGNITION_TOOL_IDS,
    FrozenMountedDocumentToolScope,
    build_mounted_document_cognition_tool_source,
)
from personagraph.tools.effects import DataEgress, EffectAction


SESSION_ID = "private-session-mounted-document-catalog"
SCOPE_SHA256 = hashlib.sha256(b"private mounted authority scope").hexdigest()
DOCUMENT_ID = "private-docstore-document-id"
VERSION_ID = "private-document-version-id"
SOURCE_SHA256 = hashlib.sha256(b"private source bytes").hexdigest()

_EXPECTED_DECLARATIONS = (
    (
        "inspect_mounted_document",
        "builtin/inspect_mounted_document",
        "inspect-mounted-document-handler-1",
    ),
    (
        "search_mounted_document",
        "builtin/search_mounted_document",
        "search-mounted-document-handler-1",
    ),
    (
        "read_mounted_document_chunks",
        "builtin/read_mounted_document_chunks",
        "read-mounted-document-chunks-handler-1",
    ),
)


def _document(
    *,
    alias: str = "mounted_document_01",
    session_id: str = SESSION_ID,
    document_version_id: str = VERSION_ID,
) -> FrozenMountedDocument:
    return FrozenMountedDocument(
        session_id=session_id,
        resource_alias=alias,
        document_id=DOCUMENT_ID,
        document_version_id=document_version_id,
        source_sha256=SOURCE_SHA256,
        processing_status="complete",
        resource_format="txt",
        media_type="text/plain",
        file_extension=".txt",
        total_chunk_count=3,
        processing_diagnostic_codes=(),
    )


def _scope(
    *,
    session_id: str = SESSION_ID,
    scope_snapshot_sha256: str = SCOPE_SHA256,
    document: FrozenMountedDocument | None = None,
) -> FrozenMountedDocumentToolScope:
    return FrozenMountedDocumentToolScope(
        session_id=session_id,
        scope_snapshot_sha256=scope_snapshot_sha256,
        documents=(document or _document(session_id=session_id),),
    )


def _registrations_and_facts():
    scope = _scope()
    source = build_mounted_document_cognition_tool_source(scope)
    assert source is not None
    registrations = tuple(
        source.catalog_snapshot.resolve(tool_id)
        for tool_id in MOUNTED_DOCUMENT_COGNITION_TOOL_IDS
    )
    return registrations, build_mounted_document_binding_facts(scope)


def test_definition_manifest_is_stable_context_free_and_ordered() -> None:
    first = build_mounted_document_tool_definition_manifest()
    second = build_mounted_document_tool_definition_manifest()

    assert tuple(item.definition.identity.tool_id for item in first) == (
        MOUNTED_DOCUMENT_COGNITION_TOOL_IDS
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

    for item, expected_action in zip(
        first,
        (EffectAction.READ, EffectAction.SEARCH, EffectAction.READ),
        strict=True,
    ):
        identity = item.definition.identity
        assert identity.contract_version == MOUNTED_DOCUMENT_COGNITION_CONTRACT_VERSION
        assert (
            identity.implementation_version
            == MOUNTED_DOCUMENT_COGNITION_IMPLEMENTATION_VERSION
        )
        (effect,) = item.definition.effect_template.effects
        assert effect.action is expected_action
        assert effect.default_scope == "*"
        assert effect.data_egress is DataEgress.CONTENT

    encoded = json.dumps(
        [item.definition.descriptor() for item in first],
        sort_keys=True,
    )
    for private_value in (
        SESSION_ID,
        DOCUMENT_ID,
        VERSION_ID,
        SOURCE_SHA256,
        SCOPE_SHA256,
        "mounted_document_01",
    ):
        assert private_value not in encoded


def test_binding_facts_commit_session_scope_alias_generation_and_freshness() -> None:
    base = build_mounted_document_binding_facts(_scope())
    other_session = "private-session-other"
    changed_session = build_mounted_document_binding_facts(
        _scope(
            session_id=other_session,
            scope_snapshot_sha256=hashlib.sha256(b"other scope").hexdigest(),
            document=_document(session_id=other_session),
        )
    )
    changed_alias = build_mounted_document_binding_facts(
        _scope(document=_document(alias="mounted_document_02"))
    )
    changed_generation = build_mounted_document_binding_facts(
        _scope(document=_document(document_version_id="private-version-next"))
    )

    assert changed_session.session_scope_sha256 != base.session_scope_sha256
    assert changed_session.scope_snapshot_sha256 != base.scope_snapshot_sha256
    assert (
        changed_alias.document_alias_projection_sha256
        != base.document_alias_projection_sha256
    )
    assert (
        changed_generation.document_generation_snapshot_sha256
        != base.document_generation_snapshot_sha256
    )
    assert (
        changed_generation.document_freshness_snapshot_sha256
        != base.document_freshness_snapshot_sha256
    )
    encoded = json.dumps(asdict(base), sort_keys=True)
    for private_value in (
        SESSION_ID,
        DOCUMENT_ID,
        VERSION_ID,
        SOURCE_SHA256,
        "mounted_document_01",
    ):
        assert private_value not in encoded


def test_frozen_source_binds_without_changing_handler_or_descriptor() -> None:
    registrations, facts = _registrations_and_facts()
    manifest = build_mounted_document_tool_definition_manifest()
    bindings = build_mounted_document_tool_bindings(
        registrations,
        facts=facts,
    )

    assert len(bindings) == 3
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
        assert isinstance(registration.handler.__self__.reader, FrozenMountedDocumentReader)


def test_binding_assertions_are_fixed_secret_free_identity_records() -> None:
    registrations, facts = _registrations_and_facts()
    bindings = build_mounted_document_tool_bindings(
        registrations,
        facts=facts,
    )

    expected_keys = {
        "schema_version",
        "binding_kind",
        "session_scope_sha256",
        "scope_snapshot_sha256",
        "document_alias_projection_sha256",
        "document_generation_snapshot_sha256",
        "document_freshness_snapshot_sha256",
        "source_fingerprint",
    }
    for binding in bindings:
        assertion = dict(binding.binding_assertion)
        assert set(assertion) == expected_keys
        assert assertion["schema_version"] == MOUNTED_DOCUMENT_BINDING_ASSERTION_SCHEMA
        assert assertion["binding_kind"] == "frozen_mounted_document_cognition"
        encoded = json.dumps(assertion, sort_keys=True)
        for private_value in (
            SESSION_ID,
            DOCUMENT_ID,
            VERSION_ID,
            SOURCE_SHA256,
            "mounted_document_01",
        ):
            assert private_value not in encoded
        for key in expected_keys - {"schema_version", "binding_kind"}:
            assert isinstance(assertion[key], str)
            assert len(assertion[key]) == 64


def test_binding_rejects_order_identity_spec_execution_and_effect_drift() -> None:
    registrations, facts = _registrations_and_facts()

    with pytest.raises(MountedDocumentCatalogError, match="canonical order"):
        build_mounted_document_tool_bindings(
            tuple(reversed(registrations)),
            facts=facts,
        )

    mutations = (
        (
            replace(registrations[0], implementation_version="different"),
            "identity drifted",
        ),
        (
            replace(
                registrations[0],
                spec=replace(registrations[0].spec, name="Changed contract"),
            ),
            "model contract drifted",
        ),
        (
            replace(
                registrations[0],
                execution_profile=replace(
                    registrations[0].execution_profile,
                    hard_timeout_s=61.0,
                ),
            ),
            "execution contract drifted",
        ),
        (
            replace(
                registrations[0],
                effect_profile=replace(
                    registrations[0].effect_profile,
                    effects=(
                        replace(
                            registrations[0].effect_profile.effects[0],
                            data_egress=DataEgress.METADATA,
                        ),
                    ),
                ),
            ),
            "effects drifted",
        ),
    )
    for changed, message in mutations:
        with pytest.raises(MountedDocumentCatalogError, match=message):
            build_mounted_document_tool_bindings(
                (changed, *registrations[1:]),
                facts=facts,
            )


def test_binding_rejects_source_or_session_fact_drift() -> None:
    registrations, facts = _registrations_and_facts()

    source_mutations = (
        replace(
            registrations[0],
            source=replace(registrations[0].source, kind=ToolSourceKind.PROVIDER),
        ),
        replace(
            registrations[0],
            source=replace(registrations[0].source, fingerprint="f" * 64),
        ),
    )
    for changed in source_mutations:
        with pytest.raises(MountedDocumentCatalogError, match="source"):
            build_mounted_document_tool_bindings(
                (changed, *registrations[1:]),
                facts=facts,
            )

    mismatched_session = replace(facts, session_scope_sha256="e" * 64)
    with pytest.raises(MountedDocumentCatalogError, match="Session scope"):
        build_mounted_document_tool_bindings(
            registrations,
            facts=mismatched_session,
        )


def test_binding_facts_require_canonical_hashes() -> None:
    _registrations, facts = _registrations_and_facts()

    with pytest.raises(MountedDocumentCatalogError, match="scope_snapshot_sha256"):
        replace(facts, scope_snapshot_sha256="not-a-digest")


def test_catalog_owner_has_no_runtime_or_l2_imports() -> None:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "src/personagraph/tools/documents/mounted_document_catalog.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert not any(
        module == "personagraph.runtime"
        or module.startswith("personagraph.runtime.")
        or module == "personagraph.l2"
        or module.startswith("personagraph.l2.")
        for module in imported_modules
    )
