"""Lane-neutral mounted-document cognition owner contracts."""

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from personagraph.tools.documents.frozen_mounted_document_reader import (
    FrozenMountedDocument,
)
from personagraph.tools.documents.mounted_document_cognition_tools import (
    MOUNTED_DOCUMENT_COGNITION_TOOL_IDS,
    FrozenMountedDocumentToolScope,
    build_mounted_document_cognition_tool_source,
)
from personagraph.tools.effects import (
    EffectAction,
    EffectResource,
    EffectScopeKind,
)


SESSION_ID = "session-mounted-document-tools"
SCOPE_SHA256 = hashlib.sha256(b"mounted document scope").hexdigest()


def _document(*, alias: str = "mounted_document_01") -> FrozenMountedDocument:
    return FrozenMountedDocument(
        session_id=SESSION_ID,
        resource_alias=alias,
        document_id="private-document-id",
        document_version_id="document-version-001",
        source_sha256=hashlib.sha256(b"document bytes").hexdigest(),
        processing_status="complete",
        resource_format="txt",
        media_type="text/plain",
        file_extension=".txt",
        total_chunk_count=2,
        processing_diagnostic_codes=(),
    )


def test_source_builds_stable_tools_without_exposing_private_identity() -> None:
    source = build_mounted_document_cognition_tool_source(
        FrozenMountedDocumentToolScope(
            session_id=SESSION_ID,
            scope_snapshot_sha256=SCOPE_SHA256,
            documents=(_document(),),
        )
    )

    assert source is not None
    assert source.tool_ids == MOUNTED_DOCUMENT_COGNITION_TOOL_IDS
    assert {
        entry.registration.tool_id
        for entry in source.catalog_snapshot.exposed()
    } == set(MOUNTED_DOCUMENT_COGNITION_TOOL_IDS)
    assert {
        entry.registration.source.source_id
        for entry in source.catalog_snapshot.exposed()
    } == {"personagraph.tools.documents.mounted-document-cognition"}
    for entry in source.catalog_snapshot.exposed():
        (effect,) = entry.registration.effect_profile.effects
        assert effect.resource is EffectResource.FILESYSTEM
        assert effect.action in {EffectAction.READ, EffectAction.SEARCH}
        assert effect.scope_kind is EffectScopeKind.SESSION
        assert effect.default_scope == SESSION_ID
    descriptor = repr(source.catalog_snapshot.to_descriptor())
    assert "private-document-id" not in descriptor
    assert "document-version-001" not in descriptor


def test_empty_frozen_scope_does_not_register_document_tools() -> None:
    source = build_mounted_document_cognition_tool_source(
        FrozenMountedDocumentToolScope(
            session_id=SESSION_ID,
            scope_snapshot_sha256=SCOPE_SHA256,
            documents=(),
        )
    )

    assert source is None


def test_tool_specs_are_identical_across_distinct_frozen_scopes() -> None:
    first = build_mounted_document_cognition_tool_source(
        FrozenMountedDocumentToolScope(
            session_id=SESSION_ID,
            scope_snapshot_sha256=SCOPE_SHA256,
            documents=(_document(),),
        )
    )
    other_session = "session-mounted-document-tools-other"
    second = build_mounted_document_cognition_tool_source(
        FrozenMountedDocumentToolScope(
            session_id=other_session,
            scope_snapshot_sha256=hashlib.sha256(b"other scope").hexdigest(),
            documents=(
                replace(
                    _document(alias="mounted_document_02"),
                    session_id=other_session,
                ),
            ),
        )
    )

    assert first is not None
    assert second is not None
    first_specs = {
        tool_id: first.catalog_snapshot.resolve(tool_id).spec.to_dict()
        for tool_id in MOUNTED_DOCUMENT_COGNITION_TOOL_IDS
    }
    second_specs = {
        tool_id: second.catalog_snapshot.resolve(tool_id).spec.to_dict()
        for tool_id in MOUNTED_DOCUMENT_COGNITION_TOOL_IDS
    }
    assert first_specs == second_specs
    assert _sha256_value(first_specs) == _sha256_value(second_specs)
    for spec in first_specs.values():
        alias_schema = spec["input_schema"]["properties"]["document_alias"]
        assert alias_schema == {
            "type": "string",
            "minLength": 1,
            "maxLength": 200,
        }


def test_frozen_scope_rejects_cross_session_duplicate_or_inexact_documents() -> None:
    first = _document()
    with pytest.raises(ValueError, match="unique"):
        FrozenMountedDocumentToolScope(
            session_id=SESSION_ID,
            scope_snapshot_sha256=SCOPE_SHA256,
            documents=(first, first),
        )
    with pytest.raises(ValueError, match="belong"):
        FrozenMountedDocumentToolScope(
            session_id=SESSION_ID,
            scope_snapshot_sha256=SCOPE_SHA256,
            documents=(replace(first, session_id="another-session"),),
        )
    with pytest.raises(ValueError, match="exact"):
        FrozenMountedDocumentToolScope(
            session_id=SESSION_ID,
            scope_snapshot_sha256=SCOPE_SHA256,
            documents=(
                replace(
                    first,
                    total_chunk_count=None,
                    processing_diagnostic_codes=None,
                ),
            ),
        )


def test_tool_owner_has_no_l2_or_runtime_imports() -> None:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "src/personagraph/tools/documents/mounted_document_cognition_tools.py"
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
        module == "personagraph.l2" or module.startswith("personagraph.l2.")
        for module in imported_modules
    )
    assert not any(
        module == "personagraph.runtime"
        or module.startswith("personagraph.runtime.")
        for module in imported_modules
    )


def _sha256_value(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
