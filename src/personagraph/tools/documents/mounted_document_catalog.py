"""Stable Definitions and contextual Bindings for mounted-document cognition.

The three model contracts are process-stable.  Session identity, the frozen
authority scope, opaque aliases, exact DocStore generations, freshness bindings,
live handlers, and effective policy scopes are contextual and therefore belong to
``ToolBinding``.  The data plane remains the single
``FrozenMountedDocumentReader`` used by ``mounted_document_cognition_tools``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import re

from ..catalog.binding import (
    BoundToolRegistration,
    ToolBinding,
    ToolDefinition,
    ToolIdentity,
)
from ..contracts import ToolSourceKind, ToolSpec
from ..effects import EffectAction, ToolEffectProfile
from ..registration import ToolExecutionProfile, ToolRegistration
from .mounted_document_cognition_tools import (
    MOUNTED_DOCUMENT_COGNITION_IMPLEMENTATION_VERSION,
    MOUNTED_DOCUMENT_COGNITION_SOURCE_DISPLAY_NAME,
    MOUNTED_DOCUMENT_COGNITION_SOURCE_ID,
    MOUNTED_DOCUMENT_COGNITION_TOOL_IDS,
    FrozenMountedDocumentToolScope,
    build_mounted_document_cognition_effect_profile,
    build_mounted_document_cognition_execution_profile,
    build_mounted_document_cognition_tool_specs,
    derive_mounted_document_cognition_source_fingerprint,
)


MOUNTED_DOCUMENT_DEFINITION_MANIFEST_SCHEMA = (
    "mounted-document-cognition-definition-manifest-v1"
)
MOUNTED_DOCUMENT_BINDING_ASSERTION_SCHEMA = (
    "mounted-document-cognition-binding-assertion-v1"
)

_SESSION_SCOPE_SCHEMA = "mounted-document-session-scope-v1"
_ALIAS_PROJECTION_SCHEMA = "mounted-document-alias-projection-v1"
_GENERATION_SNAPSHOT_SCHEMA = "mounted-document-generation-snapshot-v1"
_FRESHNESS_SNAPSHOT_SCHEMA = "mounted-document-freshness-snapshot-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class MountedDocumentCatalogError(ValueError):
    """A mounted-document Definition or contextual Binding drifted."""


@dataclass(frozen=True, slots=True)
class MountedDocumentToolDefinitionManifestItem:
    """One stable Definition plus its reviewed behavior declaration."""

    implementation_ref: str
    declared_behavior_revision: str
    definition: ToolDefinition

    def __post_init__(self) -> None:
        if not isinstance(self.implementation_ref, str) or not self.implementation_ref:
            raise ValueError("implementation_ref must not be empty")
        if (
            not isinstance(self.declared_behavior_revision, str)
            or not self.declared_behavior_revision
        ):
            raise ValueError("declared_behavior_revision must not be empty")
        if not isinstance(self.definition, ToolDefinition):
            raise TypeError("definition must be a ToolDefinition")
        if self.definition.implementation_ref != self.implementation_ref:
            raise ValueError("definition implementation_ref does not match manifest")


@dataclass(frozen=True, slots=True)
class MountedDocumentBindingFacts:
    """Secret-free identities for one exact frozen document reader scope.

    Every field is a digest.  Raw Session IDs, DocStore IDs, version IDs, source
    hashes, and alias members remain inside the existing source/handler closure.
    ``scope_snapshot_sha256`` is also the authority commitment used by the live
    source fingerprint.
    """

    session_scope_sha256: str
    scope_snapshot_sha256: str
    document_alias_projection_sha256: str
    document_generation_snapshot_sha256: str
    document_freshness_snapshot_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "session_scope_sha256",
            "scope_snapshot_sha256",
            "document_alias_projection_sha256",
            "document_generation_snapshot_sha256",
            "document_freshness_snapshot_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class _MountedDocumentDeclaration:
    tool_id: str
    implementation_ref: str
    declared_behavior_revision: str
    action: EffectAction


_DECLARATIONS = (
    _MountedDocumentDeclaration(
        "inspect_mounted_document",
        "builtin/inspect_mounted_document",
        "inspect-mounted-document-handler-1",
        EffectAction.READ,
    ),
    _MountedDocumentDeclaration(
        "search_mounted_document",
        "builtin/search_mounted_document",
        "search-mounted-document-handler-1",
        EffectAction.SEARCH,
    ),
    _MountedDocumentDeclaration(
        "read_mounted_document_chunks",
        "builtin/read_mounted_document_chunks",
        "read-mounted-document-chunks-handler-1",
        EffectAction.READ,
    ),
)


def build_mounted_document_tool_definition_manifest() -> tuple[
    MountedDocumentToolDefinitionManifestItem,
    MountedDocumentToolDefinitionManifestItem,
    MountedDocumentToolDefinitionManifestItem,
]:
    """Build three context-free Definitions in model exposure order."""

    specs = build_mounted_document_cognition_tool_specs()
    actual_ids = tuple(spec.tool_id for spec in specs)
    declared_ids = tuple(item.tool_id for item in _DECLARATIONS)
    if (
        actual_ids != MOUNTED_DOCUMENT_COGNITION_TOOL_IDS
        or declared_ids != MOUNTED_DOCUMENT_COGNITION_TOOL_IDS
        or len(set(actual_ids)) != len(actual_ids)
    ):
        raise MountedDocumentCatalogError(
            "mounted document definition order or identity drifted"
        )
    execution = build_mounted_document_cognition_execution_profile()
    manifest = tuple(
        _manifest_item(spec, declaration, execution)
        for spec, declaration in zip(specs, _DECLARATIONS, strict=True)
    )
    return (manifest[0], manifest[1], manifest[2])


def build_mounted_document_binding_facts(
    scope: FrozenMountedDocumentToolScope,
) -> MountedDocumentBindingFacts:
    """Derive only opaque contextual identities from a validated frozen scope."""

    if not isinstance(scope, FrozenMountedDocumentToolScope):
        raise TypeError("scope must be FrozenMountedDocumentToolScope")
    documents = scope.documents
    return MountedDocumentBindingFacts(
        session_scope_sha256=_canonical_sha256(
            {
                "schema_version": _SESSION_SCOPE_SCHEMA,
                "session_id": scope.session_id,
            }
        ),
        scope_snapshot_sha256=scope.scope_snapshot_sha256,
        document_alias_projection_sha256=_canonical_sha256(
            {
                "schema_version": _ALIAS_PROJECTION_SCHEMA,
                "aliases": [document.resource_alias for document in documents],
            }
        ),
        document_generation_snapshot_sha256=_canonical_sha256(
            {
                "schema_version": _GENERATION_SNAPSHOT_SCHEMA,
                "documents": [
                    {
                        "alias": document.resource_alias,
                        "document_id": document.document_id,
                        "document_version_id": document.document_version_id,
                        "source_sha256": document.source_sha256,
                        "processing_status": document.processing_status,
                        "resource_format": document.resource_format,
                        "media_type": document.media_type,
                        "file_extension": document.file_extension,
                        "total_chunk_count": document.total_chunk_count,
                        "processing_diagnostic_codes": list(
                            document.processing_diagnostic_codes or ()
                        ),
                    }
                    for document in documents
                ],
            }
        ),
        document_freshness_snapshot_sha256=_canonical_sha256(
            {
                "schema_version": _FRESHNESS_SNAPSHOT_SCHEMA,
                "documents": [
                    {
                        "alias": document.resource_alias,
                        "freshness_binding_sha256": (
                            document.freshness_binding_sha256
                        ),
                    }
                    for document in documents
                ],
            }
        ),
    )


def build_mounted_document_tool_bindings(
    registrations: Sequence[ToolRegistration],
    *,
    facts: MountedDocumentBindingFacts,
) -> tuple[ToolBinding, ToolBinding, ToolBinding]:
    """Validate and bind one exact three-tool mounted-document source."""

    if isinstance(registrations, (str, bytes)) or not isinstance(
        registrations,
        Sequence,
    ):
        raise TypeError("registrations must be a sequence of ToolRegistration values")
    if not isinstance(facts, MountedDocumentBindingFacts):
        raise TypeError("facts must be MountedDocumentBindingFacts")
    frozen = tuple(registrations)
    if len(frozen) != len(MOUNTED_DOCUMENT_COGNITION_TOOL_IDS) or any(
        not isinstance(item, ToolRegistration) for item in frozen
    ):
        raise MountedDocumentCatalogError(
            "mounted document cognition requires exactly three registrations"
        )
    if tuple(item.tool_id for item in frozen) != MOUNTED_DOCUMENT_COGNITION_TOOL_IDS:
        raise MountedDocumentCatalogError(
            "mounted document registrations are not in canonical order"
        )

    manifest = build_mounted_document_tool_definition_manifest()
    expected_source_fingerprint = (
        derive_mounted_document_cognition_source_fingerprint(
            facts.scope_snapshot_sha256
        )
    )
    bindings: list[ToolBinding] = []
    effective_scope: str | None = None
    for registration, item, declaration in zip(
        frozen,
        manifest,
        _DECLARATIONS,
        strict=True,
    ):
        definition = item.definition
        actual_identity = ToolIdentity(
            registration.tool_id,
            registration.contract_version,
            registration.implementation_version,
        )
        if actual_identity != definition.identity:
            raise MountedDocumentCatalogError(
                f"mounted document identity drifted: {registration.tool_id!r}"
            )
        if registration.spec != definition.spec:
            raise MountedDocumentCatalogError(
                f"mounted document model contract drifted: {registration.tool_id!r}"
            )
        if registration.execution_profile != definition.execution_profile:
            raise MountedDocumentCatalogError(
                f"mounted document execution contract drifted: {registration.tool_id!r}"
            )

        effects = registration.effect_profile.effects
        if len(effects) != 1 or effects[0].default_scope == "*":
            raise MountedDocumentCatalogError(
                f"mounted document effects drifted: {registration.tool_id!r}"
            )
        registration_scope = effects[0].default_scope
        expected_effects = build_mounted_document_cognition_effect_profile(
            action=declaration.action,
            default_scope=registration_scope,
        )
        if registration.effect_profile != expected_effects:
            raise MountedDocumentCatalogError(
                f"mounted document effects drifted: {registration.tool_id!r}"
            )
        if effective_scope is None:
            effective_scope = registration_scope
        elif effective_scope != registration_scope:
            raise MountedDocumentCatalogError(
                "mounted document registrations have different Session scopes"
            )

        source = registration.source
        if (
            source.kind is not ToolSourceKind.LOCAL
            or source.source_id != MOUNTED_DOCUMENT_COGNITION_SOURCE_ID
            or source.display_name != MOUNTED_DOCUMENT_COGNITION_SOURCE_DISPLAY_NAME
        ):
            raise MountedDocumentCatalogError(
                f"mounted document source drifted: {registration.tool_id!r}"
            )
        if source.fingerprint != expected_source_fingerprint:
            raise MountedDocumentCatalogError(
                f"mounted document source fingerprint drifted: {registration.tool_id!r}"
            )

        binding = ToolBinding(
            identity=definition.identity,
            definition_digest=definition.digest,
            source=source,
            handler=registration.handler,
            effect_profile=registration.effect_profile,
            binding_assertion={
                "schema_version": MOUNTED_DOCUMENT_BINDING_ASSERTION_SCHEMA,
                "binding_kind": "frozen_mounted_document_cognition",
                "session_scope_sha256": facts.session_scope_sha256,
                "scope_snapshot_sha256": facts.scope_snapshot_sha256,
                "document_alias_projection_sha256": (
                    facts.document_alias_projection_sha256
                ),
                "document_generation_snapshot_sha256": (
                    facts.document_generation_snapshot_sha256
                ),
                "document_freshness_snapshot_sha256": (
                    facts.document_freshness_snapshot_sha256
                ),
                "source_fingerprint": expected_source_fingerprint,
            },
        )
        try:
            bound = BoundToolRegistration(definition, binding)
        except ValueError as exc:
            raise MountedDocumentCatalogError(
                f"mounted document effects drifted: {registration.tool_id!r}"
            ) from exc
        if bound.descriptor() != registration.descriptor():
            raise MountedDocumentCatalogError(
                f"mounted document descriptor drifted: {registration.tool_id!r}"
            )
        bindings.append(binding)

    if effective_scope is None or _canonical_sha256(
        {
            "schema_version": _SESSION_SCOPE_SCHEMA,
            "session_id": effective_scope,
        }
    ) != facts.session_scope_sha256:
        raise MountedDocumentCatalogError(
            "mounted document Session scope does not match its binding facts"
        )
    return (bindings[0], bindings[1], bindings[2])


def _manifest_item(
    spec: ToolSpec,
    declaration: _MountedDocumentDeclaration,
    execution: ToolExecutionProfile,
) -> MountedDocumentToolDefinitionManifestItem:
    effect_template = build_mounted_document_cognition_effect_profile(
        action=declaration.action,
        default_scope="*",
    )
    definition = ToolDefinition(
        spec=spec,
        implementation_version=MOUNTED_DOCUMENT_COGNITION_IMPLEMENTATION_VERSION,
        implementation_ref=declaration.implementation_ref,
        implementation_digest=_declared_implementation_digest(
            spec=spec,
            declaration=declaration,
            effect_template=effect_template,
            execution=execution,
        ),
        effect_template=effect_template,
        execution_profile=execution,
    )
    return MountedDocumentToolDefinitionManifestItem(
        implementation_ref=declaration.implementation_ref,
        declared_behavior_revision=declaration.declared_behavior_revision,
        definition=definition,
    )


def _declared_implementation_digest(
    *,
    spec: ToolSpec,
    declaration: _MountedDocumentDeclaration,
    effect_template: ToolEffectProfile,
    execution: ToolExecutionProfile,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": MOUNTED_DOCUMENT_DEFINITION_MANIFEST_SCHEMA,
            "implementation_ref": declaration.implementation_ref,
            "declared_behavior_revision": declaration.declared_behavior_revision,
            "identity": {
                "tool_id": spec.tool_id,
                "contract_version": spec.contract_version,
                "implementation_version": (
                    MOUNTED_DOCUMENT_COGNITION_IMPLEMENTATION_VERSION
                ),
            },
            "spec": spec.to_dict(),
            "effect_template": [
                effect.to_dict() for effect in effect_template.effects
            ],
            "execution": _execution_descriptor(execution),
        }
    )


def _execution_descriptor(profile: ToolExecutionProfile) -> dict[str, object]:
    return {
        "default_timeout_s": profile.default_timeout_s,
        "hard_timeout_s": profile.hard_timeout_s,
        "max_output_bytes": profile.max_output_bytes,
        "max_transparent_retries": profile.max_transparent_retries,
        "execution_mode": profile.execution_mode.value,
        "cancellation_mode": profile.cancellation_mode.value,
        "isolation_requirement": profile.isolation_requirement.value,
        "concurrency_class": profile.concurrency_class,
    }


def _require_sha256(value: object, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MountedDocumentCatalogError(
            f"{field_name} must be a canonical SHA-256 digest"
        )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "MOUNTED_DOCUMENT_BINDING_ASSERTION_SCHEMA",
    "MOUNTED_DOCUMENT_DEFINITION_MANIFEST_SCHEMA",
    "MountedDocumentBindingFacts",
    "MountedDocumentCatalogError",
    "MountedDocumentToolDefinitionManifestItem",
    "build_mounted_document_binding_facts",
    "build_mounted_document_tool_bindings",
    "build_mounted_document_tool_definition_manifest",
]
