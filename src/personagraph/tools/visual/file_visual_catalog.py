"""Stable Definitions and contextual Bindings for shared file-version visuals.

``list_file_visuals`` has one process-stable metadata-read shape.  The current
production-capable ``read_file_visuals`` path uses the configured HTTP vision
adapter and therefore has a filesystem-read plus irreversible network-transmit
shape.  An unavailable adapter and test-only local adapters are deliberately not
made to look external: during the staged hard cut they keep their truthful direct
runtime registrations, but do not produce a read Binding for the persistent
catalog.

Supporting a future production local vision provider requires an explicitly
reviewed effect/execution-variant contract.  It must not be smuggled into the
external Definition by widening or weakening these checks.
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
from ..effects import ToolEffectProfile
from ..registration import ToolExecutionProfile, ToolRegistration
from .file_visual_tools import (
    FILE_VISUAL_IMPLEMENTATION_VERSION,
    FILE_VISUAL_SOURCE_DISPLAY_NAME,
    FILE_VISUAL_SOURCE_ID,
    FILE_VISUAL_TOOL_IDS,
    LIST_FILE_VISUALS_TOOL_ID,
    READ_FILE_VISUALS_TOOL_ID,
    build_file_visual_tool_specs,
    build_list_file_visuals_effect_profile,
    build_list_file_visuals_execution_profile,
    build_read_file_visuals_effect_profile,
    build_read_file_visuals_execution_profile,
)


FILE_VISUAL_DEFINITION_MANIFEST_SCHEMA = "file-visual-definition-manifest-v1"
FILE_VISUAL_BINDING_ASSERTION_SCHEMA = "file-visual-binding-assertion-v1"
FILE_VISUAL_SOURCE_FINGERPRINT_SCHEMA = "file-visual-source-fingerprint-v1"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class FileVisualCatalogError(ValueError):
    """A file visual Definition or contextual Binding drifted."""


@dataclass(frozen=True, slots=True)
class FileVisualToolDefinitionManifestItem:
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
class ExternalFileVisualReadBindingFacts:
    """Secret-free identities for the current external observation stack."""

    provider_identity_sha256: str
    capability_snapshot_sha256: str
    egress_policy_sha256: str
    physical_call_ledger_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "provider_identity_sha256",
            "capability_snapshot_sha256",
            "egress_policy_sha256",
            "physical_call_ledger_sha256",
        ):
            _require_sha256(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class FileVisualBindingFacts:
    """Session and File access policy facts reduced to hashes before entering a Binding.

    ``external_read`` is present only when an external adapter is both selected
    and available.  Its absence means only the list Definition can be bound.
    """

    session_scope_sha256: str
    file_access_policy_sha256: str
    external_read: ExternalFileVisualReadBindingFacts | None = None

    def __post_init__(self) -> None:
        _require_sha256(self.session_scope_sha256, "session_scope_sha256")
        _require_sha256(
            self.file_access_policy_sha256,
            "file_access_policy_sha256",
        )
        if self.external_read is not None and not isinstance(
            self.external_read,
            ExternalFileVisualReadBindingFacts,
        ):
            raise TypeError(
                "external_read must be ExternalFileVisualReadBindingFacts or None"
            )


@dataclass(frozen=True, slots=True)
class _FileVisualDeclaration:
    tool_id: str
    implementation_ref: str
    declared_behavior_revision: str


_DECLARATIONS = (
    _FileVisualDeclaration(
        LIST_FILE_VISUALS_TOOL_ID,
        "builtin/list_file_visuals",
        "file-visual-list-handler-4",
    ),
    _FileVisualDeclaration(
        READ_FILE_VISUALS_TOOL_ID,
        "builtin/read_file_visuals",
        "file-visual-external-read-handler-4",
    ),
)


def build_file_visual_tool_definition_manifest() -> tuple[
    FileVisualToolDefinitionManifestItem,
    FileVisualToolDefinitionManifestItem,
]:
    """Build the list and production-external-read Definitions."""

    specs = build_file_visual_tool_specs()
    actual_ids = tuple(spec.tool_id for spec in specs)
    declared_ids = tuple(item.tool_id for item in _DECLARATIONS)
    if (
        actual_ids != FILE_VISUAL_TOOL_IDS
        or declared_ids != FILE_VISUAL_TOOL_IDS
        or len(set(actual_ids)) != len(actual_ids)
    ):
        raise FileVisualCatalogError(
            "file visual definition order or identity drifted"
        )
    surfaces = (
        (
            specs[0],
            build_list_file_visuals_effect_profile(default_scope="*"),
            build_list_file_visuals_execution_profile(),
        ),
        (
            specs[1],
            build_read_file_visuals_effect_profile(
                default_scope="*",
                sends_externally=True,
            ),
            build_read_file_visuals_execution_profile(sends_externally=True),
        ),
    )
    manifest = tuple(
        _manifest_item(
            spec=spec,
            effect_template=effect_template,
            execution=execution,
            declaration=declaration,
        )
        for (spec, effect_template, execution), declaration in zip(
            surfaces,
            _DECLARATIONS,
            strict=True,
        )
    )
    return (manifest[0], manifest[1])


def derive_file_visual_source_fingerprint(
    facts: FileVisualBindingFacts,
    *,
    tool_id: str,
) -> str:
    """Derive the exact source identity for one eligible live Binding."""

    if not isinstance(facts, FileVisualBindingFacts):
        raise TypeError("facts must be FileVisualBindingFacts")
    if tool_id not in FILE_VISUAL_TOOL_IDS:
        raise ValueError("tool_id is not a file visual tool")
    external: dict[str, str] | None = None
    if tool_id == READ_FILE_VISUALS_TOOL_ID:
        if facts.external_read is None:
            raise FileVisualCatalogError(
                "read_file_visuals has no eligible external read binding"
            )
        external = {
            "provider_identity_sha256": (
                facts.external_read.provider_identity_sha256
            ),
            "capability_snapshot_sha256": (
                facts.external_read.capability_snapshot_sha256
            ),
            "egress_policy_sha256": (
                facts.external_read.egress_policy_sha256
            ),
            "physical_call_ledger_sha256": (
                facts.external_read.physical_call_ledger_sha256
            ),
        }
    return _canonical_sha256(
        {
            "schema_version": FILE_VISUAL_SOURCE_FINGERPRINT_SCHEMA,
            "tool_id": tool_id,
            "session_scope_sha256": facts.session_scope_sha256,
            "file_access_policy_sha256": facts.file_access_policy_sha256,
            "external_read": external,
        }
    )


def build_file_visual_tool_bindings(
    registrations: Sequence[ToolRegistration],
    *,
    facts: FileVisualBindingFacts,
) -> tuple[ToolBinding, ...]:
    """Bind list and, when eligible, the exact external read registration.

    The sequence must be the canonical one-item prefix (list only) or the full
    two-item family.  A local/unavailable read registration must not be passed as
    the second item because it does not implement the persisted external effect.
    """

    if isinstance(registrations, (str, bytes)) or not isinstance(
        registrations,
        Sequence,
    ):
        raise TypeError("registrations must be a sequence of ToolRegistration values")
    if not isinstance(facts, FileVisualBindingFacts):
        raise TypeError("facts must be FileVisualBindingFacts")
    frozen = tuple(registrations)
    if len(frozen) not in {1, 2} or any(
        not isinstance(item, ToolRegistration) for item in frozen
    ):
        raise FileVisualCatalogError(
            "file visuals require the list registration and optional external read"
        )
    actual_ids = tuple(item.tool_id for item in frozen)
    if actual_ids != FILE_VISUAL_TOOL_IDS[: len(frozen)]:
        raise FileVisualCatalogError(
            "file visual registrations are not a canonical prefix"
        )
    if len(frozen) == 2 and facts.external_read is None:
        raise FileVisualCatalogError(
            "external read registration requires external binding facts"
        )

    manifest = build_file_visual_tool_definition_manifest()
    bindings: list[ToolBinding] = []
    for registration, item in zip(
        frozen,
        manifest[: len(frozen)],
        strict=True,
    ):
        definition = item.definition
        actual_identity = ToolIdentity(
            registration.tool_id,
            registration.contract_version,
            registration.implementation_version,
        )
        if actual_identity != definition.identity:
            raise FileVisualCatalogError(
                f"file visual identity drifted: {registration.tool_id!r}"
            )
        if registration.spec != definition.spec:
            raise FileVisualCatalogError(
                f"file visual model contract drifted: {registration.tool_id!r}"
            )
        if registration.execution_profile != definition.execution_profile:
            raise FileVisualCatalogError(
                f"file visual execution contract drifted: {registration.tool_id!r}"
            )
        source = registration.source
        if (
            source.kind is not ToolSourceKind.LOCAL
            or source.source_id != FILE_VISUAL_SOURCE_ID
            or source.display_name != FILE_VISUAL_SOURCE_DISPLAY_NAME
        ):
            raise FileVisualCatalogError(
                f"file visual source drifted: {registration.tool_id!r}"
            )
        expected_fingerprint = derive_file_visual_source_fingerprint(
            facts,
            tool_id=registration.tool_id,
        )
        if source.fingerprint != expected_fingerprint:
            raise FileVisualCatalogError(
                f"file visual source fingerprint drifted: {registration.tool_id!r}"
            )
        assertion: dict[str, str] = {
            "schema_version": FILE_VISUAL_BINDING_ASSERTION_SCHEMA,
            "binding_kind": "shared_file_version_visual",
            "session_scope_sha256": facts.session_scope_sha256,
            "file_access_policy_sha256": facts.file_access_policy_sha256,
            "source_fingerprint": expected_fingerprint,
        }
        if registration.tool_id == READ_FILE_VISUALS_TOOL_ID:
            assert facts.external_read is not None
            assertion.update(
                {
                    "provider_identity_sha256": (
                        facts.external_read.provider_identity_sha256
                    ),
                    "capability_snapshot_sha256": (
                        facts.external_read.capability_snapshot_sha256
                    ),
                    "egress_policy_sha256": (
                        facts.external_read.egress_policy_sha256
                    ),
                    "physical_call_ledger_sha256": (
                        facts.external_read.physical_call_ledger_sha256
                    ),
                }
            )
        binding = ToolBinding(
            identity=definition.identity,
            definition_digest=definition.digest,
            source=source,
            handler=registration.handler,
            effect_profile=registration.effect_profile,
            binding_assertion=assertion,
        )
        try:
            bound = BoundToolRegistration(definition, binding)
        except (TypeError, ValueError) as exc:
            raise FileVisualCatalogError(
                f"file visual effects drifted: {registration.tool_id!r}"
            ) from exc
        if bound.descriptor() != registration.descriptor():
            raise FileVisualCatalogError(
                f"file visual descriptor drifted: {registration.tool_id!r}"
            )
        bindings.append(binding)
    return tuple(bindings)


def _manifest_item(
    *,
    spec: ToolSpec,
    effect_template: ToolEffectProfile,
    execution: ToolExecutionProfile,
    declaration: _FileVisualDeclaration,
) -> FileVisualToolDefinitionManifestItem:
    implementation_digest = _canonical_sha256(
        {
            "schema_version": FILE_VISUAL_DEFINITION_MANIFEST_SCHEMA,
            "implementation_ref": declaration.implementation_ref,
            "declared_behavior_revision": declaration.declared_behavior_revision,
            "identity": {
                "tool_id": spec.tool_id,
                "contract_version": spec.contract_version,
                "implementation_version": FILE_VISUAL_IMPLEMENTATION_VERSION,
            },
            "spec": spec.to_dict(),
            "effect_template": [
                effect.to_dict() for effect in effect_template.effects
            ],
            "execution": _execution_descriptor(execution),
        }
    )
    definition = ToolDefinition(
        spec=spec,
        implementation_version=FILE_VISUAL_IMPLEMENTATION_VERSION,
        implementation_ref=declaration.implementation_ref,
        implementation_digest=implementation_digest,
        effect_template=effect_template,
        execution_profile=execution,
    )
    return FileVisualToolDefinitionManifestItem(
        implementation_ref=declaration.implementation_ref,
        declared_behavior_revision=declaration.declared_behavior_revision,
        definition=definition,
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
        raise FileVisualCatalogError(
            f"{field_name} must be a canonical SHA-256 digest"
        )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "FILE_VISUAL_BINDING_ASSERTION_SCHEMA",
    "FILE_VISUAL_DEFINITION_MANIFEST_SCHEMA",
    "ExternalFileVisualReadBindingFacts",
    "FileVisualBindingFacts",
    "FileVisualCatalogError",
    "FileVisualToolDefinitionManifestItem",
    "build_file_visual_tool_bindings",
    "build_file_visual_tool_definition_manifest",
    "derive_file_visual_source_fingerprint",
]
