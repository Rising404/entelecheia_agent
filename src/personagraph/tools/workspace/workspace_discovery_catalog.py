"""Stable definitions and contextual bindings for workspace discovery tools.

The model contracts and handlers remain owned by :mod:`workspace_tools`.  This
module separates their process-stable definition manifest from the Session-bound
filesystem authority that must be frozen later.  It neither reads a Session nor
publishes definitions into the global default profile.
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
from ..effects import (
    DataEgress,
    EffectAction,
    EffectDescriptor,
    EffectResource,
    EffectScopeKind,
    Idempotency,
    Reversibility,
    ToolEffectProfile,
)
from ..registration import ToolExecutionProfile, ToolRegistration
from .workspace_tools import (
    WORKSPACE_DISCOVERY_TOOL_IDS,
    WORKSPACE_DISCOVERY_TOOL_IMPLEMENTATION_VERSION,
    build_workspace_discovery_execution_profile,
    build_workspace_discovery_tool_specs,
)


WORKSPACE_DISCOVERY_DEFINITION_MANIFEST_SCHEMA = (
    "workspace-discovery-definition-manifest-v1"
)
WORKSPACE_DISCOVERY_BINDING_ASSERTION_SCHEMA = (
    "workspace-discovery-binding-assertion-v1"
)

_WORKSPACE_DISCOVERY_SOURCE_ID = "personagraph.workspace.discovery"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class WorkspaceDiscoveryCatalogError(ValueError):
    """A definition or live binding drifted from the reviewed manifest."""


@dataclass(frozen=True, slots=True)
class WorkspaceDiscoveryToolDefinitionManifestItem:
    """One stable Definition plus its reviewed implementation declaration."""

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
class WorkspaceDiscoveryBindingFacts:
    """Secret-free fingerprints required to bind one frozen workspace scope.

    The caller hashes the read authorization rather than passing its receipt.
    The fixed assertion cannot serialize a root path or credential.
    """

    boundary_sha256: str
    read_authority_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.boundary_sha256, "boundary_sha256")
        _require_sha256(self.read_authority_sha256, "read_authority_sha256")


@dataclass(frozen=True, slots=True)
class _WorkspaceDiscoveryDeclaration:
    tool_id: str
    implementation_ref: str
    declared_behavior_revision: str
    action: EffectAction
    data_egress: DataEgress


_DECLARATIONS = (
    _WorkspaceDiscoveryDeclaration(
        "workspace_overview",
        "builtin/workspace_overview",
        "workspace-overview-handler-2",
        EffectAction.READ,
        DataEgress.METADATA,
    ),
    _WorkspaceDiscoveryDeclaration(
        "list_workspace_directory",
        "builtin/list_workspace_directory",
        "list-workspace-directory-handler-2",
        EffectAction.READ,
        DataEgress.METADATA,
    ),
    _WorkspaceDiscoveryDeclaration(
        "find_files",
        "builtin/find_files",
        "find-files-handler-2",
        EffectAction.SEARCH,
        DataEgress.METADATA,
    ),
    _WorkspaceDiscoveryDeclaration(
        "search_text_files",
        "builtin/search_text_files",
        "search-text-files-handler-2",
        EffectAction.SEARCH,
        DataEgress.CONTENT,
    ),
    _WorkspaceDiscoveryDeclaration(
        "inspect_file",
        "builtin/inspect_file",
        "inspect-file-handler-2",
        EffectAction.READ,
        DataEgress.CONTENT,
    ),
)


def build_workspace_discovery_tool_definition_manifest() -> tuple[
    WorkspaceDiscoveryToolDefinitionManifestItem,
    WorkspaceDiscoveryToolDefinitionManifestItem,
    WorkspaceDiscoveryToolDefinitionManifestItem,
    WorkspaceDiscoveryToolDefinitionManifestItem,
    WorkspaceDiscoveryToolDefinitionManifestItem,
]:
    """Build the context-free five-tool definition manifest in exposure order."""

    specs = build_workspace_discovery_tool_specs()
    actual_ids = tuple(spec.tool_id for spec in specs)
    declared_ids = tuple(item.tool_id for item in _DECLARATIONS)
    if (
        actual_ids != WORKSPACE_DISCOVERY_TOOL_IDS
        or declared_ids != WORKSPACE_DISCOVERY_TOOL_IDS
        or len(set(actual_ids)) != len(actual_ids)
    ):
        raise WorkspaceDiscoveryCatalogError(
            "workspace discovery definition order or identity drifted"
        )
    execution = build_workspace_discovery_execution_profile()
    manifest = tuple(
        _manifest_item(spec, declaration, execution)
        for spec, declaration in zip(specs, _DECLARATIONS, strict=True)
    )
    return (manifest[0], manifest[1], manifest[2], manifest[3], manifest[4])


def build_workspace_discovery_tool_bindings(
    registrations: Sequence[ToolRegistration],
    *,
    facts: WorkspaceDiscoveryBindingFacts,
) -> tuple[ToolBinding, ToolBinding, ToolBinding, ToolBinding, ToolBinding]:
    """Convert five already-frozen registrations into bare live bindings.

    Only registrations whose model and execution descriptors still match the
    stable manifest are accepted.  Constructing a temporary BoundToolRegistration
    verifies the effect-template narrowing and exact legacy descriptor projection;
    the public result remains the requested bare ``ToolBinding`` tuple.
    """

    if isinstance(registrations, (str, bytes)) or not isinstance(
        registrations,
        Sequence,
    ):
        raise TypeError("registrations must be a sequence of ToolRegistration values")
    if not isinstance(facts, WorkspaceDiscoveryBindingFacts):
        raise TypeError("facts must be WorkspaceDiscoveryBindingFacts")
    frozen = tuple(registrations)
    if len(frozen) != len(WORKSPACE_DISCOVERY_TOOL_IDS) or any(
        not isinstance(item, ToolRegistration) for item in frozen
    ):
        raise WorkspaceDiscoveryCatalogError(
            "workspace discovery requires exactly five ToolRegistration values"
        )
    manifest = build_workspace_discovery_tool_definition_manifest()
    actual_ids = tuple(item.tool_id for item in frozen)
    if actual_ids != WORKSPACE_DISCOVERY_TOOL_IDS:
        raise WorkspaceDiscoveryCatalogError(
            "workspace discovery registrations are not in canonical order"
        )

    bindings: list[ToolBinding] = []
    for registration, item in zip(frozen, manifest, strict=True):
        definition = item.definition
        actual_identity = ToolIdentity(
            registration.tool_id,
            registration.contract_version,
            registration.implementation_version,
        )
        if actual_identity != definition.identity:
            raise WorkspaceDiscoveryCatalogError(
                f"workspace discovery identity drifted: {registration.tool_id!r}"
            )
        if registration.spec != definition.spec:
            raise WorkspaceDiscoveryCatalogError(
                f"workspace discovery model contract drifted: {registration.tool_id!r}"
            )
        if registration.execution_profile != definition.execution_profile:
            raise WorkspaceDiscoveryCatalogError(
                f"workspace discovery execution contract drifted: {registration.tool_id!r}"
            )
        source = registration.source
        if (
            source.kind is not ToolSourceKind.LOCAL
            or source.source_id != _WORKSPACE_DISCOVERY_SOURCE_ID
            or source.display_name is not None
        ):
            raise WorkspaceDiscoveryCatalogError(
                f"workspace discovery source drifted: {registration.tool_id!r}"
            )
        _require_sha256(
            source.fingerprint,
            f"{registration.tool_id} source fingerprint",
        )
        binding = ToolBinding(
            identity=definition.identity,
            definition_digest=definition.digest,
            source=source,
            handler=registration.handler,
            effect_profile=registration.effect_profile,
            binding_assertion={
                "schema_version": WORKSPACE_DISCOVERY_BINDING_ASSERTION_SCHEMA,
                "binding_kind": "frozen_session_workspace_discovery",
                "boundary_sha256": facts.boundary_sha256,
                "read_authority_sha256": facts.read_authority_sha256,
                "source_fingerprint": source.fingerprint,
            },
        )
        try:
            bound = BoundToolRegistration(definition, binding)
        except ValueError as exc:
            raise WorkspaceDiscoveryCatalogError(
                f"workspace discovery effects drifted: {registration.tool_id!r}"
            ) from exc
        if bound.descriptor() != registration.descriptor():
            raise WorkspaceDiscoveryCatalogError(
                f"workspace discovery descriptor drifted: {registration.tool_id!r}"
            )
        bindings.append(binding)
    return (bindings[0], bindings[1], bindings[2], bindings[3], bindings[4])


def _manifest_item(
    spec: ToolSpec,
    declaration: _WorkspaceDiscoveryDeclaration,
    execution: ToolExecutionProfile,
) -> WorkspaceDiscoveryToolDefinitionManifestItem:
    effect_template = ToolEffectProfile(
        (
            EffectDescriptor(
                resource=EffectResource.FILESYSTEM,
                action=declaration.action,
                scope_kind=EffectScopeKind.WORKSPACE,
                default_scope="*",
                data_egress=declaration.data_egress,
                idempotency=Idempotency.IDEMPOTENT,
                reversibility=Reversibility.REVERSIBLE,
            ),
        )
    )
    implementation_digest = _declared_implementation_digest(
        spec=spec,
        implementation_ref=declaration.implementation_ref,
        declared_behavior_revision=declaration.declared_behavior_revision,
        effect_template=effect_template,
        execution=execution,
    )
    definition = ToolDefinition(
        spec=spec,
        implementation_version=WORKSPACE_DISCOVERY_TOOL_IMPLEMENTATION_VERSION,
        implementation_ref=declaration.implementation_ref,
        implementation_digest=implementation_digest,
        effect_template=effect_template,
        execution_profile=execution,
    )
    return WorkspaceDiscoveryToolDefinitionManifestItem(
        implementation_ref=declaration.implementation_ref,
        declared_behavior_revision=declaration.declared_behavior_revision,
        definition=definition,
    )


def _declared_implementation_digest(
    *,
    spec: ToolSpec,
    implementation_ref: str,
    declared_behavior_revision: str,
    effect_template: ToolEffectProfile,
    execution: ToolExecutionProfile,
) -> str:
    artifact = {
        "schema_version": WORKSPACE_DISCOVERY_DEFINITION_MANIFEST_SCHEMA,
        "implementation_ref": implementation_ref,
        "declared_behavior_revision": declared_behavior_revision,
        "identity": {
            "tool_id": spec.tool_id,
            "contract_version": spec.contract_version,
            "implementation_version": (
                WORKSPACE_DISCOVERY_TOOL_IMPLEMENTATION_VERSION
            ),
        },
        "spec": spec.to_dict(),
        "effect_template": [
            effect.to_dict() for effect in effect_template.effects
        ],
        "execution": _execution_descriptor(execution),
    }
    encoded = json.dumps(
        artifact,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        raise WorkspaceDiscoveryCatalogError(
            f"{field_name} must be a canonical SHA-256 digest"
        )


__all__ = [
    "WORKSPACE_DISCOVERY_BINDING_ASSERTION_SCHEMA",
    "WORKSPACE_DISCOVERY_DEFINITION_MANIFEST_SCHEMA",
    "WorkspaceDiscoveryBindingFacts",
    "WorkspaceDiscoveryCatalogError",
    "WorkspaceDiscoveryToolDefinitionManifestItem",
    "build_workspace_discovery_tool_bindings",
    "build_workspace_discovery_tool_definition_manifest",
]
