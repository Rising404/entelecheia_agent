"""Stable Definition and contextual Binding for workspace text writes.

The Definition contains only the process-stable model, effect, and execution
contract.  A Host later supplies secret-free hashes for the exact workspace,
UPDATE authority, protected dispatch route, and operation-ledger authority.
This module does not read Session or Runtime state and never owns approval or
dispatch decisions.
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
from .workspace_write_tools import (
    WORKSPACE_WRITE_SOURCE_DISPLAY_NAME,
    WORKSPACE_WRITE_SOURCE_ID,
    WORKSPACE_WRITE_TOOL_IMPLEMENTATION_VERSION,
    WORKSPACE_WRITE_TOOL_IDS,
    build_workspace_write_effect_profile,
    build_workspace_write_execution_profile,
    build_workspace_write_tool_spec,
)


WORKSPACE_WRITE_DEFINITION_MANIFEST_SCHEMA = (
    "workspace-write-definition-manifest-v1"
)
WORKSPACE_WRITE_BINDING_ASSERTION_SCHEMA = (
    "workspace-write-binding-assertion-v1"
)
WORKSPACE_WRITE_SOURCE_FINGERPRINT_SCHEMA = (
    "workspace-write-source-fingerprint-v1"
)

_IMPLEMENTATION_REF = "builtin/write_workspace_file"
_DECLARED_BEHAVIOR_REVISION = "write-workspace-file-handler-1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class WorkspaceWriteCatalogError(ValueError):
    """The workspace writer drifted from its reviewed Definition or Binding."""


@dataclass(frozen=True, slots=True)
class WorkspaceWriteToolDefinitionManifestItem:
    """The stable writer Definition plus its reviewed behavior declaration."""

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
class WorkspaceWriteBindingFacts:
    """Secret-free identities required to execute one protected workspace writer.

    ``protected_dispatch_sha256`` and ``operation_ledger_sha256`` identify the
    trusted route and ledger authority, not a particular ToolCall.  Exact call
    approval remains a concern of the invocation and dispatch layer.
    """

    boundary_sha256: str
    update_authority_sha256: str
    protected_dispatch_sha256: str
    operation_ledger_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.boundary_sha256, "boundary_sha256")
        _require_sha256(self.update_authority_sha256, "update_authority_sha256")
        _require_sha256(
            self.protected_dispatch_sha256,
            "protected_dispatch_sha256",
        )
        _require_sha256(
            self.operation_ledger_sha256,
            "operation_ledger_sha256",
        )


def build_workspace_write_tool_definition_manifest() -> tuple[
    WorkspaceWriteToolDefinitionManifestItem,
]:
    """Build the single context-free workspace writer Definition."""

    spec = build_workspace_write_tool_spec()
    if (spec.tool_id,) != WORKSPACE_WRITE_TOOL_IDS:
        raise WorkspaceWriteCatalogError(
            "workspace write definition identity drifted"
        )
    execution = build_workspace_write_execution_profile()
    effect_template = build_workspace_write_effect_profile(default_scope="*")
    implementation_digest = _declared_implementation_digest(
        spec=spec,
        effect_template=effect_template,
        execution=execution,
    )
    definition = ToolDefinition(
        spec=spec,
        implementation_version=WORKSPACE_WRITE_TOOL_IMPLEMENTATION_VERSION,
        implementation_ref=_IMPLEMENTATION_REF,
        implementation_digest=implementation_digest,
        effect_template=effect_template,
        execution_profile=execution,
    )
    return (
        WorkspaceWriteToolDefinitionManifestItem(
            implementation_ref=_IMPLEMENTATION_REF,
            declared_behavior_revision=_DECLARED_BEHAVIOR_REVISION,
            definition=definition,
        ),
    )


def derive_workspace_write_source_fingerprint(
    facts: WorkspaceWriteBindingFacts,
) -> str:
    """Derive the exact source identity a live registration must advertise."""

    if not isinstance(facts, WorkspaceWriteBindingFacts):
        raise TypeError("facts must be WorkspaceWriteBindingFacts")
    return _canonical_sha256(
        {
            "schema_version": WORKSPACE_WRITE_SOURCE_FINGERPRINT_SCHEMA,
            "boundary_sha256": facts.boundary_sha256,
            "update_authority_sha256": facts.update_authority_sha256,
            "protected_dispatch_sha256": facts.protected_dispatch_sha256,
            "operation_ledger_sha256": facts.operation_ledger_sha256,
        }
    )


def build_workspace_write_tool_bindings(
    registrations: Sequence[ToolRegistration],
    *,
    facts: WorkspaceWriteBindingFacts,
) -> tuple[ToolBinding]:
    """Bind one already-frozen registration to exact protected-write facts."""

    if isinstance(registrations, (str, bytes)) or not isinstance(
        registrations,
        Sequence,
    ):
        raise TypeError("registrations must be a sequence of ToolRegistration values")
    if not isinstance(facts, WorkspaceWriteBindingFacts):
        raise TypeError("facts must be WorkspaceWriteBindingFacts")
    frozen = tuple(registrations)
    if len(frozen) != 1 or not isinstance(frozen[0], ToolRegistration):
        raise WorkspaceWriteCatalogError(
            "workspace write requires exactly one ToolRegistration value"
        )
    (registration,) = frozen
    if (registration.tool_id,) != WORKSPACE_WRITE_TOOL_IDS:
        raise WorkspaceWriteCatalogError(
            "workspace write registration is not the canonical tool"
        )

    (manifest_item,) = build_workspace_write_tool_definition_manifest()
    definition = manifest_item.definition
    actual_identity = ToolIdentity(
        registration.tool_id,
        registration.contract_version,
        registration.implementation_version,
    )
    if actual_identity != definition.identity:
        raise WorkspaceWriteCatalogError("workspace write identity drifted")
    if registration.spec != definition.spec:
        raise WorkspaceWriteCatalogError("workspace write model contract drifted")
    if registration.execution_profile != definition.execution_profile:
        raise WorkspaceWriteCatalogError(
            "workspace write execution contract drifted"
        )
    effects = registration.effect_profile.effects
    if len(effects) != 2:
        raise WorkspaceWriteCatalogError("workspace write effects drifted")
    effective_scope = effects[0].default_scope
    if (
        effective_scope == "*"
        or registration.effect_profile
        != build_workspace_write_effect_profile(default_scope=effective_scope)
    ):
        raise WorkspaceWriteCatalogError("workspace write effects drifted")

    source = registration.source
    if (
        source.kind is not ToolSourceKind.LOCAL
        or source.source_id != WORKSPACE_WRITE_SOURCE_ID
        or source.display_name != WORKSPACE_WRITE_SOURCE_DISPLAY_NAME
    ):
        raise WorkspaceWriteCatalogError("workspace write source drifted")
    expected_source_fingerprint = derive_workspace_write_source_fingerprint(facts)
    if source.fingerprint != expected_source_fingerprint:
        raise WorkspaceWriteCatalogError(
            "workspace write source fingerprint drifted"
        )

    binding = ToolBinding(
        identity=definition.identity,
        definition_digest=definition.digest,
        source=source,
        handler=registration.handler,
        effect_profile=registration.effect_profile,
        binding_assertion={
            "schema_version": WORKSPACE_WRITE_BINDING_ASSERTION_SCHEMA,
            "binding_kind": "protected_workspace_write",
            "boundary_sha256": facts.boundary_sha256,
            "update_authority_sha256": facts.update_authority_sha256,
            "protected_dispatch_sha256": facts.protected_dispatch_sha256,
            "operation_ledger_sha256": facts.operation_ledger_sha256,
            "source_fingerprint": expected_source_fingerprint,
        },
    )
    try:
        bound = BoundToolRegistration(definition, binding)
    except ValueError as exc:
        raise WorkspaceWriteCatalogError("workspace write effects drifted") from exc
    if bound.descriptor() != registration.descriptor():
        raise WorkspaceWriteCatalogError("workspace write descriptor drifted")
    return (binding,)


def _declared_implementation_digest(
    *,
    spec: ToolSpec,
    effect_template: ToolEffectProfile,
    execution: ToolExecutionProfile,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": WORKSPACE_WRITE_DEFINITION_MANIFEST_SCHEMA,
            "implementation_ref": _IMPLEMENTATION_REF,
            "declared_behavior_revision": _DECLARED_BEHAVIOR_REVISION,
            "identity": {
                "tool_id": spec.tool_id,
                "contract_version": spec.contract_version,
                "implementation_version": (
                    WORKSPACE_WRITE_TOOL_IMPLEMENTATION_VERSION
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
        raise WorkspaceWriteCatalogError(
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
    "WORKSPACE_WRITE_BINDING_ASSERTION_SCHEMA",
    "WORKSPACE_WRITE_DEFINITION_MANIFEST_SCHEMA",
    "WORKSPACE_WRITE_SOURCE_FINGERPRINT_SCHEMA",
    "WorkspaceWriteBindingFacts",
    "WorkspaceWriteCatalogError",
    "WorkspaceWriteToolDefinitionManifestItem",
    "build_workspace_write_tool_bindings",
    "build_workspace_write_tool_definition_manifest",
    "derive_workspace_write_source_fingerprint",
]
