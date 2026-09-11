"""Stable Definition and contextual Binding for History retrieval.

The model contract is process-stable and lists every supported narrowing token.
Which tokens are authorized, the exact Session and committed-turn cutoff, the
retrieval generation, and the selected service/authority remain contextual
Binding facts.  The live handler is still the sole implementation assembled by
the Runtime; this module neither reads retrieval storage nor creates a second
search path.
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
from .retrieval_tools import (
    HISTORY_RETRIEVAL_SOURCE_ID,
    RETRIEVAL_TOOL_IMPLEMENTATION_VERSION,
    RETRIEVE_HISTORY_TOOL_ID,
    build_retrieve_history_effect_profile,
    build_retrieve_history_execution_profile,
    build_retrieve_history_tool_spec,
)


HISTORY_RETRIEVAL_DEFINITION_MANIFEST_SCHEMA = (
    "history-retrieval-definition-manifest-v1"
)
HISTORY_RETRIEVAL_BINDING_ASSERTION_SCHEMA = (
    "history-retrieval-binding-assertion-v1"
)
HISTORY_RETRIEVAL_SOURCE_FINGERPRINT_SCHEMA = (
    "history-retrieval-source-fingerprint-v1"
)

_IMPLEMENTATION_REF = "builtin/retrieve_history"
_DECLARED_BEHAVIOR_REVISION = "history-retrieval-handler-1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class HistoryRetrievalCatalogError(ValueError):
    """The History retriever drifted from its reviewed Definition or Binding."""


@dataclass(frozen=True, slots=True)
class HistoryRetrievalToolDefinitionManifestItem:
    """The stable Definition plus its reviewed behavior declaration."""

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
class HistoryRetrievalBindingFacts:
    """Secret-free identities for one exact frozen History retrieval route.

    Each field is a canonical digest computed by the Host.  Raw Session/Task IDs,
    turn cutoffs, source snapshots, database paths, and service objects must not be
    serialized into a persisted Binding.
    """

    effect_scope_sha256: str
    session_sha256: str
    turn_cutoff_sha256: str
    retrieval_generation_sha256: str
    allowed_scopes_sha256: str
    retrieval_service_sha256: str
    retrieval_authority_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "effect_scope_sha256",
            "session_sha256",
            "turn_cutoff_sha256",
            "retrieval_generation_sha256",
            "allowed_scopes_sha256",
            "retrieval_service_sha256",
            "retrieval_authority_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)


def build_history_retrieval_tool_definition_manifest() -> tuple[
    HistoryRetrievalToolDefinitionManifestItem,
]:
    """Build the single context-free History retrieval Definition."""

    spec = build_retrieve_history_tool_spec()
    if spec.tool_id != RETRIEVE_HISTORY_TOOL_ID:
        raise HistoryRetrievalCatalogError(
            "history retrieval definition identity drifted"
        )
    effect_template = build_retrieve_history_effect_profile(default_scope="*")
    execution = build_retrieve_history_execution_profile()
    implementation_digest = _declared_implementation_digest(
        spec=spec,
        effect_template=effect_template,
        execution=execution,
    )
    definition = ToolDefinition(
        spec=spec,
        implementation_version=RETRIEVAL_TOOL_IMPLEMENTATION_VERSION,
        implementation_ref=_IMPLEMENTATION_REF,
        implementation_digest=implementation_digest,
        effect_template=effect_template,
        execution_profile=execution,
    )
    return (
        HistoryRetrievalToolDefinitionManifestItem(
            implementation_ref=_IMPLEMENTATION_REF,
            declared_behavior_revision=_DECLARED_BEHAVIOR_REVISION,
            definition=definition,
        ),
    )


def derive_history_retrieval_source_fingerprint(
    facts: HistoryRetrievalBindingFacts,
) -> str:
    """Derive the exact live source identity from contextual facts."""

    if not isinstance(facts, HistoryRetrievalBindingFacts):
        raise TypeError("facts must be HistoryRetrievalBindingFacts")
    return _canonical_sha256(
        {
            "schema_version": HISTORY_RETRIEVAL_SOURCE_FINGERPRINT_SCHEMA,
            **_facts_descriptor(facts),
        }
    )


def build_history_retrieval_tool_bindings(
    registrations: Sequence[ToolRegistration],
    *,
    facts: HistoryRetrievalBindingFacts,
) -> tuple[ToolBinding]:
    """Bind one frozen live registration to exact History authority facts."""

    if isinstance(registrations, (str, bytes)) or not isinstance(
        registrations,
        Sequence,
    ):
        raise TypeError("registrations must be a sequence of ToolRegistration values")
    if not isinstance(facts, HistoryRetrievalBindingFacts):
        raise TypeError("facts must be HistoryRetrievalBindingFacts")
    frozen = tuple(registrations)
    if len(frozen) != 1 or not isinstance(frozen[0], ToolRegistration):
        raise HistoryRetrievalCatalogError(
            "history retrieval requires exactly one ToolRegistration value"
        )
    (registration,) = frozen
    if registration.tool_id != RETRIEVE_HISTORY_TOOL_ID:
        raise HistoryRetrievalCatalogError(
            "history retrieval registration is not the canonical tool"
        )

    (manifest_item,) = build_history_retrieval_tool_definition_manifest()
    definition = manifest_item.definition
    actual_identity = ToolIdentity(
        registration.tool_id,
        registration.contract_version,
        registration.implementation_version,
    )
    if actual_identity != definition.identity:
        raise HistoryRetrievalCatalogError("history retrieval identity drifted")
    if registration.spec != definition.spec:
        raise HistoryRetrievalCatalogError(
            "history retrieval model contract drifted"
        )
    if registration.execution_profile != definition.execution_profile:
        raise HistoryRetrievalCatalogError(
            "history retrieval execution contract drifted"
        )
    effects = registration.effect_profile.effects
    if len(effects) != 1:
        raise HistoryRetrievalCatalogError("history retrieval effects drifted")
    effective_scope = effects[0].default_scope
    if (
        effective_scope == "*"
        or registration.effect_profile
        != build_retrieve_history_effect_profile(default_scope=effective_scope)
        or _sha256_text(effective_scope) != facts.effect_scope_sha256
    ):
        raise HistoryRetrievalCatalogError("history retrieval effects drifted")

    source = registration.source
    if (
        source.kind is not ToolSourceKind.LOCAL
        or source.source_id != HISTORY_RETRIEVAL_SOURCE_ID
        or source.display_name is not None
    ):
        raise HistoryRetrievalCatalogError("history retrieval source drifted")
    expected_source_fingerprint = derive_history_retrieval_source_fingerprint(facts)
    if source.fingerprint != expected_source_fingerprint:
        raise HistoryRetrievalCatalogError(
            "history retrieval source fingerprint drifted"
        )

    binding = ToolBinding(
        identity=definition.identity,
        definition_digest=definition.digest,
        source=source,
        handler=registration.handler,
        effect_profile=registration.effect_profile,
        binding_assertion={
            "schema_version": HISTORY_RETRIEVAL_BINDING_ASSERTION_SCHEMA,
            "binding_kind": "frozen_history_retrieval",
            **_facts_descriptor(facts),
            "source_fingerprint": expected_source_fingerprint,
        },
    )
    try:
        bound = BoundToolRegistration(definition, binding)
    except ValueError as exc:
        raise HistoryRetrievalCatalogError(
            "history retrieval effects drifted"
        ) from exc
    if bound.descriptor() != registration.descriptor():
        raise HistoryRetrievalCatalogError(
            "history retrieval descriptor drifted"
        )
    return (binding,)


def _facts_descriptor(facts: HistoryRetrievalBindingFacts) -> dict[str, str]:
    return {
        "effect_scope_sha256": facts.effect_scope_sha256,
        "session_sha256": facts.session_sha256,
        "turn_cutoff_sha256": facts.turn_cutoff_sha256,
        "retrieval_generation_sha256": facts.retrieval_generation_sha256,
        "allowed_scopes_sha256": facts.allowed_scopes_sha256,
        "retrieval_service_sha256": facts.retrieval_service_sha256,
        "retrieval_authority_sha256": facts.retrieval_authority_sha256,
    }


def _declared_implementation_digest(
    *,
    spec: ToolSpec,
    effect_template: ToolEffectProfile,
    execution: ToolExecutionProfile,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": HISTORY_RETRIEVAL_DEFINITION_MANIFEST_SCHEMA,
            "implementation_ref": _IMPLEMENTATION_REF,
            "declared_behavior_revision": _DECLARED_BEHAVIOR_REVISION,
            "identity": {
                "tool_id": spec.tool_id,
                "contract_version": spec.contract_version,
                "implementation_version": RETRIEVAL_TOOL_IMPLEMENTATION_VERSION,
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
        raise HistoryRetrievalCatalogError(
            f"{field_name} must be a canonical SHA-256 digest"
        )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
    "HISTORY_RETRIEVAL_BINDING_ASSERTION_SCHEMA",
    "HISTORY_RETRIEVAL_DEFINITION_MANIFEST_SCHEMA",
    "HISTORY_RETRIEVAL_SOURCE_FINGERPRINT_SCHEMA",
    "HistoryRetrievalBindingFacts",
    "HistoryRetrievalCatalogError",
    "HistoryRetrievalToolDefinitionManifestItem",
    "build_history_retrieval_tool_bindings",
    "build_history_retrieval_tool_definition_manifest",
    "derive_history_retrieval_source_fingerprint",
]
