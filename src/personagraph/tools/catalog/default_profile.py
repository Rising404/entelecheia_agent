"""Pure materialization of a new process-default bound Tool catalog."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .model import CatalogStatus
from .binding import (
    BoundToolRegistration,
    ToolBinding,
    ToolIdentity,
)
from .persistence.records import (
    DefaultCatalogResolutionState,
    DefaultProfileAvailability,
)
from .trusted_factories import (
    ToolFactoryDriftError,
    ToolFactoryUnavailableError,
    TrustedCatalogResolutionError,
    TrustedDefaultFactoryRegistry,
)


class DefaultProfileMaterializationError(TrustedCatalogResolutionError):
    """The selected persistent default profile is internally inconsistent."""


class RequiredToolUnavailableError(DefaultProfileMaterializationError):
    """A required default profile item could not be built."""


class PersistentCatalogReader(Protocol):
    def default_resolution_state(self) -> DefaultCatalogResolutionState: ...


@dataclass(frozen=True)
class UnavailableDefaultTool:
    identity: ToolIdentity
    implementation_ref: str
    reason: str


@dataclass(frozen=True)
class MaterializedDefaultCatalog:
    """Pure bound result; Runtime remains responsible for Attempt ownership."""

    profile_revision: int
    profile_digest: str
    catalog_revision: int
    catalog_digest: str
    profile_catalog_revision: int
    registrations: tuple[BoundToolRegistration, ...]
    unavailable: tuple[UnavailableDefaultTool, ...] = ()


def materialize_default_catalog(
    repository: PersistentCatalogReader,
    registry: TrustedDefaultFactoryRegistry,
) -> MaterializedDefaultCatalog:
    """Intersect an exact profile with current lifecycle state, then bind it."""

    if not isinstance(registry, TrustedDefaultFactoryRegistry):
        raise TypeError("registry must be a TrustedDefaultFactoryRegistry")
    return _materialize_default_catalog_from_state(
        repository.default_resolution_state(),
        registry,
    )


def materialize_contextual_default_catalog(
    repository: PersistentCatalogReader,
    registry: TrustedDefaultFactoryRegistry,
    *,
    contextual_bindings: Sequence[ToolBinding] = (),
) -> MaterializedDefaultCatalog:
    """Bind the published default profile to one runtime resource context.

    The persistent profile remains the complete handler-free model surface.
    Missing contextual bindings are returned as typed ``unavailable`` entries;
    they never remove definitions from the profile itself.  Runtime owners keep
    responsibility for authorization, fresh emergency-revocation checks and
    durable invocation settlement.
    """

    if not isinstance(registry, TrustedDefaultFactoryRegistry):
        raise TypeError("registry must be a TrustedDefaultFactoryRegistry")
    return _materialize_default_catalog_from_state(
        repository.default_resolution_state(),
        registry,
        contextual_bindings=contextual_bindings,
        allow_contextual_resolution=True,
    )


def _materialize_default_catalog_from_state(
    state: DefaultCatalogResolutionState,
    registry: TrustedDefaultFactoryRegistry,
    *,
    contextual_bindings: Sequence[ToolBinding] = (),
    allow_contextual_resolution: bool = False,
) -> MaterializedDefaultCatalog:
    """Bind one already-atomic profile state without granting extra capabilities."""

    if not isinstance(state, DefaultCatalogResolutionState):
        raise TypeError("state must be a DefaultCatalogResolutionState")
    if not isinstance(registry, TrustedDefaultFactoryRegistry):
        raise TypeError("registry must be a TrustedDefaultFactoryRegistry")
    candidates = _index_contextual_bindings(contextual_bindings)
    profile = state.default_profile
    profile_entries = {
        entry.identity: entry for entry in state.profile_catalog.entries
    }
    current_entries = {
        entry.identity: entry for entry in state.current_catalog.entries
    }
    definitions = {item.identity: item for item in state.definitions}
    selected_identities = {item.identity for item in profile.items}
    unselected = tuple(sorted(set(candidates) - selected_identities))
    if unselected:
        raise DefaultProfileMaterializationError(
            "contextual bindings cannot add tools outside the default profile: "
            f"{unselected!r}"
        )
    registrations: list[BoundToolRegistration] = []
    unavailable: list[UnavailableDefaultTool] = []
    for item in profile.items:
        profile_entry = profile_entries.get(item.identity)
        if profile_entry is None:
            raise DefaultProfileMaterializationError(
                "default profile identity is absent from its catalog revision: "
                f"{item.identity!r}"
            )
        if profile_entry.status is not CatalogStatus.ACTIVE:
            raise DefaultProfileMaterializationError(
                "default profile identity was not active when published: "
                f"{item.identity!r}"
            )
        if profile_entry.definition_digest != item.definition_digest:
            raise DefaultProfileMaterializationError(
                f"default profile definition digest drifted: {item.identity!r}"
            )
        definition = definitions.get(item.identity)
        if definition is None:
            raise DefaultProfileMaterializationError(
                "selected ToolDefinition is absent from the atomic resolution state"
            )
        if definition.digest != item.definition_digest:
            raise DefaultProfileMaterializationError(
                "loaded ToolDefinition digest does not match the default profile"
            )
        factory_owned = registry.owns(definition)
        contextual = candidates.get(item.identity)
        if factory_owned and contextual is not None:
            raise DefaultProfileMaterializationError(
                "contextual binding cannot override a trusted default factory: "
                f"{item.identity!r}"
            )
        contextual_bound: BoundToolRegistration | None = None
        if contextual is not None:
            try:
                contextual_bound = BoundToolRegistration(definition, contextual)
            except (TypeError, ValueError) as exc:
                raise ToolFactoryDriftError(
                    "contextual binding does not match the persisted ToolDefinition"
                ) from exc
        current_entry = current_entries.get(item.identity)
        unavailable_reason: str | None = None
        if current_entry is None:
            unavailable_reason = "catalog_entry_absent"
        elif current_entry.definition_digest != item.definition_digest:
            raise DefaultProfileMaterializationError(
                f"current catalog definition digest drifted: {item.identity!r}"
            )
        elif current_entry.status is not CatalogStatus.ACTIVE:
            unavailable_reason = f"catalog_status:{current_entry.status.value}"
        if unavailable_reason is not None:
            _record_unavailable(
                unavailable,
                availability=item.availability,
                identity=item.identity,
                implementation_ref=definition.implementation_ref,
                reason=unavailable_reason,
            )
            continue
        if factory_owned:
            try:
                bound = registry.resolve(
                    definition,
                    expected_definition_digest=item.definition_digest,
                )
            except ToolFactoryUnavailableError as exc:
                _record_unavailable(
                    unavailable,
                    availability=item.availability,
                    identity=item.identity,
                    implementation_ref=definition.implementation_ref,
                    reason=exc.reason,
                    cause=exc,
                )
                continue
        elif contextual_bound is not None and allow_contextual_resolution:
            bound = contextual_bound
        elif allow_contextual_resolution:
            _record_unavailable(
                unavailable,
                availability=item.availability,
                identity=item.identity,
                implementation_ref=definition.implementation_ref,
                reason="contextual_binding_unavailable",
            )
            continue
        else:
            # The standalone default resolver deliberately keeps unknown factories
            # fatal. Only the unified Runtime materializer may open the contextual
            # lane, and then only for identities already selected by the profile.
            bound = registry.resolve(
                definition,
                expected_definition_digest=item.definition_digest,
            )
        registrations.append(bound)
    return MaterializedDefaultCatalog(
        profile_revision=profile.revision,
        profile_digest=profile.digest,
        catalog_revision=state.current_catalog.revision,
        catalog_digest=state.current_catalog.digest,
        profile_catalog_revision=state.profile_catalog.revision,
        registrations=tuple(registrations),
        unavailable=tuple(unavailable),
    )


def _index_contextual_bindings(
    bindings: Sequence[ToolBinding],
) -> dict[ToolIdentity, ToolBinding]:
    if isinstance(bindings, (str, bytes)):
        raise TypeError("contextual_bindings must be a sequence of ToolBinding values")
    indexed: dict[ToolIdentity, ToolBinding] = {}
    for binding in bindings:
        if not isinstance(binding, ToolBinding):
            raise TypeError(
                "contextual_bindings must contain only ToolBinding values"
            )
        if binding.identity in indexed:
            raise DefaultProfileMaterializationError(
                f"duplicate contextual binding identity: {binding.identity!r}"
            )
        indexed[binding.identity] = binding
    return indexed


def _record_unavailable(
    unavailable: list[UnavailableDefaultTool],
    *,
    availability: DefaultProfileAvailability,
    identity: ToolIdentity,
    implementation_ref: str,
    reason: str,
    cause: BaseException | None = None,
) -> None:
    if availability is DefaultProfileAvailability.REQUIRED:
        error = RequiredToolUnavailableError(
            f"required default tool is unavailable: {identity!r}: {reason}"
        )
        if cause is not None:
            raise error from cause
        raise error
    unavailable.append(
        UnavailableDefaultTool(
            identity=identity,
            implementation_ref=implementation_ref,
            reason=reason,
        )
    )


__all__ = [
    "DefaultProfileMaterializationError",
    "MaterializedDefaultCatalog",
    "RequiredToolUnavailableError",
    "UnavailableDefaultTool",
    "materialize_contextual_default_catalog",
    "materialize_default_catalog",
]
