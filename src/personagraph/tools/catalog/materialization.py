"""Materialize and exactly re-bind Attempt-owned live Tool Catalogs.

This is the single tools-layer composition boundary between the durable default
Catalog, contextual live bindings, frozen Attempt payloads, and the emergency
revocation overlay.  It does not persist Attempts or invoke tool handlers.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from types import MappingProxyType
from typing import Mapping, Protocol

from .snapshots.attempt import (
    AttemptToolBindingOwnerKind,
    AttemptToolBindingOwner,
    AttemptToolCatalogProvenance,
    FrozenAttemptToolCatalog,
    decode_frozen_attempt_tool_catalog,
)
from .model import (
    CatalogEntry,
    CatalogSnapshot,
    CatalogStatus,
    ToolCatalog,
    ToolKey,
    ToolResolutionError,
)
from .binding import (
    BoundToolRegistration,
    ToolBinding,
    ToolDefinition,
    ToolIdentity,
)
from .persistence.records import (
    DefaultCatalogResolutionState,
    DefaultProfileAvailability,
    EmergencyRevocation,
    EmergencyRevocationTarget,
)
from ..contracts import ToolCallProposal
from .default_profile import (
    DefaultProfileMaterializationError,
    RequiredToolUnavailableError,
    UnavailableDefaultTool,
    _materialize_default_catalog_from_state,
)
from .revocation import (
    EmergencyRevocationDenied,
    EmergencyRevocationGuard,
    EmergencyRevocationGuardDecision,
    EmergencyRevocationGuardReason,
    EmergencyRevocationGuardStage,
)
from ..policy import TOOL_POLICY_VERSION
from .snapshots.bound import FrozenBoundCatalog, FrozenBoundCatalogEntry
from .trusted_factories import (
    ToolFactoryUnavailableError,
    TrustedDefaultFactoryRegistry,
)


class RuntimeCatalogMaterializationError(DefaultProfileMaterializationError):
    """A live Attempt Catalog cannot be built without weakening its freeze."""


class RuntimeCatalogRepository(Protocol):
    """One authority supplies both atomic defaults and the live deny overlay."""

    def default_resolution_state(self) -> DefaultCatalogResolutionState: ...

    def list_active_emergency_revocations(
        self,
    ) -> tuple[EmergencyRevocation, ...]: ...


class MaterializedRuntimeCatalog:
    """Live registrations paired with one exact frozen Attempt envelope.

    The live CatalogSnapshot and revocation guard are deliberately private.
    Runtime callers receive handler-free definitions in the persisted exposure
    order and must call the relevant authorization method immediately before
    each future transition.  In particular, every physical retry calls
    :meth:`authorize_execute` again.  Guard decisions are not reusable tokens
    and do not replace ToolPolicy or the Runtime's operation ledger.
    """

    __slots__ = (
        "_frozen_attempt_catalog",
        "_guard",
        "_definitions",
        "_frozen_entries_by_identity",
        "_registrations_by_identity",
        "_snapshot",
        "_unavailable",
    )

    def __init__(
        self,
        *,
        snapshot: CatalogSnapshot,
        frozen_attempt_catalog: FrozenAttemptToolCatalog,
        guard: EmergencyRevocationGuard,
        unavailable: tuple[UnavailableDefaultTool, ...] = (),
    ) -> None:
        if not isinstance(snapshot, CatalogSnapshot):
            raise TypeError("snapshot must be a CatalogSnapshot")
        if not isinstance(frozen_attempt_catalog, FrozenAttemptToolCatalog):
            raise TypeError(
                "frozen_attempt_catalog must be a FrozenAttemptToolCatalog"
            )
        if not isinstance(guard, EmergencyRevocationGuard):
            raise TypeError("guard must be an EmergencyRevocationGuard")
        if not isinstance(unavailable, tuple) or any(
            not isinstance(item, UnavailableDefaultTool) for item in unavailable
        ):
            raise TypeError(
                "unavailable must contain only UnavailableDefaultTool values"
            )

        registrations_by_identity = _live_registration_index(snapshot)
        if any(entry.status is not CatalogStatus.ACTIVE for entry in snapshot.entries):
            raise RuntimeCatalogMaterializationError(
                "Attempt live and frozen catalog entries must all be active"
            )
        _require_unique_execution_names(
            tuple(registrations_by_identity),
            label="materialized live catalog",
        )
        refrozen = FrozenBoundCatalog.from_catalog_snapshot(snapshot)
        expected_frozen = frozen_attempt_catalog.frozen_catalog
        if (
            refrozen.digest != expected_frozen.digest
            or refrozen.descriptor() != expected_frozen.descriptor()
        ):
            raise RuntimeCatalogMaterializationError(
                "live catalog does not exactly match the frozen Attempt catalog"
            )
        try:
            registrations = tuple(
                registrations_by_identity[identity]
                for identity in frozen_attempt_catalog.exposure_order
            )
        except KeyError as exc:
            raise RuntimeCatalogMaterializationError(
                "exposure order references an absent live registration"
            ) from exc
        frozen_entries_by_identity = {
            entry.identity: entry for entry in expected_frozen.entries
        }
        unavailable_identities = tuple(item.identity for item in unavailable)
        if len(set(unavailable_identities)) != len(unavailable_identities):
            raise RuntimeCatalogMaterializationError(
                "unavailable diagnostics contain duplicate identities"
            )
        if set(unavailable_identities).intersection(registrations_by_identity):
            raise RuntimeCatalogMaterializationError(
                "a tool cannot be both materialized and unavailable"
            )

        self._snapshot = snapshot
        self._frozen_attempt_catalog = frozen_attempt_catalog
        self._guard = guard
        self._registrations_by_identity: Mapping[
            ToolIdentity,
            BoundToolRegistration,
        ] = MappingProxyType(registrations_by_identity)
        self._frozen_entries_by_identity: Mapping[
            ToolIdentity,
            FrozenBoundCatalogEntry,
        ] = MappingProxyType(frozen_entries_by_identity)
        self._definitions = tuple(item.definition for item in registrations)
        self._unavailable = unavailable

    @property
    def frozen_attempt_catalog(self) -> FrozenAttemptToolCatalog:
        return self._frozen_attempt_catalog

    @property
    def exposure_order(self) -> tuple[ToolIdentity, ...]:
        return self._frozen_attempt_catalog.exposure_order

    @property
    def exposed_definitions(self) -> tuple[ToolDefinition, ...]:
        """Return handler-free definitions in the persisted model exposure order."""

        return self._definitions

    @property
    def unavailable(self) -> tuple[UnavailableDefaultTool, ...]:
        """Non-authoritative diagnostics excluded from the frozen envelope."""

        return self._unavailable

    def resolve_for_proposal(
        self,
        proposal: ToolCallProposal,
    ) -> FrozenBoundCatalogEntry:
        """Resolve and guard a proposal, returning only handler-free frozen facts."""

        if not isinstance(proposal, ToolCallProposal):
            raise TypeError("proposal must be a ToolCallProposal")
        registration = self._snapshot.resolve(
            proposal.tool_id,
            proposal.contract_version,
        )
        if not isinstance(registration, BoundToolRegistration):
            raise RuntimeCatalogMaterializationError(
                "live Attempt catalog contains a non-bound registration"
            )
        self._enforce(registration, EmergencyRevocationGuardStage.RESOLVE)
        return self._frozen_entries_by_identity[registration.identity]

    def authorize_resume(
        self,
        identity: ToolIdentity,
    ) -> EmergencyRevocationGuardDecision:
        """Freshly guard resume; physical dispatch still requires EXECUTE."""

        return self._enforce_identity(
            identity,
            EmergencyRevocationGuardStage.RESUME,
        )

    def authorize_execute(
        self,
        identity: ToolIdentity,
    ) -> EmergencyRevocationGuardDecision:
        """Guard one imminent physical attempt, including every retry.

        The production bridge must keep this check adjacent to dispatch and must
        separately enforce ToolPolicy.  The returned decision is not a token.
        """

        return self._enforce_identity(
            identity,
            EmergencyRevocationGuardStage.EXECUTE,
        )

    def _enforce_identity(
        self,
        identity: ToolIdentity,
        stage: EmergencyRevocationGuardStage,
    ) -> EmergencyRevocationGuardDecision:
        if not isinstance(identity, ToolIdentity):
            raise TypeError("identity must be a ToolIdentity")
        try:
            registration = self._registrations_by_identity[identity]
        except KeyError as exc:
            raise ToolResolutionError(
                f"unknown frozen tool identity: {identity!r}"
            ) from exc
        # This also rejects a non-resolvable status from a malformed historical
        # envelope instead of letting the direct identity lookup bypass Catalog.
        resolved = self._snapshot.resolve(
            identity.tool_id,
            identity.contract_version,
        )
        if resolved is not registration:
            raise RuntimeCatalogMaterializationError(
                "frozen identity does not resolve to its exact live registration"
            )
        return self._enforce(registration, stage)

    def _enforce(
        self,
        registration: BoundToolRegistration,
        stage: EmergencyRevocationGuardStage,
    ) -> EmergencyRevocationGuardDecision:
        target = _revocation_target(registration)
        return self._guard.enforce_current(stage=stage, target=target)


class RuntimeCatalogMaterializer:
    """Build new or recovered live catalogs through one tools-owned boundary."""

    def __init__(
        self,
        repository: RuntimeCatalogRepository,
        registry: TrustedDefaultFactoryRegistry,
    ) -> None:
        if not callable(getattr(repository, "default_resolution_state", None)):
            raise TypeError("repository must provide default_resolution_state()")
        if not callable(
            getattr(repository, "list_active_emergency_revocations", None)
        ):
            raise TypeError(
                "repository must provide list_active_emergency_revocations()"
            )
        if not isinstance(registry, TrustedDefaultFactoryRegistry):
            raise TypeError("registry must be a TrustedDefaultFactoryRegistry")
        self._repository = repository
        self._registry = registry
        self._guard = EmergencyRevocationGuard(repository)

    def materialize_new(
        self,
        *,
        attempt_id: str,
        created_at: datetime,
        contextual_bindings: Sequence[ToolBinding] = (),
    ) -> MaterializedRuntimeCatalog:
        """Materialize one new Attempt from exactly one atomic state read."""

        contextual = tuple(
            _contextual_binding_index(contextual_bindings).values()
        )
        state = self._repository.default_resolution_state()
        if not isinstance(state, DefaultCatalogResolutionState):
            raise RuntimeCatalogMaterializationError(
                "default_resolution_state returned an invalid value"
            )
        profile_identities = tuple(
            item.identity for item in state.default_profile.items
        )
        _require_unique_execution_names(
            profile_identities,
            label="default profile",
        )
        defaults = _materialize_default_catalog_from_state(
            state,
            self._registry,
            contextual_bindings=contextual,
            allow_contextual_resolution=True,
        )
        registrations, unavailable = self._apply_new_attempt_resolve_guard(
            state=state,
            registrations=defaults.registrations,
            unavailable=defaults.unavailable,
        )
        _require_unique_registration_namespace(
            registrations,
            label="new live catalog",
        )
        live_catalog = ToolCatalog()
        for registration in sorted(
            registrations,
            key=lambda item: item.identity,
        ):
            live_catalog.register(registration, status=CatalogStatus.ACTIVE)
        snapshot = live_catalog.snapshot()
        frozen_catalog = FrozenBoundCatalog.from_catalog_snapshot(snapshot)
        frozen_attempt = FrozenAttemptToolCatalog(
            attempt_id=attempt_id,
            created_at=created_at,
            policy_version=TOOL_POLICY_VERSION,
            provenance=AttemptToolCatalogProvenance(
                source_catalog_revision=defaults.catalog_revision,
                source_catalog_digest=defaults.catalog_digest,
                default_profile_revision=defaults.profile_revision,
                default_profile_digest=defaults.profile_digest,
                profile_catalog_revision=defaults.profile_catalog_revision,
            ),
            frozen_catalog=frozen_catalog,
            exposure_order=tuple(item.identity for item in registrations),
            binding_owners=tuple(
                sorted(
                    (
                        AttemptToolBindingOwner(
                            identity=registration.identity,
                            kind=(
                                AttemptToolBindingOwnerKind.TRUSTED_FACTORY
                                if self._registry.owns(registration.definition)
                                else AttemptToolBindingOwnerKind.CONTEXTUAL_CANDIDATE
                            ),
                        )
                        for registration in registrations
                    ),
                    key=lambda owner: owner.identity,
                )
            ),
        )
        return MaterializedRuntimeCatalog(
            snapshot=snapshot,
            frozen_attempt_catalog=frozen_attempt,
            guard=self._guard,
            unavailable=unavailable,
        )

    def rebind_frozen(
        self,
        *,
        canonical_json: str,
        expected_sha256: str,
        expected_attempt_id: str,
        contextual_bindings: Sequence[ToolBinding] = (),
    ) -> MaterializedRuntimeCatalog:
        """Exactly restore a frozen Attempt without reading ordinary lifecycle."""

        frozen_attempt = decode_frozen_attempt_tool_catalog(
            canonical_json=canonical_json,
            expected_sha256=expected_sha256,
        )
        _require_expected_attempt_owner(
            frozen_attempt.attempt_id,
            expected_attempt_id,
        )
        if frozen_attempt.policy_version != TOOL_POLICY_VERSION:
            raise RuntimeCatalogMaterializationError(
                "frozen Attempt Tool Catalog uses a non-executable policy version"
            )
        _require_unique_execution_names(
            frozen_attempt.exposure_order,
            label="frozen Attempt catalog",
        )
        if any(
            entry.status is not CatalogStatus.ACTIVE
            for entry in frozen_attempt.frozen_catalog.entries
        ):
            raise RuntimeCatalogMaterializationError(
                "frozen Attempt catalog entries must all be active"
            )
        frozen_identities = frozenset(
            entry.identity for entry in frozen_attempt.frozen_catalog.entries
        )
        contextual = _contextual_binding_index(
            contextual_bindings,
            relevant_identities=frozen_identities,
        )
        owner_by_identity = {
            owner.identity: owner.kind
            for owner in frozen_attempt.binding_owners
        }
        registrations: dict[ToolIdentity, BoundToolRegistration] = {}
        live_entries: list[CatalogEntry] = []
        for frozen_entry in frozen_attempt.frozen_catalog.entries:
            definition = frozen_entry.definition
            candidate = contextual.get(definition.identity)
            owner_kind = owner_by_identity[definition.identity]
            try:
                factory_owned = self._registry.owns(definition)
                if owner_kind is AttemptToolBindingOwnerKind.TRUSTED_FACTORY:
                    if not factory_owned:
                        raise RuntimeCatalogMaterializationError(
                            "frozen trusted factory owner is no longer registered: "
                            f"{definition.identity!r}"
                        )
                    if candidate is not None:
                        raise RuntimeCatalogMaterializationError(
                            "contextual binding cannot override a frozen trusted "
                            f"factory owner: {definition.identity!r}"
                        )
                    registration = self._registry.resolve(
                        definition,
                        expected_definition_digest=frozen_entry.definition_digest,
                    )
                else:
                    if factory_owned:
                        raise RuntimeCatalogMaterializationError(
                            "trusted factory cannot replace a frozen contextual "
                            f"owner: {definition.identity!r}"
                        )
                    if candidate is None:
                        raise RuntimeCatalogMaterializationError(
                            "frozen contextual binding is unavailable: "
                            f"{definition.identity!r}"
                        )
                    registration = BoundToolRegistration(definition, candidate)
                frozen_entry.binding.require_exact_live_binding(
                    registration.binding
                )
            except RuntimeCatalogMaterializationError:
                raise
            except ToolFactoryUnavailableError as exc:
                raise RuntimeCatalogMaterializationError(
                    "frozen trusted factory is unavailable: "
                    f"{definition.identity!r}"
                ) from exc
            except (TypeError, ValueError) as exc:
                raise RuntimeCatalogMaterializationError(
                    "live binding does not exactly match the frozen registration: "
                    f"{definition.identity!r}"
                ) from exc
            registrations[definition.identity] = registration
            live_entries.append(
                CatalogEntry(
                    key=frozen_entry.key,
                    registration=registration,
                    status=frozen_entry.status,
                    created_revision=frozen_entry.created_revision,
                    updated_revision=frozen_entry.updated_revision,
                )
            )
        _require_unique_registration_namespace(
            tuple(registrations.values()),
            label="rebound live catalog",
        )
        snapshot = CatalogSnapshot(
            revision=frozen_attempt.frozen_catalog.revision,
            entries=tuple(live_entries),
        )
        return MaterializedRuntimeCatalog(
            snapshot=snapshot,
            frozen_attempt_catalog=frozen_attempt,
            guard=self._guard,
        )

    def _apply_new_attempt_resolve_guard(
        self,
        *,
        state: DefaultCatalogResolutionState,
        registrations: tuple[BoundToolRegistration, ...],
        unavailable: tuple[UnavailableDefaultTool, ...],
    ) -> tuple[
        tuple[BoundToolRegistration, ...],
        tuple[UnavailableDefaultTool, ...],
    ]:
        registrations_by_identity = _registration_index(
            registrations,
            label="default materialization",
        )
        unavailable_by_identity = _unavailable_index(unavailable)
        selected: list[BoundToolRegistration] = []
        diagnostics: list[UnavailableDefaultTool] = []
        for item in state.default_profile.items:
            existing_diagnostic = unavailable_by_identity.pop(
                item.identity,
                None,
            )
            if existing_diagnostic is not None:
                diagnostics.append(existing_diagnostic)
                continue
            try:
                registration = registrations_by_identity.pop(item.identity)
            except KeyError as exc:
                raise RuntimeCatalogMaterializationError(
                    "default materialization omitted a profile identity without "
                    "an unavailable diagnostic"
                ) from exc
            decision = self._guard.check_current(
                stage=EmergencyRevocationGuardStage.RESOLVE,
                target=_revocation_target(registration),
            )
            if decision.allowed:
                selected.append(registration)
                continue
            if decision.reason is not EmergencyRevocationGuardReason.ACTIVE:
                raise EmergencyRevocationDenied(decision)
            denial = EmergencyRevocationDenied(decision)
            if item.availability is DefaultProfileAvailability.REQUIRED:
                raise RequiredToolUnavailableError(
                    "required default tool is unavailable: "
                    f"{item.identity!r}: {decision.code}"
                ) from denial
            diagnostics.append(
                UnavailableDefaultTool(
                    identity=item.identity,
                    implementation_ref=registration.definition.implementation_ref,
                    reason=decision.code,
                )
            )
        if registrations_by_identity or unavailable_by_identity:
            raise RuntimeCatalogMaterializationError(
                "default materialization returned identities outside the profile"
            )
        return tuple(selected), tuple(diagnostics)


def _contextual_binding_index(
    bindings: Sequence[ToolBinding],
    *,
    relevant_identities: frozenset[ToolIdentity] | None = None,
) -> dict[ToolIdentity, ToolBinding]:
    if isinstance(bindings, (str, bytes)) or not isinstance(bindings, Sequence):
        raise TypeError(
            "contextual_bindings must be a sequence of ToolBinding values"
        )
    indexed: dict[ToolIdentity, ToolBinding] = {}
    for binding in bindings:
        if not isinstance(binding, ToolBinding):
            raise TypeError(
                "contextual_bindings must contain only ToolBinding values"
            )
        if (
            relevant_identities is not None
            and binding.identity not in relevant_identities
        ):
            continue
        if binding.identity in indexed:
            raise RuntimeCatalogMaterializationError(
                f"duplicate contextual binding identity: {binding.identity!r}"
            )
        indexed[binding.identity] = binding
    return indexed


def _registration_index(
    registrations: tuple[BoundToolRegistration, ...],
    *,
    label: str,
) -> dict[ToolIdentity, BoundToolRegistration]:
    indexed: dict[ToolIdentity, BoundToolRegistration] = {}
    for registration in registrations:
        if not isinstance(registration, BoundToolRegistration):
            raise RuntimeCatalogMaterializationError(
                f"{label} contains a non-bound registration"
            )
        if registration.identity in indexed:
            raise RuntimeCatalogMaterializationError(
                f"{label} contains duplicate identities"
            )
        indexed[registration.identity] = registration
    return indexed


def _unavailable_index(
    unavailable: tuple[UnavailableDefaultTool, ...],
) -> dict[ToolIdentity, UnavailableDefaultTool]:
    indexed: dict[ToolIdentity, UnavailableDefaultTool] = {}
    for item in unavailable:
        if not isinstance(item, UnavailableDefaultTool):
            raise RuntimeCatalogMaterializationError(
                "default materialization returned invalid unavailable diagnostics"
            )
        if item.identity in indexed:
            raise RuntimeCatalogMaterializationError(
                "default materialization returned duplicate unavailable diagnostics"
            )
        indexed[item.identity] = item
    return indexed


def _live_registration_index(
    snapshot: CatalogSnapshot,
) -> dict[ToolIdentity, BoundToolRegistration]:
    registrations: list[BoundToolRegistration] = []
    keys: list[ToolKey] = []
    for entry in snapshot.entries:
        if not isinstance(entry, CatalogEntry):
            raise RuntimeCatalogMaterializationError(
                "live catalog contains an invalid entry"
            )
        if not isinstance(entry.registration, BoundToolRegistration):
            raise RuntimeCatalogMaterializationError(
                "live catalog contains a non-bound registration"
            )
        registrations.append(entry.registration)
        keys.append(entry.key)
    if len(set(keys)) != len(keys):
        raise RuntimeCatalogMaterializationError(
            "live catalog contains duplicate ToolKeys"
        )
    return _registration_index(
        tuple(registrations),
        label="live catalog",
    )


def _require_unique_registration_namespace(
    registrations: tuple[BoundToolRegistration, ...],
    *,
    label: str,
) -> None:
    _registration_index(registrations, label=label)
    keys = tuple(
        ToolKey(item.tool_id, item.contract_version) for item in registrations
    )
    if len(set(keys)) != len(keys):
        raise RuntimeCatalogMaterializationError(
            f"{label} contains duplicate ToolKeys"
        )
    _require_unique_execution_names(
        tuple(item.identity for item in registrations),
        label=label,
    )


def _require_unique_execution_names(
    identities: tuple[ToolIdentity, ...],
    *,
    label: str,
) -> None:
    if any(not isinstance(identity, ToolIdentity) for identity in identities):
        raise RuntimeCatalogMaterializationError(
            f"{label} contains an invalid ToolIdentity"
        )
    if len(set(identities)) != len(identities):
        raise RuntimeCatalogMaterializationError(
            f"{label} contains duplicate identities"
        )
    keys = tuple(
        (identity.tool_id, identity.contract_version) for identity in identities
    )
    if len(set(keys)) != len(keys):
        raise RuntimeCatalogMaterializationError(
            f"{label} contains duplicate ToolKeys"
        )
    tool_ids = tuple(identity.tool_id for identity in identities)
    if len(set(tool_ids)) != len(tool_ids):
        raise RuntimeCatalogMaterializationError(
            f"{label} contains duplicate tool_id values"
        )


def _revocation_target(
    registration: BoundToolRegistration,
) -> EmergencyRevocationTarget:
    try:
        return EmergencyRevocationTarget.from_bound_registration(registration)
    except (TypeError, ValueError) as exc:
        raise RuntimeCatalogMaterializationError(
            "bound registration cannot form a complete revocation target"
        ) from exc


def _require_expected_attempt_owner(
    actual_attempt_id: str,
    expected_attempt_id: str,
) -> None:
    if (
        not isinstance(expected_attempt_id, str)
        or not expected_attempt_id
        or expected_attempt_id.strip() != expected_attempt_id
    ):
        raise ValueError("expected_attempt_id must be canonical non-empty text")
    if actual_attempt_id != expected_attempt_id:
        raise RuntimeCatalogMaterializationError(
            "frozen Tool Catalog belongs to a different Attempt"
        )


__all__ = [
    "MaterializedRuntimeCatalog",
    "RuntimeCatalogRepository",
    "RuntimeCatalogMaterializationError",
    "RuntimeCatalogMaterializer",
]
