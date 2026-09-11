"""可安全作为默认种子的无全局注册表工具注册。

这是能力来源，而非 Runtime 目录。调用方仍须绑定 Session/Turn 资源、取权威信息交集、
冻结执行快照，并通过自己的持久操作边界路由受保护效果。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..catalog import CatalogConflictError, CatalogStatus
from ..catalog.persistence import (
    CatalogBootstrapResult,
    DefaultProfileAvailability,
    DefaultProfileSelection,
    DefaultProfileSnapshot,
    ToolCatalogSeed,
    ToolCatalogRepository,
)
from ..catalog.binding import ToolDefinition
from ..contracts import ToolSourceDescriptor
from ..documents import (
    build_external_visual_analysis_tool_definition_manifest,
    build_format_observation_tool_definition_manifest,
)
from ..registration import ToolRegistration
from ..findings import build_execution_findings_tool_definition_manifest
from ..retrieval import (
    build_file_retrieval_tool_definition_manifest,
    build_history_retrieval_tool_definition_manifest,
)
from ..time.date_tools import DATE_TOOL_IDS, build_date_tool_registrations
from ..catalog.trusted_factories import (
    ToolFactoryUnavailableError,
    TrustedContextualBinding,
    TrustedContextualDefaultToolFactory,
    TrustedDefaultFactoryRegistry,
    TrustedDefaultToolFactory,
    TrustedStaticDefaultToolFactory,
)
from ..web.web_tools import (
    WEB_FETCH_TOOL_ID,
    WEB_SEARCH_TOOL_ID,
    WEB_TOOL_SOURCE,
    build_web_tool_registrations,
    freeze_web_search_provider_pipeline,
)
from ..workspace.workspace_discovery_catalog import (
    build_workspace_discovery_tool_definition_manifest,
)
from ..workspace.workspace_write_catalog import (
    build_workspace_write_tool_definition_manifest,
)
from ..workspace.output_file_tool import build_output_file_definition
from ..visual import (
    build_file_visual_tool_definition_manifest,
)
from ..files.file_catalog import file_tool_definition_manifest
from ..documents.file_chunk_catalog import build_file_chunk_tool_definition_manifest
from ..documents.file_inspection_catalog import file_inspection_definition_manifest
from ..tool_history import tool_history_definition_manifest


@dataclass(frozen=True)
class _ProductionDefault:
    factory: TrustedDefaultToolFactory
    availability: DefaultProfileAvailability


class ProductionDefaultProfileResolutionError(ValueError):
    """The persisted default profile is not the reviewed L1 production table."""


@dataclass(frozen=True, slots=True)
class ProductionDefaultDefinitionProfileItem:
    """One handler-free Definition in persisted default-profile order."""

    ordinal: int
    availability: DefaultProfileAvailability
    definition: ToolDefinition

    def __post_init__(self) -> None:
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
        ):
            raise ValueError("ordinal must be a non-negative integer")
        if not isinstance(self.availability, DefaultProfileAvailability):
            raise TypeError(
                "availability must be DefaultProfileAvailability"
            )
        if not isinstance(self.definition, ToolDefinition):
            raise TypeError("definition must be a ToolDefinition")


@dataclass(frozen=True, slots=True)
class ProductionDefaultDefinitionProfile:
    """One atomic, handler-free read of the canonical persisted L1 profile."""

    catalog_revision: int
    catalog_digest: str
    profile_catalog_revision: int
    profile_catalog_digest: str
    profile_revision: int
    profile_digest: str
    items: tuple[ProductionDefaultDefinitionProfileItem, ...]

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(item.definition for item in self.items)


# One pass can migrate all 26 contracts.  The bound caps only repeated CAS
# conflicts; it does not silently turn a partially migrated catalog into success.
_MAX_RECONCILIATION_ATTEMPTS = 128
_RETIRED_FILE_TOOL_IDS = frozenset({
    "list_file_candidates", "select_file_candidates",
    "prepare_file_candidates", "retrieve_file_candidates",
})


def _production_defaults() -> tuple[_ProductionDefault, ...]:
    date_registrations = _index_exact_registrations(
        build_date_tool_registrations(),
        expected_ids=DATE_TOOL_IDS,
    )
    web_registrations = _index_exact_registrations(
        build_web_tool_registrations(),
        expected_ids=(WEB_SEARCH_TOOL_ID, WEB_FETCH_TOOL_ID),
    )

    def static_default(
        *,
        implementation_ref: str,
        declared_behavior_revision: str,
        registration: ToolRegistration,
        availability: DefaultProfileAvailability,
    ) -> _ProductionDefault:
        return _ProductionDefault(
            factory=TrustedStaticDefaultToolFactory.from_registration(
                implementation_ref=implementation_ref,
                declared_behavior_revision=declared_behavior_revision,
                registration=registration,
            ),
            availability=availability,
        )

    def contextual_default(
        *,
        implementation_ref: str,
        declared_behavior_revision: str,
        registration: ToolRegistration,
        availability: DefaultProfileAvailability,
    ) -> _ProductionDefault:
        return _ProductionDefault(
            factory=TrustedContextualDefaultToolFactory.from_registration_surface(
                implementation_ref=implementation_ref,
                declared_behavior_revision=declared_behavior_revision,
                registration=registration,
                binding_factory=lambda: _build_web_search_binding(registration),
            ),
            availability=availability,
        )

    return (
        static_default(
            implementation_ref="builtin/get_today",
            declared_behavior_revision="get-today-handler-1",
            registration=date_registrations["get_today"],
            availability=DefaultProfileAvailability.REQUIRED,
        ),
        static_default(
            implementation_ref="builtin/date_after",
            declared_behavior_revision="date-after-handler-1",
            registration=date_registrations["date_after"],
            availability=DefaultProfileAvailability.REQUIRED,
        ),
        contextual_default(
            implementation_ref="builtin/web_search",
            declared_behavior_revision="bounded-web-search-handler-1",
            registration=web_registrations[WEB_SEARCH_TOOL_ID],
            availability=DefaultProfileAvailability.IF_AVAILABLE,
        ),
        static_default(
            implementation_ref="builtin/web_fetch",
            declared_behavior_revision="bounded-web-fetch-handler-1",
            registration=web_registrations[WEB_FETCH_TOOL_ID],
            availability=DefaultProfileAvailability.IF_AVAILABLE,
        ),
    )


def _build_web_search_binding(
    registration: ToolRegistration,
) -> TrustedContextualBinding:
    try:
        pipeline = freeze_web_search_provider_pipeline()
    except (TypeError, ValueError) as exc:
        raise ToolFactoryUnavailableError(
            "web_search_provider_pipeline_invalid"
        ) from exc
    if not pipeline.has_callable:
        raise ToolFactoryUnavailableError(
            "web_search_provider_pipeline_unavailable"
        )
    fingerprint = pipeline.fingerprint
    return TrustedContextualBinding(
        source=ToolSourceDescriptor(
            kind=WEB_TOOL_SOURCE.kind,
            source_id=f"{WEB_TOOL_SOURCE.source_id}.search-provider-pipeline",
            fingerprint=fingerprint,
            display_name=WEB_TOOL_SOURCE.display_name,
        ),
        handler=pipeline.search,
        effect_profile=registration.effect_profile,
        binding_assertion={
            "provider_pipeline_fingerprint": fingerprint,
            "provider_pipeline": pipeline.descriptor(),
        },
    )


def _index_exact_registrations(
    registrations: tuple[ToolRegistration, ...],
    *,
    expected_ids: tuple[str, ...],
) -> dict[str, ToolRegistration]:
    actual_ids = tuple(item.tool_id for item in registrations)
    if actual_ids != expected_ids or len(set(actual_ids)) != len(actual_ids):
        raise RuntimeError(
            "production default registration family drifted: "
            f"expected={expected_ids!r}, actual={actual_ids!r}"
        )
    return dict(zip(actual_ids, registrations, strict=True))


def build_production_default_catalog_seeds() -> tuple[ToolCatalogSeed, ...]:
    """Return the one stable production seed manifest without live bindings."""

    factory_owned = tuple(
        ToolCatalogSeed(item.factory.definition, item.availability)
        for item in _production_defaults()
    )
    workspace_contextual = tuple(
        ToolCatalogSeed(
            item.definition,
            DefaultProfileAvailability.IF_AVAILABLE,
        )
        for item in build_workspace_discovery_tool_definition_manifest()
    )
    format_contextual = tuple(
        ToolCatalogSeed(
            item.definition,
            DefaultProfileAvailability.IF_AVAILABLE,
        )
        for item in build_format_observation_tool_definition_manifest()
    )
    external_visual_contextual = tuple(
        ToolCatalogSeed(
            item.definition,
            DefaultProfileAvailability.IF_AVAILABLE,
        )
        for item in build_external_visual_analysis_tool_definition_manifest()
    )
    file_retrieval_contextual = tuple(
        ToolCatalogSeed(
            item.definition,
            DefaultProfileAvailability.IF_AVAILABLE,
        )
        for item in build_file_retrieval_tool_definition_manifest()
    )
    file_contextual = tuple(
        ToolCatalogSeed(item.definition, DefaultProfileAvailability.IF_AVAILABLE)
        for item in (
            *file_tool_definition_manifest(),
            *build_file_chunk_tool_definition_manifest(),
            *file_inspection_definition_manifest(),
        )
    )
    workspace_write_contextual = tuple(
        ToolCatalogSeed(
            item.definition,
            DefaultProfileAvailability.IF_AVAILABLE,
        )
        for item in build_workspace_write_tool_definition_manifest()
    )
    history_retrieval_contextual = tuple(
        ToolCatalogSeed(
            item.definition,
            DefaultProfileAvailability.IF_AVAILABLE,
        )
        for item in build_history_retrieval_tool_definition_manifest()
    )
    file_visual_contextual = tuple(
        ToolCatalogSeed(
            item.definition,
            DefaultProfileAvailability.IF_AVAILABLE,
        )
        for item in build_file_visual_tool_definition_manifest()
    )
    findings_contextual = tuple(
        ToolCatalogSeed(
            item.definition,
            DefaultProfileAvailability.IF_AVAILABLE,
        )
        for item in build_execution_findings_tool_definition_manifest()
    )
    tool_history_contextual = tuple(
        ToolCatalogSeed(item.definition, DefaultProfileAvailability.IF_AVAILABLE)
        for item in tool_history_definition_manifest()
    )
    return (
        *factory_owned,
        *workspace_contextual,
        *format_contextual,
        *external_visual_contextual,
        *file_retrieval_contextual,
        *file_contextual,
        *workspace_write_contextual,
        ToolCatalogSeed(build_output_file_definition(), DefaultProfileAvailability.IF_AVAILABLE),
        *history_retrieval_contextual,
        *file_visual_contextual,
        *findings_contextual,
        *tool_history_contextual,
    )


def build_production_default_factory_registry() -> (
    TrustedDefaultFactoryRegistry
):
    """Build the explicit process-local allowlist for production defaults."""

    return TrustedDefaultFactoryRegistry(
        tuple(item.factory for item in _production_defaults())
    )


def bootstrap_production_default_catalog(
    repository: ToolCatalogRepository,
) -> CatalogBootstrapResult:
    """Hard-cut every safely migratable catalog to the canonical L1 profile.

    ``bootstrap`` durably stages definitions.  Reconciliation then uses only
    audited lifecycle/CAS operations to retire an active implementation that
    conflicts with a canonical contract, activate every canonical identity, and
    publish the ordered profile.  A retired canonical identity is irreversible
    and therefore fails closed instead of being silently replaced or revived.
    """

    if not isinstance(repository, ToolCatalogRepository):
        raise TypeError("repository must be a ToolCatalogRepository")
    seeds = build_production_default_catalog_seeds()
    bootstrap = repository.bootstrap(seeds)
    return _reconcile_production_default_profile(
        repository,
        seeds=seeds,
        bootstrap=bootstrap,
    )


def _reconcile_production_default_profile(
    repository: ToolCatalogRepository,
    *,
    seeds: tuple[ToolCatalogSeed, ...],
    bootstrap: CatalogBootstrapResult,
) -> CatalogBootstrapResult:
    changed = bootstrap.changed

    for _ in range(_MAX_RECONCILIATION_ATTEMPTS):
        state = repository.default_resolution_state()
        current = state.current_catalog
        profile = state.default_profile
        entries = {entry.identity: entry for entry in current.entries}
        canonical_entries = []
        for seed in seeds:
            entry = entries.get(seed.definition.identity)
            if (
                entry is None
                or entry.definition_digest != seed.definition.digest
            ):
                raise CatalogConflictError(
                    "canonical production ToolDefinition is absent or drifted"
                )
            canonical_entries.append((seed, entry))
        retired = tuple(
            seed.definition.identity
            for seed, entry in canonical_entries
            if entry.status is CatalogStatus.RETIRED
        )
        if retired:
            raise CatalogConflictError(
                "retired canonical production ToolDefinition cannot be "
                f"reactivated: {retired!r}"
            )

        try:
            canonical_identities = {
                seed.definition.identity for seed in seeds
            }
            for obsolete in current.entries:
                if obsolete.status is CatalogStatus.RETIRED:
                    continue
                if (
                    obsolete.identity.tool_id in _RETIRED_FILE_TOOL_IDS
                    or (
                        obsolete.identity.tool_id == "read_file_chunks"
                        and obsolete.identity not in canonical_identities
                    )
                ):
                    current = repository.set_status(
                        obsolete.identity,
                        CatalogStatus.RETIRED,
                        expected_revision=current.revision,
                        actor="system.bootstrap",
                        reason="retire candidate-based file tool contracts",
                    )
                    changed = True
            for seed, entry in canonical_entries:
                if entry.status is CatalogStatus.ACTIVE:
                    continue
                canonical_identity = seed.definition.identity
                for alternative in current.entries:
                    if (
                        alternative.status is CatalogStatus.ACTIVE
                        and alternative.identity.tool_id
                        == canonical_identity.tool_id
                        and alternative.identity.contract_version
                        == canonical_identity.contract_version
                        and alternative.identity != canonical_identity
                    ):
                        previous_revision = current.revision
                        current = repository.set_status(
                            alternative.identity,
                            CatalogStatus.RETIRED,
                            expected_revision=previous_revision,
                            actor="system.bootstrap",
                            reason=(
                                "retire active implementation superseded by "
                                "canonical L1 production default"
                            ),
                        )
                        changed = changed or current.revision != previous_revision
                previous_revision = current.revision
                published = repository.set_status(
                    canonical_identity,
                    CatalogStatus.ACTIVE,
                    expected_revision=previous_revision,
                    actor="system.bootstrap",
                    reason="activate canonical L1 production default",
                )
                changed = changed or published.revision != previous_revision
                current = published
            if (
                _profile_matches_seeds(profile, seeds)
                and profile.catalog_revision == current.revision
            ):
                return CatalogBootstrapResult(
                    changed,
                    bootstrap.manifest_digest,
                    current,
                    profile,
                )
            published_profile = repository.publish_default_profile(
                tuple(
                    DefaultProfileSelection(
                        seed.definition.identity,
                        seed.availability,
                    )
                    for seed in seeds
                ),
                expected_catalog_revision=current.revision,
                expected_profile_revision=profile.revision,
                actor="system.bootstrap",
                reason="publish canonical L1 production defaults",
            )
            changed = changed or published_profile != profile
            return CatalogBootstrapResult(
                changed,
                bootstrap.manifest_digest,
                current,
                published_profile,
            )
        except CatalogConflictError:
            # Another publisher changed catalog/profile authority after our read.
            # Re-read both atomically and re-enter only a recognized state.
            latest = repository.default_resolution_state()
            if (
                latest.current_catalog.revision == current.revision
                and latest.default_profile.revision == profile.revision
            ):
                raise
            continue
    raise CatalogConflictError(
        "production default profile reconciliation did not converge"
    )


def resolve_production_default_profile(
    repository: ToolCatalogRepository,
) -> ProductionDefaultDefinitionProfile:
    """Atomically read the exact handler-free 26-item production profile."""

    if not isinstance(repository, ToolCatalogRepository):
        raise TypeError("repository must be a ToolCatalogRepository")
    state = repository.default_resolution_state()
    seeds = build_production_default_catalog_seeds()
    profile = state.default_profile
    if not _profile_matches_seeds(profile, seeds):
        raise ProductionDefaultProfileResolutionError(
            "persisted default profile is not the canonical L1 production profile"
        )
    profile_entries = {
        entry.identity: entry for entry in state.profile_catalog.entries
    }
    definitions = {item.identity: item for item in state.definitions}
    resolved: list[ProductionDefaultDefinitionProfileItem] = []
    for profile_item, seed in zip(profile.items, seeds, strict=True):
        profile_entry = profile_entries.get(profile_item.identity)
        if (
            profile_entry is None
            or profile_entry.status is not CatalogStatus.ACTIVE
            or profile_entry.definition_digest != profile_item.definition_digest
        ):
            raise ProductionDefaultProfileResolutionError(
                "canonical default profile was not active at publication"
            )
        definition = definitions.get(profile_item.identity)
        if (
            definition is None
            or definition.digest != profile_item.definition_digest
            or definition != seed.definition
        ):
            raise ProductionDefaultProfileResolutionError(
                "canonical default ToolDefinition is absent or drifted"
            )
        resolved.append(
            ProductionDefaultDefinitionProfileItem(
                ordinal=profile_item.ordinal,
                availability=profile_item.availability,
                definition=definition,
            )
        )
    return ProductionDefaultDefinitionProfile(
        catalog_revision=state.current_catalog.revision,
        catalog_digest=state.current_catalog.digest,
        profile_catalog_revision=state.profile_catalog.revision,
        profile_catalog_digest=state.profile_catalog.digest,
        profile_revision=profile.revision,
        profile_digest=profile.digest,
        items=tuple(resolved),
    )


def _profile_matches_seeds(
    profile: DefaultProfileSnapshot,
    seeds: tuple[ToolCatalogSeed, ...],
) -> bool:
    if len(profile.items) != len(seeds):
        return False
    return all(
        item.ordinal == ordinal
        and item.identity == seed.definition.identity
        and item.definition_digest == seed.definition.digest
        and item.availability is seed.availability
        for ordinal, (item, seed) in enumerate(
            zip(profile.items, seeds, strict=True)
        )
    )


__all__ = [
    "ProductionDefaultDefinitionProfile",
    "ProductionDefaultDefinitionProfileItem",
    "ProductionDefaultProfileResolutionError",
    "bootstrap_production_default_catalog",
    "build_production_default_catalog_seeds",
    "build_production_default_factory_registry",
    "resolve_production_default_profile",
]
