"""Value contracts and strict codecs for persistent Tool Catalog records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

from ....configuration.paths import STATE_DIR, resolve_exact_child_path
from ..model import CatalogStatus
from ..binding import (
    BoundToolRegistration,
    ToolBinding,
    ToolDefinition,
    ToolIdentity,
)
from ...effects import EffectAction, EffectDescriptor, EffectResource


CATALOG_DATABASE_SCHEMA_VERSION = 1
CATALOG_REVISION_SCHEMA = "tool-catalog-revision-v1"
DEFAULT_PROFILE_SCHEMA = "tool-default-profile-v1"
BOOTSTRAP_MANIFEST_SCHEMA = "tool-catalog-bootstrap-manifest-v1"
DEFAULT_TOOL_CATALOG_DATABASE_PATH = STATE_DIR / "tool_catalog.sqlite"

_CATALOG_ID = "default"
_PROFILE_ID = "default"
_SHA256_LENGTH = 64


class CatalogPersistenceError(RuntimeError):
    """The durable Tool Catalog cannot be used safely."""


class CatalogSchemaError(CatalogPersistenceError):
    """The database schema is absent, unsupported, or has drifted."""


class CatalogCorruptionError(CatalogPersistenceError):
    """Persisted catalog values fail their exact contract or integrity checks."""


class DefaultProfileAvailability(StrEnum):
    REQUIRED = "required"
    IF_AVAILABLE = "if_available"


@dataclass(frozen=True)
class DefaultProfileSelection:
    identity: ToolIdentity
    availability: DefaultProfileAvailability = (
        DefaultProfileAvailability.IF_AVAILABLE
    )

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ToolIdentity):
            raise TypeError("profile identity must be a ToolIdentity")
        if not isinstance(self.availability, DefaultProfileAvailability):
            raise TypeError("profile availability must be DefaultProfileAvailability")


@dataclass(frozen=True)
class EmergencyRevocationTarget:
    """Complete execution identity used by the pure revocation matcher."""

    identity: ToolIdentity
    implementation_digest: str
    source_fingerprint: str
    effects: tuple[EffectDescriptor, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ToolIdentity):
            raise TypeError("revocation target identity must be a ToolIdentity")
        _require_sha256(
            self.implementation_digest,
            "revocation target implementation_digest",
        )
        object.__setattr__(
            self,
            "source_fingerprint",
            _nonempty(self.source_fingerprint, "revocation target source_fingerprint"),
        )
        if not isinstance(self.effects, tuple) or not self.effects:
            raise ValueError("revocation target effects must be a non-empty tuple")
        if any(not isinstance(effect, EffectDescriptor) for effect in self.effects):
            raise TypeError(
                "revocation target effects must contain only EffectDescriptor values"
            )

    @classmethod
    def from_definition_and_binding(
        cls,
        definition: ToolDefinition,
        binding: ToolBinding,
    ) -> EmergencyRevocationTarget:
        """Validate the pair as one bound registration before matching it."""

        return cls.from_bound_registration(BoundToolRegistration(definition, binding))

    @classmethod
    def from_bound_registration(
        cls,
        registration: BoundToolRegistration,
    ) -> EmergencyRevocationTarget:
        if not isinstance(registration, BoundToolRegistration):
            raise TypeError("registration must be a BoundToolRegistration")
        fingerprint = registration.binding.source.fingerprint
        if fingerprint is None:
            raise ValueError("bound registration source fingerprint must be complete")
        return cls(
            identity=registration.identity,
            implementation_digest=registration.definition.implementation_digest,
            source_fingerprint=fingerprint,
            effects=registration.effect_profile.effects,
        )


@dataclass(frozen=True)
class EmergencyRevocationSelector:
    """Serializable deny selector; at least one exact dimension is required."""

    tool_id: str | None = None
    contract_version: str | None = None
    implementation_version: str | None = None
    implementation_digest: str | None = None
    source_fingerprint: str | None = None
    effect_resource: EffectResource | None = None
    effect_action: EffectAction | None = None

    def __post_init__(self) -> None:
        for name in (
            "tool_id",
            "contract_version",
            "implementation_version",
            "source_fingerprint",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonempty(value, name))
        if self.implementation_digest is not None:
            _require_sha256(self.implementation_digest, "implementation_digest")
        if self.effect_resource is not None and not isinstance(
            self.effect_resource,
            EffectResource,
        ):
            raise TypeError("effect_resource must be EffectResource")
        if self.effect_action is not None and not isinstance(
            self.effect_action,
            EffectAction,
        ):
            raise TypeError("effect_action must be EffectAction")
        if not any(
            value is not None
            for value in (
                self.tool_id,
                self.contract_version,
                self.implementation_version,
                self.implementation_digest,
                self.source_fingerprint,
                self.effect_resource,
                self.effect_action,
            )
        ):
            raise ValueError("emergency revocation selector must not be empty")

    def descriptor(self) -> dict[str, str | None]:
        return {
            "tool_id": self.tool_id,
            "contract_version": self.contract_version,
            "implementation_version": self.implementation_version,
            "implementation_digest": self.implementation_digest,
            "source_fingerprint": self.source_fingerprint,
            "effect_resource": (
                self.effect_resource.value
                if self.effect_resource is not None
                else None
            ),
            "effect_action": (
                self.effect_action.value if self.effect_action is not None else None
            ),
        }

    @property
    def digest(self) -> str:
        return _sha256_json(self.descriptor())

    def matches(self, target: EmergencyRevocationTarget) -> bool:
        """Conjunctively match one complete target; malformed targets are errors."""

        if not isinstance(target, EmergencyRevocationTarget):
            raise TypeError("target must be a complete EmergencyRevocationTarget")
        identity = target.identity
        scalar_dimensions = (
            (self.tool_id, identity.tool_id),
            (self.contract_version, identity.contract_version),
            (self.implementation_version, identity.implementation_version),
            (self.implementation_digest, target.implementation_digest),
            (self.source_fingerprint, target.source_fingerprint),
        )
        if any(
            expected is not None and expected != actual
            for expected, actual in scalar_dimensions
        ):
            return False
        if self.effect_resource is None and self.effect_action is None:
            return True
        return any(
            (self.effect_resource is None or effect.resource is self.effect_resource)
            and (self.effect_action is None or effect.action is self.effect_action)
            for effect in target.effects
        )


@dataclass(frozen=True)
class EmergencyRevocation:
    revocation_id: str
    selector: EmergencyRevocationSelector
    reason: str
    issued_by: str
    effective_at: datetime
    expires_at: datetime | None
    created_at: datetime
    cleared_at: datetime | None = None
    cleared_by: str | None = None
    clear_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "revocation_id", _nonempty(self.revocation_id, "revocation_id"))
        if not isinstance(self.selector, EmergencyRevocationSelector):
            raise TypeError("selector must be EmergencyRevocationSelector")
        object.__setattr__(self, "reason", _nonempty(self.reason, "reason"))
        object.__setattr__(self, "issued_by", _nonempty(self.issued_by, "issued_by"))
        for name in ("effective_at", "expires_at", "created_at", "cleared_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _utc_datetime(value, name))
        if self.expires_at is not None and self.expires_at <= self.effective_at:
            raise ValueError("expires_at must be later than effective_at")
        clear_fields = (self.cleared_at, self.cleared_by, self.clear_reason)
        if any(value is not None for value in clear_fields) and not all(
            value is not None for value in clear_fields
        ):
            raise ValueError("cleared revocation fields must be present together")
        cleared = self.cleared_at is not None
        if cleared:
            object.__setattr__(self, "cleared_by", _nonempty(self.cleared_by, "cleared_by"))
            object.__setattr__(self, "clear_reason", _nonempty(self.clear_reason, "clear_reason"))
            if self.cleared_at < self.created_at:
                raise ValueError("cleared_at cannot precede created_at")

    def active_at(self, instant: datetime) -> bool:
        selected = _utc_datetime(instant, "instant")
        return (
            self.effective_at <= selected
            and (self.expires_at is None or selected < self.expires_at)
            and (self.cleared_at is None or selected < self.cleared_at)
        )


@dataclass(frozen=True)
class ToolCatalogSeed:
    definition: ToolDefinition
    availability: DefaultProfileAvailability = (
        DefaultProfileAvailability.IF_AVAILABLE
    )

    def __post_init__(self) -> None:
        if not isinstance(self.definition, ToolDefinition):
            raise TypeError("seed definition must be a ToolDefinition")
        if not isinstance(self.availability, DefaultProfileAvailability):
            raise TypeError("seed availability must be DefaultProfileAvailability")


@dataclass(frozen=True, order=True)
class CatalogEntrySnapshot:
    identity: ToolIdentity
    definition_digest: str
    status: CatalogStatus

    def __post_init__(self) -> None:
        _require_sha256(self.definition_digest, "definition_digest")
        if not isinstance(self.status, CatalogStatus):
            raise TypeError("catalog entry status must be CatalogStatus")

    def descriptor(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "definition_digest": self.definition_digest,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class CatalogRevisionSnapshot:
    revision: int
    entries: tuple[CatalogEntrySnapshot, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise ValueError("catalog revision must be a non-negative integer")
        ordered = tuple(sorted(self.entries, key=lambda item: item.identity))
        identities = tuple(item.identity for item in ordered)
        if len(set(identities)) != len(identities):
            raise ValueError("catalog revision contains duplicate identities")
        if self.revision == 0 and ordered:
            raise ValueError("catalog revision zero must be empty")
        object.__setattr__(self, "entries", ordered)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": CATALOG_REVISION_SCHEMA,
            "revision": self.revision,
            "entries": [item.descriptor() for item in self.entries],
        }

    @property
    def digest(self) -> str:
        return _sha256_json(self.descriptor())


@dataclass(frozen=True)
class DefaultProfileItem:
    ordinal: int
    identity: ToolIdentity
    definition_digest: str
    availability: DefaultProfileAvailability

    def __post_init__(self) -> None:
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
        ):
            raise ValueError("default profile ordinal must be non-negative")
        _require_sha256(self.definition_digest, "definition_digest")
        if not isinstance(self.availability, DefaultProfileAvailability):
            raise TypeError(
                "default profile availability must be DefaultProfileAvailability"
            )

    def descriptor(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "identity": self.identity.to_dict(),
            "definition_digest": self.definition_digest,
            "availability": self.availability.value,
        }


@dataclass(frozen=True)
class DefaultProfileSnapshot:
    revision: int
    catalog_revision: int
    items: tuple[DefaultProfileItem, ...]

    def __post_init__(self) -> None:
        for name in ("revision", "catalog_revision"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        ordered = tuple(sorted(self.items, key=lambda item: item.ordinal))
        if tuple(item.ordinal for item in ordered) != tuple(range(len(ordered))):
            raise ValueError("default profile ordinals must be contiguous")
        identities = tuple(item.identity for item in ordered)
        if len(set(identities)) != len(identities):
            raise ValueError("default profile contains duplicate identities")
        object.__setattr__(self, "items", ordered)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": DEFAULT_PROFILE_SCHEMA,
            "profile_id": _PROFILE_ID,
            "revision": self.revision,
            "catalog_revision": self.catalog_revision,
            "items": [item.descriptor() for item in self.items],
        }

    @property
    def digest(self) -> str:
        return _sha256_json(self.descriptor())


@dataclass(frozen=True)
class DefaultCatalogResolutionState:
    """One atomic read used to materialize a new default execution catalog."""

    current_catalog: CatalogRevisionSnapshot
    default_profile: DefaultProfileSnapshot
    profile_catalog: CatalogRevisionSnapshot
    definitions: tuple[ToolDefinition, ...]

    def __post_init__(self) -> None:
        if self.profile_catalog.revision != self.default_profile.catalog_revision:
            raise ValueError(
                "profile catalog revision does not match the default profile"
            )
        if self.current_catalog.revision < self.profile_catalog.revision:
            raise ValueError(
                "current catalog revision cannot precede the profile catalog"
            )
        ordered = tuple(sorted(self.definitions, key=lambda item: item.identity))
        identities = tuple(item.identity for item in ordered)
        if len(set(identities)) != len(identities):
            raise ValueError("resolution state contains duplicate ToolDefinitions")
        object.__setattr__(self, "definitions", ordered)


@dataclass(frozen=True)
class CatalogBootstrapResult:
    changed: bool
    manifest_digest: str
    catalog: CatalogRevisionSnapshot
    default_profile: DefaultProfileSnapshot


@dataclass(frozen=True)
class CatalogAuditEvent:
    event_id: str
    catalog_revision: int
    action: str
    actor: str
    reason: str
    change: Mapping[str, Any]
    occurred_at: str


def _resolve_database_path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.name:
        raise ValueError("Tool Catalog database path is invalid")
    # Normalize platform aliases such as macOS ``/var -> /private/var`` before
    # applying the exact-child guard.  The guard still rejects a planted file
    # symlink at the final catalog path.
    parent = candidate.parent.resolve()
    return resolve_exact_child_path(
        parent,
        candidate.name,
        description="Tool Catalog database path",
    )


def _normalize_seeds(seeds: Sequence[ToolCatalogSeed]) -> tuple[ToolCatalogSeed, ...]:
    if isinstance(seeds, (str, bytes)):
        raise TypeError("seeds must be a sequence of ToolCatalogSeed values")
    normalized = tuple(seeds)
    if not normalized:
        raise ValueError("bootstrap seed manifest must not be empty")
    if any(not isinstance(seed, ToolCatalogSeed) for seed in normalized):
        raise TypeError("seeds must contain only ToolCatalogSeed values")
    identities = tuple(seed.definition.identity for seed in normalized)
    if len(set(identities)) != len(identities):
        raise ValueError("bootstrap seed manifest contains duplicate identities")
    contracts = tuple(
        (identity.tool_id, identity.contract_version) for identity in identities
    )
    if len(set(contracts)) != len(contracts):
        raise ValueError(
            "bootstrap default profile contains multiple implementations of one contract"
        )
    return normalized


def _normalize_profile_selections(
    selections: Sequence[DefaultProfileSelection],
) -> tuple[DefaultProfileSelection, ...]:
    if isinstance(selections, (str, bytes)):
        raise TypeError(
            "selections must be a sequence of DefaultProfileSelection values"
        )
    normalized = tuple(selections)
    if any(not isinstance(item, DefaultProfileSelection) for item in normalized):
        raise TypeError(
            "selections must contain only DefaultProfileSelection values"
        )
    identities = tuple(item.identity for item in normalized)
    if len(set(identities)) != len(identities):
        raise ValueError("default profile contains duplicate identities")
    contracts = tuple(
        (identity.tool_id, identity.contract_version) for identity in identities
    )
    if len(set(contracts)) != len(contracts):
        raise ValueError(
            "default profile contains multiple implementations of one contract"
        )
    return normalized


def _bootstrap_manifest(seeds: tuple[ToolCatalogSeed, ...]) -> dict[str, Any]:
    return {
        "schema_version": BOOTSTRAP_MANIFEST_SCHEMA,
        "entries": [
            {
                "definition": seed.definition.descriptor(),
                "definition_digest": seed.definition.digest,
                "availability": seed.availability.value,
            }
            for seed in seeds
        ],
    }


def _definition_from_row(row: sqlite3.Row) -> ToolDefinition:
    identity = ToolIdentity(
        str(row["tool_id"]),
        str(row["contract_version"]),
        str(row["implementation_version"]),
    )
    descriptor = _load_canonical_json_object(
        row["descriptor_json"],
        "ToolDefinition descriptor",
    )
    try:
        definition = ToolDefinition.from_descriptor(descriptor)
    except (TypeError, ValueError) as exc:
        raise CatalogCorruptionError(
            "persisted ToolDefinition descriptor is invalid"
        ) from exc
    if definition.identity != identity:
        raise CatalogCorruptionError(
            "persisted ToolDefinition identity does not match its row"
        )
    if definition.digest != str(row["definition_digest"]):
        raise CatalogCorruptionError(
            "persisted ToolDefinition digest does not match"
        )
    if (
        definition.implementation_ref != str(row["implementation_ref"])
        or definition.implementation_digest != str(row["implementation_digest"])
    ):
        raise CatalogCorruptionError(
            "persisted ToolDefinition implementation does not match its row"
        )
    _stored_non_negative_integer(
        row["introduced_revision"],
        "ToolDefinition introduced revision",
    )
    return definition


def _entry_snapshot_from_row(row: sqlite3.Row) -> CatalogEntrySnapshot:
    try:
        status = CatalogStatus(str(row["status"]))
    except ValueError as exc:
        raise CatalogCorruptionError("persisted catalog status is invalid") from exc
    digest = str(row["definition_digest"])
    _require_sha256(digest, "persisted definition digest", corruption=True)
    return CatalogEntrySnapshot(
        ToolIdentity(
            str(row["tool_id"]),
            str(row["contract_version"]),
            str(row["implementation_version"]),
        ),
        digest,
        status,
    )


def _profile_item_from_row(row: sqlite3.Row) -> DefaultProfileItem:
    try:
        availability = DefaultProfileAvailability(str(row["availability"]))
    except ValueError as exc:
        raise CatalogCorruptionError(
            "persisted default profile availability is invalid"
        ) from exc
    digest = str(row["definition_digest"])
    _require_sha256(digest, "persisted definition digest", corruption=True)
    return DefaultProfileItem(
        _stored_non_negative_integer(row["ordinal"], "profile ordinal"),
        ToolIdentity(
            str(row["tool_id"]),
            str(row["contract_version"]),
            str(row["implementation_version"]),
        ),
        digest,
        availability,
    )


def _decode_catalog_revision(value: dict[str, Any]) -> CatalogRevisionSnapshot:
    _require_exact_keys(value, {"schema_version", "revision", "entries"}, "catalog revision")
    if value["schema_version"] != CATALOG_REVISION_SCHEMA:
        raise CatalogCorruptionError("unsupported Tool Catalog revision descriptor")
    revision = _json_non_negative_integer(value["revision"], "catalog revision")
    raw_entries = value["entries"]
    if not isinstance(raw_entries, list):
        raise CatalogCorruptionError("catalog revision entries must be an array")
    entries: list[CatalogEntrySnapshot] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise CatalogCorruptionError("catalog revision entry must be an object")
        _require_exact_keys(
            raw,
            {"identity", "definition_digest", "status"},
            "catalog revision entry",
        )
        identity = _decode_identity(raw["identity"])
        digest = _json_sha256(raw["definition_digest"], "definition_digest")
        try:
            status = CatalogStatus(raw["status"])
        except (TypeError, ValueError) as exc:
            raise CatalogCorruptionError("catalog revision status is invalid") from exc
        entries.append(CatalogEntrySnapshot(identity, digest, status))
    snapshot = CatalogRevisionSnapshot(revision, tuple(entries))
    if snapshot.descriptor() != value:
        raise CatalogCorruptionError("Tool Catalog revision descriptor is not canonical")
    return snapshot


def _decode_default_profile(value: dict[str, Any]) -> DefaultProfileSnapshot:
    _require_exact_keys(
        value,
        {"schema_version", "profile_id", "revision", "catalog_revision", "items"},
        "default profile",
    )
    if value["schema_version"] != DEFAULT_PROFILE_SCHEMA or value["profile_id"] != _PROFILE_ID:
        raise CatalogCorruptionError("unsupported default Tool profile descriptor")
    raw_items = value["items"]
    if not isinstance(raw_items, list):
        raise CatalogCorruptionError("default profile items must be an array")
    items: list[DefaultProfileItem] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise CatalogCorruptionError("default profile item must be an object")
        _require_exact_keys(
            raw,
            {"ordinal", "identity", "definition_digest", "availability"},
            "default profile item",
        )
        try:
            availability = DefaultProfileAvailability(raw["availability"])
        except (TypeError, ValueError) as exc:
            raise CatalogCorruptionError(
                "default profile availability is invalid"
            ) from exc
        items.append(
            DefaultProfileItem(
                _json_non_negative_integer(raw["ordinal"], "profile ordinal"),
                _decode_identity(raw["identity"]),
                _json_sha256(raw["definition_digest"], "definition_digest"),
                availability,
            )
        )
    profile = DefaultProfileSnapshot(
        _json_non_negative_integer(value["revision"], "profile revision"),
        _json_non_negative_integer(value["catalog_revision"], "catalog revision"),
        tuple(items),
    )
    if profile.descriptor() != value:
        raise CatalogCorruptionError("default profile descriptor is not canonical")
    return profile


def _decode_identity(value: object) -> ToolIdentity:
    if not isinstance(value, dict):
        raise CatalogCorruptionError("ToolIdentity must be an object")
    _require_exact_keys(
        value,
        {"tool_id", "contract_version", "implementation_version"},
        "ToolIdentity",
    )
    if any(not isinstance(value[key], str) for key in value):
        raise CatalogCorruptionError("ToolIdentity fields must be strings")
    identity = ToolIdentity(
        value["tool_id"],
        value["contract_version"],
        value["implementation_version"],
    )
    if identity.to_dict() != value:
        raise CatalogCorruptionError("ToolIdentity is not canonical")
    return identity


def _audit_event_from_row(row: sqlite3.Row) -> CatalogAuditEvent:
    change = _load_canonical_json_object(row["change_json"], "catalog audit change")
    if _sha256_json(change) != str(row["change_digest"]):
        raise CatalogCorruptionError("catalog audit change digest does not match")
    return CatalogAuditEvent(
        event_id=str(row["event_id"]),
        catalog_revision=_stored_non_negative_integer(
            row["catalog_revision"],
            "audit catalog revision",
        ),
        action=_nonempty(str(row["action"]), "action"),
        actor=_nonempty(str(row["actor"]), "actor"),
        reason=_nonempty(str(row["reason"]), "reason"),
        change=change,
        occurred_at=_nonempty(str(row["occurred_at"]), "occurred_at"),
    )


def _revocation_from_row(row: sqlite3.Row) -> EmergencyRevocation:
    selector_value = _load_canonical_json_object(
        row["selector_json"],
        "emergency revocation selector",
    )
    selector = _decode_revocation_selector(selector_value)
    if selector.digest != str(row["selector_digest"]):
        raise CatalogCorruptionError(
            "emergency revocation selector digest does not match"
        )
    if str(row["disposition"]) != "deny_execution":
        raise CatalogCorruptionError(
            "emergency revocation disposition is unsupported"
        )
    try:
        return EmergencyRevocation(
            revocation_id=str(row["revocation_id"]),
            selector=selector,
            reason=str(row["reason"]),
            issued_by=str(row["issued_by"]),
            effective_at=_parse_datetime(row["effective_at"], "effective_at"),
            expires_at=_parse_optional_datetime(row["expires_at"], "expires_at"),
            created_at=_parse_datetime(row["created_at"], "created_at"),
            cleared_at=_parse_optional_datetime(row["cleared_at"], "cleared_at"),
            cleared_by=(
                str(row["cleared_by"])
                if row["cleared_by"] is not None
                else None
            ),
            clear_reason=(
                str(row["clear_reason"])
                if row["clear_reason"] is not None
                else None
            ),
        )
    except (TypeError, ValueError) as exc:
        raise CatalogCorruptionError(
            "persisted emergency revocation is invalid"
        ) from exc


def _decode_revocation_selector(
    value: dict[str, Any],
) -> EmergencyRevocationSelector:
    _require_exact_keys(
        value,
        {
            "tool_id",
            "contract_version",
            "implementation_version",
            "implementation_digest",
            "source_fingerprint",
            "effect_resource",
            "effect_action",
        },
        "emergency revocation selector",
    )
    for name in (
        "tool_id",
        "contract_version",
        "implementation_version",
        "implementation_digest",
        "source_fingerprint",
        "effect_resource",
        "effect_action",
    ):
        if value[name] is not None and not isinstance(value[name], str):
            raise CatalogCorruptionError(
                f"emergency revocation selector {name} must be text or null"
            )
    try:
        selector = EmergencyRevocationSelector(
            tool_id=value["tool_id"],
            contract_version=value["contract_version"],
            implementation_version=value["implementation_version"],
            implementation_digest=value["implementation_digest"],
            source_fingerprint=value["source_fingerprint"],
            effect_resource=(
                EffectResource(value["effect_resource"])
                if value["effect_resource"] is not None
                else None
            ),
            effect_action=(
                EffectAction(value["effect_action"])
                if value["effect_action"] is not None
                else None
            ),
        )
    except (TypeError, ValueError) as exc:
        raise CatalogCorruptionError(
            "emergency revocation selector is invalid"
        ) from exc
    if selector.descriptor() != value:
        raise CatalogCorruptionError(
            "emergency revocation selector is not canonical"
        )
    return selector


def _load_canonical_json_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise CatalogCorruptionError(f"{label} is not text JSON")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    def reject_constant(constant: str) -> None:
        raise ValueError(f"non-finite JSON number: {constant}")

    try:
        decoded = json.loads(
            value,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CatalogCorruptionError(f"{label} is not strict JSON") from exc
    if not isinstance(decoded, dict):
        raise CatalogCorruptionError(f"{label} is not a JSON object")
    if _canonical_json(decoded) != value:
        raise CatalogCorruptionError(f"{label} is not canonically encoded")
    return decoded


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise CatalogCorruptionError(f"{label} contains missing or unknown fields")


def _json_non_negative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CatalogCorruptionError(f"{label} must be a non-negative integer")
    return value


def _stored_non_negative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CatalogCorruptionError(f"{label} is invalid")
    return value


def _json_sha256(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise CatalogCorruptionError(f"{label} must be a string")
    _require_sha256(value, label, corruption=True)
    return value


def _require_sha256(value: str, label: str, *, corruption: bool = False) -> None:
    if len(value) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in value):
        error = CatalogCorruptionError if corruption else ValueError
        raise error(f"{label} must be a canonical SHA-256 digest")


def _require_non_negative_integer(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must not be empty")
    return value.strip()


def _utc_datetime(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _parse_datetime(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise CatalogCorruptionError(f"persisted {label} must be text")
    try:
        parsed = datetime.fromisoformat(value)
        normalized = _utc_datetime(parsed, label)
    except (TypeError, ValueError) as exc:
        raise CatalogCorruptionError(f"persisted {label} is invalid") from exc
    if normalized.isoformat() != value:
        raise CatalogCorruptionError(f"persisted {label} is not canonical UTC")
    return normalized


def _parse_optional_datetime(value: object, label: str) -> datetime | None:
    return None if value is None else _parse_datetime(value, label)


def _identity_values(identity: ToolIdentity) -> tuple[str, str, str]:
    return (
        identity.tool_id,
        identity.contract_version,
        identity.implementation_version,
    )


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("catalog descriptor must contain only finite JSON") from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_text(_canonical_json(value))
