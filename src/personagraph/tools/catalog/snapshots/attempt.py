"""Strict Attempt-owned envelope for one frozen bound Tool Catalog.

The nested bound catalog remains owned by :mod:`.bound`.
This module adds only the provenance needed to bind that catalog to one
Attempt.  It detects corruption through canonical encoding and a separately
persisted SHA-256 anchor; it does not establish authenticity by itself.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from ..binding import ToolIdentity
from ...policy import TOOL_POLICY_VERSION
from .bound import (
    FrozenBoundCatalog,
    FrozenBoundCatalogError,
    decode_frozen_bound_catalog,
)


ATTEMPT_TOOL_CATALOG_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class FrozenAttemptToolCatalogError(ValueError):
    """An Attempt Tool Catalog envelope is malformed or corrupt."""

    code = "frozen_attempt_tool_catalog_invalid"


class AttemptToolBindingOwnerKind(StrEnum):
    """The frozen resolver lane that owns one live handler binding."""

    TRUSTED_FACTORY = "trusted_factory"
    CONTEXTUAL_CANDIDATE = "contextual_candidate"


@dataclass(frozen=True, order=True)
class AttemptToolBindingOwner:
    """Persisted owner lane for one exact frozen tool identity."""

    identity: ToolIdentity
    kind: AttemptToolBindingOwnerKind

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ToolIdentity):
            raise TypeError("binding owner identity must be a ToolIdentity")
        if not isinstance(self.kind, AttemptToolBindingOwnerKind):
            raise TypeError("binding owner kind must be AttemptToolBindingOwnerKind")

    def descriptor(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "kind": self.kind.value,
        }


@dataclass(frozen=True)
class AttemptToolCatalogProvenance:
    """Durable global-catalog facts used to materialize one Attempt."""

    source_catalog_revision: int
    source_catalog_digest: str
    default_profile_revision: int
    default_profile_digest: str
    profile_catalog_revision: int

    def __post_init__(self) -> None:
        source_revision = _non_negative_integer(
            self.source_catalog_revision,
            "source_catalog_revision",
        )
        _require_sha256(self.source_catalog_digest, "source_catalog_digest")
        _non_negative_integer(
            self.default_profile_revision,
            "default_profile_revision",
        )
        _require_sha256(self.default_profile_digest, "default_profile_digest")
        profile_revision = _non_negative_integer(
            self.profile_catalog_revision,
            "profile_catalog_revision",
        )
        if profile_revision > source_revision:
            raise ValueError(
                "profile_catalog_revision cannot exceed source_catalog_revision"
            )

    def descriptor(self) -> dict[str, object]:
        return {
            "source_catalog_revision": self.source_catalog_revision,
            "source_catalog_digest": self.source_catalog_digest,
            "default_profile_revision": self.default_profile_revision,
            "default_profile_digest": self.default_profile_digest,
            "profile_catalog_revision": self.profile_catalog_revision,
        }


@dataclass(frozen=True)
class FrozenAttemptToolCatalog:
    """Handler-free Tool Catalog frozen for exactly one Attempt."""

    attempt_id: str
    created_at: datetime
    provenance: AttemptToolCatalogProvenance
    frozen_catalog: FrozenBoundCatalog
    exposure_order: tuple[ToolIdentity, ...]
    binding_owners: tuple[AttemptToolBindingOwner, ...]
    policy_version: str = TOOL_POLICY_VERSION

    def __post_init__(self) -> None:
        _require_canonical_text(self.attempt_id, "attempt_id")
        object.__setattr__(
            self,
            "created_at",
            _utc_datetime(self.created_at, "created_at"),
        )
        if not isinstance(self.provenance, AttemptToolCatalogProvenance):
            raise TypeError(
                "provenance must be AttemptToolCatalogProvenance"
            )
        if not isinstance(self.frozen_catalog, FrozenBoundCatalog):
            raise TypeError("frozen_catalog must be a FrozenBoundCatalog")
        if not isinstance(self.exposure_order, tuple) or any(
            not isinstance(identity, ToolIdentity)
            for identity in self.exposure_order
        ):
            raise TypeError(
                "exposure_order must contain only ToolIdentity values"
            )
        if len(set(self.exposure_order)) != len(self.exposure_order):
            raise ValueError("exposure_order contains duplicate identities")
        frozen_identities = tuple(
            entry.identity for entry in self.frozen_catalog.entries
        )
        if set(self.exposure_order) != set(frozen_identities):
            raise ValueError(
                "exposure_order identities must exactly match frozen catalog entries"
            )
        if not isinstance(self.binding_owners, tuple) or any(
            not isinstance(owner, AttemptToolBindingOwner)
            for owner in self.binding_owners
        ):
            raise TypeError(
                "binding_owners must contain only AttemptToolBindingOwner values"
            )
        owner_identities = tuple(owner.identity for owner in self.binding_owners)
        if owner_identities != tuple(sorted(owner_identities)):
            raise ValueError("binding_owners are not canonically ordered")
        if len(set(owner_identities)) != len(owner_identities):
            raise ValueError("binding_owners contain duplicate identities")
        if set(owner_identities) != set(frozen_identities):
            raise ValueError(
                "binding owner identities must exactly match frozen catalog entries"
            )
        _require_canonical_text(self.policy_version, "policy_version")

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_version": ATTEMPT_TOOL_CATALOG_SCHEMA_VERSION,
            "attempt_id": self.attempt_id,
            "created_at": self.created_at.isoformat(),
            "policy_version": self.policy_version,
            "provenance": self.provenance.descriptor(),
            "exposure_order": [
                identity.to_dict() for identity in self.exposure_order
            ],
            "binding_owners": [
                owner.descriptor() for owner in self.binding_owners
            ],
            "frozen_catalog": self.frozen_catalog.descriptor(),
        }

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.descriptor())

    @property
    def digest(self) -> str:
        return _sha256_text(self.canonical_json)


@dataclass(frozen=True)
class EncodedFrozenAttemptToolCatalog:
    """Canonical outer payload plus its separately persisted digest."""

    canonical_json: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_json, str):
            raise TypeError("canonical_json must be a string")
        _require_sha256(self.sha256, "sha256")


def encode_frozen_attempt_tool_catalog(
    snapshot: FrozenAttemptToolCatalog,
) -> EncodedFrozenAttemptToolCatalog:
    """Encode an already materialized Attempt Tool Catalog envelope."""

    if not isinstance(snapshot, FrozenAttemptToolCatalog):
        raise TypeError("snapshot must be a FrozenAttemptToolCatalog")
    return EncodedFrozenAttemptToolCatalog(
        canonical_json=snapshot.canonical_json,
        sha256=snapshot.digest,
    )


def decode_frozen_attempt_tool_catalog(
    *,
    canonical_json: str,
    expected_sha256: str,
) -> FrozenAttemptToolCatalog:
    """Verify an outer digest and decode the exact frozen envelope."""

    try:
        if not isinstance(canonical_json, str):
            raise TypeError("canonical_json must be a string")
        _require_sha256(expected_sha256, "expected_sha256")
        if _sha256_text(canonical_json) != expected_sha256:
            raise FrozenAttemptToolCatalogError(
                "frozen Attempt Tool Catalog digest does not match"
            )
        descriptor = _load_strict_json_object(canonical_json)
        if _canonical_json(descriptor) != canonical_json:
            raise FrozenAttemptToolCatalogError(
                "frozen Attempt Tool Catalog JSON is not canonical"
            )
        snapshot = _decode_snapshot_descriptor(descriptor)
        if snapshot.descriptor() != descriptor:
            raise FrozenAttemptToolCatalogError(
                "frozen Attempt Tool Catalog descriptor is not canonical"
            )
        return snapshot
    except FrozenAttemptToolCatalogError:
        raise
    except (
        FrozenBoundCatalogError,
        KeyError,
        TypeError,
        ValueError,
        RecursionError,
    ) as exc:
        raise FrozenAttemptToolCatalogError(
            "frozen Attempt Tool Catalog is invalid"
        ) from exc


def _decode_snapshot_descriptor(
    value: dict[str, Any],
) -> FrozenAttemptToolCatalog:
    _require_exact_keys(
        value,
        {
            "schema_version",
            "attempt_id",
            "created_at",
            "policy_version",
            "provenance",
            "exposure_order",
            "binding_owners",
            "frozen_catalog",
        },
        "frozen Attempt Tool Catalog",
    )
    schema_version = value["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != ATTEMPT_TOOL_CATALOG_SCHEMA_VERSION
    ):
        raise ValueError("unsupported Attempt Tool Catalog schema version")

    raw_catalog = _required_object(value, "frozen_catalog")
    nested_json = _canonical_json(raw_catalog)
    frozen_catalog = decode_frozen_bound_catalog(
        canonical_json=nested_json,
        expected_sha256=_sha256_text(nested_json),
    )
    raw_order = value["exposure_order"]
    if not isinstance(raw_order, list):
        raise TypeError("exposure_order must be a JSON array")
    raw_owners = value["binding_owners"]
    if not isinstance(raw_owners, list):
        raise TypeError("binding_owners must be a JSON array")
    return FrozenAttemptToolCatalog(
        attempt_id=_required_canonical_text(value, "attempt_id"),
        created_at=_parse_canonical_utc_datetime(value, "created_at"),
        policy_version=_required_canonical_text(value, "policy_version"),
        provenance=_decode_provenance(
            _required_object(value, "provenance")
        ),
        frozen_catalog=frozen_catalog,
        exposure_order=tuple(_decode_identity(item) for item in raw_order),
        binding_owners=tuple(_decode_binding_owner(item) for item in raw_owners),
    )


def _decode_provenance(
    value: dict[str, Any],
) -> AttemptToolCatalogProvenance:
    _require_exact_keys(
        value,
        {
            "source_catalog_revision",
            "source_catalog_digest",
            "default_profile_revision",
            "default_profile_digest",
            "profile_catalog_revision",
        },
        "Attempt Tool Catalog provenance",
    )
    return AttemptToolCatalogProvenance(
        source_catalog_revision=_non_negative_integer(
            value["source_catalog_revision"],
            "source_catalog_revision",
        ),
        source_catalog_digest=_required_sha256(
            value,
            "source_catalog_digest",
        ),
        default_profile_revision=_non_negative_integer(
            value["default_profile_revision"],
            "default_profile_revision",
        ),
        default_profile_digest=_required_sha256(
            value,
            "default_profile_digest",
        ),
        profile_catalog_revision=_non_negative_integer(
            value["profile_catalog_revision"],
            "profile_catalog_revision",
        ),
    )


def _decode_identity(value: object) -> ToolIdentity:
    if not isinstance(value, dict):
        raise TypeError("exposure identity must be a JSON object")
    _require_exact_keys(
        value,
        {"tool_id", "contract_version", "implementation_version"},
        "exposure identity",
    )
    identity = ToolIdentity(
        _required_canonical_text(value, "tool_id"),
        _required_canonical_text(value, "contract_version"),
        _required_canonical_text(value, "implementation_version"),
    )
    if identity.to_dict() != value:
        raise ValueError("exposure identity is not canonical")
    return identity


def _decode_binding_owner(value: object) -> AttemptToolBindingOwner:
    if not isinstance(value, dict):
        raise TypeError("binding owner must be a JSON object")
    _require_exact_keys(
        value,
        {"identity", "kind"},
        "binding owner",
    )
    raw_kind = _required_string(value, "kind")
    try:
        kind = AttemptToolBindingOwnerKind(raw_kind)
    except ValueError as exc:
        raise ValueError("binding owner kind is invalid") from exc
    owner = AttemptToolBindingOwner(
        identity=_decode_identity(_required_object(value, "identity")),
        kind=kind,
    )
    if owner.descriptor() != value:
        raise ValueError("binding owner descriptor is not canonical")
    return owner


def _parse_canonical_utc_datetime(
    value: dict[str, Any],
    key: str,
) -> datetime:
    raw = _required_string(value, key)
    try:
        parsed = datetime.fromisoformat(raw)
        normalized = _utc_datetime(parsed, key)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a canonical UTC datetime") from exc
    if normalized.isoformat() != raw:
        raise ValueError(f"{key} must be a canonical UTC datetime")
    return normalized


def _load_strict_json_object(payload: str) -> dict[str, Any]:
    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError, RecursionError) as exc:
        raise FrozenAttemptToolCatalogError(
            "frozen Attempt Tool Catalog is not strict JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise FrozenAttemptToolCatalogError(
            "frozen Attempt Tool Catalog must be a JSON object"
        )
    return decoded


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"{label} fields do not match the canonical contract: "
            f"missing={missing!r}, unknown={unknown!r}"
        )


def _required_object(value: dict[str, Any], key: str) -> dict[str, Any]:
    selected = value[key]
    if not isinstance(selected, dict):
        raise TypeError(f"{key} must be a JSON object")
    return selected


def _required_string(value: dict[str, Any], key: str) -> str:
    selected = value[key]
    if not isinstance(selected, str):
        raise TypeError(f"{key} must be a string")
    return selected


def _required_canonical_text(value: dict[str, Any], key: str) -> str:
    selected = _required_string(value, key)
    return _require_canonical_text(selected, key)


def _required_sha256(value: dict[str, Any], key: str) -> str:
    selected = value[key]
    _require_sha256(selected, key)
    return selected


def _require_canonical_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be canonical non-empty text")
    return value


def _non_negative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_sha256(value: object, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a canonical SHA-256 digest")


def _utc_datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


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
        raise ValueError(
            "Attempt Tool Catalog descriptor must contain only finite JSON"
        ) from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "ATTEMPT_TOOL_CATALOG_SCHEMA_VERSION",
    "AttemptToolBindingOwnerKind",
    'AttemptToolBindingOwner',
    'AttemptToolCatalogProvenance',
    'EncodedFrozenAttemptToolCatalog',
    "FrozenAttemptToolCatalogError",
    'FrozenAttemptToolCatalog',
    "decode_frozen_attempt_tool_catalog",
    "encode_frozen_attempt_tool_catalog",
]
