"""Strict codec for a handler-free, frozen bound Tool Catalog payload.

This payload is an input to a future Attempt execution snapshot, not that
complete snapshot: it intentionally has no default-profile provenance, policy
version, Attempt identity, or creation time.  The digest returned by
:func:`encode_frozen_bound_catalog` is kept outside the JSON payload so a
future store can persist the canonical payload and its trusted digest as
separate columns.  This module detects corruption; it does not provide
authenticity by itself.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from ..model import (
    CatalogEntry,
    CatalogSnapshot,
    CatalogStatus,
    ToolKey,
)
from ..binding import (
    BoundToolRegistration,
    FrozenToolBinding,
    ToolDefinition,
    ToolIdentity,
    _require_effect_binding,
)


FROZEN_BOUND_CATALOG_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class FrozenBoundCatalogError(ValueError):
    """A frozen bound catalog is malformed, non-canonical, or corrupt."""

    code = "frozen_bound_catalog_invalid"


@dataclass(frozen=True)
class FrozenBoundCatalogEntry:
    """One frozen catalog entry sufficient for a future exact re-bind check."""

    key: ToolKey
    status: CatalogStatus
    created_revision: int
    updated_revision: int
    definition: ToolDefinition
    definition_digest: str
    binding: FrozenToolBinding
    binding_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, ToolKey):
            raise TypeError("snapshot entry key must be a ToolKey")
        _require_canonical_text(self.key.tool_id, "entry tool_id")
        _require_canonical_text(
            self.key.contract_version,
            "entry contract_version",
        )
        if not isinstance(self.status, CatalogStatus):
            raise TypeError("snapshot entry status must be a CatalogStatus")
        created = _positive_integer(self.created_revision, "created_revision")
        updated = _positive_integer(self.updated_revision, "updated_revision")
        if created > updated:
            raise ValueError("created_revision cannot exceed updated_revision")
        if not isinstance(self.definition, ToolDefinition):
            raise TypeError("snapshot entry definition must be a ToolDefinition")
        _require_sha256(self.definition_digest, "definition_digest")
        if not isinstance(self.binding, FrozenToolBinding):
            raise TypeError("snapshot entry binding must be a FrozenToolBinding")
        _require_sha256(self.binding_digest, "binding_digest")

        identity = self.definition.identity
        if (self.key.tool_id, self.key.contract_version) != (
            identity.tool_id,
            identity.contract_version,
        ):
            raise ValueError("snapshot entry key does not match definition identity")
        if self.definition.digest != self.definition_digest:
            raise ValueError("snapshot entry definition digest does not match")
        if self.binding.identity != identity:
            raise ValueError("snapshot entry binding identity does not match definition")
        if self.binding.definition_digest != self.definition_digest:
            raise ValueError(
                "snapshot entry binding definition digest does not match"
            )
        if self.binding.digest != self.binding_digest:
            raise ValueError("snapshot entry binding digest does not match")
        _require_effect_binding(
            self.definition.effect_template,
            self.binding.effect_profile,
        )

    @classmethod
    def from_catalog_entry(
        cls,
        entry: CatalogEntry,
    ) -> "FrozenBoundCatalogEntry":
        registration = entry.registration
        if not isinstance(registration, BoundToolRegistration):
            raise FrozenBoundCatalogError(
                "frozen bound catalogs require only BoundToolRegistration entries"
            )
        return cls(
            key=entry.key,
            status=entry.status,
            created_revision=entry.created_revision,
            updated_revision=entry.updated_revision,
            definition=registration.definition,
            definition_digest=registration.definition_digest,
            binding=FrozenToolBinding.from_binding(registration.binding),
            binding_digest=registration.binding_digest,
        )

    @property
    def identity(self) -> ToolIdentity:
        return self.definition.identity

    def descriptor(self) -> dict[str, Any]:
        return {
            "tool_id": self.key.tool_id,
            "contract_version": self.key.contract_version,
            "status": self.status.value,
            "created_revision": self.created_revision,
            "updated_revision": self.updated_revision,
            "registration": {
                "definition": self.definition.descriptor(),
                "definition_digest": self.definition_digest,
                "binding": self.binding.descriptor(),
                "binding_digest": self.binding_digest,
            },
        }


@dataclass(frozen=True)
class FrozenBoundCatalog:
    """Handler-free catalog payload for later Attempt-level composition."""

    revision: int
    entries: tuple[FrozenBoundCatalogEntry, ...]

    def __post_init__(self) -> None:
        revision = _non_negative_integer(self.revision, "catalog revision")
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, FrozenBoundCatalogEntry)
            for entry in self.entries
        ):
            raise TypeError(
                "frozen bound catalog entries must be FrozenBoundCatalogEntry values"
            )
        keys = tuple(entry.key for entry in self.entries)
        identities = tuple(entry.identity for entry in self.entries)
        if keys != tuple(sorted(keys)):
            raise ValueError("frozen bound catalog entries are not canonically ordered")
        if len(set(keys)) != len(keys):
            raise ValueError("frozen bound catalog contains duplicate tool keys")
        if len(set(identities)) != len(identities):
            raise ValueError("frozen bound catalog contains duplicate identities")
        if revision == 0 and self.entries:
            raise ValueError("catalog revision zero must have no entries")
        if any(entry.updated_revision > revision for entry in self.entries):
            raise ValueError("snapshot entry revision exceeds catalog revision")

    @classmethod
    def from_catalog_snapshot(
        cls,
        snapshot: CatalogSnapshot,
    ) -> "FrozenBoundCatalog":
        if not isinstance(snapshot, CatalogSnapshot):
            raise TypeError("snapshot must be a CatalogSnapshot")
        return cls(
            revision=snapshot.revision,
            entries=tuple(
                FrozenBoundCatalogEntry.from_catalog_entry(entry)
                for entry in snapshot.entries
            ),
        )

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": FROZEN_BOUND_CATALOG_SCHEMA_VERSION,
            "revision": self.revision,
            "entries": [entry.descriptor() for entry in self.entries],
        }

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.descriptor())

    @property
    def digest(self) -> str:
        return _sha256_text(self.canonical_json)


@dataclass(frozen=True)
class EncodedFrozenBoundCatalog:
    """Canonical payload plus a separately persisted integrity anchor.

    Construction validates only field shape.  Integrity is established by
    :func:`decode_frozen_bound_catalog` against its expected trusted digest.
    """

    canonical_json: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_json, str):
            raise TypeError("canonical_json must be a string")
        _require_sha256(self.sha256, "sha256")


def encode_frozen_bound_catalog(
    snapshot: CatalogSnapshot | FrozenBoundCatalog,
) -> EncodedFrozenBoundCatalog:
    """Freeze and encode a catalog snapshot without retaining any handler."""

    if isinstance(snapshot, CatalogSnapshot):
        frozen = FrozenBoundCatalog.from_catalog_snapshot(snapshot)
    elif isinstance(snapshot, FrozenBoundCatalog):
        frozen = snapshot
    else:
        raise TypeError(
            "snapshot must be a CatalogSnapshot or FrozenBoundCatalog"
        )
    return EncodedFrozenBoundCatalog(
        canonical_json=frozen.canonical_json,
        sha256=frozen.digest,
    )


def decode_frozen_bound_catalog(
    *,
    canonical_json: str,
    expected_sha256: str,
) -> FrozenBoundCatalog:
    """Validate a trusted hash and decode an exact handler-free snapshot."""

    try:
        if not isinstance(canonical_json, str):
            raise TypeError("canonical_json must be a string")
        _require_sha256(expected_sha256, "expected_sha256")
        if _sha256_text(canonical_json) != expected_sha256:
            raise FrozenBoundCatalogError(
                "frozen bound catalog digest does not match"
            )
        descriptor = _load_strict_json_object(canonical_json)
        if _canonical_json(descriptor) != canonical_json:
            raise FrozenBoundCatalogError(
                "frozen bound catalog JSON is not canonical"
            )
        snapshot = _decode_snapshot_descriptor(descriptor)
        if snapshot.descriptor() != descriptor:
            raise FrozenBoundCatalogError(
                "frozen bound catalog descriptor is not canonical"
            )
        return snapshot
    except FrozenBoundCatalogError:
        raise
    except (KeyError, TypeError, ValueError, RecursionError) as exc:
        raise FrozenBoundCatalogError(
            "frozen bound catalog is invalid"
        ) from exc


def _decode_snapshot_descriptor(value: dict[str, Any]) -> FrozenBoundCatalog:
    _require_exact_keys(
        value,
        {"schema_version", "revision", "entries"},
        "frozen bound catalog",
    )
    schema_version = value["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != FROZEN_BOUND_CATALOG_SCHEMA_VERSION
    ):
        raise ValueError("unsupported frozen bound catalog schema version")
    raw_entries = value["entries"]
    if not isinstance(raw_entries, list):
        raise TypeError("frozen bound catalog entries must be an array")
    return FrozenBoundCatalog(
        revision=_non_negative_integer(value["revision"], "catalog revision"),
        entries=tuple(_decode_entry(item) for item in raw_entries),
    )


def _decode_entry(value: object) -> FrozenBoundCatalogEntry:
    if not isinstance(value, dict):
        raise TypeError("frozen bound catalog entry must be a JSON object")
    _require_exact_keys(
        value,
        {
            "tool_id",
            "contract_version",
            "status",
            "created_revision",
            "updated_revision",
            "registration",
        },
        "frozen bound catalog entry",
    )
    tool_id = _required_canonical_text(value, "tool_id")
    contract_version = _required_canonical_text(value, "contract_version")
    status_value = value["status"]
    if not isinstance(status_value, str):
        raise TypeError("snapshot entry status must be a string")
    try:
        status = CatalogStatus(status_value)
    except ValueError as exc:
        raise ValueError("snapshot entry status is invalid") from exc

    registration = _required_object(value, "registration")
    _require_exact_keys(
        registration,
        {"definition", "definition_digest", "binding", "binding_digest"},
        "frozen bound catalog registration",
    )
    definition = ToolDefinition.from_descriptor(
        _required_object(registration, "definition")
    )
    binding = FrozenToolBinding.from_descriptor(
        _required_object(registration, "binding")
    )
    return FrozenBoundCatalogEntry(
        key=ToolKey(tool_id, contract_version),
        status=status,
        created_revision=_positive_integer(
            value["created_revision"],
            "created_revision",
        ),
        updated_revision=_positive_integer(
            value["updated_revision"],
            "updated_revision",
        ),
        definition=definition,
        definition_digest=_required_sha256(
            registration,
            "definition_digest",
        ),
        binding=binding,
        binding_digest=_required_sha256(registration, "binding_digest"),
    )


def _load_strict_json_object(payload: str) -> dict[str, Any]:
    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, TypeError, ValueError, RecursionError) as exc:
        raise FrozenBoundCatalogError(
            "frozen bound catalog is not strict JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise FrozenBoundCatalogError(
            "frozen bound catalog must be a JSON object"
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


def _required_sha256(value: dict[str, Any], key: str) -> str:
    selected = value[key]
    _require_sha256(selected, key)
    return selected


def _required_canonical_text(value: dict[str, Any], key: str) -> str:
    selected = _required_string(value, key)
    _require_canonical_text(selected, key)
    return selected


def _require_canonical_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be canonical non-empty text")
    return value


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _non_negative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_sha256(value: object, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a canonical SHA-256 digest")


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
        raise ValueError("snapshot descriptor must contain only finite JSON") from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


__all__ = [
    "FROZEN_BOUND_CATALOG_SCHEMA_VERSION",
    "EncodedFrozenBoundCatalog",
    "FrozenBoundCatalog",
    "FrozenBoundCatalogEntry",
    "FrozenBoundCatalogError",
    "decode_frozen_bound_catalog",
    "encode_frozen_bound_catalog",
]
