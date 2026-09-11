from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from personagraph.tools.catalog.snapshots.attempt import (
    ATTEMPT_TOOL_CATALOG_SCHEMA_VERSION,
    AttemptToolBindingOwnerKind,
    AttemptToolBindingOwner,
    AttemptToolCatalogProvenance,
    FrozenAttemptToolCatalogError,
    FrozenAttemptToolCatalog,
    decode_frozen_attempt_tool_catalog,
    encode_frozen_attempt_tool_catalog,
)
from personagraph.tools.catalog import ToolCatalog
from personagraph.tools.catalog.binding import ToolIdentity
from personagraph.tools.policy import TOOL_POLICY_VERSION
from personagraph.tools.catalog.snapshots.bound import FrozenBoundCatalog
from personagraph.tools.time.date_tools import build_date_tool_registrations
from personagraph.tools.catalog.trusted_factories import TrustedStaticDefaultToolFactory


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _frozen_catalog() -> tuple[FrozenBoundCatalog, tuple[ToolIdentity, ...]]:
    catalog = ToolCatalog()
    exposure_order: list[ToolIdentity] = []
    for registration in build_date_tool_registrations():
        factory = TrustedStaticDefaultToolFactory.from_registration(
            implementation_ref=f"builtin/test/{registration.tool_id}",
            declared_behavior_revision=f"{registration.tool_id}-test-handler-1",
            registration=registration,
        )
        catalog.register(factory.bind(factory.definition))
        exposure_order.append(factory.identity)
    return (
        FrozenBoundCatalog.from_catalog_snapshot(catalog.snapshot()),
        tuple(exposure_order),
    )


def _snapshot(
    *,
    policy_version: str = TOOL_POLICY_VERSION,
) -> FrozenAttemptToolCatalog:
    frozen_catalog, exposure_order = _frozen_catalog()
    return FrozenAttemptToolCatalog(
        attempt_id="attempt-17",
        created_at=datetime(
            2026,
            9,
            3,
            12,
            34,
            56,
            123456,
            tzinfo=timezone.utc,
        ),
        policy_version=policy_version,
        provenance=AttemptToolCatalogProvenance(
            # The durable control-plane revision and this Attempt-local
            # ToolCatalog build revision intentionally have distinct namespaces.
            source_catalog_revision=41,
            source_catalog_digest="c" * 64,
            default_profile_revision=7,
            default_profile_digest="d" * 64,
            profile_catalog_revision=17,
        ),
        frozen_catalog=frozen_catalog,
        exposure_order=exposure_order,
        binding_owners=tuple(
            AttemptToolBindingOwner(
                identity=identity,
                kind=AttemptToolBindingOwnerKind.TRUSTED_FACTORY,
            )
            for identity in sorted(exposure_order)
        ),
    )


def _payload() -> tuple[dict[str, object], str, str]:
    encoded = encode_frozen_attempt_tool_catalog(_snapshot())
    return json.loads(encoded.canonical_json), encoded.canonical_json, encoded.sha256


def _reseal(value: object) -> tuple[str, str]:
    canonical = _canonical_json(value)
    return canonical, _sha256(canonical)


def test_attempt_catalog_round_trip_preserves_owner_and_profile_order() -> None:
    snapshot = _snapshot()

    encoded = encode_frozen_attempt_tool_catalog(snapshot)
    restored = decode_frozen_attempt_tool_catalog(
        canonical_json=encoded.canonical_json,
        expected_sha256=encoded.sha256,
    )

    assert restored == snapshot
    assert restored.canonical_json == encoded.canonical_json
    assert restored.digest == encoded.sha256
    assert restored.policy_version == TOOL_POLICY_VERSION
    assert restored.descriptor() == {
        "schema_version": ATTEMPT_TOOL_CATALOG_SCHEMA_VERSION,
        "attempt_id": "attempt-17",
        "created_at": "2026-09-03T12:34:56.123456+00:00",
        "policy_version": TOOL_POLICY_VERSION,
        "provenance": snapshot.provenance.descriptor(),
        "exposure_order": [
            identity.to_dict() for identity in snapshot.exposure_order
        ],
        "binding_owners": [
            owner.descriptor() for owner in snapshot.binding_owners
        ],
        "frozen_catalog": snapshot.frozen_catalog.descriptor(),
    }
    assert [identity.tool_id for identity in restored.exposure_order] == [
        "get_today",
        "date_after",
    ]
    assert [entry.identity.tool_id for entry in restored.frozen_catalog.entries] == [
        "date_after",
        "get_today",
    ]
    assert restored.provenance.source_catalog_revision == 41
    assert restored.frozen_catalog.revision == 2
    assert "handler" not in encoded.canonical_json
    assert "unavailable" not in encoded.canonical_json


def test_typed_timestamp_is_normalized_to_utc_before_encoding() -> None:
    snapshot = replace(
        _snapshot(),
        created_at=datetime(
            2026,
            9,
            3,
            20,
            34,
            56,
            123456,
            tzinfo=timezone(timedelta(hours=8)),
        ),
    )

    assert snapshot.created_at == datetime(
        2026,
        9,
        3,
        12,
        34,
        56,
        123456,
        tzinfo=timezone.utc,
    )
    assert snapshot.descriptor()["created_at"] == (
        "2026-09-03T12:34:56.123456+00:00"
    )


@pytest.mark.parametrize(
    "field",
    (
        "attempt_id",
        "created_at",
        "policy_version",
        "source_catalog_digest",
        "default_profile_digest",
        "exposure_order",
        "binding_owners",
        "frozen_catalog",
    ),
)
def test_outer_integrity_anchor_rejects_tampering(field: str) -> None:
    payload, _, trusted_digest = _payload()
    if field in {"source_catalog_digest", "default_profile_digest"}:
        provenance = payload["provenance"]
        assert isinstance(provenance, dict)
        provenance[field] = "e" * 64
    elif field == "exposure_order":
        order = payload[field]
        assert isinstance(order, list)
        order.reverse()
    elif field == "binding_owners":
        owners = payload[field]
        assert isinstance(owners, list)
        owner = owners[0]
        assert isinstance(owner, dict)
        owner["kind"] = AttemptToolBindingOwnerKind.CONTEXTUAL_CANDIDATE.value
    elif field == "frozen_catalog":
        catalog = payload[field]
        assert isinstance(catalog, dict)
        catalog["revision"] = 3
    else:
        payload[field] = f"changed-{field}"

    tampered = _canonical_json(payload)
    with pytest.raises(FrozenAttemptToolCatalogError, match="digest does not match"):
        decode_frozen_attempt_tool_catalog(
            canonical_json=tampered,
            expected_sha256=trusted_digest,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown_top_field",
        "missing_top_field",
        "bool_schema_version",
        "bool_source_revision",
        "uppercase_catalog_digest",
        "noncanonical_created_at",
        "noncanonical_attempt_id",
        "noncanonical_identity",
        "unknown_provenance_field",
        "unavailable_diagnostics",
        "unknown_owner_field",
        "unknown_owner_kind",
        "noncanonical_owner_order",
        "nested_unknown_field",
        "nested_digest_drift",
        "nested_catalog_revision_drift",
        "future_profile_catalog",
    ),
)
def test_resealed_structural_or_schema_drift_fails_closed(mutation: str) -> None:
    payload, _, _ = _payload()
    provenance = payload["provenance"]
    order = payload["exposure_order"]
    catalog = payload["frozen_catalog"]
    owners = payload["binding_owners"]
    assert isinstance(provenance, dict)
    assert isinstance(order, list)
    assert isinstance(catalog, dict)
    assert isinstance(owners, list)

    if mutation == "unknown_top_field":
        payload["unexpected"] = None
    elif mutation == "missing_top_field":
        del payload["policy_version"]
    elif mutation == "bool_schema_version":
        payload["schema_version"] = True
    elif mutation == "bool_source_revision":
        provenance["source_catalog_revision"] = True
    elif mutation == "uppercase_catalog_digest":
        provenance["source_catalog_digest"] = "C" * 64
    elif mutation == "noncanonical_created_at":
        payload["created_at"] = "2026-09-03T12:34:56.123456Z"
    elif mutation == "noncanonical_attempt_id":
        payload["attempt_id"] = " attempt-17"
    elif mutation == "noncanonical_identity":
        identity = order[0]
        assert isinstance(identity, dict)
        identity["tool_id"] = f" {identity['tool_id']}"
    elif mutation == "unknown_provenance_field":
        provenance["unexpected"] = None
    elif mutation == "unavailable_diagnostics":
        payload["unavailable"] = [{"reason": "not authoritative"}]
    elif mutation == "unknown_owner_field":
        owner = owners[0]
        assert isinstance(owner, dict)
        owner["unexpected"] = None
    elif mutation == "unknown_owner_kind":
        owner = owners[0]
        assert isinstance(owner, dict)
        owner["kind"] = "fallback"
    elif mutation == "noncanonical_owner_order":
        owners.reverse()
    elif mutation == "nested_unknown_field":
        catalog["unexpected"] = None
    elif mutation == "nested_digest_drift":
        entries = catalog["entries"]
        assert isinstance(entries, list)
        entry = entries[0]
        assert isinstance(entry, dict)
        registration = entry["registration"]
        assert isinstance(registration, dict)
        registration["binding_digest"] = "e" * 64
    elif mutation == "nested_catalog_revision_drift":
        catalog["revision"] = 1
    else:
        provenance["profile_catalog_revision"] = 42

    tampered, digest = _reseal(payload)
    with pytest.raises(FrozenAttemptToolCatalogError):
        decode_frozen_attempt_tool_catalog(
            canonical_json=tampered,
            expected_sha256=digest,
        )


def test_codec_rejects_noncanonical_and_duplicate_key_json() -> None:
    _, canonical, _ = _payload()
    noncanonical = canonical + "\n"
    with pytest.raises(FrozenAttemptToolCatalogError, match="not canonical"):
        decode_frozen_attempt_tool_catalog(
            canonical_json=noncanonical,
            expected_sha256=_sha256(noncanonical),
        )

    policy_field = f'"policy_version":"{TOOL_POLICY_VERSION}"'
    duplicate = canonical.replace(policy_field, f"{policy_field},{policy_field}", 1)
    assert duplicate != canonical
    with pytest.raises(FrozenAttemptToolCatalogError, match="strict JSON"):
        decode_frozen_attempt_tool_catalog(
            canonical_json=duplicate,
            expected_sha256=_sha256(duplicate),
        )


def test_exposure_order_must_be_a_duplicate_free_exact_identity_set() -> None:
    snapshot = _snapshot()
    first, second = snapshot.exposure_order

    with pytest.raises(ValueError, match="duplicate"):
        replace(snapshot, exposure_order=(first, first))
    with pytest.raises(ValueError, match="exactly match"):
        replace(snapshot, exposure_order=(first,))
    with pytest.raises(ValueError, match="exactly match"):
        replace(
            snapshot,
            exposure_order=(
                first,
                second,
                ToolIdentity("other", "1", "1"),
            ),
        )
    with pytest.raises(TypeError, match="ToolIdentity"):
        replace(snapshot, exposure_order=[first, second])  # type: ignore[arg-type]


def test_binding_owners_must_be_canonical_and_exactly_cover_frozen_entries() -> None:
    snapshot = _snapshot()
    first, second = snapshot.binding_owners

    with pytest.raises(ValueError, match="canonically ordered"):
        replace(snapshot, binding_owners=(second, first))
    with pytest.raises(ValueError, match="duplicate"):
        replace(snapshot, binding_owners=(first, first))
    with pytest.raises(ValueError, match="exactly match"):
        replace(snapshot, binding_owners=(first,))
    with pytest.raises(TypeError, match='AttemptToolBindingOwner'):
        replace(snapshot, binding_owners=[first, second])  # type: ignore[arg-type]


def test_historical_nonempty_policy_version_is_decodable_but_not_upgraded() -> None:
    historical = _snapshot(policy_version="0.9.0")

    encoded = encode_frozen_attempt_tool_catalog(historical)
    restored = decode_frozen_attempt_tool_catalog(
        canonical_json=encoded.canonical_json,
        expected_sha256=encoded.sha256,
    )

    assert restored.policy_version == "0.9.0"
    assert restored.policy_version != TOOL_POLICY_VERSION


@pytest.mark.parametrize("policy_version", ("", "  ", " 0.9.0"))
def test_policy_version_must_remain_canonical_nonempty_text(
    policy_version: str,
) -> None:
    with pytest.raises(ValueError, match="policy_version"):
        _snapshot(policy_version=policy_version)
