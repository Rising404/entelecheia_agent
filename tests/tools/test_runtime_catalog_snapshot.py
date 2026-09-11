from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from personagraph.tools.catalog import CatalogStatus, ToolCatalog
from personagraph.tools.catalog.binding import (
    BoundToolRegistration,
    FrozenToolBinding,
    ToolBinding,
    ToolDefinition,
)
from personagraph.tools.contracts import (
    ToolSourceDescriptor,
    ToolSourceKind,
    ToolSpec,
)
from personagraph.tools.effects import (
    EffectAction,
    EffectDescriptor,
    EffectResource,
    EffectScopeKind,
    ToolEffectProfile,
)
from personagraph.tools.registration import (
    ToolExecutionProfile,
    ToolRegistration,
)
from personagraph.tools.catalog.snapshots.bound import (
    FROZEN_BOUND_CATALOG_SCHEMA_VERSION,
    FrozenBoundCatalog,
    FrozenBoundCatalogError,
    decode_frozen_bound_catalog,
    encode_frozen_bound_catalog,
)


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


def _effects(*, scope: str = "*") -> ToolEffectProfile:
    return ToolEffectProfile(
        (
            EffectDescriptor(
                resource=EffectResource.FILESYSTEM,
                action=EffectAction.READ,
                scope_kind=EffectScopeKind.WORKSPACE,
                default_scope=scope,
            ),
        )
    )


def _definition(tool_id: str) -> ToolDefinition:
    return ToolDefinition(
        spec=ToolSpec(
            tool_id=tool_id,
            contract_version=f"{tool_id}-v1",
            name=f"Read {tool_id}",
            description="Read a bounded workspace value.",
            input_schema={
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
            catalog_tags=("read",),
        ),
        implementation_version="1",
        implementation_ref=f"personagraph.tools.{tool_id}",
        implementation_digest="a" * 64,
        effect_template=_effects(),
        execution_profile=ToolExecutionProfile(
            default_timeout_s=5,
            hard_timeout_s=10,
            max_output_bytes=4096,
            max_transparent_retries=0,
        ),
    )


def _binding(definition: ToolDefinition, *, scope: str) -> ToolBinding:
    return ToolBinding(
        identity=definition.identity,
        definition_digest=definition.digest,
        source=ToolSourceDescriptor(
            kind=ToolSourceKind.LOCAL,
            source_id=f"personagraph.workspace.{definition.identity.tool_id}",
            fingerprint=f"fingerprint:{definition.identity.tool_id}",
        ),
        handler=lambda payload: {"value": payload["key"]},
        effect_profile=_effects(scope=scope),
        binding_assertion={
            "scope_sha256": "b" * 64,
            "resources": {"workspace": scope},
        },
    )


def _bound(tool_id: str) -> BoundToolRegistration:
    definition = _definition(tool_id)
    return BoundToolRegistration(
        definition,
        _binding(definition, scope=f"/workspace/{tool_id}"),
    )


def _two_entry_snapshot():
    catalog = ToolCatalog()
    catalog.register(_bound("zeta"))
    catalog.register(_bound("alpha"))
    return catalog.snapshot()


def _encoded_payload() -> tuple[dict[str, object], str, str]:
    encoded = encode_frozen_bound_catalog(_two_entry_snapshot())
    return json.loads(encoded.canonical_json), encoded.canonical_json, encoded.sha256


def _reseal(value: object) -> tuple[str, str]:
    payload = _canonical_json(value)
    return payload, _sha256(payload)


def test_bound_catalog_round_trip_is_canonical_and_handler_free() -> None:
    live_snapshot = _two_entry_snapshot()

    encoded = encode_frozen_bound_catalog(live_snapshot)
    restored = decode_frozen_bound_catalog(
        canonical_json=encoded.canonical_json,
        expected_sha256=encoded.sha256,
    )

    assert restored.revision == live_snapshot.revision == 2
    assert restored.canonical_json == encoded.canonical_json
    assert restored.digest == encoded.sha256
    assert restored.descriptor()["schema_version"] == (
        FROZEN_BOUND_CATALOG_SCHEMA_VERSION
    )
    assert [entry.key.tool_id for entry in restored.entries] == ["alpha", "zeta"]
    assert [entry.descriptor() for entry in restored.entries] == (
        live_snapshot.to_bound_descriptor()["entries"]
    )
    assert all(not hasattr(entry.binding, "handler") for entry in restored.entries)
    assert "handler" not in encoded.canonical_json


def test_empty_catalog_round_trip_is_valid() -> None:
    frozen = FrozenBoundCatalog.from_catalog_snapshot(ToolCatalog().snapshot())

    encoded = encode_frozen_bound_catalog(frozen)
    restored = decode_frozen_bound_catalog(
        canonical_json=encoded.canonical_json,
        expected_sha256=encoded.sha256,
    )

    assert restored.revision == 0
    assert restored.entries == ()


def test_freeze_rejects_a_legacy_registration_that_has_no_binding_identity() -> None:
    definition = _definition("legacy")
    binding = _binding(definition, scope="/workspace/legacy")
    catalog = ToolCatalog()
    catalog.register(
        ToolRegistration(
            spec=definition.spec,
            implementation_version=definition.implementation_version,
            source=binding.source,
            handler=binding.handler,
            effect_profile=binding.effect_profile,
            execution_profile=definition.execution_profile,
        )
    )

    with pytest.raises(FrozenBoundCatalogError, match="BoundToolRegistration"):
        encode_frozen_bound_catalog(catalog.snapshot())


def test_frozen_binding_is_the_single_exact_live_rebind_guard() -> None:
    definition = _definition("alpha")
    live = _binding(definition, scope="/workspace/alpha")
    frozen = FrozenToolBinding.from_binding(live)

    assert frozen.require_exact_live_binding(live) is live
    with pytest.raises(ValueError, match="does not match"):
        frozen.require_exact_live_binding(
            _binding(definition, scope="/workspace/other")
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "definition_digest",
        "binding_digest",
        "created_revision",
        "updated_revision",
        "status",
        "entry_identity",
        "definition_identity",
        "binding_identity",
    ),
)
def test_trusted_digest_rejects_all_bound_entry_tampering(mutation: str) -> None:
    payload, _, trusted_digest = _encoded_payload()
    entry = payload["entries"][0]
    assert isinstance(entry, dict)
    registration = entry["registration"]
    assert isinstance(registration, dict)

    if mutation == "definition_digest":
        registration["definition_digest"] = "c" * 64
    elif mutation == "binding_digest":
        registration["binding_digest"] = "c" * 64
    elif mutation == "created_revision":
        entry["created_revision"] = 1
    elif mutation == "updated_revision":
        entry["updated_revision"] = 1
    elif mutation == "status":
        entry["status"] = CatalogStatus.DISABLED.value
    elif mutation == "entry_identity":
        entry["tool_id"] = "changed"
    elif mutation == "definition_identity":
        definition = registration["definition"]
        assert isinstance(definition, dict)
        identity = definition["identity"]
        assert isinstance(identity, dict)
        identity["tool_id"] = "changed"
    else:
        binding = registration["binding"]
        assert isinstance(binding, dict)
        identity = binding["identity"]
        assert isinstance(identity, dict)
        identity["tool_id"] = "changed"

    tampered_json = _canonical_json(payload)
    with pytest.raises(FrozenBoundCatalogError, match="digest does not match"):
        decode_frozen_bound_catalog(
            canonical_json=tampered_json,
            expected_sha256=trusted_digest,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "definition_digest",
        "binding_digest",
        "entry_identity",
        "definition_identity",
        "binding_identity",
        "created_revision",
        "updated_revision",
        "status_type",
        "status_value",
    ),
)
def test_resealed_structural_corruption_still_fails_closed(mutation: str) -> None:
    payload, _, _ = _encoded_payload()
    entry = payload["entries"][0]
    assert isinstance(entry, dict)
    registration = entry["registration"]
    assert isinstance(registration, dict)

    if mutation == "definition_digest":
        registration["definition_digest"] = "c" * 64
    elif mutation == "binding_digest":
        registration["binding_digest"] = "c" * 64
    elif mutation == "entry_identity":
        entry["tool_id"] = "changed"
    elif mutation == "definition_identity":
        definition = registration["definition"]
        assert isinstance(definition, dict)
        identity = definition["identity"]
        assert isinstance(identity, dict)
        identity["tool_id"] = "changed"
    elif mutation == "binding_identity":
        binding = registration["binding"]
        assert isinstance(binding, dict)
        identity = binding["identity"]
        assert isinstance(identity, dict)
        identity["tool_id"] = "changed"
    elif mutation == "created_revision":
        entry["created_revision"] = 0
    elif mutation == "updated_revision":
        entry["updated_revision"] = 3
    elif mutation == "status_type":
        entry["status"] = 1
    else:
        entry["status"] = "unknown"

    tampered_json, tampered_digest = _reseal(payload)
    with pytest.raises(FrozenBoundCatalogError):
        decode_frozen_bound_catalog(
            canonical_json=tampered_json,
            expected_sha256=tampered_digest,
        )


@pytest.mark.parametrize(
    "mutation",
    ("unknown_top_field", "unknown_entry_field", "bool_schema", "bool_revision"),
)
def test_codec_rejects_unknown_fields_and_bool_as_integer(mutation: str) -> None:
    payload, _, _ = _encoded_payload()
    if mutation == "unknown_top_field":
        payload["unexpected"] = None
    elif mutation == "unknown_entry_field":
        entry = payload["entries"][0]
        assert isinstance(entry, dict)
        entry["unexpected"] = None
    elif mutation == "bool_schema":
        payload["schema_version"] = True
    else:
        payload["revision"] = True

    tampered_json, tampered_digest = _reseal(payload)
    with pytest.raises(FrozenBoundCatalogError):
        decode_frozen_bound_catalog(
            canonical_json=tampered_json,
            expected_sha256=tampered_digest,
        )


def test_codec_rejects_noncanonical_and_duplicate_key_json() -> None:
    _, canonical, _ = _encoded_payload()
    noncanonical = canonical + "\n"
    with pytest.raises(FrozenBoundCatalogError, match="not canonical"):
        decode_frozen_bound_catalog(
            canonical_json=noncanonical,
            expected_sha256=_sha256(noncanonical),
        )

    duplicate_key = canonical.replace(
        '"revision":2',
        '"revision":2,"revision":2',
        1,
    )
    assert duplicate_key != canonical
    with pytest.raises(FrozenBoundCatalogError, match="strict JSON"):
        decode_frozen_bound_catalog(
            canonical_json=duplicate_key,
            expected_sha256=_sha256(duplicate_key),
        )


def test_codec_rejects_reordered_or_duplicate_entries_even_when_resealed() -> None:
    payload, _, _ = _encoded_payload()
    reversed_payload = deepcopy(payload)
    reversed_entries = reversed_payload["entries"]
    assert isinstance(reversed_entries, list)
    reversed_entries.reverse()

    duplicate_payload = deepcopy(payload)
    duplicate_entries = duplicate_payload["entries"]
    assert isinstance(duplicate_entries, list)
    duplicate_entries.append(deepcopy(duplicate_entries[0]))

    for invalid in (reversed_payload, duplicate_payload):
        tampered_json, tampered_digest = _reseal(invalid)
        with pytest.raises(FrozenBoundCatalogError):
            decode_frozen_bound_catalog(
                canonical_json=tampered_json,
                expected_sha256=tampered_digest,
            )


def test_codec_rejects_nested_binding_schema_drift_when_resealed() -> None:
    payload, _, _ = _encoded_payload()
    entry = payload["entries"][0]
    assert isinstance(entry, dict)
    registration = entry["registration"]
    assert isinstance(registration, dict)
    binding = registration["binding"]
    assert isinstance(binding, dict)
    source = binding["source"]
    assert isinstance(source, dict)
    source["runtime_provider"] = "forbidden"

    tampered_json, tampered_digest = _reseal(payload)
    with pytest.raises(FrozenBoundCatalogError):
        decode_frozen_bound_catalog(
            canonical_json=tampered_json,
            expected_sha256=tampered_digest,
        )
