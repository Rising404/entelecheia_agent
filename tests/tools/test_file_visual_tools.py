from __future__ import annotations

import pytest

from personagraph.tools.effects import (
    DataEgress,
    EffectAction,
    EffectResource,
    EffectScopeKind,
    Idempotency,
    Reversibility,
)
from personagraph.tools.execution import ResolvedInvocation, ToolExecutor
from personagraph.tools.visual.file_visual_tools import (
    FILE_VISUAL_IMPLEMENTATION_VERSION,
    LIST_FILE_VISUALS_CONTRACT_VERSION,
    LIST_FILE_VISUALS_TOOL_ID,
    MAX_LIST_PAGE_FILTERS,
    MAX_VISUALS_PER_PAGE,
    MAX_VISUALS_PER_READ,
    READ_FILE_VISUALS_CONTRACT_VERSION,
    READ_FILE_VISUALS_TOOL_ID,
    build_list_file_visuals_registration,
    build_read_file_visuals_registration,
)


_FILE_ID = "file_123"
_FILE_VERSION_ID = "file_version_123"
_VISUAL_REF = "visualref_" + "b" * 32
_CURSOR = "visualcursor_" + "c" * 32
_EFFECT_SCOPE = "file-corpus:session-1"
def test_file_visual_frozen_registration_identities_are_stable() -> None:
    listed = build_list_file_visuals_registration(
        handler=_list_result,
        effect_scope=_EFFECT_SCOPE,
    )
    read = build_read_file_visuals_registration(
        handler=_read_result,
        effect_scope=_EFFECT_SCOPE,
        sends_externally=False,
    )
    assert FILE_VISUAL_IMPLEMENTATION_VERSION == "3"
    assert (
        listed.contract_version,
        listed.implementation_version,
        listed.source.source_id,
    ) == (
        "file-visual-list-v2",
        "3",
        "personagraph.files.visuals",
    )
    assert (
        read.contract_version,
        read.implementation_version,
        read.source.source_id,
    ) == (
        "file-visual-read-v2",
        "3",
        "personagraph.files.visuals",
    )
    assert listed.source.fingerprint is not None
    assert read.source.fingerprint is not None


def _mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {
            *value.keys(),
            *(key for child in value.values() for key in _mapping_keys(child)),
        }
    if isinstance(value, list):
        return {key for child in value for key in _mapping_keys(child)}
    return set()


def _list_result(_payload: dict[str, object]) -> dict[str, object]:
    return {
        "contract_version": LIST_FILE_VISUALS_CONTRACT_VERSION,
        "file_id": _FILE_ID, "file_version_id": _FILE_VERSION_ID,
        "status": "ready",
        "visuals": [
            {
                "visual_unit_id": _VISUAL_REF,
                "pages": [2],
                "kind": "figure",
                "allowed_purposes": ["chart", "caption", "general"],
                "default_purpose": "general",
            }
        ],
        "next_cursor": _CURSOR,
        "reason_code": None,
    }


def _read_result(_payload: dict[str, object]) -> dict[str, object]:
    return {
        "contract_version": READ_FILE_VISUALS_CONTRACT_VERSION,
        "results": [
            {
                "file_id": _FILE_ID, "file_version_id": _FILE_VERSION_ID,
                "visual_unit_id": _VISUAL_REF,
                "pages": [2],
                "kind": "figure",
                "purpose": "chart",
                "detail": "standard",
                "region": "detected",
                "status": "completed",
                "observation": "The chart rises from 10 to 18.",
                "reason_code": None,
            }
        ],
    }


def _read_request() -> dict[str, object]:
    return {
        "requests": [
            {
                "file_id": _FILE_ID, "file_version_id": _FILE_VERSION_ID,
                "visual_unit_id": _VISUAL_REF,
                "purpose": "chart",
                "detail": "standard",
                "region": "detected",
            }
        ]
    }


def test_file_visual_contracts_have_bounded_public_schemas() -> None:
    listed = build_list_file_visuals_registration(
        handler=_list_result,
        effect_scope=_EFFECT_SCOPE,
    )
    read = build_read_file_visuals_registration(
        handler=_read_result,
        effect_scope=_EFFECT_SCOPE,
        sends_externally=False,
    )

    assert (listed.tool_id, listed.contract_version) == (
        LIST_FILE_VISUALS_TOOL_ID,
        LIST_FILE_VISUALS_CONTRACT_VERSION,
    )
    assert (read.tool_id, read.contract_version) == (
        READ_FILE_VISUALS_TOOL_ID,
        READ_FILE_VISUALS_CONTRACT_VERSION,
    )
    assert set(listed.spec.input_schema["properties"]) == {
        "file_id", "file_version_id",
        "pages",
        "cursor",
        "limit",
    }
    request_schema = read.spec.input_schema["properties"]["requests"]
    assert request_schema["maxItems"] == MAX_VISUALS_PER_READ
    assert set(request_schema["items"]["properties"]) == {
        "file_id", "file_version_id",
        "visual_unit_id",
        "purpose",
        "question",
        "detail",
        "region",
    }
    assert listed.spec.output_schema["properties"]["visuals"]["maxItems"] == (
        MAX_VISUALS_PER_PAGE
    )
    read_result_schema = read.spec.output_schema["properties"]["results"]["items"]
    assert {"observation_id", "uncertainty"} <= set(
        read_result_schema["properties"]
    )
    assert {"observation_id", "uncertainty"}.isdisjoint(
        read_result_schema["required"]
    )

    schema_keys = _mapping_keys(
        [listed.spec.to_dict(), read.spec.to_dict()]
    )
    assert {
        "path",
        "stored_rel_path",
        "attachment_id",
        "content_hash",
        "document_id",
        "document_version",
        "document_ref",
        "version_ref",
        "alias",
        "visual_alias",
        "provider_unit_id",
        "unit_id",
    }.isdisjoint(schema_keys)


def test_list_is_pure_session_scoped_metadata_read() -> None:
    registration = build_list_file_visuals_registration(
        handler=_list_result,
        effect_scope=_EFFECT_SCOPE,
    )

    assert len(registration.effect_profile.effects) == 1
    effect = registration.effect_profile.effects[0]
    assert (
        effect.resource,
        effect.action,
        effect.scope_kind,
        effect.default_scope,
        effect.data_egress,
        effect.idempotency,
        effect.reversibility,
    ) == (
        EffectResource.FILESYSTEM,
        EffectAction.READ,
        EffectScopeKind.SESSION,
        _EFFECT_SCOPE,
        DataEgress.METADATA,
        Idempotency.IDEMPOTENT,
        Reversibility.REVERSIBLE,
    )


def test_external_visual_read_declares_file_transmit_and_deduplicated_state() -> None:
    registration = build_read_file_visuals_registration(
        handler=_read_result,
        effect_scope=_EFFECT_SCOPE,
        sends_externally=True,
    )

    filesystem, network, runtime_state = registration.effect_profile.effects
    assert (
        filesystem.resource,
        filesystem.action,
        filesystem.scope_kind,
        filesystem.data_egress,
    ) == (
        EffectResource.FILESYSTEM,
        EffectAction.READ,
        EffectScopeKind.SESSION,
        DataEgress.CONTENT,
    )
    assert (
        network.resource,
        network.action,
        network.scope_kind,
        network.default_scope,
        network.data_egress,
        network.idempotency,
        network.reversibility,
    ) == (
        EffectResource.NETWORK,
        EffectAction.TRANSMIT,
        EffectScopeKind.SESSION,
        _EFFECT_SCOPE,
        DataEgress.CONTENT,
        Idempotency.NOT_IDEMPOTENT,
        Reversibility.IRREVERSIBLE,
    )
    assert (
        runtime_state.resource,
        runtime_state.action,
        runtime_state.scope_kind,
        runtime_state.default_scope,
        runtime_state.data_egress,
        runtime_state.idempotency,
        runtime_state.reversibility,
    ) == (
        EffectResource.RUNTIME_STATE,
        EffectAction.UPDATE,
        EffectScopeKind.SESSION,
        _EFFECT_SCOPE,
        DataEgress.NONE,
        Idempotency.DEDUPLICATED,
        Reversibility.UNKNOWN,
    )
    assert registration.execution_profile.max_transparent_retries == 0


def test_local_visual_read_does_not_claim_network_transmission() -> None:
    registration = build_read_file_visuals_registration(
        handler=_read_result,
        effect_scope=_EFFECT_SCOPE,
        sends_externally=False,
    )

    assert [effect.resource for effect in registration.effect_profile.effects] == [
        EffectResource.FILESYSTEM
    ]
    assert registration.execution_profile.max_transparent_retries == 1


def test_file_visual_contracts_validate_public_results() -> None:
    listed = build_list_file_visuals_registration(
        handler=_list_result,
        effect_scope=_EFFECT_SCOPE,
    )
    read = build_read_file_visuals_registration(
        handler=_read_result,
        effect_scope=_EFFECT_SCOPE,
        sends_externally=False,
    )
    executor = ToolExecutor()

    list_outcome = executor.execute(
        ResolvedInvocation(
            listed,
            {"file_id": _FILE_ID, "file_version_id": _FILE_VERSION_ID, "limit": 1},
        )
    )
    read_outcome = executor.execute(ResolvedInvocation(read, _read_request()))

    assert list_outcome.error is None
    assert read_outcome.error is None
    assert list_outcome.result["visuals"][0]["visual_unit_id"] == _VISUAL_REF
    assert read_outcome.result["results"][0]["observation"].startswith("The chart")


def test_list_accepts_one_to_eight_unique_positive_page_filters() -> None:
    observed: list[dict[str, object]] = []
    registration = build_list_file_visuals_registration(
        handler=lambda payload: observed.append(payload) or _list_result(payload),
        effect_scope=_EFFECT_SCOPE,
    )
    pages = list(range(1, MAX_LIST_PAGE_FILTERS + 1))

    outcome = ToolExecutor().execute(
        ResolvedInvocation(
            registration,
            {"file_id": _FILE_ID, "file_version_id": _FILE_VERSION_ID, "pages": pages},
        )
    )

    assert outcome.error is None
    assert observed == [{"file_id": _FILE_ID, "file_version_id": _FILE_VERSION_ID, "pages": pages}]


@pytest.mark.parametrize(
    "pages",
    (
        [],
        list(range(1, MAX_LIST_PAGE_FILTERS + 2)),
        [1, 1],
        [0],
        [-1],
        [True],
        ["2"],
    ),
    ids=(
        "empty",
        "too-many",
        "duplicate",
        "zero",
        "negative",
        "boolean",
        "string",
    ),
)
def test_list_rejects_invalid_page_filters_before_handler(pages: list[object]) -> None:
    observed: list[dict[str, object]] = []
    registration = build_list_file_visuals_registration(
        handler=lambda payload: observed.append(payload) or _list_result(payload),
        effect_scope=_EFFECT_SCOPE,
    )

    outcome = ToolExecutor().execute(
        ResolvedInvocation(
            registration,
            {"file_id": _FILE_ID, "file_version_id": _FILE_VERSION_ID, "pages": pages},
        )
    )

    assert outcome.error is not None
    assert outcome.error.code == "invalid_tool_input"
    assert observed == []


def test_read_output_accepts_optional_observation_identity_and_uncertainty() -> None:
    def enriched_result(payload: dict[str, object]) -> dict[str, object]:
        result = _read_result(payload)
        result["results"][0].update(
            {
                "observation_id": "observation_01",
                "uncertainty": 0.25,
            }
        )
        return result

    registration = build_read_file_visuals_registration(
        handler=enriched_result,
        effect_scope=_EFFECT_SCOPE,
        sends_externally=False,
    )

    outcome = ToolExecutor().execute(
        ResolvedInvocation(registration, _read_request())
    )

    assert outcome.error is None
    assert outcome.result["results"][0]["observation_id"] == "observation_01"
    assert outcome.result["results"][0]["uncertainty"] == 0.25


@pytest.mark.parametrize("private_key", ["path", "attachment_id", "visual_alias"])
def test_private_read_inputs_are_rejected_before_handler(
    private_key: str,
) -> None:
    observed: list[dict[str, object]] = []
    registration = build_read_file_visuals_registration(
        handler=lambda payload: observed.append(payload) or _read_result(payload),
        effect_scope=_EFFECT_SCOPE,
        sends_externally=False,
    )
    payload = _read_request()
    payload[private_key] = "private"

    outcome = ToolExecutor().execute(ResolvedInvocation(registration, payload))

    assert outcome.error is not None
    assert outcome.error.code == "invalid_tool_input"
    assert observed == []


def test_private_output_fields_are_rejected_at_the_contract_boundary() -> None:
    def leaking_result(payload: dict[str, object]) -> dict[str, object]:
        result = _list_result(payload)
        result["visuals"][0]["provider_unit_id"] = "raw-unit-7"
        return result

    registration = build_list_file_visuals_registration(
        handler=leaking_result,
        effect_scope=_EFFECT_SCOPE,
    )

    outcome = ToolExecutor().execute(
        ResolvedInvocation(registration, {"file_id": _FILE_ID, "file_version_id": _FILE_VERSION_ID})
    )

    assert outcome.error is not None
    assert outcome.error.code == "invalid_tool_output"


def test_strict_cardinality_and_ref_patterns_are_enforced() -> None:
    read = build_read_file_visuals_registration(
        handler=_read_result,
        effect_scope=_EFFECT_SCOPE,
        sends_externally=False,
    )
    too_many = _read_request()["requests"] * (MAX_VISUALS_PER_READ + 1)

    oversized = ToolExecutor().execute(
        ResolvedInvocation(read, {"requests": too_many})
    )
    guessed_ref = ToolExecutor().execute(
        ResolvedInvocation(
            read,
            {
                "requests": [
                    {
                        **_read_request()["requests"][0],
                        "visual_unit_id": "",
                    }
                ]
            },
        )
    )

    assert oversized.error is not None
    assert oversized.error.code == "invalid_tool_input"
    assert guessed_ref.error is not None
    assert guessed_ref.error.code == "invalid_tool_input"


@pytest.mark.parametrize("bad_scope", ["", "has spaces", "x" * 257])
def test_builders_reject_unbounded_effect_scopes(bad_scope: str) -> None:
    with pytest.raises(ValueError, match="effect_scope"):
        build_list_file_visuals_registration(
            handler=_list_result,
            effect_scope=bad_scope,
        )


def test_read_builder_requires_an_explicit_boolean_egress_fact() -> None:
    with pytest.raises(ValueError, match="sends_externally"):
        build_read_file_visuals_registration(
            handler=_read_result,
            effect_scope=_EFFECT_SCOPE,
            sends_externally="no",  # type: ignore[arg-type]
        )
