from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from personagraph.tools.catalog.binding import BoundToolRegistration, ToolBinding
from personagraph.tools.contracts import ToolSourceKind
from personagraph.tools.effects import DataEgress
from personagraph.tools.retrieval.history_retrieval_catalog import (
    HISTORY_RETRIEVAL_BINDING_ASSERTION_SCHEMA,
    HistoryRetrievalBindingFacts,
    HistoryRetrievalCatalogError,
    build_history_retrieval_tool_bindings,
    build_history_retrieval_tool_definition_manifest,
    derive_history_retrieval_source_fingerprint,
)
from personagraph.tools.retrieval.retrieval_tools import (
    HistoryRetrievalScope,
    build_retrieve_history_registration,
)


_EFFECT_SCOPE = "history-corpus:scope-authority-1"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _facts(**changes: str) -> HistoryRetrievalBindingFacts:
    values = {
        "effect_scope_sha256": _sha256(_EFFECT_SCOPE),
        "session_sha256": _sha256("private-session-1"),
        "turn_cutoff_sha256": _sha256("assistant_turn_idx:7"),
        "retrieval_generation_sha256": _sha256("private-generation-1"),
        "allowed_scopes_sha256": _sha256("current_session"),
        "retrieval_service_sha256": _sha256("private-service-1"),
        "retrieval_authority_sha256": _sha256("private-authority-1"),
    }
    values.update(changes)
    return HistoryRetrievalBindingFacts(**values)


def _registration(
    *,
    facts: HistoryRetrievalBindingFacts | None = None,
    effect_scope: str = _EFFECT_SCOPE,
):
    selected_facts = _facts() if facts is None else facts
    return build_retrieve_history_registration(
        handler=lambda payload: {"echo": dict(payload)},
        effect_scope=effect_scope,
        source_fingerprint=derive_history_retrieval_source_fingerprint(
            selected_facts
        ),
    )


def test_definition_is_stable_context_free_and_scope_generic() -> None:
    first = build_history_retrieval_tool_definition_manifest()
    second = build_history_retrieval_tool_definition_manifest()

    assert len(first) == 1
    assert first[0].definition.digest == second[0].definition.digest
    assert first[0].implementation_ref == "builtin/retrieve_history"
    assert first[0].declared_behavior_revision == "history-retrieval-handler-1"
    definition = first[0].definition
    assert tuple(
        definition.spec.input_schema["properties"]["scopes"]["items"]["enum"]
    ) == tuple(scope.value for scope in HistoryRetrievalScope)
    assert definition.effect_template.effects[0].default_scope == "*"

    encoded = json.dumps(definition.descriptor(), sort_keys=True)
    for private_value in (
        _EFFECT_SCOPE,
        "private-session-1",
        "assistant_turn_idx:7",
        "private-generation-1",
        "private-service-1",
        "private-authority-1",
    ):
        assert private_value not in encoded
    assert "handler" not in encoded


def test_live_registration_binds_exactly_to_stable_definition() -> None:
    facts = _facts()
    registration = _registration(facts=facts)
    (binding,) = build_history_retrieval_tool_bindings(
        (registration,),
        facts=facts,
    )
    definition = build_history_retrieval_tool_definition_manifest()[0].definition

    assert isinstance(binding, ToolBinding)
    bound = BoundToolRegistration(definition, binding)
    assert bound.descriptor() == registration.descriptor()
    assert bound.handler is registration.handler
    assert binding.effect_profile.effects[0].default_scope == _EFFECT_SCOPE


def test_binding_assertion_contains_only_secret_free_context_digests() -> None:
    facts = _facts()
    (binding,) = build_history_retrieval_tool_bindings(
        (_registration(facts=facts),),
        facts=facts,
    )
    assertion = dict(binding.binding_assertion)

    assert set(assertion) == {
        "schema_version",
        "binding_kind",
        "effect_scope_sha256",
        "session_sha256",
        "turn_cutoff_sha256",
        "retrieval_generation_sha256",
        "allowed_scopes_sha256",
        "retrieval_service_sha256",
        "retrieval_authority_sha256",
        "source_fingerprint",
    }
    assert assertion["schema_version"] == HISTORY_RETRIEVAL_BINDING_ASSERTION_SCHEMA
    encoded = json.dumps(assertion, sort_keys=True)
    for private_value in (
        _EFFECT_SCOPE,
        "private-session-1",
        "assistant_turn_idx:7",
        "private-generation-1",
        "private-service-1",
        "private-authority-1",
    ):
        assert private_value not in encoded
    for key, value in assertion.items():
        if key.endswith("_sha256") or key == "source_fingerprint":
            assert isinstance(value, str)
            assert len(value) == 64


def test_binding_rejects_count_identity_contract_effect_and_source_drift() -> None:
    facts = _facts()
    registration = _registration(facts=facts)
    with pytest.raises(HistoryRetrievalCatalogError, match="exactly one"):
        build_history_retrieval_tool_bindings((), facts=facts)

    effect = registration.effect_profile.effects[0]
    cases = (
        (
            replace(registration, implementation_version="different"),
            "identity drifted",
        ),
        (
            replace(
                registration,
                spec=replace(registration.spec, name="Changed contract"),
            ),
            "model contract drifted",
        ),
        (
            replace(
                registration,
                execution_profile=replace(
                    registration.execution_profile,
                    max_output_bytes=1,
                ),
            ),
            "execution contract drifted",
        ),
        (
            replace(
                registration,
                effect_profile=replace(
                    registration.effect_profile,
                    effects=(replace(effect, data_egress=DataEgress.NONE),),
                ),
            ),
            "effects drifted",
        ),
        (
            replace(
                registration,
                source=replace(registration.source, kind=ToolSourceKind.PROVIDER),
            ),
            "source drifted",
        ),
        (
            replace(
                registration,
                source=replace(
                    registration.source,
                    fingerprint=_sha256("another source"),
                ),
            ),
            "source fingerprint drifted",
        ),
    )
    for drifted, message in cases:
        with pytest.raises(HistoryRetrievalCatalogError, match=message):
            build_history_retrieval_tool_bindings((drifted,), facts=facts)

    with pytest.raises(HistoryRetrievalCatalogError, match="effects drifted"):
        build_history_retrieval_tool_bindings(
            (_registration(facts=facts, effect_scope="history-corpus:other"),),
            facts=facts,
        )


def test_context_changes_only_binding_and_source_identity() -> None:
    with pytest.raises(HistoryRetrievalCatalogError, match="session_sha256"):
        _facts(session_sha256="not-a-digest")

    first_facts = _facts()
    second_facts = _facts(session_sha256=_sha256("private-session-2"))
    definitions_before = build_history_retrieval_tool_definition_manifest()
    first = build_history_retrieval_tool_bindings(
        (_registration(facts=first_facts),),
        facts=first_facts,
    )[0]
    second = build_history_retrieval_tool_bindings(
        (_registration(facts=second_facts),),
        facts=second_facts,
    )[0]
    definitions_after = build_history_retrieval_tool_definition_manifest()

    assert definitions_before[0].definition.digest == definitions_after[0].definition.digest
    assert first.digest != second.digest
    assert first.source.fingerprint != second.source.fingerprint
