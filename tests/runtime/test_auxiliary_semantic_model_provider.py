"""现行 Auxiliary 语义验证 provider 测试。"""

from __future__ import annotations

import json
from functools import partial

import pytest

from tests.helpers.prepared_model_provider import as_prepared_test_provider

from personagraph.l2.auxiliary_execution.adapters.model_authority import (
    create_auxiliary_semantic_reviewer_model_call_authority,
)
from personagraph.l2.auxiliary_execution.verification.model_provider import (
    AuxiliarySemanticModelProfile,
    build_auxiliary_semantic_structured_provider,
    build_mock_auxiliary_semantic_result,
)
from personagraph.l2.auxiliary_execution.verification.controller import (
    AuxiliarySemanticVerificationStatus,
    run_auxiliary_semantic_verification,
)
from personagraph.session import store
from tests.runtime.test_auxiliary_semantic_verification_controller import (
    _request,
)


def test_production_semantic_provider_and_authority_settle_with_exact_replay() -> None:
    request = _request("semantic-production-provider", protected=False)
    physical = build_auxiliary_semantic_structured_provider()
    provider_calls = 0

    def provider(*args: object, **kwargs: object):
        nonlocal provider_calls
        provider_calls += 1
        return physical(*args, **kwargs)  # type: ignore[arg-type]

    first = run_auxiliary_semantic_verification(
        request,
        provider=as_prepared_test_provider(provider),
        model_call_authority_factory=(
            partial(create_auxiliary_semantic_reviewer_model_call_authority, ledger_store=store)
        ),
        emit=lambda _event: None,
    )
    replayed = run_auxiliary_semantic_verification(
        request,
        provider=as_prepared_test_provider(
            lambda *_args, **_kwargs: pytest.fail(
                "semantic settlement replay reached Provider"
            )
        ),
        model_call_authority_factory=(
            partial(create_auxiliary_semantic_reviewer_model_call_authority, ledger_store=store)
        ),
        emit=lambda _event: None,
    )

    assert first.status is AuxiliarySemanticVerificationStatus.SETTLED
    assert first.settlement is not None
    assert replayed.status is AuxiliarySemanticVerificationStatus.SETTLED
    assert replayed.settlement == first.settlement
    assert replayed.replayed is True
    assert provider_calls == 1


def test_semantic_mock_rejects_non_object_input() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        build_mock_auxiliary_semantic_result("[]")


def test_semantic_mock_canonicalizes_multi_node_reference_order() -> None:
    request = _request("semantic-multi-node-order", protected=False)
    payload = request.prompt_payload.model_dump(mode="json")
    first = payload["task_graph_proposal"]["root"]["nodes"][0]
    first["node_key"] = "root"
    second = json.loads(json.dumps(first))
    second["node_key"] = "release"
    payload["task_graph_proposal"]["root"]["nodes"] = [first, second]

    result = build_mock_auxiliary_semantic_result(
        json.dumps(payload, ensure_ascii=False)
    )

    assert all(
        item["affected_node_keys"] == ["release", "root"]
        for item in result["items"]
    )


def test_semantic_production_profile_fits_the_eight_dimension_envelope() -> None:
    profile = AuxiliarySemanticModelProfile()

    assert profile.max_output_tokens == 131_072
    assert profile.timeout_s == 600.0


def test_semantic_mock_marks_blocking_gap_as_insufficient_evidence() -> None:
    request = _request("semantic-blocking-gap", protected=False)
    payload = request.prompt_payload.model_dump(mode="json")
    payload["context_artifacts"] = [
        {
            "gaps": [
                {
                    "gap_alias": "blocking_gap_01",
                    "blocking": True,
                }
            ]
        }
    ]

    result = build_mock_auxiliary_semantic_result(
        json.dumps(payload, ensure_ascii=False)
    )

    gap_item = next(
        item for item in result["items"] if item["dimension"] == "gap_disposition"
    )
    assert gap_item["verdict"] == "insufficient_evidence"
    assert gap_item["failure_scope"] == "missing_authority"
    assert gap_item["gap_aliases"]
