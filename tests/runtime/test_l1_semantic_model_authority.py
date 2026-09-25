"""审查冻结请求与真实 issues 结果必须原样认证，禁止历史输出兼容转换。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairFeedback,
    RuntimeModelOutputRepairIssue,
    RuntimeModelStructuredPrompt,
)
from personagraph.model_io.prepared_structured_provider import (
    durable_structured_provider_prompt,
    prepare_structured_repair_request,
)
from personagraph.model_io.tier_bindings import (
    EndpointOrigin,
    ModelTier,
    ModelTierBinding,
)
from personagraph.output_protocol.l1 import L1_ATTEMPT_PROTOCOL_VERSION
from personagraph.runtime.l1.identity import canonical_json
from personagraph.runtime.l1.model_authority import (
    L1_SEMANTIC_RESULT_CONTRACT,
    L1ModelAuthorityError,
    create_l1_attempt_model_call_authority,
    create_l1_semantic_model_call_authority,
    l1_model_output_repair_policy,
)
from personagraph.runtime.model_calls.contracts import (
    DurableModelCallStateGuardRejected,
    RuntimeModelLogicalRequest,
    RuntimeModelPhysicalAttemptRequest,
    RuntimeModelPhysicalOutcome,
    RuntimeModelTypedResult,
    RuntimeModelUsage,
)


_GUARD = "a" * 64
_BINDING = ModelTierBinding(
    tier=ModelTier.FINAL_GATE,
    provider="openai-compatible",
    base_url="https://offline.invalid/v1",
    model="offline-reviewer",
    thinking_enabled=False,
    origin=EndpointOrigin.GLOBAL,
)


class _StoredRequest:
    def __init__(self) -> None:
        self.request = None
        self.physical_attempts = ()

    def get_runtime_model_logical_call(self, **_kwargs):
        # Deliberately return even wrong identities: the authority must reject
        # corrupted storage projections instead of relying on this test adapter.
        return None if self.request is None else self


def _payload():
    return {
        "schema_version": "l1-semantic-verifier-model-protocol",
        "l1_turn_run_id": "run-review",
        "attempt_id": "attempt-review",
        "system_prompt": "冻结的审查说明",
        "candidate_binding": {
            "decision_hash": "c" * 64,
            "plan_hash": "d" * 64,
            "mechanical_verification_hash": "e" * 64,
            "state_guard_hash": _GUARD,
        },
        "review_view": {
            "candidate_final_reply": "source fact",
            "request_context": {"current_user_text": "请回答"},
            "plan": {"objective": "回答"},
            "execution_context": {"stop": {"must_finalize": False}},
            "durable_evidence": {
                "results": [
                    {
                        "tool_result_id": "result-read",
                        "tool_id": "read_file_chunks",
                        "status": "succeeded",
                        "result": {
                            "chunks": [
                                {"chunk_id": "chunk-1", "content": "source fact"}
                            ]
                        },
                        "metadata": {"partial": True},
                    }
                ]
            },
        },
        "output_contract": "L1SemanticVerificationResult",
        "repair_policy": l1_model_output_repair_policy(max_physical_attempts=6),
    }


def _create(store, payload, **changes):
    kwargs = {
        "session_id": "session-review",
        "turn_id": "turn-review",
        "logical_model_call_id": "logical-review",
        "state_guard_hash": _GUARD,
        "request_payload": payload,
        "max_physical_attempts": 6,
        "store": store,
        "model_binding": _BINDING,
        "rederive_state_guard_hash": lambda: _GUARD,
    }
    kwargs.update(changes)
    return create_l1_semantic_model_call_authority(**kwargs)


def test_semantic_authority_freezes_model_material_separately_from_host_binding():
    payload = _payload()
    request = _create(_StoredRequest(), payload).logical_request
    sent = json.loads(request.structured_prompt.user_content)
    persisted = json.loads(request.request_json)
    assert sent == payload["review_view"]
    assert "candidate_binding" not in sent
    assert "acceptance_progress" not in sent
    assert persisted["candidate_binding"] == payload["candidate_binding"]
    assert request.typed_result_contract == L1_SEMANTIC_RESULT_CONTRACT


@pytest.mark.parametrize(
    "path,value",
    [
        (("l1_turn_run_id",), "other-run"),
        (("attempt_id",), "other-attempt"),
        (("candidate_binding", "decision_hash"), "f" * 64),
        (("review_view", "candidate_final_reply"), "different answer"),
        (("review_view", "request_context", "current_user_text"), "different request"),
        (("review_view", "execution_context", "stop", "must_finalize"), True),
        (
            ("review_view", "durable_evidence", "results", 0, "result"),
            {"changed": True},
        ),
        (
            ("review_view", "durable_evidence", "results", 0, "metadata"),
            {"partial": False},
        ),
        (("repair_policy", "max_physical_attempts"), 7),
        (("output_contract",), "AnotherResult"),
        (("system_prompt",), "changed prompt"),
    ],
)
def test_semantic_authority_rejects_changed_frozen_input(path, value):
    store = _StoredRequest()
    payload = _payload()
    store.request = _create(store, payload).logical_request
    changed = deepcopy(payload)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(L1ModelAuthorityError):
        _create(store, changed)


@pytest.mark.parametrize(
    "changes",
    [
        {"session_id": "another"},
        {"turn_id": "another"},
        {"logical_model_call_id": "another"},
        {"state_guard_hash": "f" * 64},
        {"max_physical_attempts": 7},
        {"model_binding": replace(_BINDING, model="another-reviewer")},
        {
            "model_binding": replace(
                _BINDING, base_url="https://another-offline.invalid/v1"
            )
        },
        {"model_binding": replace(_BINDING, api_key="nonsecret-test-credential")},
    ],
)
def test_semantic_authority_rejects_identity_endpoint_and_budget_drift(changes):
    store = _StoredRequest()
    store.request = _create(store, _payload()).logical_request
    with pytest.raises(L1ModelAuthorityError):
        _create(store, _payload(), **changes)


@pytest.mark.parametrize(
    "corruption", ["request_hash", "binding_hash", "prompt_hash", "prompt_body"]
)
def test_semantic_authority_revalidates_frozen_request_integrity(corruption):
    store = _StoredRequest()
    frozen = _create(store, _payload()).logical_request
    if corruption == "request_hash":
        frozen = frozen.model_copy(update={"request_sha256": "f" * 64})
    elif corruption == "binding_hash":
        frozen = frozen.model_copy(update={"binding_sha256": "f" * 64})
    elif corruption == "prompt_hash":
        frozen = frozen.model_copy(
            update={
                "structured_prompt": frozen.structured_prompt.model_copy(
                    update={"user_content_sha256": "f" * 64}
                )
            }
        )
    else:
        values = frozen.model_dump(
            exclude={"request_json", "request_sha256", "binding_sha256"}
        )
        values["structured_prompt"] = RuntimeModelStructuredPrompt.create(
            system_prompt="冻结的审查说明",
            user_content=canonical_json({"wrong": "body"}),
        )
        frozen = RuntimeModelLogicalRequest.create(request_payload=_payload(), **values)
    store.request = frozen
    with pytest.raises(L1ModelAuthorityError):
        _create(store, _payload())


def test_semantic_authority_replays_success_without_new_physical_attempt():
    store = _StoredRequest()
    store.request = _create(store, _payload()).logical_request
    result = {"issues": []}
    store.physical_attempts = (
        SimpleNamespace(
            request=RuntimeModelPhysicalAttemptRequest.create(
                physical_attempt_id="physical-review",
                physical_attempt_key="physical-review",
                logical_call_id=store.request.logical_call_id,
                logical_request_binding_sha256=store.request.binding_sha256,
                physical_ordinal=1,
                started_turn_id=store.request.invocation_turn_id,
                provider=store.request.provider,
                model=store.request.model,
                endpoint_fingerprint=store.request.endpoint_fingerprint,
                request_sha256=store.request.request_sha256,
                output_repair_enabled=True,
                output_repair_feedback=None,
                dispatch_authority_sha256=_GUARD,
            ),
            settlement=SimpleNamespace(
                outcome=RuntimeModelPhysicalOutcome.SUCCEEDED,
                typed_result=RuntimeModelTypedResult.create(
                    result_contract=store.request.typed_result_contract,
                    result_payload=result,
                ),
                usage=RuntimeModelUsage(),
                finish_reason="stop",
            ),
        ),
    )
    resumed = _create(store, _payload())
    resumed.require_current_state()
    replay = resumed.replay_succeeded_result()
    assert replay.physical_ordinal == 1
    assert json.loads(replay.model_result.reply) == result
    assert len(store.physical_attempts) == 1
    with pytest.raises(DurableModelCallStateGuardRejected):
        _create(
            store, _payload(), rederive_state_guard_hash=lambda: "f" * 64
        ).require_current_state()


def test_ordinary_l1_attempt_does_not_gain_prompt_drift_permission():
    store = _StoredRequest()
    kwargs = {
        "session_id": "session-review",
        "turn_id": "turn-review",
        "l1_turn_run_id": "run-review",
        "attempt_id": "attempt-review",
        "logical_model_call_id": "logical-attempt",
        "state_guard_hash": _GUARD,
        "system_prompt": "original",
        "model_payload": {
            "schema_version": L1_ATTEMPT_PROTOCOL_VERSION,
            "attempt_id": "attempt-review",
            "attempt_ordinal": 1,
        },
        "max_physical_attempts": 6,
        "store": store,
        "model_binding": _BINDING,
        "rederive_state_guard_hash": lambda: _GUARD,
    }
    store.request = create_l1_attempt_model_call_authority(**kwargs).logical_request
    with pytest.raises(L1ModelAuthorityError):
        create_l1_attempt_model_call_authority(**{**kwargs, "system_prompt": "changed"})


def test_semantic_authority_repair_keeps_exact_frozen_four_messages():
    store = _StoredRequest()
    frozen = _create(store, _payload()).logical_request
    store.request = frozen
    resumed = _create(store, _payload())
    system, user = durable_structured_provider_prompt(
        resumed,
        system_prompt="not-used",
        user_content="not-used",
    )
    rejected = "{}"
    feedback = RuntimeModelOutputRepairFeedback(
        target_contract=frozen.typed_result_contract,
        rejected_physical_ordinal=1,
        rejected_response_sha256=hashlib.sha256(rejected.encode()).hexdigest(),
        issue_coverage="complete",
        omitted_issue_count=0,
        current_issues=(
            RuntimeModelOutputRepairIssue(
                category="schema",
                code="schema.missing",
                paths=("/issues",),
                safe_explanation="此位置缺少必填字段。",
            ),
        ),
    )
    captured = []
    provider = SimpleNamespace(
        prepare=lambda *args, **kwargs: captured.append((args, kwargs))
    )
    prepare_structured_repair_request(
        provider,
        system_prompt=system,
        user_content=user,
        purpose=frozen.purpose,
    )(feedback, rejected)
    messages = captured[0][1]["repair_messages"]
    assert messages[:3] == [
        {"role": "system", "content": frozen.structured_prompt.system_prompt},
        {"role": "user", "content": frozen.structured_prompt.user_content},
        {"role": "assistant", "content": rejected},
    ]
    assert len(messages) == 4
    assert "/issues" in messages[3]["content"]
    assert "rejected_response_sha256" not in messages[3]["content"]
