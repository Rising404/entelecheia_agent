from __future__ import annotations

import hashlib
import json

from personagraph.model_io.tier_bindings import (
    EndpointOrigin,
    ModelTierBinding,
    ModelTier,
)
from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairProtocol,
)
from personagraph.l2.task_execution.delivery.candidate_gate import (
    TaskDeliveryCandidateAuthority,
    create_task_delivery_candidate_model_call_authority,
)
from personagraph.l2.task_execution.delivery.model_contracts import (
    _SYSTEM_PROMPT,
)
from personagraph.l2.work_run import TaskNodeSubject
from tests.runtime.test_auxiliary_model_authority import _MemoryLedger
from tests.runtime.test_task_delivery_validation_provider import (
    _request,
)


def _binding() -> ModelTierBinding:
    return ModelTierBinding(
        tier=ModelTier.FINAL_GATE,
        provider="openai-compatible",
        base_url="https://delivery.example/v1",
        model="delivery-model",
        api_key="test-secret",
        thinking_enabled=False,
        origin=EndpointOrigin.PROFILE,
        profile_id="delivery-final-gate",
        profile_name="Delivery final gate",
    )


def _candidate_authority() -> TaskDeliveryCandidateAuthority:
    review_request = _request()
    subject = TaskNodeSubject(
        task_id=review_request.prompt.task_id,
        graph_revision=review_request.prompt.graph_revision,
        node_id=review_request.prompt.task_id,
        node_revision=1,
    )
    return TaskDeliveryCandidateAuthority.create(
        session_id=review_request.prompt.session_id,
        invocation_turn_id=review_request.invocation_turn_id,
        subject=subject,
        task_state_version=review_request.prompt.task_state_version,
        work_run_id="delivery-work-run",
        submitted_attempt_id="delivery-attempt",
        verification_request_id="delivery-node-verification",
        verification_request_revision=1,
        output_revision=1,
        output_sha256=hashlib.sha256(
            review_request.prompt.root_output_body.encode("utf-8")
        ).hexdigest(),
        candidate_delivery_id=review_request.prompt.root_delivery_id,
        review_request=review_request,
    )


def _fresh_authorities():
    binding = _binding()
    candidate = _candidate_authority()
    candidate_call = create_task_delivery_candidate_model_call_authority(
        candidate,
        rederive_state_guard_sha256=lambda: candidate.authority_sha256,
        ledger_store=_MemoryLedger(),
        model_binding=binding,
    )
    return candidate, candidate_call, binding


def test_delivery_logical_call_freezes_current_protocol_and_exact_prompt() -> None:
    candidate, candidate_call, _binding_value = _fresh_authorities()

    expected = (
        (
            candidate_call.logical_request,
            _SYSTEM_PROMPT,
            candidate.review_request.prompt.model_dump(mode="json"),
        ),
    )
    for logical, system_prompt, user_payload in expected:
        assert logical.output_repair_protocol is (
            RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
        )
        snapshot = logical.structured_prompt
        assert snapshot is not None
        assert snapshot.system_prompt == system_prompt
        assert json.loads(snapshot.user_content) == user_payload
