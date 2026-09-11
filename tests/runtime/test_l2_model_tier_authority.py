"""L2 模型层级分发与持久权威的跨层保证。"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect

import pytest

from personagraph.l2.auxiliary_graph import AuxiliaryNodeExecutorKind
from personagraph.model_io.tier_bindings import (
    EndpointOrigin,
    ModelTierBinding,
    ModelTier,
    current_model_tier_binding,
    effective_model_tier_binding,
)
from personagraph.model_io.gateway import ModelGatewayError, ModelResult, PreparedModelCall
from personagraph.l2.auxiliary_execution.adapters import (
    model_authority as auxiliary_authority,
)
from personagraph.runtime.model_calls import requests as model_requests
from personagraph.l2.task_execution.delivery import candidate_gate
from personagraph.l2.auxiliary_execution.planning.architect import (
    request_auxiliary_graph_architect,
)
from personagraph.l2.auxiliary_execution.adapters.model_authority import (
    AuxiliaryModelAuthorityFactoryError,
    create_auxiliary_graph_architect_model_call_authority,
    create_auxiliary_work_run_model_call_authority,
)
from personagraph.model_io.endpoint_identity import (
    configured_structured_model_endpoint_identity,
)
from personagraph.l2.task_execution.delivery.candidate_gate import (
    TaskDeliveryCandidateAuthority,
    create_task_delivery_candidate_model_call_authority,
)
from personagraph.l2.task_execution.task_node.model_authority import (
    create_task_node_work_run_model_call_authority,
)
from personagraph.session.persistence.calls.runtime_model_calls import (
    RuntimeModelCallIdentityCollision,
)
from personagraph.l2.work_run import TaskNodeSubject
from tests.runtime.test_auxiliary_model_authority import (
    _MemoryLedger,
    _architect_request,
    _terminal_fail_proposal,
    _work_run_binding,
)
from tests.runtime.test_task_node_model_authority import _attempt_binding
from tests.runtime.test_task_delivery_validation_provider import (
    _request as _task_delivery_request,
)


def _tier_binding(tier: ModelTier, *, model: str) -> ModelTierBinding:
    return ModelTierBinding(
        tier=tier,
        provider="openai-compatible",
        base_url="https://tier.example/v1",
        model=model,
        api_key="tier-secret",
        thinking_enabled=False,
        origin=EndpointOrigin.PROFILE,
        profile_id=f"profile-{tier.value}",
        profile_name=f"Profile {tier.value}",
    )


def test_model_authorities_require_an_explicit_model_ledger() -> None:
    for factory in (
        create_auxiliary_graph_architect_model_call_authority,
        create_auxiliary_work_run_model_call_authority,
        create_task_node_work_run_model_call_authority,
        create_task_delivery_candidate_model_call_authority,
    ):
        parameter = inspect.signature(factory).parameters["ledger_store"]
        assert parameter.default is inspect.Parameter.empty


def test_architect_authority_uses_the_exact_frozen_architect_binding() -> None:
    request = _architect_request()
    binding = _tier_binding(ModelTier.ARCHITECT, model="architect-model")

    authority = create_auxiliary_graph_architect_model_call_authority(
        request,
        invocation_turn_id="turn_architect_invocation_01",
        rederive_state_guard_sha256=lambda: request.binding_sha256,
        ledger_store=_MemoryLedger(),
        model_binding=binding,
    )

    assert authority.logical_request.provider == binding.provider
    assert authority.logical_request.model == binding.model
    assert authority.model_binding is binding


def test_terminal_task_graph_planner_is_bound_to_architect_not_attempt() -> None:
    call = _work_run_binding(
        executor_kind=AuxiliaryNodeExecutorKind.TERMINAL_PLANNER,
    )
    architect = _tier_binding(ModelTier.ARCHITECT, model="graph-model")

    authority = create_auxiliary_work_run_model_call_authority(
        call,
        rederive_state_guard_sha256=lambda: call.state_guard_sha256,
        ledger_store=_MemoryLedger(),
        model_binding=architect,
    )

    assert authority.logical_request.model == "graph-model"
    assert authority.model_binding.tier is ModelTier.ARCHITECT


def test_ordinary_task_node_authority_uses_the_execution_binding() -> None:
    call = _attempt_binding()
    execution = _tier_binding(ModelTier.ATTEMPT, model="execution-model")

    authority = create_task_node_work_run_model_call_authority(
        call,
        rederive_state_guard_sha256=lambda: call.state_guard_sha256,
        ledger_store=_MemoryLedger(),
        model_binding=execution,
    )

    assert authority.logical_request.model == "execution-model"
    assert authority.model_binding.tier is ModelTier.ATTEMPT


def test_durable_request_scopes_the_same_binding_into_every_provider_attempt() -> None:
    request = _architect_request()
    proposal = _terminal_fail_proposal()
    ledger = _MemoryLedger()
    binding = _tier_binding(ModelTier.ARCHITECT, model="architect-model")
    observed: list[ModelTierBinding | None] = []
    authority = create_auxiliary_graph_architect_model_call_authority(
        request,
        invocation_turn_id="turn_architect_invocation_01",
        rederive_state_guard_sha256=lambda: request.binding_sha256,
        ledger_store=ledger,
        model_binding=binding,
    )

    def provider(
        _system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        observed.append(current_model_tier_binding())
        return ModelResult(
            reply=proposal.model_dump_json(),
            provider=binding.provider,
            model=binding.model,
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    request_auxiliary_graph_architect(
        request,
        invocation_turn_id="turn_architect_invocation_01",
        provider=provider,
        emit=lambda _event: None,
        durable_call=authority,
    )

    assert observed == [binding]
    assert current_model_tier_binding() is None
    assert ledger.logical is not None
    assert ledger.logical.physical_attempts[0].settlement is not None


def test_format_repair_and_transport_retry_keep_one_frozen_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """可变设置不能把一个逻辑调用拆分到不同模型模式。"""

    monkeypatch.setattr(model_requests, "_sleep", lambda _seconds: None)
    request = _architect_request()
    proposal = _terminal_fail_proposal()
    ledger = _MemoryLedger()
    frozen = _tier_binding(ModelTier.ARCHITECT, model="architect-frozen")
    changed = replace(
        frozen,
        model="architect-changed",
        thinking_enabled=True,
    )
    authority = create_auxiliary_graph_architect_model_call_authority(
        request,
        invocation_turn_id="turn_architect_invocation_01",
        rederive_state_guard_sha256=lambda: request.binding_sha256,
        ledger_store=ledger,
        model_binding=frozen,
    )
    observed: list[ModelTierBinding] = []

    def provider(
        _system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        binding = effective_model_tier_binding(ModelTier.ARCHITECT)
        observed.append(binding)
        if len(observed) == 1:
    # 在格式错误的首次响应后修改可变配置。主机格式修复与后续传输重试仍必须
    # 在该逻辑调用的原始绑定下执行。
            monkeypatch.setattr(
                auxiliary_authority,
                "resolve_tier",
                lambda _tier: changed,
            )
            reply = "not-json"
        elif len(observed) == 2:
            raise ModelGatewayError(
                "MODEL_RATE_LIMITED",
                "temporary provider limit",
                retryable=True,
            )
        else:
            reply = proposal.model_dump_json()
        return ModelResult(
            reply=reply,
            provider=binding.provider,
            model=binding.model,
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    prepared_messages: list[list[dict[str, str]] | None] = []

    def prepare(
        system_prompt: str,
        user_content: str,
        *,
        purpose: str,
        repair_messages: list[dict[str, str]] | None = None,
    ) -> PreparedModelCall:
        prepared_messages.append(repair_messages)

        def dispatch(model_call_id: str | None) -> ModelResult:
            assert model_call_id is not None
            return provider(
                system_prompt,
                user_content,
                model_call_id=model_call_id,
                purpose=purpose,
            )

        return PreparedModelCall(_dispatch=dispatch)

    provider.prepare = prepare  # type: ignore[attr-defined]

    result = request_auxiliary_graph_architect(
        request,
        invocation_turn_id="turn_architect_invocation_01",
        provider=provider,
        emit=lambda _event: None,
        durable_call=authority,
    )

    assert result.attempts == 3
    assert observed == [frozen, frozen, frozen]
    assert prepared_messages[0] is None
    assert prepared_messages[1] is not None
    assert prepared_messages[2] == prepared_messages[1]
    assert [item["role"] for item in prepared_messages[1]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert current_model_tier_binding() is None
    assert ledger.logical is not None
    assert [
        attempt.settlement.outcome.value
        for attempt in ledger.logical.physical_attempts
        if attempt.settlement is not None
    ] == ["retryable_failure", "retryable_failure", "succeeded"]


class _StrictMemoryLedger(_MemoryLedger):
    """映射生产环境中不可变逻辑请求的冲突错误。"""

    def reserve_runtime_model_logical_call(self, *, request):  # type: ignore[no-untyped-def]
        if self.logical is not None and self.logical.request != request:
            raise RuntimeModelCallIdentityCollision(
                "logical model call ID crossed immutable request authority"
            )
        return super().reserve_runtime_model_logical_call(request=request)


@pytest.mark.parametrize("binding_source", ["explicit", "configured"])
def test_recovery_rejects_model_binding_drift_before_provider_io(
    monkeypatch: pytest.MonkeyPatch,
    binding_source: str,
) -> None:
    request = _architect_request()
    ledger = _StrictMemoryLedger()
    accepted = _tier_binding(ModelTier.ARCHITECT, model="accepted-model")
    changed = replace(
        accepted,
    # 保持提供方/模型标签相同。端点身份仍必须检测 ModelResult 无法报告的物理端点变化。
        base_url="https://changed-tier.example/v1",
    )

    if binding_source == "configured":
        monkeypatch.setattr(
            auxiliary_authority,
            "resolve_tier",
            lambda _tier: accepted,
        )
        first = create_auxiliary_graph_architect_model_call_authority(
            request,
            invocation_turn_id="turn_architect_invocation_01",
            rederive_state_guard_sha256=lambda: request.binding_sha256,
            ledger_store=ledger,
        )
    else:
        first = create_auxiliary_graph_architect_model_call_authority(
            request,
            invocation_turn_id="turn_architect_invocation_01",
            rederive_state_guard_sha256=lambda: request.binding_sha256,
            ledger_store=ledger,
            model_binding=accepted,
        )
    first.reserve(turn_id="turn_architect_invocation_01")
    physical = first.begin_physical_attempt(
        turn_id="turn_architect_invocation_01",
        max_physical_attempts=6,
    )
    first.settle_physical_attempt(
        turn_id="turn_architect_invocation_01",
        physical=physical,
        outcome="retryable_failure",
        result_fingerprint="b" * 64,
        error_code="provider_retryable",
    )

    with pytest.raises(AuxiliaryModelAuthorityFactoryError):
        if binding_source == "configured":
            monkeypatch.setattr(
                auxiliary_authority,
                "resolve_tier",
                lambda _tier: changed,
            )
            create_auxiliary_graph_architect_model_call_authority(
                request,
                invocation_turn_id="turn_architect_invocation_01",
                rederive_state_guard_sha256=lambda: request.binding_sha256,
                ledger_store=ledger,
            )
        else:
            create_auxiliary_graph_architect_model_call_authority(
                request,
                invocation_turn_id="turn_architect_invocation_01",
                rederive_state_guard_sha256=lambda: request.binding_sha256,
                ledger_store=ledger,
                model_binding=changed,
            )

    assert ledger.logical is not None
    assert len(ledger.logical.physical_attempts) == 1


def test_task_delivery_candidate_authority_uses_final_gate_endpoint_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _tier_binding(ModelTier.FINAL_GATE, model="final-gate-model")
    identity = configured_structured_model_endpoint_identity(binding)
    resolved_tiers: list[ModelTier] = []

    def resolve(tier: ModelTier) -> ModelTierBinding:
        resolved_tiers.append(tier)
        return binding

    monkeypatch.setattr(candidate_gate, "resolve_tier", resolve)
    review_request = _task_delivery_request()
    subject = TaskNodeSubject(
        task_id=review_request.prompt.task_id,
        graph_revision=review_request.prompt.graph_revision,
        node_id=review_request.prompt.task_id,
        node_revision=1,
    )
    candidate = TaskDeliveryCandidateAuthority.create(
        session_id=review_request.prompt.session_id,
        invocation_turn_id=review_request.invocation_turn_id,
        subject=subject,
        task_state_version=review_request.prompt.task_state_version,
        work_run_id="work-run-final-gate",
        submitted_attempt_id="attempt-final-gate",
        verification_request_id="node-verification-final-gate",
        verification_request_revision=1,
        output_revision=1,
        output_sha256=hashlib.sha256(
            review_request.prompt.root_output_body.encode("utf-8")
        ).hexdigest(),
        candidate_delivery_id=review_request.prompt.root_delivery_id,
        review_request=review_request,
    )

    candidate_call = create_task_delivery_candidate_model_call_authority(
        candidate,
        rederive_state_guard_sha256=lambda: candidate.authority_sha256,
        ledger_store=_MemoryLedger(),
    )

    assert resolved_tiers == [ModelTier.FINAL_GATE]
    assert candidate_call.logical_request.provider == identity.provider
    assert candidate_call.logical_request.model == identity.model
    assert (
        candidate_call.logical_request.endpoint_fingerprint
        == identity.endpoint_fingerprint
    )
    assert candidate_call.model_binding is binding
    assert "tier-secret" not in candidate_call.logical_request.model_dump_json()
