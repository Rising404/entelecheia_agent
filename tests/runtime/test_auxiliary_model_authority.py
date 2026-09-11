from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from personagraph.l2.auxiliary_graph import (
    AuxiliaryNodeExecutorKind,
    AuxiliaryGraphRevisionProposalDisposition,
    AuxiliaryGraphRevisionProposal,
    AuxiliaryPlanningGoal,
    PlanningAuthorityClass,
    PlanningAuthorityProjection,
    PlanningAuthoritySourceCard,
    PlanningAuthoritySourceKind,
    PlanningCapabilityCatalogProjection,
    PlanningCapabilityDescriptor,
    PlanningCapabilityEffect,
    PlanningEpisodeBudgetUsage,
    PlanningEpisodeBudget,
    PlanningGoalPromptContext,
    TaskGraphRevisionCandidate,
    TaskGraphSemanticBaseSnapshot,
    TaskGraphSemanticLineageProjection,
    TaskGraphSemanticVerificationDimension,
    TaskGraphSemanticVerificationItem,
    TaskGraphSemanticVerificationRequest,
    TaskGraphSemanticVerificationResult,
    TaskGraphSemanticVerificationVerdict,
)
from personagraph.model_io.gateway import ModelResult, PreparedModelCall
from personagraph.l2.auxiliary_execution.planning.architect import (
    AuxiliaryGraphArchitectPrompt,
    AuxiliaryGraphArchitectRequest,
    _AUXILIARY_GRAPH_ARCHITECT_SYSTEM_PROMPT,
    request_auxiliary_graph_architect,
    serialize_auxiliary_graph_architect_prompt,
)
from personagraph.l2.auxiliary_execution.adapters.model_authority import (
    AuxiliaryModelAuthorityFactoryError,
    create_auxiliary_graph_architect_model_call_authority,
    create_auxiliary_semantic_reviewer_model_call_authority,
    create_auxiliary_work_run_model_call_authority,
)
from personagraph.l2.auxiliary_execution.verification.controller import (
    AuxiliarySemanticReviewerModelCallBinding,
)
from personagraph.l2.auxiliary_execution.work_run.controller import (
    AuxiliaryBoundModelCall,
    _bind_model_call_authority,
    _materialize_terminal_action,
)
from personagraph.l2.task_graph.contracts import (
    InSessionTaskGraphRevisionProposal,
    InSessionTaskGraphRevisionValidationContext,
    InSessionTaskSourceAnchor,
)
from personagraph.runtime.model_calls import (
    DurableModelCallStateGuardRejected,
)
from personagraph.runtime.model_calls import (
    RuntimeModelLogicalRequest,
    RuntimeModelPhysicalAttemptRequest,
    RuntimeModelPhysicalAttemptSettlement,
)
from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairProtocol,
)
from personagraph.runtime.model_calls import (
    RuntimeLogicalModelCallAuthority,
)
from personagraph.runtime.model_calls import request_model_with_retry
from personagraph.runtime.turn_events import RuntimeStage
from personagraph.l2.work_run import (
    AcceptanceVerificationFeedback,
    AttemptDecision,
    AuxiliaryNodeSubject,
    NodeVerificationResult,
    RequestUserInputAction,
    SubmitOutputWindowAction,
    OutputWindowFormat,
    VerificationVerdict,
)
from personagraph.l2.auxiliary_execution.verification.task_graph_semantic import (
    _TASK_GRAPH_SEMANTIC_VERIFICATION_SYSTEM_PROMPT,
    _parse_and_guard_task_graph_semantic_verification,
    serialize_task_graph_semantic_verification_prompt,
)
from personagraph.session.persistence.calls.runtime_model_calls import (
    StoredRuntimeModelLogicalCall,
    StoredRuntimeModelPhysicalAttempt,
)
from tests.session.test_auxiliary_semantic_verification_persistence import (
    _catalog,
    _complete_terminal_only,
    _semantic_request,
)
from tests.helpers.prepared_model_provider import as_prepared_test_provider


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
CURRENT_REPAIR_PROTOCOL = (
    RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
)


class _MemoryLedger:
    def __init__(self) -> None:
        self.logical: StoredRuntimeModelLogicalCall | None = None
        self.rejected_outputs: dict[tuple[int, str], str] = {}

    def reserve_runtime_model_logical_call(
        self,
        *,
        request: RuntimeModelLogicalRequest,
    ) -> object:
        if self.logical is None:
            self.logical = StoredRuntimeModelLogicalCall(request=request)
        elif self.logical.request != request:
            raise AssertionError("logical collision")
        return SimpleNamespace(logical_call=self.logical)

    def append_runtime_model_physical_attempt(
        self,
        *,
        request: RuntimeModelPhysicalAttemptRequest,
    ) -> object:
        assert self.logical is not None
        self.logical = StoredRuntimeModelLogicalCall(
            request=self.logical.request,
            physical_attempts=(
                *self.logical.physical_attempts,
                StoredRuntimeModelPhysicalAttempt(request=request),
            ),
        )
        return SimpleNamespace(physical_attempt_id=request.physical_attempt_id)

    def settle_runtime_model_physical_attempt(
        self,
        *,
        settlement: RuntimeModelPhysicalAttemptSettlement,
        rejected_response_text: str | None = None,
    ) -> object:
        assert self.logical is not None
        self.logical = StoredRuntimeModelLogicalCall(
            request=self.logical.request,
            physical_attempts=tuple(
                StoredRuntimeModelPhysicalAttempt(
                    request=item.request,
                    settlement=(
                        settlement
                        if item.request.physical_attempt_id
                        == settlement.physical_attempt_id
                        else item.settlement
                    ),
                )
                for item in self.logical.physical_attempts
            ),
        )
        if rejected_response_text is not None:
            feedback = settlement.next_output_repair_feedback
            assert feedback is not None
            self.rejected_outputs[
                (
                    settlement.physical_ordinal,
                    feedback.rejected_response_sha256,
                )
            ] = rejected_response_text
        return SimpleNamespace(settlement_id=settlement.settlement_id)

    def get_runtime_model_rejected_output(
        self,
        *,
        session_id: str,
        logical_call_id: str,
        rejected_physical_ordinal: int,
        rejected_response_sha256: str,
    ) -> object | None:
        if (
            self.logical is None
            or self.logical.request.session_id != session_id
            or self.logical.request.logical_call_id != logical_call_id
        ):
            return None
        response_text = self.rejected_outputs.get(
            (rejected_physical_ordinal, rejected_response_sha256)
        )
        if response_text is None:
            return None
        return SimpleNamespace(response_text=response_text)

    def get_runtime_model_logical_call(
        self,
        *,
        session_id: str,
        logical_call_id: str,
    ) -> StoredRuntimeModelLogicalCall | None:
        if (
            self.logical is None
            or self.logical.request.session_id != session_id
            or self.logical.request.logical_call_id != logical_call_id
        ):
            return None
        return self.logical


def _architect_request(
    *,
    physical_provider_tries: int = 0,
) -> AuxiliaryGraphArchitectRequest:
    goal = AuxiliaryPlanningGoal(
        session_id="session_authority_01",
        task_id="task_authority_01",
        auxiliary_graph_id="auxiliary_graph_authority_01",
        goal_id="goal_authority_01",
        base_task_graph_revision=None,
        target_task_graph_revision=1,
        creation_turn_id="turn_creation_01",
        authorization_manifest_id="authorization_manifest_01",
        budget_ledger_id="budget_authority_01",
    )
    authority = PlanningAuthorityProjection.create(
        authority_snapshot_id="authority_snapshot_01",
        authority_snapshot_sha256=SHA_A,
        cards=(
            PlanningAuthoritySourceCard(
                alias="user_authority",
                authority_class=PlanningAuthorityClass.AUTHORIZATION,
                source_kind=PlanningAuthoritySourceKind.USER_INSTRUCTION,
                source_label="Current user instruction",
                excerpt="Create a source-bound executable task graph.",
                projection_sha256=SHA_B,
            ),
        ),
    )
    capabilities = PlanningCapabilityCatalogProjection.create(
        capability_catalog_snapshot_id="capability_catalog_01",
        capability_catalog_snapshot_sha256=SHA_C,
        capabilities=(
            PlanningCapabilityDescriptor(
                capability_alias="bounded_read",
                label="Bounded read",
                description="Read an already authorized bounded resource.",
                available=True,
                effect=PlanningCapabilityEffect.READ_ONLY,
                supported_operations=("read",),
                supported_resource_kinds=("artifact",),
            ),
        ),
    )
    budget = PlanningEpisodeBudget.create(
        budget_ledger_id=goal.budget_ledger_id,
        goal_id=goal.goal_id,
        usage=PlanningEpisodeBudgetUsage(
            physical_provider_tries=physical_provider_tries,
        ),
    )
    prompt = AuxiliaryGraphArchitectPrompt.create(
        goal=PlanningGoalPromptContext(
            goal_id=goal.goal_id,
            objective="Create the exact task graph.",
            desired_output="A complete source-bound TaskGraph proposal.",
            authorization_aliases=("user_authority",),
        ),
        authority=authority,
        context_artifacts=(),
        capabilities=capabilities,
        protected_capability_grants=(),
        budget=budget,
        current_revision=None,
    )
    return AuxiliaryGraphArchitectRequest.create(
        architect_request_id="architect_request_authority_01",
        logical_call_id="architect_logical_call_authority_01",
        architect_profile_id="auxiliary_architect_v2",
        goal=goal,
        prompt_payload=prompt,
    )


def _terminal_fail_proposal() -> AuxiliaryGraphRevisionProposal:
    return AuxiliaryGraphRevisionProposal(
        disposition=AuxiliaryGraphRevisionProposalDisposition.TERMINAL_FAIL,
        expected_current_auxiliary_graph_revision=None,
        revision_reason=None,
        structure=None,
        explanation="The frozen authority cannot support a legal graph.",
        failure_reason="The required source is unavailable.",
    )


def _semantic_binding(
    prefix: str,
) -> AuxiliarySemanticReviewerModelCallBinding:
    session_id, turn_id, task_id, details, prompt_inputs = (
        _complete_terminal_only(prefix)
    )
    semantic_request = _semantic_request(
        request_id=f"{prefix}-semantic-request",
        logical_call_id=f"{prefix}-semantic-call",
        reviewer_ordinal=1,
        details=details,
        catalog=_catalog(protected=True),
        prompt_inputs=prompt_inputs,
    )
    request_payload = {
        "schema_version": "auxiliary-v2-semantic-reviewer-model-call-v1",
        "verification_result_id": f"{prefix}-semantic-result",
        "verification_request": semantic_request.model_dump(mode="json"),
    }
    request_json = json.dumps(
        request_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return AuxiliarySemanticReviewerModelCallBinding(
        logical_call_id=semantic_request.logical_call_id,
        verification_request_id=semantic_request.verification_request_id,
        verification_result_id=f"{prefix}-semantic-result",
        reviewer_ordinal=semantic_request.reviewer_ordinal,
        required_reviewer_count=semantic_request.required_reviewer_count,
        session_id=session_id,
        task_id=task_id,
        auxiliary_graph_id=semantic_request.goal.auxiliary_graph_id,
        goal_id=semantic_request.goal.goal_id,
        auxiliary_graph_revision=semantic_request.auxiliary_graph_revision,
        invocation_turn_id=turn_id,
        request_json=request_json,
        request_sha256=hashlib.sha256(request_json.encode("utf-8")).hexdigest(),
        state_guard_sha256=SHA_B,
    )


def _semantic_result(
    binding: AuxiliarySemanticReviewerModelCallBinding,
    *,
    verification_result_id: str | None = None,
) -> TaskGraphSemanticVerificationResult:
    semantic_request = TaskGraphSemanticVerificationRequest.model_validate(
        json.loads(binding.request_json)["verification_request"]
    )
    return TaskGraphSemanticVerificationResult.create(
        verification_result_id=(
            verification_result_id or binding.verification_result_id
        ),
        verification_request_id=binding.verification_request_id,
        request_binding_sha256=semantic_request.binding_sha256,
        logical_call_id=binding.logical_call_id,
        verification_profile_id=semantic_request.verification_profile_id,
        reviewer_ordinal=binding.reviewer_ordinal,
        required_reviewer_count=binding.required_reviewer_count,
        items=tuple(
            TaskGraphSemanticVerificationItem(
                dimension=dimension,
                verdict=TaskGraphSemanticVerificationVerdict.PASS,
                failure_scope=None,
                finding=f"{dimension.value} passes.",
            )
            for dimension in TaskGraphSemanticVerificationDimension
        ),
    )


def _work_run_subject() -> AuxiliaryNodeSubject:
    return AuxiliaryNodeSubject(
        task_id="task_work_run_authority_01",
        auxiliary_graph_id="auxiliary_graph_work_run_authority_01",
        auxiliary_graph_revision=3,
        node_id="auxiliary_node_work_run_authority_01",
        node_revision=2,
    )


def _work_run_binding(
    *,
    call_kind: str = "attempt_decision",
    executor_kind: AuxiliaryNodeExecutorKind = (
        AuxiliaryNodeExecutorKind.MODEL_WORK_RUN
    ),
    task_graph_base: TaskGraphSemanticBaseSnapshot | None = None,
    invocation_turn_id: str = "turn_work_run_invocation_01",
    work_run_revision: int = 7,
    system_prompt: str = "Execute only the exact frozen WorkRun request.",
    state_guard_sha256: str = SHA_A,
) -> AuxiliaryBoundModelCall:
    subject = _work_run_subject()
    common_bindings = {
        "session_id": "session_work_run_authority_01",
        "work_run_id": "work_run_authority_01",
        "subject": subject.model_dump(mode="json"),
    }
    if call_kind == "attempt_decision":
        prompt_bindings = {
            **common_bindings,
            "turn_id": invocation_turn_id,
            "work_run_revision": work_run_revision,
            "attempt_id": "attempt_work_run_authority_01",
            "attempt_ordinal": 2,
        }
        verification_request_id = None
        verification_request_revision = None
    else:
        prompt_bindings = {
            **common_bindings,
            "request_turn_id": "turn_work_run_request_01",
            "verification_request_id": "verification_work_run_authority_01",
            "locked_work_run_revision": 7,
            "submitted_attempt_id": "attempt_work_run_authority_01",
        }
        verification_request_id = "verification_work_run_authority_01"
        verification_request_revision = 4
    user_payload = {
        "bindings": prompt_bindings,
        "node": {"acceptances": []},
    }
    if task_graph_base is not None:
        user_payload["task_graph_proposal_contract"] = {
            "expected_current_graph_revision": (
                task_graph_base.base_task_graph_revision
            )
        }
        user_payload["task_graph_revision_base"] = task_graph_base.model_dump(
            mode="json"
        )
    user_content = json.dumps(
        user_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return AuxiliaryBoundModelCall.create(
        call_kind=call_kind,  # type: ignore[arg-type]
        logical_call_id=f"logical_work_run_{call_kind}_01",
        session_id="session_work_run_authority_01",
        goal_id="goal_work_run_authority_01",
        subject=subject,
        executor_kind=executor_kind,
        request_turn_id="turn_work_run_request_01",
        invocation_turn_id=invocation_turn_id,
        work_run_id="work_run_authority_01",
        work_run_revision=work_run_revision,
        attempt_id="attempt_work_run_authority_01",
        attempt_ordinal=2,
        verification_request_id=verification_request_id,
        verification_request_revision=verification_request_revision,
        system_prompt=system_prompt,
        user_content=user_content,
        state_guard_sha256=state_guard_sha256,
    )


def _terminal_task_graph_proposal() -> InSessionTaskGraphRevisionProposal:
    return InSessionTaskGraphRevisionProposal.model_validate(
        {
            "root": {
                "root_key": "root",
                "nodes": [
                    {
                        "node_key": "root",
                        "node_kind": "root",
                        "parent_node_key": None,
                        "title": "Execute the plan",
                        "objective": "Deliver the exact source-bound result.",
                        "source_anchor_ids": ["task_creation_source"],
                        "acceptance_criteria": [
                            {
                                "acceptance_id": "result_ready",
                                "criterion": "The requested result is complete.",
                                "source_anchor_ids": ["task_creation_source"],
                            }
                        ],
                        "constraints": [],
                    }
                ],
            }
        }
    )


def _terminal_task_graph_base() -> TaskGraphSemanticBaseSnapshot:
    proposal = _terminal_task_graph_proposal()
    return TaskGraphSemanticBaseSnapshot.create(
        base_task_graph_revision=1,
        root_node_alias="base_root",
        nodes=(
            {
                "node_alias": "base_root",
                "node_revision": 1,
                "node_kind": "root",
                "parent_node_alias": None,
                "title": "Existing plan",
                "objective": "Deliver the existing authorized plan.",
                "source_anchor_aliases": ("task_creation_source",),
                "acceptance_criteria": (
                    proposal.root.nodes[0].acceptance_criteria[0],
                ),
                "constraints": (),
                "status": "proposed",
            },
        ),
        source_snapshot_sha256=SHA_C,
    )


def _terminal_task_graph_context() -> InSessionTaskGraphRevisionValidationContext:
    source_text = "Create the exact source-bound result."
    return InSessionTaskGraphRevisionValidationContext(
        session_id="session_work_run_authority_01",
        source_turn_id="turn_work_run_invocation_01",
        target_insession_task_id="task_work_run_authority_01",
        expected_current_graph_revision=1,
        source_anchors=(
            InSessionTaskSourceAnchor(
                anchor_id="task_creation_source",
                source_turn_id="turn_work_run_invocation_01",
                source_kind="previously_authorized_task_state",
                start=0,
                end=len(source_text),
                excerpt=source_text,
            ),
        ),
        authorization_anchor_ids=("task_creation_source",),
        required_anchor_ids=("task_creation_source",),
    )


def _positive_terminal_candidate(
    *,
    base_node_alias: str = "base_root",
) -> TaskGraphRevisionCandidate:
    proposal = _terminal_task_graph_proposal()
    return TaskGraphRevisionCandidate(
        proposal=proposal,
        lineage=(
            TaskGraphSemanticLineageProjection(
                proposal_node_key="root",
                disposition="reuse",
                base_node_alias=base_node_alias,
            ),
        ),
    )


def test_architect_factory_builds_private_config_free_exact_logical_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "anthropic-compatible")
    monkeypatch.setenv("PERSONAGRAPH_MODEL", "configured-model-v1")
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "sk-private-do-not-persist")
    monkeypatch.setenv(
        "PERSONAGRAPH_BASE_URL",
        "https://user:private@private.example/v1?token=secret",
    )
    request = _architect_request()
    rederived: list[str] = []

    def rederive() -> str:
        rederived.append(request.binding_sha256)
        return request.binding_sha256

    authority = create_auxiliary_graph_architect_model_call_authority(
        request,
        invocation_turn_id="turn_architect_invocation_01",
        rederive_state_guard_sha256=rederive,
        ledger_store=_MemoryLedger(),
    )
    logical = authority.logical_request

    assert isinstance(authority, RuntimeLogicalModelCallAuthority)
    assert logical.logical_call_id == request.logical_call_id
    assert logical.session_id == request.goal.session_id
    assert logical.task_id == request.goal.task_id
    assert logical.auxiliary_graph_id == request.goal.auxiliary_graph_id
    assert logical.goal_id == request.goal.goal_id
    assert logical.execution_subject_id is None
    assert logical.invocation_turn_id == "turn_architect_invocation_01"
    assert logical.call_kind == "auxiliary_graph_architect"
    assert logical.purpose == "runtime_auxiliary_graph_architect_v2"
    assert logical.provider == "anthropic-compatible"
    assert logical.model == "configured-model-v1"
    assert logical.request_contract == "auxiliary-graph-architect-request-v1"
    assert json.loads(logical.request_json) == request.model_dump(mode="json")
    assert logical.output_repair_protocol is CURRENT_REPAIR_PROTOCOL
    assert logical.structured_prompt is not None
    assert logical.structured_prompt.system_prompt == (
        _AUXILIARY_GRAPH_ARCHITECT_SYSTEM_PROMPT
    )
    assert logical.structured_prompt.user_content == (
        serialize_auxiliary_graph_architect_prompt(request)
    )
    assert logical.typed_result_contract == "auxiliary-graph-revision-proposal-v2"
    assert logical.max_physical_attempts == 6
    assert logical.state_guard_sha256 == request.binding_sha256
    assert len(logical.endpoint_fingerprint) == 64
    serialized = logical.model_dump_json()
    assert "sk-private-do-not-persist" not in serialized
    assert "private.example" not in serialized
    assert "user:private" not in serialized

    authority.require_current_state()
    assert rederived == [request.binding_sha256]


def test_auxiliary_authority_factories_require_explicit_model_ledger() -> None:
    architect_request = _architect_request()
    semantic_binding = _semantic_binding("semantic-ledger-required")
    work_run_binding = _work_run_binding()

    with pytest.raises(TypeError, match="ledger_store"):
        create_auxiliary_graph_architect_model_call_authority(
            architect_request,
            invocation_turn_id="turn_architect_invocation_01",
            rederive_state_guard_sha256=lambda: architect_request.binding_sha256,
        )  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="ledger_store"):
        create_auxiliary_semantic_reviewer_model_call_authority(
            semantic_binding,
            rederive_state_guard_sha256=lambda: semantic_binding.state_guard_sha256,
        )  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="ledger_store"):
        create_auxiliary_work_run_model_call_authority(
            work_run_binding,
            rederive_state_guard_sha256=lambda: work_run_binding.state_guard_sha256,
        )  # type: ignore[call-arg]


def test_architect_factory_has_stable_identity_and_exact_typed_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    request = _architect_request()
    proposal = _terminal_fail_proposal()
    ledger = _MemoryLedger()
    provider_calls = 0

    def factory() -> RuntimeLogicalModelCallAuthority:
        return create_auxiliary_graph_architect_model_call_authority(
            request,
            invocation_turn_id="turn_architect_invocation_01",
            rederive_state_guard_sha256=lambda: request.binding_sha256,
            ledger_store=ledger,
        )

    first = factory()
    second = factory()
    assert first.logical_request == second.logical_request

    def provider(
        _system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return ModelResult(
            reply=proposal.model_dump_json(),
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    fresh = request_auxiliary_graph_architect(
        request,
        invocation_turn_id="turn_architect_invocation_01",
        provider=as_prepared_test_provider(provider),
        emit=lambda _event: None,
        durable_call=first,
    )
    replayed = request_auxiliary_graph_architect(
        request,
        invocation_turn_id="turn_architect_invocation_01",
        provider=as_prepared_test_provider(
            lambda *_args, **_kwargs: pytest.fail("replay reached Provider")
        ),
        emit=lambda _event: None,
        durable_call=second,
    )

    assert provider_calls == 1
    assert fresh.value == replayed.value
    assert replayed.replayed is True
    assert ledger.logical is not None
    typed = ledger.logical.physical_attempts[0].settlement.typed_result
    assert typed is not None
    assert typed.result_contract == "auxiliary-graph-revision-proposal-v2"
    assert typed.parsed() == proposal.model_dump(mode="json")


def test_architect_factory_rejects_stale_request_tamper_and_guard_drift() -> None:
    request = _architect_request()
    tampered = request.model_copy(
        update={"architect_profile_id": "different_architect_profile"}
    )
    with pytest.raises(
        AuxiliaryModelAuthorityFactoryError,
        match="Architect request",
    ):
        create_auxiliary_graph_architect_model_call_authority(
            tampered,
            invocation_turn_id="turn_architect_invocation_01",
            rederive_state_guard_sha256=lambda: request.binding_sha256,
            ledger_store=_MemoryLedger(),
        )

    authority = create_auxiliary_graph_architect_model_call_authority(
        request,
        invocation_turn_id="turn_architect_invocation_01",
        rederive_state_guard_sha256=lambda: SHA_C,
        ledger_store=_MemoryLedger(),
    )
    with pytest.raises(DurableModelCallStateGuardRejected):
        authority.require_current_state()


def test_semantic_factory_exactly_matches_controller_binding_and_replay_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("PERSONAGRAPH_MODEL", "semantic-model-v1")
    binding = _semantic_binding("semantic-authority-exact")
    authority = create_auxiliary_semantic_reviewer_model_call_authority(
        binding,
        rederive_state_guard_sha256=lambda: binding.state_guard_sha256,
        ledger_store=_MemoryLedger(),
    )
    logical = authority.logical_request

    assert logical.logical_call_id == binding.logical_call_id
    assert logical.session_id == binding.session_id
    assert logical.task_id == binding.task_id
    assert logical.auxiliary_graph_id == binding.auxiliary_graph_id
    assert logical.goal_id == binding.goal_id
    assert logical.execution_subject_id is None
    assert logical.invocation_turn_id == binding.invocation_turn_id
    assert logical.call_kind == "task_graph_semantic_verification"
    assert logical.purpose == binding.purpose
    assert logical.provider == "anthropic-compatible"
    assert logical.model == "semantic-model-v1"
    assert logical.request_contract == binding.request_contract
    assert logical.request_json == binding.request_json
    assert logical.request_sha256 == binding.request_sha256
    assert logical.output_repair_protocol is CURRENT_REPAIR_PROTOCOL
    assert logical.structured_prompt is not None
    semantic_request = TaskGraphSemanticVerificationRequest.model_validate(
        json.loads(binding.request_json)["verification_request"]
    )
    assert logical.structured_prompt.system_prompt == (
        _TASK_GRAPH_SEMANTIC_VERIFICATION_SYSTEM_PROMPT
    )
    assert logical.structured_prompt.user_content == (
        serialize_task_graph_semantic_verification_prompt(semantic_request)
    )
    assert logical.typed_result_contract == binding.typed_result_contract
    assert logical.max_physical_attempts == binding.max_physical_attempts
    assert logical.state_guard_sha256 == binding.state_guard_sha256

    result = _semantic_result(binding)
    replay_payload = authority.typed_result_payload(
        model_result=ModelResult(
            reply='{"items":[]}',
            provider=logical.provider,
            model=logical.model,
            latency_ms=1,
            model_call_id="physical_semantic_01",
        ),
        value=result,
    )
    assert replay_payload == {
        "items": [
            {
                **item.model_dump(mode="json"),
                "failure_scope": (
                    None
                    if item.failure_scope is None
                    else item.failure_scope.value
                ),
            }
            for item in result.items
        ]
    }
    reparsed = _parse_and_guard_task_graph_semantic_verification(
        json.dumps(
            replay_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    assert reparsed == result.items


def test_semantic_factory_is_stable_and_rejects_binding_or_result_tamper() -> None:
    binding = _semantic_binding("semantic-authority-tamper")

    def factory() -> RuntimeLogicalModelCallAuthority:
        return create_auxiliary_semantic_reviewer_model_call_authority(
            binding,
            rederive_state_guard_sha256=lambda: binding.state_guard_sha256,
            ledger_store=_MemoryLedger(),
        )

    assert factory().logical_request == factory().logical_request

    tampered = binding.model_copy(update={"request_json": '{"changed":true}'})
    with pytest.raises(
        AuxiliaryModelAuthorityFactoryError,
        match="semantic reviewer binding",
    ):
        create_auxiliary_semantic_reviewer_model_call_authority(
            tampered,
            rederive_state_guard_sha256=lambda: binding.state_guard_sha256,
            ledger_store=_MemoryLedger(),
        )

    envelope = json.loads(binding.request_json)
    envelope["verification_request"]["binding_sha256"] = SHA_C
    request_json = json.dumps(
        envelope,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    self_hash_tampered = binding.model_copy(
        update={
            "request_json": request_json,
            "request_sha256": hashlib.sha256(
                request_json.encode("utf-8")
            ).hexdigest(),
        }
    )
    with pytest.raises(
        AuxiliaryModelAuthorityFactoryError,
        match="semantic reviewer request",
    ):
        create_auxiliary_semantic_reviewer_model_call_authority(
            self_hash_tampered,
            rederive_state_guard_sha256=lambda: binding.state_guard_sha256,
            ledger_store=_MemoryLedger(),
        )

    authority = factory()
    with pytest.raises(
        AuxiliaryModelAuthorityFactoryError,
        match="verification result",
    ):
        authority.typed_result_payload(
            model_result=ModelResult(
                reply='{"items":[]}',
                provider=authority.logical_request.provider,
                model=authority.logical_request.model,
                latency_ms=1,
                model_call_id="physical_semantic_01",
            ),
            value=_semantic_result(
                binding,
                verification_result_id="another_semantic_result",
            ),
        )


def test_semantic_factory_separates_immutable_origin_from_current_dispatch_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    origin = _semantic_binding("semantic-authority-cross-turn")
    ledger = _MemoryLedger()
    origin_authority = create_auxiliary_semantic_reviewer_model_call_authority(
        origin,
        rederive_state_guard_sha256=lambda: origin.state_guard_sha256,
        ledger_store=ledger,
    )
    origin_authority.reserve(turn_id=origin.invocation_turn_id)

    continued = origin.model_copy(
        update={
            "invocation_turn_id": "turn_semantic_dispatch_02",
            "state_guard_sha256": SHA_C,
        }
    )
    continued_authority = create_auxiliary_semantic_reviewer_model_call_authority(
        continued,
        rederive_state_guard_sha256=lambda: SHA_C,
        ledger_store=ledger,
    )

    assert continued_authority.logical_request == origin_authority.logical_request
    assert continued_authority.logical_request.invocation_turn_id == (
        origin.invocation_turn_id
    )
    assert continued_authority.dispatch_state_guard_sha256 == SHA_C
    assert continued_authority.dispatch_binding_sha256 is not None
    continued_authority.require_current_state()

    stale_dispatch = create_auxiliary_semantic_reviewer_model_call_authority(
        continued,
        rederive_state_guard_sha256=lambda: SHA_A,
        ledger_store=ledger,
    )
    with pytest.raises(DurableModelCallStateGuardRejected):
        stale_dispatch.require_current_state()

    changed_payload = json.loads(continued.request_json)
    changed_payload["verification_result_id"] = "different-semantic-result"
    changed_json = json.dumps(
        changed_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    changed_request = continued.model_copy(
        update={
            "verification_result_id": "different-semantic-result",
            "request_json": changed_json,
            "request_sha256": hashlib.sha256(
                changed_json.encode("utf-8")
            ).hexdigest(),
        }
    )
    with pytest.raises(
        AuxiliaryModelAuthorityFactoryError,
        match="dispatch lease",
    ):
        create_auxiliary_semantic_reviewer_model_call_authority(
            changed_request,
            rederive_state_guard_sha256=lambda: SHA_C,
            ledger_store=ledger,
        )


def test_work_run_factory_is_secret_free_and_binds_complete_attempt_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "anthropic")
    monkeypatch.setenv("PERSONAGRAPH_MODEL", "work-run-model-v1")
    monkeypatch.setenv("PERSONAGRAPH_API_KEY", "sk-work-run-private")
    monkeypatch.setenv(
        "PERSONAGRAPH_BASE_URL",
        "https://user:secret@work-run-private.example/v1?token=private",
    )
    binding = _work_run_binding()
    authority = create_auxiliary_work_run_model_call_authority(
        binding,
        rederive_state_guard_sha256=lambda: binding.state_guard_sha256,
        ledger_store=_MemoryLedger(),
    )
    logical = authority.logical_request

    assert logical.logical_call_id == binding.logical_call_id
    assert logical.session_id == binding.session_id
    assert logical.task_id == binding.task_id
    assert logical.auxiliary_graph_id == binding.auxiliary_graph_id
    assert logical.goal_id == binding.goal_id
    assert logical.execution_subject_id == binding.execution_subject_id
    assert logical.invocation_turn_id == binding.invocation_turn_id
    assert logical.call_kind == binding.call_kind
    assert logical.purpose == binding.purpose
    assert logical.request_contract == binding.request_contract
    assert logical.request_json == binding.request_json
    assert logical.request_sha256 == binding.request_sha256
    assert logical.output_repair_protocol is CURRENT_REPAIR_PROTOCOL
    assert logical.structured_prompt is not None
    assert logical.structured_prompt.system_prompt == binding.system_prompt
    assert logical.structured_prompt.user_content == binding.user_content
    assert logical.typed_result_contract == binding.typed_result_contract
    assert logical.max_physical_attempts == binding.max_physical_attempts
    assert logical.state_guard_sha256 == binding.state_guard_sha256
    request_payload = json.loads(logical.request_json)
    assert request_payload["authority"]["attempt_id"] == binding.attempt_id
    assert request_payload["authority"]["attempt_ordinal"] == 2
    assert request_payload["authority"]["work_run_revision"] == 7
    assert request_payload["authority"]["subject"] == binding.subject.model_dump(
        mode="json"
    )
    serialized = logical.model_dump_json()
    assert "sk-work-run-private" not in serialized
    assert "work-run-private.example" not in serialized
    assert "user:secret" not in serialized


def test_work_run_factory_exactly_replays_typed_attempt_without_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    binding = _work_run_binding()
    ledger = _MemoryLedger()
    decision = AttemptDecision(
        action=RequestUserInputAction(
            question="Which exact source should this node use?"
        )
    )
    provider_calls = 0

    def factory() -> RuntimeLogicalModelCallAuthority:
        return create_auxiliary_work_run_model_call_authority(
            binding,
            rederive_state_guard_sha256=lambda: binding.state_guard_sha256,
            ledger_store=ledger,
        )

    def provider(model_call_id: str) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return ModelResult(
            reply=decision.model_dump_json(),
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=binding.purpose,
        )

    def validate(result: ModelResult) -> AttemptDecision:
        return AttemptDecision.model_validate_json(result.reply)

    first = request_model_with_retry(
        turn_id=binding.invocation_turn_id,
        session_id=binding.session_id,
        purpose=binding.purpose,
        stage=RuntimeStage.L2_PLAN,
        prepare_request=lambda: PreparedModelCall(_dispatch=provider),
        prepare_repair_request=lambda _feedback, _rejected: PreparedModelCall(
            _dispatch=provider
        ),
        validate=validate,
        emit=lambda _event: None,
        durable_call=factory(),
    )
    replayed = request_model_with_retry(
        turn_id=binding.invocation_turn_id,
        session_id=binding.session_id,
        purpose=binding.purpose,
        stage=RuntimeStage.L2_PLAN,
        prepare_request=lambda: PreparedModelCall(
            _dispatch=lambda _model_call_id: pytest.fail("replay reached Provider")
        ),
        prepare_repair_request=lambda _feedback, _rejected: PreparedModelCall(
            _dispatch=lambda _model_call_id: pytest.fail("replay reached Provider")
        ),
        validate=validate,
        emit=lambda _event: None,
        durable_call=factory(),
    )

    assert provider_calls == 1
    assert first.value == replayed.value == decision
    assert replayed.replayed is True
    assert ledger.logical is not None
    typed = ledger.logical.physical_attempts[0].settlement.typed_result
    assert typed is not None
    assert typed.result_contract == binding.typed_result_contract
    assert typed.parsed() == decision.model_dump(mode="json")


def test_terminal_work_run_replay_restores_provider_facing_task_graph_action() -> None:
    binding = _work_run_binding(
        executor_kind=AuxiliaryNodeExecutorKind.TERMINAL_PLANNER
    )
    proposal = _terminal_task_graph_proposal()
    normalized = AttemptDecision(
        action=SubmitOutputWindowAction(
            content=proposal.model_dump_json(),
            format=OutputWindowFormat.PLAIN_TEXT,
        )
    )
    authority = create_auxiliary_work_run_model_call_authority(
        binding,
        rederive_state_guard_sha256=lambda: binding.state_guard_sha256,
        ledger_store=_MemoryLedger(),
    )

    payload = authority.typed_result_payload(
        model_result=ModelResult(
            reply="{}",
            provider=authority.logical_request.provider,
            model=authority.logical_request.model,
            latency_ms=1,
            model_call_id="physical_terminal_work_run_01",
        ),
        value=normalized,
    )

    assert payload == {
        "acceptance_updates": [],
        "action": {
            "kind": "submit_task_graph",
            "proposal": proposal.model_dump(mode="json"),
        },
    }


def test_positive_terminal_initial_response_and_durable_replay_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    base = _terminal_task_graph_base()
    candidate = _positive_terminal_candidate()
    binding = _work_run_binding(
        executor_kind=AuxiliaryNodeExecutorKind.TERMINAL_PLANNER,
        task_graph_base=base,
    )
    ledger = _MemoryLedger()
    provider_payload = {
        "acceptance_updates": [],
        "action": {
            "kind": "submit_task_graph",
            "proposal": candidate.proposal.model_dump(mode="json"),
            "lineage": [
                item.model_dump(mode="json") for item in candidate.lineage
            ],
        },
    }
    provider_reply = json.dumps(
        provider_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    provider_calls = 0

    def factory() -> RuntimeLogicalModelCallAuthority:
        return create_auxiliary_work_run_model_call_authority(
            binding,
            rederive_state_guard_sha256=lambda: binding.state_guard_sha256,
            ledger_store=ledger,
        )

    def provider(model_call_id: str) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return ModelResult(
            reply=provider_reply,
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=binding.purpose,
        )

    def validate(result: ModelResult) -> AttemptDecision:
        materialized = _materialize_terminal_action(
            json.loads(result.reply),
            context=_terminal_task_graph_context(),
            base_snapshot=base,
        )
        return AttemptDecision.model_validate(materialized)

    fresh = request_model_with_retry(
        turn_id=binding.invocation_turn_id,
        session_id=binding.session_id,
        purpose=binding.purpose,
        stage=RuntimeStage.L2_PLAN,
        prepare_request=lambda: PreparedModelCall(_dispatch=provider),
        prepare_repair_request=lambda _feedback, _rejected: PreparedModelCall(
            _dispatch=provider
        ),
        validate=validate,
        emit=lambda _event: None,
        durable_call=factory(),
    )
    replayed = request_model_with_retry(
        turn_id=binding.invocation_turn_id,
        session_id=binding.session_id,
        purpose=binding.purpose,
        stage=RuntimeStage.L2_PLAN,
        prepare_request=lambda: PreparedModelCall(
            _dispatch=lambda _model_call_id: pytest.fail("replay reached Provider")
        ),
        prepare_repair_request=lambda _feedback, _rejected: PreparedModelCall(
            _dispatch=lambda _model_call_id: pytest.fail("replay reached Provider")
        ),
        validate=validate,
        emit=lambda _event: None,
        durable_call=factory(),
    )

    assert provider_calls == 1
    assert fresh.replayed is False
    assert replayed.replayed is True
    assert fresh.value == replayed.value
    assert isinstance(fresh.value.action, SubmitOutputWindowAction)
    assert fresh.value.action.content == candidate.model_dump_json()
    assert ledger.logical is not None
    typed = ledger.logical.physical_attempts[0].settlement.typed_result
    assert typed is not None
    assert typed.parsed() == provider_payload


def test_positive_terminal_typed_replay_rejects_bare_and_tampered_candidates() -> None:
    base = _terminal_task_graph_base()
    binding = _work_run_binding(
        executor_kind=AuxiliaryNodeExecutorKind.TERMINAL_PLANNER,
        task_graph_base=base,
    )
    authority = create_auxiliary_work_run_model_call_authority(
        binding,
        rederive_state_guard_sha256=lambda: binding.state_guard_sha256,
        ledger_store=_MemoryLedger(),
    )
    model_result = ModelResult(
        reply="{}",
        provider=authority.logical_request.provider,
        model=authority.logical_request.model,
        latency_ms=1,
        model_call_id="physical_positive_terminal_guard_01",
    )

    bare = AttemptDecision(
        action=SubmitOutputWindowAction(
            content=_terminal_task_graph_proposal().model_dump_json(),
            format=OutputWindowFormat.PLAIN_TEXT,
        )
    )
    with pytest.raises(
        AuxiliaryModelAuthorityFactoryError,
        match="proposal failed fresh validation",
    ):
        authority.typed_result_payload(model_result=model_result, value=bare)

    unknown_lineage = _positive_terminal_candidate(
        base_node_alias="unprojected_base"
    )
    crossed = AttemptDecision(
        action=SubmitOutputWindowAction(
            content=unknown_lineage.model_dump_json(),
            format=OutputWindowFormat.PLAIN_TEXT,
        )
    )
    with pytest.raises(
        AuxiliaryModelAuthorityFactoryError,
        match="proposal failed fresh validation",
    ):
        authority.typed_result_payload(model_result=model_result, value=crossed)

    candidate = _positive_terminal_candidate()
    noncanonical = AttemptDecision(
        action=SubmitOutputWindowAction(
            content=json.dumps(
                candidate.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            ),
            format=OutputWindowFormat.PLAIN_TEXT,
        )
    )
    with pytest.raises(
        AuxiliaryModelAuthorityFactoryError,
        match="proposal failed fresh validation",
    ):
        authority.typed_result_payload(
            model_result=model_result,
            value=noncanonical,
        )


def test_base_null_terminal_typed_replay_rejects_positive_candidate() -> None:
    binding = _work_run_binding(
        executor_kind=AuxiliaryNodeExecutorKind.TERMINAL_PLANNER
    )
    authority = create_auxiliary_work_run_model_call_authority(
        binding,
        rederive_state_guard_sha256=lambda: binding.state_guard_sha256,
        ledger_store=_MemoryLedger(),
    )
    decision = AttemptDecision(
        action=SubmitOutputWindowAction(
            content=_positive_terminal_candidate().model_dump_json(),
            format=OutputWindowFormat.PLAIN_TEXT,
        )
    )
    with pytest.raises(
        AuxiliaryModelAuthorityFactoryError,
        match="proposal failed fresh validation",
    ):
        authority.typed_result_payload(
            model_result=ModelResult(
                reply="{}",
                provider=authority.logical_request.provider,
                model=authority.logical_request.model,
                latency_ms=1,
                model_call_id="physical_base_null_candidate_guard_01",
            ),
            value=decision,
        )


def test_work_run_factory_rejects_state_drift_and_self_hash_tamper_before_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    binding = _work_run_binding()
    authority = create_auxiliary_work_run_model_call_authority(
        binding,
        rederive_state_guard_sha256=lambda: SHA_B,
        ledger_store=_MemoryLedger(),
    )
    provider_calls = 0

    def provider(_model_call_id: str) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("state drift reached Provider")

    with pytest.raises(DurableModelCallStateGuardRejected):
        request_model_with_retry(
            turn_id=binding.invocation_turn_id,
            session_id=binding.session_id,
            purpose=binding.purpose,
            stage=RuntimeStage.L2_PLAN,
            prepare_request=lambda: PreparedModelCall(_dispatch=provider),
            prepare_repair_request=lambda _feedback, _rejected: PreparedModelCall(
                _dispatch=provider
            ),
            validate=lambda result: AttemptDecision.model_validate_json(
                result.reply
            ),
            emit=lambda _event: None,
            durable_call=authority,
        )
    assert provider_calls == 0

    tampered = binding.model_copy(update={"work_run_revision": 8})
    with pytest.raises(
        AuxiliaryModelAuthorityFactoryError,
        match="WorkRun binding",
    ):
        create_auxiliary_work_run_model_call_authority(
            tampered,
            rederive_state_guard_sha256=lambda: binding.state_guard_sha256,
            ledger_store=_MemoryLedger(),
        )


def test_work_run_factory_reuses_only_exact_semantic_request_across_dispatch_leases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    origin = _work_run_binding()
    ledger = _MemoryLedger()
    origin_authority = create_auxiliary_work_run_model_call_authority(
        origin,
        rederive_state_guard_sha256=lambda: origin.state_guard_sha256,
        ledger_store=ledger,
    )
    origin_authority.reserve(turn_id=origin.invocation_turn_id)

    continued_payload = json.loads(origin.user_content)
    continued_payload["bindings"]["turn_id"] = "turn_work_run_invocation_02"
    continued_payload["bindings"]["work_run_revision"] = 8
    continued_user_content = json.dumps(
        continued_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    continued = AuxiliaryBoundModelCall.create(
        call_kind="attempt_decision",
        logical_call_id=origin.logical_call_id,
        session_id=origin.session_id,
        goal_id=origin.goal_id,
        subject=origin.subject,
        executor_kind=origin.executor_kind,
        request_turn_id=origin.request_turn_id,
        invocation_turn_id="turn_work_run_invocation_02",
        work_run_id=origin.work_run_id,
        work_run_revision=8,
        attempt_id=origin.attempt_id,
        attempt_ordinal=origin.attempt_ordinal,
        verification_request_id=None,
        verification_request_revision=None,
        system_prompt=origin.system_prompt,
        user_content=continued_user_content,
        state_guard_sha256=SHA_B,
    )
    continued_authority = create_auxiliary_work_run_model_call_authority(
        continued,
        rederive_state_guard_sha256=lambda: SHA_B,
        ledger_store=ledger,
    )

    assert continued_authority.logical_request == origin_authority.logical_request
    assert continued_authority.dispatch_binding_sha256 == continued.binding_sha256
    assert continued_authority.dispatch_state_guard_sha256 == SHA_B
    continued_authority.require_current_state()
    stale_guard = create_auxiliary_work_run_model_call_authority(
        continued,
        rederive_state_guard_sha256=lambda: SHA_A,
        ledger_store=ledger,
    )
    with pytest.raises(DurableModelCallStateGuardRejected):
        stale_guard.require_current_state()

    drifted_payload = json.loads(continued_user_content)
    drifted_payload["node"] = {"acceptances": [], "objective": "changed"}
    drifted_user_content = json.dumps(
        drifted_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    drifted = AuxiliaryBoundModelCall.create(
        call_kind="attempt_decision",
        logical_call_id=origin.logical_call_id,
        session_id=origin.session_id,
        goal_id=origin.goal_id,
        subject=origin.subject,
        executor_kind=origin.executor_kind,
        request_turn_id=origin.request_turn_id,
        invocation_turn_id="turn_work_run_invocation_02",
        work_run_id=origin.work_run_id,
        work_run_revision=8,
        attempt_id=origin.attempt_id,
        attempt_ordinal=origin.attempt_ordinal,
        verification_request_id=None,
        verification_request_revision=None,
        system_prompt=origin.system_prompt,
        user_content=drifted_user_content,
        state_guard_sha256=SHA_B,
    )
    with pytest.raises(
        AuxiliaryModelAuthorityFactoryError,
        match="dispatch lease",
    ):
        create_auxiliary_work_run_model_call_authority(
            drifted,
            rederive_state_guard_sha256=lambda: SHA_B,
            ledger_store=ledger,
        )


def test_work_run_verification_replay_and_controller_strict_runtime_match() -> None:
    binding = _work_run_binding(call_kind="node_verification")
    authority = create_auxiliary_work_run_model_call_authority(
        binding,
        rederive_state_guard_sha256=lambda: binding.state_guard_sha256,
        ledger_store=_MemoryLedger(),
    )
    result = NodeVerificationResult(
        verification_request_id=binding.verification_request_id,
        verification_request_revision=binding.verification_request_revision,
        work_run_id=binding.work_run_id,
        locked_work_run_revision=binding.work_run_revision,
        submitted_attempt_id=binding.attempt_id,
        acceptance_progress_revision=3,
        subject=binding.subject,
        output_revision=2,
        acceptance_results=(
            AcceptanceVerificationFeedback(
                acceptance_id="acceptance_work_run_authority_01",
                verdict=VerificationVerdict.PASSED,
                finding="The locked output satisfies the criterion.",
            ),
        ),
        all_pass=True,
    )
    payload = authority.typed_result_payload(
        model_result=ModelResult(
            reply='{"acceptance_results":[]}',
            provider=authority.logical_request.provider,
            model=authority.logical_request.model,
            latency_ms=1,
            model_call_id="physical_work_run_verification_01",
        ),
        value=result,
    )
    assert payload == {
        "acceptance_results": [
            item.model_dump(mode="json") for item in result.acceptance_results
        ]
    }

    crossed = RuntimeLogicalModelCallAuthority(
        logical_request=authority.logical_request.model_copy(
            update={"task_id": "another_task"}
        ),
        state_guard_sha256=lambda: binding.state_guard_sha256,
        typed_replay_payload_builder=lambda _result, value: value,
        store=_MemoryLedger(),
    )
    with pytest.raises(
        TypeError,
        match="differs from WorkRun binding",
    ):
        _bind_model_call_authority(
            factory=lambda _binding, **_kwargs: crossed,
            binding=binding,
            rederive=lambda: binding.state_guard_sha256,
        )
