"""现行 Auxiliary WorkRun controller 的行为与边界测试。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from functools import partial
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from personagraph.l2.auxiliary_graph import (
    TaskGraphRevisionCandidate,
    TaskGraphSemanticBaseSnapshot,
)
from personagraph.l2.auxiliary_graph.dependency_projection import (
    AuxiliaryDependencyBundle,
    AuxiliaryDependencyInputLimits,
    AuxiliaryDependencyInputUnsupported,
)
from personagraph.l2.task_graph.contracts import (
    InSessionTaskAcceptanceProposal,
    InSessionTaskGraphRevisionProposal,
    InSessionTaskGraphRevisionValidationContext,
    InSessionTaskSourceAnchor,
)
from personagraph.l2.task_graph.task_matching import InSessionTaskMatchesProposal
from personagraph.model_io.gateway import ModelGatewayError, ModelResult, PreparedModelCall
from personagraph.l2.auxiliary_execution.work_run import (
    controller as auxiliary_work_run_controller_module,
)
from personagraph.l2.task_execution.attempts.decision import (
    AttemptDecisionContext,
    AttemptDecisionInputLimits,
    AttemptUserInput,
    PriorToolResultProjection,
    PriorToolResultsProjection,
)
from personagraph.l2.auxiliary_execution.driver import (
    canonical_auxiliary_graph_driver_state_guard,
)
from personagraph.l2.auxiliary_execution.work_run.controller import (
    AuxiliaryWorkRunStatus,
    AuxiliaryWorkRunRequest,
    AuxiliaryWorkRunProfile,
    AuxiliaryWorkRunIdPlan,
    derive_auxiliary_work_run_ids,
    run_auxiliary_model_node,
)
from personagraph.l2.auxiliary_execution.adapters.model_authority import (
    create_auxiliary_work_run_model_call_authority,
)
from personagraph.runtime.model_calls import (
    DurableModelCallStateGuardRejected,
    DurableModelCallTerminalState,
)
from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairProtocol,
    RuntimeModelStructuredPrompt,
)
from personagraph.runtime.turn_deadline import TurnDeadline
from personagraph.model_io.output_validation import ModelOutputValidationError
from personagraph.tools.policy import (
    ProtectedToolExecutionAuthority,
)
from personagraph.l2.task_execution.tool_bridge.protected_dispatch import (
    RuntimeProtectedToolDispatcher,
)
from personagraph.l2.task_execution.tool_bridge.persistence_contracts import (
    ToolBridgeCallPersistence,
    ToolBridgePersistencePlan,
)
from personagraph.l2.task_execution.tool_bridge.work_run_bridge import SqliteWorkRunToolBridge
from personagraph.persistent_turn_content.findings import EXECUTION_FINDINGS_TOOL_IDS
from personagraph.l2.task_execution.verification.decision import NodeVerificationInputLimits
from personagraph.l2.task_execution.task_node.dependencies import (
    TaskNodeDependencyDeliveries,
    TaskNodeDependencyInputLimits,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import terminal as terminal_store
from personagraph.session.l2_store import continuation as continuation_store
from personagraph.session.l2_store import verification as verification_store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import planning as planning_store
from personagraph.session.persistence.l2.auxiliary_graph import auxiliary_graphs
from personagraph.session.persistence.l2.task_graph import insession_tasks as insession_task_records
from personagraph.tools.catalog import CatalogSnapshot, ToolCatalog
from personagraph.tools.contracts import (
    ToolSourceDescriptor,
    ToolSourceKind,
    ToolSpec,
)
from personagraph.tools.effects import (
    DataEgress,
    EffectAction,
    EffectDescriptor,
    EffectResource,
    EffectScopeKind,
    ToolEffectProfile,
)
from personagraph.tools.policy import AuthorityFacts, ScopeGrant
from personagraph.tools.registration import ToolExecutionProfile, ToolRegistration
from personagraph.l2.work_run import (
    AuxiliaryNodeSubject,
    DownstreamVerificationDisposition,
    DownstreamVerificationFeedback,
    OutputWindow,
    ToolResultStatus,
    ToolResult,
    WorkRunStatus,
    initialize_acceptance_progress,
)
from tests.session.test_planning_artifact_seal_persistence import _primitive
from tests.helpers.prepared_model_provider import as_prepared_test_provider


USER_TEXT = "请分析材料并形成一个可执行计划"


def test_model_work_run_prompt_uses_toolspec_and_generic_coverage_state() -> None:
    prompt = auxiliary_work_run_controller_module._MODEL_WORK_RUN_SYSTEM_PROMPT

    assert "各 ToolSpec 的 description、input_schema、output_schema" in prompt
    assert "cursor、coverage、truncated 或 complete" in prompt
    assert "不得把一次局部观察默认当作完整覆盖" in prompt
    assert "inspect_mounted_document" not in prompt
    assert "search_mounted_document" not in prompt
    assert "read_mounted_document_chunks" not in prompt


def test_active_auxiliary_prompts_do_not_advertise_failure_only_feedback() -> None:
    for prompt in (
        auxiliary_work_run_controller_module._MODEL_WORK_RUN_SYSTEM_PROMPT,
        auxiliary_work_run_controller_module._USER_GATE_SYSTEM_PROMPT,
        auxiliary_work_run_controller_module._TERMINAL_PLANNER_SYSTEM_PROMPT,
        auxiliary_work_run_controller_module._POSITIVE_BASE_TERMINAL_PLANNER_SYSTEM_PROMPT,
        auxiliary_work_run_controller_module._VERIFICATION_SYSTEM_PROMPT,
    ):
        assert "host_repair_feedback" not in prompt


def test_terminal_planner_prompts_require_minimal_sufficient_irreducible_graph() -> None:
    for prompt in (
        auxiliary_work_run_controller_module._TERMINAL_PLANNER_SYSTEM_PROMPT,
        auxiliary_work_run_controller_module._POSITIVE_BASE_TERMINAL_PLANNER_SYSTEM_PROMPT,
    ):
        assert "最少充分节点" in prompt
        assert "最短必要依赖链" in prompt
        assert "无法安全并入" in prompt
        assert "固定阶段模板" in prompt


def test_standard_profile_and_node_identity_plan_are_production_reentrant() -> None:
    profile = AuxiliaryWorkRunProfile()
    assert profile.dependency_input_limits.max_items == 64
    assert profile.attempt_input_limits.dependency_delivery_limits.max_items == 0

    subject = AuxiliaryNodeSubject(
        task_id="task-01",
        auxiliary_graph_id="aux-graph-01",
        auxiliary_graph_revision=2,
        node_id="node-01",
        node_revision=3,
    )
    first = derive_auxiliary_work_run_ids(
        session_id="session-01",
        subject=subject,
    )
    replay = derive_auxiliary_work_run_ids(
        session_id="session-01",
        subject=subject,
    )
    next_revision = derive_auxiliary_work_run_ids(
        session_id="session-01",
        subject=subject.model_copy(update={"node_revision": 4}),
    )

    assert first == replay
    assert first != next_revision
    assert len(set(first.model_dump().values())) == len(first.model_dump())
    assert all(len(value) <= 200 for value in first.model_dump().values())
    assert first.for_attempt(first.attempt_id, 1) == first.attempt_id
    assert first.for_attempt(first.attempt_id, 2).endswith(":attempt-2")


class _ReplyProvider:
    def __init__(
        self,
        replies: Iterable[str | BaseException],
        *,
        provider: str = "fake",
        model: str = "fake-model",
    ) -> None:
        self.replies = list(replies)
        self.provider = provider
        self.model = model
        self.calls: list[tuple[str, str, str]] = []
        self.system_prompts: list[str] = []
        self.prepared_calls: list[dict[str, object]] = []

    def prepare(
        self,
        system_prompt: str,
        user_content: str,
        *,
        purpose: str,
        repair_messages: list[dict[str, str]] | None = None,
    ) -> PreparedModelCall:
        prepared: dict[str, object] = {
            "system_prompt": system_prompt,
            "user_content": user_content,
            "purpose": purpose,
        }
        if repair_messages is not None:
            prepared["repair_messages"] = repair_messages
        self.prepared_calls.append(prepared)

        def dispatch(model_call_id: str | None) -> ModelResult:
            assert model_call_id is not None
            return self(
                system_prompt,
                user_content,
                model_call_id=model_call_id,
                purpose=purpose,
            )

        return PreparedModelCall(_dispatch=dispatch)

    def __call__(
        self,
        system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        self.system_prompts.append(system_prompt)
        self.calls.append((model_call_id, purpose, user_content))
        if not self.replies:
            raise AssertionError("provider received an unexpected call")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return ModelResult(
            reply=reply,
            provider=self.provider,
            model=self.model,
            latency_ms=1,
            model_call_id=model_call_id,
            finish_reason="stop",
        )


class _PreparedReplyProvider:
    def __init__(
        self,
        replies: Iterable[str | BaseException],
        *,
        provider: str = "fake",
        model: str = "fake-model",
    ) -> None:
        self.replies = list(replies)
        self.provider = provider
        self.model = model
        self.calls: list[tuple[str, str, str]] = []
        self.system_prompts: list[str] = []
        self.prepared_calls: list[dict[str, object]] = []

    def __call__(self, *_args: object, **_kwargs: object) -> ModelResult:
        raise AssertionError("prepared provider must not use its legacy path")

    def prepare(
        self,
        system_prompt: str,
        user_content: str,
        *,
        purpose: str,
        repair_messages: list[dict[str, str]] | None = None,
    ) -> object:
        prepared: dict[str, object] = {
            "system_prompt": system_prompt,
            "user_content": user_content,
            "purpose": purpose,
        }
        if repair_messages is not None:
            prepared["repair_messages"] = repair_messages
        self.prepared_calls.append(prepared)
        owner = self

        class _Prepared:
            def dispatch(self, *, model_call_id: str) -> ModelResult:
                owner.system_prompts.append(system_prompt)
                owner.calls.append((model_call_id, purpose, user_content))
                if not owner.replies:
                    raise AssertionError("provider received an unexpected call")
                reply = owner.replies.pop(0)
                if isinstance(reply, BaseException):
                    raise reply
                return ModelResult(
                    reply=reply,
                    provider=owner.provider,
                    model=owner.model,
                    latency_ms=1,
                    model_call_id=model_call_id,
                    finish_reason="stop",
                )

        return _Prepared()


class _PassVerifier:
    def __init__(
        self,
        failures: Iterable[BaseException] = (),
        verdicts: Iterable[str] = (),
        *,
        provider: str = "fake",
        model: str = "fake-verifier",
    ) -> None:
        self.failures = list(failures)
        self.verdicts = list(verdicts)
        self.provider = provider
        self.model = model
        self.calls: list[tuple[str, str, str]] = []
        self.system_prompts: list[str] = []

    def prepare(
        self,
        system_prompt: str,
        user_content: str,
        *,
        purpose: str,
        repair_messages: list[dict[str, str]] | None = None,
    ) -> PreparedModelCall:
        del repair_messages

        def dispatch(model_call_id: str | None) -> ModelResult:
            assert model_call_id is not None
            return self(
                system_prompt,
                user_content,
                model_call_id=model_call_id,
                purpose=purpose,
            )

        return PreparedModelCall(_dispatch=dispatch)

    def __call__(
        self,
        system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        self.system_prompts.append(system_prompt)
        self.calls.append((model_call_id, purpose, user_content))
        if self.failures:
            raise self.failures.pop(0)
        prompt = json.loads(user_content)
        verdict = self.verdicts.pop(0) if self.verdicts else "passed"
        acceptance_ids = [
            item["acceptance_id"] for item in prompt["node"]["acceptances"]
        ]
        reply = {
            "acceptance_results": [
                {
                    "acceptance_id": acceptance_id,
                    "verdict": verdict,
                    "finding": "锁定输出满足此项验收条件。",
                    "missing_requirements": (
                        [] if verdict == "passed" else ["需要修订输出。"]
                    ),
                }
                for acceptance_id in acceptance_ids
            ]
        }
        return ModelResult(
            reply=json.dumps(reply, ensure_ascii=False),
            provider=self.provider,
            model=self.model,
            latency_ms=1,
            model_call_id=model_call_id,
            finish_reason="stop",
        )


class _RevalidatingAuthority:
    def __init__(self, binding, rederive) -> None:
        self.semantic_call_id = binding.logical_call_id
        self.logical_request = SimpleNamespace(
            structured_prompt=RuntimeModelStructuredPrompt.create(
                system_prompt=binding.system_prompt,
                user_content=binding.user_content,
            ),
            output_repair_protocol=(
                RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
            ),
            typed_result_contract=binding.typed_result_contract,
        )
        self._expected = binding.state_guard_sha256
        self._rederive = rederive
        self._physical_ordinal = 0
        self._repair_feedback = None

    def require_current_state(self) -> None:
        if self._rederive() != self._expected:
            raise DurableModelCallStateGuardRejected("test dependency drift")

    def reserve(self, *, turn_id: str):
        return {"turn_id": turn_id}

    def replay_succeeded_result(self):
        return None

    def begin_physical_attempt(
        self,
        *,
        turn_id: str,
        max_physical_attempts: int,
        output_repair_enabled: bool = False,
        output_repair_feedback=None,
    ):
        assert output_repair_enabled is True
        assert output_repair_feedback == self._repair_feedback
        self._physical_ordinal += 1
        authority = self

        class _Physical:
            physical_attempt_id = (
                f"{authority.semantic_call_id}:physical:{authority._physical_ordinal}"
            )
            physical_ordinal = authority._physical_ordinal
            provider = "fake"
            model = "fake-model"

            @property
            def model_call_id(self) -> str:
                return self.physical_attempt_id

        assert self._physical_ordinal <= max_physical_attempts
        assert turn_id
        return _Physical()

    def settle_physical_attempt(self, **kwargs):
        if kwargs.get("next_output_repair_feedback") is not None:
            self._repair_feedback = kwargs["next_output_repair_feedback"]
        return kwargs

    def recover_output_repair_feedback(self):
        return self._repair_feedback

    def terminal_state_error(self, message: str):
        return DurableModelCallTerminalState(message)

    def typed_result_payload(self, *, model_result, value):
        return {"model_result": model_result, "value": value}

    def success_fingerprint(self, result):
        return str(result)

    def failure_fingerprint(self, *, error, provider_result):
        return f"{error}:{provider_result}"


def _window_revision(session_id: str) -> int:
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    return int(window["state_version"])


def _clock():
    current = 0.0

    def tick() -> float:
        nonlocal current
        current += 1.0
        return current

    return tick


def _seed_task() -> tuple[str, str, str]:
    session_id = store.create_session("Entelecheia")
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="aux-v2-controller",
        source="auxiliary_v2_work_run_controller_test",
        user_text=USER_TEXT,
        lease_owner="aux-v2-controller-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    applied = task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=turn_id,
        apply_id="aux-v2-controller-task",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "new_root",
                        "local_key": "root",
                        "title": "分析材料",
                        "objective": "分析材料并形成执行计划",
                        "source_excerpt": USER_TEXT,
                    }
                ]
            }
        ),
        exposed_catalog_ids=(),
        expected_window_revision=_window_revision(session_id),
    )
    return session_id, turn_id, applied.created_insession_task_ids_by_local_key["root"]


def _acceptance() -> InSessionTaskAcceptanceProposal:
    return InSessionTaskAcceptanceProposal(
        acceptance_id="grounded",
        criterion="输出必须与任务目标一致且可核对",
        source_anchor_ids=("task_creation_source",),
    )


def _commit_graph(
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
    terminal_only: bool = False,
) -> None:
    acceptance = _acceptance()
    terminal = auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
        local_node_key="synthesize",
        node_kind="synthesize",
        executor_kind="terminal_planner",
        title="形成任务图",
        objective="形成一个完整、来源受约束的 TaskGraph 提案",
        source_anchor_ids=("task_creation_source",),
        acceptance_criteria=(acceptance,),
        output_contract="task_graph_revision_proposal_v2",
    )
    nodes = (terminal,)
    edges: tuple[auxiliary_graphs.AuxiliaryGraphEdgeProposalRecord, ...] = ()
    if not terminal_only:
        investigate = auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
            local_node_key="investigate",
            node_kind="analyze",
            executor_kind="model_work_run",
            title="调查材料",
            objective="输出一份与任务目标一致的材料调查结论",
            source_anchor_ids=("task_creation_source",),
            acceptance_criteria=(acceptance,),
            output_contract="planning_context_v1",
            capability_profile_id="readonly_documents_v1",
        )
        nodes = (investigate, terminal)
        edges = (
            auxiliary_graphs.AuxiliaryGraphEdgeProposalRecord(
                dependency_node_key="investigate",
                consumer_node_key="synthesize",
            ),
        )
    auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=1,
        expected_base_task_graph_revision=None,
        expected_control_state_version=None,
        expected_current_auxiliary_graph_revision=None,
        apply_id="aux-v2-controller-graph",
        goal_objective="形成受来源约束的可执行任务图",
        proposal=auxiliary_graphs.AuxiliaryGraphRevisionProposalRecord(
            revision_reason="initial",
            terminal_node_key="synthesize",
            nodes=nodes,
            edges=edges,
        ),
        authority_context={"anchors": []},
        budget_profile={"profile_id": "planning-test-v1"},
        auxiliary_graph_id="aux-v2-controller-graph",
        goal_id="aux-v2-controller-goal",
    )


def _commit_user_gate_graph(
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
) -> None:
    acceptance = _acceptance()
    clarify = auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
        local_node_key="clarify",
        node_kind="clarify",
        executor_kind="user_gate",
        title="补齐阻塞信息",
        objective="请明确希望计划覆盖最近几年。",
        source_anchor_ids=("task_creation_source",),
        acceptance_criteria=(acceptance,),
        output_contract="user_response_v1",
    )
    terminal = auxiliary_graphs.AuxiliaryGraphNodeProposalRecord(
        local_node_key="synthesize",
        node_kind="synthesize",
        executor_kind="terminal_planner",
        title="形成任务图",
        objective="使用用户补充的信息形成完整 TaskGraph 提案",
        source_anchor_ids=("task_creation_source",),
        acceptance_criteria=(acceptance,),
        output_contract="task_graph_revision_proposal_v2",
    )
    auxiliary_graphs.commit_auxiliary_graph_revision(
        store._deps(),
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
        expected_task_state_version=1,
        expected_base_task_graph_revision=None,
        expected_control_state_version=None,
        expected_current_auxiliary_graph_revision=None,
        apply_id="aux-v2-controller-user-gate-graph",
        goal_objective="形成受用户补充信息约束的可执行任务图",
        proposal=auxiliary_graphs.AuxiliaryGraphRevisionProposalRecord(
            revision_reason="initial",
            terminal_node_key="synthesize",
            nodes=(clarify, terminal),
            edges=(
                auxiliary_graphs.AuxiliaryGraphEdgeProposalRecord(
                    dependency_node_key="clarify",
                    consumer_node_key="synthesize",
                ),
            ),
        ),
        authority_context={"anchors": []},
        budget_profile={"profile_id": "planning-test-v1"},
        auxiliary_graph_id="aux-v2-controller-user-gate-graph",
        goal_id="aux-v2-controller-user-gate-goal",
    )


def _subject(
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
    executor: str,
) -> AuxiliaryNodeSubject:
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    match = next(
        item for item in frontier.ready_fresh if item.executor_kind.value == executor
    )
    return match.subject


def _profile() -> AuxiliaryWorkRunProfile:
    dependency = TaskNodeDependencyInputLimits(
        profile_id="aux-v2-dependencies",
        max_items=0,
        max_serialized_utf8_bytes=1024,
    )
    return AuxiliaryWorkRunProfile(
        attempt_input_limits=AttemptDecisionInputLimits(
            profile_id="aux-v2-attempt",
            max_prior_tool_result_items=16,
            max_prior_tool_results_serialized_utf8_bytes=64_000,
            dependency_delivery_limits=dependency,
            max_serialized_utf8_bytes=256_000,
        ),
        verification_input_limits=NodeVerificationInputLimits(
            profile_id="aux-v2-verification",
            max_acceptance_items=64,
            max_supporting_tool_result_items=32,
            dependency_delivery_limits=dependency,
            max_serialized_utf8_bytes=256_000,
        ),
        dependency_input_limits=AuxiliaryDependencyInputLimits(
            profile_id="aux-v2-dependency-input",
            max_items=64,
            max_serialized_utf8_bytes=256_000,
        ),
    )


@pytest.mark.parametrize(
    "status",
    (
        ToolResultStatus.REJECTED,
        ToolResultStatus.FAILED,
        ToolResultStatus.TIMED_OUT,
        ToolResultStatus.CANCELLED,
        ToolResultStatus.COMPLETION_UNCONFIRMED,
    ),
)
def test_attempt_parser_rejects_non_succeeded_result_as_acceptance_support(
    status: ToolResultStatus,
) -> None:
    subject = AuxiliaryNodeSubject(
        task_id="task-01",
        auxiliary_graph_id="aux-graph-01",
        auxiliary_graph_revision=2,
        node_id="node-01",
        node_revision=1,
    )
    acceptance = InSessionTaskAcceptanceProposal(
        acceptance_id="grounded",
        criterion="结论必须有成功工具结果支持",
        source_anchor_ids=("task_creation_source",),
    )
    profile = _profile()
    context = AttemptDecisionContext(
        session_id="session-01",
        turn_id="turn-01",
        work_run_id="work-run-01",
        work_run_revision=6,
        attempt_id="attempt-02",
        attempt_ordinal=2,
        user_input=AttemptUserInput(content=USER_TEXT),
        subject=subject,
        node_title="调查材料",
        node_objective="形成有证据支持的调查结论",
        acceptances=(acceptance,),
        acceptance_progress=initialize_acceptance_progress(
            work_run_id="work-run-01",
            subject=subject,
            acceptance_ids=(acceptance.acceptance_id,),
        ),
        output_window=OutputWindow(
            work_run_id="work-run-01",
            updated_turn_id="turn-01",
        ),
        dependency_deliveries=TaskNodeDependencyDeliveries(),
        prior_tool_results=PriorToolResultsProjection(
            items=(
                PriorToolResultProjection(
                    tool_id="document.read",
                    tool_version="1.0.0",
                    result=ToolResult(
                        status=status,
                        tool_result_id="result-failed",
                        tool_call_id="call-failed",
                        attempt_id="attempt-01",
                        ordinal=1,
                        output=None,
                        error_code="tool_did_not_succeed",
                        error_message="The tool did not produce successful evidence.",
                    ),
                ),
            )
        ),
        input_limits=profile.attempt_input_limits,
        allowed_tools=(),
    )
    reply = json.dumps(
        {
            "acceptance_updates": [
                {
                    "acceptance_id": "grounded",
                    "model_claimed_satisfied": True,
                    "supporting_tool_result_ids": ["result-failed"],
                }
            ],
            "action": {
                "kind": "request_user_input",
                "question": "请补充材料。",
            },
        },
        ensure_ascii=False,
    )

    with pytest.raises(ModelOutputValidationError, match="AcceptanceProgress"):
        auxiliary_work_run_controller_module._parse_attempt_decision(
            reply,
            context=context,
            terminal=False,
            task_graph_context=None,
        )


def test_auxiliary_attempt_prompt_compacts_historical_findings_snapshots() -> None:
    old_claim = "this evicted finding must not re-enter the model view"
    subject = AuxiliaryNodeSubject(
        task_id="task-01",
        auxiliary_graph_id="aux-graph-01",
        auxiliary_graph_revision=2,
        node_id="node-01",
        node_revision=1,
    )
    acceptance = InSessionTaskAcceptanceProposal(
        acceptance_id="grounded",
        criterion="结论必须有成功工具结果支持",
        source_anchor_ids=("task_creation_source",),
    )
    context = AttemptDecisionContext(
        session_id="session-01",
        turn_id="turn-01",
        work_run_id="work-run-01",
        work_run_revision=2,
        attempt_id="attempt-02",
        attempt_ordinal=2,
        user_input=AttemptUserInput(content=USER_TEXT),
        subject=subject,
        node_title="调查材料",
        node_objective="形成有证据支持的调查结论",
        acceptances=(acceptance,),
        acceptance_progress=initialize_acceptance_progress(
            work_run_id="work-run-01",
            subject=subject,
            acceptance_ids=(acceptance.acceptance_id,),
        ),
        output_window=OutputWindow(
            work_run_id="work-run-01",
            updated_turn_id="turn-01",
        ),
        dependency_deliveries=TaskNodeDependencyDeliveries(),
        prior_tool_results=PriorToolResultsProjection(
            items=(
                PriorToolResultProjection(
                    tool_id="record_execution_findings",
                    tool_version="1.3.0",
                    result=ToolResult(
                        status=ToolResultStatus.SUCCEEDED,
                        tool_result_id="result-findings",
                        tool_call_id="call-findings",
                        attempt_id="attempt-01",
                        ordinal=1,
                        output={
                            "schema_version": "execution-findings-tool-result-v1",
                            "active_projection": {
                                "projection_sha256": "c" * 64,
                                "active_entries": [{"claim": old_claim}],
                            },
                        },
                    ),
                ),
            )
        ),
        input_limits=_profile().attempt_input_limits,
        allowed_tools=(),
    )

    payload = auxiliary_work_run_controller_module._attempt_prompt_payload(
        context,
        terminal=False,
        task_graph_context=None,
        task_graph_semantic_base_snapshot=None,
        auxiliary_dependency_payload={"items": []},
    )

    assert payload["prior_tool_results"]["items"][0]["output"][
        "active_projection"
    ] == {
        "schema_version": "execution-findings-active-projection-reference-v1",
        "compacted": True,
        "projection_sha256": "c" * 64,
    }
    assert old_claim not in json.dumps(payload)


def _ids(prefix: str) -> AuxiliaryWorkRunIdPlan:
    return AuxiliaryWorkRunIdPlan(
        work_run_id=f"{prefix}-run",
        create_work_run_apply_id=f"{prefix}-create",
        attempt_id=f"{prefix}-attempt",
        start_attempt_apply_id=f"{prefix}-start",
        attempt_decision_apply_id=f"{prefix}-decide",
        attempt_model_call_id=f"{prefix}-attempt-model",
        prepare_verification_apply_id=f"{prefix}-prepare",
        verification_request_id=f"{prefix}-verification",
        verification_model_call_id=f"{prefix}-verification-model",
        commit_verification_apply_id=f"{prefix}-settle",
        completion_id=f"{prefix}-completion",
    )


def _context(
    *, session_id: str, turn_id: str, task_id: str
) -> InSessionTaskGraphRevisionValidationContext:
    return InSessionTaskGraphRevisionValidationContext(
        session_id=session_id,
        source_turn_id=turn_id,
        target_insession_task_id=task_id,
        expected_current_graph_revision=None,
        source_anchors=(
            InSessionTaskSourceAnchor(
                anchor_id="task_creation_source",
                source_turn_id=turn_id,
                source_kind="previously_authorized_task_state",
                start=0,
                end=len(USER_TEXT),
                excerpt=USER_TEXT,
            ),
        ),
        authorization_anchor_ids=("task_creation_source",),
        required_anchor_ids=("task_creation_source",),
    )


def _positive_context(
    *, session_id: str, turn_id: str, task_id: str
) -> InSessionTaskGraphRevisionValidationContext:
    return _context(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    ).model_copy(update={"expected_current_graph_revision": 1})


def _semantic_base_snapshot() -> TaskGraphSemanticBaseSnapshot:
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
                    InSessionTaskAcceptanceProposal(
                        acceptance_id="plan_ready",
                        criterion="The plan is complete and reviewable.",
                        source_anchor_ids=("task_creation_source",),
                    ),
                ),
                "constraints": (),
                "status": "proposed",
            },
        ),
        source_snapshot_sha256="a" * 64,
    )


def _request(
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
    subject: AuxiliaryNodeSubject,
    executor: str,
    prefix: str,
    allow_user_input: bool = True,
) -> AuxiliaryWorkRunRequest:
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    return AuxiliaryWorkRunRequest(
        session_id=session_id,
        turn_id=turn_id,
        subject=subject,
        executor_kind=executor,
        initial_driver_state_guard_sha256=(
            canonical_auxiliary_graph_driver_state_guard(frontier)
        ),
        id_plan=_ids(prefix),
        allow_user_input=allow_user_input,
        task_graph_validation_context=(
            _context(session_id=session_id, turn_id=turn_id, task_id=task_id)
            if executor == "terminal_planner"
            else None
        ),
    )


def _submit_text() -> str:
    return json.dumps(
        {
            "acceptance_updates": [
                {
                    "acceptance_id": "grounded",
                    "model_claimed_satisfied": True,
                }
            ],
            "action": {
                "kind": "submit_output_window",
                "content": "材料调查已经完成，结论与当前任务目标一致。",
                "format": "plain_text",
            },
        },
        ensure_ascii=False,
    )


def _accept_user_gate_answer() -> str:
    return json.dumps(
        {
            "acceptance_updates": [
                {
                    "acceptance_id": "grounded",
                    "model_claimed_satisfied": True,
                }
            ],
            "action": {
                "kind": "submit_output_window",
                "content": "accept_answer",
                "format": "plain_text",
            },
        },
        ensure_ascii=False,
    )


def _task_graph_submit() -> str:
    proposal = InSessionTaskGraphRevisionProposal.model_validate(
        {
            "root": {
                "root_key": "root",
                "nodes": [
                    {
                        "node_key": "root",
                        "node_kind": "root",
                        "parent_node_key": None,
                        "title": "执行计划",
                        "objective": "交付一份可核对的完整执行计划",
                        "source_anchor_ids": ["task_creation_source"],
                        "acceptance_criteria": [
                            {
                                "acceptance_id": "plan_ready",
                                "criterion": "执行计划完整且可核对",
                                "source_anchor_ids": ["task_creation_source"],
                            }
                        ],
                        "constraints": [],
                    }
                ],
            }
        }
    )
    return json.dumps(
        {
            "acceptance_updates": [
                {
                    "acceptance_id": "grounded",
                    "model_claimed_satisfied": True,
                }
            ],
            "action": {
                "kind": "submit_task_graph",
                "proposal": proposal.model_dump(mode="json"),
            },
        },
        ensure_ascii=False,
    )


def _run(
    request: AuxiliaryWorkRunRequest,
    *,
    attempt_provider: _ReplyProvider,
    verifier: _PassVerifier,
    terminal_downstream_gate=None,
    deadline=None,
    model_call_authority_factory=None,
):
    return run_auxiliary_model_node(
        request,
        profile=_profile(),
        capability_catalogs={
            "readonly_documents_v1": CatalogSnapshot(revision=1, entries=())
        },
        attempt_provider=attempt_provider,
        verification_provider=verifier,
        emit=lambda _event: None,
        monotonic_clock=_clock(),
        deadline=deadline,
        model_call_authority_factory=model_call_authority_factory,
        terminal_downstream_gate=terminal_downstream_gate,
    )


def _interrupt_and_accept_followup_turn(
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
    client_request_id: str,
    stage: str = "VERIFICATION",
) -> str:
    marked = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=_window_revision(session_id),
        stage=stage,
        interruption_reason="process_lost_during_verification",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(marked["state_version"]),
        end_reason="process_lost_during_verification",
        error_code="PROCESS_LOST",
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=client_request_id,
        source="auxiliary_v2_work_run_controller_test",
        user_text="继续完成同一个验证请求",
        lease_owner="aux-v2-controller-test",
    )
    next_turn_id = str(accepted["turn"]["turn_id"])
    insession_task_records.link_turn_to_insession_tasks(
        store._deps(),
        session_id=session_id,
        turn_id=next_turn_id,
        insession_task_ids=(task_id,),
        expected_window_revision=_window_revision(session_id),
    )
    return next_turn_id


def _prepare_model_dependent_terminal(
    *,
    prefix: str,
) -> tuple[str, str, str, AuxiliaryWorkRunRequest]:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(session_id=session_id, turn_id=turn_id, task_id=task_id)
    source = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    source_request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=source,
        executor="model_work_run",
        prefix=f"{prefix}-source",
    )
    source_result = _run(
        source_request,
        attempt_provider=_ReplyProvider([_submit_text()]),
        verifier=_PassVerifier(),
    )
    assert source_result.status is AuxiliaryWorkRunStatus.COMPLETED
    terminal = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="terminal_planner",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=terminal,
        executor="terminal_planner",
        prefix=f"{prefix}-terminal",
    )
    return session_id, turn_id, task_id, request


def test_model_work_run_completes_and_exact_reentry_skips_both_models() -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(session_id=session_id, turn_id=turn_id, task_id=task_id)
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="model-node",
    )
    attempt = _ReplyProvider([_submit_text()])
    verifier = _PassVerifier()

    first = _run(request, attempt_provider=attempt, verifier=verifier)
    replay = _run(request, attempt_provider=attempt, verifier=verifier)

    assert first.status is AuxiliaryWorkRunStatus.COMPLETED
    assert replay == first
    assert first.completion_id == "model-node-completion"
    assert [item[0] for item in attempt.calls] == ["model-node-attempt-model"]
    assert [item[0] for item in verifier.calls] == [
        "model-node-verification-model"
    ]
    assert {
        item["tool_id"]
        for item in json.loads(attempt.calls[0][2])["allowed_tools"]
    } == EXECUTION_FINDINGS_TOOL_IDS
    assert json.loads(attempt.calls[0][2])["execution_findings"][
        "ledger_revision"
    ] == 0
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id="model-node-run",
    )
    assert {
        entry["tool_id"] for entry in stored.attempts[0].catalog_snapshot["entries"]
    } == EXECUTION_FINDINGS_TOOL_IDS
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    assert [item.executor_kind.value for item in frontier.ready_fresh] == [
        "terminal_planner"
    ]
    with store._connect() as conn:
        assert int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_auxiliary_node_completions_v2"
            ).fetchone()[0]
        ) == 1


def test_findings_feature_off_hides_attempt_surface_but_keeps_companion() -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(session_id=session_id, turn_id=turn_id, task_id=task_id)
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="model-node-findings-off",
    )
    attempt = _ReplyProvider([_submit_text()])

    completed = run_auxiliary_model_node(
        request,
        profile=_profile().model_copy(
            update={"execution_findings_enabled": False}
        ),
        capability_catalogs={
            "readonly_documents_v1": CatalogSnapshot(revision=1, entries=())
        },
        attempt_provider=attempt,
        verification_provider=_PassVerifier(),
        emit=lambda _event: None,
        monotonic_clock=_clock(),
    )

    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    payload = json.loads(attempt.calls[0][2])
    assert payload["allowed_tools"] == []
    assert payload["execution_findings"] is None
    assert "execution_findings" not in attempt.system_prompts[0]
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id="model-node-findings-off-run",
    )
    assert stored.attempts[0].catalog_snapshot["entries"] == []
    findings = store.get_execution_findings_ledger_for_owner(
        owner_kind="work_run",
        execution_owner_id="model-node-findings-off-run",
    )
    assert findings is not None
    assert findings.ledger.status.value == "closed"
    assert findings.ledger.revision == 0


def test_model_work_run_retries_with_exact_host_contract_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "personagraph.runtime.model_calls.requests._sleep",
        lambda _seconds: None,
    )
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(session_id=session_id, turn_id=turn_id, task_id=task_id)
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="model-node-repair",
    )
    rejected = json.dumps(
        {
            "acceptance_updates": [],
            "action": {
                "kind": "submit_output_window",
                "content": "wrong format enum",
                "format": "json",
            },
        }
    )
    attempt = _ReplyProvider([rejected, _submit_text()])

    completed = _run(
        request,
        attempt_provider=attempt,
        verifier=_PassVerifier(),
    )

    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    assert len(attempt.calls) == 2
    assert "model_claimed_satisfied" in attempt.system_prompts[0]
    assert "supporting_tool_result_ids" in attempt.system_prompts[0]
    first_prompt = json.loads(attempt.calls[0][2])
    assert "host_repair_feedback" not in first_prompt
    assert attempt.calls[0][2] == attempt.calls[1][2]
    repair_messages = attempt.prepared_calls[1]["repair_messages"]
    assert [message["role"] for message in repair_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert repair_messages[2]["content"] == rejected
    feedback = json.loads(
        repair_messages[3]["content"].split("Host 修复清单：", 1)[1]
    )
    assert set(feedback) == {"current_issues"}
    assert any(
        issue["safe_explanation"] == "该位置的值不在目标合同允许的范围内。"
        and "/action/format" in issue["paths"]
        for issue in feedback["current_issues"]
    )
    assert rejected not in attempt.calls[1][2]


def test_prepared_model_work_run_repair_uses_four_message_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "personagraph.runtime.model_calls.requests._sleep",
        lambda _seconds: None,
    )
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(session_id=session_id, turn_id=turn_id, task_id=task_id)
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="prepared-model-node-repair",
    )
    secret_marker = "PRIVATE_REJECTED_AUXILIARY_ATTEMPT"
    rejected = json.dumps(
        {
            "acceptance_updates": [],
            "action": {
                "kind": "submit_output_window",
                "content": secret_marker,
                "format": "json",
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    attempt = _PreparedReplyProvider([rejected, _submit_text()])

    completed = _run(
        request,
        attempt_provider=attempt,
        verifier=_PassVerifier(),
    )

    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    assert len(attempt.prepared_calls) == 2
    first, repair = attempt.prepared_calls
    assert "repair_messages" not in first
    assert first["system_prompt"] == repair["system_prompt"]
    assert first["user_content"] == repair["user_content"]
    assert "host_repair_feedback" not in str(first["system_prompt"])
    messages = repair["repair_messages"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[0]["content"] == first["system_prompt"]
    assert messages[1]["content"] == first["user_content"]
    assert messages[2]["content"] == rejected
    repair_instruction = messages[3]["content"]
    assert rejected not in repair_instruction
    envelope = json.loads(repair_instruction.split("Host 修复清单：", 1)[1])
    assert set(envelope) == {"current_issues"}
    assert "当前清单可能不完整" not in repair_instruction
    assert any(
        "/action/format" in issue["paths"]
        for issue in envelope["current_issues"]
    )
    assert secret_marker not in json.dumps(envelope, ensure_ascii=False)


def test_durable_attempt_binds_each_output_repair_prompt_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    monkeypatch.setattr(
        "personagraph.runtime.model_calls.requests._sleep",
        lambda _seconds: None,
    )
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(session_id=session_id, turn_id=turn_id, task_id=task_id)
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="durable-exact-request",
    )
    rejected = json.dumps(
        {
            "acceptance_updates": [],
            "action": {
                "kind": "submit_output_window",
                "content": "wrong format enum",
                "format": "json",
            },
        }
    )
    attempt = _PreparedReplyProvider(
        [rejected, _submit_text()],
        provider="mock",
        model="mock-structured",
    )

    completed = _run(
        request,
        attempt_provider=attempt,
        verifier=_PassVerifier(provider="mock", model="mock-structured"),
        model_call_authority_factory=(
            partial(create_auxiliary_work_run_model_call_authority, ledger_store=store)
        ),
    )

    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    assert len(attempt.calls) == 2
    assert len(attempt.prepared_calls) == 2
    first_prepared, repair_prepared = attempt.prepared_calls
    assert first_prepared["system_prompt"] == repair_prepared["system_prompt"]
    assert first_prepared["user_content"] == repair_prepared["user_content"]
    assert attempt.calls[0][2] == attempt.calls[1][2]
    repair_messages = repair_prepared["repair_messages"]
    assert [item["role"] for item in repair_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert repair_messages[0]["content"] == first_prepared["system_prompt"]
    assert repair_messages[1]["content"] == first_prepared["user_content"]
    assert repair_messages[2]["content"] == rejected
    repair_feedback = json.loads(
        repair_messages[3]["content"].split("Host 修复清单：", 1)[1]
    )
    assert set(repair_feedback) == {"current_issues"}
    assert repair_feedback["current_issues"]
    assert rejected not in repair_messages[3]["content"]
    logical = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id="durable-exact-request-attempt-model",
    )
    assert logical is not None
    ledgered = json.loads(logical.request.request_json)
    assert json.dumps(
        ledgered["user_content"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) == first_prepared["user_content"]
    assert logical.request.max_physical_attempts == 6
    assert all(
        item.request.request_sha256 == logical.request.request_sha256
        for item in logical.physical_attempts
    )
    first_physical, second_physical = logical.physical_attempts
    assert first_physical.request.output_repair_enabled is True
    assert first_physical.request.output_repair_feedback is None
    assert first_physical.settlement is not None
    assert first_physical.settlement.next_output_repair_feedback is not None
    assert second_physical.request.output_repair_feedback == (
        first_physical.settlement.next_output_repair_feedback
    )
    assert first_physical.request.dispatch_request_sha256 != (
        second_physical.request.dispatch_request_sha256
    )


def test_durable_attempt_transport_retry_keeps_current_repair_prompt_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    monkeypatch.setattr(
        "personagraph.runtime.model_calls.requests._sleep",
        lambda _seconds: None,
    )
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(session_id=session_id, turn_id=turn_id, task_id=task_id)
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="durable-repair-transport",
    )
    rejected = json.dumps(
        {
            "acceptance_updates": [],
            "action": {
                "kind": "submit_output_window",
                "content": "wrong format enum",
                "format": "json",
            },
        }
    )
    attempt = _PreparedReplyProvider(
        [
            rejected,
            ModelGatewayError(
                "MODEL_CALL_TIMEOUT",
                "transient failure after repair was selected",
                retryable=True,
            ),
            _submit_text(),
        ],
        provider="mock",
        model="mock-structured",
    )

    completed = _run(
        request,
        attempt_provider=attempt,
        verifier=_PassVerifier(provider="mock", model="mock-structured"),
        model_call_authority_factory=(
            partial(create_auxiliary_work_run_model_call_authority, ledger_store=store)
        ),
    )

    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    assert len(attempt.calls) == 3
    assert attempt.calls[1][2] == attempt.calls[2][2]
    assert len(attempt.prepared_calls) == 3
    first_prepared, failed_repair, succeeded_repair = attempt.prepared_calls
    assert "repair_messages" not in first_prepared
    assert failed_repair["repair_messages"] == succeeded_repair["repair_messages"]
    repair_messages = failed_repair["repair_messages"]
    assert repair_messages[2]["content"] == rejected
    assert repair_messages[:2] == [
        {
            "role": "system",
            "content": first_prepared["system_prompt"],
        },
        {
            "role": "user",
            "content": first_prepared["user_content"],
        },
    ]
    logical = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id="durable-repair-transport-attempt-model",
    )
    assert logical is not None
    first, transport_failed, succeeded = logical.physical_attempts
    assert first.request.output_repair_feedback is None
    assert transport_failed.request.output_repair_feedback is not None
    assert succeeded.request.output_repair_feedback == (
        transport_failed.request.output_repair_feedback
    )
    assert succeeded.request.dispatch_request_sha256 == (
        transport_failed.request.dispatch_request_sha256
    )


def test_model_output_dependency_flows_through_terminal_and_verifier(
    monkeypatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(session_id=session_id, turn_id=turn_id, task_id=task_id)
    model_subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    model_request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=model_subject,
        executor="model_work_run",
        prefix="dependency-source",
    )
    assert _run(
        model_request,
        attempt_provider=_ReplyProvider([_submit_text()]),
        verifier=_PassVerifier(),
    ).status is AuxiliaryWorkRunStatus.COMPLETED

    terminal_subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="terminal_planner",
    )
    terminal_request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=terminal_subject,
        executor="terminal_planner",
        prefix="dependent-terminal",
    )
    provider = _ReplyProvider([_task_graph_submit()])
    verifier = _PassVerifier()
    original_resolve = auxiliary_graph_store.resolve_auxiliary_dependencies
    resolved_projection_hashes: list[str] = []

    def resolve_and_record(**kwargs):
        bundle = original_resolve(**kwargs)
        resolved_projection_hashes.append(bundle.projection_sha256)
        return bundle

    monkeypatch.setattr(
        auxiliary_graph_store,
        "resolve_auxiliary_dependencies",
        resolve_and_record,
    )
    result = _run(
        terminal_request,
        attempt_provider=provider,
        verifier=verifier,
    )

    assert result.status is AuxiliaryWorkRunStatus.COMPLETED
    attempt_dependency = json.loads(provider.calls[0][2])[
        "auxiliary_dependency_bundle"
    ]
    verification_dependency = json.loads(verifier.calls[0][2])[
        "auxiliary_dependency_bundle"
    ]
    assert attempt_dependency == verification_dependency
    assert len(resolved_projection_hashes) == 3
    assert len(set(resolved_projection_hashes)) == 1
    assert attempt_dependency["untrusted_dependency_data"] is True
    assert [item["dependency_kind"] for item in attempt_dependency["items"]] == [
        "model_output"
    ]
    assert attempt_dependency["items"][0]["content"] == (
        "材料调查已经完成，结论与当前任务目标一致。"
    )


def test_dependency_free_terminal_completes_proposal_without_task_graph_commit() -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        terminal_only=True,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="terminal_planner",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="terminal_planner",
        prefix="terminal-only",
    )

    attempt = _ReplyProvider([_task_graph_submit()])
    verifier = _PassVerifier()
    result = _run(
        request,
        attempt_provider=attempt,
        verifier=verifier,
    )

    assert result.status is AuxiliaryWorkRunStatus.COMPLETED
    attempt_dependency = json.loads(attempt.calls[0][2])[
        "auxiliary_dependency_bundle"
    ]
    verifier_dependency = json.loads(verifier.calls[0][2])[
        "auxiliary_dependency_bundle"
    ]
    assert attempt_dependency == verifier_dependency
    assert attempt_dependency["items"] == []
    assert attempt_dependency["untrusted_dependency_data"] is True
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id="terminal-only-run",
    )
    InSessionTaskGraphRevisionProposal.model_validate_json(
        stored.output_window.content
    )
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    assert task.current_graph_revision is None
def test_terminal_prompt_projects_typed_anchor_roles_and_exact_graph_contract() -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        terminal_only=True,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="terminal_planner",
    )
    creation = _context(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    ).source_anchors[0]
    context = InSessionTaskGraphRevisionValidationContext(
        session_id=session_id,
        source_turn_id=turn_id,
        target_insession_task_id=task_id,
        expected_current_graph_revision=None,
        source_anchors=(
            creation,
            InSessionTaskSourceAnchor(
                anchor_id="document_fact",
                source_turn_id=turn_id,
                source_kind="retrieved_document",
                start=0,
                end=24,
                excerpt="Document evidence excerpt",
            ),
            InSessionTaskSourceAnchor(
                anchor_id="missing_appendix",
                source_turn_id=turn_id,
                source_kind="gap",
                gap_blocking=True,
                start=0,
                end=26,
                excerpt="Required appendix unavailable",
            ),
        ),
        authorization_anchor_ids=("task_creation_source",),
        required_anchor_ids=("task_creation_source", "document_fact"),
    )
    typed_anchors = (
        auxiliary_work_run_controller_module._task_graph_source_anchor_contracts(
            context
        )
    )
    assert typed_anchors == [
        {
            "anchor_id": "task_creation_source",
            "role": "authorization",
            "required": True,
            "blocking": False,
            "source_kind": "previously_authorized_task_state",
            "excerpt": USER_TEXT,
            "excerpt_sha256": hashlib.sha256(USER_TEXT.encode("utf-8")).hexdigest(),
        },
        {
            "anchor_id": "document_fact",
            "role": "evidence",
            "required": True,
            "blocking": False,
            "source_kind": "retrieved_document",
            "excerpt": "Document evidence excerpt",
            "excerpt_sha256": hashlib.sha256(
                "Document evidence excerpt".encode("utf-8")
            ).hexdigest(),
        },
        {
            "anchor_id": "missing_appendix",
            "role": "gap",
            "required": False,
            "blocking": True,
            "source_kind": "gap",
            "excerpt": "Required appendix unavailable",
            "excerpt_sha256": hashlib.sha256(
                "Required appendix unavailable".encode("utf-8")
            ).hexdigest(),
        },
    ]
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="terminal_planner",
        prefix="typed-terminal-contract",
    )
    attempt = _ReplyProvider([_task_graph_submit()])
    verifier = _PassVerifier()

    completed = _run(request, attempt_provider=attempt, verifier=verifier)

    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    contract = json.loads(attempt.calls[0][2])["task_graph_proposal_contract"]
    verification_prompt = json.loads(verifier.calls[0][2])
    assert verification_prompt["task_graph_proposal_contract"] == contract
    assert "task_graph_revision_base" not in verification_prompt
    assert contract["source_anchors"] == typed_anchors[:1]
    terminal_system = attempt.system_prompts[0]
    assert "^[a-z][a-z0-9_-]{0,63}$" in terminal_system
    assert "constraints 必须固定输出 []" in terminal_system
    assert "node 和 Acceptance 两层" in terminal_system
    assert "blocking=true" in terminal_system
    assert "语义等价的重复节点" in terminal_system
    assert "max_nodes_per_task" in terminal_system
    assert "excerpt_sha256" in terminal_system
    assert "不能扩大 role=authorization" in terminal_system
    assert "Host 先执行叶子" in terminal_system
    assert "root=C" in terminal_system
    assert "根节点不是空的协调壳" in terminal_system
    assert "执行完成报告" in terminal_system
    assert "base-null" in terminal_system
    assert "不存在 model-owned lineage" in terminal_system
    assert "planning-only" in terminal_system
    assert "TaskGraph 叶节点重新读取或观察" in terminal_system
    verifier_system = verifier.system_prompts[0]
    assert (
        "acceptance_id、verdict、finding、missing_requirements"
        in verifier_system
    )
    assert '"acceptance_results"' in verifier_system
    assert "auxiliary_dependency_bundle" in verifier_system
    assert "planning-only" in verifier_system
    assert "不得要求 proposal 显式命名" in verifier_system


def test_positive_base_terminal_prompt_and_material_require_exact_lineage() -> None:
    context = _positive_context(
        session_id="session-01",
        turn_id="turn-01",
        task_id="task-01",
    )
    base_snapshot = _semantic_base_snapshot()
    subject = AuxiliaryNodeSubject(
        task_id="task-01",
        auxiliary_graph_id="aux-graph-01",
        auxiliary_graph_revision=2,
        node_id="terminal-01",
        node_revision=1,
    )
    request = AuxiliaryWorkRunRequest(
        session_id="session-01",
        turn_id="turn-01",
        subject=subject,
        executor_kind="terminal_planner",
        initial_driver_state_guard_sha256="b" * 64,
        id_plan=_ids("positive-terminal"),
        task_graph_validation_context=context,
        task_graph_semantic_base_snapshot=base_snapshot,
    )
    assert (
        "kind、proposal、lineage"
        in auxiliary_work_run_controller_module._attempt_system_prompt(request)
    )
    assert (
        "task_graph_revision_base.nodes[].node_alias"
        in auxiliary_work_run_controller_module._attempt_system_prompt(request)
    )

    acceptance = InSessionTaskAcceptanceProposal(
        acceptance_id="grounded",
        criterion="The revision candidate is complete.",
        source_anchor_ids=("task_creation_source",),
    )
    attempt_context = AttemptDecisionContext(
        session_id="session-01",
        turn_id="turn-01",
        work_run_id="positive-terminal-run",
        work_run_revision=1,
        attempt_id="positive-terminal-attempt",
        attempt_ordinal=1,
        user_input=AttemptUserInput(content=USER_TEXT),
        subject=subject,
        node_title="Revise the TaskGraph",
        node_objective="Produce the complete next TaskGraph snapshot.",
        acceptances=(acceptance,),
        acceptance_progress=initialize_acceptance_progress(
            work_run_id="positive-terminal-run",
            subject=subject,
            acceptance_ids=("grounded",),
        ),
        output_window=OutputWindow(
            work_run_id="positive-terminal-run",
            updated_turn_id="turn-01",
        ),
        dependency_deliveries=TaskNodeDependencyDeliveries(),
        input_limits=_profile().attempt_input_limits,
    )
    prompt = auxiliary_work_run_controller_module._attempt_prompt_payload(
        attempt_context,
        terminal=True,
        task_graph_context=context,
        task_graph_semantic_base_snapshot=base_snapshot,
        auxiliary_dependency_payload={"items": []},
    )
    assert prompt["task_graph_revision_base"] == base_snapshot.model_dump(
        mode="json"
    )
    assert (
        prompt["task_graph_proposal_contract"][
            "expected_current_graph_revision"
        ]
        == 1
    )

    raw = json.loads(_task_graph_submit())
    raw["action"]["lineage"] = [
        {
            "proposal_node_key": "root",
            "disposition": "reuse",
            "base_node_alias": "base_root",
        }
    ]
    materialized = auxiliary_work_run_controller_module._materialize_terminal_action(
        raw,
        context=context,
        base_snapshot=base_snapshot,
    )
    candidate = TaskGraphRevisionCandidate.model_validate_json(
        materialized["action"]["content"]
    )
    assert candidate.lineage[0].base_node_alias == "base_root"
    assert materialized["action"]["content"] == candidate.model_dump_json()
    assert (
        auxiliary_work_run_controller_module._parse_terminal_output_material(
            candidate.model_dump_json(),
            context=context,
            base_snapshot=base_snapshot,
        )
        == candidate
    )

    unknown = json.loads(_task_graph_submit())
    unknown["action"]["lineage"] = [
        {
            "proposal_node_key": "root",
            "disposition": "reuse",
            "base_node_alias": "unknown_base",
        }
    ]
    with pytest.raises(ValueError, match="unknown base alias"):
        auxiliary_work_run_controller_module._materialize_terminal_action(
            unknown,
            context=context,
            base_snapshot=base_snapshot,
        )

    missing_lineage = json.loads(_task_graph_submit())
    with pytest.raises(ValueError, match="terminal action"):
        auxiliary_work_run_controller_module._materialize_terminal_action(
            missing_lineage,
            context=context,
            base_snapshot=base_snapshot,
        )


def test_positive_base_terminal_verifier_receives_revision_contract_and_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _positive_context(
        session_id="session-01",
        turn_id="turn-01",
        task_id="task-01",
    )
    base_snapshot = _semantic_base_snapshot()
    sentinel_verification_context = object()
    monkeypatch.setattr(
        auxiliary_work_run_controller_module,
        "build_node_verification_prompt_payload",
        lambda received: (
            {"node": {"acceptances": []}}
            if received is sentinel_verification_context
            else pytest.fail("unexpected verification context")
        ),
    )

    verification_prompt = (
        auxiliary_work_run_controller_module
        ._build_auxiliary_verification_prompt_payload(
            sentinel_verification_context,
            auxiliary_dependency_payload={"items": []},
            task_graph_context=context,
            task_graph_semantic_base_snapshot=base_snapshot,
        )
    )

    assert verification_prompt["task_graph_revision_base"] == (
        base_snapshot.model_dump(mode="json")
    )
    assert (
        verification_prompt["task_graph_proposal_contract"]
        ["expected_current_graph_revision"]
        == 1
    )
    verifier_system = (
        auxiliary_work_run_controller_module._VERIFICATION_SYSTEM_PROMPT
    )
    assert "TaskGraphRevisionCandidate" in verifier_system
    assert "target_graph_revision" in verifier_system
    assert "不得要求" in verifier_system


def test_terminal_request_requires_base_snapshot_iff_context_is_positive_base() -> None:
    subject = AuxiliaryNodeSubject(
        task_id="task-01",
        auxiliary_graph_id="aux-graph-01",
        auxiliary_graph_revision=2,
        node_id="terminal-01",
        node_revision=1,
    )
    common = {
        "session_id": "session-01",
        "turn_id": "turn-01",
        "subject": subject,
        "executor_kind": "terminal_planner",
        "initial_driver_state_guard_sha256": "b" * 64,
        "id_plan": _ids("request-base-binding"),
    }
    with pytest.raises(ValidationError, match="requires exactly one"):
        AuxiliaryWorkRunRequest(
            **common,
            task_graph_validation_context=_positive_context(
                session_id="session-01",
                turn_id="turn-01",
                task_id="task-01",
            ),
        )
    with pytest.raises(ValidationError, match="requires exactly one"):
        AuxiliaryWorkRunRequest(
            **common,
            task_graph_validation_context=_context(
                session_id="session-01",
                turn_id="turn-01",
                task_id="task-01",
            ),
            task_graph_semantic_base_snapshot=_semantic_base_snapshot(),
        )
    with pytest.raises(ValidationError, match="revision differs"):
        AuxiliaryWorkRunRequest(
            **common,
            task_graph_validation_context=_positive_context(
                session_id="session-01",
                turn_id="turn-01",
                task_id="task-01",
            ).model_copy(update={"expected_current_graph_revision": 2}),
            task_graph_semantic_base_snapshot=_semantic_base_snapshot(),
        )


def test_host_context_dependency_flows_through_terminal_and_verifier(
    tmp_path,
) -> None:
    session_id, turn_id, task_id, command, primitive_result = _primitive(tmp_path)
    sealed = planning_store.seal_auxiliary_host_primitive_result(
        command=command,
        result=primitive_result,
    )
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    terminal = frontier.ready_fresh[0]
    context = terminal_store.build_auxiliary_terminal_task_graph_validation_context(
        session_id=session_id,
        invocation_turn_id=turn_id,
        task_id=task_id,
    )
    request = AuxiliaryWorkRunRequest(
        session_id=session_id,
        turn_id=turn_id,
        subject=terminal.subject,
        executor_kind="terminal_planner",
        initial_driver_state_guard_sha256=(
            canonical_auxiliary_graph_driver_state_guard(frontier)
        ),
        id_plan=_ids("host-dependent-terminal"),
        task_graph_validation_context=context,
    )
    terminal_reply = json.loads(_task_graph_submit())
    terminal_reply["acceptance_updates"][0]["acceptance_id"] = (
        "source_understood"
    )
    attempt = _ReplyProvider([json.dumps(terminal_reply, ensure_ascii=False)])
    verifier = _PassVerifier()

    completed = _run(request, attempt_provider=attempt, verifier=verifier)

    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    attempt_dependency = json.loads(attempt.calls[0][2])[
        "auxiliary_dependency_bundle"
    ]
    verifier_dependency = json.loads(verifier.calls[0][2])[
        "auxiliary_dependency_bundle"
    ]
    assert attempt_dependency == verifier_dependency
    assert [item["dependency_kind"] for item in attempt_dependency["items"]] == [
        "host_context"
    ]
    item = attempt_dependency["items"][0]
    assert item["completion_id"] == sealed.artifact_id
    assert item["artifact"] == (
        primitive_result.prompt_inputs.context_artifact.model_dump(mode="json")
    )


def test_dependency_oversize_fails_before_work_run_or_model_dispatch() -> None:
    session_id, turn_id, task_id, request = _prepare_model_dependent_terminal(
        prefix="dependency-oversize"
    )
    profile = _profile().model_copy(
        update={
            "dependency_input_limits": AuxiliaryDependencyInputLimits(
                profile_id="dependency-oversize",
                max_items=64,
                max_serialized_utf8_bytes=64,
            )
        }
    )
    attempt = _ReplyProvider([_task_graph_submit()])
    verifier = _PassVerifier()

    stopped = run_auxiliary_model_node(
        request,
        profile=profile,
        capability_catalogs={},
        attempt_provider=attempt,
        verification_provider=verifier,
        emit=lambda _event: None,
        monotonic_clock=_clock(),
    )

    assert stopped.status is AuxiliaryWorkRunStatus.FAILED_CLOSED
    assert stopped.reason_code == "v2_dependency_input_too_large"
    assert stopped.work_run_id is None
    assert attempt.calls == []
    assert verifier.calls == []
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    assert frontier.recoverable == ()


def test_dependency_store_tamper_is_typed_before_model_dispatch() -> None:
    _session_id, _turn_id, _task_id, request = _prepare_model_dependent_terminal(
        prefix="dependency-controller-tamper"
    )
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_work_run_output_windows SET snapshot_json='{}' "
            "WHERE work_run_id='dependency-controller-tamper-source-run'"
        )
    attempt = _ReplyProvider([_task_graph_submit()])
    verifier = _PassVerifier()

    stopped = _run(request, attempt_provider=attempt, verifier=verifier)

    assert (
        stopped.status
        is AuxiliaryWorkRunStatus.DEPENDENCY_PROJECTION_UNAVAILABLE
    )
    assert stopped.reason_code == "v2_dependency_projection_store_rejected"
    assert stopped.work_run_id is None
    assert attempt.calls == []
    assert verifier.calls == []


def test_verifier_refetch_rejects_dependency_tampered_after_attempt_call() -> None:
    _session_id, _turn_id, _task_id, request = _prepare_model_dependent_terminal(
        prefix="dependency-verifier-tamper"
    )
    attempt_calls = 0

    def tampering_attempt_provider(
        _system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        nonlocal attempt_calls
        attempt_calls += 1
        assert purpose == "runtime_auxiliary_v2_attempt_decision"
        with store._connect() as conn:
            conn.execute(
                "UPDATE insession_work_run_output_windows SET snapshot_json='{}' "
                "WHERE work_run_id='dependency-verifier-tamper-source-run'"
            )
        return ModelResult(
            reply=_task_graph_submit(),
            provider="fake",
            model="fake-model",
            latency_ms=1,
            model_call_id=model_call_id,
            finish_reason="stop",
        )

    verifier = _PassVerifier()
    stopped = run_auxiliary_model_node(
        request,
        profile=_profile(),
        capability_catalogs={},
        attempt_provider=as_prepared_test_provider(tampering_attempt_provider),
        verification_provider=verifier,
        emit=lambda _event: None,
        monotonic_clock=_clock(),
    )

    assert (
        stopped.status
        is AuxiliaryWorkRunStatus.DEPENDENCY_PROJECTION_UNAVAILABLE
    )
    assert stopped.reason_code == "v2_dependency_projection_store_rejected"
    assert stopped.work_run_id == "dependency-verifier-tamper-terminal-run"
    assert attempt_calls == 1
    assert verifier.calls == []


def test_dependency_unsupported_input_is_typed_before_model_dispatch(
    monkeypatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        terminal_only=True,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="terminal_planner",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="terminal_planner",
        prefix="dependency-unsupported",
    )

    def reject_serialization(*_args, **_kwargs):
        raise AuxiliaryDependencyInputUnsupported("injected unsupported input")

    monkeypatch.setattr(
        "personagraph.l2.auxiliary_execution.work_run.controller."
        "serialize_auxiliary_dependency_model_payload",
        reject_serialization,
    )
    attempt = _ReplyProvider([_task_graph_submit()])

    stopped = _run(
        request,
        attempt_provider=attempt,
        verifier=_PassVerifier(),
    )

    assert stopped.status is AuxiliaryWorkRunStatus.FAILED_CLOSED
    assert stopped.reason_code == "v2_dependency_input_unsupported"
    assert stopped.work_run_id is None
    assert attempt.calls == []


def test_dependency_projection_drift_rejects_durable_attempt_before_provider(
    monkeypatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        terminal_only=True,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="terminal_planner",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="terminal_planner",
        prefix="dependency-guard-drift",
    )
    original_resolve = auxiliary_graph_store.resolve_auxiliary_dependencies
    calls = 0

    def resolve_then_drift(**kwargs):
        nonlocal calls
        calls += 1
        bundle = original_resolve(**kwargs)
        if calls < 3:
            return bundle
        return AuxiliaryDependencyBundle.create(
            session_id=bundle.session_id,
            task_id=bundle.task_id,
            auxiliary_graph_id=bundle.auxiliary_graph_id,
            auxiliary_graph_revision=bundle.auxiliary_graph_revision,
            consumer_subject=bundle.consumer_subject,
            consumer_node_alias=bundle.consumer_node_alias,
            structure_sha256="f" * 64,
            items=bundle.items,
        )

    monkeypatch.setattr(
        auxiliary_graph_store,
        "resolve_auxiliary_dependencies",
        resolve_then_drift,
    )
    attempt = _ReplyProvider([_task_graph_submit()])

    stopped = run_auxiliary_model_node(
        request,
        profile=_profile(),
        capability_catalogs={},
        attempt_provider=attempt,
        verification_provider=_PassVerifier(),
        emit=lambda _event: None,
        monotonic_clock=_clock(),
        model_call_authority_factory=lambda binding, **kwargs: (
            _RevalidatingAuthority(
                binding,
                kwargs["rederive_state_guard_sha256"],
            )
        ),
    )

    assert stopped.status is AuxiliaryWorkRunStatus.FAILED_CLOSED
    assert stopped.reason_code == "v2_attempt_model_state_guard_changed"
    assert calls == 3
    assert attempt.calls == []


def test_durable_attempt_authority_rejection_is_typed_before_provider() -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        terminal_only=True,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="terminal_planner",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="terminal_planner",
        prefix="model-authority-rejected",
    )
    attempt = _ReplyProvider([_task_graph_submit()])

    def reject_authority(_binding, **_kwargs):
        raise ValueError("configured durable authority rejected the binding")

    stopped = run_auxiliary_model_node(
        request,
        profile=_profile(),
        capability_catalogs={},
        attempt_provider=attempt,
        verification_provider=_PassVerifier(),
        emit=lambda _event: None,
        monotonic_clock=_clock(),
        model_call_authority_factory=reject_authority,
    )

    assert stopped.status is AuxiliaryWorkRunStatus.FAILED_CLOSED
    assert stopped.reason_code == "v2_attempt_model_authority_rejected"
    assert attempt.calls == []


def test_production_work_run_authority_persists_attempt_and_verifier_ledgers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        terminal_only=True,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="terminal_planner",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="terminal_planner",
        prefix="production-model-authority",
    )
    attempt = _ReplyProvider(
        [_task_graph_submit()],
        provider="mock",
        model="mock-structured",
    )
    verifier = _PassVerifier(
        provider="mock",
        model="mock-structured",
    )

    result = run_auxiliary_model_node(
        request,
        profile=_profile(),
        capability_catalogs={},
        attempt_provider=attempt,
        verification_provider=verifier,
        emit=lambda _event: None,
        monotonic_clock=_clock(),
        model_call_authority_factory=(
            partial(create_auxiliary_work_run_model_call_authority, ledger_store=store)
        ),
    )

    assert result.status is AuxiliaryWorkRunStatus.COMPLETED
    attempt_call_id = request.id_plan.for_attempt(
        request.id_plan.attempt_model_call_id,
        1,
    )
    verification_call_id = request.id_plan.for_attempt(
        request.id_plan.verification_model_call_id,
        1,
    )
    attempt_logical = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=attempt_call_id,
    )
    verification_logical = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=verification_call_id,
    )
    assert attempt_logical is not None
    assert verification_logical is not None
    assert attempt_logical.request.execution_subject_id is not None
    assert (
        verification_logical.request.execution_subject_id
        == attempt_logical.request.execution_subject_id
    )
    assert len(attempt_logical.physical_attempts) == 1
    assert len(verification_logical.physical_attempts) == 1
    assert attempt_logical.physical_attempts[0].settlement is not None
    assert verification_logical.physical_attempts[0].settlement is not None


def test_production_work_run_authority_replays_after_host_commit_response_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        terminal_only=True,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="terminal_planner",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="terminal_planner",
        prefix="production-model-response-loss",
    )
    attempt = _ReplyProvider(
        [_task_graph_submit()],
        provider="mock",
        model="mock-structured",
    )
    verifier = _PassVerifier(
        provider="mock",
        model="mock-structured",
    )
    original_commit = work_run_store.commit_work_run_output_action
    lost = False

    def lose_first_host_commit_response(**kwargs):
        nonlocal lost
        if not lost:
            lost = True
            raise RuntimeError("simulated Host commit response loss")
        return original_commit(**kwargs)

    monkeypatch.setattr(
        work_run_store,
        "commit_work_run_output_action",
        lose_first_host_commit_response,
    )
    run_kwargs = {
        "profile": _profile(),
        "capability_catalogs": {},
        "attempt_provider": attempt,
        "verification_provider": verifier,
        "emit": lambda _event: None,
        "monotonic_clock": _clock(),
        "model_call_authority_factory": (
            partial(create_auxiliary_work_run_model_call_authority, ledger_store=store)
        ),
    }
    with pytest.raises(RuntimeError, match="simulated Host commit response loss"):
        run_auxiliary_model_node(request, **run_kwargs)
    completed = run_auxiliary_model_node(request, **run_kwargs)

    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    assert len(attempt.calls) == 1
    assert len(verifier.calls) == 1
    attempt_call_id = request.id_plan.for_attempt(
        request.id_plan.attempt_model_call_id,
        1,
    )
    logical = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=attempt_call_id,
    )
    assert logical is not None
    assert len(logical.physical_attempts) == 1


def test_auxiliary_verifier_repairs_precise_contract_error() -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(session_id=session_id, turn_id=turn_id, task_id=task_id)
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="model-node-verifier-repair",
    )
    secret_marker = "PRIVATE_REJECTED_AUXILIARY_REVIEW"
    verifier = _ReplyProvider(
        [
            json.dumps({"private_extra": secret_marker}),
            json.dumps(
                {
                    "acceptance_results": [
                        {
                            "acceptance_id": "grounded",
                            "verdict": "passed",
                            "finding": "锁定输出满足此项验收条件。",
                            "missing_requirements": [],
                        }
                    ]
                },
                ensure_ascii=False,
            ),
        ]
    )

    completed = _run(
        request,
        attempt_provider=_ReplyProvider([_submit_text()]),
        verifier=verifier,
    )

    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    assert len(verifier.calls) == 2
    assert verifier.calls[0][0] == verifier.calls[1][0]
    first_prompt = json.loads(verifier.calls[0][2])
    assert "host_repair_feedback" not in first_prompt
    assert verifier.calls[0][2] == verifier.calls[1][2]
    repair_messages = verifier.prepared_calls[1]["repair_messages"]
    feedback = json.loads(
        repair_messages[3]["content"].split("Host 修复清单：", 1)[1]
    )
    explanations = {issue["safe_explanation"] for issue in feedback["current_issues"]}
    assert explanations == {"该位置含有目标合同未声明的额外字段。", "目标合同要求此位置必须存在。"}
    issue_paths = {
        path
        for issue in feedback["current_issues"]
        for path in issue["paths"]
    }
    assert issue_paths == {"", "/acceptance_results"}
    assert "private_extra" not in json.dumps(feedback, ensure_ascii=False)
    assert secret_marker not in json.dumps(feedback, ensure_ascii=False)


def test_prepared_auxiliary_verifier_repair_uses_four_message_context() -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(session_id=session_id, turn_id=turn_id, task_id=task_id)
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="prepared-model-node-verifier-repair",
    )
    secret_marker = "PRIVATE_REJECTED_AUXILIARY_REVIEW"
    rejected = json.dumps(
        {"private_extra": secret_marker},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    accepted = json.dumps(
        {
            "acceptance_results": [
                {
                    "acceptance_id": "grounded",
                    "verdict": "passed",
                    "finding": "锁定输出满足此项验收条件。",
                    "missing_requirements": [],
                }
            ]
        },
        ensure_ascii=False,
    )
    verifier = _PreparedReplyProvider([rejected, accepted])

    completed = _run(
        request,
        attempt_provider=_ReplyProvider([_submit_text()]),
        verifier=verifier,
    )

    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    assert len(verifier.prepared_calls) == 2
    first, repair = verifier.prepared_calls
    assert "repair_messages" not in first
    assert first["system_prompt"] == repair["system_prompt"]
    assert first["user_content"] == repair["user_content"]
    assert "host_repair_feedback" not in str(first["system_prompt"])
    messages = repair["repair_messages"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[2]["content"] == rejected
    repair_instruction = messages[3]["content"]
    assert rejected not in repair_instruction
    envelope = json.loads(repair_instruction.split("Host 修复清单：", 1)[1])
    assert set(envelope) == {"current_issues"}
    assert "当前清单可能不完整" not in repair_instruction
    issue_paths = {
        path
        for issue in envelope["current_issues"]
        for path in issue["paths"]
    }
    assert "/acceptance_results" in issue_paths
    assert "" in issue_paths
    assert secret_marker not in json.dumps(envelope, ensure_ascii=False)


def test_prepared_auxiliary_nonpass_verdict_is_not_output_repair() -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(session_id=session_id, turn_id=turn_id, task_id=task_id)
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="prepared-verification-nonpass",
    )

    def verification_reply(verdict: str) -> str:
        return json.dumps(
            {
                "acceptance_results": [
                    {
                        "acceptance_id": "grounded",
                        "verdict": verdict,
                        "finding": "按当前证据作出业务验收结论。",
                        "missing_requirements": (
                            [] if verdict == "passed" else ["需要修订输出。"]
                        ),
                    }
                ]
            },
            ensure_ascii=False,
        )

    verifier = _PreparedReplyProvider(
        [verification_reply("not_satisfied"), verification_reply("passed")]
    )

    completed = _run(
        request,
        attempt_provider=_ReplyProvider([_submit_text(), _submit_text()]),
        verifier=verifier,
    )

    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    assert completed.attempt_id == "prepared-verification-nonpass-attempt:attempt-2"
    assert len(verifier.prepared_calls) == 2
    assert all(
        "repair_messages" not in prepared
        for prepared in verifier.prepared_calls
    )


def test_dependency_projection_drift_rejects_durable_verifier_before_provider(
    monkeypatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        terminal_only=True,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="terminal_planner",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="terminal_planner",
        prefix="dependency-verifier-guard-drift",
    )
    original_resolve = auxiliary_graph_store.resolve_auxiliary_dependencies
    calls = 0

    def resolve_then_drift_for_verifier(**kwargs):
        nonlocal calls
        calls += 1
        bundle = original_resolve(**kwargs)
        if calls < 8:
            return bundle
        return AuxiliaryDependencyBundle.create(
            session_id=bundle.session_id,
            task_id=bundle.task_id,
            auxiliary_graph_id=bundle.auxiliary_graph_id,
            auxiliary_graph_revision=bundle.auxiliary_graph_revision,
            consumer_subject=bundle.consumer_subject,
            consumer_node_alias=bundle.consumer_node_alias,
            structure_sha256="e" * 64,
            items=bundle.items,
        )

    monkeypatch.setattr(
        auxiliary_graph_store,
        "resolve_auxiliary_dependencies",
        resolve_then_drift_for_verifier,
    )
    attempt = _ReplyProvider([_task_graph_submit()])
    verifier = _PassVerifier()

    stopped = run_auxiliary_model_node(
        request,
        profile=_profile(),
        capability_catalogs={},
        attempt_provider=attempt,
        verification_provider=verifier,
        emit=lambda _event: None,
        monotonic_clock=_clock(),
        model_call_authority_factory=lambda binding, **kwargs: (
            _RevalidatingAuthority(
                binding,
                kwargs["rederive_state_guard_sha256"],
            )
        ),
    )

    assert stopped.status is AuxiliaryWorkRunStatus.FAILED_CLOSED
    assert stopped.reason_code == "v2_verification_model_state_guard_changed"
    assert calls == 8
    assert len(attempt.calls) == 1
    assert verifier.calls == []


def test_missing_capability_catalog_is_typed_and_mutates_nothing() -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(session_id=session_id, turn_id=turn_id, task_id=task_id)
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="missing-capability",
    )
    provider = _ReplyProvider([_submit_text()])

    result = run_auxiliary_model_node(
        request,
        profile=_profile(),
        capability_catalogs={},
        attempt_provider=provider,
        verification_provider=_PassVerifier(),
        emit=lambda _event: None,
        monotonic_clock=_clock(),
    )

    assert (
        result.status
        is AuxiliaryWorkRunStatus.CAPABILITY_CATALOG_UNAVAILABLE
    )
    assert provider.calls == []
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    assert frontier.recoverable == ()


def test_waiting_user_action_settles_through_the_only_seam() -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(session_id=session_id, turn_id=turn_id, task_id=task_id)
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="waiting-user",
    )
    provider = _ReplyProvider(
        [
            json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": "请补充希望覆盖的时间范围。",
                    },
                },
                ensure_ascii=False,
            )
        ]
    )
    verifier = _PassVerifier()

    first = _run(request, attempt_provider=provider, verifier=verifier)

    assert first.status is AuxiliaryWorkRunStatus.WAITING_USER
    assert first.reason_code == "v2_waiting_user_requires_store_continuation"
    assert len(provider.calls) == 1
    assert verifier.calls == []
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id="waiting-user-run",
    )
    assert stored.work_run.status.value == "waiting_user"
    assert stored.current_attempt_id is None
    assert stored.attempts[0].action == "request_user_input"
    assert stored.attempts[0].decision is not None


def test_closed_world_user_gate_fails_before_work_run_or_model_call() -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_user_gate_graph(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="user_gate",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="user_gate",
        prefix="closed-world-user-gate",
        allow_user_input=False,
    )
    provider = _ReplyProvider([])

    result = _run(
        request,
        attempt_provider=provider,
        verifier=_PassVerifier(),
    )

    assert result.status is AuxiliaryWorkRunStatus.FAILED_CLOSED
    assert result.reason_code == "v2_closed_world_user_gate_forbidden"
    assert result.work_run_id is None
    assert provider.calls == []


def test_closed_world_model_work_run_repairs_question_before_persistence() -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(session_id=session_id, turn_id=turn_id, task_id=task_id)
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="closed-world-model-run",
        allow_user_input=False,
    )
    provider = _ReplyProvider(
        [
            json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": "请提供材料截图。",
                    },
                },
                ensure_ascii=False,
            ),
            _submit_text(),
        ]
    )

    result = _run(
        request,
        attempt_provider=provider,
        verifier=_PassVerifier(),
    )

    assert result.status is AuxiliaryWorkRunStatus.COMPLETED
    assert len(provider.calls) == 2
    assert all(
        "request_user_input 被禁止" in prompt
        for prompt in provider.system_prompts
    )
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id="closed-world-model-run-run",
    )
    assert stored.work_run.status is WorkRunStatus.COMPLETED
    assert all(item.action != "request_user_input" for item in stored.attempts)


def test_waiting_user_answer_starts_stable_attempt_two_and_completes() -> None:
    session_id, question_turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=question_turn_id,
        task_id=task_id,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=question_turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    question = "请补充希望覆盖的时间范围。"
    answer = "覆盖最近三年，并优先分析最近十二个月。"
    attempt = _ReplyProvider(
        [
            json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": question,
                    },
                },
                ensure_ascii=False,
            ),
            _submit_text(),
        ]
    )
    verifier = _PassVerifier()
    question_request = _request(
        session_id=session_id,
        turn_id=question_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="answer",
    )

    waiting = _run(
        question_request,
        attempt_provider=attempt,
        verifier=verifier,
    )
    assert waiting.status is AuxiliaryWorkRunStatus.WAITING_USER
    finalized = store.finalize_turn_execution(
        session_id=session_id,
        turn_id=question_turn_id,
        expected_window_revision=_window_revision(session_id),
        processing_level="L2",
        assistant_content=question,
        post_commit_job_kinds=(),
    )
    store.release_turn_execution_window(
        session_id=session_id,
        turn_id=question_turn_id,
        expected_window_revision=int(finalized["window"]["state_version"]),
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="answer-followup",
        source="auxiliary_v2_work_run_controller_test",
        user_text=answer,
        lease_owner="aux-v2-controller-test",
    )
    answer_turn_id = str(accepted["turn"]["turn_id"])
    insession_task_records.link_turn_to_insession_tasks(
        store._deps(),
        session_id=session_id,
        turn_id=answer_turn_id,
        insession_task_ids=(task_id,),
        expected_window_revision=_window_revision(session_id),
    )
    answer_request = _request(
        session_id=session_id,
        turn_id=answer_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="answer",
    )

    completed = _run(answer_request, attempt_provider=attempt, verifier=verifier)

    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    assert completed.attempt_id == "answer-attempt:attempt-2"
    assert completed.completion_id == "answer-completion:attempt-2"
    second_prompt = json.loads(attempt.calls[1][2])
    assert second_prompt["user_input"] == {
        "content": answer,
        "prior_waiting_user_question": question,
    }
    stored = work_run_store.get_work_run(session_id=session_id, work_run_id="answer-run")
    assert tuple(item.attempt.ordinal for item in stored.attempts) == (1, 2)
    assert stored.attempts[1].turn_id == answer_turn_id
    assert stored.attempts[1].input_turn_id == answer_turn_id
    assert (
        stored.attempts[1].predecessor_question_attempt_id == "answer-attempt"
    )


def test_user_gate_rejects_question_that_rewrites_frozen_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "personagraph.runtime.model_calls.requests._sleep",
        lambda _seconds: None,
    )
    session_id, turn_id, task_id = _seed_task()
    _commit_user_gate_graph(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="user_gate",
    )
    expected_question = "请明确希望计划覆盖最近几年。"
    attempt = _ReplyProvider(
        [
            json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": "请问您大概想看多长时间范围？",
                    },
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": expected_question,
                    },
                },
                ensure_ascii=False,
            ),
        ]
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="user_gate",
        prefix="user-gate-exact-question",
    )

    waiting = _run(
        request,
        attempt_provider=attempt,
        verifier=_PassVerifier(),
    )

    assert waiting.status is AuxiliaryWorkRunStatus.WAITING_USER
    assert len(attempt.calls) == 2
    first_prompt = json.loads(attempt.calls[0][2])
    second_prompt = json.loads(attempt.calls[1][2])
    assert first_prompt["user_gate_contract"]["expected_question"] == (
        expected_question
    )
    assert second_prompt["user_gate_contract"]["expected_question"] == (
        expected_question
    )
    assert "host_repair_feedback" not in second_prompt
    repair_messages = attempt.prepared_calls[1]["repair_messages"]
    feedback = json.loads(
        repair_messages[3]["content"].split("Host 修复清单：", 1)[1]
    )
    assert any(
        issue["paths"] == ["/action"]
        and "用户澄清门 action 未遵守冻结问题" in issue["safe_explanation"]
        for issue in feedback["current_issues"]
    )
    pending = continuation_store.get_auxiliary_pending_user_question(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert pending is not None and pending.question == expected_question


def test_user_gate_persists_question_and_exact_answer_attempt_unblocks_terminal() -> None:
    session_id, question_turn_id, task_id = _seed_task()
    _commit_user_gate_graph(
        session_id=session_id,
        turn_id=question_turn_id,
        task_id=task_id,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=question_turn_id,
        task_id=task_id,
        executor="user_gate",
    )
    question = "请明确希望计划覆盖最近几年。"
    answer = "覆盖最近三年。"
    attempt = _ReplyProvider(
        [
            json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": question,
                    },
                },
                ensure_ascii=False,
            ),
            _accept_user_gate_answer(),
        ]
    )
    verifier = _PassVerifier()
    question_request = _request(
        session_id=session_id,
        turn_id=question_turn_id,
        task_id=task_id,
        subject=subject,
        executor="user_gate",
        prefix="user-gate",
    )

    waiting = _run(
        question_request,
        attempt_provider=attempt,
        verifier=verifier,
    )

    assert waiting.status is AuxiliaryWorkRunStatus.WAITING_USER
    pending = continuation_store.get_auxiliary_pending_user_question(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert pending is not None
    assert pending.question == question
    assert pending.subject == subject
    first_prompt = json.loads(attempt.calls[0][2])
    assert first_prompt["user_gate_contract"] == {
        "schema_version": "auxiliary-user-gate-attempt-v1",
        "phase": "ask",
        "expected_question": question,
        "host_materialized_output_contract": "auxiliary-user-response-v1",
    }
    session_pending = continuation_store.list_pending_user_questions(session_id=session_id)
    assert len(session_pending) == 1
    assert session_pending[0].work_run_id == pending.work_run_id
    assert session_pending[0].question == question
    assert work_run_store.list_turn_completed_verified_delivery_ids(
        session_id=session_id,
        turn_id=question_turn_id,
    ) == ()
    finalized = store.finalize_turn_execution(
        session_id=session_id,
        turn_id=question_turn_id,
        expected_window_revision=_window_revision(session_id),
        processing_level="L2",
        assistant_content=question,
        post_commit_job_kinds=(),
    )
    store.release_turn_execution_window(
        session_id=session_id,
        turn_id=question_turn_id,
        expected_window_revision=int(finalized["window"]["state_version"]),
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="user-gate-answer-followup",
        source="auxiliary_v2_work_run_controller_test",
        user_text=answer,
        lease_owner="aux-v2-controller-test",
    )
    answer_turn_id = str(accepted["turn"]["turn_id"])
    task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=answer_turn_id,
        apply_id="user-gate-answer-match",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "existing_root",
                        "insession_task_id": task_id,
                        "source_excerpt": answer,
                        "execute_current": True,
                    }
                ]
            }
        ),
        exposed_catalog_ids=(task_id,),
        expected_window_revision=_window_revision(session_id),
    )
    answer_request = _request(
        session_id=session_id,
        turn_id=answer_turn_id,
        task_id=task_id,
        subject=subject,
        executor="user_gate",
        prefix="user-gate",
    )

    completed = _run(answer_request, attempt_provider=attempt, verifier=verifier)
    replayed_after_response_loss = _run(
        answer_request,
        attempt_provider=attempt,
        verifier=verifier,
    )

    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    assert replayed_after_response_loss == completed
    assert completed.attempt_id == "user-gate-attempt:attempt-2"
    assert len(attempt.calls) == 2
    second_prompt = json.loads(attempt.calls[1][2])
    assert second_prompt["user_input"] == {
        "content": answer,
        "prior_waiting_user_question": question,
    }
    stored = work_run_store.get_work_run(session_id=session_id, work_run_id="user-gate-run")
    assert tuple(item.attempt.ordinal for item in stored.attempts) == (1, 2)
    assert stored.attempts[1].predecessor_question_attempt_id == "user-gate-attempt"
    gate_output = json.loads(stored.output_window.content)
    assert gate_output == {
        "answer": answer,
        "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "question": question,
        "schema_version": "auxiliary-user-response-v1",
    }
    assert continuation_store.list_pending_user_questions(session_id=session_id) == ()
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=answer_turn_id,
        insession_task_id=task_id,
    )
    assert tuple(item.executor_kind.value for item in frontier.ready_fresh) == (
        "terminal_planner",
    )


def test_user_gate_rejects_incomplete_question_turn_without_revision_trigger() -> None:
    session_id, question_turn_id, task_id = _seed_task()
    _commit_user_gate_graph(
        session_id=session_id,
        turn_id=question_turn_id,
        task_id=task_id,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=question_turn_id,
        task_id=task_id,
        executor="user_gate",
    )
    question = "请明确希望计划覆盖最近几年。"
    answer = "覆盖最近三年。"
    attempt = _ReplyProvider(
        [
            json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": question,
                    },
                },
                ensure_ascii=False,
            )
        ]
    )
    waiting = _run(
        _request(
            session_id=session_id,
            turn_id=question_turn_id,
            task_id=task_id,
            subject=subject,
            executor="user_gate",
            prefix="user-gate-incomplete-no-trigger",
        ),
        attempt_provider=attempt,
        verifier=_PassVerifier(),
    )
    assert waiting.status is AuxiliaryWorkRunStatus.WAITING_USER

    interrupted = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=question_turn_id,
        expected_window_revision=_window_revision(session_id),
        stage="RESPONSE",
        interruption_reason="WAITING_USER",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=question_turn_id,
        expected_window_revision=int(interrupted["state_version"]),
        end_reason="host_stopped",
        error_code="WAITING_USER",
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="user-gate-incomplete-no-trigger-answer",
        source="auxiliary_v2_work_run_controller_test",
        user_text=answer,
        lease_owner="aux-v2-controller-test",
    )
    answer_turn_id = str(accepted["turn"]["turn_id"])
    task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=answer_turn_id,
        apply_id="user-gate-incomplete-no-trigger-match",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "existing_root",
                        "insession_task_id": task_id,
                        "source_excerpt": answer,
                        "execute_current": True,
                    }
                ]
            }
        ),
        exposed_catalog_ids=(task_id,),
        expected_window_revision=_window_revision(session_id),
    )

    rejected = _run(
        _request(
            session_id=session_id,
            turn_id=answer_turn_id,
            task_id=task_id,
            subject=subject,
            executor="user_gate",
            prefix="user-gate-incomplete-no-trigger",
        ),
        attempt_provider=attempt,
        verifier=_PassVerifier(),
    )

    assert rejected.status is AuxiliaryWorkRunStatus.FAILED_CLOSED
    assert rejected.reason_code == "v2_waiting_user_continuation_authority_rejected"
    assert len(attempt.calls) == 1
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id="user-gate-incomplete-no-trigger-run",
    )
    assert stored.work_run.status is WorkRunStatus.WAITING_USER
    assert len(stored.attempts) == 1


def test_user_gate_does_not_consume_plain_linked_text_without_lane_authority() -> None:
    session_id, question_turn_id, task_id = _seed_task()
    _commit_user_gate_graph(
        session_id=session_id,
        turn_id=question_turn_id,
        task_id=task_id,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=question_turn_id,
        task_id=task_id,
        executor="user_gate",
    )
    question = "请明确希望计划覆盖最近几年。"
    attempt = _ReplyProvider(
        [
            json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": question,
                    },
                },
                ensure_ascii=False,
            )
        ]
    )
    verifier = _PassVerifier()
    question_request = _request(
        session_id=session_id,
        turn_id=question_turn_id,
        task_id=task_id,
        subject=subject,
        executor="user_gate",
        prefix="user-gate-unbound",
    )
    waiting = _run(question_request, attempt_provider=attempt, verifier=verifier)
    assert waiting.status is AuxiliaryWorkRunStatus.WAITING_USER
    finalized = store.finalize_turn_execution(
        session_id=session_id,
        turn_id=question_turn_id,
        expected_window_revision=_window_revision(session_id),
        processing_level="L2",
        assistant_content=question,
        post_commit_job_kinds=(),
    )
    store.release_turn_execution_window(
        session_id=session_id,
        turn_id=question_turn_id,
        expected_window_revision=int(finalized["window"]["state_version"]),
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="user-gate-unrelated-followup",
        source="auxiliary_v2_work_run_controller_test",
        user_text="顺便告诉我当前进度。",
        lease_owner="aux-v2-controller-test",
    )
    unrelated_turn_id = str(accepted["turn"]["turn_id"])
    insession_task_records.link_turn_to_insession_tasks(
        store._deps(),
        session_id=session_id,
        turn_id=unrelated_turn_id,
        insession_task_ids=(task_id,),
        expected_window_revision=_window_revision(session_id),
    )
    unrelated_request = _request(
        session_id=session_id,
        turn_id=unrelated_turn_id,
        task_id=task_id,
        subject=subject,
        executor="user_gate",
        prefix="user-gate-unbound",
    )

    still_waiting = _run(
        unrelated_request,
        attempt_provider=attempt,
        verifier=verifier,
    )

    assert still_waiting.status is AuxiliaryWorkRunStatus.WAITING_USER
    assert still_waiting.reason_code == "v2_waiting_for_authorized_user_answer_turn"
    assert len(attempt.calls) == 1
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id="user-gate-unbound-run",
    )
    assert len(stored.attempts) == 1
    assert stored.work_run.status is WorkRunStatus.WAITING_USER


def test_model_and_verification_interruptions_are_same_turn_replayable() -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(session_id=session_id, turn_id=turn_id, task_id=task_id)
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="interrupted",
    )
    attempt = _ReplyProvider(
        [
            ModelGatewayError(
                "FAKE_TERMINAL",
                "fake terminal failure",
                retryable=False,
            ),
            _submit_text(),
        ]
    )
    verifier = _PassVerifier(
        failures=(
            ModelGatewayError(
                "FAKE_VERIFY_TERMINAL",
                "fake verifier failure",
                retryable=False,
            ),
        )
    )

    first = _run(request, attempt_provider=attempt, verifier=verifier)
    second = _run(request, attempt_provider=attempt, verifier=verifier)
    third = _run(request, attempt_provider=attempt, verifier=verifier)

    assert first.status is AuxiliaryWorkRunStatus.MODEL_INTERRUPTED
    assert (
        second.status
        is AuxiliaryWorkRunStatus.VERIFICATION_INTERRUPTED
    )
    assert third.status is AuxiliaryWorkRunStatus.COMPLETED
    assert [item[0] for item in attempt.calls] == [
        "interrupted-attempt-model",
        "interrupted-attempt-model",
    ]
    assert [item[0] for item in verifier.calls] == [
        "interrupted-verification-model",
        "interrupted-verification-model",
    ]


@pytest.mark.parametrize("failure_stage", ("attempt", "verification"))
def test_durable_terminal_model_failure_is_not_advertised_as_replayable(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(session_id=session_id, turn_id=turn_id, task_id=task_id)
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix=f"durable-terminal-{failure_stage}",
    )
    terminal_error = ModelGatewayError(
        "MODEL_CALL_FAILED",
        "the configured Provider credentials are invalid",
        retryable=False,
        details={"reason": "invalid_credentials"},
    )
    attempt = _ReplyProvider(
        [terminal_error] if failure_stage == "attempt" else [_submit_text()],
        provider="mock",
        model="mock-structured",
    )
    verifier = _PassVerifier(
        failures=(terminal_error,) if failure_stage == "verification" else (),
        provider="mock",
        model="mock-structured",
    )

    first = _run(
        request,
        attempt_provider=attempt,
        verifier=verifier,
        model_call_authority_factory=(
            partial(create_auxiliary_work_run_model_call_authority, ledger_store=store)
        ),
    )
    provider_call_count = len(attempt.calls) + len(verifier.calls)
    replay = _run(
        request,
        attempt_provider=_ReplyProvider(
            [],
            provider="mock",
            model="mock-structured",
        ),
        verifier=_PassVerifier(provider="mock", model="mock-structured"),
        model_call_authority_factory=(
            partial(create_auxiliary_work_run_model_call_authority, ledger_store=store)
        ),
    )

    assert first.status is AuxiliaryWorkRunStatus.FAILED_CLOSED
    assert replay.status is AuxiliaryWorkRunStatus.FAILED_CLOSED
    assert first.reason_code == (
        f"v2_{failure_stage}_model_call_terminal_failure"
    )
    assert replay.reason_code == first.reason_code
    assert provider_call_count == (1 if failure_stage == "attempt" else 2)


def test_active_undecided_attempt_resumes_on_an_exact_new_turn() -> None:
    session_id, first_turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    first_request = _request(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="resume",
    )
    attempt = _ReplyProvider(
        [
            ModelGatewayError(
                "FAKE_TERMINAL",
                "fake terminal failure",
                retryable=False,
            ),
            _submit_text(),
        ]
    )
    verifier = _PassVerifier()

    interrupted = _run(
        first_request,
        attempt_provider=attempt,
        verifier=verifier,
    )
    assert (
        interrupted.status
        is AuxiliaryWorkRunStatus.MODEL_INTERRUPTED
    )
    marked = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=_window_revision(session_id),
        stage="L2_PLAN",
        interruption_reason="process_lost",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=int(marked["state_version"]),
        end_reason="process_lost",
        error_code="PROCESS_LOST",
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="resume-followup",
        source="auxiliary_v2_work_run_controller_test",
        user_text="继续执行",
        lease_owner="aux-v2-controller-test",
    )
    second_turn_id = str(accepted["turn"]["turn_id"])
    insession_task_records.link_turn_to_insession_tasks(
        store._deps(),
        session_id=session_id,
        turn_id=second_turn_id,
        insession_task_ids=(task_id,),
        expected_window_revision=_window_revision(session_id),
    )
    second_request = _request(
        session_id=session_id,
        turn_id=second_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="resume",
    )

    completed = _run(second_request, attempt_provider=attempt, verifier=verifier)

    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    assert [item[0] for item in attempt.calls] == [
        "resume-attempt-model",
        "resume-attempt-model",
    ]
    stored = work_run_store.get_work_run(session_id=session_id, work_run_id="resume-run")
    assert len(stored.attempts) == 1
    assert stored.attempts[0].turn_id == second_turn_id
    assert stored.attempts[0].input_turn_id == first_turn_id


def test_decided_protected_tool_attempt_recovers_without_resending_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, first_turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    prefix = "protected-recovery"
    id_plan = _ids(prefix)
    handler_invocations = 0
    authority_revalidations = 0

    def transmit(arguments: dict[str, str]) -> dict[str, str]:
        nonlocal handler_invocations
        handler_invocations += 1
        return {"value": arguments["value"].upper()}

    def revalidate_authority() -> bool:
        nonlocal authority_revalidations
        authority_revalidations += 1
        return True

    registration = ToolRegistration(
        spec=ToolSpec(
            tool_id="vision.analyze",
            contract_version="contract-1",
            name="Analyze image",
            description="Analyze one image through an approved external provider.",
            input_schema={
                "type": "object",
                "required": ["value"],
                "properties": {"value": {"type": "string"}},
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "required": ["value"],
                "properties": {"value": {"type": "string"}},
                "additionalProperties": False,
            },
            catalog_tags=("research",),
        ),
        implementation_version="1+protected@1",
        source=ToolSourceDescriptor(
            ToolSourceKind.LOCAL,
            "auxiliary-v2-protected-recovery-test",
        ),
        handler=transmit,
        effect_profile=ToolEffectProfile(
            (
                EffectDescriptor(
                    EffectResource.NETWORK,
                    EffectAction.TRANSMIT,
                    EffectScopeKind.SESSION,
                    data_egress=DataEgress.CONTENT,
                ),
            )
        ),
        execution_profile=ToolExecutionProfile(),
    )
    catalog = ToolCatalog()
    catalog.register(registration)
    snapshot = catalog.snapshot()

    def persistence_plan(request):
        return ToolBridgePersistencePlan(
            decision_apply_id=request.apply_id,
            close_apply_id=f"{request.attempt_id}:tool-close",
            calls=tuple(
                ToolBridgeCallPersistence(
                    tool_call_id=call.tool_call_id,
                    tool_result_id=f"{call.tool_call_id}:result",
                    result_apply_id=f"{call.tool_call_id}:result-apply",
                )
                for call in request.decision.action.calls
            ),
        )

    bridge = SqliteWorkRunToolBridge(
        catalog_snapshot=snapshot,
        persistence_plan_factory=persistence_plan,
        authority=AuthorityFacts(
            approval_grants=(
                ScopeGrant(
                    EffectResource.NETWORK,
                    EffectAction.TRANSMIT,
                    EffectScopeKind.SESSION,
                    "*",
                ),
            )
        ),
        protected_dispatcher=RuntimeProtectedToolDispatcher(
            provider_identity_sha256="c" * 64,
            ledger_store=store,
        ),
        protected_authority_by_key={
            (registration.tool_id, registration.contract_version): (
                ProtectedToolExecutionAuthority(
                    approval_receipt_ids=("approval-receipt-01",),
                    execution_backend_identity_sha256="d" * 64,
                    revalidate=revalidate_authority,
                )
            )
        },
    )
    first_request = _request(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix=prefix,
    )
    first_provider = _ReplyProvider(
        (
            json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "call_tools",
                        "calls": [
                            {
                                "tool_id": registration.tool_id,
                                "arguments": {"value": "chart"},
                            }
                        ],
                    },
                }
            ),
        )
    )
    original_append = work_run_store.append_work_run_tool_result

    def die_after_provider_success(**_kwargs: object):
        raise SystemExit("injected process death after protected provider success")

    monkeypatch.setattr(
        work_run_store,
        "append_work_run_tool_result",
        die_after_provider_success,
    )
    with pytest.raises(SystemExit, match="after protected provider success"):
        run_auxiliary_model_node(
            first_request,
            profile=_profile(),
            capability_catalogs={"readonly_documents_v1": snapshot},
            capability_tool_bridges={"readonly_documents_v1": bridge},
            attempt_provider=first_provider,
            verification_provider=_PassVerifier(),
            emit=lambda _event: None,
            monotonic_clock=_clock(),
        )

    assert handler_invocations == 1
    stranded = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    assert stranded.attempts[0].action == "call_tools"
    assert stranded.attempts[0].decision is not None
    assert stranded.tool_results == ()
    stable_tool_call_id = id_plan.tool_call_id(1, 1)
    assert tuple(item.call.tool_call_id for item in stranded.tool_calls) == (
        stable_tool_call_id,
    )

    monkeypatch.setattr(
        work_run_store,
        "append_work_run_tool_result",
        original_append,
    )
    second_turn_id = _interrupt_and_accept_followup_turn(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        client_request_id="protected-recovery-followup",
        stage="TOOL",
    )
    second_request = _request(
        session_id=session_id,
        turn_id=second_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix=prefix,
    )
    bridge_without_protected_recovery = SqliteWorkRunToolBridge(
        catalog_snapshot=snapshot,
        persistence_plan_factory=persistence_plan,
        authority=AuthorityFacts(
            approval_grants=(
                ScopeGrant(
                    EffectResource.NETWORK,
                    EffectAction.TRANSMIT,
                    EffectScopeKind.SESSION,
                    "*",
                ),
            )
        ),
    )
    denied_provider = _ReplyProvider(())
    denied = run_auxiliary_model_node(
        second_request,
        profile=_profile(),
        capability_catalogs={"readonly_documents_v1": snapshot},
        capability_tool_bridges={
            "readonly_documents_v1": bridge_without_protected_recovery
        },
        attempt_provider=denied_provider,
        verification_provider=_PassVerifier(),
        emit=lambda _event: None,
        monotonic_clock=_clock(),
    )
    assert denied.status is AuxiliaryWorkRunStatus.FAILED_CLOSED
    assert denied.reason_code == "v2_decided_tool_recovery_authority_rejected"
    assert denied_provider.calls == []
    assert work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    ).attempts[0].turn_id == first_turn_id

    followup_provider = _ReplyProvider(
        (
            json.dumps(
                {
                    "acceptance_updates": [],
                    "action": {
                        "kind": "request_user_input",
                        "question": "请确认是否继续。",
                    },
                },
                ensure_ascii=False,
            ),
        )
    )

    resumed = run_auxiliary_model_node(
        second_request,
        profile=_profile(),
        capability_catalogs={"readonly_documents_v1": snapshot},
        capability_tool_bridges={"readonly_documents_v1": bridge},
        attempt_provider=followup_provider,
        verification_provider=_PassVerifier(),
        emit=lambda _event: None,
        monotonic_clock=_clock(),
    )

    assert resumed.status is AuxiliaryWorkRunStatus.WAITING_USER
    assert len(followup_provider.calls) == 1
    assert handler_invocations == 1
    assert authority_revalidations >= 2
    recovered = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=id_plan.work_run_id,
    )
    assert recovered.attempts[0].turn_id == second_turn_id
    assert recovered.attempts[0].input_turn_id == first_turn_id
    assert recovered.tool_results[0].tool_call_id == stable_tool_call_id
    logical = store.get_runtime_tool_logical_call(
        session_id=session_id,
        logical_tool_call_id=stable_tool_call_id,
    )
    assert logical is not None
    assert logical.request.invocation_turn_id == first_turn_id
    assert logical.request.provider_identity_sha256 == "d" * 64
    assert len(logical.physical_attempts) == 1


def test_retryable_durable_attempt_continues_same_logical_call_across_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """轮次租约变化不得创建新的语义模型请求。"""

    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    session_id, first_turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    first_request = _request(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="durable-retry-resume",
    )
    attempt = _ReplyProvider(
        [
            ModelGatewayError(
                "MODEL_CALL_TIMEOUT",
                "the first Turn lost its remaining wall-clock budget",
                retryable=True,
            ),
            _submit_text(),
        ],
        provider="mock",
        model="mock-structured",
    )
    verifier = _PassVerifier(provider="mock", model="mock-structured")

    class _ExpireBeforeSecondPhysicalAttempt:
        def __init__(self) -> None:
            self.checks = 0

        def expired(self) -> bool:
            self.checks += 1
            return self.checks > 1

        def remaining_s(self) -> float:
    # 保持首次物理请求内部一致：截止时间只在下一次循环边界检查时到期。
            return 1.0

    first = _run(
        first_request,
        attempt_provider=attempt,
        verifier=verifier,
        deadline=_ExpireBeforeSecondPhysicalAttempt(),
        model_call_authority_factory=(
            partial(create_auxiliary_work_run_model_call_authority, ledger_store=store)
        ),
    )

    assert first.status is AuxiliaryWorkRunStatus.TURN_LIMIT_REACHED
    logical_before = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id="durable-retry-resume-attempt-model",
    )
    assert logical_before is not None
    assert logical_before.request.invocation_turn_id == first_turn_id
    assert len(logical_before.physical_attempts) == 1
    assert (
        logical_before.physical_attempts[0].settlement.outcome.value
        == "retryable_failure"
    )

    second_turn_id = _interrupt_and_accept_followup_turn(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        client_request_id="durable-retry-resume-followup",
    )
    second_request = _request(
        session_id=session_id,
        turn_id=second_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="durable-retry-resume",
    )

    completed = _run(
        second_request,
        attempt_provider=attempt,
        verifier=verifier,
        model_call_authority_factory=(
            partial(create_auxiliary_work_run_model_call_authority, ledger_store=store)
        ),
    )

    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    logical_after = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id="durable-retry-resume-attempt-model",
    )
    assert logical_after is not None
    assert logical_after.request == logical_before.request
    assert tuple(
        item.request.started_turn_id for item in logical_after.physical_attempts
    ) == (first_turn_id, second_turn_id)
    assert tuple(
        item.settlement.outcome.value for item in logical_after.physical_attempts
    ) == ("retryable_failure", "succeeded")
    assert [call[0] for call in attempt.calls] == [
        logical_after.physical_attempts[0].request.model_call_id,
        logical_after.physical_attempts[1].request.model_call_id,
    ]


def test_durable_attempt_recovers_output_repair_prompt_across_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    session_id, first_turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    first_request = _request(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="durable-repair-resume",
    )
    rejected = json.dumps(
        {
            "acceptance_updates": [],
            "action": {
                "kind": "submit_output_window",
                "content": "wrong format enum",
                "format": "json",
            },
        }
    )
    attempt = _ReplyProvider(
        [rejected, _submit_text()],
        provider="mock",
        model="mock-structured",
    )
    verifier = _PassVerifier(provider="mock", model="mock-structured")

    class _ExpireBeforeSecondPhysicalAttempt:
        def __init__(self) -> None:
            self.checks = 0

        def expired(self) -> bool:
            self.checks += 1
            return self.checks > 1

        def remaining_s(self) -> float:
    # 保持首次物理请求内部一致：截止时间只在下一次循环边界检查时到期。
            return 1.0

    first = _run(
        first_request,
        attempt_provider=attempt,
        verifier=verifier,
        deadline=_ExpireBeforeSecondPhysicalAttempt(),
        model_call_authority_factory=(
            partial(create_auxiliary_work_run_model_call_authority, ledger_store=store)
        ),
    )

    assert first.status is AuxiliaryWorkRunStatus.TURN_LIMIT_REACHED
    logical_before = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id="durable-repair-resume-attempt-model",
    )
    assert logical_before is not None
    assert len(logical_before.physical_attempts) == 1
    first_settlement = logical_before.physical_attempts[0].settlement
    assert first_settlement is not None
    assert first_settlement.next_output_repair_feedback is not None

    prepared_calls: list[dict[str, object]] = []

    class _PreparedLegacyRecovery:
        def __init__(
            self,
            system_prompt: str,
            user_content: str,
            purpose: str,
        ) -> None:
            self.system_prompt = system_prompt
            self.user_content = user_content
            self.purpose = purpose

        def dispatch(self, *, model_call_id: str) -> ModelResult:
            return attempt(
                self.system_prompt,
                self.user_content,
                model_call_id=model_call_id,
                purpose=self.purpose,
            )

    def prepare_after_upgrade(
        system_prompt: str,
        user_content: str,
        *,
        purpose: str,
        repair_messages: list[dict[str, str]] | None = None,
    ) -> _PreparedLegacyRecovery:
        prepared: dict[str, object] = {
            "system_prompt": system_prompt,
            "user_content": user_content,
            "purpose": purpose,
        }
        if repair_messages is not None:
            prepared["repair_messages"] = repair_messages
        prepared_calls.append(prepared)
        return _PreparedLegacyRecovery(system_prompt, user_content, purpose)

    attempt.prepare = prepare_after_upgrade  # type: ignore[attr-defined]

    second_turn_id = _interrupt_and_accept_followup_turn(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        client_request_id="durable-repair-resume-followup",
    )
    second_request = _request(
        session_id=session_id,
        turn_id=second_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="durable-repair-resume",
    )

    completed = _run(
        second_request,
        attempt_provider=attempt,
        verifier=verifier,
        model_call_authority_factory=(
            partial(create_auxiliary_work_run_model_call_authority, ledger_store=store)
        ),
    )

    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    first_prompt, recovered_prompt = (
        json.loads(call[2]) for call in attempt.calls
    )
    assert "host_repair_feedback" not in first_prompt
    assert recovered_prompt == first_prompt
    assert rejected not in attempt.calls[1][2]
    assert prepared_calls
    repair_prepared = next(
        item for item in prepared_calls if "repair_messages" in item
    )
    repair_messages = repair_prepared["repair_messages"]
    assert [item["role"] for item in repair_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert repair_messages[0]["content"] == repair_prepared["system_prompt"]
    assert repair_messages[1]["content"] == repair_prepared["user_content"]
    assert repair_messages[2]["content"] == rejected
    recovered_feedback = json.loads(
        repair_messages[3]["content"].split("Host 修复清单：", 1)[1]
    )
    assert recovered_feedback == {"current_issues": [
        {"paths": list(issue.paths), "safe_explanation": issue.safe_explanation}
        for issue in first_settlement.next_output_repair_feedback.current_issues
    ]}
    assert rejected not in repair_messages[3]["content"]
    logical_after = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id="durable-repair-resume-attempt-model",
    )
    assert logical_after is not None
    assert logical_after.request == logical_before.request
    assert tuple(
        item.request.started_turn_id for item in logical_after.physical_attempts
    ) == (first_turn_id, second_turn_id)
    assert logical_after.physical_attempts[1].request.output_repair_feedback == (
        first_settlement.next_output_repair_feedback
    )


@pytest.mark.parametrize("settled_uncertain", (False, True))
def test_pending_or_uncertain_durable_attempt_waits_for_external_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
    settled_uncertain: bool,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    session_id, first_turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    first_request = _request(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="durable-external-wait",
    )

    class _SimulatedProcessLoss(BaseException):
        pass

    attempt = _ReplyProvider(
        [_SimulatedProcessLoss("provider dispatch result was not observed")],
        provider="mock",
        model="mock-structured",
    )
    captured_authorities = []

    def capture_authority(binding, **kwargs):
        authority = create_auxiliary_work_run_model_call_authority(
            binding,
            ledger_store=store,
            **kwargs,
        )
        captured_authorities.append(authority)
        return authority

    with pytest.raises(_SimulatedProcessLoss):
        _run(
            first_request,
            attempt_provider=attempt,
            verifier=_PassVerifier(provider="mock", model="mock-structured"),
            model_call_authority_factory=capture_authority,
        )
    assert len(captured_authorities) == 1
    if settled_uncertain:
        captured_authorities[0].settle_pending_external(
            turn_id=first_turn_id,
            outcome="uncertain",
            result_fingerprint="c" * 64,
            provider_request_id="provider-request-awaiting-reconciliation",
            error_code="provider_result_uncertain",
        )
    snapshot = captured_authorities[0].inspect_recovery()
    assert snapshot.disposition == (
        "waiting_uncertain" if settled_uncertain else "waiting_pending"
    )

    second_turn_id = _interrupt_and_accept_followup_turn(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        client_request_id=(
            "durable-uncertain-followup"
            if settled_uncertain
            else "durable-pending-followup"
        ),
    )
    second_request = _request(
        session_id=session_id,
        turn_id=second_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="durable-external-wait",
    )
    waiting = _run(
        second_request,
        attempt_provider=_ReplyProvider(
            [],
            provider="mock",
            model="mock-structured",
        ),
        verifier=_PassVerifier(provider="mock", model="mock-structured"),
        model_call_authority_factory=(
            partial(create_auxiliary_work_run_model_call_authority, ledger_store=store)
        ),
    )

    assert waiting.status is AuxiliaryWorkRunStatus.WAITING_EXTERNAL
    assert waiting.reason_code == (
        "v2_attempt_model_call_waiting_external_reconciliation"
    )
    logical = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id="durable-external-wait-attempt-model",
    )
    assert logical is not None
    assert len(logical.physical_attempts) == 1
    assert len(attempt.calls) == 1


def test_pending_verification_resumes_across_turns_and_settles_once() -> None:
    session_id, first_turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    attempt = _ReplyProvider([_submit_text()])
    verifier = _PassVerifier(
        failures=(
            ModelGatewayError(
                "FAKE_VERIFY_FIRST_TURN",
                "first verifier invocation was interrupted",
                retryable=False,
            ),
            ModelGatewayError(
                "FAKE_VERIFY_SECOND_TURN",
                "second verifier invocation was interrupted",
                retryable=False,
            ),
        )
    )
    first_request = _request(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="verification-resume",
    )

    first = _run(first_request, attempt_provider=attempt, verifier=verifier)

    assert first.status is AuxiliaryWorkRunStatus.VERIFICATION_INTERRUPTED
    after_first = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id="verification-resume-run",
    )
    with store._connect() as conn:
        request_before = tuple(
            conn.execute(
                "SELECT verification_request_id, request_turn_id, "
                "submitted_attempt_id, output_revision, request_binding_hash, "
                "request_revision, status FROM "
                "insession_work_run_verification_requests "
                "WHERE verification_request_id=?",
                ("verification-resume-verification",),
            ).fetchone()
        )
    assert request_before[-2:] == (1, "pending")
    first_charge_count = len(after_first.budget_charges)
    first_active_seconds = after_first.work_run.budget.active_seconds_consumed
    second_turn_id = _interrupt_and_accept_followup_turn(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        client_request_id="verification-resume-second-turn",
    )
    second_request = _request(
        session_id=session_id,
        turn_id=second_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="verification-resume",
    )

    second = _run(second_request, attempt_provider=attempt, verifier=verifier)

    assert second.status is AuxiliaryWorkRunStatus.VERIFICATION_INTERRUPTED
    after_second = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id="verification-resume-run",
    )
    assert len(after_second.budget_charges) == first_charge_count
    assert (
        after_second.work_run.budget.active_seconds_consumed
        == first_active_seconds
    )
    third_turn_id = _interrupt_and_accept_followup_turn(
        session_id=session_id,
        turn_id=second_turn_id,
        task_id=task_id,
        client_request_id="verification-resume-third-turn",
    )
    third_request = _request(
        session_id=session_id,
        turn_id=third_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="verification-resume",
    )

    completed = _run(third_request, attempt_provider=attempt, verifier=verifier)

    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED, (
        completed.reason_code
    )
    assert attempt.calls[0][0] == "verification-resume-attempt-model"
    assert [item[0] for item in verifier.calls] == [
        "verification-resume-verification-model",
        "verification-resume-verification-model",
        "verification-resume-verification-model",
    ]
    settled = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id="verification-resume-run",
    )
    assert len(settled.budget_charges) == first_charge_count + 1
    assert (
        settled.work_run.budget.active_seconds_consumed
        == first_active_seconds + 1.0
    )
    replayed = _run(
        third_request,
        attempt_provider=_ReplyProvider([]),
        verifier=_PassVerifier(),
    )
    assert replayed == completed
    replayed_state = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id="verification-resume-run",
    )
    assert replayed_state.work_run.budget == settled.work_run.budget
    assert replayed_state.budget_charges == settled.budget_charges
    with store._connect() as conn:
        receipt_rows = conn.execute(
            "SELECT invocation_turn_id FROM "
            "insession_auxiliary_v2_execution_apply_receipts "
            "WHERE work_run_id=? AND operation='resume_verification' "
            "ORDER BY created_at, apply_id",
            ("verification-resume-run",),
        ).fetchall()
        request_after = tuple(
            conn.execute(
                "SELECT verification_request_id, request_turn_id, "
                "submitted_attempt_id, output_revision, request_binding_hash, "
                "request_revision, status FROM "
                "insession_work_run_verification_requests "
                "WHERE verification_request_id=?",
                ("verification-resume-verification",),
            ).fetchone()
        )
    assert tuple(str(row[0]) for row in receipt_rows) == (
        second_turn_id,
        third_turn_id,
    )
    assert request_after[:5] == request_before[:5]
    assert request_after[-2:] == (4, "completed")


def test_retryable_durable_verification_continues_same_logical_call_across_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    session_id, first_turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    attempt = _ReplyProvider(
        [_submit_text()],
        provider="mock",
        model="mock-structured",
    )
    verifier = _PassVerifier(
        failures=(
            ModelGatewayError(
                "MODEL_CALL_TIMEOUT",
                "verification retry crosses the Turn wall-clock boundary",
                retryable=True,
            ),
        ),
        provider="mock",
        model="mock-structured",
    )

    class _ExpireBeforeVerificationRetry:
        def __init__(self) -> None:
            self.checks = 0

        def expired(self) -> bool:
            self.checks += 1
            return self.checks > 2

        def remaining_s(self) -> float:
    # 尝试与首次验证器调用仍获授权；在下一次循环边界检查时注入到期状态。
            return 1.0

    first_request = _request(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="durable-verification-resume",
    )
    first = _run(
        first_request,
        attempt_provider=attempt,
        verifier=verifier,
        deadline=_ExpireBeforeVerificationRetry(),
        model_call_authority_factory=(
            partial(create_auxiliary_work_run_model_call_authority, ledger_store=store)
        ),
    )

    assert first.status is AuxiliaryWorkRunStatus.TURN_LIMIT_REACHED
    logical_before = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id="durable-verification-resume-verification-model",
    )
    assert logical_before is not None
    assert len(logical_before.physical_attempts) == 1
    assert (
        logical_before.physical_attempts[0].settlement.outcome.value
        == "retryable_failure"
    )

    second_turn_id = _interrupt_and_accept_followup_turn(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        client_request_id="durable-verification-resume-followup",
    )
    second_request = _request(
        session_id=session_id,
        turn_id=second_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="durable-verification-resume",
    )
    completed = _run(
        second_request,
        attempt_provider=attempt,
        verifier=verifier,
        model_call_authority_factory=(
            partial(create_auxiliary_work_run_model_call_authority, ledger_store=store)
        ),
    )

    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    logical_after = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id="durable-verification-resume-verification-model",
    )
    assert logical_after is not None
    assert logical_after.request == logical_before.request
    assert tuple(
        item.request.started_turn_id for item in logical_after.physical_attempts
    ) == (first_turn_id, second_turn_id)
    assert tuple(
        item.settlement.outcome.value for item in logical_after.physical_attempts
    ) == ("retryable_failure", "succeeded")


def test_pending_durable_verification_is_not_redispatched_after_turn_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    session_id, first_turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        executor="model_work_run",
    )

    class _SimulatedVerifierProcessLoss(BaseException):
        pass

    attempt = _ReplyProvider(
        [_submit_text()],
        provider="mock",
        model="mock-structured",
    )
    verifier = _PassVerifier(
        failures=(
            _SimulatedVerifierProcessLoss(
                "verification response was not durably observed"
            ),
        ),
        provider="mock",
        model="mock-structured",
    )
    first_request = _request(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="durable-pending-verification",
    )
    with pytest.raises(_SimulatedVerifierProcessLoss):
        _run(
            first_request,
            attempt_provider=attempt,
            verifier=verifier,
            model_call_authority_factory=(
                partial(create_auxiliary_work_run_model_call_authority, ledger_store=store)
            ),
        )

    second_turn_id = _interrupt_and_accept_followup_turn(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        client_request_id="durable-pending-verification-followup",
    )
    second_request = _request(
        session_id=session_id,
        turn_id=second_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="durable-pending-verification",
    )
    waiting = _run(
        second_request,
        attempt_provider=_ReplyProvider(
            [],
            provider="mock",
            model="mock-structured",
        ),
        verifier=_PassVerifier(provider="mock", model="mock-structured"),
        model_call_authority_factory=(
            partial(create_auxiliary_work_run_model_call_authority, ledger_store=store)
        ),
    )

    assert waiting.status is AuxiliaryWorkRunStatus.WAITING_EXTERNAL
    assert waiting.reason_code == (
        "v2_verification_model_call_waiting_external_reconciliation"
    )
    logical = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id="durable-pending-verification-verification-model",
    )
    assert logical is not None
    assert len(logical.physical_attempts) == 1
    assert logical.physical_attempts[0].settlement is None
    assert len(verifier.calls) == 1


def test_applied_verification_resume_response_loss_reenters_prepared_request(
    monkeypatch,
) -> None:
    session_id, first_turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    attempt = _ReplyProvider([_submit_text()])
    verifier = _PassVerifier(
        failures=(
            ModelGatewayError(
                "FAKE_VERIFY_RESPONSE_LOSS",
                "source Turn verifier invocation was interrupted",
                retryable=False,
            ),
        )
    )
    first_request = _request(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="verification-resume-response-loss",
    )
    interrupted = _run(
        first_request,
        attempt_provider=attempt,
        verifier=verifier,
    )
    assert interrupted.status is AuxiliaryWorkRunStatus.VERIFICATION_INTERRUPTED
    before_resume = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id="verification-resume-response-loss-run",
    )
    next_turn_id = _interrupt_and_accept_followup_turn(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        client_request_id="verification-resume-response-loss-turn",
    )
    next_request = _request(
        session_id=session_id,
        turn_id=next_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="verification-resume-response-loss",
    )
    original_resume = continuation_store.resume_auxiliary_verification

    def apply_then_lose_response(*, command):
        original_resume(command=command)
        raise continuation_store.AuxiliaryContinuationPersistenceError(
            "simulated response loss after commit"
        )

    monkeypatch.setattr(
        continuation_store,
        "resume_auxiliary_verification",
        apply_then_lose_response,
    )
    lost = _run(next_request, attempt_provider=attempt, verifier=verifier)

    assert lost.status is AuxiliaryWorkRunStatus.FAILED_CLOSED
    assert lost.reason_code == "v2_verification_resume_authority_rejected"
    after_lost_response = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id="verification-resume-response-loss-run",
    )
    assert after_lost_response.work_run.budget == before_resume.work_run.budget
    assert after_lost_response.budget_charges == before_resume.budget_charges
    assert len(verifier.calls) == 1

    monkeypatch.setattr(
        continuation_store,
        "resume_auxiliary_verification",
        original_resume,
    )
    completed = _run(next_request, attempt_provider=attempt, verifier=verifier)

    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    assert len(verifier.calls) == 2
    settled = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id="verification-resume-response-loss-run",
    )
    assert len(settled.budget_charges) == len(before_resume.budget_charges) + 1
    assert (
        settled.work_run.budget.active_seconds_consumed
        == before_resume.work_run.budget.active_seconds_consumed + 1.0
    )
    with store._connect() as conn:
        receipt_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM "
                "insession_auxiliary_v2_execution_apply_receipts "
                "WHERE work_run_id=? AND operation='resume_verification'",
                ("verification-resume-response-loss-run",),
            ).fetchone()[0]
        )
    assert receipt_count == 1


@pytest.mark.parametrize(
    "rejection_kind",
    ("stale-request-revision", "apply-id-collision"),
)
def test_verification_resume_stale_or_collision_fails_before_provider(
    monkeypatch,
    rejection_kind: str,
) -> None:
    session_id, first_turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    attempt = _ReplyProvider([_submit_text()])
    verifier = _PassVerifier(
        failures=(
            ModelGatewayError(
                "FAKE_VERIFY_AUTHORITY_REJECTION",
                "source Turn verifier invocation was interrupted",
                retryable=False,
            ),
        )
    )
    first_request = _request(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="verification-resume-rejected",
    )
    interrupted = _run(
        first_request,
        attempt_provider=attempt,
        verifier=verifier,
    )
    assert interrupted.status is AuxiliaryWorkRunStatus.VERIFICATION_INTERRUPTED
    before = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id="verification-resume-rejected-run",
    )
    next_turn_id = _interrupt_and_accept_followup_turn(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        client_request_id="verification-resume-rejected-turn",
    )
    next_request = _request(
        session_id=session_id,
        turn_id=next_turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="verification-resume-rejected",
    )

    def reject_resume(*, command):
        assert command.verification_request_id == (
            "verification-resume-rejected-verification"
        )
        if rejection_kind == "stale-request-revision":
            raise verification_store.TaskNodeVerificationRequestRevisionConflict(
                expected=1,
                actual=2,
            )
        raise continuation_store.AuxiliaryContinuationApplyIdCollision(
            "simulated verification resume apply collision"
        )

    monkeypatch.setattr(
        continuation_store,
        "resume_auxiliary_verification",
        reject_resume,
    )
    stopped = _run(next_request, attempt_provider=attempt, verifier=verifier)

    assert stopped.status is AuxiliaryWorkRunStatus.FAILED_CLOSED
    assert stopped.reason_code == "v2_verification_resume_authority_rejected"
    assert len(attempt.calls) == 1
    assert len(verifier.calls) == 1
    after = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id="verification-resume-rejected-run",
    )
    assert after.work_run.budget == before.work_run.budget
    assert after.budget_charges == before.budget_charges


def test_nonpass_verification_drives_stable_second_attempt_and_completion() -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(session_id=session_id, turn_id=turn_id, task_id=task_id)
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="repair",
    )
    attempt = _ReplyProvider([_submit_text(), _submit_text()])
    verifier = _PassVerifier(verdicts=("not_satisfied", "passed"))

    result = _run(request, attempt_provider=attempt, verifier=verifier)

    assert result.status is AuxiliaryWorkRunStatus.COMPLETED
    assert result.attempt_id == "repair-attempt:attempt-2"
    assert result.verification_request_id == "repair-verification:attempt-2"
    assert result.completion_id == "repair-completion:attempt-2"
    assert [item[0] for item in attempt.calls] == [
        "repair-attempt-model",
        "repair-attempt-model:attempt-2",
    ]
    assert [item[0] for item in verifier.calls] == [
        "repair-verification-model",
        "repair-verification-model:attempt-2",
    ]
    second_prompt = json.loads(attempt.calls[1][2])
    assert second_prompt["verification_feedback"] is not None
    assert (
        second_prompt["verification_feedback"]["acceptance_results"][0]["verdict"]
        == "not_satisfied"
    )


def test_terminal_downstream_retry_uses_same_work_run_attempt_before_freeze() -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        terminal_only=True,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="terminal_planner",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="terminal_planner",
        prefix="terminal-semantic-retry",
    )
    attempt = _ReplyProvider([_task_graph_submit(), _task_graph_submit()])
    verifier = _PassVerifier()
    gate_calls = 0

    def gate(context, result):
        nonlocal gate_calls
        gate_calls += 1
        assert result.all_pass is True
        assert context.subject == subject
        if gate_calls == 1:
            with store._connect() as conn:
                completion_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM insession_auxiliary_node_completions_v2 "
                        "WHERE work_run_id=?",
                        (context.work_run_id,),
                    ).fetchone()[0]
                )
                frozen_at = conn.execute(
                    "SELECT frozen_at FROM insession_work_run_output_windows "
                    "WHERE work_run_id=?",
                    (context.work_run_id,),
                ).fetchone()[0]
            assert completion_count == 0
            assert frozen_at is None
            return (
                DownstreamVerificationFeedback(
                    gate_id="task_graph_semantic",
                    disposition=(
                        DownstreamVerificationDisposition.RETRY_ATTEMPT
                    ),
                    finding="终端提案漏掉一个必要 Acceptance。",
                    repair_objective="保留图结构并补齐该 Acceptance。",
                    source_result_id="semantic-result-terminal-retry-1",
                    source_result_sha256="1" * 64,
                    affected_subject_ids=(subject.node_id,),
                ),
            )
        return (
            DownstreamVerificationFeedback(
                gate_id="task_graph_semantic",
                disposition=DownstreamVerificationDisposition.PASS,
                finding="终端提案已通过独立语义审查。",
                source_result_id="semantic-result-terminal-retry-2",
                source_result_sha256="2" * 64,
                affected_subject_ids=(),
            ),
        )

    completed = _run(
        request,
        attempt_provider=attempt,
        verifier=verifier,
        terminal_downstream_gate=gate,
    )

    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    assert completed.work_run_id == request.id_plan.work_run_id
    assert completed.attempt_id.endswith(":attempt-2")
    assert gate_calls == 2
    assert len(attempt.calls) == 2
    second_prompt = json.loads(attempt.calls[1][2])
    downstream = second_prompt["verification_feedback"]["downstream_results"]
    assert downstream[0]["gate_id"] == "task_graph_semantic"
    assert downstream[0]["disposition"] == "retry_attempt"


def test_terminal_semantic_gate_latency_does_not_exhaust_node_work_run() -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        terminal_only=True,
    )
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="terminal_planner",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="terminal_planner",
        prefix="terminal-semantic-latency",
    )
    now = 0.0

    def clock() -> float:
        nonlocal now
        current = now
        now += 1.0
        return current

    def slow_passing_gate(_context, _result):
        nonlocal now
        now += 901.0
        return (
            DownstreamVerificationFeedback(
                gate_id="task_graph_semantic",
                disposition=DownstreamVerificationDisposition.PASS,
                finding="独立语义审查通过。",
                source_result_id="semantic-result-slow-pass",
                source_result_sha256="3" * 64,
                affected_subject_ids=(),
            ),
        )

    completed = run_auxiliary_model_node(
        request,
        profile=_profile(),
        capability_catalogs={
            "readonly_documents_v1": CatalogSnapshot(revision=1, entries=())
        },
        attempt_provider=_ReplyProvider([_task_graph_submit()]),
        verification_provider=_PassVerifier(),
        emit=lambda _event: None,
        monotonic_clock=clock,
        terminal_downstream_gate=slow_passing_gate,
    )

    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=request.id_plan.work_run_id,
    )
    assert stored.work_run.budget.active_seconds_consumed == 2.0


def test_turn_deadline_preserves_the_exact_attempt_for_same_turn_recovery() -> None:
    session_id, turn_id, task_id = _seed_task()
    _commit_graph(session_id=session_id, turn_id=turn_id, task_id=task_id)
    subject = _subject(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        executor="model_work_run",
    )
    request = _request(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        subject=subject,
        executor="model_work_run",
        prefix="deadline",
    )
    attempt = _ReplyProvider([_submit_text()])
    verifier = _PassVerifier()

    stopped = run_auxiliary_model_node(
        request,
        profile=_profile(),
        capability_catalogs={
            "readonly_documents_v1": CatalogSnapshot(revision=1, entries=())
        },
        attempt_provider=attempt,
        verification_provider=verifier,
        emit=lambda _event: None,
        monotonic_clock=_clock(),
        deadline=TurnDeadline(expires_at_monotonic=0.0),
    )
    completed = _run(request, attempt_provider=attempt, verifier=verifier)

    assert (
        stopped.status
        is AuxiliaryWorkRunStatus.TURN_LIMIT_REACHED
    )
    assert stopped.reason_code == "v2_turn_deadline_exceeded_with_active_attempt"
    assert stopped.attempt_id == "deadline-attempt"
    assert completed.status is AuxiliaryWorkRunStatus.COMPLETED
    assert [item[0] for item in attempt.calls] == ["deadline-attempt-model"]
