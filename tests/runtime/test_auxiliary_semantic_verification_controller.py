"""现行 Auxiliary TaskGraph 语义验证 controller 测试。"""

from __future__ import annotations

import json
import threading
from concurrent.futures import Future
from functools import partial

import pytest

from personagraph.l2.auxiliary_graph import (
    PlanningAuthorityClass,
    TaskGraphSemanticVerificationRequest,
    TaskGraphSemanticVerificationDimension,
    TaskGraphSemanticVerificationDisposition,
    TaskGraphSemanticVerificationVerdict,
)
from personagraph.l2.task_graph.task_matching import InSessionTaskMatchesProposal
from personagraph.model_io.gateway import ModelGatewayError, ModelResult, PreparedModelCall
from personagraph.l2.auxiliary_execution.verification import controller
from personagraph.l2.auxiliary_execution.application import (
    AuxiliaryApplicationPorts,
    AuxiliaryApplicationRequest,
    AuxiliaryApplicationStatus,
    run_auxiliary_application_to_boundary,
)
from personagraph.l2.auxiliary_execution.adapters.model_authority import (
    create_auxiliary_semantic_reviewer_model_call_authority,
)
from personagraph.runtime.model_calls import (
    RuntimeModelLogicalRequest,
)
from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairProtocol,
    RuntimeModelStructuredPrompt,
)
from personagraph.runtime.model_calls import MAX_MODEL_ATTEMPTS
from personagraph.runtime.model_calls import (
    RuntimeLogicalModelCallAuthority,
)
from personagraph.l2.auxiliary_execution.verification.task_graph_semantic import (
    _TASK_GRAPH_SEMANTIC_VERIFICATION_SYSTEM_PROMPT,
    serialize_task_graph_semantic_verification_prompt,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import verification as verification_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import semantic_verification as semantic_store
from personagraph.session.persistence.l2.task_graph import insession_tasks as insession_task_records
from tests.runtime.test_auxiliary_planning_controller import (
    _no_mounted_documents,
    _seed_task,
)
from tests.session.test_auxiliary_semantic_verification_persistence import (
    _catalog,
    _complete_terminal_only,
    _semantic_request,
)
from tests.helpers.prepared_model_provider import as_prepared_test_provider


def _reviewer_ordinal_for_model_call(model_call_id: str) -> int:
    """在不依赖线程顺序的情况下解析正式审查器身份。"""

    with store._connect() as conn:
        row = conn.execute(
            "SELECT logical.request_json "
            "FROM insession_runtime_model_physical_attempts AS physical "
            "JOIN insession_runtime_model_logical_calls AS logical "
            "ON logical.logical_call_id=physical.logical_call_id "
            "WHERE physical.physical_attempt_key=?",
            (model_call_id,),
        ).fetchone()
    assert row is not None
    logical = json.loads(str(row["request_json"]))
    bound_request = json.loads(logical["request_json"])
    return int(bound_request["verification_request"]["reviewer_ordinal"])


class _SemanticProvider:
    def __init__(
        self,
        *,
        failed_reviewer_ordinal: int | None = None,
        failed_dimension: TaskGraphSemanticVerificationDimension = (
            TaskGraphSemanticVerificationDimension.GOAL_COVERAGE
        ),
        failure_scope: str = "terminal_proposal",
        failures: tuple[BaseException, ...] = (),
        provider: str = "semantic_test_provider",
        model: str = "semantic_test_model",
    ) -> None:
        self.calls: list[str] = []
        self.failed_reviewer_ordinal = failed_reviewer_ordinal
        self.failed_dimension = failed_dimension
        self.failure_scope = failure_scope
        self.failures = list(failures)
        self.provider = provider
        self.model = model
        self._lock = threading.Lock()
        self._failed_once = False

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
        _system_prompt: str,
        _user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        assert purpose == "runtime_task_graph_semantic_verification"
        reviewer_ordinal = _reviewer_ordinal_for_model_call(model_call_id)
        with self._lock:
            self.calls.append(model_call_id)
            failure = self.failures.pop(0) if self.failures else None
            should_fail_review = (
                reviewer_ordinal == self.failed_reviewer_ordinal
                and not self._failed_once
            )
            if should_fail_review:
                self._failed_once = True
        if failure is not None:
            raise failure
        payload = json.loads(_user_content)
        evidence_aliases = {
            item["alias"]
            for item in payload["authority"]["cards"]
            if item["authority_class"] == PlanningAuthorityClass.EVIDENCE.value
        }
        referenced_aliases = {
            alias
            for node in payload["task_graph_proposal"]["root"]["nodes"]
            for alias in (
                *node["source_anchor_ids"],
                *(
                    alias
                    for acceptance in node["acceptance_criteria"]
                    for alias in acceptance["source_anchor_ids"]
                ),
            )
        }
        required_evidence = sorted(evidence_aliases & referenced_aliases)
        gaps = sorted(
            item["gap_alias"]
            for artifact in payload["context_artifacts"]
            for item in artifact["gaps"]
        )
        node_key = payload["task_graph_proposal"]["root"]["root_key"]
        reply = {
            "items": [
                {
                    "dimension": dimension.value,
                    "verdict": (
                        TaskGraphSemanticVerificationVerdict.FAIL.value
                        if should_fail_review
                        and dimension is self.failed_dimension
                        else TaskGraphSemanticVerificationVerdict.PASS.value
                    ),
                    "failure_scope": (
                        self.failure_scope
                        if should_fail_review
                        and dimension is self.failed_dimension
                        else None
                    ),
                    "finding": (
                        f"{dimension.value} requires revision."
                        if should_fail_review
                        and dimension is self.failed_dimension
                        else f"{dimension.value} passed frozen review."
                    ),
                    "affected_node_keys": [node_key],
                    "evidence_aliases": (
                        required_evidence
                        if dimension
                        is TaskGraphSemanticVerificationDimension.EVIDENCE_GROUNDING
                        else []
                    ),
                    "gap_aliases": (
                        gaps
                        if dimension
                        is TaskGraphSemanticVerificationDimension.GAP_DISPOSITION
                        else []
                    ),
                }
                for dimension in TaskGraphSemanticVerificationDimension
            ]
        }
        return ModelResult(
            reply=json.dumps(reply, ensure_ascii=False),
            provider=self.provider,
            model=self.model,
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )


class _OverlappingSemanticProvider(_SemanticProvider):
    """除非两个审查器提供方调用都在进行，否则测试失败。"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._barrier = threading.Barrier(2)
        self._active_lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def __call__(self, *args, **kwargs) -> ModelResult:
        with self._active_lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            self._barrier.wait(timeout=5.0)
            return super().__call__(*args, **kwargs)
        finally:
            with self._active_lock:
                self._active -= 1


class _BlockingSemanticProvider(_SemanticProvider):
    """让两个提供方调用保持打开，以便观察实时 STARTED 事件。"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._active_lock = threading.Lock()
        self._active = 0
        self.max_active = 0
        self.both_entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, *args, **kwargs) -> ModelResult:
        with self._active_lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            if self._active == 2:
                self.both_entered.set()
        try:
            if not self.release.wait(timeout=5.0):
                raise AssertionError("blocked semantic Provider was not released")
            return super().__call__(*args, **kwargs)
        finally:
            with self._active_lock:
                self._active -= 1


def _authority_factory(*, open_pending: bool = False):
    bindings = []
    pending_opened: set[str] = set()

    def factory(binding, *, rederive_state_guard_sha256):
        bindings.append(binding)
        semantic_request = TaskGraphSemanticVerificationRequest.model_validate(
            json.loads(binding.request_json)["verification_request"]
        )
        logical = RuntimeModelLogicalRequest.create(
            logical_call_id=binding.logical_call_id,
            session_id=binding.session_id,
            task_id=binding.task_id,
            auxiliary_graph_id=binding.auxiliary_graph_id,
            goal_id=binding.goal_id,
            invocation_turn_id=binding.invocation_turn_id,
            call_kind="task_graph_semantic_verification",
            purpose=binding.purpose,
            provider="semantic_test_provider",
            model="semantic_test_model",
            endpoint_fingerprint="a" * 64,
            request_contract=binding.request_contract,
            request_payload=json.loads(binding.request_json),
            output_repair_protocol=(
                RuntimeModelOutputRepairProtocol.FOUR_MESSAGE_WHOLE_RESPONSE_REGENERATION
            ),
            structured_prompt=RuntimeModelStructuredPrompt.create(
                system_prompt=_TASK_GRAPH_SEMANTIC_VERIFICATION_SYSTEM_PROMPT,
                user_content=serialize_task_graph_semantic_verification_prompt(
                    semantic_request
                ),
            ),
            typed_result_contract=binding.typed_result_contract,
            max_physical_attempts=binding.max_physical_attempts,
            state_guard_sha256=binding.state_guard_sha256,
        )
        authority = RuntimeLogicalModelCallAuthority(
            logical_request=logical,
            state_guard_sha256=rederive_state_guard_sha256,
            typed_replay_payload_builder=lambda _model_result, value: {
                "items": [
                    {
                        **item.model_dump(mode="json"),
                        "failure_scope": (
                            None
                            if item.failure_scope is None
                            else item.failure_scope.value
                        ),
                    }
                    for item in value.items
                ]
            },
            store=store,
        )
        if open_pending and binding.logical_call_id not in pending_opened:
            authority.reserve(turn_id=binding.invocation_turn_id)
            authority.begin_physical_attempt(
                turn_id=binding.invocation_turn_id,
                max_physical_attempts=MAX_MODEL_ATTEMPTS,
                output_repair_enabled=True,
            )
            pending_opened.add(binding.logical_call_id)
        return authority

    return factory, bindings


def _request(prefix: str, *, protected: bool):
    session_id, turn_id, task_id, details, prompt_inputs = (
        _complete_terminal_only(prefix)
    )
    catalog = _catalog(protected=protected)
    seed = _semantic_request(
        request_id=f"{prefix}-seed-request",
        logical_call_id=f"{prefix}-seed-call",
        reviewer_ordinal=1,
        details=details,
        catalog=catalog,
        prompt_inputs=prompt_inputs,
    )
    frontier = auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
        session_id=session_id,
        turn_id=turn_id,
        insession_task_id=task_id,
    )
    reviewer_count = 2 if protected else 1
    return controller.AuxiliarySemanticVerificationRequest(
        frontier=frontier,
        prompt_payload=seed.prompt_payload,
        capability_catalog=catalog,
        reviewers=tuple(
            controller.AuxiliarySemanticReviewerIdPlan(
                reviewer_ordinal=ordinal,
                verification_profile_id=f"{prefix}-semantic-verifier",
                verification_request_id=f"{prefix}-request-{ordinal}",
                logical_call_id=f"{prefix}-call-{ordinal}",
                verification_result_id=f"{prefix}-result-{ordinal}",
            )
            for ordinal in range(1, reviewer_count + 1)
        ),
        settlement_id=f"{prefix}-settlement",
    )


def _run(request, provider, factory, *, deadline=None):
    return controller.run_auxiliary_semantic_verification(
        request,
        provider=as_prepared_test_provider(provider),
        model_call_authority_factory=factory,
        emit=lambda _event: None,
        deadline=deadline,
    )


def _window_revision(session_id: str) -> int:
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    return int(window["state_version"])


def _handoff_semantic_turn(
    request: controller.AuxiliarySemanticVerificationRequest,
    *,
    client_request_id: str,
) -> controller.AuxiliarySemanticVerificationRequest:
    marked = store.mark_turn_execution_interrupted(
        session_id=request.frontier.session_id,
        turn_id=request.frontier.turn_id,
        expected_window_revision=_window_revision(request.frontier.session_id),
        stage="VERIFICATION",
        interruption_reason="semantic_model_call_cross_turn_test",
    )
    store.settle_interrupted_turn_execution(
        session_id=request.frontier.session_id,
        turn_id=request.frontier.turn_id,
        expected_window_revision=int(marked["state_version"]),
        end_reason="semantic_model_call_cross_turn_test",
        error_code="PROCESS_LOST",
    )
    accepted = store.accept_turn_execution(
        session_id=request.frontier.session_id,
        client_request_id=client_request_id,
        source="auxiliary_v2_semantic_controller_test",
        user_text="继续完成同一个语义验证请求",
        lease_owner="aux-v2-semantic-controller-test",
    )
    next_turn_id = str(accepted["turn"]["turn_id"])
    insession_task_records.link_turn_to_insession_tasks(
        store._deps(),
        session_id=request.frontier.session_id,
        turn_id=next_turn_id,
        insession_task_ids=(request.frontier.task_id,),
        expected_window_revision=_window_revision(request.frontier.session_id),
    )
    return request.model_copy(
        update={
            "frontier": auxiliary_graph_store.project_auxiliary_graph_execution_frontier(
                session_id=request.frontier.session_id,
                turn_id=next_turn_id,
                insession_task_id=request.frontier.task_id,
            )
        }
    )


class _ExpireBeforeSecondPhysicalAttempt:
    def __init__(self) -> None:
        self.checks = 0

    def expired(self) -> bool:
        self.checks += 1
        return self.checks > 1

    def remaining_s(self) -> float:
        return 0.0 if self.checks > 1 else 30.0


def _handoff_execution_turn(
    *,
    session_id: str,
    turn_id: str,
    task_id: str,
    client_request_id: str,
) -> str:
    marked = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=_window_revision(session_id),
        stage="VERIFICATION",
        interruption_reason="candidate_semantic_cross_turn_test",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(marked["state_version"]),
        end_reason="candidate_semantic_cross_turn_test",
        error_code="PROCESS_LOST",
    )
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=client_request_id,
        source="auxiliary_v2_candidate_semantic_test",
        user_text="继续完成同一个候选图验证请求",
        lease_owner="aux-v2-candidate-semantic-test",
    )
    next_turn_id = str(accepted["turn"]["turn_id"])
    followup_text = "继续完成同一个候选图验证请求"
    task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=next_turn_id,
        apply_id=f"{client_request_id}-match",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "existing_root",
                        "insession_task_id": task_id,
                        "source_excerpt": followup_text,
                        "execute_current": True,
                    }
                ]
            }
        ),
        exposed_catalog_ids=(task_id,),
        expected_window_revision=_window_revision(session_id),
    )
    return next_turn_id


def test_one_reviewer_settles_and_reentry_skips_provider() -> None:
    request = _request("semantic-controller-one", protected=False)
    provider = _SemanticProvider()
    factory, bindings = _authority_factory()

    settled = _run(request, provider, factory)

    assert settled.status is controller.AuxiliarySemanticVerificationStatus.SETTLED
    assert settled.reason_code == "semantic_quorum_settled"
    assert settled.completed_reviewer_ordinals == (1,)
    assert settled.settlement is not None
    assert settled.settlement.required_reviewer_count == 1
    assert settled.replayed is False
    assert len(provider.calls) == 1
    assert len(bindings) == 1

    replayed = _run(request, provider, factory)

    assert replayed.status is controller.AuxiliarySemanticVerificationStatus.SETTLED
    assert replayed.settlement == settled.settlement
    assert replayed.replayed is True
    assert len(provider.calls) == 1
    assert len(bindings) == 1


def test_host_policy_dispatches_two_independent_reviewers() -> None:
    request = _request("semantic-controller-two", protected=True)
    provider = _SemanticProvider()
    factory, bindings = _authority_factory()

    settled = _run(request, provider, factory)

    assert settled.status is controller.AuxiliarySemanticVerificationStatus.SETTLED
    assert settled.completed_reviewer_ordinals == (1, 2)
    assert settled.settlement is not None
    assert settled.settlement.required_reviewer_count == 2
    assert tuple(item.reviewer_ordinal for item in settled.settlement.requests) == (
        1,
        2,
    )
    assert len(provider.calls) == 2
    assert len(set(provider.calls)) == 2
    assert [item.reviewer_ordinal for item in bindings] == [1, 2]


def test_two_reviewer_provider_io_overlaps_but_commits_and_emits_by_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request("semantic-controller-concurrent", protected=True)
    provider = _BlockingSemanticProvider()
    factory, bindings = _authority_factory()
    events = []
    committed_ordinals: list[int] = []
    results = []
    errors: list[BaseException] = []
    original_commit = semantic_store.commit_auxiliary_semantic_verification_result

    def capture_commit(*, command):
        committed_ordinals.append(command.result.reviewer_ordinal)
        return original_commit(command=command)

    monkeypatch.setattr(
        semantic_store,
        "commit_auxiliary_semantic_verification_result",
        capture_commit,
    )

    def run_controller() -> None:
        try:
            results.append(
                controller.run_auxiliary_semantic_verification(
                    request,
                    provider=provider,
                    model_call_authority_factory=factory,
                    emit=events.append,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=run_controller)
    worker.start()
    try:
        assert provider.both_entered.wait(timeout=5.0)
    # 两个慢调用仍阻塞在提供方内部，但 UI 事件端口已经按固定顺序观察到两个
    # STARTED 事件。
        assert [event.status.value for event in events] == ["started", "started"]
        assert [
            _reviewer_ordinal_for_model_call(event.model_call_id)
            for event in events
            if event.model_call_id is not None
        ] == [1, 2]
        assert committed_ordinals == []
    finally:
        provider.release.set()
        worker.join(timeout=10.0)

    assert not worker.is_alive()
    assert errors == []
    assert len(results) == 1
    settled = results[0]

    assert settled.status is controller.AuxiliarySemanticVerificationStatus.SETTLED
    assert provider.max_active == 2
    assert [item.reviewer_ordinal for item in bindings] == [1, 2]
    assert committed_ordinals == [1, 2]
    event_ordinals = [
        _reviewer_ordinal_for_model_call(event.model_call_id)
        for event in events
        if event.model_call_id is not None
    ]
    assert set(event_ordinals) == {1, 2}
    assert event_ordinals == [1, 2, 1, 2]
    assert [event.status.value for event in events] == [
        "started",
        "started",
        "completed",
        "completed",
    ]


def test_terminal_candidate_two_reviewer_pre_review_overlaps_provider_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request("semantic-candidate-concurrent", protected=True)
    request = request.model_copy(
        update={
            "terminal_candidate_binding": (
                controller.AuxiliaryTerminalCandidateBinding(
                    work_run_id="candidate-concurrent-work-run",
                    submitted_attempt_id="candidate-concurrent-attempt",
                    node_verification_request_id="candidate-concurrent-node-review",
                    output_revision=1,
                    output_snapshot_sha256="a" * 64,
                )
            )
        }
    )
    # 夹具含有已完成的终态节点。让测试聚焦候选控制器扇出，同时仍验证真实的
    # 持久模型账本与提供方边界。
    monkeypatch.setattr(
        controller,
        "_rederive_terminal_candidate_state_guard",
        lambda _request, *, semantic_request, binding: binding.state_guard_sha256,
    )
    provider = _OverlappingSemanticProvider()
    factory, bindings = _authority_factory()
    events = []

    review = controller.review_auxiliary_terminal_candidate_semantics(
        request,
        provider=provider,
        model_call_authority_factory=factory,
        emit=events.append,
    )

    assert review.route.value == "pass"
    assert provider.max_active == 2
    assert [item.reviewer_ordinal for item in bindings] == [1, 2]
    event_ordinals = [
        _reviewer_ordinal_for_model_call(event.model_call_id)
        for event in events
        if event.model_call_id is not None
    ]
    assert set(event_ordinals) == {1, 2}
    assert event_ordinals == [1, 2, 1, 2]


def test_reviewer_fail_settles_revise_and_reentry_skips_provider() -> None:
    request = _request("semantic-controller-revise", protected=True)
    provider = _SemanticProvider(failed_reviewer_ordinal=2)
    factory, bindings = _authority_factory()

    settled = _run(request, provider, factory)

    assert settled.status is controller.AuxiliarySemanticVerificationStatus.SETTLED
    assert settled.reason_code == "semantic_quorum_settled"
    assert settled.settlement is not None
    assert (
        settled.settlement.host_disposition
        is TaskGraphSemanticVerificationDisposition.REVISE
    )
    assert len(provider.calls) == 2
    assert len(bindings) == 2

    replayed = _run(request, provider, factory)

    assert replayed.status is controller.AuxiliarySemanticVerificationStatus.SETTLED
    assert replayed.settlement == settled.settlement
    assert replayed.replayed is True
    assert len(provider.calls) == 2
    assert len(bindings) == 2


def test_response_loss_replays_durable_model_result_without_provider(
    monkeypatch,
) -> None:
    request = _request("semantic-controller-loss", protected=False)
    provider = _SemanticProvider()
    factory, _bindings = _authority_factory()
    original = semantic_store.commit_auxiliary_semantic_verification_result

    def lose_before_semantic_commit(*, command):
        raise semantic_store.AuxiliarySemanticVerificationPersistenceError(
            "simulated response loss after durable Provider settlement"
        )

    monkeypatch.setattr(
        semantic_store,
        "commit_auxiliary_semantic_verification_result",
        lose_before_semantic_commit,
    )
    lost = _run(request, provider, factory)
    assert lost.status is controller.AuxiliarySemanticVerificationStatus.FAILED_CLOSED
    assert len(provider.calls) == 1
    assert semantic_store.get_auxiliary_semantic_verification_result(
        session_id=request.frontier.session_id,
        verification_result_id=request.reviewers[0].verification_result_id,
    ) is None

    monkeypatch.setattr(
        semantic_store,
        "commit_auxiliary_semantic_verification_result",
        original,
    )
    recovered = _run(request, provider, factory)

    assert recovered.status is controller.AuxiliarySemanticVerificationStatus.SETTLED
    assert recovered.settlement is not None
    assert len(provider.calls) == 1


def test_stale_frontier_and_tampered_catalog_fail_before_provider() -> None:
    request = _request("semantic-controller-tamper", protected=False)
    provider = _SemanticProvider()
    factory, _bindings = _authority_factory()

    stale = _run(
        request.model_copy(
            update={
                "frontier": request.frontier.model_copy(
                    update={
                        "revision_state_version": (
                            request.frontier.revision_state_version + 1
                        )
                    }
                )
            }
        ),
        provider,
        factory,
    )
    assert stale.status is controller.AuxiliarySemanticVerificationStatus.FAILED_CLOSED
    assert stale.reason_code == "semantic_frontier_authority_rejected"
    assert provider.calls == []

    settled = _run(request, provider, factory)
    assert settled.status is controller.AuxiliarySemanticVerificationStatus.SETTLED
    provider.calls.clear()
    with store._connect() as conn:
        conn.execute(
            "UPDATE insession_auxiliary_capability_catalog_items "
            "SET descriptor_json='{}' WHERE capability_catalog_snapshot_id=?",
            (request.capability_catalog.capability_catalog_snapshot_id,),
        )
    tampered = _run(request, provider, factory)
    assert tampered.status is controller.AuxiliarySemanticVerificationStatus.FAILED_CLOSED
    assert tampered.reason_code == "semantic_stored_authority_rejected"
    assert provider.calls == []


def test_pending_durable_dispatch_returns_waiting_external_without_provider() -> None:
    request = _request("semantic-controller-pending", protected=False)
    provider = _SemanticProvider()
    factory, _bindings = _authority_factory(open_pending=True)

    waiting = _run(request, provider, factory)

    assert waiting.status is controller.AuxiliarySemanticVerificationStatus.WAITING_EXTERNAL
    assert waiting.reason_code == "semantic_reviewer_model_call_waiting_external"
    assert waiting.completed_reviewer_ordinals == ()
    assert waiting.pending_reviewer_ordinal == 1
    assert waiting.settlement is None
    assert provider.calls == []


def test_two_pre_event_waiting_reviewers_advance_the_ordinal_gate() -> None:
    request = _request("semantic-controller-two-pending", protected=True)
    provider = _SemanticProvider()
    factory, _bindings = _authority_factory(open_pending=True)
    events = []

    waiting = controller.run_auxiliary_semantic_verification(
        request,
        provider=provider,
        model_call_authority_factory=factory,
        emit=events.append,
    )

    assert waiting.status is controller.AuxiliarySemanticVerificationStatus.WAITING_EXTERNAL
    assert waiting.pending_reviewer_ordinal == 1
    assert waiting.completed_reviewer_ordinals == ()
    assert events == []
    assert provider.calls == []


def test_first_event_emit_failure_releases_the_next_reviewer() -> None:
    request = _request("semantic-controller-emit-failure", protected=True)
    provider = _SemanticProvider()
    factory, _bindings = _authority_factory()
    events = []

    class _EmitFailure(RuntimeError):
        pass

    def fail_first_emit(event) -> None:
        events.append(event)
        if len(events) == 1:
            raise _EmitFailure("first reviewer event sink failed")

    with pytest.raises(_EmitFailure):
        controller.run_auxiliary_semantic_verification(
            request,
            provider=provider,
            model_call_authority_factory=factory,
            emit=fail_first_emit,
        )

    # 审查器 1 从未越过失败的 STARTED 发射；审查器 2 则被释放、调用提供方，
    # 并发射其余生命周期事件。
    assert len(provider.calls) == 1
    assert _reviewer_ordinal_for_model_call(provider.calls[0]) == 2
    assert [
        _reviewer_ordinal_for_model_call(event.model_call_id)
        for event in events
        if event.model_call_id is not None
    ] == [1, 2, 2]

    first_logical = store.get_runtime_model_logical_call(
        session_id=request.frontier.session_id,
        logical_call_id=request.reviewers[0].logical_call_id,
    )
    assert first_logical is not None
    assert len(first_logical.physical_attempts) == 1
    first_settlement = first_logical.physical_attempts[0].settlement
    assert first_settlement is not None
    assert first_settlement.outcome.value == "retryable_failure"
    assert first_settlement.error_code == "RUNTIME_EVENT_EMIT_FAILED"

    recovered = controller.run_auxiliary_semantic_verification(
        request,
        provider=provider,
        model_call_authority_factory=factory,
        emit=lambda _event: None,
    )

    assert recovered.status is (
        controller.AuxiliarySemanticVerificationStatus.SETTLED
    )
    # 审查器 1 获得一次授权重试；审查器 2 重放已经持久化的成功结果，
    # 而不是开启重复提供方 I/O。
    assert len(provider.calls) == 2
    retried_first = store.get_runtime_model_logical_call(
        session_id=request.frontier.session_id,
        logical_call_id=request.reviewers[0].logical_call_id,
    )
    assert retried_first is not None
    assert tuple(
        attempt.settlement.outcome.value
        for attempt in retried_first.physical_attempts
        if attempt.settlement is not None
    ) == ("retryable_failure", "succeeded")


def test_partial_executor_submit_failure_joins_started_reviewer_before_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request("semantic-controller-submit-failure", protected=True)
    provider = _SemanticProvider()
    factory, _bindings = _authority_factory()

    class _SubmitFailure(RuntimeError):
        pass

    class _FailSecondSubmitExecutor:
        instances: list["_FailSecondSubmitExecutor"] = []

        def __init__(self, **_kwargs) -> None:
            self.submit_count = 0
            self.threads: list[threading.Thread] = []
            self.handoffs = []
            self.worker_alive_on_exit: bool | None = None
            self.__class__.instances.append(self)

        def __enter__(self):
            return self

        def submit(self, fn, /, *args, **kwargs):
            self.submit_count += 1
            if self.submit_count == 2:
                raise _SubmitFailure("second reviewer submit failed")
            future: Future = Future()
            assert future.set_running_or_notify_cancel()
            first_event_queue = kwargs["first_event_queue"]

            def run() -> None:
                try:
                    future.set_result(fn(*args, **kwargs))
                except BaseException as exc:
                    future.set_exception(exc)

            thread = threading.Thread(target=run)
            self.threads.append(thread)
            thread.start()
    # 让部分提交竞争具有确定性：只有审查器 1 已开启持久物理尝试并阻塞在首事件
    # 门禁后，才提交审查器 2。
            handoff = first_event_queue.get(timeout=5.0)
            self.handoffs.append(handoff)
            first_event_queue.put(handoff)
            return future

        def __exit__(self, _exc_type, _exc, _traceback) -> bool:
            for thread in self.threads:
                thread.join(timeout=0.2)
            self.worker_alive_on_exit = any(
                thread.is_alive() for thread in self.threads
            )
    # 即使实现损坏，也不得永久泄漏测试线程。
            for handoff in self.handoffs:
                if not handoff.released.is_set():
                    handoff.error = _SubmitFailure("test cleanup")
                    handoff.released.set()
            for thread in self.threads:
                thread.join(timeout=5.0)
            return False

    monkeypatch.setattr(
        controller,
        "ThreadPoolExecutor",
        _FailSecondSubmitExecutor,
    )

    with pytest.raises(_SubmitFailure, match="second reviewer submit failed"):
        controller.run_auxiliary_semantic_verification(
            request,
            provider=provider,
            model_call_authority_factory=factory,
            emit=lambda _event: None,
        )

    executor = _FailSecondSubmitExecutor.instances[0]
    assert executor.worker_alive_on_exit is False
    assert provider.calls == []
    first_logical = store.get_runtime_model_logical_call(
        session_id=request.frontier.session_id,
        logical_call_id=request.reviewers[0].logical_call_id,
    )
    assert first_logical is not None
    assert len(first_logical.physical_attempts) == 1
    first_settlement = first_logical.physical_attempts[0].settlement
    assert first_settlement is not None
    assert first_settlement.outcome.value == "retryable_failure"


def test_partial_submit_does_not_wait_for_handoff_from_unstarted_future(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request("semantic-controller-unstarted-future", protected=True)
    provider = _SemanticProvider()
    factory, _bindings = _authority_factory()

    class _SubmitFailure(RuntimeError):
        pass

    class _WorkerNeverStarted(RuntimeError):
        pass

    class _FailBeforeWorkerStartsExecutor:
        instances: list["_FailBeforeWorkerStartsExecutor"] = []

        def __init__(self, **_kwargs) -> None:
            self.submit_count = 0
            self.first_event_queue = None
            self.__class__.instances.append(self)

        def __enter__(self):
            return self

        def submit(self, _fn, /, *_args, **kwargs):
            self.submit_count += 1
            if self.submit_count == 2:
                raise _SubmitFailure("second reviewer submit failed")
            self.first_event_queue = kwargs["first_event_queue"]
            future: Future = Future()
            future.set_exception(
                _WorkerNeverStarted("executor rejected worker before invocation")
            )
            return future

        def __exit__(self, _exc_type, _exc, _traceback) -> bool:
            return False

    monkeypatch.setattr(
        controller,
        "ThreadPoolExecutor",
        _FailBeforeWorkerStartsExecutor,
    )
    errors: list[BaseException] = []
    finished = threading.Event()

    def run_controller() -> None:
        try:
            controller.run_auxiliary_semantic_verification(
                request,
                provider=provider,
                model_call_authority_factory=factory,
                emit=lambda _event: None,
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    worker = threading.Thread(target=run_controller)
    worker.start()
    finished_without_rescue = finished.wait(timeout=0.5)
    if not finished_without_rescue:
    # 为已知损坏实现提供兜底，确保回归测试绝不向测试套件其余部分泄漏阻塞的
    # 非守护线程。
        executor = _FailBeforeWorkerStartsExecutor.instances[0]
        assert executor.first_event_queue is not None
        executor.first_event_queue.put(
            controller._SemanticReviewerFirstEventHandoff(event=None)
        )
    worker.join(timeout=5.0)

    assert finished_without_rescue
    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], _SubmitFailure)
    assert provider.calls == []


def test_partial_submit_releases_work_enqueued_before_submit_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request("semantic-controller-enqueued-submit-failure", protected=True)
    provider = _SemanticProvider()
    factory, _bindings = _authority_factory()

    class _SubmitFailure(RuntimeError):
        pass

    class _EnqueueThenFailExecutor:
        instances: list["_EnqueueThenFailExecutor"] = []

        def __init__(self, **_kwargs) -> None:
            self.submit_count = 0
            self.threads: list[threading.Thread] = []
            self.worker_alive_on_exit: bool | None = None
            self.__class__.instances.append(self)

        def __enter__(self):
            return self

        def submit(self, fn, /, *args, **kwargs):
            self.submit_count += 1
            future: Future = Future()
            assert future.set_running_or_notify_cancel()
            first_event_queue = kwargs["first_event_queue"]

            def run() -> None:
                try:
                    future.set_result(fn(*args, **kwargs))
                except BaseException as exc:
                    future.set_exception(exc)

            thread = threading.Thread(target=run)
            self.threads.append(thread)
            thread.start()
    # 模拟 ThreadPoolExecutor 的棘手失败模式：工作已经运行或入队，但 submit
    # 在向协调器返回 Future 前抛出异常。
            handoff = first_event_queue.get(timeout=5.0)
            first_event_queue.put(handoff)
            if self.submit_count == 2:
                raise _SubmitFailure("second reviewer was enqueued before submit failed")
            return future

        def __exit__(self, _exc_type, _exc, _traceback) -> bool:
            for thread in self.threads:
                thread.join(timeout=5.0)
            self.worker_alive_on_exit = any(
                thread.is_alive() for thread in self.threads
            )
            return False

    monkeypatch.setattr(
        controller,
        "ThreadPoolExecutor",
        _EnqueueThenFailExecutor,
    )

    with pytest.raises(_SubmitFailure, match="enqueued before submit failed"):
        controller.run_auxiliary_semantic_verification(
            request,
            provider=provider,
            model_call_authority_factory=factory,
            emit=lambda _event: None,
        )

    executor = _EnqueueThenFailExecutor.instances[0]
    assert executor.worker_alive_on_exit is False
    assert provider.calls == []
    for reviewer in request.reviewers:
        logical = store.get_runtime_model_logical_call(
            session_id=request.frontier.session_id,
            logical_call_id=reviewer.logical_call_id,
        )
        assert logical is not None
        assert len(logical.physical_attempts) == 1
        settlement = logical.physical_attempts[0].settlement
        assert settlement is not None
        assert settlement.outcome.value == "retryable_failure"
        assert settlement.error_code == "RUNTIME_EVENT_EMIT_FAILED"


def test_retryable_semantic_reviewer_reuses_origin_logical_call_on_next_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    request = _request("semantic-controller-cross-turn-retry", protected=False)
    provider = _SemanticProvider(
        failures=(
            ModelGatewayError(
                "MODEL_CALL_TIMEOUT",
                "the first Turn exhausted its wall-clock lease",
                retryable=True,
            ),
        ),
        provider="mock",
        model="mock-structured",
    )

    first = _run(
        request,
        provider,
        partial(
            create_auxiliary_semantic_reviewer_model_call_authority,
            ledger_store=store,
        ),
        deadline=_ExpireBeforeSecondPhysicalAttempt(),
    )

    assert first.status is controller.AuxiliarySemanticVerificationStatus.TURN_LIMIT_REACHED
    logical_before = store.get_runtime_model_logical_call(
        session_id=request.frontier.session_id,
        logical_call_id=request.reviewers[0].logical_call_id,
    )
    assert logical_before is not None
    assert logical_before.request.invocation_turn_id == request.frontier.turn_id
    assert len(logical_before.physical_attempts) == 1
    assert logical_before.physical_attempts[0].settlement.outcome.value == (
        "retryable_failure"
    )

    continued = _handoff_semantic_turn(
        request,
        client_request_id="semantic-controller-cross-turn-retry-followup",
    )
    settled = _run(
        continued,
        provider,
        partial(
            create_auxiliary_semantic_reviewer_model_call_authority,
            ledger_store=store,
        ),
    )

    assert settled.status is controller.AuxiliarySemanticVerificationStatus.SETTLED, (
        settled.reason_code
    )
    logical_after = store.get_runtime_model_logical_call(
        session_id=request.frontier.session_id,
        logical_call_id=request.reviewers[0].logical_call_id,
    )
    assert logical_after is not None
    assert logical_after.request == logical_before.request
    assert tuple(
        item.request.started_turn_id for item in logical_after.physical_attempts
    ) == (request.frontier.turn_id, continued.frontier.turn_id)
    assert tuple(
        item.settlement.outcome.value for item in logical_after.physical_attempts
    ) == ("retryable_failure", "succeeded")
    assert len(provider.calls) == 2


def test_two_reviewer_quorum_resumes_only_pending_reviewer_on_next_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    request = _request("semantic-controller-cross-turn-quorum", protected=True)
    passing = _SemanticProvider(provider="mock", model="mock-structured")

    class _SecondReviewerRetryable:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.failed = False
            self.lock = threading.Lock()

        def __call__(self, *args, model_call_id: str, **kwargs):
            reviewer_ordinal = _reviewer_ordinal_for_model_call(model_call_id)
            with self.lock:
                should_fail = reviewer_ordinal == 2 and not self.failed
                if should_fail:
                    self.failed = True
                    self.calls.append(model_call_id)
            if should_fail:
                raise ModelGatewayError(
                    "MODEL_CALL_TIMEOUT",
                    "second reviewer crossed the Turn lease",
                    retryable=True,
                )
            result = passing(*args, model_call_id=model_call_id, **kwargs)
            with self.lock:
                self.calls.append(model_call_id)
            return result

    provider = _SecondReviewerRetryable()

    class _ExpireAfterSecondReviewerFailure:
        def expired(self) -> bool:
            with provider.lock:
                return len(provider.calls) >= 2

        def remaining_s(self) -> float:
            with provider.lock:
                return 0.0 if len(provider.calls) >= 2 else 30.0

    first = _run(
        request,
        provider,
        partial(
            create_auxiliary_semantic_reviewer_model_call_authority,
            ledger_store=store,
        ),
        deadline=_ExpireAfterSecondReviewerFailure(),
    )

    assert first.status is controller.AuxiliarySemanticVerificationStatus.TURN_LIMIT_REACHED
    assert first.completed_reviewer_ordinals == (1,)
    first_result = semantic_store.get_auxiliary_semantic_verification_result(
        session_id=request.frontier.session_id,
        verification_result_id=request.reviewers[0].verification_result_id,
    )
    assert first_result is not None
    second_before = store.get_runtime_model_logical_call(
        session_id=request.frontier.session_id,
        logical_call_id=request.reviewers[1].logical_call_id,
    )
    assert second_before is not None
    assert len(second_before.physical_attempts) == 1

    continued = _handoff_semantic_turn(
        request,
        client_request_id="semantic-controller-cross-turn-quorum-followup",
    )
    settled = _run(
        continued,
        provider,
        partial(
            create_auxiliary_semantic_reviewer_model_call_authority,
            ledger_store=store,
        ),
    )

    assert settled.status is controller.AuxiliarySemanticVerificationStatus.SETTLED, (
        settled.reason_code
    )
    assert settled.completed_reviewer_ordinals == (1, 2)
    assert settled.settlement is not None
    second_after = store.get_runtime_model_logical_call(
        session_id=request.frontier.session_id,
        logical_call_id=request.reviewers[1].logical_call_id,
    )
    assert second_after is not None
    assert second_after.request == second_before.request
    assert tuple(
        item.request.started_turn_id for item in second_after.physical_attempts
    ) == (request.frontier.turn_id, continued.frontier.turn_id)
    assert len(provider.calls) == 3


def test_earlier_reviewer_failure_buffers_later_success_for_exact_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    request = _request("semantic-controller-ordinal-prefix", protected=True)
    passing = _SemanticProvider(provider="mock", model="mock-structured")

    class _FirstReviewerRetryable:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.failed = False
            self.lock = threading.Lock()
            self.first_dispatch_barrier = threading.Barrier(2)

        def __call__(self, *args, model_call_id: str, **kwargs):
            reviewer_ordinal = _reviewer_ordinal_for_model_call(model_call_id)
            with self.lock:
                first_dispatch = not self.failed
            if first_dispatch:
                self.first_dispatch_barrier.wait(timeout=5.0)
            with self.lock:
                should_fail = reviewer_ordinal == 1 and not self.failed
                if should_fail:
                    self.failed = True
                    self.calls.append(model_call_id)
            if should_fail:
                raise ModelGatewayError(
                    "MODEL_CALL_TIMEOUT",
                    "first reviewer crossed the Turn lease",
                    retryable=True,
                )
            result = passing(*args, model_call_id=model_call_id, **kwargs)
            with self.lock:
                self.calls.append(model_call_id)
            return result

    provider = _FirstReviewerRetryable()

    class _ExpireAfterFirstReviewerFailure:
        def expired(self) -> bool:
            with provider.lock:
                return provider.failed

        def remaining_s(self) -> float:
            with provider.lock:
                return 0.0 if provider.failed else 30.0

    first = _run(
        request,
        provider,
        partial(
            create_auxiliary_semantic_reviewer_model_call_authority,
            ledger_store=store,
        ),
        deadline=_ExpireAfterFirstReviewerFailure(),
    )

    assert first.status is controller.AuxiliarySemanticVerificationStatus.TURN_LIMIT_REACHED
    assert first.completed_reviewer_ordinals == ()
    assert semantic_store.get_auxiliary_semantic_verification_result(
        session_id=request.frontier.session_id,
        verification_result_id=request.reviewers[1].verification_result_id,
    ) is None
    second_logical = store.get_runtime_model_logical_call(
        session_id=request.frontier.session_id,
        logical_call_id=request.reviewers[1].logical_call_id,
    )
    assert second_logical is not None
    assert second_logical.physical_attempts[-1].settlement is not None
    assert second_logical.physical_attempts[-1].settlement.outcome.value == "succeeded"

    continued = _handoff_semantic_turn(
        request,
        client_request_id="semantic-controller-ordinal-prefix-followup",
    )
    settled = _run(
        continued,
        provider,
        partial(
            create_auxiliary_semantic_reviewer_model_call_authority,
            ledger_store=store,
        ),
    )

    assert settled.status is controller.AuxiliarySemanticVerificationStatus.SETTLED
    assert settled.completed_reviewer_ordinals == (1, 2)
    # 两次首轮提供方调用，加上审查器 1 的授权重试。已经持久化的审查器 2
    # 成功结果会在不执行提供方 I/O 的情况下重新验证。
    assert len(provider.calls) == 3
    assert sum(
        _reviewer_ordinal_for_model_call(model_call_id) == 2
        for model_call_id in provider.calls
    ) == 1


@pytest.mark.parametrize("settled_uncertain", (False, True))
def test_pending_or_uncertain_semantic_reviewer_waits_after_turn_handoff(
    monkeypatch: pytest.MonkeyPatch,
    settled_uncertain: bool,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    request = _request(
        f"semantic-controller-cross-turn-external-{settled_uncertain}",
        protected=False,
    )

    class _SimulatedProcessLoss(BaseException):
        pass

    provider = _SemanticProvider(
        failures=(_SimulatedProcessLoss("provider result was not observed"),),
        provider="mock",
        model="mock-structured",
    )
    captured = []

    def capture_authority(binding, **kwargs):
        authority = create_auxiliary_semantic_reviewer_model_call_authority(
            binding,
            ledger_store=store,
            **kwargs,
        )
        captured.append(authority)
        return authority

    with pytest.raises(_SimulatedProcessLoss):
        _run(request, provider, capture_authority)
    assert len(captured) == 1
    if settled_uncertain:
        captured[0].settle_pending_external(
            turn_id=request.frontier.turn_id,
            outcome="uncertain",
            result_fingerprint="c" * 64,
            provider_request_id="semantic-provider-request-unresolved",
            error_code="provider_result_uncertain",
        )
    assert captured[0].inspect_recovery().disposition == (
        "waiting_uncertain" if settled_uncertain else "waiting_pending"
    )

    continued = _handoff_semantic_turn(
        request,
        client_request_id=(
            "semantic-controller-cross-turn-uncertain-followup"
            if settled_uncertain
            else "semantic-controller-cross-turn-pending-followup"
        ),
    )
    no_provider = _SemanticProvider(provider="mock", model="mock-structured")
    waiting = _run(
        continued,
        no_provider,
        partial(
            create_auxiliary_semantic_reviewer_model_call_authority,
            ledger_store=store,
        ),
    )

    assert waiting.status is controller.AuxiliarySemanticVerificationStatus.WAITING_EXTERNAL, (
        waiting.reason_code
    )
    assert waiting.pending_reviewer_ordinal == 1
    logical = store.get_runtime_model_logical_call(
        session_id=request.frontier.session_id,
        logical_call_id=request.reviewers[0].logical_call_id,
    )
    assert logical is not None
    assert logical.request.invocation_turn_id == request.frontier.turn_id
    assert len(logical.physical_attempts) == 1
    assert no_provider.calls == []


def test_durable_semantic_success_replays_after_precommit_loss_on_next_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    request = _request("semantic-controller-cross-turn-loss", protected=False)
    provider = _SemanticProvider(provider="mock", model="mock-structured")
    original = semantic_store.commit_auxiliary_semantic_verification_result

    def lose_before_semantic_commit(*, command):
        raise semantic_store.AuxiliarySemanticVerificationPersistenceError(
            "simulated process loss after durable Provider settlement"
        )

    monkeypatch.setattr(
        semantic_store,
        "commit_auxiliary_semantic_verification_result",
        lose_before_semantic_commit,
    )
    lost = _run(
        request,
        provider,
        partial(
            create_auxiliary_semantic_reviewer_model_call_authority,
            ledger_store=store,
        ),
    )
    assert lost.status is controller.AuxiliarySemanticVerificationStatus.FAILED_CLOSED
    assert len(provider.calls) == 1

    monkeypatch.setattr(
        semantic_store,
        "commit_auxiliary_semantic_verification_result",
        original,
    )
    continued = _handoff_semantic_turn(
        request,
        client_request_id="semantic-controller-cross-turn-loss-followup",
    )
    no_provider = _SemanticProvider(provider="mock", model="mock-structured")
    recovered = _run(
        continued,
        no_provider,
        partial(
            create_auxiliary_semantic_reviewer_model_call_authority,
            ledger_store=store,
        ),
    )

    assert recovered.status is controller.AuxiliarySemanticVerificationStatus.SETTLED, (
        recovered.reason_code
    )
    assert recovered.settlement is not None
    assert no_provider.calls == []
    logical = store.get_runtime_model_logical_call(
        session_id=request.frontier.session_id,
        logical_call_id=request.reviewers[0].logical_call_id,
    )
    assert logical is not None
    assert logical.request.invocation_turn_id == request.frontier.turn_id
    assert len(logical.physical_attempts) == 1


def test_retryable_terminal_candidate_semantic_call_continues_on_next_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    session_id, first_turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    semantic_provider = _SemanticProvider(
        failures=(
            ModelGatewayError(
                "MODEL_CALL_TIMEOUT",
                "candidate semantic retry crossed the Turn lease",
                retryable=True,
            ),
        ),
        provider="mock",
        model="mock-structured",
    )

    class _ExpireAfterFirstSemanticPhysicalAttempt:
        def expired(self) -> bool:
            return bool(semantic_provider.calls)

        def remaining_s(self) -> float:
            return 0.0 if semantic_provider.calls else 30.0

    first = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=first_turn_id,
            task_id=task_id,
            max_effect_steps=8,
            deadline=_ExpireAfterFirstSemanticPhysicalAttempt(),
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            semantic_provider=semantic_provider,
            semantic_model_call_authority_factory=(
                partial(create_auxiliary_semantic_reviewer_model_call_authority, ledger_store=store)
            ),
        ),
    )

    assert first.status is AuxiliaryApplicationStatus.TURN_LIMIT_REACHED
    with store._connect() as conn:
        row = conn.execute(
            "SELECT logical_call_id FROM insession_runtime_model_logical_calls "
            "WHERE session_id=? AND call_kind='task_graph_semantic_verification'",
            (session_id,),
        ).fetchone()
    assert row is not None
    logical_call_id = str(row["logical_call_id"])
    logical_before = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=logical_call_id,
    )
    assert logical_before is not None
    assert logical_before.request.invocation_turn_id == first_turn_id
    assert len(logical_before.physical_attempts) == 1
    assert logical_before.physical_attempts[0].settlement.outcome.value == (
        "retryable_failure"
    )

    second_turn_id = _handoff_execution_turn(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        client_request_id="candidate-semantic-retry-followup",
    )
    completed = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=second_turn_id,
            task_id=task_id,
            max_effect_steps=8,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            semantic_provider=semantic_provider,
            semantic_model_call_authority_factory=(
                partial(create_auxiliary_semantic_reviewer_model_call_authority, ledger_store=store)
            ),
        ),
    )

    assert completed.status is AuxiliaryApplicationStatus.COMMITTED, (
        completed.reason_code
    )
    logical_after = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=logical_call_id,
    )
    assert logical_after is not None
    assert logical_after.request == logical_before.request
    assert tuple(
        item.request.started_turn_id for item in logical_after.physical_attempts
    ) == (first_turn_id, second_turn_id)
    assert tuple(
        item.settlement.outcome.value for item in logical_after.physical_attempts
    ) == ("retryable_failure", "succeeded")
    assert len(semantic_provider.calls) == 2


@pytest.mark.parametrize("settled_uncertain", (False, True))
def test_pending_or_uncertain_terminal_candidate_waits_after_turn_handoff(
    monkeypatch: pytest.MonkeyPatch,
    settled_uncertain: bool,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    session_id, first_turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)

    class _SimulatedCandidateProcessLoss(BaseException):
        pass

    semantic_provider = _SemanticProvider(
        failures=(
            _SimulatedCandidateProcessLoss(
                "candidate semantic Provider result was not observed"
            ),
        ),
        provider="mock",
        model="mock-structured",
    )
    captured = []

    def capture_authority(binding, **kwargs):
        authority = create_auxiliary_semantic_reviewer_model_call_authority(
            binding,
            ledger_store=store,
            **kwargs,
        )
        captured.append(authority)
        return authority

    with pytest.raises(_SimulatedCandidateProcessLoss):
        run_auxiliary_application_to_boundary(
            AuxiliaryApplicationRequest(
                session_id=session_id,
                turn_id=first_turn_id,
                task_id=task_id,
                max_effect_steps=8,
            ),
            ports=AuxiliaryApplicationPorts(
                model_ledger_store=store,
                emit=lambda _event: None,
                semantic_provider=semantic_provider,
                semantic_model_call_authority_factory=capture_authority,
            ),
        )
    assert len(captured) == 1
    if settled_uncertain:
        captured[0].settle_pending_external(
            turn_id=first_turn_id,
            outcome="uncertain",
            result_fingerprint="d" * 64,
            provider_request_id="candidate-semantic-request-unresolved",
            error_code="provider_result_uncertain",
        )
    assert captured[0].inspect_recovery().disposition == (
        "waiting_uncertain" if settled_uncertain else "waiting_pending"
    )

    second_turn_id = _handoff_execution_turn(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        client_request_id=(
            "candidate-semantic-uncertain-followup"
            if settled_uncertain
            else "candidate-semantic-pending-followup"
        ),
    )
    no_provider = _SemanticProvider(provider="mock", model="mock-structured")
    waiting = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=second_turn_id,
            task_id=task_id,
            max_effect_steps=8,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            semantic_provider=no_provider,
            semantic_model_call_authority_factory=(
                partial(create_auxiliary_semantic_reviewer_model_call_authority, ledger_store=store)
            ),
        ),
    )

    assert waiting.status is AuxiliaryApplicationStatus.WAITING_EXTERNAL, (
        waiting.reason_code
    )
    assert waiting.reason_code == "work_run_model_call_waiting_external"
    logical = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=captured[0].semantic_call_id,
    )
    assert logical is not None
    assert logical.request.invocation_turn_id == first_turn_id
    assert len(logical.physical_attempts) == 1
    assert no_provider.calls == []


def test_terminal_candidate_semantic_success_replays_after_node_precommit_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PERSONAGRAPH_MODEL_PROVIDER", "mock")
    session_id, first_turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    semantic_provider = _SemanticProvider(provider="mock", model="mock-structured")
    original_commit = verification_store.commit_auxiliary_node_verification_result

    class _SimulatedNodePrecommitLoss(BaseException):
        pass

    def lose_after_semantic_success(**kwargs):
        if semantic_provider.calls:
            raise _SimulatedNodePrecommitLoss(
                "semantic success was durable before node verification commit"
            )
        return original_commit(**kwargs)

    monkeypatch.setattr(
        verification_store,
        "commit_auxiliary_node_verification_result",
        lose_after_semantic_success,
    )
    with pytest.raises(_SimulatedNodePrecommitLoss):
        run_auxiliary_application_to_boundary(
            AuxiliaryApplicationRequest(
                session_id=session_id,
                turn_id=first_turn_id,
                task_id=task_id,
                max_effect_steps=8,
            ),
            ports=AuxiliaryApplicationPorts(
                model_ledger_store=store,
                emit=lambda _event: None,
                semantic_provider=semantic_provider,
                semantic_model_call_authority_factory=(
                    partial(create_auxiliary_semantic_reviewer_model_call_authority, ledger_store=store)
                ),
            ),
        )
    assert len(semantic_provider.calls) == 1
    with store._connect() as conn:
        row = conn.execute(
            "SELECT logical_call_id FROM insession_runtime_model_logical_calls "
            "WHERE session_id=? AND call_kind='task_graph_semantic_verification'",
            (session_id,),
        ).fetchone()
    assert row is not None
    logical_call_id = str(row["logical_call_id"])
    logical_before = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=logical_call_id,
    )
    assert logical_before is not None
    assert len(logical_before.physical_attempts) == 1
    assert logical_before.physical_attempts[0].settlement.outcome.value == (
        "succeeded"
    )

    monkeypatch.setattr(
        verification_store,
        "commit_auxiliary_node_verification_result",
        original_commit,
    )
    second_turn_id = _handoff_execution_turn(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
        client_request_id="candidate-semantic-precommit-loss-followup",
    )
    no_provider = _SemanticProvider(provider="mock", model="mock-structured")
    completed = run_auxiliary_application_to_boundary(
        AuxiliaryApplicationRequest(
            session_id=session_id,
            turn_id=second_turn_id,
            task_id=task_id,
            max_effect_steps=8,
        ),
        ports=AuxiliaryApplicationPorts(
            model_ledger_store=store,
            emit=lambda _event: None,
            semantic_provider=no_provider,
            semantic_model_call_authority_factory=(
                partial(create_auxiliary_semantic_reviewer_model_call_authority, ledger_store=store)
            ),
        ),
    )

    assert completed.status is AuxiliaryApplicationStatus.COMMITTED, (
        completed.reason_code
    )
    assert no_provider.calls == []
    logical_after = store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=logical_call_id,
    )
    assert logical_after is not None
    assert logical_after.request == logical_before.request
    assert len(logical_after.physical_attempts) == 1
