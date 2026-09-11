from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from personagraph.l2.auxiliary_graph import (
    AuxiliaryGraphRevisionReason,
    PlanningAuthorityAnchor,
    PlanningAuthorityClass,
    PlanningAuthorityOriginKind,
    PlanningAuthorityProjection,
    PlanningAuthoritySnapshot,
    PlanningAuthoritySourceCard,
    PlanningAuthoritySourceKind,
)
from personagraph.model_io.gateway import ModelResult
from personagraph.l2.auxiliary_execution.planning import (
    replanning_controller as replanning,
)
from personagraph.l2.auxiliary_execution.planning.replanning_controller import (
    AuxiliaryReplanningError,
    AuxiliaryReplanningRequest,
    AuxiliaryReplanningStatus,
    run_auxiliary_replanning,
)
from personagraph.l2.auxiliary_execution.planning.model_provider import (
    build_mock_auxiliary_architect_proposal,
)
from personagraph.l2.auxiliary_execution.planning.profiles import (
    MODEL_ANALYSIS_CAPABILITY,
    MOUNTED_DOCUMENT_READ_CAPABILITY,
)
from personagraph.l2.auxiliary_execution.planning.mounted_document_authority import (
    freeze_mounted_document_planning_authority,  # noqa: F401
)
from personagraph.session import store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import planning as planning_store
from tests.session.test_auxiliary_replan_trigger_persistence import (
    _settled_non_pass,
)


class _ReplanningArchitectProvider:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        _system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        assert purpose == "runtime_auxiliary_graph_architect_v2"
        payload = json.loads(user_content)
        self.calls.append(payload)
        trigger = payload["replan_trigger"]
        current = payload["current_revision"]
        assert trigger is not None
        assert trigger["semantic_host_disposition"] == "revise"
        assert trigger["expected_current_auxiliary_graph_revision"] == current[
            "auxiliary_graph_revision"
        ]
        assert trigger["reviewers"][0]["items"]

        structure = deepcopy(current["structure"])
        for node in structure["nodes"]:
            node["origin_node_alias"] = node["node_key"]
        reply = {
            "schema_version": "auxiliary-graph-revision-proposal-v2",
            "disposition": "revise_revision",
            "expected_current_auxiliary_graph_revision": current[
                "auxiliary_graph_revision"
            ],
            "revision_reason": "verification_failed",
            "structure": structure,
            "explanation": (
                "Revise the exact current graph after the bound semantic failure."
            ),
            "blocking_gap_ids": [],
            "requested_user_question": None,
            "failure_reason": None,
        }
        return ModelResult(
            reply=json.dumps(reply, ensure_ascii=False),
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )


def _request(session_id: str, turn_id: str, task_id: str, settlement, *, maximum=3):
    return AuxiliaryReplanningRequest(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        settlement=settlement,
        max_autonomous_replans=maximum,
    )




def test_non_pass_settlement_replans_exact_n_to_n_plus_one_and_replays() -> None:
    session_id, turn_id, task_id, before, settlement = _settled_non_pass(
        "replan-controller-happy"
    )
    provider = _ReplanningArchitectProvider()
    request = _request(session_id, turn_id, task_id, settlement)

    replanned = run_auxiliary_replanning(
        request,
        provider=provider,
        ledger_store=store,
        emit=lambda _event: None,
    )

    assert replanned.status is AuxiliaryReplanningStatus.REPLANNED
    assert replanned.reason_code == "semantic_replan_revision_committed"
    assert replanned.trigger is not None
    assert replanned.architect_decision is not None
    assert replanned.revision_commit is not None
    assert replanned.application is not None
    assert replanned.model_call_id is not None
    assert store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=replanned.architect_decision.logical_call_id,
    ) is not None
    assert replanned.model_attempts == 1
    assert replanned.model_replayed is False
    assert replanned.trigger_replayed is False
    assert replanned.revision_commit.status == "applied"
    assert replanned.application_replayed is False
    assert len(provider.calls) == 1

    after = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert after is not None
    assert after.auxiliary_graph_revision == before.auxiliary_graph_revision + 1
    assert after.parent_auxiliary_graph_revision == before.auxiliary_graph_revision
    assert after.reason == AuxiliaryGraphRevisionReason.VERIFICATION_FAILED.value
    assert after.goal_id == before.goal_id
    assert after.budget is not None and before.budget is not None
    assert after.budget.budget_ledger_id == before.budget.budget_ledger_id
    assert replanned.application.applied_auxiliary_graph_revision == (
        after.auxiliary_graph_revision
    )
    assert planning_store.get_active_auxiliary_replan_trigger(
        session_id=session_id,
        task_id=task_id,
    ) is None
    assert planning_store.count_auxiliary_replan_triggers(
        session_id=session_id,
        task_id=task_id,
        goal_id=before.goal_id,
    ) == 1

    replayed = run_auxiliary_replanning(
        request,
        provider=lambda *_args, **_kwargs: pytest.fail(
            "an applied replan reached the Architect Provider"
        ),
        ledger_store=store,
        emit=lambda _event: None,
    )

    assert replayed.status is AuxiliaryReplanningStatus.ALREADY_REPLANNED
    assert replayed.reason_code == "semantic_replan_already_applied"
    assert replayed.trigger == replanned.trigger
    assert replayed.application == replanned.application
    assert replayed.architect_decision is None
    assert replayed.revision_commit is None
    assert replayed.model_call_id is None
    assert len(provider.calls) == 1




def test_replan_architect_preserves_current_resource_authority_and_adds_review_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """语义证据投影不得抹除当前图别名。"""

    session_id, _turn_id, task_id, details, settlement = _settled_non_pass(
        "replan-controller-authority-union"
    )
    assert details.authority_snapshot is not None
    authorization_anchor = next(
        item
        for item in details.authority_snapshot.anchors
        if item.authority_class is PlanningAuthorityClass.AUTHORIZATION
    )
    mounted_anchor = PlanningAuthorityAnchor(
        anchor_id="authority_anchor_mounted_document_01",
        authority_snapshot_id=details.authority_snapshot_id,
        projection_alias="mounted_document_01",
        authority_class=PlanningAuthorityClass.EVIDENCE,
        origin_kind=PlanningAuthorityOriginKind.WORKSPACE_RESOURCE,
        origin_id="document-version-01",
        source_revision=1,
        content_sha256="a" * 64,
        item_ordinal=1,
        projection_sha256="b" * 64,
        freshness_binding_sha256="c" * 64,
    )
    authority_snapshot = PlanningAuthoritySnapshot.create(
        authority_snapshot_id=details.authority_snapshot_id,
        session_id=details.session_id,
        task_id=details.task_id,
        auxiliary_graph_id=details.auxiliary_graph_id,
        goal_id=details.goal_id,
        source_turn_id=details.source_turn_id,
        anchors=tuple(
            sorted(
                (authorization_anchor, mounted_anchor),
                key=lambda item: item.projection_alias,
            )
        ),
    )
    authorization_card = next(
        item
        for item in settlement.requests[0].prompt_payload.authority.cards
        if item.authority_class is PlanningAuthorityClass.AUTHORIZATION
    )
    initial_mounted_card = PlanningAuthoritySourceCard(
        alias="mounted_document_01",
        authority_class=PlanningAuthorityClass.EVIDENCE,
        source_kind=PlanningAuthoritySourceKind.DOCUMENT,
        source_label="Mounted controlled evidence pack",
        excerpt="One mounted document is available for bounded reading.",
        projection_sha256=mounted_anchor.projection_sha256,
    )
    initial_authority = PlanningAuthorityProjection.create(
        authority_snapshot_id="auxauth_bootstrap_authority_union",
        authority_snapshot_sha256="e" * 64,
        cards=tuple(
            sorted(
                (
                    authorization_card,
                    initial_mounted_card,
                ),
                key=lambda item: item.alias,
            )
        ),
    )
    observation_cards = tuple(
        PlanningAuthoritySourceCard(
            alias=f"document_01_obs_{ordinal:03d}",
            authority_class=PlanningAuthorityClass.EVIDENCE,
            source_kind=PlanningAuthoritySourceKind.DOCUMENT,
            source_label=f"Verified document observation {ordinal}",
            excerpt=f"Verified bounded observation {ordinal}.",
            projection_sha256=f"{ordinal:064x}",
        )
        for ordinal in range(1, 21)
    )
    review_authority = PlanningAuthorityProjection.create(
        authority_snapshot_id=authority_snapshot.authority_snapshot_id,
        authority_snapshot_sha256=authority_snapshot.snapshot_sha256,
        cards=tuple(
            sorted(
                (
                    authorization_card,
                    initial_mounted_card,
                    *observation_cards,
                ),
                key=lambda item: item.alias,
            )
        ),
    )
    nodes = tuple(
        node.model_copy(
            update={
                "input_resource_aliases": ("mounted_document_01",),
                "source_anchor_ids": tuple(
                    sorted(
                        {
                            *node.source_anchor_ids,
                            "mounted_document_01",
                        }
                    )
                )
            }
        )
        for node in details.nodes
    )
    current = details.model_copy(
        update={
            "authority_snapshot": authority_snapshot,
            "authority_snapshot_sha256": authority_snapshot.snapshot_sha256,
            "nodes": nodes,
        }
    )
    prompt_payload = settlement.requests[0].prompt_payload.model_copy(
        update={"authority": review_authority}
    )
    semantic_request = settlement.requests[0].model_copy(
        update={"prompt_payload": prompt_payload}
    )
    semantic_settlement = settlement.model_copy(
        update={"requests": (semantic_request,)}
    )
    initial_completion = SimpleNamespace(
        binding=SimpleNamespace(
            session_id=session_id,
            task_id=task_id,
            auxiliary_graph_id=details.auxiliary_graph_id,
            goal_id=details.goal_id,
            architect_request=SimpleNamespace(
                prompt_payload=SimpleNamespace(authority=initial_authority)
            ),
        ),
        committed_auxiliary_graph_revision=details.auxiliary_graph_revision,
        committed_authority_snapshot_id=authority_snapshot.authority_snapshot_id,
        committed_authority_snapshot_sha256=authority_snapshot.snapshot_sha256,
    )
    monkeypatch.setattr(
        replanning.planning_store,
        "get_auxiliary_initial_planning_completion",
        lambda **_kwargs: initial_completion,
    )
    replan_trigger = replanning._build_architect_trigger(
        details=details,
        settlement=settlement,
    )

    request = replanning._build_architect_request(
        details=current,
        settlement=semantic_settlement,
        replan_trigger=replan_trigger,
    )

    expected_aliases = tuple(
        sorted(
            {
                *(item.alias for item in observation_cards),
                "mounted_document_01",
                authorization_card.alias,
            }
        )
    )
    assert tuple(
        item.alias for item in request.prompt_payload.authority.cards
    ) == expected_aliases

    # 持久化夹具早于生产规划能力目录。提供实时文档规划请求所携带的两个相同
    # 可用能力，使该断言验证资源分类，而不是因夹具的旧版目录失败。
    mock_payload = json.loads(request.prompt_payload.model_dump_json())
    mock_payload["capabilities"]["capabilities"] = [
        {
            "capability_alias": MODEL_ANALYSIS_CAPABILITY,
            "available": True,
        },
        {
            "capability_alias": MOUNTED_DOCUMENT_READ_CAPABILITY,
            "available": True,
        },
    ]
    mock_proposal = build_mock_auxiliary_architect_proposal(
        json.dumps(mock_payload, ensure_ascii=False)
    )
    mock_nodes = mock_proposal["structure"]["nodes"]
    mock_resource_aliases = {
        alias
        for node in mock_nodes
        for alias in node["input_resource_aliases"]
    }
    assert mock_resource_aliases == {"mounted_document_01"}
    assert not mock_resource_aliases.intersection(
        item.alias for item in observation_cards
    )

    next_snapshot_id = "auxauth_replanned_authority_union"
    next_snapshot = PlanningAuthoritySnapshot.create(
        authority_snapshot_id=next_snapshot_id,
        session_id=current.session_id,
        task_id=current.task_id,
        auxiliary_graph_id=current.auxiliary_graph_id,
        goal_id=current.goal_id,
        source_turn_id=current.source_turn_id,
        anchors=tuple(
            item.model_copy(
                update={"authority_snapshot_id": next_snapshot_id}
            )
            for item in authority_snapshot.anchors
        ),
    )
    next_current = current.model_copy(
        update={
            "auxiliary_graph_revision": current.auxiliary_graph_revision + 1,
            "authority_snapshot_id": next_snapshot.authority_snapshot_id,
            "authority_snapshot_sha256": next_snapshot.snapshot_sha256,
            "authority_snapshot": next_snapshot,
        }
    )
    next_review_authority = PlanningAuthorityProjection.create(
        authority_snapshot_id=next_snapshot.authority_snapshot_id,
        authority_snapshot_sha256=next_snapshot.snapshot_sha256,
        cards=review_authority.cards,
    )

    rebound = replanning._build_replan_architect_authority(
        details=next_current,
        semantic_authority=next_review_authority,
    )

    assert rebound.authority_snapshot_id == next_snapshot.authority_snapshot_id
    assert tuple(item.alias for item in rebound.cards) == tuple(
        item.alias for item in request.prompt_payload.authority.cards
    )

    mounted_card = next(
        item for item in initial_authority.cards if item.alias == "mounted_document_01"
    )
    colliding_authority = PlanningAuthorityProjection.create(
        authority_snapshot_id=authority_snapshot.authority_snapshot_id,
        authority_snapshot_sha256=authority_snapshot.snapshot_sha256,
        cards=tuple(
            sorted(
                (
                    authorization_card,
                    mounted_card.model_copy(
                        update={"excerpt": "Conflicting semantic projection."}
                    ),
                ),
                key=lambda item: item.alias,
            )
        ),
    )
    with pytest.raises(
        AuxiliaryReplanningError,
        match="collides with planning authority",
    ):
        replanning._build_replan_architect_authority(
            details=current,
            semantic_authority=colliding_authority,
        )


def test_model_result_replays_after_failure_before_revision_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id, before, settlement = _settled_non_pass(
        "replan-controller-model-replay"
    )
    provider = _ReplanningArchitectProvider()
    request = _request(session_id, turn_id, task_id, settlement, maximum=1)
    real_commit = auxiliary_graph_store.commit_auxiliary_graph_revision
    fail_once = True

    def fail_before_commit(**kwargs):
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise RuntimeError("simulated response loss before revision commit")
        return real_commit(**kwargs)

    monkeypatch.setattr(
        auxiliary_graph_store,
        "commit_auxiliary_graph_revision",
        fail_before_commit,
    )
    with pytest.raises(RuntimeError, match="response loss"):
        run_auxiliary_replanning(
            request,
            provider=provider,
            ledger_store=store,
            emit=lambda _event: None,
        )

    assert len(provider.calls) == 1
    current = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert current is not None
    assert current.auxiliary_graph_revision == before.auxiliary_graph_revision
    assert planning_store.count_auxiliary_replan_triggers(
        session_id=session_id,
        task_id=task_id,
        goal_id=before.goal_id,
    ) == 1

    recovered = run_auxiliary_replanning(
        request,
        provider=provider,
        ledger_store=store,
        emit=lambda _event: None,
    )

    assert recovered.status is AuxiliaryReplanningStatus.REPLANNED
    assert recovered.model_replayed is True
    assert recovered.model_attempts == 1
    assert recovered.trigger_replayed is True
    assert len(provider.calls) == 1


def test_revision_commit_response_loss_consumes_active_trigger_without_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id, before, settlement = _settled_non_pass(
        "replan-controller-commit-loss"
    )
    provider = _ReplanningArchitectProvider()
    request = _request(session_id, turn_id, task_id, settlement, maximum=1)
    real_commit = auxiliary_graph_store.commit_auxiliary_graph_revision
    lose_once = True

    def commit_then_lose(**kwargs):
        nonlocal lose_once
        committed = real_commit(**kwargs)
        if lose_once:
            lose_once = False
            raise RuntimeError("simulated response loss after revision commit")
        return committed

    monkeypatch.setattr(
        auxiliary_graph_store,
        "commit_auxiliary_graph_revision",
        commit_then_lose,
    )
    with pytest.raises(RuntimeError, match="response loss"):
        run_auxiliary_replanning(
            request,
            provider=provider,
            ledger_store=store,
            emit=lambda _event: None,
        )

    current = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert current is not None
    assert current.auxiliary_graph_revision == before.auxiliary_graph_revision + 1
    assert planning_store.get_active_auxiliary_replan_trigger(
        session_id=session_id,
        task_id=task_id,
    ) is not None
    assert len(provider.calls) == 1

    recovered = run_auxiliary_replanning(
        request,
        provider=lambda *_args, **_kwargs: pytest.fail(
            "post-commit recovery reached the Architect Provider"
        ),
        ledger_store=store,
        emit=lambda _event: None,
    )

    assert recovered.status is AuxiliaryReplanningStatus.ALREADY_REPLANNED
    assert recovered.reason_code == "semantic_replan_revision_recovered"
    assert recovered.application is not None
    assert recovered.trigger_replayed is True
    assert recovered.model_call_id is None
    assert planning_store.get_active_auxiliary_replan_trigger(
        session_id=session_id,
        task_id=task_id,
    ) is None
    assert len(provider.calls) == 1


def test_replan_limit_stops_before_trigger_or_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id, details, settlement = _settled_non_pass(
        "replan-controller-limit"
    )
    calls = 0

    def provider(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("replan limit reached the Provider")

    monkeypatch.setattr(
        planning_store,
        "count_auxiliary_replan_triggers",
        lambda **_kwargs: 1,
    )
    limited = run_auxiliary_replanning(
        _request(session_id, turn_id, task_id, settlement, maximum=1),
        provider=provider,
        ledger_store=store,
        emit=lambda _event: None,
    )

    assert limited.status is AuxiliaryReplanningStatus.LIMIT_REACHED
    assert limited.reason_code == "autonomous_replan_limit_reached"
    assert limited.trigger is None
    assert limited.architect_decision is None
    assert limited.application is None
    assert calls == 0
    assert planning_store.get_active_auxiliary_replan_trigger(
        session_id=session_id,
        task_id=task_id,
    ) is None
    current = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert current == details


def test_cross_task_settlement_fails_closed_before_provider() -> None:
    session_id, turn_id, task_id, _details, _settlement = _settled_non_pass(
        "replan-controller-owner"
    )
    other_settlement = _settlement.model_copy(
        update={"task_id": "foreign-task"}
    )
    provider_calls = 0

    def provider(*_args, **_kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("cross-task settlement reached the Provider")

    with pytest.raises(AuxiliaryReplanningError, match="settlement"):
        run_auxiliary_replanning(
            _request(session_id, turn_id, task_id, other_settlement),
            provider=provider,
            ledger_store=store,
            emit=lambda _event: None,
        )

    assert provider_calls == 0
