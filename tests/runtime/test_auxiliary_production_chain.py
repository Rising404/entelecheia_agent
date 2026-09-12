from __future__ import annotations

from dataclasses import replace
import json

import pytest

from tests.helpers.auxiliary_project import (
    auxiliary_project_authority,  # noqa: F401
    authorized_auxiliary_documents,
)

from tests.documents._authority import ingest_registered_document_fixture
from personagraph.input_processing.documents import readers
from personagraph.workspace.documents import application as docstore
from personagraph.model_io.gateway import ModelResult
from personagraph.l2.auxiliary_execution.planning.model_provider import (
    build_auxiliary_architect_structured_provider,
)
from personagraph.l2.auxiliary_execution.application import (
    AuxiliaryApplicationPorts,
    AuxiliaryApplicationResult,
    AuxiliaryApplicationStatus,
)
from personagraph.l2.auxiliary_execution.planning.mounted_document_authority import (
    build_mounted_resource_read_port,
    freeze_mounted_document_planning_authority,
)
from personagraph.l2.auxiliary_execution.production_chain import (
    AuxiliaryProductionChainPorts,
    AuxiliaryProductionChainRequest,
    AuxiliaryProductionChainStatus,
    run_auxiliary_to_verified_delivery,
)
from personagraph.l2.auxiliary_execution.delivery.composition import (
    AuxiliaryTaskDeliveryPorts,
    AuxiliaryTaskDeliveryResult,
    AuxiliaryTaskDeliveryStatus,
)
from personagraph.l2.task_execution.work_run.model_providers import (
    WorkRunStructuredModelProfile,
    build_attempt_structured_provider,
)
from personagraph.l2.auxiliary_execution import production_chain
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import terminal as terminal_store
from personagraph.session.l2_store import verification as verification_store
from personagraph.session.l2_store import auxiliary_graph as auxiliary_graph_store
from personagraph.session.l2_store import task_delivery as task_delivery_store
from tests.runtime.test_auxiliary_planning_controller import (
    _ingest_planning_document,
    _no_mounted_documents,
    _seed_task,
    _virtual_mounted_freshness,
)
from tests.input_processing.documents.conftest import _build, _text_stream
from tests.runtime.test_auxiliary_visual_resource import (
    _RenderingVisionAdapter,
    _ingest_visual_source,
)
from tests.runtime.test_task_delivery_candidate_gate import _candidate_payload
from tests.session.test_auxiliary_task_graph_commit_persistence import (
    _settled_positive_base,
)
from tests.helpers.prepared_model_provider import as_prepared_test_provider


@pytest.fixture(autouse=True)
def _virtual_planning_sources_are_physically_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """合成的挂载链路夹具使用虚构私有路径。"""

    monkeypatch.setenv(readers.ENGINE_ENV_VAR, readers.NATIVE_ENGINE)
    monkeypatch.setattr(
        docstore,
        "check_mounted_document_freshness",
        _virtual_mounted_freshness,
    )


def test_production_chain_reaches_verified_delivery_and_reentry_is_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    def emit(_event):
        return None
    request = AuxiliaryProductionChainRequest(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        max_auxiliary_effect_steps=8,
    )
    ports = AuxiliaryProductionChainPorts(
        auxiliary=AuxiliaryApplicationPorts(model_ledger_store=store, emit=emit),
        delivery=AuxiliaryTaskDeliveryPorts(model_ledger_store=store, emit=emit),
    )

    first = run_auxiliary_to_verified_delivery(request, ports=ports)

    assert first.status is AuxiliaryProductionChainStatus.DELIVERY_READY
    assert first.reason_code == "whole_task_candidate_pass"
    assert first.final_delivery_id is not None
    assert first.publication_body is not None
    assert first.publication_body.strip()
    assert first.publication_format == "markdown"
    assert first.replayed_delivery is False
    with store._connect() as conn:
        before = tuple(
            int(value)
            for value in conn.execute(
                "SELECT "
                "(SELECT COUNT(*) FROM insession_runtime_model_logical_calls "
                " WHERE session_id=?), "
                "(SELECT COUNT(*) FROM insession_work_runs "
                " WHERE session_id=? AND insession_task_id=?), "
                "(SELECT COUNT(*) FROM insession_task_node_deliveries "
                " WHERE session_id=? AND insession_task_id=?)",
                (session_id, session_id, task_id, session_id, task_id),
            ).fetchone()
        )

    def model_must_not_run(*_args, **_kwargs):
        raise AssertionError("completed chain replay must not invoke a model")

    replayed = run_auxiliary_to_verified_delivery(
        request,
        ports=AuxiliaryProductionChainPorts(
            auxiliary=AuxiliaryApplicationPorts(
                model_ledger_store=store,
                emit=emit,
                planning_provider=as_prepared_test_provider(
                    model_must_not_run
                ),
                attempt_provider=as_prepared_test_provider(model_must_not_run),
                verification_provider=as_prepared_test_provider(
                    model_must_not_run
                ),
                semantic_provider=as_prepared_test_provider(model_must_not_run),
            ),
            delivery=AuxiliaryTaskDeliveryPorts(
                model_ledger_store=store,
                emit=emit,
                attempt_provider=as_prepared_test_provider(model_must_not_run),
                verification_provider=as_prepared_test_provider(
                    model_must_not_run
                ),
            ),
        ),
    )

    assert replayed.status is AuxiliaryProductionChainStatus.DELIVERY_READY
    assert replayed.reason_code == "whole_task_candidate_pass_replayed"
    assert replayed.final_delivery_id == first.final_delivery_id
    assert replayed.publication_body == first.publication_body
    assert replayed.replayed_delivery is True
    with store._connect() as conn:
        after = tuple(
            int(value)
            for value in conn.execute(
                "SELECT "
                "(SELECT COUNT(*) FROM insession_runtime_model_logical_calls "
                " WHERE session_id=?), "
                "(SELECT COUNT(*) FROM insession_work_runs "
                " WHERE session_id=? AND insession_task_id=?), "
                "(SELECT COUNT(*) FROM insession_task_node_deliveries "
                " WHERE session_id=? AND insession_task_id=?)",
                (session_id, session_id, task_id, session_id, task_id),
            ).fetchone()
        )
    assert after == before


def test_production_chain_autonomously_revises_then_publishes_only_n_plus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auxiliary_results = iter(
        (
            AuxiliaryApplicationResult(
                status=AuxiliaryApplicationStatus.COMMITTED,
                reason_code="task_graph_revision_one_committed",
                effect_steps=3,
            ),
            AuxiliaryApplicationResult(
                status=AuxiliaryApplicationStatus.COMMITTED,
                reason_code="task_graph_positive_revision_committed",
                effect_steps=4,
            ),
        )
    )
    delivery_results = iter(
        (
            AuxiliaryTaskDeliveryResult(
                status=AuxiliaryTaskDeliveryStatus.REVISION_REQUIRED,
                reason_code="whole_task_validation_revision_required",
            ),
            AuxiliaryTaskDeliveryResult(
                status=AuxiliaryTaskDeliveryStatus.DELIVERY_READY,
                reason_code="verified_root_delivery_ready",
                final_delivery_id="delivery-revision-two",
                publication_body="Only the corrected N+1 answer is publishable.",
                publication_format="markdown",
            ),
        )
    )
    admitted_budgets: list[int] = []

    def run_auxiliary(request, *, ports):
        del ports
        admitted_budgets.append(request.max_effect_steps)
        return next(auxiliary_results)

    monkeypatch.setattr(
        production_chain,
        "run_auxiliary_application_to_boundary",
        run_auxiliary,
    )
    monkeypatch.setattr(
        production_chain,
        "run_auxiliary_committed_task_to_delivery",
        lambda request, *, ports: next(delivery_results),
    )

    result = run_auxiliary_to_verified_delivery(
        AuxiliaryProductionChainRequest(
            session_id="session-loop",
            turn_id="turn-loop",
            task_id="task-loop",
            max_auxiliary_effect_steps=10,
            max_task_graph_revision_cycles=2,
        ),
        ports=AuxiliaryProductionChainPorts(
            auxiliary=AuxiliaryApplicationPorts(model_ledger_store=store, emit=lambda _event: None),
            delivery=AuxiliaryTaskDeliveryPorts(model_ledger_store=store, emit=lambda _event: None),
        ),
    )

    assert result.status is AuxiliaryProductionChainStatus.DELIVERY_READY
    assert result.task_graph_revision_cycles == 1
    assert admitted_budgets == [10, 7]
    assert result.final_delivery_id == "delivery-revision-two"
    assert result.publication_body == (
        "Only the corrected N+1 answer is publishable."
    )


def test_production_chain_revision_budget_exhaustion_never_publishes_failed_n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        production_chain,
        "run_auxiliary_application_to_boundary",
        lambda request, *, ports: AuxiliaryApplicationResult(
            status=AuxiliaryApplicationStatus.COMMITTED,
            reason_code="task_graph_revision_one_committed",
            effect_steps=request.max_effect_steps,
        ),
    )
    monkeypatch.setattr(
        production_chain,
        "run_auxiliary_committed_task_to_delivery",
        lambda request, *, ports: AuxiliaryTaskDeliveryResult(
            status=AuxiliaryTaskDeliveryStatus.REVISION_REQUIRED,
            reason_code="whole_task_validation_revision_required",
        ),
    )

    result = run_auxiliary_to_verified_delivery(
        AuxiliaryProductionChainRequest(
            session_id="session-budget",
            turn_id="turn-budget",
            task_id="task-budget",
            max_auxiliary_effect_steps=1,
        ),
        ports=AuxiliaryProductionChainPorts(
            auxiliary=AuxiliaryApplicationPorts(model_ledger_store=store, emit=lambda _event: None),
            delivery=AuxiliaryTaskDeliveryPorts(model_ledger_store=store, emit=lambda _event: None),
        ),
    )

    assert (
        result.status
        is AuxiliaryProductionChainStatus.STEP_LIMIT_REACHED
    )
    assert result.reason_code == "task_graph_revision_auxiliary_budget_exhausted"
    assert result.final_delivery_id is None
    assert result.publication_body is None


def test_persisted_chain_runs_model_planned_n_to_n_plus_one_validation_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    _no_mounted_documents(monkeypatch, session_id)
    def emit(_event):
        return None

    architect_payloads: list[dict[str, object]] = []
    physical_architect = build_auxiliary_architect_structured_provider()

    def architect_provider(*args, **kwargs) -> ModelResult:
        architect_payloads.append(json.loads(str(args[1])))
        return physical_architect(*args, **kwargs)

    physical_auxiliary_attempt = build_attempt_structured_provider(
        WorkRunStructuredModelProfile(
            attempt_max_output_tokens=8_192,
            verification_max_output_tokens=4_096,
            timeout_s=60.0,
        )
    )
    positive_terminal_calls = 0

    def auxiliary_attempt_provider(*args, **kwargs) -> ModelResult:
        nonlocal positive_terminal_calls
        request_payload = json.loads(str(args[1]))
        result = physical_auxiliary_attempt(*args, **kwargs)
        base = request_payload.get("task_graph_revision_base")
        if base is None:
            return result
        positive_terminal_calls += 1
        assert isinstance(base, dict)
        base_nodes = base["nodes"]
        assert isinstance(base_nodes, list) and base_nodes
        base_alias = base_nodes[0]["node_alias"]
        response = json.loads(result.reply)
        proposal_nodes = response["action"]["proposal"]["root"]["nodes"]
        proposal_nodes[0]["objective"] = (
            f"{proposal_nodes[0]['objective']} Include the required release step."
        )
        proposal_nodes[0]["acceptance_criteria"][0]["criterion"] = (
            "The corrected delivery includes the required release step."
        )
        response["action"]["lineage"] = [
            {
                "proposal_node_key": node["node_key"],
                "disposition": "revise",
                "base_node_alias": base_alias,
            }
            for node in proposal_nodes
        ]
        return replace(
            result,
            reply=json.dumps(response, ensure_ascii=False),
        )

    validation_revisions: list[int] = []

    def task_candidate_validation_provider(
        _system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        prompt = json.loads(user_content)
        graph_revision = int(prompt["graph_revision"])
        validation_revisions.append(graph_revision)
        requires_revision = graph_revision == 1
        payload = _candidate_payload(
            route=("replan_task_graph" if requires_revision else "pass"),
            root_node_id=str(prompt["task_id"]),
        )
        return ModelResult(
            reply=json.dumps(payload),
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    request = AuxiliaryProductionChainRequest(
        session_id=session_id,
        turn_id=turn_id,
        task_id=task_id,
        max_auxiliary_effect_steps=24,
        max_task_graph_revision_cycles=2,
    )
    first = run_auxiliary_to_verified_delivery(
        request,
        ports=AuxiliaryProductionChainPorts(
            auxiliary=AuxiliaryApplicationPorts(
                model_ledger_store=store,
                emit=emit,
                planning_provider=as_prepared_test_provider(architect_provider),
                attempt_provider=as_prepared_test_provider(
                    auxiliary_attempt_provider
                ),
            ),
                delivery=AuxiliaryTaskDeliveryPorts(
                    model_ledger_store=store,
                    emit=emit,
                    task_candidate_validation_provider=as_prepared_test_provider(
                        task_candidate_validation_provider
                    ),
                ),
        ),
    )

    assert first.status is AuxiliaryProductionChainStatus.DELIVERY_READY
    assert first.task_graph_revision_cycles == 1
    assert first.final_delivery_id is not None
    assert first.publication_body is not None
    assert validation_revisions == [1, 2]
    assert positive_terminal_calls == 1
    assert len(architect_payloads) == 2
    assert architect_payloads[0].get("task_graph_revision_trigger") is None
    positive_prompt = architect_payloads[1]
    assert positive_prompt["task_graph_revision_trigger"][
        "base_graph_revision"
    ] == 1
    assert positive_prompt["task_graph_revision_trigger"][
        "target_graph_revision"
    ] == 2
    assert positive_prompt["task_graph_semantic_base"][
        "base_task_graph_revision"
    ] == 1

    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    assert task.current_graph_revision == 2
    assert task.status.value == "completed"
    resolved = verification_store.get_task_node_delivery(
        session_id=session_id,
        delivery_id=first.final_delivery_id,
    )
    assert resolved.delivery.subject.graph_revision == 2
    assert task_delivery_store.get_active_task_graph_revision_trigger(
        session_id=session_id,
        task_id=task_id,
    ) is None
    with store._connect() as conn:
        graph_revisions = tuple(
            int(row[0])
            for row in conn.execute(
                "SELECT graph_revision FROM insession_task_graph_revisions "
                "WHERE insession_task_id=? ORDER BY graph_revision",
                (task_id,),
            ).fetchall()
        )
        before_replay = tuple(
            int(value)
            for value in conn.execute(
                "SELECT "
                "(SELECT COUNT(*) FROM insession_runtime_model_logical_calls "
                " WHERE session_id=?), "
                "(SELECT COUNT(*) FROM insession_work_runs "
                " WHERE session_id=? AND insession_task_id=?), "
                "(SELECT COUNT(*) FROM insession_task_node_deliveries "
                " WHERE session_id=? AND insession_task_id=?), "
                "(SELECT COUNT(*) FROM insession_task_delivery_validation_requests "
                " WHERE session_id=? AND insession_task_id=?), "
                "(SELECT COUNT(*) FROM insession_task_graph_revision_triggers "
                " WHERE session_id=? AND insession_task_id=?), "
                "(SELECT COUNT(*) FROM "
                " insession_task_graph_revision_trigger_applications "
                " WHERE session_id=? AND insession_task_id=?)",
                (
                    session_id,
                    session_id,
                    task_id,
                    session_id,
                    task_id,
                    session_id,
                    task_id,
                    session_id,
                    task_id,
                    session_id,
                    task_id,
                ),
            ).fetchone()
        )
    assert graph_revisions == (1, 2)
    assert before_replay[3:] == (2, 1, 1)

    def model_must_not_run(*_args, **_kwargs):
        raise AssertionError("completed N+1 replay reached a model Provider")

    replayed = run_auxiliary_to_verified_delivery(
        request,
        ports=AuxiliaryProductionChainPorts(
            auxiliary=AuxiliaryApplicationPorts(
                model_ledger_store=store,
                emit=emit,
                planning_provider=as_prepared_test_provider(
                    model_must_not_run
                ),
                attempt_provider=as_prepared_test_provider(model_must_not_run),
                verification_provider=as_prepared_test_provider(
                    model_must_not_run
                ),
                semantic_provider=as_prepared_test_provider(model_must_not_run),
            ),
            delivery=AuxiliaryTaskDeliveryPorts(
                model_ledger_store=store,
                emit=emit,
                attempt_provider=as_prepared_test_provider(model_must_not_run),
                verification_provider=as_prepared_test_provider(
                    model_must_not_run
                ),
            ),
        ),
    )
    assert replayed.status is AuxiliaryProductionChainStatus.DELIVERY_READY
    assert replayed.final_delivery_id == first.final_delivery_id
    assert replayed.replayed_delivery is True
    with store._connect() as conn:
        after_replay = tuple(
            int(value)
            for value in conn.execute(
                "SELECT "
                "(SELECT COUNT(*) FROM insession_runtime_model_logical_calls "
                " WHERE session_id=?), "
                "(SELECT COUNT(*) FROM insession_work_runs "
                " WHERE session_id=? AND insession_task_id=?), "
                "(SELECT COUNT(*) FROM insession_task_node_deliveries "
                " WHERE session_id=? AND insession_task_id=?), "
                "(SELECT COUNT(*) FROM insession_task_delivery_validation_requests "
                " WHERE session_id=? AND insession_task_id=?), "
                "(SELECT COUNT(*) FROM insession_task_graph_revision_triggers "
                " WHERE session_id=? AND insession_task_id=?), "
                "(SELECT COUNT(*) FROM "
                " insession_task_graph_revision_trigger_applications "
                " WHERE session_id=? AND insession_task_id=?)",
                (
                    session_id,
                    session_id,
                    task_id,
                    session_id,
                    task_id,
                    session_id,
                    task_id,
                    session_id,
                    task_id,
                    session_id,
                    task_id,
                ),
            ).fetchone()
        )
    assert after_replay == before_replay


def test_candidate_replan_runs_exact_two_to_three_with_single_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command, _base_root_id = _settled_positive_base(
        "candidate-v2-production-two-to-three"
    )
    committed = terminal_store.commit_auxiliary_task_graph_proposal(command=command)
    assert committed.committed_graph_revision == 2
    session_id = command.session_id
    turn_id = command.source_turn_id
    task_id = command.task_id
    _no_mounted_documents(monkeypatch, session_id)
    before_auxiliary = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert before_auxiliary is not None
    before_goal_id = before_auxiliary.goal_id

    architect_payloads: list[dict[str, object]] = []
    physical_architect = build_auxiliary_architect_structured_provider()

    def architect_provider(*args, **kwargs) -> ModelResult:
        architect_payloads.append(json.loads(str(args[1])))
        return physical_architect(*args, **kwargs)

    physical_auxiliary_attempt = build_attempt_structured_provider(
        WorkRunStructuredModelProfile(
            attempt_max_output_tokens=8_192,
            verification_max_output_tokens=4_096,
            timeout_s=60.0,
        )
    )

    def auxiliary_attempt_provider(*args, **kwargs) -> ModelResult:
        request_payload = json.loads(str(args[1]))
        result = physical_auxiliary_attempt(*args, **kwargs)
        base = request_payload.get("task_graph_revision_base")
        if base is None:
            return result
        assert isinstance(base, dict)
        base_nodes = base["nodes"]
        assert isinstance(base_nodes, list) and base_nodes
        base_alias = base_nodes[0]["node_alias"]
        response = json.loads(result.reply)
        proposal_nodes = response["action"]["proposal"]["root"]["nodes"]
        proposal_nodes[0]["objective"] = (
            f"{proposal_nodes[0]['objective']} Include the missing release step."
        )
        proposal_nodes[0]["acceptance_criteria"][0]["criterion"] = (
            "The corrected delivery includes the missing release step."
        )
        response["action"]["lineage"] = [
            {
                "proposal_node_key": node["node_key"],
                "disposition": "revise",
                "base_node_alias": base_alias,
            }
            for node in proposal_nodes
        ]
        return replace(result, reply=json.dumps(response, ensure_ascii=False))

    candidate_revisions: list[int] = []

    def candidate_provider(
        _system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        prompt = json.loads(user_content)
        graph_revision = int(prompt["graph_revision"])
        candidate_revisions.append(graph_revision)
        route = "replan_task_graph" if graph_revision == 2 else "pass"
        return ModelResult(
            reply=json.dumps(
                _candidate_payload(route=route, root_node_id=task_id),
                ensure_ascii=False,
            ),
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    # 该夹具从已提交的兼容图 N=2 开始，而应用引导入口有意只接受新的无基础目标
    # 或已认证的活跃触发器。仅准入一次既有提交；下方每个“N=2 失败 → 正向规划
    # → N=3”转换仍使用真实生产实现。
    real_run_auxiliary = production_chain.run_auxiliary_application_to_boundary
    auxiliary_entries = 0

    def admit_existing_then_run_positive(request, *, ports):
        nonlocal auxiliary_entries
        auxiliary_entries += 1
        if auxiliary_entries == 1:
            task = task_graph_store.get_insession_task_details(
                request.session_id,
                request.task_id,
            )
            assert task is not None and task.current_graph_revision == 2
            return AuxiliaryApplicationResult(
                status=AuxiliaryApplicationStatus.COMMITTED,
                reason_code="preexisting_task_graph_revision_two_committed",
                effect_steps=0,
            )
        return real_run_auxiliary(request, ports=ports)

    monkeypatch.setattr(
        production_chain,
        "run_auxiliary_application_to_boundary",
        admit_existing_then_run_positive,
    )

    result = run_auxiliary_to_verified_delivery(
        AuxiliaryProductionChainRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            max_auxiliary_effect_steps=24,
            max_task_graph_revision_cycles=2,
        ),
        ports=AuxiliaryProductionChainPorts(
            auxiliary=AuxiliaryApplicationPorts(
                model_ledger_store=store,
                emit=lambda _event: None,
                planning_provider=as_prepared_test_provider(architect_provider),
                attempt_provider=as_prepared_test_provider(
                    auxiliary_attempt_provider
                ),
            ),
            delivery=AuxiliaryTaskDeliveryPorts(
                model_ledger_store=store,
                emit=lambda _event: None,
                task_candidate_validation_provider=as_prepared_test_provider(
                    candidate_provider
                ),
            ),
        ),
    )

    assert result.status is AuxiliaryProductionChainStatus.DELIVERY_READY
    assert result.task_graph_revision_cycles == 1
    assert result.final_delivery_id is not None
    assert result.publication_body is not None
    assert auxiliary_entries == 2
    assert candidate_revisions == [2, 3]
    assert len(architect_payloads) == 1
    positive_prompt = architect_payloads[0]
    assert positive_prompt["task_graph_revision_trigger"][
        "base_graph_revision"
    ] == 2
    assert positive_prompt["task_graph_revision_trigger"][
        "target_graph_revision"
    ] == 3
    assert positive_prompt["task_graph_semantic_base"][
        "base_task_graph_revision"
    ] == 2

    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    assert task.current_graph_revision == 3
    assert task.status.value == "completed"
    resolved = verification_store.get_task_node_delivery(
        session_id=session_id,
        delivery_id=result.final_delivery_id,
    )
    assert resolved.delivery.subject.graph_revision == 3
    after_auxiliary = auxiliary_graph_store.get_auxiliary_graph_for_task(
        session_id=session_id,
        insession_task_id=task_id,
    )
    assert after_auxiliary is not None
    assert after_auxiliary.goal_id != before_goal_id
    assert task_delivery_store.get_active_task_graph_revision_trigger(
        session_id=session_id,
        task_id=task_id,
    ) is None
    with store._connect() as conn:
        graph_revisions = tuple(
            int(row[0])
            for row in conn.execute(
                "SELECT graph_revision FROM insession_task_graph_revisions "
                "WHERE insession_task_id=? ORDER BY graph_revision",
                (task_id,),
            ).fetchall()
        )
        reviewer_calls = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_runtime_model_logical_calls "
                "WHERE session_id=? AND call_kind='task_delivery_validation'",
                (session_id,),
            ).fetchone()[0]
        )
        candidate_calls = int(
            conn.execute(
                "SELECT COUNT(*) FROM insession_runtime_model_logical_calls "
                "WHERE session_id=? AND call_kind="
                "'task_delivery_candidate_validation'",
                (session_id,),
            ).fetchone()[0]
        )
        trigger_rows = tuple(
            conn.execute(
                "SELECT base_graph_revision, target_graph_revision "
                "FROM insession_task_graph_revision_triggers "
                "WHERE session_id=? AND insession_task_id=?",
                (session_id, task_id),
            ).fetchall()
        )
        root_delivery_revisions = tuple(
            int(row[0])
            for row in conn.execute(
                "SELECT graph_revision FROM insession_task_node_deliveries "
                "WHERE session_id=? AND insession_task_id=? "
                "AND insession_task_node_id=? ORDER BY graph_revision",
                (session_id, task_id, task_id),
            ).fetchall()
        )
    assert graph_revisions == (1, 2, 3)
    assert reviewer_calls == 0
    assert candidate_calls == 2
    assert [tuple(row) for row in trigger_rows] == [(2, 3)]
    assert root_delivery_revisions == (2, 3)

    finalized = store.finalize_authoritative_referenced_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=int(
            store.get_turn_execution_window(session_id)["state_version"]
        ),
        post_commit_job_kinds=(),
    )
    assert finalized["node_delivery_ids"] == (result.final_delivery_id,)
    assert store.get_turns(session_id)[1]["content"] == result.publication_body


def test_mounted_document_evidence_crosses_the_full_chain_to_delivery() -> None:
    session_id, turn_id, task_id = _seed_task()
    evidence_text = (
        "Synthetic PDF evidence: Project Birch reached 84 percent accuracy "
        "and its deadline is October 15, 2031."
    )
    ingested_document = _ingest_planning_document(
        session_id=session_id,
        path="/private/synthetic/project-birch.pdf",
        title="Synthetic Project Birch Report",
        content=evidence_text,
    )
    def emit(_event):
        return None

    result = run_auxiliary_to_verified_delivery(
        AuxiliaryProductionChainRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            desired_output="根据挂载 PDF 生成有证据约束的回答",
            max_auxiliary_effect_steps=8,
        ),
        ports=AuxiliaryProductionChainPorts(
            auxiliary=AuxiliaryApplicationPorts(
                task_document_scope=authorized_auxiliary_documents(
                    session_id=session_id,
                    turn_id=turn_id,
                    task_id=task_id,
                    document_ids=(str(ingested_document["doc_id"]),),
                ),
                model_ledger_store=store,
                emit=emit,
            ),
            delivery=AuxiliaryTaskDeliveryPorts(model_ledger_store=store, emit=emit),
        ),
    )

    assert result.status is AuxiliaryProductionChainStatus.DELIVERY_READY
    assert result.publication_body is not None
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    used_anchor_ids = {
        str(anchor_id)
        for node in task.nodes
        for anchor_id in node["source_anchor_ids"]
    }
    evidence_anchor_ids = used_anchor_ids - {"task_creation_source"}
    assert evidence_anchor_ids
    with store._connect() as conn:
        primitive = conn.execute(
            "SELECT status, settled_artifact_id FROM "
            "insession_auxiliary_planning_primitive_invocations "
            "WHERE session_id=?",
            (session_id,),
        ).fetchone()
    assert primitive is not None
    assert primitive["status"] == "settled"
    assert primitive["settled_artifact_id"] is not None


def test_real_pdf_reader_and_docstore_feed_the_verified_delivery_chain(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    pdf_path = tmp_path / "synthetic-project-birch.pdf"
    pdf_path.write_bytes(
        _build(
            [
                {
                    "content": _text_stream(
                        [
                            "Project Birch synthetic report",
                            "Accuracy reached 84 percent",
                            "Deadline is October 15 2031",
                        ]
                    )
                }
            ]
        )
    )
    monkeypatch.setenv(readers.ENGINE_ENV_VAR, readers.NATIVE_ENGINE)

    ingested = ingest_registered_document_fixture(
        str(pdf_path),
        session_id=session_id,
    )

    assert ingested["ok"] is True
    assert ingested["processing_status"] == "complete"
    assert ingested["needs_vision"] is False
    def emit(_event):
        return None
    result = run_auxiliary_to_verified_delivery(
        AuxiliaryProductionChainRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            desired_output="解读 Project Birch PDF 并输出有证据约束的结果",
            max_auxiliary_effect_steps=8,
        ),
        ports=AuxiliaryProductionChainPorts(
            auxiliary=AuxiliaryApplicationPorts(
                task_document_scope=authorized_auxiliary_documents(
                    session_id=session_id,
                    turn_id=turn_id,
                    task_id=task_id,
                    document_ids=(str(ingested["doc_id"]),),
                ),
                model_ledger_store=store,
                emit=emit,
            ),
            delivery=AuxiliaryTaskDeliveryPorts(model_ledger_store=store, emit=emit),
        ),
    )

    assert result.status is AuxiliaryProductionChainStatus.DELIVERY_READY
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    assert task.current_graph_revision == 1
    assert any(
        anchor_id != "task_creation_source"
        for node in task.nodes
        for anchor_id in node["source_anchor_ids"]
    )


@pytest.mark.parametrize("office_format", ("docx", "pptx"))
def test_real_office_reader_and_docstore_feed_the_verified_delivery_chain(
    tmp_path,
    office_format: str,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    path = tmp_path / f"synthetic-project-birch.{office_format}"
    if office_format == "docx":
        docx = pytest.importorskip("docx")
        document = docx.Document()
        document.add_heading("Project Birch", level=1)
        document.add_paragraph("Accuracy reached 84 percent.")
        document.add_paragraph("Deadline is October 15, 2031.")
        document.save(str(path))
    else:
        pptx = pytest.importorskip("pptx")
        presentation = pptx.Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = "Project Birch"
        slide.placeholders[1].text = (
            "Accuracy reached 84 percent. Deadline is October 15, 2031."
        )
        presentation.save(str(path))

    ingested = ingest_registered_document_fixture(
        str(path),
        session_id=session_id,
    )

    assert ingested["ok"] is True
    assert ingested["processing_status"] == "complete"
    def emit(_event):
        return None
    result = run_auxiliary_to_verified_delivery(
        AuxiliaryProductionChainRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            desired_output=(
                f"解读 Project Birch {office_format.upper()} 并输出有证据约束的结果"
            ),
            max_auxiliary_effect_steps=8,
        ),
        ports=AuxiliaryProductionChainPorts(
            auxiliary=AuxiliaryApplicationPorts(
                task_document_scope=authorized_auxiliary_documents(
                    session_id=session_id,
                    turn_id=turn_id,
                    task_id=task_id,
                    document_ids=(str(ingested["doc_id"]),),
                ),
                model_ledger_store=store,
                emit=emit,
            ),
            delivery=AuxiliaryTaskDeliveryPorts(model_ledger_store=store, emit=emit),
        ),
    )

    assert result.status is AuxiliaryProductionChainStatus.DELIVERY_READY
    task = task_graph_store.get_insession_task_details(session_id, task_id)
    assert task is not None
    assert any(
        anchor_id != "task_creation_source"
        for node in task.nodes
        for anchor_id in node["source_anchor_ids"]
    )


@pytest.mark.parametrize("image_suffix", (".png", ".jpg"))
def test_real_image_visual_observation_reaches_verified_delivery(
    tmp_path,
    image_suffix: str,
) -> None:
    session_id, turn_id, task_id = _seed_task()
    ingested_document = _ingest_visual_source(
        tmp_path / f"synthetic-chart{image_suffix}",
        session_id=session_id,
    )
    mounted = freeze_mounted_document_planning_authority(session_id=session_id)
    assert len(mounted.visual_bindings) == 1
    adapter = _RenderingVisionAdapter()
    read_port = build_mounted_resource_read_port(
        mounted_authority=mounted,
        vision_adapter=adapter,
    )
    def emit(_event):
        return None

    result = run_auxiliary_to_verified_delivery(
        AuxiliaryProductionChainRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            desired_output="读取图表数值并形成有视觉证据约束的结果",
            max_auxiliary_effect_steps=8,
        ),
        ports=AuxiliaryProductionChainPorts(
            auxiliary=AuxiliaryApplicationPorts(
                task_document_scope=authorized_auxiliary_documents(
                    session_id=session_id,
                    turn_id=turn_id,
                    task_id=task_id,
                    document_ids=(str(ingested_document["doc_id"]),),
                ),
                model_ledger_store=store,
                emit=emit,
                resource_read_port=read_port,
            ),
            delivery=AuxiliaryTaskDeliveryPorts(model_ledger_store=store, emit=emit),
        ),
    )

    assert result.status is AuxiliaryProductionChainStatus.DELIVERY_READY
    assert len(adapter.seen) == 1
    assert adapter.seen[0].mime_type == (
        "image/jpeg" if image_suffix == ".jpg" else "image/png"
    )
    with store._connect() as conn:
        artifacts = tuple(
            str(row["artifact_json"])
            for row in conn.execute(
                "SELECT artifact_json FROM "
                "insession_auxiliary_planning_context_artifacts "
                "WHERE session_id=?",
                (session_id,),
            ).fetchall()
        )
    assert any("A blue bar reaches 42" in artifact for artifact in artifacts)
