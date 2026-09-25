from __future__ import annotations

import itertools
import json

import pytest

from personagraph.l2.task_execution.task_graph import controller as controller_module
from personagraph.l2.task_execution.work_run import (
    turn_controller as turn_controller_module,
)
from personagraph.l2.task_graph.contracts import (
    InSessionTaskGraphRevisionProposal,
    InSessionTaskGraphRevisionValidationContext,
    InSessionTaskSourceAnchor,
)
from personagraph.l2.task_graph.task_matching import InSessionTaskMatchesProposal
from personagraph.model_io.gateway import ModelGatewayError, ModelResult
from personagraph.l2.task_execution.paper_prompt_context import PaperAttemptContext
from personagraph.l2.task_execution.task_graph.controller import (
    TaskNodeToolRuntimeBinding,
    TaskNodeToolRuntimePlan,
    TaskNodeToolRuntime,
    TaskGraphWorkRunProfile,
    TaskGraphWorkRunRequest,
    run_task_graph_work_runs,
)
from personagraph.l2.task_execution.tool_bridge.work_run_bridge import SqliteWorkRunToolBridge
from personagraph.persistent_turn_content.findings import EXECUTION_FINDINGS_TOOL_IDS
from personagraph.l2.task_execution.work_run.turn_controller import WorkRunTurnStableIdPlan
from personagraph.l2.task_execution.work_run.model_providers import (
    WorkRunStructuredModelProfile,
    build_attempt_structured_provider,
    build_verification_structured_provider,
)
from personagraph.session import store
from personagraph.session.l2_store import task_graph as task_graph_store
from personagraph.session.l2_store import continuation as continuation_store
from personagraph.session.l2_store import verification as verification_store
from personagraph.session.l2_store import work_run as work_run_store
from personagraph.session.persistence.l2.task_graph import insession_tasks as insession_task_records
from personagraph.l2.task_graph.paper_resource_contracts import (
    PaperDocumentBinding,
    PaperResourceSnapshot,
)
from personagraph.session.persistence.l2.work_run.work_execution import (
    RecoverableTaskNodeExecutionCandidate,
)
from personagraph.tools.catalog import ToolCatalog
from personagraph.tools.contracts import (
    ToolSourceDescriptor,
    ToolSourceKind,
    ToolSpec,
)
from personagraph.tools.effects import (
    EffectAction,
    EffectDescriptor,
    EffectResource,
    EffectScopeKind,
    ToolEffectProfile,
)
from personagraph.tools.registration import ToolExecutionProfile, ToolRegistration
from personagraph.l2.work_run import TaskNodeSubject
from tests.helpers.prepared_model_provider import as_prepared_test_provider


USER_TEXT = "请完成一份包含研究和核对步骤的方案"


def _window_revision(session_id: str) -> int:
    window = store.get_turn_execution_window(session_id)
    assert window is not None
    return int(window["state_version"])


def _clock():
    values = itertools.count()
    return lambda: float(next(values))


def _model_profile() -> WorkRunStructuredModelProfile:
    return WorkRunStructuredModelProfile(
        attempt_max_output_tokens=4096,
        verification_max_output_tokens=4096,
        timeout_s=60,
    )


def test_default_profile_reserves_a_separate_complex_planning_envelope() -> None:
    profile = TaskGraphWorkRunProfile()

    task_node = profile.structured_model_profile()
    auxiliary = profile.auxiliary_structured_model_profile()

    assert task_node.attempt_max_output_tokens == 131_072
    assert task_node.verification_max_output_tokens == 131_072
    assert task_node.timeout_s == 45.0
    assert auxiliary.attempt_max_output_tokens == 131_072
    assert auxiliary.verification_max_output_tokens == 131_072
    assert auxiliary.timeout_s == 180.0


def _paper_resources(*, session_id: str, task_id: str) -> PaperAttemptContext:
    return PaperAttemptContext.from_snapshot(
        PaperResourceSnapshot.create(
            session_id=session_id,
            task_id=task_id,
            bound_graph_revision=1,
            bound_task_state_version=1,
            retrieval_data_version_id="rdv-task-graph-paper",
            retrieval_generation_fingerprint="retrieval-task-graph-paper",
            encoder_fingerprint="deterministic-lexical@1",
            documents=(
                PaperDocumentBinding(
                    paper_key="P1",
                    document_id="private-task-graph-document",
                    source_version_id="private-task-graph-version",
                    title="TaskGraph paper",
                    source_sha256="e" * 64,
                    processing_status="complete",
                    admitted_chunk_count=3,
                    chunk_manifest_sha256="f" * 64,
                    admitted_text_page_start=1,
                    admitted_text_page_end=5,
                ),
            ),
        )
    )


def _two_paper_resources(*, session_id: str, task_id: str) -> PaperAttemptContext:
    return PaperAttemptContext.from_snapshot(
        PaperResourceSnapshot.create(
            session_id=session_id,
            task_id=task_id,
            bound_graph_revision=1,
            bound_task_state_version=1,
            retrieval_data_version_id="rdv-task-graph-two-paper",
            retrieval_generation_fingerprint="retrieval-task-graph-two-paper",
            encoder_fingerprint="deterministic-lexical@1",
            documents=tuple(
                PaperDocumentBinding(
                    paper_key=paper_key,
                    document_id=f"private-{paper_key.lower()}",
                    source_version_id=f"private-version-{paper_key.lower()}",
                    title=f"TaskGraph {paper_key} paper",
                    source_sha256=sha * 64,
                    processing_status="complete",
                    admitted_chunk_count=3,
                    chunk_manifest_sha256=manifest_sha * 64,
                    admitted_text_page_start=1,
                    admitted_text_page_end=5,
                )
                for paper_key, sha, manifest_sha in (
                    ("P1", "a", "b"),
                    ("P2", "c", "d"),
                )
            ),
        )
    )


def _readonly_registration(*, handler, tool_id: str = "read.once") -> ToolRegistration:
    return ToolRegistration(
        spec=ToolSpec(
            tool_id=tool_id,
            contract_version="contract-1",
            name=f"Read once ({tool_id})",
            description="Controlled TaskGraph readonly recovery tool.",
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
            catalog_tags=("read",),
        ),
        implementation_version="implementation-1",
        source=ToolSourceDescriptor(
            ToolSourceKind.LOCAL,
            "task-graph-recovery-test",
        ),
        handler=handler,
        effect_profile=ToolEffectProfile(
            (
                EffectDescriptor(
                    EffectResource.MEMORY,
                    EffectAction.READ,
                    EffectScopeKind.LOCAL,
                ),
            )
        ),
        execution_profile=ToolExecutionProfile(),
    )


def _seed_graph(*, child_titles: tuple[str, ...]) -> tuple[str, str, str, int]:
    session_id = store.create_session("Entelecheia")
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id="multi-node-controller-turn",
        source="runtime_test",
        user_text=USER_TEXT,
        lease_owner="multi-node-controller-test",
    )
    turn_id = str(accepted["turn"]["turn_id"])
    applied = task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=turn_id,
        apply_id="multi-node-controller-match",
        proposal=InSessionTaskMatchesProposal.model_validate(
            {
                "task_matches": [
                    {
                        "match_type": "new_root",
                        "local_key": "plan",
                        "title": "完整方案",
                        "objective": "形成经过研究和核对的完整方案",
                        "source_excerpt": USER_TEXT,
                    }
                ]
            }
        ),
        exposed_catalog_ids=(),
        expected_window_revision=_window_revision(session_id),
    )
    task_id = applied.created_insession_task_ids_by_local_key["plan"]
    nodes: list[dict[str, object]] = [
        {
            "node_key": "root",
            "node_kind": "root",
            "title": "完整方案",
            "objective": "汇总直接子节点并形成最终方案",
            "source_anchor_ids": ["request"],
            "acceptance_criteria": [
                {
                    "acceptance_id": "root_complete",
                    "criterion": "最终方案完整且可执行",
                    "source_anchor_ids": ["request"],
                }
            ],
        }
    ]
    for index, title in enumerate(child_titles, start=1):
        nodes.append(
            {
                "node_key": f"child_{index}",
                "node_kind": "subtask",
                "parent_node_key": "root",
                "title": title,
                "objective": f"完成{title}",
                "source_anchor_ids": ["request"],
                "acceptance_criteria": [
                    {
                        "acceptance_id": f"child_{index}_complete",
                        "criterion": f"{title}结果可核对",
                        "source_anchor_ids": ["request"],
                    }
                ],
            }
        )
    committed = insession_task_records.commit_insession_task_graph_revision(
        store._deps(),
        session_id=session_id,
        source_turn_id=turn_id,
        target_insession_task_id=task_id,
        expected_current_graph_revision=None,
        expected_task_state_version=1,
        expected_window_revision=int(applied.window_state_version or 0),
        apply_id="multi-node-controller-graph",
        proposal=InSessionTaskGraphRevisionProposal.model_validate(
            {"root": {"root_key": "root", "nodes": nodes}}
        ),
        trusted_context=InSessionTaskGraphRevisionValidationContext(
            session_id=session_id,
            source_turn_id=turn_id,
            target_insession_task_id=task_id,
            expected_current_graph_revision=None,
            source_anchors=(
                InSessionTaskSourceAnchor(
                    anchor_id="request",
                    source_turn_id=turn_id,
                    source_kind="current_user_instruction",
                    start=0,
                    end=len(USER_TEXT),
                    excerpt=USER_TEXT,
                ),
            ),
            authorization_anchor_ids=("request",),
            required_anchor_ids=("request",),
        ),
    )
    return session_id, turn_id, task_id, committed.window_state_version


def test_drives_leaf_then_canonical_root_and_returns_only_final_delivery():
    session_id, turn_id, task_id, revision = _seed_graph(
        child_titles=("研究步骤",)
    )
    attempt_payloads: list[dict[str, object]] = []
    verification_payloads: list[dict[str, object]] = []
    base_attempt = build_attempt_structured_provider(_model_profile())
    base_verification = build_verification_structured_provider(_model_profile())

    def attempt_provider(system, user, **kwargs):  # type: ignore[no-untyped-def]
        attempt_payloads.append(json.loads(user))
        return base_attempt(system, user, **kwargs)

    def verification_provider(system, user, **kwargs):  # type: ignore[no-untyped-def]
        verification_payloads.append(json.loads(user))
        return base_verification(system, user, **kwargs)

    result = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            expected_window_revision=revision,
        ),
        monotonic_clock=_clock(),
        attempt_provider=as_prepared_test_provider(attempt_provider),
        verification_provider=as_prepared_test_provider(verification_provider),
        emit=lambda _event: None,
    )

    assert result.status == "completed"
    assert result.final_delivery_id is not None
    assert len(result.work_run_ids) == 2
    assert len(attempt_payloads) == len(verification_payloads) == 2
    assert [item["source_context"] for item in attempt_payloads] == [
        item["source_context"] for item in verification_payloads
    ]
    for payload in (*attempt_payloads, *verification_payloads):
        source_context = payload["source_context"]
        assert source_context["node_source_anchor_ids"] == ["request"]
        assert source_context["anchors"][0]["excerpt"] == USER_TEXT
        assert len(source_context["authority_sha256"]) == 64
    assert result.pending_question_attempt_ids == ()
    assert work_run_store.get_completed_task_final_delivery_id(
        session_id=session_id,
        task_id=task_id,
    ) == result.final_delivery_id
    resolved = verification_store.get_task_node_delivery(
        session_id=session_id,
        delivery_id=result.final_delivery_id,
    )
    assert resolved.delivery.subject.node_id == task_id
    with store._connect() as conn:
        requests = conn.execute(
            "SELECT dependency_delivery_ids_json FROM "
            "insession_work_run_verification_requests ORDER BY created_at"
        ).fetchall()
        assert len(requests) == 2
        assert json.loads(requests[0][0]) == []
        assert len(json.loads(requests[1][0])) == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_task_node_deliveries"
        ).fetchone()[0] == 2
        assert conn.execute(
            "SELECT COUNT(*) FROM session_turns WHERE role='assistant'"
        ).fetchone()[0] == 0


def test_findings_feature_off_hides_model_surface_but_keeps_owner_companion():
    session_id, turn_id, task_id, revision = _seed_graph(child_titles=())
    attempts: list[tuple[str, dict[str, object]]] = []
    base_attempt = build_attempt_structured_provider(_model_profile())

    def attempt_provider(system, user, **kwargs):  # type: ignore[no-untyped-def]
        attempts.append((system, json.loads(user)))
        return base_attempt(system, user, **kwargs)

    result = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            expected_window_revision=revision,
            profile=TaskGraphWorkRunProfile(
                execution_findings_enabled=False
            ),
        ),
        monotonic_clock=_clock(),
        attempt_provider=as_prepared_test_provider(attempt_provider),
        verification_provider=build_verification_structured_provider(
            _model_profile()
        ),
        emit=lambda _event: None,
    )

    assert result.status == "completed"
    assert attempts
    for system_prompt, payload in attempts:
        assert "execution_findings 是 Host" not in system_prompt
        assert payload["execution_findings"] is None
        assert EXECUTION_FINDINGS_TOOL_IDS.isdisjoint(
            item["tool_id"] for item in payload["allowed_tools"]
        )
    assert len(result.work_run_ids) == 1
    findings = store.get_execution_findings_ledger_for_owner(
        owner_kind="work_run",
        execution_owner_id=result.work_run_ids[0],
    )
    assert findings is not None
    assert findings.ledger.status.value == "closed"


def test_preflights_and_applies_exact_node_scoped_catalog_and_paper_context():
    session_id, turn_id, task_id, revision = _seed_graph(
        child_titles=("P1 dossier", "P2 dossier")
    )
    spec_documents = (
        PaperDocumentBinding(
            paper_key="P1",
            document_id="private-p1",
            source_version_id="private-v1",
            title="Paper one",
            source_sha256="a" * 64,
            processing_status="complete",
            admitted_chunk_count=1,
            chunk_manifest_sha256="b" * 64,
            admitted_text_page_start=1,
            admitted_text_page_end=1,
        ),
        PaperDocumentBinding(
            paper_key="P2",
            document_id="private-p2",
            source_version_id="private-v2",
            title="Paper two",
            source_sha256="c" * 64,
            processing_status="complete",
            admitted_chunk_count=1,
            chunk_manifest_sha256="d" * 64,
            admitted_text_page_start=1,
            admitted_text_page_end=1,
        ),
    )
    full_context = PaperAttemptContext.from_snapshot(
        PaperResourceSnapshot.create(
            session_id=session_id,
            task_id=task_id,
            bound_graph_revision=1,
            bound_task_state_version=1,
            retrieval_data_version_id="rdv-node-scope",
            retrieval_generation_fingerprint="generation-node-scope",
            encoder_fingerprint="deterministic-lexical@1",
            documents=spec_documents,
        )
    )
    details = task_graph_store.get_insession_task_details(session_id, task_id)
    assert details is not None
    title_by_node = {
        str(node["insession_task_node_id"]): str(node["title"])
        for node in details.nodes
    }

    def snapshot_for(tool_id: str):
        catalog = ToolCatalog()
        catalog.register(
            _readonly_registration(handler=lambda payload: payload, tool_id=tool_id)
        )
        return catalog.snapshot()

    p1_snapshot = snapshot_for("read.p1")
    p2_snapshot = snapshot_for("read.p2")
    root_snapshot = ToolCatalog().snapshot()
    preflighted: list[str] = []

    def runtime_for(subject):  # type: ignore[no-untyped-def]
        title = title_by_node[subject.node_id]
        preflighted.append(title)
        if title == "P1 dossier":
            return TaskNodeToolRuntime(
                catalog_snapshot=p1_snapshot,
                tool_bridge=object(),  # 提交路径绝不会分发 ToolCall
                paper_resources=full_context.for_papers("P1"),
            )
        if title == "P2 dossier":
            return TaskNodeToolRuntime(
                catalog_snapshot=p2_snapshot,
                tool_bridge=object(),
                paper_resources=full_context.for_papers("P2"),
            )
        return TaskNodeToolRuntime(catalog_snapshot=root_snapshot)

    base_attempt = build_attempt_structured_provider(_model_profile())
    payloads: dict[str, dict[str, object]] = {}

    def attempt_provider(system, user, **kwargs):  # type: ignore[no-untyped-def]
        payload = json.loads(user)
        payloads[str(payload["node"]["title"])] = payload
        return base_attempt(system, user, **kwargs)

    result = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            expected_window_revision=revision,
            paper_resources=full_context,
        ),
        monotonic_clock=_clock(),
        attempt_provider=as_prepared_test_provider(attempt_provider),
        verification_provider=build_verification_structured_provider(_model_profile()),
        emit=lambda _event: None,
        node_tool_runtime_factory=runtime_for,
    )

    assert result.status == "completed"
    assert set(preflighted) == {"完整方案", "P1 dossier", "P2 dossier"}
    assert [item["tool_id"] for item in payloads["P1 dossier"]["allowed_tools"]] == [
        "read.p1"
    ]
    assert [item["tool_id"] for item in payloads["P2 dossier"]["allowed_tools"]] == [
        "read.p2"
    ]
    assert [paper["alias"] for paper in payloads["P1 dossier"]["paper_resources"]["papers"]] == ["P1"]
    assert [paper["alias"] for paper in payloads["P2 dossier"]["paper_resources"]["papers"]] == ["P2"]
    assert {
        item["tool_id"] for item in payloads["完整方案"]["allowed_tools"]
    } == EXECUTION_FINDINGS_TOOL_IDS
    assert payloads["完整方案"]["execution_findings"]["ledger_revision"] == 0
    assert payloads["完整方案"]["execution_findings"]["active_entries"] == []
    assert "paper_resources" not in payloads["完整方案"]


def test_node_runtime_preflight_failure_writes_no_work_run_prefix():
    session_id, turn_id, task_id, revision = _seed_graph(
        child_titles=("P1 dossier", "P2 dossier")
    )
    details = task_graph_store.get_insession_task_details(session_id, task_id)
    assert details is not None
    title_by_node = {
        str(node["insession_task_node_id"]): str(node["title"])
        for node in details.nodes
    }

    def runtime_for(subject):  # type: ignore[no-untyped-def]
        if title_by_node[subject.node_id] == "P2 dossier":
            raise RuntimeError("late node runtime unavailable")
        return TaskNodeToolRuntime(catalog_snapshot=ToolCatalog().snapshot())

    result = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            expected_window_revision=revision,
        ),
        monotonic_clock=_clock(),
        emit=lambda _event: None,
        node_tool_runtime_factory=runtime_for,
    )

    assert result.status == "failed_closed"
    assert result.failure_code == "task_node_tool_runtime_unavailable"
    assert result.work_run_ids == ()
    with store._connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM insession_work_runs WHERE insession_task_id=?",
            (task_id,),
        ).fetchone()[0] == 0


def test_first_waiting_child_detaches_and_stops_task_lane_with_exact_one_question():
    session_id, turn_id, task_id, revision = _seed_graph(
        child_titles=("需要补充", "独立核对")
    )
    default_attempt = build_attempt_structured_provider(_model_profile())

    def attempt_provider(
        system_prompt: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        payload = json.loads(user_content)
        if payload["node"]["title"] == "需要补充":
            return ModelResult(
                reply=(
                    '{"acceptance_updates":[],"action":'
                    '{"kind":"request_user_input",'
                    '"question":"请补充第一分支需要的材料。"}}'
                ),
                provider="test",
                model="test",
                latency_ms=1,
                model_call_id=model_call_id,
            )
        return default_attempt(
            system_prompt,
            user_content,
            model_call_id=model_call_id,
            purpose=purpose,
        )

    result = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            expected_window_revision=revision,
        ),
        monotonic_clock=_clock(),
        attempt_provider=as_prepared_test_provider(attempt_provider),
        verification_provider=build_verification_structured_provider(
            _model_profile()
        ),
        emit=lambda _event: None,
    )

    assert result.status == "waiting_user"
    assert len(result.work_run_ids) == 1
    assert len(result.pending_question_attempt_ids) == 1
    window = store.get_turn_execution_window(session_id)
    assert window is not None and window["current_work_run_id"] is None
    pending = continuation_store.list_pending_user_questions(session_id=session_id)
    assert [item.question_attempt_id for item in pending] == list(
        result.pending_question_attempt_ids
    )
    details = task_graph_store.get_insession_task_details(session_id, task_id)
    assert details is not None and details.status.value == "active"
    statuses = {str(node["title"]): str(node["status"]) for node in details.nodes}
    assert statuses["需要补充"] == "awaiting_user"
    assert statuses["独立核对"] == "proposed"
    assert statuses["完整方案"] == "proposed"


def test_closed_world_task_graph_repairs_question_without_persisting_wait() -> None:
    session_id, turn_id, task_id, revision = _seed_graph(child_titles=())
    prompts: list[str] = []
    payloads: list[dict[str, object]] = []
    repair_message_roles: list[tuple[str, ...]] = []

    def attempt_provider(system_prompt, user_content, **kwargs):
        prompt = json.loads(user_content)
        prompts.append(system_prompt)
        payloads.append(prompt)
        repair_messages = kwargs.get("repair_messages")
        if repair_messages is None:
            action = {
                "kind": "request_user_input",
                "question": "请提供附件截图。",
            }
            updates = []
        else:
            assert isinstance(repair_messages, list)
            repair_message_roles.append(
                tuple(str(message["role"]) for message in repair_messages)
            )
            assert repair_messages[1]["content"] == user_content
            action = {
                "kind": "submit_output_window",
                "content": "基于当前授权材料给出最佳证据回答。",
                "format": "plain_text",
            }
            updates = [
                {
                    "acceptance_id": item["acceptance_id"],
                    "model_claimed_satisfied": True,
                    "supporting_tool_result_ids": [],
                    "empty_support_justification": {
                        "schema_version": "empty-support-justification-v1",
                        "reason_code": "candidate_is_primary_artifact",
                        "explanation": "本次提交正文是该条件要求的主要交付物。",
                    },
                }
                for item in prompt["node"]["acceptances"]
            ]
        return ModelResult(
            reply=json.dumps(
                {"acceptance_updates": updates, "action": action},
                ensure_ascii=False,
            ),
            provider="test",
            model="test",
            latency_ms=1,
            model_call_id=kwargs["model_call_id"],
        )

    result = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            expected_window_revision=revision,
            allow_user_input=False,
        ),
        monotonic_clock=_clock(),
        attempt_provider=as_prepared_test_provider(attempt_provider),
        verification_provider=build_verification_structured_provider(
            _model_profile()
        ),
        emit=lambda _event: None,
    )

    assert result.status == "completed"
    assert len(payloads) == 2
    assert payloads[0] == payloads[1]
    assert all("host_repair_feedback" not in payload for payload in payloads)
    assert repair_message_roles == [("system", "user", "assistant", "user")]
    assert all("request_user_input 被禁止" in prompt for prompt in prompts)
    assert continuation_store.list_pending_user_questions(session_id=session_id) == ()
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=result.work_run_ids[0],
    )
    assert all(item.action != "request_user_input" for item in stored.attempts)


def test_answer_turn_continues_exact_waiting_task_node_work_run():
    session_id, question_turn_id, task_id, revision = _seed_graph(child_titles=())
    question = "请明确最终回答面向专家还是普通读者。"
    answer = "面向普通读者。"
    provider_payloads: list[dict[str, object]] = []

    def attempt_provider(system_prompt, user_content, **kwargs):
        payload = json.loads(user_content)
        provider_payloads.append(payload)
        if payload["user_input"]["prior_waiting_user_question"] is None:
            return ModelResult(
                reply=json.dumps(
                    {
                        "acceptance_updates": [],
                        "action": {
                            "kind": "request_user_input",
                            "question": question,
                        },
                    },
                    ensure_ascii=False,
                ),
                provider="test",
                model="test",
                latency_ms=1,
                model_call_id=kwargs["model_call_id"],
            )
        return ModelResult(
            reply=json.dumps(
                {
                    "acceptance_updates": [
                        {
                            "acceptance_id": item["acceptance_id"],
                            "model_claimed_satisfied": True,
                            "supporting_tool_result_ids": [],
                            "empty_support_justification": {
                                "schema_version": (
                                    "empty-support-justification-v1"
                                ),
                                "reason_code": "candidate_is_primary_artifact",
                                "explanation": (
                                    "本次提交正文就是该条件要求的主要交付物。"
                                ),
                            },
                        }
                        for item in payload["node"]["acceptances"]
                    ],
                    "action": {
                        "kind": "submit_output_window",
                        "content": "已按普通读者所需的表达方式完成回答。",
                        "format": "plain_text",
                    },
                },
                ensure_ascii=False,
            ),
            provider="test",
            model="test",
            latency_ms=1,
            model_call_id=kwargs["model_call_id"],
        )

    waiting = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=session_id,
            turn_id=question_turn_id,
            task_id=task_id,
            expected_window_revision=revision,
        ),
        monotonic_clock=_clock(),
        attempt_provider=as_prepared_test_provider(attempt_provider),
        verification_provider=build_verification_structured_provider(
            _model_profile()
        ),
        emit=lambda _event: None,
    )

    assert waiting.status == "waiting_user"
    finalized = store.finalize_turn_execution(
        session_id=session_id,
        turn_id=question_turn_id,
        expected_window_revision=waiting.window_state_version,
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
        client_request_id="multi-node-controller-answer-turn",
        source="runtime_test",
        user_text=answer,
        lease_owner="multi-node-controller-test",
    )
    answer_turn_id = str(accepted["turn"]["turn_id"])
    applied = task_graph_store.apply_insession_task_matches(
        session_id=session_id,
        source_turn_id=answer_turn_id,
        apply_id="multi-node-controller-answer-match",
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

    completed = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=session_id,
            turn_id=answer_turn_id,
            task_id=task_id,
            expected_window_revision=int(applied.window_state_version or 0),
        ),
        monotonic_clock=_clock(),
        attempt_provider=as_prepared_test_provider(attempt_provider),
        verification_provider=build_verification_structured_provider(
            _model_profile()
        ),
        emit=lambda _event: None,
    )

    assert completed.status == "completed"
    assert completed.pending_question_attempt_ids == ()
    assert [payload["user_input"] for payload in provider_payloads] == [
        {
            "content": USER_TEXT,
            "prior_waiting_user_question": None,
        },
        {
            "content": answer,
            "prior_waiting_user_question": question,
        },
    ]
    work_run = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=waiting.work_run_ids[0],
    )
    assert tuple(item.attempt.ordinal for item in work_run.attempts) == (1, 2)
    assert continuation_store.list_pending_user_questions(session_id=session_id) == ()


def test_work_run_cap_counts_dispatches_not_state_machine_iterations():
    child_titles = tuple(f"步骤 {index}" for index in range(1, 17))
    session_id, turn_id, task_id, revision = _seed_graph(
        child_titles=child_titles
    )

    result = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=session_id,
            turn_id=turn_id,
            task_id=task_id,
            expected_window_revision=revision,
            profile=TaskGraphWorkRunProfile(max_work_runs_per_turn=17),
        ),
        monotonic_clock=_clock(),
        emit=lambda _event: None,
    )

    assert result.status == "completed"
    assert len(result.work_run_ids) == 17


def test_task_graph_discovers_and_resumes_readonly_active_idle_after_process_death(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, first_turn_id, task_id, revision = _seed_graph(child_titles=())
    paper_resources = _paper_resources(
        session_id=session_id,
        task_id=task_id,
    )
    handler_calls = 0
    provider_calls = 0
    provider_payloads: list[dict[str, object]] = []

    def read_once(payload: dict[str, str]) -> dict[str, str]:
        nonlocal handler_calls
        handler_calls += 1
        return {"value": payload["value"]}

    catalog = ToolCatalog()
    catalog.register(_readonly_registration(handler=read_once))
    snapshot = catalog.snapshot()

    def persistence_plan(request):
        suffix = ":workrun"
        assert request.work_run_id.endswith(suffix)
        id_plan = WorkRunTurnStableIdPlan(
            namespace=request.work_run_id[: -len(suffix)]
        )
        return id_plan.tool_bridge_persistence_plan(request)

    bridge = SqliteWorkRunToolBridge(
        catalog_snapshot=snapshot,
        persistence_plan_factory=persistence_plan,
    )

    def provider(
        _system: str,
        user_content: str,
        *,
        model_call_id: str,
        purpose: str,
    ) -> ModelResult:
        nonlocal provider_calls
        assert purpose == "runtime_work_run_attempt_decision"
        provider_calls += 1
        provider_payloads.append(json.loads(user_content))
        action = (
            {
                "kind": "call_tools",
                "calls": [
                    {
                        "tool_id": "read.once",
                        "arguments": {"value": "durable"},
                    }
                ],
            }
            if provider_calls == 1
            else {
                "kind": "request_user_input",
                "question": "请确认是否继续。",
            }
        )
        return ModelResult(
            reply=json.dumps(
                {"acceptance_updates": [], "action": action},
                ensure_ascii=False,
            ),
            provider="test",
            model="test",
            latency_ms=1,
            model_call_id=model_call_id,
        )

    original_close = work_run_store.close_work_run_attempt

    def close_then_die(**kwargs: object):
        original_close(**kwargs)
        raise SystemExit("injected TaskGraph process death after readonly close")

    monkeypatch.setattr(work_run_store, "close_work_run_attempt", close_then_die)
    with pytest.raises(SystemExit, match="TaskGraph process death"):
        run_task_graph_work_runs(
            TaskGraphWorkRunRequest(
                session_id=session_id,
                turn_id=first_turn_id,
                task_id=task_id,
                expected_window_revision=revision,
                paper_resources=paper_resources,
            ),
            monotonic_clock=_clock(),
            catalog_snapshot=snapshot,
            attempt_provider=as_prepared_test_provider(provider),
            verification_provider=as_prepared_test_provider(
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    AssertionError("tool/ask path must not invoke verification")
                )
            ),
            emit=lambda _event: None,
            tool_bridge=bridge,
        )

    frontier = work_run_store.project_task_node_execution_frontier(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
    )
    assert len(frontier.recoverable) == 1
    candidate = frontier.recoverable[0]
    assert candidate.current_attempt_id is None
    assert handler_calls == 1

    marked = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=first_turn_id,
        expected_window_revision=_window_revision(session_id),
        stage="TOOL",
        interruption_reason="task_graph_process_lost_after_close",
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
        client_request_id="task-graph-resume-readonly-idle",
        source="runtime_test",
        user_text="继续",
        lease_owner="multi-node-controller-test",
    )
    second_turn_id = str(accepted["turn"]["turn_id"])
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (
                session_id,
                second_turn_id,
                task_id,
                "2026-08-14T00:45:00+00:00",
            ),
        )
    monkeypatch.setattr(work_run_store, "close_work_run_attempt", original_close)

    resumed = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=session_id,
            turn_id=second_turn_id,
            task_id=task_id,
            expected_window_revision=_window_revision(session_id),
            paper_resources=paper_resources,
        ),
        monotonic_clock=_clock(),
        catalog_snapshot=snapshot,
        attempt_provider=as_prepared_test_provider(provider),
        verification_provider=as_prepared_test_provider(
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("tool/ask path must not invoke verification")
            )
        ),
        emit=lambda _event: None,
        tool_bridge=bridge,
    )

    assert resumed.status == "waiting_user"
    assert resumed.work_run_ids == (candidate.work_run_id,)
    assert handler_calls == 1
    assert provider_calls == 2
    expected_paper_resources = paper_resources.to_dict()
    assert [payload["paper_resources"] for payload in provider_payloads] == [
        expected_paper_resources,
        expected_paper_resources,
    ]
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=candidate.work_run_id,
    )
    assert tuple(item.attempt.ordinal for item in stored.attempts) == (1, 2)
    assert len(stored.tool_results) == 1


def _accept_task_graph_recovery_turn(
    *,
    session_id: str,
    interrupted_turn_id: str,
    task_id: str,
    suffix: str,
) -> tuple[str, int]:
    marked = store.mark_turn_execution_interrupted(
        session_id=session_id,
        turn_id=interrupted_turn_id,
        expected_window_revision=_window_revision(session_id),
        stage="VERIFICATION",
        interruption_reason="task_graph_verification_process_lost",
    )
    store.settle_interrupted_turn_execution(
        session_id=session_id,
        turn_id=interrupted_turn_id,
        expected_window_revision=int(marked["state_version"]),
        end_reason="task_graph_verification_process_lost",
        error_code="PROCESS_LOST",
    )
    with store._connect() as conn:
        old_turn = conn.execute(
            "SELECT status FROM runtime_turns WHERE turn_id=?",
            (interrupted_turn_id,),
        ).fetchone()
    assert old_turn is not None and str(old_turn["status"]) == "incomplete"
    accepted = store.accept_turn_execution(
        session_id=session_id,
        client_request_id=f"task-graph-verification-recovery-{suffix}",
        source="runtime_test",
        user_text="继续验证",
        lease_owner="multi-node-controller-test",
    )
    recovery_turn_id = str(accepted["turn"]["turn_id"])
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO insession_task_turn_links "
            "(session_id, turn_id, insession_task_id, insession_task_node_id, "
            "relation, created_at) VALUES (?, ?, ?, NULL, 'referenced', ?)",
            (
                session_id,
                recovery_turn_id,
                task_id,
                "2026-08-15T00:00:00+00:00",
            ),
        )
    return recovery_turn_id, _window_revision(session_id)


def _seed_interrupted_task_graph_verification(
    *,
    suffix: str,
    child_titles: tuple[str, ...] = (),
) -> tuple[str, str, str, int, RecoverableTaskNodeExecutionCandidate]:
    session_id, first_turn_id, task_id, revision = _seed_graph(
        child_titles=child_titles
    )
    interrupted = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=session_id,
            turn_id=first_turn_id,
            task_id=task_id,
            expected_window_revision=revision,
        ),
        monotonic_clock=_clock(),
        verification_provider=as_prepared_test_provider(
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ModelGatewayError(
                    "MODEL_UNAVAILABLE",
                    "injected TaskGraph verification interruption",
                    retryable=False,
                )
            )
        ),
        emit=lambda _event: None,
    )
    assert interrupted.status == "verification_interrupted"
    frontier = work_run_store.project_task_node_execution_frontier(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
    )
    assert len(frontier.recoverable) == 1
    candidate = frontier.recoverable[0]
    assert candidate.work_run_reason == "verification_technical_failure"
    recovery_turn_id, recovery_revision = _accept_task_graph_recovery_turn(
        session_id=session_id,
        interrupted_turn_id=first_turn_id,
        task_id=task_id,
        suffix=suffix,
    )
    return (
        session_id,
        recovery_turn_id,
        task_id,
        recovery_revision,
        candidate,
    )


def test_task_graph_recovers_pending_verification_after_prepare_process_death(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, first_turn_id, task_id, revision = _seed_graph(child_titles=())
    original_invoke = turn_controller_module._invoke_node_verification

    def prepare_then_die(request, **_kwargs):  # type: ignore[no-untyped-def]
        verification_store.prepare_task_node_verification(
            session_id=request.session_id,
            turn_id=request.turn_id,
            work_run_id=request.work_run_id,
            expected_work_run_revision=request.expected_work_run_revision,
            expected_progress_revision=request.expected_progress_revision,
            expected_output_revision=request.expected_output_revision,
            expected_window_revision=request.expected_window_revision,
            apply_id=request.prepare_apply_id,
            verification_request_id=request.verification_request_id,
        )
        raise SystemExit("injected TaskGraph process death after verification prepare")

    monkeypatch.setattr(
        turn_controller_module,
        "_invoke_node_verification",
        prepare_then_die,
    )
    with pytest.raises(SystemExit, match="process death"):
        run_task_graph_work_runs(
            TaskGraphWorkRunRequest(
                session_id=session_id,
                turn_id=first_turn_id,
                task_id=task_id,
                expected_window_revision=revision,
            ),
            monotonic_clock=_clock(),
            emit=lambda _event: None,
        )
    frontier = work_run_store.project_task_node_execution_frontier(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
    )
    assert len(frontier.recoverable) == 1
    candidate = frontier.recoverable[0]
    assert candidate.work_run_reason == "verification_pending"
    request_id = candidate.current_verification_request_id
    assert request_id is not None
    record_before = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id=request_id,
    )
    assert record_before.request.status.value == "pending"
    same_turn = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=session_id,
            turn_id=first_turn_id,
            task_id=task_id,
            expected_window_revision=_window_revision(session_id),
        ),
        monotonic_clock=_clock(),
        verification_provider=as_prepared_test_provider(
            lambda *_args, **_kwargs: pytest.fail(
                "same-Turn pending verification must not reach the verifier"
            )
        ),
        emit=lambda _event: None,
    )
    assert same_turn.status == "failed_closed"
    assert same_turn.failure_code == "verification_recovery_authority_invalid"

    recovery_turn_id, recovery_revision = _accept_task_graph_recovery_turn(
        session_id=session_id,
        interrupted_turn_id=first_turn_id,
        task_id=task_id,
        suffix="pending",
    )
    monkeypatch.setattr(
        turn_controller_module,
        "_invoke_node_verification",
        original_invoke,
    )
    recovered = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=session_id,
            turn_id=recovery_turn_id,
            task_id=task_id,
            expected_window_revision=recovery_revision,
        ),
        monotonic_clock=_clock(),
        emit=lambda _event: None,
    )

    assert recovered.status == "completed", recovered
    assert recovered.work_run_ids == (candidate.work_run_id,)
    record_after = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id=request_id,
    )
    assert record_after.request.verification_request_id == request_id
    assert record_after.request.request_turn_id == first_turn_id
    assert record_after.request.status.value == "completed"
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=candidate.work_run_id,
    )
    assert len(stored.attempts) == 1
    assert len(stored.tool_results) == 0
    id_plan = WorkRunTurnStableIdPlan(
        namespace=candidate.work_run_id[: -len(":workrun")]
    )
    with store._connect() as conn:
        apply_ids = {
            str(row["apply_id"])
            for row in conn.execute(
                "SELECT apply_id FROM insession_work_run_apply_receipts "
                "WHERE work_run_id=?",
                (candidate.work_run_id,),
            ).fetchall()
        }
    assert id_plan.verification_recovery_apply_id(
        1,
        record_before.request.revision,
        recovery_turn_id,
    ) in apply_ids
    assert id_plan.verification_commit_apply_id(
        1,
        record_before.request.revision + 1,
    ) in apply_ids


def test_task_graph_recovers_submit_committed_before_verification_prepare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id, first_turn_id, task_id, revision = _seed_graph(child_titles=())
    original_invoke = turn_controller_module._invoke_node_verification

    def die_before_prepare(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise SystemExit("injected TaskGraph process death before verification prepare")

    monkeypatch.setattr(
        turn_controller_module,
        "_invoke_node_verification",
        die_before_prepare,
    )
    with pytest.raises(SystemExit, match="before verification prepare"):
        run_task_graph_work_runs(
            TaskGraphWorkRunRequest(
                session_id=session_id,
                turn_id=first_turn_id,
                task_id=task_id,
                expected_window_revision=revision,
            ),
            monotonic_clock=_clock(),
            emit=lambda _event: None,
        )

    frontier = work_run_store.project_task_node_execution_frontier(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
    )
    assert len(frontier.recoverable) == 1
    candidate = frontier.recoverable[0]
    assert candidate.work_run_reason == "verification_pending"
    assert candidate.current_verification_request_id is None
    assert candidate.current_attempt_id is None
    stored_before = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=candidate.work_run_id,
    )
    assert len(stored_before.attempts) == 1
    submitted = stored_before.attempts[0]
    assert submitted.action == "submit_output_window"
    id_plan = WorkRunTurnStableIdPlan(
        namespace=candidate.work_run_id[: -len(":workrun")]
    )
    request_id = id_plan.verification_request_id(submitted.attempt.ordinal)

    recovery_turn_id, recovery_revision = _accept_task_graph_recovery_turn(
        session_id=session_id,
        interrupted_turn_id=first_turn_id,
        task_id=task_id,
        suffix="unprepared",
    )
    monkeypatch.setattr(
        turn_controller_module,
        "_invoke_node_verification",
        original_invoke,
    )
    recovered = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=session_id,
            turn_id=recovery_turn_id,
            task_id=task_id,
            expected_window_revision=recovery_revision,
        ),
        monotonic_clock=_clock(),
        emit=lambda _event: None,
    )

    assert recovered.status == "completed", recovered
    assert recovered.work_run_ids == (candidate.work_run_id,)
    record = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id=request_id,
    )
    assert record.request.verification_request_id == request_id
    assert record.request.request_turn_id == recovery_turn_id
    assert record.request.status.value == "completed"
    stored_after = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=candidate.work_run_id,
    )
    assert len(stored_after.attempts) == 1
    assert len(stored_after.tool_results) == 0


def test_task_graph_recovers_interrupted_verification_without_new_attempt() -> None:
    (
        session_id,
        recovery_turn_id,
        task_id,
        recovery_revision,
        candidate,
    ) = _seed_interrupted_task_graph_verification(suffix="interrupted")
    request_id = candidate.current_verification_request_id
    assert request_id is not None
    before = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id=request_id,
    )
    assert before.request.status.value == "interrupted"

    recovered = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=session_id,
            turn_id=recovery_turn_id,
            task_id=task_id,
            expected_window_revision=recovery_revision,
        ),
        monotonic_clock=_clock(),
        emit=lambda _event: None,
    )

    assert recovered.status == "completed"
    assert recovered.work_run_ids == (candidate.work_run_id,)
    after = verification_store.get_task_node_verification_record(
        session_id=session_id,
        verification_request_id=request_id,
    )
    assert after.request.verification_request_id == request_id
    assert after.request.status.value == "completed"
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=candidate.work_run_id,
    )
    assert len(stored.attempts) == 1
    assert len(stored.tool_results) == 0


@pytest.mark.parametrize(
    "corruption",
    ["duplicate", "wrong_task", "wrong_node", "stale_revision"],
)
def test_task_graph_verification_recovery_authority_corruption_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    (
        session_id,
        recovery_turn_id,
        task_id,
        recovery_revision,
        candidate,
    ) = _seed_interrupted_task_graph_verification(suffix=f"corrupt-{corruption}")
    original_frontier = work_run_store.project_task_node_execution_frontier

    def corrupted_frontier(**kwargs):  # type: ignore[no-untyped-def]
        projected = original_frontier(**kwargs)
        if corruption == "duplicate":
            recoverable = (candidate, candidate)
        elif corruption in {"wrong_task", "wrong_node"}:
            subject_update = (
                {"task_id": "crossed-task"}
                if corruption == "wrong_task"
                else {"node_id": "crossed-node"}
            )
            recoverable = (
                candidate.model_copy(
                    update={
                        "subject": candidate.subject.model_copy(
                            update=subject_update
                        )
                    }
                ),
            )
        else:
            recoverable = (
                candidate.model_copy(
                    update={"work_run_revision": candidate.work_run_revision + 1}
                ),
            )
        return projected.model_copy(update={"recoverable": recoverable})

    monkeypatch.setattr(
        work_run_store,
        "project_task_node_execution_frontier",
        corrupted_frontier,
    )
    provider_calls = 0

    def must_not_verify(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("corrupt recovery authority reached the verifier")

    result = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=session_id,
            turn_id=recovery_turn_id,
            task_id=task_id,
            expected_window_revision=recovery_revision,
        ),
        monotonic_clock=_clock(),
        verification_provider=as_prepared_test_provider(must_not_verify),
        emit=lambda _event: None,
    )

    assert result.status == "failed_closed"
    assert result.failure_code in {
        "verification_recovery_authority_ambiguous",
        "verification_recovery_authority_invalid",
    }
    assert provider_calls == 0
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=candidate.work_run_id,
    )
    assert len(stored.attempts) == 1
    assert len(stored.tool_results) == 0


def test_task_graph_verification_recovery_fences_late_old_revisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        session_id,
        recovery_turn_id,
        task_id,
        recovery_revision,
        candidate,
    ) = _seed_interrupted_task_graph_verification(suffix="late-revision")
    original_recover = controller_module.recover_task_node_work_run_verification
    provider_calls = 0

    def race_then_recover(request, **kwargs):  # type: ignore[no-untyped-def]
        verification_store.resume_task_node_verification(
            session_id=request.session_id,
            turn_id=request.turn_id,
            work_run_id=request.work_run_id,
            verification_request_id=request.verification_request_id,
            expected_work_run_revision=request.expected_work_run_revision,
            expected_verification_request_revision=(
                request.expected_verification_request_revision
            ),
            expected_window_revision=request.expected_window_revision,
            apply_id="task-graph-race-advanced-verification",
        )
        return original_recover(request, **kwargs)

    def must_not_verify(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("stale verification request reached the verifier")

    monkeypatch.setattr(
        controller_module,
        "recover_task_node_work_run_verification",
        race_then_recover,
    )
    result = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=session_id,
            turn_id=recovery_turn_id,
            task_id=task_id,
            expected_window_revision=recovery_revision,
        ),
        monotonic_clock=_clock(),
        verification_provider=as_prepared_test_provider(must_not_verify),
        emit=lambda _event: None,
    )

    assert result.status == "failed_closed"
    assert result.failure_code in {
        "turn_window_stale",
        "existing_nonterminal_work_run",
        "verification_recovery_failed",
    }
    assert provider_calls == 0
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=candidate.work_run_id,
    )
    assert len(stored.attempts) == 1
    assert len(stored.tool_results) == 0


@pytest.mark.parametrize(
    ("interrupted_title", "expected_alias", "expected_tool_id"),
    [
        ("P1 dossier", "P1", "read.p1.recovery"),
        ("P2 dossier", "P2", "read.p2.recovery"),
        ("完整方案", None, None),
    ],
)
def test_verification_recovery_uses_exact_frozen_node_runtime_scope(
    monkeypatch: pytest.MonkeyPatch,
    interrupted_title: str,
    expected_alias: str | None,
    expected_tool_id: str | None,
) -> None:
    session_id, first_turn_id, task_id, revision = _seed_graph(
        child_titles=("P1 dossier", "P2 dossier")
    )
    details = task_graph_store.get_insession_task_details(session_id, task_id)
    assert details is not None and details.current_graph_revision == 1
    paper_resources = _two_paper_resources(
        session_id=session_id,
        task_id=task_id,
    )
    handler_calls = 0

    def unused_handler(payload):  # type: ignore[no-untyped-def]
        nonlocal handler_calls
        handler_calls += 1
        return payload

    def snapshot_for(tool_id: str):
        catalog = ToolCatalog()
        catalog.register(
            _readonly_registration(handler=unused_handler, tool_id=tool_id)
        )
        return catalog.snapshot()

    p1_bridge = object()
    p2_bridge = object()
    p1_runtime = TaskNodeToolRuntime(
        catalog_snapshot=snapshot_for("read.p1.recovery"),
        tool_bridge=p1_bridge,  # 提交/恢复路径绝不会分发 ToolCall
        paper_resources=paper_resources.for_papers("P1"),
    )
    p2_runtime = TaskNodeToolRuntime(
        catalog_snapshot=snapshot_for("read.p2.recovery"),
        tool_bridge=p2_bridge,
        paper_resources=paper_resources.for_papers("P2"),
    )
    root_runtime = TaskNodeToolRuntime(catalog_snapshot=ToolCatalog().snapshot())
    runtime_by_title = {
        "P1 dossier": p1_runtime,
        "P2 dossier": p2_runtime,
        "完整方案": root_runtime,
    }
    bindings: list[TaskNodeToolRuntimeBinding] = []
    for node in details.nodes:
        subject = TaskNodeSubject(
            task_id=task_id,
            graph_revision=1,
            node_id=str(node["insession_task_node_id"]),
            node_revision=int(node["node_revision"]),
        )
        bindings.append(
            TaskNodeToolRuntimeBinding(
                subject=subject,
                runtime=runtime_by_title[str(node["title"])],
            )
        )
    runtime_plan = TaskNodeToolRuntimePlan(
        session_id=session_id,
        task_id=task_id,
        graph_revision=1,
        default_runtime=root_runtime,
        bindings=tuple(bindings),
    )
    base_verifier = build_verification_structured_provider(_model_profile())

    def interrupt_selected(
        system_prompt: str,
        user_content: str,
        **kwargs: object,
    ) -> ModelResult:
        payload = json.loads(user_content)
        if payload["node"]["title"] == interrupted_title:
            raise ModelGatewayError(
                "MODEL_UNAVAILABLE",
                f"interrupt {interrupted_title}",
                retryable=False,
            )
        return base_verifier(
            system_prompt,
            user_content,
            model_call_id=str(kwargs["model_call_id"]),
            purpose=str(kwargs["purpose"]),
        )

    interrupted = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=session_id,
            turn_id=first_turn_id,
            task_id=task_id,
            expected_window_revision=revision,
            paper_resources=paper_resources,
        ),
        monotonic_clock=_clock(),
        verification_provider=as_prepared_test_provider(interrupt_selected),
        emit=lambda _event: None,
        node_tool_runtime_plan=runtime_plan,
    )
    assert interrupted.status == "verification_interrupted"
    frontier = work_run_store.project_task_node_execution_frontier(
        session_id=session_id,
        turn_id=first_turn_id,
        task_id=task_id,
    )
    recovery_candidates = tuple(
        item
        for item in frontier.recoverable
        if item.work_run_reason == "verification_technical_failure"
    )
    assert len(recovery_candidates) == 1
    candidate = recovery_candidates[0]
    recovery_turn_id, recovery_revision = _accept_task_graph_recovery_turn(
        session_id=session_id,
        interrupted_turn_id=first_turn_id,
        task_id=task_id,
        suffix=f"scope-{expected_alias or 'root'}",
    )
    original_recover = controller_module.recover_task_node_work_run_verification
    observed: list[tuple[tuple[str, ...], tuple[str, ...], object | None]] = []

    def observe_recovery(request, **kwargs):  # type: ignore[no-untyped-def]
        aliases = (
            tuple(
                str(item["alias"])
                for item in request.paper_resources.to_dict()["papers"]
            )
            if request.paper_resources is not None
            else ()
        )
        tool_ids = tuple(
            entry.registration.spec.tool_id
            for entry in kwargs["catalog_snapshot"].exposed()
        )
        observed.append((aliases, tool_ids, kwargs.get("tool_bridge")))
        return original_recover(request, **kwargs)

    monkeypatch.setattr(
        controller_module,
        "recover_task_node_work_run_verification",
        observe_recovery,
    )
    recovered = run_task_graph_work_runs(
        TaskGraphWorkRunRequest(
            session_id=session_id,
            turn_id=recovery_turn_id,
            task_id=task_id,
            expected_window_revision=recovery_revision,
            paper_resources=paper_resources,
        ),
        monotonic_clock=_clock(),
        verification_provider=base_verifier,
        emit=lambda _event: None,
        node_tool_runtime_plan=runtime_plan,
    )

    assert recovered.status == "completed"
    expected_aliases = (expected_alias,) if expected_alias is not None else ()
    expected_tool_ids = (
        (expected_tool_id,) if expected_tool_id is not None else ()
    )
    expected_bridge = (
        p1_bridge
        if expected_alias == "P1"
        else p2_bridge
        if expected_alias == "P2"
        else None
    )
    assert observed == [(expected_aliases, expected_tool_ids, expected_bridge)]
    stored = work_run_store.get_work_run(
        session_id=session_id,
        work_run_id=candidate.work_run_id,
    )
    assert len(stored.attempts) == 1
    assert len(stored.tool_results) == 0
    assert handler_calls == 0
