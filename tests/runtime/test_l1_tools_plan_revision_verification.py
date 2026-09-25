"""L1 工具、计划修订与验证的端到端锁定测试。"""

from __future__ import annotations
import json
from functools import partial
from pathlib import Path
import pytest
from personagraph.input_processing.vision.contracts import (
    VisionCapabilitySnapshot,
    VisionPurpose,
)
from personagraph.model_io.gateway import ModelResult, PreparedModelCall
from personagraph.output_protocol.l1 import L1ResultReference
from personagraph.persistent_turn_content.evidence import l1_calls_by_ref
from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairFeedback,
    RuntimeModelOutputRepairIssue,
)
from personagraph.runtime.l1 import controller as l1_controller
from personagraph.runtime.l1 import model as l1_model
from personagraph.runtime.l1 import semantic_verification as l1_semantic_verification
from personagraph.runtime.l1.identity import canonical_json, sha256_json
from personagraph.runtime.l1.tool_runtime import build_l1_tool_runtime
from personagraph.runtime import entry
from personagraph.runtime.entry.ingress import model as ingress_model
from personagraph.runtime.entry.routing.policy import (
    TurnRoutingPolicy,
    freeze_turn_routing_policy,
)
from personagraph.session import store as session_store
from personagraph.tools.composition.default_catalog import (
    build_production_default_catalog_seeds,
)
from personagraph.tools.catalog.persistence import (
    EmergencyRevocationSelector,
    ToolCatalogRepository,
)
from tests.helpers.prepared_model_provider import (
    as_prepared_test_provider as _as_prepared_test_provider,
)

as_prepared_test_provider = partial(_as_prepared_test_provider, add_l1_notes=True)


def _model_result(payload: dict[str, object], model_call_id: object) -> ModelResult:
    return ModelResult(
        reply=json.dumps(payload, ensure_ascii=False),
        provider="mock",
        model="mock-structured",
        latency_ms=1,
        model_call_id=str(model_call_id),
    )


def _prepared_structured_sequence(
    payloads: list[dict[str, object] | str],
    calls: list[dict[str, object]],
    *,
    modernize_evidence: bool = True,
):
    """返回只支持准备阶段、并记录每次精确对话的提供方。"""
    remaining = list(payloads)

    def provider(*_args: object, **_kwargs: object) -> ModelResult:
        raise AssertionError("a prepared L1 provider must not use direct dispatch")

    def prepare(
        system_prompt: str, user_content: str, **kwargs: object
    ) -> PreparedModelCall:
        if not remaining:
            raise AssertionError("prepared response sequence was exhausted")
        payload = remaining.pop(0)
        calls.append(
            {"system_prompt": system_prompt, "user_content": user_content, **kwargs}
        )

        def dispatch(model_call_id: str | None) -> ModelResult:
            assert model_call_id is not None
            if modernize_evidence and not isinstance(payload, str):
                return _model_result(payload, model_call_id)
            return ModelResult(
                reply=payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False),
                provider="mock",
                model="mock-structured",
                latency_ms=1,
                model_call_id=model_call_id,
            )

        return PreparedModelCall(_dispatch=dispatch)

    provider.prepare = prepare
    return as_prepared_test_provider(provider)


def _install_l1_classifier(monkeypatch) -> None:
    monkeypatch.setattr(
        ingress_model,
        "complete_structured",
        lambda *_args, **kwargs: _model_result(
            {"processing_level": "L1", "task_matches": []}, kwargs["model_call_id"]
        ),
    )


def _l1_policy():
    return freeze_turn_routing_policy(
        TurnRoutingPolicy(l1_enabled=True, l2_enabled=False), source="request_override"
    )


def _create_l1_session(tmp_path: Path) -> str:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return session_store.create_session("Entelecheia", working_dir=str(workspace))


def _run_l1(
    *,
    session_id: str,
    user_text: str,
    request_id: str,
    features: dict[str, object] | None = None,
):
    ingress_model.complete_structured = as_prepared_test_provider(
        ingress_model.complete_structured
    )
    l1_model.complete_structured = as_prepared_test_provider(
        l1_model.complete_structured
    )
    l1_semantic_verification.complete_structured = as_prepared_test_provider(
        l1_semantic_verification.complete_structured
    )
    return entry.run_entry_turn(
        user_input=user_text,
        features={"context_guard_limit": 24000, **(features or {})},
        session_id=session_id,
        client_request_id=request_id,
        routing_policy=_l1_policy(),
        store=session_store,
    )


def test_l1_uses_the_exact_persisted_production_definition_profile(
    tmp_path: Path,
) -> None:
    """L1 的模型工具面直接来自唯一持久默认表。"""
    session_id = _create_l1_session(tmp_path)
    with session_store.session_database_scope(session_id):
        runtime = build_l1_tool_runtime(session_id)
    expected = tuple(
        (seed.definition.identity for seed in build_production_default_catalog_seeds())
    )
    actual = tuple((definition.identity for definition in runtime.definitions))
    assert actual
    assert actual == expected
    model_catalog = runtime.model_catalog()
    assert tuple((item["tool_id"] for item in model_catalog)) == tuple(
        identity.tool_id
        for identity in expected
        if identity.tool_id in runtime.registrations_by_tool_id
        and identity.tool_id not in runtime.disabled_tool_ids
    )
    assert all(
        (
            set(item)
            == {
                "tool_id",
                "contract_version",
                "name",
                "description",
                "input_schema",
                "catalog_tags",
                "effects",
            }
            for item in model_catalog
        )
    )
    assert all(("output_schema" not in item for item in model_catalog))
    assert all(("input_schema" in item for item in model_catalog))
    assert all(("effects" in item for item in model_catalog))
    assert all((definition.spec.output_schema for definition in runtime.definitions))
    persisted_catalog = json.loads(runtime.catalog_snapshot_json)
    assert all(
        (
            "output_schema" in item["definition"]["spec"]
            for item in persisted_catalog["tools"]
        )
    )
    full_model_catalog_json = canonical_json(
        [
            {
                **definition.spec.to_dict(),
                "effects": [
                    effect.to_dict() for effect in definition.effect_template.effects
                ],
            }
            for definition in runtime.definitions
        ]
    )
    assert (
        len(runtime.model_catalog_json.encode("utf-8"))
        < len(full_model_catalog_json.encode("utf-8")) / 2
    )


def test_l1_rechecks_emergency_revocation_at_resolve_and_execute(
    tmp_path: Path,
) -> None:
    session_id = _create_l1_session(tmp_path)
    with session_store.session_database_scope(session_id):
        resolve_runtime = build_l1_tool_runtime(session_id)
    repository = ToolCatalogRepository()
    repository.issue_emergency_revocation(
        "deny-date-resolution",
        EmergencyRevocationSelector(tool_id="get_today"),
        issued_by="test.security",
        reason="prove L1 reads a fresh resolve-stage deny",
    )
    denied_at_resolve = resolve_runtime.prepare(
        tool_id="get_today", arguments={}, remaining_tool_calls=1
    )
    assert denied_at_resolve.rejected_outcome is not None
    assert denied_at_resolve.rejected_outcome.error is not None
    assert denied_at_resolve.rejected_outcome.error.code == "tool_emergency_revoked"
    repository.clear_emergency_revocation(
        "deny-date-resolution",
        cleared_by="test.security",
        reason="prepare an independently executable call",
    )
    with session_store.session_database_scope(session_id):
        execute_runtime = build_l1_tool_runtime(session_id)
    prepared = execute_runtime.prepare(
        tool_id="get_today", arguments={}, remaining_tool_calls=1
    )
    assert prepared.rejected_outcome is None
    repository.issue_emergency_revocation(
        "deny-date-execution",
        EmergencyRevocationSelector(tool_id="get_today"),
        issued_by="test.security",
        reason="prove L1 rechecks immediately before execution",
    )
    denied_at_execute = execute_runtime.execute_prepared(
        prepared, deadline_monotonic=float("inf")
    )
    assert denied_at_execute.outcome.error is not None
    assert denied_at_execute.outcome.error.code == "tool_emergency_revoked"


def test_l1_keeps_visual_tools_bound_and_checks_each_selected_source(
    monkeypatch, tmp_path, bound_partitioned_session
) -> None:

    class ExternalVision:
        transmits_externally = True

        def capabilities(self):
            return VisionCapabilitySnapshot(
                available=True,
                provider="configured",
                model="configured",
                endpoint_identity="configured",
                processor_fingerprint="configured-external@1",
                supported_purposes=tuple(VisionPurpose),
            )

    monkeypatch.setattr(
        "personagraph.tools.workspace.session_read_source.default_vision_adapter",
        lambda: ExternalVision(),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "chart.png").write_bytes(b"non-empty source for authorization")
    session_id = bound_partitioned_session(working_dir=workspace)
    visual_tools = {"analyze_image", "analyze_pdf_page"}
    unauthorized = build_l1_tool_runtime(session_id)
    unauthorized_model_ids = {item["tool_id"] for item in unauthorized.model_catalog()}
    assert visual_tools <= unauthorized_model_ids
    assert visual_tools <= set(unauthorized.registrations_by_tool_id)
    denied = unauthorized.prepare(
        tool_id="analyze_image",
        arguments={
            "path": "chart.png",
            "purpose": "chart",
            "detail": "standard",
            "region": "page",
        },
        remaining_tool_calls=1,
    )
    assert denied.rejected_outcome is None
    assert denied.requires_protected_dispatch is True
    authorized = build_l1_tool_runtime(session_id)
    authorized_ids = set(authorized.registrations_by_tool_id)
    assert visual_tools <= authorized_ids
    prepared = authorized.prepare(
        tool_id="analyze_image",
        arguments={
            "path": "chart.png",
            "purpose": "chart",
            "detail": "standard",
            "region": "page",
        },
        remaining_tool_calls=1,
    )
    assert prepared.rejected_outcome is None
    assert prepared.policy["disposition"] == "allow"
    assert prepared.requires_protected_dispatch is True


def test_l1_executes_a_real_bound_workspace_tool(
    monkeypatch, tmp_path, bound_partitioned_session
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "brief.txt").write_text(
        "Project Northstar is ready for review.", encoding="utf-8"
    )
    session_id = bound_partitioned_session(working_dir=workspace)
    user_text = "读取 brief.txt 的内容。"
    _install_l1_classifier(monkeypatch)
    payloads: list[dict[str, object]] = []

    def decide(_system: str, user_content: str, **kwargs: object) -> ModelResult:
        payload = json.loads(user_content)
        payloads.append(payload)
        if len(payloads) == 1:
            assert {item["tool_id"] for item in payload["tool_catalog"]} >= {
                "workspace_overview",
                "inspect_file",
            }
            decision: dict[str, object] = {
                "plan": {
                    "objective": "读取指定文件",
                    "acceptances": [{"criterion": "读取 brief.txt 并报告内容"}],
                },
                "action": {
                    "kind": "call_tools",
                    "calls": [
                        {"tool_id": "inspect_file", "arguments": {"path": "brief.txt"}}
                    ],
                },
            }
        else:
            tool_result = payload["prior_tool_results"][0]
            assert tool_result["status"] == "succeeded"
            text = tool_result["result"]["items"][0]["text"]
            assert text == "Project Northstar is ready for review."
            decision = {
                "plan": None,
                "action": {"kind": "submit_final_reply", "reply": text},
            }
        return _model_result(decision, kwargs["model_call_id"])

    monkeypatch.setattr("personagraph.runtime.l1.model.complete_structured", decide)
    result = _run_l1(
        session_id=session_id, user_text=user_text, request_id="l1-real-workspace-tool"
    )
    assert result.status == "completed"
    assert result.reply == "Project Northstar is ready for review."
    execution = session_store.get_l1_turn_execution(
        session_id=session_id, turn_id=result.turn_id
    )
    assert execution is not None
    assert execution["tool_calls"][0]["tool_id"] == "inspect_file"
    assert execution["tool_calls"][0]["status"] == "succeeded"
    assert (
        json.loads(execution["tool_calls"][0]["policy_json"])["disposition"] == "allow"
    )
    verification = json.loads(execution["state"]["verification_report_json"])
    assert verification["contract_version"] == "l1-final-reply-verification"
    assert verification["passed"] is True
    semantic = json.loads(execution["state"]["semantic_verification_report_json"])
    assert semantic["contract_version"] == "l1-semantic-verification-receipt-v2"
    assert semantic["disposition"] == "passed"
    assert semantic["trigger"]["reasons"] == ["policy_always"]
    logical = session_store.get_runtime_model_logical_call(
        session_id=session_id, logical_call_id=semantic["reviewer_logical_call_id"]
    )
    assert logical is not None
    assert logical.request.call_kind == "l1_semantic_verifier"
    assert logical.request.purpose == "runtime_l1_semantic_verifier"


def test_l1_mechanical_rejection_identifies_the_exact_bad_support_binding(
    monkeypatch, tmp_path, bound_partitioned_session
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    private_tool_content = "PRIVATE TOOL RESULT MUST NOT ENTER REPAIR FEEDBACK"
    (workspace / "brief.txt").write_text(private_tool_content, encoding="utf-8")
    session_id = bound_partitioned_session(working_dir=workspace)
    user_text = "读取 brief.txt，然后给出一句概括。"
    _install_l1_classifier(monkeypatch)
    payloads: list[dict[str, object]] = []
    call_ref = ""
    rejected_call_ref = ""
    original_materialize = l1_controller.materialize_decision_references

    def admit_rejected_result_for_gate_test(proposal, *, execution):
        # Bypass only the admission eligibility check to exercise the independent gate.
        bound = original_materialize(proposal.model_copy(update={"references": ()}), execution=execution)
        calls = l1_calls_by_ref(execution)
        return bound.model_copy(update={"references": tuple(
            L1ResultReference(
                tool_call_id=calls[ref.call_ref]["tool_call_id"],
                result_sha256=calls[ref.call_ref]["outcome_hash"],
                chunk_id=ref.chunk_id,
            ) for ref in proposal.references
        )})

    monkeypatch.setattr(
        l1_controller, "materialize_decision_references", admit_rejected_result_for_gate_test
    )

    def decide(_system: str, user_content: str, **kwargs: object) -> ModelResult:
        nonlocal call_ref, rejected_call_ref
        payload = json.loads(user_content)
        payloads.append(payload)
        if len(payloads) == 1:
            decision: dict[str, object] = {
                "plan": {
                    "objective": "读取文件并给出概括",
                    "acceptances": [
                        {"criterion": "读取 brief.txt"},
                        {"criterion": "给出一句概括"},
                    ],
                },
                "action": {
                    "kind": "call_tools",
                    "calls": [
                        {"tool_id": "inspect_file", "arguments": {"path": "brief.txt"}},
                        {
                            "tool_id": "inspect_file",
                            "arguments": {"path": "../outside.txt"},
                        },
                    ],
                },
            }
        elif len(payloads) == 2:
            tool_results = payload["prior_tool_results"]
            result = next(
                (item for item in tool_results if item["status"] == "succeeded")
            )
            rejected = next(
                (item for item in tool_results if item["status"] != "succeeded")
            )
            call_ref = result["call_ref"]
            rejected_call_ref = rejected["call_ref"]
            decision = {
                "plan": None,
                "references": [{"call_ref": rejected_call_ref}],
                "action": {"kind": "submit_final_reply", "reply": "已读取并概括文件。"},
            }
        else:
            rejection = payload["verification_feedback"]
            assert rejection["source"] == "mechanical"
            assert (
                rejection["issues"][0]["code"] == "supporting_tool_result_not_succeeded"
            )
            assert rejection["issues"][0]["call_ref"] == rejected_call_ref
            repair_projection = {
                "feedback": rejection["feedback"],
                "issues": rejection["issues"],
            }
            assert private_tool_content not in json.dumps(
                repair_projection, ensure_ascii=False
            )
            decision = {
                "plan": None,
                "references": [{"call_ref": call_ref}],
                "action": {"kind": "submit_final_reply", "reply": "已读取并概括文件。"},
            }
        return _model_result(decision, kwargs["model_call_id"])

    monkeypatch.setattr("personagraph.runtime.l1.model.complete_structured", decide)
    result = _run_l1(
        session_id=session_id,
        user_text=user_text,
        request_id="l1-specific-mechanical-repair-feedback",
        features={"l1_semantic_verification_mode": "off"},
    )
    assert result.status == "completed"
    assert result.reply == "已读取并概括文件。"
    assert len(payloads) == 3
    execution = session_store.get_l1_turn_execution(
        session_id=session_id, turn_id=result.turn_id
    )
    assert execution is not None
    assert [attempt["status"] for attempt in execution["attempts"]] == [
        "closed",
        "closed",
        "closed",
    ]
    assert [attempt["action_kind"] for attempt in execution["attempts"]] == [
        "call_tools",
        "submit_final_reply",
        "submit_final_reply",
    ]


def test_l1_workspace_write_requires_explicit_approval_and_durable_dispatch(
    monkeypatch, tmp_path, bound_partitioned_session
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = bound_partitioned_session(working_dir=workspace)
    arguments = {"path": "result.txt", "content": "写入已完成。"}
    without_approval = build_l1_tool_runtime(session_id)
    assert "write_workspace_file" in {
        item["tool_id"] for item in without_approval.model_catalog()
    }
    assert "write_workspace_file" in without_approval.registrations_by_tool_id
    denied = without_approval.prepare(
        tool_id="write_workspace_file", arguments=arguments, remaining_tool_calls=1
    )
    assert denied.rejected_outcome is not None
    assert denied.rejected_outcome.error is not None
    assert denied.rejected_outcome.error.code == "tool_policy_authorization_required"
    grant = session_store.grant_session_workspace_write_authority(session_id)
    approved = build_l1_tool_runtime(session_id)
    prepared = approved.prepare(
        tool_id="write_workspace_file", arguments=arguments, remaining_tool_calls=1
    )
    assert prepared.rejected_outcome is None
    assert prepared.requires_protected_dispatch is True
    direct = approved.execute_prepared(prepared, deadline_monotonic=10000000.0)
    assert direct.outcome.error is not None
    assert direct.outcome.error.code == "protected_tool_dispatch_required"
    assert not (workspace / "result.txt").exists()
    _install_l1_classifier(monkeypatch)
    model_payloads: list[dict[str, object]] = []

    def decide(_system: str, user_content: str, **kwargs: object) -> ModelResult:
        payload = json.loads(user_content)
        model_payloads.append(payload)
        if len(model_payloads) == 1:
            assert "write_workspace_file" in {
                item["tool_id"] for item in payload["tool_catalog"]
            }
            decision: dict[str, object] = {
                "plan": {
                    "objective": "将确认文字写入工作目录",
                    "acceptances": [{"criterion": "写入 result.txt"}],
                },
                "action": {
                    "kind": "call_tools",
                    "calls": [
                        {"tool_id": "write_workspace_file", "arguments": arguments}
                    ],
                },
            }
        else:
            tool_result = payload["prior_tool_results"][0]
            assert tool_result["status"] == "succeeded"
            assert tool_result["result"]["path"] == "result.txt"
            decision = {
                "plan": None,
                "action": {
                    "kind": "submit_final_reply",
                    "reply": "已写入 result.txt。",
                },
            }
        return _model_result(decision, kwargs["model_call_id"])

    monkeypatch.setattr("personagraph.runtime.l1.model.complete_structured", decide)
    result = _run_l1(
        session_id=session_id,
        user_text="请把确认文字写入工作目录。",
        request_id="l1-approved-workspace-write",
    )
    assert result.status == "completed"
    assert (workspace / "result.txt").read_text(encoding="utf-8") == "写入已完成。"
    execution = session_store.get_l1_turn_execution(
        session_id=session_id, turn_id=result.turn_id
    )
    assert execution is not None
    tool_call = execution["tool_calls"][0]
    assert tool_call["tool_id"] == "write_workspace_file"
    assert tool_call["status"] == "succeeded"
    assert tool_call["execution_class"] == "protected_effect"
    assert tool_call["protected_phase"] == "succeeded"
    assert tool_call["approval_receipt_ids_json"] == canonical_json([grant.grant_id])
    receipt = json.loads(tool_call["protected_receipt_json"])
    assert receipt["physical_attempt_id"] == tool_call["physical_attempt_id"]
    assert receipt["outcome_status"] == "succeeded"


def test_l1_finds_prepares_retrieves_and_reads_file_through_protected_dispatch(
    monkeypatch, tmp_path, bound_partitioned_session
) -> None:
    """通过 L1 的真实分发器验证 File/FileVersion 工作流。

    测试覆盖路径发现、状态检查、按需准备、检索与精读。该端到端锁还证明：
    在后续只读工具调用消费已准备版本前，准备操作已通过受保护分发完成。
    """
    monkeypatch.setenv("PERSONAGRAPH_DOCUMENT_READER", "native")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for ordinal in range(128):
        (workspace / f"candidate-{ordinal:03d}.md").write_text(
            f"bounded inventory fixture {ordinal}", encoding="utf-8"
        )
    (workspace / "zz-benchmark.md").write_text(
        "# Benchmark\n\nHarbor accuracy is 84.25 percent after three trials.",
        encoding="utf-8",
    )
    session_id = bound_partitioned_session(working_dir=workspace)
    _install_l1_classifier(monkeypatch)
    decisions: list[dict[str, object]] = []
    file_id = ""

    def decide(_system: str, user_content: str, **kwargs: object) -> ModelResult:
        nonlocal file_id
        payload = json.loads(user_content)
        decisions.append(payload)
        if len(decisions) == 1:
            tool_catalog = {item["tool_id"]: item for item in payload["tool_catalog"]}
            assert {
                "find_files",
                "check_files_state",
                "prepare_files",
                "retrieve_files",
                "read_file_chunks",
            } <= set(tool_catalog)
            assert not {
                "select_file_candidates",
                "prepare_file_candidates",
                "retrieve_file_candidates",
            } & set(tool_catalog)
            assert "search_documents" not in tool_catalog
            decision: dict[str, object] = {
                "plan": {
                    "objective": "读取 Harbor benchmark 的准确率",
                    "acceptances": [{"criterion": "读取 benchmark 中的准确率"}],
                },
                "action": {
                    "kind": "call_tools",
                    "calls": [
                        {
                            "tool_id": "find_files",
                            "arguments": {"name": "zz-benchmark.md"},
                        }
                    ],
                },
            }
        elif len(decisions) == 2:
            discovery = payload["prior_tool_results"][-1]
            assert discovery["status"] == "succeeded"
            discovered_items = discovery["result"]["items"]
            assert len(discovered_items) == 1
            assert discovered_items[0]["path"] == "zz-benchmark.md"
            assert "candidate_id" not in discovered_items[0]
            decision = {
                "plan": None,
                "action": {
                    "kind": "call_tools",
                    "calls": [
                        {
                            "tool_id": "check_files_state",
                            "arguments": {"files": [{"path": "zz-benchmark.md"}]},
                        }
                    ],
                },
            }
        elif len(decisions) == 3:
            checked = payload["prior_tool_results"][-1]
            assert checked["status"] == "succeeded"
            checked_item = checked["result"]["results"][0]
            assert checked_item["status"] == "not_ingested"
            assert "file_ref" not in checked_item
            assert checked_item["file_id"] is None
            decision = {
                "plan": None,
                "action": {
                    "kind": "call_tools",
                    "calls": [
                        {
                            "tool_id": "prepare_files",
                            "arguments": {"files": [{"path": "zz-benchmark.md"}]},
                        }
                    ],
                },
            }
        elif len(decisions) == 4:
            prepared = payload["prior_tool_results"][-1]
            assert prepared["status"] == "succeeded"
            prepared_item = prepared["result"]["results"][0]
            assert prepared_item["status"] == "ready"
            file_id = prepared_item["file_id"]
            assert file_id
            assert prepared_item["file_version_id"]
            assert prepared_item["document_version_id"]
            decision = {
                "plan": None,
                "action": {
                    "kind": "call_tools",
                    "calls": [
                        {
                            "tool_id": "retrieve_files",
                            "arguments": {
                                "queries": ["Harbor accuracy"],
                                "file_ids": [file_id],
                            },
                        }
                    ],
                },
            }
        elif len(decisions) == 5:
            retrieved = payload["prior_tool_results"][-1]
            assert retrieved["status"] == "succeeded"
            evidence = retrieved["result"]["files"][0]
            assert evidence["file_id"] == file_id
            assert evidence["document_version_id"]
            decision = {
                "plan": None,
                "action": {
                    "kind": "call_tools",
                    "calls": [
                        {
                            "tool_id": "read_file_chunks",
                            "arguments": {
                                "targets": [
                                    {
                                        "chunk_ids": [
                                            evidence["chunks"][0]["chunk_id"]
                                        ],
                                    }
                                ]
                            },
                        }
                    ],
                },
            }
        else:
            exact = payload["prior_tool_results"][-1]
            assert exact["status"] == "succeeded"
            assert (
                "84.25 percent" in exact["result"]["results"][0]["chunks"][0]["content"]
            )
            decision = {
                "plan": None,
                "action": {
                    "kind": "submit_final_reply",
                    "reply": "Harbor accuracy is 84.25 percent.",
                },
            }
        return _model_result(decision, kwargs["model_call_id"])

    monkeypatch.setattr("personagraph.runtime.l1.model.complete_structured", decide)
    result = _run_l1(
        session_id=session_id,
        user_text="请查询 Harbor benchmark 的准确率。",
        request_id="l1-file-version-protected-vertical",
        features={
            "file_retrieval_write_enabled": True,
            "file_retrieval_read_enabled": True,
            "l1_retrieval_tools_enabled": True,
            "l1_semantic_verification_mode": "off",
        },
    )
    assert result.status == "completed"
    assert result.reply == "Harbor accuracy is 84.25 percent."
    execution = session_store.get_l1_turn_execution(
        session_id=session_id, turn_id=result.turn_id
    )
    assert execution is not None
    tool_calls = execution["tool_calls"]
    calls_by_tool = {item["tool_id"]: item for item in tool_calls}
    assert set(calls_by_tool) == {
        "find_files",
        "check_files_state",
        "prepare_files",
        "retrieve_files",
        "read_file_chunks",
    }
    assert calls_by_tool["prepare_files"]["execution_class"] == "protected_effect"
    assert calls_by_tool["prepare_files"]["protected_phase"] == "succeeded"
    assert calls_by_tool["find_files"]["execution_class"] == "read_only"
    assert calls_by_tool["check_files_state"]["execution_class"] == "read_only"
    assert calls_by_tool["retrieve_files"]["execution_class"] == "read_only"
    assert calls_by_tool["read_file_chunks"]["execution_class"] == "read_only"
    prepared_result = json.loads(calls_by_tool["prepare_files"]["outcome_json"])[
        "result"
    ]["results"][0]
    read_arguments = json.loads(calls_by_tool["read_file_chunks"]["arguments_json"])
    assert set(read_arguments["targets"][0]) == {"chunk_ids"}
    read_result = json.loads(calls_by_tool["read_file_chunks"]["outcome_json"])[
        "result"
    ]["results"][0]
    assert read_result["file_id"] == prepared_result["file_id"]
    assert read_result["document_version_id"] == prepared_result["document_version_id"]
    assert prepared_result["file_id"] in json.dumps(decisions)


def test_l1_workspace_write_is_not_replayed_after_dispatch_interrupt(
    monkeypatch, tmp_path, bound_partitioned_session
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = bound_partitioned_session(working_dir=workspace)
    session_store.grant_session_workspace_write_authority(session_id)
    _install_l1_classifier(monkeypatch)
    arguments = {"path": "once.txt", "content": "只应写入一次。"}

    def decide(_system: str, _user_content: str, **kwargs: object) -> ModelResult:
        return _model_result(
            {
                "plan": {
                    "objective": "写入一次文件",
                    "acceptances": [{"criterion": "写入 once.txt"}],
                },
                "action": {
                    "kind": "call_tools",
                    "calls": [
                        {"tool_id": "write_workspace_file", "arguments": arguments}
                    ],
                },
            },
            kwargs["model_call_id"],
        )

    monkeypatch.setattr("personagraph.runtime.l1.model.complete_structured", decide)
    from personagraph.runtime.l1.protected_tool_dispatch import ToolExecutor

    original_execute = ToolExecutor.execute
    executions = 0

    def interrupt_after_write(self, invocation):
        nonlocal executions
        executions += 1
        original_execute(self, invocation)
        raise KeyboardInterrupt("injected after protected workspace write")

    monkeypatch.setattr(ToolExecutor, "execute", interrupt_after_write)
    with pytest.raises(KeyboardInterrupt, match="after protected workspace write"):
        _run_l1(
            session_id=session_id,
            user_text="请只写入一次。",
            request_id="l1-workspace-write-interrupt",
        )
    assert executions == 1
    assert (workspace / "once.txt").read_text(encoding="utf-8") == "只应写入一次。"
    inspection = session_store.inspect_turn_execution(session_id)
    turn_id = str(inspection["turn"]["turn_id"])
    execution = session_store.get_l1_turn_execution(
        session_id=session_id, turn_id=turn_id
    )
    assert execution is not None
    interrupted_call = execution["tool_calls"][0]
    assert interrupted_call["status"] == "pending"
    assert interrupted_call["protected_phase"] == "dispatching"
    _run_l1(
        session_id=session_id,
        user_text="请只写入一次。",
        request_id="l1-workspace-write-interrupt",
    )
    assert executions == 1
    assert (workspace / "once.txt").read_text(encoding="utf-8") == "只应写入一次。"


def test_l1_semantic_rejection_returns_to_the_next_think_act_step(
    monkeypatch, tmp_path: Path
) -> None:
    session_id = _create_l1_session(tmp_path)
    user_text = "请回答二加二等于几。"
    _install_l1_classifier(monkeypatch)
    generator_prompts: list[str] = []
    generator_payloads: list[dict[str, object]] = []
    reviewer_candidates: list[str] = []

    def decide(system: str, user_content: str, **kwargs: object) -> ModelResult:
        generator_prompts.append(system)
        generator_payloads.append(json.loads(user_content))
        reply = "二加二等于五。" if len(generator_prompts) == 1 else "二加二等于四。"
        return _model_result(
            {
                "plan": {
                    "objective": "回答算术问题",
                    "acceptances": [{"criterion": "回答二加二的结果"}],
                }
                if len(generator_prompts) == 1
                else None,
                "action": {"kind": "submit_final_reply", "reply": reply},
            },
            kwargs["model_call_id"],
        )

    def review(_system: str, user_content: str, **kwargs: object) -> ModelResult:
        payload = json.loads(user_content)
        reply = payload["candidate_final_reply"]
        reviewer_candidates.append(reply)
        if reply.endswith("五。"):
            result: dict[str, object] = {"issues": [{"message": "算术结论不正确。"}]}
        else:
            result = {"issues": []}
        return _model_result(result, kwargs["model_call_id"])

    monkeypatch.setattr("personagraph.runtime.l1.model.complete_structured", decide)
    monkeypatch.setattr(
        "personagraph.runtime.l1.semantic_verification.complete_structured", review
    )
    monkeypatch.setattr(
        "personagraph.runtime.model_calls.requests._sleep", lambda _: None
    )
    result = _run_l1(
        session_id=session_id,
        user_text=user_text,
        request_id="l1-semantic-repair",
        features={"l1_semantic_verification_mode": "always"},
    )
    assert result.status == "completed"
    assert result.reply == "二加二等于四。"
    assert reviewer_candidates == ["二加二等于五。", "二加二等于四。"]
    assert len(generator_prompts) == 2
    assert generator_prompts[0] == generator_prompts[1]
    assert generator_payloads[1]["verification_feedback"]["source"] == "semantic"
    assert "算术结论不正确" in json.dumps(
        generator_payloads[1]["verification_feedback"], ensure_ascii=False
    )
    execution = session_store.get_l1_turn_execution(
        session_id=session_id, turn_id=result.turn_id
    )
    assert execution is not None
    assert execution["state"]["attempts_started"] == 2
    assert [attempt["status"] for attempt in execution["attempts"]] == [
        "closed",
        "closed",
    ]
    assert [attempt["action_kind"] for attempt in execution["attempts"]] == [
        "submit_final_reply",
        "submit_final_reply",
    ]
    semantic = json.loads(execution["state"]["semantic_verification_report_json"])
    assert semantic["disposition"] == "passed"
    assert semantic["trigger"]["reasons"] == ["policy_always"]
    assert semantic["reviewer_result"] == {"issues": []}


def test_l1_semantic_reviewer_receives_source_and_retries_factual_conflict(
    monkeypatch, tmp_path: Path
) -> None:
    """通用审查收到用户材料与候选，事实冲突被反馈后可重新提交。"""
    session_id = _create_l1_session(tmp_path)
    user_text = "材料写明签署日期为 2031-07-08，Cedar 为 960 ms。请准确复述这两个值。"
    _install_l1_classifier(monkeypatch)
    decision_calls = 0
    reviewer_prompts: list[str] = []
    reviewer_payloads: list[dict[str, object]] = []

    def decide(_system: str, _user_content: str, **kwargs: object) -> ModelResult:
        nonlocal decision_calls
        decision_calls += 1
        reply = (
            "签署日期为 2021-07-08，Cedar 为 960 ms。"
            if decision_calls == 1
            else "签署日期为 2031-07-08，Cedar 为 960 ms。"
        )
        return _model_result(
            {
                "plan": {
                    "objective": "准确复述材料中的日期与延迟",
                    "acceptances": [{"criterion": "复述签署日期与 Cedar 延迟"}],
                }
                if decision_calls == 1
                else None,
                "action": {"kind": "submit_final_reply", "reply": reply},
            },
            kwargs["model_call_id"],
        )

    def review(system_prompt: str, user_content: str, **kwargs: object) -> ModelResult:
        payload = json.loads(user_content)
        reviewer_prompts.append(system_prompt)
        reviewer_payloads.append(payload)
        wrong_year = "2021-07-08" in payload["candidate_final_reply"]
        return _model_result(
            {
                "issues": [
                    {"message": "候选日期 2021-07-08 与材料日期 2031-07-08 冲突。"}
                ]
                if wrong_year
                else []
            },
            kwargs["model_call_id"],
        )

    monkeypatch.setattr("personagraph.runtime.l1.model.complete_structured", decide)
    monkeypatch.setattr(
        "personagraph.runtime.l1.semantic_verification.complete_structured", review
    )
    result = _run_l1(
        session_id=session_id,
        user_text=user_text,
        request_id="l1-literal-audit",
        features={"l1_semantic_verification_mode": "always"},
    )
    assert result.status == "completed"
    assert result.reply == "签署日期为 2031-07-08，Cedar 为 960 ms。"
    assert reviewer_payloads[0]["request_context"]["current_user_text"] == user_text
    assert (
        reviewer_payloads[0]["candidate_final_reply"]
        == "签署日期为 2021-07-08，Cedar 为 960 ms。"
    )
    assert all(("事实依据" in prompt for prompt in reviewer_prompts))
    assert all(
        (
            "未明确指定语言时，以用户请求使用的语言为准" in prompt
            for prompt in reviewer_prompts
        )
    )


def test_l1_recovers_after_verifier_feedback_was_persisted_before_process_loss(
    monkeypatch, tmp_path: Path
) -> None:
    session_id = _create_l1_session(tmp_path)
    user_text = "请回答二加二等于几。"
    _install_l1_classifier(monkeypatch)
    generator_payloads: list[dict[str, object]] = []
    reviewer_candidates: list[str] = []

    def decide(_system: str, user_content: str, **kwargs: object) -> ModelResult:
        payload = json.loads(user_content)
        generator_payloads.append(payload)
        first = len(generator_payloads) == 1
        reply = "二加二等于五。" if first else "二加二等于四。"
        return _model_result(
            {
                "plan": {
                    "objective": "回答算术问题",
                    "acceptances": [{"criterion": "回答二加二的结果"}],
                }
                if first
                else None,
                "action": {"kind": "submit_final_reply", "reply": reply},
            },
            kwargs["model_call_id"],
        )

    def review(_system: str, user_content: str, **kwargs: object) -> ModelResult:
        payload = json.loads(user_content)
        reply = payload["candidate_final_reply"]
        reviewer_candidates.append(reply)
        revise = reply.endswith("五。")
        return _model_result(
            {"issues": [{"message": "算术结论不正确。"}] if revise else []},
            kwargs["model_call_id"],
        )

    monkeypatch.setattr("personagraph.runtime.l1.model.complete_structured", decide)
    monkeypatch.setattr(
        "personagraph.runtime.l1.semantic_verification.complete_structured", review
    )
    original_reject = session_store.reject_l1_final_reply_candidate

    def persist_then_lose(**kwargs: object) -> dict[str, object]:
        original_reject(**kwargs)
        raise KeyboardInterrupt("injected verifier-feedback response loss")

    monkeypatch.setattr(
        session_store, "reject_l1_final_reply_candidate", persist_then_lose
    )
    with pytest.raises(KeyboardInterrupt, match="verifier-feedback response loss"):
        _run_l1(
            session_id=session_id,
            user_text=user_text,
            request_id="l1-semantic-rejection-recovery",
            features={"l1_semantic_verification_mode": "always"},
        )
    monkeypatch.setattr(
        session_store, "reject_l1_final_reply_candidate", original_reject
    )
    resumed = entry.resume_active_l1_entry_turn(
        session_id=session_id, features={}, store=session_store
    )
    assert not isinstance(resumed, str)
    assert resumed.status == "completed"
    assert resumed.reply == "二加二等于四。"
    assert reviewer_candidates == ["二加二等于五。", "二加二等于四。"]
    assert generator_payloads[1]["verification_feedback"]["source"] == "semantic"
    execution = session_store.get_l1_turn_execution(
        session_id=session_id, turn_id=resumed.turn_id
    )
    assert execution is not None
    assert [attempt["status"] for attempt in execution["attempts"]] == [
        "closed",
        "closed",
    ]
    assert [attempt["action_kind"] for attempt in execution["attempts"]] == [
        "submit_final_reply",
        "submit_final_reply",
    ]


def test_l1_store_rejects_a_semantic_receipt_bound_to_another_candidate(
    monkeypatch, tmp_path: Path
) -> None:
    session_id = _create_l1_session(tmp_path)
    user_text = "请直接确认收到。"
    _install_l1_classifier(monkeypatch)
    monkeypatch.setattr(
        "personagraph.runtime.l1.model.complete_structured",
        lambda *_args, **kwargs: _model_result(
            {
                "plan": {
                    "objective": "确认收到",
                    "acceptances": [{"criterion": "确认收到用户消息"}],
                },
                "action": {"kind": "submit_final_reply", "reply": "收到。"},
            },
            kwargs["model_call_id"],
        ),
    )
    original_commit = session_store.commit_l1_attempt_decision

    def tamper(**kwargs: object):
        report_json = kwargs.get("semantic_verification_report_json")
        if isinstance(report_json, str):
            report = json.loads(report_json)
            report["decision_hash"] = "0" * 64
            replacement = canonical_json(report)
            kwargs = {
                **kwargs,
                "semantic_verification_report_json": replacement,
                "semantic_verification_report_hash": sha256_json(report),
            }
        return original_commit(**kwargs)

    monkeypatch.setattr(session_store, "commit_l1_attempt_decision", tamper)
    result = _run_l1(
        session_id=session_id,
        user_text=user_text,
        request_id="l1-semantic-receipt-tamper",
        features={"l1_semantic_verification_mode": "always"},
    )
    assert result.status == "incomplete"
    assert result.error_code == "INTERNAL_FAILURE"
    execution = session_store.get_l1_turn_execution(
        session_id=session_id, turn_id=result.turn_id
    )
    assert execution is not None
    assert execution["run"]["status"] == "failed"
    assert execution["state"]["final_reply"] is None
    assert (
        session_store.get_committed_turn_pair(session_id, f"commit_{result.turn_id}")
        is None
    )


def test_l1_semantic_verifier_repairs_one_invalid_output(
    monkeypatch, tmp_path: Path
) -> None:
    session_id = _create_l1_session(tmp_path)
    user_text = "请确认语义验证格式修复。"
    _install_l1_classifier(monkeypatch)
    monkeypatch.setattr(
        "personagraph.runtime.l1.model.complete_structured",
        lambda *_args, **kwargs: _model_result(
            {
                "plan": {
                    "objective": "确认格式修复",
                    "acceptances": [{"criterion": "确认语义验证格式修复"}],
                },
                "action": {"kind": "submit_final_reply", "reply": "已确认。"},
            },
            kwargs["model_call_id"],
        ),
    )
    reviewer_prompts: list[str] = []
    reviewer_repairs: list[list[dict[str, str]]] = []

    def review(system: str, _user_content: str, **kwargs: object) -> ModelResult:
        reviewer_prompts.append(system)
        repair_messages = kwargs.get("repair_messages")
        if repair_messages is not None:
            assert isinstance(repair_messages, list)
            reviewer_repairs.append(repair_messages)
        payload: dict[str, object]
        if len(reviewer_prompts) == 1:
            payload = {}
        else:
            payload = {"issues": []}
        return _model_result(payload, kwargs["model_call_id"])

    monkeypatch.setattr(
        "personagraph.runtime.l1.semantic_verification.complete_structured", review
    )
    monkeypatch.setattr(
        "personagraph.runtime.model_calls.requests._sleep", lambda _: None
    )
    result = _run_l1(
        session_id=session_id,
        user_text=user_text,
        request_id="l1-semantic-format-repair",
        features={"l1_semantic_verification_mode": "always"},
    )
    assert result.status == "completed"
    assert len(reviewer_prompts) == 2
    assert "acceptance_id 可省略" in reviewer_prompts[0]
    assert "related_result_refs" not in reviewer_prompts[0]
    assert reviewer_prompts[1] == reviewer_prompts[0]
    assert len(reviewer_repairs) == 1
    messages = reviewer_repairs[0]
    assert [item["role"] for item in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    repair_envelope = json.loads(messages[3]["content"].split("Host 修复清单：", 1)[1])
    assert set(repair_envelope) == {"current_issues"}
    assert {
        path for issue in repair_envelope["current_issues"] for path in issue["paths"]
    } == {"/issues"}
    assert "不得改写候选" in messages[0]["content"]
    execution = session_store.get_l1_turn_execution(
        session_id=session_id, turn_id=result.turn_id
    )
    assert execution is not None
    semantic = json.loads(execution["state"]["semantic_verification_report_json"])
    logical = session_store.get_runtime_model_logical_call(
        session_id=session_id, logical_call_id=semantic["reviewer_logical_call_id"]
    )
    assert logical is not None
    assert len(logical.physical_attempts) == 2
    feedback = logical.physical_attempts[0].settlement.next_output_repair_feedback
    assert feedback is not None
    assert {issue.code for issue in feedback.current_issues} == {"schema.missing"}
    assert {path for issue in feedback.current_issues for path in issue.paths} == {
        "/issues"
    }


@pytest.mark.parametrize(
    "malformed_action",
    [
        {"kind": "submit_final_reply"},
        {"kind": "submit_final_reply", "reply": {"PRIVATE_SENTINEL": "not a string"}},
    ],
)
def test_l1_prepared_decision_repair_uses_four_message_context(
    monkeypatch, tmp_path: Path, malformed_action
) -> None:
    session_id = _create_l1_session(tmp_path)
    user_text = "请完整确认这个请求。"
    _install_l1_classifier(monkeypatch)
    plan = {"objective": "确认请求", "acceptances": [{"criterion": "完整确认请求"}]}
    malformed = {
        "plan": plan,
        "note": "提交当前答复候选。",
        "action": malformed_action,
    }
    corrected = {
        "plan": plan,
        "note": "提交当前答复候选。",
        "action": {"kind": "submit_final_reply", "reply": "我已完整确认这个请求。"},
    }
    prepared_calls: list[dict[str, object]] = []
    provider = _prepared_structured_sequence([malformed, corrected], prepared_calls)
    monkeypatch.setattr("personagraph.runtime.l1.model.complete_structured", provider)
    result = _run_l1(
        session_id=session_id,
        user_text=user_text,
        request_id="l1-prepared-decision-repair",
        features={"l1_semantic_verification_mode": "off"},
    )
    assert result.status == "completed"
    assert result.reply == "我已完整确认这个请求。"
    assert len(prepared_calls) == 2
    first, repair = prepared_calls
    assert "repair_messages" not in first
    assert first["system_prompt"] == repair["system_prompt"]
    assert first["user_content"] == repair["user_content"]
    messages = repair["repair_messages"]
    assert isinstance(messages, list)
    assert [item["role"] for item in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[0]["content"] == first["system_prompt"]
    assert messages[1]["content"] == first["user_content"]
    assert messages[2]["content"] == _model_result(malformed, "ignored").reply
    repair_envelope = json.loads(messages[3]["content"].split("Host 修复清单：", 1)[1])
    assert {
        "schema_version",
        "message_contract",
        "rejected_response_sha256",
    }.isdisjoint(repair_envelope)
    assert set(repair_envelope) == {"current_issues"}
    assert {tuple(item["paths"]) for item in repair_envelope["current_issues"]} >= {
        ("/action/reply",)
    }
    assert "PRIVATE_SENTINEL" not in json.dumps(repair_envelope)
    execution = session_store.get_l1_turn_execution(
        session_id=session_id, turn_id=result.turn_id
    )
    assert execution is not None
    logical = session_store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=execution["attempts"][0]["logical_model_call_id"],
    )
    assert logical is not None
    policy = json.loads(logical.request.request_json)["repair_policy"]
    assert policy["feedback_contract"] == "runtime-model-output-repair-feedback-v2"
    feedback = logical.physical_attempts[0].settlement.next_output_repair_feedback
    assert isinstance(feedback, RuntimeModelOutputRepairFeedback)


def test_l1_prepared_json_syntax_repair_keeps_parser_location(
    monkeypatch, tmp_path: Path
) -> None:
    session_id = _create_l1_session(tmp_path)
    _install_l1_classifier(monkeypatch)
    malformed = '{\n  "note": "PRIVATE_SENTINEL",\n  "action": }'
    with pytest.raises(json.JSONDecodeError) as parse_failure:
        json.loads(malformed)
    corrected = {
        "note": "提交确认。",
        "plan": {"objective": "确认请求", "acceptances": [{"criterion": "确认请求"}]},
        "action": {"kind": "submit_final_reply", "reply": "已确认请求。"},
    }
    prepared_calls: list[dict[str, object]] = []
    provider = _prepared_structured_sequence([malformed, corrected], prepared_calls)
    monkeypatch.setattr("personagraph.runtime.l1.model.complete_structured", provider)
    result = _run_l1(
        session_id=session_id, user_text="请确认请求。", request_id="l1-json-location",
        features={"l1_semantic_verification_mode": "off"},
    )
    assert result.status == "completed"
    assert len(prepared_calls) == 2
    messages = prepared_calls[1]["repair_messages"]
    assert messages[2]["content"] == malformed
    envelope = json.loads(messages[3]["content"].split("Host 修复清单：", 1)[1])
    assert envelope == {"current_issues": [{
        "paths": [""],
        "safe_explanation": "输出不是有效的 JSON object。",
        "json_line": parse_failure.value.lineno,
        "json_column": parse_failure.value.colno,
    }]}
    assert "PRIVATE_SENTINEL" not in json.dumps(envelope)
    execution = session_store.get_l1_turn_execution(
        session_id=session_id, turn_id=result.turn_id,
    )
    logical = session_store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=execution["attempts"][0]["logical_model_call_id"],
    )
    feedback = logical.physical_attempts[0].settlement.next_output_repair_feedback
    issue = feedback.current_issues[0]
    assert issue.code == "json_syntax.invalid_json"
    assert (issue.json_line, issue.json_column) == (
        parse_failure.value.lineno, parse_failure.value.colno,
    )
    assert logical.physical_attempts[1].request.output_repair_feedback == feedback


def test_l1_prepared_host_repair_accepts_the_full_issue_byte_budget(
    monkeypatch, tmp_path: Path
) -> None:
    session_id = _create_l1_session(tmp_path)
    _install_l1_classifier(monkeypatch)
    issue = RuntimeModelOutputRepairIssue(
        category="host_guard", code="host_guard.test_bounded_feedback",
        paths=("/action",), safe_explanation="x" * 500,
    )
    original_admit = l1_controller._admit_decision
    rejected = False

    def admit_once(decision, **kwargs):
        nonlocal rejected
        if not rejected:
            rejected = True
            raise l1_controller.L1ControllerFailure(
                l1_controller.RuntimeErrorCode.MODEL_OUTPUT_INVALID,
                "PRIVATE_SENTINEL", repair_issue=issue,
            )
        return original_admit(decision, **kwargs)

    monkeypatch.setattr(l1_controller, "_admit_decision", admit_once)
    decision = {
        "note": "确认请求。",
        "plan": {"objective": "确认请求", "acceptances": [{"criterion": "确认请求"}]},
        "action": {"kind": "submit_final_reply", "reply": "已确认。"},
    }
    prepared_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        l1_model, "complete_structured",
        _prepared_structured_sequence([decision, decision], prepared_calls),
    )
    result = _run_l1(
        session_id=session_id, user_text="请确认。", request_id="l1-full-issue-budget",
        features={"l1_semantic_verification_mode": "off"},
    )
    assert result.status == "completed"
    assert len(prepared_calls) == 2
    instruction = prepared_calls[1]["repair_messages"][3]["content"]
    envelope = json.loads(instruction.split("Host 修复清单：", 1)[1])
    assert envelope["current_issues"][0]["safe_explanation"] == issue.safe_explanation
    assert "PRIVATE_SENTINEL" not in instruction


def test_l1_prepared_findings_revision_repair_keeps_argument_location(
    monkeypatch, tmp_path: Path
) -> None:
    session_id = _create_l1_session(tmp_path)
    _install_l1_classifier(monkeypatch)
    plan = {"objective": "确认请求", "acceptances": [{"criterion": "确认请求"}]}
    prepared_calls: list[dict[str, object]] = []
    provider = _prepared_structured_sequence([
        {
            "plan": plan,
            "action": {"kind": "call_tools", "calls": [{
                "tool_id": "record_execution_findings",
                "arguments": {"expected_ledger_revision": "PRIVATE_SENTINEL"},
            }]},
        },
        {"plan": plan, "action": {"kind": "submit_final_reply", "reply": "已确认。"}},
    ], prepared_calls)
    monkeypatch.setattr(l1_model, "complete_structured", provider)
    result = _run_l1(
        session_id=session_id, user_text="请确认。", request_id="l1-findings-cas-repair",
        features={"l1_semantic_verification_mode": "off"},
    )
    assert result.status == "completed"
    assert len(prepared_calls) == 2
    messages = prepared_calls[1]["repair_messages"]
    envelope = json.loads(messages[3]["content"].split("Host 修复清单：", 1)[1])
    assert envelope["current_issues"][0]["paths"] == [
        "/action/calls/0/arguments/expected_ledger_revision",
    ]
    assert "省略" in envelope["current_issues"][0]["safe_explanation"]
    assert "PRIVATE_SENTINEL" not in json.dumps(envelope)
    execution = session_store.get_l1_turn_execution(
        session_id=session_id, turn_id=result.turn_id,
    )
    assert execution["state"]["attempts_started"] == 1
    assert execution["tool_calls"] == []
    logical = session_store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=execution["attempts"][0]["logical_model_call_id"],
    )
    feedback = logical.physical_attempts[0].settlement.next_output_repair_feedback
    assert feedback.current_issues[0].code == "host_guard.l1_findings_revision_mismatch"
    assert feedback.issue_coverage.value == "first_only"
    assert logical.physical_attempts[1].request.output_repair_feedback == feedback


def test_l1_prepared_finalization_repair_reports_exact_host_guard(
    monkeypatch, tmp_path, bound_partitioned_session
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "brief.txt").write_text("ready", encoding="utf-8")
    session_id = bound_partitioned_session(working_dir=workspace)
    user_text = "读取 brief.txt；若最后一步无法继续，就如实提交部分结果。"
    _install_l1_classifier(monkeypatch)
    plan = {
        "objective": "读取文件并交付结果",
        "acceptances": [{"criterion": "读取 brief.txt 并交付结果"}],
    }
    first_tool_call = {
        "plan": plan,
        "action": {
            "kind": "call_tools",
            "calls": [{"tool_id": "inspect_file", "arguments": {"path": "brief.txt"}}],
        },
    }
    rejected_final_tool_call = {
        "plan": None,
        "action": {
            "kind": "call_tools",
            "calls": [
                {
                    "tool_id": "inspect_file",
                    "arguments": {"path": "PRIVATE_PROVIDER_SENTINEL"},
                }
            ],
        },
    }
    corrected_submission = {
        "plan": None,
        "action": {
            "kind": "submit_final_reply",
            "reply": "已停止继续调用工具，并如实提交当前部分结果。",
        },
    }
    prepared_calls: list[dict[str, object]] = []
    provider = _prepared_structured_sequence(
        [first_tool_call, rejected_final_tool_call, corrected_submission],
        prepared_calls,
    )
    monkeypatch.setattr("personagraph.runtime.l1.model.complete_structured", provider)
    result = _run_l1(
        session_id=session_id,
        user_text=user_text,
        request_id="l1-prepared-finalization-repair",
        features={"l1_max_attempts": 2, "l1_semantic_verification_mode": "off"},
    )
    assert result.status == "completed"
    assert result.reply == "已停止继续调用工具，并如实提交当前部分结果。"
    assert len(prepared_calls) == 3
    messages = prepared_calls[2]["repair_messages"]
    assert isinstance(messages, list)
    assert [item["role"] for item in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    rejected_payload = json.loads(messages[2]["content"])
    assert rejected_payload["action"] == rejected_final_tool_call["action"]
    assert isinstance(rejected_payload["note"], str)
    assert "execution_notes" not in rejected_payload
    repair_envelope = json.loads(messages[3]["content"].split("Host 修复清单：", 1)[1])
    assert set(repair_envelope) == {"current_issues"}
    assert repair_envelope["current_issues"][0]["paths"] == ["/action/kind"]
    explanation = repair_envelope["current_issues"][0]["safe_explanation"]
    assert "execution_limits.finalization_required=true" in explanation
    assert "submit_final_reply" in explanation
    assert "PRIVATE_PROVIDER_SENTINEL" not in json.dumps(
        repair_envelope, ensure_ascii=False
    )
    execution = session_store.get_l1_turn_execution(
        session_id=session_id, turn_id=result.turn_id,
    )
    logical = session_store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=execution["attempts"][1]["logical_model_call_id"],
    )
    feedback = logical.physical_attempts[0].settlement.next_output_repair_feedback
    assert feedback.current_issues[0].code == "host_guard.l1_finalization_required"
    assert feedback.issue_coverage.value == "first_only"
    assert logical.physical_attempts[1].request.output_repair_feedback == feedback


def test_l1_prepared_semantic_host_guard_repair_has_exact_paths(
    monkeypatch, tmp_path: Path
) -> None:
    session_id = _create_l1_session(tmp_path)
    user_text = "请确认语义审查边界。"
    _install_l1_classifier(monkeypatch)
    monkeypatch.setattr(
        "personagraph.runtime.l1.model.complete_structured",
        lambda *_args, **kwargs: _model_result(
            {
                "plan": {
                    "objective": "确认边界",
                    "acceptances": [{"criterion": "确认语义审查边界"}],
                },
                "action": {
                    "kind": "submit_final_reply",
                    "reply": "已确认语义审查边界。",
                },
            },
            kwargs["model_call_id"],
        ),
    )
    invalid_host_binding = {
        "issues": [
            {"message": "引用了当前输入不存在的身份。", "acceptance_id": "wrong"}
        ]
    }
    passing = {"issues": []}
    prepared_calls: list[dict[str, object]] = []
    provider = _prepared_structured_sequence(
        [invalid_host_binding, passing], prepared_calls
    )
    monkeypatch.setattr(
        "personagraph.runtime.l1.semantic_verification.complete_structured", provider
    )
    result = _run_l1(
        session_id=session_id,
        user_text=user_text,
        request_id="l1-prepared-semantic-guard-repair",
        features={"l1_semantic_verification_mode": "always"},
    )
    assert result.status == "completed"
    assert len(prepared_calls) == 2
    first, repair = prepared_calls
    assert "repair_messages" not in first
    assert first["system_prompt"] == repair["system_prompt"]
    assert first["user_content"] == repair["user_content"]
    messages = repair["repair_messages"]
    assert isinstance(messages, list)
    assert (
        messages[2]["content"] == _model_result(invalid_host_binding, "ignored").reply
    )
    assert "不得改写候选" in messages[0]["content"]
    repair_envelope = json.loads(messages[3]["content"].split("Host 修复清单：", 1)[1])
    assert set(repair_envelope) == {"current_issues"}
    assert {tuple(item["paths"]) for item in repair_envelope["current_issues"]} == {
        ("/issues/0/acceptance_id",),
    }


def test_l1_semantic_revise_is_a_business_result_not_a_format_repair(
    monkeypatch, tmp_path: Path
) -> None:
    session_id = _create_l1_session(tmp_path)
    user_text = "请先给出候选回答，再按审查意见修订。"
    _install_l1_classifier(monkeypatch)
    decision_calls = 0

    def decide(_system: str, _user: str, **kwargs: object) -> ModelResult:
        nonlocal decision_calls
        decision_calls += 1
        return _model_result(
            {
                "plan": {
                    "objective": "给出并修订回答",
                    "acceptances": [{"criterion": "给出候选回答并按意见修订"}],
                }
                if decision_calls == 1
                else None,
                "action": {
                    "kind": "submit_final_reply",
                    "reply": "初稿。" if decision_calls == 1 else "修订稿。",
                },
            },
            kwargs["model_call_id"],
        )

    monkeypatch.setattr("personagraph.runtime.l1.model.complete_structured", decide)
    semantic_calls: list[dict[str, object]] = []
    semantic_provider = _prepared_structured_sequence(
        [{"issues": [{"message": "候选回答仍需按审查意见修订。"}]}, {"issues": []}],
        semantic_calls,
    )
    monkeypatch.setattr(
        "personagraph.runtime.l1.semantic_verification.complete_structured",
        semantic_provider,
    )
    result = _run_l1(
        session_id=session_id,
        user_text=user_text,
        request_id="l1-semantic-revise-business-result",
        features={"l1_semantic_verification_mode": "always"},
    )
    assert result.status == "completed"
    assert result.reply == "修订稿。"
    assert decision_calls == 2
    assert len(semantic_calls) == 2
    assert all(("repair_messages" not in item for item in semantic_calls))


def test_l1_repairs_an_invalid_revision_inside_the_same_attempt(
    monkeypatch, tmp_path: Path
) -> None:
    session_id = _create_l1_session(tmp_path)
    user_text = "查询今天日期并解释结果。"
    _install_l1_classifier(monkeypatch)
    prompts: list[str] = []
    repairs: list[list[dict[str, str]]] = []

    def decide(system: str, user_content: str, **kwargs: object) -> ModelResult:
        prompts.append(system)
        repair_messages = kwargs.get("repair_messages")
        if repair_messages is not None:
            assert isinstance(repair_messages, list)
            repairs.append(repair_messages)
        payload = json.loads(user_content)
        if len(prompts) == 1:
            decision: dict[str, object] = {
                "plan": {
                    "objective": "查询日期",
                    "acceptances": [{"criterion": "查询今天日期"}],
                },
                "action": {
                    "kind": "call_tools",
                    "calls": [{"tool_id": "get_today", "arguments": {}}],
                },
            }
        else:
            tool_result = payload["prior_tool_results"][-1]
            if len(prompts) == 2:
                acceptances = [{"criterion": "解释日期结果"}]
            else:
                acceptances = [
                    payload["plan"]["acceptances"][0],
                    {"criterion": "解释日期结果"},
                ]
            decision = {
                "plan": {"objective": "查询并解释日期", "acceptances": acceptances},
                "action": {
                    "kind": "submit_final_reply",
                    "reply": f"今天是 {tool_result['result']['date']}，这是当前日期。",
                },
            }
        return _model_result(decision, kwargs["model_call_id"])

    monkeypatch.setattr("personagraph.runtime.l1.model.complete_structured", decide)
    monkeypatch.setattr(
        "personagraph.runtime.model_calls.requests._sleep", lambda _: None
    )
    result = _run_l1(
        session_id=session_id, user_text=user_text, request_id="l1-plan-revision-repair"
    )
    assert result.status == "completed"
    assert len(prompts) == 3
    assert prompts == [prompts[0]] * 3
    assert len(repairs) == 1
    repair_envelope = json.loads(
        repairs[0][3]["content"].split("Host 修复清单：", 1)[1]
    )
    assert repair_envelope["current_issues"][0]["paths"] == ["/plan/acceptances"]
    assert "不能删除" in repair_envelope["current_issues"][0]["safe_explanation"]
    execution = session_store.get_l1_turn_execution(
        session_id=session_id, turn_id=result.turn_id
    )
    assert execution is not None
    assert execution["state"]["attempts_started"] == 2
    assert [item["revision"] for item in execution["plan_revisions"]] == [1, 2]
    first_plan = json.loads(execution["plan_revisions"][0]["plan_json"])
    second_plan = json.loads(execution["plan_revisions"][1]["plan_json"])
    assert first_plan["schema_version"] == "l1-plan-v3"
    assert second_plan["schema_version"] == "l1-plan-v3"
    assert (
        first_plan["acceptances"][0]["acceptance_id"]
        == second_plan["acceptances"][0]["acceptance_id"]
    )


def test_l1_scope_and_verification_accepts_a_repeated_current_plan(
    monkeypatch, tmp_path, bound_partitioned_session
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (tmp_path / "outside.txt").write_text(
        "TOP SECRET OUTSIDE CONTENT", encoding="utf-8"
    )
    session_id = bound_partitioned_session(working_dir=workspace)
    user_text = "读取工作目录外的 outside.txt；如果不允许请如实说明。"
    _install_l1_classifier(monkeypatch)
    prompts: list[str] = []
    payloads: list[dict[str, object]] = []
    reviewer_candidates: list[str] = []
    current_plan_proposal: dict[str, object] = {
        "objective": "尝试读取指定文件并如实交付",
        "acceptances": [{"criterion": "读取 outside.txt 或说明权限限制"}],
    }

    def decide(system: str, user_content: str, **kwargs: object) -> ModelResult:
        prompts.append(system)
        payload = json.loads(user_content)
        payloads.append(payload)
        if len(prompts) == 1:
            decision: dict[str, object] = {
                "plan": current_plan_proposal,
                "action": {
                    "kind": "call_tools",
                    "calls": [
                        {
                            "tool_id": "inspect_file",
                            "arguments": {"path": "../outside.txt"},
                        }
                    ],
                },
            }
        else:
            if len(prompts) >= 3:
                assert payload["verification_feedback"]["source"] == "semantic"
                assert payload["prior_tool_results"] == []
            else:
                tool_result = payload["prior_tool_results"][0]
                assert tool_result["status"] == "failed"
            if len(prompts) == 2:
                reply = "文件已经读取。"
            else:
                reply = "授权工作目录之外的 outside.txt 不允许读取。"
            decision = {
                "plan": payload["plan"] if len(prompts) >= 3 else None,
                "action": {"kind": "submit_final_reply", "reply": reply},
            }
        return _model_result(decision, kwargs["model_call_id"])

    def review(_system: str, user_content: str, **kwargs: object) -> ModelResult:
        payload = json.loads(user_content)
        reply = payload["candidate_final_reply"]
        reviewer_candidates.append(reply)
        if reply == "文件已经读取。":
            assert payload["execution_context"]["tool_calls"][0]["status"] == "failed"
            result: dict[str, object] = {
                "issues": [{"message": "候选回答声称读取成功，但工具观察为失败。"}]
            }
        else:
            result = {"issues": []}
        return _model_result(result, kwargs["model_call_id"])

    monkeypatch.setattr("personagraph.runtime.l1.model.complete_structured", decide)
    monkeypatch.setattr(
        "personagraph.runtime.l1.semantic_verification.complete_structured", review
    )
    monkeypatch.setattr(
        "personagraph.runtime.model_calls.requests._sleep", lambda _: None
    )
    result = _run_l1(
        session_id=session_id, user_text=user_text, request_id="l1-verification-repair"
    )
    assert len(prompts) == 3
    assert result.status == "completed"
    assert result.reply == "授权工作目录之外的 outside.txt 不允许读取。"
    assert prompts[0] == prompts[1] == prompts[2]
    assert payloads[2]["verification_feedback"]["source"] == "semantic"
    assert reviewer_candidates == [
        "文件已经读取。",
        "授权工作目录之外的 outside.txt 不允许读取。",
    ]
    execution = session_store.get_l1_turn_execution(
        session_id=session_id, turn_id=result.turn_id
    )
    assert execution is not None
    assert [item["revision"] for item in execution["plan_revisions"]] == [1]
    final_decision = json.loads(execution["attempts"][-1]["decision_json"])
    assert final_decision["plan"] is None
    final_logical_call = session_store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=execution["attempts"][-1]["logical_model_call_id"],
    )
    assert final_logical_call is not None
    assert len(final_logical_call.physical_attempts) == 1
    assert execution["tool_calls"][0]["status"] == "failed"
    assert "TOP SECRET OUTSIDE CONTENT" not in str(
        execution["tool_calls"][0]["outcome_json"]
    )
    assert execution["attempts"][-1]["status"] == "closed"
    assert execution["attempts"][-1]["action_kind"] == "submit_final_reply"
    assert (
        session_store.get_committed_turn_pair(session_id, f"commit_{result.turn_id}")
        is not None
    )


def test_l1_store_rejects_a_mismatched_tool_settlement(
    monkeypatch, tmp_path: Path
) -> None:
    session_id = _create_l1_session(tmp_path)
    user_text = "查询今天日期。"
    _install_l1_classifier(monkeypatch)
    monkeypatch.setattr(
        "personagraph.runtime.l1.model.complete_structured",
        lambda *_args, **kwargs: _model_result(
            {
                "plan": {
                    "objective": "查询日期",
                    "acceptances": [{"criterion": "查询今天日期"}],
                },
                "action": {
                    "kind": "call_tools",
                    "calls": [{"tool_id": "get_today", "arguments": {}}],
                },
            },
            kwargs["model_call_id"],
        ),
    )
    settle = session_store.settle_l1_tool_call

    def corrupt_settlement(**kwargs: object):
        assert json.loads(str(kwargs["outcome_json"]))["status"] == "succeeded"
        return settle(**{**kwargs, "outcome_status": "failed"})

    monkeypatch.setattr(session_store, "settle_l1_tool_call", corrupt_settlement)
    result = _run_l1(
        session_id=session_id,
        user_text=user_text,
        request_id="l1-mismatched-tool-settlement",
    )
    assert result.status == "incomplete"
    assert result.error_code == "INTERNAL_FAILURE"
    execution = session_store.get_l1_turn_execution(
        session_id=session_id, turn_id=result.turn_id
    )
    assert execution is not None
    assert execution["run"]["status"] == "failed"
    assert execution["tool_calls"][0]["status"] == "pending"
    assert execution["tool_calls"][0]["outcome_json"] is None
    assert (
        session_store.get_committed_turn_pair(session_id, f"commit_{result.turn_id}")
        is None
    )
