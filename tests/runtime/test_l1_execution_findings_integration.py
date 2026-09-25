from __future__ import annotations

import json
from pathlib import Path

from personagraph.model_io.gateway import ModelResult
from personagraph.runtime import entry
from personagraph.runtime.entry.ingress import model as ingress_model
from personagraph.runtime.l1.execution_notes import record_committed_execution_notes
from personagraph.runtime.l1.tool_context import project_recent_tool_results
from personagraph.persistent_turn_content.findings import (
    ExecutionFindingStatus,
    ExecutionFindingsOwnerKind,
)
from personagraph.runtime.entry.routing.policy import (
    TurnRoutingPolicy,
    freeze_turn_routing_policy,
)
from personagraph.session import store as session_store
from tests.helpers.evidence_submission import add_empty_support_justifications
from tests.helpers.prepared_model_provider import as_prepared_test_provider


def test_l1_previous_document_result_keeps_all_chunk_bodies() -> None:
    tool_result_batch = {
        "schema_version": "l1-attempt-tool-results-v1",
        "attempt_id": "attempt-read",
        "tool_results": [
            {
                "tool_call_id": "call-read",
                "call_ordinal": 1,
                "tool_id": "read_document_chunks",
                "status": "succeeded",
                "result_sha256": "c" * 64,
                "result": {
                    "file_id": "file-paper",
                    "file_version_id": "version-1",
                    "chunks": [
                        {
                            "chunk_id": "chunk-0",
                            "sequence": 0,
                            "content": "already summarized",
                        },
                        {
                            "chunk_id": "chunk-1",
                            "sequence": 1,
                            "content": "must remain visible",
                        },
                    ],
                },
            }
        ],
    }
    prior_tool_results, projection = project_recent_tool_results(
        {
            "attempts": [
                {
                    "attempt_id": "attempt-read",
                    "ordinal": 1,
                    "tool_results_json": json.dumps(tool_result_batch),
                }
            ]
        }
    )

    projected_result = prior_tool_results[0]
    assert projected_result["result"] is not None
    assert projected_result["result"]["chunks"][0]["content"] == "already summarized"
    assert projected_result["result"]["chunks"][1]["content"] == "must remain visible"
    assert "result_partially_compacted" not in projected_result
    assert projection == {
        "policy": "previous_attempt_only",
        "source_attempt_id": "attempt-read",
        "durable_tool_result_count": 1,
        "projected_tool_result_count": 1,
        "omitted_tool_result_count": 0,
        "compacted_tool_result_count": 0,
        "omitted_audit_metadata_field_count": 0,
    }


def test_l1_findings_receipt_is_bounded_and_older_source_is_omitted() -> None:
    old_claim = "this evicted finding must not re-enter the model view"
    source_batch = {
        "schema_version": "l1-attempt-tool-results-v1",
        "attempt_id": "attempt-source",
        "tool_results": [
            {
                "tool_call_id": "call-source",
                "call_ordinal": 1,
                "tool_id": "source-tool",
                "status": "succeeded",
                "result_sha256": "a" * 64,
                "result": {"body": "source bytes already summarized"},
            }
        ],
    }
    findings_batch = {
        "schema_version": "l1-attempt-tool-results-v1",
        "attempt_id": "attempt-findings",
        "tool_results": [
            {
                "tool_call_id": "call-findings",
                "call_ordinal": 1,
                "tool_id": "record_execution_findings",
                "status": "succeeded",
                "result_sha256": "b" * 64,
                "result": {
                    "schema_version": "execution-findings-tool-result-v1",
                    "status": "applied",
                    "ledger_revision": 1,
                    "affected_entry_ids": ["finding-1"],
                    "active_projection": {
                        "projection_sha256": "c" * 64,
                        "active_entries": [{"claim": old_claim}],
                    },
                    "replayed": False,
                },
            },
        ],
    }

    prior, projection = project_recent_tool_results(
        {
            "attempts": [
                {
                    "attempt_id": "attempt-source",
                    "ordinal": 1,
                    "tool_results_json": json.dumps(source_batch),
                },
                {
                    "attempt_id": "attempt-findings",
                    "ordinal": 2,
                    "tool_results_json": json.dumps(findings_batch),
                },
            ]
        }
    )

    assert [item["tool_call_id"] for item in prior] == ["call-findings"]
    compacted_projection = prior[0]["result"]["active_projection"]
    assert compacted_projection == {
        "schema_version": "execution-findings-active-projection-reference-v1",
        "compacted": True,
        "projection_sha256": "c" * 64,
    }
    assert old_claim not in json.dumps(prior)
    assert projection == {
        "policy": "previous_attempt_only",
        "source_attempt_id": "attempt-findings",
        "durable_tool_result_count": 2,
        "projected_tool_result_count": 1,
        "omitted_tool_result_count": 1,
        "compacted_tool_result_count": 1,
        "omitted_audit_metadata_field_count": 0,
    }


def test_l1_non_tool_previous_attempt_does_not_resurrect_older_results() -> None:
    source_batch = {
        "schema_version": "l1-attempt-tool-results-v1",
        "attempt_id": "attempt-source",
        "tool_results": [
            {
                "tool_call_id": "call-source",
                "call_ordinal": 1,
                "tool_id": "source-tool",
                "status": "succeeded",
                "result_sha256": "a" * 64,
                "result": {"body": "older source body"},
            }
        ],
    }

    prior, projection = project_recent_tool_results(
        {
            "attempts": [
                {
                    "attempt_id": "attempt-source",
                    "ordinal": 1,
                    "tool_results_json": json.dumps(source_batch),
                },
                {
                    "attempt_id": "attempt-submit",
                    "ordinal": 2,
                    "action_kind": "submit_final_reply",
                    "tool_results_json": None,
                },
            ]
        }
    )

    assert prior == []
    assert projection == {
        "policy": "previous_attempt_only",
        "source_attempt_id": "attempt-submit",
        "durable_tool_result_count": 1,
        "projected_tool_result_count": 0,
        "omitted_tool_result_count": 1,
        "compacted_tool_result_count": 0,
        "omitted_audit_metadata_field_count": 0,
    }


def test_l1_findings_history_suppresses_malformed_unbounded_output() -> None:
    leaked_claim = "a malformed legacy claim must not enter the model view"
    batch = {
        "schema_version": "l1-attempt-tool-results-v1",
        "attempt_id": "attempt-findings",
        "tool_results": [
            {
                "tool_call_id": "call-findings",
                "call_ordinal": 1,
                "tool_id": "record_execution_findings",
                "status": "succeeded",
                "result_sha256": "b" * 64,
                "result": {"claim": leaked_claim},
            }
        ],
    }

    prior, projection = project_recent_tool_results(
        {
            "attempts": [
                {
                    "attempt_id": "attempt-findings",
                    "ordinal": 1,
                    "tool_results_json": json.dumps(batch),
                }
            ]
        }
    )

    assert prior[0]["result"] == {
        "schema_version": "execution-findings-tool-result-reference-v1",
        "compacted": True,
        "active_projection": {
            "schema_version": ("execution-findings-active-projection-reference-v1"),
            "compacted": True,
        },
    }
    assert leaked_claim not in json.dumps(prior)
    assert projection == {
        "policy": "previous_attempt_only",
        "source_attempt_id": "attempt-findings",
        "durable_tool_result_count": 1,
        "projected_tool_result_count": 1,
        "omitted_tool_result_count": 0,
        "compacted_tool_result_count": 1,
        "omitted_audit_metadata_field_count": 0,
    }


def test_l1_records_and_reinjects_execution_findings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = session_store.create_session(
        "Entelecheia",
        working_dir=str(workspace),
    )
    user_text = "查询今天日期，记录发现后再回答。"
    policy = freeze_turn_routing_policy(
        TurnRoutingPolicy(l1_enabled=True, l2_enabled=False),
        source="request_override",
    )

    monkeypatch.setattr(
        ingress_model,
        "complete_structured",
        as_prepared_test_provider(
            lambda *_args, **kwargs: _model_result(
                {"processing_level": "L1", "task_matches": []},
                kwargs["model_call_id"],
            ),
        ),
    )
    model_payloads: list[dict[str, object]] = []
    source_result_ref = ""

    def decide(_system: str, user_content: str, **kwargs: object) -> ModelResult:
        nonlocal source_result_ref
        payload = json.loads(user_content)
        model_payloads.append(payload)
        if len(model_payloads) == 1:
            findings_schema = next(
                item["input_schema"] for item in payload["tool_catalog"]
                if item["tool_id"] == "record_execution_findings"
            )
            assert set(findings_schema["$defs"]["sourceRef"]["properties"]) == {
                "call_ref", "chunk_id",
            }
            decision: dict[str, object] = {
                "plan": {
                    "objective": "查询日期并持久化本轮发现",
                    "acceptances": [
                        {
                            "criterion": "查询并报告今天日期",
                        }
                    ],
                },
                "action": {
                    "kind": "call_tools",
                    "calls": [
                        {
                            "tool_id": "get_today",
                            "arguments": {},
                        }
                    ],
                },
            }
        elif len(model_payloads) == 2:
            result = payload["prior_tool_results"][0]
            source_result_ref = result["call_ref"]
            assert "tool_call_id" not in result and "result_sha256" not in result
            findings = payload["execution_findings"]
            assert "ledger_revision" not in findings
            assert findings["active_entry_count"] == 1
            assert len(findings["notes"]) == 1
            assert findings["notes"][0]["entry_id"]
            decision = {
                "plan": None,
                "action": {
                    "kind": "call_tools",
                    "calls": [
                        {
                            "tool_id": "record_execution_findings",
                            "arguments": {
                                "items": [
                                    {
                                        "kind": "finding",
                                        "claim": "The date tool returned a current date.",
                                        "source_refs": [
                                            {"call_ref": source_result_ref}
                                        ],
                                    }
                                ],
                            },
                        }
                    ],
                },
            }
        else:
            findings = payload["execution_findings"]
            assert "ledger_revision" not in findings
            assert findings["active_entry_count"] == 3
            assert (
                sum(
                    item["summary"] == "The date tool returned a current date."
                    for item in findings["notes"]
                )
                == 1
            )
            explicit_note = next(
                item for item in findings["notes"]
                if item["summary"] == "The date tool returned a current date."
            )
            assert explicit_note["source_refs"] == [{"call_ref": source_result_ref}]
            prior_tool_results = payload["prior_tool_results"]
            assert len(prior_tool_results) == 1
            findings_receipt = prior_tool_results[0]
            assert findings_receipt["tool_id"] == "record_execution_findings"
            assert findings_receipt["call_ref"] != source_result_ref
            assert findings_receipt["result"]["active_projection"]["compacted"] is True
            assert "The date tool returned a current date." not in json.dumps(
                findings_receipt["result"]
            )
            assert findings_receipt["arguments"]["items"][0]["claim"] == (
                "The date tool returned a current date."
            )
            assert findings_receipt["arguments"]["items"][0]["source_refs"] == [
                {"call_ref": source_result_ref},
            ]
            assert payload["tool_result_projection"] == {
                "durable_tool_result_count": 2,
                "projected_tool_result_count": 1,
                "omitted_tool_result_count": 1,
            }
            decision = {
                "plan": None,
                "references": [{"call_ref": source_result_ref}],
                "action": {
                    "kind": "submit_final_reply",
                    "reply": "日期已查询，且本轮发现已记录。",
                },
            }
        return _model_result(decision, kwargs["model_call_id"])

    monkeypatch.setattr(
        "personagraph.runtime.l1.model.complete_structured",
        as_prepared_test_provider(decide, add_l1_notes=True),
    )

    semantic_payloads: list[dict[str, object]] = []

    def review(
        _system: str,
        user_content: str,
        **kwargs: object,
    ) -> ModelResult:
        payload = json.loads(user_content)
        semantic_payloads.append(payload)
        supporting_results = payload["durable_evidence"]["results"]
        assert len(supporting_results) == 1
        assert supporting_results[0]["call_ref"] == source_result_ref
        assert "tool_call_id" not in supporting_results[0]
        assert supporting_results[0]["result"] is not None
        return _model_result(
            {"issues": []},
            kwargs["model_call_id"],
        )

    monkeypatch.setattr(
        "personagraph.runtime.l1.semantic_verification.complete_structured",
        as_prepared_test_provider(review),
    )
    reserve = session_store.reserve_l1_tool_call
    ledger_revisions_before_tools = []

    def reserve_after_notes(**kwargs):
        snapshot = session_store.get_execution_findings_ledger_for_owner(
            owner_kind=ExecutionFindingsOwnerKind.L1_TURN_RUN,
            execution_owner_id=kwargs["l1_turn_run_id"],
        )
        ledger_revisions_before_tools.append(
            (kwargs["tool_id"], snapshot.ledger.revision)
        )
        if kwargs["tool_id"] == "get_today":
            execution = session_store.get_l1_turn_execution(
                session_id=kwargs["session_id"],
                turn_id=kwargs["turn_id"],
            )
            attempt = next(
                item
                for item in execution["attempts"]
                if item["attempt_id"] == kwargs["attempt_id"]
            )
            for _ in range(2):
                record_committed_execution_notes(
                    store=session_store,
                    ledger_id=snapshot.ledger.ledger_id,
                    attempt=attempt,
                    execution=execution,
                )
            replayed = session_store.get_execution_findings_ledger_for_owner(
                owner_kind=ExecutionFindingsOwnerKind.L1_TURN_RUN,
                execution_owner_id=kwargs["l1_turn_run_id"],
            )
            assert replayed.ledger.revision == snapshot.ledger.revision
            assert replayed.ledger.entry_revisions == snapshot.ledger.entry_revisions
        return reserve(**kwargs)

    monkeypatch.setattr(session_store, "reserve_l1_tool_call", reserve_after_notes)
    result = entry.run_entry_turn(
        user_input=user_text,
        features={
            "context_guard_limit": 24_000,
            "l1_max_attempts": 12,
            "l1_max_tool_calls_per_attempt": 8,
        },
        session_id=session_id,
        client_request_id="l1-findings-integration",
        routing_policy=policy,
        store=session_store,
    )

    assert result.status == "completed", (
        result.error_code,
        len(model_payloads),
        {
            "findings": model_payloads[-1].get("execution_findings"),
            "prior": model_payloads[-1].get("prior_tool_results"),
            "projection": model_payloads[-1].get("tool_result_projection"),
        }
        if model_payloads
        else None,
    )
    assert len(semantic_payloads) == 1
    execution = session_store.get_l1_turn_execution(
        session_id=session_id,
        turn_id=result.turn_id,
    )
    assert execution is not None
    run_id = str(execution["run"]["l1_turn_run_id"])
    ledger = session_store.get_execution_findings_ledger_for_owner(
        owner_kind=ExecutionFindingsOwnerKind.L1_TURN_RUN,
        execution_owner_id=run_id,
    )
    assert ledger is not None
    assert ledger.ledger.status.value == "closed"
    assert ledger.ledger.revision == 4
    assert ledger_revisions_before_tools == [
        ("get_today", 1),
        ("record_execution_findings", 2),
    ]
    assert ledger.ledger.entry_revisions[0].status is (ExecutionFindingStatus.ACTIVE)
    calls_by_tool = {str(item["tool_id"]): item for item in execution["tool_calls"]}
    assert set(calls_by_tool) == {"get_today", "record_execution_findings"}
    assert (
        calls_by_tool["record_execution_findings"]["execution_class"] == "runtime_state"
    )
    # 模型只提交短坐标；Host 保存真实调用、结果摘要和 CAS。
    recorded_arguments = json.loads(
        calls_by_tool["record_execution_findings"]["arguments_json"]
    )
    assert recorded_arguments["expected_ledger_revision"] == 2
    source = recorded_arguments["items"][0]["source_refs"][0]
    assert source == {
        "tool_result_id": calls_by_tool["get_today"]["tool_call_id"],
        "result_sha256": calls_by_tool["get_today"]["outcome_hash"],
        "chunk_id": None,
    }


def _model_result(payload: dict[str, object], model_call_id: object) -> ModelResult:
    return ModelResult(
        reply=json.dumps(
            add_empty_support_justifications(payload),
            ensure_ascii=False,
        ),
        provider="mock",
        model="mock-structured",
        latency_ms=1,
        model_call_id=str(model_call_id),
    )


def test_prepared_provider_notes_are_explicit_and_preserve_deliberate_invalid_outputs():
    plan = {"acceptances": [{"acceptance_id": "answer"}]}
    decision = {"plan": None, "action": {"kind": "submit_final_reply", "reply": "ok"}}

    def provider(*_args, **kwargs):
        return _model_result(decision, kwargs["model_call_id"])

    plain = as_prepared_test_provider(provider)
    request = json.dumps({"plan": plan})
    result = plain.prepare("system", request, purpose="runtime_l1_attempt").dispatch(
        model_call_id="plain"
    )
    assert "note" not in json.loads(result.reply)
    enriched = as_prepared_test_provider(provider, add_l1_notes=True)
    result = enriched.prepare("system", request, purpose="runtime_l1_attempt").dispatch(
        model_call_id="with-notes"
    )
    assert json.loads(result.reply)["note"]
    decision["note"] = ""
    invalid = enriched.prepare(
        "system", request, purpose="runtime_l1_attempt"
    ).dispatch(model_call_id="invalid")
    assert json.loads(invalid.reply)["note"] == ""
