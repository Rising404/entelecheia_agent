"""可执行的 L1 模型/工具/观察/最终交付垂直切片。"""

from __future__ import annotations

from functools import partial

import json
import sqlite3
from types import SimpleNamespace

import pytest

from personagraph.context_budget import ContextBudgetExceeded
from personagraph.model_io.gateway import ModelResult
from personagraph.model_io.tier_bindings import (
    EndpointOrigin,
    ModelTierBinding,
    ModelTier,
)
from personagraph.runtime import entry
from personagraph.runtime.entry import application as entry_application
from personagraph.runtime.entry.ingress import model as ingress_model
from personagraph.runtime.l1 import controller as l1_controller
from personagraph.runtime.l1.controller import run_l1_turn
from personagraph.runtime.l1 import execution_config as l1_execution_config
from personagraph.runtime.l1 import tool_runtime as l1_tool_runtime
from personagraph.runtime.turn.contracts import AcceptedEntryTurn
from personagraph.runtime.entry.routing.policy import (
    TurnRoutingPolicy,
    freeze_turn_routing_policy,
)
from personagraph.runtime.turn_deadline import TurnDeadline
from personagraph.session import store as session_store
from personagraph.session.attachments.application import accept_upload
from personagraph.tools.catalog import CatalogConflictError
from personagraph.tools.composition import default_catalog
from personagraph.tools.workspace import session_read_source
from tests.helpers.evidence_submission import add_empty_support_justifications
from tests.helpers.prepared_model_provider import (
    as_prepared_test_provider as _as_prepared_test_provider,
    repair_feedback_from_provider_kwargs,
)

# 本文件的脚本响应关注执行/恢复；显式升级为当前的必填公开笔记协议。
as_prepared_test_provider = partial(_as_prepared_test_provider, add_l1_notes=True)


def _create_l1_session(tmp_path) -> str:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return session_store.create_session(
        "Entelecheia",
        working_dir=str(workspace),
    )


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


def _resume_l1(
    *,
    accepted: AcceptedEntryTurn,
    features: dict[str, object],
):
    execution = session_store.get_l1_turn_execution(
        session_id=accepted.session_id,
        turn_id=accepted.turn_id,
    )
    assert execution is not None
    inspection = session_store.inspect_turn_execution(accepted.session_id)
    context = entry_application.build_entry_context(
        accepted=accepted,
        features=features,
        store=session_store,
        deadline=TurnDeadline.starting_now(300),
    )
    return run_l1_turn(
        accepted=accepted,
        context=context,
        l1_turn_run_id=execution["run"]["l1_turn_run_id"],
        initial_turn_window_revision=int(inspection["window"]["state_version"]),
        deadline=TurnDeadline.starting_now(300),
        emit=lambda _event: None,
        store=session_store,
        lease_owner=entry_application._runtime_entry_lease_owner(),
    )


@pytest.mark.parametrize(
    "failure_type",
    [
        OSError,
        sqlite3.OperationalError,
        session_store.SessionCatalogError,
        session_store.SessionStoreError,
    ],
)
def test_l1_workspace_materialization_normalizes_expected_runtime_failures(
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[Exception],
) -> None:
    def fail_workspace(*_args: object, **_kwargs: object) -> None:
        raise failure_type("injected workspace failure")

    monkeypatch.setattr(
        session_read_source,
        "build_session_workspace_readonly_runtime",
        fail_workspace,
    )

    with pytest.raises(
        l1_tool_runtime.L1WorkspaceUnavailableError,
        match="fixed workspace could not be materialized",
    ) as exc_info:
        l1_tool_runtime.build_l1_tool_runtime("session-workspace-failure")

    assert isinstance(exc_info.value.__cause__, failure_type)


def test_l1_workspace_materialization_normalizes_missing_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_workspace(*_args: object, **_kwargs: object) -> None:
        raise ValueError("session does not exist")

    monkeypatch.setattr(
        session_read_source,
        "build_session_workspace_readonly_runtime",
        fail_workspace,
    )

    with pytest.raises(l1_tool_runtime.L1WorkspaceUnavailableError):
        l1_tool_runtime.build_l1_tool_runtime("missing-session")


@pytest.mark.parametrize("failure_type", [TypeError, AssertionError, ValueError])
def test_l1_workspace_materialization_does_not_hide_programming_errors(
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[Exception],
) -> None:
    def fail_workspace(*_args: object, **_kwargs: object) -> None:
        raise failure_type("injected programming error")

    monkeypatch.setattr(
        session_read_source,
        "build_session_workspace_readonly_runtime",
        fail_workspace,
    )

    with pytest.raises(failure_type, match="injected programming error"):
        l1_tool_runtime.build_l1_tool_runtime("session-programming-error")


def test_l1_workspace_failure_stops_before_l1_model_io(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    session_id = _create_l1_session(tmp_path)
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

    def fail_workspace(*_args: object, **_kwargs: object) -> None:
        raise sqlite3.OperationalError("workspace database unavailable")

    monkeypatch.setattr(
        session_read_source,
        "build_session_workspace_readonly_runtime",
        fail_workspace,
    )
    l1_model_calls = 0

    def count_l1_model_call(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal l1_model_calls
        l1_model_calls += 1
        raise AssertionError("workspace failure must stop before L1 model I/O")

    monkeypatch.setattr(
        "personagraph.runtime.l1.model.complete_structured",
        as_prepared_test_provider(count_l1_model_call),
    )

    result = entry.run_entry_turn(
        user_input="读取工作区中的文档。",
        features={"context_guard_limit": 24_000},
        session_id=session_id,
        client_request_id="l1-workspace-runtime-failure",
        routing_policy=policy,
        store=session_store,
    )

    assert result.status == "incomplete"
    assert result.processing_level == "L1"
    assert result.end_reason == "l1_workspace_unavailable"
    assert result.error_code == "L1_RUNTIME_NOT_READY"
    assert l1_model_calls == 0


def test_l1_catalog_normalizes_expected_control_plane_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    session_id = _create_l1_session(tmp_path)

    def fail_catalog(*_args: object, **_kwargs: object) -> None:
        raise CatalogConflictError("injected catalog conflict")

    monkeypatch.setattr(
        default_catalog,
        "bootstrap_production_default_catalog",
        fail_catalog,
    )

    with pytest.raises(l1_tool_runtime.L1ToolCatalogUnavailableError) as exc_info:
        l1_tool_runtime.build_l1_tool_runtime(session_id)

    assert isinstance(exc_info.value.__cause__, CatalogConflictError)


def test_l1_catalog_does_not_hide_programming_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    session_id = _create_l1_session(tmp_path)

    def fail_catalog(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("injected catalog programming error")

    monkeypatch.setattr(
        default_catalog,
        "bootstrap_production_default_catalog",
        fail_catalog,
    )

    with pytest.raises(AssertionError, match="injected catalog programming error"):
        l1_tool_runtime.build_l1_tool_runtime(session_id)


def test_l1_provider_budget_failure_keeps_its_durable_failure_code(
    monkeypatch,
) -> None:
    failure_codes: list[str] = []

    class Store:
        def fail_l1_turn_run(self, **kwargs: object) -> None:
            failure_codes.append(str(kwargs["failure_code"]))

    def reject(**_kwargs: object):
        raise ContextBudgetExceeded(limit=1000, estimated_tokens=1001)

    monkeypatch.setattr(l1_controller, "_run_l1_turn_impl", reject)

    with pytest.raises(ContextBudgetExceeded):
        l1_controller.run_l1_turn(
            accepted=SimpleNamespace(
                session_id="session-context-budget",
                turn_id="turn-context-budget",
            ),
            context=object(),
            l1_turn_run_id="l1run-context-budget",
            initial_turn_window_revision=1,
            deadline=TurnDeadline.starting_now(60),
            emit=lambda _event: None,
            store=Store(),
            lease_owner=None,
        )

    assert failure_codes == ["CONTEXT_BUDGET_EXCEEDED"]


def test_l1_executes_a_durable_tool_loop_and_commits_one_formal_reply(
    monkeypatch,
    tmp_path,
) -> None:
    session_id = _create_l1_session(tmp_path)
    user_text = "告诉我今天的日期，并说明你确实查询过。"
    policy = freeze_turn_routing_policy(
        TurnRoutingPolicy(l1_enabled=True, l2_enabled=False),
        source="request_override",
    )

    def classify(*_args: object, **kwargs: object) -> ModelResult:
        return _model_result(
            {"processing_level": "L1", "task_matches": []},
            kwargs["model_call_id"],
        )

    model_payloads: list[dict[str, object]] = []

    def decide(_system: str, user_content: str, **kwargs: object) -> ModelResult:
        payload = json.loads(user_content)
        model_payloads.append(payload)
        if len(model_payloads) == 1:
            decision: dict[str, object] = {
                "plan": {
                    "objective": "查询并报告今天的日期",
                    "acceptances": [
                        {
                            "criterion": "查询今天日期并向用户报告",
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
        else:
            tool_result = payload["prior_tool_results"][0]
            tool_result_id = tool_result["tool_result_id"]
            date = tool_result["result"]["date"]
            decision = {
                "plan": None,
                "references": [{"tool_result_id": tool_result_id}],
                "action": {
                    "kind": "submit_final_reply",
                    "reply": f"今天是 {date}。",
                },
            }
        return _model_result(decision, kwargs["model_call_id"])

    monkeypatch.setattr(
        ingress_model,
        "complete_structured",
        as_prepared_test_provider(classify),
    )
    monkeypatch.setattr(
        "personagraph.runtime.l1.model.complete_structured",
        as_prepared_test_provider(decide),
    )

    result = entry.run_entry_turn(
        user_input=user_text,
        features={
            "context_guard_limit": 24_000,
            "l1_max_attempts": 12,
            "l1_max_tool_calls_per_attempt": 8,
        },
        session_id=session_id,
        client_request_id="l1-tool-loop",
        routing_policy=policy,
        store=session_store,
    )

    assert result.status == "completed", result
    assert result.processing_level == "L1"
    assert result.reply is not None and result.reply.startswith("今天是 ")
    assert len(model_payloads) == 2
    assert model_payloads[0]["prior_tool_results"] == []
    assert len(model_payloads[1]["prior_tool_results"]) == 1
    assert "verification_feedback" not in model_payloads[1]
    assert model_payloads[1]["tool_result_projection"] == {
        "durable_tool_result_count": 1,
        "projected_tool_result_count": 1,
        "omitted_tool_result_count": 0,
    }
    assert {item["tool_id"] for item in model_payloads[0]["tool_catalog"]} >= {
        "get_today",
        "date_after",
    }

    execution = session_store.get_l1_turn_execution(
        session_id=session_id,
        turn_id=result.turn_id,
    )
    assert execution is not None
    assert execution["run"]["status"] == "completed"
    assert execution["state"]["stage"] == "completed"
    assert execution["state"]["attempts_started"] == 2
    assert [attempt["status"] for attempt in execution["attempts"]] == [
        "closed",
        "closed",
    ]
    assert [attempt["action_kind"] for attempt in execution["attempts"]] == [
        "call_tools",
        "submit_final_reply",
    ]
    assert len(execution["tool_calls"]) == 1
    assert execution["tool_calls"][0]["status"] == "succeeded"
    logical_calls = [
        session_store.get_runtime_model_logical_call(
            session_id=session_id,
            logical_call_id=attempt["logical_model_call_id"],
        )
        for attempt in execution["attempts"]
    ]
    assert all(call is not None for call in logical_calls)
    assert [
        call.physical_attempts[-1].settlement.outcome.value for call in logical_calls
    ] == ["succeeded", "succeeded"]
    assert session_store.list_insession_task_catalog(session_id) == ()

    replayed = entry.run_entry_turn(
        user_input=user_text,
        features={"context_guard_limit": 24_000},
        session_id=session_id,
        client_request_id="l1-tool-loop",
        store=session_store,
    )
    assert replayed == result


def test_l1_atomic_bootstrap_is_startup_recoverable_before_controller_entry(
    monkeypatch,
    tmp_path,
) -> None:
    session_id = _create_l1_session(tmp_path)
    user_text = "确认原子初始化后的任务可以恢复。"
    policy = freeze_turn_routing_policy(
        TurnRoutingPolicy(l1_enabled=True, l2_enabled=False),
        source="request_override",
    )
    classification_calls = 0

    def classify(*_args: object, **kwargs: object) -> ModelResult:
        nonlocal classification_calls
        classification_calls += 1
        return _model_result(
            {"processing_level": "L1", "task_matches": []},
            kwargs["model_call_id"],
        )

    monkeypatch.setattr(
        ingress_model,
        "complete_structured",
        as_prepared_test_provider(classify),
    )
    original_controller = entry_application._run_l1_controller_and_finalize

    def lose_before_controller(**_kwargs: object):
        raise KeyboardInterrupt("injected post-bootstrap process loss")

    monkeypatch.setattr(
        entry_application,
        "_run_l1_controller_and_finalize",
        lose_before_controller,
    )
    with pytest.raises(KeyboardInterrupt, match="post-bootstrap process loss"):
        entry.run_entry_turn(
            user_input=user_text,
            features={"context_guard_limit": 24_000},
            session_id=session_id,
            client_request_id="l1-atomic-bootstrap-loss",
            routing_policy=policy,
            store=session_store,
        )

    inspection = session_store.inspect_turn_execution(session_id)
    turn_id = str(inspection["turn"]["turn_id"])  # type: ignore[index]
    execution = session_store.get_l1_turn_execution(
        session_id=session_id,
        turn_id=turn_id,
    )
    assert execution is not None
    assert execution["run"]["status"] == "active"
    assert execution["state"]["stage"] == "bootstrap"
    assert execution["state"]["execution_config_json"]
    assert execution["state"]["execution_config_hash"]
    assert execution["state"]["corpus_manifest_contract_version"]
    assert execution["state"]["corpus_manifest_json"]
    assert execution["state"]["corpus_manifest_hash"]

    recovered_feature_snapshots: list[dict[str, object]] = []

    def record_recovered_features(**kwargs: object):
        recovered_feature_snapshots.append(dict(kwargs["features"]))  # type: ignore[arg-type]
        return original_controller(**kwargs)

    monkeypatch.setattr(
        entry_application,
        "_run_l1_controller_and_finalize",
        record_recovered_features,
    )
    monkeypatch.setattr(
        "personagraph.runtime.l1.model.complete_structured",
        as_prepared_test_provider(
            lambda *_args, **kwargs: _model_result(
                {
                    "plan": {
                        "objective": "确认恢复成功",
                        "acceptances": [
                            {
                                "criterion": "确认原子初始化后的任务可以恢复",
                            }
                        ],
                    },
                    "action": {
                        "kind": "submit_final_reply",
                        "reply": "原子初始化后的任务已恢复。",
                    },
                },
                kwargs["model_call_id"],
            ),
        ),
    )
    resumed = entry.resume_active_l1_entry_turn(
        session_id=session_id,
        features={"context_guard_limit": 1},
        store=session_store,
    )

    assert not isinstance(resumed, str)
    assert resumed.status == "completed"
    assert resumed.reply == "原子初始化后的任务已恢复。"
    assert classification_calls == 1
    assert recovered_feature_snapshots[0]["context_guard_limit"] == 24_000


def test_l1_recovery_does_not_freeze_unadopted_workspace_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    bound_partitioned_session,
) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("original corpus", encoding="utf-8")
    session_id = bound_partitioned_session(working_dir=tmp_path)
    user_text = "确认这次恢复可以继续。"
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
    original_controller = entry_application._run_l1_controller_and_finalize

    def lose_after_manifest(**_kwargs: object):
        raise KeyboardInterrupt("injected post-manifest process loss")

    monkeypatch.setattr(
        entry_application,
        "_run_l1_controller_and_finalize",
        lose_after_manifest,
    )
    with pytest.raises(KeyboardInterrupt, match="post-manifest process loss"):
        entry.run_entry_turn(
            user_input=user_text,
            features={"context_guard_limit": 24_000},
            session_id=session_id,
            client_request_id="l1-corpus-drift",
            routing_policy=policy,
            store=session_store,
        )

    source.write_text("changed corpus with a different size", encoding="utf-8")
    monkeypatch.setattr(
        entry_application,
        "_run_l1_controller_and_finalize",
        original_controller,
    )

    provider_calls = 0

    def complete_after_recovery(
        *_args: object,
        **kwargs: object,
    ) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return _model_result(
            {
                "plan": {
                    "objective": "确认恢复继续",
                    "acceptances": [
                        {
                            "criterion": "确认这次恢复可以继续",
                        }
                    ],
                },
                "action": {
                    "kind": "submit_final_reply",
                    "reply": "这次恢复可以继续。",
                },
            },
            kwargs["model_call_id"],
        )

    monkeypatch.setattr(
        "personagraph.runtime.l1.model.complete_structured",
        as_prepared_test_provider(complete_after_recovery),
    )

    resumed = entry.resume_active_l1_entry_turn(
        session_id=session_id,
        features={"context_guard_limit": 24_000},
        store=session_store,
    )

    assert not isinstance(resumed, str)
    assert resumed.status == "completed"
    assert resumed.processing_level == "L1"
    assert resumed.reply == "这次恢复可以继续。"
    assert provider_calls == 1


def test_l1_recovery_rejects_changed_turn_attachment_before_model_io(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    bound_partitioned_session,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = bound_partitioned_session(working_dir=workspace)
    payload = b"original attachment corpus"
    with session_store.session_database_scope(session_id):
        attachment = accept_upload(
            session_id=session_id,
            raw_name="evidence.txt",
            declared_media_type="text/plain",
            payload=payload,
            store=session_store,
        )
    user_text = "读取本轮附件。"
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
    original_controller = entry_application._run_l1_controller_and_finalize

    def lose_after_manifest(**_kwargs: object):
        raise KeyboardInterrupt("injected attachment-manifest process loss")

    monkeypatch.setattr(
        entry_application,
        "_run_l1_controller_and_finalize",
        lose_after_manifest,
    )
    with pytest.raises(
        KeyboardInterrupt,
        match="attachment-manifest process loss",
    ):
        entry.run_entry_turn(
            user_input=user_text,
            features={"context_guard_limit": 24_000},
            session_id=session_id,
            client_request_id="l1-attachment-corpus-drift",
            attachment_ids=(attachment.attachment_id,),
            routing_policy=policy,
            store=session_store,
        )

    with session_store.session_database_scope(session_id):
        record = session_store.get_attachment(attachment.attachment_id)
    assert record is not None
    stored_path = workspace / str(record["stored_rel_path"])
    replacement = bytes((payload[0] ^ 1,)) + payload[1:]
    assert len(replacement) == len(payload)
    stored_path.write_bytes(replacement)
    monkeypatch.setattr(
        entry_application,
        "_run_l1_controller_and_finalize",
        original_controller,
    )

    def forbid_model_call(*_args: object, **_kwargs: object) -> ModelResult:
        raise AssertionError("attachment drift must stop before L1 model I/O")

    monkeypatch.setattr(
        "personagraph.runtime.l1.model.complete_structured",
        as_prepared_test_provider(forbid_model_call),
    )
    resumed = entry.resume_active_l1_entry_turn(
        session_id=session_id,
        features={"context_guard_limit": 24_000},
        store=session_store,
    )

    assert not isinstance(resumed, str)
    assert resumed.status == "incomplete"
    assert resumed.processing_level == "L1"
    assert resumed.error_code == "L1_RUNTIME_NOT_READY"


def test_l1_recovery_does_not_invent_missing_historical_corpus_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    session_id = _create_l1_session(tmp_path)
    user_text = "恢复时不要猜测旧语料范围。"
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
    original_controller = entry_application._run_l1_controller_and_finalize

    def lose_after_bootstrap(**_kwargs: object):
        raise KeyboardInterrupt("injected historical-manifest process loss")

    monkeypatch.setattr(
        entry_application,
        "_run_l1_controller_and_finalize",
        lose_after_bootstrap,
    )
    with pytest.raises(KeyboardInterrupt, match="historical-manifest process loss"):
        entry.run_entry_turn(
            user_input=user_text,
            features={"context_guard_limit": 24_000},
            session_id=session_id,
            client_request_id="l1-missing-corpus-authority",
            routing_policy=policy,
            store=session_store,
        )

    inspection = session_store.inspect_turn_execution(session_id)
    turn_id = str(inspection["turn"]["turn_id"])  # type: ignore[index]
    with sqlite3.connect(session_store.DB_PATH) as conn:
        conn.execute(
            "UPDATE l1_turn_run_states SET "
            "corpus_manifest_contract_version=NULL, corpus_manifest_json=NULL, "
            "corpus_manifest_hash=NULL WHERE turn_id=?",
            (turn_id,),
        )
    monkeypatch.setattr(
        entry_application,
        "_run_l1_controller_and_finalize",
        original_controller,
    )

    def forbid_model_call(*_args: object, **_kwargs: object) -> ModelResult:
        raise AssertionError("missing corpus authority must stop before model I/O")

    monkeypatch.setattr(
        "personagraph.runtime.l1.model.complete_structured",
        as_prepared_test_provider(forbid_model_call),
    )
    resumed = entry.resume_active_l1_entry_turn(
        session_id=session_id,
        features={"context_guard_limit": 24_000},
        store=session_store,
    )

    assert not isinstance(resumed, str)
    assert resumed.status == "incomplete"
    assert resumed.processing_level == "L1"
    assert resumed.error_code == "L1_RUNTIME_NOT_READY"


def test_l1_startup_recovery_fails_closed_after_model_endpoint_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    session_id = _create_l1_session(tmp_path)
    user_text = "不要在恢复时偷偷更换模型配置。"
    policy = freeze_turn_routing_policy(
        TurnRoutingPolicy(l1_enabled=True, l2_enabled=False),
        source="request_override",
    )

    def binding(
        *,
        thinking_enabled: bool,
        base_url: str = "https://models.example.test/v1",
    ) -> ModelTierBinding:
        return ModelTierBinding(
            tier=ModelTier.L1,
            provider="openai-compatible",
            base_url=base_url,
            model="model-for-l1",
            api_key="test-credential",
            thinking_enabled=thinking_enabled,
            origin=EndpointOrigin.PROFILE,
            profile_id="profile-l1-recovery",
            profile_name="L1 recovery profile",
        )

    accepted_binding = binding(thinking_enabled=False)
    monkeypatch.setattr(
        l1_execution_config,
        "resolve_tier",
        lambda _tier: accepted_binding,
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
    original_controller = entry_application._run_l1_controller_and_finalize

    def lose_before_controller(**_kwargs: object):
        raise KeyboardInterrupt("injected post-bootstrap configuration loss")

    monkeypatch.setattr(
        entry_application,
        "_run_l1_controller_and_finalize",
        lose_before_controller,
    )
    with pytest.raises(KeyboardInterrupt, match="configuration loss"):
        entry.run_entry_turn(
            user_input=user_text,
            features={"context_guard_limit": 24_000},
            session_id=session_id,
            client_request_id="l1-endpoint-drift",
            routing_policy=policy,
            store=session_store,
        )

    changed_binding = binding(
        thinking_enabled=True,
        base_url="https://changed-models.example.test/v1",
    )
    monkeypatch.setattr(
        l1_execution_config,
        "resolve_tier",
        lambda _tier: changed_binding,
    )
    monkeypatch.setattr(
        entry_application,
        "_run_l1_controller_and_finalize",
        original_controller,
    )
    resumed = entry.resume_active_l1_entry_turn(
        session_id=session_id,
        features={"context_guard_limit": 1},
        store=session_store,
    )

    assert not isinstance(resumed, str)
    assert resumed.status == "incomplete"
    assert resumed.processing_level == "L1"
    assert resumed.error_code == "MODEL_CONFIGURATION_FAILURE"
    execution = session_store.get_l1_turn_execution(
        session_id=session_id,
        turn_id=resumed.turn_id,
    )
    assert execution is not None
    assert execution["run"]["status"] == "failed"
    assert execution["state"]["failure_code"] == "MODEL_CONFIGURATION_FAILURE"


def test_l1_repairs_one_host_semantic_rejection_inside_the_same_attempt(
    monkeypatch,
    tmp_path,
) -> None:
    session_id = _create_l1_session(tmp_path)
    user_text = "直接确认你已经理解这个请求。"
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
    prompts: list[str] = []
    model_call_ids: list[str] = []
    provider_feedbacks: list[dict[str, object] | None] = []

    def decide(system: str, _user: str, **kwargs: object) -> ModelResult:
        prompts.append(system)
        model_call_ids.append(str(kwargs["model_call_id"]))
        provider_feedbacks.append(repair_feedback_from_provider_kwargs(kwargs))
        plan = {
            "objective": "确认理解请求",
            "acceptances": [
                {
                    "criterion": "确认已经理解",
                }
            ],
        }
        if len(prompts) == 1:
            action: dict[str, object] = {
                "kind": "call_tools",
                "calls": [
                    {
                        "tool_id": "tool_that_does_not_exist",
                        "arguments": {},
                    }
                ],
            }
        else:
            action = {
                "kind": "submit_final_reply",
                "reply": "我已经理解这个请求。",
            }
        (
            []
            if len(prompts) == 1
            else [
                {
                    "acceptance_id": "confirm",
                    "model_claimed_satisfied": True,
                    "supporting_tool_result_ids": [],
                    "empty_support_justification": {
                        "schema_version": "empty-support-justification-v1",
                        "reason_code": "provided_context_sufficient",
                        "explanation": "确认内容可由当前请求直接完成。",
                    },
                }
            ]
        )
        return _model_result(
            {
                "plan": plan,
                "action": action,
            },
            kwargs["model_call_id"],
        )

    monkeypatch.setattr(
        "personagraph.runtime.l1.model.complete_structured",
        as_prepared_test_provider(decide),
    )
    monkeypatch.setattr(
        "personagraph.runtime.model_calls.requests._sleep", lambda _: None
    )

    result = entry.run_entry_turn(
        user_input=user_text,
        features={"context_guard_limit": 24_000},
        session_id=session_id,
        client_request_id="l1-semantic-repair",
        routing_policy=policy,
        store=session_store,
    )

    assert result.status == "completed"
    assert len(prompts) == 2
    assert model_call_ids[0] != model_call_ids[1]
    assert "source_quote" not in prompts[0]
    assert "note" in prompts[0]
    assert prompts[1] == prompts[0]
    assert provider_feedbacks[0] is None
    repair_feedback = provider_feedbacks[1]
    assert repair_feedback is not None
    assert {
        "schema_version",
        "message_contract",
        "rejected_response_sha256",
    }.isdisjoint(repair_feedback)
    assert set(repair_feedback) == {"current_issues"}
    assert repair_feedback["current_issues"][0]["paths"] == ["/action/calls/0/tool_id"]
    execution = session_store.get_l1_turn_execution(
        session_id=session_id,
        turn_id=result.turn_id,
    )
    assert execution is not None
    assert execution["state"]["attempts_started"] == 1
    assert execution["tool_calls"] == []
    logical = session_store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=execution["attempts"][0]["logical_model_call_id"],
    )
    assert logical is not None
    assert len(logical.physical_attempts) == 2
    assert [
        attempt.settlement.outcome.value for attempt in logical.physical_attempts
    ] == ["retryable_failure", "succeeded"]
    feedback = logical.physical_attempts[0].settlement.next_output_repair_feedback
    assert feedback is not None
    assert feedback.current_issues[0].code == "host_guard.l1_unavailable_tool"
    assert logical.physical_attempts[1].request.output_repair_feedback == feedback


def test_l1_exhausted_format_repair_marks_the_run_failed(
    monkeypatch,
    tmp_path,
) -> None:
    session_id = _create_l1_session(tmp_path)
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
    monkeypatch.setattr(
        "personagraph.runtime.l1.model.complete_structured",
        as_prepared_test_provider(
            lambda *_args, **kwargs: ModelResult(
                reply="not-json",
                provider="mock",
                model="mock-structured",
                latency_ms=1,
                model_call_id=str(kwargs["model_call_id"]),
            ),
        ),
    )
    monkeypatch.setattr(
        "personagraph.runtime.model_calls.requests._sleep", lambda _: None
    )

    result = entry.run_entry_turn(
        user_input="完成一个有界回答",
        features={"context_guard_limit": 24_000},
        session_id=session_id,
        client_request_id="l1-format-exhausted",
        routing_policy=policy,
        store=session_store,
    )

    assert result.status == "completed"
    assert result.error_code == "MODEL_OUTPUT_INVALID"
    assert result.reply.startswith("本轮未完成")
    assert "not-json" not in result.reply
    execution = session_store.get_l1_turn_execution(
        session_id=session_id,
        turn_id=result.turn_id,
    )
    assert execution is not None
    assert execution["run"]["status"] == "failed"
    assert execution["state"]["stage"] == "failed"
    assert execution["state"]["failure_code"] == "MODEL_BAD_RESPONSE"
    assert len(execution["attempts"]) == 1
    committed = session_store.get_committed_turn_pair(
        session_id,
        f"commit_{result.turn_id}",
    )
    assert committed is not None
    assert committed["assistant_content"] == result.reply


def test_l1_replays_durable_model_success_after_pre_decision_response_loss(
    monkeypatch,
    tmp_path,
) -> None:
    session_id = _create_l1_session(tmp_path)
    user_text = "确认这次单轮任务已经完成。"
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
    provider_calls = 0

    def decide(*_args: object, **kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return _model_result(
            {
                "plan": {
                    "objective": "确认单轮任务完成",
                    "acceptances": [
                        {
                            "criterion": "确认本轮任务已经完成",
                        }
                    ],
                },
                "action": {
                    "kind": "submit_final_reply",
                    "reply": "这次单轮任务已经完成。",
                },
            },
            kwargs["model_call_id"],
        )

    monkeypatch.setattr(
        "personagraph.runtime.l1.model.complete_structured",
        as_prepared_test_provider(decide),
    )
    original_commit_decision = session_store.commit_l1_attempt_decision
    accepted_turns = []

    def lose_response_after_model_settlement(**_kwargs: object) -> dict[str, object]:
        raise KeyboardInterrupt("injected process loss")

    monkeypatch.setattr(
        session_store,
        "commit_l1_attempt_decision",
        lose_response_after_model_settlement,
    )
    features = {"context_guard_limit": 24_000}
    with pytest.raises(KeyboardInterrupt, match="injected process loss"):
        entry.run_entry_turn(
            user_input=user_text,
            features=features,
            session_id=session_id,
            client_request_id="l1-model-response-loss",
            routing_policy=policy,
            on_turn_accepted=accepted_turns.append,
            store=session_store,
        )

    assert provider_calls == 1
    accepted = accepted_turns[0]
    execution = session_store.get_l1_turn_execution(
        session_id=session_id,
        turn_id=accepted.turn_id,
    )
    assert execution is not None
    assert execution["attempts"][0]["status"] == "active"
    logical = session_store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=execution["attempts"][0]["logical_model_call_id"],
    )
    assert logical is not None
    assert logical.physical_attempts[-1].settlement.outcome.value == "succeeded"

    monkeypatch.setattr(
        session_store,
        "commit_l1_attempt_decision",
        original_commit_decision,
    )
    resumed = entry.run_entry_turn(
        user_input=user_text,
        features={
            "context_guard_limit": 24_000,
        },
        session_id=session_id,
        client_request_id="l1-model-response-loss",
        routing_policy=policy,
        store=session_store,
    )

    assert resumed.status == "completed"
    assert resumed.reply == "这次单轮任务已经完成。"
    assert provider_calls == 1
    recovered_execution = session_store.get_l1_turn_execution(
        session_id=session_id,
        turn_id=accepted.turn_id,
    )
    assert recovered_execution is not None
    assert recovered_execution["attempts"][0]["status"] == "closed"
    assert recovered_execution["attempts"][0]["action_kind"] == "submit_final_reply"
    assert recovered_execution["run"]["status"] == "completed"
    assert recovered_execution["state"]["stage"] == "completed"


def test_l1_startup_recovery_fails_closed_on_unsettled_model_dispatch(
    monkeypatch,
    tmp_path,
) -> None:
    session_id = _create_l1_session(tmp_path)
    user_text = "确认这个模型请求只会发送一次。"
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
    provider_calls = 0

    def lose_during_provider_dispatch(*_args: object, **_kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        raise KeyboardInterrupt("injected pending-model process loss")

    monkeypatch.setattr(
        "personagraph.runtime.l1.model.complete_structured",
        as_prepared_test_provider(lose_during_provider_dispatch),
    )
    with pytest.raises(KeyboardInterrupt, match="pending-model process loss"):
        entry.run_entry_turn(
            user_input=user_text,
            features={"context_guard_limit": 24_000},
            session_id=session_id,
            client_request_id="l1-pending-model-loss",
            routing_policy=policy,
            store=session_store,
        )

    def forbidden_model_call(*_args: object, **_kwargs: object) -> ModelResult:
        raise AssertionError("an unsettled Provider dispatch must not be resent")

    monkeypatch.setattr(
        "personagraph.runtime.l1.model.complete_structured",
        as_prepared_test_provider(forbidden_model_call),
    )
    resumed = entry.resume_active_l1_entry_turn(
        session_id=session_id,
        features={"context_guard_limit": 24_000},
        store=session_store,
    )

    assert not isinstance(resumed, str)
    assert resumed.status == "incomplete"
    assert resumed.processing_level == "L1"
    assert resumed.end_reason == "model_completion_unconfirmed"
    assert resumed.error_code == "MODEL_COMPLETION_UNCONFIRMED"
    assert provider_calls == 1
    execution = session_store.get_l1_turn_execution(
        session_id=session_id,
        turn_id=resumed.turn_id,
    )
    assert execution is not None
    assert execution["run"]["status"] == "failed"
    assert execution["state"]["failure_code"] == "MODEL_COMPLETION_UNCONFIRMED"
    logical = session_store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=execution["attempts"][0]["logical_model_call_id"],
    )
    assert logical is not None
    assert len(logical.physical_attempts) == 1
    assert logical.physical_attempts[0].settlement is None


def test_l1_request_replay_fails_closed_on_an_unsettled_tool_call(
    monkeypatch,
    tmp_path,
) -> None:
    from personagraph.runtime.l1.tool_runtime import L1ToolRuntime

    session_id = _create_l1_session(tmp_path)
    user_text = "查询今天日期。"
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
    provider_calls = 0

    def decide(*_args: object, **kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return _model_result(
            {
                "plan": {
                    "objective": "查询今天日期",
                    "acceptances": [
                        {
                            "criterion": "查询今天日期",
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
            },
            kwargs["model_call_id"],
        )

    monkeypatch.setattr(
        "personagraph.runtime.l1.model.complete_structured",
        as_prepared_test_provider(decide),
    )
    original_execute = L1ToolRuntime.execute_prepared

    def lose_after_reservation(*_args: object, **_kwargs: object):
        raise KeyboardInterrupt("injected pending-tool process loss")

    monkeypatch.setattr(L1ToolRuntime, "execute_prepared", lose_after_reservation)
    features = {"context_guard_limit": 24_000}
    with pytest.raises(KeyboardInterrupt, match="pending-tool process loss"):
        entry.run_entry_turn(
            user_input=user_text,
            features=features,
            session_id=session_id,
            client_request_id="l1-pending-tool-loss",
            routing_policy=policy,
            store=session_store,
        )

    monkeypatch.setattr(L1ToolRuntime, "execute_prepared", original_execute)
    replayed = entry.run_entry_turn(
        user_input=user_text,
        features={"context_guard_limit": 24_000},
        session_id=session_id,
        client_request_id="l1-pending-tool-loss",
        routing_policy=policy,
        store=session_store,
    )

    assert replayed.status == "incomplete"
    assert replayed.processing_level == "L1"
    assert replayed.error_code == "TOOL_COMPLETION_UNCONFIRMED"
    assert provider_calls == 1
    execution = session_store.get_l1_turn_execution(
        session_id=session_id,
        turn_id=replayed.turn_id,
    )
    assert execution is not None
    assert execution["run"]["status"] == "failed"
    assert execution["tool_calls"][0]["status"] == "pending"


def test_l1_resumes_a_settled_tool_batch_before_attempt_close(
    monkeypatch,
    tmp_path,
) -> None:
    session_id = _create_l1_session(tmp_path)
    user_text = "查询今天日期后告诉我。"
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
    provider_calls = 0

    def decide(_system: str, user_content: str, **kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        payload = json.loads(user_content)
        if payload["plan"] is None:
            decision = {
                "plan": {
                    "objective": "查询今天日期",
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
        else:
            tool_result = payload["prior_tool_results"][0]
            decision = {
                "plan": None,
                "action": {
                    "kind": "submit_final_reply",
                    "reply": f"今天是 {tool_result['result']['date']}。",
                },
            }
        return _model_result(decision, kwargs["model_call_id"])

    monkeypatch.setattr(
        "personagraph.runtime.l1.model.complete_structured",
        as_prepared_test_provider(decide),
    )
    original_close_attempt = session_store.close_l1_attempt
    accepted_turns = []

    def lose_before_attempt_close(**_kwargs: object) -> dict[str, object]:
        raise KeyboardInterrupt("injected Attempt close loss")

    monkeypatch.setattr(
        session_store,
        "close_l1_attempt",
        lose_before_attempt_close,
    )
    features = {"context_guard_limit": 24_000}
    with pytest.raises(KeyboardInterrupt, match="Attempt close loss"):
        entry.run_entry_turn(
            user_input=user_text,
            features=features,
            session_id=session_id,
            client_request_id="l1-observation-loss",
            routing_policy=policy,
            on_turn_accepted=accepted_turns.append,
            store=session_store,
        )

    accepted = accepted_turns[0]
    execution = session_store.get_l1_turn_execution(
        session_id=session_id,
        turn_id=accepted.turn_id,
    )
    assert execution is not None
    assert execution["attempts"][0]["status"] == "active"
    assert execution["tool_calls"][0]["status"] == "succeeded"
    assert provider_calls == 1

    monkeypatch.setattr(
        session_store,
        "close_l1_attempt",
        original_close_attempt,
    )
    resumed = entry.run_entry_turn(
        user_input=user_text,
        features={"context_guard_limit": 24_000},
        session_id=session_id,
        client_request_id="l1-observation-loss",
        routing_policy=policy,
        store=session_store,
    )

    assert resumed.status == "completed"
    assert resumed.reply.startswith("今天是 ")
    assert provider_calls == 2
    recovered_execution = session_store.get_l1_turn_execution(
        session_id=session_id,
        turn_id=accepted.turn_id,
    )
    assert recovered_execution is not None
    assert [attempt["status"] for attempt in recovered_execution["attempts"]] == [
        "closed",
        "closed",
    ]


def test_l1_returns_a_persisted_finalizing_answer_without_another_model_call(
    monkeypatch,
    tmp_path,
) -> None:
    session_id = _create_l1_session(tmp_path)
    user_text = "给出本轮最终回答。"
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
    provider_calls = 0

    def decide(*_args: object, **kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        return _model_result(
            {
                "plan": {
                    "objective": "给出最终回答",
                    "acceptances": [
                        {
                            "criterion": "给出本轮最终回答",
                        }
                    ],
                },
                "action": {
                    "kind": "submit_final_reply",
                    "reply": "这是本轮最终回答。",
                },
            },
            kwargs["model_call_id"],
        )

    monkeypatch.setattr(
        "personagraph.runtime.l1.model.complete_structured",
        as_prepared_test_provider(decide),
    )
    original_commit_decision = session_store.commit_l1_attempt_decision
    accepted_turns = []

    def accept_then_lose(**kwargs: object) -> dict[str, object]:
        original_commit_decision(**kwargs)
        raise KeyboardInterrupt("injected finalizing response loss")

    monkeypatch.setattr(
        session_store,
        "commit_l1_attempt_decision",
        accept_then_lose,
    )
    features = {"context_guard_limit": 24_000}
    with pytest.raises(KeyboardInterrupt, match="finalizing response loss"):
        entry.run_entry_turn(
            user_input=user_text,
            features=features,
            session_id=session_id,
            client_request_id="l1-finalizing-loss",
            routing_policy=policy,
            on_turn_accepted=accepted_turns.append,
            store=session_store,
        )

    accepted = accepted_turns[0]
    execution = session_store.get_l1_turn_execution(
        session_id=session_id,
        turn_id=accepted.turn_id,
    )
    assert execution is not None
    assert execution["attempts"][0]["status"] == "closed"
    assert execution["attempts"][0]["action_kind"] == "submit_final_reply"
    assert execution["state"]["stage"] == "finalizing"
    assert provider_calls == 1

    monkeypatch.setattr(
        session_store,
        "commit_l1_attempt_decision",
        original_commit_decision,
    )
    resumed = entry.resume_active_l1_entry_turn(
        session_id=session_id,
        features={"context_guard_limit": 24_000},
        store=session_store,
    )

    assert not isinstance(resumed, str)
    assert resumed.status == "completed"
    assert resumed.reply == "这是本轮最终回答。"
    assert provider_calls == 1


def test_l1_recovers_durable_host_repair_feedback_after_process_loss(
    monkeypatch,
    tmp_path,
) -> None:
    session_id = _create_l1_session(tmp_path)
    user_text = "确认你理解这项要求。"
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
    provider_calls = 0

    prompts: list[str] = []
    provider_feedbacks: list[dict[str, object] | None] = []

    def decide(system: str, *_args: object, **kwargs: object) -> ModelResult:
        nonlocal provider_calls
        provider_calls += 1
        prompts.append(system)
        provider_feedbacks.append(repair_feedback_from_provider_kwargs(kwargs))
        if provider_calls == 1:
            action: dict[str, object] = {
                "kind": "call_tools",
                "calls": [
                    {
                        "tool_id": "missing_tool",
                        "arguments": {},
                    }
                ],
            }
        else:
            action = {
                "kind": "submit_final_reply",
                "reply": "我已经理解这项要求。",
            }
        return _model_result(
            {
                "plan": {
                    "objective": "确认理解",
                    "acceptances": [
                        {
                            "criterion": "确认理解要求",
                        }
                    ],
                },
                "action": action,
            },
            kwargs["model_call_id"],
        )

    monkeypatch.setattr(
        "personagraph.runtime.l1.model.complete_structured",
        as_prepared_test_provider(decide),
    )
    original_append_attempt = session_store.append_runtime_model_physical_attempt
    append_calls = 0

    def lose_before_repair_dispatch(**kwargs: object):
        nonlocal append_calls
        append_calls += 1
        if append_calls == 2:
            raise KeyboardInterrupt("injected pre-repair process loss")
        return original_append_attempt(**kwargs)

    monkeypatch.setattr(
        session_store,
        "append_runtime_model_physical_attempt",
        lose_before_repair_dispatch,
    )
    accepted_turns = []
    features = {"context_guard_limit": 24_000}
    with pytest.raises(KeyboardInterrupt, match="pre-repair process loss"):
        entry.run_entry_turn(
            user_input=user_text,
            features=features,
            session_id=session_id,
            client_request_id="l1-repair-feedback-loss",
            routing_policy=policy,
            on_turn_accepted=accepted_turns.append,
            store=session_store,
        )

    assert provider_calls == 1
    monkeypatch.setattr(
        session_store,
        "append_runtime_model_physical_attempt",
        original_append_attempt,
    )
    resumed = entry.run_entry_turn(
        user_input=user_text,
        features={"context_guard_limit": 24_000},
        session_id=session_id,
        client_request_id="l1-repair-feedback-loss",
        routing_policy=policy,
        store=session_store,
    )

    assert resumed.status == "completed"
    assert resumed.reply == "我已经理解这项要求。"
    assert provider_calls == 2
    assert prompts[1] == prompts[0]
    assert provider_feedbacks[0] is None
    repair_feedback = provider_feedbacks[1]
    assert repair_feedback is not None
    assert repair_feedback["current_issues"][0]["paths"] == ["/action/calls/0/tool_id"]
    execution = session_store.get_l1_turn_execution(
        session_id=session_id,
        turn_id=accepted_turns[0].turn_id,
    )
    assert execution is not None
    assert execution["run"]["status"] == "completed"
    logical = session_store.get_runtime_model_logical_call(
        session_id=session_id,
        logical_call_id=execution["attempts"][0]["logical_model_call_id"],
    )
    assert logical is not None
    assert len(logical.physical_attempts) == 2
    feedback = logical.physical_attempts[0].settlement.next_output_repair_feedback
    assert feedback is not None
    assert logical.physical_attempts[1].request.output_repair_feedback == feedback
