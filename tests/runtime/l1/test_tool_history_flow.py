"""实际 L1 入口：正文退出、目录发现、原文回读与原始证据验证闭环。"""

import json

import pytest

from personagraph.model_io.gateway import ModelResult
from personagraph.persistent_turn_content.evidence import l1_tool_result_id
from personagraph.runtime import entry
from personagraph.runtime.entry.routing.policy import TurnRoutingPolicy, freeze_turn_routing_policy
from personagraph.session import store as session_store
from tests.helpers.prepared_model_provider import as_prepared_test_provider, repair_feedback_from_provider_kwargs


USER_TEXT = "查询今天日期并回答。"
PLAN = {"objective": "查询并报告日期", "acceptances": [{"criterion": "查询日期并回答"}]}


def _result(value, kwargs):
    return ModelResult(reply=json.dumps(value, ensure_ascii=False), provider="mock", model="mock-structured",
                       latency_ms=1, model_call_id=kwargs["model_call_id"])


def _note(*, decision_summary, observation_summary=""):
    return observation_summary + decision_summary


@pytest.mark.parametrize("reject_first_answer", [False, True])
def test_history_read_keeps_original_evidence_after_body_eviction(
    monkeypatch, tmp_path, bound_partitioned_session, reject_first_answer,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    session_id = bound_partitioned_session(working_dir=workspace)
    payloads = []
    original = {}
    review_count = 0
    repair_count = 0
    read_requests = []

    def decide(_system, user_content, **kwargs):
        nonlocal repair_count
        payload = json.loads(user_content)
        payloads.append(payload)
        prior = payload["prior_tool_results"]
        call = None
        if payload["plan"] is None:
            assert prior == []
            call = ("get_today", {})
            note = _note(decision_summary="先取得当前日期。")
        elif payload.get("verification_feedback") is not None:
            assert prior == []  # 上一步是提交，不可复活更早的回读正文。
            assert payload["tool_result_projection"]["omitted_tool_result_count"] == 3
            note = _note(decision_summary="根据审查意见修订表述，日期依据不变。")
        elif prior[0]["tool_id"] == "get_today":
            assert len(prior) == 1
            original.update(prior[0])
            call = ("list_tool_results", {})
            note = _note(
                observation_summary=f"日期工具返回 {original['result']['date']}，后续沿用此结果。",
                decision_summary="查看历史结果目录以确认原始结果可回读。",
            )
        elif prior[0]["tool_id"] == "list_tool_results":
            assert len(prior) == 1
            assert original["result"] not in [r["result"] for r in prior]
            assert payload["tool_result_projection"]["omitted_tool_result_count"] == 1
            rows = prior[0]["result"]["results"]
            assert rows[0]["source"]["tool_result_id"] == original["tool_result_id"]
            call = ("read_tool_result", {
                "tool_result_id": rows[0]["source"]["tool_result_id"], "path": "/result",
            })
            note = _note(
                observation_summary="历史目录给出了原始日期结果的读取身份。",
                decision_summary="回读已保存的原始日期结果进行核对。",
            )
            read_requests.append(user_content)
        else:
            assert len(prior) == 1 and prior[0]["tool_id"] == "read_tool_result"
            response = prior[0]["result"]
            assert response["source"]["tool_result_id"] == original["tool_result_id"]
            assert response["partial"] is False
            original["readback_value"] = response["value"]
            assert response["value"] == original["result"]
            assert any(original["result"]["date"] in n["summary"]
                       for n in payload["execution_findings"]["notes"])
            note = _note(
                observation_summary="已核对原始结果，可按保存的日期回答。",
                decision_summary="提交有原始日期证据支持的答复。",
            )
        decision = {
            "plan": PLAN if payload["plan"] is None else None,
            "note": note,
            "action": {"kind": "call_tools", "calls": [{"tool_id": call[0], "arguments": call[1]}]}
            if call else {"kind": "submit_final_reply", "reply": f"今天是 {original['result']['date']}。"},
        }
        if call is None:
            decision["references"] = [{"tool_result_id": original["tool_result_id"]}]
        if call and call[0] == "read_tool_result":
            feedback = repair_feedback_from_provider_kwargs(kwargs)
            if feedback is None:
                decision["unexpected_field"] = True  # 同一步结构重试不消费 prior_tool_results。
            else:
                repair_count += 1
                assert read_requests[-2] == user_content
        return _result(decision, kwargs)

    def review(_system, user_content, **kwargs):
        nonlocal review_count
        review_count += 1
        payload = json.loads(user_content)
        submitted = payload["durable_evidence"]["results"]
        assert len(submitted) == 1
        assert submitted[0]["tool_result_id"] == original["tool_result_id"]
        assert submitted[0]["result"] == original["result"]
        reject = reject_first_answer and review_count == 1
        return _result({
            "issues": [{"message": "请核对并明确说明日期。"}] if reject else []}, kwargs)

    monkeypatch.setattr("personagraph.runtime.l1.model.complete_structured", as_prepared_test_provider(decide))
    monkeypatch.setattr("personagraph.runtime.entry.ingress.model.complete_structured", as_prepared_test_provider(
        lambda *_args, **kwargs: _result({"processing_level": "L1", "task_matches": []}, kwargs),
    ))
    monkeypatch.setattr("personagraph.runtime.l1.semantic_verification.complete_structured", as_prepared_test_provider(review))
    monkeypatch.setattr("personagraph.runtime.model_calls.requests._sleep", lambda _: None)
    result = entry.run_entry_turn(
        user_input=USER_TEXT, features={"context_guard_limit": 24_000, "l1_max_attempts": 6,
                                      "l1_max_tool_calls_per_attempt": 4},
        session_id=session_id, client_request_id=f"tool-history-{reject_first_answer}",
        routing_policy=freeze_turn_routing_policy(TurnRoutingPolicy(l1_enabled=True, l2_enabled=False),
                                                source="request_override"),
        store=session_store,
    )
    assert result.status == "completed", (result.error_code, len(payloads))
    execution = session_store.get_l1_turn_execution(session_id=session_id, turn_id=result.turn_id)
    assert [c["tool_id"] for c in execution["tool_calls"]].count("get_today") == 1
    assert len(execution["tool_calls"]) == 3
    original_call = next(call for call in execution["tool_calls"] if call["tool_id"] == "get_today")
    assert json.loads(original_call["outcome_json"])["result"] == original["readback_value"]
    assert l1_tool_result_id(
        tool_call_id=original_call["tool_call_id"], result_sha256=original_call["outcome_hash"],
    ) == original["tool_result_id"]
    assert len(execution["attempts"]) == 4 + int(reject_first_answer)
    assert repair_count == 1
    assert review_count == 1 + int(reject_first_answer)
