"""测试适配器不得改写共享 Provider 或其他测试持有的包装器。"""

from __future__ import annotations

import json

from personagraph.model_io import gateway
from personagraph.model_io.gateway import ModelResult, PreparedModelCall
from tests.helpers.prepared_model_provider import as_prepared_test_provider


def _result() -> ModelResult:
    return ModelResult(
        reply=json.dumps({
            "plan": {"objective": "answer", "acceptances": [{"criterion": "answer"}]},
            "references": [{"tool_result_id": "native-result"}],
            "action": {"kind": "submit_final_reply", "reply": "answer"},
        }),
        provider="mock", model="mock", latency_ms=0,
    )


def test_l1_adapter_does_not_replace_the_shared_gateway_prepare(monkeypatch):
    original = gateway.complete_structured.prepare
    # 失败回归本身也需隔离旧实现的污染，防止影响本进程的后续测试。
    monkeypatch.setattr(gateway.complete_structured, "prepare", original)

    adapted = as_prepared_test_provider(gateway.complete_structured, add_l1_notes=True)

    assert gateway.complete_structured.prepare is original
    assert original is gateway.prepare_complete_structured
    assert adapted is not gateway.complete_structured
    assert adapted.prepare is not original


def test_prepared_wrappers_leave_original_and_inner_configuration_unchanged():
    prepared_calls = []
    dispatched_calls = []

    def provider(*_args, **_kwargs):
        raise AssertionError("prepared dispatch must not call the direct Provider")

    def prepare(system, user, **kwargs):
        prepared_calls.append((system, user, kwargs))

        def dispatch(call_id):
            dispatched_calls.append(call_id)
            return _result()

        return PreparedModelCall(_dispatch=dispatch)

    provider.prepare = prepare
    notes = as_prepared_test_provider(provider, add_l1_notes=True)
    notes_prepare = notes.prepare
    unchanged = as_prepared_test_provider(notes)

    assert provider.prepare is prepare
    assert notes.prepare is notes_prepare
    assert unchanged is notes
    assert as_prepared_test_provider(provider) is provider

    request = "{}"
    results = [
        json.loads(current.prepare(
            "system", request, purpose="runtime_l1_attempt",
        ).dispatch(model_call_id=f"call-{index}").reply)
        for index, current in enumerate((provider, notes, unchanged))
    ]
    assert "note" not in results[0]
    assert results[1]["note"] == results[2]["note"]
    assert results[1]["note"]
    assert all(result["references"] == [{"tool_result_id": "native-result"}]
               for result in results)
    assert all(result["plan"] == results[0]["plan"] for result in results)
    assert len(prepared_calls) == 3
    assert dispatched_calls == ["call-0", "call-1", "call-2"]


def test_legacy_provider_gets_a_separate_prepared_wrapper_and_keyword_filter():
    seen = []

    def provider(system, user, *, model_call_id, purpose, kept=None):
        seen.append((system, user, model_call_id, purpose, kept))
        return _result()

    adapted = as_prepared_test_provider(provider)

    assert adapted is not provider
    assert not hasattr(provider, "prepare")
    prepared = adapted.prepare(
        "system", "{}", purpose="parser_test", kept="yes", ignored="no",
    )
    assert seen == []
    assert prepared.dispatch(model_call_id="test-call") == _result()
    assert seen == [("system", "{}", "test-call", "parser_test", "yes")]
