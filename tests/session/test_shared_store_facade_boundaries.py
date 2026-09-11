"""共享 Store 门面仅在调用时解析依赖，并保留底层结果与异常身份。"""

from unittest.mock import Mock, call

import pytest

from personagraph.session import (
    history_store_facade,
    runtime_call_ledger_store_facade,
    turn_input_store_facade,
)
from personagraph.session.persistence.calls import runtime_model_calls, runtime_tool_calls
from personagraph.session.persistence.history import turns
from personagraph.session.persistence.turns import attachments


@pytest.mark.parametrize(
    "facade_module,facade_class,owner,method,args,kwargs,error_name",
    [
        (
            history_store_facade,
            history_store_facade.HistoryStoreFacade,
            turns,
            "get_turns",
            ("session",),
            {},
            "TranscriptDeliveryReferenceError",
        ),
        (
            turn_input_store_facade,
            turn_input_store_facade.TurnInputStoreFacade,
            attachments,
            "get_attachment",
            ("attachment",),
            {},
            "AttachmentBindingError",
        ),
        (
            runtime_call_ledger_store_facade,
            runtime_call_ledger_store_facade.RuntimeCallLedgerStoreFacade,
            runtime_model_calls,
            "get_runtime_model_logical_call",
            (),
            {"session_id": "session", "logical_call_id": "model-call"},
            "RuntimeModelCallPersistenceError",
        ),
        (
            runtime_call_ledger_store_facade,
            runtime_call_ledger_store_facade.RuntimeCallLedgerStoreFacade,
            runtime_tool_calls,
            "get_runtime_tool_logical_call",
            (),
            {"session_id": "session", "logical_tool_call_id": "tool-call"},
            "RuntimeToolCallPersistenceError",
        ),
    ],
)
def test_shared_facade_resolves_dependencies_once_per_call(
    monkeypatch, facade_module, facade_class, owner, method, args, kwargs, error_name
):
    first_deps, second_deps, result = object(), object(), object()
    deps_factory = Mock(side_effect=(first_deps, second_deps))
    operation = Mock(return_value=result)
    monkeypatch.setattr(owner, method, operation)

    facade = facade_class(deps_factory=deps_factory)
    deps_factory.assert_not_called()
    assert getattr(facade, method)(*args, **kwargs) is result
    assert getattr(facade, method)(*args, **kwargs) is result
    assert deps_factory.call_args_list == [call(), call()]
    assert operation.call_args_list == [
        call(first_deps, *args, **kwargs),
        call(second_deps, *args, **kwargs),
    ]
    assert getattr(facade_module, error_name) is getattr(owner, error_name)
