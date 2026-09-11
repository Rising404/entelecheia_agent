"""提供方中立模型账本 Store 端口的边界覆盖。"""

from __future__ import annotations

import inspect
from typing import get_type_hints

from personagraph.runtime.model_calls import contracts
from personagraph.session.runtime_call_ledger_store_facade import (
    RuntimeCallLedgerStoreFacade,
)


def test_ledger_store_port_preserves_keyword_only_persistence_contracts() -> None:
    expected = {
        "reserve_runtime_model_logical_call": (
            ("self", "request"),
            {"request": contracts.RuntimeModelLogicalRequest},
            object,
        ),
        "append_runtime_model_physical_attempt": (
            ("self", "request"),
            {"request": contracts.RuntimeModelPhysicalAttemptRequest},
            object,
        ),
        "settle_runtime_model_physical_attempt": (
            ("self", "settlement", "rejected_response_text"),
            {
                "settlement": contracts.RuntimeModelPhysicalAttemptSettlement,
                "rejected_response_text": str | None,
            },
            object,
        ),
        "get_runtime_model_logical_call": (
            ("self", "session_id", "logical_call_id"),
            {"session_id": str, "logical_call_id": str},
            object | None,
        ),
        "get_runtime_model_rejected_output": (
            (
                "self",
                "session_id",
                "logical_call_id",
                "rejected_physical_ordinal",
                "rejected_response_sha256",
            ),
            {
                "session_id": str,
                "logical_call_id": str,
                "rejected_physical_ordinal": int,
                "rejected_response_sha256": str,
            },
            object | None,
        ),
    }
    for method_name, (parameters, typed_parameters, expected_return_type) in expected.items():
        method = getattr(contracts.RuntimeModelLedgerStore, method_name)
        signature = inspect.signature(method)
        hints = get_type_hints(method)
        assert tuple(signature.parameters) == parameters
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for name, parameter in signature.parameters.items()
            if name != "self"
        )
        for name, expected_type in typed_parameters.items():
            assert hints[name] == expected_type
        assert hints["return"] == expected_return_type


def test_session_facade_preserves_the_model_ledger_call_shape() -> None:
    for method_name in (
        "reserve_runtime_model_logical_call",
        "append_runtime_model_physical_attempt",
        "settle_runtime_model_physical_attempt",
        "get_runtime_model_logical_call",
        "get_runtime_model_rejected_output",
    ):
        contract = inspect.signature(
            getattr(contracts.RuntimeModelLedgerStore, method_name)
        )
        facade = inspect.signature(
            getattr(RuntimeCallLedgerStoreFacade, method_name)
        )

        assert tuple(facade.parameters) == tuple(contract.parameters)
        for name, parameter in contract.parameters.items():
            facade_parameter = facade.parameters[name]
            assert facade_parameter.kind is parameter.kind
            assert facade_parameter.default == parameter.default
