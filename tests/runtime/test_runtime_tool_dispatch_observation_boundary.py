"""持久运行时工具分发观察的边界覆盖。"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from personagraph.runtime.tool_calls import contracts


class _Result(BaseModel):
    page: int
    text: str


def _values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "session_id": "session-1",
        "work_run_id": "work-run-1",
        "attempt_id": "attempt-1",
        "logical_tool_call_id": "call-1",
        "logical_request_binding_sha256": "a" * 64,
        "physical_attempt_id": "physical-1",
        "physical_request_binding_sha256": "b" * 64,
        "physical_ordinal": 1,
        "provider_identity_sha256": "c" * 64,
        "tool_id": "document-read",
        "contract_version": "document-read-v1",
        "implementation_version": "document-read-impl-v1",
        "outcome": contracts.RuntimeToolPhysicalOutcome.SUCCEEDED,
        "provider_request_id": "provider-1",
        "duration_ms": 9,
        "error_code": None,
    }
    values.update(overrides)
    return values


def test_dispatch_observation_preserves_result_and_error_semantics() -> None:
    succeeded = contracts.RuntimeToolDispatchObservation.create(
        result=_Result(page=2, text="evidence"),
        **_values(),
    )
    assert succeeded.parsed_result() == {"page": 2, "text": "evidence"}

    failed = contracts.RuntimeToolDispatchObservation.create(
        **_values(
            outcome=contracts.RuntimeToolPhysicalOutcome.RETRYABLE_FAILURE,
            error_code="temporary-transport",
        )
    )
    with pytest.raises(
        contracts.RuntimeToolCallAuthorityError,
        match="failed dispatch observation has no typed result",
    ):
        failed.parsed_result()

    with pytest.raises(ValueError, match="only a succeeded dispatch observation"):
        contracts.RuntimeToolDispatchObservation.create(**_values())
