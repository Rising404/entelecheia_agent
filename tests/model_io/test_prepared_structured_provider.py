from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairFeedback,
    RuntimeModelOutputRepairIssue,
    RuntimeModelStructuredPrompt,
)
from personagraph.model_io.prepared_structured_provider import (
    durable_structured_provider_prompt,
    prepare_structured_repair_request,
    prepare_structured_request,
)


def test_durable_structured_prompt_replaces_current_renderer_exactly() -> None:
    snapshot = RuntimeModelStructuredPrompt.create(
        system_prompt="frozen system",
        user_content='{"frozen":"user"}',
    )
    durable = SimpleNamespace(
        logical_request=SimpleNamespace(structured_prompt=snapshot)
    )

    assert durable_structured_provider_prompt(
        durable,
        system_prompt="changed system",
        user_content='{"changed":"user"}',
    ) == ("frozen system", '{"frozen":"user"}')


def _feedback(rejected_response_text: str) -> RuntimeModelOutputRepairFeedback:
    return RuntimeModelOutputRepairFeedback(
        target_contract="l1-decision-proposal-v1",
        rejected_physical_ordinal=2,
        rejected_response_sha256=hashlib.sha256(
            rejected_response_text.encode("utf-8")
        ).hexdigest(),
        issue_coverage="complete",
        omitted_issue_count=0,
        current_issues=(
            RuntimeModelOutputRepairIssue(
                category="schema",
                code="schema.missing",
                paths=("/action/completion_report/obligation_reports",),
                safe_explanation="目标合同要求此位置必须存在。",
            ),
        ),
    )


class _PreparedProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def prepare(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        return {"prepared": len(self.calls)}


def test_structured_preparer_forwards_fresh_provider_kwargs() -> None:
    provider = _PreparedProvider()
    prepare_request = prepare_structured_request(
        provider,
        system_prompt="system",
        user_content="user",
        purpose="contract",
        prepare_kwargs=lambda: {"max_tokens": 2048, "json_mode": True},
    )

    assert prepare_request() == {"prepared": 1}
    assert provider.calls == [
        (
            ("system", "user"),
            {
                "purpose": "contract",
                "max_tokens": 2048,
                "json_mode": True,
            },
        )
    ]


def test_repair_preparer_builds_exact_four_message_contract() -> None:
    provider = _PreparedProvider()
    rejected = '{"action":{"type":"final_answer"}}'
    feedback = _feedback(rejected)
    prepare_repair = prepare_structured_repair_request(
        provider,
        system_prompt="frozen system",
        user_content='{"frozen":"user"}',
        purpose="l1_decision",
    )

    assert prepare_repair(feedback, rejected) == {"prepared": 1}
    args, kwargs = provider.calls[0]
    assert args == ("frozen system", '{"frozen":"user"}')
    assert kwargs["purpose"] == "l1_decision"
    messages = kwargs["repair_messages"]
    assert isinstance(messages, list)
    assert messages[:3] == [
        {"role": "system", "content": "frozen system"},
        {"role": "user", "content": '{"frozen":"user"}'},
        {"role": "assistant", "content": rejected},
    ]
    assert messages[3]["role"] == "user"
    repair_instruction = messages[3]["content"]
    assert isinstance(repair_instruction, str)
    visible = json.loads(repair_instruction.split("Host 修复清单：", 1)[1])
    assert visible == {"current_issues": [{
        "paths": ["/action/completion_report/obligation_reports"],
        "safe_explanation": "目标合同要求此位置必须存在。",
    }]}
    assert "完整 JSON" in repair_instruction
    assert "可能不完整" not in repair_instruction
    for host_only in (
        feedback.rejected_response_sha256, feedback.target_contract,
        "schema_version", "rejected_physical_ordinal", "issue_coverage", "schema.missing",
    ):
        assert host_only not in repair_instruction
    assert rejected not in repair_instruction


def test_repair_preparer_forwards_fresh_provider_kwargs() -> None:
    provider = _PreparedProvider()
    rejected = '{"bad":true}'
    timeout = 7.5
    prepare_repair = prepare_structured_repair_request(
        provider,
        system_prompt="system",
        user_content="user",
        purpose="l1-decision",
        prepare_kwargs=lambda: {
            "mock_payload": {"ok": True},
            "max_tokens": 2048,
            "timeout_s": timeout,
            "json_mode": True,
        },
    )

    prepare_repair(_feedback(rejected), rejected)
    _args, kwargs = provider.calls[0]
    assert kwargs["mock_payload"] == {"ok": True}
    assert kwargs["max_tokens"] == 2048
    assert kwargs["timeout_s"] == timeout
    assert kwargs["json_mode"] is True


def test_repair_preparer_rejects_a_body_that_does_not_match_envelope() -> None:
    provider = _PreparedProvider()
    prepare_repair = prepare_structured_repair_request(
        provider,
        system_prompt="system",
        user_content="user",
        purpose="contract",
    )

    with pytest.raises(ValueError, match="SHA-256"):
        prepare_repair(_feedback("expected"), "different")
    assert provider.calls == []


def test_repair_preparer_requires_contract_and_text_body() -> None:
    provider = _PreparedProvider()
    prepare_repair = prepare_structured_repair_request(
        provider,
        system_prompt="system",
        user_content="user",
        purpose="contract",
    )
    with pytest.raises(TypeError, match="contract"):
        prepare_repair(object(), "rejected")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="text"):
        prepare_repair(_feedback("rejected"), b"rejected")  # type: ignore[arg-type]
    assert provider.calls == []


def test_structured_helpers_fail_closed_for_provider_without_prepare() -> None:
    with pytest.raises(TypeError, match="must expose prepare"):
        prepare_structured_request(
            object(),
            system_prompt="system",
            user_content="user",
            purpose="contract",
        )
    with pytest.raises(TypeError, match="must expose prepare"):
        prepare_structured_repair_request(
            object(),
            system_prompt="system",
            user_content="user",
            purpose="contract",
        )


def test_repair_projection_preserves_durable_feedback_and_exact_rejected_text() -> None:
    provider = _PreparedProvider()
    rejected = ' {"content":"保持原始排版与 hash 字样"}\n'
    feedback = _feedback(rejected)
    original = feedback.model_dump_json()

    prepare = prepare_structured_repair_request(
        provider, system_prompt="system", user_content="user", purpose="contract",
    )
    prepare(feedback, rejected)
    messages = provider.calls[0][1]["repair_messages"]
    assert messages[2] == {"role": "assistant", "content": rejected}
    visible = json.loads(messages[3]["content"].split("Host 修复清单：", 1)[1])
    assert visible["current_issues"][0]["safe_explanation"] == feedback.current_issues[0].safe_explanation
    assert feedback.model_dump_json() == original


@pytest.mark.parametrize("coverage", ["partial", "first_only", "truncated"])
def test_incomplete_repair_explains_remaining_checks_without_protocol_metadata(coverage) -> None:
    provider = _PreparedProvider()
    feedback = RuntimeModelOutputRepairFeedback.model_validate({
        **_feedback("bad").model_dump(),
        "issue_coverage": coverage,
        "omitted_issue_count": 1 if coverage == "truncated" else 0,
    })
    prepare_structured_repair_request(
        provider, system_prompt="系统", user_content="问题", purpose="contract",
    )(feedback, "bad")
    instruction = provider.calls[0][1]["repair_messages"][3]["content"]
    assert "当前清单可能不完整，请同时检查其余字段。" in instruction
    assert coverage not in instruction


def test_json_syntax_repair_preserves_line_and_column() -> None:
    provider = _PreparedProvider()
    feedback = RuntimeModelOutputRepairFeedback.model_validate({
        **_feedback("bad").model_dump(),
        "current_issues": [{
            "category": "json_syntax", "code": "json_syntax.invalid_json", "paths": [""],
            "safe_explanation": "字符串未闭合。", "json_line": 3, "json_column": 12,
        }],
    })
    prepare_structured_repair_request(
        provider, system_prompt="system", user_content="user", purpose="contract",
    )(feedback, "bad")
    instruction = provider.calls[0][1]["repair_messages"][3]["content"]
    visible = json.loads(instruction.split("Host 修复清单：", 1)[1])
    assert visible == {"current_issues": [{
        "paths": [""], "safe_explanation": "字符串未闭合。", "json_line": 3, "json_column": 12,
    }]}
