"""同一 issues 合同用于模型 Schema、解析、真实持久结果和 Host 判定。"""

import json

import pytest
from pydantic import ValidationError

from personagraph.output_protocol.l1 import L1PlanProposal, materialize_l1_plan
from personagraph.runtime.l1.identity import sha256_json
from personagraph.runtime.l1.semantic_contracts import (
    L1SemanticVerificationResult,
    derive_l1_semantic_verification_trigger,
    L1SemanticVerificationMode,
)
from personagraph.runtime.l1.semantic_verification import (
    _L1_SEMANTIC_SYSTEM_PROMPT,
    _validate_semantic_result,
    L1SemanticVerificationInvocation,
    build_l1_semantic_verification_receipt,
)


def _plan():
    return materialize_l1_plan(
        L1PlanProposal(objective="回答", acceptances=({"criterion": "回答"},)),
        input_message_id="input-1",
        user_text="回答",
    )


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"issues": None},
        {"issues": [{"message": "  "}]},
        {"issues": [], "verdict": "pass"},
        {"issues": [], "summary": "passed"},
    ],
)
def test_malformed_review_never_defaults_to_pass(raw):
    with pytest.raises(ValidationError):
        L1SemanticVerificationResult.model_validate(raw)


def test_pass_and_revise_are_derived_without_fabricating_persisted_fields():
    accepted = L1SemanticVerificationResult.model_validate({"issues": []})
    assert accepted.verdict == "pass"
    assert accepted.model_dump(mode="json") == {"issues": []}
    rejected = L1SemanticVerificationResult.model_validate(
        {"issues": [{"message": " 缺少实际依据。 "}]}
    )
    assert rejected.verdict == "revise"
    assert rejected.issues[0].message == "缺少实际依据。"
    assert set(rejected.model_dump()) == {"issues"}
    assert "缺少实际依据" in rejected.safe_feedback()


def test_optional_acceptance_id_is_checked_only_when_supplied():
    plan = _plan()
    _validate_semantic_result(
        L1SemanticVerificationResult(issues=[{"message": "缺证据"}]), plan=plan
    )
    _validate_semantic_result(
        L1SemanticVerificationResult(
            issues=[
                {
                    "message": "未回应此项",
                    "acceptance_id": plan.acceptances[0].acceptance_id,
                }
            ]
        ),
        plan=plan,
    )
    with pytest.raises(ValueError, match="acceptance_id"):
        _validate_semantic_result(
            L1SemanticVerificationResult(
                issues=[
                    {
                        "message": "未回应此项",
                        "acceptance_id": "made_up",
                    }
                ]
            ),
            plan=plan,
        )


def test_prompt_embeds_the_same_complete_schema_as_validation():
    schema = json.loads(_L1_SEMANTIC_SYSTEM_PROMPT.splitlines()[-1])
    assert schema == L1SemanticVerificationResult.model_json_schema()
    assert schema["required"] == ["issues"]
    assert set(schema["properties"]) == {"issues"}
    assert "可交付不等于全部任务完成" in _L1_SEMANTIC_SYSTEM_PROMPT


def test_prompt_documents_review_inputs_in_four_ordered_sections():
    # These guard the input documentation, not the real model's review quality.
    prompt = _L1_SEMANTIC_SYSTEM_PROMPT
    headings = ("一、审查职责", "二、只读输入", "三、判断标准", "四、输出")
    positions = [prompt.index(heading) for heading in headings]
    assert positions == sorted(positions)
    input_table = prompt[positions[1] : positions[2]]
    for field in (
        "request_context.current_user_text",
        "request_context.history_pairs",
        "request_context.session_summary",
        "request_context.attachments",
        "plan.objective",
        "plan.acceptances[].acceptance_id",
        "plan.acceptances[].criterion",
        "candidate_final_reply",
        "execution_context.stop",
        "execution_context.tool_calls",
        "execution_context.model_notes",
        "execution_context.candidate_note",
        "durable_evidence.results",
        "durable_evidence.results[].result_scope",
        "durable_evidence.omitted_recent_result_count",
        "durable_evidence.selection",
        "durable_evidence.evidence_scope",
        "verification_feedback",
    ):
        assert field in input_table


def test_prompt_distinguishes_bounded_uncertainty_from_unsupported_absence_claims():
    # Preserve the agreed review boundaries without pretending to measure accuracy.
    prompt = _L1_SEMANTIC_SYSTEM_PROMPT
    assert "两类情况均可通过" in prompt
    assert "不要求证明全文不存在答案或已经穷尽所有查找" in prompt
    assert "被省略不等于未执行或不存在" in prompt
    assert "不包含上一版候选全文" in prompt
    assert "不是必须沿用的结论" in prompt
    assert "不能仅因没有确定答案、还可以继续查找" in prompt


def test_prompt_distinguishes_execution_facts_missing_evidence_and_contradictions():
    # This tests instruction coverage, not a real model's ability to obey it.
    prompt = _L1_SEMANTIC_SYSTEM_PROMPT
    assert "status=ready 不等于正文已读" in prompt
    assert "未尝试读取不等于权限拒绝" in prompt
    assert "对应文件或能力的工具错误" in prompt
    assert "摘要未提及某事实不等于否定该事实" in prompt
    assert "区分材料直接矛盾与当前证据不足" in prompt
    assert "尚未取得足够正文" in prompt


@pytest.mark.parametrize(
    ("label", "verdict"),
    [("通过示例：", "pass"), ("不通过示例：", "revise")],
)
def test_prompt_output_examples_match_the_actual_contract(label, verdict):
    lines = _L1_SEMANTIC_SYSTEM_PROMPT.splitlines()
    example = lines[lines.index(label) + 1]
    result = L1SemanticVerificationResult.model_validate_json(example)
    assert result.verdict == verdict
    _validate_semantic_result(result, plan=_plan())


def test_receipt_binds_authentic_issues_result_not_an_invented_completion_table():
    result = L1SemanticVerificationResult(issues=[])
    trigger = derive_l1_semantic_verification_trigger(
        mode=L1SemanticVerificationMode.ALWAYS,
        acceptance_count=1,
        tool_result_count=0,
    )
    receipt = build_l1_semantic_verification_receipt(
        trigger=trigger,
        decision_hash="a" * 64,
        plan_hash="b" * 64,
        mechanical_verification_hash="c" * 64,
        state_guard_hash="d" * 64,
        checked_acceptances=1,
        checked_tool_results=0,
        invocation=L1SemanticVerificationInvocation(
            result, "review-1", sha256_json(result), 1
        ),
    )
    assert receipt.model_dump(mode="json")["reviewer_result"] == {"issues": []}
    assert receipt.reviewer_result_hash == sha256_json({"issues": []})
    assert "model_claimed_satisfied" not in receipt.model_dump_json()
    with pytest.raises(ValueError, match="did not pass"):
        build_l1_semantic_verification_receipt(
            trigger=trigger,
            decision_hash="a" * 64,
            plan_hash="b" * 64,
            mechanical_verification_hash="c" * 64,
            state_guard_hash="d" * 64,
            checked_acceptances=1,
            checked_tool_results=0,
            invocation=L1SemanticVerificationInvocation(
                L1SemanticVerificationResult(issues=[{"message": "候选不可靠"}]),
                "review-1",
                "e" * 64,
                1,
            ),
        )
