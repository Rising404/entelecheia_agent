"""有界 L1 lane 的确定性结构/来源 gate。

本模块只检查已提交标识与字段关系，不判断所引证据是否在语义上证明 Acceptance。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ...tools.findings.contracts import EXECUTION_FINDINGS_TOOL_IDS
from ...tools.tool_history.definitions import TOOL_HISTORY_TOOL_IDS
from ...persistent_turn_content import L1Plan, l1_tool_result_id
from ...output_protocol.l1 import L1ResultReference
from .semantic_evidence import L1EvidenceProjectionError, project_referenced_results


L1_VERIFICATION_CONTRACT_VERSION = "l1-final-reply-verification"


class L1VerificationStateError(RuntimeError):
    """已持久化执行证据格式错误或内部不一致。"""


@dataclass(frozen=True, slots=True)
class L1VerificationIssue:
    code: str
    acceptance_id: str | None = None
    tool_result_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "acceptance_id": self.acceptance_id,
            "tool_result_id": self.tool_result_id,
        }


@dataclass(frozen=True, slots=True)
class L1VerificationResult:
    contract_version: str
    passed: bool
    issues: tuple[L1VerificationIssue, ...]
    checked_acceptances: int
    checked_tool_results: int

    def safe_feedback(self) -> str:
        codes = tuple(dict.fromkeys(issue.code for issue in self.issues))
        return "L1 final-reply verification failed: " + ", ".join(codes)

    def safe_details(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "checked_acceptances": self.checked_acceptances,
            "checked_tool_results": self.checked_tool_results,
        }


@dataclass(frozen=True, slots=True)
class _ToolResultEvidence:
    tool_result_id: str
    tool_call_id: str
    tool_id: str
    status: str


def verify_l1_final_reply(
    *,
    plan: L1Plan,
    reply: str,
    references: tuple[L1ResultReference, ...],
    execution: Mapping[str, object],
) -> L1VerificationResult:
    """纯机械门：校验正文与可选真实来源，不生成逐项完成声明。

    提交引用必须对应成功 ToolResult，findings 写入结果
    不能当作独立证据；pending/损坏的持久工具状态是 Host 状态错误。
    本函数不调用模型，也不裁定证据能否语义上证明答案，
    这些交给独立 semantic reviewer。
    """

    evidence = _load_tool_result_evidence(execution)
    if any(item.status == "pending" for item in evidence.values()):
        raise L1VerificationStateError(
            "L1 final-reply verification encountered a pending ToolCall"
        )

    issues: list[L1VerificationIssue] = []
    acceptance_ids = tuple(item.acceptance_id for item in plan.acceptances)
    sources = {
        (item.source.message_id, item.source.content_sha256)
        for item in plan.acceptances
    }
    if len(sources) != 1:
        issues.append(L1VerificationIssue("plan_message_source_mismatch"))
    for reference in references:
        result_id = reference.tool_result_id
        if result_id:
            result = evidence.get(result_id)
            if result is None:
                issues.append(
                    L1VerificationIssue(
                        "unknown_supporting_tool_result",
                        None,
                        result_id,
                    )
                )
                continue
            if result.status != "succeeded":
                issues.append(
                    L1VerificationIssue(
                        "supporting_tool_result_not_succeeded",
                        None,
                        result_id,
                    )
                )
            if result.tool_id in EXECUTION_FINDINGS_TOOL_IDS:
                issues.append(
                    L1VerificationIssue(
                        "execution_findings_result_not_admissible_evidence",
                        None,
                        result_id,
                    )
                )
            if result.tool_id in TOOL_HISTORY_TOOL_IDS:
                issues.append(
                    L1VerificationIssue(
                        "tool_history_receipt_not_admissible_evidence",
                        None,
                        result_id,
                    )
                )

    if not reply.strip():
        issues.append(L1VerificationIssue("final_reply_empty"))

    if not issues and references:
        try:
            project_referenced_results(references=references, execution=execution)
        except L1EvidenceProjectionError as exc:
            issues.append(L1VerificationIssue(str(exc)))

    return L1VerificationResult(
        contract_version=L1_VERIFICATION_CONTRACT_VERSION,
        passed=not issues,
        issues=tuple(issues),
        checked_acceptances=len(acceptance_ids),
        checked_tool_results=len(evidence),
    )


def protected_l1_acceptance_ids(
    *,
    plan: L1Plan,
    execution: Mapping[str, object],
) -> frozenset[str]:
    """任一 ToolCall 准入后冻结当前 Acceptance 标识。"""

    evidence = _load_tool_result_evidence(execution)
    if not evidence:
        return frozenset()
    return frozenset(item.acceptance_id for item in plan.acceptances)


def _load_tool_result_evidence(
    execution: Mapping[str, object],
) -> dict[str, _ToolResultEvidence]:
    raw_calls = execution.get("tool_calls")
    if not isinstance(raw_calls, list):
        raise L1VerificationStateError(
            "L1 execution projection omitted its ToolCall list"
        )
    evidence: dict[str, _ToolResultEvidence] = {}
    for raw in raw_calls:
        if not isinstance(raw, Mapping):
            raise L1VerificationStateError("L1 ToolCall projection has the wrong shape")
        tool_call_id = raw.get("tool_call_id")
        tool_id = raw.get("tool_id")
        status = raw.get("status")
        result_sha256 = raw.get("outcome_hash")
        if (
            not isinstance(tool_call_id, str)
            or not tool_call_id
            or not isinstance(tool_id, str)
            or not tool_id
            or not isinstance(status, str)
        ):
            raise L1VerificationStateError(
                "L1 ToolCall projection omitted verification facts"
            )
        if status == "pending":
            result_id = f"pending:{tool_call_id}"
        else:
            if not isinstance(result_sha256, str) or len(result_sha256) != 64:
                raise L1VerificationStateError(
                    "settled L1 ToolCall omitted its result sha256"
                )
            result_id = l1_tool_result_id(
                tool_call_id=tool_call_id,
                result_sha256=result_sha256,
            )
        if result_id in evidence:
            raise L1VerificationStateError(
                "L1 execution projection contains duplicate ToolResult IDs"
            )
        evidence[result_id] = _ToolResultEvidence(
            tool_result_id=result_id,
            tool_call_id=tool_call_id,
            tool_id=tool_id,
            status=status,
        )
    return evidence


__all__ = [
    "L1_VERIFICATION_CONTRACT_VERSION",
    "L1VerificationIssue",
    "L1VerificationResult",
    "L1VerificationStateError",
    "protected_l1_acceptance_ids",
    "verify_l1_final_reply",
]
