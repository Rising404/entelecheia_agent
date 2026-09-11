"""模型请求的安全拒绝诊断与公开错误分类。"""

from __future__ import annotations

import hashlib
import json
import logging

from personagraph.model_io.gateway import ModelGatewayError, ModelResult
from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairFeedback,
)

from ..turn_events import RuntimeErrorCode


_LOG = logging.getLogger(__name__)


def record_model_output_rejection(
    *,
    purpose: str,
    turn_id: str,
    session_id: str | None,
    model_call_id: str,
    logical_model_call_id: str,
    physical_ordinal: int,
    feedback: RuntimeModelOutputRepairFeedback | None,
    fallback_code: str,
    model_result: ModelResult | None,
    repair_scheduled: bool,
) -> None:
    """记录安全拒绝元数据，不复制被拒的模型正文。"""

    try:
        from personagraph.trajectory import record_rejected_output

        rejected_sha256 = (
            feedback.rejected_response_sha256
            if feedback is not None
            else _diagnostic_rejected_response_sha256(model_result)
        )
        observation: dict[str, object] = {
            "schema_version": "runtime-model-output-rejection-observation-v1",
            "purpose": purpose,
            "logical_model_call_id": logical_model_call_id,
            "physical_ordinal": physical_ordinal,
            "rejected_response_sha256": rejected_sha256,
            "repair_scheduled": repair_scheduled,
        }
        if isinstance(feedback, RuntimeModelOutputRepairFeedback):
            observation.update(
                {
                    "target_contract": feedback.target_contract,
                    "issue_coverage": feedback.issue_coverage.value,
                    "omitted_issue_count": feedback.omitted_issue_count,
                    "issues": [
                        {
                            "category": issue.category.value,
                            "code": issue.code,
                            "paths": list(issue.paths),
                            "safe_explanation": issue.safe_explanation,
                        }
                        for issue in feedback.current_issues
                    ],
                }
            )
        else:
            observation.update(
                {
                    "target_contract": None,
                    "issue_coverage": "first_only",
                    "omitted_issue_count": 0,
                    "issues": [
                        {
                            "category": "host_guard",
                            "code": fallback_code,
                            "paths": [""],
                        }
                    ],
                }
            )
        record_rejected_output(
            stage=f"model_output_validation:{purpose}",
            rejected=json.dumps(
                observation,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            reason_code=fallback_code,
            session_id=session_id,
            turn_id=turn_id,
            model_call_id=model_call_id,
        )
    except Exception as exc:
        # 诊断绝不能改变模型调用结果；同时防护 recorder 导入和序列化失败。
        _LOG.error(
            "model output rejection trajectory failed purpose=%s turn_id=%s "
            "error_type=%s",
            purpose,
            turn_id,
            type(exc).__name__,
        )
        return


def record_terminal_model_failure(
    *,
    purpose: str,
    turn_id: str,
    session_id: str | None,
    model_call_id: str,
    reason_code: str,
    attempts: int,
    duration_ms: int,
) -> None:
    """把共享重试包装器的最终失败投影为一条无正文 trajectory。"""

    try:
        from personagraph.trajectory import record_model_request_failure

        record_model_request_failure(
            model_call_id=model_call_id,
            purpose=purpose,
            reason_code=reason_code,
            attempts=attempts,
            duration_ms=duration_ms,
            session_id=session_id,
            turn_id=turn_id,
        )
    except Exception as exc:
        # 观测失败不能替换原本即将抛出的模型失败。
        _LOG.error(
            "terminal model failure trajectory failed purpose=%s turn_id=%s "
            "model_call_id=%s error_type=%s",
            purpose,
            turn_id,
            model_call_id,
            type(exc).__name__,
        )
        return


def runtime_error_code(error: ModelGatewayError) -> RuntimeErrorCode:
    """把 provider 错误投影为稳定、无内容的 Runtime 事件分类。"""

    if error.code == "TURN_DEADLINE_EXCEEDED":
        return RuntimeErrorCode.TURN_DEADLINE_EXCEEDED
    if error.code == "MODEL_CALL_TIMEOUT":
        return RuntimeErrorCode.MODEL_TIMEOUT
    if error.code == "MODEL_BAD_RESPONSE":
        return RuntimeErrorCode.MODEL_OUTPUT_INVALID
    if not error.retryable and error.details.get("physical_error_retryable") is not True:
        return RuntimeErrorCode.MODEL_CONFIGURATION_FAILURE
    return RuntimeErrorCode.MODEL_TRANSPORT_FAILURE


def _diagnostic_rejected_response_sha256(result: ModelResult | None) -> str:
    """生成无正文诊断摘要；允许不可编码 surrogate 的确定性转义。"""

    reply = "" if result is None else result.reply
    return hashlib.sha256(
        reply.encode("utf-8", errors="backslashreplace")
    ).hexdigest()


__all__ = [
    "record_model_output_rejection",
    "record_terminal_model_failure",
    "runtime_error_code",
]
