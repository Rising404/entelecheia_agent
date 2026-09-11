"""完整 Task 根 Delivery 结构化验证的重试外观。

固定提示词、不透明模型契约标识及严格回复解码位于
:mod:`personagraph.l2.task_execution.delivery.model_contracts`；此外观持有持久调用、重试、事件以及
现行仅根节点执行生命周期。
"""

from __future__ import annotations

import json
from typing import Callable

from pydantic import ValidationError

from personagraph.l2.task_graph import (
    TaskDeliveryValidationDisposition,
    TaskDeliveryValidationRequest,
    TaskDeliveryValidationResult,
    require_root_only_task_delivery_execution_retry,
)
from personagraph.model_io.gateway import ModelResult
from personagraph.runtime.model_calls.contracts import DurableLogicalModelCallAuthority
from personagraph.model_io.output_repair_contracts import (
    RuntimeModelOutputRepairIssue,
    RuntimeModelOutputRepairIssueCategory,
    RuntimeModelOutputRepairIssueCoverage,
)
from personagraph.model_io.output_validation import ModelOutputValidationError
from personagraph.runtime.model_calls.requests import (
    ModelRequestResult,
    request_model_with_retry,
)
from personagraph.runtime.turn_deadline import TurnDeadline
from personagraph.model_io.prepared_structured_provider import (
    durable_structured_provider_prompt,
    prepare_structured_repair_request,
    prepare_structured_request,
)
from personagraph.model_io.structured_output_repair import (
    project_validation_error_issues,
    safe_validation_error_reason,
)
from .model_contracts import (
    PURPOSE,
    REQUEST_CONTRACT,
    RESULT_CONTRACT,
    TaskDeliveryValidationStructuredProvider,
    _SYSTEM_PROMPT,
    decode_task_delivery_validation_response as _decode_response,
    task_delivery_validation_model_payload,
)
from personagraph.runtime.turn_events import RuntimeStage, TurnEvent


def _dispatch_task_delivery_validation(
    request: TaskDeliveryValidationRequest,
    *,
    provider: TaskDeliveryValidationStructuredProvider,
    emit: Callable[[TurnEvent], object],
    validate: Callable[[ModelResult], TaskDeliveryValidationResult],
    deadline: TurnDeadline | None,
    durable_call: DurableLogicalModelCallAuthority | None,
) -> ModelRequestResult[TaskDeliveryValidationResult]:
    """执行现行结构化调度、修复和持久模型调用生命周期。"""

    if durable_call is not None and (
        durable_call.semantic_call_id != request.logical_call_id
    ):
        raise ValueError("Task delivery validator durable call identity mismatch")
    user_content = json.dumps(
        request.prompt.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    system_prompt, user_content = durable_structured_provider_prompt(
        durable_call,
        system_prompt=_SYSTEM_PROMPT,
        user_content=user_content,
    )
    prepared_request = prepare_structured_request(
        provider,
        system_prompt=system_prompt,
        user_content=user_content,
        purpose=PURPOSE,
    )
    prepared_repair = prepare_structured_repair_request(
        provider,
        system_prompt=system_prompt,
        user_content=user_content,
        purpose=PURPOSE,
    )
    return request_model_with_retry(
        turn_id=request.invocation_turn_id,
        session_id=request.prompt.session_id,
        purpose=PURPOSE,
        stage=RuntimeStage.VERIFICATION,
        prepare_request=prepared_request,
        prepare_repair_request=prepared_repair,
        repair_target_contract=RESULT_CONTRACT,
        validate=validate,
        emit=emit,
        deadline=deadline,
        durable_call=durable_call,
        logical_model_call_id=request.logical_call_id,
    )


def request_task_delivery_validation(
    request: TaskDeliveryValidationRequest,
    *,
    provider: TaskDeliveryValidationStructuredProvider,
    emit: Callable[[TurnEvent], object],
    deadline: TurnDeadline | None = None,
    durable_call: DurableLogicalModelCallAuthority | None = None,
) -> ModelRequestResult[TaskDeliveryValidationResult]:
    """请求现行四路整任务审查，不直接变更持久化状态。"""

    def validate(model_result: ModelResult) -> TaskDeliveryValidationResult:
        try:
            validated = _decode_response(
                request=request,
                reply=model_result.reply,
            )
            if (
                validated.disposition
                is TaskDeliveryValidationDisposition.RETRY_EXECUTION
            ):
                require_root_only_task_delivery_execution_retry(
                    result=validated,
                    root_node_id=request.prompt.task_id,
                )
            return validated
        except json.JSONDecodeError as exc:
            raise ModelOutputValidationError(
                "invalid whole-Task delivery validation result",
                repair_code="task_delivery_validation_v2_contract_invalid",
                safe_repair_reason=(
                    "Return one complete syntactically valid JSON object covering "
                    "every required delivery validation dimension exactly once."
                ),
                repair_issues=(
                    RuntimeModelOutputRepairIssue(
                        category=(
                            RuntimeModelOutputRepairIssueCategory.JSON_SYNTAX
                        ),
                        code="json_syntax.task_delivery_validation.invalid_json",
                        paths=("",),
                        json_line=exc.lineno,
                        json_column=exc.colno,
                        safe_explanation=(
                            "输出不是完整且语法有效的 JSON object。"
                        ),
                    ),
                ),
                repair_issue_coverage=(
                    RuntimeModelOutputRepairIssueCoverage.FIRST_ONLY
                ),
            ) from exc
        except ValidationError as exc:
            projection = project_validation_error_issues(
                exc,
                contract=TaskDeliveryValidationResult,
            )
            raise ModelOutputValidationError(
                "invalid whole-Task delivery validation result",
                repair_code="task_delivery_validation_v2_contract_invalid",
                safe_repair_reason=safe_validation_error_reason(
                    exc,
                    contract=TaskDeliveryValidationResult,
                    fallback=(
                        "The response violates the whole-Task delivery "
                        "validation contract."
                    ),
                ),
                repair_issues=projection.issues,
                repair_issue_coverage=projection.issue_coverage,
                omitted_repair_issue_count=projection.omitted_issue_count,
            ) from exc
        except (TypeError, ValueError) as exc:
            error_text = str(exc)
            if error_text == "duplicate JSON object key":
                issue = RuntimeModelOutputRepairIssue(
                    category=RuntimeModelOutputRepairIssueCategory.JSON_SYNTAX,
                    code="json_syntax.task_delivery_validation.duplicate_key",
                    paths=("",),
                    safe_explanation="JSON object 不得包含重复键。",
                )
            elif error_text.startswith("non-finite JSON number is forbidden"):
                issue = RuntimeModelOutputRepairIssue(
                    category=RuntimeModelOutputRepairIssueCategory.JSON_SYNTAX,
                    code=(
                        "json_syntax.task_delivery_validation.non_finite_number"
                    ),
                    paths=("",),
                    safe_explanation="JSON 只能使用有限数字。",
                )
            elif error_text == (
                "delivery validation finding cites an unknown node"
            ):
                issue = RuntimeModelOutputRepairIssue(
                    category=RuntimeModelOutputRepairIssueCategory.HOST_GUARD,
                    code=(
                        "host_guard.task_delivery_validation."
                        "unknown_node_reference"
                    ),
                    paths=("/findings",),
                    safe_explanation=(
                        "affected_node_ids 只能引用冻结请求中已声明的 "
                        "node_id。"
                    ),
                )
            elif error_text == (
                "delivery validation finding cites an unknown anchor"
            ):
                issue = RuntimeModelOutputRepairIssue(
                    category=RuntimeModelOutputRepairIssueCategory.HOST_GUARD,
                    code=(
                        "host_guard.task_delivery_validation."
                        "unknown_evidence_reference"
                    ),
                    paths=("/findings",),
                    safe_explanation=(
                        "evidence_anchor_ids 只能引用冻结请求中已声明的 "
                        "anchor_id。"
                    ),
                )
            elif error_text == (
                "execution repair must affect only the canonical root node"
            ):
                issue = RuntimeModelOutputRepairIssue(
                    category=RuntimeModelOutputRepairIssueCategory.HOST_GUARD,
                    code=(
                        "host_guard.task_delivery_validation."
                        "execution_repair_not_root_only"
                    ),
                    paths=("/findings",),
                    safe_explanation=(
                        "execution_output 订正只能引用当前 canonical root；"
                        "child 缺陷必须分类为 task_graph_design。"
                    ),
                )
            else:
                issue = RuntimeModelOutputRepairIssue(
                    category=RuntimeModelOutputRepairIssueCategory.HOST_GUARD,
                    code="host_guard.task_delivery_validation.contract_invalid",
                    paths=("",),
                    safe_explanation=(
                        "输出未通过整任务交付验证的 Host 合同规则。"
                    ),
                )
            raise ModelOutputValidationError(
                "invalid whole-Task delivery validation result",
                repair_code="task_delivery_validation_v2_contract_invalid",
                safe_repair_reason=(
                    "Return one JSON object covering every required delivery "
                    "validation dimension exactly once with a valid fault domain."
                ),
                repair_issues=(issue,),
                repair_issue_coverage=(
                    RuntimeModelOutputRepairIssueCoverage.FIRST_ONLY
                ),
            ) from exc

    return _dispatch_task_delivery_validation(
        request,
        provider=provider,
        emit=emit,
        validate=validate,
        deadline=deadline,
        durable_call=durable_call,
    )


__all__ = [
    "PURPOSE",
    "REQUEST_CONTRACT",
    "RESULT_CONTRACT",
    "TaskDeliveryValidationStructuredProvider",
    "request_task_delivery_validation",
    "task_delivery_validation_model_payload",
]
