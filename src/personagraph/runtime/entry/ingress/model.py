"""Entry ingress 的有界模型分类调用。"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from ..context.attachments import AttachmentProjection
from ..context.contracts import EntryContext
from personagraph.model_io.gateway import (
    DEFAULT_MODEL_TIMEOUT_S,
    ModelResult,
    complete_structured,
)
from personagraph.model_io.tier_bindings import ModelTier, resolve_tier
from .model_contracts import EntryClassification
from ...model_calls.policy import MAX_MODEL_ATTEMPTS
from ...model_calls.requests import request_model_with_retry
from ....model_io.output_validation import ModelOutputValidationError
from ....model_io.output_repair_contracts import (
    RuntimeModelOutputRepairIssue,
    RuntimeModelOutputRepairIssueCoverage,
)
from ....model_io.prepared_structured_provider import (
    prepare_structured_repair_request,
    prepare_structured_request,
)
from ....model_io.structured_output_repair import (
    project_validation_error_issues,
    safe_validation_error_reason,
)
from ...turn.contracts import ProcessingLevel
from ...turn_events import EntryEventEmitter, RuntimeStage
from ...turn_deadline import TurnDeadline
from personagraph.model_io.output_language import PLANNING_OUTPUT_LANGUAGE_CLAUSE


_CLASSIFIER_SYSTEM_PROMPT_BODY = """你是 Entelecheia 的入口分类器。这一步只为本轮选择处理等级。你不回答用户、不调用工具、不创建持久任务、不生成任务图；输出只是待 Host 校验的提案。

## 一、你会收到什么

| 字段 | 是否必有 | 内容 |
|---|---|---|
| current_user_text | 必有 | 用户本轮原文，是判断意图的唯一依据 |
| allowed_processing_levels | 必有 | Host 本轮允许的等级，只能从中选择 |
| history_pairs | 可为空 | 之前轮次的对话 |
| session_summary | 可为空 | 更早对话的摘要 |
| new_files | 可为空 | Host 已确认的本轮文件元数据：media_type、kind、access |
| host_facts | 必有 | attachment_count、session_summary_status |
| recent_incomplete_turn | 可为空 | 上一轮未正常结束的事实 |

关于 new_files：它只表示 Host 已确认文件存在，并登记了类型与按需访问状态；它不代表用户意图，不得据此补写用户没有在 current_user_text 中表达的任务目标。access=on_demand 表示入口阶段尚未读取内容。

## 二、怎么判

选等级：依据用户本轮实际要什么，只能从 allowed_processing_levels 中选，不得自行升降级。若用户只询问数量、名称、类型或大小等清单元数据，可选 L0；若用户要求读取、总结、引用、比较或以其他方式理解文件内容，且 L1 在允许范围内，必须选 L1 以使用文件 candidate 工具。"""


_CLASSIFIER_OUTPUT_PROMPT = """## 三、输出什么

只输出一个 JSON object，键且仅键为 processing_level。不要输出解释、Markdown 或额外字段。

示例：
{"processing_level":"L0"}

字段含义：
- processing_level：本轮等级，必须出自 allowed_processing_levels。"""

_ENTRY_ROUTER_MAX_OUTPUT_TOKENS = 32_768


def classify_turn(
    context: EntryContext,
    emit: EntryEventEmitter,
    deadline: TurnDeadline | None = None,
) -> EntryClassification:
    """Router 模型调用：把有界 EntryContext 投影成 processing level 提案。

    输入包括当前文本、选定历史/摘要、恢复事实和 allowed_processing_levels。
    new_files 仅含粗粒度类型/访问事实，故意不含文件名和正文。绑定 ModelTier.ROUTER，
    经结构化解析与级别守卫后才返回；允许 L1 并不等于强制选择 L1。
    此处不执行工具；实际 lane 分派由 Entry 的 admission / routing 接续完成。
    """
    allowed_levels = context.routing_policy.allowed_processing_levels
    payload = {
        "current_user_text": context.envelope.user_text,
        "history_pairs": context.history_pairs,
        "session_summary": context.session_summary,
        "new_files": _classifier_attachment_facts(context.attachments),
        "host_facts": {
            "attachment_count": len(context.envelope.attachments),
            "session_summary_status": context.session_summary_status,
        },
        "recent_incomplete_turn": _recovery_fact(context),
        "allowed_processing_levels": allowed_levels,
    }
    system_prompt = _classifier_system_prompt(allowed_levels)
    user_content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    mock_payload = {"processing_level": "L0"}
    structured_provider = complete_structured
    # Classification 属于共享 ingress 权威，而非任一执行 lane。一次性冻结其 Router
    # 绑定，使每次物理重试与修复都使用同一 endpoint，并实际携带已配置 thinking 控制。
    model_binding = resolve_tier(ModelTier.ROUTER)

    prepare_request = prepare_structured_request(
        structured_provider,
        system_prompt=system_prompt,
        user_content=user_content,
        purpose="runtime_entry_classify",
        prepare_kwargs=lambda: {
            "mock_payload": mock_payload,
            "max_tokens": _ENTRY_ROUTER_MAX_OUTPUT_TOKENS,
            "timeout_s": DEFAULT_MODEL_TIMEOUT_S,
            "json_mode": True,
            "binding": model_binding,
        },
    )

    def validate(result: ModelResult) -> EntryClassification:
        try:
            classification = EntryClassification.model_validate(json.loads(result.reply))
        except ValidationError as exc:
            projection = project_validation_error_issues(
                exc,
                contract=EntryClassification,
            )
            raise ModelOutputValidationError(
                "invalid entry classification",
                repair_code="entry_classification.contract_invalid",
                safe_repair_reason=safe_validation_error_reason(
                    exc,
                    contract=EntryClassification,
                    fallback=(
                        "The response violates the entry classification contract."
                    ),
                ),
                repair_issues=projection.issues,
                repair_issue_coverage=projection.issue_coverage,
                omitted_repair_issue_count=projection.omitted_issue_count,
            ) from exc
        except json.JSONDecodeError as exc:
            raise ModelOutputValidationError(
                "invalid entry classification",
                repair_code="entry_classification.invalid_json",
                safe_repair_reason=(
                    "Return one complete syntactically valid JSON object containing "
                    "only processing_level."
                ),
                repair_issues=(
                    RuntimeModelOutputRepairIssue(
                        category="json_syntax",
                        code="json_syntax.invalid_json",
                        paths=("",),
                        json_line=exc.lineno,
                        json_column=exc.colno,
                        safe_explanation="输出不是完整合法的 JSON object。",
                    ),
                ),
                repair_issue_coverage=(
                    RuntimeModelOutputRepairIssueCoverage.FIRST_ONLY
                ),
            ) from exc
        except TypeError as exc:
            raise ModelOutputValidationError(
                "invalid entry classification",
                repair_code="entry_classification.response_not_text",
                safe_repair_reason="Return one complete JSON object encoded as text.",
                repair_issues=(
                    RuntimeModelOutputRepairIssue(
                        category="json_syntax",
                        code="json_syntax.response_not_text",
                        paths=("",),
                        safe_explanation="输出必须是一份 JSON object 文本。",
                    ),
                ),
            ) from exc
        if not context.routing_policy.allows(classification.processing_level):
            raise ModelOutputValidationError(
                "entry classification selected a disabled processing level",
                repair_code="entry_classification.level_disabled",
                safe_repair_reason=(
                    "processing_level 未启用；allowed_processing_levels="
                    + ",".join(allowed_levels)
                ),
                repair_issues=(
                    RuntimeModelOutputRepairIssue(
                        category="host_guard",
                        code="entry_classification.level_disabled",
                        paths=("/processing_level",),
                        safe_explanation=(
                            "processing_level 必须从本轮 Host 允许的等级中选择。"
                        ),
                    ),
                ),
                repair_issue_coverage=(
                    RuntimeModelOutputRepairIssueCoverage.COMPLETE
                ),
            )
        return classification

    prepare_repair_request = prepare_structured_repair_request(
        structured_provider,
        system_prompt=system_prompt,
        user_content=user_content,
        purpose="runtime_entry_classify",
        prepare_kwargs=lambda: {
            "mock_payload": mock_payload,
            "max_tokens": _ENTRY_ROUTER_MAX_OUTPUT_TOKENS,
            "timeout_s": DEFAULT_MODEL_TIMEOUT_S,
            "json_mode": True,
            "binding": model_binding,
        },
    )
    return request_model_with_retry(
        turn_id=context.envelope.turn_id,
        session_id=context.envelope.session_id,
        purpose="runtime_entry_classify",
        stage=RuntimeStage.CLASSIFY,
        prepare_request=prepare_request,
        prepare_repair_request=prepare_repair_request,
        repair_target_contract="entry-classification-v1",
        validate=validate,
        emit=emit,
        max_attempts=MAX_MODEL_ATTEMPTS,
        deadline=deadline,
    ).value


def _classifier_system_prompt(
    allowed_levels: tuple[ProcessingLevel, ...],
) -> str:
    allowed = ", ".join(allowed_levels)
    route_guidance = "只处理当前请求。"
    return (
        f"{_CLASSIFIER_SYSTEM_PROMPT_BODY}\n\n"
        f"{route_guidance}\n\n{_CLASSIFIER_OUTPUT_PROMPT}\n\n"
        f"{PLANNING_OUTPUT_LANGUAGE_CLAUSE}\n\n"
        f"本轮 Host 允许的 processing_level 仅为：{allowed}。\n"
        "L0：无需工具即可直接回答的闲聊、解释或情绪回应。\n"
        "L1：能在当前 Turn 内通过有界计划、工具和验证完成，不建立跨 Turn 任务图。\n"
        "不得输出未列入 allowed_processing_levels 的等级，也不得自行升降级。"
    )


def _classifier_attachment_facts(projection: AttachmentProjection) -> list[dict[str, Any]]:
    """用于路由决策、关于本 turn 文件的可信 host 事实。

    classifier 绝不会看到附件内容——路由只需知道文件存在及其粗粒度类型。
    文件名有意省略，因为它们是作者控制的文本，并非用户当前意图的证据。
    """

    return [
        {
            "media_type": item.media_type,
            "kind": item.kind.value,
            "access": item.access.value,
        }
        for item in projection.items
    ]


def _recovery_fact(context: EntryContext) -> dict[str, str] | None:
    """只向 route classifier 暴露安全的中断元数据。"""

    recovery = context.recovery_projection
    if recovery is None:
        return None
    return {
        "turn_id": recovery.turn_id,
        "end_reason": recovery.end_reason,
        "error_code": recovery.error_code or "",
    }
