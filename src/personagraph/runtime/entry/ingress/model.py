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
    runtime_model_output_repair_issue_sort_key,
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


_CLASSIFIER_SYSTEM_PROMPT_BODY = """你是 Entelecheia 的入口分类器。这一步为本轮选择处理等级；只有下面明确启用长期任务匹配时，才提出 task_matches。你不回答用户、不调用工具、不创建持久任务、不生成任务图；输出只是待 Host 校验的提案。

## 一、你会收到什么

| 字段 | 是否必有 | 内容 |
|---|---|---|
| current_user_text | 必有 | 用户本轮原文，是判断意图的唯一依据 |
| allowed_processing_levels | 必有 | Host 本轮允许的等级，只能从中选择 |
| history_pairs | 可为空 | 之前轮次的对话 |
| session_summary | 可为空 | 更早对话的摘要 |
| new_files | 可为空 | Host 已确认的本轮文件元数据：media_type、kind、access |
| host_facts | 必有 | attachment_count、session_summary_status；非 L1 模式另含长期任务控制状态 |
| recent_incomplete_turn | 可为空 | 上一轮未正常结束的事实 |

关于 new_files：它只表示 Host 已确认文件存在，并登记了类型与按需访问状态；它不代表用户意图，不得据此补写用户没有在 current_user_text 中表达的任务目标。access=on_demand 表示入口阶段尚未读取内容。

## 二、怎么判

选等级：依据用户本轮实际要什么，只能从 allowed_processing_levels 中选，不得自行升降级。若用户只询问数量、名称、类型或大小等清单元数据，可选 L0；若用户要求读取、总结、引用、比较或以其他方式理解文件内容，且 L1 在允许范围内，必须选 L1 以使用文件 candidate 工具。"""

_CLASSIFIER_TASK_MATCH_PROMPT = """本轮启用长期任务匹配。task_catalog 是可关联的已有根任务目录；task_catalog_truncated 为 true 表示目录被截断，不得猜测未列出的 ID。host_facts 另含 has_pending_decision、has_active_work_run 两项长期任务控制状态。

匹配任务：每项只能是以下四种结构之一。
- 新的独立根任务：{"match_type":"new_root","local_key":"本次输出内唯一的临时键","title":"短标题","objective":"目标摘要","source_excerpt":"当前用户原文中的唯一连续片段"}
- 关联已有根任务：{"match_type":"existing_root","insession_task_id":"task_catalog 中的 ID","source_excerpt":"当前用户原文中的唯一连续片段","execute_current":true或false}
- 已有根任务下的新分支意图：{"match_type":"existing_root_branch","insession_task_id":"task_catalog 中的根任务 ID","branch_key":"本次输出内唯一的临时键","branch_summary":"分支意图摘要","source_excerpt":"当前用户原文中的唯一连续片段","execute_current":true或false}
- 用户明确要求替换已有根任务目标并立即按新目标执行：{"match_type":"existing_root_target_change","insession_task_id":"task_catalog 中的根任务 ID","replacement_objective":"替换后的完整目标摘要","source_excerpt":"当前用户原文中明确表达目标替换的唯一连续片段","execute_current":true}

共同规则：
- source_excerpt 必须逐字来自 current_user_text，且在其中只出现一次；重复短语要扩成能唯一定位的更长片段。
- 最多 24 项，每次输出最多包含 3 个 new_root。超出时优先合并属于同一用户目标的要求，不得通过更换 local_key 重复提案。
- new_root 只用于相互独立的新目标；相关要求应归入同一个根任务，不要按背景、偏好、约束、疑问机械拆根。
- existing_root、existing_root_branch 和 existing_root_target_change 的 insession_task_id 只能从 task_catalog 选择；不得猜测被截断或未展示的 ID。task_catalog 为空时不得输出这三种匹配。
- local_key 与 branch_key 只在本次输出内使用，必须以小写字母开头，仅含小写字母、数字、下划线或连字符。
- new_root、existing_root_branch、existing_root_target_change、以及 execute_current=true 的 existing_root 必须选择 L2。L0 的 task_matches 只能为空，或只含 execute_current=false 的 existing_root。L1 的 task_matches 必须为空。
- existing_root_branch 只是"本轮可能要在该根任务下形成分支"的入口意图，不代表已经创建了节点。

关于 existing_root_target_change：只用于用户原文明示要替换整个根目标的情况。普通继续、补充背景、增加约束、回答待答问题或新增分支，都必须继续使用 existing_root 或 existing_root_branch，绝不能推断成目标变更。replacement_objective 必须是用户本轮明确要求的新目标，execute_current 必须为 true，不得从旧目标、附件名或 Host 状态自行补写。

关于 execute_current：它只表示用户本轮要求继续执行这个已有根任务，不表示用户已经充分回答了待答问题，也不授权你选择执行单元或检查点。
- 当 task_catalog 中某个根任务带有 pending_user_question，且 current_user_text 在语义上直接回答了该问题时，应为该 existing_root 输出 execute_current=true；短答案不需要重复"继续任务"。
- 用户只是询问、讨论、补充背景或转向其他任务时，execute_current 必须为 false。弹窗来源和待答问题本身不能替用户证明 true；只有当前原文确实回答问题或明确要求继续时才可为 true。

有任务关联时的输出示例：
{"processing_level":"L2","task_matches":[{"match_type":"new_root","local_key":"q4_report","title":"整理季度报告","objective":"汇总本季度销售数据并指出异常","source_excerpt":"..."}]}
示例中的 source_excerpt 与 ID 必须换成当前输入中的真实值。"""

_CLASSIFIER_OUTPUT_PROMPT = """## 三、输出什么

只输出一个 JSON object，键且仅键为 processing_level、task_matches。不要输出解释、Markdown 或额外字段。

没有任务关联时：
{"processing_level":"L0","task_matches":[]}

字段含义：
- processing_level：本轮等级，必须出自 allowed_processing_levels。
- task_matches：任务匹配提案数组；未启用匹配或没有任务关联时输出空数组。"""

_ENTRY_ROUTER_MAX_OUTPUT_TOKENS = 32_768


def classify_turn(
    context: EntryContext,
    emit: EntryEventEmitter,
    deadline: TurnDeadline | None = None,
) -> EntryClassification:
    """Router 模型调用：把有界 EntryContext 投影成 processing level / task-match 提案。

    输入包括当前文本、选定历史/摘要、恢复事实和 allowed_processing_levels；仅非 L1
    模式额外提供任务目录和长期任务控制状态。
    new_files 仅含粗粒度类型/访问事实，故意不含文件名和正文。绑定 ModelTier.ROUTER，
    经结构化解析、级别与任务引用守卫后才返回；允许 L1 并不等于强制选择 L1。
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
    if not context.routing_policy.allows("L1"):
        payload["host_facts"].update(
            has_pending_decision=bool(context.snapshot.pending_decision_id),
            has_active_work_run=bool(context.snapshot.active_run_id),
        )
        payload.update(
            task_catalog=[
                item.model_dump(mode="json", exclude_none=True)
                for item in context.task_catalog.items
            ],
            task_catalog_truncated=context.task_catalog.truncated,
        )
    system_prompt = _classifier_system_prompt(allowed_levels)
    user_content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    mock_payload = {"processing_level": "L0", "task_matches": []}
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
                    "only processing_level and task_matches."
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
        if context.routing_policy.allows("L1") and classification.task_matches:
            raise ModelOutputValidationError(
                "L1 mode cannot reference durable Tasks",
                repair_code="entry_classification.task_matches_disabled",
                safe_repair_reason="本轮未启用长期任务匹配，task_matches 必须为空数组。",
            )
        task_match_error_codes = (
            _entry_task_match_guard_error_codes(
                classification,
                authoritative_user_text=context.envelope.user_text or "",
                trusted_root_catalog=context.task_catalog,
            )
            if classification.task_matches
            else ()
        )
        if task_match_error_codes:
            raise ModelOutputValidationError(
                "entry task matches failed deterministic guard: "
                + ",".join(task_match_error_codes),
                repair_code="entry_classification.task_matches_rejected",
                safe_repair_reason=(
                    "task_matches 未通过来源锚点或任务目录校验；请只使用当前原文中"
                    "唯一连续的 source_excerpt 和 task_catalog 中真实存在的任务。"
                ),
                repair_issues=tuple(
                    RuntimeModelOutputRepairIssue(
                        category="host_guard",
                        code=f"entry_classification.task_matches.{code}",
                        paths=("/task_matches",),
                        safe_explanation=(
                            "task_matches 未通过来源锚点、唯一性、数量或任务目录校验。"
                        ),
                    )
                    for code in task_match_error_codes
                ),
                repair_issue_coverage=(
                    RuntimeModelOutputRepairIssueCoverage.COMPLETE
                ),
            )
        invalid_level_match_indexes = tuple(
            index
            for index, match in enumerate(classification.task_matches)
            if classification.processing_level == "L0"
            and (
                match.match_type != "existing_root"
                or bool(getattr(match, "execute_current", False))
            )
        )
        if invalid_level_match_indexes:
            raise ModelOutputValidationError(
                "L0 task matches may only reference an existing root",
                repair_code="entry_classification.level_task_match_mismatch",
                safe_repair_reason=(
                    "L0 只能只读引用 existing_root，且 execute_current 必须为 false"
                ),
                repair_issues=tuple(
                    sorted(
                        (
                            RuntimeModelOutputRepairIssue(
                                category="host_guard",
                                code=(
                                    "entry_classification."
                                    "level_task_match_mismatch"
                                ),
                                paths=(f"/task_matches/{index}",),
                                safe_explanation=(
                                    "L0 只能只读引用 existing_root，且 "
                                    "execute_current 必须为 false。"
                                ),
                            )
                            for index in invalid_level_match_indexes
                        ),
                        key=runtime_model_output_repair_issue_sort_key,
                    )
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
    task_matching = (
        "本轮是 L1 模式，只处理当前请求，不关联或续接长期任务；task_matches 必须为空数组。"
        if "L1" in allowed_levels
        else _CLASSIFIER_TASK_MATCH_PROMPT
    )
    return (
        f"{_CLASSIFIER_SYSTEM_PROMPT_BODY}\n\n"
        f"{task_matching}\n\n{_CLASSIFIER_OUTPUT_PROMPT}\n\n"
        f"{PLANNING_OUTPUT_LANGUAGE_CLAUSE}\n\n"
        f"本轮 Host 允许的 processing_level 仅为：{allowed}。\n"
        "L0：无需工具即可直接回答的闲聊、解释或情绪回应。\n"
        "L1：能在当前 Turn 内通过有界计划、工具和验证完成，不建立跨 Turn 任务图。\n"
        "L2：需要跨 Turn 持久任务生命周期、真正续接已有 Task，或长期任务图。\n"
        "不得输出未列入 allowed_processing_levels 的等级，也不得自行升降级。"
    )


def _entry_task_match_guard_error_codes(
    classification: EntryClassification,
    *,
    authoritative_user_text: str,
    trusted_root_catalog: object,
) -> tuple[str, ...]:
    """Validate L0 references without importing the L2 TaskGraph package."""

    if not classification.task_matches:
        return ()
    if classification.processing_level == "L2":
        from ....l2.task_graph.task_matching import guard_insession_task_matches

        guarded = guard_insession_task_matches(
            classification.task_matches_proposal(),
            authoritative_user_text=authoritative_user_text,
            trusted_root_catalog=trusted_root_catalog,
        )
        return tuple(code.value for code in guarded.error_codes)

    errors: set[str] = set()
    catalog_items = tuple(getattr(trusted_root_catalog, "items", ()))
    catalog_ids = [getattr(item, "insession_task_id", None) for item in catalog_items]
    if len(catalog_ids) != len(set(catalog_ids)):
        errors.add("duplicate_catalog_task_id")
    known_task_ids = set(catalog_ids)
    identities: set[tuple[object, ...]] = set()
    for match in classification.task_matches:
        task_id = getattr(match, "insession_task_id", None)
        source_excerpt = getattr(match, "source_excerpt", None)
        identity = (
            getattr(match, "match_type", None),
            task_id,
            source_excerpt,
        )
        if identity in identities:
            errors.add("duplicate_task_match")
        identities.add(identity)
        if not isinstance(task_id, str) or not task_id.strip():
            errors.add("blank_insession_task_id")
        elif task_id not in known_task_ids:
            errors.add("unknown_insession_task_id")
        if not isinstance(source_excerpt, str) or not source_excerpt.strip():
            errors.add("blank_source_excerpt")
            continue
        start = authoritative_user_text.find(source_excerpt)
        if start < 0:
            errors.add("source_excerpt_not_found")
        elif authoritative_user_text.find(source_excerpt, start + 1) >= 0:
            errors.add("source_excerpt_ambiguous")
    return tuple(sorted(errors))


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
