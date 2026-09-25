"""工具历史列表/正文读取的公开协议，不访问运行记录。"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field

from personagraph.persistent_turn_content.evidence import L1_CALL_REF_PATTERN
from personagraph.persistent_turn_content.tool_results import (
    DEFAULT_HISTORY_LIST_LIMIT,
    MAX_HISTORY_LIST_LIMIT,
    DEFAULT_HISTORY_READ_LIMIT,
    MAX_HISTORY_READ_LIMIT,
    MAX_HISTORY_OFFSET,
    ToolResultHistoryPage,
    ToolResultContentPage,
)
from ..contracts import ToolSourceDescriptor, ToolSourceKind, ToolSpec
from ..effects import (
    DataEgress,
    EffectAction,
    EffectDescriptor,
    EffectResource,
    EffectScopeKind,
    Idempotency,
    Reversibility,
    ToolEffectProfile,
)
from ..registration import ToolExecutionProfile, ToolHandler, ToolRegistration

TOOL_HISTORY_TOOL_IDS = ("list_tool_results", "read_tool_result")


class ListToolResultsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    offset: int = Field(default=0, ge=0, le=MAX_HISTORY_OFFSET)
    limit: int = Field(
        default=DEFAULT_HISTORY_LIST_LIMIT, ge=1, le=MAX_HISTORY_LIST_LIMIT
    )


class ReadToolResultInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    call_ref: str = Field(
        pattern=L1_CALL_REF_PATTERN,
        description="当前执行中原调用的短引用，如 c2.1；必须原样使用工具结果或目录提供的 call_ref。",
    )
    path: str = Field(default="", max_length=2000, description="业务内容的 JSON Pointer；默认根对象，正文如 /result/text。")
    offset: int = Field(default=0, ge=0, le=MAX_HISTORY_OFFSET, description="当前 path 内的起始偏移。")
    limit: int = Field(
        default=DEFAULT_HISTORY_READ_LIMIT, ge=1, le=MAX_HISTORY_READ_LIMIT,
        description="字符串按 Unicode 字符，数组按条目、对象按字段计数；另受单次字节预算保护。",
    )


def build_tool_history_registration(
    *,
    tool_id: str,
    handler: ToolHandler,
    effect_scope: str,
) -> ToolRegistration:
    if tool_id not in TOOL_HISTORY_TOOL_IDS or not effect_scope:
        raise ValueError("tool history requires known tool and exact execution scope")
    listing = tool_id == "list_tool_results"
    input_contract = ListToolResultsInput if listing else ReadToolResultInput
    output_contract = ToolResultHistoryPage if listing else ToolResultContentPage
    description = (
        "分页查看当前执行已保存的工具结果目录，包括失败结果，不重跑原工具。按原步骤、批内调用顺序排列；"
        "每项 source 给出原始工具名、状态和 call_ref 短引用；arguments_summary 是有界参数摘要，"
        "不是完整参数。通过 next_offset 续页。需要正文时调用 read_tool_result。"
        "失败的 findings 写入参数可能包含无效来源，目录会明确标为 arguments_unavailable 并省略；可回读 error 修正。"
        "引用依据时使用原始source身份，不能把本次目录调用的回执当成原始结果。"
        if listing
        else "按 call_ref（如 c2.1）回读本次执行的结构化业务结果，不重跑工具。path 为 JSON Pointer，"
        "默认根对象；正文可用 /result/text，错误可用 /error。offset/limit 对字符串按 Unicode 字符，"
        "数组按条目、对象按字段计数。next_offset 续页；expand_paths 指向当前页放不下的完整成员，"
        "可再次指定 path 深入读取。原始 source 身份用于引用；错误不是成功事实证据。"
        "findings 写入工具的历史回执只展示带 compacted 标记的简要回执，不回读旧 active_projection 正文；"
        "该页 source_content_compacted=true 且 partial=true，当前笔记正文由 execution_findings 统一提供。"
    )
    return ToolRegistration(
        spec=ToolSpec(
            tool_id=tool_id,
            contract_version="tool-history-v3",
            name="查看工具结果目录" if listing else "回读工具结果正文",
            description=description,
            input_schema=input_contract.model_json_schema(),
            output_schema=output_contract.model_json_schema(),
            catalog_tags=("execution", "read"),
        ),
        implementation_version="3",
        source=ToolSourceDescriptor(
            kind=ToolSourceKind.LOCAL,
            source_id="personagraph.tools.tool_history",
            fingerprint=hashlib.sha256(f"{tool_id}:3".encode()).hexdigest(),
        ),
        handler=handler,
        effect_profile=ToolEffectProfile(
            (
                EffectDescriptor(
                    resource=EffectResource.RUNTIME_STATE,
                    action=EffectAction.READ,
                    scope_kind=EffectScopeKind.EXECUTION,
                    default_scope=effect_scope,
                    data_egress=DataEgress.METADATA if listing else DataEgress.CONTENT,
                    idempotency=Idempotency.IDEMPOTENT,
                    reversibility=Reversibility.REVERSIBLE,
                ),
            )
        ),
        execution_profile=ToolExecutionProfile(
            default_timeout_s=10,
            hard_timeout_s=30,
            max_output_bytes=512_000,
            max_transparent_retries=0,
            concurrency_class="tool_history_read",
        ),
    )
