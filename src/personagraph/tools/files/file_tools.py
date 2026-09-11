"""直接路径/File ID 的状态检查与批量准备协议，不拥有文件业务。"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from ..contracts import ToolSourceDescriptor, ToolSourceKind, ToolSpec
from ..effects import (
    DataEgress, EffectAction, EffectDescriptor, EffectResource, EffectScopeKind,
    Idempotency, Reversibility, ToolEffectProfile,
)
from ..registration import CancellationMode, ToolExecutionProfile, ToolRegistration

CHECK_FILES_STATE_TOOL_ID = "check_files_state"
PREPARE_FILES_TOOL_ID = "prepare_files"
FILE_TOOLS_CONTRACT_VERSION = "file-state-v1"
MAX_FILES = 64


def derive_file_preparation_tool_request_id(logical_tool_call_id: str, input_index: int) -> str:
    """为持久 ToolCall 的一个原始输入项绑定唯一文档准备请求。"""
    if not isinstance(logical_tool_call_id, str) or not logical_tool_call_id.strip():
        raise ValueError("logical_tool_call_id must be a non-empty string")
    if type(input_index) is not int or not 0 <= input_index < MAX_FILES:
        raise ValueError("input_index must identify one file input")
    digest = hashlib.sha256(f"{logical_tool_call_id}:{input_index}".encode()).hexdigest()
    return f"file-tool-request:{digest}"


def build_file_state_registration(
    *, tool_id: str, handler: Callable, effect_scope: str,
) -> ToolRegistration:
    if tool_id not in {CHECK_FILES_STATE_TOOL_ID, PREPARE_FILES_TOOL_ID}:
        raise ValueError("unknown file state tool")
    prepares = tool_id == PREPARE_FILES_TOOL_ID
    implementation_version = "4" if prepares else "2"
    effects = [EffectDescriptor(
        resource=EffectResource.FILESYSTEM, action=EffectAction.READ,
        scope_kind=EffectScopeKind.SESSION, default_scope=effect_scope,
        data_egress=DataEgress.METADATA, idempotency=Idempotency.IDEMPOTENT,
        reversibility=Reversibility.REVERSIBLE,
    )]
    if prepares:
        effects.append(EffectDescriptor(
            resource=EffectResource.RUNTIME_STATE, action=EffectAction.UPDATE,
            scope_kind=EffectScopeKind.EXECUTION, default_scope=effect_scope,
            data_egress=DataEgress.NONE, idempotency=Idempotency.DEDUPLICATED,
            reversibility=Reversibility.REVERSIBLE,
        ))
    description = (
        "批量显式解析指定文件并建立当前会话可检索的内容。每项直接给工作区相对路径"
        "或真实 file_id，不要求先检查；可选 file_version_id 防止处理错误版本。"
        "复用相同源版本与处理配方的共享任务；仅对本次指定文件登记、解析、索引及挂载。"
        "后台处理中会挂起本次调用，准备及当前会话挂载完成后才返回；"
        "无需反复调用 check_files_state 等待。等待沿用 Host 的本轮剩余时间与取消信号，"
        "不额外消耗模型 Attempt；结束等待不会取消其他会话共享的后台任务。"
        "ready_indices 表示准备完成、已可检索；不代表已经读取正文，也不是仍待解析。"
        "相同文件版本已 ready 时无需重复 prepare_files，应按任务需要检索或读取正文。"
        "not_ready_indices 包含处理中、变化和失败项，具体原因逐项返回。"
        "reason_code 明确标为 terminal 的失败，在文件版本与处理条件未改变时"
        "原样重试不会恢复；应选择其他材料或能力，或说明文件无法读取造成的信息缺口。"
        "它不修改原文件，也不自动调用外部视觉模型。"
        if prepares else
        "只读检查指定工作区相对路径或真实 File ID 的当前内容版本、"
        "文本入库和本会话可用性；可选 file_version_id 检查精确版本。"
        "不创建 File、任务或挂载，不解析文件。返回 ready、changed、not_ingested、"
        "pending、partial 或 unavailable，并给出原因与现有真实身份；"
        "未登记文件的 file_id 为 null。检查不是 prepare_files 的强制前置步骤。"
    )
    return ToolRegistration(
        spec=ToolSpec(
            tool_id=tool_id, contract_version=FILE_TOOLS_CONTRACT_VERSION,
            name="Prepare files" if prepares else "Check files state",
            description=description, input_schema=file_input_schema(),
            output_schema=file_output_schema(), catalog_tags=("file", "document", "status"),
        ),
        implementation_version=implementation_version,
        source=ToolSourceDescriptor(
            kind=ToolSourceKind.LOCAL, source_id="personagraph.tools.files",
            fingerprint=hashlib.sha256(
                f"{tool_id}:{FILE_TOOLS_CONTRACT_VERSION}:{implementation_version}".encode(),
            ).hexdigest(),
        ),
        handler=handler, effect_profile=ToolEffectProfile(tuple(effects)),
        execution_profile=ToolExecutionProfile(
            default_timeout_s=None if prepares else 45,
            hard_timeout_s=None if prepares else 90,
            max_output_bytes=100_000, max_transparent_retries=0 if prepares else 1,
            cancellation_mode=CancellationMode.COOPERATIVE if prepares else CancellationMode.NONE,
            concurrency_class="file_preparation" if prepares else "file_state_read",
        ),
    )


def file_input_schema() -> dict:
    identifier = {"type": "string", "minLength": 1, "maxLength": 128}
    target = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "path": {
                "type": "string",
                "minLength": 1,
                "maxLength": 2000,
                "description": (
                    "工作区相对路径；优先使用 file_catalog.relative_path 或目录工具"
                    "返回的 path。name 只是显示文件名，不等于 path；只有文件确在"
                    "工作区根目录时，文件名本身才可作为完整相对路径。"
                ),
            },
            "file_id": identifier, "file_version_id": identifier,
        },
        "oneOf": [{"required": ["path"]}, {"required": ["file_id"]}],
    }
    return {
        "type": "object", "additionalProperties": False, "required": ["files"],
        "properties": {"files": {
            "type": "array", "minItems": 1, "maxItems": MAX_FILES,
            "uniqueItems": True, "items": target,
        }},
    }


def file_output_schema() -> dict:
    nullable_id = {"type": ["string", "null"], "minLength": 1, "maxLength": 256}
    fields = {
        "input_index": {"type": "integer", "minimum": 0, "maximum": MAX_FILES - 1},
        "status": {"enum": ["ready", "changed", "not_ingested", "pending", "partial", "unavailable"]},
        "file_id": nullable_id, "file_version_id": nullable_id,
        "file_name": {"type": ["string", "null"], "maxLength": 512},
        "relative_path": {"type": ["string", "null"], "maxLength": 2000},
        "document_id": nullable_id, "document_version_id": nullable_id,
        "reason_code": nullable_id, "reused": {"type": "boolean"},
    }
    index_list = {"type": "array", "maxItems": MAX_FILES, "uniqueItems": True,
                  "items": {"type": "integer", "minimum": 0, "maximum": MAX_FILES - 1}}
    properties = {
        "contract_version": {"const": FILE_TOOLS_CONTRACT_VERSION},
        "results": {"type": "array", "minItems": 1, "maxItems": MAX_FILES, "items": {
            "type": "object", "additionalProperties": False,
            "required": list(fields), "properties": fields,
        }},
        "ready_indices": index_list, "not_ready_indices": index_list,
    }
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}
