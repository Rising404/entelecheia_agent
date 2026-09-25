"""由 Host 派发的执行发现工具之模型可见契约。"""

from __future__ import annotations

from typing import Any

from ...persistent_turn_content.findings import (
    EXECUTION_FINDING_CLAIM_MAX_CHARACTERS,
    ExecutionFindingKind,
    ExecutionFindingSourceRef,
)
from ...persistent_turn_content.findings import (
    RECORD_EXECUTION_FINDINGS_TOOL_ID,
    REVISE_EXECUTION_FINDING_TOOL_ID,
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
from ..registration import ToolExecutionProfile, ToolRegistration


EXECUTION_FINDINGS_TOOL_CONTRACT_VERSION = "2.0.0"
EXECUTION_FINDINGS_TOOL_IMPLEMENTATION_VERSION = "host-runtime-state-v3"
EXECUTION_FINDINGS_SOURCE_ID = "personagraph.execution_findings"
EXECUTION_FINDINGS_SOURCE_DISPLAY_NAME = "Execution Findings Ledger"
EXECUTION_FINDINGS_DEFAULT_EFFECT_SCOPE = "current_execution_findings"

_CURRENT_MAX_SOURCE_REFS_PER_ENTRY = 16
_CURRENT_MAX_SCOPE_KEYS_PER_ENTRY = 24

_CURRENT_RECORD_NAME = "向任务台账追加记录"
_CURRENT_RECORD_DESCRIPTION = (
    "台账是当前运行期间持续保留的一份临时笔记，用于记录模型自己认为有价值的发现与观察、"
    "已做决定和尚未解决的问题；它不是长期记忆，也不等同于最终答案或已经核实的证据。"
    "本工具向台账追加 1–4 条新记录，不修改或撤回已有记录。L1 的版本绑定由 Host 完成。"
    "每条记录可以关联当前计划中的相关目标，并可附上工具结果或文档片段作为依据；依据允许为空。"
    "台账内容不代表绝对事实，但可以帮助模型追溯已有结果和决策历史，并判断如何继续任务。"
    "模型可见记录构成有界 FIFO 工作集；新记录进入队尾，容量或字节预算超限时最早记录先退出"
    "投影，但其持久审计 revision 不会被删除。单次写入长度和台账总量受系统限制。"
)
_CURRENT_REVISE_NAME = "修改或撤回任务台账记录"
_CURRENT_REVISE_DESCRIPTION = (
    "台账是当前运行期间持续保留的一份临时笔记，用于记录模型自己认为有价值的发现与观察、"
    "已做决定和尚未解决的问题；它不是长期记忆，也不等同于最终答案或已经核实的证据。"
    "本工具用于处理台账中仍然有效的已有记录：内容需要更正或补充时，使用 supersede 提交完整的"
    "新内容；记录错误或不再适用时，使用 retract 并说明原因。修改和撤回都会保留旧版本，不会"
    "直接覆盖或删除历史。调用时填写目标记录 ID，L1 的版本绑定由 Host 完成。替代内容可以关联当前计划中"
    "的相关目标，并可附上工具结果或文档片段作为依据；依据允许为空。台账内容不代表绝对事实，"
    "但可以帮助模型追溯已有结果和决策历史，并判断如何继续任务。supersede 后的新 revision 会刷新到"
    "FIFO 队尾，retract 会将目标移出工作集；已经出队的旧记录不会因其他记录撤回而自动回到投影。"
    "单次修改数量和台账总量受系统限制。"
)


def build_execution_findings_tool_specs() -> tuple[ToolSpec, ToolSpec]:
    """返回稳定、按暴露顺序排列的两个模型契约。"""

    return (
        ToolSpec(
            tool_id=RECORD_EXECUTION_FINDINGS_TOOL_ID,
            contract_version=EXECUTION_FINDINGS_TOOL_CONTRACT_VERSION,
            name=_CURRENT_RECORD_NAME,
            description=_CURRENT_RECORD_DESCRIPTION,
            input_schema=_record_input_schema(),
            output_schema=_output_schema(),
            catalog_tags=("execution", "write", "status"),
        ),
        ToolSpec(
            tool_id=REVISE_EXECUTION_FINDING_TOOL_ID,
            contract_version=EXECUTION_FINDINGS_TOOL_CONTRACT_VERSION,
            name=_CURRENT_REVISE_NAME,
            description=_CURRENT_REVISE_DESCRIPTION,
            input_schema=_revise_input_schema(),
            output_schema=_output_schema(),
            catalog_tags=("execution", "write", "status"),
        ),
    )


def build_execution_findings_effect_profile(
    *,
    default_scope: str = EXECUTION_FINDINGS_DEFAULT_EFFECT_SCOPE,
) -> ToolEffectProfile:
    """返回稳定效果形状；Catalog Definition 可用 ``*`` 作为范围模板。"""

    return ToolEffectProfile(
        (
            EffectDescriptor(
                resource=EffectResource.RUNTIME_STATE,
                action=EffectAction.UPDATE,
                scope_kind=EffectScopeKind.EXECUTION,
                default_scope=default_scope,
                data_egress=DataEgress.NONE,
                idempotency=Idempotency.DEDUPLICATED,
                reversibility=Reversibility.REVERSIBLE,
            ),
        )
    )


def build_execution_findings_execution_profile() -> ToolExecutionProfile:
    """返回两个台账写工具共享的稳定执行约束。"""

    return ToolExecutionProfile(
        default_timeout_s=2,
        hard_timeout_s=5,
        max_output_bytes=64_000,
        max_transparent_retries=0,
        concurrency_class="runtime_state",
    )


def build_execution_findings_tool_registrations(
    *,
    effect_scope: str = EXECUTION_FINDINGS_DEFAULT_EFFECT_SCOPE,
    source_fingerprint: str | None = None,
) -> tuple[ToolRegistration, ...]:
    """返回两个工具契约；其写入方标识由 Runtime 提供。"""

    effect_profile = build_execution_findings_effect_profile(
        default_scope=effect_scope
    )
    execution_profile = build_execution_findings_execution_profile()
    source = ToolSourceDescriptor(
        kind=ToolSourceKind.LOCAL,
        source_id=EXECUTION_FINDINGS_SOURCE_ID,
        fingerprint=source_fingerprint,
        display_name=EXECUTION_FINDINGS_SOURCE_DISPLAY_NAME,
    )
    record_spec, revise_spec = build_execution_findings_tool_specs()
    return (
        ToolRegistration(
            spec=record_spec,
            implementation_version=EXECUTION_FINDINGS_TOOL_IMPLEMENTATION_VERSION,
            source=source,
            handler=_host_dispatch_required,
            effect_profile=effect_profile,
            execution_profile=execution_profile,
        ),
        ToolRegistration(
            spec=revise_spec,
            implementation_version=EXECUTION_FINDINGS_TOOL_IMPLEMENTATION_VERSION,
            source=source,
            handler=_host_dispatch_required,
            effect_profile=effect_profile,
            execution_profile=execution_profile,
        ),
    )


def _record_input_schema() -> dict[str, Any]:
    expected_revision: dict[str, Any] = {
        "type": "integer",
        "minimum": 0,
        "description": "可选 CAS 版本；L1 省略时由 Host 绑定当前冻结快照，显式旧版本会被拒绝。",
    }
    items: dict[str, Any] = {
        "type": "array",
        "minItems": 1,
        "maxItems": 4,
        "items": {"$ref": "#/$defs/recordItem"},
        "description": "本次要追加的台账记录，最少 1 条、最多 4 条。",
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {
            "expected_ledger_revision": expected_revision,
            "items": items,
        },
        "$defs": _shared_definitions(),
    }


def _revise_input_schema() -> dict[str, Any]:
    definitions = _shared_definitions()
    supersede_properties: dict[str, Any] = {
        "operation": {"const": "supersede"},
        "entry_id": {"$ref": "#/$defs/id"},
        "kind": {"enum": [item.value for item in ExecutionFindingKind]},
        "claim": {
            "type": "string",
            "minLength": 1,
            "maxLength": EXECUTION_FINDING_CLAIM_MAX_CHARACTERS,
        },
        "source_refs": {
            "type": "array",
            "maxItems": _CURRENT_MAX_SOURCE_REFS_PER_ENTRY,
            "items": {"$ref": "#/$defs/sourceRef"},
        },
        "scope_keys": {"$ref": "#/$defs/scopeKeys"},
    }
    retract_properties: dict[str, Any] = {
        "operation": {"const": "retract"},
        "entry_id": {"$ref": "#/$defs/id"},
        "reason": {"type": "string", "minLength": 1, "maxLength": 400},
    }
    supersede_properties["operation"]["description"] = (
        "用一份完整的新内容替代目标记录，同时保留旧版本。"
    )
    supersede_properties["entry_id"]["description"] = (
        "要修改的当前有效台账记录 ID。"
    )
    retract_properties["operation"]["description"] = (
        "撤回错误或不再适用的目标记录，同时保留历史。"
    )
    retract_properties["entry_id"]["description"] = (
        "要撤回的当前有效台账记录 ID。"
    )
    retract_properties["reason"]["description"] = "撤回这条记录的原因。"
    definitions.update(
        {
            "supersedeItem": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "operation",
                    "entry_id",
                    "kind",
                    "claim",
                ],
                "properties": supersede_properties,
            },
            "retractItem": {
                "type": "object",
                "additionalProperties": False,
                "required": ["operation", "entry_id", "reason"],
                "properties": retract_properties,
            },
        }
    )
    expected_revision: dict[str, Any] = {
        "type": "integer",
        "minimum": 0,
        "description": "可选 CAS 版本；L1 省略时由 Host 绑定当前冻结快照，显式旧版本会被拒绝。",
    }
    items: dict[str, Any] = {
        "type": "array",
        "minItems": 1,
        "maxItems": 4,
        "items": {
            "oneOf": [
                {"$ref": "#/$defs/supersedeItem"},
                {"$ref": "#/$defs/retractItem"},
            ]
        },
        "description": "本次要修改或撤回的记录，最少 1 条、最多 4 条。",
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {
            "expected_ledger_revision": expected_revision,
            "items": items,
        },
        "$defs": definitions,
    }


def _shared_definitions() -> dict[str, Any]:
    identifier = {
        "type": "string",
        "pattern": "^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$",
    }
    definitions: dict[str, Any] = {
        "id": identifier,
        "scopeKeys": {
            "type": "array",
            "maxItems": _CURRENT_MAX_SCOPE_KEYS_PER_ENTRY,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "maxLength": 200},
        },
        "sourceRef": ExecutionFindingSourceRef.model_json_schema(),
        "recordItem": {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind", "claim"],
            "properties": {
                "operation": {"const": "record"},
                "kind": {"enum": [item.value for item in ExecutionFindingKind]},
                "claim": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": EXECUTION_FINDING_CLAIM_MAX_CHARACTERS,
                },
                "source_refs": {
                    "type": "array",
                    "maxItems": _CURRENT_MAX_SOURCE_REFS_PER_ENTRY,
                    "items": {"$ref": "#/$defs/sourceRef"},
                },
                "scope_keys": {"$ref": "#/$defs/scopeKeys"},
            },
        },
    }
    definitions["scopeKeys"]["description"] = (
        "可选的相关计划目标标识；填写时从当前计划中逐字选择，省略不代表所有目标。"
    )
    record_properties = definitions["recordItem"]["properties"]
    record_properties["kind"]["description"] = (
        "finding 表示发现或观察，decision 表示已做决定，gap 表示未解决问题。"
    )
    record_properties["claim"]["description"] = "本条临时笔记的完整内容。"
    record_properties["source_refs"]["description"] = (
        "可选依据；允许为空，填写时必须精确引用当前运行中的真实结果或文档片段。"
    )
    return definitions


def _output_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "status",
            "ledger_revision",
            "affected_entry_ids",
            "active_projection",
            "replayed",
        ],
        "properties": {
            "schema_version": {"const": "execution-findings-tool-result-v1"},
            "status": {"const": "applied"},
            "ledger_revision": {"type": "integer", "minimum": 1},
            "affected_entry_ids": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1, "maxLength": 200},
            },
            "active_projection": {"type": "object"},
            "replayed": {"type": "boolean"},
        },
    }


def _host_dispatch_required(_arguments: dict[str, Any]) -> dict[str, Any]:
    raise RuntimeError(
        "execution findings tools require the Host runtime-state dispatcher"
    )


__all__ = [
    "EXECUTION_FINDINGS_DEFAULT_EFFECT_SCOPE",
    "EXECUTION_FINDINGS_SOURCE_DISPLAY_NAME",
    "EXECUTION_FINDINGS_SOURCE_ID",
    "EXECUTION_FINDINGS_TOOL_CONTRACT_VERSION",
    "EXECUTION_FINDINGS_TOOL_IMPLEMENTATION_VERSION",
    "build_execution_findings_effect_profile",
    "build_execution_findings_execution_profile",
    "build_execution_findings_tool_registrations",
    "build_execution_findings_tool_specs",
]
