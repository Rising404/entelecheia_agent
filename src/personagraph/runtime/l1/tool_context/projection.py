"""仅把紧邻前一步的工具批次投影到下一次决策，不消费或改写持久结果。

本模块不处理 API 重试、数据库或证据验证。controller 对已冻结的模型请求直接复用，
只有开启新逻辑步骤才调用这里。语义验证按原始引用独立读取持久证据。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from ....persistent_turn_content.findings import EXECUTION_FINDINGS_TOOL_IDS
from ....persistent_turn_content.evidence import (
    format_l1_call_ref, l1_call_refs_by_id, project_findings_arguments,
)
from ....output_protocol.l1_persistence import normalize_l1_finding_arguments
from ....tools.findings.projection import project_execution_findings_tool_output_for_model


class ToolResultProjectionError(ValueError):
    """持久结果形状不符合 L1 输入投影合同。"""


_SENSITIVE_ARGUMENT_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "password",
        "secret",
        "token",
        "access_token",
        "refresh_token",
    }
)
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")
_MAX_ARGUMENT_DEPTH = 8
_MAX_ARGUMENT_CONTAINER_ITEMS = 64
_MAX_ARGUMENT_STRING_CHARACTERS = 4_096
_TOOL_EXECUTION_AUDIT_METADATA_KEYS = frozenset(
    {
        "tool_id",
        "contract_version",
        "implementation_version",
        "execution_mode",
    }
)


def project_tool_call_arguments(
    call: Mapping[str, object],
) -> dict[str, Any]:
    """校验并安全投影一次持久 ToolCall 的真实规范化参数。

    参数 JSON/hash 仍以 Store 中的原始记录为权威；本函数只生成有界展示副本。
    查询、页码、offset 等普通业务参数保持原值，凭据和绝对本机路径不会重新进入
    模型上下文。任何省略都会在 ``arguments_projection`` 中显式计数。
    """

    raw = call.get("arguments_json")
    expected_hash = call.get("arguments_hash")
    if not isinstance(raw, str) or not isinstance(expected_hash, str):
        raise ToolResultProjectionError(
            "persisted L1 ToolCall arguments have no JSON/hash authority"
        )
    try:
        arguments = json.loads(raw)
        canonical = json.dumps(
            arguments,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        raise ToolResultProjectionError(
            "persisted L1 ToolCall arguments are invalid JSON"
        ) from None
    if (
        not isinstance(arguments, dict)
        or canonical != raw
        or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
        or hashlib.sha256(raw.encode("utf-8")).hexdigest() != expected_hash
    ):
        raise ToolResultProjectionError(
            "persisted L1 ToolCall arguments failed authority validation"
        )

    counts = {"redacted": 0, "truncated": 0}
    projected = _project_argument_value(arguments, counts=counts, depth=0)
    assert isinstance(projected, dict)
    return {
        "arguments": projected,
        "arguments_projection": {
            "source": "persisted_normalized_arguments",
            "arguments_sha256": expected_hash,
            "complete": counts["redacted"] == counts["truncated"] == 0,
            "redacted_value_count": counts["redacted"],
            "truncated_value_count": counts["truncated"],
        },
    }


def _project_argument_value(
    value: Any,
    *,
    counts: dict[str, int],
    depth: int,
    key: str | None = None,
) -> Any:
    if key is not None and key.casefold() in _SENSITIVE_ARGUMENT_KEYS:
        counts["redacted"] += 1
        return "<redacted>"
    if isinstance(value, str):
        if value.startswith(("/", "~/")) or _WINDOWS_ABSOLUTE_PATH.match(value):
            counts["redacted"] += 1
            return "<absolute-path-redacted>"
        if len(value) > _MAX_ARGUMENT_STRING_CHARACTERS:
            counts["truncated"] += 1
            return value[:_MAX_ARGUMENT_STRING_CHARACTERS]
        return value
    if value is None or isinstance(value, bool | int | float):
        return value
    if depth >= _MAX_ARGUMENT_DEPTH:
        counts["truncated"] += 1
        return "<depth-truncated>"
    if isinstance(value, Mapping):
        items = list(value.items())
        if len(items) > _MAX_ARGUMENT_CONTAINER_ITEMS:
            counts["truncated"] += len(items) - _MAX_ARGUMENT_CONTAINER_ITEMS
            items = items[:_MAX_ARGUMENT_CONTAINER_ITEMS]
        return {
            str(item_key): _project_argument_value(
                item_value,
                counts=counts,
                depth=depth + 1,
                key=str(item_key),
            )
            for item_key, item_value in items
        }
    if isinstance(value, list | tuple):
        items = list(value)
        if len(items) > _MAX_ARGUMENT_CONTAINER_ITEMS:
            counts["truncated"] += len(items) - _MAX_ARGUMENT_CONTAINER_ITEMS
            items = items[:_MAX_ARGUMENT_CONTAINER_ITEMS]
        return [
            _project_argument_value(item, counts=counts, depth=depth + 1)
            for item in items
        ]
    counts["truncated"] += 1
    return "<unsupported-value>"


def project_recent_tool_results(
    execution: dict[str, object],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """保留整个前一步批次；前一步无结果时不向更早步骤追溯。

    不以 findings 引用决定正文是否存活，也不留下随步骤线性增长的占位结果。
    结果总数供模型了解仍有历史可回读，具体 ID 可经历史目录工具分页发现。
    调用前置条件：Store 按 ordinal 排序，controller 已结算前一步；本层不重新判定
    运行状态。原始结果的 hash 与引用身份由持久读取端口及验证器独立复核。
    """

    attempts = execution.get("attempts")
    if not isinstance(attempts, list) or any(not isinstance(a, dict) for a in attempts):
        raise ToolResultProjectionError("persisted L1 Attempts have the wrong shape")
    durable_count = 0
    latest: list[dict[str, Any]] = []
    source_attempt_id = None
    for index, attempt in enumerate(attempts):
        batch = _read_batch(attempt)
        durable_count += len(batch)
        if index == len(attempts) - 1:
            latest = batch
            source_attempt_id = attempt.get("attempt_id")
    calls_present = "tool_calls" in execution
    calls_by_id = _tool_calls_by_id(execution) if calls_present else {}
    arguments_projected = 0
    if calls_present:
        for result in latest:
            call_id = result.get("tool_call_id")
            call = calls_by_id.get(call_id) if isinstance(call_id, str) else None
            if call is None or any(
                call.get(key) != expected
                for key, expected in (
                    ("attempt_id", result.get("attempt_id")),
                    ("tool_id", result.get("tool_id")),
                    ("status", result.get("status")),
                    ("outcome_hash", result.get("result_sha256")),
                )
            ):
                raise ToolResultProjectionError(
                    "persisted L1 ToolCall disagrees with its result batch"
                )
            result.update(project_tool_call_arguments(call))
            result["call_ordinal"] = call["call_ordinal"]
            arguments_projected += 1
    omitted_audit_metadata_fields = sum(
        _project_tool_result_metadata(result) for result in latest
    )
    compacted_count = 0
    for result in latest:
        result["call_ref"] = format_l1_call_ref(
            attempt_ordinal=result["attempt_ordinal"],
            call_ordinal=result["call_ordinal"],
        )
        result.pop("tool_result_id", None)
        if result.get("tool_id") in EXECUTION_FINDINGS_TOOL_IDS:
            if result.get("status") != "succeeded":
                # 被拒参数可能根本不符合 findings 协议；已核验原件摘要，不把它解析成证据。
                result.pop("arguments", None)
                result.pop("arguments_projection", None)
                result["arguments_unavailable"] = True
                result["arguments_unavailable_reason"] = "unsuccessful_findings_call"
            elif "arguments" in result:
                result["arguments"] = project_findings_arguments(
                    normalize_l1_finding_arguments(
                        result["arguments"], tool_calls=execution["tool_calls"],
                    ),
                    call_refs=l1_call_refs_by_id(execution),
                )
            # 台账正文由当前 active projection 唯一提供，旧写入回执不能绕过其有界工作集。
            result["result"] = project_execution_findings_tool_output_for_model(result.get("result"))
            result["result_partially_compacted"] = True
            result["compaction_source"] = "execution_findings"
            compacted_count += 1
    report = {
        "policy": "previous_attempt_only",
        "source_attempt_id": source_attempt_id,
        "durable_tool_result_count": durable_count,
        "projected_tool_result_count": len(latest),
        "omitted_tool_result_count": durable_count - len(latest),
        "compacted_tool_result_count": compacted_count,
        "omitted_audit_metadata_field_count": omitted_audit_metadata_fields,
    }
    if calls_present:
        report["arguments_projected_tool_result_count"] = arguments_projected
    return latest, report


def _project_tool_result_metadata(result: dict[str, Any]) -> int:
    """移除通用执行审计字段，保留工具声明的范围/分页等结果语义。"""

    metadata = result.get("metadata")
    if metadata is None:
        return 0
    if not isinstance(metadata, dict):
        raise ToolResultProjectionError(
            "persisted L1 ToolResult metadata has the wrong shape"
        )
    omitted = sum(key in metadata for key in _TOOL_EXECUTION_AUDIT_METADATA_KEYS)
    projected = {
        key: value
        for key, value in metadata.items()
        if key not in _TOOL_EXECUTION_AUDIT_METADATA_KEYS
    }
    if projected:
        result["metadata"] = projected
    else:
        result.pop("metadata", None)
    return omitted


def _tool_calls_by_id(
    execution: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    calls = execution.get("tool_calls")
    if not isinstance(calls, list) or any(not isinstance(call, Mapping) for call in calls):
        raise ToolResultProjectionError("persisted L1 ToolCalls have the wrong shape")
    indexed: dict[str, Mapping[str, object]] = {}
    for call in calls:
        call_id = call.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id or call_id in indexed:
            raise ToolResultProjectionError(
                "persisted L1 ToolCall identity is missing or duplicated"
            )
        indexed[call_id] = call
    return indexed


def _read_batch(attempt: dict[str, Any]) -> list[dict[str, Any]]:
    raw = attempt.get("tool_results_json")
    if raw is None:
        return []
    if not isinstance(raw, str):
        raise ToolResultProjectionError("persisted L1 ToolResults are not JSON text")
    try:
        batch = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ToolResultProjectionError("persisted L1 ToolResults are invalid JSON") from exc
    if not isinstance(batch, dict) or not isinstance(batch.get("tool_results"), list):
        raise ToolResultProjectionError("persisted L1 ToolResults have the wrong shape")
    results = batch["tool_results"]
    if any(not isinstance(result, dict) for result in results):
        raise ToolResultProjectionError("persisted L1 ToolResult has the wrong shape")
    # json.loads 已创建独立副本；附加来源身份不会触及数据库里的原始 JSON/hash。
    for result in results:
        result["attempt_id"] = attempt.get("attempt_id")
        result["attempt_ordinal"] = attempt.get("ordinal")
    return results
