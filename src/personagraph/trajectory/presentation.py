"""从现有去重快照生成可读轨迹；不读取其它账本或创建第二份运行状态。"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any


_KINDS = {
    "model_call": "模型调用", "tool_call": "工具动作", "retrieval": "检索",
    "recording_failure": "记录故障",
}
_ROLES = {
    "system": "系统要求", "user": "输入", "assistant": "模型输出",
    "thinking": "模型思考记录", "tool_arguments": "工具输入", "tool_result": "工具结果",
    "query": "查询", "evidence": "召回条目", "rejected_output": "拒绝原因",
}
_OUTCOMES = {"ok": "已返回", "rejected": "校验拒绝", "failed": "失败"}


def render_trajectory(snapshot: Mapping[str, Any]) -> str:
    """按调用合并返回/拒绝/终态记录，正文就地展开，重复 blob 用阅读链接代替。"""
    groups: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for step in snapshot["steps"]:
        identity = (
            step.get("model_call_id") if step["kind"] == "model_call" else None
        ) or step["step_id"]
        key = (step.get("session_id"), step.get("turn_id"), step["kind"], identity)
        groups.setdefault(key, []).append(step)
    blobs = snapshot["blobs"]
    origins: dict[tuple[object, ...], int] = {}
    for number, records in enumerate(groups.values(), 1):
        for record in records:
            for part in record["parts"]:
                value = _object(blobs[part["blob_sha256"]]["text"])
                if _is_rejection(value):
                    origins[_response_key(record, value["rejected_response_sha256"])] = number

    lines = [
        "# 执行轨迹", "",
        "按调用分组；一组可能包含多次返回、校验拒绝和终态汇总，不等于 HTTP 请求次数。",
        "正文只来自已保存轨迹；被省略或截断的内容不会推测补全。", "",
    ]
    seen: dict[str, str] = {}
    for number, records in enumerate(groups.values(), 1):
        first = records[0]
        lines.extend([
            f'<a id="call-{number}"></a>', "",
            f"## {number}. {_KINDS.get(first['kind'], first['kind'])}", "",
            _fence(str(first.get("purpose") or first["kind"])), "",
        ])
        for ordinal, record in enumerate(records, 1):
            status = _OUTCOMES.get(record["outcome"], record["outcome"])
            lines.extend([f"### {status}", "", _fence(_record_label(record)), ""])
            for part in record["parts"]:
                digest = part["blob_sha256"]
                label = _ROLES.get(part["role"], part["role"])
                if digest in seen:
                    lines.extend([f"{label}：[与前文相同](#{seen[digest]})。", ""])
                    continue
                anchor = f"part-{number}-{ordinal}-{part['seq']}"
                seen[digest] = anchor
                blob = blobs[digest]
                lines.extend([f'<a id="{anchor}"></a>', "", f"{label}：", ""])
                lines.extend(_content(blob["text"], record, origins))
                if blob.get("truncated"):
                    lines.extend([f"注意：仅保存前缀；原文共 {blob['byte_count']} 字节。", ""])
    return "\n".join(lines).rstrip() + "\n"


def _content(text: str, record: Mapping[str, Any], origins: Mapping[tuple, int]) -> list[str]:
    value = _object(text)
    if _is_rejection(value):
        lines = []
        for issue in value.get("issues", []):
            reason = issue.get("safe_explanation") or issue.get("code") or "未保存具体原因"
            lines.extend([_fence(f"{', '.join(issue.get('paths') or ['/'])}: {reason}"), ""])
        lines.extend([
            "已安排后续修复。" if value.get("repair_scheduled") else "未安排后续修复。", "",
        ])
        return lines
    if value is not None and str(value.get("reference_kind", "")).startswith("trajectory-redacted-"):
        original = origins.get(_response_key(record, value.get("response_sha256")))
        lines = [
            "正文未复制到轨迹；以下是来源标记，不保证该正文存在可读取的副本。", "",
            _fence(json.dumps(value, ensure_ascii=False, indent=2)), "",
        ]
        if value["reference_kind"] == "trajectory-redacted-rejected-model-output" and original:
            lines.extend([f"修复关系：[此前被拒的调用](#call-{original})。", ""])
        return lines
    try:
        text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except (ValueError, TypeError):
        pass
    return [_fence(text), ""]


def _record_label(record: Mapping[str, Any]) -> str:
    fields = {
        "时间": record["occurred_at"], "调用": record.get("model_call_id"),
        "原因": record.get("reason_code"), "耗时(ms)": record.get("duration_ms"),
        **record.get("metrics", {}),
    }
    return " · ".join(f"{key}: {value}" for key, value in fields.items() if value is not None)


def _object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _is_rejection(value: dict[str, Any] | None) -> bool:
    return value is not None and value.get("schema_version") == "runtime-model-output-rejection-observation-v1"


def _response_key(record: Mapping[str, Any], digest: object) -> tuple[object, ...]:
    return record.get("session_id"), record.get("turn_id"), digest


def _fence(text: str) -> str:
    # 模型/文档正文可能包含 Markdown；选择更长 fence，避免把正文当阅读器指令或 HTML。
    length = max((len(match) for match in re.findall(r"`+", text)), default=0)
    marker = "`" * max(3, length + 1)
    return f"{marker}\n{text}\n{marker}"


__all__ = ["render_trajectory"]
