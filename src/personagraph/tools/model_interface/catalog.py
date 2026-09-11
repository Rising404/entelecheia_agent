"""从现行 ToolSpec 投影必要的模型目录，不派生第二套描述或参数协议。"""

from __future__ import annotations

from collections.abc import Collection
from copy import deepcopy


def project_model_tool_catalog(
    catalog: list[dict], *, available_tool_ids: Collection[str] | None = None,
) -> list[dict]:
    """原样保留工具说明与输入约束；绑定、effects、版本与输出校验留给 Host。"""
    return [
        {
            "tool_id": entry["tool_id"],
            "description": entry["description"],
            "input_schema": deepcopy(entry["input_schema"]),
        }
        for entry in catalog
        if available_tool_ids is None or entry["tool_id"] in available_tool_ids
    ]
