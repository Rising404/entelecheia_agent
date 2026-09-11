"""Document retrieval 的轻量受信作用域策略。

本模块只拥有结构常量和纯判断，不加载 Document Store。Tooling 可以用同一组常量构造
Host 冻结的 ``SourceFilter``，Document Source 再在打开权威访问时解释该作用域。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..contracts import SourceFilter


DOCUMENT_EVIDENCE_SCOPE_KEY = "document_evidence_scope"
DOCUMENT_USER_EVIDENCE_SCOPE = "user_evidence"


def requests_user_evidence_scope(source_filter: SourceFilter) -> bool:
    """返回 Host 是否要求只召回非 Agent 产出的文档证据。"""

    return (
        source_filter.as_mapping().get(DOCUMENT_EVIDENCE_SCOPE_KEY)
        == DOCUMENT_USER_EVIDENCE_SCOPE
    )


def is_user_evidence_document(
    document: Mapping[str, Any],
    *,
    path_area: str | None,
) -> bool:
    """根据独立来源账本与路径分类排除 Agent 产物。

    两项权威事实任一把文档标为 Agent 输出时即拒绝；旧文档没有来源记录或路径分类时
    保持既有 workspace 语义，避免无依据地缩小用户已挂载语料。
    """

    file_origin = str(document.get("file_origin") or "").strip()
    normalized_path_area = str(path_area or "").strip()
    return (
        file_origin != "agent_output"
        and normalized_path_area not in {"agent_output", "output"}
    )


__all__ = [
    "DOCUMENT_EVIDENCE_SCOPE_KEY",
    "DOCUMENT_USER_EVIDENCE_SCOPE",
    "is_user_evidence_document",
    "requests_user_evidence_scope",
]
