"""把已验证 ``RetrievedContext`` 投影成模型可读文本。

本模块只做确定性序列化：保留来源状态、coverage、引用、写入 guard 和最终已打包条目的顺序，
并转义进入属性位置的公开元数据。它不重新检索、排序、裁剪、计算预算或判断答案是否充分，
这些决定必须在构造 ``RetrievedContext`` 前完成。

条目正文即使来自已授权来源，进入模型后仍只是外部证据而非 system instruction；最终 Prompt
Builder 负责把本投影放在正确的信任层。该投影同时服务 Runtime、轨迹与测试，不属于某一种
Tool，因此保留在 Retrieval 根级公共边界。
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

from .contracts import RetrievedContext, SourceUnitRef


@dataclass(frozen=True, slots=True)
class SerializedRetrievedContext:
    """可用于 Prompt 的文本，以及生成它的精确已验证 Unit。"""

    text: str
    item_refs: tuple[SourceUnitRef, ...]


class RetrievedContextPromptSerializer:
    """序列化完整检索结果，不创造或省略证据语义。

    返回的 ``item_refs`` 与正文条目同序，供调用方把 Prompt 内容与已验证来源重新关联；调用方
    不应解析生成的 XML-like 文本来恢复权威身份。
    """

    def serialize(self, context: RetrievedContext) -> SerializedRetrievedContext:
        lines = [
            "<retrieved_context"
            f' status="{context.status.value}"'
            f' packed_tokens="{context.packed_tokens}"'
            f' configured_token_limit="{context.configured_token_limit}">'
        ]
        for source_type, outcome in context.source_outcomes.items():
            attributes = [
                f'source="{source_type.value}"',
                f'availability="{outcome.availability.value}"',
                f'retrieval="{outcome.retrieval.value}"',
            ]
            if outcome.reason_code:
                attributes.append(f'reason_code="{escape(outcome.reason_code, quote=True)}"')
            attributes.extend(
                f'{key}="{escape(value, quote=True)}"'
                for key, value in sorted(outcome.coverage_facts.items())
            )
            lines.append(f"<retrieval_source_status {' '.join(attributes)} />")

        guard = context.long_term_memory_write_guard
        guard_attributes = [
            f'status="{guard.status.value}"',
            f'evaluated_sources="{",".join(source.value for source in guard.evaluated_sources)}"',
        ]
        if guard.blocking_sources:
            guard_attributes.append(
                f'blocking_sources="{",".join(source.value for source in guard.blocking_sources)}"'
            )
        if guard.reason_codes:
            guard_attributes.append(
                f'reason_codes="{escape(",".join(guard.reason_codes), quote=True)}"'
            )
        lines.append(f"<long_term_memory_write_guard {' '.join(guard_attributes)} />")
        for position, item in enumerate(context.items, start=1):
            citation = _citation_text(item.citation)
            lines.extend(
                (
                    "<retrieved_item"
                    f' position="{position}"'
                    f' source="{item.ref.source_type.value}"'
                    f' revision="{escape(item.ref.source_revision, quote=True)}"'
                    f' citation="{escape(citation, quote=True)}">',
                    item.content,
                    "</retrieved_item>",
                )
            )
        lines.append("</retrieved_context>")
        return SerializedRetrievedContext(
            text="\n".join(lines),
            item_refs=tuple(item.ref for item in context.items),
        )


def _citation_text(citation) -> str:
    """稳定定位符和有界公开事实，绝不包含私有诊断细节。"""

    return "; ".join(
        f"{key}={value}"
        for key, value in sorted((str(key), str(value)) for key, value in citation.items())
    )
