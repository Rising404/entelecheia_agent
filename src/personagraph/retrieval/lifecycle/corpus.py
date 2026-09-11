"""独立检索语料库的冻结来源成员关系。

两个语料库刻意共享契约和实现代码，同时拥有各自的 SQLite 目录和 generation 生命周期。
将成员关系保存在此处，可为组合层、消费者和测试提供统一的规范路由权威源。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts import CorpusKey, SourceType


@dataclass(frozen=True, slots=True)
class RetrievalCorpusBinding:
    key: CorpusKey
    source_types: tuple[SourceType, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.key, CorpusKey):
            raise ValueError("key must be a CorpusKey")
        normalized = tuple(self.source_types)
        if not normalized or any(not isinstance(item, SourceType) for item in normalized):
            raise ValueError("source_types must be a non-empty SourceType tuple")
        if len(set(normalized)) != len(normalized):
            raise ValueError("source_types must not contain duplicates")
        object.__setattr__(self, "source_types", normalized)

    @property
    def generation_source_types(self) -> tuple[str, ...]:
        """generation 指纹使用的规范字符串形式。"""

        return tuple(sorted(source_type.value for source_type in self.source_types))

    def contains(self, source_type: SourceType) -> bool:
        return source_type in self.source_types


FILE_CORPUS = RetrievalCorpusBinding(
    key=CorpusKey.FILE,
    source_types=(SourceType.DOCUMENT, SourceType.PICTURE),
)

# L1 Session 检索使用独立的 History 物理语料，但只包含当前会话。
SESSION_CORPUS = RetrievalCorpusBinding(
    key=CorpusKey.HISTORY,
    source_types=(SourceType.CURRENT_SESSION,),
)
