"""项目 File 检索的 Turn 版本冻结边界。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..lifecycle.generation import require_exact_active_generation
from .document_maintenance import (
    DocumentRetrievalComposition,
    build_document_retrieval_composition,
)


@dataclass(frozen=True, slots=True)
class FileTurnRetrievalBinding:
    """一个已接受 Turn 所预期的精确 File 配方。

    在首个已授权文件完成持久摄取前，第一代 Document 可能尚未 ACTIVE。
    冻结其确定性版本 ID 仍可防止恢复的 Turn 静默采用另一配方。
    """

    composition: DocumentRetrievalComposition
    data_version_id: str

    def __post_init__(self) -> None:
        if self.data_version_id != self.composition.generation_spec.version_id:
            raise ValueError("File Turn binding changed retrieval generation")


def prepare_file_retrieval_for_turn(
    *,
    features: Mapping[str, Any],
) -> FileTurnRetrievalBinding | None:
    """冻结精确 File 世代，但不发布空索引。"""

    if features.get("file_retrieval_read_enabled", False) is not True:
        return None
    composition = build_document_retrieval_composition()
    active = composition.foundation.catalog.active_data_version()
    if active is not None:
        active = require_exact_active_generation(
            composition.foundation.catalog,
            composition.generation_spec,
        )
        data_version_id = active.id
    else:
        data_version_id = composition.generation_spec.version_id
    return FileTurnRetrievalBinding(
        composition=composition,
        data_version_id=data_version_id,
    )


__all__ = [
    "FileTurnRetrievalBinding",
    "prepare_file_retrieval_for_turn",
]
