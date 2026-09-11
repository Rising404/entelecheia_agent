"""通用派生索引覆盖探针的 Document 专用组合。"""

from __future__ import annotations

from ...contracts import SourceType
from ...ports import SourceIndexBindingReaderPort
from .base import (
    DEFAULT_MAX_SOURCE_INDEX_COVERAGE_BINDINGS,
    SourceDerivedIndexCoverageProbe,
)
from ...sqlite_store import SqliteRetrievalCatalog


DEFAULT_MAX_DOCUMENT_COVERAGE_BINDINGS = DEFAULT_MAX_SOURCE_INDEX_COVERAGE_BINDINGS


class DocumentIndexCoverageProbe(SourceDerivedIndexCoverageProbe):
    """报告当前已挂载 Document 未被索引完全覆盖的情况。"""

    def __init__(
        self,
        *,
        catalog: SqliteRetrievalCatalog,
        binding_reader: SourceIndexBindingReaderPort,
        maximum_bindings: int = DEFAULT_MAX_DOCUMENT_COVERAGE_BINDINGS,
    ) -> None:
        super().__init__(
            catalog=catalog,
            binding_reader=binding_reader,
            source_type=SourceType.DOCUMENT,
            reason_prefix="document",
            maximum_bindings=maximum_bindings,
        )
