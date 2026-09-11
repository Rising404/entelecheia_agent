"""通用派生索引覆盖探针的当前 Session 组合。"""

from __future__ import annotations

from ...ports import SourceIndexBindingReaderPort
from .base import (
    DEFAULT_MAX_SOURCE_INDEX_COVERAGE_BINDINGS,
    SourceDerivedIndexCoverageProbe,
)
from ...contracts import SourceType
from ...sqlite_store import SqliteRetrievalCatalog


DEFAULT_MAX_CURRENT_SESSION_COVERAGE_BINDINGS = DEFAULT_MAX_SOURCE_INDEX_COVERAGE_BINDINGS


class CurrentSessionIndexCoverageProbe(SourceDerivedIndexCoverageProbe):
    """报告已提交对话对未被索引完全覆盖的情况。"""

    def __init__(
        self,
        *,
        catalog: SqliteRetrievalCatalog,
        binding_reader: SourceIndexBindingReaderPort,
        maximum_bindings: int = DEFAULT_MAX_CURRENT_SESSION_COVERAGE_BINDINGS,
    ) -> None:
        super().__init__(
            catalog=catalog,
            binding_reader=binding_reader,
            source_type=SourceType.CURRENT_SESSION,
            reason_prefix="current_session",
            maximum_bindings=maximum_bindings,
        )
