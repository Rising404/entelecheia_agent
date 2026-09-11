"""Picture-observation specialization of derived-index coverage."""

from __future__ import annotations

from ...contracts import SourceType
from ...ports import SourceIndexBindingReaderPort
from ...sqlite_store import SqliteRetrievalCatalog
from .base import (
    DEFAULT_MAX_SOURCE_INDEX_COVERAGE_BINDINGS,
    SourceDerivedIndexCoverageProbe,
)


DEFAULT_MAX_PICTURE_COVERAGE_BINDINGS = (
    DEFAULT_MAX_SOURCE_INDEX_COVERAGE_BINDINGS
)


class PictureIndexCoverageProbe(SourceDerivedIndexCoverageProbe):
    """Report active picture observations missing from the pinned derived index."""

    def __init__(
        self,
        *,
        catalog: SqliteRetrievalCatalog,
        binding_reader: SourceIndexBindingReaderPort,
        maximum_bindings: int = DEFAULT_MAX_PICTURE_COVERAGE_BINDINGS,
    ) -> None:
        super().__init__(
            catalog=catalog,
            binding_reader=binding_reader,
            source_type=SourceType.PICTURE,
            reason_prefix="picture",
            maximum_bindings=maximum_bindings,
        )


__all__ = [
    "DEFAULT_MAX_PICTURE_COVERAGE_BINDINGS",
    "PictureIndexCoverageProbe",
]
