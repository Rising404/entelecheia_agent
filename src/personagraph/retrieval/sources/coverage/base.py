"""证明一个 Source 已被派生检索数据覆盖的通用只读机制。

权威侧适配器只提供当前指针绑定。本模块将它们与固定且就绪的 RetrievalDataVersion
比较，不读取来源内容、不扩大作用域，也不写入同步状态。
"""

from __future__ import annotations

from ...contracts import (
    ContextCoverageLimitation,
    SourceAccess,
    SourceAvailability,
    SourceIndexBindingSnapshot,
    SourceType,
)
from ...ports import SourceIndexBindingReaderPort
from ...sqlite_store import (
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
    SqliteRetrievalCatalog,
)


DEFAULT_MAX_SOURCE_INDEX_COVERAGE_BINDINGS = 5_000


class SourceDerivedIndexCoverageProbe:
    """检测一个就绪 Source 的派生索引覆盖是否不完整。"""

    def __init__(
        self,
        *,
        catalog: SqliteRetrievalCatalog,
        binding_reader: SourceIndexBindingReaderPort,
        source_type: SourceType,
        reason_prefix: str,
        maximum_bindings: int = DEFAULT_MAX_SOURCE_INDEX_COVERAGE_BINDINGS,
    ) -> None:
        if not isinstance(source_type, SourceType):
            raise ValueError("source_type must be a SourceType")
        if not isinstance(reason_prefix, str) or not reason_prefix.strip():
            raise ValueError("reason_prefix must be a non-empty string")
        if (
            isinstance(maximum_bindings, bool)
            or not isinstance(maximum_bindings, int)
            or maximum_bindings <= 0
        ):
            raise ValueError("maximum_bindings must be a positive integer")
        self._catalog = catalog
        self._binding_reader = binding_reader
        self._source_type = source_type
        self._reason_prefix = reason_prefix
        self._maximum_bindings = maximum_bindings

    @property
    def source_type(self) -> SourceType:
        return self._source_type

    def check_source_coverage(
        self,
        access: SourceAccess,
        *,
        retrieval_data_version_id: str | None,
    ) -> ContextCoverageLimitation | None:
        """返回安全限制信息，而非泄露跨存储失败。"""

        if access.source_type is not self.source_type:
            raise ValueError("derived-index coverage requires the matching SourceAccess")
        if access.availability is not SourceAvailability.READY:
            raise ValueError("derived-index coverage requires ready SourceAccess")
        try:
            return self._check_ready_source_coverage(
                access,
                retrieval_data_version_id=retrieval_data_version_id,
            )
        except Exception:
            return self._limitation("coverage_check_unavailable")

    def _check_ready_source_coverage(
        self,
        access: SourceAccess,
        *,
        retrieval_data_version_id: str | None,
    ) -> ContextCoverageLimitation | None:
        if not retrieval_data_version_id:
            return self._limitation("data_version_unavailable")
        binding_snapshot = self._binding_reader.get_current_index_binding_snapshot(
            access,
            maximum_bindings=self._maximum_bindings,
        )
        if not isinstance(binding_snapshot, SourceIndexBindingSnapshot):
            raise TypeError("binding reader returned an invalid snapshot")
        if not binding_snapshot.source_snapshot_is_current:
            return self._limitation("coverage_incomplete")
        if not binding_snapshot.binding_enumeration_complete:
            return self._limitation("coverage_limit_exceeded")

        data_version = self._catalog.get_data_version(retrieval_data_version_id)
        if (
            data_version is None
            or data_version.role is not RetrievalDataVersionRole.ACTIVE
            or data_version.state is not RetrievalDataVersionState.READY
        ):
            return self._limitation("data_version_unavailable")

        expected = {
            (
                binding.source_unit_id,
                binding.source_revision,
                binding.indexed_content_hash,
            )
            for binding in binding_snapshot.bindings
        }
        if not expected:
            return None
        covered = self._catalog.active_ready_source_binding_keys(
            data_version_id=data_version.id,
            source_filter=access.source_filter,
            keys=tuple(sorted(expected)),
        )
        if expected.issubset(covered):
            return None
        return self._limitation("coverage_incomplete")

    def _limitation(self, suffix: str) -> ContextCoverageLimitation:
        return ContextCoverageLimitation(
            source_type=self.source_type,
            reason_code=f"{self._reason_prefix}_index_{suffix}",
        )
