"""基于已挂载 Document generation 的具体规划资源读取器。

核心资源感知契约有意接受注入 port。本模块是 Entelecheia 现有 Document store 的生产
组合：加载一个 Session 作用域当前 generation，检查每个有类型分块哈希，移除私有
路径，并返回有界证据投影。它绝不重新解析源字节或调用模型；文档 ingest 留下的视觉
覆盖仍作为显式 gap，交给后续视觉能力处理。
"""

from __future__ import annotations

import hashlib

from personagraph.l2.auxiliary_graph.contracts import PlanningObservationStatus
from personagraph.l2.planning.resource_perception import (
    MAX_RAW_STATEMENT_CHARACTERS,
    FrozenPlanningResource,
    PlanningResourceCoverage,
    PlanningResourceEvidenceKind,
    PlanningResourceEvidenceUnit,
    PlanningResourceGapReason,
    PlanningResourceReadOutcome,
    PlanningResourceReadPortError,
    PlanningResourceReadRequest,
)
from personagraph.tools.documents.frozen_mounted_document_reader import (
    FrozenMountedDocument,
    FrozenMountedDocumentReadError,
    FrozenMountedDocumentReadFailure,
    FrozenMountedDocumentReader,
)


class MountedDocumentPlanningResourceReadPort:
    """从持久化分块精确读取一个冻结挂载 Document 版本。"""

    def __init__(
        self,
        allowed_documents: tuple[FrozenMountedDocument, ...] | None = None,
        *,
        reader: FrozenMountedDocumentReader | None = None,
    ) -> None:
        self._reader = reader or FrozenMountedDocumentReader()
        if allowed_documents is None:
            # Exact control-plane composition supplies the complete snapshot
            # binding.  A direct Host primitive request still carries a core
            # frozen resource identity and uses the same validated reader.
            self._allowed_documents_by_alias = None
            return
        aliases = tuple(item.resource_alias for item in allowed_documents)
        if len(aliases) != len(set(aliases)):
            raise ValueError("mounted Document read scope has duplicate aliases")
        self._allowed_documents_by_alias = {
            item.resource_alias: item for item in allowed_documents
        }

    def read_frozen_resource(
        self,
        request: PlanningResourceReadRequest,
    ) -> PlanningResourceReadOutcome:
        if not isinstance(request, PlanningResourceReadRequest):
            raise TypeError("request must be a PlanningResourceReadRequest")
        resource = request.resource
        if self._allowed_documents_by_alias is None:
            document = _document_from_resource(resource)
        else:
            document = self._allowed_documents_by_alias.get(
                resource.resource_alias
            )
            if document is None or not _resource_matches_document(
                resource,
                document,
            ):
                raise PlanningResourceReadPortError(
                    status=PlanningObservationStatus.BLOCKED,
                    reason=PlanningResourceGapReason.ACCESS_BLOCKED,
                )
        try:
            snapshot = self._reader.read_window(
                document,
                start_sequence=0,
                maximum_chunks=request.max_evidence_units,
            )
        except FrozenMountedDocumentReadError as exc:
            if exc.failure is FrozenMountedDocumentReadFailure.STALE:
                return PlanningResourceReadOutcome(
                    status=PlanningObservationStatus.STALE,
                    observed_resource_version=(
                        exc.observed_document_version_id
                    ),
                    observed_content_sha256=exc.observed_source_sha256,
                    observed_coverage=_optional_coverage(
                        exc.observed_processing_status
                    ),
                    gap_reasons=(PlanningResourceGapReason.RESOURCE_STALE,),
                )
            if exc.failure is FrozenMountedDocumentReadFailure.UNAVAILABLE:
                status = PlanningObservationStatus.BLOCKED
                reason = PlanningResourceGapReason.ACCESS_BLOCKED
            else:
                status = PlanningObservationStatus.FAILED
                reason = PlanningResourceGapReason.READ_FAILED
            raise PlanningResourceReadPortError(
                status=status,
                reason=reason,
                private_detail=type(exc).__name__,
            ) from exc

        observed_coverage = _coverage(snapshot.processing_status)

        evidence: list[PlanningResourceEvidenceUnit] = []
        projection_bounded = snapshot.truncated
        for chunk in snapshot.chunks:
            # 持久化分块保留来源边缘空白，因为它在 Markdown 与纯文本中可能有意义。
            # 规划证据具有更严格的规范文本边界，因此这里只规范化外缘；已存储分块及其
            # 版本绑定 producer id 保持不变，内部空白也原样保留。
            statement = chunk.content.strip()
            if not statement:
                raise PlanningResourceReadPortError(
                    status=PlanningObservationStatus.FAILED,
                    reason=PlanningResourceGapReason.READ_FAILED,
                    private_detail="empty_canonical_document_chunk",
                )
            if len(statement) > MAX_RAW_STATEMENT_CHARACTERS:
                statement = statement[:MAX_RAW_STATEMENT_CHARACTERS].rstrip()
                projection_bounded = True
            statement_sha256 = hashlib.sha256(statement.encode("utf-8")).hexdigest()
            evidence.append(
                PlanningResourceEvidenceUnit(
                    source_unit_id=_source_unit_id(
                        document.document_id,
                        snapshot.document_version_id,
                        chunk.producer_chunk_id,
                    ),
                    statement=statement,
                    locator=_public_locator(
                        resource,
                        sequence=chunk.sequence,
                        source_pages=chunk.source_pages,
                    ),
                    content_sha256=statement_sha256,
                    evidence_kind=PlanningResourceEvidenceKind.DOCUMENT_TEXT,
                )
            )

        if not evidence:
            return PlanningResourceReadOutcome(
                status=PlanningObservationStatus.NO_MATCH,
                observed_resource_version=snapshot.document_version_id,
                observed_content_sha256=snapshot.source_sha256,
                observed_coverage=observed_coverage,
                gap_reasons=(PlanningResourceGapReason.NO_RELEVANT_CONTENT,),
            )

        reasons: set[PlanningResourceGapReason] = set()
        if observed_coverage is PlanningResourceCoverage.PARTIAL:
            reasons.add(PlanningResourceGapReason.INCOMPLETE_COVERAGE)
        if projection_bounded:
            reasons.add(PlanningResourceGapReason.PROJECTION_BOUNDED)
        status = (
            PlanningObservationStatus.PARTIAL
            if reasons
            else PlanningObservationStatus.SUCCESS
        )
        return PlanningResourceReadOutcome(
            status=status,
            observed_resource_version=snapshot.document_version_id,
            observed_content_sha256=snapshot.source_sha256,
            observed_coverage=observed_coverage,
            evidence=tuple(evidence),
            gap_reasons=tuple(sorted(reasons, key=lambda item: item.value)),
        )


def _coverage(processing_status: str) -> PlanningResourceCoverage:
    try:
        return PlanningResourceCoverage(processing_status)
    except ValueError as exc:
        raise ValueError("mounted Document has unsupported processing coverage") from exc


def _document_from_resource(
    resource: FrozenPlanningResource,
) -> FrozenMountedDocument:
    return FrozenMountedDocument(
        session_id=resource.session_id,
        resource_alias=resource.resource_alias,
        document_id=resource.resource_id,
        document_version_id=resource.resource_version,
        source_sha256=resource.content_sha256,
        processing_status=resource.coverage.value,
        resource_format=resource.resource_format.value,
        media_type=resource.media_type,
        file_extension=resource.file_extension,
    )


def _resource_matches_document(
    resource: FrozenPlanningResource,
    document: FrozenMountedDocument,
) -> bool:
    return bool(
        resource.session_id == document.session_id
        and resource.resource_alias == document.resource_alias
        and resource.resource_id == document.document_id
        and resource.resource_version == document.document_version_id
        and resource.content_sha256 == document.source_sha256
        and resource.coverage.value == document.processing_status
        and resource.resource_format.value == document.resource_format
        and resource.media_type == document.media_type
        and resource.file_extension == document.file_extension
    )


def _optional_coverage(
    processing_status: str | None,
) -> PlanningResourceCoverage | None:
    if processing_status is None:
        return None
    try:
        return _coverage(processing_status)
    except ValueError:
        return None


def _source_unit_id(document_id: str, version_id: str, producer_id: str) -> str:
    digest = hashlib.sha256(
        "\x1f".join((document_id, version_id, producer_id)).encode("utf-8")
    ).hexdigest()
    return f"document_unit_{digest[:32]}"


def _public_locator(
    resource: FrozenPlanningResource,
    *,
    sequence: int,
    source_pages: tuple[int, ...],
) -> str:
    prefix = f"resource:{resource.resource_alias}"
    if len(source_pages) == 1:
        return f"{prefix}#page={source_pages[0]}&chunk={sequence}"
    if source_pages:
        pages = ",".join(str(page) for page in source_pages)
        return f"{prefix}#pages={pages}&chunk={sequence}"
    return f"{prefix}#chunk={sequence}"


__all__ = ["MountedDocumentPlanningResourceReadPort"]
