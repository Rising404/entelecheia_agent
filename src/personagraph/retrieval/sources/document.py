"""从项目 Document Store 连接到统一检索的只读适配器。

适配器拥有 Document Source 的稳定 ID、结构作用域和权威重新校验。检索数据库绝不
直接访问领域表，从而让仅含指针的派生数据始终可替换。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any, Literal

from ...workspace.documents import application as docstore
from ...session import store as session_store
from ..contracts import (
    DOCUMENT_PROCESSING_COVERAGE_CODES,
    SourceAccess,
    SourceAvailability,
    SourceFilter,
    SourceIndexBinding,
    SourceIndexBindingSnapshot,
    SourceType,
    SourceUnit,
    SourceUnitRef,
)
from ..lifecycle.outbox import RetrievalUpdateEvent
from .identity import (
    mounted_document_chunk_ref_and_content,
    mounted_document_chunk_source_unit_id,
    parse_mounted_document_chunk_source_unit_id,
)
from ..lifecycle.sync import IndexableSourceUnit
from .document_policy import (
    DOCUMENT_EVIDENCE_SCOPE_KEY,
    DOCUMENT_USER_EVIDENCE_SCOPE,
    is_user_evidence_document,
    requests_user_evidence_scope,
)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _scope_value(source_filter: SourceFilter, key: str) -> str | None:
    return source_filter.as_mapping().get(key)


class MountedDocumentChunkSourceAdapter:
    """只面向当前且已切块 DocSet 块的 Document 适配器。

    受信请求可以指定一份已挂载文档（``session_id`` + ``doc_id``），也可以指定某个
    Session 中所有已挂载文档（仅 ``session_id``）。已索引的带类型 Unit 属于共享项目
    数据，只保留 ``doc_id``；搜索和获取内容前都会重新校验 Session 能力。
    """

    source_type = SourceType.DOCUMENT

    def __init__(
        self,
        *,
        preflight_mode: Literal["durable", "read_only"] = "durable",
    ) -> None:
        """选择打开 Document 检索访问时证明新鲜度的方式。

        ``durable`` 保留历史行为：打开访问会记录 DocStore 检索快照供后续审计。
        ``read_only`` 面向已经作出显式状态变更决策、且需要严格只读检索路径的调用方。
        它仍获取相同的权威和覆盖事实，但把快照身份保存在内存中，而不要求 DocStore
        持久化。
        """

        if preflight_mode not in {"durable", "read_only"}:
            raise ValueError("preflight_mode must be 'durable' or 'read_only'")
        self._preflight_mode = preflight_mode

    def catalog_source_filters(
        self,
        access: SourceAccess,
    ) -> tuple[SourceFilter, ...]:
        """将一个已授权 Session 视图解析为项目索引过滤器。

        Retrieval Unit 由绑定到本项目的每个 Session 共享，因此在 ``scope_json`` 中只存
        ``doc_id``。已打开的访问许可提供精确的已挂载当前文档集合；搜索对每份获准文档
        各运行一次，使未挂载项目文件无法占用 Session 候选预算。
        """

        if access.source_type is not self.source_type:
            raise ValueError("Document catalog filters require Document access")
        if access.availability is not SourceAvailability.READY:
            raise ValueError("Document catalog filters require ready access")
        requested_doc_id = _scope_value(access.source_filter, "doc_id")
        document_ids = tuple(sorted(access.source_revision_map))
        if requested_doc_id is not None:
            if document_ids != (requested_doc_id,):
                return ()
            document_ids = (requested_doc_id,)
        return tuple(
            SourceFilter.from_mapping(self.source_type, {"doc_id": doc_id})
            for doc_id in document_ids
        )

    def open_retrieval_access(self, source_filter: SourceFilter) -> SourceAccess:
        doc_id = _scope_value(source_filter, "doc_id")
        session_id = _scope_value(source_filter, "session_id")
        evidence_scope = _scope_value(source_filter, DOCUMENT_EVIDENCE_SCOPE_KEY)
        if evidence_scope not in {None, DOCUMENT_USER_EVIDENCE_SCOPE}:
            return SourceAccess(
                self.source_type,
                source_filter,
                SourceAvailability.BLOCKED,
                reason_code="document_evidence_scope_unsupported",
            )
        if not session_id:
            return SourceAccess(
                self.source_type,
                source_filter,
                SourceAvailability.BLOCKED,
                reason_code="document_session_id_missing",
            )
        try:
            session_availability = _session_retrieval_availability(session_id)
            if session_availability is not SourceAvailability.READY:
                return SourceAccess(
                    self.source_type,
                    source_filter,
                    session_availability,
                    reason_code=f"document_session_{session_availability.value}",
                )
            report = _apply_document_evidence_scope(
                self._open_freshness_report(session_id, doc_id),
                session_id=session_id,
                source_filter=source_filter,
            )
        except Exception:
            return SourceAccess(
                self.source_type,
                source_filter,
                SourceAvailability.UNAVAILABLE,
                reason_code="document_freshness_check_failed",
            )

        if report.get("ok"):
            if report.get("status") == "no_documents":
                return SourceAccess(self.source_type, source_filter, SourceAvailability.EMPTY)
            version_map = _document_version_map(report)
            snapshot_id = (
                _read_only_document_source_snapshot_id(
                    session_id=session_id,
                    source_filter=source_filter,
                    version_map=version_map,
                )
                if self._preflight_mode == "read_only" and version_map is not None
                else report.get("snapshot_id")
            )
            if (
                not isinstance(snapshot_id, str)
                or not snapshot_id.strip()
                or version_map is None
            ):
                return SourceAccess(
                    self.source_type,
                    source_filter,
                    SourceAvailability.UNAVAILABLE,
                    reason_code="document_freshness_snapshot_unavailable",
                )
            return SourceAccess(
                self.source_type,
                source_filter,
                SourceAvailability.READY,
                source_snapshot_id=snapshot_id,
                source_revision_map=version_map,
                coverage_facts=_document_source_coverage_facts(report),
            )

        return SourceAccess(
            self.source_type,
            source_filter,
            SourceAvailability.BLOCKED,
            reason_code=_document_freshness_reason(report),
        )

    def _open_freshness_report(
        self,
        session_id: str,
        doc_id: str | None,
    ) -> dict[str, Any]:
        """打开一个作用域，不改变默认持久路径。"""

        if self._preflight_mode == "read_only":
            return docstore.check_mounted_document_freshness(session_id, doc_id)
        return docstore.preflight_mounted_documents(session_id, doc_id)

    def revalidate_retrieval_access(self, access: SourceAccess) -> SourceAccess:
        """重新检查 Document 新鲜度和挂载作用域，不执行写入。

        初始访问要么拥有持久审计快照，要么派生只读内存身份。最终检查不会增加权威写入；
        它只在发布检索候选或无匹配结果前，证明原始受信作用域仍描述当前文件和挂载。
        """

        if access.source_type is not self.source_type:
            raise ValueError("Document access revalidation requires a Document SourceAccess")
        source_filter = access.source_filter
        doc_id = _scope_value(source_filter, "doc_id")
        session_id = _scope_value(source_filter, "session_id")
        evidence_scope = _scope_value(source_filter, DOCUMENT_EVIDENCE_SCOPE_KEY)
        if evidence_scope not in {None, DOCUMENT_USER_EVIDENCE_SCOPE}:
            return SourceAccess(
                self.source_type,
                source_filter,
                SourceAvailability.BLOCKED,
                reason_code="document_evidence_scope_unsupported",
            )
        if not session_id:
            return SourceAccess(
                self.source_type,
                source_filter,
                SourceAvailability.BLOCKED,
                reason_code="document_session_id_missing",
            )
        try:
            session_availability = _session_retrieval_availability(session_id)
            if session_availability is not SourceAvailability.READY:
                return SourceAccess(
                    self.source_type,
                    source_filter,
                    session_availability,
                    reason_code=f"document_session_{session_availability.value}",
                )
            report = _apply_document_evidence_scope(
                docstore.check_mounted_document_freshness(session_id, doc_id),
                session_id=session_id,
                source_filter=source_filter,
            )
        except Exception:
            return SourceAccess(
                self.source_type,
                source_filter,
                SourceAvailability.UNAVAILABLE,
                reason_code="document_freshness_recheck_failed",
            )
        if report.get("ok"):
            if report.get("status") == "no_documents":
                # 仅当来源打开时就是空的，``EMPTY`` 才是正常答案。若此前就绪的 Session
                # 全局作用域在搜索期间失去所有已挂载文档，将旧搜索发布为普通空来源会隐藏
                # 真实来源变化，并可能绕过必需来源阻止条件。
                if access.source_revision_map:
                    return SourceAccess(
                        self.source_type,
                        source_filter,
                        SourceAvailability.BLOCKED,
                        reason_code="document_source_changed_during_retrieval",
                    )
                return SourceAccess(self.source_type, source_filter, SourceAvailability.EMPTY)
            version_map = _document_version_map(report)
            if version_map is None:
                return SourceAccess(
                    self.source_type,
                    source_filter,
                    SourceAvailability.UNAVAILABLE,
                    reason_code="document_freshness_snapshot_unavailable",
                )
            return SourceAccess(
                self.source_type,
                source_filter,
                SourceAvailability.READY,
                source_snapshot_id=access.source_snapshot_id,
                source_revision_map=version_map,
                coverage_facts=_document_source_coverage_facts(report),
            )
        return SourceAccess(
            self.source_type,
            source_filter,
            SourceAvailability.BLOCKED,
            reason_code=_document_freshness_reason(report),
        )

    def fetch_units(
        self,
        access: SourceAccess,
        refs: Sequence[SourceUnitRef],
    ) -> Sequence[SourceUnit]:
        if access.source_type is not self.source_type or access.availability is not SourceAvailability.READY:
            return ()
        source_filter = access.source_filter
        doc_id = _scope_value(source_filter, "doc_id")
        session_id = _scope_value(source_filter, "session_id")
        if (
            not session_id
            or _session_retrieval_availability(session_id) is not SourceAvailability.READY
        ):
            return ()
        units: list[SourceUnit] = []
        for ref in refs:
            if ref.source_type is not self.source_type:
                continue
            identity = parse_mounted_document_chunk_source_unit_id(ref.source_unit_id)
            if identity is None or (
                not identity.is_project_scoped
                and identity.session_id != session_id
            ):
                continue
            if identity.is_typed:
                identity_doc_id = str(identity.doc_id)
                expected_version_id = access.source_revision_map.get(identity_doc_id)
                if not expected_version_id or ref.source_revision != expected_version_id:
                    continue
                chunk = docstore.get_current_typed_document_chunk(
                    identity_doc_id,
                    str(identity.producer_chunk_id),
                    expected_version_id=expected_version_id,
                    session_id=session_id,
                )
            else:
                chunk = docstore.get_current_document_chunk(
                    str(identity.storage_chunk_id),
                    session_id=session_id,
                )
            if chunk is None or (doc_id and str(chunk["doc_id"]) != doc_id):
                continue
            current_doc_id = str(chunk["doc_id"])
            if access.source_revision_map.get(current_doc_id) != str(chunk["source_version_id"]):
                continue
            current_ref, normalized_content = self._ref_and_content_from_chunk(
                chunk,
                session_id=session_id,
            )
            if current_ref != ref:
                continue
            citation = {
                "file_id": str(chunk.get("file_id") or ""),
                "file_version_id": str(chunk.get("file_version_id") or ""),
                "doc_id": current_doc_id,
                "chunk_id": str(chunk["id"]),
                "source_version_id": str(chunk["source_version_id"]),
                "location": str(chunk.get("loc") or ""),
                "title": str(chunk.get("title") or ""),
                "origin": _chunk_origin(chunk, session_id=session_id),
                "source_path": str(chunk.get("path") or ""),
                "source_mtime_ns": str(chunk.get("source_mtime_ns") or ""),
                "source_added_at": str(chunk.get("added_at") or ""),
                **_document_processing_citation(chunk),
            }
            if chunk.get("producer_chunk_id") is not None:
                citation.update({
                    "producer_chunk_id": str(chunk["producer_chunk_id"]),
                    "span_json": str(chunk.get("span_json") or ""),
                    "metadata_json": str(chunk.get("metadata_json") or ""),
                    "content_sha256": str(chunk.get("content_sha256") or ""),
                    "chunk_contract_version": str(chunk.get("chunk_contract_version") or ""),
                })
            units.append(
                SourceUnit(
                    ref=current_ref,
                    content=normalized_content,
                    citation=citation,
                )
            )
        return tuple(units)

    def get_current_index_binding_snapshot(
        self,
        access: SourceAccess,
        *,
        maximum_bindings: int,
    ) -> SourceIndexBindingSnapshot:
        """暴露当前 Document 块指针，不读取块内容。

        独立覆盖探针负责与派生检索数据比较。本适配器仍是唯一读取 Document 权威表的
        RAG 模块，因此探针无法扩大来源作用域或创建第二条权威路径。
        """

        if access.source_type is not self.source_type:
            raise ValueError("Document binding snapshots require a Document SourceAccess")
        if access.availability is not SourceAvailability.READY:
            raise ValueError("Document binding snapshots require ready SourceAccess")
        session_id = _scope_value(access.source_filter, "session_id")
        if not session_id:
            return SourceIndexBindingSnapshot(source_snapshot_is_current=False)
        requested_doc_id = _scope_value(access.source_filter, "doc_id")
        if requested_doc_id and set(access.source_revision_map) != {requested_doc_id}:
        # 伪造或过期访问不得利用此元数据桥枚举同一 Session 中另一份已挂载文档。
            return SourceIndexBindingSnapshot(source_snapshot_is_current=False)
        authority_snapshot = docstore.get_mounted_document_index_binding_snapshot(
            session_id,
            version_map=access.source_revision_map,
            maximum_bindings=maximum_bindings,
        )
        return SourceIndexBindingSnapshot(
            source_snapshot_is_current=authority_snapshot.source_snapshot_is_current,
            bindings=tuple(
                SourceIndexBinding(
                    source_unit_id=mounted_document_chunk_source_unit_id(
                        session_id=session_id,
                        storage_chunk_id=binding.chunk_id,
                        doc_id=(binding.doc_id if binding.producer_chunk_id is not None else None),
                        producer_chunk_id=binding.producer_chunk_id,
                    ),
                    source_revision=binding.source_version_id,
                    indexed_content_hash=binding.indexed_content_hash,
                )
                for binding in authority_snapshot.bindings
            ),
            binding_enumeration_complete=authority_snapshot.binding_enumeration_complete,
        )

    def read_for_index(self, event: RetrievalUpdateEvent) -> IndexableSourceUnit | None:
        if event.ref.source_type is not self.source_type:
            return None
        source_unit = self.read_current_for_reconcile(event.ref)
        if source_unit is None or source_unit.ref != event.ref:
            return None
        return source_unit

    def read_current_for_reconcile(self, ref: SourceUnitRef) -> IndexableSourceUnit | None:
        if ref.source_type is not self.source_type:
            return None
        identity = parse_mounted_document_chunk_source_unit_id(ref.source_unit_id)
        if identity is None:
            return None
        session_id = identity.session_id
        if (
            session_id is not None
            and _session_retrieval_availability(session_id)
            is not SourceAvailability.READY
        ):
            return None
        if identity.is_typed:
            chunk = docstore.get_current_typed_document_chunk(
                str(identity.doc_id),
                str(identity.producer_chunk_id),
                expected_version_id=ref.source_revision,
                session_id=session_id,
            )
        else:
            chunk = docstore.get_current_document_chunk(
                str(identity.storage_chunk_id),
                session_id=session_id,
            )
        if chunk is None:
            return None
        if session_id is not None and not docstore.is_mounted(
            str(chunk["doc_id"]),
            session_id,
        ):
            return None
        ref, normalized_content = self._ref_and_content_from_chunk(
            chunk,
            session_id=session_id or "project-index",
        )
        source_scope = {"doc_id": str(chunk["doc_id"])}
        if not identity.is_project_scoped:
            assert session_id is not None
            source_scope["session_id"] = session_id
        return IndexableSourceUnit(
            ref=ref,
            source_filter=SourceFilter.from_mapping(
                self.source_type,
                source_scope,
            ),
            content=normalized_content,
        )

    def list_indexable_units_for_backfill(
        self,
        source_filter: SourceFilter,
    ) -> tuple[IndexableSourceUnit, ...]:
        if source_filter.source_type is not self.source_type:
            return ()
        doc_id = _scope_value(source_filter, "doc_id")
        session_id = _scope_value(source_filter, "session_id")
        if session_id and (
            _session_retrieval_availability(session_id) is not SourceAvailability.READY
        ):
            return ()
        if doc_id:
            document_ids = (doc_id,)
        elif session_id:
            document_ids = tuple(
                str(document["id"]) for document in docstore.mounted_docs(session_id)
            )
        else:
            # Generation rollout 是项目级维护操作，不借用任一 Session 的挂载能力。
            # 空 Document scope 明确表示当前项目的完整语料，而非旧的全局文档库。
            document_ids = tuple(
                str(document["id"]) for document in docstore.list_documents()
            )
        units: list[IndexableSourceUnit] = []
        for current_doc_id in document_ids:
            chunks = docstore.list_current_document_chunks(current_doc_id, session_id=session_id)
            for chunk in chunks:
                if session_id is None and chunk.get("producer_chunk_id") is None:
                    raise ValueError(
                        "project document backfill requires typed chunk identities"
                    )
                ref, normalized_content = self._ref_and_content_from_chunk(
                    chunk,
                    session_id=session_id or "project-index",
                )
                unit_scope_values = {"doc_id": current_doc_id}
                if chunk.get("producer_chunk_id") is None:
                    assert session_id is not None
                    unit_scope_values["session_id"] = session_id
                units.append(
                    IndexableSourceUnit(
                        ref=ref,
                        source_filter=SourceFilter.from_mapping(
                            self.source_type,
                            unit_scope_values,
                        ),
                        content=normalized_content,
                    )
                )
        return tuple(units)

    @staticmethod
    def _ref_from_chunk(chunk: dict[str, Any], *, session_id: str) -> SourceUnitRef:
        ref, _ = MountedDocumentChunkSourceAdapter._ref_and_content_from_chunk(
            chunk,
            session_id=session_id,
        )
        return ref

    @staticmethod
    def _ref_and_content_from_chunk(
        chunk: dict[str, Any],
        *,
        session_id: str,
    ) -> tuple[SourceUnitRef, str]:
        return mounted_document_chunk_ref_and_content(
            session_id=session_id,
            chunk_id=str(chunk["id"]),
            source_version_id=str(chunk["source_version_id"]),
            content=str(chunk["content"]),
            doc_id=(
                str(chunk["doc_id"])
                if chunk.get("producer_chunk_id") is not None
                else None
            ),
            producer_chunk_id=(
                str(chunk["producer_chunk_id"])
                if chunk.get("producer_chunk_id") is not None
                else None
            ),
        )



def _document_freshness_reason(report: dict[str, Any]) -> str:
    """返回安全稳定的来源状态原因，不包含路径或内容。"""

    statuses = {
        str(document.get("status"))
        for document in report.get("documents", ())
        if isinstance(document, dict) and document.get("status")
    }
    if len(statuses) == 1:
        return f"document_{next(iter(statuses))}"
    return "document_freshness_blocked"


def _apply_document_evidence_scope(
    report: dict[str, Any],
    *,
    session_id: str,
    source_filter: SourceFilter,
) -> dict[str, Any]:
    """把 user-evidence scope 投影到打开访问的文档修订集合。

    ``source_revision_map`` 决定后续 catalog filter，因此必须在 SourceAccess 建立和重校验
    时收窄，而不能等 Top-K 或 evidence 返回后再丢弃命中。
    """

    if not requests_user_evidence_scope(source_filter):
        return report
    allowed_document_ids = {
        str(document["id"])
        for document in docstore.mounted_docs(session_id)
        if document.get("id")
        and is_user_evidence_document(
            document,
            path_area=_document_path_area(session_id, document.get("path")),
        )
    }
    scoped = dict(report)
    documents = report.get("documents")
    if isinstance(documents, list):
        scoped["documents"] = [
            document
            for document in documents
            if isinstance(document, dict)
            and str(document.get("doc_id") or "") in allowed_document_ids
        ]
    version_map = report.get("version_map")
    if isinstance(version_map, Mapping):
        scoped["version_map"] = {
            str(document_id): str(version_id)
            for document_id, version_id in version_map.items()
            if str(document_id) in allowed_document_ids
        }
    remaining_document_ids = set(scoped.get("version_map", {}))
    if not remaining_document_ids and isinstance(scoped.get("documents"), list):
        remaining_document_ids = {
            str(document.get("doc_id"))
            for document in scoped["documents"]
            if isinstance(document, dict) and document.get("doc_id")
        }
    if report.get("ok") and not remaining_document_ids:
        scoped.update(ok=True, status="no_documents", documents=[], version_map={})
    return scoped


def _document_path_area(session_id: str, path: object) -> str | None:
    """读取路径来源分类；不可判定时沿用 workspace 默认语义。"""

    from ...workspace.files.attachments import classify_session_path

    try:
        area = classify_session_path(session_id, str(path or ""))
    except (OSError, ValueError):
        return None
    return str(area) if area is not None else None


def _read_only_document_source_snapshot_id(
    *,
    session_id: str,
    source_filter: SourceFilter,
    version_map: Mapping[str, str],
) -> str:
    """为已验证 Document 读取返回确定性的内存身份。

    只读打开无法复用 DocStore 的持久审计快照，因此改为对精确受信作用域和当前版本映射
    计算哈希。这样 SourceAccess 能保持相同的非空快照不变量，同时不把私有 ID 放进外部
    形态的快照值。
    """

    material = {
        "session_id": session_id,
        "source_type": source_filter.source_type.value,
        "scope": dict(sorted(source_filter.as_mapping().items())),
        "version_map": sorted(version_map.items()),
    }
    encoded = json.dumps(
        material,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "document-readonly:" + _content_hash(encoded)


def _chunk_origin(chunk: Mapping[str, Any], *, session_id: str) -> str:
    """标明文本由谁写入：用户，还是 agent 自己此前的输出。

    二者最终进入同一索引，召回后读取方式也完全相同，这会让模型把自己的旧草稿当成来源
    材料再次引用。在引用中标明来源才能让模型区分二者；该信息会与引用其余部分一起进入
    Prompt。
    """

    file_origin = str(chunk.get("file_origin") or "").strip()
    if file_origin == "agent_output":
        return "agent_output"
    if file_origin == "user_upload":
        return "user_upload"
    if file_origin == "workspace_existing":
        return "workspace"

    from ...workspace.files.attachments import (
        SessionStorageArea,
        classify_session_path,
    )

    try:
        area = classify_session_path(session_id, str(chunk.get("path") or ""))
    except (OSError, ValueError):
        return "workspace"
    if area is SessionStorageArea.OUTPUT:
        return "agent_output"
    if area is SessionStorageArea.INPUT:
        return "user_upload"
    return "workspace"


def _document_processing_citation(document: dict[str, Any]) -> dict[str, str]:
    """将有界的当前版本覆盖事实投影为字符串引用。

    读取器诊断细节不会进入 Prompt。原因码足以告诉模型为什么获准文本不完整，同时保留
    ``Mapping[str, str]`` Source 引用契约。
    """

    status = str(document.get("processing_status") or "legacy_unknown")
    if status not in {"complete", "partial", "legacy_unknown"}:
        status = "legacy_unknown"
    diagnostics = _document_processing_diagnostics(document, status=status)
    if diagnostics is None:
        diagnostic_codes = "null"
        needs_vision = "unknown"
    else:
        codes = sorted({
            diagnostic["code"]
            for diagnostic in diagnostics
            if isinstance(diagnostic.get("code"), str)
            and diagnostic["code"] in DOCUMENT_PROCESSING_COVERAGE_CODES
        })
        diagnostic_codes = json.dumps(
            codes,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        needs_vision = "true" if "page_needs_vision" in codes else "false"
    coverage_gap = (
        "true" if status == "partial"
        else "false" if status == "complete"
        else "unknown"
    )
    return {
        "processing_status": status,
        "processing_diagnostic_codes": diagnostic_codes,
        "needs_vision": needs_vision,
        "coverage_gap": coverage_gap,
    }


def _document_processing_diagnostics(
    document: dict[str, Any],
    *,
    status: str,
) -> list[dict[str, Any]] | None:
    if status == "legacy_unknown":
        return None
    raw = document.get("diagnostics_json")
    if raw is None and "diagnostics" in document:
        raw = document.get("diagnostics")
    if raw is None:
        return []
    try:
        diagnostics = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(diagnostics, list) or not all(
        isinstance(diagnostic, dict) for diagnostic in diagnostics
    ):
        return None
    return diagnostics


def _document_source_coverage_facts(report: dict[str, Any]) -> dict[str, str]:
    """聚合已打开 DocSet 范围内有界的当前版本覆盖。

    这些事实同时存在于 Source 访问和单独引用上，因此无匹配或上下文打包遗漏无法抹去
    所选文档作用域中有部分内容不可读取这一事实。
    """

    documents = [
        document
        for document in report.get("documents", ())
        if isinstance(document, dict)
    ]
    if not documents:
        return {}
    projections = [_document_processing_citation(document) for document in documents]
    statuses = {projection["processing_status"] for projection in projections}
    processing_status = (
        "partial"
        if "partial" in statuses
        else "legacy_unknown"
        if "legacy_unknown" in statuses
        else "complete"
    )
    diagnostic_codes: set[str] = set()
    for projection in projections:
        try:
            raw_codes = json.loads(projection["processing_diagnostic_codes"])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(raw_codes, list):
            diagnostic_codes.update(
                code
                for code in raw_codes
                if isinstance(code, str) and code in DOCUMENT_PROCESSING_COVERAGE_CODES
            )
    needs_values = {projection["needs_vision"] for projection in projections}
    needs_vision = (
        "true" if "true" in needs_values
        else "unknown" if "unknown" in needs_values
        else "false"
    )
    coverage_gap = (
        "true" if processing_status == "partial"
        else "unknown" if processing_status == "legacy_unknown"
        else "false"
    )
    return {
        "processing_status": processing_status,
        "processing_diagnostic_codes": json.dumps(
            sorted(diagnostic_codes),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "needs_vision": needs_vision,
        "coverage_gap": coverage_gap,
    }


def _document_version_map(report: dict[str, Any]) -> dict[str, str] | None:
    """将已验证 docstore 报告规范化为非空 revision 映射。"""

    raw_version_map = report.get("version_map")
    if isinstance(raw_version_map, dict):
        normalized = {str(key): str(value) for key, value in raw_version_map.items()}
    else:
        documents = report.get("documents")
        if not isinstance(documents, list):
            return None
        normalized = {
            str(document["doc_id"]): str(document["version_id"])
            for document in documents
            if isinstance(document, dict)
            and isinstance(document.get("doc_id"), str)
            and document.get("doc_id")
            and isinstance(document.get("version_id"), str)
            and document.get("version_id")
        }
    if not normalized or any(not key.strip() or not value.strip() for key, value in normalized.items()):
        return None
    return normalized


def _session_retrieval_availability(session_id: str) -> SourceAvailability:
    """Session 进入回收站后立即关闭其所有作用域内检索访问。

    会话目录状态由 Session Store 管理，Document 挂载与内容则由项目 Document Store
    管理。派生索引异步更新，因此此受信读取侧门禁会防止软删除会话在延迟窗口中返回仍有
    索引的 Unit。``archived`` 仍可读取；绑定缺失或不可读时以 ``unavailable`` 关闭失败。
    """

    try:
        session = session_store.get_session(session_id)
    except Exception:
        return SourceAvailability.UNAVAILABLE
    if session is None:
        return SourceAvailability.UNAVAILABLE
    return (
        SourceAvailability.BLOCKED
        if session.get("status") == "trashed"
        else SourceAvailability.READY
    )
