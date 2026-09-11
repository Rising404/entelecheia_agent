"""将 retrieve_files 接到原生 File / FileVersion 的只读检索数据面。

adapter 负责范围绑定、实时访问权复查与最终 evidence 投影；共享 retrieval port
执行查询。轨迹审计应以最终投影为准，不能把后来被权限/来源检查丢弃的命中记成
已交给模型的证据。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any

from ...retrieval.contracts import FILE_RETRIEVAL_RRF_LIMIT_PER_QUERY, SourceType
from ...retrieval.execution import checkpoint, measure
from ...retrieval.ports import RetrievalCancelled
from ...retrieval.lifecycle.generation import RetrievalGenerationSpec
from ...retrieval.sources.identity import (
    mounted_document_chunk_ref_and_content,
    picture_observation_source_unit_id,
)
from ...retrieval.tooling.contracts import (
    FileRetrievalOrigin, FileRetrievalReadinessResult, FileRetrievalReadinessStatus,
    FileRetrievalScope, FrozenFileRetrievalBinding, RetrievalCorpus,
    RetrievalStatus, RetrievalToolRequest, RetrievalToolResult,
)
from ...retrieval.tooling.service import RetrievalServiceToolPort
from ...workspace.documents.reading import (
    CurrentFileChunk,
    CurrentFileDocument,
    get_current_file_document,
    read_current_file_chunk,
)
from ...workspace.files.access import AuthorizedFileSource
from ..effects import EffectScopeKind
from ..execution import ToolBusinessFailure
from ..execution_context import current_tool_execution
from .execution_bridge import file_retrieval_execution
from .file_retrieval_tools import (
    DEFAULT_FILE_RETRIEVAL_RESULT_LIMIT, DEFAULT_FILE_RETRIEVAL_TOKEN_LIMIT,
    MAX_FILES_PER_RETRIEVE, MAX_CHUNK_CONTENT_CHARS, MAX_QUERY_CHARS,
    MAX_FILE_RETRIEVAL_OUTPUT_BYTES,
    MAX_FILE_RETRIEVAL_ITEMS, MAX_FILE_RETRIEVAL_QUERIES, MAX_FILE_RETRIEVAL_TOKEN_LIMIT,
    RETRIEVE_FILES_CONTRACT_VERSION, build_retrieve_files_registration,
)
from .public_projection import sanitize_public_locator


@dataclass
class _FileReadinessBridge:
    runtime: "FileRetrievalRuntime"
    sources: Mapping[str, AuthorizedFileSource]
    observed: dict[str, FileRetrievalReadinessResult] = field(default_factory=dict)

    def ensure_ready(self, request, binding):
        if (
            request.session_id != self.runtime.session_id
            or request.scope_snapshot_id != self.runtime.scope_id
            or request.retrieval_data_version != self.runtime.generation_spec.version_id
            or binding.source_id != binding.file_id
            or binding.authority_id != binding.file_id
        ):
            raise ValueError("retrieval request crossed File authority")
        source = self.sources.get(binding.file_id)
        status = FileRetrievalReadinessStatus.BLOCKED
        reason = "file_access_unavailable"
        document = None
        if source is not None and self.runtime.revalidate(source):
            document = self.runtime.read_document(
                file_id=source.file_id, file_version_id=source.file_version_id,
                session_id=self.runtime.session_id,
            )
            status = FileRetrievalReadinessStatus.READY if document else FileRetrievalReadinessStatus.PENDING
            reason = None if document else "file_not_prepared"
        result = FileRetrievalReadinessResult(
            status=status, file_id=binding.file_id, reason_code=reason,
            retrieval_data_version=self.runtime.generation_spec.version_id,
            document_id=document.document_id if document else None,
            document_version_id=document.document_version_id if document else None,
        )
        self.observed[binding.file_id] = result
        return result


@dataclass(frozen=True, slots=True)
class FileRetrievalRuntime:
    session_id: str = field(repr=False)
    scope_id: str = field(repr=False)
    generation_spec: RetrievalGenerationSpec = field(repr=False)
    resolve_file: Callable[..., AuthorizedFileSource] = field(repr=False)
    revalidate: Callable[[AuthorizedFileSource], bool] = field(repr=False)
    port_factory: Callable = field(repr=False)
    picture_file_authority: object | None = field(default=None, repr=False)
    read_document: Callable[..., CurrentFileDocument | None] = field(
        default=get_current_file_document, repr=False,
    )
    read_chunk: Callable[..., CurrentFileChunk | None] = field(
        default=read_current_file_chunk, repr=False,
    )
    filesystem_scope_kind: EffectScopeKind = EffectScopeKind.WORKSPACE

    def __post_init__(self):
        if not self.session_id or not self.scope_id:
            raise ValueError("File retrieval requires a Session and authority scope")
        if not all(callable(value) for value in (
            self.resolve_file, self.revalidate, self.port_factory, self.read_document, self.read_chunk,
        )):
            raise TypeError("File retrieval ports must be callable")

    @property
    def effect_scope(self):
        return "file-corpus:" + hashlib.sha256(self.scope_id.encode()).hexdigest()

    def registration(self):
        return build_retrieve_files_registration(
            handler=self.retrieve, effect_scope=self.effect_scope,
            filesystem_scope_kind=self.filesystem_scope_kind,
        )

    def retrieve(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with file_retrieval_execution():
            return self._retrieve(payload)

    def _retrieve(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """解析 queries / file_ids，绑定冻结 generation，再返回当前可授权的最终 evidence。

        file_ids 缺省检索 Session 语料；显式列表只收窄范围。支持 deferred audit 的 port
        先返回未记录结果，_project 复查后才记 final_projection 与丢弃诊断，避免轨迹
        收录未实际交付的命中。这里消费工具给定的 query，不额外调用 LLM 生成 query。
        """

        checkpoint()
        queries, file_ids, limit, token_limit = _parse_request(payload)
        sources, access_gaps = self._resolve_selected(file_ids)
        scope = FileRetrievalScope.SESSION_CORPUS if file_ids is None else FileRetrievalScope.SELECTED_FILES
        if file_ids is not None and not sources:
            return _response(queries, scope, [], access_gaps, len(file_ids), 0, blocked=True)
        bridge = _FileReadinessBridge(self, sources)
        bindings = tuple(FrozenFileRetrievalBinding(
            source_id=source.file_id, authority_id=source.file_id,
            origin=_retrieval_origin(source.origin), project_id=source.project_id,
            file_id=source.file_id, file_version_id=source.file_version_id,
            expected_content_sha256=source.fingerprint.sha256,
        ) for source in sources.values())
        inventory = self.picture_file_authority.freeze_inventory() if (
            file_ids is None and self.picture_file_authority is not None
        ) else None
        request = RetrievalToolRequest(
            request_id=_request_id(self.scope_id, queries, file_ids), corpus=RetrievalCorpus.FILES,
            query=queries[0], query_variants=queries[1:], limit=limit,
            context_token_limit=token_limit, session_id=self.session_id,
            scope_snapshot_id=self.scope_id, retrieval_data_version=self.generation_spec.version_id,
            file_scope=scope, file_bindings=bindings,
            session_file_bindings=inventory.bindings if inventory else (),
            file_inventory_complete=inventory.complete if inventory else True,
        )
        port = self.port_factory(bridge)
        audit = None
        publication: dict[str, Any] = {"response": None, "diagnostics": ()}
        invocation = current_tool_execution()

        def record_settlement(status: str) -> None:
            response = publication["response"]
            diagnostics = publication["diagnostics"]
            if status != "succeeded" or response is None:
                response = {
                    "status": "blocked", "evidence": [],
                    "gaps": [{"code": f"tool_{status}", "blocking": True}],
                }
                diagnostics = (*diagnostics, {
                    "stage": "tool_settlement", "code": f"tool_{status}",
                    "status": status, "blocking": True, "known_count": 1,
                })
            if audit is not None:
                audit.record(final_projection=response, diagnostics=diagnostics)

        try:
            deferred = getattr(port, "retrieve_file_readonly_unrecorded", None)
            if callable(deferred):
                result, audit = deferred(request)
                if not callable(getattr(audit, "record", None)):
                    raise TypeError("invalid deferred retrieval audit")
                if invocation is not None:
                    invocation.on_settled(record_settlement)
            else:
                result = port.retrieve(request)
        except RetrievalCancelled:
            raise
        except Exception as exc:
            raise ToolBusinessFailure("retrieval_backend_unavailable", "The prepared corpus could not complete the query.") from exc
        diagnostics = []
        checkpoint()
        if not isinstance(result, RetrievalToolResult):
            response = _response(queries, scope, [], [_gap("retrieval_backend_contract_violation")], len(file_ids or ()), 0, blocked=True)
        else:
            with measure("tool_projection"):
                response, diagnostics = self._project(request, result, bridge, access_gaps)
        publication.update(response=response, diagnostics=tuple(diagnostics))
        checkpoint()
        # Audit-after-projection：只记录最终可交付引用，连同 Host 丢弃命中的诊断。
        if audit is not None and invocation is None:
            record_settlement("succeeded")
        return response

    def _resolve_selected(self, file_ids):
        sources, gaps = {}, []
        for file_id in file_ids or ():
            checkpoint()
            try:
                source = self.resolve_file(file_id=file_id)
                if source is None or source.file_id != file_id or not self.revalidate(source):
                    raise ValueError("File access unavailable")
                if str(source.origin) == "agent_output":
                    raise ValueError("agent outputs are outside the evidence corpus")
                sources[file_id] = source
            except Exception:
                gaps.append(_gap("file_access_unavailable", file_id=file_id))
        return sources, gaps

    def _project(self, request, result, bridge, access_gaps):
        requested = len(bridge.sources) + len(access_gaps)
        if (
            not result.scope_is_current or result.scope_snapshot_id != self.scope_id
            or result.retrieval_data_version != self.generation_spec.version_id
        ):
            return _response(request.queries, request.file_scope, [], [_gap("file_scope_stale")], requested, 0, blocked=True), []
        evidence, diagnostics, gaps = [], [], list(access_gaps)
        for gap in result.gaps:
            if gap.authority_id is not None and gap.authority_id not in bridge.sources:
                return _response(request.queries, request.file_scope, [], [_gap("retrieval_authority_mismatch")], requested, 0, blocked=True), []
            gaps.append({"code": gap.code, "blocking": gap.blocking,
                         "file_id": gap.authority_id, "known_count": gap.known_count})
        for raw in result.evidence:
            checkpoint()
            try:
                projected = self._project_evidence(request, raw, bridge.sources)
                evidence.append(projected)
                if raw.citation.get("location") and projected["locator"] is None:
                    gaps.append(_gap("unsafe_locator_omitted", file_id=projected["file_id"], blocking=False))
            except Exception:
                gaps.append(_gap("evidence_authority_unavailable", file_id=raw.authority_id))
                diagnostics.append({"stage": "tool_projection", "code": "evidence_authority_unavailable",
                                    "status": "dropped", "blocking": True, "known_count": 1,
                                    "file_id": raw.authority_id, "source_type": raw.source_type.value,
                                    "source_unit_id": raw.source_unit_id, "source_revision": raw.source_revision})
        evidence.sort(key=lambda item: (item["rank"], item["evidence_type"], item.get("chunk_id") or item.get("observation_id")))
        ready = sum(value.status is FileRetrievalReadinessStatus.READY for value in bridge.observed.values())
        return _response(
            request.queries, request.file_scope, evidence[:request.limit], gaps,
            requested, ready, blocked=result.status is RetrievalStatus.BLOCKED,
            partial=result.status is RetrievalStatus.PARTIAL,
            truncated=result.truncated or len(evidence) > request.limit or any(item["snippet_truncated"] for item in evidence),
        ), diagnostics

    def _project_evidence(self, request, raw, selected):
        """把一个检索命中重新绑定到可访问的精确文件版本，再生成公开证据。

        检查 query lane、File lineage、scope、来源与内容 hash；投影前后都复查访问权，
        防止检索期间撤权或版本变化。任一失败交给 _project 丢弃并记 gap/diagnostic，
        不用新版本正文替换旧命中的身份。
        """

        if raw.source_type not in {SourceType.DOCUMENT, SourceType.PICTURE}:
            raise ValueError("unexpected source type")
        if raw.query_index >= len(request.queries):
            raise ValueError("evidence query lane is outside this request")
        if not raw.query_matches or any(
            match.query_index >= len(request.queries) or match.rank > FILE_RETRIEVAL_RRF_LIMIT_PER_QUERY
            for match in raw.query_matches
        ):
            raise ValueError("evidence query_matches are outside this request")
        citation = raw.citation
        file_id, file_version_id = citation.get("file_id"), citation.get("file_version_id")
        if not file_id or not file_version_id:
            raise ValueError("evidence has no exact File lineage")
        if request.file_scope is FileRetrievalScope.SELECTED_FILES:
            source = selected.get(file_id)
            if source is None or raw.authority_id != file_id or source.file_version_id != file_version_id:
                raise ValueError("evidence crossed the selected File scope")
        else:
            if raw.authority_id is not None:
                raise ValueError("unexpected per-File authority on corpus result")
            source = self.resolve_file(file_id=file_id, file_version_id=file_version_id)
        if source is None or source.file_id != file_id or source.file_version_id != file_version_id:
            raise ValueError("evidence File identity mismatched")
        if str(source.origin) == "agent_output" or not self.revalidate(source):
            raise ValueError("evidence File authority changed")
        if raw.source_type is SourceType.DOCUMENT:
            result, public_content, public_hash, recorded_at = self._project_document(raw, source)
        else:
            result = self._project_picture(raw, source)
            recorded_at = citation.get("created_at")
            public_content, public_hash = raw.content, raw.indexed_content_hash
        if not raw.content or hashlib.sha256(raw.content.encode()).hexdigest() != raw.indexed_content_hash:
            raise ValueError("evidence content hash mismatched")
        if not self.revalidate(source):
            raise ValueError("File authority changed during projection")
        result.update({
            "file_id": file_id,
            "file_version_id": file_version_id,
            "snippet": public_content[:MAX_CHUNK_CONTENT_CHARS],
            "snippet_truncated": len(public_content) > MAX_CHUNK_CONTENT_CHARS,
            "content_character_count": len(public_content),
            "content_sha256": public_hash,
            "rank": raw.rank,
            "query_index": raw.query_index,
            "query_matches": [{
                "query_index": match.query_index, "rank": match.rank,
                "fused_rank": match.fused_rank, "fusion_score": match.fusion_score,
                "reranker_score": match.reranker_score,
            } for match in raw.query_matches],
            "source": _source_metadata(source, recorded_at),
        })
        return result

    def _project_document(self, raw, source):
        chunk = self.read_chunk(
            file_id=source.file_id,
            file_version_id=source.file_version_id,
            document_version_id=raw.source_revision,
            session_id=self.session_id,
            chunk_id=raw.citation.get("chunk_id"),
        )
        if chunk is None or chunk.document.document_id != raw.citation.get("doc_id"):
            raise ValueError("evidence chunk unavailable")
        ref, content = mounted_document_chunk_ref_and_content(
            session_id=self.session_id,
            chunk_id=chunk.chunk_id,
            source_version_id=chunk.document.document_version_id,
            content=chunk.content,
            doc_id=chunk.document.document_id,
            producer_chunk_id=chunk.producer_chunk_id,
        )
        if (ref.source_unit_id, ref.source_revision, ref.indexed_content_hash, content) != (
            raw.source_unit_id, raw.source_revision, raw.indexed_content_hash, raw.content,
        ):
            raise ValueError("retrieved chunk failed source verification")
        current_document = self.read_document(
            file_id=source.file_id,
            file_version_id=source.file_version_id,
            session_id=self.session_id,
        )
        if current_document != chunk.document:
            raise ValueError("Document generation changed during evidence projection")
        result = {
            "evidence_type": "document_chunk",
            "document_id": chunk.document.document_id,
            "document_version_id": chunk.document.document_version_id,
            "chunk_id": chunk.chunk_id,
            "chunk_sequence": chunk.sequence,
            "locator": sanitize_public_locator(chunk.locator),
        }
        return result, chunk.content, chunk.content_sha256, chunk.document.added_at

    def _project_picture(self, raw, source):
        if self.picture_file_authority is None:
            raise ValueError("Picture authority unavailable")
        citation = raw.citation
        decision = self.picture_file_authority.authorize(
            session_id=self.session_id,
            file_id=source.file_id,
            file_version_id=source.file_version_id,
            picture_id=citation.get("picture_id"),
        )
        if not decision.allowed:
            raise ValueError("Picture evidence access denied")
        identities = {
            key: citation.get(key) for key in ("picture_id", "picture_unit_id", "observation_id")
        }
        if any(not value for value in identities.values()):
            raise ValueError("Picture evidence identity unavailable")
        if raw.source_unit_id != picture_observation_source_unit_id(identities["observation_id"]):
            raise ValueError("Picture observation SourceUnit identity mismatched")
        return {
            "evidence_type": "picture_observation",
            **identities,
            "locator": _picture_public_locator(citation),
        }


def build_file_retrieval_runtime(
    *, session_id, scope_id, generation_spec, resolve_file, revalidate,
    file_foundation=None, port_factory=None, trajectory_turn_id=None,
    picture_file_authority=None, read_document=get_current_file_document,
    read_chunk=read_current_file_chunk, filesystem_scope_kind=EffectScopeKind.WORKSPACE,
):
    if port_factory is None:
        if file_foundation is None:
            raise ValueError("File retrieval requires a foundation")
        def port_factory(bridge):
            return RetrievalServiceToolPort(
                    file_foundation=file_foundation, file_readiness=bridge, file_readonly=True,
                    trajectory_turn_id=trajectory_turn_id,
                )
    return FileRetrievalRuntime(
        session_id=session_id, scope_id=scope_id, generation_spec=generation_spec,
        resolve_file=resolve_file, revalidate=revalidate, port_factory=port_factory,
        picture_file_authority=picture_file_authority, read_document=read_document,
        read_chunk=read_chunk, filesystem_scope_kind=filesystem_scope_kind,
    )


def _parse_request(payload):
    if isinstance(payload, Mapping) and ({"limit", "per_query_limit"} & set(payload)):
        raise ToolBusinessFailure("invalid_request", "limit 与 per_query_limit 已退役；请使用 result_limit（合并去重后总计 1-96 块）。")
    if not isinstance(payload, Mapping) or set(payload) - {"queries", "file_ids", "result_limit", "context_token_limit"}:
        raise ToolBusinessFailure("invalid_request", "The retrieval request is invalid.")
    queries = _strings(payload.get("queries"), MAX_FILE_RETRIEVAL_QUERIES, MAX_QUERY_CHARS, "queries", trim=True)
    file_ids = _strings(payload["file_ids"], MAX_FILES_PER_RETRIEVE, 256, "file_ids") if "file_ids" in payload else None
    limit = _integer(payload.get("result_limit", DEFAULT_FILE_RETRIEVAL_RESULT_LIMIT), MAX_FILE_RETRIEVAL_ITEMS, "result_limit")
    tokens = _integer(payload.get("context_token_limit", DEFAULT_FILE_RETRIEVAL_TOKEN_LIMIT), MAX_FILE_RETRIEVAL_TOKEN_LIMIT, "context_token_limit")
    return queries, file_ids, limit, tokens


def _strings(value, maximum, max_chars, name, *, trim=False):
    if not isinstance(value, list) or not 1 <= len(value) <= maximum or any(not isinstance(item, str) for item in value):
        raise ToolBusinessFailure("invalid_request", f"{name} must be a bounded non-empty string list.")
    values = tuple(item.strip() if trim else item for item in value)
    if any(not item or len(item) > max_chars or (not trim and item != item.strip()) for item in values) or len(values) != len(set(values)):
        raise ToolBusinessFailure("invalid_request", f"{name} must contain unique bounded values.")
    return values


def _integer(value, maximum, name):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ToolBusinessFailure("invalid_request", f"{name} is outside the supported range.")
    return value


def _gap(code, *, file_id=None, blocking=True):
    return {"code": code, "blocking": blocking, "file_id": file_id, "known_count": 1}


def _response(queries, scope, evidence, gaps, requested, ready, *, blocked=False, partial=False, truncated=False):
    evidence = [] if blocked else evidence
    status = "blocked" if blocked else "partial" if partial or gaps or truncated else "complete"
    response = {"contract_version": RETRIEVE_FILES_CONTRACT_VERSION, "retrieval_scope": scope.value,
            "status": status, "outcome": "matched" if evidence else "no_match" if status == "complete" else "not_established",
            "queries": list(queries), "evidence": evidence, "gaps": gaps[:MAX_FILES_PER_RETRIEVE],
            "coverage": {"requested_file_count": requested, "document_ready_file_count": ready,
                         "returned_evidence_count": len(evidence),
                         "document_chunk_count": sum(item["evidence_type"] == "document_chunk" for item in evidence),
                         "picture_observation_count": sum(item["evidence_type"] == "picture_observation" for item in evidence)},
            "truncated": bool(truncated or len(gaps) > MAX_FILES_PER_RETRIEVE)}
    omitted = 0
    while len(json.dumps(response, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")) > MAX_FILE_RETRIEVAL_OUTPUT_BYTES:
        if not evidence:
            raise ToolBusinessFailure("invalid_response", "File retrieval metadata exceeds the response byte budget.")
        evidence.pop()
        omitted += 1
        response["status"], response["truncated"] = "partial", True
        response["outcome"] = "matched" if evidence else "not_established"
        response["gaps"] = [*gaps[:MAX_FILES_PER_RETRIEVE - 1], {
            "code": "response_byte_limit_exceeded", "blocking": False,
            "file_id": None, "known_count": omitted,
        }]
        response["coverage"].update({
            "returned_evidence_count": len(evidence),
            "document_chunk_count": sum(item["evidence_type"] == "document_chunk" for item in evidence),
            "picture_observation_count": sum(item["evidence_type"] == "picture_observation" for item in evidence),
        })
    return response


def _request_id(scope_id, queries, file_ids):
    material = json.dumps([scope_id, queries, file_ids], ensure_ascii=False, separators=(",", ":"))
    return "file_retrieve_" + hashlib.sha256(material.encode()).hexdigest()


def _source_metadata(source, recorded_at):
    origin = _retrieval_origin(source.origin).value
    name = source.file_name.replace("\\", "/").rsplit("/", 1)[-1]
    return {"file_name": name if name and len(name) <= 512 and not any(ord(c) < 32 for c in name) else None,
            "relative_path": None if origin == "user_upload" else sanitize_public_locator(source.relative_path, maximum=2000),
            "origin": origin, "source_modified_at": _source_mtime(source.fingerprint.mtime_ns),
            "corpus_recorded_at": _safe_timestamp(recorded_at)}


def _retrieval_origin(origin):
    return FileRetrievalOrigin("workspace" if str(origin) == "workspace_existing" else str(origin))


def _source_mtime(value):
    try:
        return datetime.fromtimestamp(int(value) / 1_000_000_000, tz=timezone.utc).isoformat() if not isinstance(value, bool) and int(value) >= 0 else None
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _safe_timestamp(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 64:
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).isoformat()
    except ValueError:
        return None


def _picture_public_locator(citation):
    kind = {"pdf_page": "page", "pptx_slide": "slide"}.get(citation.get("surface_kind"))
    ordinal = citation.get("surface_ordinal")
    return f"{kind} {int(ordinal)}" if kind and isinstance(ordinal, str) and ordinal.isdecimal() and 1 <= int(ordinal) < 1_000_000 else None


__all__ = ["FileRetrievalRuntime", "build_file_retrieval_runtime"]
