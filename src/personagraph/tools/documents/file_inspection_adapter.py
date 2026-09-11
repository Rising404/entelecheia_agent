"""将会话文件授权和文档查询服务连接到字面查找/分块目录协议。"""

from collections.abc import Callable
from dataclasses import dataclass

from personagraph.workspace.documents.inspection import inspect_current_file_document, find_literal_text
from personagraph.workspace.documents.reading import get_current_file_document
from personagraph.workspace.files.access import FileAccessError
from ..execution import ToolBusinessFailure
from ..schema_validation import ToolSchemaCompiler, SchemaValidationError
from ..retrieval.public_projection import sanitize_public_locator
from .file_inspection_tools import FILE_INSPECTION_TOOL_IDS, build_file_inspection_registration


@dataclass(frozen=True, slots=True)
class FileInspectionRuntime:
    session_id: str
    effect_scope: str
    resolve_file: Callable
    revalidate: Callable
    inspect_document: Callable = inspect_current_file_document
    read_document: Callable = get_current_file_document

    @property
    def registrations(self):
        return tuple(build_file_inspection_registration(
            tool_id=tool_id, handler=self.search if tool_id == "search_file_text" else self.inspect,
            effect_scope=self.effect_scope,
        ) for tool_id in FILE_INSPECTION_TOOL_IDS)

    def inspect(self, payload):
        return self._run(payload, searching=False)

    def search(self, payload):
        return self._run(payload, searching=True)

    def _run(self, payload, *, searching):
        tool_id = "search_file_text" if searching else "inspect_file_chunks"
        registration = next(item for item in self.registrations if item.tool_id == tool_id)
        try:
            ToolSchemaCompiler().compile(registration.spec.input_schema, role="input").validate(payload)
        except SchemaValidationError as exc:
            raise ToolBusinessFailure("invalid_request", str(exc)) from exc
        if not searching:
            for target in payload["targets"]:
                if ("chunk_ids" in target or "chunk_sequences" in target) and (
                    "start_sequence" in target or "limit" in target
                ):
                    raise ToolBusinessFailure("invalid_request", "Use selectors or a sequence window, not both.")
        return {"contract_version": "file-inspection-v1", "results": [
            self._target(target, payload, searching=searching) for target in payload["targets"]
        ]}

    def _target(self, target, payload, *, searching):
        result = _empty_result(target, searching)
        try:
            source = self.resolve_file(file_id=target["file_id"])
            if source is None or source.file_id != target["file_id"] or not self.revalidate(source):
                result["reason_code"] = "file_access_unavailable"
                return result
            snapshot = self.inspect_document(
                file_id=source.file_id, file_version_id=source.file_version_id,
                document_version_id=target["document_version_id"], session_id=self.session_id,
                include_source_text=searching,
            )
            if snapshot is None:
                result["reason_code"] = "document_version_unavailable"
                return result
            if (snapshot.document.file_id, snapshot.document.file_version_id,
                snapshot.document.document_version_id) != (
                source.file_id, source.file_version_id, target["document_version_id"],
            ):
                result["reason_code"] = "document_version_unavailable"
                return result
            if searching:
                values = find_literal_text(snapshot, text=payload["text"],
                                           case_sensitive=payload.get("case_sensitive", False),
                                           whitespace=payload.get("whitespace", "exact"),
                                           match_offset=payload.get("match_offset", 0), limit=payload.get("limit", 50))
                for match in values["matches"]:
                    match["locators"] = [sanitize_public_locator(locator) or "" for locator in match["locators"]]
            else:
                values = _locations(snapshot, target)
            if not self.revalidate(source):
                result["reason_code"] = "file_access_changed"
                return result
            if self.read_document(file_id=source.file_id, file_version_id=source.file_version_id,
                                  session_id=self.session_id) != snapshot.document:
                result["reason_code"] = "document_version_changed"
                return result
            result.update(values, file_version_id=source.file_version_id,
                          processing_status=snapshot.document.processing_status)
            result["status"] = ("unavailable" if result["reason_code"] else
                                "partial" if (result.get("unavailable_selectors") or
                                              snapshot.document.processing_status == "partial") else "ready")
            return result
        except FileAccessError:
            result["reason_code"] = "file_access_unavailable"
            return result
        except (ValueError, TypeError):
            result["reason_code"] = "document_snapshot_invalid"
            return result


def _locations(snapshot, target):
    selectors = target.get("chunk_ids", target.get("chunk_sequences"))
    missing = []
    if selectors is None:
        start, limit = target.get("start_sequence", 0), target.get("limit", 32)
        selected = snapshot.chunks[start:start + limit]
        next_sequence = start + len(selected)
        next_sequence = next_sequence if next_sequence < len(snapshot.chunks) else None
    else:
        index = {getattr(chunk, "chunk_id" if "chunk_ids" in target else "sequence"): chunk
                 for chunk in snapshot.chunks}
        selected = [index[selector] for selector in selectors if selector in index]
        missing = [selector for selector in selectors if selector not in index]
        next_sequence = None
    total = snapshot.document.total_chunk_count
    return {"total_chunk_count": total, "first_sequence": 0 if total else None,
            "last_sequence": total - 1 if total else None,
            "physical_page_count": snapshot.document.physical_page_count, "next_sequence": next_sequence,
            "chunks": [{"chunk_id": chunk.chunk_id, "sequence": chunk.sequence,
                        "locator": sanitize_public_locator(chunk.locator) or "", "source_pages": list(chunk.source_pages)} for chunk in selected],
            "unavailable_selectors": missing}


def _empty_result(target, searching):
    result = {"file_id": target["file_id"], "document_version_id": target["document_version_id"],
              "file_version_id": None, "status": "unavailable", "reason_code": None, "processing_status": None}
    if searching:
        result.update(scan_complete=False, total_match_count=None, scanned_element_count=0,
                      next_match_offset=None, matches_truncated=False, matches=[])
    else:
        result.update(total_chunk_count=None, first_sequence=None, last_sequence=None,
                      physical_page_count=None, next_sequence=None, chunks=[], unavailable_selectors=[])
    return result
