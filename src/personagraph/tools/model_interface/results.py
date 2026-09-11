"""模型业务结果投影：保留原生身份和正文，审计字段不重复注入。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy


_FILE_LIST_TOOLS = {
    "check_files_state", "prepare_files", "read_file_chunks", "inspect_file_chunks",
    "search_file_text", "read_file_visuals",
}
_FORMAT_TOOLS = {
    "read_text", "read_pdf_text", "read_word", "read_slides", "inspect_image",
    "analyze_image", "analyze_pdf_page",
}
_FILE_AUDIT = {"content_sha256", "source_content_sha256"}
_RETRIEVAL_AUDIT = {"rank", "query_index", "query_matches", "fusion_score", "reranker_score"}
_FILE_IDENTITY = ("file_id", "file_version_id", "document_id", "document_version_id")
_RESULT_SCOPE_METADATA = {
    "truncated", "partial", "has_more", "omitted_count", "omitted_chunk_count",
    "truncation_reason", "cursor", "next_cursor", "next_offset",
}


def project_tool_result_metadata(metadata: object) -> dict:
    """共享结果范围/续读提示；不把执行审计或未知扩展元信息送给模型。"""
    if not isinstance(metadata, Mapping):
        return {}
    return {key: deepcopy(value) for key, value in metadata.items() if key in _RESULT_SCOPE_METADATA}


def project_file_record(record: dict) -> dict:
    """只清理协议级 hash；不改写正文或替换文件/文档/版本身份。"""
    result = deepcopy(record)
    _drop(result, _FILE_AUDIT)
    return result


def project_tool_result(tool_id: str, result: object) -> object:
    """每种工具只处理已知协议位置；任意用户字典和正文不做递归删字段。"""
    value = deepcopy(result)
    if not isinstance(value, dict):
        return value
    if (
        tool_id in _FILE_LIST_TOOLS | {"retrieve_files"}
        and set(value) == {"chunk", "source_context"}
        and isinstance(value["chunk"], dict)
        and isinstance(value["source_context"], list)
    ):
        return _selected_chunk(value)
    if tool_id in _FILE_LIST_TOOLS:
        _project_rows(value, "results", _file_result)
        _project_rows(value, "unavailable_targets", project_file_record)
    elif tool_id == "retrieve_files":
        rows = value.pop("evidence", None)
        if isinstance(rows, list):
            value["files"] = _group_file_evidence(rows)
    elif tool_id in {"list_file_visuals", "create_output_file"}:
        value = project_file_record(value)
    elif tool_id == "list_tool_results":
        _project_rows(value, "results", _history_item)
    elif tool_id == "read_tool_result":
        if isinstance(value.get("source"), dict):
            value["source"] = _history_source(value["source"])
        # value 已由 history 工具选择并分页；不能当成当前工具协议再次解释。
    elif tool_id in _FORMAT_TOOLS:
        if isinstance(value.get("source"), dict):
            value["source"].pop("sha256", None)
        for item in value.get("images", []):
            if isinstance(item, dict):
                _drop(item, {"sent_sha256", "source_sha256"})
    elif tool_id in {"record_execution_findings", "revise_execution_finding"}:
        _drop(value, {"schema_version", "source_schema_version", "replayed"})
    else:
        return value
    _drop(value, {"contract_version", "schema_version"})
    return value


def _group_file_evidence(rows: list) -> list[dict]:
    groups: dict[tuple, dict] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        item = _file_result(raw)
        _drop(item, _RETRIEVAL_AUDIT)
        key = tuple(item.get(field) for field in _FILE_IDENTITY)
        if key not in groups:
            groups[key] = {
                field: item[field] for field in _FILE_IDENTITY if field in item
            }
            source = item.get("source")
            if isinstance(source, dict):
                groups[key]["source"] = {
                    field: value for field, value in source.items()
                    if field not in {"corpus_recorded_at"}
                }
        for field in (*_FILE_IDENTITY, "source"):
            item.pop(field, None)
        kind = item.pop("evidence_type", None)
        collection = "chunks" if kind == "document_chunk" else "observations"
        groups[key].setdefault(collection, []).append(item)
    return list(groups.values())


def _file_result(item: dict) -> dict:
    result = project_file_record(item)
    for chunk in result.get("chunks", []):
        if isinstance(chunk, dict):
            _drop(chunk, _FILE_AUDIT)
    return result


def _selected_chunk(value: dict) -> dict:
    """只处理 Host 精确取块形成的封套，正文中的用户字典保持原样。"""
    audit = _FILE_AUDIT | _RETRIEVAL_AUDIT | {"contract_version", "schema_version", "corpus_recorded_at"}
    chunk = _file_result(value["chunk"])
    _drop(chunk, audit)
    if isinstance(chunk.get("source"), dict):
        _drop(chunk["source"], audit)
    contexts = []
    for context in value["source_context"]:
        if isinstance(context, dict):
            _drop(context, audit)
            if context:
                contexts.append(context)
    return {"chunk": chunk, "source_context": contexts}


def _history_source(source: dict) -> dict:
    return {
        key: deepcopy(source[key])
        for key in ("tool_result_id", "tool_id", "status") if key in source
    }


def _history_item(item: dict) -> dict:
    result = deepcopy(item)
    result["source"] = _history_source(result["source"])
    result.pop("arguments_sha256", None)
    return result


def _drop(value: dict, keys: set[str]) -> None:
    for key in keys:
        value.pop(key, None)


def _project_rows(value: dict, key: str, project) -> None:
    rows = value.get(key)
    if isinstance(rows, list):
        value[key] = [project(item) if isinstance(item, dict) else item for item in rows]
