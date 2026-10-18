"""L1 文件短引用：从冻结附件和已验证回执重建，准入时还原原生身份。

这里只转换工具协议中的身份位置，不改正文、不执行工具、不建立可变注册表。
编号按冻结附件的路径顺序及持久调用坐标追加；旧结果退出展示不会重编号。
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json

from ....persistent_turn_content.evidence import l1_calls_by_ref, parse_l1_call_ref


_SELECTOR_ARRAYS = {
    "prepare_files": "files", "check_files_state": "files",
    "read_file_chunks": "targets", "inspect_file_chunks": "targets",
    "search_file_text": "targets", "read_file_visuals": "requests",
}
_FILE_TOOLS = {*_SELECTOR_ARRAYS, "list_file_visuals", "retrieve_files"}
_IDENTITY_FIELDS = {"file_id", "file_version_id"}
_CATALOG_FIELDS = (
    "file_id", "file_version_id", "name", "relative_path", "media_type", "size_bytes", "origin",
)
_REF_SCHEMA = {"type": "string", "pattern": r"^f[1-9][0-9]*$"}


class FileReferenceAdmissionError(ValueError):
    def __init__(self, path: str, available: tuple[str, ...]):
        self.path = path
        self.available = available
        super().__init__("file selector must use an available L1 file_ref")


def build_file_reference_catalog(
    attachments: Sequence[Mapping], execution: Mapping,
) -> list[dict]:
    """同一文件版本只占一个位置；新观察的版本追加，不覆盖旧绑定。"""
    records = {}

    def add(row):
        if not isinstance(row, Mapping):
            return
        identity = (row.get("file_id"), row.get("file_version_id"))
        if not all(isinstance(value, str) and value for value in identity):
            return  # 未登记/无法解析的文件没有可绑定身份，仍可由原路径准备。
        if identity in records:
            return
        entry = {key: deepcopy(row[key]) for key in _CATALOG_FIELDS if key in row}
        source = row.get("source")
        if isinstance(source, Mapping):
            for key in ("relative_path", "media_type"):
                if key in source:
                    entry.setdefault(key, source[key])
            entry.setdefault("name", source.get("file_name"))
        if row.get("file_name"):
            entry.setdefault("name", row["file_name"])
        records[identity] = {"file_ref": f"f{len(records) + 1}", **entry}

    for row in sorted(attachments, key=lambda item: (
        item["relative_path"], item["file_id"], item["file_version_id"],
    )):
        add(row)
    calls = l1_calls_by_ref(execution)
    for ref in sorted(calls, key=parse_l1_call_ref):
        call = calls[ref]
        if call["status"] != "succeeded" or call["tool_id"] not in _FILE_TOOLS | {"create_output_file"}:
            continue
        raw = call.get("outcome_json")
        if not isinstance(raw, str) or hashlib.sha256(raw.encode()).hexdigest() != call.get("outcome_hash"):
            raise ValueError("file reference source outcome hash mismatch")
        outcome = json.loads(raw)
        if outcome.get("status") != call["status"]:
            raise ValueError("file reference source status mismatch")
        for row in _result_records(call["tool_id"], outcome.get("result")):
            add(row)
    return list(records.values())


def _result_records(tool_id: str, result: object):
    if not isinstance(result, dict):
        return
    if tool_id in _FILE_TOOLS and set(result) == {"chunk", "source_context"}:
        yield result["chunk"]
        yield from result["source_context"]
        return
    if tool_id in {"list_file_visuals", "create_output_file"}:
        yield result
    elif tool_id in _SELECTOR_ARRAYS:
        yield from result.get("results", [])
        yield from result.get("unavailable_targets", [])
    elif tool_id == "retrieve_files":
        yield from result.get("evidence", result.get("files", []))
        yield from result.get("gaps", [])


class FileReferences:
    """一份 Host 请求快照的文件引用；调用方不能提供其他 Run 的路由。"""

    def __init__(self, catalog: Sequence[Mapping]):
        self.catalog = deepcopy(list(catalog))
        self.by_ref = {row["file_ref"]: row for row in self.catalog}
        self.by_identity = {(row["file_id"], row["file_version_id"]): row["file_ref"] for row in self.catalog}
        if len(self.by_ref) != len(self.catalog) or len(self.by_identity) != len(self.catalog):
            raise ValueError("duplicate file reference binding")

    def project_record(self, row: dict) -> dict:
        projected = deepcopy(row)
        if "file_id" not in row:
            return projected
        ref = self.by_identity.get((row.get("file_id"), row.get("file_version_id")))
        if ref is None and row.get("file_version_id") is None:
            matches = [value for (file_id, _), value in self.by_identity.items() if file_id == row.get("file_id")]
            if len(matches) == 1:
                ref = matches[0]
        for key in _IDENTITY_FIELDS:
            projected.pop(key, None)
        if ref is not None:
            projected["file_ref"] = ref
        return projected

    def project_result(self, tool_id: str, result: object) -> object:
        projected = deepcopy(result)
        for row in _result_records(tool_id, projected):
            if isinstance(row, dict):
                replacement = self.project_record(row)
                row.clear()
                row.update(replacement)
        return projected

    def project_arguments(self, tool_id: str, arguments: dict) -> dict:
        projected = deepcopy(arguments)
        if tool_id == "retrieve_files" and "file_ids" in projected:
            refs = [self.project_record({"file_id": identity}).get("file_ref") for identity in projected.pop("file_ids")]
            projected["file_refs"] = refs
        for row in _selectors(tool_id, projected):
            replacement = self.project_record(row)
            row.clear()
            row.update(replacement)
        return projected

    def materialize_arguments(self, tool_id: str, arguments: dict, *, path: str) -> dict:
        """先还原身份再走现行工具合同与授权；不猜错拼 ID，不吞掉未知字段。"""
        result = deepcopy(arguments)

        def resolve(ref, location):
            if not isinstance(ref, str) or ref not in self.by_ref:
                raise FileReferenceAdmissionError(location, tuple(self.by_ref)[:16])
            return self.by_ref[ref]

        if tool_id == "retrieve_files":
            if "file_ids" in result:
                raise FileReferenceAdmissionError(path + "/file_ids", tuple(self.by_ref)[:16])
            if "file_refs" in result:
                refs = result.pop("file_refs")
                if not isinstance(refs, list):
                    raise FileReferenceAdmissionError(path + "/file_refs", tuple(self.by_ref)[:16])
                result["file_ids"] = [resolve(ref, f"{path}/file_refs/{i}")["file_id"] for i, ref in enumerate(refs)]
        for i, row in enumerate(_selectors(tool_id, result)):
            array = _SELECTOR_ARRAYS.get(tool_id)
            location = f"{path}/{array}/{i}" if array else path
            if _IDENTITY_FIELDS & row.keys():
                raise FileReferenceAdmissionError(location, tuple(self.by_ref)[:16])
            if "file_ref" not in row:
                continue
            source = resolve(row.pop("file_ref"), location + "/file_ref")
            row["file_id"] = source["file_id"]
            if tool_id in {"prepare_files", "check_files_state", "list_file_visuals", "read_file_visuals"}:
                row["file_version_id"] = source["file_version_id"]
        return result


def _selectors(tool_id: str, arguments: dict):
    if tool_id == "list_file_visuals":
        yield arguments
    elif tool_id in _SELECTOR_ARRAYS:
        rows = arguments.get(_SELECTOR_ARRAYS[tool_id])
        if isinstance(rows, list):
            yield from (row for row in rows if isinstance(row, dict))


def project_file_tool_schema(entry: dict) -> dict:
    """只派生身份字段，数组上限、互斥条件及其他业务字段仍来自 ToolSpec。"""
    tool_id = entry["tool_id"]
    if tool_id not in _FILE_TOOLS:
        return entry
    projected = deepcopy(entry)
    schema = projected["input_schema"]
    if tool_id == "retrieve_files":
        field = schema["properties"].pop("file_ids")
        field["items"] = deepcopy(_REF_SCHEMA)
        schema["properties"]["file_refs"] = field
    else:
        array = _SELECTOR_ARRAYS.get(tool_id)
        target = schema["properties"][array]["items"] if array else schema
        target["properties"].pop("file_id")
        target["properties"].pop("file_version_id", None)
        target["properties"]["file_ref"] = deepcopy(_REF_SCHEMA)
        _replace_required_fields(target)
    projected["description"] = (
        projected["description"].replace("file_version_id", "文件版本（Host 自动绑定）")
        .replace("file_ids", "file_refs").replace("file_id", "file_ref")
        + " L1 使用当前 file_catalog 的 file_ref（如 f1）；不要填写内部文件 ID 或版本 ID。"
    )
    return projected


def _replace_required_fields(schema: dict) -> None:
    if "required" in schema:
        schema["required"] = list(dict.fromkeys(
            "file_ref" if name in _IDENTITY_FIELDS else name for name in schema["required"]
        ))
    for key in ("oneOf", "anyOf", "allOf"):
        for child in schema.get(key, []):
            _replace_required_fields(child)
    if "not" in schema:
        _replace_required_fields(schema["not"])
