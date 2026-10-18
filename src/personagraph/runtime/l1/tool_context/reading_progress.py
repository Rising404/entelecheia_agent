"""有界的工具阅读目录：保留范围和续读入口，不复制正文或推断任务完成。"""

from collections.abc import Mapping
from copy import deepcopy
import hashlib
from itertools import islice
import json

from ....persistent_turn_content.evidence import l1_calls_by_ref
from ....persistent_turn_content.findings import EXECUTION_FINDINGS_TOOL_IDS
from ....tools.tool_history.definitions import TOOL_HISTORY_TOOL_IDS
from ..identity import canonical_json
from .projection import ToolResultProjectionError, project_tool_call_arguments

MAX_PROGRESS_RESULTS = 48
MAX_PROGRESS_BYTES = 64_000
_CURSOR_READERS = {"read_text", "read_pdf_text", "read_word", "read_slides"}


def summarize_reading_result(result: object) -> dict:
    """字段只描述实际返回内容；selected_pages 是请求范围，不是已读页。"""
    if not isinstance(result, Mapping):
        return {}
    summary = {
        key: deepcopy(result[key])
        for key in ("outcome", "truncated", "next_cursor", "next_offset")
        if key in result
    }
    if "status" in result:
        summary["result_status"] = result["status"]
    coverage = result.get("coverage")
    if isinstance(coverage, Mapping):
        summary["coverage"] = deepcopy(dict(coverage))
    elements = result.get("elements")
    if isinstance(elements, list):
        summary["returned_element_count"] = len(elements)
        summary["returned_pages"] = sorted(
            {
                item["locator"]["page"]
                for item in elements
                if isinstance(item, dict)
                and isinstance(item.get("locator"), dict)
                and isinstance(item["locator"].get("page"), int)
            }
        )
    for key in ("observations", "evidence", "unavailable_targets"):
        if isinstance(result.get(key), list):
            summary[f"{key}_count"] = len(result[key])
    return summary


def project_reading_progress(
    execution: Mapping[str, object],
    *,
    attachment_catalog=(),
) -> dict:
    """从持久调用重建目录；相同参数合并计数，保留最新回执的固定坐标。"""
    indexed = {}
    for call_ref, call in l1_calls_by_ref(execution).items():
        tool_id = call["tool_id"]
        if tool_id in (*EXECUTION_FINDINGS_TOOL_IDS, *TOOL_HISTORY_TOOL_IDS):
            continue
        if call.get("status") == "pending":
            continue
        raw = call.get("outcome_json")
        if not isinstance(raw, str) or hashlib.sha256(
            raw.encode()
        ).hexdigest() != call.get("outcome_hash"):
            raise ToolResultProjectionError("reading progress outcome hash mismatch")
        outcome = json.loads(raw)
        argument_view = project_tool_call_arguments(call)
        arguments = argument_view["arguments"]
        key = (tool_id, call["arguments_hash"])
        previous = indexed.pop(key, {})
        item = {
            "call_ref": call_ref,
            "tool_id": tool_id,
            "status": call["status"],
            **argument_view,
            "identical_request_count": previous.get("identical_request_count", 0) + 1,
            **summarize_reading_result(outcome.get("result")),
            "read_result": {
                "tool_id": "read_tool_result",
                "arguments": {
                    "call_ref": call_ref,
                    "path": "/result",
                },
            },
        }
        # 内部摘要只用于完整性校验；模型使用固定 call_ref。
        item["arguments_projection"].pop("arguments_sha256", None)
        error = outcome.get("error")
        if isinstance(error, Mapping):
            item["error_code"] = error.get("code")
        cursor = item.get("next_cursor")
        if (
            tool_id in _CURSOR_READERS
            and isinstance(cursor, dict)
            and argument_view["arguments_projection"]["complete"]
        ):
            item["continuation"] = {
                "tool_id": tool_id,
                "arguments": {**arguments, **cursor},
            }
        indexed[key] = item
    attachment_reading = project_attachment_reading(
        indexed.values(), attachment_catalog
    )
    selected = []
    size = len(canonical_json(attachment_reading).encode())
    for item in reversed(tuple(indexed.values())):
        item_size = len(canonical_json(item).encode()) + 1
        if (
            len(selected) == MAX_PROGRESS_RESULTS
            or size + item_size > MAX_PROGRESS_BYTES
        ):
            break
        selected.append(item)
        size += item_size
    return {
        "results": list(reversed(selected)),
        "attachment_reading": attachment_reading,
        "omitted_count": len(indexed) - len(selected),
        "scope": "调用范围与续读目录，不是正文证据；省略项可经 list_tool_results 回查。",
    }


def project_attachment_reading(records, attachment_catalog):
    """Summarize frozen attachment coverage and the furthest continuation per selection.

    Seeing some text on a page does not establish complete page coverage. A pending
    cursor marks only that selection's furthest returned boundary, not all earlier text.
    """
    paths = {item["relative_path"] for item in attachment_catalog}
    documents, selections = {}, {}
    for record in records:
        if record["tool_id"] not in _CURSOR_READERS or record["status"] != "succeeded":
            continue
        arguments = record.get("arguments", {})
        path = arguments.get("path")
        if path not in paths or not record.get("arguments_projection", {}).get(
            "complete"
        ):
            continue
        coverage = record.get("coverage") or {}
        document = documents.setdefault(path, {"pages": set(), "physical_pages": 0})
        document["pages"].update(record.get("returned_pages", []))
        document["physical_pages"] = max(
            document["physical_pages"], coverage.get("physical_pages") or 0
        )
        selection = {
            key: value
            for key, value in arguments.items()
            if key
            not in {"element_offset", "character_offset", "max_elements", "max_chars"}
        }
        key = (
            record["tool_id"],
            json.dumps(selection, ensure_ascii=False, sort_keys=True),
        )
        cursor = record.get("next_cursor")
        if cursor is None and "next_cursor" not in record:
            continue
        # 同一页范围内按元素/字符位置排序；已到末尾的读取不会退回旧前缀。
        position = (
            (1, 0, 0)
            if cursor is None
            else (0, cursor["element_offset"], cursor["character_offset"])
        )
        previous = selections.get(key)
        if previous is None or position > previous[0]:
            continuation = (
                None
                if cursor is None
                else {
                    "source_call_ref": record["call_ref"],
                    "tool_id": record["tool_id"],
                    "arguments": {**arguments, **cursor},
                }
            )
            selections[key] = (position, continuation)
    rows = []
    for path, document in documents.items():
        total, seen = document["physical_pages"], document["pages"]
        row = {"path": path, "pages_with_returned_text": sorted(seen)}
        if total:
            missing = list(
                islice((page for page in range(1, total + 1) if page not in seen), 128)
            )
            row.update(
                physical_pages=total,
                pages_without_returned_text=missing,
                omitted_missing_page_count=total - len(seen) - len(missing),
            )
        rows.append(row)
    pending = [item for _, item in selections.values() if item is not None]
    summary = {
        "documents": rows[-MAX_PROGRESS_RESULTS:],
        "pending_reads": pending[-MAX_PROGRESS_RESULTS:],
        "omitted_count": max(0, len(rows) - MAX_PROGRESS_RESULTS)
        + max(0, len(pending) - MAX_PROGRESS_RESULTS),
        "scope": "Only frozen attachments. Returned text on a page does not prove full-page coverage. Pending cursors are the furthest observed boundaries for their original selections.",
    }
    while (
        len(json.dumps(summary, ensure_ascii=False).encode()) > MAX_PROGRESS_BYTES // 2
    ):
        (summary["pending_reads"] or summary["documents"]).pop(0)
        summary["omitted_count"] += 1
    return summary
