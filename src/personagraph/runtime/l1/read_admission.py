"""L1 阅读动作准入：完整游标，以及冻结来源上的无进展重复请求。"""

from collections.abc import Mapping, Sequence
import json

from ...model_io.output_repair_contracts import RuntimeModelOutputRepairIssue
from ...output_protocol.actions import CallToolsAction
from ...persistent_turn_content.evidence import l1_calls_by_ref
from .identity import canonical_json
from .tool_context.reading_progress import project_reading_progress

_NATIVE_READERS = {"read_text", "read_pdf_text", "read_word", "read_slides"}


def check_read_admission(
    action: CallToolsAction,
    *,
    execution: Mapping[str, object],
    attachment_catalog: Sequence[Mapping[str, object]],
    tool_catalog: Sequence[Mapping[str, object]],
) -> RuntimeModelOutputRepairIssue | None:
    """只约束当前附件版本与不可变历史；不缓存结果，不禁止工作区文件重新读取。

    同参数、同结果允许一次复核；第三次应续读或改变查找方式。不同返回结果、不同
    页码/偏移以及可重试失败不因此拒绝；不可变历史上已确认不存在的路径须先修正。
    授权和实际工具校验仍由工具执行边界负责。
    """
    incomplete = tuple(
        f"/action/calls/{index}/arguments"
        for index, call in enumerate(action.calls)
        if call.tool_id in _NATIVE_READERS
        and call.arguments.get("character_offset")
        and "element_offset" not in call.arguments
    )
    if incomplete:
        return RuntimeModelOutputRepairIssue(
            category="host_guard",
            code="host_guard.l1_incomplete_read_cursor",
            paths=incomplete,
            safe_explanation=(
                "续读游标不完整：character_offset 是某个元素内部的偏移，不是整个文件偏移。"
                "请同时复制原 next_cursor 的 element_offset 和 character_offset；"
                "reading_progress.attachment_reading.pending_reads 给出完整参数；不要只复制一个字段。"
            ),
        )
    paths = {item.get("relative_path") for item in attachment_catalog}
    defaults = {
        item["tool_id"]: {
            name: schema["default"]
            for name, schema in item["input_schema"].get("properties", {}).items()
            if "default" in schema
        }
        for item in tool_catalog
    }
    calls = l1_calls_by_ref(execution)
    repeated = []
    originals = []
    invalid_paths = []
    repeated_sources = set()
    for index, proposed in enumerate(action.calls):
        frozen_read = (
            proposed.tool_id in _NATIVE_READERS
            and proposed.arguments.get("path") in paths
        )
        if not frozen_read and proposed.tool_id != "read_tool_result":
            continue
        request_key = canonical_json(
            defaults.get(proposed.tool_id, {}) | proposed.arguments
        )
        matches = [
            (ref, call)
            for ref, call in calls.items()
            if call.get("tool_id") == proposed.tool_id
            and isinstance(call.get("arguments_json"), str)
            and canonical_json(
                defaults.get(proposed.tool_id, {}) | json.loads(call["arguments_json"])
            )
            == request_key
        ]
        if proposed.tool_id == "read_tool_result" and any(
            call.get("status") == "failed"
            and isinstance(call.get("outcome_json"), str)
            and (json.loads(call["outcome_json"]).get("error") or {}).get("code")
            in {"history_path_unavailable", "invalid_history_path"}
            for _, call in matches
        ):
            invalid_paths.append(f"/action/calls/{index}/arguments/path")
            continue
        matches = [
            (ref, call) for ref, call in matches if call.get("status") == "succeeded"
        ]
        if len(matches) < 2 or matches[-1][1].get("outcome_hash") != matches[-2][1].get(
            "outcome_hash"
        ):
            continue
        repeated.append(f"/action/calls/{index}/arguments")
        originals.append(matches[-1][0])
        if frozen_read:
            repeated_sources.add((proposed.tool_id, proposed.arguments["path"]))
    if invalid_paths:
        return RuntimeModelOutputRepairIssue(
            category="host_guard",
            code="host_guard.l1_invalid_history_path_repeated",
            paths=tuple(invalid_paths),
            safe_explanation=(
                "该 call_ref 的同一路径已返回 history_path_unavailable 或 invalid_history_path；"
                "已保存的结果不可变，原样重试不能产生该字段。请保持正确 call_ref，先读 /result，"
                "再按实际字段或 expand_paths 选择路径，不要假设所有工具都有 /result/text。"
            ),
        )
    if repeated:
        continuation_hint = ""
        if repeated_sources:
            progress = project_reading_progress(
                execution, attachment_catalog=attachment_catalog
            )
            pending = progress["attachment_reading"]["pending_reads"]
            continuation = next(
                (
                    item
                    for item in pending
                    if (item["tool_id"], item["arguments"]["path"]) in repeated_sources
                ),
                None,
            )
            if continuation:
                arguments = continuation["arguments"]
                continuation_hint = (
                    f"可续读 {continuation['source_call_ref']} 的原页范围："
                    f"element_offset={arguments['element_offset']}，"
                    f"character_offset={arguments['character_offset']}。"
                )
        return RuntimeModelOutputRepairIssue(
            category="host_guard",
            code="host_guard.l1_repeated_read_without_progress",
            paths=tuple(repeated),
            safe_explanation=(
                "等价读取已两次返回同一内容。"
                f"原回执（最多8项）：{', '.join(tuple(dict.fromkeys(originals))[:8])}。"
                f"{continuation_hint}"
                "next_cursor 参数见 reading_progress.attachment_reading.pending_reads；"
                "也可换范围或检索词。历史回读按 next_offset / expand_paths 深入，勿原样重试。"
            ),
        )
    return None
