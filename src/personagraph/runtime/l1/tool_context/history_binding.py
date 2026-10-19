"""把当前 L1 执行的存储读取端口接到通用工具历史接口。

启动时只冻结分区和 owner，不要求 TurnRun 已存在；真正执行工具时才读取既有结果。
模型参数没有 Session/Run 路由权，工具层也不需要依赖 Session 数据库实现。
"""

from dataclasses import dataclass
from typing import Callable

from ....session import store as session_store
from ....tools.tool_history import ToolHistoryRuntime
from ....tools.model_interface import project_tool_result
from ....persistent_turn_content.evidence import l1_calls_by_ref, project_findings_arguments
from ....persistent_turn_content.tool_results import (
    MAX_ARGUMENT_SUMMARY_CHARACTERS, ToolHistoryReadPort, ToolHistoryError,
)
from ....output_protocol.l1_persistence import normalize_l1_finding_arguments
from ..corpus_contracts import derive_l1_turn_run_id
from ..identity import canonical_json
from ..model_view.file_references import FileReferences, build_file_reference_catalog
from ..model_view.tool_references import ToolReferences, build_tool_reference_catalog
from ....persistent_turn_content.findings import EXECUTION_FINDINGS_TOOL_IDS, ExecutionFindingsOwnerKind
from .projection import project_tool_call_arguments


@dataclass(frozen=True)
class _ModelHistoryReader:
    """存储端口仍返回原始身份；这里只改目录摘要，在截断前投影完整参数。"""

    native: ToolHistoryReadPort
    load_view: Callable[..., tuple[FileReferences, ToolReferences, dict]]

    @property
    def authority_sha256(self):
        return self.native.authority_sha256

    def read_result(self, **arguments):
        return self.native.read_result(**arguments)

    def list_results(self, **arguments):
        page = self.native.list_results(**arguments)
        files, references, execution = self.load_view(
            include_entries=any(item.source.tool_id in EXECUTION_FINDINGS_TOOL_IDS for item in page.results),
        )
        calls = l1_calls_by_ref(execution)
        call_refs = {call["tool_call_id"]: ref for ref, call in calls.items()}
        items = []
        for item in page.results:
            if item.arguments_unavailable:
                items.append(item)
                continue
            source = item.source
            call = calls.get(source.call_ref)
            identity = None if call is None else (
                call["tool_call_id"], call["tool_id"], call["outcome_hash"], call["arguments_hash"],
            )
            if identity != (source.tool_call_id, source.tool_id, source.result_sha256, item.arguments_sha256):
                raise ToolHistoryError("tool_result_integrity_invalid")
            argument_view = project_tool_call_arguments(call)
            projected = argument_view["arguments"]
            if source.tool_id in EXECUTION_FINDINGS_TOOL_IDS:
                projected = project_findings_arguments(
                    normalize_l1_finding_arguments(projected, tool_calls=execution["tool_calls"]), call_refs=call_refs,
                )
            projected = files.project_arguments(source.tool_id, references.project_arguments(source.tool_id, projected))
            summary = canonical_json(projected)
            items.append(item.model_copy(update={
                "arguments_summary": summary[:MAX_ARGUMENT_SUMMARY_CHARACTERS],
                "arguments_truncated": (
                    len(summary) > MAX_ARGUMENT_SUMMARY_CHARACTERS
                    or not argument_view["arguments_projection"]["complete"]
                ),
            }))
        return page.model_copy(update={"results": tuple(items)})


def bind_l1_tool_history(*, session_id: str, turn_id: str, attachment_catalog=()) -> ToolHistoryRuntime:
    run_id = derive_l1_turn_run_id(session_id=session_id, turn_id=turn_id)
    reader = session_store.build_l1_tool_history_reader(
        session_id=session_id, l1_turn_run_id=run_id,
    )
    def load_view(*, include_entries):
        execution = session_store.get_l1_turn_execution(session_id=session_id, turn_id=turn_id)
        files = FileReferences(build_file_reference_catalog(attachment_catalog, execution))
        revisions = ()
        if include_entries:
            snapshot = session_store.get_execution_findings_ledger_for_owner(
                owner_kind=ExecutionFindingsOwnerKind.L1_TURN_RUN, execution_owner_id=run_id,
            )
            if snapshot is None:
                raise ValueError("L1 findings ledger is unavailable for history projection")
            revisions = snapshot.ledger.entry_revisions
        references = ToolReferences(build_tool_reference_catalog(execution, entry_revisions=revisions))
        return files, references, execution

    def project_result(tool_id, result):
        files, references, _ = load_view(include_entries=tool_id in EXECUTION_FINDINGS_TOOL_IDS)
        return files.project_result(tool_id, references.project_result(tool_id, project_tool_result(tool_id, result)))

    return ToolHistoryRuntime(
        port=_ModelHistoryReader(reader, load_view), effect_scope=f"l1_turn_run:{run_id}", result_projector=project_result,
    )
