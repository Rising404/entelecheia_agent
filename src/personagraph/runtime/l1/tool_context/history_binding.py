"""把当前 L1 执行的存储读取端口接到通用工具历史接口。

启动时只冻结分区和 owner，不要求 TurnRun 已存在；真正执行工具时才读取既有结果。
模型参数没有 Session/Run 路由权，工具层也不需要依赖 Session 数据库实现。
"""

from ....session import store as session_store
from ....tools.tool_history import ToolHistoryRuntime
from ....tools.model_interface import project_tool_result
from ..corpus_contracts import derive_l1_turn_run_id
from ..model_view.file_references import FileReferences, build_file_reference_catalog


def bind_l1_tool_history(*, session_id: str, turn_id: str, attachment_catalog=()) -> ToolHistoryRuntime:
    run_id = derive_l1_turn_run_id(session_id=session_id, turn_id=turn_id)
    reader = session_store.build_l1_tool_history_reader(
        session_id=session_id, l1_turn_run_id=run_id,
    )
    def project_result(tool_id, result):
        execution = session_store.get_l1_turn_execution(session_id=session_id, turn_id=turn_id)
        files = FileReferences(build_file_reference_catalog(attachment_catalog, execution))
        return files.project_result(tool_id, project_tool_result(tool_id, result))

    return ToolHistoryRuntime(
        port=reader, effect_scope=f"l1_turn_run:{run_id}", result_projector=project_result,
    )
