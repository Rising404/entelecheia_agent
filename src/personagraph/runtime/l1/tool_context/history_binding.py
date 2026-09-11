"""把当前 L1 执行的存储读取端口接到通用工具历史接口。

启动时只冻结分区和 owner，不要求 TurnRun 已存在；真正执行工具时才读取既有结果。
模型参数没有 Session/Run 路由权，工具层也不需要依赖 Session 数据库实现。
"""

from ....session import store as session_store
from ....tools.tool_history import ToolHistoryRuntime
from ..corpus_contracts import derive_l1_turn_run_id


def bind_l1_tool_history(*, session_id: str, turn_id: str) -> ToolHistoryRuntime:
    run_id = derive_l1_turn_run_id(session_id=session_id, turn_id=turn_id)
    reader = session_store.build_l1_tool_history_reader(
        session_id=session_id, l1_turn_run_id=run_id,
    )
    return ToolHistoryRuntime(port=reader, effect_scope=f"l1_turn_run:{run_id}")
