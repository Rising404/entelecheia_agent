"""记录每个步骤的实际内容。

运行时事件流记录步骤发生过；这里记录步骤收到什么、产生什么。二者通过
``model_call_id`` / ``turn_id`` 关联。

轨迹是诊断数据，而非对话：它绝不会重新进入提示，不会展示给用户，也不能作为
解释模型行为原因的证据。
"""

from .contracts import (
    MAX_BLOB_BYTES,
    Blob,
    Part,
    PartRole,
    Step,
    StepKind,
    StepOutcome,
    text_blob,
)
from .exporting import TrajectoryArtifact, export_trajectory
from .presentation import render_trajectory
from .store import (
    TrajectoryReadError,
    TrajectorySchemaError,
    TrajectoryStore,
    active_store,
)
from .scope import TurnLinkage, current_turn_linkage, turn_linkage_scope
from .recorder import (
    record_model_call,
    record_model_request_failure,
    record_rejected_output,
    record_retrieval,
    record_tool_call,
)

__all__ = [
    "MAX_BLOB_BYTES",
    "Blob",
    "Part",
    "PartRole",
    "Step",
    "StepKind",
    "StepOutcome",
    "TrajectoryArtifact",
    "TrajectoryReadError",
    "TrajectorySchemaError",
    "TrajectoryStore",
    "TurnLinkage",
    "active_store",
    "current_turn_linkage",
    "export_trajectory",
    "record_model_call",
    "record_model_request_failure",
    "record_rejected_output",
    "record_retrieval",
    "record_tool_call",
    "render_trajectory",
    "text_blob",
    "turn_linkage_scope",
]
