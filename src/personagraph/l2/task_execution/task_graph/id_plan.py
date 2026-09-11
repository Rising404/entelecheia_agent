"""供 TaskGraph WorkRun 控制器使用的纯稳定 ID 推导。

这些函数将已有权威的 TaskNode 与 Turn 标识转换为稳定持久 ID。它们不访问 Session 状态、
不选择工作、不调用提供商，也不推进 WorkRun。
"""

from __future__ import annotations

import hashlib

from personagraph.l2.work_run import TaskNodeSubject
from personagraph.l2.task_execution.work_run.stable_ids import WorkRunTurnStableIdPlan


def derive_task_graph_work_run_stable_ids(
    *,
    session_id: str,
    turn_id: str,
    subject: TaskNodeSubject,
) -> WorkRunTurnStableIdPlan:
    """推导一个新 TaskGraph WorkRun 的确定性命名空间。"""

    identity = "\0".join(
        (
            session_id,
            turn_id,
            subject.task_id,
            str(subject.graph_revision),
            subject.node_id,
            str(subject.node_revision),
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:40]
    return WorkRunTurnStableIdPlan(namespace=f"task-graph-v1-{digest}")


def recover_task_graph_work_run_stable_ids(
    work_run_id: str,
) -> WorkRunTurnStableIdPlan:
    """从持久 TaskGraph WorkRun ID 恢复原始命名空间。"""

    suffix = ":workrun"
    if not work_run_id.endswith(suffix):
        raise ValueError("recoverable WorkRun ID has no stable controller namespace")
    return WorkRunTurnStableIdPlan(namespace=work_run_id[: -len(suffix)])


def derive_task_graph_safe_lane_detach_apply_id(
    *,
    turn_id: str,
    work_run_id: str,
) -> str:
    """推导一次 TaskGraph 安全泳道分离的幂等收据。"""

    digest = hashlib.sha256(f"{turn_id}\0{work_run_id}".encode("utf-8")).hexdigest()
    return f"task-graph-detach-v1:{digest}"


__all__ = [
    "derive_task_graph_safe_lane_detach_apply_id",
    "derive_task_graph_work_run_stable_ids",
    "recover_task_graph_work_run_stable_ids",
]
