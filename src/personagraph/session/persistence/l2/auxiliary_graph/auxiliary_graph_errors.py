"""AuxiliaryGraph 持久化权威边界共享的关闭式失败错误。

图、依赖、语义验证、重规划和 Host 原语边界都需要此稳定错误谱系，但都不应仅为声明或
捕获错误就导入完整图记录实现。
"""

from __future__ import annotations

from ..work_run.work_execution import WorkExecutionPersistenceError


class AuxiliaryGraphPersistenceError(WorkExecutionPersistenceError):
    """AuxiliaryGraph 权威不变量以关闭方式失败。"""


class AuxiliaryGraphApplyIdCollision(AuxiliaryGraphPersistenceError):
    """AuxiliaryGraph 应用 ID 被复用于不同载荷。"""


__all__ = [
    "AuxiliaryGraphApplyIdCollision",
    "AuxiliaryGraphPersistenceError",
]
