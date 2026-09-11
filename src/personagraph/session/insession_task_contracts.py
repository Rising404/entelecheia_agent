"""持久 ``insession_task`` 存储的小型跨层错误契约。"""

from __future__ import annotations


class InSessionTaskPersistenceError(RuntimeError):
    """无法按请求安全读取或提交任务图。"""


class InSessionTaskApplyIdCollision(InSessionTaskPersistenceError):
    """同一幂等键被复用于语义不同的提交。"""


class InSessionTaskGraphRevisionConflict(InSessionTaskPersistenceError):
    """调用方尝试修订不再是当前版本的图。"""

    def __init__(self, *, expected: int | None, actual: int | None) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"insession task graph revision conflict: expected {expected}, actual {actual}"
        )


class InSessionTaskStateVersionConflict(InSessionTaskPersistenceError):
    """调用方尝试从过期 Task 状态版本修改 Task。"""

    def __init__(self, *, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"insession task state version conflict: expected {expected}, actual {actual}"
        )


class InSessionTaskGraphRevisionNotSupported(InSessionTaskPersistenceError):
    """请求的图修订模式尚无安全持久化协议。"""

    def __init__(self) -> None:
        super().__init__(
            "insession task graph revisions above revision 1 are not supported"
        )
