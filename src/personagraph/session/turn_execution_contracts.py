"""持久 Turn 执行存储的小型跨层契约。

Runtime 编排可以依赖这些异常事实，而无需导入 SQLite 持久化模块。存储实现仍是事务和
schema 的唯一所有者。
"""

from __future__ import annotations


class TurnExecutionPersistenceError(RuntimeError):
    """持久 Turn 执行不变量遭到违反。"""


class TurnExecutionBusyError(TurnExecutionPersistenceError):
    """Session 已经拥有尚未释放的 Window。"""

    def __init__(self, window: dict[str, object]) -> None:
        self.window = window
        self.details = {
            "session_id": window["session_id"],
            "turn_id": window["turn_id"],
            "window_state": window["window_state"],
            "window_revision": window["state_version"],
        }
        super().__init__("TURN_IN_PROGRESS")


class TurnExecutionWindowRevisionConflict(TurnExecutionPersistenceError):
    """过期调用方尝试在另一项转换后修改 Window。"""

    def __init__(self, *, expected: int, actual: int) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"turn execution window revision conflict: expected {expected}, actual {actual}"
        )


class TurnExecutionLeaseConflict(TurnExecutionPersistenceError):
    """调用方观察到的 Window 租约在变更前发生了变化。"""

    def __init__(self, *, expected_heartbeat_at: str | None, actual_heartbeat_at: str | None) -> None:
        self.expected_heartbeat_at = expected_heartbeat_at
        self.actual_heartbeat_at = actual_heartbeat_at
        super().__init__("turn execution window lease changed during mutation")
