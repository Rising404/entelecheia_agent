"""用 ContextVar 传递 accepted Turn 与 trajectory 的请求内归属（turn linkage）。

归属 scope 与 Session 数据库 scope 是两个独立边界：前者补齐记录 ID，后者决定
Store 写到哪里。线程池须 copy_context；新后台工作须显式绑定其原始 Turn。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True, slots=True)
class TurnLinkage:
    session_id: str
    turn_id: str


_CURRENT_TURN_LINKAGE: ContextVar[TurnLinkage | None] = ContextVar(
    "personagraph_trajectory_turn_linkage",
    default=None,
)


@contextmanager
def turn_linkage_scope(*, session_id: str, turn_id: str) -> Iterator[TurnLinkage]:
    """在当前执行上下文绑定一个 accepted Turn，退出时恢复外层绑定。

    token/reset 让嵌套调用和异常退出都不泄漏归属；它不创建 Turn、不选择数据库，
    也不会自动把 ContextVar 复制进新工作线程。
    """

    linkage = TurnLinkage(
        session_id=_require_identifier(session_id, field="session_id"),
        turn_id=_require_identifier(turn_id, field="turn_id"),
    )
    token = _CURRENT_TURN_LINKAGE.set(linkage)
    try:
        yield linkage
    finally:
        _CURRENT_TURN_LINKAGE.reset(token)


def current_turn_linkage() -> TurnLinkage | None:
    """只读返回当前上下文的 Turn 归属；未绑定返回 None，不推测最近一个 Turn。"""

    return _CURRENT_TURN_LINKAGE.get()


def resolve_turn_linkage(
    *,
    session_id: str | None,
    turn_id: str | None,
) -> tuple[str | None, str | None]:
    """从当前 scope 补齐缺失 ID，并拒绝显式 ID 与活动 Turn 冲突。

    没有 scope 时保留调用方给出的值，是否有可写 Session 数据库仍由 Store 检查。
    不能用另一 Turn 的显式参数覆盖活动 scope，否则同一调用的轨迹会串到别轮。
    """

    scoped = current_turn_linkage()
    if scoped is None:
        return session_id, turn_id
    if session_id is not None and session_id != scoped.session_id:
        raise ValueError(
            "explicit session_id conflicts with the active Turn scope"
        )
    if turn_id is not None and turn_id != scoped.turn_id:
        raise ValueError(
            "explicit turn_id conflicts with the active Turn scope"
        )
    return (
        session_id if session_id is not None else scoped.session_id,
        turn_id if turn_id is not None else scoped.turn_id,
    )


def _require_identifier(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"trajectory {field} must be a non-empty string")
    return value


__all__ = [
    "TurnLinkage",
    "current_turn_linkage",
    "resolve_turn_linkage",
    "turn_linkage_scope",
]
