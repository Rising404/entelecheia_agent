"""读取由一个已接受 Entry Turn 持有的活动 Window。

此组件只执行异常侧 Entry 变更或租约敏感延续前所需的失败关闭读取验证；不会推进、
释放或以其他方式更改 Turn Window。
"""

from __future__ import annotations

from typing import Protocol

from ...turn.contracts import AcceptedEntryTurn


class EntryActiveWindowReadPort(Protocol):
    """针对一个 Session 的一次权威 execution-window 读取。"""

    def inspect_turn_execution(self, session_id: str) -> dict[str, object]: ...


def load_authoritative_active_turn_window(
    *,
    accepted: AcceptedEntryTurn,
    store: EntryActiveWindowReadPort,
    expected_lease_owner: str | None = None,
) -> dict[str, object] | None:
    """在任何异常侧变更前重新加载所属活动 Turn 游标。"""

    try:
        inspected = store.inspect_turn_execution(accepted.session_id)
        window = inspected.get("window")
        if (
            not isinstance(window, dict)
            or str(window.get("turn_id") or "") != accepted.turn_id
            or str(window.get("window_state") or "") != "active"
            or (
                expected_lease_owner is not None
                and str(window.get("lease_owner") or "")
                != expected_lease_owner
            )
        ):
            return None
        revision = window.get("state_version")
        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 1
        ):
            return None
        return window
    except Exception:
        return None


__all__ = [
    "EntryActiveWindowReadPort",
    "load_authoritative_active_turn_window",
]
