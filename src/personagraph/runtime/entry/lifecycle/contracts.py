"""Entry 生命周期回调的轻量契约。"""

from __future__ import annotations

from collections.abc import Callable

from ...turn.contracts import AcceptedEntryTurn


EntryAcceptedEmitter = Callable[[AcceptedEntryTurn], None]


__all__ = ["EntryAcceptedEmitter"]
