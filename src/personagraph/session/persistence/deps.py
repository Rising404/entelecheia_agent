"""Session 存储实现模块共享的窄持久化依赖。"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class StoreDeps:
    init_db: Callable[[], None]
    connect: Callable[[], sqlite3.Connection]
    now: Callable[[], str]
    new_id: Callable[[], str]
