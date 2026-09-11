from __future__ import annotations

from typing import Any

from ...session import store as session_store
from .errors import ApiError
from .views import session_summary

def is_safe_id(value: str) -> bool:
    return bool(value) and all(ch.isalnum() or ch in {"_", "-"} for ch in value)

def require_session(session_id: str) -> dict[str, Any]:
    session = session_store.get_session(session_id)
    if session is None:
        raise ApiError("SESSION_NOT_FOUND", "会话不存在", status=404, details={"session_id": session_id})
    return session_summary(session)


def require_workspace_session(session_id: str) -> None:
    """执行 workspace 范围操作前确保 Session 存在。"""

    if session_store.get_session(session_id) is None:
        raise ApiError("SESSION_NOT_FOUND", "会话不存在", status=404, details={"session_id": session_id})


def required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None or str(value).strip() == "":
        raise ApiError("MISSING_FIELD", "缺少必要字段", details={"field": key})
    return str(value)

def empty_to_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if text == "" else text

def optional_int(value: Any, default: int | None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ApiError("INVALID_INT", "参数必须是整数", details={"value": value}) from exc
