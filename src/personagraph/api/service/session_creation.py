"""创建请求的幂等准入与错误投影；目录物化仍由 Session API 创建入口负责。"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from ...session.catalog import (
    CreationRequestConflict,
    CreationRequestInProgress,
    SessionCatalog,
    validate_session_id,
)
from .errors import ApiError


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionCreationRequest:
    request_id: str | None = None
    session_id: str | None = None


@contextmanager
def accept_creation_request(payload: dict[str, Any]) -> Iterator[SessionCreationRequest]:
    if "client_request_id" not in payload:
        yield SessionCreationRequest()
        return
    try:
        request_id = validate_session_id(payload["client_request_id"])
    except ValueError as exc:
        raise ApiError("INVALID_CLIENT_REQUEST_ID", "创建请求 ID 格式无效") from exc
    # 摘要绑定调用者提交的意图，而不是可变的默认根或后来人工修改的标题。
    parameters = {key: payload.get(key) or None for key in ("title", "folder_id", "working_dir")}
    request_hash = hashlib.sha256(
        json.dumps(parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    catalog = SessionCatalog()
    try:
        session_id = catalog.reserve_creation_request(request_id, request_hash)
    except CreationRequestConflict as exc:
        raise ApiError(
            "SESSION_CREATION_REQUEST_CONFLICT", "该创建请求 ID 已用于其他参数", status=409,
        ) from exc
    except CreationRequestInProgress as exc:
        raise ApiError(
            "SESSION_CREATION_IN_PROGRESS", "会话创建结果尚未确认，请稍后重试同一请求", status=409,
        ) from exc
    try:
        yield SessionCreationRequest(request_id, session_id)
    except Exception as exc:
        if session_id is None:
            try:
                released = catalog.release_unpublished_creation(request_id)
                if released and isinstance(exc, ApiError):
                    exc.details["creation_not_committed"] = True
            except Exception:
                LOGGER.exception("could not release unpublished Session creation request")
        raise
