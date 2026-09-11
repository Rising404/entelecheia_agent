"""提交后失败的只读展示与显式恢复命令；不运行索引或摘要处理器。"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping

from ...runtime.post_commit.scheduler import schedule_turn_post_commit_jobs
from ...session import store as session_store
from ...session.turn_execution_contracts import TurnExecutionPersistenceError
from .common import require_session
from .errors import ApiError


_LOG = logging.getLogger(__name__)
_FIELDS = frozenset({
    "turn_id", "request_id", "action", "expected_window_revision",
    "expected_failed_job_digest", "job_ids", "confirm_stale",
})


def post_commit_control_view(inspection: Mapping[str, object]) -> dict | None:
    """只投影当前失败集合及其 Store 摘要，GET 不认领、重试或释放窗口。"""
    window, jobs = inspection.get("window"), inspection.get("post_commit_jobs")
    if not isinstance(window, Mapping) or window.get("window_state") != "post_commit_pending":
        return None
    failed = [
        {key: job.get(key) for key in ("job_id", "job_kind", "reason_code")}
        for job in jobs if isinstance(job, Mapping) and job.get("status") == "terminal_failed"
    ] if isinstance(jobs, list) else []
    return {
        "turn_id": window.get("turn_id"),
        "window_revision": window.get("state_version"),
        "failed_job_digest": inspection.get("failed_job_digest"),
        "failed_jobs": failed,
    }


def control_turn_post_commit_jobs(session_id: str, payload: dict) -> dict:
    """认证后的显式 retry/waive；精确集合、窗口和幂等性仍由 Store 事务裁定。

    HTTP 身份认证由统一 server 边界完成，Session 数据库范围由 router 绑定。
    actor 由服务端指定，不能从 JSON 自报；跳过必须另外确认上下文可能过时。
    """
    command = _validated_command(payload)
    session = require_session(session_id)
    if session.get("status") != "active":
        raise ApiError("SESSION_READ_ONLY", "请先恢复该会话再处理失败任务", status=409)
    try:
        result = session_store.apply_turn_post_commit_job_control(
            session_id=session_id, actor="authenticated_local_user", **command,
        )
    except (TurnExecutionPersistenceError, ValueError) as exc:
        raise ApiError(
            "POST_COMMIT_CONTROL_CONFLICT",
            "失败任务或会话状态已经变化，请刷新后重新确认",
            status=409,
        ) from exc
    window = result.get("window")
    # 旧回执的 exact replay 不能顺带唤醒另一轮工作；只调度仍由本命令 Turn 持有的待办。
    if (isinstance(window, Mapping) and window.get("turn_id") == command["turn_id"]
            and window.get("window_state") == "post_commit_pending"):
        # 命令已持久化后调度失败不能伪装写入失败；宿主恢复器会再次发现待办。
        try:
            schedule_turn_post_commit_jobs(session_id=session_id, store=session_store)
        except Exception:
            _LOG.exception("could not wake post-commit worker after explicit control")
    return {"replayed": bool(result["replayed"])}


def _validated_command(payload: dict) -> dict:
    if not isinstance(payload, dict) or set(payload) - _FIELDS:
        raise ApiError("INVALID_POST_COMMIT_CONTROL", "恢复请求包含未知字段", status=400)
    command = {}
    for key in ("turn_id", "request_id"):
        value = payload.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", value):
            raise ApiError("INVALID_POST_COMMIT_CONTROL", f"{key} 格式不正确", status=400)
        command[key] = value
    revision = payload.get("expected_window_revision")
    digest = payload.get("expected_failed_job_digest")
    jobs = payload.get("job_ids")
    if type(revision) is not int or revision < 1:
        raise ApiError("INVALID_POST_COMMIT_CONTROL", "窗口版本必须是正整数", status=400)
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ApiError("INVALID_POST_COMMIT_CONTROL", "失败集合标识格式不正确", status=400)
    if (
        not isinstance(jobs, list) or not 1 <= len(jobs) <= 16
        or any(not isinstance(job, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", job) for job in jobs)
        or len(set(jobs)) != len(jobs)
    ):
        raise ApiError("INVALID_POST_COMMIT_CONTROL", "失败任务列表格式不正确", status=400)
    action = payload.get("action")
    if not isinstance(action, str) or action not in {"retry", "waive"}:
        raise ApiError("INVALID_POST_COMMIT_CONTROL", "只允许重试或明确跳过", status=400)
    if "confirm_stale" in payload and type(payload["confirm_stale"]) is not bool:
        raise ApiError("INVALID_POST_COMMIT_CONTROL", "跳过确认必须为布尔值", status=400)
    if action == "waive" and payload.get("confirm_stale") is not True:
        raise ApiError("POST_COMMIT_CONFIRMATION_REQUIRED", "跳过前需确认后续上下文可能不完整", status=400)
    command.update(
        action=action, expected_window_revision=revision,
        expected_failed_job_digest=digest, job_ids=tuple(jobs),
    )
    return command


__all__ = ["control_turn_post_commit_jobs", "post_commit_control_view"]
