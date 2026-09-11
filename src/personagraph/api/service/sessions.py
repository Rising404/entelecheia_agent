from __future__ import annotations

import logging
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

from ...configuration.features import load_features
from ...context_budget import ContextBudgetExceeded
from ...identity import DEFAULT_AGENT_ID
from personagraph.model_io.gateway import ModelGatewayError
from ...runtime.concurrency import SessionRunBusyError
from ..runtime_outcome import build_error_outcome
from ...workspace.files.attachments import MAX_ATTACHMENTS_PER_TURN
from ...runtime.entry import run_entry_turn
from ...runtime.turn.contracts import AcceptedEntryTurn
from ...runtime.entry.routing.policy import (
    ResolvedTurnRoutingPolicy,
    TurnRoutingPolicyError,
    available_processing_levels,
    parse_turn_routing_policy_snapshot,
    resolve_turn_routing_policy,
)
from ...runtime.post_commit.scheduler import schedule_turn_post_commit_jobs
from ...runtime.entry.lifecycle.window_audit import TurnWindowBlockedError
from ...session import store as session_store
from ...session.insession_task_contracts import InSessionTaskPersistenceError
from .common import empty_to_none, optional_int, require_session, required_str
from .errors import ApiError
from .post_commit import post_commit_control_view
from .views import (
    insession_task_detail_view,
    model_api_error,
    pending_user_question_view,
    session_summary,
    turn_result_view,
    turn_view,
)


_LOG = logging.getLogger(__name__)

_DEFAULT_SESSION_NAME = "新会话"
_DEFAULT_PROJECT_NAME_LIMIT = 48
_UNSAFE_PROJECT_NAME_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _current_local_date() -> date:
    return datetime.now().astimezone().date()


def _safe_default_project_name(title: str | None) -> str:
    """将不可信会话标题收窄为一个可移植且可通过 workspace guard 的目录段。"""

    from ...configuration.paths import deny_reason

    normalized = unicodedata.normalize("NFKC", str(title or ""))
    normalized = " ".join(normalized.split())
    normalized = _UNSAFE_PROJECT_NAME_CHARACTERS.sub("_", normalized)
    normalized = normalized.strip(" ._")[:_DEFAULT_PROJECT_NAME_LIMIT]
    if not normalized:
        normalized = _DEFAULT_SESSION_NAME
    # 诸如 ``token``、``.env`` 的标题会被统一 workspace guard 拒绝；目录仍然必须
    # 可用，因此仅在这类少见标题下退回中性名称，而不是绕过安全规则。
    probe = Path(f"2000-01-01_{normalized}")
    if deny_reason(probe) is not None:
        return _DEFAULT_SESSION_NAME
    return normalized


def _default_project_root() -> Path:
    """解析模型可见的默认 Project 根，并再次守住源码与敏感目录边界。"""

    from ...configuration import paths
    from ...configuration.app_settings import default_projects_directory
    from ...configuration.paths import deny_reason

    configured = Path(default_projects_directory()).expanduser()
    if not configured.is_absolute():
        raise ApiError(
            "DEFAULT_PROJECT_ROOT_INVALID",
            "默认工作目录配置无效",
            status=409,
            details={"reason": "path_not_absolute"},
        )
    try:
        root = configured.resolve(strict=False)
        source_root = Path(paths.PROJECT_ROOT).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ApiError(
            "DEFAULT_PROJECT_ROOT_INVALID",
            "默认工作目录配置无效",
            status=409,
            details={"reason": "path_resolution_failed"},
        ) from exc
    if root == source_root or root.is_relative_to(source_root):
        raise ApiError(
            "DEFAULT_PROJECT_ROOT_INVALID",
            "默认工作目录不能位于源码目录中",
            status=409,
            details={"reason": "inside_source_checkout"},
        )
    reason = deny_reason(root)
    if reason is not None:
        raise ApiError(
            "DEFAULT_PROJECT_ROOT_INVALID",
            "默认工作目录被安全策略拒绝",
            status=409,
            details={"reason": reason},
        )
    return root


def _create_default_project_directory(title: str | None) -> Path:
    """原子创建 ``日期_会话名`` 目录；同名并发创建使用稳定数字后缀。"""

    from ...configuration.paths import deny_reason

    root = _default_project_root()
    try:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise ApiError(
            "DEFAULT_PROJECT_CREATE_FAILED",
            "无法创建默认工作目录",
            status=409,
            details={"reason": "root_create_failed"},
        ) from exc
    if not root.is_dir():
        raise ApiError(
            "DEFAULT_PROJECT_CREATE_FAILED",
            "无法创建默认工作目录",
            status=409,
            details={"reason": "root_not_directory"},
        )

    base_name = f"{_current_local_date().isoformat()}_{_safe_default_project_name(title)}"
    for ordinal in range(1, 10_001):
        name = base_name if ordinal == 1 else f"{base_name}-{ordinal}"
        candidate = root / name
        reason = deny_reason(candidate)
        if reason is not None:
            raise ApiError(
                "DEFAULT_PROJECT_CREATE_FAILED",
                "默认工作目录名称被安全策略拒绝",
                status=409,
                details={"reason": reason},
            )
        try:
            candidate.mkdir(mode=0o700, exist_ok=False)
        except FileExistsError:
            continue
        except OSError as exc:
            raise ApiError(
                "DEFAULT_PROJECT_CREATE_FAILED",
                "无法创建默认工作目录",
                status=409,
                details={"reason": "directory_create_failed"},
            ) from exc
        return candidate.resolve()
    raise ApiError(
        "DEFAULT_PROJECT_CREATE_FAILED",
        "无法分配唯一的默认工作目录",
        status=409,
        details={"reason": "name_space_exhausted"},
    )


def list_sessions(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    status = str(params.get("status") or "active")
    folder_id = empty_to_none(params.get("folder_id"))
    query = empty_to_none(params.get("query"))
    limit = optional_int(params.get("limit"), default=50)
    if query:
        rows = session_store.search_sessions(
            query,
            status=status,
            folder_id=folder_id,
            limit=limit,
        )
    else:
        rows = session_store.list_sessions(
            status=status,
            folder_id=folder_id,
            limit=limit,
        )
    return {"sessions": [session_summary(row) for row in rows]}


def _raise_workspace_layout_error(exc: Exception) -> None:
    from ...workspace.binding import WorkspaceLayoutError

    details: dict[str, str] = {}
    if isinstance(exc, WorkspaceLayoutError):
        details["reason"] = exc.code
    raise ApiError(
        "WORKSPACE_LAYOUT_REJECTED",
        "该目录无法安全建立或识别 agent 私有工作区",
        status=409,
        details=details,
    ) from exc


def _remove_empty_default_project(path: Path | None) -> None:
    """仅回收本次创建且仍为空的目录；绝不递归删除用户或运行时内容。"""

    if path is None:
        return
    try:
        path.rmdir()
    except OSError:
        _LOG.warning(
            "default project directory could not be removed after session creation failed",
            extra={"path": str(path)},
        )


def create_session(payload: dict[str, Any]) -> dict[str, Any]:
    from .session_creation import accept_creation_request

    with accept_creation_request(payload) as request:
        if request.session_id is not None:
            session = session_store.get_session(request.session_id)
            if session is None:
                raise ApiError(
                    "SESSION_CREATION_RESULT_UNAVAILABLE", "该请求对应的会话已不存在，不能重复创建", status=410,
                )
            return {"session": session_summary(session)}
        return _create_session(payload, creation_request_id=request.request_id)


def _create_session(
    payload: dict[str, Any], *, creation_request_id: str | None,
) -> dict[str, Any]:
    # ``persona_id`` 仍是持久兼容列，但客户端不再选择角色卡。每个新 Session 都绑定到产品
    # 身份，且该值绝不会用于 prompt 构建。
    persona_id = DEFAULT_AGENT_ID
    title = empty_to_none(payload.get("title"))
    folder_id = empty_to_none(payload.get("folder_id"))
    store_creation_started = False
    raw_directory = payload.get("working_dir")
    default_project: Path | None = None
    if raw_directory is None or raw_directory == "":
        default_project = _create_default_project_directory(title)
        working_directory = default_project
    else:
        from ...configuration.workspace import validate_workspace_directory

        try:
            working_directory = validate_workspace_directory(raw_directory, must_exist=True)
        except ValueError as exc:
            raise ApiError("INVALID_WORKING_DIR", str(exc), details={"field": "working_dir"}) from exc
    try:
        # Project 绑定是元数据，并非创建隐藏逐 Session 输入/输出 workspace 的请求。上传和
        # 生成文件位于绑定的 Project 树中，并登记到其 documents.sqlite。
        store_creation_started = True
        session_id = session_store.create_session(
            persona_id,
            title=title,
            folder_id=folder_id,
            working_dir=str(working_directory),
            # 产品创建只有在读取 authority 同步落盘后才能发布。低层 Store 的默认仍保持
            # 宽松，供迁移夹具和内部调用方使用。
            require_workspace_read_authority=True,
            **({"creation_request_id": creation_request_id} if creation_request_id is not None else {}),
        )
    except Exception as exc:
        # Store 明确证明尚未取得分区/Project 所有权时，默认目录仍完全归本次失败
        # 操作所有，可安全做一次非递归空目录回收。进入 Project remember 后则保留
        # 目录：locator 可能已提交，事后“查无引用再删”会引入并发 publish 的 TOCTOU。
        if not store_creation_started or isinstance(
            exc,
            session_store.SessionCreationNotOwnedError,
        ):
            _remove_empty_default_project(default_project)
        if isinstance(exc, (OSError, ValueError)):
            _raise_workspace_layout_error(exc)
        raise
    return {"session": require_session(session_id)}


def get_session(session_id: str) -> dict[str, Any]:
    session = require_session(session_id)
    turns = session_store.get_turns(session_id)
    execution = session_store.inspect_turn_execution(session_id)
    runtime_routing = _session_runtime_routing_view(
        session_id=session_id,
    )
    try:
        pending_questions = session_store.list_pending_user_questions(
            session_id=session_id,
        )
        pending_question_views = [
            pending_user_question_view(item) for item in pending_questions
        ]
    except (InSessionTaskPersistenceError, ValueError) as exc:
        # 返回空列表会使持久问题在损坏或 authority join 断裂后从弹窗消失。应保留上一份
        # 客户端投影，并让本次刷新保守失败。
        raise ApiError(
            "PENDING_USER_QUESTIONS_UNAVAILABLE",
            "待回答问题暂时无法读取；任务状态未被修改",
            status=409,
            details={"session_id": session_id},
        ) from exc
    return {
        "session": session,
        "turns": [turn_view(turn) for turn in turns],
        "pending_user_questions": pending_question_views,
        "turn_window": turn_execution_window_view(
            execution.get("window") if isinstance(execution, dict) else None,
            post_commit_jobs=(
                execution.get("post_commit_jobs")
                if isinstance(execution, dict)
                else None
            ),
        ),
        "runtime_routing": runtime_routing,
        "post_commit": post_commit_control_view(execution),
    }


def _session_runtime_routing_view(
    *,
    session_id: str,
) -> dict[str, Any]:
    """投影有效 Session 默认值，但不公开 Store 记录。"""

    try:
        stored = session_store.get_session_turn_routing_policy(session_id)
        resolved = resolve_turn_routing_policy(
            request_policy=None,
            request_policy_provided=False,
            stored_session_policy_json=(
                str(stored["policy_json"]) if stored is not None else None
            ),
        )
    except (TurnRoutingPolicyError, ValueError) as exc:
        raise ApiError(
            "STORED_RUNTIME_POLICY_INVALID",
            "会话运行模式无法读取；会话状态未被修改",
            status=409,
            details={"session_id": session_id},
        ) from exc

    snapshot = resolved.snapshot
    return {
        "schema_version": snapshot.schema_version,
        "source": snapshot.source,
        "policy": snapshot.policy.to_dict(),
        "allowed_processing_levels": list(snapshot.allowed_processing_levels),
        "available_processing_levels": list(available_processing_levels()),
    }


def get_insession_task_details(
    session_id: str,
    insession_task_id: str,
) -> dict[str, Any]:
    """返回一个 Session 范围的只读任务卡片投影。

    ``insession_task`` 被刻意设为不同于旧长期任务 API。不能使用任务 ID 发现其他 Session
    的任务：Store 读取和此公开端点都绑定到 ``session_id``。
    """

    require_session(session_id)
    try:
        from ...session.l2_store import task_graph as task_graph_store

        details = task_graph_store.get_insession_task_details(
            session_id,
            insession_task_id,
        )
        if details is None:
            raise ApiError(
                "INSESSION_TASK_NOT_FOUND",
                "会话内任务不存在或不属于当前会话",
                status=404,
                details={"insession_task_id": insession_task_id},
            )
        return {"task": insession_task_detail_view(details)}
    except InSessionTaskPersistenceError as exc:
        # 损坏的历史源 manifest 不能投影为看似合理的任务卡片。不要公开原始
        # SQLite/manifest 诊断。
        raise ApiError(
            "INSESSION_TASK_DETAILS_UNAVAILABLE",
            "会话内任务详情暂不可用；未修改任务状态",
            status=409,
            details={"insession_task_id": insession_task_id},
        ) from exc
    except ValueError as exc:
        # 将格式错误的 Store 投影视同不可读 manifest：此端点只读，必须保守失败而非编造部分
        # UI 表示。
        raise ApiError(
            "INSESSION_TASK_DETAILS_UNAVAILABLE",
            "会话内任务详情暂不可用；未修改任务状态",
            status=409,
            details={"insession_task_id": insession_task_id},
        ) from exc


def turn_execution_window_view(
    window: dict[str, object] | None,
    *,
    post_commit_jobs: object = None,
) -> dict[str, Any] | None:
    """只公开聊天 UI 所需的当前执行槽位事实。"""

    if not isinstance(window, dict):
        return None
    state = str(window.get("window_state") or "")
    if state not in {"empty", "active", "post_commit_pending", "interrupted"}:
        return None
    revision = window.get("state_version")
    if not isinstance(revision, int):
        return None
    view: dict[str, Any] = {
        "turn_id": str(window["turn_id"]) if window.get("turn_id") else None,
        "window_state": state,
        "window_revision": revision,
        "stage": str(window["stage"]) if window.get("stage") else None,
        "interruption_reason": (
            str(window["interruption_reason"])
            if window.get("interruption_reason")
            else None
        ),
    }
    if state == "post_commit_pending":
        failed_codes = _post_commit_failure_codes(post_commit_jobs)
        view["post_commit_status"] = "failed" if failed_codes else "pending"
        view["post_commit_error_codes"] = failed_codes
    return view


def _post_commit_failure_codes(post_commit_jobs: object) -> list[str]:
    """投影终止派生状态失败，但不公开 job 内部细节。"""

    if not isinstance(post_commit_jobs, list):
        return []
    codes = {
        str(job.get("reason_code"))
        for job in post_commit_jobs
        if isinstance(job, dict)
        and job.get("status") == "terminal_failed"
        and isinstance(job.get("reason_code"), str)
        and str(job["reason_code"]).strip()
    }
    return sorted(codes)


def get_runtime_events(session_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """返回新 Runtime 有界且安全的活动尾部或追赶页。

    供 SSE 断连后使用 after=event_id 追赶持久阶段事件；它不重新执行 Turn。
    游标必须属于当前 Session。返回的是 project_turn_event 的公开投影，
    不是含模型消息/工具内容的 trajectory；两类记录用途不同。
    """
    require_session(session_id)
    params = params or {}
    from ...runtime.turn_events import TurnEvent, project_turn_event
    max_page_size = session_store.MAX_RUNTIME_TURN_EVENT_PAGE_SIZE

    limit = optional_int(params.get("limit"), default=100)
    if limit is None or not 1 <= limit <= max_page_size:
        raise ApiError(
            "INVALID_RUNTIME_EVENT_LIMIT",
            f"limit 必须在 1 到 {max_page_size} 之间",
            details={"limit": limit, "max": max_page_size},
        )
    after = empty_to_none(params.get("after"))
    try:
        page = session_store.list_runtime_turn_events(
            session_id, after=after, limit=limit
        )
    except session_store.UnknownRuntimeTurnEventCursor as exc:
        raise ApiError(
            "RUNTIME_EVENT_CURSOR_NOT_FOUND",
            "运行事件游标不存在或不属于当前会话",
            status=409,
            details={"after": after, "session_id": session_id},
        ) from exc
    public_events = []
    for item in page["events"]:
        persisted = dict(item)
        sequence = persisted.pop("sequence")
        public_events.append(
            project_turn_event(
                TurnEvent.model_validate(persisted), sequence=int(sequence)
            ).model_dump(mode="json")
        )
    return {"enabled": True, **page, "events": public_events}


def patch_session(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    require_session(session_id)

    if "working_dir" in payload:
        raise ApiError(
            "SESSION_WORKING_DIR_MANAGED",
            "会话工作目录创建后不可更换或解绑",
            status=409,
            details={"field": "working_dir", "session_id": session_id},
        )

    title = payload.get("title")
    if title is not None and not session_store.rename_session(session_id, str(title)):
        raise ApiError("SESSION_NOT_FOUND", "会话不存在", status=404, details={"session_id": session_id})

    if "folder_id" in payload and not session_store.move_session(session_id, empty_to_none(payload.get("folder_id"))):
        raise ApiError("FOLDER_NOT_FOUND", "文件夹不存在或会话不存在", status=404, details={"session_id": session_id})

    action = empty_to_none(payload.get("status_action"))
    if action:
        from ...session import service as session_service

        # trash/restore 走产品级入口：它们要跨库带上会话的私有记忆，
        # 只调 session_store 会把记忆留在原地。
        actions = {
            "archive": session_store.archive_session,
            "unarchive": session_store.unarchive_session,
            "trash": session_service.trash_session_fully,
            "restore": session_service.restore_session_fully,
        }
        fn = actions.get(action)
        if fn is None:
            raise ApiError("INVALID_STATUS_ACTION", "不支持的会话状态操作", details={"status_action": action})
        if not fn(session_id):
            raise ApiError("INVALID_SESSION_STATE", "当前会话状态不支持该操作", details={"status_action": action})

    return {"session": require_session(session_id)}

def purge_session(session_id: str) -> dict[str, Any]:
    """永久删除一个 Session 及所有仅被它引用的内容。

    它与移入回收站转换分开，而不是另一种 ``status_action``：前者可逆，本操作不可逆；
    同一端点有时表示“移动”、有时表示“销毁”，无法正确表达这种差异。
    """
    from ...session import service as session_service

    try:
        deleted = session_service.purge_session_fully(session_id)
    except Exception as err:
        # 跑过运行时的会话删不掉：持久化层的 purge 只清了引用 goal 的一部分表，
        # 剩下的行把 goal 钉住，外键就拦下整笔删除。这里不替它补删——那些表归
        # 运行时管——但要说清楚是什么卡住了，别让用户看到一句"服务内部错误"。
        #
        # 认字符串而不是认异常类型：异常类型是存储层的，这一层不该把它 import 进来。
        if "FOREIGN KEY constraint failed" not in str(err):
            raise
        raise ApiError(
            "SESSION_PURGE_BLOCKED",
            "这个会话运行过任务，残留的运行时记录挡住了彻底删除。可以先放回收站。",
            status=409,
            details={"session_id": session_id},
        ) from err
    if not deleted:
        raise ApiError("SESSION_NOT_FOUND", "会话不存在", status=404,
                       details={"session_id": session_id})
    return {"purged": True, "session_id": session_id}


def empty_session_trash() -> dict[str, Any]:
    """永久删除所有已回收 Session，并报告数量。"""
    from ...session import service as session_service

    return {"purged": session_service.empty_trash_fully()}


def list_folders(params: dict[str, Any] | None = None) -> dict[str, Any]:
    status = empty_to_none((params or {}).get("status")) or "active"
    if status not in {"active", "archived", "trashed", "all"}:
        raise ApiError("INVALID_FOLDER_STATUS", "status 取值非法", details={"status": status})
    return {"folders": session_store.folder_tree(status=status)}

def create_folder(payload: dict[str, Any]) -> dict[str, Any]:
    name = required_str(payload, "name")
    parent_id = empty_to_none(payload.get("parent_id"))
    if parent_id is not None and session_store.get_folder(parent_id) is None:
        raise ApiError("FOLDER_NOT_FOUND", "父文件夹不存在", status=404, details={"parent_id": parent_id})
    folder_id = session_store.create_folder(name, parent_id=parent_id)
    return {"folder": session_store.get_folder(folder_id)}

def patch_folder(folder_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """改文件夹：重命名 / 移动父级 / 归档·回收·恢复（后两者级联子树+会话）。"""
    if session_store.get_folder(folder_id) is None:
        raise ApiError("FOLDER_NOT_FOUND", "文件夹不存在", status=404, details={"folder_id": folder_id})
    name = empty_to_none(payload.get("name"))
    if name and not session_store.rename_folder(folder_id, name):
        raise ApiError("FOLDER_RENAME_FAILED", "重命名失败", details={"folder_id": folder_id})
    if "parent_id" in payload:
        ok, reason = session_store.move_folder(folder_id, empty_to_none(payload.get("parent_id")))
        if not ok:
            raise ApiError("FOLDER_MOVE_FAILED", "移动失败（可能造成环或父级不存在）", status=400,
                           details={"reason": reason})
    status = empty_to_none(payload.get("status"))
    if status:
        ok, reason = session_store.set_folder_status(folder_id, status)
        if not ok:
            raise ApiError("FOLDER_STATUS_FAILED", "状态变更失败", status=400, details={"reason": reason})
    return {"folder": session_store.get_folder(folder_id)}

def delete_folder(folder_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    ok, reason = session_store.delete_folder(folder_id)
    if not ok:
        code = "FOLDER_NOT_FOUND" if reason == "folder_not_found" else "FOLDER_NOT_EMPTY"
        status = 404 if code == "FOLDER_NOT_FOUND" else 409
        raise ApiError(code, "文件夹删除失败", status=status, details={"reason": reason})
    return {"ok": True, "folder_id": folder_id, "deleted": True}

def _requested_attachment_ids(payload: dict[str, Any]) -> tuple[str, ...]:
    """读取消息声明的附件 id，但不信任这些值。"""
    raw = payload.get("attachment_ids") or []
    if isinstance(raw, str):
        raw = [item for item in raw.split(",") if item.strip()]
    if not isinstance(raw, list):
        raise ApiError("INVALID_ATTACHMENT_IDS", "attachment_ids 必须是数组", status=400)
    ids = tuple(str(item).strip() for item in raw if str(item).strip())
    if len(ids) > MAX_ATTACHMENTS_PER_TURN:
        raise ApiError(
            "TOO_MANY_ATTACHMENTS",
            "本轮附件数量超过上限",
            status=400,
            details={"max_attachments_per_turn": MAX_ATTACHMENTS_PER_TURN, "count": len(ids)},
        )
    return ids


def chat_turn(
    payload: dict[str, Any],
    *,
    on_stream_event: Callable[[dict[str, Any]], None] | None = None,
    on_turn_accepted: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """将一次 Chat API 请求交给 Runtime，并投影用户可见结果。

    这里校验请求标识、Session、附件声明和 routing policy；真正的 Turn 接受、
    幂等重放与执行窗口（Turn Window）由 run_entry_turn / Store 持有。
    on_turn_accepted 表示输入已经持久接受，on_stream_event 传递执行进展，
    两者都不等于最终答复已提交。正式答复提交后唤醒 post-commit worker，
    派生状态未结算时窗口仍可能是 post_commit_pending。

    JSON route 与 SSE handler 都调用本函数，区别只在回调是否传入。调用方应已建立
    Session 数据库 scope；返回前可能已启动后台 post-commit worker，因此 SSE final
    的发送时刻与后台摘要执行没有固定先后保证。
    """

    session_id = required_str(payload, "session_id")
    client_request_id = required_str(payload, "client_request_id")
    if len(client_request_id) > session_store.MAX_TURN_EXECUTION_IDENTIFIER_LENGTH:
        raise ApiError(
            "CLIENT_REQUEST_ID_TOO_LONG",
            "请求标识超过安全长度限制",
            status=400,
            details={
                "max_length": session_store.MAX_TURN_EXECUTION_IDENTIFIER_LENGTH,
            },
        )
    require_session(session_id)
    message = required_str(payload, "message")
    config = empty_to_none(payload.get("config"))
    features = load_features(config)
    attachment_ids = _requested_attachment_ids(payload)
    try:
        routing_policy = _resolve_chat_turn_routing_policy(
            payload=payload,
            session_id=session_id,
            client_request_id=client_request_id,
        )
    except TurnRoutingPolicyError as exc:
        raise ApiError(
            exc.code,
            str(exc),
            status=(
                409
                if exc.code in {
                    "CLIENT_REQUEST_ID_REUSED",
                    "STORED_RUNTIME_POLICY_INVALID",
                }
                else 400
            ),
        ) from exc
        # Host 重启期间可能丢失 daemon worker。开始新用户请求不代表可以绕过被持有的 Window，
        # 但这是在后台恢复持久派生状态 job 的安全机会。Entry audit 仍是接受该新请求的
        # 权威依据。
    _schedule_pending_turn_post_commit_jobs(session_id)
    try:
        result = run_entry_turn(
            user_input=message,
            features=features,
            session_id=session_id,
            client_request_id=client_request_id,
            on_stream_event=on_stream_event,
            on_turn_accepted=(
                (lambda accepted: on_turn_accepted(_accepted_turn_view(accepted)))
                if on_turn_accepted is not None
                else None
            ),
            attachment_ids=attachment_ids,
            routing_policy=routing_policy.snapshot,
            persist_routing_policy_as_session_default=(
                routing_policy.persist_as_session_default
            ),
            store=session_store,
        )
    except session_store.AttachmentBindingError as exc:
        # 刻意拒绝整个 Turn：用户附加文件后收到忽略该文件的回答，比被告知发送失败受到的
        # 误导更严重。
        raise ApiError(
            "ATTACHMENT_BINDING_FAILED",
            "附件无法绑定到本轮，请重新添加后发送",
            status=409,
            details={"reason": exc.reason, "attachment_ids": list(exc.attachment_ids)},
        ) from exc
    except ModelGatewayError as exc:
        raise model_api_error(exc) from exc
    except ContextBudgetExceeded as exc:
        details = {
            "guard_limit": exc.limit,
            "estimated_tokens": exc.estimated_tokens,
            "degraded": list(exc.degraded),
        }
        raise ApiError(
            "CONTEXT_BUDGET_EXCEEDED",
            "上下文超过安全预算，已在模型调用前停止",
            status=422,
            details=details,
            outcome=build_error_outcome(
                code="CONTEXT_BUDGET_EXCEEDED",
                domain="context",
                message=str(exc),
                retryable=False,
                details=details,
            ).to_dict(),
        ) from exc
    except SessionRunBusyError as exc:
        raise _session_busy_api_error(exc) from exc
    except (session_store.TurnExecutionBusyError, TurnWindowBlockedError) as exc:
        raise _turn_window_busy_api_error(exc) from exc
    except session_store.TurnExecutionRequestIdCollision as exc:
        raise ApiError(
            "CLIENT_REQUEST_ID_REUSED",
            "同一请求标识不能用于不同的消息或附件",
            status=409,
            details={"session_id": session_id},
        ) from exc
    if result.window_state == "post_commit_pending":
        # 正式交付已经提交。调度失败不能把它变成 API 失败，也不能导致客户端重试用户消息；
        # 保留的 Window/job 仍可在后续 Host 调用时恢复。
        _schedule_pending_turn_post_commit_jobs(session_id)
    from .session_titles import schedule_session_title

    schedule_session_title(session_id)
    return {
        "result": turn_result_view(result),
        "session": session_summary(require_session(session_id)),
    }


def _resolve_chat_turn_routing_policy(
    *,
    payload: dict[str, Any],
    session_id: str,
    client_request_id: str,
) -> ResolvedTurnRoutingPolicy:
    """解析新策略，同时保持已接受请求重放稳定。

    已存在 request-id 时优先验证原 Turn 的 snapshot/hash；本次显式 policy 若不同
    就拒绝复用 ID。只有新请求才按 request > Session > default 解析，避免重试因
    会话默认值变更而悄悄换 lane。此处解析/读取，实际保存仍随 Turn 接受事务完成。
    """

    existing = session_store.get_turn_execution_for_client_request(
        session_id=session_id,
        client_request_id=client_request_id,
    )
    request_policy_provided = "runtime_policy" in payload
    if existing is not None:
        record = existing.get("routing_policy")
        if record is None:
            raise TurnRoutingPolicyError(
                "STORED_RUNTIME_POLICY_INVALID",
                "the accepted Turn has no routing-policy snapshot",
            )
        elif isinstance(record, dict):
            snapshot_json = record.get("snapshot_json")
            snapshot_hash = record.get("snapshot_hash")
            if not isinstance(snapshot_json, str) or not isinstance(
                snapshot_hash, str
            ):
                raise TurnRoutingPolicyError(
                    "STORED_RUNTIME_POLICY_INVALID",
                    "the accepted Turn routing-policy snapshot is incomplete",
                )
            snapshot = parse_turn_routing_policy_snapshot(
                snapshot_json,
                expected_sha256=snapshot_hash,
            )
        else:
            raise TurnRoutingPolicyError(
                "STORED_RUNTIME_POLICY_INVALID",
                "the accepted Turn routing-policy snapshot is invalid",
            )
        if request_policy_provided:
            requested = resolve_turn_routing_policy(
                request_policy=payload.get("runtime_policy"),
                request_policy_provided=True,
                stored_session_policy_json=None,
            ).snapshot.policy
            if requested != snapshot.policy:
                raise TurnRoutingPolicyError(
                    "CLIENT_REQUEST_ID_REUSED",
                    "the request routing policy differs from the accepted Turn",
                )
        return ResolvedTurnRoutingPolicy(
            snapshot=snapshot,
            persist_as_session_default=False,
        )

    try:
        stored = session_store.get_session_turn_routing_policy(session_id)
    except ValueError as exc:
        raise TurnRoutingPolicyError(
            "STORED_RUNTIME_POLICY_INVALID",
            "the stored Session routing policy failed its integrity check",
        ) from exc
    return resolve_turn_routing_policy(
        request_policy=payload.get("runtime_policy"),
        request_policy_provided=request_policy_provided,
        stored_session_policy_json=(
            str(stored["policy_json"]) if stored is not None else None
        ),
    )


def _schedule_pending_turn_post_commit_jobs(session_id: str) -> None:
    """尽力唤醒 Host 以执行持久提交后工作。

    此组合层 helper 刻意不检查 job payload，也不同步调用供应商。只有当存储表明当前 Session
    仍处于对应 Window 状态时，它才启动 Runtime 所有的 worker。
    """

    try:
        inspection = session_store.inspect_turn_execution(session_id)
        window = inspection.get("window")
        if not isinstance(window, dict) or window.get("window_state") != "post_commit_pending":
            return
        schedule_turn_post_commit_jobs(session_id=session_id, store=session_store)
    except Exception:
        _LOG.exception("could not schedule durable turn post-commit work")


def _accepted_turn_view(accepted: AcceptedEntryTurn) -> dict[str, Any]:
    """只有持久准入完成后才发出 SSE 接受记录。"""

    return {
        "session_id": accepted.session_id,
        "turn_id": accepted.turn_id,
        "client_request_id": accepted.client_request_id,
        "window_revision": accepted.window_revision,
        "replayed": accepted.replayed,
    }


def _turn_window_busy_api_error(
    exc: session_store.TurnExecutionBusyError | TurnWindowBlockedError,
) -> ApiError:
    details = dict(exc.details)
    return ApiError(
        "TURN_IN_PROGRESS",
        "该会话上一轮尚未安全收束，请等待完成或稍后重试",
        status=409,
        details=details,
    )

def _session_busy_api_error(exc: SessionRunBusyError) -> ApiError:
    details = dict(exc.details)
    return ApiError(
        exc.code,
        "该会话已有任务正在运行，请稍后重试",
        status=409,
        details=details,
        outcome=build_error_outcome(
            code=exc.code,
            domain="runtime",
            message=exc.message,
            retryable=exc.retryable,
            details=details,
        ).to_dict(),
    )
