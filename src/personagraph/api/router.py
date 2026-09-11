"""HTTP 方法/路径到应用服务的声明式路由，区别于模型选择 L0/L1 的语义路由。

普通 /api/chat：dispatch_response → CHAT_BODY → Session scope → service.chat_turn。
/api/chat/stream 只借用路径识别与同一正文合同，由 server 专用 SSE handler 交付。
此模块选择应用用例，不决定 Agent 的 Plan、ToolCall 或处理级别。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from . import service
from . import security as api_security
from ..session import store as session_store
from .contracts import (
    BodySchema,
    CHAT_BODY,
    CONFIG_BODY,
    MODEL_PROFILE_BODY,
    DOCUMENT_BODY,
    DOCUMENT_INGEST_JOB_BODY,
    DOCUMENT_INGEST_RETRY_BODY,
    FOLDER_BODY,
    MAX_REQUEST_TARGET_BYTES,
    OBJECT_BODY,
    PROJECT_BODY,
    PROJECT_ORDER_BODY,
    SESSION_CONTEXT_CORRECTION_BODY,
    AUTONAME_BODY,
    SESSION_BODY,
)


@dataclass(frozen=True)
class RouteResponse:
    """JSON payload 及路由选择的成功 HTTP 状态。"""

    payload: dict[str, Any]
    status: int = 200

    def __post_init__(self) -> None:
        if not 200 <= self.status < 300:
            raise ValueError("RouteResponse status must be a successful HTTP status")


RouteResult = dict[str, Any] | RouteResponse
Handler = Callable[[dict[str, str], dict[str, Any], dict[str, Any]], RouteResult]


@dataclass(frozen=True)
class Route:
    method: str
    path: tuple[str, ...]
    handler: Handler
    body: BodySchema = OBJECT_BODY

    def match(self, method: str, parts: list[str]) -> dict[str, str] | None:
        """匹配方法和路径段，提取有长度界限的占位参数；不验证业务身份或执行 handler。

        返回空 dict 也表示匹配成功，None 才表示不匹配。Session ID 的格式和多来源一致性
        在 _route_session_id 检查，不能仅凭路径占位符接受任意数据库标识。
        """

        if method != self.method or len(parts) != len(self.path):
            return None
        params: dict[str, str] = {}
        for pattern, actual in zip(self.path, parts, strict=True):
            if pattern.startswith("{") and pattern.endswith("}"):
                name = pattern[1:-1]
                if not actual or len(actual) > 128:
                    return None
                params[name] = actual
            elif pattern != actual:
                return None
        return params


def _service_handler(fn: Callable[..., dict[str, Any]], *param_names: str) -> Handler:
    """将声明式 route 参数适配成 service 的位置参数与 JSON body。

    工厂创建的是薄调用包装；正文校验和 Session scope 在 dispatch_response 中完成，
    handler 不另起线程，也不重新决定请求应进入哪个业务函数。
    """

    def handler(params: dict[str, str], body: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
        args = [params[name] for name in param_names]
        return fn(*args, body) if args else fn(body)

    return handler


def _get_with_params(fn: Callable[..., dict[str, Any]], *param_names: str) -> Handler:
    def handler(params: dict[str, str], body: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
        args = [params[name] for name in param_names]
        return fn(*args, query) if args else fn(query)

    return handler


def _get_no_query(fn: Callable[..., dict[str, Any]], *param_names: str) -> Handler:
    def handler(params: dict[str, str], body: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
        return fn(*(params[name] for name in param_names))

    return handler


def _health_payload(
    _params: dict[str, str],
    _body: dict[str, Any],
    query: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": True, "service": "personagraph-api"}
    challenge = query.get("challenge")
    if challenge is None:
        return payload
    try:
        payload["identity"] = api_security.api_identity_proof(challenge)
    except ValueError as exc:
        raise service.ApiError(
            "INVALID_IDENTITY_CHALLENGE",
            "本地 API 身份挑战格式不正确",
            status=400,
        ) from exc
    return payload


ROUTES: tuple[Route, ...] = (
    Route("GET", ("api", "health"), _health_payload),
    Route("GET", ("api", "status"), lambda _p, _b, _q: service.system_status()),
    Route("GET", ("api", "config"), lambda _p, _b, _q: service.get_config()),
    Route("POST", ("api", "config"), lambda _p, b, _q: service.update_config(b), CONFIG_BODY),
    Route("PUT", ("api", "config"), lambda _p, b, _q: service.update_config(b), CONFIG_BODY),
    Route("GET", ("api", "model-profiles"), lambda _p, _b, q: service.list_model_profiles(q)),
    Route("POST", ("api", "model-profiles"),
          lambda _p, b, _q: service.create_model_profile(b), MODEL_PROFILE_BODY),
    Route("PATCH", ("api", "model-profiles", "{profile_id}"),
          _service_handler(service.update_model_profile, "profile_id"), MODEL_PROFILE_BODY),
    Route("DELETE", ("api", "model-profiles", "{profile_id}"),
          _get_no_query(service.delete_model_profile, "profile_id")),
    Route("POST", ("api", "model-profiles", "{profile_id}", "activate"),
          _get_no_query(service.activate_model_profile, "profile_id")),
    # 明文单独一条路由：列表永远不带 key，只有明确要看时才走这里。
    Route("GET", ("api", "model-profiles", "{profile_id}", "secret"),
          _get_no_query(service.reveal_model_profile_secret, "profile_id")),
    Route("PATCH", ("api", "config"), lambda _p, b, _q: service.update_config(b), CONFIG_BODY),

    Route("GET", ("api", "sessions"), lambda _p, _b, q: service.list_sessions(q)),
    Route("POST", ("api", "sessions"), lambda _p, b, _q: service.create_session(b), SESSION_BODY),
    Route("GET", ("api", "sessions", "{session_id}"), _get_no_query(service.get_session, "session_id")),
    Route("POST", ("api", "sessions", "{session_id}", "post-commit", "control"),
          _service_handler(service.control_turn_post_commit_jobs, "session_id")),
    Route(
        "GET",
        ("api", "sessions", "{session_id}", "insession-tasks", "{insession_task_id}"),
        _get_no_query(service.get_insession_task_details, "session_id", "insession_task_id"),
    ),
    Route("GET", ("api", "sessions", "{session_id}", "runtime-events"), _get_with_params(service.get_runtime_events, "session_id")),
    Route("GET", ("api", "sessions", "{session_id}", "attachments"), _get_with_params(service.list_session_attachments, "session_id")),
    Route("DELETE", ("api", "sessions", "{session_id}", "attachments", "{attachment_id}"),
          lambda params, body, _query: service.delete_session_attachment(params["session_id"], params["attachment_id"])),
    Route("PATCH", ("api", "sessions", "{session_id}"), _service_handler(service.patch_session, "session_id"), SESSION_BODY),
    # 永久删除单独一条路由，不混进 status_action：那些都是可逆的，这条不是。
    Route("DELETE", ("api", "sessions", "{session_id}"),
          _get_no_query(service.purge_session, "session_id")),
    Route("GET", ("api", "appearance", "background"), lambda _p, _b, _q: service.get_background()),
    Route("DELETE", ("api", "appearance", "background"), lambda _p, _b, _q: service.clear_background()),
    # 项目就是会话绑定的那个目录，按目录分组；注册表只记住空项目。
    Route("GET", ("api", "projects"), lambda _p, _b, q: service.list_projects(q)),
    Route("POST", ("api", "projects"), lambda _p, b, _q: service.create_project(b), PROJECT_BODY),
    Route("PATCH", ("api", "projects"), lambda _p, b, _q: service.rename_project(b), PROJECT_BODY),
    Route("POST", ("api", "projects", "pin"), lambda _p, b, _q: service.pin_project(b), PROJECT_BODY),
    Route("POST", ("api", "projects", "reorder"),
          lambda _p, b, _q: service.reorder_projects(b), PROJECT_ORDER_BODY),
    Route("POST", ("api", "projects", "forget"), lambda _p, b, _q: service.forget_project(b), PROJECT_BODY),
    Route("POST", ("api", "sessions", "{session_id}", "autoname"),
          _service_handler(service.name_session_from_turn, "session_id"), AUTONAME_BODY),
    Route("POST", ("api", "sessions", "trash", "empty"),
          lambda _p, _b, _q: service.empty_session_trash()),
    Route("GET", ("api", "sessions", "{session_id}", "session-context"), _get_with_params(service.get_session_context, "session_id")),
    Route("GET", ("api", "sessions", "{session_id}", "session-context", "explain"), _get_with_params(service.explain_session_context_state, "session_id")),
    Route("GET", ("api", "sessions", "{session_id}", "session-context", "export"), _get_with_params(service.export_session_context, "session_id")),
    Route("POST", ("api", "sessions", "{session_id}", "session-context", "clear"), _service_handler(service.clear_session_context, "session_id")),
    Route("POST", ("api", "sessions", "{session_id}", "session-context", "corrections"), _service_handler(service.create_session_context_correction, "session_id"), SESSION_CONTEXT_CORRECTION_BODY),
    Route("POST", ("api", "sessions", "{session_id}", "session-context", "repair", "preview"), _service_handler(service.preview_session_context_repair, "session_id")),
    Route("POST", ("api", "sessions", "{session_id}", "session-context", "repair", "apply"), _service_handler(service.apply_session_context_repair, "session_id")),

    # 工作区文件管理（FR-8）：会话 working_dir 内真实文件/文件夹增删改（删除→系统回收站）
    Route("GET", ("api", "sessions", "{session_id}", "files"), _get_with_params(service.list_workspace_files, "session_id")),
    Route("POST", ("api", "sessions", "{session_id}", "files"), _service_handler(service.create_workspace_entry, "session_id")),
    Route("DELETE", ("api", "sessions", "{session_id}", "files"), _service_handler(service.delete_workspace_entry, "session_id")),
    Route("GET", ("api", "sessions", "{session_id}", "file"), _get_with_params(service.read_workspace_file, "session_id")),
    Route("PUT", ("api", "sessions", "{session_id}", "file"), _service_handler(service.write_workspace_file, "session_id")),

    Route("GET", ("api", "folders"), lambda _p, _b, q: service.list_folders(q)),
    Route("POST", ("api", "folders"), lambda _p, b, _q: service.create_folder(b), FOLDER_BODY),
    Route("PATCH", ("api", "folders", "{folder_id}"), _service_handler(service.patch_folder, "folder_id"), FOLDER_BODY),
    Route("DELETE", ("api", "folders", "{folder_id}"), _service_handler(service.delete_folder, "folder_id")),

    Route("POST", ("api", "chat"), lambda _p, b, _q: service.chat_turn(b), CHAT_BODY),

    Route("GET", ("api", "documents"), lambda _p, _b, q: service.list_documents(q)),
    Route("GET", ("api", "documents", "{document_id}"), _get_with_params(service.get_document, "document_id")),
    Route("PATCH", ("api", "documents", "{document_id}"), _service_handler(service.patch_document, "document_id"), DOCUMENT_BODY),
    Route("DELETE", ("api", "documents", "{document_id}"), _get_with_params(service.delete_document, "document_id")),
    Route("POST", ("api", "documents", "{document_id}", "detach"), _service_handler(service.detach_document, "document_id"), DOCUMENT_BODY),

    Route(
        "GET",
        ("api", "document-ingest-jobs"),
        lambda _p, _b, q: service.list_document_ingest_jobs(q),
    ),
    Route(
        "POST",
        ("api", "document-ingest-jobs"),
        lambda _p, b, _q: RouteResponse(
            service.enqueue_document_ingest_job(b),
            status=202,
        ),
        DOCUMENT_INGEST_JOB_BODY,
    ),
    Route(
        "GET",
        ("api", "document-ingest-jobs", "{job_id}"),
        _get_with_params(service.get_document_ingest_job, "job_id"),
    ),
    Route(
        "POST",
        ("api", "document-ingest-jobs", "{job_id}", "retry"),
        lambda params, body, _query: RouteResponse(
            service.retry_document_ingest_job(params["job_id"], body),
            status=202,
        ),
        DOCUMENT_INGEST_RETRY_BODY,
    ),

)


def dispatch_response(method: str, target: str, body: dict[str, Any]) -> RouteResponse:
    """普通 JSON 请求主链：匹配路由 → 校验已声明字段 → 绑定 Session → 调用 service。

    path 先 URL decode，再与 ROUTES 匹配；存在 Session 标识时，整个 service 调用
    都位于 session_database_scope 内，使 Runtime、工具与 trajectory 共用所属存储。
    返回 RouteResponse 只携带成功 HTTP 状态和 payload；异常交给 server 的传输边界。
    streaming chat 不从这里发送响应，见 ApiHandler._handle_chat_stream。
    """
    if len(target.encode("utf-8")) > MAX_REQUEST_TARGET_BYTES:
        raise service.ApiError(
            "REQUEST_TARGET_TOO_LARGE",
            "请求路径或查询参数超过限制",
            status=414,
            details={"max_bytes": MAX_REQUEST_TARGET_BYTES},
        )
    parsed = urlparse(target)
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if not parts or parts[0] != "api":
        raise service.ApiError("NOT_FOUND", "API path not found", status=404)
    query = _query_params(parsed.query)
    for route in ROUTES:
        params = route.match(method, parts)
        if params is not None:
            validated_body = route.body.validate(body)
            session_id = _route_session_id(route, params, validated_body, query)
            if session_id is None:
                result = route.handler(params, validated_body, query)
            else:
                with session_store.session_database_scope(session_id):
                    result = route.handler(params, validated_body, query)
            return result if isinstance(result, RouteResponse) else RouteResponse(result)
    raise service.ApiError("NOT_FOUND", "API path not found", status=404)


def _route_session_id(
    route: Route,
    params: dict[str, str],
    body: dict[str, Any],
    query: dict[str, Any],
) -> str | None:
    """从 path、已声明 body 字段和 query 汇总 Session ID，拒绝格式错误与彼此冲突。

    无标识返回 None，表示本路由不建立 Session scope；有标识只完成格式/一致性
    校验，Session 存在性及访问具体对象的业务检查仍由 service 负责。
    """

    path_session_id = params.get("session_id")
    body_session_id = (
        body.get("session_id") if "session_id" in route.body.fields else None
    )
    query_session_id = query.get("session_id")
    candidates = [
        value
        for value in (path_session_id, body_session_id, query_session_id)
        if value is not None
    ]
    if not candidates:
        return None
    if any(not isinstance(value, str) or not value.strip() for value in candidates):
        raise service.ApiError(
            "INVALID_SESSION_ID",
            "session_id 格式不正确",
            status=400,
        )
    if len(set(candidates)) != 1:
        raise service.ApiError(
            "SESSION_ID_MISMATCH",
            "路径、请求体与查询参数中的 session_id 不一致",
            status=400,
        )
    session_id = str(candidates[0])
    try:
        session_store.validate_session_id(session_id)
    except ValueError as exc:
        raise service.ApiError(
            "INVALID_SESSION_ID",
            "session_id 格式不正确",
            status=400,
        ) from exc
    return session_id


ATTACHMENT_UPLOAD_PATH = ("api", "sessions", "{session_id}", "attachments")


def attachment_upload_session_id(path: str) -> str | None:
    """当路径表示二进制附件上传时返回 Session id。

    上传不能经过 JSON dispatcher：其正文是原始字节，远大于 JSON 正文限制。
    """
    parts = [unquote(part) for part in urlparse(path).path.strip("/").split("/") if part]
    if len(parts) != 4:
        return None
    if parts[0] != "api" or parts[1] != "sessions" or parts[3] != "attachments":
        return None
    return parts[2] or None


def is_chat_stream_target(target: str) -> bool:
    parsed = urlparse(target)
    return [unquote(part) for part in parsed.path.split("/") if part] == ["api", "chat", "stream"]


def _query_params(query: str) -> dict[str, Any]:
    raw = parse_qs(query, keep_blank_values=True)
    return {key: values[-1] if values else "" for key, values in raw.items()}

_BACKGROUND_UPLOAD_PATH = ("api", "appearance", "background")
_BACKGROUND_ASSET_PATH = ("api", "appearance", "background", "asset")


def _path_parts(path: str) -> tuple[str, ...]:
    return tuple(part for part in path.split("?", 1)[0].strip("/").split("/") if part)


def is_background_upload(path: str) -> bool:
    """背景是原始字节上传，和附件一样绕开 JSON 请求体。"""

    return _path_parts(path) == _BACKGROUND_UPLOAD_PATH


def is_background_asset(path: str) -> bool:
    """这一条要回二进制，不能走 JSON 出口。"""

    return _path_parts(path) == _BACKGROUND_ASSET_PATH
