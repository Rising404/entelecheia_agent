"""本地 HTTP Host：从进程启动到 JSON / SSE 传输的阅读入口。

main → run → ApiHandler._handle。普通 JSON 经 router.dispatch_response；
/api/chat/stream 由 _handle_chat_stream 建立 Session scope 后直调 service.chat_turn，
两种传输最终共用同一个 Runtime。服务启动、连接写入与业务提交是不同生命周期，
阅读时继续跟到 runtime/entry/application.py，而不是在收到 HTTP 200 时判断执行成功。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import service
from .contracts import (
    CHAT_BODY,
    MAX_REQUEST_BODY_BYTES,
    MAX_REQUEST_TARGET_BYTES,
    parse_json_object,
)
from ..workspace.files.attachments import MAX_ATTACHMENT_BYTES
from .service.appearance import MAX_BYTES as BACKGROUND_MAX_BYTES
from .router import (
    attachment_upload_session_id,
    is_background_asset,
    is_background_upload,
    dispatch_response,
    is_chat_stream_target,
)
from . import security as api_security
from .sse import SseDelivery, is_client_disconnect
from ..session import store as session_store


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DOCUMENT_MAINTENANCE_STOP_TIMEOUT_SECONDS = 5.0
POST_COMMIT_STOP_TIMEOUT_SECONDS = 5.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

class ApiHandler(BaseHTTPRequestHandler):
    """薄 HTTP 传输层；路由与正文契约位于 handler 之外。"""

    server_version = "EntelecheiaAPI/0.2"

    def do_OPTIONS(self) -> None:  # noqa: N802 - 标准库钩子名称
        try:
            self._require_allowed_origin()
            self._try_send_json({"ok": True})
        except service.ApiError as exc:
            self._try_send_json(exc.to_payload(), status=exc.status)

    def do_GET(self) -> None:  # noqa: N802
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")

    def do_PUT(self) -> None:  # noqa: N802
        self._handle("PUT")

    def do_PATCH(self) -> None:  # noqa: N802
        self._handle("PATCH")

    def do_DELETE(self) -> None:  # noqa: N802
        self._handle("DELETE")

    def log_message(self, fmt: str, *args: Any) -> None:
        # 默认保持 API server 安静；Electron 可自行呈现错误。
        return

    def _handle(self, method: str) -> None:
        """所有业务 HTTP 方法共用的传输入口：来源/认证 → 路径限额 → 专用或 JSON 路由。

        Chat SSE 与二进制上传先分流，不能都交给 _read_json。普通请求经
        dispatch_response 完成 BodySchema、Session scope 和 service 调用后再发送 JSON。
        ApiError 保留公开错误与 HTTP 状态；其他异常在此收敛为 INTERNAL_ERROR。
        业务执行与响应写入分开，客户端断连不能作为业务失败或再次执行的依据。
        """

        try:
            self._require_allowed_origin()
            self._require_authorization()
            if len(self.path.encode("utf-8")) > MAX_REQUEST_TARGET_BYTES:
                raise service.ApiError(
                    "REQUEST_TARGET_TOO_LARGE",
                    "请求路径或查询参数超过限制",
                    status=414,
                    details={"max_bytes": MAX_REQUEST_TARGET_BYTES},
                )
            # Transport split：SSE 自己管理响应帧；普通 JSON 路由随后进入 dispatcher。
            if method == "POST" and _is_chat_stream_path(self.path):
                self._handle_chat_stream()
                return
            if method == "POST" and is_background_upload(self.path):
                self._try_send_json(self._handle_background_upload())
                return
            if method == "GET" and is_background_asset(self.path):
                self._handle_background_asset()
                return
            upload_session_id = (
                attachment_upload_session_id(self.path) if method == "POST" else None
            )
            if upload_session_id is not None:
                safe_session_id = _database_session_id(upload_session_id)
                with session_store.session_database_scope(safe_session_id):
                    payload = self._handle_attachment_upload(safe_session_id)
                self._try_send_json(payload)
                return
            response = dispatch_response(method, self.path, self._read_json())
        except service.ApiError as exc:
            self._try_send_json(exc.to_payload(), status=exc.status)
        except Exception:  # pragma: no cover - 防御性 HTTP 边界
            self._try_send_json(
                service.ApiError("INTERNAL_ERROR", "服务内部错误", status=500).to_payload(),
                status=500,
            )
        else:
            self._try_send_json(response.payload, status=response.status)

    def _require_allowed_origin(self) -> None:
        """按 Origin policy 检查请求来源；这是来源边界，不代表 API token 已认证。

        OPTIONS 预检也调用它，具体允许值由 api/security.py 的策略统一决定。
        """

        origin = self.headers.get("Origin")
        if not api_security.origin_allowed(origin):
            raise service.ApiError(
                "ORIGIN_FORBIDDEN",
                "请求来源不在本地 API 白名单中",
                status=403,
            )

    def _require_authorization(self) -> None:
        """在路由/正文处理前检查 API 认证，并保留 health 与已配置开发来源的显式例外。

        不在此检查 Session 是否存在或授予工具权限；那些分别属于 service 与 Runtime。
        """

        if _is_public_health_path(self.path):
            return
        origin = self.headers.get("Origin")
        if api_security.unauthenticated_dev_origin_allowed(origin):
            return
        if not api_security.token_matches(self.headers.get("Authorization")):
            raise service.ApiError(
                "API_AUTH_REQUIRED",
                "本地 API 认证失败",
                status=401,
            )

    def _handle_attachment_upload(self, session_id: str) -> dict[str, Any]:
        """在 JSON 正文契约之外接收一次原始二进制上传。

        _handle 已先绑定目标 Session 数据库；上传产生附件记录，后续 chat_turn 还要通过
        attachment_ids 将其绑定到精确 Turn。上传成功本身不意味着模型已经读取文件正文。
        """
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise service.ApiError("INVALID_CONTENT_LENGTH", "上传必须提供 Content-Length", status=411)
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise service.ApiError("INVALID_CONTENT_LENGTH", "Content-Length 必须是整数", status=400) from exc
        if length < 0:
            raise service.ApiError("INVALID_CONTENT_LENGTH", "Content-Length 不能为负数", status=400)
        if length > MAX_ATTACHMENT_BYTES:
        # 在读取前拒绝，避免利用超大上传占用内存或连接。
            raise service.ApiError(
                "ATTACHMENT_TOO_LARGE",
                "文件超过单个附件大小上限",
                status=413,
                details={"max_bytes": MAX_ATTACHMENT_BYTES, "size_bytes": length},
            )
        payload = self.rfile.read(length) if length else b""
        if len(payload) != length:
            raise service.ApiError("UPLOAD_INCOMPLETE", "上传未完整接收，请重试", status=400)
        return service.upload_attachment(
            session_id,
            payload=payload,
            filename=service.decode_upload_filename(self.headers.get(service.FILENAME_HEADER)),
            declared_media_type=self.headers.get("Content-Type"),
        )

    def _handle_background_upload(self) -> dict[str, Any]:
        """像附件一样，以原始字节接收外观背景。"""

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise service.ApiError("INVALID_CONTENT_LENGTH", "上传必须提供 Content-Length", status=411)
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise service.ApiError("INVALID_CONTENT_LENGTH", "Content-Length 必须是整数", status=400) from exc
        if length < 0:
            raise service.ApiError("INVALID_CONTENT_LENGTH", "Content-Length 不能为负数", status=400)
        if length > BACKGROUND_MAX_BYTES:
            # 和附件一样：先拒再读，别让一个超大文件先把内存占住。
            raise service.ApiError(
                "BACKGROUND_TOO_LARGE",
                "背景文件超过大小上限",
                status=413,
                details={"max_bytes": BACKGROUND_MAX_BYTES, "size_bytes": length},
            )
        payload = self.rfile.read(length) if length else b""
        if len(payload) != length:
            raise service.ApiError("UPLOAD_INCOMPLETE", "上传未完整接收，请重试", status=400)
        return service.store_background(
            media_type=self.headers.get("Content-Type") or "",
            payload=payload,
            now=_utc_now(),
        )

    def _handle_background_asset(self) -> None:
        """唯一以字节而非 JSON 响应的路由。"""

        try:
            payload, media_type = service.read_background_asset()
        except service.ApiError as exc:
            self._try_send_json(exc.to_payload(), status=exc.status)
            return
        self.send_response(200)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Accept-Ranges", "none")
        # 换背景时 URL 会带上新的版本号，所以这里可以放心让它缓存久一点。
        self.send_header("Cache-Control", "private, max-age=86400")
        self._send_cors_headers()
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _read_json(self) -> dict[str, Any]:
        """有界读取 HTTP JSON 字节，再交给 parse_json_object 验证对象形状。

        Content-Length 在读取前校验，避免先分配超大正文；缺失或零长度视为空对象。
        此处只检查传输/JSON 形状，CHAT_BODY 的字段类型与 service 的必填/语义校验
        仍在后面，不能把成功 decode 当作可信 Turn 已接受。
        """

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return {}
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise service.ApiError("INVALID_CONTENT_LENGTH", "Content-Length 必须是整数", status=400) from exc
        if length < 0:
            raise service.ApiError("INVALID_CONTENT_LENGTH", "Content-Length 不能为负数", status=400)
        if length > MAX_REQUEST_BODY_BYTES:
            raise service.ApiError(
                "REQUEST_BODY_TOO_LARGE",
                "请求体超过 1 MiB 限制",
                status=413,
                details={"max_bytes": MAX_REQUEST_BODY_BYTES},
            )
        if length == 0:
            return {}
        content_type = str(self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if content_type and content_type != "application/json":
            raise service.ApiError(
                "UNSUPPORTED_MEDIA_TYPE",
                "请求体必须使用 application/json",
                status=415,
            )
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise service.ApiError("INCOMPLETE_REQUEST_BODY", "请求体未完整接收", status=400)
        return parse_json_object(raw)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _try_send_json(self, payload: dict[str, Any], status: int = 200) -> bool:
        """发送一个 JSON 响应，并将 peer 断开视为仅传输失败。

        此包装器被刻意放在路由执行之外。应用 OSError 仍必须变成 INTERNAL_ERROR；只有写入
        此响应时抛出的异常才能归类为客户端断连。
        """
        try:
            self._send_json(payload, status=status)
        except OSError as exc:
            if not is_client_disconnect(exc):
                raise
            return False
        return True

    def _handle_chat_stream(self) -> None:
        """Chat SSE 专用入口：校验正文、绑定 Session scope，直调共享 chat_turn 服务。

        此路径绕过普通 JSON dispatcher，但不绕过 CHAT_BODY / service / Runtime 准入。
        HTTP 200 先开启事件流；accepted 在输入事务提交后发出，running 是传输提示，
        runtime_event 是持久阶段事件，final 装载服务返回值，error 装载 API 异常。
        final 仍可能携带 incomplete，或重放请求观察到的 running 结果，须读 result.status
        和 window_state。SseDelivery 识别断连后停止写 socket，不取消已接受的 Runtime 执行。
        """

        body = CHAT_BODY.validate(self._read_json())
        session_id = _database_session_id(body.get("session_id"))
        with session_store.session_database_scope(session_id):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._send_cors_headers()
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            # Header 已提交：后续服务错误只能发 SSE error，不能再改写 HTTP 状态码。
            delivery = SseDelivery(self._send_sse)
            try:
                def emit_model_event(item: dict[str, Any]) -> None:
                    event = str(item.get("event") or "message")
                    payload = {key: value for key, value in item.items() if key != "event"}
                    delivery.send(event, payload)

                def emit_accepted_turn(item: dict[str, Any]) -> None:
            # ``accepted`` 只有在 Turn/input/window 事务提交后才由应用服务发出。因此重连可以
            # 使用其 Turn id，无须猜测服务器是否接受了 POST。
                    delivery.send("accepted", item)
                    delivery.send("running", {"turn_id": item["turn_id"]})

                result = service.chat_turn(
                    body,
                    on_stream_event=emit_model_event,
                    on_turn_accepted=emit_accepted_turn,
                )
                # final 是本次服务调用的终帧；业务成功与窗口释放仍看 result 内的状态。
                delivery.send("final", result)
            except service.ApiError as exc:
                delivery.send("error", exc.to_payload() | {"status": exc.status})
            except Exception:  # pragma: no cover - 防御性流边界
                error = service.ApiError("INTERNAL_ERROR", "服务内部错误", status=500)
                delivery.send("error", error.to_payload() | {"status": 500})

    def _send_sse(self, event: str, payload: dict[str, Any]) -> None:
        """把一个公开事件编码为 SSE frame 并 flush；只负责传输，不持久化或修改业务状态。

        runtime_event 的 event_id 写入 SSE id；断线后的事件追赶由 runtime-events API
        的 after 游标完成。不要在此混入 prompt、工具正文或 trajectory 私有诊断。
        """

        data = json.dumps(payload, ensure_ascii=False)
        runtime_event = payload.get("runtime_event")
        if event == "runtime_event" and isinstance(runtime_event, dict):
            event_id = str(runtime_event.get("event_id") or "").strip()
            if event_id:
                self.wfile.write(f"id: {event_id}\n".encode("utf-8"))
        self.wfile.write(f"event: {event}\n".encode("utf-8"))
        for line in data.splitlines() or ["{}"]:
            self.wfile.write(f"data: {line}\n".encode("utf-8"))
        self.wfile.write(b"\n")
        self.wfile.flush()

    def _send_cors_headers(self) -> None:
        origin = self.headers.get("Origin")
        if origin and api_security.origin_allowed(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,PATCH,DELETE,OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Authorization,Content-Type,X-Attachment-Filename",
        )


def _is_chat_stream_path(path: str) -> bool:
    return is_chat_stream_target(path)


def _is_public_health_path(path: str) -> bool:
    from urllib.parse import urlparse

    return urlparse(path).path.rstrip("/") == "/api/health"


def _database_session_id(value: Any) -> str:
    """将不可信 Session 参数校验为可安全选择数据库的 ID，不查询 Session 是否存在。

    SSE / 二进制上传未走普通 router，因此必须在进入 session_database_scope 前
    显式执行同类校验；Session 存在性继续由对应 service 验证。
    """

    if not isinstance(value, str) or not value.strip():
        raise service.ApiError(
            "INVALID_SESSION_ID",
            "session_id 格式不正确",
            status=400,
        )
    try:
        return session_store.validate_session_id(value)
    except ValueError as exc:
        raise service.ApiError(
            "INVALID_SESSION_ID",
            "session_id 格式不正确",
            status=400,
        ) from exc


def _build_document_maintenance_lifecycle():
    """避免维护依赖进入 HTTP 传输导入路径。"""

    from ..workspace.ingestion.composition import (
        build_project_document_maintenance_supervisor,
    )
    from ..tools.visual.publication_recovery import build_visual_publication_recovery

    return build_project_document_maintenance_supervisor(
        after_pass_factory=build_visual_publication_recovery,
    )


def _recover_active_l1_turns() -> int:
    """API Host 重启后重新发现已初始化的 L1 Turn。

    委托 recovery_worker 发现并调度已有 L1 run，不在 HTTP 启动线程直接运行模型。
    返回值是被调度/已有 worker 的会话数，不是恢复成功或新接受的 Turn 数。
    """

    from ..runtime.l1.recovery_worker import recover_active_l1_turns

    return recover_active_l1_turns()


def _build_turn_post_commit_lifecycle():
    """HTTP 宿主只拥有启停，任务发现/重试/租约结算仍归 Runtime。"""
    from ..runtime.post_commit.lifecycle import build_turn_post_commit_lifecycle

    return build_turn_post_commit_lifecycle()


def _bootstrap_tool_catalog() -> None:
    """在接受请求前发布进程内置的持久默认 Tool Catalog。

    这里只发布内置 Definition / profile；每个 Turn 的文件、工作区等可执行 binding
    仍由 L1 bootstrap 组合，启动成功不表示所有工具在所有会话中都可执行。
    """

    from ..tools.catalog.persistence import ToolCatalogRepository
    from ..tools.composition.default_catalog import (
        bootstrap_production_default_catalog,
    )

    bootstrap_production_default_catalog(ToolCatalogRepository())


def run(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    api_token: str | None = None,
) -> None:
    """启动回环 HTTP Host，并在接收请求前准备目录、维护服务与恢复任务。

    先占用端口再初始化共享 API token，避免竞争启动者覆盖正在服务的凭据。
    Tool Catalog / 文档维护 / 恢复调度完成启动步骤后才 server_activate；
    recovery 调度不等于所有旧 Turn 已恢复完毕。ThreadingHTTPServer 为请求分配线程，
    Session 隔离与同会话串行执行分别由 scope 和 Runtime guard 负责。
    finally 先停止提交后接单并有界等待，再停止文档维护和监听；未结算任务留待重启恢复。
    """

    if host not in {"127.0.0.1", "localhost"}:
        raise service.ApiError(
            "UNSAFE_BIND_HOST",
            "UI API 只允许绑定 127.0.0.1/localhost",
            status=400,
            details={"host": host},
        )
    # Reserve the address before rotating the shared token.  Concurrent sidecars
    # must lose here, not after overwriting the credential used by the winner.
    httpd = ThreadingHTTPServer((host, port), ApiHandler, bind_and_activate=False)
    document_maintenance = None
    post_commit_lifecycle = None
    try:
        httpd.server_bind()
        try:
            api_security.initialize_api_token(api_token)
        except ValueError as exc:
            raise service.ApiError("INVALID_API_TOKEN", str(exc), status=400) from exc
        _bootstrap_tool_catalog()
        document_maintenance = _build_document_maintenance_lifecycle()
        document_maintenance.start()
        post_commit_lifecycle = _build_turn_post_commit_lifecycle()
        post_commit_lifecycle.start()
        _recover_active_l1_turns()
        # 启动材料已就绪，后台恢复可继续并发运行；从此开始接收客户端连接。
        httpd.server_activate()
        print(f"Entelecheia API listening on http://{host}:{port}")
        httpd.serve_forever()
    finally:
        try:
            if post_commit_lifecycle is not None:
                stopped = post_commit_lifecycle.stop(timeout_seconds=POST_COMMIT_STOP_TIMEOUT_SECONDS)
                if not stopped:
                    import logging

                    logging.getLogger(__name__).warning(
                        "post-commit shutdown deadline reached; durable work remains for restart"
                    )
        finally:
            try:
                if document_maintenance is not None:
                    document_maintenance.stop(
                        timeout_seconds=DOCUMENT_MAINTENANCE_STOP_TIMEOUT_SECONDS
                    )
            finally:
                httpd.server_close()


def main(argv: list[str] | None = None) -> None:
    """CLI 启动入口：加载本机配置，解析监听参数，再交给 run 管理 Host 生命周期。

    pyproject.toml 的 entelecheia-api 指向这里；导入 server 模块本身不启动监听，
    load_dotenv 也只在显式启动时运行。
    """

    from ..configuration.paths import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the Entelecheia UI API server.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--api-token", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    run(args.host, args.port, api_token=args.api_token)


if __name__ == "__main__":
    main()
