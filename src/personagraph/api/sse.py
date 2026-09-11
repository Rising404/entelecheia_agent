"""请求局部的 SSE 交付状态；连接断开与 Runtime 生命周期保持分离。"""

from __future__ import annotations

import errno
from collections.abc import Callable
from typing import Any


_DISCONNECT_ERRNOS = {
    errno.ECONNABORTED,
    errno.ECONNRESET,
    errno.ENOTCONN,
    errno.EPIPE,
}


def is_client_disconnect(exc: BaseException) -> bool:
    """返回 SSE 写入是否因客户端离开而失败。"""
    if isinstance(exc, (BrokenPipeError, ConnectionAbortedError, ConnectionResetError)):
        return True
    return isinstance(exc, OSError) and exc.errno in _DISCONNECT_ERRNOS


class SseDelivery:
    """与 Runtime 执行隔离、请求局部的尽力 SSE 交付。

    peer 断开后，后续发送变为 no-op。非连接错误仍会传播，确保序列化和应用缺陷绝不会被
    误认为普通传输丢失。
    """

    def __init__(self, send: Callable[[str, dict[str, Any]], None]) -> None:
        self._send = send
        self._connected = True

    @property
    def connected(self) -> bool:
        return self._connected

    def send(self, event: str, payload: dict[str, Any]) -> bool:
        """尽力发送事件；识别到 peer 断连后记住 disconnected，后续发送直接返回 False。

        这里只停止传输，既不取消模型/工具，也不回滚已接受 Turn。非断连类 OSError
        继续传播，避免把序列化或服务缺陷当作用户正常关闭页面。
        """

        if not self._connected:
            return False
        try:
            self._send(event, payload)
        except OSError as exc:
            if not is_client_disconnect(exc):
                raise
            self._connected = False
            return False
        return True
