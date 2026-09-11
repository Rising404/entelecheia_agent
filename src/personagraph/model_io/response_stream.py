"""临时响应流事件的请求局部交付通道。

本模块只持有 ``answer_start`` 和 ``delta`` 等响应片段所用的进程内回调绑定。
持久且不含内容的 Runtime 生命周期事实属于 :mod:`personagraph.runtime.turn_events`。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator


ResponseStreamSink = Callable[[dict[str, Any]], None]
_RESPONSE_STREAM_SINK: ContextVar[ResponseStreamSink | None] = ContextVar(
    "personagraph_response_stream_sink",
    default=None,
)


@contextmanager
def bind_response_stream(callback: ResponseStreamSink | None) -> Iterator[None]:
    """在上下文持续期间绑定一个请求局部响应接收器。"""

    token = _RESPONSE_STREAM_SINK.set(callback)
    try:
        yield
    finally:
        _RESPONSE_STREAM_SINK.reset(token)


def emit_response_stream_event(event: str, **payload: Any) -> None:
    """绑定请求接收器时交付一个临时响应事件。"""

    callback = _RESPONSE_STREAM_SINK.get()
    if callback is not None:
        callback({"event": event, **payload})


def response_stream_bound() -> bool:
    """返回当前上下文是否具有响应流接收器。"""

    return _RESPONSE_STREAM_SINK.get() is not None
