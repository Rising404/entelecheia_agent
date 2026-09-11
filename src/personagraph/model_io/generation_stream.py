"""用户可见模型生成与请求局部响应流的协调。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from .response_stream import (
    bind_response_stream,
    emit_response_stream_event,
    response_stream_bound,
)


_STREAM_GENERATION: ContextVar["_StreamGeneration | None"] = ContextVar(
    "personagraph_model_stream_generation", default=None
)
_TOOL_CALL_KEY = '"tool_calls"'


@dataclass
class _StreamGeneration:
    """图 Turn 中的一个用户可见候选回复。

    工具调用采用纯 JSON 协议。在确认模型正在生成普通文本前暂存有歧义的前缀，避免图在
    决定下一节点时，将完整工具命令短暂显示在聊天记录中。
    """

    generation_id: str
    visible: bool
    buffered: str = ""
    released: bool = False
    tool_call_candidate: bool = False
    finished: bool = False

    def emit_delta(self, text: str) -> None:
        if not self.visible or self.finished or not text:
            return
        if self.released:
            _emit_stream_event("delta", generation_id=self.generation_id, text=text)
            return
        self.buffered += text
        if self.tool_call_candidate or _looks_like_tool_call_prefix(self.buffered):
            self.tool_call_candidate = True
            return
        if not self.released:
            self.released = True
            _emit_stream_event("delta", generation_id=self.generation_id, text=self.buffered)
            self.buffered = ""
            return

    def complete(self) -> None:
        if not self.visible or self.finished:
            return
        if self.buffered:
            _emit_stream_event("delta", generation_id=self.generation_id, text=self.buffered)
            self.buffered = ""
        self.finished = True
        _emit_stream_event("answer_complete", generation_id=self.generation_id)

    def discard(self) -> None:
        if not self.visible or self.finished:
            return
        self.buffered = ""
        self.finished = True
        _emit_stream_event("answer_discard", generation_id=self.generation_id)


@contextmanager
def stream_events(callback: Callable[[dict[str, Any]], None] | None) -> Iterator[None]:
    """绑定 API 请求局部的 SSE 接收器，且不把回调序列化进图状态。"""
    with bind_response_stream(callback):
        yield


@contextmanager
def stream_generation(generation_id: str, *, visible: bool = True) -> Iterator[None]:
    """将供应商增量限定在一个图生成候选中。"""
    enabled = visible and response_stream_bound()
    generation = _StreamGeneration(generation_id=generation_id, visible=enabled)
    token = _STREAM_GENERATION.set(generation)
    if enabled:
        _emit_stream_event("answer_start", generation_id=generation_id)
    try:
        yield
    except BaseException:
        generation.discard()
        raise
    finally:
        _STREAM_GENERATION.reset(token)


def complete_stream_generation(*, discard: bool = False) -> None:
    generation = _STREAM_GENERATION.get()
    if generation is None:
        return
    if discard:
        generation.discard()
    else:
        generation.complete()


def streaming_generation_active() -> bool:
    generation = _STREAM_GENERATION.get()
    return generation is not None and generation.visible


def emit_stream_delta(text: str) -> None:
    generation = _STREAM_GENERATION.get()
    if generation is not None:
        generation.emit_delta(text)


def _emit_stream_event(event: str, **payload: Any) -> None:
    emit_response_stream_event(event, **payload)


def _looks_like_tool_call_prefix(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return False
    # provider 可以在一条 SSE delta 中给出不完整前缀，也可以给出完整 JSON 控制对象。
    # 接受左花括号后的 JSON 空白，但只暂存 `tool_calls` 键；普通 JSON 响应仍可立即显示。
    key_prefix = stripped[1:].lstrip()
    return not key_prefix or (
        _TOOL_CALL_KEY.startswith(key_prefix)
        or key_prefix.startswith(_TOOL_CALL_KEY)
    )
