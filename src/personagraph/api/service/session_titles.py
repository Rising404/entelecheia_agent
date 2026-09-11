"""Session 首轮显示标题：读取已提交对话、独立生成、条件更新，不参与 Turn 结算。"""

from __future__ import annotations

import json
import re
import logging
from concurrent.futures import Future
from threading import Lock, Thread
from typing import Any

from ...configuration import paths
from ...session import store as session_store

LOGGER = logging.getLogger(__name__)
_PENDING: dict[tuple[str, str, str], Future] = {}
_PENDING_LOCK = Lock()


def schedule_session_title(session_id: str) -> Future | None:
    """首轮提交后启动非阻塞命名；同一进程的并发请求共享生成结果。"""
    try:
        key = (str(paths.STATE_DIR), str(session_store.DB_PATH), session_id)
        with _PENDING_LOCK:
            if key in _PENDING:
                return _PENDING[key]
            future = Future()
            _PENDING[key] = future
        try:
            Thread(
                target=_run_title_job, args=(key, future),
                name=f"session-title-{session_id[:24]}", daemon=True,
            ).start()
        except Exception:
            with _PENDING_LOCK:
                _PENDING.pop(key, None)
            raise
        return future
    except Exception:
        LOGGER.warning("could not schedule session title", exc_info=True)
        return None


def _run_title_job(key: tuple[str, str, str], future: Future) -> None:
    try:
        with session_store.session_database_scope(key[2]):
            future.set_result(generate_session_title(key[2]))
    except Exception:
        LOGGER.warning("session title worker failed", exc_info=True)
        future.set_result({"session_id": key[2], "title": None})
    finally:
        with _PENDING_LOCK:
            _PENDING.pop(key, None)


def name_session_from_turn(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """前端等待服务端命名结果；不信任客户端提交的对话文本。"""
    future = schedule_session_title(session_id)
    return future.result() if future is not None else {"session_id": session_id, "title": None}


def generate_session_title(session_id: str) -> dict[str, Any]:
    session = session_store.get_session(session_id)
    if session is None:
        return {"session_id": session_id, "title": None}
    original = session.get("title")
    result = {"session_id": session_id, "title": original}
    # 仅完整的首次正式对话可触发，失败/未提交输入不能成为命名依据。
    state = session_store.get_committed_turn_pair_index_state(session_id)
    if state.pair_count != 1:
        return result
    pairs = session_store.list_committed_turn_pairs(session_id, limit=1)
    if not pairs:
        return result
    question = str(pairs[0].get("user_content") or "").strip()
    answer = str(pairs[0].get("assistant_content") or "").strip()
    fallback = " ".join(question.split())[:24]
    if not fallback or original not in (None, "", "新会话", fallback):
        return result
    title = ""
    for _ in range(2):
        title = _proposed_title(question, answer, fallback)
        if title:
            break
    if title:
        session_store.replace_session_title(session_id, expected=original, title=title[:60])
    current = session_store.get_session(session_id)
    return {"session_id": session_id, "title": current.get("title") if current else None}


def _bare_title(text: str) -> str:
    """把"没包 JSON 的裸标题"当标题收下，但只在它确实像个标题的时候。

    放宽的边界要窄：多行、过长、或者带 JSON/围栏残留的，都更可能是模型在解释或者
    出错，拿去当标题会比退回兜底更难看。
    """

    content = (text or "").strip().strip("`").strip()
    if not content or "\n" in content or len(content) > 40:
        return ""
    if any(ch in content for ch in "{}[]"):
        return ""
    return content


def _model_json_object(text: str) -> dict[str, Any] | None:
    """从模型回复里取出一个 JSON object。

    模型很常见地把 JSON 包在 ```json 围栏里。原来这里只做 ``strip("`")``，剥掉反引号
    之后还剩一个 ``json`` 前缀，``json.loads`` 直接抛异常 —— 于是每次都落到兜底，
    而兜底正是"把用户原话截断"。表面现象就是"自动起名根本没生效"，但日志里看到的
    只是一次解析失败，很难联想到标题上去。
    """

    content = (text or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content).strip()
    try:
        parsed = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        # 模型有时会在 JSON 前后加一句话。取第一个完整的花括号块。
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _proposed_title(question: str, answer: str, fallback: str) -> str:
    from personagraph.model_io.gateway import complete_structured

    try:
        result = complete_structured(
            "给一段对话起一个短标题，用于会话列表。"
            "只输出 JSON：{\"title\":\"...\"}。"
            "标题不超过 16 个字，直接说这次对话要解决什么，"
            "不要引号、不要标点结尾、不要「关于」「讨论」这类填充词。",
            f"用户：{question[:600]}\n\n助手：{answer[:600]}",
            mock_payload={"title": fallback},
            max_tokens=1200,
            json_mode=True,
            purpose="session_autoname",
        )
        title = str((_model_json_object(result.reply) or {}).get("title") or "").strip()
        if not title:
            # 提示词要求只输出 JSON，但模型有时就直接把标题甩回来
            title = _bare_title(result.reply)
        if not title:
            LOGGER.warning(
                "session autoname got a reply it could not read: %r", (result.reply or "")[:160]
            )
    except Exception:
        # 起名失败不该让这一轮看起来出了问题：退回截断的第一句，
        # 它至少还是这次对话说过的话。但要留下痕迹
        LOGGER.warning("session autoname could not name this turn", exc_info=True)
        return ""
    return title
