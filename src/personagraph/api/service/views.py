from __future__ import annotations

from typing import Any, Protocol, get_args

from personagraph.model_io.gateway import ModelGatewayError
from ...runtime.turn.contracts import EntryTurnResult
from ..runtime_outcome import build_error_outcome
from ...session.entry_task_contracts import EntryTaskStatus
from .errors import ApiError


_INSESSION_TASK_STATUSES = frozenset(get_args(EntryTaskStatus))


class _InSessionTaskDetailsView(Protocol):
    """HTTP projection consumes shape, not the L2 TaskGraph implementation."""

    insession_task_id: str
    title: str
    status: object
    current_graph_revision: int | None
    nodes: tuple[dict[str, object], ...]
    related_turn_count: int


def session_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "title": row.get("title"),
        "status": row.get("status"),
        "folder_id": row.get("folder_id"),
        "working_dir": row.get("working_dir"),
        "created_at": row.get("created_at"),
        "last_active_at": row.get("last_active_at"),
    }


def turn_view(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_idx": row.get("turn_idx"),
        "role": row.get("role"),
        "content": row.get("content"),
        "created_at": row.get("created_at"),
    }


def pending_user_question_view(value: Any) -> dict[str, Any]:
    """只公开交互，绝不公开 WorkRun/Attempt 恢复 authority。"""

    task_id = getattr(value, "insession_task_id", None)
    question = getattr(value, "question", None)
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("pending user question has no Task identity")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("pending user question has no content")
    return {
        "insession_task_id": task_id,
        "question": question,
    }


def insession_task_detail_view(details: _InSessionTaskDetailsView) -> dict[str, Any]:
    """为聊天 UI 投影一个 Session 内任务，不包含 audit 内部细节。

    Store 级详情包含源 anchor、授权事实、约束和节点所有验收标准，因为它同时也是持久控制面
    读取模型。紧凑的用户任务卡片不需要这些内容，在此公开会意外把读取端点变成证据 API。
    应刻意缩小 HTTP 投影，并在序列化前验证无类型节点字典。
    """

    nodes = tuple(_insession_task_node_view(node) for node in details.nodes)
    if details.current_graph_revision is None:
        if nodes:
            raise ValueError("in-session task shell cannot contain graph nodes")
    else:
        root_node = _validated_insession_task_root(nodes, details.insession_task_id)
        if root_node["title"] != details.title:
            raise ValueError("in-session task root title does not match its task")

    status = getattr(details.status, "value", details.status)
    if not isinstance(status, str) or status not in _INSESSION_TASK_STATUSES:
        raise ValueError("in-session task detail contains an invalid status")
    return {
        "insession_task_id": details.insession_task_id,
        "title": details.title,
        "status": status,
        "current_graph_revision": details.current_graph_revision,
        "nodes": list(nodes),
        "related_turn_count": details.related_turn_count,
    }


def _insession_task_node_view(node: dict[str, object]) -> dict[str, Any]:
    """在一个 Store 节点跨越 API 边界前验证并最小化它。"""

    node_id = node.get("insession_task_node_id")
    parent_node_id = node.get("parent_insession_task_node_id")
    node_kind = node.get("node_kind")
    title = node.get("title")
    status = node.get("status")
    ordinal = node.get("ordinal")
    if (
        not isinstance(node_id, str)
        or not node_id
        or (
            parent_node_id is not None
            and (not isinstance(parent_node_id, str) or not parent_node_id)
        )
        or node_kind not in {"root", "subtask"}
        or not isinstance(title, str)
        or not title.strip()
        or not isinstance(status, str)
        or status not in _INSESSION_TASK_STATUSES
        or not isinstance(ordinal, int)
        or isinstance(ordinal, bool)
        or ordinal < 0
    ):
        raise ValueError("in-session task detail contains an invalid node projection")
    return {
        "insession_task_node_id": node_id,
        "parent_insession_task_node_id": parent_node_id,
        "node_kind": node_kind,
        "ordinal": ordinal,
        "title": title,
        "status": status,
    }


def _validated_insession_task_root(
    nodes: tuple[dict[str, Any], ...],
    insession_task_id: str,
) -> dict[str, Any]:
    """确认小型 API 投影仍表示一棵有根树。"""

    nodes_by_id = {node["insession_task_node_id"]: node for node in nodes}
    if len(nodes_by_id) != len(nodes):
        raise ValueError("in-session task detail has duplicate node identifiers")
    root_nodes = [node for node in nodes if node["node_kind"] == "root"]
    if len(root_nodes) != 1:
        raise ValueError("in-session task detail must contain exactly one root node")
    root_node = root_nodes[0]
    if (
        root_node["insession_task_node_id"] != insession_task_id
        or root_node["parent_insession_task_node_id"] is not None
    ):
        raise ValueError("in-session task detail root does not match its task")

    for node in nodes:
        node_id = node["insession_task_node_id"]
        parent_node_id = node["parent_insession_task_node_id"]
        if node["node_kind"] == "root":
            continue
        if parent_node_id not in nodes_by_id or parent_node_id == node_id:
            raise ValueError("in-session task detail contains an invalid parent reference")
        visited = {node_id}
        cursor = parent_node_id
        while cursor != insession_task_id:
            if cursor in visited:
                raise ValueError("in-session task detail contains a parent cycle")
            visited.add(cursor)
            parent = nodes_by_id.get(cursor)
            if parent is None:
                raise ValueError("in-session task detail contains an unreachable node")
            cursor = parent["parent_insession_task_node_id"]
            if cursor is None:
                raise ValueError("in-session task detail contains a second root")
    return root_node

def model_api_error(exc: ModelGatewayError) -> ApiError:
    """将抛出的 ModelGatewayError 转成公开 API 错误、HTTP 状态和 outcome。

    Runtime 已正常返回的 incomplete EntryTurnResult 不经过这里；它仍由
    turn_result_view 投影。区分“API 调用抛错”与“已接受 Turn 中断”，避免混淆重试入口。
    """

    details = dict(exc.details)
    outcome = build_error_outcome(
        code=exc.code,
        domain="model",
        message=exc.message,
        retryable=exc.retryable,
        details=details,
    ).to_dict()
    status = model_error_http_status(exc)
    return ApiError(
        exc.code,
        model_error_message(exc.code),
        status=status,
        details=details,
        outcome=outcome,
    )

def model_error_http_status(exc: ModelGatewayError) -> int:
    if exc.code == "MODEL_CALL_TIMEOUT":
        return 504
    if exc.code == "MODEL_BAD_RESPONSE":
        return 502
    status_code = exc.details.get("status_code")
    if isinstance(status_code, int):
        if status_code in {401, 403}:
            return 502
        if status_code == 429 or status_code >= 500:
            return 503
        return 502
    if not exc.retryable and exc.details.get("physical_error_retryable") is not True:
        return 400
    return 503

def model_error_message(code: str) -> str:
    if code == "MODEL_CALL_TIMEOUT":
        return "模型调用超时，可在确认网络后重试本轮"
    if code == "MODEL_BAD_RESPONSE":
        return "模型返回结构异常"
    return "模型调用失败"

def turn_result_view(result: EntryTurnResult) -> dict[str, Any]:
    """将 EntryTurnResult 转成 JSON / SSE final 共用的公开结果，不重新判断执行是否成功。

    status 表达 Turn 结果，window_state 表达当前执行槽位；completed 与
    post_commit_pending 可以同时成立。reply 来自 Entry 的权威投影，不能在这里
    用候选或最后一条模型文本补出一份正式回答。
    """

    return {
        "session_id": result.session_id,
        "turn_id": result.turn_id,
        "status": result.status,
        "processing_level": result.processing_level,
        "reply": result.reply,
        "end_reason": result.end_reason,
        "error_code": result.error_code,
        "related_insession_task_ids": list(result.related_insession_task_ids),
        "work_run_ids": list(result.work_run_ids),
        "window_state": result.window_state,
        "window_revision": result.window_revision,
        "available_controls": list(result.available_controls),
        "delivery": result.delivery,
        "pending_decision": result.pending_decision,
    }
