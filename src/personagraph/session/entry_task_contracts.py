"""Entry 所需的中性 Task 读取契约。

这些值只描述 Host 已验证后可交给 Entry 的最小投影。它们不携带 TaskGraph、
WorkRun 或 AuxiliaryGraph 的执行契约，因此 Session 的普通读取路径无需初始化 L2。
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..output_protocol import AcceptanceUpdate


EntryTaskStatus = Literal[
    "proposed",
    "active",
    "awaiting_user",
    "waiting_external",
    "interrupted",
    "blocked",
    "cancelled",
    "completed",
]


class _EntryTaskContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EntryTaskCatalogItem(_EntryTaskContract):
    """可安全放入 Entry 分类上下文的一项根 Task 事实。"""

    insession_task_id: str = Field(min_length=1, max_length=128)
    goal_summary: str = Field(min_length=1, max_length=2_241)
    status: EntryTaskStatus
    current_graph_revision: int | None = Field(default=None, ge=1)
    pending_user_question: str | None = Field(
        default=None,
        min_length=1,
        max_length=2_000,
    )


class EntryPendingTaskQuestion(_EntryTaskContract):
    """Entry 路由只需知道问题正文及其根 Task 归属。"""

    insession_task_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=2_000)


def parse_entry_pending_question_decision_json(value: str) -> str:
    """严格解析 Host 接受的 ``request_user_input`` 决策并返回问题正文。

    该 helper 是 L2-free 的窄契约边界，供 Entry 投影和 transcript formal-content
    解析复用。它拒绝 duplicate key、非标准数值、额外字段和 Pydantic coercion；
    不承担其他 Attempt action 的解析职责。
    """

    if not isinstance(value, str) or not value:
        raise ValueError("pending-question decision JSON is missing")
    try:
        decision = json.loads(
            value,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("pending-question decision JSON is corrupt") from exc
    if not isinstance(decision, dict) or set(decision) != {
        "acceptance_updates",
        "action",
    }:
        raise ValueError("pending-question decision has an invalid shape")
    updates = decision["acceptance_updates"]
    if not isinstance(updates, list):
        raise ValueError("pending-question acceptance updates are invalid")
    for raw_update in updates:
        if (
            not isinstance(raw_update, dict)
            or not isinstance(raw_update.get("model_claimed_satisfied"), bool)
            or not isinstance(raw_update.get("supporting_tool_result_ids", []), list)
        ):
            raise ValueError("pending-question acceptance update is not canonical")
        try:
            update = AcceptanceUpdate.model_validate(raw_update)
        except ValueError as exc:
            raise ValueError(
                "pending-question acceptance update is not canonical"
            ) from exc
        if json.dumps(
            update.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ) != json.dumps(
            raw_update,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ):
            raise ValueError("pending-question acceptance update is not canonical")
    action = decision["action"]
    if (
        not isinstance(action, dict)
        or set(action) != {"kind", "question"}
        or action["kind"] != "request_user_input"
    ):
        raise ValueError("pending-question action has an invalid shape")
    question = action["question"]
    if not isinstance(question, str) or not question or len(question) > 2_000:
        raise ValueError("pending-question content is invalid")
    return question


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON constant: {value}")


__all__ = [
    'EntryPendingTaskQuestion',
    "EntryTaskCatalogItem",
    'EntryTaskStatus',
    "parse_entry_pending_question_decision_json",
]
