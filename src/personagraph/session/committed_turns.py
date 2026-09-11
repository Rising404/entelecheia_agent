"""Session 已提交 Turn 对的窄只读查询辅助函数。"""

from __future__ import annotations

from collections.abc import Mapping


def latest_committed_turn_pair(
    store: object,
    session_id: str,
) -> Mapping[str, object] | None:
    """返回最近的完整 Turn 对，并验证检索截止点所需字段。"""

    pairs = getattr(store, "list_committed_turn_pairs")(
        session_id,
        limit=1,
    )
    if not pairs:
        return None
    pair = pairs[-1]
    if not isinstance(pair, Mapping) or "assistant_turn_idx" not in pair:
        raise ValueError("committed Turn cutoff authority is unavailable")
    return pair
