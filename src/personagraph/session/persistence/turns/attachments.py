"""聊天附件及其与 Turn 绑定关系的持久化。"""

from __future__ import annotations

from typing import Any

from ..deps import StoreDeps


_COLUMNS = (
    "attachment_id, session_id, turn_id, origin, original_name, stored_rel_path, "
    "media_type, declared_media_type, size_bytes, content_hash, kind, "
    "project_id, file_id, file_version_id, created_at, bound_at"
)
_QUALIFIED_ATTACHMENT_COLUMNS = ", ".join(
    f"attachment.{column.strip()}" for column in _COLUMNS.split(",")
)
_TURN_ATTACHMENT_READ_MODEL_COLUMNS = (
    f"{_QUALIFIED_ATTACHMENT_COLUMNS}, "
    "binding.binding_ordinal AS binding_ordinal, "
    "turn_input.message_id AS input_message_id, "
    "input_file.ordinal AS input_file_ordinal, "
    "input_file.project_id AS input_project_id, "
    "input_file.file_id AS input_file_id, "
    "input_file.file_version_id AS input_file_version_id"
)


class AttachmentBindingError(ValueError):
    """一个或多个附件 ID 无法绑定到请求的 Turn。"""

    def __init__(self, reason: str, attachment_ids: tuple[str, ...]) -> None:
        super().__init__(f"{reason}: {', '.join(attachment_ids)}")
        self.reason = reason
        self.attachment_ids = attachment_ids


def create_attachment(
    deps: StoreDeps,
    *,
    attachment_id: str,
    session_id: str,
    origin: str,
    original_name: str,
    stored_rel_path: str,
    media_type: str,
    declared_media_type: str | None,
    size_bytes: int,
    content_hash: str,
    kind: str,
    project_id: str | None = None,
    file_id: str | None = None,
    file_version_id: str | None = None,
) -> dict[str, Any]:
    """记录一个已存储上传，它尚未附加到任何 Turn。"""

    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        if conn.execute("SELECT 1 FROM sessions WHERE id=?", (session_id,)).fetchone() is None:
            raise ValueError(f"unknown session: {session_id}")
        conn.execute(
            f"INSERT INTO session_attachments ({_COLUMNS}) "
            "VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)",
            (attachment_id, session_id, origin, original_name, stored_rel_path,
             media_type, declared_media_type, size_bytes, content_hash, kind,
             project_id, file_id, file_version_id, now),
        )
        row = conn.execute(
            f"SELECT {_COLUMNS} FROM session_attachments WHERE attachment_id=?",
            (attachment_id,),
        ).fetchone()
    return dict(row)


def get_attachment(deps: StoreDeps, attachment_id: str) -> dict[str, Any] | None:
    deps.init_db()
    with deps.connect() as conn:
        row = conn.execute(
            f"SELECT {_COLUMNS} FROM session_attachments WHERE attachment_id=?",
            (attachment_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def list_turn_attachments(deps: StoreDeps, session_id: str, turn_id: str) -> list[dict[str, Any]]:
    """按 Turn 绑定顺序列出附件；旧记录无 ordinal 时回退到上传顺序。"""

    deps.init_db()
    with deps.connect() as conn:
        rows = conn.execute(
            f"SELECT {_TURN_ATTACHMENT_READ_MODEL_COLUMNS} "
            "FROM session_attachments AS attachment "
            "LEFT JOIN runtime_turn_attachment_bindings AS binding "
            "ON binding.turn_id=attachment.turn_id "
            "AND binding.attachment_id=attachment.attachment_id "
            "LEFT JOIN runtime_turn_inputs AS turn_input "
            "ON turn_input.turn_id=attachment.turn_id "
            "AND turn_input.session_id=attachment.session_id "
            "LEFT JOIN runtime_turn_input_file_refs AS input_file "
            "ON input_file.message_id=turn_input.message_id "
            "AND input_file.ordinal=binding.binding_ordinal "
            "WHERE attachment.session_id=? AND attachment.turn_id=? "
            "ORDER BY CASE WHEN binding.binding_ordinal IS NULL THEN 1 ELSE 0 END, "
            "binding.binding_ordinal, attachment.created_at, attachment.attachment_id",
            (session_id, turn_id),
        ).fetchall()
    return [dict(row) for row in rows]


def list_unbound_attachments(deps: StoreDeps, session_id: str) -> list[dict[str, Any]]:
    """列出用户尚未发送的上传。

    这些上传会被保留而非清扫：用户上传后离开页面再返回时，文件仍应等待发送。配额与 GC
    属于另一项决策（061 §8.2）。
    """

    deps.init_db()
    with deps.connect() as conn:
        rows = conn.execute(
            f"SELECT {_COLUMNS} FROM session_attachments "
            "WHERE session_id=? AND turn_id IS NULL ORDER BY created_at, attachment_id",
            (session_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def bind_attachments_to_turn(
    deps: StoreDeps,
    *,
    session_id: str,
    turn_id: str,
    attachment_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    """把先前上传的文件附加到一个 Turn，操作全部成功或全部失败。

    每个 ID 都必须属于当前 Session 且仍未绑定。部分绑定会被拒绝，而不会静默丢弃用户
    明确添加的附件；与其在不告知用户的情况下忽略附件，不如让整个 Turn 明确失败。
    """

    if not attachment_ids:
        return []
    if len(set(attachment_ids)) != len(attachment_ids):
        raise AttachmentBindingError("duplicate attachment ids", attachment_ids)

    deps.init_db()
    now = deps.now()
    with deps.connect() as conn:
        return bind_attachments_to_turn_in_transaction(
            conn,
            session_id=session_id,
            turn_id=turn_id,
            attachment_ids=attachment_ids,
            bound_at=now,
        )


def bind_attachments_to_turn_in_transaction(
    conn,
    *,
    session_id: str,
    turn_id: str,
    attachment_ids: tuple[str, ...],
    bound_at: str,
) -> list[dict[str, Any]]:
    """使用调用方现有 SQLite 事务绑定附件。

    Turn 接受事务必须同时公开受信输入、附件绑定和执行 Window。把校验留在此处，可避免
    组合事务重复实现附件所有权检查，或意外偏离独立绑定路径。
    """

    if not attachment_ids:
        return []
    if len(set(attachment_ids)) != len(attachment_ids):
        raise AttachmentBindingError("duplicate attachment ids", attachment_ids)
    placeholders = ",".join("?" for _ in attachment_ids)
    rows = conn.execute(
        f"SELECT attachment_id, session_id, turn_id FROM session_attachments "
        f"WHERE attachment_id IN ({placeholders})",
        attachment_ids,
    ).fetchall()
    found = {str(row["attachment_id"]): row for row in rows}

    missing = tuple(item for item in attachment_ids if item not in found)
    if missing:
        raise AttachmentBindingError("unknown attachment", missing)
    foreign = tuple(
        item for item in attachment_ids if str(found[item]["session_id"]) != session_id
    )
    if foreign:
        raise AttachmentBindingError("attachment belongs to another session", foreign)
    already = tuple(item for item in attachment_ids if found[item]["turn_id"] is not None)
    if already:
        raise AttachmentBindingError("attachment already bound to a turn", already)

    conn.execute(
        f"UPDATE session_attachments SET turn_id=?, bound_at=? "
        f"WHERE attachment_id IN ({placeholders})",
        (turn_id, bound_at, *attachment_ids),
    )
    bound = conn.execute(
        f"SELECT {_COLUMNS} FROM session_attachments "
        f"WHERE attachment_id IN ({placeholders}) ORDER BY created_at, attachment_id",
        attachment_ids,
    ).fetchall()
    return [dict(row) for row in bound]


def delete_unbound_attachment(deps: StoreDeps, *, session_id: str, attachment_id: str) -> dict[str, Any] | None:
    """丢弃用户暂存后又在发送前移除的上传。

    只有未绑定行可以删除。附件一旦属于已提交 Turn，就成为对话记录的一部分；删除它会让
    对话引用已经不存在的材料。
    """

    deps.init_db()
    with deps.connect() as conn:
        row = conn.execute(
            f"SELECT {_COLUMNS} FROM session_attachments "
            "WHERE attachment_id=? AND session_id=? AND turn_id IS NULL",
            (attachment_id, session_id),
        ).fetchone()
        if row is None:
            return None
        conn.execute("DELETE FROM session_attachments WHERE attachment_id=?", (attachment_id,))
    return dict(row)


def session_attachment_total_bytes(deps: StoreDeps, session_id: str) -> int:
    deps.init_db()
    with deps.connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(size_bytes), 0) AS total FROM session_attachments WHERE session_id=?",
            (session_id,),
        ).fetchone()
    return int(row["total"])
