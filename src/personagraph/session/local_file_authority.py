"""一个 Session 的持久、仅限本地文件读写权威信息。

本模块只判断 Session 对本地路径的访问范围，不判断内容是否构成证据。
外发默认同意由 tools/policy 统一决定，不在本地读取账本中另设批准。

本地权威信息有两个来源：

* 当前 Project 已登记的用户上传文件和 Agent 产物可隐式读取；
* 用户可以一次性批准 Session 已配置的工作根目录。该授予会持久化，
  并绑定到根目录的规范路径、设备及 inode。

后一种绑定使配置根目录变更、同路径目录被替换或跨越符号链接边界时采用失败关闭策略。
调用方仍须在 I/O 前立即应用普通路径边界；此账本是授权事实，而非文件描述符能力，
自身无法消除文件系统检查时与使用时竞争。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from ..workspace.files import attachments as attachment_storage


_BINDING_SCHEMA_VERSION = 1


class LocalFileAction(StrEnum):
    """此权威信息可以显式授予的本地工作区操作。"""

    READ = "read"
    SEARCH = "search"
    UPDATE = "update"


class LocalFileAuthorizationReason(StrEnum):
    """作出一个仅限本地授权决策的原因。"""

    IMPLICIT_SESSION_INPUT = "implicit_session_input"
    IMPLICIT_SESSION_OUTPUT = "implicit_session_output"
    EXPLICIT_WORKING_ROOT_GRANT = "explicit_working_root_grant"
    EXPLICIT_WORKING_ROOT_WRITE_GRANT = "explicit_working_root_write_grant"
    AUTHORIZATION_REQUIRED = "authorization_required"
    INVALID_SESSION = "invalid_session"
    PATH_NOT_FOUND = "path_not_found"
    WORKING_ROOT_UNAVAILABLE = "working_root_unavailable"
    PATH_OUTSIDE_AUTHORIZED_SCOPE = "path_outside_authorized_scope"


_READ_ALLOWED_REASONS = frozenset(
    {
        LocalFileAuthorizationReason.IMPLICIT_SESSION_INPUT,
        LocalFileAuthorizationReason.IMPLICIT_SESSION_OUTPUT,
        LocalFileAuthorizationReason.EXPLICIT_WORKING_ROOT_GRANT,
    }
)
_ALLOWED_REASONS = _READ_ALLOWED_REASONS | frozenset(
    {LocalFileAuthorizationReason.EXPLICIT_WORKING_ROOT_WRITE_GRANT}
)


@dataclass(frozen=True)
class LocalFileAuthorizationDecision:
    """针对一次本地工作区操作的决策，不混入证据或外发授权。"""

    allowed: bool
    action: LocalFileAction
    reason: LocalFileAuthorizationReason
    canonical_path: str | None = None
    grant_id: str | None = None
    storage_area: attachment_storage.SessionStorageArea | None = None

    def __post_init__(self) -> None:
        if self.allowed != (self.reason in _ALLOWED_REASONS):
            raise ValueError("decision allowed flag does not match its reason")
        if self.reason in {
            LocalFileAuthorizationReason.EXPLICIT_WORKING_ROOT_GRANT,
            LocalFileAuthorizationReason.EXPLICIT_WORKING_ROOT_WRITE_GRANT,
        }:
            if not self.grant_id or self.storage_area is not None:
                raise ValueError("an explicit decision requires only a grant receipt")
        elif self.allowed:
            if self.grant_id is not None or self.storage_area is None:
                raise ValueError("an implicit decision requires only a storage area")
        elif self.grant_id is not None or self.storage_area is not None:
            raise ValueError("a denied decision cannot carry authority receipts")

@dataclass(frozen=True)
class SessionWorkspaceReadGrant:
    """对一个 Session 精确工作目录对象的持久批准。"""

    grant_id: str
    session_id: str
    canonical_root: str
    root_device: int
    root_inode: int
    root_binding_sha256: str
    granted_at: datetime
    revoked_at: datetime | None = None

    @property
    def actions(self) -> tuple[LocalFileAction, LocalFileAction]:
        """单次批准始终覆盖两个非变更本地操作。"""

        return (LocalFileAction.READ, LocalFileAction.SEARCH)


@dataclass(frozen=True)
class SessionWorkspaceWriteGrant:
    """允许智能体在一个根目录下写入的显式可撤销批准。"""

    grant_id: str
    session_id: str
    canonical_root: str
    root_device: int
    root_inode: int
    root_binding_sha256: str
    granted_at: datetime
    revoked_at: datetime | None = None

    @property
    def actions(self) -> tuple[LocalFileAction]:
        return (LocalFileAction.UPDATE,)


@dataclass(frozen=True)
class _RootBinding:
    session_id: str
    canonical_root: str
    root_device: int
    root_inode: int
    root_binding_sha256: str


class SqliteSessionFileAuthority:
    """持久化并解析 Session 本地 READ/SEARCH 授予。

    ``working_root`` 必须是 Host 当前为该 Session 持久化的根目录。每次观测都会记录。
    观测到另一个根目录标识会撤销之前的活动授予，从而避免切换离开后，在已观测到变化的情况下
    又静默复用旧批准。
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._explicit_path = Path(path) if path is not None else None

    @property
    def path(self) -> Path:
        if self._explicit_path is not None:
            return self._explicit_path
        from . import store as session_store

        return session_store.current_session_database_path()

    def grant_workspace_root(
        self,
        *,
        session_id: str,
        working_root: Path | str,
    ) -> SessionWorkspaceReadGrant:
        """一次性批准当前根目录，并返回幂等回执。"""

        _require_session_id(session_id)
        binding = _resolve_root_binding(session_id=session_id, working_root=working_root)
        now = _now()
        with self._connect(session_id=session_id) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._synchronize_observed_root(connection, binding=binding, observed_at=now)
            row = connection.execute(
                """
                SELECT *
                  FROM session_workspace_read_grants
                 WHERE session_id=? AND root_binding_sha256=? AND revoked_at IS NULL
                """,
                (session_id, binding.root_binding_sha256),
            ).fetchone()
            if row is not None:
                return _row_to_grant(row)

            grant = SessionWorkspaceReadGrant(
                grant_id=f"sfrg_{uuid.uuid4().hex[:24]}",
                session_id=session_id,
                canonical_root=binding.canonical_root,
                root_device=binding.root_device,
                root_inode=binding.root_inode,
                root_binding_sha256=binding.root_binding_sha256,
                granted_at=now,
            )
            connection.execute(
                """
                INSERT INTO session_workspace_read_grants (
                    grant_id,
                    session_id,
                    canonical_root,
                    root_device,
                    root_inode,
                    root_binding_sha256,
                    granted_at,
                    revoked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    grant.grant_id,
                    grant.session_id,
                    grant.canonical_root,
                    str(grant.root_device),
                    str(grant.root_inode),
                    grant.root_binding_sha256,
                    grant.granted_at.isoformat(),
                ),
            )
            return grant

    def grant_workspace_write(
        self,
        *,
        session_id: str,
        working_root: Path | str,
    ) -> SessionWorkspaceWriteGrant:
        """显式批准智能体在当前根目录下的 UPDATE 效果。

        此操作特意与 :meth:`grant_workspace_root` 分离。选择目录会授予模型本地认知能力，
        绝不会静默授予修改用户文件的权限。
        """

        _require_session_id(session_id)
        binding = _resolve_root_binding(session_id=session_id, working_root=working_root)
        now = _now()
        with self._connect(session_id=session_id) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._synchronize_observed_root(connection, binding=binding, observed_at=now)
            row = connection.execute(
                """
                SELECT *
                  FROM session_workspace_write_grants
                 WHERE session_id=? AND root_binding_sha256=? AND revoked_at IS NULL
                """,
                (session_id, binding.root_binding_sha256),
            ).fetchone()
            if row is not None:
                return _row_to_write_grant(row)

            grant = SessionWorkspaceWriteGrant(
                grant_id=f"sfwg_{uuid.uuid4().hex[:24]}",
                session_id=session_id,
                canonical_root=binding.canonical_root,
                root_device=binding.root_device,
                root_inode=binding.root_inode,
                root_binding_sha256=binding.root_binding_sha256,
                granted_at=now,
            )
            connection.execute(
                """
                INSERT INTO session_workspace_write_grants (
                    grant_id,
                    session_id,
                    canonical_root,
                    root_device,
                    root_inode,
                    root_binding_sha256,
                    granted_at,
                    revoked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    grant.grant_id,
                    grant.session_id,
                    grant.canonical_root,
                    str(grant.root_device),
                    str(grant.root_inode),
                    grant.root_binding_sha256,
                    grant.granted_at.isoformat(),
                ),
            )
            return grant

    def authorize(
        self,
        *,
        session_id: str,
        candidate: Path | str,
        action: LocalFileAction | str,
        working_root: Path | str | None = None,
    ) -> LocalFileAuthorizationDecision:
        """解析一个现有路径的仅限本地权威信息。

        首先检查托管 Session 输入/输出，且无需配置工作根目录。所有其他路径都同时需要当前根目录
        和匹配且未撤销的授予。
        """

        normalized_action = LocalFileAction(action)
        if not _is_valid_session_id(session_id):
            return _denied(
                action=normalized_action,
                reason=LocalFileAuthorizationReason.INVALID_SESSION,
            )

        canonical_candidate = _resolve_existing(candidate)
        if canonical_candidate is None:
            return _denied(
                action=normalized_action,
                reason=LocalFileAuthorizationReason.PATH_NOT_FOUND,
            )

        if normalized_action is LocalFileAction.UPDATE:
            return self._authorize_workspace_write(
                session_id=session_id,
                candidate=canonical_candidate,
                working_root=working_root,
            )

        try:
            storage_area = attachment_storage.classify_session_path(
                session_id,
                canonical_candidate,
            )
        except ValueError:
            return _denied(
                action=normalized_action,
                reason=LocalFileAuthorizationReason.INVALID_SESSION,
                canonical_path=str(canonical_candidate),
            )
        # 上方 ``resolve(strict=True)`` 证明决策时路径存在；显式检查让该条件在此权威接缝中可见。
        if storage_area is not None and canonical_candidate.exists():
            reason = (
                LocalFileAuthorizationReason.IMPLICIT_SESSION_INPUT
                if storage_area is attachment_storage.SessionStorageArea.INPUT
                else LocalFileAuthorizationReason.IMPLICIT_SESSION_OUTPUT
            )
            return LocalFileAuthorizationDecision(
                allowed=True,
                action=normalized_action,
                reason=reason,
                canonical_path=str(canonical_candidate),
                storage_area=storage_area,
            )

        if working_root is None:
            return _denied(
                action=normalized_action,
                reason=LocalFileAuthorizationReason.AUTHORIZATION_REQUIRED,
                canonical_path=str(canonical_candidate),
            )
        try:
            binding = _resolve_root_binding(
                session_id=session_id,
                working_root=working_root,
            )
        except (OSError, ValueError):
            return _denied(
                action=normalized_action,
                reason=LocalFileAuthorizationReason.WORKING_ROOT_UNAVAILABLE,
                canonical_path=str(canonical_candidate),
            )

        canonical_root = Path(binding.canonical_root)
        if not canonical_candidate.is_relative_to(canonical_root):
        # 即使此特定路径位于新根目录之外，也要记录新观测到的根目录。配置根目录变化会使旧权威信息
        # 失效，而不取决于哪个候选项最先暴露该变化。
            self._observe_root(binding)
            return _denied(
                action=normalized_action,
                reason=LocalFileAuthorizationReason.PATH_OUTSIDE_AUTHORIZED_SCOPE,
                canonical_path=str(canonical_candidate),
            )

        now = _now()
        with self._connect(session_id=session_id) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._synchronize_observed_root(connection, binding=binding, observed_at=now)
            row = connection.execute(
                """
                SELECT grant_id
                  FROM session_workspace_read_grants
                 WHERE session_id=? AND root_binding_sha256=? AND revoked_at IS NULL
                """,
                (session_id, binding.root_binding_sha256),
            ).fetchone()
        if row is None:
            return _denied(
                action=normalized_action,
                reason=LocalFileAuthorizationReason.AUTHORIZATION_REQUIRED,
                canonical_path=str(canonical_candidate),
            )
        return LocalFileAuthorizationDecision(
            allowed=True,
            action=normalized_action,
            reason=LocalFileAuthorizationReason.EXPLICIT_WORKING_ROOT_GRANT,
            canonical_path=str(canonical_candidate),
            grant_id=str(row["grant_id"]),
        )

    def _authorize_workspace_write(
        self,
        *,
        session_id: str,
        candidate: Path,
        working_root: Path | str | None,
    ) -> LocalFileAuthorizationDecision:
        """解析 UPDATE 权威信息，但绝不隐式授予。"""

        action = LocalFileAction.UPDATE
        if working_root is None:
            return _denied(
                action=action,
                reason=LocalFileAuthorizationReason.AUTHORIZATION_REQUIRED,
                canonical_path=str(candidate),
            )
        try:
            binding = _resolve_root_binding(
                session_id=session_id,
                working_root=working_root,
            )
        except (OSError, ValueError):
            return _denied(
                action=action,
                reason=LocalFileAuthorizationReason.WORKING_ROOT_UNAVAILABLE,
                canonical_path=str(candidate),
            )
        canonical_root = Path(binding.canonical_root)
        if not candidate.is_relative_to(canonical_root):
            self._observe_root(binding)
            return _denied(
                action=action,
                reason=LocalFileAuthorizationReason.PATH_OUTSIDE_AUTHORIZED_SCOPE,
                canonical_path=str(candidate),
            )
        now = _now()
        with self._connect(session_id=session_id) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._synchronize_observed_root(connection, binding=binding, observed_at=now)
            row = connection.execute(
                """
                SELECT grant_id
                  FROM session_workspace_write_grants
                 WHERE session_id=? AND root_binding_sha256=? AND revoked_at IS NULL
                """,
                (session_id, binding.root_binding_sha256),
            ).fetchone()
        if row is None:
            return _denied(
                action=action,
                reason=LocalFileAuthorizationReason.AUTHORIZATION_REQUIRED,
                canonical_path=str(candidate),
            )
        return LocalFileAuthorizationDecision(
            allowed=True,
            action=action,
            reason=(
                LocalFileAuthorizationReason.EXPLICIT_WORKING_ROOT_WRITE_GRANT
            ),
            canonical_path=str(candidate),
            grant_id=str(row["grant_id"]),
        )

    def revoke(self, *, session_id: str, grant_id: str | None = None) -> int:
        """撤销一个回执，或某个 Session 的全部活动根目录回执。

        广泛 Session 撤销特意同时包含读写授予。传入特定回执时仍可跨两个授予表保持精确，
        因为 ID 具有不同前缀。
        """

        _require_session_id(session_id)
        now = _now().isoformat()
        with self._connect(session_id=session_id) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if grant_id is None:
                read_cursor = connection.execute(
                    """
                    UPDATE session_workspace_read_grants
                       SET revoked_at=?
                     WHERE session_id=? AND revoked_at IS NULL
                    """,
                    (now, session_id),
                )
                write_cursor = connection.execute(
                    """
                    UPDATE session_workspace_write_grants
                       SET revoked_at=?
                     WHERE session_id=? AND revoked_at IS NULL
                    """,
                    (now, session_id),
                )
                return read_cursor.rowcount + write_cursor.rowcount
            else:
                read_cursor = connection.execute(
                    """
                    UPDATE session_workspace_read_grants
                       SET revoked_at=?
                     WHERE session_id=? AND grant_id=? AND revoked_at IS NULL
                    """,
                    (now, session_id, grant_id),
                )
                write_cursor = connection.execute(
                    """
                    UPDATE session_workspace_write_grants
                       SET revoked_at=?
                     WHERE session_id=? AND grant_id=? AND revoked_at IS NULL
                    """,
                    (now, session_id, grant_id),
                )
                return read_cursor.rowcount + write_cursor.rowcount

    def list_grants(self, *, session_id: str) -> tuple[SessionWorkspaceReadGrant, ...]:
        """返回 Session 的授予历史，包括已撤销回执。"""

        _require_session_id(session_id)
        with self._connect(session_id=session_id) as connection:
            rows = connection.execute(
                """
                SELECT *
                  FROM session_workspace_read_grants
                 WHERE session_id=?
                 ORDER BY granted_at, grant_id
                """,
                (session_id,),
            ).fetchall()
        return tuple(_row_to_grant(row) for row in rows)

    def list_write_grants(
        self,
        *,
        session_id: str,
    ) -> tuple[SessionWorkspaceWriteGrant, ...]:
        """返回写入批准历史，包括已撤销回执。"""

        _require_session_id(session_id)
        with self._connect(session_id=session_id) as connection:
            rows = connection.execute(
                """
                SELECT *
                  FROM session_workspace_write_grants
                 WHERE session_id=?
                 ORDER BY granted_at, grant_id
                """,
                (session_id,),
            ).fetchall()
        return tuple(_row_to_write_grant(row) for row in rows)

    def _observe_root(self, binding: _RootBinding) -> None:
        with self._connect(session_id=binding.session_id) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._synchronize_observed_root(
                connection,
                binding=binding,
                observed_at=_now(),
            )

    @staticmethod
    def _synchronize_observed_root(
        connection: sqlite3.Connection,
        *,
        binding: _RootBinding,
        observed_at: datetime,
    ) -> None:
        previous = connection.execute(
            """
            SELECT root_binding_sha256
              FROM session_workspace_root_observations
             WHERE session_id=?
            """,
            (binding.session_id,),
        ).fetchone()
        changed = (
            previous is not None
            and previous["root_binding_sha256"] != binding.root_binding_sha256
        )
        if changed:
            connection.execute(
                """
                UPDATE session_workspace_read_grants
                   SET revoked_at=?
                 WHERE session_id=? AND revoked_at IS NULL
                """,
                (observed_at.isoformat(), binding.session_id),
            )
            connection.execute(
                """
                UPDATE session_workspace_write_grants
                   SET revoked_at=?
                 WHERE session_id=? AND revoked_at IS NULL
                """,
                (observed_at.isoformat(), binding.session_id),
            )
        connection.execute(
            """
            INSERT INTO session_workspace_root_observations (
                session_id,
                canonical_root,
                root_device,
                root_inode,
                root_binding_sha256,
                observed_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                canonical_root=excluded.canonical_root,
                root_device=excluded.root_device,
                root_inode=excluded.root_inode,
                root_binding_sha256=excluded.root_binding_sha256,
                observed_at=excluded.observed_at
            """,
            (
                binding.session_id,
                binding.canonical_root,
                str(binding.root_device),
                str(binding.root_inode),
                binding.root_binding_sha256,
                observed_at.isoformat(),
            ),
        )

    def _connect(self, *, session_id: str) -> sqlite3.Connection:
        path = self.path
        if self._explicit_path is None:
            from . import store as session_store

            if session_store.uses_partitioned_storage():
                bound_session_id = session_store.current_session_id()
                if bound_session_id != session_id:
                    raise session_store.SessionStoreError(
                        f"file authority for session {session_id!r} cannot use "
                        f"active session scope {bound_session_id!r}"
                    )
            session_store.init_db()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        if self._explicit_path is not None:
            self._ensure_schema(connection)
        return connection

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS session_workspace_read_grants (
                    grant_id            TEXT PRIMARY KEY,
                    session_id          TEXT NOT NULL,
                    canonical_root      TEXT NOT NULL,
                    root_device         TEXT NOT NULL,
                    root_inode          TEXT NOT NULL,
                    root_binding_sha256 TEXT NOT NULL,
                    granted_at          TEXT NOT NULL,
                    revoked_at          TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_session_workspace_active_grant
                    ON session_workspace_read_grants(session_id)
                 WHERE revoked_at IS NULL
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_workspace_grant_history
                    ON session_workspace_read_grants(session_id, granted_at, grant_id)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS session_workspace_root_observations (
                    session_id          TEXT PRIMARY KEY,
                    canonical_root      TEXT NOT NULL,
                    root_device         TEXT NOT NULL,
                    root_inode          TEXT NOT NULL,
                    root_binding_sha256 TEXT NOT NULL,
                    observed_at         TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS session_workspace_write_grants (
                    grant_id            TEXT PRIMARY KEY,
                    session_id          TEXT NOT NULL,
                    canonical_root      TEXT NOT NULL,
                    root_device         TEXT NOT NULL,
                    root_inode          TEXT NOT NULL,
                    root_binding_sha256 TEXT NOT NULL,
                    granted_at          TEXT NOT NULL,
                    revoked_at          TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_session_workspace_active_write_grant
                    ON session_workspace_write_grants(session_id)
                 WHERE revoked_at IS NULL
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_workspace_write_grant_history
                    ON session_workspace_write_grants(session_id, granted_at, grant_id)
                """
            )


def _resolve_existing(candidate: Path | str) -> Path | None:
    try:
        resolved = Path(candidate).expanduser().resolve(strict=True)
        if not resolved.exists():
            return None
        return resolved
    except (OSError, RuntimeError, ValueError):
        return None


def _resolve_root_binding(
    *,
    session_id: str,
    working_root: Path | str,
) -> _RootBinding:
    canonical = Path(working_root).expanduser().resolve(strict=True)
    if not canonical.is_dir():
        raise ValueError("the Session working root must be an existing directory")
    stat_result = canonical.stat()
    payload = json.dumps(
        {
            "canonical_root": str(canonical),
            "root_device": str(stat_result.st_dev),
            "root_inode": str(stat_result.st_ino),
            "schema_version": _BINDING_SCHEMA_VERSION,
            "session_id": session_id,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _RootBinding(
        session_id=session_id,
        canonical_root=str(canonical),
        root_device=stat_result.st_dev,
        root_inode=stat_result.st_ino,
        root_binding_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _row_to_grant(row: sqlite3.Row) -> SessionWorkspaceReadGrant:
    return SessionWorkspaceReadGrant(
        grant_id=str(row["grant_id"]),
        session_id=str(row["session_id"]),
        canonical_root=str(row["canonical_root"]),
        root_device=int(row["root_device"]),
        root_inode=int(row["root_inode"]),
        root_binding_sha256=str(row["root_binding_sha256"]),
        granted_at=datetime.fromisoformat(str(row["granted_at"])),
        revoked_at=(
            datetime.fromisoformat(str(row["revoked_at"]))
            if row["revoked_at"] is not None
            else None
        ),
    )


def _row_to_write_grant(row: sqlite3.Row) -> SessionWorkspaceWriteGrant:
    return SessionWorkspaceWriteGrant(
        grant_id=str(row["grant_id"]),
        session_id=str(row["session_id"]),
        canonical_root=str(row["canonical_root"]),
        root_device=int(row["root_device"]),
        root_inode=int(row["root_inode"]),
        root_binding_sha256=str(row["root_binding_sha256"]),
        granted_at=datetime.fromisoformat(str(row["granted_at"])),
        revoked_at=(
            datetime.fromisoformat(str(row["revoked_at"]))
            if row["revoked_at"] is not None
            else None
        ),
    )


def _denied(
    *,
    action: LocalFileAction,
    reason: LocalFileAuthorizationReason,
    canonical_path: str | None = None,
) -> LocalFileAuthorizationDecision:
    return LocalFileAuthorizationDecision(
        allowed=False,
        action=action,
        reason=reason,
        canonical_path=canonical_path,
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _is_valid_session_id(session_id: str) -> bool:
    return bool(
        isinstance(session_id, str)
        and session_id.strip()
        and len(session_id) <= 160
        and "/" not in session_id
        and "\\" not in session_id
        and session_id not in {".", ".."}
    )


def _require_session_id(session_id: str) -> None:
    if not _is_valid_session_id(session_id):
        raise ValueError(f"invalid Session id: {session_id!r}")


__all__ = [
    'LocalFileAction',
    'LocalFileAuthorizationDecision',
    'LocalFileAuthorizationReason',
    'SessionWorkspaceReadGrant',
    'SessionWorkspaceWriteGrant',
    'SqliteSessionFileAuthority',
]
