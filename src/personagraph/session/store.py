"""会话与对话历史的持久化存储（SQLite）。

权威来源：`session_turns` 表（用户可见对话历史与普通聊天续接）。
LangGraph 的 SqliteSaver checkpointer 另行管理运行态图状态恢复，不作为聊天记录来源。
本模块负责会话注册、历史日志、文件夹与工作记忆派生态的读写。

P1 实现：建表 + 会话创建/读取/列出 + 历史追加/读取 + last_active 更新。
P2 实现：重命名/归档/软删/恢复/purge/文件夹。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
import sqlite3
import shutil
import threading
import uuid
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .persistence.l1.tool_history import L1ToolHistoryReader

from . import history_store_facade as history_store_facade_records
from . import lifecycle_store_facade as lifecycle_store_facade_records
from . import runtime_call_ledger_store_facade as runtime_call_ledger_store_facade_records
from . import turn_input_store_facade as turn_input_store_facade_records
from .runtime_call_ledger_store_facade import (
    RuntimeModelCallIdentityCollision,
    RuntimeModelCallPersistenceError,
    RuntimeModelCallTerminalState,
    RuntimeModelLedgerMutationResult,
    RuntimeModelLogicalRequest,
    RuntimeModelPhysicalAttemptRequest,
    RuntimeModelPhysicalAttemptSettlement,
    RuntimeToolCallIdentityCollision,
    RuntimeToolCallPersistenceError,
    RuntimeToolCallTerminalState,
    RuntimeToolCallWaitingExternalState,
    RuntimeToolLedgerMutationResult,
    RuntimeToolLogicalRequest,
    RuntimeToolPhysicalAttemptRequest,
    RuntimeToolPhysicalAttemptSettlement,
    StoredRuntimeModelLogicalCall,
    StoredRuntimeModelPhysicalAttempt,
    StoredRuntimeModelRejectedOutput,
    StoredRuntimeToolLogicalCall,
    StoredRuntimeToolPhysicalAttempt,
)
from .turn_input_store_facade import (
    AttachmentBindingError,
    MAX_RUNTIME_TURN_EVENT_PAGE_SIZE,
    UnknownRuntimeTurnEventCursor,
)
from .history_store_facade import TranscriptDeliveryReferenceError
from .persistence.turns import entry_tasks as entry_task_records
from .persistence.metadata import sessions as session_records
from .persistence.metadata import queries as session_queries
from .persistence.turns import post_commit_jobs as post_commit_job_records
from .persistence import execution_findings as execution_findings_records
from .persistence import document_mounts as document_mount_records
from .persistence.l1 import turn_runs as l1_turn_run_records
from .persistence.history import session_summaries as session_summary_records
from .persistence.turns import turn_execution as turn_execution_records
from .persistence.turns import turn_routing_policies as turn_routing_policy_records
from .persistence.deps import StoreDeps
from .persistence.schema import SCHEMA_VERSION, initialize_schema
from .catalog import (
    SessionCatalog,
    SessionCatalogError,
    session_db_relative_path,
    validate_session_id,
)
from .entry_task_contracts import (
    EntryPendingTaskQuestion,
    EntryTaskCatalogItem,
)
from ..persistent_turn_content.findings import (
    ExecutionFindingsLedgerCreateResult,
    ExecutionFindingsMutationCommand,
    ExecutionFindingsMutationResult,
    ExecutionFindingsOwnerKind,
    ExecutionFindingsQuota,
    ExecutionFindingsSnapshot,
)
from .session_summary import (
    SessionSummaryProgressError,
    SessionSummaryStateConflict,
    SessionSummaryStateError,
    SessionSummaryState,
    SessionSummaryStatus,
    SessionSummaryTurnPair,
    SessionSummaryUpdate,
)

TurnExecutionBusyError = turn_execution_records.TurnExecutionBusyError
TurnExecutionRequestIdCollision = turn_execution_records.TurnExecutionRequestIdCollision
L1TurnRunPersistenceError = l1_turn_run_records.L1TurnRunPersistenceError
ExecutionFindingsPersistenceError = (
    execution_findings_records.ExecutionFindingsPersistenceError
)
ExecutionFindingsOwnerClosed = (
    execution_findings_records.ExecutionFindingsOwnerClosed
)
ExecutionFindingsRevisionConflict = (
    execution_findings_records.ExecutionFindingsRevisionConflict
)
ExecutionFindingsMutationIdentityCollision = (
    execution_findings_records.ExecutionFindingsMutationIdentityCollision
)
ExecutionFindingsQuotaExceeded = (
    execution_findings_records.ExecutionFindingsQuotaExceeded
)
ExecutionFindingsSourceReferenceInvalid = (
    execution_findings_records.ExecutionFindingsSourceReferenceInvalid
)
ExecutionFindingsScopeInvalid = (
    execution_findings_records.ExecutionFindingsScopeInvalid
)
ExecutionFindingsStoredAuthorityCorrupt = (
    execution_findings_records.ExecutionFindingsStoredAuthorityCorrupt
)
TurnExecutionWindowRevisionConflict = turn_execution_records.TurnExecutionWindowRevisionConflict
TurnExecutionLeaseConflict = turn_execution_records.TurnExecutionLeaseConflict
TurnExecutionFinalizationConflict = turn_execution_records.TurnExecutionFinalizationConflict
FormalTurnReference = turn_execution_records.FormalTurnReference
TurnPostCommitJobsPending = turn_execution_records.TurnPostCommitJobsPending
TurnPostCommitJobLeaseError = turn_execution_records.TurnPostCommitJobLeaseError

MAX_TURN_EXECUTION_IDENTIFIER_LENGTH = (
    turn_execution_records.MAX_TURN_EXECUTION_IDENTIFIER_LENGTH
)
# 生产环境绝不打开进程级全局 Session 数据库。``DB_PATH`` 是显式的旧版测试接缝，
# 其未改动的哨兵值有意设为不可创建的 SQLite 目标。
_DEFAULT_DB_PATH = Path(os.devnull)
DB_PATH = _DEFAULT_DB_PATH
_SESSION_DATABASE_BINDING: ContextVar[tuple[str, Path] | None] = ContextVar(
    "personagraph_session_database_binding",
    default=None,
)


class SessionStoreError(RuntimeError):
    """Session 持久化在安全存储操作开始前失败。"""


class SessionCreationRollbackError(SessionStoreError):
    """未发布 Session 的补偿未能完整完成；其默认 Project 必须保留。"""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"failed to roll back unpublished session {session_id!r}")
        self.session_id = session_id


class SessionCreationNotOwnedError(SessionStoreError):
    """Session 创建在取得 Project/分区所有权前失败。"""


class SessionCreationClaimConflict(SessionCreationNotOwnedError):
    """生成的 Session ID 已由另一个权威记录或分区占用。"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _normalized_database_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _uses_explicit_database_override() -> bool:
    """当 DB_PATH 被覆盖时，保留既有测试/迁移接缝。"""

    return _normalized_database_path(DB_PATH) != _normalized_database_path(
        _DEFAULT_DB_PATH
    )


def uses_partitioned_storage() -> bool:
    """是否由生产 catalog 与逐 Session 数据库负责持久化。"""

    return not _uses_explicit_database_override()


def _catalog() -> SessionCatalog:
    return SessionCatalog()


@contextmanager
def _project_documents_scope(session_id: str) -> Iterator[None]:
    """绑定由一个 catalog locator 指定的项目文档权威状态。"""

    record = _catalog().get_session(session_id)
    project_id = record.get("project_id") if record is not None else None
    if project_id is None:
        yield
        return

    from . import project_catalog
    from ..workspace.storage.context import bind as bind_project_documents
    from ..workspace.storage.database import DocumentDatabase

    try:
        project = project_catalog.get_by_id(str(project_id))
        if project is None:
            raise SessionStoreError(
                f"session {session_id!r} names unknown project {project_id!r}"
            )
        database = DocumentDatabase(
            project_id=project.project_id,
            project_root=project.canonical_root,
            db_path=project.documents_db_path,
        )
    except SessionStoreError:
        raise
    except (OSError, RuntimeError, sqlite3.Error, TypeError, ValueError) as exc:
        raise SessionStoreError(
            f"failed to resolve project documents for session {session_id!r}"
        ) from exc

    with bind_project_documents(database):
        yield


class _SessionDocumentMountPort:
    """把 Session 持久能力适配到 Workspace Document 的窄端口。"""

    @staticmethod
    def current_session_id() -> str:
        return current_session_id()

    @staticmethod
    def mount_document(document_id: str, session_id: str) -> bool:
        return mount_document(document_id, session_id)

    @staticmethod
    def unmount_document(document_id: str, session_id: str) -> bool:
        return unmount_document(document_id, session_id)

    @staticmethod
    def is_document_mounted(document_id: str, session_id: str) -> bool:
        return is_document_mounted(document_id, session_id)

    @staticmethod
    def list_document_mounts(session_id: str) -> list[dict[str, Any]]:
        return list_document_mounts(session_id)

    @staticmethod
    def create_document_retrieval_snapshot(
        *,
        snapshot_id: str,
        session_id: str,
        manifest_hash: str,
        manifest_json: str,
        reason: str,
        created_at: str,
    ) -> None:
        create_document_retrieval_snapshot(
            snapshot_id=snapshot_id,
            session_id=session_id,
            manifest_hash=manifest_hash,
            manifest_json=manifest_json,
            reason=reason,
            created_at=created_at,
        )

    @staticmethod
    def get_document_retrieval_snapshot(
        snapshot_id: str,
        session_id: str,
    ) -> dict[str, Any] | None:
        return get_document_retrieval_snapshot(snapshot_id, session_id)


_SESSION_DOCUMENT_MOUNT_PORT = _SessionDocumentMountPort()


@contextmanager
def document_mount_port_scope() -> Iterator[None]:
    """向调用方组合 Session 所拥有的文档挂载持久能力。"""

    from ..workspace.documents.mounting import bind_document_mount_port

    with bind_document_mount_port(_SESSION_DOCUMENT_MOUNT_PORT):
        yield


@contextmanager
def session_database_scope(session_id: str) -> Iterator[Path]:
    """绑定一个 Session 数据库及其可选项目文档权威状态。

    此绑定限定于 request/task 局部，绝不修改 ``DB_PATH``，因此多个 API 线程为
    不同 Session 服务时仍然安全。

    JSON dispatcher、SSE handler 和后台 worker 都应从这个 scope 进入配套存储。
    同 Session 嵌套复用绑定，跨 Session 嵌套拒绝；退出时恢复 ContextVar。
    它决定数据写到哪里，不代表已获得 Session run guard，也不自动设置 trajectory 的
    turn_id；后者在 accepted Turn 执行边界绑定，新 Thread 必须重新进入存储 scope。
    """

    safe_id = validate_session_id(session_id)
    current = _SESSION_DATABASE_BINDING.get()
    if current is not None and current[0] != safe_id:
        raise SessionStoreError(
            f"cannot route session {safe_id!r} inside active scope {current[0]!r}"
        )
    if current is not None:
        # 请求已限定作用域时，service facade 仍可能通过 ``get_session`` 回到这里。
        # 复用两个 ContextVar，避免遮蔽 Session 绑定或尝试被禁止的嵌套项目文档绑定。
        yield current[1]
        return
    explicit_database_override = _uses_explicit_database_override()
    path = (
        _normalized_database_path(DB_PATH)
        if explicit_database_override
        else _catalog().session_db_path(safe_id)
    )
    token = _SESSION_DATABASE_BINDING.set((safe_id, path))
    try:
        with document_mount_port_scope():
            if explicit_database_override:
                yield path
            else:
                with _project_documents_scope(safe_id):
                    yield path
    finally:
        _SESSION_DATABASE_BINDING.reset(token)


@contextmanager
def session_database_route_scope(session_id: str) -> Iterator[Path]:
    """在已绑定精确 Project 时仅补充目标 Session 数据库路由。

    Project 文档维护线程已经持有一个不可嵌套的 DocumentDatabase ContextVar。普通
    :func:`session_database_scope` 会再次绑定 Project，因此不能用于后台 job。此窄入口
    先证明 Session catalog、Project catalog 与当前 DocumentDatabase 完全一致，再只
    设置 Session 数据库 ContextVar；它不放宽普通请求的作用域规则。
    """

    safe_id = validate_session_id(session_id)
    if _uses_explicit_database_override():
        raise SessionStoreError(
            "session database route scope requires partitioned project state"
        )
    from ..workspace.storage.context import (
        current as current_project_documents,
    )
    from . import project_catalog

    database = current_project_documents()
    if database is None:
        raise SessionStoreError(
            "session database route scope requires a bound project"
        )
    record = _catalog().get_session(safe_id)
    project_id = str(record.get("project_id") or "") if record is not None else ""
    if not project_id or project_id != database.project_id:
        raise SessionStoreError(
            "session database route scope requires the same project"
        )
    project = project_catalog.get_by_id(project_id)
    if project is None or (
        Path(project.canonical_root).expanduser().resolve()
        != database.project_root.expanduser().resolve()
        or Path(project.documents_db_path).expanduser().resolve()
        != database.db_path.expanduser().resolve()
    ):
        raise SessionStoreError(
            "session database route scope project authority changed"
        )
    path = _catalog().session_db_path(safe_id)
    current = _SESSION_DATABASE_BINDING.get()
    if current is not None:
        if current[0] != safe_id:
            raise SessionStoreError(
                f"cannot route session {safe_id!r} inside active scope {current[0]!r}"
            )
        if current[1].expanduser().resolve() != path.expanduser().resolve():
            raise SessionStoreError(
                "active session database route does not match the catalog"
            )
        with document_mount_port_scope():
            yield current[1]
        return
    token = _SESSION_DATABASE_BINDING.set((safe_id, path))
    try:
        with document_mount_port_scope():
            yield path
    finally:
        _SESSION_DATABASE_BINDING.reset(token)


def _active_database_path() -> Path:
    if _uses_explicit_database_override():
        return _normalized_database_path(DB_PATH)
    binding = _SESSION_DATABASE_BINDING.get()
    if binding is None:
        raise SessionStoreError(
            "a production session database operation requires "
            "session_database_scope(session_id)"
        )
    return binding[1]


def current_session_database_path() -> Path:
    """返回当前 Session 配套存储应使用的数据库。

    显式覆盖 ``DB_PATH`` 是既有测试与迁移工具的兼容入口，和本模块其他存储
    操作一样直接使用该路径。生产分区模式不提供进程级回退：调用方必须先进入
    ``session_database_scope(session_id)``。
    """

    if _uses_explicit_database_override():
        return _normalized_database_path(DB_PATH)
    binding = _SESSION_DATABASE_BINDING.get()
    if binding is None:
        raise SessionStoreError(
            "no session database is bound; enter "
            "session_database_scope(session_id) first"
        )
    return binding[1]


def current_session_id() -> str:
    """返回绑定到当前 request/task 的 Session 标识。"""

    binding = _SESSION_DATABASE_BINDING.get()
    if binding is None:
        raise SessionStoreError(
            "no session database is bound; enter "
            "session_database_scope(session_id) first"
        )
    return binding[0]


def mount_document(
    document_id: str,
    session_id: str,
    *,
    mounted_at: str | None = None,
) -> bool:
    """为项目持有的 Document 持久化一项 Session 能力。"""

    init_db()
    with _connect() as conn:
        return document_mount_records.mount_document(
            conn,
            session_id=session_id,
            document_id=document_id,
            mounted_at=mounted_at or _now(),
        )


def unmount_document(document_id: str, session_id: str) -> bool:
    init_db()
    with _connect() as conn:
        return document_mount_records.unmount_document(
            conn,
            session_id=session_id,
            document_id=document_id,
        )


def is_document_mounted(document_id: str, session_id: str) -> bool:
    init_db()
    with _connect() as conn:
        return document_mount_records.is_document_mounted(
            conn,
            session_id=session_id,
            document_id=document_id,
        )


def list_document_mounts(session_id: str) -> list[dict[str, Any]]:
    init_db()
    with _connect() as conn:
        return document_mount_records.list_document_mounts(
            conn,
            session_id=session_id,
        )


def create_document_retrieval_snapshot(
    *,
    snapshot_id: str,
    session_id: str,
    manifest_hash: str,
    manifest_json: str,
    reason: str,
    created_at: str | None = None,
) -> None:
    init_db()
    with _connect() as conn:
        document_mount_records.create_retrieval_snapshot(
            conn,
            snapshot_id=snapshot_id,
            session_id=session_id,
            manifest_hash=manifest_hash,
            manifest_json=manifest_json,
            reason=reason,
            created_at=created_at or _now(),
        )


def get_document_retrieval_snapshot(
    snapshot_id: str,
    session_id: str,
) -> dict[str, Any] | None:
    init_db()
    with _connect() as conn:
        return document_mount_records.get_retrieval_snapshot(
            conn,
            snapshot_id=snapshot_id,
            session_id=session_id,
        )


def _connect() -> sqlite3.Connection:
    conn: sqlite3.Connection | None = None
    database_path = _active_database_path()
    try:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(database_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    except (OSError, sqlite3.Error) as exc:
        if conn is not None:
            conn.close()
        raise SessionStoreError(
            f"Failed to open session database at {database_path}"
        ) from exc


def connect_session_context_authority() -> sqlite3.Connection:
    """打开当前绑定的 Session 数据库以持久化 SessionContext。

    SessionContext 过去会直接连接进程级全局 ``DB_PATH``。在分区生产环境中，
    该路径只是已退役的测试哨兵，因此直接连接可能重新创建旧的共享会话历史数据库。
    """

    init_db()
    return _connect()


def current_store_deps() -> StoreDeps:
    """Return a fresh dependency bundle bound to the current Session DB route."""

    return StoreDeps(init_db=init_db, connect=_connect, now=_now, new_id=_new_id)


def _deps() -> StoreDeps:
    return current_store_deps()


_RUNTIME_CALL_LEDGER_STORE_FACADE = (
    runtime_call_ledger_store_facade_records.build_runtime_call_ledger_store_facade(
        deps_factory=lambda: _deps(),
    )
)
reserve_runtime_model_logical_call = (
    _RUNTIME_CALL_LEDGER_STORE_FACADE.reserve_runtime_model_logical_call
)
append_runtime_model_physical_attempt = (
    _RUNTIME_CALL_LEDGER_STORE_FACADE.append_runtime_model_physical_attempt
)
settle_runtime_model_physical_attempt = (
    _RUNTIME_CALL_LEDGER_STORE_FACADE.settle_runtime_model_physical_attempt
)
get_runtime_model_logical_call = (
    _RUNTIME_CALL_LEDGER_STORE_FACADE.get_runtime_model_logical_call
)
get_runtime_model_rejected_output = (
    _RUNTIME_CALL_LEDGER_STORE_FACADE.get_runtime_model_rejected_output
)
reserve_runtime_tool_logical_call = (
    _RUNTIME_CALL_LEDGER_STORE_FACADE.reserve_runtime_tool_logical_call
)
append_runtime_tool_physical_attempt = (
    _RUNTIME_CALL_LEDGER_STORE_FACADE.append_runtime_tool_physical_attempt
)
settle_runtime_tool_physical_attempt = (
    _RUNTIME_CALL_LEDGER_STORE_FACADE.settle_runtime_tool_physical_attempt
)
get_runtime_tool_logical_call = (
    _RUNTIME_CALL_LEDGER_STORE_FACADE.get_runtime_tool_logical_call
)

_HISTORY_STORE_FACADE = history_store_facade_records.build_history_store_facade(
    deps_factory=lambda: _deps(),
)
_LOCAL_GET_SESSION = _HISTORY_STORE_FACADE.get_session
_LOCAL_LIST_SESSIONS = _HISTORY_STORE_FACADE.list_sessions
_LOCAL_SEARCH_SESSIONS = _HISTORY_STORE_FACADE.search_sessions
get_turns = _HISTORY_STORE_FACADE.get_turns
_LOCAL_GET_TURNS = get_turns
get_turn = _HISTORY_STORE_FACADE.get_turn
get_committed_turn_pair = _HISTORY_STORE_FACADE.get_committed_turn_pair
list_committed_turn_pairs = _HISTORY_STORE_FACADE.list_committed_turn_pairs
get_committed_turn_pair_index_state = (
    _HISTORY_STORE_FACADE.get_committed_turn_pair_index_state
)
get_committed_turn_pair_index_binding_snapshot = (
    _HISTORY_STORE_FACADE.get_committed_turn_pair_index_binding_snapshot
)
get_user_turn_markers = _HISTORY_STORE_FACADE.get_user_turn_markers
get_history_messages = _HISTORY_STORE_FACADE.get_history_messages


def _merge_catalog_session(
    local: dict[str, Any],
    catalog_record: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(local)
    merged["project_id"] = catalog_record.get("project_id")
    merged["db_path"] = catalog_record["db_path"]
    if catalog_record.get("folder_id") is not None and merged.get("folder_id") is None:
        merged["folder_id"] = catalog_record["folder_id"]
    return merged


def _read_catalog_session(catalog_record: dict[str, Any]) -> dict[str, Any]:
    session_id = str(catalog_record["id"])
    expected_relative_path = session_db_relative_path(session_id)
    if catalog_record.get("db_path") != expected_relative_path:
        raise SessionStoreError(
            f"catalog locator for session {session_id!r} is not canonical"
        )
    database_path = _catalog().session_db_path(session_id)
    if not database_path.is_file():
        raise SessionStoreError(
            f"catalog locator for session {session_id!r} has no session database"
        )
    with session_database_scope(session_id):
        local = _LOCAL_GET_SESSION(session_id)
    if local is None:
        raise SessionStoreError(
            f"session database for {session_id!r} has no identity row"
        )
    return _merge_catalog_session(local, catalog_record)


def _partitioned_turns(session_id: str) -> list[dict[str, Any]]:
    database_path = _catalog().session_db_path(session_id)
    if not database_path.is_file():
        raise SessionStoreError(
            f"catalog locator for session {session_id!r} has no session database"
        )
    with session_database_scope(session_id):
        return _LOCAL_GET_TURNS(session_id)


def get_session(session_id: str) -> dict[str, Any] | None:
    if _uses_explicit_database_override():
        return _LOCAL_GET_SESSION(session_id)
    record = _catalog().get_session(session_id)
    return _read_catalog_session(record) if record is not None else None


def _ordered_catalog_sessions() -> Iterator[dict[str, Any]]:
    yield from sorted(
        (_read_catalog_session(record) for record in _catalog().list_sessions()),
        key=lambda row: str(row.get("last_active_at") or ""),
        reverse=True,
    )


def list_session_ids_for_post_commit_recovery() -> tuple[str, ...]:
    """枚举恢复候选身份，不展开会话数据库或会话正文。

    GUI 列表需要逐库读取完整会话信息；后台恢复不能复用它，否则一个已损坏或
    丢失的数据库会阻断全部健康会话。这里只读共享目录，具体状态由恢复器逐项
    进入 Session scope 检查，并分别报告失败。显式单库接缝仍只读取其本地身份表。
    """

    rows = (
        _LOCAL_LIST_SESSIONS(status="all")
        if _uses_explicit_database_override()
        else _catalog().list_sessions()
    )
    return tuple(str(row["id"]) for row in rows if row.get("status") != "trashed")


def list_sessions(
    include_archived: bool = False,
    include_trashed: bool = False,
    *,
    status: str = "active",
    folder_id: str | None = None,
    persona_id: str | None = None,
    query: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if _uses_explicit_database_override():
        return _LOCAL_LIST_SESSIONS(
            include_archived,
            include_trashed,
            status=status,
            folder_id=folder_id,
            persona_id=persona_id,
            query=query,
            limit=limit,
        )
    return session_queries.list_sessions(
        _ordered_catalog_sessions(),
        _partitioned_turns,
        include_archived=include_archived,
        include_trashed=include_trashed,
        status=status,
        folder_id=folder_id,
        persona_id=persona_id,
        query=query,
        limit=limit,
    )


def search_sessions(
    query: str,
    status: str = "active",
    folder_id: str | None = None,
    persona_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    if _uses_explicit_database_override():
        return _LOCAL_SEARCH_SESSIONS(
            query,
            status,
            folder_id,
            persona_id,
            limit,
        )
    return session_queries.search_sessions(
        _ordered_catalog_sessions(),
        _partitioned_turns,
        query=query,
        status=status,
        folder_id=folder_id,
        persona_id=persona_id,
        limit=limit,
    )

_LIFECYCLE_STORE_FACADE = (
    lifecycle_store_facade_records.build_session_lifecycle_store_facade(
        deps_factory=lambda: _deps(),
    )
)
_LOCAL_RENAME_SESSION = _LIFECYCLE_STORE_FACADE.rename_session
_LOCAL_ARCHIVE_SESSION = _LIFECYCLE_STORE_FACADE.archive_session
_LOCAL_UNARCHIVE_SESSION = _LIFECYCLE_STORE_FACADE.unarchive_session
_LOCAL_TRASH_SESSION = _LIFECYCLE_STORE_FACADE.trash_session
_LOCAL_RESTORE_SESSION = _LIFECYCLE_STORE_FACADE.restore_session
_LOCAL_PURGE_SESSION = _LIFECYCLE_STORE_FACADE.purge_session
_LOCAL_MOVE_SESSION = _LIFECYCLE_STORE_FACADE.move_session
_LOCAL_LIST_TRASHED = _LIFECYCLE_STORE_FACADE.list_trashed
_LOCAL_CREATE_FOLDER = _LIFECYCLE_STORE_FACADE.create_folder
_LOCAL_GET_FOLDER = _LIFECYCLE_STORE_FACADE.get_folder
_LOCAL_LIST_FOLDERS = _LIFECYCLE_STORE_FACADE.list_folders
_LOCAL_RENAME_FOLDER = _LIFECYCLE_STORE_FACADE.rename_folder
_LOCAL_DELETE_FOLDER = _LIFECYCLE_STORE_FACADE.delete_folder
_LOCAL_MOVE_FOLDER = _LIFECYCLE_STORE_FACADE.move_folder
_LOCAL_SET_FOLDER_STATUS = _LIFECYCLE_STORE_FACADE.set_folder_status
_LOCAL_FOLDER_TREE = _LIFECYCLE_STORE_FACADE.folder_tree


def _restore_partitioned_session_metadata(
    session_id: str,
    snapshot: Mapping[str, Any],
) -> None:
    """catalog 镜像变更失败时执行尽力补偿。"""

    with session_database_scope(session_id):
        init_db()
        with _connect() as conn:
            conn.execute(
                "UPDATE sessions SET title=?, folder_id=?, status=?, archived_at=?, "
                "deleted_at=?, previous_status=?, working_dir=? WHERE id=?",
                (
                    snapshot.get("title"),
                    snapshot.get("folder_id"),
                    snapshot.get("status"),
                    snapshot.get("archived_at"),
                    snapshot.get("deleted_at"),
                    snapshot.get("previous_status"),
                    snapshot.get("working_dir"),
                    session_id,
                ),
            )


def _partitioned_session_lifecycle_mutation(
    session_id: str,
    mutation,
    *,
    catalog_fields,
) -> bool:
    catalog = _catalog()
    if catalog.get_session(session_id) is None:
        return False
    with session_database_scope(session_id):
        before = _LOCAL_GET_SESSION(session_id)
        if before is None or not mutation():
            return False
        after = _LOCAL_GET_SESSION(session_id)
    if after is None:
        _restore_partitioned_session_metadata(session_id, before)
        raise SessionStoreError(f"session {session_id!r} disappeared during mutation")
    try:
        if catalog.update_session(session_id, **catalog_fields(after)) is None:
            raise SessionStoreError(f"catalog locator for session {session_id!r} disappeared")
    except Exception:
        _restore_partitioned_session_metadata(session_id, before)
        raise
    return True


_SESSION_TITLE_LOCK = threading.RLock()


def rename_session(session_id: str, title: str) -> bool:
    with _SESSION_TITLE_LOCK:
        return _rename_session(session_id, title)


def _rename_session(session_id: str, title: str) -> bool:
    if _uses_explicit_database_override():
        return _LOCAL_RENAME_SESSION(session_id, title)
    return _partitioned_session_lifecycle_mutation(
        session_id,
        lambda: _LOCAL_RENAME_SESSION(session_id, title),
        catalog_fields=lambda _after: {"title": title},
    )


def replace_session_title(session_id: str, *, expected: str | None, title: str) -> bool:
    """自动标题 CAS；不修改 Project、目录、状态或会话历史。"""
    with _SESSION_TITLE_LOCK:
        def replace() -> bool:
            return session_records.replace_session_title(_deps(), session_id, expected=expected, title=title)

        if _uses_explicit_database_override():
            return replace()
        return _partitioned_session_lifecycle_mutation(
            session_id, replace, catalog_fields=lambda after: {"title": after["title"]},
        )


def archive_session(session_id: str) -> bool:
    if _uses_explicit_database_override():
        return _LOCAL_ARCHIVE_SESSION(session_id)
    return _partitioned_session_lifecycle_mutation(
        session_id,
        lambda: _LOCAL_ARCHIVE_SESSION(session_id),
        catalog_fields=lambda after: {"status": after["status"]},
    )


def unarchive_session(session_id: str) -> bool:
    if _uses_explicit_database_override():
        return _LOCAL_UNARCHIVE_SESSION(session_id)
    return _partitioned_session_lifecycle_mutation(
        session_id,
        lambda: _LOCAL_UNARCHIVE_SESSION(session_id),
        catalog_fields=lambda after: {"status": after["status"]},
    )


def trash_session(session_id: str) -> bool:
    if _uses_explicit_database_override():
        return _LOCAL_TRASH_SESSION(session_id)
    return _partitioned_session_lifecycle_mutation(
        session_id,
        lambda: _LOCAL_TRASH_SESSION(session_id),
        catalog_fields=lambda after: {"status": after["status"]},
    )


def restore_session(session_id: str) -> bool:
    if _uses_explicit_database_override():
        return _LOCAL_RESTORE_SESSION(session_id)
    return _partitioned_session_lifecycle_mutation(
        session_id,
        lambda: _LOCAL_RESTORE_SESSION(session_id),
        catalog_fields=lambda after: {"status": after["status"]},
    )


def purge_session(session_id: str) -> bool:
    if _uses_explicit_database_override():
        return _LOCAL_PURGE_SESSION(session_id)
    catalog = _catalog()
    tombstone = catalog.begin_session_purge(session_id)
    if tombstone is None:
        return False
    database_path = catalog.session_db_path(session_id)
    session_directory = database_path.parent
    try:
        if session_directory.exists():
            shutil.rmtree(session_directory)
        _INITIALIZED_PATHS.discard(str(database_path.resolve()))
    except OSError:
        # ``rmtree`` 可能在只删除部分 Session 分区后失败。保留待处理墓碑可隐藏这部分
        # 状态并保证后续重试安全；重新暴露旧 locator 会把可恢复清除变成损坏的活动
        # Session 实例。
        raise
    if not catalog.complete_session_purge(session_id):
        raise SessionStoreError(
            f"session {session_id!r} payload was purged but its catalog tombstone "
            "could not be completed"
        )
    return True


def _set_partitioned_session_folder(session_id: str, folder_id: str | None) -> bool:
    catalog = _catalog()
    if folder_id is not None and catalog.get_folder(folder_id) is None:
        return False
    catalog_record = catalog.get_session(session_id)
    if catalog_record is None:
        return False
    with session_database_scope(session_id):
        local = _LOCAL_GET_SESSION(session_id)
        if local is None:
            return False
        init_db()
        with _connect() as conn:
            updated = bool(
                conn.execute(
                    "UPDATE sessions SET folder_id=? WHERE id=?",
                    (folder_id, session_id),
                ).rowcount
            )
    if not updated:
        return False
    try:
        if catalog.update_session(session_id, folder_id=folder_id) is None:
            raise SessionStoreError(f"catalog locator for session {session_id!r} disappeared")
    except Exception:
        _restore_partitioned_session_metadata(session_id, local)
        raise
    return True


def move_session(session_id: str, folder_id: str | None) -> bool:
    if _uses_explicit_database_override():
        return _LOCAL_MOVE_SESSION(session_id, folder_id)
    return _set_partitioned_session_folder(session_id, folder_id)


def list_trashed() -> list[dict[str, Any]]:
    if _uses_explicit_database_override():
        return _LOCAL_LIST_TRASHED()
    return list_sessions(status="trashed")


def create_folder(name: str, parent_id: str | None = None) -> str:
    if _uses_explicit_database_override():
        return _LOCAL_CREATE_FOLDER(name, parent_id)
    folder_id = _new_id()
    _catalog().create_folder(folder_id=folder_id, name=name, parent_id=parent_id)
    return folder_id


def get_folder(folder_id: str) -> dict[str, Any] | None:
    if _uses_explicit_database_override():
        return _LOCAL_GET_FOLDER(folder_id)
    return _catalog().get_folder(folder_id)


def _catalog_folder_with_count(folder: dict[str, Any]) -> dict[str, Any]:
    item = dict(folder)
    item["session_count"] = sum(
        record["status"] != "trashed"
        for record in _catalog().list_sessions(folder_id=str(folder["id"]))
    )
    return item


def list_folders() -> list[dict[str, Any]]:
    if _uses_explicit_database_override():
        return _LOCAL_LIST_FOLDERS()
    return [_catalog_folder_with_count(row) for row in _catalog().list_folders()]


def rename_folder(folder_id: str, name: str) -> bool:
    if _uses_explicit_database_override():
        return _LOCAL_RENAME_FOLDER(folder_id, name)
    return _catalog().update_folder(folder_id, name=name) is not None


def delete_folder(folder_id: str) -> tuple[bool, str]:
    if _uses_explicit_database_override():
        return _LOCAL_DELETE_FOLDER(folder_id)
    return _catalog().delete_folder(folder_id)


def move_folder(
    folder_id: str,
    new_parent_id: str | None,
) -> tuple[bool, str]:
    if _uses_explicit_database_override():
        return _LOCAL_MOVE_FOLDER(folder_id, new_parent_id)
    if _catalog().get_folder(folder_id) is None:
        return False, "folder_not_found"
    if new_parent_id is not None and _catalog().get_folder(new_parent_id) is None:
        return False, "parent_not_found"
    try:
        updated = _catalog().update_folder(folder_id, parent_id=new_parent_id)
    except ValueError as exc:
        if "descendant" in str(exc):
            return False, "cannot_move_into_descendant"
        if "own parent" in str(exc):
            return False, "cannot_move_into_self"
        raise
    return (updated is not None, "moved" if updated is not None else "folder_not_found")


def _catalog_descendant_folder_ids(folder_id: str) -> list[str]:
    folders = {str(row["id"]): row for row in _catalog().list_folders()}
    if folder_id not in folders:
        return []
    selected = [folder_id]
    cursor = 0
    while cursor < len(selected):
        parent = selected[cursor]
        selected.extend(
            child_id
            for child_id, row in folders.items()
            if row.get("parent_id") == parent and child_id not in selected
        )
        cursor += 1
    return selected


def set_folder_status(
    folder_id: str,
    status: str,
) -> tuple[bool, str]:
    if _uses_explicit_database_override():
        return _LOCAL_SET_FOLDER_STATUS(folder_id, status)
    if status not in {"active", "archived", "trashed"}:
        return False, "invalid_status"
    folder_ids = _catalog_descendant_folder_ids(folder_id)
    if not folder_ids:
        return False, "folder_not_found"
    session_records_to_change = [
        record
        for child_id in folder_ids
        for record in _catalog().list_sessions(folder_id=child_id)
    ]
    for record in session_records_to_change:
        session_id = str(record["id"])
        current = str(record["status"])
        if status == "trashed" and current in {"active", "archived"}:
            trash_session(session_id)
        elif status == "archived" and current == "active":
            archive_session(session_id)
        elif status == "active" and current == "trashed":
            restore_session(session_id)
        elif status == "active" and current == "archived":
            unarchive_session(session_id)
    for child_id in folder_ids:
        _catalog().update_folder(child_id, status=status)
    return True, f"folder_{status}"


def folder_tree(status: str = "active") -> list[dict[str, Any]]:
    if _uses_explicit_database_override():
        return _LOCAL_FOLDER_TREE(status)
    if status not in {"active", "archived", "trashed", "all"}:
        raise ValueError(f"invalid folder status: {status}")
    rows = _catalog().list_folders(status=None if status == "all" else status)
    nodes = {
        str(row["id"]): {**_catalog_folder_with_count(row), "children": []}
        for row in rows
    }
    roots: list[dict[str, Any]] = []
    for node in nodes.values():
        parent = nodes.get(str(node.get("parent_id")))
        if parent is None:
            roots.append(node)
        else:
            parent["children"].append(node)
    return roots

_TURN_INPUT_STORE_FACADE = (
    turn_input_store_facade_records.build_turn_input_store_facade(
        deps_factory=lambda: _deps(),
    )
)
create_runtime_turn = _TURN_INPUT_STORE_FACADE.create_runtime_turn
complete_runtime_turn = _TURN_INPUT_STORE_FACADE.complete_runtime_turn
create_attachment = _TURN_INPUT_STORE_FACADE.create_attachment
get_attachment = _TURN_INPUT_STORE_FACADE.get_attachment
list_turn_attachments = _TURN_INPUT_STORE_FACADE.list_turn_attachments
list_unbound_attachments = _TURN_INPUT_STORE_FACADE.list_unbound_attachments
bind_attachments_to_turn = _TURN_INPUT_STORE_FACADE.bind_attachments_to_turn
delete_unbound_attachment = _TURN_INPUT_STORE_FACADE.delete_unbound_attachment
session_attachment_total_bytes = _TURN_INPUT_STORE_FACADE.session_attachment_total_bytes
list_pending_carry_over_turns = _TURN_INPUT_STORE_FACADE.list_pending_carry_over_turns
list_turn_input_segments = _TURN_INPUT_STORE_FACADE.list_turn_input_segments
attach_runtime_turn_carry_over = _TURN_INPUT_STORE_FACADE.attach_runtime_turn_carry_over
append_runtime_turn_event = _TURN_INPUT_STORE_FACADE.append_runtime_turn_event
list_runtime_turn_events = _TURN_INPUT_STORE_FACADE.list_runtime_turn_events


# 与 memory.store 同理：ThreadingHTTPServer 首启并发建表串行化，防 "database is locked"。
_INIT_LOCK = threading.Lock()
_INITIALIZED_PATHS: set[str] = set()


def init_db() -> None:
    database_path = _active_database_path()
    key = str(database_path)
    if key in _INITIALIZED_PATHS:
        return
    with _INIT_LOCK:
        if key in _INITIALIZED_PATHS:
            return
        _init_db_locked()
        _INITIALIZED_PATHS.add(key)


def _init_db_locked() -> None:
    database_path = _active_database_path()
    try:
        with _connect() as conn:
            initialize_schema(conn)
    except SessionStoreError:
        raise
    except sqlite3.Error as exc:
        raise SessionStoreError(
            f"Failed to initialize session database schema at {database_path}"
        ) from exc


def _create_partitioned_session(
    *,
    session_id: str,
    persona_id: str,
    title: str | None,
    folder_id: str | None,
    working_dir: str | None,
    project_id: str | None,
    require_workspace_read_authority: bool,
    creation_request_id: str | None = None,
) -> str:
    if folder_id is not None and _catalog().get_folder(folder_id) is None:
        raise ValueError(f"unknown session folder: {folder_id}")
    with session_database_scope(session_id):
        session_records.create_session(
            _deps(),
            persona_id,
            title,
            None,
            working_dir,
            session_id=session_id,
        )
    # 产品 API 的 Session 在公开 catalog locator 前，必须已经拥有可用的精确读取
    # authority。此时 session_database_scope 可按已声明分区路径工作，无需先发布 catalog。
    if working_dir is not None and require_workspace_read_authority:
        _grant_workspace_read_authority(
            session_id=session_id,
            working_dir=working_dir,
        )
    _catalog().create_session(
        session_id=session_id,
        project_id=project_id,
        title=title,
        persona_id=persona_id,
        folder_id=folder_id,
        creation_request_id=creation_request_id,
    )
    return session_id


def _claim_partitioned_session_directory(session_id: str) -> Path:
    """原子声明一个全新 Session 分区；已存在状态绝不属于本调用。"""

    catalog = _catalog()
    if (
        catalog.get_session(session_id) is not None
        or catalog.get_session_purge_tombstone(session_id) is not None
    ):
        raise SessionCreationClaimConflict(
            f"session id {session_id!r} already exists"
        )
    database_path = catalog.session_db_path(session_id)
    sessions_root = database_path.parent.parent
    sessions_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        database_path.parent.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        raise SessionCreationClaimConflict(
            f"session id {session_id!r} already exists"
        ) from exc
    # 再检查一次 catalog，覆盖首次检查与目录声明之间的并发发布。一旦另一方的
    # locator/tombstone 已经可见，该精确路径就属于它的权威状态；即使本调用赢得了
    # mkdir，也不得再删除目录，否则会让合法 locator 指向消失的载荷。
    if (
        catalog.get_session(session_id) is not None
        or catalog.get_session_purge_tombstone(session_id) is not None
    ):
        raise SessionCreationClaimConflict(
            f"session id {session_id!r} already exists"
        )
    return database_path.parent


def _rollback_unpublished_partitioned_session(session_id: str) -> None:
    """清除尚未返回给调用方的精确 Session 载荷与 locator。"""

    catalog = _catalog()
    database_path = catalog.session_db_path(session_id)
    if catalog.get_session(session_id) is not None:
        if not purge_session(session_id):
            raise SessionStoreError(
                f"unpublished session {session_id!r} could not be purged"
            )
    elif database_path.parent.exists():
        # 先持久声明清理意图，再触碰载荷。这样即使 ``rmtree`` 只完成一部分后失败，
        # 下次进程启动/创建 Session 也能发现并继续清理这个从未公开的分区。
        catalog.begin_unpublished_session_purge(session_id)
        try:
            shutil.rmtree(database_path.parent)
        except OSError:
            raise
        if not catalog.complete_session_purge(session_id):
            raise SessionStoreError(
                f"unpublished session {session_id!r} payload was purged but its "
                "cleanup tombstone could not be completed"
            )
    _INITIALIZED_PATHS.discard(str(database_path.resolve()))


def _retry_pending_unpublished_session_cleanups(*, limit: int = 16) -> int:
    """有界重试从未发布 Session 的残留分区，不阻塞无关新建操作。"""

    catalog = _catalog()
    cleaned = 0
    for tombstone in catalog.list_pending_unpublished_session_purges(limit=limit):
        session_id = str(tombstone["session_id"])
        database_path = catalog.session_db_path(session_id)
        if str(tombstone["db_path"]) != session_db_relative_path(session_id):
            continue
        try:
            if database_path.parent.exists():
                shutil.rmtree(database_path.parent)
            if catalog.complete_session_purge(session_id):
                _INITIALIZED_PATHS.discard(str(database_path.resolve()))
                cleaned += 1
        except (OSError, SessionCatalogError, sqlite3.Error):
            # durable pending 行保留；后续进程/创建入口会再次尝试。
            continue
    return cleaned


_MAX_SESSION_ID_CLAIM_ATTEMPTS = 8


def create_session(
    persona_id: str,
    title: str | None = None,
    folder_id: str | None = None,
    working_dir: str | None = None,
    *,
    require_workspace_read_authority: bool = False,
    creation_request_id: str | None = None,
) -> str:
    if not isinstance(require_workspace_read_authority, bool):
        raise TypeError("require_workspace_read_authority must be a bool")
    partitioned = not _uses_explicit_database_override()
    if creation_request_id is not None and not partitioned:
        raise SessionCreationNotOwnedError("idempotent creation requires partitioned Session state")
    if partitioned:
        try:
            _retry_pending_unpublished_session_cleanups()
        except Exception as exc:
            raise SessionCreationNotOwnedError(
                "session creation could not inspect pending cleanup claims"
            ) from exc
    session_id = ""
    session_created = False
    partitioned_payload_owned = False
    try:
        if not partitioned:
            session_id = session_records.create_session(
                _deps(), persona_id, title, folder_id, working_dir
            )
            session_created = True
        else:
            for attempt in range(_MAX_SESSION_ID_CLAIM_ATTEMPTS):
                session_id = _new_id()
                try:
                    _claim_partitioned_session_directory(session_id)
                except SessionCreationClaimConflict:
                    if attempt + 1 >= _MAX_SESSION_ID_CLAIM_ATTEMPTS:
                        raise
                    continue
                except Exception as exc:
                    # 只有已证明的 ID/claim 冲突才允许重抽。其他异常既可能是目录或
                    # catalog 故障，也可能不可重试；此时尚未取得分区所有权。
                    raise SessionCreationNotOwnedError(
                        "session creation failed before claiming a partition"
                    ) from exc
                partitioned_payload_owned = True
                break
            if not partitioned_payload_owned:
                raise SessionCreationNotOwnedError(
                    "session creation could not claim a new partition"
                )
            project_id: str | None = None
            if working_dir is not None:
                from . import project_catalog

                project_id = project_catalog.remember(working_dir).project_id
            _create_partitioned_session(
                session_id=session_id,
                persona_id=persona_id,
                title=title,
                folder_id=folder_id,
                working_dir=working_dir,
                project_id=project_id,
                creation_request_id=creation_request_id,
                require_workspace_read_authority=(
                    require_workspace_read_authority
                ),
            )
            session_created = True
        if working_dir is not None and not (
            partitioned and require_workspace_read_authority
        ):
            if require_workspace_read_authority:
                _grant_workspace_read_authority(
                    session_id=session_id,
                    working_dir=working_dir,
                )
            else:
                _record_workspace_read_authority(
                    session_id=session_id,
                    working_dir=working_dir,
                )
        return session_id
    except Exception:
        if partitioned and partitioned_payload_owned:
            try:
                _rollback_unpublished_partitioned_session(session_id)
            except Exception as rollback_error:
                raise SessionCreationRollbackError(session_id) from rollback_error
        elif session_created:
            session_records.purge_session(_deps(), session_id)
        raise


def grant_session_workspace_write_authority(session_id: str):
    """记录一次明确的人工批准，允许 agent 写入 workspace。

    绑定工作目录仍只授予 READ/SEARCH 权限。呈现用户批准 UI 的调用方使用这一狭窄
    Host 操作，允许针对该精确根标识的 ``UPDATE`` effect。
    """

    session = get_session(session_id)
    if session is None:
        raise ValueError("session does not exist")
    working_dir = session.get("working_dir")
    if not isinstance(working_dir, str) or not working_dir.strip():
        raise ValueError("session has no bound working directory")
    with session_database_scope(session_id):
        return _workspace_file_authority().grant_workspace_write(
            session_id=session_id,
            working_root=working_dir,
        )


def revoke_session_workspace_write_authority(
    session_id: str,
    *,
    grant_id: str | None = None,
) -> int:
    """撤销某 Session 的一项或全部显式 workspace 写入批准。"""

    with session_database_scope(session_id):
        authority = _workspace_file_authority()
        if grant_id is None:
            active = authority.list_write_grants(session_id=session_id)
            revoked = 0
            for grant in active:
                if grant.revoked_at is None:
                    revoked += authority.revoke(
                        session_id=session_id,
                        grant_id=grant.grant_id,
                    )
            return revoked
        return authority.revoke(session_id=session_id, grant_id=grant_id)


def _workspace_file_authority():
    """在显式 Host 根绑定接缝处解析狭窄账本。

    创建 Session 时写入的 ``working_dir`` 是唯一选择本地作用域的 Host 操作。其
    持久化 receipt 让后续模型读取无需逐文件询问。由于 Runtime 也消费 Session
    Store，此导入保持局部。
    """

    from .local_file_authority import SqliteSessionFileAuthority

    return SqliteSessionFileAuthority()


def _grant_workspace_read_authority(
    *,
    session_id: str,
    working_dir: str,
) -> None:
    with session_database_scope(session_id):
        _workspace_file_authority().grant_workspace_root(
            session_id=session_id,
            working_root=working_dir,
        )


def _record_workspace_read_authority(
    *,
    session_id: str,
    working_dir: str,
) -> None:
    try:
        _grant_workspace_read_authority(
            session_id=session_id,
            working_dir=working_dir,
        )
    except (OSError, RuntimeError, ValueError):
        # 低层 Store 历来接受未解析根目录。保留该兼容性，但绝不将不可用路径转成读取
        # 权威；产品 API 创建会要求 authority 同步成功，否则不会发布 Session。
        return


def list_turn_linked_work_run_ids(
    *,
    session_id: str,
    turn_id: str,
) -> tuple[str, ...]:
    """按持久化链接顺序返回所有精确的 Turn 关联 WorkRun id。"""

    return entry_task_records.list_turn_linked_work_run_ids(
        _deps(),
        session_id=session_id,
        turn_id=turn_id,
    )


def list_pending_user_questions(
    *,
    session_id: str,
) -> tuple[EntryPendingTaskQuestion, ...]:
    """投影 Entry/UI 可见、仍等待回答的当前任务问题。"""

    return entry_task_records.list_entry_pending_task_questions(
        _deps(),
        session_id=session_id,
    )


def list_insession_task_catalog(
    session_id: str,
    *,
    limit: int | None = None,
) -> tuple[EntryTaskCatalogItem, ...]:
    return entry_task_records.list_entry_task_catalog(
        _deps(), session_id, limit=limit
    )


def list_turn_insession_task_ids(
    session_id: str,
    turn_id: str,
) -> tuple[str, ...]:
    """读取为某 Turn 持久化的有序根 Task 关联。"""
    return entry_task_records.list_turn_insession_task_ids(
        _deps(), session_id, turn_id
    )


def accept_turn_execution(
    *,
    session_id: str,
    client_request_id: str,
    source: str,
    user_text: str,
    attachment_ids: tuple[str, ...] = (),
    turn_id: str | None = None,
    input_message_id: str | None = None,
    lease_owner: str | None = None,
    initial_stage: str = "INGRESS",
    routing_policy_source: str | None = None,
    routing_policy_snapshot_json: str | None = None,
    routing_policy_snapshot_hash: str | None = None,
    session_routing_policy_json: str | None = None,
    session_routing_policy_hash: str | None = None,
    execution_snapshot_json: str | None = None,
    execution_snapshot_sha256: str | None = None,
) -> dict[str, object]:
    """持久地接受一个新 Runtime Turn，或返回其幂等重放。

    这是 facade 到 session/persistence/turns/turn_execution.py 的组合接缝：
    _deps() 注入当前 scope 的连接、时钟与 ID 工厂，事务规则只由记录 owner 实现。
    追踪真实 SQL/幂等语义应继续跟到 turn_execution_records.accept_turn_execution。
    """

    return turn_execution_records.accept_turn_execution(
        _deps(),
        session_id=session_id,
        client_request_id=client_request_id,
        source=source,
        user_text=user_text,
        attachment_ids=attachment_ids,
        turn_id=turn_id,
        input_message_id=input_message_id,
        lease_owner=lease_owner,
        initial_stage=initial_stage,
        routing_policy_source=routing_policy_source,
        routing_policy_snapshot_json=routing_policy_snapshot_json,
        routing_policy_snapshot_hash=routing_policy_snapshot_hash,
        session_routing_policy_json=session_routing_policy_json,
        session_routing_policy_hash=session_routing_policy_hash,
        execution_snapshot_json=execution_snapshot_json,
        execution_snapshot_sha256=execution_snapshot_sha256,
    )


def get_session_turn_routing_policy(
    session_id: str,
) -> dict[str, object] | None:
    """读取最新的显式 Session 级路由选择（如有）。"""

    return turn_routing_policy_records.get_session_turn_routing_policy(
        _deps(),
        session_id=session_id,
    )


def create_l1_turn_run(
    *,
    session_id: str,
    turn_id: str,
    routing_policy_snapshot_hash: str,
    expected_window_revision: int,
    expected_lease_owner: str | None = None,
    l1_turn_run_id: str | None = None,
    deadline_at: str | None = None,
    max_attempts: int | None = None,
    max_tool_calls_per_attempt: int | None = None,
    catalog_snapshot_json: str | None = None,
    catalog_snapshot_hash: str | None = None,
    execution_config_json: str | None = None,
    execution_config_hash: str | None = None,
    corpus_manifest_contract_version: str | None = None,
    corpus_manifest_json: str | None = None,
    corpus_manifest_hash: str | None = None,
) -> dict[str, object]:
    """创建或重放一个 L1 聚合，并可选择执行原子初始化。"""

    return l1_turn_run_records.create_l1_turn_run(
        _deps(),
        session_id=session_id,
        turn_id=turn_id,
        routing_policy_snapshot_hash=routing_policy_snapshot_hash,
        expected_window_revision=expected_window_revision,
        expected_lease_owner=expected_lease_owner,
        l1_turn_run_id=l1_turn_run_id,
        deadline_at=deadline_at,
        max_attempts=max_attempts,
        max_tool_calls_per_attempt=max_tool_calls_per_attempt,
        catalog_snapshot_json=catalog_snapshot_json,
        catalog_snapshot_hash=catalog_snapshot_hash,
        execution_config_json=execution_config_json,
        execution_config_hash=execution_config_hash,
        corpus_manifest_contract_version=corpus_manifest_contract_version,
        corpus_manifest_json=corpus_manifest_json,
        corpus_manifest_hash=corpus_manifest_hash,
    )


def get_l1_turn_run(
    *,
    session_id: str,
    turn_id: str,
) -> dict[str, object] | None:
    return l1_turn_run_records.get_l1_turn_run(
        _deps(),
        session_id=session_id,
        turn_id=turn_id,
    )


def create_execution_findings_ledger(
    *,
    session_id: str,
    owner_kind: ExecutionFindingsOwnerKind | str,
    execution_owner_id: str,
    quota: ExecutionFindingsQuota | None = None,
) -> ExecutionFindingsLedgerCreateResult:
    """为一个执行单元创建或重放 Host 持有的账本。"""

    return execution_findings_records.create_execution_findings_ledger(
        _deps(),
        session_id=session_id,
        owner_kind=owner_kind,
        execution_owner_id=execution_owner_id,
        quota=quota,
    )


def get_execution_findings_ledger(
    *,
    ledger_id: str,
) -> ExecutionFindingsSnapshot | None:
    return execution_findings_records.get_execution_findings_ledger(
        _deps(),
        ledger_id=ledger_id,
    )


def get_execution_findings_ledger_for_owner(
    *,
    owner_kind: ExecutionFindingsOwnerKind | str,
    execution_owner_id: str,
) -> ExecutionFindingsSnapshot | None:
    return execution_findings_records.get_execution_findings_ledger_for_owner(
        _deps(),
        owner_kind=owner_kind,
        execution_owner_id=execution_owner_id,
    )


def apply_execution_findings_mutation(
    *,
    command: ExecutionFindingsMutationCommand,
) -> ExecutionFindingsMutationResult:
    return execution_findings_records.apply_execution_findings_mutation(
        _deps(),
        command=command,
    )


def close_execution_findings_ledger(
    *,
    ledger_id: str,
    expected_ledger_revision: int,
) -> ExecutionFindingsSnapshot:
    return execution_findings_records.close_execution_findings_ledger(
        _deps(),
        ledger_id=ledger_id,
        expected_ledger_revision=expected_ledger_revision,
    )


def claim_l1_turn_run_resume(
    *,
    session_id: str,
    turn_id: str,
    lease_owner: str,
    lease_seconds: int,
) -> dict[str, object]:
    """隔离并认领一个精确且未完成的 L1 request-id 重放。"""

    return l1_turn_run_records.claim_l1_turn_run_resume(
        _deps(),
        session_id=session_id,
        turn_id=turn_id,
        lease_owner=lease_owner,
        lease_seconds=lease_seconds,
    )


def renew_l1_turn_run_resume_lease(
    *,
    session_id: str,
    turn_id: str,
    l1_turn_run_id: str,
    lease_owner: str,
) -> bool:
    """为一个精确且由重放持有的 L1 执行租约续约心跳。"""

    return l1_turn_run_records.renew_l1_turn_run_resume_lease(
        _deps(),
        session_id=session_id,
        turn_id=turn_id,
        l1_turn_run_id=l1_turn_run_id,
        lease_owner=lease_owner,
    )


def initialize_l1_turn_run(**kwargs: object) -> dict[str, object]:
    return l1_turn_run_records.initialize_l1_turn_run(_deps(), **kwargs)


def start_l1_attempt(**kwargs: object) -> dict[str, object]:
    return l1_turn_run_records.start_l1_attempt(_deps(), **kwargs)


def get_l1_attempt_state_guard(**kwargs: object) -> str:
    return l1_turn_run_records.get_l1_attempt_state_guard(_deps(), **kwargs)


def commit_l1_attempt_decision(**kwargs: object) -> dict[str, object]:
    return l1_turn_run_records.commit_l1_attempt_decision(_deps(), **kwargs)


def reject_l1_final_reply_candidate(**kwargs: object) -> dict[str, object]:
    return l1_turn_run_records.reject_l1_final_reply_candidate(_deps(), **kwargs)


def reserve_l1_tool_call(**kwargs: object) -> dict[str, object]:
    return l1_turn_run_records.reserve_l1_tool_call(_deps(), **kwargs)


def begin_l1_protected_tool_dispatch(**kwargs: object) -> dict[str, object]:
    return l1_turn_run_records.begin_l1_protected_tool_dispatch(_deps(), **kwargs)


def settle_l1_tool_call(**kwargs: object) -> dict[str, object]:
    return l1_turn_run_records.settle_l1_tool_call(_deps(), **kwargs)


def settle_l1_protected_tool_dispatch(**kwargs: object) -> dict[str, object]:
    return l1_turn_run_records.settle_l1_protected_tool_dispatch(_deps(), **kwargs)


def close_l1_attempt(**kwargs: object) -> dict[str, object]:
    return l1_turn_run_records.close_l1_attempt(_deps(), **kwargs)


def get_l1_turn_execution(**kwargs: object) -> dict[str, object] | None:
    return l1_turn_run_records.get_l1_turn_execution(_deps(), **kwargs)


def build_l1_tool_history_reader(*, session_id: str, l1_turn_run_id: str) -> L1ToolHistoryReader:
    """绑定当前数据库和执行范围；真正读取始终使用只读连接，不初始化存储。"""
    return L1ToolHistoryReader(database_path=current_session_database_path(),
                               session_id=session_id, l1_turn_run_id=l1_turn_run_id)


def fail_l1_turn_run(**kwargs: object) -> None:
    l1_turn_run_records.fail_l1_turn_run(_deps(), **kwargs)


def get_turn_execution_window(session_id: str) -> dict[str, object] | None:
    """读取 Session 的持久化当前执行槽位。"""

    return turn_execution_records.get_turn_execution_window(_deps(), session_id)


def get_turn_execution_input(
    *,
    session_id: str,
    turn_id: str,
) -> dict[str, object]:
    """读取为某 Runtime Turn 接受的一份不可变用户输入。"""

    return turn_execution_records.get_turn_execution_input(
        _deps(), session_id=session_id, turn_id=turn_id
    )


def get_turn_execution_for_client_request(
    *,
    session_id: str,
    client_request_id: str,
) -> dict[str, object] | None:
    """通过公开幂等键读取一个现有已接受 Turn。"""

    return turn_execution_records.get_turn_execution_for_client_request(
        _deps(),
        session_id=session_id,
        client_request_id=client_request_id,
    )


def inspect_turn_execution(session_id: str) -> dict[str, object]:
    """读取存储层持有的当前执行槽位恢复事实。"""

    return turn_execution_records.inspect_turn_execution(_deps(), session_id)


def advance_turn_execution_window(
    *,
    session_id: str,
    turn_id: str,
    expected_window_revision: int,
    stage: str,
    lease_owner: str | None = None,
    last_event_sequence: int | None = None,
) -> dict[str, object]:
    """持久化某活动 Turn 的下一 Runtime 阶段。"""

    return turn_execution_records.advance_turn_execution_window(
        _deps(),
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=expected_window_revision,
        stage=stage,
        lease_owner=lease_owner,
        last_event_sequence=last_event_sequence,
    )


def mark_turn_execution_interrupted(
    *,
    session_id: str,
    turn_id: str,
    expected_window_revision: int,
    stage: str,
    interruption_reason: str | None,
    expected_heartbeat_at: str | None = None,
    require_heartbeat_match: bool = False,
    expected_lease_owner: str | None = None,
) -> dict[str, object]:
    """持久化已知停止标记；最终分类由下一 Turn 审计负责。"""

    return turn_execution_records.mark_turn_execution_interrupted(
        _deps(),
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=expected_window_revision,
        stage=stage,
        interruption_reason=interruption_reason,
        expected_heartbeat_at=expected_heartbeat_at,
        require_heartbeat_match=require_heartbeat_match,
        expected_lease_owner=expected_lease_owner,
    )


def settle_interrupted_turn_execution(
    *,
    session_id: str,
    turn_id: str,
    expected_window_revision: int,
    end_reason: str,
    error_code: str | None,
) -> dict[str, object]:
    """将一个已停止 Turn 审计为 ``incomplete``，并释放其旧 Window。"""

    return turn_execution_records.settle_interrupted_turn_execution(
        _deps(),
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=expected_window_revision,
        end_reason=end_reason,
        error_code=error_code,
    )


def finalize_turn_execution(
    *,
    session_id: str,
    turn_id: str,
    expected_window_revision: int,
    processing_level: str,
    assistant_content: str,
    post_commit_job_kinds: tuple[str, ...],
    expected_lease_owner: str | None = None,
    l1_terminal_failure_code: str | None = None,
) -> dict[str, object]:
    """以原子方式提交正式交付和所有必需的派生状态任务。"""

    return turn_execution_records.finalize_turn_execution(
        _deps(),
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=expected_window_revision,
        processing_level=processing_level,  # type: ignore[arg-type]
        assistant_content=assistant_content,
        post_commit_job_kinds=post_commit_job_kinds,
        expected_lease_owner=expected_lease_owner,
        l1_terminal_failure_code=l1_terminal_failure_code,
    )


def finalize_verified_turn_execution(
    *,
    session_id: str,
    turn_id: str,
    delivery_id: str,
    expected_window_revision: int,
    post_commit_job_kinds: tuple[str, ...],
    expected_lease_owner: str | None = None,
) -> dict[str, object]:
    """发布一个已验证 NodeDelivery，而不复制其冻结正文。"""

    return turn_execution_records.finalize_verified_turn_execution(
        _deps(),
        session_id=session_id,
        turn_id=turn_id,
        delivery_id=delivery_id,
        expected_window_revision=expected_window_revision,
        post_commit_job_kinds=post_commit_job_kinds,
        expected_lease_owner=expected_lease_owner,
    )


def finalize_verified_turn_deliveries(
    *,
    session_id: str,
    turn_id: str,
    delivery_ids: tuple[str, ...],
    expected_window_revision: int,
    post_commit_job_kinds: tuple[str, ...],
    pending_question_attempt_ids: tuple[str, ...] = (),
    expected_lease_owner: str | None = None,
) -> dict[str, object]:
    """发布有序 Delivery，随后发布所有待处理问题引用。"""

    return turn_execution_records.finalize_verified_turn_deliveries(
        _deps(),
        session_id=session_id,
        turn_id=turn_id,
        delivery_ids=delivery_ids,
        pending_question_attempt_ids=pending_question_attempt_ids,
        expected_window_revision=expected_window_revision,
        post_commit_job_kinds=post_commit_job_kinds,
        expected_lease_owner=expected_lease_owner,
    )


def finalize_referenced_turn_execution(
    *,
    session_id: str,
    turn_id: str,
    reference_items: tuple[FormalTurnReference, ...],
    expected_window_revision: int,
    post_commit_job_kinds: tuple[str, ...],
    expected_lease_owner: str | None = None,
) -> dict[str, object]:
    """发布一个精确、按来源排序的混合正式引用元组。"""

    return turn_execution_records.finalize_referenced_turn_execution(
        _deps(),
        session_id=session_id,
        turn_id=turn_id,
        reference_items=reference_items,
        expected_window_revision=expected_window_revision,
        post_commit_job_kinds=post_commit_job_kinds,
        expected_lease_owner=expected_lease_owner,
    )


def finalize_authoritative_referenced_turn_execution(
    *,
    session_id: str,
    turn_id: str,
    expected_window_revision: int,
    post_commit_job_kinds: tuple[str, ...],
    expected_lease_owner: str | None = None,
) -> dict[str, object]:
    """以原子方式推导并发布精确正式引用元组。"""

    return turn_execution_records.finalize_authoritative_referenced_turn_execution(
        _deps(),
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=expected_window_revision,
        post_commit_job_kinds=post_commit_job_kinds,
        expected_lease_owner=expected_lease_owner,
    )


def mark_authoritative_no_public_turn_stop(
    *,
    session_id: str,
    turn_id: str,
    expected_window_revision: int,
) -> dict[str, object]:
    """证明所有执行 lane 均已停止且没有公开正文，并予以标记。"""

    return turn_execution_records.mark_authoritative_no_public_turn_stop(
        _deps(),
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=expected_window_revision,
    )


def list_turn_post_commit_jobs(turn_id: str) -> list[dict[str, object]]:
    return post_commit_job_records.list_turn_post_commit_jobs(_deps(), turn_id)


def claim_due_turn_post_commit_jobs(
    *,
    session_id: str,
    worker_id: str,
    lease_seconds: int,
    limit: int = 16,
) -> list[dict[str, object]]:
    return post_commit_job_records.claim_due_turn_post_commit_jobs(
        _deps(),
        session_id=session_id,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        limit=limit,
    )


def mark_turn_post_commit_job_applied(
    *,
    job_id: str,
    worker_id: str,
) -> dict[str, object]:
    return post_commit_job_records.mark_turn_post_commit_job_applied(
        _deps(), job_id=job_id, worker_id=worker_id
    )


def reconcile_turn_post_commit_job_applied(
    *,
    session_id: str,
    turn_id: str,
    job_id: str,
    expected_job_kind: str,
) -> dict[str, object]:
    """Settle a replay-safe post-commit job after re-proving its exact effect."""

    return post_commit_job_records.reconcile_turn_post_commit_job_applied(
        _deps(),
        session_id=session_id,
        turn_id=turn_id,
        job_id=job_id,
        expected_job_kind=expected_job_kind,
    )


def mark_turn_post_commit_job_failed(
    *,
    job_id: str,
    worker_id: str,
    reason_code: str,
    retry_after_seconds: int | None,
) -> dict[str, object]:
    return post_commit_job_records.mark_turn_post_commit_job_failed(
        _deps(),
        job_id=job_id,
        worker_id=worker_id,
        reason_code=reason_code,
        retry_after_seconds=retry_after_seconds,
    )


def get_committed_turn_pair_for_post_commit(
    *,
    session_id: str,
    turn_id: str,
) -> dict[str, object] | None:
    """为派生状态 worker 返回一个精确 Runtime 对话记录 pair。"""

    return post_commit_job_records.get_committed_turn_pair_for_post_commit(
        _deps(),
        session_id=session_id,
        turn_id=turn_id,
    )


def get_session_summary_state(session_id: str) -> SessionSummaryState | None:
    """读取有类型且处于稳定边界的 session-summary 投影。"""

    return post_commit_job_records.get_session_summary_state(_deps(), session_id)


def list_committed_turn_pairs_for_summary(
    session_id: str,
    *,
    after_turn_id: str | None,
    retain_recent_pairs: int = session_summary_records.SESSION_SUMMARY_MIN_RETAIN_RECENT_PAIRS,
    limit: int = 32,
) -> tuple[SessionSummaryTurnPair, ...]:
    """在受保护热点历史之外读取一页有界时间序列。"""

    return post_commit_job_records.list_committed_turn_pairs_for_summary(
        _deps(),
        session_id,
        after_turn_id=after_turn_id,
        retain_recent_pairs=retain_recent_pairs,
        limit=limit,
    )


def commit_session_summary_post_commit_job(
    *,
    session_id: str,
    job_id: str,
    worker_id: str,
    expected_state_version: int,
    expected_summarized_through_turn_id: str | None,
    update: SessionSummaryUpdate,
) -> SessionSummaryState:
    """以原子方式持久化一个已验证 summary 更新并结算其任务。"""

    return post_commit_job_records.commit_session_summary_post_commit_job(
        _deps(),
        session_id=session_id,
        job_id=job_id,
        worker_id=worker_id,
        expected_state_version=expected_state_version,
        expected_summarized_through_turn_id=expected_summarized_through_turn_id,
        update=update,
    )


def fail_session_summary_post_commit_job(
    *,
    session_id: str,
    job_id: str,
    worker_id: str,
    expected_state_version: int,
    expected_summarized_through_turn_id: str | None,
    status: SessionSummaryStatus,
    reason_code: str,
    retry_after_seconds: int | None,
) -> SessionSummaryState:
    """以原子方式记录 summary 降级并结算其 worker 持有任务。"""

    return post_commit_job_records.fail_session_summary_post_commit_job(
        _deps(),
        session_id=session_id,
        job_id=job_id,
        worker_id=worker_id,
        expected_state_version=expected_state_version,
        expected_summarized_through_turn_id=expected_summarized_through_turn_id,
        status=status,
        reason_code=reason_code,
        retry_after_seconds=retry_after_seconds,
    )


def apply_turn_post_commit_job_control(
    *,
    session_id: str,
    turn_id: str,
    expected_window_revision: int,
    request_id: str,
    action: str,
    job_ids: tuple[str, ...],
    expected_failed_job_digest: str,
    actor: str,
) -> dict[str, object]:
    return post_commit_job_records.apply_turn_post_commit_job_control(
        _deps(),
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=expected_window_revision,
        request_id=request_id,
        action=action,  # type: ignore[arg-type]
        job_ids=job_ids,
        expected_failed_job_digest=expected_failed_job_digest,
        actor=actor,
    )


def release_turn_execution_window(
    *,
    session_id: str,
    turn_id: str,
    expected_window_revision: int,
) -> dict[str, object]:
    """所有提交后工作结算后，释放 Session 以供下一 Turn 使用。"""

    return post_commit_job_records.release_turn_execution_window(
        _deps(),
        session_id=session_id,
        turn_id=turn_id,
        expected_window_revision=expected_window_revision,
    )


# ---------------------------------------------------------------------------
# P2 会话管理：重命名 / 归档 / 软删(回收站) / 恢复 / 移动 / 物理清除
# ---------------------------------------------------------------------------

def _exists(session_id: str) -> bool:
    return get_session(session_id) is not None


def export_session(session_id: str, fmt: str = "md") -> str | None:
    """将 session 及其 turn 渲染为 markdown 或 json。"""
    sess = get_session(session_id)
    if sess is None:
        return None
    turns = get_turns(session_id)
    from .context import evidence
    evidence_events = [item.to_dict() for item in evidence.list_event_evidence(session_id)]
    public_session = {
        key: value for key, value in sess.items() if key != "persona_id"
    }
    fmt = fmt.lower()
    if fmt == "json":
        return json.dumps(
            {
                "session": public_session,
                "turns": turns,
                "evidence_events": evidence_events,
            },
            ensure_ascii=False,
            indent=2,
        )
    if fmt != "md":
        raise ValueError(f"unsupported session export format: {fmt}")
    lines = [
        f"# {sess.get('title') or 'Untitled Session'}",
        "",
        f"- session_id: `{sess['id']}`",
        f"- status: `{sess['status']}`",
        f"- folder_id: `{sess.get('folder_id') or '-'}`",
        f"- created_at: `{sess.get('created_at') or '-'}`",
        f"- last_active_at: `{sess.get('last_active_at') or '-'}`",
        "",
        "## Turns",
        "",
    ]
    for turn in turns:
        who = "User" if turn["role"] == "user" else "Assistant"
        lines.extend([
            f"### {turn['turn_idx']}. {who}",
            "",
            turn["content"],
            "",
        ])
    if evidence_events:
        lines.extend(["## Evidence Events", ""])
        for event in evidence_events:
            lines.append(
                f"- {event['kind']} | {event['source_ref']} | {event['content_excerpt']}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
