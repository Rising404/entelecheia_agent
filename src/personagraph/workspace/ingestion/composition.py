"""默认文件准备的跨域组合：注入 Session 权限、Project 路由与检索实现。

仅本组合边界依赖 Retrieval 的公开构造工厂；worker、owner 与持久 job 不负责
选择模型或 backend，也不从 Retrieval 借用文件/会话操作。
"""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import uuid

from personagraph.session import project_catalog, store as session_store
from personagraph.session.workspace_authority import (
    validate_current_session_workspace_authority,
)
from personagraph.workspace.documents import application as docstore
from personagraph.workspace.files import FileSource, WorkspaceFileAuthority
from personagraph.workspace.storage.context import (
    ProjectDocumentContextError,
    bind as bind_project_documents,
    current as current_project_documents,
)
from personagraph.workspace.storage.database import DocumentDatabase
from personagraph.retrieval.operations.document_maintenance import (
    build_document_ingestion_worker,
    build_document_retrieval_composition,
)
from personagraph.retrieval.ports import BgeM3EncoderPort, RerankerPort
from personagraph.retrieval.profile import DocumentRetrievalProfile
from . import execution as ingestion_execution
from .execution import DocumentIngestExecutionOwner
from .delivery import FilePreparationDelivery
from .lifecycle import DocumentMaintenanceWorkerLifecycle
from .supervisor import (
    ProjectDocumentMaintenanceSupervisor,
    ProjectLifecycleBuilder,
    ProjectLoader,
)
from .worker import DocumentMaintenanceWorker


def build_document_maintenance_lifecycle(
    *,
    worker_id: str | None = None,
    retrieval_db_path: Path | str | None = None,
    profile: DocumentRetrievalProfile | None = None,
    encoder: BgeM3EncoderPort | None = None,
    reranker: RerankerPort | None = None,
    after_pass: Callable[[], object] | None = None,
) -> DocumentMaintenanceWorkerLifecycle:
    """基于持久化 ingest 与 Outbox 状态构建一个进程内轮询器。

    构造过程会初始化 SQLite schema 并为已配置 encoder 生成指纹，但有意不执行索引。
    产品默认使用严格、本地且固定 revision 的 BGE-M3 hybrid generation。轻量测试或
    紧急回滚必须显式选择 lexical profile；缺失资产在构造期失败，不会由请求触发下载。
    """

    composition = build_document_retrieval_composition(
        retrieval_db_path=retrieval_db_path,
        profile=profile,
        encoder=encoder,
        reranker=reranker,
    )
    project_database = current_project_documents()
    if project_database is None:
        raise ProjectDocumentContextError(
            "document maintenance requires a bound project database"
        )
    resolved_worker_id = worker_id or (
        f"api-document-maintenance:{os.getpid()}:{uuid.uuid4().hex}"
    )
    worker = build_document_ingestion_worker(
        composition,
        worker_id=resolved_worker_id,
        connect_documents=project_database.open_connection,
        validate_source_authority=validate_current_session_workspace_authority,
        link_project_file=_project_file_linker(project_database),
        request_delivery=_request_delivery(project_database).deliver_pending,
        job_scope_factory=_background_job_session_scope,
    )
    lifecycle = DocumentMaintenanceWorkerLifecycle(
        worker,
        after_pass=after_pass,
        scope_factory=lambda: bind_project_documents(project_database),
        on_started=lambda current: ingestion_execution.register_background_ingest_owner(
            project_database,
            worker=worker,
            lifecycle=current,
        ),
        on_stopped=lambda current: ingestion_execution.unregister_background_ingest_owner(
            project_database,
            lifecycle=current,
        ),
    )
    return lifecycle


def _background_job_session_scope(session_id: str):
    """在 lifecycle 已绑定 Project 的前提下补充单个 job 的 Session 路由。"""

    return session_store.session_database_route_scope(session_id)


def _request_delivery(database: DocumentDatabase) -> FilePreparationDelivery:
    return FilePreparationDelivery(
        connect_documents=database.open_connection,
        validate_source_authority=validate_current_session_workspace_authority,
        mount_document=docstore.mount_document,
        is_mounted=docstore.is_mounted,
        session_scope_factory=_background_job_session_scope,
    )


def _project_file_linker(
    database: DocumentDatabase,
) -> Callable[[str, str | None], tuple[str, str]]:
    """绑定一个 Project 的文件登记能力，供 ingestion worker 显式调用。"""

    root = database.project_root.expanduser().resolve()

    def link(
        canonical_path: str,
        media_type: str | None,
    ) -> tuple[str, str]:
        path = Path(canonical_path).expanduser().resolve()
        relative_path = path.relative_to(root).as_posix()
        registration = WorkspaceFileAuthority(database).ensure_current_path(
            relative_path,
            source=FileSource.WORKSPACE_EXISTING,
            media_type=media_type,
        )
        return (
            registration.file.file_id,
            registration.version.file_version_id,
        )

    return link


def build_project_document_maintenance_supervisor(
    *,
    project_loader: ProjectLoader | None = None,
    lifecycle_builder: ProjectLifecycleBuilder | None = None,
    poll_interval_seconds: float = 1.0,
    profile: DocumentRetrievalProfile | None = None,
    encoder: BgeM3EncoderPort | None = None,
    reranker: RerankerPort | None = None,
    after_pass_factory: Callable[[DocumentDatabase], Callable[[], object]] | None = None,
) -> ProjectDocumentMaintenanceSupervisor:
    """构造 API Host 所拥有的逐 Project 文档维护 supervisor。"""

    if project_loader is None:
        project_loader = project_catalog.list_registered_projects

    if lifecycle_builder is None:

        def lifecycle_builder(project, database: DocumentDatabase):
            if current_project_documents() is not database:
                raise RuntimeError(
                    "project maintenance lifecycle must be built inside its project scope"
                )
            return build_document_maintenance_lifecycle(
                worker_id=(
                    f"api-document-maintenance:{os.getpid()}:"
                    f"{project.project_id}:{uuid.uuid4().hex}"
                ),
                profile=profile,
                encoder=encoder,
                reranker=reranker,
                after_pass=(after_pass_factory(database) if after_pass_factory else None),
            )

    return ProjectDocumentMaintenanceSupervisor(
        project_loader=project_loader,
        lifecycle_builder=lifecycle_builder,
        poll_interval_seconds=poll_interval_seconds,
    )



def resolve_document_ingest_owner() -> DocumentIngestExecutionOwner:
    """为当前 Project 注入默认检索配方，不拥有第二份执行状态。"""

    database = _require_project_database()
    profile = DocumentRetrievalProfile.from_environment()
    return ingestion_execution.resolve_document_ingest_owner(
        database,
        profile_key=profile.fingerprint(),
        build_worker=lambda: _build_synchronous_ingest_worker(database, profile),
    )


def synchronous_ingest_worker() -> DocumentMaintenanceWorker:
    """离线入口使用同一 Project owner，默认配方只在组合边界解析。"""

    database = _require_project_database()
    profile = DocumentRetrievalProfile.from_environment()
    return ingestion_execution.synchronous_ingest_worker(
        database,
        profile_key=profile.fingerprint(),
        build_worker=lambda: _build_synchronous_ingest_worker(database, profile),
    )


def _require_project_database() -> DocumentDatabase:
    database = current_project_documents()
    if database is None:
        raise ProjectDocumentContextError(
            "document ingest owner requires a bound project database"
        )
    return database


def _build_synchronous_ingest_worker(
    project_database: DocumentDatabase,
    profile: DocumentRetrievalProfile,
) -> DocumentMaintenanceWorker:
    composition = build_document_retrieval_composition(profile=profile)
    return build_document_ingestion_worker(
        composition,
        worker_id=f"synchronous-ingest:{os.getpid()}:{uuid.uuid4().hex}",
        connect_documents=project_database.open_connection,
        validate_source_authority=validate_current_session_workspace_authority,
        link_project_file=_project_file_linker(project_database),
        request_delivery=_request_delivery(project_database).deliver_pending,
        job_scope_factory=_background_job_session_scope,
    )


__all__ = [
    "build_document_maintenance_lifecycle",
    "build_project_document_maintenance_supervisor",
    "resolve_document_ingest_owner",
    "synchronous_ingest_worker",
]
