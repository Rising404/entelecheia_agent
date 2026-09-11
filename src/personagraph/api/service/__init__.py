"""按 API 领域分组的稳定 HTTP 应用服务 facade。

普通 Session/L1 API 导入不得初始化已暂停的 L2 执行栈。
"""

from .errors import ApiError
from .appearance import (
    clear_background,
    get_background,
    read_background_asset,
    store_background,
)
from .projects import (
    create_project,
    forget_project,
    list_projects,
    pin_project,
    reorder_projects,
    rename_project,
)
from .session_titles import name_session_from_turn
from .post_commit import control_turn_post_commit_jobs
from .attachments import (
    FILENAME_HEADER,
    decode_upload_filename,
    delete_session_attachment,
    list_session_attachments,
    upload_attachment,
)
from .status import (
    activate_model_profile,
    create_model_profile,
    delete_model_profile,
    get_config,
    list_model_profiles,
    reveal_model_profile_secret,
    system_status,
    update_config,
    update_model_profile,
)
from .sessions import (
    empty_session_trash,
    purge_session,
    chat_turn, create_folder, create_session, delete_folder, get_runtime_events,
    get_insession_task_details, get_session, list_folders,
    list_sessions, patch_folder, patch_session,
)
from .session_context import (
    apply_session_context_repair, clear_session_context,
    create_session_context_correction, explain_session_context_state,
    export_session_context, get_session_context, preview_session_context_repair,
)
from .workspace_documents import (
    delete_document,
    detach_document,
    get_document,
    list_documents,
    patch_document,
)
from .document_ingest import (
    enqueue_document_ingest_job,
    get_document_ingest_job,
    list_document_ingest_jobs,
    retry_document_ingest_job,
)
from .workspace_files import (
    create_workspace_entry,
    delete_workspace_entry,
    list_workspace_files,
    read_workspace_file,
    write_workspace_file,
)


__all__ = [
    "ApiError",
    "FILENAME_HEADER",
    "activate_model_profile",
    "apply_session_context_repair",
    "chat_turn",
    "clear_background",
    "clear_session_context",
    "create_folder",
    "create_model_profile",
    "create_project",
    "create_session",
    "create_session_context_correction",
    "create_workspace_entry",
    "control_turn_post_commit_jobs",
    "decode_upload_filename",
    "delete_document",
    "delete_folder",
    "delete_model_profile",
    "delete_session_attachment",
    "delete_workspace_entry",
    "detach_document",
    "empty_session_trash",
    "enqueue_document_ingest_job",
    "explain_session_context_state",
    "export_session_context",
    "forget_project",
    "get_background",
    "get_config",
    "get_document",
    "get_document_ingest_job",
    "get_insession_task_details",
    "get_runtime_events",
    "get_session",
    "get_session_context",
    "list_document_ingest_jobs",
    "list_documents",
    "list_folders",
    "list_model_profiles",
    "list_projects",
    "list_session_attachments",
    "list_sessions",
    "list_workspace_files",
    "name_session_from_turn",
    "patch_document",
    "patch_folder",
    "patch_session",
    "pin_project",
    "preview_session_context_repair",
    "purge_session",
    "read_background_asset",
    "read_workspace_file",
    "rename_project",
    "reorder_projects",
    "retry_document_ingest_job",
    "reveal_model_profile_secret",
    "store_background",
    "system_status",
    "update_config",
    "update_model_profile",
    "upload_attachment",
    "write_workspace_file",
]
