"""Workspace 文件身份、不可变版本与上传准入的窄入口。"""

from .admission import (
    WorkspaceFileAuthority,
    find_current_file_in_connection,
    get_file_with_version_in_transaction,
)
from .contracts import (
    FileRegistrationConflict,
    FileSource,
    ProjectFileChangedError,
    ProjectFileError,
    ProjectFilePathError,
    ProjectFileSizeLimitError,
    ProjectFileRecord,
    ProjectFileVersionRecord,
    RegisteredProjectFileVersion,
)
from .observation import normalize_project_relative_path
from .outputs import ProjectOutputConflict, ProjectOutputError, ProjectOutputService
from .uploads import (
    ProjectUploadConflict,
    ProjectUploadError,
    ProjectUploadPathError,
    ProjectUploadService,
    StoredProjectUpload,
)


__all__ = [
    "FileRegistrationConflict",
    "FileSource",
    "ProjectFileChangedError",
    "ProjectFileError",
    "ProjectFilePathError",
    "ProjectFileSizeLimitError",
    "ProjectFileRecord",
    "WorkspaceFileAuthority",
    "find_current_file_in_connection",
    "get_file_with_version_in_transaction",
    "ProjectFileVersionRecord",
    "ProjectUploadConflict",
    "ProjectUploadError",
    "ProjectUploadPathError",
    "ProjectUploadService",
    "RegisteredProjectFileVersion",
    "StoredProjectUpload",
    "normalize_project_relative_path",
    "ProjectOutputConflict",
    "ProjectOutputError",
    "ProjectOutputService",
]
