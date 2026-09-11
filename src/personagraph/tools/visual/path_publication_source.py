"""把路径视觉问答接到现有 File 权威；只登记当前文件版本，不解析文档。"""

from collections.abc import Callable

from ...workspace.files import WorkspaceFileAuthority
from ...workspace.files.access import AuthorizedFileSource, FileAccess, FileAccessError
from ...workspace.storage.database import DocumentDatabase


def build_visual_path_resolver(
    *, database: DocumentDatabase, validate_path: Callable[[str], bool],
) -> Callable[[str], AuthorizedFileSource]:
    access = FileAccess(database=database, validate_path=validate_path)
    authority = WorkspaceFileAuthority(database)

    def resolve(relative_path: str) -> AuthorizedFileSource:
        source = access.resolve_path(relative_path)
        if source.file_version_id is None:
            # 未登记或字节已变时，只建立 File/版本身份；没有 prepare_files 调用。
            authority.ensure_current_path(
                source.relative_path, source=source.origin,
                file_id=source.file_id, media_type=source.media_type,
            )
            current = access.resolve_path(relative_path)
            if current.fingerprint != source.fingerprint:
                raise FileAccessError("file_content_changed")
            source = current
        if not source.file_id or not source.file_version_id or not access.revalidate(source):
            raise FileAccessError("file_content_changed")
        return source

    return resolve
