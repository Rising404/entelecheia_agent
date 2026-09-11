"""文件工具的当前 Session 访问适配；复用既有根授予和托管文件权威。"""

from dataclasses import dataclass
from pathlib import Path

from personagraph.session import store as session_store
from personagraph.session.local_file_authority import SqliteSessionFileAuthority
from personagraph.session.workspace_authority import validate_current_session_workspace_authority
from ..workspace.workspace_tools import FrozenWorkspaceToolBoundary


@dataclass(frozen=True, slots=True)
class SessionFileToolAuthority:
    session_id: str
    boundary: FrozenWorkspaceToolBoundary
    grant_id: str

    def validate_current(self) -> bool:
        try:
            self.boundary.require_current_root()
            session = session_store.get_session(self.session_id)
            if (
                session is None or session.get("status") == "trashed"
                or not session.get("working_dir")
                or Path(session["working_dir"]).resolve() != self.boundary.root
            ):
                return False
            return any(
                grant.grant_id == self.grant_id and grant.revoked_at is None
                and grant.canonical_root == str(self.boundary.root)
                and grant.root_device == self.boundary.root_device
                and grant.root_inode == self.boundary.root_inode
                for grant in SqliteSessionFileAuthority().list_grants(session_id=self.session_id)
            )
        except (OSError, RuntimeError, ValueError):
            return False

    def validate_path(self, canonical_path: str) -> bool:
        if not self.validate_current():
            return False
        try:
            # 上传文件已归 Project 的“附件/”；沿用显式根授予，不以 File ID 授权。
            # 保留目录 .personagraph 不会继承根读取权限。
            return validate_current_session_workspace_authority(self.session_id, canonical_path)
        except (OSError, RuntimeError, ValueError):
            return False

    def validate_source(self, session_id: str, canonical_path: str) -> bool:
        return session_id == self.session_id and self.validate_path(canonical_path)
