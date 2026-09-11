"""Session-authorized exact File inventory for published Picture observations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import hashlib
import json

from ...retrieval.sources.picture import PictureFileAccessDecision, PictureFileAccessAuthorityPort
from ...retrieval.tooling.contracts import FrozenFileVersionBinding
from ...workspace.documents.mounting import mounted_document_ids
from ...workspace.documents.reading import get_current_document_file
from ...workspace.storage.context import require_current


@dataclass(frozen=True, slots=True)
class PictureFileInventorySnapshot:
    bindings: tuple[FrozenFileVersionBinding, ...]
    complete: bool
    authority_snapshot_id: str | None = field(default=None, repr=False)

    def __post_init__(self):
        values = tuple(self.bindings)
        if any(not isinstance(item, FrozenFileVersionBinding) for item in values):
            raise TypeError("Picture inventory requires exact File versions")
        if len(values) != len(set(values)):
            raise ValueError("Picture inventory contains duplicate File versions")
        if not isinstance(self.complete, bool):
            raise TypeError("Picture inventory completeness must be boolean")
        if self.complete and (
            not isinstance(self.authority_snapshot_id, str)
            or not self.authority_snapshot_id.startswith("picture_access_")
            or len(self.authority_snapshot_id) != len("picture_access_") + 64
        ):
            raise ValueError("complete Picture inventory requires a snapshot")
        if not self.complete and (values or self.authority_snapshot_id is not None):
            raise ValueError("incomplete Picture inventory must fail closed")
        object.__setattr__(self, "bindings", values)


@dataclass(frozen=True, slots=True)
class PictureRetrievalFileAuthority(PictureFileAccessAuthorityPort):
    """No project scan: only explicitly authorized files and Session mounts.

    The injected inventory must resolve current Session attachment/workspace
    grants on every call. IDs are identity, never an independent access grant.
    """

    session_id: str = field(repr=False)
    authorized_file_bindings: Callable[[], Iterable[FrozenFileVersionBinding]] = field(repr=False)
    mounted_ids: Callable = field(default=mounted_document_ids, repr=False)
    read_mounted_file: Callable = field(default=get_current_document_file, repr=False)

    def __post_init__(self):
        if not self.session_id or not all(callable(value) for value in (
            self.authorized_file_bindings, self.mounted_ids, self.read_mounted_file,
        )):
            raise ValueError("Picture authority requires explicit Session inventory ports")

    def freeze_inventory(self):
        try:
            project_id = require_current().project_id
            values = set(self.authorized_file_bindings())
            if any(not isinstance(item, FrozenFileVersionBinding) or item.project_id != project_id for item in values):
                raise ValueError("Picture inventory crossed Project authority")
            for document_id in self.mounted_ids(self.session_id):
                document = self.read_mounted_file(document_id=document_id, session_id=self.session_id)
                if document is None:
                    raise ValueError("mounted Document has no current File lineage")
                values.add(FrozenFileVersionBinding(
                    project_id=project_id, file_id=document.file_id,
                    file_version_id=document.file_version_id,
                ))
            bindings = tuple(sorted(values, key=lambda item: (item.project_id, item.file_id, item.file_version_id)))
            material = json.dumps([self.session_id, [[item.project_id, item.file_id, item.file_version_id] for item in bindings]], separators=(",", ":"))
            return PictureFileInventorySnapshot(
                bindings=bindings, complete=True,
                authority_snapshot_id="picture_access_" + hashlib.sha256(material.encode()).hexdigest(),
            )
        except Exception:
            return PictureFileInventorySnapshot(bindings=(), complete=False)

    def authorize(self, *, session_id, file_id, file_version_id, picture_id):
        inventory = self.freeze_inventory()
        allowed = session_id == self.session_id and inventory.complete and any(
            item.file_id == file_id and item.file_version_id == file_version_id
            for item in inventory.bindings
        )
        return PictureFileAccessDecision(
            session_id=session_id, file_id=file_id, file_version_id=file_version_id,
            picture_id=picture_id, allowed=allowed,
            authority_snapshot_id=inventory.authority_snapshot_id if allowed else None,
            reason_code=None if allowed else "picture_file_access_denied",
        )


__all__ = ["PictureFileInventorySnapshot", "PictureRetrievalFileAuthority"]
