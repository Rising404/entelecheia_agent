"""Picture access comes from explicit current Session inventory, never a scan."""

from personagraph.retrieval.tooling.contracts import FrozenFileVersionBinding
from personagraph.tools.retrieval.picture_file_authority import PictureRetrievalFileAuthority
from personagraph.workspace.storage.context import bind
from personagraph.workspace.storage.database import DocumentDatabase


def test_inventory_membership_changes_and_session_revocation_are_rechecked(tmp_path):
    database=DocumentDatabase("project-a",tmp_path,tmp_path/"documents.sqlite")
    values=[FrozenFileVersionBinding("project-a","file-a","fv-a")]
    authority=PictureRetrievalFileAuthority("session-a",lambda:tuple(values),mounted_ids=lambda _:())
    with bind(database):
        first=authority.freeze_inventory()
        assert authority.authorize(session_id="session-a",file_id="file-a",file_version_id="fv-a",picture_id=None).allowed
        assert not authority.authorize(session_id="other-session",file_id="file-a",file_version_id="fv-a",picture_id=None).allowed
        values[:]=[FrozenFileVersionBinding("project-a","file-a","fv-b")]
        second=authority.freeze_inventory()
        assert first.authority_snapshot_id != second.authority_snapshot_id
        assert not authority.authorize(session_id="session-a",file_id="file-a",file_version_id="fv-a",picture_id=None).allowed
        values.clear()
        assert not authority.authorize(session_id="session-a",file_id="file-a",file_version_id="fv-b",picture_id=None).allowed
    assert not database.db_path.exists()


def test_cross_project_and_unavailable_inventory_fail_closed(tmp_path):
    database=DocumentDatabase("project-a",tmp_path,tmp_path/"documents.sqlite")
    authority=PictureRetrievalFileAuthority("session-a",lambda:(FrozenFileVersionBinding("other","file-a","fv-a"),),mounted_ids=lambda _:())
    with bind(database):
        snapshot=authority.freeze_inventory()
    assert snapshot.complete is False and snapshot.bindings==()
