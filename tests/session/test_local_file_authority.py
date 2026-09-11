"""持久会话本地文件读取权威的窄边界。"""

from __future__ import annotations

from pathlib import Path

import pytest

from personagraph.workspace.files import attachments as storage
from personagraph.workspace.storage.context import current
from personagraph.workspace.files import (
    FileSource,
    WorkspaceFileAuthority,
)
from personagraph.session.local_file_authority import (
    LocalFileAction,
    LocalFileAuthorizationReason,
    SqliteSessionFileAuthority,
)


@pytest.fixture
def project_session(tmp_path: Path, partitioned_project_state):
    from personagraph.session import store as session_store

    root = tmp_path / "project"
    root.mkdir()
    session_id = session_store.create_session(
        "Entelecheia",
        working_dir=str(root),
    )
    return session_id, root


@pytest.fixture
def ledger_path(tmp_path: Path) -> Path:
    return tmp_path / "session-file-authorizations.sqlite"


def _write(path: Path, text: str = "content") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_default_authority_honors_explicit_legacy_database_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from personagraph.session import store as session_store

    database_path = tmp_path / "legacy-session.sqlite"
    root = tmp_path / "workspace"
    root.mkdir()
    monkeypatch.setattr(session_store, "DB_PATH", database_path)

    authority = SqliteSessionFileAuthority()
    grant = authority.grant_workspace_root(session_id="s1", working_root=root)

    assert authority.path == database_path.resolve()
    assert grant.session_id == "s1"


def test_one_grant_covers_read_and_search_and_survives_restart(
    ledger_path: Path,
    tmp_path: Path,
) -> None:
    working_root = tmp_path / "workspace"
    document = _write(working_root / "papers" / "paper.pdf")
    first = SqliteSessionFileAuthority(ledger_path)

    before = first.authorize(
        session_id="s1",
        working_root=working_root,
        candidate=document,
        action=LocalFileAction.READ,
    )
    assert before.allowed is False
    assert before.reason is LocalFileAuthorizationReason.AUTHORIZATION_REQUIRED

    grant = first.grant_workspace_root(session_id="s1", working_root=working_root)
    restarted = SqliteSessionFileAuthority(ledger_path)
    read = restarted.authorize(
        session_id="s1",
        working_root=working_root,
        candidate=document,
        action=LocalFileAction.READ,
    )
    search = restarted.authorize(
        session_id="s1",
        working_root=working_root,
        candidate=working_root / "papers",
        action=LocalFileAction.SEARCH,
    )

    assert grant.actions == (LocalFileAction.READ, LocalFileAction.SEARCH)
    assert read.allowed is True
    assert read.grant_id == grant.grant_id
    assert search.allowed is True
    assert search.grant_id == grant.grant_id


def test_repeating_approval_for_the_same_root_is_idempotent(
    ledger_path: Path,
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    ledger = SqliteSessionFileAuthority(ledger_path)

    first = ledger.grant_workspace_root(session_id="s1", working_root=root)
    second = ledger.grant_workspace_root(session_id="s1", working_root=root)

    assert second == first
    assert len(ledger.list_grants(session_id="s1")) == 1


def test_a_grant_never_reaches_another_session_or_path_outside_the_root(
    ledger_path: Path,
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    inside = _write(root / "inside.md")
    outside = _write(tmp_path / "outside.md")
    ledger = SqliteSessionFileAuthority(ledger_path)
    ledger.grant_workspace_root(session_id="s1", working_root=root)

    other_session = ledger.authorize(
        session_id="s2",
        working_root=root,
        candidate=inside,
        action=LocalFileAction.READ,
    )
    outside_scope = ledger.authorize(
        session_id="s1",
        working_root=root,
        candidate=outside,
        action=LocalFileAction.READ,
    )

    assert other_session.allowed is False
    assert other_session.reason is LocalFileAuthorizationReason.AUTHORIZATION_REQUIRED
    assert outside_scope.allowed is False
    assert (
        outside_scope.reason
        is LocalFileAuthorizationReason.PATH_OUTSIDE_AUTHORIZED_SCOPE
    )


def test_observing_a_changed_working_root_permanently_revokes_the_old_grant(
    ledger_path: Path,
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    first_file = _write(first_root / "one.md")
    second_root = tmp_path / "second"
    second_file = _write(second_root / "two.md")
    ledger = SqliteSessionFileAuthority(ledger_path)
    old_grant = ledger.grant_workspace_root(session_id="s1", working_root=first_root)

    changed = ledger.authorize(
        session_id="s1",
        working_root=second_root,
        candidate=second_file,
        action=LocalFileAction.READ,
    )
    switched_back = ledger.authorize(
        session_id="s1",
        working_root=first_root,
        candidate=first_file,
        action=LocalFileAction.READ,
    )

    assert changed.allowed is False
    assert changed.reason is LocalFileAuthorizationReason.AUTHORIZATION_REQUIRED
    assert switched_back.allowed is False
    history = ledger.list_grants(session_id="s1")
    assert len(history) == 1
    assert history[0].grant_id == old_grant.grant_id
    assert history[0].revoked_at is not None


def test_replacing_the_directory_at_the_same_path_invalidates_approval(
    ledger_path: Path,
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    original_file = _write(root / "document.md")
    ledger = SqliteSessionFileAuthority(ledger_path)
    grant = ledger.grant_workspace_root(session_id="s1", working_root=root)

    old_directory = tmp_path / "old-workspace"
    root.rename(old_directory)
    replacement_file = _write(root / "document.md", "replacement")
    decision = ledger.authorize(
        session_id="s1",
        working_root=root,
        candidate=replacement_file,
        action=LocalFileAction.READ,
    )

    assert original_file.read_text(encoding="utf-8") == "replacement"
    assert (old_directory / "document.md").exists() is True
    assert root.stat().st_ino != grant.root_inode
    assert decision.allowed is False
    assert decision.reason is LocalFileAuthorizationReason.AUTHORIZATION_REQUIRED
    assert ledger.list_grants(session_id="s1")[0].revoked_at is not None


def test_revocation_takes_effect_and_persists(
    ledger_path: Path,
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    document = _write(root / "document.md")
    ledger = SqliteSessionFileAuthority(ledger_path)
    grant = ledger.grant_workspace_root(session_id="s1", working_root=root)

    assert ledger.revoke(session_id="s1", grant_id=grant.grant_id) == 1
    assert ledger.revoke(session_id="s1", grant_id=grant.grant_id) == 0
    decision = SqliteSessionFileAuthority(ledger_path).authorize(
        session_id="s1",
        working_root=root,
        candidate=document,
        action=LocalFileAction.SEARCH,
    )

    assert decision.allowed is False
    assert decision.reason is LocalFileAuthorizationReason.AUTHORIZATION_REQUIRED


def test_registered_user_and_agent_files_are_implicitly_local_readable(
    project_session,
    ledger_path: Path,
) -> None:
    from personagraph.session import store as session_store

    session_id, root = project_session
    with session_store.session_database_scope(session_id):
        stored = storage.store_attachment(
            session_id=session_id,
            attachment_id="att_1",
            raw_name="paper.pdf",
            payload=b"project evidence",
        )
        agent_output = _write(root / "draft.md")
        database = current()
        assert database is not None
        WorkspaceFileAuthority(database).register_path(
            "draft.md",
            source=FileSource.AGENT_OUTPUT,
        )
        ledger = SqliteSessionFileAuthority(ledger_path)
        input_decision = ledger.authorize(
            session_id=session_id,
            candidate=stored.absolute_path,
            action=LocalFileAction.READ,
        )
        output_decision = ledger.authorize(
            session_id=session_id,
            candidate=agent_output,
            action=LocalFileAction.SEARCH,
        )

    assert input_decision.allowed is True
    assert input_decision.reason is LocalFileAuthorizationReason.IMPLICIT_SESSION_INPUT
    assert input_decision.storage_area is storage.SessionStorageArea.INPUT
    assert output_decision.allowed is True
    assert output_decision.reason is LocalFileAuthorizationReason.IMPLICIT_SESSION_OUTPUT
    assert output_decision.storage_area is storage.SessionStorageArea.OUTPUT


def test_project_storage_authority_requires_a_registered_existing_safe_path(
    project_session,
    ledger_path: Path,
    tmp_path: Path,
) -> None:
    from personagraph.session import store as session_store

    session_id, root = project_session
    missing = root / "missing.pdf"
    outside = _write(tmp_path / "outside.pdf")
    planted = root / "looks-managed.pdf"
    planted.symlink_to(outside)
    with session_store.session_database_scope(session_id):
        ledger = SqliteSessionFileAuthority(ledger_path)
        absent = ledger.authorize(
            session_id=session_id,
            candidate=missing,
            action=LocalFileAction.READ,
        )
        escaped = ledger.authorize(
            session_id=session_id,
            candidate=planted,
            action=LocalFileAction.READ,
        )

    assert absent.reason is LocalFileAuthorizationReason.PATH_NOT_FOUND
    assert escaped.allowed is False
    assert escaped.reason is LocalFileAuthorizationReason.AUTHORIZATION_REQUIRED


def test_a_working_root_grant_does_not_follow_a_symlink_outside(
    ledger_path: Path,
    tmp_path: Path,
) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = _write(tmp_path / "outside.md")
    planted = root / "inside-looking.md"
    planted.symlink_to(outside)
    ledger = SqliteSessionFileAuthority(ledger_path)
    ledger.grant_workspace_root(session_id="s1", working_root=root)

    decision = ledger.authorize(
        session_id="s1",
        working_root=root,
        candidate=planted,
        action=LocalFileAction.READ,
    )

    assert decision.allowed is False
    assert (
        decision.reason
        is LocalFileAuthorizationReason.PATH_OUTSIDE_AUTHORIZED_SCOPE
    )


def test_mutating_actions_are_not_in_the_authority_vocabulary(
    ledger_path: Path,
    tmp_path: Path,
) -> None:
    document = _write(tmp_path / "document.md")
    ledger = SqliteSessionFileAuthority(ledger_path)

    with pytest.raises(ValueError):
        ledger.authorize(
            session_id="s1",
            candidate=document,
            action="write",
        )
