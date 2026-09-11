from __future__ import annotations

import pytest

from personagraph.api import service
from personagraph.api.service import sessions as session_service
from personagraph.api.service.errors import ApiError
from personagraph.runtime.post_commit.runner import process_due_turn_post_commit_jobs
from personagraph.session import store as session_store


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 512


@pytest.fixture
def project_root(tmp_path, partitioned_project_state):
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture
def session(project_root):
    return session_store.create_session(
        "Entelecheia",
        working_dir=str(project_root),
    )


def _upload(session_id, name, payload, declared=None):
    with session_store.session_database_scope(session_id):
        return service.upload_attachment(
            session_id, payload=payload, filename=name, declared_media_type=declared
        )["attachment"]


def _list(session_id):
    with session_store.session_database_scope(session_id):
        return service.list_session_attachments(session_id)


def _chat(payload):
    with session_store.session_database_scope(str(payload["session_id"])):
        return service.chat_turn(payload)


def _delete(session_id, attachment_id):
    with session_store.session_database_scope(session_id):
        return service.delete_session_attachment(session_id, attachment_id)


def test_every_type_is_accepted_and_labelled(session):
    """准入面向所有文件；可读性通过声明表达，而不在入口处强制。"""
    image = _upload(session, "shot.png", PNG, "image/png")
    audio = _upload(session, "voice.mp3", b"ID3\x03" + b"\x00" * 64, "audio/mpeg")

    assert image["kind"] == "image" and image["readable"] is True
    assert audio["kind"] == "audio" and audio["readable"] is False


def test_declared_content_type_never_overrides_the_bytes(session):
    attachment = _upload(session, "invoice.pdf", PNG, declared="application/pdf")
    assert attachment["media_type"] == "image/png"


def test_uploads_stay_listed_until_they_are_sent(session):
    _upload(session, "a.png", PNG)
    _upload(session, "b.png", PNG + b"x")
    listing = _list(session)

    assert len(listing["attachments"]) == 2
    assert listing["limits"]["max_attachments_per_turn"] >= 1


def test_upload_has_exactly_one_project_owned_copy(session, project_root):
    attachment = _upload(session, "../../paper.png", PNG)

    with session_store.session_database_scope(session):
        record = session_store.get_attachment(attachment["attachment_id"])
    assert record is not None
    managed = project_root / str(record["stored_rel_path"])
    assert managed.read_bytes() == PNG
    assert list(project_root.rglob("paper.png")) == [managed]
    assert not (project_root / "input").exists()
    assert not (project_root / "output").exists()


def test_upload_is_refused_past_the_single_file_bound(session, monkeypatch):
    monkeypatch.setattr(
        "personagraph.session.attachments.application.MAX_ATTACHMENT_BYTES", 16
    )
    with pytest.raises(ApiError) as exc:
        with session_store.session_database_scope(session):
            service.upload_attachment(
                session, payload=b"x" * 64, filename="big.bin", declared_media_type=None
            )
    assert exc.value.code == "ATTACHMENT_TOO_LARGE"


def test_upload_is_refused_past_the_session_quota(session, monkeypatch):
    monkeypatch.setattr(
        "personagraph.session.attachments.application.MAX_SESSION_ATTACHMENT_BYTES", 600
    )
    _upload(session, "a.png", PNG)
    with pytest.raises(ApiError) as exc:
        with session_store.session_database_scope(session):
            service.upload_attachment(
                session, payload=PNG, filename="b.png", declared_media_type=None
            )
    assert exc.value.code == "SESSION_ATTACHMENT_QUOTA_EXCEEDED"


def test_filename_header_is_decoded_without_being_trusted():
    assert service.decode_upload_filename("%E5%9B%BE.png") == "图.png"
    assert service.decode_upload_filename(None) == "file"
    assert service.decode_upload_filename("") == "file"


def test_sending_an_unknown_attachment_id_fails_the_whole_turn(session):
    """忽略用户附加的文件，比明确拒绝发送更糟。"""
    with pytest.raises(ApiError) as exc:
        _chat({
            "session_id": session, "message": "看看这个", "attachment_ids": ["att_nope"],
            "client_request_id": "unknown-attachment",
        })
    assert exc.value.code == "ATTACHMENT_BINDING_FAILED"
    assert exc.value.status == 409
    with session_store.session_database_scope(session):
        assert session_store.get_turns(session) == []


def test_an_attachment_cannot_be_sent_twice(session, monkeypatch):
    # 此处测试的是附件所有权。同步完成新的隐藏摘要任务，避免该测试转而验证
    # 有意设置的单轮执行窗口门禁。
    monkeypatch.setattr(
        session_service,
        "schedule_turn_post_commit_jobs",
        lambda *, session_id, store: process_due_turn_post_commit_jobs(
            session_id=session_id,
            store=store,
            worker_id="attachment-test-summary-worker",
        ),
    )
    attachment = _upload(session, "a.png", PNG)
    _chat({
        "session_id": session, "message": "第一次", "attachment_ids": [attachment["attachment_id"]],
        "client_request_id": "first-attachment",
    })
    with pytest.raises(ApiError) as exc:
        _chat({
            "session_id": session, "message": "第二次",
            "attachment_ids": [attachment["attachment_id"]],
            "client_request_id": "second-attachment",
        })
    assert exc.value.code == "ATTACHMENT_BINDING_FAILED"


def test_another_sessions_attachment_cannot_be_borrowed(session, project_root):
    other = session_store.create_session("Entelecheia", working_dir=str(project_root))
    stolen = _upload(other, "a.png", PNG)
    with pytest.raises(ApiError) as exc:
        _chat({
            "session_id": session, "message": "借一个", "attachment_ids": [stolen["attachment_id"]],
            "client_request_id": "borrowed-attachment",
        })
    assert exc.value.code == "ATTACHMENT_BINDING_FAILED"


def test_too_many_attachments_in_one_turn_is_refused(session):
    with pytest.raises(ApiError) as exc:
        _chat({
            "session_id": session, "message": "很多",
            "attachment_ids": [f"att_{index}" for index in range(64)],
            "client_request_id": "too-many-attachments",
        })
    assert exc.value.code == "TOO_MANY_ATTACHMENTS"


def test_a_sent_attachment_is_bound_to_the_turn_that_sent_it(session):
    attachment = _upload(session, "notes.md", "会议要点".encode())
    result = _chat({
        "session_id": session, "message": "总结一下",
        "attachment_ids": [attachment["attachment_id"]],
        "client_request_id": "bound-attachment",
    })
    turn_id = result["result"]["turn_id"]

    with session_store.session_database_scope(session):
        bound = session_store.list_turn_attachments(session, turn_id)
        unbound = session_store.list_unbound_attachments(session)
    assert [row["attachment_id"] for row in bound] == [attachment["attachment_id"]]
    assert unbound == []


def test_purging_a_session_keeps_project_owned_files(session, project_root):
    """Session 生命周期不得删除可能被其他 Session 使用的 Project 文件。"""
    from personagraph.session import service as session_service

    attachment = _upload(session, "a.png", PNG)
    with session_store.session_database_scope(session):
        record = session_store.get_attachment(attachment["attachment_id"])
    assert record is not None
    project_file = project_root / str(record["stored_rel_path"])
    assert project_file.is_file()

    session_service.purge_session_fully(session)

    assert project_file.read_bytes() == PNG


def test_a_staged_attachment_can_be_discarded_before_sending(session, project_root):
    """编辑器的移除按钮必须真正删除附件，而不只是隐藏。"""
    attachment = _upload(session, "a.png", PNG)
    with session_store.session_database_scope(session):
        record = session_store.get_attachment(attachment["attachment_id"])
    assert record is not None
    project_file = project_root / str(record["stored_rel_path"])

    result = _delete(session, attachment["attachment_id"])

    assert result["deleted"] is True
    assert _list(session)["attachments"] == []
    assert project_file.is_file()


def test_a_sent_attachment_cannot_be_discarded(session):
    """它属于已提交轮次；删除它会让对话记录失去归属。"""
    attachment = _upload(session, "notes.md", "会议要点".encode())
    _chat({
        "session_id": session, "message": "总结", "attachment_ids": [attachment["attachment_id"]],
        "client_request_id": "sent-attachment",
    })

    with pytest.raises(ApiError) as exc:
        _delete(session, attachment["attachment_id"])
    assert exc.value.code == "ATTACHMENT_NOT_REMOVABLE"
    assert exc.value.status == 404


def test_another_sessions_attachment_cannot_be_discarded(session, project_root):
    other = session_store.create_session("Entelecheia", working_dir=str(project_root))
    stolen = _upload(other, "a.png", PNG)

    with pytest.raises(ApiError) as exc:
        _delete(session, stolen["attachment_id"])
    assert exc.value.code == "ATTACHMENT_NOT_REMOVABLE"
    # 真正的所有者仍然持有它。
    assert len(_list(other)["attachments"]) == 1
