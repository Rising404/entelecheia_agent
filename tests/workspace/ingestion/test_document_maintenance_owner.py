from __future__ import annotations

import pytest

from personagraph.workspace.ingestion.composition import (
    build_document_maintenance_lifecycle,
    resolve_document_ingest_owner,
    synchronous_ingest_worker,
)
from personagraph.workspace.ingestion.execution import DocumentIngestOwnerConflict
from personagraph.retrieval.profile import DocumentRetrievalProfile
from tests.documents._authority import bound_project_document_authority


def test_active_background_lifecycle_is_the_only_project_ingest_owner(tmp_path) -> None:
    """prepare 必须 wake 已有 owner，而不是为同一 Project 构造第二个 encoder。"""

    with bound_project_document_authority(tmp_path):
        lifecycle = build_document_maintenance_lifecycle(
            profile=DocumentRetrievalProfile.lexical(),
        )
        assert lifecycle.start() is True
        try:
            owner = resolve_document_ingest_owner()

            assert owner.kind == "background"
            assert owner.can_run_synchronously is False
            with pytest.raises(
                DocumentIngestOwnerConflict,
                match="background document ingest owner is active",
            ):
                synchronous_ingest_worker()
        finally:
            assert lifecycle.stop(timeout_seconds=2.0) is True


def test_stopped_background_lifecycle_releases_the_project_owner(tmp_path) -> None:
    with bound_project_document_authority(tmp_path):
        lifecycle = build_document_maintenance_lifecycle(
            profile=DocumentRetrievalProfile.lexical(),
        )
        assert lifecycle.start() is True
        assert resolve_document_ingest_owner().kind == "background"
        assert lifecycle.stop(timeout_seconds=2.0) is True

        owner = resolve_document_ingest_owner()

        assert owner.kind == "synchronous"
        assert owner.can_run_synchronously is True


def test_second_background_lifecycle_cannot_claim_the_same_project(tmp_path) -> None:
    """失败的重复 start 不得覆盖或泄漏首个 owner 的注册。"""

    with bound_project_document_authority(tmp_path):
        first = build_document_maintenance_lifecycle(
            profile=DocumentRetrievalProfile.lexical(),
        )
        second = build_document_maintenance_lifecycle(
            profile=DocumentRetrievalProfile.lexical(),
        )
        assert first.start() is True
        try:
            with pytest.raises(
                DocumentIngestOwnerConflict,
                match="already has an active document ingest owner",
            ):
                second.start()
            assert second.is_running is False
            assert resolve_document_ingest_owner().kind == "background"
        finally:
            assert first.stop(timeout_seconds=2.0) is True

        # 首个 lifecycle 正常 stop 后，失败过的实例也可以重新获得 owner。
        assert second.start() is True
        assert second.stop(timeout_seconds=2.0) is True
