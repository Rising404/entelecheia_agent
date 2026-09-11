from __future__ import annotations

from personagraph.retrieval.contracts import CorpusKey, SourceType
from personagraph.retrieval.lifecycle.corpus import FILE_CORPUS, SESSION_CORPUS
from personagraph.retrieval.foundation import (
    build_file_retrieval_foundation,
)


def test_active_retrieval_corpora_have_explicit_source_membership() -> None:
    assert FILE_CORPUS.key is CorpusKey.FILE
    assert FILE_CORPUS.source_types == (
        SourceType.DOCUMENT,
        SourceType.PICTURE,
    )
    assert SESSION_CORPUS.key is CorpusKey.HISTORY
    assert SESSION_CORPUS.source_types == (SourceType.CURRENT_SESSION,)


def test_file_foundation_installs_document_and_picture_adapters(tmp_path) -> None:
    foundation = build_file_retrieval_foundation(
        db_path=tmp_path / "retrieval.sqlite"
    )

    assert foundation.corpus_key is CorpusKey.FILE
    assert set(foundation.service._sources) == {
        SourceType.DOCUMENT,
        SourceType.PICTURE,
    }
    assert set(foundation.service._readonly_source_execution._sources) == {
        SourceType.DOCUMENT,
        SourceType.PICTURE,
    }
    assert foundation.sync_service.source_types == (
        SourceType.DOCUMENT,
        SourceType.PICTURE,
    )
    assert set(foundation.source_adapters) == {
        SourceType.DOCUMENT,
        SourceType.PICTURE,
    }
