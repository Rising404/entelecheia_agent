from __future__ import annotations

from types import SimpleNamespace

import pytest

from personagraph.retrieval.lifecycle.generation import (
    RetrievalGenerationMismatch,
    RetrievalGenerationSpec,
)
from personagraph.retrieval.operations import turn_binding
from personagraph.retrieval.sqlite_store import (
    RetrievalDataVersion,
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
)


class _Catalog:
    def __init__(self, active=None) -> None:
        self._active = active

    def active_data_version(self):
        return self._active


def _composition(*, active=None, version: str = "file-generation-v1"):
    return SimpleNamespace(
        generation_spec=SimpleNamespace(
            version_id=version,
            fingerprint="f" * 64,
        ),
        foundation=SimpleNamespace(catalog=_Catalog(active)),
    )


def test_file_turn_generation_binding_is_default_off_and_does_not_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def build():
        nonlocal calls
        calls += 1
        return _composition()

    monkeypatch.setattr(
        turn_binding,
        "build_document_retrieval_composition",
        build,
    )

    assert turn_binding.prepare_file_retrieval_for_turn(features={}) is None
    assert calls == 0
    binding = turn_binding.prepare_file_retrieval_for_turn(
        features={"file_retrieval_read_enabled": True}
    )
    assert binding is not None
    assert binding.data_version_id == "file-generation-v1"
    assert calls == 1


def test_file_turn_generation_binding_rejects_an_incompatible_active_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = RetrievalGenerationSpec(
        encoder_fingerprint="encoder-v1",
        chunker_fingerprint="chunker-v1",
        document_chunk_contract_version=1,
        retrieval_catalog_schema_version=1,
        source_identity_contract="source-v1",
        index_recipe="recipe-v1",
        source_types=("document",),
    )
    active = RetrievalDataVersion(
        id="another-generation",
        fingerprint="e" * 64,
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
        created_at="2026-08-27T00:00:00+00:00",
        activated_at="2026-08-27T00:00:00+00:00",
    )
    monkeypatch.setattr(
        turn_binding,
        "build_document_retrieval_composition",
        lambda: SimpleNamespace(
            generation_spec=spec,
            foundation=SimpleNamespace(catalog=_Catalog(active)),
        ),
    )

    with pytest.raises(RetrievalGenerationMismatch, match="active_generation"):
        turn_binding.prepare_file_retrieval_for_turn(
            features={"file_retrieval_read_enabled": True}
        )
