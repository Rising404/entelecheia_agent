from __future__ import annotations

from dataclasses import dataclass

from personagraph.retrieval.contracts import (
    SourceAccess,
    SourceAvailability,
    SourceFilter,
    SourceIndexBinding,
    SourceIndexBindingSnapshot,
    SourceType,
)
from personagraph.retrieval.sources.coverage.picture import PictureIndexCoverageProbe
from personagraph.retrieval.sqlite_store import (
    RetrievalDataVersion,
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
)


@dataclass
class _BindingReader:
    snapshot: SourceIndexBindingSnapshot

    def get_current_index_binding_snapshot(self, access, *, maximum_bindings):
        assert access.source_type is SourceType.PICTURE
        assert maximum_bindings > 0
        return self.snapshot


@dataclass
class _Catalog:
    covered: frozenset[tuple[str, str, str]]

    def get_data_version(self, data_version_id):
        assert data_version_id == "v1"
        return RetrievalDataVersion(
            id="v1",
            fingerprint="test",
            role=RetrievalDataVersionRole.ACTIVE,
            state=RetrievalDataVersionState.READY,
            created_at="2026-09-04T12:00:00+00:00",
            activated_at="2026-09-04T12:00:00+00:00",
        )

    def active_ready_source_binding_keys(self, *, data_version_id, source_filter, keys):
        assert data_version_id == "v1"
        assert source_filter.source_type is SourceType.PICTURE
        return self.covered.intersection(keys)


def _access() -> SourceAccess:
    return SourceAccess(
        source_type=SourceType.PICTURE,
        source_filter=SourceFilter.from_mapping(
            SourceType.PICTURE,
            {"file_id": "file-1", "file_version_id": "version-1"},
        ),
        availability=SourceAvailability.READY,
        source_snapshot_id="snapshot-1",
        source_revision_map={"observation-1": "a" * 64},
    )


def test_picture_coverage_requires_each_active_fifo_binding() -> None:
    binding_key = ("picture-observation-v1:b2JzLTE", "a" * 64, "b" * 64)
    binding = SourceIndexBinding(*binding_key)
    complete = PictureIndexCoverageProbe(
        catalog=_Catalog(frozenset({binding_key})),
        binding_reader=_BindingReader(SourceIndexBindingSnapshot(True, (binding,))),
    ).check_source_coverage(_access(), retrieval_data_version_id="v1")
    missing = PictureIndexCoverageProbe(
        catalog=_Catalog(frozenset()),
        binding_reader=_BindingReader(SourceIndexBindingSnapshot(True, (binding,))),
    ).check_source_coverage(_access(), retrieval_data_version_id="v1")

    assert complete is None
    assert missing is not None
    assert missing.reason_code == "picture_index_coverage_incomplete"
