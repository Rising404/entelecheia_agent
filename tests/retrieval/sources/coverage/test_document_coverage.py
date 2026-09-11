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
from personagraph.retrieval.sources.coverage.document import DocumentIndexCoverageProbe
from personagraph.retrieval.sqlite_store import (
    RetrievalDataVersion,
    RetrievalDataVersionRole,
    RetrievalDataVersionState,
)


def _access() -> SourceAccess:
    return SourceAccess(
        source_type=SourceType.DOCUMENT,
        source_filter=SourceFilter.from_mapping(
            SourceType.DOCUMENT,
            {"doc_id": "doc-1", "session_id": "s1"},
        ),
        availability=SourceAvailability.READY,
        source_snapshot_id="snapshot-1",
        source_revision_map={"doc-1": "version-1"},
    )


@dataclass
class BindingReader:
    snapshot: SourceIndexBindingSnapshot

    def get_current_index_binding_snapshot(
        self,
        access: SourceAccess,
        *,
        maximum_bindings: int,
    ) -> SourceIndexBindingSnapshot:
        assert access.source_type is SourceType.DOCUMENT
        assert maximum_bindings > 0
        return self.snapshot


@dataclass
class Catalog:
    active: RetrievalDataVersion | None
    covered: frozenset[tuple[str, str, str]] = frozenset()
    fail: bool = False
    seen_keys: tuple[tuple[str, str, str], ...] = ()

    def get_data_version(self, data_version_id):
        if self.fail:
            raise RuntimeError("cross-store failure")
        if self.active is None or self.active.id != data_version_id:
            return None
        return self.active

    def active_ready_source_binding_keys(self, *, data_version_id, source_filter, keys):
        if self.fail:
            raise RuntimeError("cross-store failure")
        assert data_version_id == "v1"
        assert source_filter.source_type is SourceType.DOCUMENT
        self.seen_keys = tuple(keys)
        return self.covered


def _active_version() -> RetrievalDataVersion:
    return RetrievalDataVersion(
        id="v1",
        fingerprint="test",
        role=RetrievalDataVersionRole.ACTIVE,
        state=RetrievalDataVersionState.READY,
        created_at="2026-08-11T00:00:00+00:00",
        activated_at="2026-08-11T00:00:00+00:00",
    )


def test_document_coverage_probe_requires_every_current_binding_but_not_document_text():
    binding = SourceIndexBinding("s1:chunk-1", "version-1", "hash-1")
    reader = BindingReader(SourceIndexBindingSnapshot(True, (binding,)))
    catalog = Catalog(
        active=_active_version(),
        covered=frozenset({("s1:chunk-1", "version-1", "hash-1")}),
    )

    result = DocumentIndexCoverageProbe(catalog=catalog, binding_reader=reader).check_source_coverage(
        _access(),
        retrieval_data_version_id="v1",
    )

    assert result is None
    assert catalog.seen_keys == (("s1:chunk-1", "version-1", "hash-1"),)


def test_document_coverage_probe_rejects_a_ready_binding_with_the_wrong_content_hash():
    binding = SourceIndexBinding("s1:chunk-1", "version-1", "expected-hash")
    result = DocumentIndexCoverageProbe(
        catalog=Catalog(
            active=_active_version(),
            covered=frozenset({("s1:chunk-1", "version-1", "wrong-hash")}),
        ),
        binding_reader=BindingReader(SourceIndexBindingSnapshot(True, (binding,))),
    ).check_source_coverage(_access(), retrieval_data_version_id="v1")

    assert result is not None
    assert result.reason_code == "document_index_coverage_incomplete"


def test_document_coverage_probe_reports_safe_partial_reasons_for_lag_and_cross_store_failure():
    binding = SourceIndexBinding("s1:chunk-1", "version-1", "hash-1")
    reader = BindingReader(SourceIndexBindingSnapshot(True, (binding,)))

    incomplete = DocumentIndexCoverageProbe(
        catalog=Catalog(active=_active_version()),
        binding_reader=reader,
    ).check_source_coverage(_access(), retrieval_data_version_id="v1")
    assert incomplete is not None
    assert incomplete.reason_code == "document_index_coverage_incomplete"

    unavailable_data_version = DocumentIndexCoverageProbe(
        catalog=Catalog(active=None),
        binding_reader=reader,
    ).check_source_coverage(_access(), retrieval_data_version_id="v1")
    assert unavailable_data_version is not None
    assert unavailable_data_version.reason_code == "document_index_data_version_unavailable"

    unavailable_probe = DocumentIndexCoverageProbe(
        catalog=Catalog(active=_active_version(), fail=True),
        binding_reader=reader,
    ).check_source_coverage(_access(), retrieval_data_version_id="v1")
    assert unavailable_probe is not None
    assert unavailable_probe.reason_code == "document_index_coverage_check_unavailable"


def test_document_coverage_probe_allows_a_verified_document_without_text_chunks():
    result = DocumentIndexCoverageProbe(
        catalog=Catalog(active=_active_version()),
        binding_reader=BindingReader(SourceIndexBindingSnapshot(True, ())),
    ).check_source_coverage(_access(), retrieval_data_version_id="v1")

    assert result is None


def test_document_coverage_probe_fails_safe_when_binding_enumeration_hits_its_limit():
    binding = SourceIndexBinding("s1:chunk-1", "version-1", "hash-1")
    result = DocumentIndexCoverageProbe(
        catalog=Catalog(active=_active_version()),
        binding_reader=BindingReader(
            SourceIndexBindingSnapshot(
                True,
                (binding,),
                binding_enumeration_complete=False,
            )
        ),
    ).check_source_coverage(_access(), retrieval_data_version_id="v1")

    assert result is not None
    assert result.reason_code == "document_index_coverage_limit_exceeded"


def test_document_coverage_probe_never_treats_an_unpinned_request_as_complete():
    binding = SourceIndexBinding("s1:chunk-1", "version-1", "hash-1")
    result = DocumentIndexCoverageProbe(
        catalog=Catalog(
            active=_active_version(),
            covered=frozenset({("s1:chunk-1", "version-1", "hash-1")}),
        ),
        binding_reader=BindingReader(SourceIndexBindingSnapshot(True, (binding,))),
    ).check_source_coverage(_access(), retrieval_data_version_id=None)

    assert result is not None
    assert result.reason_code == "document_index_data_version_unavailable"


def test_document_coverage_probe_rejects_a_previous_ready_data_version():
    previous = RetrievalDataVersion(
        id="v1",
        fingerprint="test",
        role=RetrievalDataVersionRole.PREVIOUS,
        state=RetrievalDataVersionState.READY,
        created_at="2026-08-11T00:00:00+00:00",
        activated_at="2026-08-11T00:00:00+00:00",
    )
    binding = SourceIndexBinding("s1:chunk-1", "version-1", "hash-1")
    result = DocumentIndexCoverageProbe(
        catalog=Catalog(
            active=previous,
            covered=frozenset({("s1:chunk-1", "version-1", "hash-1")}),
        ),
        binding_reader=BindingReader(SourceIndexBindingSnapshot(True, (binding,))),
    ).check_source_coverage(_access(), retrieval_data_version_id="v1")

    assert result is not None
    assert result.reason_code == "document_index_data_version_unavailable"
