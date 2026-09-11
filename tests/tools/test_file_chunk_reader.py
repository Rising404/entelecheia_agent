"""Exact File reads reject stale versions and preserve bounded verified output."""

from dataclasses import replace
import hashlib
from types import SimpleNamespace

import pytest

from personagraph.tools.documents.file_chunk_reader import build_file_chunk_reader_runtime
from personagraph.tools.documents.file_chunk_tools import MAX_CHUNK_CONTENT_CHARS
from personagraph.tools.execution import ToolBusinessFailure
from personagraph.workspace.documents.reading import CurrentFileChunk, CurrentFileDocument


def _reader(content="exact text", **overrides):
    source=SimpleNamespace(file_id="file-a",file_version_id="fv-a")
    doc=CurrentFileDocument("file-a","fv-a","doc-a","dv-a","a"*64,1,"complete",None,None)
    chunk=CurrentFileChunk(doc,"chunk-a","producer-a",3,content,hashlib.sha256(content.encode()).hexdigest(),"page 7",(7,))
    kwargs=dict(session_id="session-a",scope_id="scope-a",resolve_file=lambda **_:source,
                revalidate=lambda _:True,read_document=lambda **_:doc,read_chunk=lambda **_:chunk,
                read_chunk_document=lambda **_:doc)
    kwargs.update(overrides)
    return build_file_chunk_reader_runtime(**kwargs)


def _payload(**selector):
    return {"targets":[{"file_id":"file-a","document_version_id":"dv-a",**selector}]}


def test_exact_chunk_id_and_sequence_share_the_same_output():
    reader=_reader()
    first=reader.read_chunks(_payload(chunk_ids=["chunk-a"]))
    second=reader.read_chunks(_payload(chunk_sequences=[3]))
    assert first==second
    assert first["results"][0]["file_version_id"]=="fv-a"
    assert first["results"][0]["chunks"][0]["source_pages"]==[7]


def test_native_chunk_ids_resolve_lineage_without_a_prior_file_selection():
    reader = _reader()
    result = reader.read_chunks({"targets": [{"chunk_ids": ["chunk-a"]}]})
    assert result == reader.read_chunks(_payload(chunk_sequences=[3]))


@pytest.mark.parametrize("constraint", [{}, {"file_id": "wrong"}, {"document_version_id": "old"}])
def test_native_chunk_lookup_rejects_missing_or_conflicting_lineage(constraint):
    reader = _reader(read_chunk_document=(lambda **_: None) if not constraint else _reader().read_chunk_document,
                     read_chunk=lambda **_: pytest.fail("must not read rejected native ID"))
    result = reader.read_chunks({"targets": [{"chunk_ids": ["missing"], **constraint}]})
    assert result["results"] == []
    assert result["unavailable_targets"][0]["reason_code"] == "chunk_unavailable"


def test_native_chunk_ids_do_not_bypass_file_authorization():
    reader = _reader(revalidate=lambda _: False,
                     read_chunk=lambda **_: pytest.fail("must not read revoked File"))
    result = reader.read_chunks({"targets": [{"chunk_ids": ["chunk-a"]}]})
    assert result["results"][0]["chunks"] == []
    assert result["unavailable_targets"][0]["reason_code"] == "file_access_unavailable"


def test_stale_document_version_never_reads_another_generation():
    reader=_reader(read_chunk=lambda **_:pytest.fail("must not read stale target"))
    payload=_payload(chunk_ids=["chunk-a"])
    payload["targets"][0]["document_version_id"]="old-version"
    result=reader.read_chunks(payload)
    assert result["results"][0]["chunks"]==[]
    assert result["unavailable_targets"][0]["reason_code"]=="document_version_unavailable"


def test_revocation_after_read_discards_all_content():
    decisions=iter((True,False))
    result=_reader(revalidate=lambda _:next(decisions)).read_chunks(_payload(chunk_ids=["chunk-a"]))
    assert result["results"][0]["chunks"]==[]
    assert result["unavailable_targets"][0]["reason_code"]=="file_access_changed"


def test_same_file_reprocessing_during_read_invalidates_the_document_result():
    document=CurrentFileDocument("file-a","fv-a","doc-a","dv-a","a"*64,1,"complete",None,None)
    observations=iter((document,replace(document,document_version_id="dv-b")))
    result=_reader(read_document=lambda **_:next(observations)).read_chunks(_payload(chunk_ids=["chunk-a"]))
    assert result["results"][0]["chunks"]==[]
    assert result["unavailable_targets"][0]["reason_code"]=="document_version_changed"


def test_content_is_bounded_and_keeps_full_chunk_hash():
    text="x"*(MAX_CHUNK_CONTENT_CHARS+1)
    result=_reader(content=text).read_chunks(_payload(chunk_ids=["chunk-a"]))
    chunk=result["results"][0]["chunks"][0]
    assert len(chunk["content"])==MAX_CHUNK_CONTENT_CHARS
    assert chunk["content_sha256"]==hashlib.sha256(text.encode()).hexdigest()
    assert chunk["content_truncated"] is True and result["truncated"] is True


@pytest.mark.parametrize("selector", [{}, {"chunk_ids":[]}, {"chunk_sequences":[True]},
    {"chunk_ids":["chunk-a"],"chunk_sequences":[3]}, {"chunk_refs":["old"]}])
def test_invalid_selectors_are_rejected_before_reading(selector):
    with pytest.raises(ToolBusinessFailure):
        _reader().read_chunks(_payload(**selector))
