"""文本格式必须能穿过 reader 路由、解析与切块，而不只出现在工具说明里。"""

from pathlib import Path

import pytest

from personagraph.input_processing.documents.preparation import (
    PreparedDocumentIngest,
    prepare_document_path,
)
from personagraph.input_processing.documents.readers import (
    configured_processor_fingerprint,
    get_reader,
)
from personagraph.input_processing.documents.readers.plain_text import read_plain_text
from personagraph.input_processing.files import detect_type
from personagraph.input_processing.files.text_formats import TEXT_MEDIA_TYPES


@pytest.mark.parametrize("suffix", tuple(TEXT_MEDIA_TYPES))
def test_text_format_identity_and_native_ingest_route_stay_aligned(suffix):
    # HTML 可选择 Docling；原生文本兜底仍须与上传识别共享完整的格式集合。
    detected = detect_type(b"plain text", f"sample{suffix}")
    assert detected.extension == suffix
    assert detected.media_type == TEXT_MEDIA_TYPES[suffix]
    assert get_reader(Path(f"sample{suffix}"), engine="native") is read_plain_text


@pytest.mark.parametrize("suffix", (".json", ".jsonl", ".ndjson", ".JSONL", ".NDJSON"))
def test_json_text_formats_reach_the_shared_reader_and_chunker(tmp_path, suffix):
    path = tmp_path / f"records{suffix}"
    text = '{"内容":"起始记录"}\n' + '{"内容":"中间材料"}\n' * 2000 + '{"内容":"尾部哨兵"}\n'
    path.write_text(text, encoding="utf-8")

    assert get_reader(path) is read_plain_text
    assert configured_processor_fingerprint(path) is not None
    prepared = prepare_document_path(str(path))

    assert isinstance(prepared, PreparedDocumentIngest)
    assert prepared.processing_status == "complete"
    assert len(prepared.document_chunks) > 1
    assert "尾部哨兵" in prepared.document_chunks[-1].text
    for element in prepared.source_elements:
        start, end = element.locator.char_range
        assert element.text == text[start:end]
