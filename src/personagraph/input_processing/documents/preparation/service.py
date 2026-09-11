"""把一个稳定本地来源解析并切分为无持久副作用的准备结果。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ....configuration.paths import deny_reason
from ...files import (
    SourceChangedDuringReadError,
    SourceSizeLimitError,
    fingerprint_file,
)
from ..chunking import ChunkingProfile, chunk_document
from ..contracts import (
    ProcessingAdmissionStatus,
    ProcessingResult,
)
from ..readers import (
    LegacyOfficeBridge,
    configured_legacy_office_bridge,
    get_reader,
    read_document,
    supported_suffixes,
)
from .contracts import DocumentPrepareFailure, PreparedDocumentIngest


def prepare_document_path(
    path_str: str,
    *,
    chunking_profile: ChunkingProfile | None = None,
    legacy_office_bridge: LegacyOfficeBridge | None = None,
) -> PreparedDocumentIngest | DocumentPrepareFailure:
    """读取、分类并切分一个稳定源，但不写入 authority。"""

    path = Path(path_str).expanduser().resolve()
    reason = deny_reason(path)
    if reason:
        return DocumentPrepareFailure(reason, {"path": str(path)})
    if not path.is_file():
        return DocumentPrepareFailure("not_a_file", {"path": str(path)})
    if (
        path.suffix.lower() in {".doc", ".ppt"}
        and legacy_office_bridge is None
        and configured_legacy_office_bridge() is None
    ):
        return DocumentPrepareFailure(
            "unsupported_legacy_office",
            {
                "path": str(path),
                "hint": "convert to DOCX/PPTX; sandboxed legacy conversion is not enabled",
            },
        )
    legacy_office = path.suffix.lower() in {".doc", ".ppt"}
    reader = (
        get_reader(path, legacy_office_bridge=legacy_office_bridge)
        if legacy_office
        else get_reader(path)
    )
    if reader is None:
        return DocumentPrepareFailure(
            "unsupported_format",
            {"supported": supported_suffixes()},
        )

    try:
        before = fingerprint_file(path)
        result = (
            read_document(path, legacy_office_bridge=legacy_office_bridge)
            if legacy_office
            else read_document(path)
        )
        after = fingerprint_file(path)
    except SourceSizeLimitError:
        return DocumentPrepareFailure(
            "too_large",
            {"path": str(path)},
        )
    except SourceChangedDuringReadError:
        return DocumentPrepareFailure(
            "source_changed_during_ingest",
            {"path": str(path)},
        )
    except OSError as exc:
        return DocumentPrepareFailure(f"read_error:{type(exc).__name__}", {})
    except Exception as exc:
        return DocumentPrepareFailure(f"read_error:{type(exc).__name__}", {})

    if before.sha256 != after.sha256:
        return DocumentPrepareFailure(
            "source_changed_during_ingest",
            {"path": str(path)},
        )
    if result.admission_status is ProcessingAdmissionStatus.REJECTED:
        failure = _incomplete_processing_result(path, result)
        return DocumentPrepareFailure(
            str(failure.pop("reason")),
            {key: value for key, value in failure.items() if key != "ok"},
        )

    source_elements = result.text_elements()
    elements = tuple(result.legacy_elements())
    if not any((element.get("content") or "").strip() for element in elements):
        return DocumentPrepareFailure(
            "no_text_extracted",
            {"hint": "可能是扫描件/图片型文档，多模态阶段才支持"},
        )
    profile = chunking_profile or ChunkingProfile()
    document_chunks = chunk_document(result, source_key=str(path), profile=profile)
    diagnostics = tuple(diagnostic.to_dict() for diagnostic in result.diagnostics)
    preview = "\n".join(
        str(element.get("content") or "") for element in elements[:3]
    )
    return PreparedDocumentIngest(
        canonical_path=str(path),
        title=path.stem,
        mime=path.suffix.lstrip("."),
        elements=elements,
        source_elements=source_elements,
        document_chunks=document_chunks,
        source_fingerprint=after,
        processor_fingerprint=str(result.processor),
        chunker_fingerprint=profile.fingerprint(),
        processing_status=result.admission_status.value,
        processing_diagnostics=diagnostics,
        needs_vision=result.needs_vision,
        summary_preview=preview,
        page_manifest=result.page_manifest,
    )


def _incomplete_processing_result(path: Path, result: ProcessingResult) -> dict[str, Any]:
    """返回稳定且安全的失败，不暴露部分文档文本。"""

    return {
        "ok": False,
        "reason": "document_processing_incomplete",
        "path": str(path),
        "needs_vision": result.needs_vision,
        "diagnostics": [diagnostic.to_dict() for diagnostic in result.diagnostics],
    }
