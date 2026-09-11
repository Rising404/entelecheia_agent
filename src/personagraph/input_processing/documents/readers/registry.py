"""格式适配器：把磁盘字节转换为格式中立元素。

reader 的唯一职责就是该转换。它不得写数据库、调用模型、切块、总结或挂载任何内容——这些
是独立阶段，需要在没有解析器时也可替换、可测试。

对 Docling 覆盖的格式，存在两个引擎：

``native``
    本包内的 reader。不需要模型、不需要下载，耗时毫秒级。结构来自显式标记（DOCX heading、
    Markdown 井号）或几何信息；后者只适用于简单单栏 PDF。

``docling``
    使用检测模型对渲染页面执行布局分析，从没有语义标记的 PDF 中恢复结构，并为扫描页面
    提供 OCR。代价是加载模型，且每个文档耗时数秒。

引擎选择由 Host 决定，绝不是逐调用偏好；因此 chunk 记录的 processor 指纹始终能标识实际
生成它的引擎。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from ..contracts import (
    DiagnosticCode,
    ProcessingDiagnostic,
    ProcessingResult,
    ProcessorFingerprint,
)
from ...files import MAX_DOCUMENT_FILE_BYTES
from .docling_reader import (
    assess_pdf_ingest_geometry,
    configured_docling_recipe,
    docling_available,
    fingerprint as docling_fingerprint,
    read_with_docling,
)
from .office import DOCX_READER, PPTX_READER, read_docx, read_pptx
from .image import (
    configured_image_processor_fingerprint,
    read_image,
    validate_image_payload,
)
from .legacy_office import (
    LegacyOfficeBridge,
    configured_legacy_office_bridge,
)
from .pdf import READER_NAME as PDF_READER_NAME
from .pdf import READER_VERSION as PDF_READER_VERSION
from .pdf import (
    merge_pdf_annotation_inventory,
    merge_pdf_native_table_inventory,
    pdf_reader_requires_password,
    read_pdf,
)
from .plain_text import READER_NAME as PLAIN_TEXT_READER_NAME
from .plain_text import READER_VERSION as PLAIN_TEXT_READER_VERSION
from .plain_text import read_plain_text


DocumentReader = Callable[[Path], ProcessingResult]

ENGINE_ENV_VAR = "PERSONAGRAPH_DOCUMENT_ENGINE"
NATIVE_ENGINE = "native"
DOCLING_ENGINE = "docling"

_TEXT_SUFFIXES = (
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".log", ".yaml", ".yml",
    ".py", ".js", ".ts", ".html", ".css", ".sql", ".sh", ".toml", ".xml", ".ini",
    ".rs", ".go", ".java", ".c", ".h", ".cpp", ".rb",
)

_NATIVE_READERS: dict[str, DocumentReader] = {
    **{suffix: read_plain_text for suffix in _TEXT_SUFFIXES},
    ".png": read_image,
    ".jpg": read_image,
    ".jpeg": read_image,
    ".pdf": read_pdf,
    ".docx": read_docx,
    ".pptx": read_pptx,
}

# 以下格式值得承担布局分析成本，且原生 reader 无法提供更强源覆盖契约。PDF 保持布局优先；
# xlsx 没有原生适配器，HTML 否则会回退到有界纯文本而非结构化布局。DOCX 和 PPTX 刻意保持
# 原生处理：OOXML reader 会保留 Word story 与 PowerPoint speaker note，同时显式报告每个
# 不受支持的视觉结构。
#
# Markdown 被刻意排除。其结构在文本中已显式表达，原生 reader 可精确恢复；渲染后再对像素
# 运行布局检测模型纯属浪费。
_DOCLING_SUFFIXES = frozenset({".pdf", ".xlsx", ".html"})


class UnsupportedDocumentFormat(ValueError):
    """没有为此文件格式注册适配器。"""

    def __init__(self, suffix: str) -> None:
        super().__init__(f"no document reader for {suffix or '(no suffix)'}")
        self.suffix = suffix


class UnsupportedLegacyOfficeFormat(UnsupportedDocumentFormat):
    """二进制 DOC/PPT 需要未来的 sandbox 转换桥梁。"""

    reason_code = "unsupported_legacy_office"

    def __init__(self, suffix: str) -> None:
        super().__init__(suffix)
        self.args = (
            f"legacy Office {suffix} requires a sandboxed conversion bridge",
        )


def configured_engine() -> str:
    """解析本安装所使用的引擎。

    安装可选 ``documents-layout`` extra *本身就是*启用操作，因此存在 Docling 时默认使用。
    若同时要求安装和环境变量，只会产生携带依赖却永远无法受益的部署。

    ``PERSONAGRAPH_DOCUMENT_ENGINE=native`` 会强制恢复快速、无模型的 reader；Docling
    缺失时也会自动降级到这些 reader，而不会导致所有摄取失败。
    """

    requested = (os.environ.get(ENGINE_ENV_VAR) or "").strip().lower()
    if requested == NATIVE_ENGINE:
        return NATIVE_ENGINE
    if requested == DOCLING_ENGINE and not docling_available():
        return NATIVE_ENGINE
    return DOCLING_ENGINE if docling_available() else NATIVE_ENGINE


def get_reader(
    path: Path,
    *,
    engine: str | None = None,
    legacy_office_bridge: LegacyOfficeBridge | None = None,
) -> DocumentReader | None:
    suffix = path.suffix.lower()
    if suffix in {".doc", ".ppt"}:
        bridge = legacy_office_bridge or configured_legacy_office_bridge()
        return None if bridge is None else bridge.read
    resolved = engine or configured_engine()
    if suffix == ".pdf":
        reader, _rejection = _configured_pdf_reader(path, resolved)
        return reader
    if resolved == DOCLING_ENGINE and suffix in _DOCLING_SUFFIXES and docling_available():
        return read_with_docling
    return _NATIVE_READERS.get(suffix)


def _configured_pdf_reader(
    path: Path,
    resolved_engine: str,
) -> tuple[DocumentReader, ProcessingDiagnostic | None]:
    """根据一次非渲染评估选择 Docling 或原生 PDF 解析。"""

    layout_available = (
        resolved_engine == DOCLING_ENGINE and docling_available()
    )
    preferred = read_with_docling if layout_available else read_pdf
    try:
        source_size = path.stat().st_size
    except OSError:
        # 能力发现传入只有名称的 Path。尚无物理源可分类时，保留已配置 recipe。
        return preferred, None
    if source_size > MAX_DOCUMENT_FILE_BYTES:
        # 此拒绝由公共字节准入检查负责，且必须在 pypdf 看到潜在恶意超大源之前运行。
        return preferred, None
    recipe = configured_docling_recipe() if layout_available else None
    assessment = assess_pdf_ingest_geometry(path, recipe=recipe)
    if assessment.rejection is not None:
        return preferred, assessment.rejection
    if layout_available and assessment.eager_eligible:
        return read_with_docling, None
    return read_pdf, None


def configured_processor_fingerprint(
    path: Path,
    *,
    engine: str | None = None,
    legacy_office_bridge: LegacyOfficeBridge | None = None,
) -> ProcessorFingerprint | None:
    """在不读取源内容的情况下解析已配置 reader recipe。"""

    suffix = path.suffix.lower()
    bridge = legacy_office_bridge
    if suffix in {".doc", ".ppt"}:
        bridge = bridge or configured_legacy_office_bridge()
        return None if bridge is None else bridge.processor_fingerprint(suffix)
    reader = get_reader(path, engine=engine)
    if reader is None:
        return None
    return _reader_processor_fingerprint(reader)


def _reader_processor_fingerprint(reader: DocumentReader) -> ProcessorFingerprint:
    if reader is read_with_docling:
        return docling_fingerprint()
    if reader is read_plain_text:
        return ProcessorFingerprint(PLAIN_TEXT_READER_NAME, PLAIN_TEXT_READER_VERSION)
    if reader is read_image:
        return configured_image_processor_fingerprint()
    if reader is read_pdf:
        return ProcessorFingerprint(PDF_READER_NAME, PDF_READER_VERSION)
    if reader is read_docx:
        return DOCX_READER
    if reader is read_pptx:
        return PPTX_READER
    raise RuntimeError("configured document reader has no frozen processor fingerprint")


def supported_suffixes(
    *,
    engine: str | None = None,
    legacy_office_bridge: LegacyOfficeBridge | None = None,
) -> list[str]:
    resolved = engine or configured_engine()
    suffixes = set(_NATIVE_READERS)
    if resolved == DOCLING_ENGINE and docling_available():
        suffixes |= _DOCLING_SUFFIXES
    if legacy_office_bridge is not None or configured_legacy_office_bridge() is not None:
        suffixes |= {".doc", ".ppt"}
    return sorted(suffixes)


def read_document(
    path: Path,
    *,
    engine: str | None = None,
    legacy_office_bridge: LegacyOfficeBridge | None = None,
) -> ProcessingResult:
    """读取一个文档；没有适配器声明支持其格式时抛出异常。"""

    if (
        path.suffix.lower() in {".doc", ".ppt"}
        and legacy_office_bridge is None
        and configured_legacy_office_bridge() is None
    ):
        raise UnsupportedLegacyOfficeFormat(path.suffix.lower())
    pdf_rejection: ProcessingDiagnostic | None = None
    if path.suffix.lower() == ".pdf":
        reader, pdf_rejection = _configured_pdf_reader(
            path,
            engine or configured_engine(),
        )
    else:
        reader = get_reader(
            path,
            engine=engine,
            legacy_office_bridge=legacy_office_bridge,
        )
    if reader is None:
        raise UnsupportedDocumentFormat(path.suffix.lower())
    pdf_has_xfa = False
    if path.suffix.lower() == ".pdf":
        processor = _reader_processor_fingerprint(reader)
        # 在要求 pypdf 解析安全 trailer 前先根据元数据拒绝。否则，原生或 Docling reader 尚未
        # 有机会执行公共 64 MiB 准入上限，巨大的本地文件就会先突破它。
        try:
            source_size = path.stat().st_size
        except OSError:
            source_size = None
        if source_size is not None and source_size > MAX_DOCUMENT_FILE_BYTES:
            return ProcessingResult(
                elements=(),
                processor=processor,
                diagnostics=(ProcessingDiagnostic(
                    DiagnosticCode.LIMIT_REACHED,
                    detail="document file byte limit reached",
                ),),
            )
        if pdf_rejection is not None:
            return ProcessingResult(
                elements=(),
                processor=processor,
                diagnostics=(pdf_rejection,),
            )
        if _pdf_requires_password(path):
            return ProcessingResult(
                elements=(),
                processor=processor,
                diagnostics=(ProcessingDiagnostic(DiagnosticCode.PASSWORD_REQUIRED),),
            )
        pdf_has_xfa = _pdf_has_xfa(path)
    result = reader(path)
    if path.suffix.lower() == ".pdf" and reader is read_with_docling:
        result = merge_pdf_native_table_inventory(path, result)
        result = merge_pdf_annotation_inventory(path, result)
    if pdf_has_xfa:
        result = ProcessingResult(
            elements=result.elements,
            processor=result.processor,
            diagnostics=(*result.diagnostics, ProcessingDiagnostic(
                DiagnosticCode.PARSER_PARTIAL,
                detail="PDF XFA form data is not supported",
            )),
            page_manifest=result.page_manifest,
        )
    return result


def _pdf_requires_password(path: Path) -> bool:
    """检查 PDF 安全设置，但不阻止空用户密码。"""

    try:
        from pypdf import PdfReader

        return pdf_reader_requires_password(PdfReader(str(path), strict=False))
    except Exception:
        return False


def _pdf_has_xfa(path: Path) -> bool:
    """检测两个已配置 reader 都未建立清单的语义 XFA 表单数据。"""

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path), strict=False)
        root = reader.trailer.get("/Root")
        root = root.get_object() if hasattr(root, "get_object") else root
        acroform = None if root is None else root.get("/AcroForm")
        acroform = (
            acroform.get_object()
            if hasattr(acroform, "get_object")
            else acroform
        )
        return bool(acroform is not None and "/XFA" in acroform)
    except Exception:
        return False


__all__ = [
    "DOCLING_ENGINE",
    "ENGINE_ENV_VAR",
    "NATIVE_ENGINE",
    "DocumentReader",
    "UnsupportedDocumentFormat",
    "UnsupportedLegacyOfficeFormat",
    "LegacyOfficeBridge",
    "configured_legacy_office_bridge",
    "configured_engine",
    "configured_processor_fingerprint",
    "docling_available",
    "get_reader",
    "read_image",
    "validate_image_payload",
    "read_docx",
    "read_document",
    "read_pdf",
    "read_plain_text",
    "read_pptx",
    "read_with_docling",
    "supported_suffixes",
]
