"""文档输入处理的窄入口。

包初始化不加载 reader、vision 或存储实现；这些适配器之间存在合法的
反向组合路径，eager re-export 会将它们变成包级循环依赖。为保留现有公开符号，
本模块只在实际访问时延迟解析所属模块。
"""

from __future__ import annotations

from importlib import import_module
from typing import Final


_EXPORT_MODULES: Final[dict[str, str]] = {
    # chunking
    "ChunkSpan": ".chunking",
    "ChunkingProfile": ".chunking",
    "DocumentChunk": ".chunking",
    "chunk_document": ".chunking",
    "chunker_fingerprint": ".chunking",
    # contracts
    "DiagnosticCode": ".contracts",
    "DocumentElement": ".contracts",
    "DocumentLocator": ".contracts",
    "DocumentNonTextKind": ".contracts",
    'DocumentNonTextUnit': ".contracts",
    "DocumentPageInventoryStatus": ".contracts",
    'DocumentPageManifest': ".contracts",
    'DocumentPageRecord': ".contracts",
    "DocumentPageState": ".contracts",
    "DocumentTextEvidenceOrigin": ".contracts",
    'DocumentTextEvidence': ".contracts",
    "ElementKind": ".contracts",
    "ProcessingAdmissionStatus": ".contracts",
    "ProcessingDiagnostic": ".contracts",
    "ProcessingResult": ".contracts",
    "ProcessorFingerprint": ".contracts",
    "make_element_id": ".contracts",
    # coverage
    "PageAuthorityCoverageGap": ".validation",
    "PaperPageAuthorityEligibility": ".validation",
    "evaluate_page_authority_eligibility": ".validation",
    "evaluate_paper_page_authority_eligibility": ".validation",
    # readers
    "DOCLING_ENGINE": ".readers",
    "NATIVE_ENGINE": ".readers",
    "UnsupportedDocumentFormat": ".readers",
    "UnsupportedLegacyOfficeFormat": ".readers",
    "configured_engine": ".readers",
    "docling_available": ".readers",
    "get_reader": ".readers",
    "read_document": ".readers",
    "supported_suffixes": ".readers",
    # pure preparation
    "DocumentPrepareFailure": ".preparation",
    "PreparedDocumentIngest": ".preparation",
    "prepare_document_path": ".preparation",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORT_MODULES})
