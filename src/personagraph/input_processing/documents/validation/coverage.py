"""基于不可变页面与 chunk authority 的机械文档资格判断。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..chunking import DocumentChunk
from ..contracts import (
    DiagnosticCode,
    DocumentNonTextKind,
    DocumentPageManifest,
    DocumentPageInventoryStatus,
    DocumentPageState,
    ProcessingResult,
)


PAPER_MIN_PHYSICAL_PAGES = 10
PAPER_MAX_PHYSICAL_PAGES = 30
PAPER_REQUIRED_DETECTOR_CAPABILITIES = frozenset({
    "figure_inventory",
    "formula_inventory",
    "nontext_unit_inventory",
    "physical_page_inventory",
    "table_inventory",
    "text_element_source_pages",
    "typed_page_diagnostics",
})


@dataclass(frozen=True, slots=True)
class PaperPageAuthorityEligibility:
    eligible: bool
    reason_codes: tuple[str, ...] = ()
    coverage_gaps: tuple[PageAuthorityCoverageGap, ...] = ()

    def __post_init__(self) -> None:
        if self.eligible == bool(self.reason_codes):
            raise ValueError("eligible papers have no reason codes; rejected papers require one")


@dataclass(frozen=True, slots=True)
class PageAuthorityCoverageGap:
    """一个仍需 dossier 级解释的已定位非文本单元。"""

    unit_id: str | None
    kind: DocumentNonTextKind | None
    source_pages: tuple[int, ...]
    gap_code: str
    requires_visual_read: bool

    def __post_init__(self) -> None:
        if (self.unit_id is not None and not self.unit_id.strip()) or not self.gap_code.strip():
            raise ValueError("coverage gap identity must not be empty")
        if (self.unit_id is None) != (self.kind is None):
            raise ValueError("coverage gap unit identity and kind must appear together")


class PageBoundChunk(Protocol):
    """临时 chunk 与持久 Host 绑定共享的最小形状。"""

    element_ids: tuple[str, ...]
    source_pages: tuple[int, ...]


def evaluate_paper_page_authority_eligibility(
    result: ProcessingResult,
    chunks: Sequence[DocumentChunk],
) -> PaperPageAuthorityEligibility:
    """判断机械上是否支持页面完整的 dossier 工作。

    span 端点被刻意忽略。只有各 chunk 的精确 ``source_pages`` 和元素 ID 才能建立页面覆盖。
    """

    return evaluate_page_authority_eligibility(
        result.page_manifest,
        chunks,
        processing_diagnostic_codes=tuple(
            diagnostic.code.value for diagnostic in result.diagnostics
        ),
    )


def evaluate_page_authority_eligibility(
    manifest: DocumentPageManifest | None,
    chunks: Sequence[PageBoundChunk],
    *,
    processing_diagnostic_codes: Sequence[str] = (),
) -> PaperPageAuthorityEligibility:
    """直接根据持久 Host 投影评估同一道门。"""

    reasons: list[str] = []
    if manifest is None:
        return PaperPageAuthorityEligibility(
            False,
            ("page_manifest_unavailable",),
        )
    if manifest.inventory_status is not DocumentPageInventoryStatus.COMPLETE:
        reasons.append("page_inventory_not_complete")
    if not (
        PAPER_MIN_PHYSICAL_PAGES
        <= manifest.physical_page_count
        <= PAPER_MAX_PHYSICAL_PAGES
    ):
        reasons.append("physical_page_count_out_of_range")
    if not PAPER_REQUIRED_DETECTOR_CAPABILITIES <= set(
        manifest.detector_capabilities
    ):
        reasons.append("detector_capability_incomplete")
    if any(page.state is DocumentPageState.UNREADABLE for page in manifest.pages):
        reasons.append("unreadable_page")
    allowed_diagnostic_codes = {
        DiagnosticCode.PAGE_EMPTY.value,
        DiagnosticCode.PAGE_NEEDS_VISION.value,
    }
    if any(code not in allowed_diagnostic_codes for code in processing_diagnostic_codes):
        reasons.append("silent_loss_diagnostic")
    manifest_codes = {
        diagnostic.code.value
        for page in manifest.pages
        for diagnostic in page.diagnostics
    }
    if (
        DiagnosticCode.PAGE_NEEDS_VISION.value in processing_diagnostic_codes
        and DiagnosticCode.PAGE_NEEDS_VISION.value not in manifest_codes
    ):
        reasons.append("silent_loss_diagnostic")
    for page in manifest.pages:
        page_codes = {diagnostic.code for diagnostic in page.diagnostics}
        requires_visual = any(
            unit.requires_visual_read for unit in page.nontext_units
        )
        if requires_visual != (DiagnosticCode.PAGE_NEEDS_VISION in page_codes):
            reasons.append("silent_loss_diagnostic")
        if (
            (DiagnosticCode.PAGE_EMPTY in page_codes)
            != (page.state is DocumentPageState.NO_EXTRACTABLE_CONTENT)
        ):
            reasons.append("silent_loss_diagnostic")

    element_pages: dict[str, set[int]] = {}
    for page in manifest.pages:
        for element_id in page.text_element_ids:
            element_pages.setdefault(element_id, set()).add(page.page_number)
    covered_ids: set[str] = set()
    for chunk in chunks:
        if not chunk.source_pages:
            reasons.append("chunk_source_pages_missing")
            continue
        expected_pages: set[int] = set()
        unknown_id = False
        for element_id in chunk.element_ids:
            pages = element_pages.get(element_id)
            if pages is None:
                unknown_id = True
                continue
            covered_ids.add(element_id)
            expected_pages.update(pages)
        if unknown_id or expected_pages != set(chunk.source_pages):
            reasons.append("chunk_source_pages_mismatch")
    if set(element_pages) - covered_ids:
        reasons.append("text_element_not_chunked")

    gaps_by_id: dict[str, PageAuthorityCoverageGap] = {}
    for page in manifest.pages:
        if page.state is DocumentPageState.NO_EXTRACTABLE_CONTENT:
            gaps_by_id[f"page:{page.page_number}"] = PageAuthorityCoverageGap(
                unit_id=None,
                kind=None,
                source_pages=(page.page_number,),
                gap_code="no_extractable_content",
                requires_visual_read=False,
            )
        for unit in page.nontext_units:
            gaps_by_id.setdefault(
                unit.unit_id,
                PageAuthorityCoverageGap(
                    unit_id=unit.unit_id,
                    kind=unit.kind,
                    source_pages=unit.source_pages,
                    gap_code=(
                        "visual_interpretation_pending"
                        if unit.requires_visual_read
                        else "nontext_semantics_pending"
                    ),
                    requires_visual_read=unit.requires_visual_read,
                ),
            )
    unique_reasons = tuple(dict.fromkeys(reasons))
    return PaperPageAuthorityEligibility(
        not unique_reasons,
        unique_reasons,
        tuple(gaps_by_id.values()),
    )
