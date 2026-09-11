"""先分类各页面，再决定读取方式的 PDF reader。

PDF 并非单一类型。若把每一页都当作可提取文本，扫描文档就会变成对无人读取页面的自信回答：
提取器返回空字符串，空字符串因“无意义”被丢弃，而模型最终误以为自己收到了文档。

因此，每一页都会先分类；无法产生文本的页面会生成显式图像元素及诊断——绝不会静默跳过。
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path

from ..contracts import (
    DiagnosticCode,
    DocumentElement,
    DocumentLocator,
    DocumentNonTextKind,
    DocumentNonTextUnit,
    DocumentPageInventoryStatus,
    DocumentPageManifest,
    DocumentPageRecord,
    DocumentPageState,
    ElementKind,
    ProcessingDiagnostic,
    ProcessingResult,
    ProcessorFingerprint,
    make_element_id,
)
from ...files import SourceSizeLimitError, fingerprint_file


READER_NAME = "pdfplumber"
READER_VERSION = "8"

# 字符数少于此值的页面会被视为没有可用文本层，即使技术上提取到了内容：扫描器经常会从页眉
# 或 OCR 水印中留下少量游离字形。
MIN_TEXT_CHARS_PER_PAGE = 16

MAX_PAGES = 2000
# 保留历史上的小文档宽限，对长 PDF 按比例扩展，同时维持绝对内存边界。每个物理页面 128
# 个元素足以覆盖按行提取，又不会让压缩的恶意源生成无界内存元素图。
MAX_ELEMENTS = 20000
MAX_ELEMENTS_PER_PAGE = 128
MAX_DOCUMENT_ELEMENTS = 200000
MAX_ELEMENT_CHARS = 20000

# 只有当检测到的网格具备足够证据，能证明提取并未止于表头或少量单元格时，原生文本才能关闭
# 布局 reader 的显式视觉表格 gap。这些保守阈值有意优先保留视觉 gap，而非错误宣称完整理解。
MIN_NATIVE_TABLE_ROWS_FOR_RECONCILIATION = 2
MIN_NATIVE_TABLE_COLUMNS_FOR_RECONCILIATION = 2
MIN_NATIVE_TABLE_CELL_COVERAGE_FOR_RECONCILIATION = 0.75


class PageKind(StrEnum):
    TEXT = "text"
    SCANNED = "scanned"
    MIXED = "mixed"
    EMPTY = "empty"


class PdfPasswordRequired(RuntimeError):
    """文档已加密；不会尝试绕过加密。"""


def pdf_reader_requires_password(reader: object) -> bool:
    """返回加密 reader 是否需要非空用户密码。

    许多公开 PDF 带有 owner-password 权限标志，却刻意将用户密码留空。这些源无须密钥即可
    读取，不能仅因 ``is_encrypted`` 为 true 就拒绝。
    """

    if not bool(getattr(reader, "is_encrypted", False)):
        return False
    decrypt = getattr(reader, "decrypt", None)
    if not callable(decrypt):
        return True
    try:
        return not bool(decrypt(""))
    except Exception:
        return True


def _document_element_limit(physical_page_count: int) -> int:
    if physical_page_count < 0:
        raise ValueError("physical_page_count must not be negative")
    return min(
        MAX_DOCUMENT_ELEMENTS,
        max(MAX_ELEMENTS, physical_page_count * MAX_ELEMENTS_PER_PAGE),
    )


def classify_page(
    char_count: int,
    image_count: int,
    *,
    vector_count: int = 0,
) -> PageKind:
    """根据解析器在页面上的实际发现判断页面类型。"""

    has_text = char_count >= MIN_TEXT_CHARS_PER_PAGE
    has_images = image_count > 0 or vector_count > 0
    if has_text and has_images:
        return PageKind.MIXED
    if has_text:
        return PageKind.TEXT
    if has_images:
        return PageKind.SCANNED
    return PageKind.EMPTY


def read_pdf(
    path: Path,
    *,
    _native_table_quality: dict[str, bool] | None = None,
    _native_vector_bboxes: dict[
        int, tuple[tuple[float, float, float, float], ...] | None
    ] | None = None,
) -> ProcessingResult:
    """将一个 PDF 读取为格式中立元素和如实诊断。"""

    import pdfplumber
    from pdfplumber.utils.exceptions import PdfminerException

    processor = ProcessorFingerprint(READER_NAME, READER_VERSION)
    elements: list[DocumentElement] = []
    diagnostics: list[ProcessingDiagnostic] = []
    try:
        source_key = fingerprint_file(path).sha256
    except SourceSizeLimitError:
        return ProcessingResult(
            elements=(),
            processor=processor,
            diagnostics=(ProcessingDiagnostic(
                DiagnosticCode.LIMIT_REACHED,
                detail="document file byte limit reached",
            ),),
        )
    except PermissionError as exc:
        return ProcessingResult(
            elements=(),
            processor=processor,
            diagnostics=(ProcessingDiagnostic(
                DiagnosticCode.PERMISSION_DENIED,
                detail=type(exc).__name__,
            ),),
        )
    except OSError as exc:
        return ProcessingResult(
            elements=(),
            processor=processor,
            diagnostics=(ProcessingDiagnostic(
                DiagnosticCode.CORRUPT_SOURCE,
                detail=type(exc).__name__,
            ),),
        )

    try:
        pdf = pdfplumber.open(str(path))
    except PdfminerException as exc:
        code = (
            DiagnosticCode.PASSWORD_REQUIRED
            if "password" in str(exc).lower()
            else DiagnosticCode.CORRUPT_SOURCE
        )
        return ProcessingResult(
            elements=(),
            processor=processor,
            diagnostics=(ProcessingDiagnostic(code, detail=type(exc).__name__),),
        )

    with pdf:
        physical_page_count = len(pdf.pages)
        if physical_page_count > MAX_PAGES:
            return ProcessingResult(
                elements=(),
                processor=processor,
                diagnostics=(ProcessingDiagnostic(
                    DiagnosticCode.LIMIT_REACHED,
                    detail=(
                        f"physical page count {physical_page_count} exceeds "
                        f"the supported limit {MAX_PAGES}"
                    ),
                ),),
            )
        element_limit = _document_element_limit(physical_page_count)
        page_records: list[DocumentPageRecord] = []
        for page_index, page in enumerate(pdf.pages, start=1):
            try:
                if len(elements) >= element_limit:
                    diagnostic = ProcessingDiagnostic(
                        DiagnosticCode.LIMIT_REACHED,
                        DocumentLocator(page=page_index),
                        detail="element budget exhausted",
                    )
                    diagnostics.append(diagnostic)
                    page_records.append(DocumentPageRecord(
                        page_number=page_index,
                        state=DocumentPageState.UNREADABLE,
                        diagnostics=(diagnostic,),
                    ))
                    continue
                page_records.append(_read_page(
                    page, page_index, source_key,
                    elements=elements, diagnostics=diagnostics,
                    native_table_quality=_native_table_quality,
                    native_vector_bboxes=_native_vector_bboxes,
                    element_limit=element_limit,
                ))
            finally:
                _close_page_cache(page)

    if not elements and not diagnostics:
        diagnostics.append(ProcessingDiagnostic(DiagnosticCode.EMPTY_SOURCE))
    page_manifest = DocumentPageManifest(
        physical_page_count=physical_page_count,
        inventory_status=(
            DocumentPageInventoryStatus.COMPLETE
            if physical_page_count
            else DocumentPageInventoryStatus.UNAVAILABLE
        ),
        detector_fingerprint=f"{processor}:physical-pages-v1",
        detector_capabilities=tuple(sorted({
            "nontext_unit_inventory",
            "physical_page_inventory",
            "raster_image_inventory",
            "table_inventory",
            "text_element_source_pages",
            "typed_page_diagnostics",
            "vector_graphics_inventory",
        })),
        pages=tuple(page_records),
    )
    return ProcessingResult(
        elements=tuple(elements),
        processor=processor,
        diagnostics=tuple(diagnostics),
        page_manifest=page_manifest,
    )


def merge_pdf_annotation_inventory(
    path: Path,
    result: ProcessingResult,
) -> ProcessingResult:
    """将有界 PDF 表单/评论 authority 合并到另一个 PDF reader。

    布局 reader 适合 OCR 和页面结构，但可能静默遗漏 AcroForm 值与语义 annotation。此窄
    pass 只读取 annotation 清单，并追加缺失文本或已定位视觉 gap，同时保留主 reader 的
    processor 身份和页面结构。
    """

    import pdfplumber

    manifest = result.page_manifest
    if manifest is None or manifest.physical_page_count == 0:
        return result
    try:
        pdf = pdfplumber.open(str(path))
    except Exception as exc:
        return ProcessingResult(
            elements=result.elements,
            processor=result.processor,
            diagnostics=(*result.diagnostics, ProcessingDiagnostic(
                DiagnosticCode.PARSER_PARTIAL,
                detail=f"AnnotationInventory:{type(exc).__name__}",
            )),
            page_manifest=manifest,
        )

    source_key = fingerprint_file(path).sha256
    elements = list(result.elements)
    diagnostics = list(result.diagnostics)
    records = list(manifest.pages)
    changed = False
    with pdf:
        if len(pdf.pages) != manifest.physical_page_count:
            return ProcessingResult(
                elements=result.elements,
                processor=result.processor,
                diagnostics=(*result.diagnostics, ProcessingDiagnostic(
                    DiagnosticCode.PARSER_PARTIAL,
                    detail="AnnotationInventoryPageCountDrift",
                )),
                page_manifest=manifest,
            )
        for page_number, page in enumerate(pdf.pages, start=1):
            try:
                raw_annotations = tuple(getattr(page, "annots", ()) or ())
            except Exception as exc:
                raw_annotations = ()
                annotation_failure = ProcessingDiagnostic(
                    DiagnosticCode.PARSER_PARTIAL,
                    DocumentLocator(page=page_number),
                    detail=f"AnnotationInventory:{type(exc).__name__}",
                )
            else:
                annotation_failure = None
            if not raw_annotations and annotation_failure is None:
                continue

            record = records[page_number - 1]
            page_elements = tuple(
                element
                for element in elements
                if page_number in element.source_pages
            )
            existing_text = tuple(
                _normalized_annotation_text(element.text)
                for element in page_elements
                if (element.text or "").strip()
                and element.locator.section_path[:1] == ("pdf_annotation",)
            )
            annotations = tuple(
                raw
                for raw in raw_annotations
                if not _annotation_already_present(raw, existing_text)
            )
            element_start = len(elements)
            diagnostic_start = len(diagnostics)
            ordinal = 1 + max(
                (
                    element.locator.ordinal
                    for element in page_elements
                    if element.locator.ordinal is not None
                ),
                default=-1,
            )
            new_units: tuple[DocumentNonTextUnit, ...] = ()
            if annotations:
                _ordinal, new_units = _emit_annotations(
                    annotations,
                    page,
                    page_number,
                    source_key,
                    elements,
                    diagnostics,
                    ordinal,
                )
            if annotation_failure is not None:
                diagnostics.append(annotation_failure)
            new_elements = tuple(elements[element_start:])
            new_diagnostics = tuple(diagnostics[diagnostic_start:])
            if not new_elements and not new_diagnostics and not new_units:
                continue
            changed = True
            text_ids = (
                *record.text_element_ids,
                *(
                    element.element_id
                    for element in new_elements
                    if (element.text or "").strip()
                ),
            )
            units = (*record.nontext_units, *new_units)
            page_diagnostics = (*record.diagnostics, *new_diagnostics)
            records[page_number - 1] = DocumentPageRecord(
                page_number=page_number,
                state=_merged_page_state(
                    prior=record.state,
                    text_ids=text_ids,
                    units=units,
                    diagnostics=page_diagnostics,
                ),
                text_element_ids=text_ids,
                nontext_units=units,
                diagnostics=page_diagnostics,
            )

    if not changed:
        return result
    merged_manifest = DocumentPageManifest(
        physical_page_count=manifest.physical_page_count,
        inventory_status=manifest.inventory_status,
        detector_fingerprint=manifest.detector_fingerprint,
        detector_capabilities=tuple(sorted({
            *manifest.detector_capabilities,
            "semantic_annotation_inventory",
        })),
        pages=tuple(records),
    )
    return ProcessingResult(
        elements=tuple(elements),
        processor=result.processor,
        diagnostics=tuple(diagnostics),
        page_manifest=merged_manifest,
    )


def merge_pdf_native_table_inventory(
    path: Path,
    result: ProcessingResult,
) -> ProcessingResult:
    """将有界原生表格文本融合到布局 reader 结果中。

    Docling 是布局和 OCR 的更强默认方案，但它可能把机械可提取的带边框表格归类为未解析
    视觉对象。原生 reader 已持有有界、类型化的表格清单，因此保留主页面 authority，只追加
    其未表示的表格。
    """

    manifest = result.page_manifest
    if manifest is None or manifest.physical_page_count == 0:
        return result
    native_table_quality: dict[str, bool] = {}
    native_vector_bboxes: dict[
        int, tuple[tuple[float, float, float, float], ...] | None
    ] = {}
    native = read_pdf(
        path,
        _native_table_quality=native_table_quality,
        _native_vector_bboxes=native_vector_bboxes,
    )
    native_manifest = native.page_manifest
    if (
        native_manifest is None
        or native_manifest.physical_page_count != manifest.physical_page_count
    ):
        return ProcessingResult(
            elements=result.elements,
            processor=result.processor,
            diagnostics=(*result.diagnostics, ProcessingDiagnostic(
                DiagnosticCode.PARSER_PARTIAL,
                detail="Native PDF table inventory unavailable",
            )),
            page_manifest=manifest,
        )

    source_key = fingerprint_file(path).sha256
    page_heights = _pdf_page_heights(
        path,
        expected_page_count=manifest.physical_page_count,
    )
    elements = list(result.elements)
    elements_by_id = {element.element_id: element for element in elements}
    records = list(manifest.pages)
    removed_element_ids: set[str] = set()
    removed_diagnostics: set[ProcessingDiagnostic] = set()
    changed = False
    for page_number in range(1, manifest.physical_page_count + 1):
        record = records[page_number - 1]
        primary_page_elements = [
            element for element in elements if page_number in element.source_pages
        ]
        represented_table_counts: dict[str, int] = {}
        for element in primary_page_elements:
            if element.kind is not ElementKind.TABLE or not (element.text or "").strip():
                continue
            normalized = _normalized_table_text(element.text)
            represented_table_counts[normalized] = (
                represented_table_counts.get(normalized, 0) + 1
            )
        ordinal = 1 + max(
            (
                element.locator.ordinal
                for element in primary_page_elements
                if element.locator.ordinal is not None
            ),
            default=-1,
        )
        native_record = native_manifest.pages[page_number - 1]
        native_tables: list[DocumentElement] = []
        if not any(
            diagnostic.code
            in {DiagnosticCode.CORRUPT_SOURCE, DiagnosticCode.LIMIT_REACHED}
            for diagnostic in native_record.diagnostics
        ):
            for table in native.elements:
                if (
                    table.kind is not ElementKind.TABLE
                    or page_number not in table.source_pages
                ):
                    continue
                normalized = _normalized_table_text(table.text)
                if not normalized:
                    continue
                represented_count = represented_table_counts.get(normalized, 0)
                if represented_count:
                    represented_table_counts[normalized] = represented_count - 1
                    continue
                native_tables.append(table)

        resolved_by_native_index = _match_unresolved_visual_tables(
            record=record,
            native_record=native_record,
            native_tables=tuple(native_tables),
            primary_elements_by_id=elements_by_id,
            page_height=(
                page_heights[page_number - 1]
                if len(page_heights) == manifest.physical_page_count
                else None
            ),
            trusted_native_indices=frozenset(
                index
                for index, table in enumerate(native_tables)
                if native_vector_bboxes.get(page_number) is not None
                and native_table_quality.get(table.element_id, False)
            ),
            native_vector_bboxes=native_vector_bboxes.get(page_number) or (),
        )
        replacement_units: dict[str, DocumentNonTextUnit] = {}
        new_elements: list[DocumentElement] = []
        appended_units: list[DocumentNonTextUnit] = []
        for native_index, table in enumerate(native_tables):
            resolved = resolved_by_native_index.get(native_index)
            if resolved is not None:
                locator = resolved.locator
                fused = DocumentElement(
                    element_id=make_element_id(source_key, locator, table.text),
                    kind=ElementKind.TABLE,
                    text=table.text,
                    locator=locator,
                    source_pages=resolved.source_pages,
                )
                replacement_units[resolved.unit_id] = DocumentNonTextUnit(
                    unit_id=resolved.unit_id,
                    kind=DocumentNonTextKind.TABLE,
                    source_pages=resolved.source_pages,
                    text_element_ids=(fused.element_id,),
                    element_id=fused.element_id,
                    locator=locator,
                    requires_visual_read=False,
                )
                if resolved.element_id is not None:
                    removed_element_ids.add(resolved.element_id)
                for diagnostic in record.diagnostics:
                    if (
                        diagnostic.code is DiagnosticCode.PAGE_NEEDS_VISION
                        and diagnostic.locator == resolved.locator
                        and diagnostic.detail == _TABLE_VISUAL_DIAGNOSTIC_DETAIL
                    ):
                        removed_diagnostics.add(diagnostic)
                new_elements.append(fused)
                continue

            locator = DocumentLocator(
                page=page_number,
                ordinal=ordinal,
                section_path=("pdf_native_table",),
            )
            fused = DocumentElement(
                element_id=make_element_id(source_key, locator, table.text),
                kind=ElementKind.TABLE,
                text=table.text,
                locator=locator,
                source_pages=(page_number,),
            )
            unit = DocumentNonTextUnit(
                unit_id=_nontext_unit_id(
                    source_key,
                    DocumentNonTextKind.TABLE,
                    page_number,
                    ordinal,
                ),
                kind=DocumentNonTextKind.TABLE,
                source_pages=(page_number,),
                text_element_ids=(fused.element_id,),
                element_id=fused.element_id,
                locator=locator,
                requires_visual_read=False,
            )
            new_elements.append(fused)
            appended_units.append(unit)
            ordinal += 1
        if not new_elements:
            continue
        changed = True
        elements.extend(new_elements)
        text_ids = (*record.text_element_ids, *(item.element_id for item in new_elements))
        units = (
            *(
                replacement_units.get(unit.unit_id, unit)
                for unit in record.nontext_units
            ),
            *appended_units,
        )
        diagnostics = tuple(
            diagnostic
            for diagnostic in record.diagnostics
            if diagnostic not in removed_diagnostics
        )
        records[page_number - 1] = DocumentPageRecord(
            page_number=page_number,
            state=_merged_page_state(
                prior=record.state,
                text_ids=text_ids,
                units=units,
                diagnostics=diagnostics,
            ),
            text_element_ids=text_ids,
            nontext_units=units,
            diagnostics=diagnostics,
        )

    if not changed:
        return result
    merged_manifest = DocumentPageManifest(
        physical_page_count=manifest.physical_page_count,
        inventory_status=manifest.inventory_status,
        detector_fingerprint=manifest.detector_fingerprint,
        detector_capabilities=tuple(sorted({
            *manifest.detector_capabilities,
            "native_table_inventory",
        })),
        pages=tuple(records),
    )
    return ProcessingResult(
        elements=tuple(
            element
            for element in elements
            if element.element_id not in removed_element_ids
        ),
        processor=result.processor,
        diagnostics=tuple(
            diagnostic
            for diagnostic in result.diagnostics
            if diagnostic not in removed_diagnostics
        ),
        page_manifest=merged_manifest,
    )


def _normalized_table_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).casefold()


_TABLE_RECONCILIATION_MIN_IOU = 0.7
_TABLE_VISUAL_DIAGNOSTIC_DETAIL = "table requires visual interpretation"


def _match_unresolved_visual_tables(
    *,
    record: DocumentPageRecord,
    native_record: DocumentPageRecord,
    native_tables: tuple[DocumentElement, ...],
    primary_elements_by_id: Mapping[str, DocumentElement],
    page_height: float | None,
    trusted_native_indices: frozenset[int],
    native_vector_bboxes: tuple[tuple[float, float, float, float], ...],
) -> dict[int, DocumentNonTextUnit]:
    """只为原生表格回退返回无歧义几何匹配。

    Docling 报告左下原点 PDF 坐标，而 pdfplumber 报告左上原点坐标。计数和解析器局部序号
    不是身份：页面可能同时包含互不相关的栅格表格与原生表格。因此只有当两个 box 高度重叠，
    且两边各自恰好只有一个合格对应项时，才解析 gap。
    """

    if page_height is None or not math.isfinite(page_height) or page_height <= 0:
        return {}
    unresolved: list[DocumentNonTextUnit] = []
    for unit in record.nontext_units:
        evidence = (
            primary_elements_by_id.get(unit.element_id)
            if unit.element_id is not None
            else None
        )
        if (
            unit.kind is DocumentNonTextKind.TABLE
            and unit.requires_visual_read
            and unit.source_pages == (record.page_number,)
            and not unit.text_element_ids
            and unit.locator.bbox is not None
            and evidence is not None
            and evidence.kind is ElementKind.IMAGE
            and evidence.needs_vision
            and not (evidence.text or "").strip()
            and any(
                diagnostic.code is DiagnosticCode.PAGE_NEEDS_VISION
                and diagnostic.locator == unit.locator
                and diagnostic.detail == _TABLE_VISUAL_DIAGNOSTIC_DETAIL
                for diagnostic in record.diagnostics
            )
        ):
            unresolved.append(unit)
    if not unresolved or not native_tables:
        return {}

    native_visual_units = tuple(
        unit
        for unit in native_record.nontext_units
        if unit.kind in {DocumentNonTextKind.FIGURE, DocumentNonTextKind.FORMULA}
        and unit.requires_visual_read
        and unit.locator.bbox is not None
    )
    eligible_native_indices = frozenset(
        index
        for index, table in enumerate(native_tables)
        if index in trusted_native_indices
        and table.locator.bbox is not None
    )
    candidates_by_unit: dict[str, tuple[int, ...]] = {}
    units_by_native: dict[int, list[DocumentNonTextUnit]] = {
        index: [] for index in range(len(native_tables))
    }
    for unit in unresolved:
        assert unit.locator.bbox is not None
        primary_bbox = _bottom_left_to_top_left_bbox(
            unit.locator.bbox,
            page_height=page_height,
        )
        primary_gap_has_visual_content = any(
            _visual_unit_blocks_table(primary_bbox, visual_unit)
            for visual_unit in native_visual_units
        ) or any(
            _bbox_intersection_area(primary_bbox, vector_bbox) > 0
            for vector_bbox in native_vector_bboxes
        )
        candidates = (
            ()
            if primary_gap_has_visual_content
            else tuple(
                index
                for index, table in enumerate(native_tables)
                if index in eligible_native_indices
                and table.locator.bbox is not None
                and _bbox_iou(primary_bbox, table.locator.bbox)
                >= _TABLE_RECONCILIATION_MIN_IOU
            )
        )
        candidates_by_unit[unit.unit_id] = candidates
        for index in candidates:
            units_by_native[index].append(unit)

    resolved: dict[int, DocumentNonTextUnit] = {}
    for unit in unresolved:
        candidates = candidates_by_unit[unit.unit_id]
        if len(candidates) != 1:
            continue
        native_index = candidates[0]
        if len(units_by_native[native_index]) != 1:
            continue
        resolved[native_index] = unit
    return resolved


def _bottom_left_to_top_left_bbox(
    bbox: tuple[float, float, float, float],
    *,
    page_height: float,
) -> tuple[float, float, float, float]:
    left, bottom, right, top = bbox
    return (left, page_height - top, right, page_height - bottom)


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x0 = max(left[0], right[0])
    top = max(left[1], right[1])
    x1 = min(left[2], right[2])
    bottom = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, bottom - top)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _visual_unit_blocks_table(
    table_bbox: tuple[float, float, float, float],
    visual_unit: DocumentNonTextUnit,
) -> bool:
    visual_bbox = visual_unit.locator.bbox
    assert visual_bbox is not None
    intersection = _bbox_intersection_area(table_bbox, visual_bbox)
    if intersection <= 0:
        return False
        # 栅格图像、公式或矢量标记可能恰好携带文本提取器遗漏的单元格内容。目前矢量清单按页
        # 聚合，因此面积比例测试可能被远处无关标记稀释。故任何真实重叠都会保持显式视觉 gap
        # 开放。这可能为整页背景保留无害 gap，但绝不会把混合表格转换成错误完整结果。
    return True


def _bbox_intersection_area(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    return max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0,
        min(left[3], right[3]) - max(left[1], right[1]),
    )


def _pdf_page_heights(
    path: Path,
    *,
    expected_page_count: int,
) -> tuple[float, ...]:
    """在 pdfplumber 坐标系中读取页面高度，并保守失败。"""

    try:
        import pdfplumber

        with pdfplumber.open(path) as document:
            if len(document.pages) != expected_page_count:
                return ()
            heights = tuple(float(page.height) for page in document.pages)
    except Exception:
        return ()
    if any(not math.isfinite(height) or height <= 0 for height in heights):
        return ()
    return heights


def _annotation_already_present(
    raw: object,
    existing_text: tuple[str, ...],
) -> bool:
    if not isinstance(raw, Mapping):
        return False
    data = raw.get("data")
    annotation = data if isinstance(data, Mapping) else raw
    subtype = _pdf_name(_mapping_value(annotation, "Subtype"))
    content = _annotation_text(annotation, subtype=subtype)
    normalized = _normalized_annotation_text(content)
    if not normalized:
        return False
    return normalized in existing_text


def _normalized_annotation_text(value: str | None) -> str:
    return " ".join(str(value or "").split()).casefold()


def _merged_page_state(
    *,
    prior: DocumentPageState,
    text_ids: tuple[str, ...],
    units: tuple[DocumentNonTextUnit, ...],
    diagnostics: tuple[ProcessingDiagnostic, ...],
) -> DocumentPageState:
    if any(
        diagnostic.code in {DiagnosticCode.CORRUPT_SOURCE, DiagnosticCode.LIMIT_REACHED}
        for diagnostic in diagnostics
    ):
        return DocumentPageState.UNREADABLE
    if text_ids and units:
        return DocumentPageState.MIXED
    if text_ids:
        return DocumentPageState.TEXT
    if units:
        return DocumentPageState.VISUAL_ONLY
    return prior


def _read_page(
    page,
    page_index: int,
    source_key: str,
    *,
    elements: list[DocumentElement],
    diagnostics: list[ProcessingDiagnostic],
    native_table_quality: dict[str, bool] | None = None,
    native_vector_bboxes: dict[
        int, tuple[tuple[float, float, float, float], ...] | None
    ] | None = None,
    element_limit: int | None = None,
) -> DocumentPageRecord:
    diagnostic_start = len(diagnostics)
    element_start = len(elements)
    try:
        chars = tuple(page.chars)
        images = tuple(page.images)
        vector_objects = tuple(page.curves) + tuple(page.rects) + tuple(page.lines)
    except Exception as exc:
        diagnostic = ProcessingDiagnostic(
            DiagnosticCode.CORRUPT_SOURCE,
            DocumentLocator(page=page_index),
            detail=type(exc).__name__,
        )
        diagnostics.append(diagnostic)
        return DocumentPageRecord(
            page_number=page_index,
            state=DocumentPageState.UNREADABLE,
            diagnostics=(diagnostic,),
        )

    try:
        annotations = tuple(getattr(page, "annots", ()) or ())
    except Exception as exc:
        annotations = ()
        diagnostics.append(ProcessingDiagnostic(
            DiagnosticCode.PARSER_PARTIAL,
            DocumentLocator(page=page_index),
            detail=f"AnnotationInventory:{type(exc).__name__}",
        ))

    kind = classify_page(
        len(chars),
        len(images),
        vector_count=len(vector_objects),
    )
    if kind is PageKind.EMPTY and annotations:
    # 表单字段或评论可能是页面上唯一的语义对象。
        kind = PageKind.TEXT
    if kind is PageKind.EMPTY:
        diagnostic = ProcessingDiagnostic(
            DiagnosticCode.PAGE_EMPTY, DocumentLocator(page=page_index)
        )
        diagnostics.append(diagnostic)
        return DocumentPageRecord(
            page_number=page_index,
            state=DocumentPageState.NO_EXTRACTABLE_CONTENT,
            diagnostics=tuple(diagnostics[diagnostic_start:]),
        )

    ordinal = 0
    nontext_units: list[DocumentNonTextUnit] = []
    if kind in {PageKind.TEXT, PageKind.MIXED}:
        (
            ordinal,
            table_units,
            table_bboxes,
            table_grid_segments,
            table_cell_bboxes,
        ) = _emit_tables(
            page,
            page_index,
            source_key,
            elements,
            diagnostics,
            ordinal,
            native_table_quality=native_table_quality,
            element_limit=element_limit,
        )
        nontext_units.extend(table_units)
        ordinal = _emit_text_lines(
            page,
            page_index,
            source_key,
            elements,
            diagnostics,
            ordinal,
            excluded_bboxes=table_bboxes,
            element_limit=element_limit,
        )
    ordinal, annotation_units = _emit_annotations(
        annotations,
        page,
        page_index,
        source_key,
        elements,
        diagnostics,
        ordinal,
        element_limit=element_limit,
    )
    nontext_units.extend(annotation_units)
    if kind in {PageKind.TEXT, PageKind.MIXED}:
        if not any(
            (element.text or "").strip()
            for element in elements[element_start:]
        ) and not any(
            diagnostic.code in {DiagnosticCode.CORRUPT_SOURCE, DiagnosticCode.LIMIT_REACHED}
            for diagnostic in diagnostics[diagnostic_start:]
        ):
            diagnostics.append(ProcessingDiagnostic(
                DiagnosticCode.CORRUPT_SOURCE,
                DocumentLocator(page=page_index),
                detail="TextLayerYieldedNoElements",
            ))

    if images:
        # 页面携带只有视觉能力才能读取的内容。明确说明这一点，调用方才能报告“这些页面是我
        # 无法读取的图像”，而不是把部分文档描述为完整文档。
        for image in images:
            image_element = _image_element(
                page,
                page_index,
                source_key,
                ordinal,
                bbox=_bbox_of(image),
            )
            if not _element_slot_available(
                elements,
                diagnostics,
                image_element.locator,
                element_limit=element_limit,
            ):
                break
            elements.append(image_element)
            nontext_units.append(DocumentNonTextUnit(
                unit_id=_nontext_unit_id(
                    source_key,
                    DocumentNonTextKind.FIGURE,
                    page_index,
                    ordinal,
                ),
                kind=DocumentNonTextKind.FIGURE,
                source_pages=(page_index,),
                element_id=image_element.element_id,
                locator=image_element.locator,
                requires_visual_read=True,
            ))
            ordinal += 1
        # 带边框表格已由其类型化 TABLE 单元表示；网格线是布局支架，不是第二个未读视觉对象。
    unread_vector_objects = tuple(
        item for item in vector_objects
        if not _is_table_grid_scaffolding(
            item,
            table_bboxes=table_bboxes,
            table_grid_segments=table_grid_segments,
            table_cell_bboxes=table_cell_bboxes,
        )
    ) if kind in {PageKind.TEXT, PageKind.MIXED} else vector_objects
    if native_vector_bboxes is not None:
        located_vector_bboxes = tuple(_bbox_of(item) for item in unread_vector_objects)
        native_vector_bboxes[page_index] = (
            tuple(bbox for bbox in located_vector_bboxes if bbox is not None)
            if all(bbox is not None for bbox in located_vector_bboxes)
            else None
        )
    if unread_vector_objects:
        vector_element = _image_element(
            page,
            page_index,
            source_key,
            ordinal,
            bbox=_union_bboxes(
                tuple(
                    bbox
                    for item in unread_vector_objects
                    if (bbox := _bbox_of(item)) is not None
                )
            ),
        )
        if _element_slot_available(
            elements,
            diagnostics,
            vector_element.locator,
            element_limit=element_limit,
        ):
            elements.append(vector_element)
            nontext_units.append(DocumentNonTextUnit(
                unit_id=_nontext_unit_id(
                    source_key,
                    DocumentNonTextKind.VECTOR_GRAPHICS,
                    page_index,
                    ordinal,
                ),
                kind=DocumentNonTextKind.VECTOR_GRAPHICS,
                source_pages=(page_index,),
                element_id=vector_element.element_id,
                locator=vector_element.locator,
                requires_visual_read=True,
            ))
    if images or unread_vector_objects:
        diagnostics.append(ProcessingDiagnostic(
            DiagnosticCode.PAGE_NEEDS_VISION,
            DocumentLocator(page=page_index),
            detail=(
                "visual-only page"
                if kind is PageKind.SCANNED
                else "page contains unread visual units"
            ),
        ))

    page_diagnostics = tuple(diagnostics[diagnostic_start:])
    page_elements = tuple(elements[element_start:])
    text_element_ids = tuple(
        element.element_id
        for element in page_elements
        if (element.text or "").strip()
    )
    fatal = any(
        diagnostic.code in {DiagnosticCode.CORRUPT_SOURCE, DiagnosticCode.LIMIT_REACHED}
        for diagnostic in page_diagnostics
    )
    if fatal:
        state = DocumentPageState.UNREADABLE
    elif text_element_ids and nontext_units:
        state = DocumentPageState.MIXED
    elif text_element_ids:
        state = DocumentPageState.TEXT
    elif nontext_units:
        state = DocumentPageState.VISUAL_ONLY
    else:
        # 只有当解析器分类与提取器不一致时才能到达此路径；将该分歧记录为致命类型化事实。
        diagnostic = ProcessingDiagnostic(
            DiagnosticCode.CORRUPT_SOURCE,
            DocumentLocator(page=page_index),
            detail="ClassifiedPageYieldedNoInventory",
        )
        diagnostics.append(diagnostic)
        page_diagnostics = (*page_diagnostics, diagnostic)
        state = DocumentPageState.UNREADABLE
    return DocumentPageRecord(
        page_number=page_index,
        state=state,
        text_element_ids=text_element_ids,
        nontext_units=tuple(nontext_units),
        diagnostics=page_diagnostics,
    )


def _emit_tables(
    page,
    page_index,
    source_key,
    elements,
    diagnostics,
    ordinal: int,
    *,
    native_table_quality: dict[str, bool] | None = None,
    element_limit: int | None = None,
) -> tuple[
    int,
    tuple[DocumentNonTextUnit, ...],
    tuple[tuple[float, float, float, float], ...],
    tuple[tuple[float, float, float, float], ...],
    tuple[tuple[float, float, float, float], ...],
]:
    units: list[DocumentNonTextUnit] = []
    table_bboxes: list[tuple[float, float, float, float]] = []
    table_grid_segments: set[tuple[float, float, float, float]] = set()
    table_cell_bboxes: set[tuple[float, float, float, float]] = set()
    try:
        finder = getattr(page, "find_tables", None)
        if callable(finder):
            found_tables = tuple(finder())
            table_entries = tuple(
                (
                    item.extract(),
                    _bbox_of({
                        "x0": item.bbox[0],
                        "top": item.bbox[1],
                        "x1": item.bbox[2],
                        "bottom": item.bbox[3],
                    }),
                )
                for item in found_tables
            )
            table_bboxes = [
                bbox for _table, bbox in table_entries if bbox is not None
            ]
            for item in found_tables:
                for cell in tuple(getattr(item, "cells", ()) or ()):
                    try:
                        x0, top, x1, bottom = (float(value) for value in cell)
                    except (TypeError, ValueError):
                        continue
                    cell_bbox = (x0, top, x1, bottom)
                    table_cell_bboxes.add(cell_bbox)
                    table_grid_segments.update({
                        (x0, top, x1, top),
                        (x0, bottom, x1, bottom),
                        (x0, top, x0, bottom),
                        (x1, top, x1, bottom),
                    })
        else:
            table_entries = tuple(
                (table, None) for table in page.extract_tables()
            )
    except Exception as exc:
        diagnostics.append(ProcessingDiagnostic(
            DiagnosticCode.CORRUPT_SOURCE,
            DocumentLocator(page=page_index),
            detail=f"TableExtraction:{type(exc).__name__}",
        ))
        return ordinal, (), (), (), ()
    for table, table_bbox in table_entries:
        text = _render_table(table)
        if not text.strip():
            continue
        locator = DocumentLocator(
            page=page_index,
            ordinal=ordinal,
            bbox=table_bbox,
        )
        if not _element_slot_available(
            elements,
            diagnostics,
            locator,
            element_limit=element_limit,
        ):
            break
        stored_text = text[:MAX_ELEMENT_CHARS]
        if len(text) > MAX_ELEMENT_CHARS:
            diagnostics.append(ProcessingDiagnostic(
                DiagnosticCode.LIMIT_REACHED,
                locator,
                detail="table element character limit reached",
            ))
        element = DocumentElement(
            element_id=make_element_id(source_key, locator, stored_text),
            kind=ElementKind.TABLE,
            text=stored_text,
            locator=locator,
            source_pages=(page_index,),
        )
        elements.append(element)
        if native_table_quality is not None:
            native_table_quality[element.element_id] = (
                _native_table_grid_is_complete_enough(table)
            )
        units.append(DocumentNonTextUnit(
            unit_id=_nontext_unit_id(
                source_key,
                DocumentNonTextKind.TABLE,
                page_index,
                ordinal,
            ),
            kind=DocumentNonTextKind.TABLE,
            source_pages=(page_index,),
            text_element_ids=(element.element_id,),
            element_id=element.element_id,
            locator=locator,
            requires_visual_read=False,
        ))
        ordinal += 1
    return (
        ordinal,
        tuple(units),
        tuple(table_bboxes),
        tuple(sorted(table_grid_segments)),
        tuple(sorted(table_cell_bboxes)),
    )


def _native_table_grid_is_complete_enough(
    table: object,
) -> bool:
    """判断原生表格文本是否可以关闭显式视觉 gap。

    仅非空渲染并不足够：提取可能只返回表头，却静默丢失正文。保留原始网格形状，并要求小型
    矩形表格、至少两行有内容、每列均被覆盖，以及冻结的最小有内容单元格比例。
    """

    if not isinstance(table, (list, tuple)):
        return False
    rows = tuple(row for row in table if isinstance(row, (list, tuple)))
    if len(rows) != len(table) or len(rows) < MIN_NATIVE_TABLE_ROWS_FOR_RECONCILIATION:
        return False
    column_count = len(rows[0]) if rows else 0
    if column_count < MIN_NATIVE_TABLE_COLUMNS_FOR_RECONCILIATION:
        return False
    if any(len(row) != column_count for row in rows):
        return False
    populated = tuple(
        tuple(bool(str(cell or "").strip()) for cell in row)
        for row in rows
    )
    if sum(any(row) for row in populated) < MIN_NATIVE_TABLE_ROWS_FOR_RECONCILIATION:
        return False
    if any(not any(row[column] for row in populated) for column in range(column_count)):
        return False
    populated_count = sum(sum(row) for row in populated)
    return (
        populated_count / (len(rows) * column_count)
        >= MIN_NATIVE_TABLE_CELL_COVERAGE_FOR_RECONCILIATION
    )


def _emit_text_lines(
    page,
    page_index,
    source_key,
    elements,
    diagnostics,
    ordinal: int,
    *,
    excluded_bboxes: tuple[tuple[float, float, float, float], ...] = (),
    element_limit: int | None = None,
) -> int:
    try:
        lines = page.extract_text_lines()
    except Exception as exc:
        diagnostics.append(ProcessingDiagnostic(
            DiagnosticCode.CORRUPT_SOURCE,
            DocumentLocator(page=page_index),
            detail=type(exc).__name__,
        ))
        return ordinal

    for line in lines:
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        if _bbox_inside_any(_bbox_of(line), excluded_bboxes):
            continue
        locator = DocumentLocator(
            page=page_index,
            ordinal=ordinal,
            bbox=_bbox_of(line),
        )
        if not _element_slot_available(
            elements,
            diagnostics,
            locator,
            element_limit=element_limit,
        ):
            break
        stored_text = text[:MAX_ELEMENT_CHARS]
        if len(text) > MAX_ELEMENT_CHARS:
            diagnostics.append(ProcessingDiagnostic(
                DiagnosticCode.LIMIT_REACHED,
                locator,
                detail="text element character limit reached",
            ))
        elements.append(DocumentElement(
            element_id=make_element_id(source_key, locator, stored_text),
            kind=ElementKind.PARAGRAPH,
            text=stored_text,
            locator=locator,
            source_pages=(page_index,),
        ))
        ordinal += 1
    return ordinal


def _emit_annotations(
    annotations,
    page,
    page_index: int,
    source_key: str,
    elements: list[DocumentElement],
    diagnostics: list[ProcessingDiagnostic],
    ordinal: int,
    *,
    element_limit: int | None = None,
) -> tuple[int, tuple[DocumentNonTextUnit, ...]]:
    """提取表单/评论文本，并保留不受支持的语义 annotation。"""

    units: list[DocumentNonTextUnit] = []
    for raw in annotations:
        if not isinstance(raw, Mapping):
            continue
        data = raw.get("data")
        annotation = data if isinstance(data, Mapping) else raw
        subtype = _pdf_name(_mapping_value(annotation, "Subtype"))
        locator = DocumentLocator(
            page=page_index,
            ordinal=ordinal,
            bbox=_bbox_of(dict(raw)) or _bbox_of(dict(annotation)),
            section_path=("pdf_annotation", subtype.lower()),
        )
        content = _annotation_text(annotation, subtype=subtype)

        if subtype == "Link" and not content:
            # 不带替代/评论文本的链接矩形属于导航，而非缺失的文档主张。
            continue

        if content:
            if not _element_slot_available(
                elements,
                diagnostics,
                locator,
                element_limit=element_limit,
            ):
                break
            stored_text = content[:MAX_ELEMENT_CHARS]
            if len(content) > MAX_ELEMENT_CHARS:
                diagnostics.append(ProcessingDiagnostic(
                    DiagnosticCode.LIMIT_REACHED,
                    locator,
                    detail="annotation text character limit reached",
                ))
            elements.append(DocumentElement(
                element_id=make_element_id(source_key, locator, stored_text),
                kind=ElementKind.PARAGRAPH,
                text=stored_text,
                locator=locator,
                source_pages=(page_index,),
            ))
            ordinal += 1
            continue

            # 存在语义 annotation，但未产生安全文本。保留一个已定位非文本单元，而不是宣称
            # 页面完整。
        visual = _image_element(
            page,
            page_index,
            source_key,
            ordinal,
            bbox=locator.bbox,
            section_path=locator.section_path,
        )
        if not _element_slot_available(
            elements,
            diagnostics,
            visual.locator,
            element_limit=element_limit,
        ):
            break
        elements.append(visual)
        unit = DocumentNonTextUnit(
            unit_id=_nontext_unit_id(
                source_key,
                DocumentNonTextKind.VECTOR_GRAPHICS,
                page_index,
                ordinal,
            ),
            kind=DocumentNonTextKind.VECTOR_GRAPHICS,
            source_pages=(page_index,),
            element_id=visual.element_id,
            locator=visual.locator,
            requires_visual_read=True,
        )
        units.append(unit)
        diagnostics.append(ProcessingDiagnostic(
            DiagnosticCode.PAGE_NEEDS_VISION,
            visual.locator,
            detail=f"PDF {subtype} annotation has no extractable text",
        ))
        ordinal += 1
    return ordinal, tuple(units)


def _annotation_text(
    annotation: Mapping[object, object],
    *,
    subtype: str,
) -> str | None:
    content = None
    if subtype == "Widget":
        field = _pdf_scalar_text(_mapping_value(annotation, "T"))
        value = _pdf_scalar_text(
            _mapping_value(annotation, "V")
            or _mapping_value(annotation, "DV")
        )
        if value:
            content = f"{field}: {value}" if field else value
    if content is None:
    # 高亮/遮盖/墨迹/文件 annotation 都可能携带人工评论。子类型白名单会静默丢失该内容，
    # 因此安全标量 Contents 字段适用于整个格式。
        content = _pdf_scalar_text(_mapping_value(annotation, "Contents"))
    return content


def _mapping_value(value: Mapping[object, object], key: str) -> object | None:
    return value.get(key) if key in value else value.get(f"/{key}")


def _pdf_name(value: object) -> str:
    name = getattr(value, "name", value)
    return str(name or "").lstrip("/")


def _pdf_scalar_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        text = str(value)
    else:
        name = getattr(value, "name", None)
        if not isinstance(name, str):
            return None
        text = name
    text = " ".join(text.split())
    return text or None


def _element_slot_available(
    elements: list[DocumentElement],
    diagnostics: list[ProcessingDiagnostic],
    locator: DocumentLocator,
    *,
    element_limit: int | None = None,
) -> bool:
    """不预留状态；生成内容将越过上限时只报告一次。"""

    active_limit = MAX_ELEMENTS if element_limit is None else element_limit
    if len(elements) < active_limit:
        return True
    if not any(
        diagnostic.code is DiagnosticCode.LIMIT_REACHED
        and diagnostic.detail == "element budget exhausted"
        and diagnostic.locator.page == locator.page
        for diagnostic in diagnostics
    ):
        diagnostics.append(ProcessingDiagnostic(
            DiagnosticCode.LIMIT_REACHED,
            locator,
            detail="element budget exhausted",
        ))
    return False


def _close_page_cache(page: object) -> None:
    """投影后释放 pdfplumber 的逐页对象/文本缓存。"""

    close = getattr(page, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception:
        # 外层 PDF 上下文仍持有源句柄。缓存清理失败不能抹掉其他方面如实的解析结果。
        return


def _image_element(
    page,
    page_index: int,
    source_key: str,
    ordinal: int,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    section_path: tuple[str, ...] = (),
) -> DocumentElement:
    locator = DocumentLocator(
        page=page_index,
        ordinal=ordinal,
        section_path=section_path,
        bbox=bbox or (0.0, 0.0, float(page.width or 0), float(page.height or 0)),
    )
    return DocumentElement(
        element_id=make_element_id(source_key, locator, None),
        kind=ElementKind.IMAGE,
        text=None,
        locator=locator,
        needs_vision=True,
        source_pages=(page_index,),
    )


def _nontext_unit_id(
    source_key: str,
    kind: DocumentNonTextKind,
    page_number: int,
    ordinal: int,
) -> str:
    digest = hashlib.sha256(
        f"{source_key}\x1f{kind.value}\x1f{page_number}\x1f{ordinal}".encode("utf-8")
    ).hexdigest()
    return f"ntu_{digest[:24]}"


def _bbox_of(line: dict) -> tuple[float, float, float, float] | None:
    try:
        return (
            float(line["x0"]), float(line["top"]),
            float(line["x1"]), float(line["bottom"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _union_bboxes(
    bboxes: tuple[tuple[float, float, float, float], ...],
) -> tuple[float, float, float, float] | None:
    if not bboxes:
        return None
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def _is_table_grid_scaffolding(
    item: Mapping[object, object],
    *,
    table_bboxes: tuple[tuple[float, float, float, float], ...] | list[
        tuple[float, float, float, float]
    ],
    table_grid_segments: tuple[tuple[float, float, float, float], ...],
    table_cell_bboxes: tuple[tuple[float, float, float, float], ...],
    tolerance: float = 0.5,
) -> bool:
    """只抑制已证明的表格边框，绝不抑制单元格内任意图形。"""

    bbox = _bbox_of(item)
    if bbox is None:
        return False
    object_type = str(_mapping_value(item, "object_type") or "").lower()
    if object_type == "line":
        x0, top, x1, bottom = bbox
        if abs(top - bottom) <= tolerance:
            intervals = tuple(
                (segment[0], segment[2])
                for segment in table_grid_segments
                if abs(segment[1] - segment[3]) <= tolerance
                and abs(segment[1] - top) <= tolerance
            )
            return _intervals_cover(x0, x1, intervals, tolerance=tolerance)
        if abs(x0 - x1) <= tolerance:
            intervals = tuple(
                (segment[1], segment[3])
                for segment in table_grid_segments
                if abs(segment[0] - segment[2]) <= tolerance
                and abs(segment[0] - x0) <= tolerance
            )
            return _intervals_cover(top, bottom, intervals, tolerance=tolerance)
        return False
    if object_type == "rect":
        if any(
            all(abs(left - right) <= tolerance for left, right in zip(bbox, expected))
            for expected in (*tuple(table_bboxes), *table_cell_bboxes)
        ):
            return True
        if bool(_mapping_value(item, "fill")) and not bool(
            _mapping_value(item, "stroke")
        ):
            rect_width = max(0.0, bbox[2] - bbox[0])
            rect_area = rect_width * max(0.0, bbox[3] - bbox[1])
            fill_top_edges = tuple(
                (segment[0], segment[2])
                for segment in table_grid_segments
                if abs(segment[1] - segment[3]) <= tolerance
                and abs(segment[1] - bbox[1]) <= tolerance
            )
            fill_bottom_edges = tuple(
                (segment[0], segment[2])
                for segment in table_grid_segments
                if abs(segment[1] - segment[3]) <= tolerance
                and abs(segment[1] - bbox[3]) <= tolerance
            )
            for table_bbox in table_bboxes:
                table_width = max(0.0, table_bbox[2] - table_bbox[0])
                shared_left = max(bbox[0], table_bbox[0])
                shared_right = min(bbox[2], table_bbox[2])
                if (
                    rect_area > 0
                    and table_width > 0
                    and shared_right > shared_left
                    and rect_width / table_width >= 0.9
                    and _bbox_intersection_area(bbox, table_bbox) / rect_area >= 0.95
                    and _intervals_cover(
                        shared_left,
                        shared_right,
                        fill_top_edges,
                        tolerance=tolerance,
                    )
                    and _intervals_cover(
                        shared_left,
                        shared_right,
                        fill_bottom_edges,
                        tolerance=tolerance,
                    )
                ):
        # PDF 生成器通常会绘制全宽行色带，并超出表格检测所用线条几个点。填充且无描边的
        # 色带属于布局，而非单元格图形。
                    return True
        # 某些生成器会把交替行背景绘制成横跨多个单元格的矩形。当四边都与已检测网格重合时，
        # 即使它们并非某个单元格的精确 bbox，也属于布局。
        x0, top, x1, bottom = bbox
        horizontal_top = tuple(
            (segment[0], segment[2])
            for segment in table_grid_segments
            if abs(segment[1] - segment[3]) <= tolerance
            and abs(segment[1] - top) <= tolerance
        )
        horizontal_bottom = tuple(
            (segment[0], segment[2])
            for segment in table_grid_segments
            if abs(segment[1] - segment[3]) <= tolerance
            and abs(segment[1] - bottom) <= tolerance
        )
        vertical_left = tuple(
            (segment[1], segment[3])
            for segment in table_grid_segments
            if abs(segment[0] - segment[2]) <= tolerance
            and abs(segment[0] - x0) <= tolerance
        )
        vertical_right = tuple(
            (segment[1], segment[3])
            for segment in table_grid_segments
            if abs(segment[0] - segment[2]) <= tolerance
            and abs(segment[0] - x1) <= tolerance
        )
        return (
            _intervals_cover(x0, x1, horizontal_top, tolerance=tolerance)
            and _intervals_cover(x0, x1, horizontal_bottom, tolerance=tolerance)
            and _intervals_cover(top, bottom, vertical_left, tolerance=tolerance)
            and _intervals_cover(top, bottom, vertical_right, tolerance=tolerance)
        )
    # 即使 bounding box 恰好位于已检测表格内，曲线和未知对象仍保留为视觉 gap。
    return False


def _intervals_cover(
    start: float,
    end: float,
    intervals: tuple[tuple[float, float], ...],
    *,
    tolerance: float,
) -> bool:
    lower, upper = sorted((start, end))
    cursor = lower
    for interval_start, interval_end in sorted(
        (tuple(sorted(interval)) for interval in intervals),
        key=lambda interval: interval[0],
    ):
        if interval_end < cursor - tolerance:
            continue
        if interval_start > cursor + tolerance:
            return False
        cursor = max(cursor, interval_end)
        if cursor >= upper - tolerance:
            return True
    return cursor >= upper - tolerance


def _bbox_inside_any(
    bbox: tuple[float, float, float, float] | None,
    containers: tuple[tuple[float, float, float, float], ...] | list[
        tuple[float, float, float, float]
    ],
    *,
    tolerance: float = 1.0,
) -> bool:
    if bbox is None:
        return False
    x0, top, x1, bottom = bbox
    return any(
        x0 >= outer_x0 - tolerance
        and top >= outer_top - tolerance
        and x1 <= outer_x1 + tolerance
        and bottom <= outer_bottom + tolerance
        for outer_x0, outer_top, outer_x1, outer_bottom in containers
    )


def _render_table(table: list[list[str | None]]) -> str:
    """将表格展平为文本，但不假装它是正文。"""

    rows = [
        " | ".join((cell or "").replace("\n", " ").strip() for cell in row)
        for row in table
        if any((cell or "").strip() for cell in row)
    ]
    return "\n".join(rows)
