"""与格式无关的文档元素契约。

本模块的核心在于：读取器输出中不保留任何用于辨别原始格式的痕迹。在它出现
以前，每个读取器都会自行发明位置显示字符串——``"p1"``、``"段1"``、
``"slide1"``、``"L1-L40"``——导致任何需要使用位置信息的消费者都必须解析
格式专用文本。这也使 PDF 方面的改进无法惠及其他格式：页码与边界框没有中立
通道可供传递。

这里的定位器是具有可选字段的类型化对象。每种格式填充自己拥有的信息，其余
字段保持 ``None``；人类可读字符串只在唯一一处完成渲染。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum


class ElementKind(StrEnum):
    """一段文档内容的*本质*，独立于其编码方式。"""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    IMAGE = "image"
    CAPTION = "caption"
    PAGE_BREAK = "page_break"


class DocumentTextEvidenceOrigin(StrEnum):
    """非原生文档文本的可信机械来源。"""

    OCR = "ocr"


class DiagnosticCode(StrEnum):
    """reader 对无法完整处理的源可报告事项的封闭集合。部分成功必须始终携带其中一项，
    而不能静默返回较少元素。"""

    PASSWORD_REQUIRED = "password_required"
    PERMISSION_DENIED = "permission_denied"
    CORRUPT_SOURCE = "corrupt_source"
    EMPTY_SOURCE = "empty_source"
    PAGE_NEEDS_VISION = "page_needs_vision"
    PAGE_EMPTY = "page_empty"
    OCR_ENGINE_FAILED = "ocr_engine_failed"
    OCR_TEXT_UNCERTAIN = "ocr_text_uncertain"
    PARSER_PARTIAL = "parser_partial"
    LIMIT_REACHED = "limit_reached"
    FALLBACK_READER_USED = "fallback_reader_used"


class ProcessingAdmissionStatus(StrEnum):
    """reader 结果是否可以跨越持久文档边界。

    只有 ``complete`` 和 ``partial`` 是可持久化文档状态。``rejected`` 保持为类型化决定，
    避免调用方把干净的空结果或致命诊断与可准入的部分覆盖混淆。
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
    REJECTED = "rejected"


class DocumentPageState(StrEnum):
    """在一个物理源页面上机械观测到的内容。"""

    TEXT = "text"
    MIXED = "mixed"
    VISUAL_ONLY = "visual_only"
    NO_EXTRACTABLE_CONTENT = "no_extractable_content"
    UNREADABLE = "unreadable"


class DocumentPageInventoryStatus(StrEnum):
    """detector 是否证明了它所报告的物理页面全集。"""

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class DocumentNonTextKind(StrEnum):
    """页面覆盖核算所需的封闭非文本分类。"""

    FIGURE = "figure"
    VECTOR_GRAPHICS = "vector_graphics"
    FORMULA = "formula"
    TABLE = "table"
    PAGE_VISUAL = "page_visual"


_PARTIAL_DIAGNOSTIC_CODES = frozenset({
    DiagnosticCode.PAGE_NEEDS_VISION,
    DiagnosticCode.PAGE_EMPTY,
    DiagnosticCode.OCR_TEXT_UNCERTAIN,
    DiagnosticCode.PARSER_PARTIAL,
    DiagnosticCode.FALLBACK_READER_USED,
})

_FATAL_DIAGNOSTIC_CODES = frozenset({
    DiagnosticCode.PASSWORD_REQUIRED,
    DiagnosticCode.PERMISSION_DENIED,
    DiagnosticCode.CORRUPT_SOURCE,
    DiagnosticCode.EMPTY_SOURCE,
    DiagnosticCode.OCR_ENGINE_FAILED,
    DiagnosticCode.LIMIT_REACHED,
})


@dataclass(frozen=True)
class DocumentLocator:
    """元素所在位置，以格式所支持的术语表示。

    所有字段都被刻意设为可选：纯文本文件有字符范围但没有页面；PDF 有页面和 box，但没有
    section path；DOCX 有 heading path，但没有几何信息。消费者读取能理解的字段并忽略
    其余字段——这一特性使某种格式可以提高精度，而无须改变其他格式或下游。
    """

    page: int | None = None
    ordinal: int | None = None
    section_path: tuple[str, ...] = ()
    bbox: tuple[float, float, float, float] | None = None
    char_range: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.page is not None and self.page < 1:
            raise ValueError("page numbers are 1-based")
        if self.ordinal is not None and self.ordinal < 0:
            raise ValueError("ordinal must not be negative")
        if self.bbox is not None:
            x0, top, x1, bottom = self.bbox
            if x1 < x0 or bottom < top:
                raise ValueError("bbox must be ordered (x0, top, x1, bottom)")
        if self.char_range is not None:
            start, end = self.char_range
            if start < 0 or end < start:
                raise ValueError("char_range must be an ordered non-negative span")

    @property
    def is_empty(self) -> bool:
        return (
            self.page is None
            and self.ordinal is None
            and not self.section_path
            and self.bbox is None
            and self.char_range is None
        )

    def describe(self) -> str:
        """渲染简短且便于阅读的位置。

        这是唯一生成显示字符串的位置。reader 不得自行构建，否则格式专属词汇会重新泄漏到
        引用和 prompt 中。
        """

        parts: list[str] = []
        if self.page is not None:
            parts.append(f"p{self.page}")
        if self.section_path:
            parts.append(" › ".join(self.section_path))
        if self.char_range is not None:
            parts.append(f"c{self.char_range[0]}-{self.char_range[1]}")
        if not parts and self.ordinal is not None:
            parts.append(f"#{self.ordinal}")
        elif parts and self.ordinal is not None and self.page is not None:
            parts[0] = f"{parts[0]}#{self.ordinal}"
        return " ".join(parts) if parts else "?"


@dataclass(frozen=True)
class ProcessorFingerprint:
    """产生结果的 reader；当 reader 本身变化时，可据此使派生数据失效。"""

    reader: str
    version: str

    def __str__(self) -> str:
        return f"{self.reader}@{self.version}"


@dataclass(frozen=True)
class ProcessingDiagnostic:
    """有关不完整或降级处理的一项结构化事实。"""

    code: DiagnosticCode
    locator: DocumentLocator = field(default_factory=DocumentLocator)
    detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "at": self.locator.describe(),
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class DocumentNonTextUnit:
    """在一个或多个精确物理页面上观测到的一个非正文单元。

    ``requires_visual_read`` 被刻意设为独立于是否已提取文本。图像可能带有 OCR 文本，但其
    视觉主张仍未被读取；而拥有忠实文本表示的表格或公式可被机械核算，无须假装它不存在。
    """

    unit_id: str
    kind: DocumentNonTextKind
    source_pages: tuple[int, ...]
    text_element_ids: tuple[str, ...] = ()
    element_id: str | None = None
    locator: DocumentLocator = field(default_factory=DocumentLocator)
    requires_visual_read: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.unit_id, str) or not self.unit_id.strip():
            raise ValueError("nontext unit_id must not be empty")
        _validate_exact_pages(self.source_pages, field_name="nontext source_pages")
        _validate_unique_ids(self.text_element_ids, field_name="text_element_ids")
        if self.element_id is not None and not self.element_id.strip():
            raise ValueError("nontext element_id must not be empty")
        if self.locator.page is not None and self.locator.page not in self.source_pages:
            raise ValueError("nontext locator page must belong to source_pages")

    def to_dict(self) -> dict[str, object]:
        return {
            "unit_id": self.unit_id,
            "kind": self.kind.value,
            "source_pages": list(self.source_pages),
            "text_element_ids": list(self.text_element_ids),
            "element_id": self.element_id,
            "locator": _locator_to_dict(self.locator),
            "requires_visual_read": self.requires_visual_read,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DocumentNonTextUnit:
        try:
            source_pages = _integer_tuple(value["source_pages"])
            text_element_ids = _string_tuple(value.get("text_element_ids", ()))
            raw_locator = value.get("locator", {})
            if not isinstance(raw_locator, Mapping):
                raise ValueError("nontext locator must be an object")
            element_id = value.get("element_id")
            if element_id is not None and not isinstance(element_id, str):
                raise ValueError("nontext element_id must be a string or null")
            visual = value.get("requires_visual_read", False)
            if not isinstance(visual, bool):
                raise ValueError("requires_visual_read must be boolean")
            return cls(
                unit_id=_required_string(value.get("unit_id"), "unit_id"),
                kind=DocumentNonTextKind(value.get("kind")),
                source_pages=source_pages,
                text_element_ids=text_element_ids,
                element_id=element_id,
                locator=_locator_from_dict(raw_locator),
                requires_visual_read=visual,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid nontext unit payload") from exc


@dataclass(frozen=True, slots=True)
class DocumentPageRecord:
    """一个从 1 开始编号的物理页面的完整观测清单。"""

    page_number: int
    state: DocumentPageState
    text_element_ids: tuple[str, ...] = ()
    nontext_units: tuple[DocumentNonTextUnit, ...] = ()
    diagnostics: tuple[ProcessingDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.page_number, bool) or self.page_number < 1:
            raise ValueError("page_number must be a positive integer")
        _validate_unique_ids(self.text_element_ids, field_name="text_element_ids")
        unit_ids = tuple(unit.unit_id for unit in self.nontext_units)
        _validate_unique_ids(unit_ids, field_name="nontext unit ids")
        if any(self.page_number not in unit.source_pages for unit in self.nontext_units):
            raise ValueError("page record may only contain units sourced from that page")
        if any(
            diagnostic.locator.page != self.page_number
            for diagnostic in self.diagnostics
        ):
            raise ValueError("page diagnostics must carry the exact physical page")
        if self.state is DocumentPageState.TEXT and (
            not self.text_element_ids or self.nontext_units
        ):
            raise ValueError("text page requires text and no nontext units")
        if self.state is DocumentPageState.MIXED and (
            not self.text_element_ids or not self.nontext_units
        ):
            raise ValueError("mixed page requires text and nontext units")
        if self.state is DocumentPageState.VISUAL_ONLY and (
            self.text_element_ids or not self.nontext_units
        ):
            raise ValueError("visual_only page requires nontext units and no text")
        if self.state is DocumentPageState.NO_EXTRACTABLE_CONTENT and (
            self.text_element_ids or self.nontext_units
        ):
            raise ValueError("no_extractable_content page must have an empty inventory")
        if self.state is DocumentPageState.UNREADABLE and not self.diagnostics:
            raise ValueError("unreadable page requires a typed diagnostic")

    def to_dict(self) -> dict[str, object]:
        return {
            "page_number": self.page_number,
            "state": self.state.value,
            "text_element_ids": list(self.text_element_ids),
            "nontext_units": [unit.to_dict() for unit in self.nontext_units],
            "diagnostics": [_diagnostic_to_dict(item) for item in self.diagnostics],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DocumentPageRecord:
        try:
            raw_units = value.get("nontext_units", ())
            raw_diagnostics = value.get("diagnostics", ())
            if not _is_object_sequence(raw_units) or not _is_object_sequence(
                raw_diagnostics
            ):
                raise ValueError("page units and diagnostics must be object sequences")
            page_number = _required_integer(value.get("page_number"), "page_number")
            return cls(
                page_number=page_number,
                state=DocumentPageState(value.get("state")),
                text_element_ids=_string_tuple(value.get("text_element_ids", ())),
                nontext_units=tuple(
                    DocumentNonTextUnit.from_dict(item) for item in raw_units
                ),
                diagnostics=tuple(
                    _diagnostic_from_dict(item, expected_page=page_number)
                    for item in raw_diagnostics
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid page record payload") from exc


PAGE_MANIFEST_CONTRACT_VERSION = 1


@dataclass(frozen=True, slots=True)
class DocumentPageManifest:
    """由一个 detector 生成、可自认证的物理页面清单。"""

    physical_page_count: int
    inventory_status: DocumentPageInventoryStatus
    detector_fingerprint: str
    detector_capabilities: tuple[str, ...]
    pages: tuple[DocumentPageRecord, ...]
    manifest_sha256: str = ""
    contract_version: int = PAGE_MANIFEST_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PAGE_MANIFEST_CONTRACT_VERSION:
            raise ValueError("unsupported page manifest contract_version")
        if not isinstance(self.physical_page_count, int) or isinstance(
            self.physical_page_count, bool
        ):
            raise ValueError("physical_page_count must be an integer")
        if self.physical_page_count < 0:
            raise ValueError("physical_page_count must not be negative")
        if not self.detector_fingerprint.strip():
            raise ValueError("detector_fingerprint must not be empty")
        _validate_unique_ids(
            self.detector_capabilities,
            field_name="detector_capabilities",
            require_sorted=True,
        )
        expected_numbers = tuple(range(1, self.physical_page_count + 1))
        actual_numbers = tuple(page.page_number for page in self.pages)
        if actual_numbers != expected_numbers:
            raise ValueError("page manifest must contain continuous physical pages 1..N")
        units: dict[str, DocumentNonTextUnit] = {}
        observed_unit_pages: dict[str, set[int]] = {}
        for page in self.pages:
            for unit in page.nontext_units:
                existing = units.setdefault(unit.unit_id, unit)
                if existing != unit:
                    raise ValueError("nontext unit identity must be stable across pages")
                observed_unit_pages.setdefault(unit.unit_id, set()).add(page.page_number)
        if any(
            tuple(sorted(observed_unit_pages[unit_id])) != unit.source_pages
            for unit_id, unit in units.items()
        ):
            raise ValueError("nontext unit must be inventoried on every exact source page")
        if self.inventory_status is DocumentPageInventoryStatus.UNAVAILABLE:
            if self.physical_page_count != 0 or self.pages:
                raise ValueError("unavailable page inventory cannot claim physical pages")
        elif self.physical_page_count < 1:
            raise ValueError("available page inventory requires at least one page")
        expected_hash = hashlib.sha256(
            _canonical_json(self._unsigned_dict()).encode("utf-8")
        ).hexdigest()
        if self.manifest_sha256:
            if self.manifest_sha256 != expected_hash:
                raise ValueError("page manifest_sha256 does not authenticate payload")
        else:
            object.__setattr__(self, "manifest_sha256", expected_hash)

    def _unsigned_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "physical_page_count": self.physical_page_count,
            "inventory_status": self.inventory_status.value,
            "detector_fingerprint": self.detector_fingerprint,
            "detector_capabilities": list(self.detector_capabilities),
            "pages": [page.to_dict() for page in self.pages],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._unsigned_dict(), "manifest_sha256": self.manifest_sha256}

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> DocumentPageManifest:
        try:
            raw_pages = value.get("pages", ())
            if not _is_object_sequence(raw_pages):
                raise ValueError("manifest pages must be an object sequence")
            return cls(
                contract_version=_required_integer(
                    value.get("contract_version"), "contract_version"
                ),
                physical_page_count=_required_integer(
                    value.get("physical_page_count"), "physical_page_count"
                ),
                inventory_status=DocumentPageInventoryStatus(
                    value.get("inventory_status")
                ),
                detector_fingerprint=_required_string(
                    value.get("detector_fingerprint"), "detector_fingerprint"
                ),
                detector_capabilities=_string_tuple(
                    value.get("detector_capabilities", ())
                ),
                pages=tuple(DocumentPageRecord.from_dict(item) for item in raw_pages),
                manifest_sha256=_required_string(
                    value.get("manifest_sha256"), "manifest_sha256"
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid page manifest payload") from exc

    @classmethod
    def from_json(cls, value: str) -> DocumentPageManifest:
        try:
            payload = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("invalid page manifest JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("page manifest JSON must contain an object")
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class DocumentTextEvidence:
    """一个机械推断文本元素的 prompt 安全 provenance。

    源单元引用将 OCR 文本关联回页面 manifest 中精确的非文本单元。零置信度文本被刻意设计为
    无法表示：reader 必须发出类型化不确定性 gap，而不是把它伪装成普通正文。
    """

    origin: DocumentTextEvidenceOrigin
    confidence: float
    source_unit_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.origin, DocumentTextEvidenceOrigin):
            raise ValueError("unsupported text evidence origin")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0.0 < float(self.confidence) <= 1.0
        ):
            raise ValueError("text evidence confidence must be within (0, 1]")
        object.__setattr__(self, "confidence", float(self.confidence))
        if (
            not isinstance(self.source_unit_id, str)
            or not 1 <= len(self.source_unit_id) <= 160
            or any(
                not (character.isascii() and (character.isalnum() or character in "_.:-"))
                for character in self.source_unit_id
            )
        ):
            raise ValueError("text evidence source_unit_id must be a prompt-safe identifier")

    @property
    def uncertainty(self) -> float:
        return 1.0 - self.confidence

    def to_prompt_dict(self) -> dict[str, object]:
        return {
            "origin": self.origin.value,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "source_unit_id": self.source_unit_id,
        }


@dataclass(frozen=True)
class DocumentElement:
    """文档中一段可寻址内容。"""

    element_id: str
    kind: ElementKind
    text: str | None
    locator: DocumentLocator
    needs_vision: bool = False
    source_pages: tuple[int, ...] = ()
    text_evidence: DocumentTextEvidence | None = None

    def __post_init__(self) -> None:
        if self.kind is ElementKind.IMAGE:
            if self.text is not None and not self.text.strip():
                raise ValueError("an image element must carry a caption or nothing")
        elif self.kind is not ElementKind.PAGE_BREAK and not (self.text or "").strip():
            # 无文本元素进入 prompt 后与缺失元素无法区分，这正是扫描页变成虚构内容的原因。
            # 应改为发出诊断。
            raise ValueError(f"{self.kind.value} element requires text")
        if self.source_pages:
            _validate_exact_pages(self.source_pages, field_name="element source_pages")
            if self.locator.page is None or self.locator.page not in self.source_pages:
                raise ValueError("element locator page must belong to source_pages")
        elif self.locator.page is not None:
            object.__setattr__(self, "source_pages", (self.locator.page,))
        if self.text_evidence is not None:
            if not isinstance(self.text_evidence, DocumentTextEvidence):
                raise ValueError("text_evidence must be a DocumentTextEvidence")
            if not (self.text or "").strip():
                raise ValueError("text evidence requires an emitted text element")


def make_element_id(source_key: str, locator: DocumentLocator, text: str | None) -> str:
    """根据位置和内容派生稳定 id。

    刻意不使用计数器：重新处理未变化的源必须产生相同 id，使派生数据（摘要、embedding）
    可被复用；文档其他位置的编辑也不能让其后所有内容重新编号。
    """

    digest = hashlib.sha256()
    digest.update(source_key.encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(repr((
        locator.page, locator.ordinal, locator.section_path,
        locator.bbox, locator.char_range,
    )).encode("utf-8"))
    digest.update(b"\x1f")
    digest.update((text or "").encode("utf-8"))
    return f"el_{digest.hexdigest()[:24]}"


@dataclass(frozen=True)
class ProcessingResult:
    """一个 reader 从一个源产生的全部结果，包括它无法完成的事项。"""

    elements: tuple[DocumentElement, ...]
    processor: ProcessorFingerprint
    diagnostics: tuple[ProcessingDiagnostic, ...] = ()
    page_manifest: DocumentPageManifest | None = None

    def __post_init__(self) -> None:
        element_ids = tuple(element.element_id for element in self.elements)
        _validate_unique_ids(element_ids, field_name="document element ids")
        if self.page_manifest is None:
            return
        _validate_manifest_elements(self.page_manifest, self.elements)
        _validate_manifest_diagnostics(
            self.page_manifest,
            self.diagnostics,
        )
        if (
            self.page_manifest.inventory_status
            is DocumentPageInventoryStatus.PARTIAL
            and not self.diagnostics
        ):
            raise ValueError("partial page inventory requires a typed diagnostic")

    @property
    def needs_vision(self) -> bool:
        return any(element.needs_vision for element in self.elements) or any(
            diagnostic.code is DiagnosticCode.PAGE_NEEDS_VISION
            for diagnostic in self.diagnostics
        )

    @property
    def is_complete(self) -> bool:
        """没有内容被跳过、降级或保留为不可读时为 True。"""

        return not self.diagnostics

    @property
    def admission_status(self) -> ProcessingAdmissionStatus:
        """为持久准入对结果分类，并保守失败。

        完整性与准入被刻意用于回答不同问题：旧 ``is_complete`` 标志只报告是否发出诊断，
        而准入还要求存在可用文本，并区分明确容忍的覆盖 gap 与致命处理失败。
        """

        if not self.text_elements():
            return ProcessingAdmissionStatus.REJECTED

        diagnostic_codes = {diagnostic.code for diagnostic in self.diagnostics}
        if (
            any(element.needs_vision for element in self.elements)
            and DiagnosticCode.PAGE_NEEDS_VISION not in diagnostic_codes
        ):
            return ProcessingAdmissionStatus.REJECTED
        if diagnostic_codes & _FATAL_DIAGNOSTIC_CODES:
            return ProcessingAdmissionStatus.REJECTED
        if not diagnostic_codes:
            return ProcessingAdmissionStatus.COMPLETE
        if diagnostic_codes <= _PARTIAL_DIAGNOSTIC_CODES:
            return ProcessingAdmissionStatus.PARTIAL
        return ProcessingAdmissionStatus.REJECTED

    def text_elements(self) -> tuple[DocumentElement, ...]:
        return tuple(element for element in self.elements if (element.text or "").strip())

    def legacy_elements(self) -> list[dict[str, object]]:
        """投影为旧 ``{"content", "loc"}`` 形状。

        保留此投影，使两个现有消费者在迁移期间继续工作。字符串来自
        ``DocumentLocator.describe`` 而非 reader，因此即使兼容路径也不再携带格式专属词汇。
        """

        projected: list[dict[str, object]] = []
        for element in self.text_elements():
            item: dict[str, object] = {
                "content": element.text or "",
                "loc": element.locator.describe(),
            }
            if element.text_evidence is not None:
                item["text_evidence"] = element.text_evidence.to_prompt_dict()
            projected.append(item)
        return projected


def _validate_manifest_elements(
    manifest: DocumentPageManifest,
    elements: tuple[DocumentElement, ...],
) -> None:
    expected_by_page: dict[int, set[str]] = {
        page.page_number: set() for page in manifest.pages
    }
    elements_by_id = {element.element_id: element for element in elements}
    known_ids = set(elements_by_id)
    visual_unit_counts: dict[str, int] = {}
    text_evidence_unit_counts: dict[str, int] = {}
    validated_unit_ids: set[str] = set()
    for element in elements:
        if not (element.text or "").strip():
            continue
        if not element.source_pages:
            raise ValueError("page-manifest text element requires exact source_pages")
        for page_number in element.source_pages:
            if page_number not in expected_by_page:
                raise ValueError("text element source_pages exceed physical page inventory")
            expected_by_page[page_number].add(element.element_id)
    for page in manifest.pages:
        if set(page.text_element_ids) != expected_by_page[page.page_number]:
            raise ValueError("page manifest text element inventory is not exact")
        for unit in page.nontext_units:
            if unit.unit_id in validated_unit_ids:
                continue
            validated_unit_ids.add(unit.unit_id)
            linked_ids = set(unit.text_element_ids)
            if unit.element_id is not None:
                linked_ids.add(unit.element_id)
            if not linked_ids <= known_ids:
                raise ValueError("page manifest nontext unit references unknown elements")
            if any(
                not (elements_by_id[element_id].text or "").strip()
                for element_id in unit.text_element_ids
            ):
                raise ValueError("nontext text_element_ids must reference emitted text")
            for element_id in unit.text_element_ids:
                text_element = elements_by_id[element_id]
                text_evidence = text_element.text_evidence
                if (
                    text_evidence is None
                    or text_evidence.source_unit_id != unit.unit_id
                ):
                    continue
                if not set(text_element.source_pages) <= set(unit.source_pages):
                    raise ValueError("text evidence source_pages exceed its source unit")
                text_evidence_unit_counts[element_id] = (
                    text_evidence_unit_counts.get(element_id, 0) + 1
                )
            if unit.element_id is not None:
                evidence = elements_by_id[unit.element_id]
                if evidence.source_pages != unit.source_pages:
                    raise ValueError("nontext evidence source_pages are not exact")
                if evidence.needs_vision and evidence.locator != unit.locator:
                    raise ValueError("nontext evidence locator is not exact")
                if evidence.needs_vision:
                    if not unit.requires_visual_read:
                        raise ValueError("visual evidence requires a visual-read unit")
                    visual_unit_counts[evidence.element_id] = (
                        visual_unit_counts.get(evidence.element_id, 0) + 1
                    )
    if any(
        visual_unit_counts.get(element.element_id, 0) != 1
        for element in elements
        if element.needs_vision
    ):
        raise ValueError("page manifest visual element inventory is not exact")
    if any(
        text_evidence_unit_counts.get(element.element_id, 0) != 1
        for element in elements
        if element.text_evidence is not None
    ):
        raise ValueError("text evidence source unit linkage is not exact")


def _validate_manifest_diagnostics(
    manifest: DocumentPageManifest,
    diagnostics: tuple[ProcessingDiagnostic, ...],
) -> None:
    expected_by_page = {
        page.page_number: tuple(page.diagnostics) for page in manifest.pages
    }
    actual_by_page: dict[int, list[ProcessingDiagnostic]] = {
        page.page_number: [] for page in manifest.pages
    }
    for diagnostic in diagnostics:
        if diagnostic.locator.page is None:
            continue
        if diagnostic.locator.page not in actual_by_page:
            raise ValueError("page diagnostic exceeds physical page inventory")
        actual_by_page[diagnostic.locator.page].append(diagnostic)
    for page_number, expected in expected_by_page.items():
        if tuple(actual_by_page[page_number]) != expected:
            raise ValueError("page manifest diagnostics inventory is not exact")
        requires_visual_read = any(
            unit.requires_visual_read
            for unit in manifest.pages[page_number - 1].nontext_units
        )
        has_visual_gap = any(
            diagnostic.code is DiagnosticCode.PAGE_NEEDS_VISION
            for diagnostic in expected
        )
        if requires_visual_read != has_visual_gap:
            raise ValueError(
                "page visual-read inventory and PAGE_NEEDS_VISION must agree"
            )


def _validate_exact_pages(value: tuple[int, ...], *, field_name: str) -> None:
    if not value or any(isinstance(page, bool) or not isinstance(page, int) or page < 1 for page in value):
        raise ValueError(f"{field_name} must contain positive integers")
    if value != tuple(sorted(set(value))):
        raise ValueError(f"{field_name} must be a sorted exact set")


def _validate_unique_ids(
    value: tuple[str, ...],
    *,
    field_name: str,
    require_sorted: bool = False,
) -> None:
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{field_name} must not contain duplicates")
    if require_sorted and value != tuple(sorted(value)):
        raise ValueError(f"{field_name} must be sorted")


def _locator_to_dict(locator: DocumentLocator) -> dict[str, object]:
    return {
        "page": locator.page,
        "ordinal": locator.ordinal,
        "section_path": list(locator.section_path),
        "bbox": list(locator.bbox) if locator.bbox is not None else None,
        "char_range": list(locator.char_range) if locator.char_range is not None else None,
    }


def _locator_from_dict(value: Mapping[str, object]) -> DocumentLocator:
    page = value.get("page")
    ordinal = value.get("ordinal")
    bbox = value.get("bbox")
    char_range = value.get("char_range")
    return DocumentLocator(
        page=None if page is None else _required_integer(page, "page"),
        ordinal=None if ordinal is None else _required_integer(ordinal, "ordinal"),
        section_path=_string_tuple(value.get("section_path", ())),
        bbox=None if bbox is None else _float_quad(bbox, "bbox"),
        char_range=None if char_range is None else _integer_pair(char_range, "char_range"),
    )


def _diagnostic_to_dict(value: ProcessingDiagnostic) -> dict[str, object]:
    return {
        "code": value.code.value,
        "locator": _locator_to_dict(value.locator),
        "detail": value.detail,
    }


def _diagnostic_from_dict(
    value: Mapping[str, object],
    *,
    expected_page: int,
) -> ProcessingDiagnostic:
    raw_locator = value.get("locator")
    if not isinstance(raw_locator, Mapping):
        raise ValueError("diagnostic locator must be an object")
    locator = _locator_from_dict(raw_locator)
    if locator.page != expected_page:
        raise ValueError("diagnostic page does not match page record")
    detail = value.get("detail")
    if detail is not None and not isinstance(detail, str):
        raise ValueError("diagnostic detail must be a string or null")
    return ProcessingDiagnostic(
        code=DiagnosticCode(value.get("code")),
        locator=locator,
        detail=detail,
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _required_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _integer_tuple(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("expected an integer sequence")
    return tuple(_required_integer(item, "sequence item") for item in value)


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("expected a string sequence")
    if not all(isinstance(item, str) for item in value):
        raise ValueError("expected a string sequence")
    return tuple(value)


def _float_quad(value: object, field_name: str) -> tuple[float, float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 4:
        raise ValueError(f"{field_name} must contain four numbers")
    try:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain four numbers") from exc


def _integer_pair(value: object, field_name: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{field_name} must contain two integers")
    return (
        _required_integer(value[0], field_name),
        _required_integer(value[1], field_name),
    )


def _is_object_sequence(value: object) -> bool:
    return (
        not isinstance(value, (str, bytes))
        and isinstance(value, Sequence)
        and all(isinstance(item, Mapping) for item in value)
    )
