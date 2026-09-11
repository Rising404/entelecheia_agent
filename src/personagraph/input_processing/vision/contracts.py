"""本地 OCR 证据与可选视觉语义的不可变契约。

OCR 与视觉理解被刻意设计为不同结果类型。OCR 可以忠实转录像素，却不理解承载文本的图像；
因此 OCR 成功绝不能用于宣称视觉语义完整。
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 仅供类型检查：运行时导入会与 imaging.payload 形成环
    from ..documents.contracts import DocumentLocator
    from .imaging.payload import VisionPayload

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class OcrStatus(StrEnum):
    """一次显式选择 OCR 后端调用的结果。"""

    SUCCESS = "success"
    BLANK = "blank"
    FAILED = "failed"


class OcrFailureCode(StrEnum):
    """稳定失败分类；异常消息绝不会成为 API 状态。"""

    BACKEND_UNAVAILABLE = "backend_unavailable"
    BACKEND_FAILED = "backend_failed"
    INVALID_OUTPUT = "invalid_output"


@dataclass(frozen=True, slots=True)
class PixelSize:
    width: int
    height: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or self.width < 1
            or self.height < 1
        ):
            raise ValueError("pixel dimensions must be positive integers")

    @property
    def pixel_count(self) -> int:
        return self.width * self.height


BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class OcrLine:
    """一行逐字 OCR 文本及按左上原点排序的 box。

    ``bbox_norm`` 位于包含端点的 0..1 坐标空间。``bbox_px`` 是定向图像像素中的相同区域。
    二者都使用 ``(left, top, right, bottom)``，而非后端专属 x/y/width/height。
    """

    text: str
    confidence: float
    bbox_norm: BBox
    bbox_px: BBox

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("OCR line text must not be empty")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("OCR line confidence must be within 0..1")
        object.__setattr__(self, "confidence", float(self.confidence))
        _validate_bbox(self.bbox_norm, field_name="bbox_norm", normalized=True)
        _validate_bbox(self.bbox_px, field_name="bbox_px", normalized=False)


DEFAULT_OCR_LANGUAGE_HINTS = ("zh-Hans", "en-US")


@dataclass(frozen=True, slots=True)
class OcrRequest:
    """发送给本地 OCR 的精确像素身份与几何信息。"""

    source_unit_id: str
    source_sha256: str
    image_sha256: str
    page: int
    pixel_size: PixelSize
    dpi: tuple[float, float] | None
    language_hints: tuple[str, ...] = DEFAULT_OCR_LANGUAGE_HINTS

    def __post_init__(self) -> None:
        _require_text(self.source_unit_id, "source_unit_id")
        _require_sha256(self.source_sha256, "source_sha256")
        _require_sha256(self.image_sha256, "image_sha256")
        if isinstance(self.page, bool) or not isinstance(self.page, int) or self.page < 1:
            raise ValueError("OCR page must be a positive integer")
        if not isinstance(self.pixel_size, PixelSize):
            raise ValueError("pixel_size must be a PixelSize")
        _validate_dpi(self.dpi)
        _validate_nonempty_unique_texts(self.language_hints, "language_hints")


@dataclass(frozen=True, slots=True)
class OcrBackendResult:
    """附加源元数据和计时之前，仅属于后端的结果。"""

    status: OcrStatus
    lines: tuple[OcrLine, ...] = ()
    warnings: tuple[str, ...] = ()
    failure_code: OcrFailureCode | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, OcrStatus):
            raise ValueError("unsupported OCR status")
        if not isinstance(self.lines, tuple) or any(
            not isinstance(line, OcrLine) for line in self.lines
        ):
            raise ValueError("lines must contain OcrLine values")
        _validate_ocr_outcome(self.status, self.lines, self.failure_code)
        _validate_unique_texts(self.warnings, "warnings")

    @classmethod
    def success(
        cls,
        lines: tuple[OcrLine, ...],
        *,
        warnings: tuple[str, ...] = (),
    ) -> OcrBackendResult:
        return cls(OcrStatus.SUCCESS, lines=lines, warnings=warnings)

    @classmethod
    def blank(cls, *, warnings: tuple[str, ...] = ()) -> OcrBackendResult:
        return cls(OcrStatus.BLANK, warnings=warnings)

    @classmethod
    def failed(
        cls,
        failure_code: OcrFailureCode,
        *,
        warnings: tuple[str, ...] = (),
    ) -> OcrBackendResult:
        return cls(
            OcrStatus.FAILED,
            failure_code=failure_code,
            warnings=warnings,
        )


@dataclass(frozen=True, slots=True)
class OcrResult:
    """一次 OCR 尝试的完整不可变证据记录。"""

    status: OcrStatus
    engine_fingerprint: str
    source_unit_id: str
    source_sha256: str
    image_sha256: str
    page: int
    pixel_size: PixelSize
    dpi: tuple[float, float] | None
    language_hints: tuple[str, ...]
    lines: tuple[OcrLine, ...] = ()
    elapsed_ms: int = 0
    warnings: tuple[str, ...] = ()
    failure_code: OcrFailureCode | None = None

    def __post_init__(self) -> None:
        _require_text(self.engine_fingerprint, "engine_fingerprint")
        OcrRequest(
            source_unit_id=self.source_unit_id,
            source_sha256=self.source_sha256,
            image_sha256=self.image_sha256,
            page=self.page,
            pixel_size=self.pixel_size,
            dpi=self.dpi,
            language_hints=self.language_hints,
        )
        if not isinstance(self.status, OcrStatus):
            raise ValueError("unsupported OCR status")
        if not isinstance(self.lines, tuple) or any(
            not isinstance(line, OcrLine) for line in self.lines
        ):
            raise ValueError("lines must contain OcrLine values")
        _validate_ocr_outcome(self.status, self.lines, self.failure_code)
        if (
            isinstance(self.elapsed_ms, bool)
            or not isinstance(self.elapsed_ms, int)
            or self.elapsed_ms < 0
        ):
            raise ValueError("elapsed_ms must be a non-negative integer")
        _validate_unique_texts(self.warnings, "warnings")
        for line in self.lines:
            _, _, right, bottom = line.bbox_px
            if right > self.pixel_size.width or bottom > self.pixel_size.height:
                raise ValueError("bbox_px exceeds the oriented pixel size")

    @classmethod
    def from_request(
        cls,
        request: OcrRequest,
        *,
        status: OcrStatus,
        engine_fingerprint: str,
        lines: tuple[OcrLine, ...] = (),
        elapsed_ms: int,
        warnings: tuple[str, ...] = (),
        failure_code: OcrFailureCode | None = None,
    ) -> OcrResult:
        return cls(
            status=status,
            engine_fingerprint=engine_fingerprint,
            source_unit_id=request.source_unit_id,
            source_sha256=request.source_sha256,
            image_sha256=request.image_sha256,
            page=request.page,
            pixel_size=request.pixel_size,
            dpi=request.dpi,
            language_hints=request.language_hints,
            lines=lines,
            elapsed_ms=elapsed_ms,
            warnings=warnings,
            failure_code=failure_code,
        )


class VisionPurpose(StrEnum):
    CAPTION = "caption"
    CHART = "chart"
    FORMULA = "formula"
    GENERAL = "general"
    QUESTION = "question"


MAX_VISION_QUESTION_CHARS = 4000
MAX_VISION_CALL_ID_CHARS = 300


def normalize_vision_question(
    purpose: VisionPurpose, question: object
) -> str | None:
    """校验自由问题与用途的组合，所有工具入口复用同一规则。"""

    if not isinstance(purpose, VisionPurpose):
        raise ValueError("purpose must be a VisionPurpose")
    if purpose is not VisionPurpose.QUESTION:
        if question is not None:
            raise ValueError("question is only allowed when purpose=question")
        return None
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty text when purpose=question")
    normalized = question.strip()
    if "\x00" in normalized:
        raise ValueError("question must not contain NUL characters")
    if len(normalized) > MAX_VISION_QUESTION_CHARS:
        raise ValueError(
            f"question must contain at most {MAX_VISION_QUESTION_CHARS} characters"
        )
    return normalized


def validate_vision_call_identity(
    purpose: VisionPurpose, logical_tool_call_id: str | None
) -> None:
    """问答需要 Host 已冻结的调用身份；不能用临时随机数模拟恢复身份。"""

    if logical_tool_call_id is None and purpose is not VisionPurpose.QUESTION:
        return
    if (
        not isinstance(logical_tool_call_id, str)
        or not logical_tool_call_id
        or logical_tool_call_id != logical_tool_call_id.strip()
        or len(logical_tool_call_id) > MAX_VISION_CALL_ID_CHARS
        or any(
            character.isspace() or not character.isprintable()
            for character in logical_tool_call_id
        )
    ):
        raise ValueError("logical_tool_call_id must be a non-empty Host call identity")


class VisionDetail(StrEnum):
    """一次分析值得使用多少像素细节。

    由模型选择，因为只有模型知道自己是在问“这是什么图”还是“读取坐标轴标签”。在密集图表
    上测得，该阶梯分别消耗 379 / 1009 / 2014 个输入 token，三档都能正确读取，因此保守
    选择成本很低；首次尝试无法解析时也可选择高档。

    刻意不提供提交原图选项：供应商会按自身规则缩小超大图像；对 2400 万像素源，这会静默
    损坏精确字符串，且成本高于这里任何一档。
    """

    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"

    @property
    def pixel_budget(self) -> int:
        return _DETAIL_PIXEL_BUDGET[self]


_DETAIL_PIXEL_BUDGET: dict["VisionDetail", int] = {}


class VisionRegion(StrEnum):
    """提交一个单元周围多大范围的页面内容。

    这是模型逐级提升的档位，而不是由它编造的坐标。模型只看过渲染裁剪图，没有依据选择
    PDF point；但它能发现柱形数值在图像顶部缺失，并据此请求更大范围。
    """

    DETECTED = "detected"
    EXPANDED = "expanded"
    PAGE = "page"


class VisionStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


_DETAIL_PIXEL_BUDGET.update({
    VisionDetail.LOW: 350_000,
    VisionDetail.STANDARD: 1_000_000,
    VisionDetail.HIGH: 2_000_000,
})


@dataclass(frozen=True, slots=True)
class VisionRequest:
    """拟用于语义分析图像的可审计身份。"""

    source_unit_id: str
    source_sha256: str
    image_sha256: str
    locator: "DocumentLocator"
    mime_type: str
    pixel_size: PixelSize
    byte_count: int
    purpose: VisionPurpose
    prompt_contract_version: str
    # 供应商准备路径。构建此请求前已经完成 disclosure 分类；PDF 始终保留
    # 父文档路径与页码，只在外部分派前按需渲染。外部分派会附加下方精确已准备字节。
    image_path: str
    detail: VisionDetail = VisionDetail.STANDARD
    region: VisionRegion = VisionRegion.DETECTED
    disclosure_receipt_id: str | None = None
    question: str | None = None
    # Host 签发的稳定工具调用身份，不属于模型可填写的参数；问答用它区分新调用与恢复。
    logical_tool_call_id: str | None = None
    # 仅供 Host 使用，且刻意排除在 repr/equality 之外。它绝不能投影到模型可见请求 JSON，
    # 但通过持久发送适配器传递不可变字节，正是把验证、预留和供应商传输绑定到同一 payload
    # 的方式。
    prepared_payload: "VisionPayload | None" = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        # 延迟导入，使 ``personagraph.input_processing.vision`` 在新进程中仍可导入。立即导入
        # document 包会加载其图像 reader，而该 reader 本身依赖本模块。
        from ..documents.contracts import DocumentLocator

        _require_text(self.source_unit_id, "source_unit_id")
        _require_sha256(self.source_sha256, "source_sha256")
        _require_sha256(self.image_sha256, "image_sha256")
        if not isinstance(self.locator, DocumentLocator):
            raise ValueError("locator must be a DocumentLocator")
        if not isinstance(self.pixel_size, PixelSize):
            raise ValueError("pixel_size must be a PixelSize")
        if not isinstance(self.purpose, VisionPurpose):
            raise ValueError("purpose must be a VisionPurpose")
        object.__setattr__(
            self, "question", normalize_vision_question(self.purpose, self.question)
        )
        validate_vision_call_identity(self.purpose, self.logical_tool_call_id)
        if self.mime_type not in {"image/jpeg", "image/png"}:
            raise ValueError("unsupported vision request mime_type")
        if (
            isinstance(self.byte_count, bool)
            or not isinstance(self.byte_count, int)
            or self.byte_count < 1
        ):
            raise ValueError("byte_count must be a positive integer")
        _require_text(self.prompt_contract_version, "prompt_contract_version")
        _require_text(self.image_path, "image_path")
        if not isinstance(self.detail, VisionDetail):
            raise ValueError("detail must be a VisionDetail")
        if not isinstance(self.region, VisionRegion):
            raise ValueError("region must be a VisionRegion")
        if self.disclosure_receipt_id is not None:
            _require_text(self.disclosure_receipt_id, "disclosure_receipt_id")
        if self.prepared_payload is not None:
            from .imaging import VisionPayload

            prepared = self.prepared_payload
            if not isinstance(prepared, VisionPayload):
                raise ValueError("prepared_payload must be a VisionPayload")
            if not isinstance(prepared.data, bytes) or not prepared.data:
                raise ValueError("prepared_payload data must be immutable bytes")
            if prepared.mime_type not in {"image/jpeg", "image/png"}:
                raise ValueError("prepared_payload has an unsupported mime_type")
            if not isinstance(prepared.pixel_size, PixelSize):
                raise ValueError("prepared_payload pixel_size must be a PixelSize")
            if not isinstance(prepared.resampled, bool):
                raise ValueError("prepared_payload resampled must be boolean")
            if hashlib.sha256(prepared.data).hexdigest() != prepared.sent_sha256:
                raise ValueError("prepared_payload sent_sha256 does not match its bytes")
        # 这是已验证文件/渲染的身份。source_sha256 保持为父文档身份（尤其是原始 PDF），
        # 而 sent_sha256 是渲染/重采样后的 PNG。
            if prepared.source_sha256 != self.image_sha256:
                raise ValueError(
                    "prepared_payload source_sha256 does not match image_sha256"
                )

    @property
    def page(self) -> int | None:
        return self.locator.page


@dataclass(frozen=True, slots=True)
class VisionCapabilitySnapshot:
    available: bool
    provider: str
    model: str
    endpoint_identity: str
    processor_fingerprint: str
    supported_purposes: tuple[VisionPurpose, ...] = ()
    reason_code: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.provider, "provider")
        _require_text(self.model, "model")
        _require_text(self.endpoint_identity, "endpoint_identity")
        _require_text(self.processor_fingerprint, "processor_fingerprint")
        if not isinstance(self.available, bool):
            raise ValueError("available must be boolean")
        if not isinstance(self.supported_purposes, tuple) or any(
            not isinstance(purpose, VisionPurpose)
            for purpose in self.supported_purposes
        ):
            raise ValueError("supported_purposes must contain VisionPurpose values")
        if len(self.supported_purposes) != len(set(self.supported_purposes)):
            raise ValueError("supported_purposes must not contain duplicates")
        if self.available:
            if not self.supported_purposes or self.reason_code is not None:
                raise ValueError("available vision capability requires purposes and no reason")
        elif self.supported_purposes or not self.reason_code:
            raise ValueError("unavailable vision capability requires a reason and no purposes")


@dataclass(frozen=True, slots=True)
class VisionObservation:
    observation_id: str
    kind: str
    text: str
    uncertainty: float | None = None

    def __post_init__(self) -> None:
        _require_text(self.observation_id, "observation_id")
        _require_text(self.kind, "kind")
        _require_text(self.text, "text")
        # Absence means the provider supplied no score, not certainty or failure.
        if self.uncertainty is None:
            return
        if (
            isinstance(self.uncertainty, bool)
            or not isinstance(self.uncertainty, (int, float))
            or not math.isfinite(float(self.uncertainty))
            or not 0.0 <= float(self.uncertainty) <= 1.0
        ):
            raise ValueError("uncertainty must be within 0..1")
        object.__setattr__(self, "uncertainty", float(self.uncertainty))


@dataclass(frozen=True, slots=True)
class VisionFailureDiagnostics:
    """可持久化的失败事实；不接收异常消息、响应正文或任意 HTTP 头。"""

    phase: str
    elapsed_ms: int
    timeout_s: float
    completion_uncertain: bool
    exception_type: str | None = None
    cause_type: str | None = None
    http_status: int | None = None
    retry_after_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, str) or self.phase not in {
            "request", "response_decode", "response_validate"
        }:
            raise ValueError("unsupported vision failure phase")
        for name in ("exception_type", "cause_type"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str)
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", value) is None
            ):
                raise ValueError(f"{name} must be a bounded exception class name")
        if (
            isinstance(self.elapsed_ms, bool)
            or not isinstance(self.elapsed_ms, int)
            or self.elapsed_ms < 0
        ):
            raise ValueError("elapsed_ms must be a non-negative integer")
        if not isinstance(self.completion_uncertain, bool):
            raise ValueError("completion_uncertain must be boolean")
        if self.http_status is not None and (
            isinstance(self.http_status, bool)
            or not isinstance(self.http_status, int)
            or not 100 <= self.http_status <= 599
        ):
            raise ValueError("http_status must be a valid HTTP status")
        for name in ("timeout_s", "retry_after_s"):
            value = getattr(self, name)
            if value is None and name == "retry_after_s":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
                or (name == "timeout_s" and value == 0)
            ):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, float(value))

    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        return {
            "phase": self.phase,
            "exception_type": self.exception_type,
            "cause_type": self.cause_type,
            "http_status": self.http_status,
            "retry_after_s": self.retry_after_s,
            "elapsed_ms": self.elapsed_ms,
            "timeout_s": self.timeout_s,
            "completion_uncertain": self.completion_uncertain,
        }


@dataclass(frozen=True, slots=True)
class VisionResult:
    status: VisionStatus
    provider: str
    model: str
    endpoint_identity: str
    processor_fingerprint: str
    input_sha256: str
    output_sha256: str | None = None
    observations: tuple[VisionObservation, ...] = ()
    unresolved_gap_refs: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    failure_code: str | None = None
    failure_diagnostics: VisionFailureDiagnostics | None = None

    def __post_init__(self) -> None:
        _require_text(self.provider, "provider")
        _require_text(self.model, "model")
        _require_text(self.endpoint_identity, "endpoint_identity")
        _require_text(self.processor_fingerprint, "processor_fingerprint")
        _require_sha256(self.input_sha256, "input_sha256")
        if self.output_sha256 is not None:
            _require_sha256(self.output_sha256, "output_sha256")
        if not isinstance(self.status, VisionStatus):
            raise ValueError("status must be a VisionStatus")
        if not isinstance(self.observations, tuple) or any(
            not isinstance(item, VisionObservation) for item in self.observations
        ):
            raise ValueError("observations must contain VisionObservation values")
        _validate_unique_texts(self.unresolved_gap_refs, "unresolved_gap_refs")
        _validate_unique_texts(self.warnings, "warnings")
        if self.failure_code is not None:
            _require_text(self.failure_code, "failure_code")
        if self.failure_diagnostics is not None and (
            not isinstance(self.failure_diagnostics, VisionFailureDiagnostics)
            or self.status is not VisionStatus.FAILED
        ):
            raise ValueError("failure_diagnostics requires a failed vision result")
        if self.status in {VisionStatus.UNAVAILABLE, VisionStatus.FAILED}:
            if self.observations or not self.failure_code or not self.unresolved_gap_refs:
                raise ValueError("unavailable/failed vision result requires only failure state")
        elif self.status is VisionStatus.COMPLETED:
            if not self.observations or self.failure_code or self.unresolved_gap_refs:
                raise ValueError("completed vision result requires resolved observations")
        elif not self.observations or not self.unresolved_gap_refs:
            raise ValueError("partial vision result requires observations and gaps")


def _validate_ocr_outcome(
    status: OcrStatus,
    lines: tuple[OcrLine, ...],
    failure_code: OcrFailureCode | None,
) -> None:
    if status is OcrStatus.SUCCESS:
        if not lines or failure_code is not None:
            raise ValueError("successful OCR requires lines and no failure_code")
    elif status is OcrStatus.BLANK:
        if lines or failure_code is not None:
            raise ValueError("blank OCR cannot carry lines or failure_code")
    elif status is OcrStatus.FAILED:
        if lines or failure_code is None:
            raise ValueError("failed OCR requires failure_code and no lines")
    else:  # 防御绕过 enum 类型标注的调用方
        raise ValueError("unsupported OCR status")


def _validate_bbox(value: object, *, field_name: str, normalized: bool) -> None:
    if not isinstance(value, tuple) or len(value) != 4:
        raise ValueError(f"{field_name} must contain four numbers")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in value
    ):
        raise ValueError(f"{field_name} must contain finite numbers")
    left, top, right, bottom = (float(item) for item in value)
    if left < 0 or top < 0 or right < left or bottom < top:
        raise ValueError(f"{field_name} must be an ordered non-negative box")
    if normalized and (right > 1 or bottom > 1):
        raise ValueError("bbox_norm must be within 0..1")


def _validate_dpi(value: tuple[float, float] | None) -> None:
    if value is None:
        return
    if not isinstance(value, tuple) or len(value) != 2 or any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        or float(item) <= 0
        for item in value
    ):
        raise ValueError("dpi must contain two positive finite numbers")


def _require_sha256(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase sha256 hex digest")


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")


def _validate_nonempty_unique_texts(value: tuple[str, ...], field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    _validate_unique_texts(value, field_name)


def _validate_unique_texts(value: tuple[str, ...], field_name: str) -> None:
    if not isinstance(value, tuple) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{field_name} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{field_name} must not contain duplicates")


__all__ = [
    "MAX_VISION_QUESTION_CHARS",
    "MAX_VISION_CALL_ID_CHARS",
    "normalize_vision_question",
    "validate_vision_call_identity",
    "VisionDetail",
    "VisionRegion",
    "DEFAULT_OCR_LANGUAGE_HINTS",
    "OcrBackendResult",
    "OcrFailureCode",
    'OcrLine',
    'OcrRequest',
    'OcrResult',
    "OcrStatus",
    "PixelSize",
    'VisionCapabilitySnapshot',
    'VisionObservation',
    "VisionFailureDiagnostics",
    "VisionPurpose",
    'VisionRequest',
    'VisionResult',
    "VisionStatus",
]
