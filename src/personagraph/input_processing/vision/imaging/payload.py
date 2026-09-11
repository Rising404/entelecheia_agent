"""将请求的图像路径转换为适合发送的字节，或返回类型化拒绝。

请求携带路径，因为 receipt 和诊断会引用它；像素只在此处发送时获取，且绝不保留。该路径上
所有可能的问题——空字符串、文件缺失、格式无法解码、图像大到无法忠实读取——都是模型可以
处理的*结果*，而不是它永远看不到的异常。

缩小在此处而非供应商处发生。对本项目端点的测量表明，提交 2400 万像素原图会静默损坏
一个精确字符串，同时消耗 2523 个输入 token；同一源在本地重采样到 100 万像素后，只用
972 个 token 就能正确读取。无论如何供应商都会调整超大输入，因此真正的选择不是是否缩放，
而是我们是否知道模型看到了什么。
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from importlib import metadata
from pathlib import Path

from ...files import (
    SourceChangedDuringReadError,
    SourceSizeLimitError,
    fingerprint_file,
)
from ..contracts import PixelSize, VisionDetail, VisionRegion, VisionRequest


# 任何重采样前对*源*的边界。其作用是完全阻止病态文件被解码，而非拒绝普通照片：2400 万
# 像素相机画面远低于两项上限。
MAX_SOURCE_BYTES = 40 * 1024 * 1024
MAX_SOURCE_PIXELS = 80_000_000


# 为单个图像渲染的页面应接近细节预算，但邮票大小的区域不能为达到预算而放大 50 倍：超过
# 某个点后，新增像素只是插值，而非信息。
MIN_RENDER_SCALE = 0.5
MAX_RENDER_SCALE = 8.0

PDF_SUFFIX = ".pdf"

PREPARED_VISUAL_ARTIFACT_RECEIPT_CONTRACT = "prepared-visual-artifact-receipt-v1"
PREPARED_VISUAL_RENDER_RECIPE_CONTRACT = "prepared-visual-render-recipe-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_RENDER_RECIPE_BYTES = 32 * 1024

# detector 的 box 只是图像位置近似值，且往往过紧。在柱形图上测得，严格按报告 box 裁剪会
# 移除最高柱的印刷数值——模型随后正确报告未显示数值，但这只是对被我们破坏的问题给出的
# 如实回答。margin 会增加少量周围空白，却能恢复坐标轴标签和数据标注所在的边缘。
REGION_MARGIN_RATIO = 0.06
MIN_REGION_MARGIN_PT = 6.0

# “expanded”比 detector 报告区域宽多少。它是给已看过裁剪图、认为有内容缺失的模型使用的
# 档位，因此必须与默认值有明显差异，而不能只是轻微调整。
EXPANDED_REGION_FACTOR = 2.2

# 当检测 box 与嵌入图像矩形在较小者上达到此重叠比例时，二者视为同一图像。
REGION_MATCH_RATIO = 0.3


class PayloadFailure(StrEnum):
    """无法为分析准备图像的封闭原因集合。"""

    PATH_MISSING = "image_path_missing"
    PAGE_OUT_OF_RANGE = "image_page_out_of_range"
    REGION_INVALID = "image_region_invalid"
    PATH_NOT_A_FILE = "image_path_not_a_file"
    SOURCE_TOO_LARGE = "image_source_too_large"
    UNSUPPORTED_FORMAT = "image_format_unsupported"
    DECODE_FAILED = "image_decode_failed"
    READ_FAILED = "image_read_failed"
    SOURCE_CHANGED = "image_source_changed"


@dataclass(frozen=True)
class VisionPayload:
    """将被传输的精确内容，以及其来源证明。"""

    data: bytes
    mime_type: str
    pixel_size: PixelSize
    sent_sha256: str
    source_sha256: str
    resampled: bool
    prepared_detail: VisionDetail | None = None

    @property
    def descriptor(self) -> str:
        """本次准备的身份，用于结果指纹。"""

        shape = f"{self.pixel_size.width}x{self.pixel_size.height}"
        return f"{shape}{'+resampled' if self.resampled else ''}"


@dataclass(frozen=True, slots=True)
class PreparedVisualArtifactReceipt:
    """不含字节或路径的、实际已准备视觉产物回执。

    ``sent_sha256`` 绑定最终发送编码；``pixel_sha256`` 则在固定 RGBA 域中绑定
    解码后的像素，避免 PNG 编码器元数据变化被误认为视觉内容变化。渲染 recipe 是规范
    JSON，只描述有界准备过程和公开 locator，不携带私有文件路径。
    """

    sent_sha256: str
    pixel_sha256: str
    width: int
    height: int
    media_type: str
    preparation_fingerprint: str
    canonical_render_recipe: str

    @property
    def contract_version(self) -> str:
        return PREPARED_VISUAL_ARTIFACT_RECEIPT_CONTRACT

    def __post_init__(self) -> None:
        for name in ("sent_sha256", "pixel_sha256", "preparation_fingerprint"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase sha256 digest")
        if (
            isinstance(self.width, bool)
            or isinstance(self.height, bool)
            or not isinstance(self.width, int)
            or not isinstance(self.height, int)
            or self.width < 1
            or self.height < 1
        ):
            raise ValueError("prepared visual dimensions must be positive integers")
        if self.media_type != "image/png":
            raise ValueError("prepared visual media_type must be canonical image/png")
        encoded = self.canonical_render_recipe.encode("utf-8")
        if not encoded or len(encoded) > _MAX_RENDER_RECIPE_BYTES:
            raise ValueError("prepared visual render recipe is empty or too large")
        try:
            recipe = json.loads(self.canonical_render_recipe)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("prepared visual render recipe is invalid JSON") from exc
        if _canonical_json(recipe) != self.canonical_render_recipe:
            raise ValueError("prepared visual render recipe must be canonical JSON")
        _validate_render_recipe(recipe)
        implementation = recipe.get("implementation")
        if not isinstance(implementation, dict) or (
            _sha256_json(implementation) != self.preparation_fingerprint
        ):
            raise ValueError("preparation fingerprint does not bind the implementation")

        result = recipe["result"]
        if (
            result["sent_sha256"] != self.sent_sha256
            or result["width"] != self.width
            or result["height"] != self.height
            or result["media_type"] != self.media_type
        ):
            raise ValueError("prepared visual receipt conflicts with its render recipe")

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "PreparedVisualArtifactReceipt":
        """Rehydrate only the exact versioned, byte-free receipt schema."""

        if not isinstance(payload, Mapping) or set(payload) != {
            "contract_version",
            "height",
            "media_type",
            "pixel_sha256",
            "preparation_fingerprint",
            "render_recipe",
            "sent_sha256",
            "width",
        }:
            raise ValueError("prepared visual receipt has unsupported fields")
        if payload["contract_version"] != PREPARED_VISUAL_ARTIFACT_RECEIPT_CONTRACT:
            raise ValueError("prepared visual receipt contract is unsupported")
        recipe = payload["render_recipe"]
        if not isinstance(recipe, Mapping):
            raise ValueError("prepared visual render recipe must be an object")
        try:
            canonical_recipe = _canonical_json(dict(recipe))
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError("prepared visual render recipe is invalid") from exc
        return cls(
            sent_sha256=payload["sent_sha256"],
            pixel_sha256=payload["pixel_sha256"],
            width=payload["width"],
            height=payload["height"],
            media_type=payload["media_type"],
            preparation_fingerprint=payload["preparation_fingerprint"],
            canonical_render_recipe=canonical_recipe,
        )

    def as_payload(self) -> dict[str, object]:
        """返回可安全嵌入 durable publication envelope 的 JSON 值。"""

        return {
            "contract_version": self.contract_version,
            "height": self.height,
            "media_type": self.media_type,
            "pixel_sha256": self.pixel_sha256,
            "preparation_fingerprint": self.preparation_fingerprint,
            "render_recipe": json.loads(self.canonical_render_recipe),
            "sent_sha256": self.sent_sha256,
            "width": self.width,
        }


@dataclass(frozen=True)
class PayloadRefusal:
    """失败的准备，并携带模型可读原因。"""

    failure: PayloadFailure
    detail: str


@dataclass(frozen=True, slots=True)
class PreparedVisualArtifactBundle:
    """An inseparable prepared request and its exact byte-free receipt."""

    prepared_request: VisionRequest
    receipt: PreparedVisualArtifactReceipt

    def __post_init__(self) -> None:
        validate_prepared_visual_artifact_receipt(
            self.prepared_request,
            self.receipt,
        )

    @property
    def payload(self) -> VisionPayload:
        payload = self.prepared_request.prepared_payload
        assert isinstance(payload, VisionPayload)
        return payload


def prepare_payload_with_receipt(
    request: VisionRequest,
    *,
    detail: VisionDetail | None = None,
) -> PreparedVisualArtifactBundle | PayloadRefusal:
    """Prepare once and bind the effective request, bytes and receipt together."""

    if not isinstance(request, VisionRequest):
        raise TypeError("request must be a VisionRequest")
    if request.prepared_payload is not None:
        raise ValueError("canonical visual preparation requires an unprepared request")
    effective_detail = detail or request.detail
    payload = prepare_payload(request, detail=effective_detail)
    if isinstance(payload, PayloadRefusal):
        return payload
    # ``VisionRequest`` correctly rejects a prepared payload whose bytes no
    # longer belong to the frozen image identity.  Detect that expected race
    # before ``dataclasses.replace`` so callers receive the typed preparation
    # refusal instead of a constructor exception.
    if payload.source_sha256 != request.image_sha256:
        return PayloadRefusal(
            failure=PayloadFailure.SOURCE_CHANGED,
            detail="visual source changed during canonical preparation",
        )
    prepared_request = replace(
        request,
        detail=effective_detail,
        pixel_size=payload.pixel_size,
        prepared_payload=payload,
    )
    receipt = build_prepared_visual_artifact_receipt(prepared_request, payload)
    return PreparedVisualArtifactBundle(
        prepared_request=prepared_request,
        receipt=receipt,
    )


def build_prepared_visual_artifact_receipt(
    request: VisionRequest,
    payload: VisionPayload,
) -> PreparedVisualArtifactReceipt:
    """从实际待发送 payload 构造无字节、无路径的内部回执。

    这是独立 builder，因而不改变 ``prepare_payload`` 的既有返回类型。调用方可在完成
    disclosure 与准备后、进入 durable provider ledger 前显式构造回执。
    """

    if not isinstance(request, VisionRequest):
        raise TypeError("request must be a VisionRequest")
    if not isinstance(payload, VisionPayload):
        raise TypeError("payload must be a VisionPayload")
    if request.prepared_payload is None or request.prepared_payload != payload:
        raise ValueError("prepared visual receipt must bind request.prepared_payload")
    if not isinstance(payload.data, bytes) or not payload.data:
        raise ValueError("prepared visual payload must contain immutable bytes")
    if hashlib.sha256(payload.data).hexdigest() != payload.sent_sha256:
        raise ValueError("prepared visual sent hash does not match its bytes")
    if payload.source_sha256 != request.image_sha256:
        raise ValueError("prepared visual source hash does not match the request")
    if payload.mime_type != "image/png":
        raise ValueError("prepared visual payload must use canonical image/png")
    if payload.prepared_detail is not request.detail:
        raise ValueError("prepared visual detail does not match the bound request")

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(payload.data)) as opened:
            if opened.format != "PNG" or getattr(opened, "n_frames", 1) != 1:
                raise ValueError(
                    "prepared visual bytes must be one canonical PNG frame"
                )
            opened.load()
            if opened.size != (payload.pixel_size.width, payload.pixel_size.height):
                raise ValueError("prepared visual pixel size does not match its bytes")
            rgba = opened.convert("RGBA")
            rgba_bytes = rgba.tobytes()
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("prepared visual bytes cannot be decoded") from exc

    source_kind = (
        "pdf-region"
        if Path(request.image_path).suffix.lower() == PDF_SUFFIX
        else "raster-image"
    )
    implementation: dict[str, object] = {
        # 这是已持久化 receipt 的实现身份，不是可导入模块路径。目录迁移不能
        # 让相同像素 recipe 的既有 receipt 无故失效。
        "component": "personagraph.input_processing.vision.payload",
        "output_encoding": "png-optimize-false",
        "output_pixel_hash_domain": "personagraph-picture-rgba-pixels-v1",
        "pillow_version": _distribution_version("Pillow"),
        "raster_resample_filter": "pillow-lanczos",
        "source_kind": source_kind,
    }
    if source_kind == "pdf-region":
        implementation.update(
            {
                "expanded_region_factor": EXPANDED_REGION_FACTOR,
                "max_render_scale": MAX_RENDER_SCALE,
                "min_region_margin_pt": MIN_REGION_MARGIN_PT,
                "min_render_scale": MIN_RENDER_SCALE,
                "pdf_renderer": "pypdfium2",
                "pypdfium2_version": _distribution_version("pypdfium2"),
                "region_margin_ratio": REGION_MARGIN_RATIO,
            }
        )
    locator = request.locator
    recipe = {
        "contract_version": PREPARED_VISUAL_RENDER_RECIPE_CONTRACT,
        "implementation": implementation,
        "request": {
            "detail": payload.prepared_detail.value,
            "locator": {
                "bbox": list(locator.bbox) if locator.bbox is not None else None,
                "char_range": (
                    list(locator.char_range) if locator.char_range is not None else None
                ),
                "ordinal": locator.ordinal,
                "page": locator.page,
                "section_hierarchy_sha256": _sha256_json(list(locator.section_path)),
            },
            "region": request.region.value,
            "source_kind": source_kind,
        },
        "result": {
            "height": payload.pixel_size.height,
            "media_type": payload.mime_type,
            "resampled": payload.resampled,
            "sent_sha256": payload.sent_sha256,
            "source_sha256": payload.source_sha256,
            "width": payload.pixel_size.width,
        },
    }
    recipe_json = _canonical_json(recipe)
    pixel_domain = (
        b"personagraph-picture-rgba-pixels-v1\0"
        + f"{rgba.width}x{rgba.height}\0RGBA\0".encode("ascii")
        + rgba_bytes
    )
    return PreparedVisualArtifactReceipt(
        sent_sha256=payload.sent_sha256,
        pixel_sha256=hashlib.sha256(pixel_domain).hexdigest(),
        width=payload.pixel_size.width,
        height=payload.pixel_size.height,
        media_type=payload.mime_type,
        preparation_fingerprint=_sha256_json(implementation),
        canonical_render_recipe=recipe_json,
    )


def validate_prepared_visual_artifact_receipt(
    request: VisionRequest,
    receipt: PreparedVisualArtifactReceipt,
) -> None:
    """Recompute a receipt from the exact bound bytes without reopening the source."""

    if not isinstance(request, VisionRequest):
        raise TypeError("request must be a VisionRequest")
    if not isinstance(receipt, PreparedVisualArtifactReceipt):
        raise TypeError("receipt must be a PreparedVisualArtifactReceipt")
    payload = request.prepared_payload
    if not isinstance(payload, VisionPayload):
        raise ValueError("prepared visual receipt requires request.prepared_payload")
    expected = build_prepared_visual_artifact_receipt(request, payload)
    if expected != receipt:
        raise ValueError("prepared visual receipt does not match the bound payload")


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_render_recipe(value: object) -> None:
    """Validate the closed v1 recipe schema rather than guessing from strings."""

    if not isinstance(value, dict) or set(value) != {
        "contract_version",
        "implementation",
        "request",
        "result",
    }:
        raise ValueError("prepared visual render recipe has unsupported fields")
    if value["contract_version"] != PREPARED_VISUAL_RENDER_RECIPE_CONTRACT:
        raise ValueError("prepared visual render recipe contract is unsupported")

    implementation = value["implementation"]
    request = value["request"]
    result = value["result"]
    if (
        not isinstance(implementation, dict)
        or not isinstance(request, dict)
        or not isinstance(result, dict)
    ):
        raise ValueError("prepared visual render recipe sections must be objects")

    source_kind = implementation.get("source_kind")
    if source_kind not in {"pdf-region", "raster-image"}:
        raise ValueError("prepared visual render recipe source kind is unsupported")
    implementation_keys = {
        "component",
        "output_encoding",
        "output_pixel_hash_domain",
        "pillow_version",
        "raster_resample_filter",
        "source_kind",
    }
    if source_kind == "pdf-region":
        implementation_keys.update(
            {
                "expanded_region_factor",
                "max_render_scale",
                "min_region_margin_pt",
                "min_render_scale",
                "pdf_renderer",
                "pypdfium2_version",
                "region_margin_ratio",
            }
        )
    if set(implementation) != implementation_keys:
        raise ValueError("prepared visual implementation has unsupported fields")
    if (
        implementation["component"] != "personagraph.input_processing.vision.payload"
        or implementation["output_encoding"] != "png-optimize-false"
        or implementation["output_pixel_hash_domain"]
        != "personagraph-picture-rgba-pixels-v1"
        or implementation["raster_resample_filter"] != "pillow-lanczos"
    ):
        raise ValueError("prepared visual implementation contract is unsupported")
    _validate_distribution_version(implementation["pillow_version"])
    if source_kind == "pdf-region":
        if (
            implementation["pdf_renderer"] != "pypdfium2"
            or implementation["expanded_region_factor"] != EXPANDED_REGION_FACTOR
            or implementation["max_render_scale"] != MAX_RENDER_SCALE
            or implementation["min_region_margin_pt"] != MIN_REGION_MARGIN_PT
            or implementation["min_render_scale"] != MIN_RENDER_SCALE
            or implementation["region_margin_ratio"] != REGION_MARGIN_RATIO
        ):
            raise ValueError("prepared visual PDF implementation is unsupported")
        _validate_distribution_version(implementation["pypdfium2_version"])

    if set(request) != {"detail", "locator", "region", "source_kind"}:
        raise ValueError("prepared visual request recipe has unsupported fields")
    if request["detail"] not in {item.value for item in VisionDetail}:
        raise ValueError("prepared visual request detail is unsupported")
    if request["region"] not in {item.value for item in VisionRegion}:
        raise ValueError("prepared visual request region is unsupported")
    if request["source_kind"] != source_kind:
        raise ValueError("prepared visual request source kind conflicts")
    locator = request["locator"]
    if not isinstance(locator, dict) or set(locator) != {
        "bbox",
        "char_range",
        "ordinal",
        "page",
        "section_hierarchy_sha256",
    }:
        raise ValueError("prepared visual request locator has unsupported fields")
    _validate_optional_number_tuple(locator["bbox"], length=4)
    _validate_optional_integer_tuple(locator["char_range"], length=2)
    _validate_optional_nonnegative_integer(locator["ordinal"])
    _validate_optional_positive_integer(locator["page"])
    _validate_sha256(locator["section_hierarchy_sha256"], "section hierarchy")

    if set(result) != {
        "height",
        "media_type",
        "resampled",
        "sent_sha256",
        "source_sha256",
        "width",
    }:
        raise ValueError("prepared visual result recipe has unsupported fields")
    if (
        not isinstance(result["width"], int)
        or isinstance(result["width"], bool)
        or result["width"] < 1
        or not isinstance(result["height"], int)
        or isinstance(result["height"], bool)
        or result["height"] < 1
    ):
        raise ValueError("prepared visual result dimensions are invalid")
    if result["media_type"] != "image/png" or not isinstance(result["resampled"], bool):
        raise ValueError("prepared visual result encoding is invalid")
    _validate_sha256(result["sent_sha256"], "sent")
    _validate_sha256(result["source_sha256"], "source")


def _validate_distribution_version(value: object) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_-]*", value) is None
    ):
        raise ValueError("prepared visual dependency version is invalid")


def _validate_sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"prepared visual {label} hash is invalid")


def _validate_optional_number_tuple(value: object, *, length: int) -> None:
    if value is None:
        return
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        raise ValueError("prepared visual locator number tuple is invalid")


def _validate_optional_integer_tuple(value: object, *, length: int) -> None:
    if value is None:
        return
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in value
        )
    ):
        raise ValueError("prepared visual locator integer tuple is invalid")


def _validate_optional_nonnegative_integer(value: object) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 0
    ):
        raise ValueError("prepared visual locator ordinal is invalid")


def _validate_optional_positive_integer(value: object) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 1
    ):
        raise ValueError("prepared visual locator page is invalid")


def prepare_payload(
    request: VisionRequest,
    *,
    detail: VisionDetail | None = None,
) -> VisionPayload | PayloadRefusal:
    """发送前加载、验证、规范化并限制一张图像。"""

    level = detail or request.detail
    # ``VisionRequest`` 已在构造时拒绝空路径，因此这是第二道防线而非第一道；它确保以其他
    # 方式组装请求的任何调用方调用 ``prepare_payload`` 时都能得到完整定义的结果。
    raw_path = (request.image_path or "").strip()
    if not raw_path:
        return PayloadRefusal(PayloadFailure.PATH_MISSING, "no image path was supplied")

    path = Path(raw_path).expanduser()
    if not path.is_file():
        return PayloadRefusal(
            PayloadFailure.PATH_NOT_A_FILE, "image path is not a readable file"
        )

    if path.suffix.lower() == PDF_SUFFIX:
        # 已挂载 PDF 已跨越文档源准入边界。在此复用该边界的流式指纹，而不是把完整源复制到
        # 内存或应用普通图像字节上限。
        try:
            source_fingerprint = fingerprint_file(path)
        except SourceSizeLimitError:
            return PayloadRefusal(
                PayloadFailure.SOURCE_TOO_LARGE,
                "PDF exceeds the document source byte bound",
            )
        except SourceChangedDuringReadError:
            return PayloadRefusal(
                PayloadFailure.SOURCE_CHANGED,
                "PDF source changed while it was being verified",
            )
        except OSError as exc:
            return PayloadRefusal(PayloadFailure.READ_FAILED, type(exc).__name__)

        # PDF 内的图像没有自己的文件。只在此渲染其页面，使普通图像解码边界独立于 PDF 大小。
        region = _render_pdf_region(path, request, level)
        if isinstance(region, PayloadRefusal):
            return region
        try:
            after_render = path.stat()
        except OSError as exc:
            return PayloadRefusal(PayloadFailure.READ_FAILED, type(exc).__name__)
        if (
            after_render.st_size != source_fingerprint.size_bytes
            or after_render.st_mtime_ns != source_fingerprint.mtime_ns
        ):
            return PayloadRefusal(
                PayloadFailure.SOURCE_CHANGED,
                "PDF source changed while its page was being rendered",
            )
        return _encode(
            region,
            source_sha256=source_fingerprint.sha256,
            resampled=True,
            prepared_detail=level,
        )

    try:
        source = path.read_bytes()
    except OSError as exc:
        return PayloadRefusal(PayloadFailure.READ_FAILED, type(exc).__name__)
    if not source:
        return PayloadRefusal(PayloadFailure.DECODE_FAILED, "image file is empty")
    if len(source) > MAX_SOURCE_BYTES:
        return PayloadRefusal(
            PayloadFailure.SOURCE_TOO_LARGE,
            f"{len(source)} bytes exceeds the {MAX_SOURCE_BYTES} byte source bound",
        )

    from PIL import Image

    try:
        with Image.open(io.BytesIO(source)) as opened:
            if opened.width * opened.height > MAX_SOURCE_PIXELS:
                return PayloadRefusal(
                    PayloadFailure.SOURCE_TOO_LARGE,
                    f"{opened.width}x{opened.height} exceeds the pixel bound",
                )
            opened.load()
            image = opened.convert("RGB")
    except Exception as exc:
        # Pillow 会抛出多类解码器错误，而 HEIC 在此根本没有解码器。只保留 class 名作为
        # 细节，确保文件内容无法进入日志或 prompt。
        return PayloadRefusal(PayloadFailure.UNSUPPORTED_FORMAT, type(exc).__name__)

    budget = level.pixel_budget
    pixels = image.width * image.height
    resampled = pixels > budget
    if resampled:
        scale = (budget / pixels) ** 0.5
        target = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(target, Image.LANCZOS)

    return _encode(
        image,
        source_sha256=hashlib.sha256(source).hexdigest(),
        resampled=resampled,
        prepared_detail=level,
    )


def _encode(
    image,
    *,
    source_sha256: str,
    resampled: bool,
    prepared_detail: VisionDetail,
) -> VisionPayload:
    """序列化实际将被传输的图片。

    所有图片都使用 PNG：它是无损格式，因此重采样仍是本管线唯一损失信息之处；统一输出格式
    还能让发送哈希在多次运行间可复现。
    """

    buffer = io.BytesIO()
    image.save(buffer, "PNG", optimize=False)
    data = buffer.getvalue()
    return VisionPayload(
        data=data,
        mime_type="image/png",
        pixel_size=PixelSize(image.width, image.height),
        sent_sha256=hashlib.sha256(data).hexdigest(),
        source_sha256=source_sha256,
        resampled=resampled,
        prepared_detail=prepared_detail,
    )


def _render_pdf_region(
    path: Path, request: VisionRequest, level: VisionDetail
) -> "object":
    """按大致请求的像素预算栅格化一个页面区域。

    比例来自区域而非页面，因此小图像会按自身所需细节渲染，而不是采用适合整页的设置。
    """

    import pypdfium2 as pdfium

    page_number = request.locator.page or 1
    try:
        document = pdfium.PdfDocument(str(path))
    except Exception as exc:
        return PayloadRefusal(PayloadFailure.DECODE_FAILED, type(exc).__name__)

        # PDFium 持有原生句柄；交给垃圾回收器会在关闭时发出警告，并使文件在长寿命进程中保持
        # 映射，因此从此处退出的每条路径都会关闭文档。
    try:
        if not 1 <= page_number <= len(document):
            return PayloadRefusal(
                PayloadFailure.PAGE_OUT_OF_RANGE,
                f"page {page_number} is outside a {len(document)}-page document",
            )
        page = document[page_number - 1]
        try:
            page_width, page_height = page.get_size()
            region = _resolve_region(page, request, page_width, page_height)
            if region is None:
                return PayloadRefusal(
                    PayloadFailure.REGION_INVALID, "the requested region is empty"
                )
            left, top, right, bottom = region
            span = max(1.0, (right - left) * (bottom - top))
            scale = min(
                MAX_RENDER_SCALE,
                max(MIN_RENDER_SCALE, (level.pixel_budget / span) ** 0.5),
            )
            rendered = page.render(scale=scale).to_pil().convert("RGB")
        finally:
            page.close()
    except Exception as exc:
        return PayloadRefusal(PayloadFailure.DECODE_FAILED, type(exc).__name__)
    finally:
        document.close()

    box = (
        max(0, int(left * scale)),
        max(0, int(top * scale)),
        min(rendered.width, int(right * scale)),
        min(rendered.height, int(bottom * scale)),
    )
    if box[2] - box[0] < 1 or box[3] - box[1] < 1:
        return PayloadRefusal(
            PayloadFailure.REGION_INVALID, "the region rendered to nothing"
        )
    return rendered.crop(box)


def _resolve_region(
    page, request: VisionRequest, width: float, height: float
) -> tuple[float, float, float, float] | None:
    """决定渲染页面的哪个矩形区域。"""

    if request.region is VisionRegion.PAGE:
        return (0.0, 0.0, width, height)

    exact = _embedded_image_rect(page, request.locator.bbox, height)
    base = (
        _clamp_region(exact, width, height, margin=False)
        if exact is not None
        else _clamp_region(request.locator.bbox, width, height)
    )
    if base is None or request.region is VisionRegion.DETECTED:
        return base
    return _grow(base, EXPANDED_REGION_FACTOR, width, height)


def _embedded_image_rect(
    page, bbox: tuple[float, float, float, float] | None, height: float
) -> tuple[float, float, float, float] | None:
    """返回此 box 所指嵌入图像的精确矩形。

    布局 detector 报告它*认为*内容所在的位置；放置在 PDF 中的栅格图像则在文件中记录了
    精确矩形。对某图表的测量显示，detector box 比真实位置低 75pt，并裁掉了最高柱的印刷
    数值，因此当二者明显描述同一图像时，以权威矩形为准。
    """

    if bbox is None:
        return None
    try:
        import pypdfium2 as pdfium

        candidates = []
        for obj in page.get_objects():
            if obj.type != pdfium.raw.FPDF_PAGEOBJ_IMAGE:
                continue
            left, bottom_pdf, right, top_pdf = obj.get_bounds()
            # PDF 空间使用左下原点；locator 使用左上原点。
            candidates.append((left, height - top_pdf, right, height - bottom_pdf))
    except Exception:
        return None

    best, best_share = None, 0.0
    for rect in candidates:
        share = _overlap_share(bbox, rect)
        if share > best_share:
            best, best_share = rect, share
    return best if best_share >= REGION_MATCH_RATIO else None


def _overlap_share(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """交集面积占较小矩形的比例。"""

    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return intersection / smaller if smaller > 0 else 0.0


def _grow(
    region: tuple[float, float, float, float],
    factor: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    """以中心为基准扩大区域，同时保持在页面内。"""

    left, top, right, bottom = region
    centre_x, centre_y = (left + right) / 2, (top + bottom) / 2
    half_w = (right - left) * factor / 2
    half_h = (bottom - top) * factor / 2
    return (
        max(0.0, centre_x - half_w),
        max(0.0, centre_y - half_h),
        min(width, centre_x + half_w),
        min(height, centre_y + half_h),
    )


def _clamp_region(
    bbox: tuple[float, float, float, float] | None,
    width: float,
    height: float,
    *,
    margin: bool = True,
) -> tuple[float, float, float, float] | None:
    """稍微扩大 locator box，随后将其限制在页面内。

    两步都很重要。margin 恢复过紧 detector box 裁掉的内容；clamp 避免越出页面的 box
    导致拒绝，因为那是 detector 伪影，而非丢失图像的理由。
    """

    if bbox is None:
        return (0.0, 0.0, width, height)
    left, top, right, bottom = bbox
    if margin:
        # 只有估算 box 需要余量；精确矩形本就正确。
        margin_x = max(MIN_REGION_MARGIN_PT, (right - left) * REGION_MARGIN_RATIO)
        margin_y = max(MIN_REGION_MARGIN_PT, (bottom - top) * REGION_MARGIN_RATIO)
        left, right = left - margin_x, right + margin_x
        top, bottom = top - margin_y, bottom + margin_y
    left, right = max(0.0, min(left, width)), max(0.0, min(right, width))
    top, bottom = max(0.0, min(top, height)), max(0.0, min(bottom, height))
    if right - left < 1.0 or bottom - top < 1.0:
        return None
    return (left, top, right, bottom)


__all__ = [
    "MAX_SOURCE_BYTES",
    "MAX_SOURCE_PIXELS",
    "EXPANDED_REGION_FACTOR",
    "MAX_RENDER_SCALE",
    "MIN_REGION_MARGIN_PT",
    "REGION_MARGIN_RATIO",
    "MIN_RENDER_SCALE",
    "PDF_SUFFIX",
    "PREPARED_VISUAL_ARTIFACT_RECEIPT_CONTRACT",
    "PREPARED_VISUAL_RENDER_RECIPE_CONTRACT",
    "PayloadFailure",
    "PayloadRefusal",
    "PreparedVisualArtifactBundle",
    "PreparedVisualArtifactReceipt",
    "VisionPayload",
    "build_prepared_visual_artifact_receipt",
    "prepare_payload",
    "prepare_payload_with_receipt",
    "validate_prepared_visual_artifact_receipt",
]
