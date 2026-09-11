"""基于 Docling 的 reader，并映射到本包的中立契约。

Docling 使用检测模型对渲染页面执行布局分析，因此能够在没有语义标记的 PDF 中区分 heading
与正文。本包的原生 reader 无法做到这一点；它们根据几何信息推断结构，而这只适用于简单的
单栏文档。

此适配器刻意把 Docling 隐藏在 ``ProcessingResult`` 后。调用方始终通过
:func:`read_document` 访问文档，永远不会知道元素由哪个引擎生成——这也让 processor 指纹
成为有意义的失效键，而非无人读取的标签。
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import platform
import sys
import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

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
from ...files import fingerprint_file
from .pdf import MAX_PAGES as MAX_DOCUMENT_PAGES
from .pdf import pdf_reader_requires_password


READER_NAME = "docling"
ADAPTER_RECIPE_VERSION = "4"

MAX_ELEMENTS = 20000
MAX_ELEMENT_CHARS = 20000
MAX_DOCUMENT_FILE_BYTES = 64 * 1024 * 1024
MAX_RENDER_SIDE_PIXELS = 16_384
MAX_RENDER_PAGE_PIXELS = 40_000_000
MAX_RENDER_TOTAL_PIXELS = 400_000_000
PDF_BACKEND_SUPERSAMPLING_FACTOR = 1.5
# 即使完整文档布局被推迟，PDF 绝对准入仍保持有界。一个参考页面是 US Letter，采用当前
# Docling 管线冻结的相同最坏比例；1000 个此类页面约为 98 亿像素。这是路由/准入预算，
# 绝不是立即栅格化源的请求。
PDF_INGEST_REFERENCE_PAGE_WIDTH_POINTS = 612
PDF_INGEST_REFERENCE_PAGE_HEIGHT_POINTS = 792
MAX_PDF_INGEST_REFERENCE_PAGES = 1000
PDF_INGEST_POLICY_RENDER_SCALE = 3.0
MAX_ARTIFACT_FILES = 512
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
FROZEN_ARTIFACTS_MARKER = "personagraph:frozen-local-docling-artifacts"
ANNOTATION_INVENTORY_COMPONENT = "pdf-annotation-inventory-v2"
NATIVE_TABLE_INVENTORY_COMPONENT = "pdf-native-table-inventory-v2"
OCR_LANGUAGE_HINTS = ("zh-Hans", "en-US")
PDF_RENDER_PREFLIGHT_ALGORITHM = "pdf-mediabox-userunit-render-budget-v1"

_TABLE_MODEL_REPO_ID = "docling-project/docling-models"
_TABLE_MODEL_REVISION = "v2.3.0"

_INFERENCE_DISTRIBUTIONS = (
    "docling-parse",
    "huggingface-hub",
    "numpy",
    "pdfminer.six",
    "pdfplumber",
    "pillow",
    "pypdf",
    "pypdfium2",
    "safetensors",
    "torch",
    "torchvision",
    "transformers",
)
_REQUIRED_INFERENCE_DISTRIBUTIONS = frozenset({
    "docling-parse",
    "huggingface-hub",
    "numpy",
    "pdfminer.six",
    "pdfplumber",
    "pillow",
    "pypdf",
    "pypdfium2",
    "safetensors",
    "torch",
    "transformers",
})

# Docling 自身分类映射到本系统分类。未映射标签会变成段落而非被丢弃：丢失内容比丢失类型
# 区分更糟，新的 Docling 标签绝不能静默删除文本。
_LABEL_TO_KIND: dict[str, ElementKind] = {
    "title": ElementKind.HEADING,
    "section_header": ElementKind.HEADING,
    "paragraph": ElementKind.PARAGRAPH,
    "text": ElementKind.PARAGRAPH,
    "list_item": ElementKind.LIST_ITEM,
    "table": ElementKind.TABLE,
    "document_index": ElementKind.TABLE,
    "picture": ElementKind.IMAGE,
    "caption": ElementKind.CAPTION,
    "page_header": ElementKind.PARAGRAPH,
    "page_footer": ElementKind.PARAGRAPH,
    "footnote": ElementKind.PARAGRAPH,
    "formula": ElementKind.PARAGRAPH,
    "code": ElementKind.PARAGRAPH,
}


class DoclingUnavailable(RuntimeError):
    """Docling 未安装，因此无法选择此引擎。"""


@dataclass(frozen=True, slots=True)
class _DoclingResourceLimits:
    max_file_size_bytes: int
    max_num_pages: int
    render_scale: float
    backend_supersampling_factor: float
    max_render_side_pixels: int
    max_render_page_pixels: int
    max_render_total_pixels: int


class _PdfPreflightFailure(RuntimeError):
    """在 Docling 能够渲染 PDF 之前触发的 prompt 安全类型化停止。"""

    def __init__(self, code: DiagnosticCode, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class PdfRenderGeometryAssessment:
    """路由与 Docling 共享的一项非栅格化 PDF 几何决定。"""

    total_pixels: int
    eager_limit_detail: str | None = None
    rejection: ProcessingDiagnostic | None = None

    @property
    def eager_eligible(self) -> bool:
        return self.rejection is None and self.eager_limit_detail is None


@lru_cache(maxsize=1)
def docling_version() -> str:
    try:
        return importlib_metadata.version("docling")
    except importlib_metadata.PackageNotFoundError as exc:
        raise DoclingUnavailable("docling is not installed") from exc


def docling_available() -> bool:
    try:
        docling_version()
        configured_docling_recipe()
    except (DoclingUnavailable, OSError, ValueError):
        return False
    return True


@dataclass(frozen=True, slots=True)
class DoclingProcessorRecipe:
    """读取所用精确转换器配置的规范身份。

    该 JSON 被刻意设为不可变且自描述。它包含解析器/模型包版本、具体 OCR 实现及选项、模型
    身份、所有有效 PDF 管线选项，以及 Host 资源上限。``_pipeline_options_from_recipe``
    根据此值重建；若当前默认值无法精确复现它，则拒绝运行。
    """

    canonical_json: str
# （身份键、Docling artifacts 根目录子项、不可变本地快照）
    artifact_sources: tuple[tuple[str, str, str], ...] = ()

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def identity(self) -> dict[str, Any]:
        value = json.loads(self.canonical_json)
        if not isinstance(value, dict):  # pragma: no cover - 构造过程已证明这一点
            raise RuntimeError("Docling recipe identity must be an object")
        return value


def _distribution_version(name: str) -> str:
    """即使选择了可选后端依赖，也返回稳定标记。"""

    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def _ocr_options():
    """显式选择 OCR 引擎，而不是交由自动选择。

    Docling 默认为 ``auto``，其选择依赖可能随版本变化的内部启发式规则——这会在任何指纹
    都不变化的情况下静默改变提取文本乃至 chunk 内容。为引擎命名可让这一决定由我们掌控。

    macOS 上优先使用 Apple Vision framework：它无须下载模型，且比随附替代方案快一个
    数量级。其他平台默认使用 RapidOCR。
    """

    if sys.platform == "darwin":
        try:
            from docling.datamodel.pipeline_options import OcrMacOptions

            return OcrMacOptions(lang=list(OCR_LANGUAGE_HINTS))
        except Exception:
            pass
    try:
        from docling.datamodel.pipeline_options import RapidOcrOptions

        return RapidOcrOptions()
    except Exception:
        return None


def _pipeline_options_payload(options: Any) -> dict[str, Any]:
    payload = options.model_dump(mode="json", serialize_as_any=True)
    if not isinstance(payload, dict):  # pragma: no cover - Docling 契约防御分支
        raise RuntimeError("Docling pipeline options did not serialize to an object")
    if payload.get("artifacts_path") is not None:
    # 机器本地暂存路径属于运行细节，而非语义。规范 artifact commit 与内容哈希另存于 recipe，
    # 它们才是跨机器使派生数据失效的依据。
        payload["artifacts_path"] = FROZEN_ARTIFACTS_MARKER
    return payload


def _component_identity(options: Any) -> dict[str, Any]:
    payload = options.model_dump(mode="json", serialize_as_any=True)
    if not isinstance(payload, dict):  # pragma: no cover - Docling 契约防御分支
        raise RuntimeError("Docling component options did not serialize to an object")
    model_spec = payload.get("model_spec")
    return {
        "class": f"{type(options).__module__}.{type(options).__qualname__}",
        "kind": str(getattr(options, "kind", "")),
        "model_spec": model_spec if isinstance(model_spec, dict) else {},
        "options": payload,
    }


def _new_pipeline_options(
    ocr_options: Any,
    *,
    artifacts_path: Path | None = None,
) -> Any:
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    options = PdfPipelineOptions()
    # 采用此引擎正是为了扫描页面；关闭 OCR 会使它们返回空内容，而本设计正要避免该失败。
    options.do_ocr = True
    options.ocr_options = ocr_options
    options.artifacts_path = artifacts_path
    return options


def _ocr_package_versions(backend: str, options: Any) -> dict[str, str]:
    names: list[str]
    if backend == "ocrmac":
        names = ["ocrmac"]
    elif backend == "rapidocr":
        backend_distribution = {
            "onnxruntime": "onnxruntime",
            "openvino": "openvino",
            "paddle": "paddlepaddle",
            "torch": "torch",
        }.get(str(getattr(options, "backend", "")))
        names = ["rapidocr"]
        if backend_distribution:
            names.append(backend_distribution)
    else:
        names = [backend]
    return {name: _distribution_version(name) for name in sorted(set(names))}


def _ocr_model_identity(backend: str, options: Any) -> dict[str, Any]:
    if backend == "ocrmac":
        return {
            "resolver": "apple-vision-framework",
            "framework": str(getattr(options, "framework", "vision")),
            "recognition": str(getattr(options, "recognition", "")),
        # Vision 实现随 macOS 而非 Python wheel 发布，因此 OS 版本就是其有效模型/包版本。
            "macos_version": platform.mac_ver()[0],
            "kernel_release": platform.release(),
        }
    if backend == "rapidocr":
        return {
            "resolver": "rapidocr-frozen-local-artifacts",
            "inference_backend": str(getattr(options, "backend", "")),
            "languages": list(getattr(options, "lang", ()) or ()),
        }
    return {"resolver": f"{backend}-explicit-options"}


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _artifact_manifest(root: Path) -> dict[str, Any]:
    """为所有可能参与冻结模型快照的文件计算哈希。"""

    if not root.is_dir():
        raise DoclingUnavailable("a required local Docling artifact is unavailable")
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        size = candidate.stat().st_size
        total_bytes += size
        if len(files) >= MAX_ARTIFACT_FILES or total_bytes > MAX_ARTIFACT_BYTES:
            raise DoclingUnavailable("local Docling artifact manifest exceeds safety caps")
        files.append(
            {
                "path": candidate.relative_to(root).as_posix(),
                "sha256": _sha256_path(candidate),
                "size_bytes": size,
            }
        )
    if not files:
        raise DoclingUnavailable("a required local Docling artifact snapshot is empty")
    return {
        "file_count": len(files),
        "files": files,
        "total_bytes": total_bytes,
    }


def _resolve_hf_snapshot(repo_id: str, revision: str) -> tuple[str, Path]:
    """在无网络访问的情况下解析已缓存的 HF commit。"""

    from huggingface_hub.constants import HF_HUB_CACHE

    safe_ref_characters = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    )
    if not revision or any(char not in safe_ref_characters for char in revision):
        raise DoclingUnavailable("Docling model revision is not a safe local ref")
    repo_cache = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
    if len(revision) == 40 and all(char in "0123456789abcdef" for char in revision):
        commit = revision
    else:
        ref_path = repo_cache / "refs" / revision
        try:
            with ref_path.open("r", encoding="ascii") as source:
                commit = source.read(81).strip()
        except OSError as exc:
            raise DoclingUnavailable(
                "a required pinned Docling model is not cached locally"
            ) from exc
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise DoclingUnavailable("cached Docling model ref is not an immutable commit")
    snapshot = repo_cache / "snapshots" / commit
    if not snapshot.is_dir():
        raise DoclingUnavailable("cached Docling model snapshot is unavailable")
    return commit, snapshot


def _resolved_artifacts(
    backend: str,
    pipeline_options: Any,
) -> tuple[dict[str, Any], tuple[tuple[str, str, str], ...]]:
    layout_spec = pipeline_options.layout_options.model_spec
    layout_repo = str(layout_spec.repo_id)
    layout_revision = str(layout_spec.revision or "main")
    layout_commit, layout_snapshot = _resolve_hf_snapshot(
        layout_repo,
        layout_revision,
    )
    table_commit, table_snapshot = _resolve_hf_snapshot(
        _TABLE_MODEL_REPO_ID,
        _TABLE_MODEL_REVISION,
    )
    required_layout_files = (
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
    )
    required_table_files = (
        "model_artifacts/tableformer/accurate/tableformer_accurate.safetensors",
        "model_artifacts/tableformer/accurate/tm_config.json",
    )
    if not all((layout_snapshot / name).is_file() for name in required_layout_files):
        raise DoclingUnavailable("cached Docling layout snapshot is incomplete")
    if not all((table_snapshot / name).is_file() for name in required_table_files):
        raise DoclingUnavailable("cached Docling table snapshot is incomplete")
    artifacts: dict[str, Any] = {
        "layout": {
            "manifest": _artifact_manifest(layout_snapshot),
            "repo_id": layout_repo,
            "requested_revision": layout_revision,
            "resolved_commit": layout_commit,
        },
        "table": {
            "manifest": _artifact_manifest(table_snapshot),
            "repo_id": _TABLE_MODEL_REPO_ID,
            "requested_revision": _TABLE_MODEL_REVISION,
            "resolved_commit": table_commit,
        },
    }
    sources: list[tuple[str, str, str]] = [
        (
            "layout",
            layout_repo.replace("/", "--"),
            str(layout_snapshot),
        ),
        (
            "table",
            _TABLE_MODEL_REPO_ID.replace("/", "--"),
            str(table_snapshot),
        ),
    ]
    if backend == "rapidocr":
        from docling.datamodel.settings import settings

        rapidocr_root = Path(settings.cache_dir) / "models" / "RapidOcr"
        artifacts["ocr"] = {
            "manifest": _artifact_manifest(rapidocr_root),
            "repo_id": "docling-local/RapidOcr",
            "requested_revision": "local-prefetched",
            "resolved_commit": None,
        }
        sources.append(("ocr", "RapidOcr", str(rapidocr_root)))
    return artifacts, tuple(sources)


def _layout_image_processor_identity(snapshot: Path) -> dict[str, Any]:
    """解析与 Docling 引擎相同的默认 processor 选择。"""

    from transformers import AutoImageProcessor

    # Docling 2.119 省略 ``use_fast``。保留这一事实及其解析到的具体 class/default，避免
    # Transformers 默认值翻转却隐藏在相同 processor 指纹之后。
    processor = AutoImageProcessor.from_pretrained(
        str(snapshot),
        local_files_only=True,
    )
    raw_options = processor.to_dict()
    options = json.loads(json.dumps(
        raw_options,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ))
    return {
        "class": f"{type(processor).__module__}.{type(processor).__qualname__}",
        "options": options,
        "resolved_is_fast": bool(getattr(processor, "is_fast", False)),
        "use_fast_argument": None,
        "use_fast_policy": "docling-call-omits-argument;resolved-choice-is-bound",
    }


def _inference_runtime_identity(
    pipeline_options: Any,
    artifact_sources: tuple[tuple[str, str, str], ...],
) -> dict[str, Any]:
    from docling.utils.accelerator_utils import decide_device

    sources = {key: Path(path) for key, _target, path in artifact_sources}
    layout_snapshot = sources.get("layout")
    if layout_snapshot is None:
        raise DoclingUnavailable("Docling layout runtime has no frozen snapshot")
    packages = {
        name: _distribution_version(name)
        for name in _INFERENCE_DISTRIBUTIONS
    }
    if any(
        packages[name] == "not-installed"
        for name in _REQUIRED_INFERENCE_DISTRIBUTIONS
    ):
        raise DoclingUnavailable("a required Docling inference runtime is unavailable")
    engine_options = pipeline_options.layout_options.engine_options
    raw_engine = getattr(engine_options, "engine_type", "")
    engine = str(getattr(raw_engine, "value", raw_engine))
    resolved_device = str(decide_device(
        pipeline_options.accelerator_options.device
    ))
    return {
        "layout_engine": engine,
        "layout_engine_options": engine_options.model_dump(
            mode="json",
            serialize_as_any=True,
        ),
        "layout_image_processor": _layout_image_processor_identity(
            layout_snapshot
        ),
        "packages": packages,
        "resolved_accelerator": resolved_device,
        "table_accelerator": (
            "cpu" if resolved_device == "mps" else resolved_device
        ),
        "table_engine": str(
            getattr(pipeline_options.table_structure_options, "kind", "")
        ),
        "table_engine_options": (
            pipeline_options.table_structure_options.model_dump(
                mode="json",
                serialize_as_any=True,
            )
        ),
    }


def _pipeline_render_scale(pipeline_options: Any) -> float:
    """返回此冻结管线可能请求的最大页面像素比例。"""

    candidates = [float(pipeline_options.images_scale)]
    if bool(pipeline_options.do_ocr):
        candidates.append(float(pipeline_options.ocr_options.scale))
    if any(not math.isfinite(value) or value <= 0 for value in candidates):
        raise DoclingUnavailable("Docling configured a non-finite render scale")
    return max(candidates)


def _frozen_artifact_root(recipe: DoclingProcessorRecipe) -> Path:
    """验证 recipe 字节，并按 Docling 离线布局公开它们。"""

    identity = recipe.identity
    expected_artifacts = identity.get("artifacts")
    if not isinstance(expected_artifacts, dict) or not recipe.artifact_sources:
        raise DoclingUnavailable("the Docling recipe has no frozen local artifacts")
    staging = (
        Path(tempfile.gettempdir())
        / "personagraph-docling-artifacts"
        / recipe.digest
    )
    staging.mkdir(mode=0o700, parents=True, exist_ok=True)
    if staging.is_symlink() or not staging.is_dir():
        raise DoclingUnavailable("Docling artifact staging root is not a directory")
    for key, target_name, raw_source in recipe.artifact_sources:
        expected = expected_artifacts.get(key)
        if not isinstance(expected, dict):
            raise DoclingUnavailable("Docling artifact identity is incomplete")
        source = Path(raw_source).resolve(strict=True)
        if _artifact_manifest(source) != expected.get("manifest"):
            raise DoclingUnavailable("local Docling artifact bytes changed after pinning")
        target = staging / target_name
        try:
            target.symlink_to(source, target_is_directory=True)
        except FileExistsError:
            pass
        try:
            resolved_target = target.resolve(strict=True)
        except OSError as exc:
            raise DoclingUnavailable("Docling artifact staging link is invalid") from exc
        if not target.is_symlink() or resolved_target != source:
            raise DoclingUnavailable("Docling artifact staging link does not match recipe")
    return staging


@lru_cache(maxsize=1)
def configured_docling_recipe() -> DoclingProcessorRecipe:
    """冻结由指纹计算与转换共同使用的唯一 recipe。"""

    ocr_options = _ocr_options()
    if ocr_options is None:
        raise DoclingUnavailable("no explicit Docling OCR backend is available")
    backend = str(getattr(ocr_options, "kind", "auto"))
    if backend == "auto":
        raise DoclingUnavailable("Docling OCR auto-selection is not a frozen recipe")
    ocr_packages = _ocr_package_versions(backend, ocr_options)
    if "not-installed" in ocr_packages.values():
        raise DoclingUnavailable("the selected Docling OCR backend is not installed")
    pipeline_options = _new_pipeline_options(
        ocr_options,
        artifacts_path=Path(FROZEN_ARTIFACTS_MARKER),
    )
    artifacts, artifact_sources = _resolved_artifacts(backend, pipeline_options)
    inference_runtime = _inference_runtime_identity(
        pipeline_options,
        artifact_sources,
    )
    ocr_payload = ocr_options.model_dump(mode="json", serialize_as_any=True)
    identity = {
        "adapter_components": [
            ANNOTATION_INVENTORY_COMPONENT,
            NATIVE_TABLE_INVENTORY_COMPONENT,
        ],
        "adapter_recipe_version": ADAPTER_RECIPE_VERSION,
        "artifacts": artifacts,
        "inference_runtime": inference_runtime,
        "packages": {
            name: _distribution_version(name)
            for name in ("docling", "docling-core", "docling-ibm-models")
        },
        "ocr": {
            "backend": backend,
            "class": f"{type(ocr_options).__module__}.{type(ocr_options).__qualname__}",
            "model": _ocr_model_identity(backend, ocr_options),
            "options": ocr_payload,
            "packages": ocr_packages,
        },
        "pipeline": {
            "class": (
                f"{type(pipeline_options).__module__}."
                f"{type(pipeline_options).__qualname__}"
            ),
            "components": {
                "layout": _component_identity(pipeline_options.layout_options),
                "table": _component_identity(
                    pipeline_options.table_structure_options
                ),
            },
            "options": _pipeline_options_payload(pipeline_options),
        },
        "resource_limits": {
            "max_file_size_bytes": MAX_DOCUMENT_FILE_BYTES,
            "max_num_pages": MAX_DOCUMENT_PAGES,
            "pdf_render_geometry": {
                "algorithm": PDF_RENDER_PREFLIGHT_ALGORITHM,
                "backend_supersampling_factor": PDF_BACKEND_SUPERSAMPLING_FACTOR,
                "max_page_pixels": MAX_RENDER_PAGE_PIXELS,
                "max_side_pixels": MAX_RENDER_SIDE_PIXELS,
                "max_total_pixels": MAX_RENDER_TOTAL_PIXELS,
                "render_scale": _pipeline_render_scale(pipeline_options),
            },
        },
    }
    return DoclingProcessorRecipe(
        json.dumps(
            identity,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        artifact_sources=artifact_sources,
    )


def fingerprint(
    recipe: DoclingProcessorRecipe | None = None,
) -> ProcessorFingerprint:
    """标识一次读取使用的精确包、模型、选项和上限。"""

    frozen = recipe or configured_docling_recipe()
    identity = frozen.identity
    return ProcessorFingerprint(
        READER_NAME,
        (
            f"{identity['packages']['docling']}+{identity['ocr']['backend']}+"
            f"recipe{frozen.digest}+adapter{ADAPTER_RECIPE_VERSION}"
        ),
    )


def ocr_engine_name() -> str:
    """本安装将使用的 OCR 引擎名称。"""

    return str(configured_docling_recipe().identity["ocr"]["backend"])


def _pipeline_options_from_recipe(recipe: DoclingProcessorRecipe) -> Any:
    """重建并验证冻结在 ``recipe`` 中的精确选项。"""

    from docling.datamodel.pipeline_options import OcrMacOptions, RapidOcrOptions

    identity = recipe.identity
    ocr_identity = identity["ocr"]
    ocr_class = {
        "ocrmac": OcrMacOptions,
        "rapidocr": RapidOcrOptions,
    }.get(str(ocr_identity["backend"]))
    if ocr_class is None:
        raise DoclingUnavailable("the frozen Docling OCR backend is unsupported")
    ocr_options = ocr_class.model_validate(ocr_identity["options"])
    options = _new_pipeline_options(
        ocr_options,
        artifacts_path=_frozen_artifact_root(recipe),
    )
    if (
        _inference_runtime_identity(options, recipe.artifact_sources)
        != identity["inference_runtime"]
    ):
        raise RuntimeError("current Docling inference runtime does not match recipe")
    if _pipeline_options_payload(options) != identity["pipeline"]["options"]:
        # 默认值已变化，但包/版本身份未变化。若继续沿用旧指纹，派生数据将无法审计。
        raise RuntimeError("current Docling options do not match the frozen recipe")
    return options


@lru_cache(maxsize=1)
def _converter(recipe: DoclingProcessorRecipe | None = None):
    """只构建一次转换器。

    Docling 首次使用时会加载布局模型；若为每个文档重建转换器，每次摄取都要承担该成本。
    """

    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    frozen = recipe or configured_docling_recipe()
    options = _pipeline_options_from_recipe(frozen)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeError(f"invalid Docling recipe resource limit: {field}")
    return value


def _resource_limits_from_recipe(
    recipe: DoclingProcessorRecipe,
) -> _DoclingResourceLimits:
    """读取已计算进此转换 recipe 哈希的精确上限。"""

    resource_limits = recipe.identity.get("resource_limits")
    if not isinstance(resource_limits, dict):
        raise RuntimeError("Docling recipe has no resource limit identity")
    render = resource_limits.get("pdf_render_geometry")
    if not isinstance(render, dict):
        raise RuntimeError("Docling recipe has no PDF render geometry limits")
    if render.get("algorithm") != PDF_RENDER_PREFLIGHT_ALGORITHM:
        raise RuntimeError("Docling recipe uses an unsupported PDF preflight")
    render_scale = render.get("render_scale")
    backend_factor = render.get("backend_supersampling_factor")
    for value, field in (
        (render_scale, "render scale"),
        (backend_factor, "backend supersampling factor"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise RuntimeError(f"invalid Docling recipe {field}")
    limits = _DoclingResourceLimits(
        max_file_size_bytes=_positive_int(
            resource_limits.get("max_file_size_bytes"),
            field="max_file_size_bytes",
        ),
        max_num_pages=_positive_int(
            resource_limits.get("max_num_pages"),
            field="max_num_pages",
        ),
        render_scale=float(render_scale),
        backend_supersampling_factor=float(backend_factor),
        max_render_side_pixels=_positive_int(
            render.get("max_side_pixels"),
            field="max_side_pixels",
        ),
        max_render_page_pixels=_positive_int(
            render.get("max_page_pixels"),
            field="max_page_pixels",
        ),
        max_render_total_pixels=_positive_int(
            render.get("max_total_pixels"),
            field="max_total_pixels",
        ),
    )
    if limits.max_render_total_pixels < limits.max_render_page_pixels:
        raise RuntimeError("Docling total render budget is smaller than one page")
    return limits


def _finite_pdf_number(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite PDF geometry")
    return number


def _default_pdf_ingest_limits() -> _DoclingResourceLimits:
    """返回 Docling 本身不可用时采用的冻结 Host 策略。"""

    return _DoclingResourceLimits(
        max_file_size_bytes=MAX_DOCUMENT_FILE_BYTES,
        max_num_pages=MAX_DOCUMENT_PAGES,
        render_scale=PDF_INGEST_POLICY_RENDER_SCALE,
        backend_supersampling_factor=PDF_BACKEND_SUPERSAMPLING_FACTOR,
        max_render_side_pixels=MAX_RENDER_SIDE_PIXELS,
        max_render_page_pixels=MAX_RENDER_PAGE_PIXELS,
        max_render_total_pixels=MAX_RENDER_TOTAL_PIXELS,
    )


def _absolute_pdf_ingest_total_pixels(limits: _DoclingResourceLimits) -> int:
    effective_scale = limits.render_scale * limits.backend_supersampling_factor
    reference_width = math.ceil(
        PDF_INGEST_REFERENCE_PAGE_WIDTH_POINTS * effective_scale
    )
    reference_height = math.ceil(
        PDF_INGEST_REFERENCE_PAGE_HEIGHT_POINTS * effective_scale
    )
    return reference_width * reference_height * MAX_PDF_INGEST_REFERENCE_PAGES


def assess_pdf_ingest_geometry(
    path: Path,
    *,
    recipe: DoclingProcessorRecipe | None = None,
) -> PdfRenderGeometryAssessment:
    """在不渲染 PDF 的情况下，将其分类为 eager、deferred 或 rejected。

    如果存在 ``recipe``，它会提供精确的已配置 Docling 比例与 eager 上限。若不存在，上方
    稳定 Host 准入策略会让仅原生安装仍受相同 PDF 绝对边界约束。
    """

    limits = (
        _resource_limits_from_recipe(recipe)
        if recipe is not None
        else _default_pdf_ingest_limits()
    )
    try:
        return _assess_pdf_ingest_geometry(path, limits)
    except _PdfPreflightFailure as exc:
        return PdfRenderGeometryAssessment(
            total_pixels=0,
            rejection=ProcessingDiagnostic(exc.code, detail=exc.detail),
        )


def _assess_pdf_ingest_geometry(
    path: Path,
    limits: _DoclingResourceLimits,
) -> PdfRenderGeometryAssessment:
    """只检查一次页面几何，并保留首个仅针对 eager 的超限项。"""

    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path), strict=False)
        if pdf_reader_requires_password(reader):
            raise _PdfPreflightFailure(
                DiagnosticCode.PASSWORD_REQUIRED,
                "pdf_password_required",
            )

        root = reader.trailer.get("/Root")
        page_tree = root.get("/Pages") if isinstance(root, Mapping) else None
        declared_count = (
            page_tree.get("/Count") if isinstance(page_tree, Mapping) else None
        )
        if declared_count is not None and int(declared_count) > limits.max_num_pages:
            raise _PdfPreflightFailure(
                DiagnosticCode.LIMIT_REACHED,
                "pdf_page_count_limit_reached",
            )

        total_pixels = 0
        page_count = 0
        eager_limit_detail: str | None = None
        absolute_total_pixels = _absolute_pdf_ingest_total_pixels(limits)
        effective_scale = limits.render_scale * limits.backend_supersampling_factor
        for page_number, page in enumerate(reader.pages, start=1):
            page_count = page_number
            if page_number > limits.max_num_pages:
                raise _PdfPreflightFailure(
                    DiagnosticCode.LIMIT_REACHED,
                    "pdf_page_count_limit_reached",
                )
            box = page.mediabox
            user_unit = _finite_pdf_number(page.get("/UserUnit", 1))
            if user_unit <= 0:
                raise ValueError("invalid PDF UserUnit")
            width_points = abs(
                _finite_pdf_number(box.right) - _finite_pdf_number(box.left)
            ) * user_unit
            height_points = abs(
                _finite_pdf_number(box.top) - _finite_pdf_number(box.bottom)
            ) * user_unit
            if width_points <= 0 or height_points <= 0:
                raise ValueError("invalid PDF MediaBox")
            rendered_width = math.ceil(width_points * effective_scale)
            rendered_height = math.ceil(height_points * effective_scale)
            if eager_limit_detail is None and (
                rendered_width > limits.max_render_side_pixels
                or rendered_height > limits.max_render_side_pixels
            ):
                eager_limit_detail = (
                    f"pdf_render_side_limit_reached:page={page_number}"
                )
            page_pixels = rendered_width * rendered_height
            if (
                eager_limit_detail is None
                and page_pixels > limits.max_render_page_pixels
            ):
                eager_limit_detail = (
                    f"pdf_render_page_pixel_limit_reached:page={page_number}"
                )
            total_pixels += page_pixels
            if (
                eager_limit_detail is None
                and total_pixels > limits.max_render_total_pixels
            ):
                eager_limit_detail = (
                    f"pdf_render_total_pixel_limit_reached:page={page_number}"
                )
            if total_pixels > absolute_total_pixels:
                raise _PdfPreflightFailure(
                    DiagnosticCode.LIMIT_REACHED,
                    f"pdf_ingest_total_pixel_limit_reached:page={page_number}",
                )
        if page_count == 0:
            raise ValueError("PDF has no pages")
        return PdfRenderGeometryAssessment(
            total_pixels=total_pixels,
            eager_limit_detail=eager_limit_detail,
        )
    except _PdfPreflightFailure:
        raise
    except Exception as exc:
        raise _PdfPreflightFailure(
            DiagnosticCode.CORRUPT_SOURCE,
            f"pdf_geometry_preflight_failed:{type(exc).__name__}",
        ) from exc


def _pdf_render_geometry_preflight(
    path: Path,
    limits: _DoclingResourceLimits,
) -> None:
    """在不解码或栅格化页面的情况下，限制最坏页面渲染成本。"""

    assessment = _assess_pdf_ingest_geometry(path, limits)
    if assessment.eager_limit_detail is not None:
        raise _PdfPreflightFailure(
            DiagnosticCode.LIMIT_REACHED,
            assessment.eager_limit_detail,
        )


_CONVERSION_STATUSES = frozenset({
    "pending",
    "started",
    "failure",
    "success",
    "partial_success",
    "skipped",
})
_CONVERSION_ERROR_CATEGORIES = frozenset({
    "policy",
    "capacity",
    "source_unavailable",
    "target_unavailable",
    "timeout",
    "internal",
    "backend_failure",
    "inference_failure",
    "unknown",
})
_CONVERSION_COMPONENTS = frozenset({
    "document_backend",
    "model",
    "doc_assembler",
    "user_input",
    "pipeline",
})


def _safe_enum_value(value: Any, allowed: frozenset[str]) -> str:
    raw = value if isinstance(value, str) else getattr(value, "value", None)
    return raw if isinstance(raw, str) and raw in allowed else "unknown"


def _conversion_errors(value: Any) -> tuple[Any, ...]:
    # Docling 契约要求 list。拒绝自定义 iterable，避免意外结果对象在清理诊断时执行代码。
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(value)


def _safe_conversion_error_fields(error: Any) -> tuple[str, str, int | None]:
    category = _safe_enum_value(
        getattr(error, "category", None),
        _CONVERSION_ERROR_CATEGORIES,
    )
    component = _safe_enum_value(
        getattr(error, "component_type", None),
        _CONVERSION_COMPONENTS,
    )
    page_number = getattr(error, "page_no", None)
    if (
        isinstance(page_number, bool)
        or not isinstance(page_number, int)
        or page_number < 1
    ):
        page_number = None
        # 刻意不读取 error_message 或 module_name：二者都是自由文本，可能包含失败解析器/模型
        # 发出的源文本。
    return category, component, page_number


def _rejected_conversion_result(
    processor: ProcessorFingerprint,
    *,
    status: str,
    errors: tuple[Any, ...],
) -> ProcessingResult:
    safe_errors = tuple(_safe_conversion_error_fields(error) for error in errors)
    if len(safe_errors) <= 1:
        category, component, _page = safe_errors[0] if safe_errors else (
            "unknown",
            "unknown",
            None,
        )
        fields = f"category={category};component={component}"
    else:
        categories = ",".join(sorted({item[0] for item in safe_errors}))
        components = ",".join(sorted({item[1] for item in safe_errors}))
        fields = f"categories={categories};components={components}"
    code = (
        DiagnosticCode.LIMIT_REACHED
        if any(item[0] in {"policy", "capacity"} for item in safe_errors)
        else DiagnosticCode.CORRUPT_SOURCE
    )
    return ProcessingResult(
        elements=(),
        processor=processor,
        diagnostics=(ProcessingDiagnostic(
            code,
            detail=f"docling_status={status};{fields}",
        ),),
    )


def _partial_conversion_diagnostics(
    *,
    status: str,
    errors: tuple[Any, ...],
    physical_pages: tuple[int, ...],
) -> tuple[ProcessingDiagnostic, ...]:
    valid_pages = set(physical_pages)
    safe_by_page: dict[int | None, set[tuple[str, str]]] = {}
    for error in errors:
        category, component, page_number = _safe_conversion_error_fields(error)
        if page_number not in valid_pages:
            page_number = None
        safe_by_page.setdefault(page_number, set()).add((category, component))
    if not safe_by_page:
        safe_by_page[None] = {("unknown", "unknown")}

    diagnostics: list[ProcessingDiagnostic] = []
    for page_number in sorted(safe_by_page, key=lambda page: page or 0):
        records = sorted(safe_by_page[page_number])
        if len(records) == 1:
            category, component = records[0]
            fields = f"category={category};component={component}"
        else:
            categories = ",".join(sorted({record[0] for record in records}))
            components = ",".join(sorted({record[1] for record in records}))
            fields = f"categories={categories};components={components}"
        diagnostics.append(ProcessingDiagnostic(
            DiagnosticCode.PARSER_PARTIAL,
            DocumentLocator(page=page_number),
            detail=f"docling_status={status};{fields}",
        ))
    return tuple(diagnostics)


def read_with_docling(path: Path) -> ProcessingResult:
    """通过 Docling 将一个文档转换为中立元素。"""

    recipe = configured_docling_recipe()
    processor = fingerprint(recipe)
    limits = _resource_limits_from_recipe(recipe)
    try:
        source_size = path.stat().st_size
    except OSError as exc:
        return ProcessingResult(
            elements=(),
            processor=processor,
            diagnostics=(ProcessingDiagnostic(
                DiagnosticCode.CORRUPT_SOURCE, detail=type(exc).__name__
            ),),
        )
    if source_size > limits.max_file_size_bytes:
        return ProcessingResult(
            elements=(),
            processor=processor,
            diagnostics=(ProcessingDiagnostic(
                DiagnosticCode.LIMIT_REACHED,
                detail=(
                    f"source size {source_size} exceeds supported limit "
                    f"{limits.max_file_size_bytes}"
                ),
            ),),
        )
    if path.suffix.lower() == ".pdf":
        try:
            _pdf_render_geometry_preflight(path, limits)
        except _PdfPreflightFailure as exc:
            return ProcessingResult(
                elements=(),
                processor=processor,
                diagnostics=(ProcessingDiagnostic(
                    exc.code,
                    detail=exc.detail,
                ),),
            )
    try:
        source_key = fingerprint_file(path).sha256
        conversion = _convert_with_known_dependency_deprecations(
            _converter(recipe),
            path=path,
            limits=limits,
        )
        status = _safe_enum_value(
            getattr(conversion, "status", None),
            _CONVERSION_STATUSES,
        )
        conversion_errors = _conversion_errors(
            getattr(conversion, "errors", None)
        )
    except Exception as exc:
        # Docling 会抛出多种解析器错误。对调用方而言它们含义相同——无法读取此源；只保留
        # class 名作为细节，确保没有文档内容进入日志。
        code = (
            DiagnosticCode.LIMIT_REACHED
            if _is_docling_limit_error(exc)
            else DiagnosticCode.CORRUPT_SOURCE
        )
        return ProcessingResult(
            elements=(),
            processor=processor,
            diagnostics=(ProcessingDiagnostic(
                code, detail=type(exc).__name__
            ),),
        )

    if status not in {"success", "partial_success"}:
        return _rejected_conversion_result(
            processor,
            status=status,
            errors=conversion_errors,
        )

    document = getattr(conversion, "document", None)
    try:
        physical_pages = _physical_page_numbers(document)
    except ValueError:
        return ProcessingResult(
            elements=(),
            processor=processor,
            diagnostics=(ProcessingDiagnostic(
                DiagnosticCode.CORRUPT_SOURCE,
                detail="PhysicalPageInventoryUnavailable",
            ),),
        )
    page_heights = _page_heights(document, physical_pages)

    elements: list[DocumentElement] = []
    diagnostics: list[ProcessingDiagnostic] = list(
        _partial_conversion_diagnostics(
            status=status,
            errors=conversion_errors,
            physical_pages=physical_pages,
        )
        if status == "partial_success" or conversion_errors
        else ()
    )
    nontext_by_page: dict[int, list[DocumentNonTextUnit]] = {
        page: [] for page in physical_pages
    }
    section: list[str] = []
    ordinal = 0
    element_budget_exhausted = False

    for item, _tree_level in document.iterate_items():
        if len(elements) >= MAX_ELEMENTS:
            element_budget_exhausted = True
            break
        label = str(getattr(item, "label", ""))
        kind = _LABEL_TO_KIND.get(label, ElementKind.PARAGRAPH)
        text = str(getattr(item, "text", "") or "").strip()
        if label == "document_index" and not text:
            text = _document_index_markdown(item, document)
        source_pages = _source_pages_for(item)
        if not source_pages or not set(source_pages) <= set(physical_pages):
            diagnostics.append(ProcessingDiagnostic(
                DiagnosticCode.CORRUPT_SOURCE,
                detail=f"UnlocatedDocumentItem:{label or 'unknown'}",
            ))
            continue
        locator = _locator_for(
            item,
            ordinal=ordinal,
            section=section,
            page_heights=page_heights,
        )

        if kind is ElementKind.HEADING and text:
            # 追加前截断到 heading 自身深度。无条件追加会使每个 heading 都成为前一个的子项，
            # 从而让平面文档生成不断加深、但并不存在的层级路径。
            depth = max(1, int(getattr(item, "level", 1) or 1))
            section = [*section[: depth - 1], text]
            locator = _locator_for(
                item,
                ordinal=ordinal,
                section=section,
                page_heights=page_heights,
            )

        nontext_kind = _nontext_kind_for_label(label)
        if nontext_kind is not None:
            stored_text = text[:MAX_ELEMENT_CHARS]
            if len(text) > MAX_ELEMENT_CHARS:
                for page_number in source_pages:
                    diagnostics.append(ProcessingDiagnostic(
                        DiagnosticCode.LIMIT_REACHED,
                        _page_locator(locator, page_number),
                        detail=f"{label or 'document'} element character limit reached",
                    ))
            if stored_text:
                emitted_kind = (
                    ElementKind.CAPTION
                    if nontext_kind is DocumentNonTextKind.FIGURE
                    else kind
                )
                element = DocumentElement(
                    element_id=make_element_id(source_key, locator, stored_text),
                    kind=emitted_kind,
                    text=stored_text,
                    locator=locator,
                    source_pages=source_pages,
                )
                elements.append(element)
                text_element_ids = (element.element_id,)
            else:
                element = DocumentElement(
                    element_id=make_element_id(source_key, locator, None),
                    kind=ElementKind.IMAGE,
                    text=None,
                    locator=locator,
                    needs_vision=True,
                    source_pages=source_pages,
                )
                elements.append(element)
                text_element_ids = ()
            requires_visual = (
                nontext_kind is DocumentNonTextKind.FIGURE or not stored_text
            )
            unit = DocumentNonTextUnit(
                unit_id=_nontext_unit_id(
                    source_key,
                    nontext_kind,
                    source_pages,
                    ordinal,
                ),
                kind=nontext_kind,
                source_pages=source_pages,
                text_element_ids=text_element_ids,
                element_id=element.element_id,
                locator=locator,
                requires_visual_read=requires_visual,
            )
            for page_number in source_pages:
                nontext_by_page[page_number].append(unit)
            if requires_visual:
                for page_number in source_pages:
                    diagnostics.append(ProcessingDiagnostic(
                        DiagnosticCode.PAGE_NEEDS_VISION,
                        _page_locator(locator, page_number),
                        detail=f"{nontext_kind.value} requires visual interpretation",
                    ))
            ordinal += 1
            continue

        if not text:
            for page_number in source_pages:
                diagnostics.append(ProcessingDiagnostic(
                    DiagnosticCode.CORRUPT_SOURCE,
                    _page_locator(locator, page_number),
                    detail=f"EmptyDoclingItem:{label or 'unknown'}",
                ))
            continue
        stored_text = text[:MAX_ELEMENT_CHARS]
        if len(text) > MAX_ELEMENT_CHARS:
            for page_number in source_pages:
                diagnostics.append(ProcessingDiagnostic(
                    DiagnosticCode.LIMIT_REACHED,
                    _page_locator(locator, page_number),
                    detail="text element character limit reached",
                ))
        elements.append(DocumentElement(
            element_id=make_element_id(source_key, locator, stored_text),
            kind=kind,
            text=stored_text,
            locator=locator,
            source_pages=source_pages,
        ))
        ordinal += 1

    if element_budget_exhausted:
        for page_number in physical_pages:
            diagnostics.append(ProcessingDiagnostic(
                DiagnosticCode.LIMIT_REACHED,
                DocumentLocator(page=page_number),
                detail="element budget exhausted",
            ))

    text_ids_by_page: dict[int, list[str]] = {page: [] for page in physical_pages}
    for element in elements:
        if not (element.text or "").strip():
            continue
        for page_number in element.source_pages:
            text_ids_by_page[page_number].append(element.element_id)
    diagnostics_by_page: dict[int, list[ProcessingDiagnostic]] = {
        page: [] for page in physical_pages
    }
    for diagnostic in diagnostics:
        if diagnostic.locator.page in diagnostics_by_page:
            diagnostics_by_page[int(diagnostic.locator.page)].append(diagnostic)

    page_records: list[DocumentPageRecord] = []
    for page_number in physical_pages:
        text_ids = tuple(text_ids_by_page[page_number])
        nontext_units = tuple(nontext_by_page[page_number])
        page_diagnostics = diagnostics_by_page[page_number]
        if not text_ids and not nontext_units and not any(
            diagnostic.code in {DiagnosticCode.CORRUPT_SOURCE, DiagnosticCode.LIMIT_REACHED}
            for diagnostic in page_diagnostics
        ):
            empty = ProcessingDiagnostic(
                DiagnosticCode.PAGE_EMPTY,
                DocumentLocator(page=page_number),
            )
            diagnostics.append(empty)
            page_diagnostics.append(empty)
        if any(
            diagnostic.code in {DiagnosticCode.CORRUPT_SOURCE, DiagnosticCode.LIMIT_REACHED}
            for diagnostic in page_diagnostics
        ):
            state = DocumentPageState.UNREADABLE
        elif text_ids and nontext_units:
            state = DocumentPageState.MIXED
        elif text_ids:
            state = DocumentPageState.TEXT
        elif nontext_units:
            state = DocumentPageState.VISUAL_ONLY
        else:
            state = DocumentPageState.NO_EXTRACTABLE_CONTENT
        page_records.append(DocumentPageRecord(
            page_number=page_number,
            state=state,
            text_element_ids=text_ids,
            nontext_units=nontext_units,
            diagnostics=tuple(page_diagnostics),
        ))

    page_manifest = DocumentPageManifest(
        physical_page_count=len(physical_pages),
        inventory_status=DocumentPageInventoryStatus.COMPLETE,
        detector_fingerprint=f"{processor}:physical-pages-v1",
        detector_capabilities=tuple(sorted({
            "figure_inventory",
            "formula_inventory",
            "layout_element_inventory",
            "nontext_unit_inventory",
            "physical_page_inventory",
            "table_inventory",
            "text_element_source_pages",
            "typed_page_diagnostics",
        })),
        pages=tuple(page_records),
    )
    return ProcessingResult(
        elements=tuple(elements),
        processor=processor,
        diagnostics=tuple(diagnostics),
        page_manifest=page_manifest,
    )


def _convert_with_known_dependency_deprecations(
    converter: Any,
    *,
    path: Path,
    limits: _DoclingResourceLimits,
) -> Any:
    """隔离两个上游弃用项，同时不隐藏 reader 警告。

    Docling 2.119 会读取自身已弃用的 ``generate_table_images`` 选项，而其 Torch 依赖会导入
    ``torch.jit.script_method``。正确以 warnings-as-errors 运行测试的项目，否则会把可读文档
    变成类型化损坏结果。两个过滤器都绑定到发出警告的依赖模块和精确消息；包括未来 Docling
    弃用在内的其他警告仍会跨越正常失败边界，必须主动评审。
    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"This field is deprecated\. Use `generate_page_images=True` "
                r"and call `TableItem\.get_image\(\)` to extract table images "
                r"from page images\."
            ),
            category=DeprecationWarning,
            module=r"docling\.pipeline\.standard_pdf_pipeline",
        )
        warnings.filterwarnings(
            "ignore",
            message=(
                r"`torch\.jit\.script_method` is deprecated\. Please switch "
                r"to `torch\.compile` or `torch\.export`\."
            ),
            category=DeprecationWarning,
            module=r"torch\.jit\._script",
        )
        return converter.convert(
            str(path),
            max_num_pages=limits.max_num_pages,
            max_file_size=limits.max_file_size_bytes,
            raises_on_error=False,
        )


def _is_docling_limit_error(error: BaseException) -> bool:
    """跨包装异常识别 Docling 的策略超限失败。"""

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        rendered = f"{type(current).__name__}:{current}".lower()
        if any(marker in rendered for marker in (
            "max_num_pages",
            "max_file_size",
            "maximum number of pages",
            "maximum file size",
        )):
            return True
        current = current.__cause__ or current.__context__
    return False


def _physical_page_numbers(document: Any) -> tuple[int, ...]:
    pages = getattr(document, "pages", None)
    if not isinstance(pages, Mapping) or not pages:
        raise ValueError("physical page inventory unavailable")
    numbers: list[int] = []
    for key, page in pages.items():
        page_number = getattr(page, "page_no", key)
        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number < 1
            or key != page_number
        ):
            raise ValueError("physical page identity is not exact")
        numbers.append(page_number)
    ordered = tuple(sorted(numbers))
    if ordered != tuple(range(1, len(ordered) + 1)):
        raise ValueError("physical page inventory is not continuous")
    return ordered


def _page_heights(
    document: Any,
    physical_pages: tuple[int, ...],
) -> dict[int, float]:
    """只返回精确的正 Docling 页面高度。

    当前 Docling ``PageItem`` 对象要求 ``size``，且每个 ``BoundingBox`` 都携带坐标原点。
    保持此提取器的防御性，可让旧版或格式错误的转换对象保留文本，同时拒绝为左下原点 box
    编造几何信息。
    """

    pages = getattr(document, "pages", None)
    if not isinstance(pages, Mapping):
        return {}
    heights: dict[int, float] = {}
    for page_number in physical_pages:
        size = getattr(pages.get(page_number), "size", None)
        raw_height = getattr(size, "height", None)
        if isinstance(raw_height, bool):
            continue
        try:
            height = float(raw_height)
        except (TypeError, ValueError):
            continue
        if math.isfinite(height) and height > 0:
            heights[page_number] = height
    return heights


def _source_pages_for(item: Any) -> tuple[int, ...]:
    provenance = getattr(item, "prov", None) or []
    pages: set[int] = set()
    for record in provenance:
        page_number = getattr(record, "page_no", None)
        if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
            return ()
        pages.add(page_number)
    return tuple(sorted(pages))


def _nontext_kind_for_label(label: str) -> DocumentNonTextKind | None:
    return {
        "picture": DocumentNonTextKind.FIGURE,
        "formula": DocumentNonTextKind.FORMULA,
        "table": DocumentNonTextKind.TABLE,
        "document_index": DocumentNonTextKind.TABLE,
    }.get(label)


def _document_index_markdown(item: Any, document: Any) -> str:
    """对 Docling 表格后端索引项执行尽力文本投影。

    ``document_index`` 是 ``TableItem``，其有用内容位于 ``TableData`` 而非 ``item.text``。
    只有结构有界且至少有一个真实单元格的表格会被序列化。任何项局部失败都返回空字符串，
    使现有 TABLE visual-gap 路径保持权威，而不是拒绝其他方面可读的文档。
    """

    data = getattr(item, "data", None)
    cells = getattr(data, "table_cells", None)
    if not isinstance(cells, (list, tuple)) or len(cells) > MAX_ELEMENTS:
        return ""
    has_real_cell_text = False
    cell_text_chars = 0
    for cell in cells:
        cell_text = getattr(cell, "text", None)
        if not isinstance(cell_text, str):
            continue
        cell_text_chars += len(cell_text)
        if cell_text_chars > MAX_ELEMENT_CHARS:
            return ""
        if cell_text.strip():
            has_real_cell_text = True
    if not has_real_cell_text:
        return ""

    rows = getattr(data, "num_rows", None)
    columns = getattr(data, "num_cols", None)
    if (
        isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows < 1
        or isinstance(columns, bool)
        or not isinstance(columns, int)
        or columns < 1
        or rows * columns > MAX_ELEMENTS
    ):
        return ""

    exporter = getattr(item, "export_to_markdown", None)
    if not callable(exporter):
        return ""
    try:
        markdown = exporter(doc=document)
    except Exception:
        return ""
    if not isinstance(markdown, str) or len(markdown) > MAX_ELEMENT_CHARS:
        return ""
    return markdown.strip()


def _nontext_unit_id(
    source_key: str,
    kind: DocumentNonTextKind,
    source_pages: tuple[int, ...],
    ordinal: int,
) -> str:
    digest = hashlib.sha256(
        repr((source_key, kind.value, source_pages, ordinal)).encode("utf-8")
    ).hexdigest()
    return f"ntu_{digest[:24]}"


def _page_locator(locator: DocumentLocator, page_number: int) -> DocumentLocator:
    return DocumentLocator(
        page=page_number,
        ordinal=locator.ordinal,
        section_path=locator.section_path,
        bbox=locator.bbox if locator.page == page_number else None,
        char_range=locator.char_range if locator.page == page_number else None,
    )


def _locator_for(
    item: Any,
    *,
    ordinal: int,
    section: list[str],
    page_heights: Mapping[int, float],
) -> DocumentLocator:
    provenance = getattr(item, "prov", None) or []
    first = provenance[0] if provenance else None
    page_number = getattr(first, "page_no", None) if first else None
    return DocumentLocator(
        page=page_number,
        ordinal=ordinal,
        section_path=tuple(section),
        bbox=_bbox_of(first, page_height=page_heights.get(page_number)),
        char_range=_charspan_of(first),
    )


def _bbox_of(
    provenance: Any,
    *,
    page_height: float | None,
) -> tuple[float, float, float, float] | None:
    """把具有显式原点的 Docling bbox 规范化为左上原点几何。

    ``DocumentLocator`` 与视觉 cropper 使用左上原点的 ``(left, top, right, bottom)``。
    Docling 可输出任一原点。因此左下原点 box 需要物理页面高度；如果高度或原点任一不可用，
    丢弃 bbox 比裁剪页面的镜像部分更安全。视觉 Runtime 随后回退到完整页面。
    """

    box = getattr(provenance, "bbox", None) if provenance else None
    if box is None:
        return None
    raw_origin = getattr(box, "coord_origin", None)
    origin = (
        raw_origin
        if isinstance(raw_origin, str)
        else getattr(raw_origin, "value", None)
    )
    if origin not in {"TOPLEFT", "BOTTOMLEFT"}:
        return None
    try:
        left, right = float(box.l), float(box.r)
        first_y, second_y = float(box.t), float(box.b)
    except (AttributeError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (left, right, first_y, second_y)):
        return None
    if right < left:
        left, right = right, left
    if origin == "TOPLEFT":
        top, bottom = sorted((first_y, second_y))
        return (left, top, right, bottom)

    if (
        page_height is None
        or isinstance(page_height, bool)
        or not math.isfinite(page_height)
        or page_height <= 0
    ):
        return None
    lower, upper = sorted((first_y, second_y))
    top, bottom = page_height - upper, page_height - lower
    return (left, top, right, bottom)


def _charspan_of(provenance: Any) -> tuple[int, int] | None:
    span = getattr(provenance, "charspan", None) if provenance else None
    if span is None:
        return None
    try:
        start, end = int(span[0]), int(span[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (start, end) if 0 <= start <= end else None
