"""在一个冻结工作区内执行有界且格式专属的观测。

这些工具暴露文档读取器，但不会把模型提供的路径变成环境文件系统权威。Host 会冻结工作区根目录，
所有路径（包括符号链接）都在该根目录下解析，且结果始终说明已观测多少内容以及如何继续有界读取。

本模块特意不向 Runtime 注册工具。它是注册构建器，使组合根可以在一个目录快照中精确纳入此界面。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ...input_processing.documents.contracts import (
    DiagnosticCode,
    DocumentElement,
    DocumentLocator,
    DocumentNonTextKind,
    ProcessingDiagnostic,
    ProcessingResult,
)
from ...input_processing.documents.readers import (
    UnsupportedDocumentFormat,
    UnsupportedLegacyOfficeFormat,
    read_document,
)
from ...input_processing.documents.readers.image import read_image
from ...input_processing.documents.readers.plain_text import read_plain_text
from ...input_processing.vision.contracts import (
    PixelSize,
    VisionDetail,
    VisionPurpose,
    VisionRegion,
    VisionRequest,
)
from ...input_processing.vision.imaging import PayloadRefusal, prepare_payload
from ...workspace.discovery import DEFAULT_TIMEOUT_S
from ...workspace.files.access import AuthorizedFileSource, FileAccessError
from ...runtime.model_calls.vision import (
    MountedVisualCallLedgerError,
    MountedVisualCallWaitingExternal,
)
from ..contracts import ToolSourceDescriptor, ToolSourceKind, ToolSpec
from ..effects import (
    DataEgress,
    EffectAction,
    EffectDescriptor,
    EffectResource,
    EffectScopeKind,
    Idempotency,
    Reversibility,
    ToolEffectProfile,
)
from ..execution import ToolBusinessFailure
from ..execution_context import current_tool_execution
from .format_observation_source_authority import (
    MAX_PATH_CHARS,
    _assert_unchanged,
    _guard_workspace_handler,
    _resolve_file,
    _source_identity,
)
from ..registration import CancellationMode, ToolExecutionProfile, ToolRegistration
from ..visual.visual_tool_boundary import (
    FrozenVisualToolBoundary,
    VisualUnitRef,
)
from ..visual.visual_observation_service import (
    MAX_UNITS_PER_CALL,
    VisualObservationRequest,
    VisualObservationService,
)
from ..visual.question_contract import (
    parse_visual_question,
    visual_call_identity,
    visual_question_constraint,
    visual_question_schema,
)
from ..visual.project_observation_publication import (
    VisualObservationPublisher, VisualPublicationIndexFailure,
)
from ..visual.failure_reporting import failure_diagnostics_schema
from ..workspace.workspace_tools import FrozenWorkspaceToolBoundary


FORMAT_OBSERVATION_TOOL_CONTRACT_VERSION = "format-observation-v1"
FORMAT_OBSERVATION_TOOL_IMPLEMENTATION_VERSION = "2"
VISUAL_OBSERVATION_TOOL_CONTRACT_VERSION = "visual-observation-v2"
VISUAL_OBSERVATION_TOOL_IMPLEMENTATION_VERSION = "3"
FORMAT_OBSERVATION_LOCAL_TOOL_IDS = (
    "read_text",
    "read_pdf_text",
    "read_word",
    "read_slides",
    "inspect_image",
)
EXTERNAL_VISUAL_ANALYSIS_TOOL_IDS = (
    "analyze_image",
    "analyze_pdf_page",
)

DEFAULT_MAX_CHARS = 6_000
MAX_CHARS = 8_000
DEFAULT_MAX_ELEMENTS = 12
MAX_ELEMENTS = 16
MAX_CURSOR_OFFSET = 1_000_000
# ToolExecutor 会拒绝整个超过 90,000 字节的调用。读取器文本已有游标边界，但来源元数据没有：
# 一个有效 DOCX 标题可能长达 20,000 个字符，并被复制到后续每个元素的 ``section_path`` 和
# ``display`` 中。此处保留第二个更小的投影预算，使处理器能够如实返回部分结果，
# 而不是在执行器边界丢失整个结果。
MAX_PROJECTED_OUTPUT_BYTES = 80_000
MAX_LOCATOR_SECTION_SEGMENTS = 8
MAX_LOCATOR_SECTION_BYTES = 1_024
MAX_LOCATOR_DISPLAY_BYTES = 1_280
MAX_PROJECTED_SELECTED_PAGES = 256
MAX_DIAGNOSTIC_ITEMS = 16
MAX_DIAGNOSTIC_CODE_BYTES = 128
MAX_DIAGNOSTIC_FIELD_BYTES = 768
MAX_OBSERVATION_TEXT_BYTES = 4_096

TEXT_SUFFIXES = frozenset(
    {
        ".txt",
        ".md",
        ".markdown",
        ".csv",
        ".tsv",
        ".json",
        ".log",
        ".yaml",
        ".yml",
        ".py",
        ".js",
        ".ts",
        ".html",
        ".css",
        ".sql",
        ".sh",
        ".toml",
        ".xml",
        ".ini",
        ".rs",
        ".go",
        ".java",
        ".c",
        ".h",
        ".cpp",
        ".rb",
    }
)
PDF_SUFFIXES = frozenset({".pdf"})
WORD_SUFFIXES = frozenset({".docx", ".doc"})
SLIDE_SUFFIXES = frozenset({".pptx", ".ppt"})
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


def build_format_observation_tool_registrations(
    boundary: FrozenWorkspaceToolBoundary,
) -> tuple[ToolRegistration, ...]:
    """为一个冻结工作区构建五个独立本地读取工具。

    本地检查不依赖模型或人工批准。外部语义分析由
    :func:`build_external_visual_analysis_tool_registrations` 单独暴露，
    使其 NETWORK TRANSMIT 效果无法隐藏在可选参数之后。
    """

    if not isinstance(boundary, FrozenWorkspaceToolBoundary):
        raise TypeError("boundary must be a FrozenWorkspaceToolBoundary")
    source = ToolSourceDescriptor(
        kind=ToolSourceKind.LOCAL,
        source_id="personagraph.input_processing.format_observation",
    )
    execution = build_format_observation_execution_profile(
        timeout_s=boundary.timeout_s,
    )
    effect = _read_effect(boundary)
    specs = {spec.tool_id: spec for spec in build_format_observation_tool_specs()}

    def registration(
        tool_id: str,
        handler: Callable[[dict[str, Any]], dict[str, Any]],
        effect_profile: ToolEffectProfile = effect,
    ) -> ToolRegistration:
        spec = specs[tool_id]
        implementation_version = (
            VISUAL_OBSERVATION_TOOL_IMPLEMENTATION_VERSION
            if tool_id == "inspect_image"
            else FORMAT_OBSERVATION_TOOL_IMPLEMENTATION_VERSION
        )
        return ToolRegistration(
            spec=spec,
            implementation_version=implementation_version,
            source=source,
            handler=_guard_workspace_handler(boundary, handler),
            effect_profile=effect_profile,
            execution_profile=execution,
        )

    return (
        registration(
            "read_text",
            lambda payload: _read_text(boundary, payload),
        ),
        registration(
            "read_pdf_text",
            lambda payload: _read_pdf_text(boundary, payload),
        ),
        registration(
            "read_word",
            lambda payload: _read_word(boundary, payload),
        ),
        registration(
            "read_slides",
            lambda payload: _read_slides(boundary, payload),
        ),
        registration(
            "inspect_image",
            lambda payload: _inspect_image(boundary, payload),
        ),
    )


def build_format_observation_tool_specs() -> tuple[ToolSpec, ...]:
    """Return the five local observation contracts in canonical order."""

    def spec(
        *,
        tool_id: str,
        contract_version: str,
        name: str,
        description: str,
        input_schema: dict[str, Any],
    ) -> ToolSpec:
        return ToolSpec(
            tool_id=tool_id,
            contract_version=contract_version,
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=_output_schema(),
            catalog_tags=("document", "file", "read"),
        )

    return (
        spec(
            tool_id="read_text",
            contract_version=FORMAT_OBSERVATION_TOOL_CONTRACT_VERSION,
            name="Read a text file",
            description=(
                "按工作区相对路径读取有界 UTF-8 文本或 Markdown；截断后使用 "
                "next_cursor 续读。path 应取 file_catalog.relative_path 或目录工具"
                "返回的 path，不能仅把显示字段 name 当作路径。"
            ),
            input_schema=_text_input_schema(),
        ),
        spec(
            tool_id="read_pdf_text",
            contract_version=FORMAT_OBSERVATION_TOOL_CONTRACT_VERSION,
            name="Read text from PDF pages",
            description=(
                "按工作区相对路径从选定的 1-based PDF 页码范围提取有界原生文本；"
                "仅视觉内容会保留为明确的覆盖缺口。path 应取 "
                "file_catalog.relative_path 或目录工具返回的 path，不能仅把显示字段 "
                "name 当作路径。"
            ),
            input_schema=_paged_text_input_schema("page"),
        ),
        spec(
            tool_id="read_word",
            contract_version=FORMAT_OBSERVATION_TOOL_CONTRACT_VERSION,
            name="Read a Word document",
            description=(
                "按工作区相对路径读取 DOCX 的有界文本和结构定位；旧 DOC 仅在配置"
                "沙箱转换桥时可读。path 应取 file_catalog.relative_path 或目录工具"
                "返回的 path，不能仅把显示字段 name 当作路径。"
            ),
            input_schema=_text_input_schema(),
        ),
        spec(
            tool_id="read_slides",
            contract_version=FORMAT_OBSERVATION_TOOL_CONTRACT_VERSION,
            name="Read presentation slides",
            description=(
                "按工作区相对路径读取所选 PPTX 幻灯片范围内的有界正文与备注；"
                "未配置显式后端时，幻灯片渲染会报告不可用。path 应取 "
                "file_catalog.relative_path 或目录工具返回的 path，不能仅把显示字段 "
                "name 当作路径。"
            ),
            input_schema=_slides_input_schema(),
        ),
        spec(
            tool_id="inspect_image",
            contract_version=VISUAL_OBSERVATION_TOOL_CONTRACT_VERSION,
            name="Inspect a PNG or JPEG",
            description=(
                "按工作区相对路径读取有界本地 OCR 证据和安全图片元数据。原始像素"
                "不会进入普通模型 ToolResult；语义视觉理解使用 analyze_image。"
                "path 应取 file_catalog.relative_path 或目录工具返回的 path，不能仅把"
                "显示字段 name 当作路径。"
            ),
            input_schema=_image_input_schema(),
        ),
    )


def build_format_observation_execution_profile(
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ToolExecutionProfile:
    """Return the shared execution contract for local format observation."""

    return ToolExecutionProfile(
        default_timeout_s=max(30.0, timeout_s),
        hard_timeout_s=max(120.0, timeout_s * 2),
        # WorkRun retains prior complete ToolResults below 128 KiB. Keep the
        # registration below that ceiling so observations are not silently lost.
        max_output_bytes=90_000,
        max_transparent_retries=0,
        concurrency_class="format_observation",
    )


def build_external_visual_analysis_tool_specs() -> tuple[ToolSpec, ToolSpec]:
    """Return the two provider-neutral visual-analysis contracts."""

    def spec(
        *,
        tool_id: str,
        name: str,
        description: str,
        input_schema: dict[str, Any],
    ) -> ToolSpec:
        return ToolSpec(
            tool_id=tool_id,
            contract_version=VISUAL_OBSERVATION_TOOL_CONTRACT_VERSION,
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=_output_schema(),
            catalog_tags=("document", "file", "read", "fetch"),
        )

    return (
        spec(
            tool_id="analyze_image",
            name="理解一张 PNG 或 JPEG",
            description=(
                "将工作区相对路径指定的一张 PNG/JPEG 交给配置的视觉模型读取内容，"
                "指定用途、清晰度和观察范围。可读文件默认允许外发，无需另外请求用户批准；"
                "purpose=question 时必须在 question 中写出具体自然语言问题；"
                "其它用途省略 question 或设为 null。"
                "执行前仍检查文件可读性和版本。path 应取 "
                "file_catalog.relative_path 或目录工具返回的 path，不能仅把显示字段 "
                "name 当作路径。"
            ),
            input_schema=_analyze_image_input_schema(),
        ),
        spec(
            tool_id="analyze_pdf_page",
            name="理解指定 PDF 页面",
            description=(
                "按工作区相对路径渲染指定 PDF 的页面，逐页交给配置的视觉模型读取"
                "页面整体内容及各区域关系。可在同一 Attempt 中多次调用，"
                "分别选择不同页面或问题；多个调用按 calls 顺序串行执行。"
                f"在 pages 中合并本次要看的 1..{MAX_UNITS_PER_CALL} 个不重复的一基页码，"
                "按给定顺序返回每页观察或失败原因。无需先运行 prepare_files；"
                "单页索引失败不丢弃其他页的成功观察；失败页会保留具体失败码，"
                "视觉回执已保存不等于索引已成功，也不保证自动恢复。"
                "只按需读取指定页面，不会遍历页内已拆分的 visual_unit。"
                "所选页面共用用途、问题、清晰度和观察范围。可读文件默认允许外发，无需另外请求用户批准；"
                "purpose=question 时必须在 question 中写出具体自然语言问题；"
                "其它用途省略 question 或设为 null。"
                "执行前仍检查文件可读性和版本。path 应取 "
                "file_catalog.relative_path 或目录工具返回的 path，不能仅把显示字段 "
                "name 当作路径。"
            ),
            input_schema=_analyze_pdf_page_input_schema(),
        ),
    )


def build_external_visual_analysis_execution_profile(tool_id: str) -> ToolExecutionProfile:
    """Return the provider-neutral execution contract for visual analysis.

    Provider timeouts and retry behavior belong to the adapter/physical-call
    binding.  The Tool execution envelope is deliberately stable across
    workspace boundaries and provider selections.
    """

    if tool_id not in EXTERNAL_VISUAL_ANALYSIS_TOOL_IDS:
        raise ValueError("unknown external visual analysis tool")
    # 单个物理视觉请求仍沿用 120 秒；多页工具按最大页数给串行请求留出总窗口，
    # 不修改 provider 自身时限，也不影响图像工具及检索时限。
    timeout_s = 120.0 * (MAX_UNITS_PER_CALL if tool_id == "analyze_pdf_page" else 1)
    return ToolExecutionProfile(
        default_timeout_s=timeout_s,
        hard_timeout_s=timeout_s,
        max_output_bytes=90_000,
        max_transparent_retries=0,
        concurrency_class="external_visual_analysis",
        cancellation_mode=CancellationMode.COOPERATIVE,
    )


def build_external_visual_analysis_tool_registrations(
    boundary: FrozenWorkspaceToolBoundary,
    *,
    vision_adapter=None,
    resolve_visual_file: Callable[[str], AuthorizedFileSource] | None = None,
    visual_publisher: VisualObservationPublisher | None = None,
) -> tuple[ToolRegistration, ToolRegistration]:
    """构建两个显式外发的视觉分析工具。

    每次逻辑调用选择一张图像或一个有界 PDF 页码批次。这些注册始终声明 FILESYSTEM READ
    加 NETWORK TRANSMIT；提供商当前是否可用绝不会削弱冻结效果契约。
    """

    if not isinstance(boundary, FrozenWorkspaceToolBoundary):
        raise TypeError("boundary must be a FrozenWorkspaceToolBoundary")
    source = ToolSourceDescriptor(
        kind=ToolSourceKind.LOCAL,
        source_id="personagraph.input_processing.external_visual_analysis",
    )
    observation_service = VisualObservationService(
        adapter=vision_adapter,
    )
    effect = _external_vision_effect(boundary)
    specs = {spec.tool_id: spec for spec in build_external_visual_analysis_tool_specs()}

    def registration(
        tool_id: str,
        handler: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> ToolRegistration:
        return ToolRegistration(
            spec=specs[tool_id],
            # Provider/capability identity is contextual binding state, not a
            # revision of this stable handler contract.
            implementation_version=VISUAL_OBSERVATION_TOOL_IMPLEMENTATION_VERSION,
            source=source,
            handler=_guard_workspace_handler(boundary, handler),
            effect_profile=effect,
            execution_profile=build_external_visual_analysis_execution_profile(tool_id),
        )

    return (
        registration(
            "analyze_image",
            lambda payload: _analyze_image(
                boundary,
                payload,
                observation_service=observation_service,
                resolve_visual_file=resolve_visual_file,
                visual_publisher=visual_publisher,
            ),
        ),
        registration(
            "analyze_pdf_page",
            lambda payload: _analyze_pdf_page(
                boundary,
                payload,
                observation_service=observation_service,
                resolve_visual_file=resolve_visual_file,
                visual_publisher=visual_publisher,
            ),
        ),
    )


def _read_effect(boundary: FrozenWorkspaceToolBoundary) -> ToolEffectProfile:
    return ToolEffectProfile(
        (
            EffectDescriptor(
                resource=EffectResource.FILESYSTEM,
                action=EffectAction.READ,
                scope_kind=EffectScopeKind.WORKSPACE,
                default_scope=str(boundary.root),
                resource_argument="path",
                data_egress=DataEgress.CONTENT,
                idempotency=Idempotency.IDEMPOTENT,
                reversibility=Reversibility.REVERSIBLE,
            ),
        )
    )


def _external_vision_effect(
    boundary: FrozenWorkspaceToolBoundary,
) -> ToolEffectProfile:
    effects = list(_read_effect(boundary).effects)
    effects.append(
        EffectDescriptor(
            resource=EffectResource.NETWORK,
            action=EffectAction.TRANSMIT,
            scope_kind=EffectScopeKind.SESSION,
            default_scope=boundary.session_id,
            data_egress=DataEgress.CONTENT,
            # 两次发送相同像素仍属于两次披露，且首次披露无法撤销。持久防重复发送处理属于
            # 受保护派发器，而非乐观的效果声明。
            idempotency=Idempotency.NOT_IDEMPOTENT,
            reversibility=Reversibility.IRREVERSIBLE,
        )
    )
    effects.append(
        EffectDescriptor(
            resource=EffectResource.RUNTIME_STATE,
            action=EffectAction.UPDATE,
            scope_kind=EffectScopeKind.SESSION,
            default_scope=boundary.session_id,
            idempotency=Idempotency.IDEMPOTENT,
            reversibility=Reversibility.REVERSIBLE,
        )
    )
    return ToolEffectProfile(tuple(effects))


def _read_text(
    boundary: FrozenWorkspaceToolBoundary, payload: Mapping[str, Any]
) -> dict[str, Any]:
    target, relative = _resolve_file(boundary, payload, TEXT_SUFFIXES)
    source = _source_identity(target)
    result = read_plain_text(target)
    _assert_unchanged(target, source)
    return _document_view(
        result,
        format_name="text",
        relative_path=relative,
        source=source,
        payload=payload,
        selected_pages=(),
        physical_pages=None,
    )


def _read_pdf_text(
    boundary: FrozenWorkspaceToolBoundary, payload: Mapping[str, Any]
) -> dict[str, Any]:
    target, relative = _resolve_file(boundary, payload, PDF_SUFFIXES)
    source = _source_identity(target)
    result = read_document(target, engine="native")
    _assert_unchanged(target, source)
    physical_pages = _physical_pages(result)
    selected_pages = (
        _page_selection(
            payload,
            physical_pages=physical_pages,
            start_key="start_page",
            end_key="end_page",
        )
        if physical_pages is not None
        else ()
    )
    return _document_view(
        result,
        format_name="pdf",
        relative_path=relative,
        source=source,
        payload=payload,
        selected_pages=selected_pages,
        physical_pages=physical_pages,
    )


def _read_word(
    boundary: FrozenWorkspaceToolBoundary, payload: Mapping[str, Any]
) -> dict[str, Any]:
    target, relative = _resolve_file(boundary, payload, WORD_SUFFIXES)
    source = _source_identity(target)
    try:
        result = read_document(target, engine="native")
    except UnsupportedLegacyOfficeFormat:
        return _legacy_office_gap(
            format_name="word", relative_path=relative, source=source
        )
    except UnsupportedDocumentFormat as exc:  # 防御性桥接契约守卫。
        raise ToolBusinessFailure(
            "format_unavailable",
            "No Word reader is configured for this source.",
            {"actual_suffix": exc.suffix},
        ) from exc
    _assert_unchanged(target, source)
    return _document_view(
        result,
        format_name="word",
        relative_path=relative,
        source=source,
        payload=payload,
        selected_pages=(),
        physical_pages=None,
    )


def _read_slides(
    boundary: FrozenWorkspaceToolBoundary, payload: Mapping[str, Any]
) -> dict[str, Any]:
    target, relative = _resolve_file(boundary, payload, SLIDE_SUFFIXES)
    source = _source_identity(target)
    try:
        result = read_document(target, engine="native")
    except UnsupportedLegacyOfficeFormat:
        return _legacy_office_gap(
            format_name="slides", relative_path=relative, source=source
        )
    except UnsupportedDocumentFormat as exc:  # 防御性桥接契约守卫。
        raise ToolBusinessFailure(
            "format_unavailable",
            "No slide reader is configured for this source.",
            {"actual_suffix": exc.suffix},
        ) from exc
    _assert_unchanged(target, source)
    physical_pages = _physical_pages(result)
    selected_pages = (
        _page_selection(
            payload,
            physical_pages=physical_pages,
            start_key="start_slide",
            end_key="end_slide",
        )
        if physical_pages is not None
        else ()
    )
    render_visuals = _boolean(payload.get("render_visuals"), False, "render_visuals")
    extra_diagnostics: tuple[dict[str, Any], ...] = ()
    if render_visuals:
        rendering = {
            "requested": True,
            "status": "unavailable",
            "reason_code": "slide_render_backend_not_configured",
        }
        extra_diagnostics = (
            {
                "code": "slide_render_backend_unavailable",
                "at": "?",
                "detail": (
                    "No explicit sandboxed PPT/PPTX render backend is configured."
                ),
                "retry_hint": "Read slide text now; use source assets or configure a renderer.",
            },
        )
    else:
        rendering = {
            "requested": False,
            "status": "not_requested",
            "reason_code": None,
        }
    return _document_view(
        result,
        format_name="slides",
        relative_path=relative,
        source=source,
        payload=payload,
        selected_pages=selected_pages,
        physical_pages=physical_pages,
        extra_diagnostics=extra_diagnostics,
        rendering=rendering,
    )


def _inspect_image(
    boundary: FrozenWorkspaceToolBoundary,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    target, relative = _resolve_file(boundary, payload, IMAGE_SUFFIXES)
    source = _source_identity(target)
    detail = _vision_detail(payload.get("detail"))
    result = read_image(target)
    images: tuple[dict[str, Any], ...] = ()
    extra_diagnostics: tuple[dict[str, Any], ...] = ()
    prepared = prepare_payload(
        _vision_request(
            target,
            source,
            page=1,
            detail=detail,
            mime_type=(
                "image/png" if target.suffix.lower() == ".png" else "image/jpeg"
            ),
        )
    )
    if isinstance(prepared, PayloadRefusal):
        extra_diagnostics = (_payload_diagnostic(prepared, at="p1"),)
        rendering = {
            "requested": True,
            "status": "unavailable",
            "reason_code": prepared.failure.value,
        }
    else:
        images = (_image_view(prepared, page=1, detail=detail),)
        rendering = {
            "requested": True,
            "status": "available",
            "reason_code": None,
        }
    _assert_unchanged(target, source)
    return _document_view(
        result,
        format_name="image",
        relative_path=relative,
        source=source,
        payload=payload,
        selected_pages=(1,),
        physical_pages=1,
        images=images,
        extra_diagnostics=extra_diagnostics,
        rendering=rendering,
        analysis=_analysis_summary(
            requested=False,
            purpose=None,
            detail=None,
            region=None,
            observations=(),
        ),
    )


def _analyze_image(
    boundary: FrozenWorkspaceToolBoundary,
    payload: Mapping[str, Any],
    *,
    observation_service: VisualObservationService,
    resolve_visual_file: Callable[[str], AuthorizedFileSource] | None = None,
    visual_publisher: VisualObservationPublisher | None = None,
) -> dict[str, Any]:
    target, relative = _resolve_file(boundary, payload, IMAGE_SUFFIXES)
    source = _source_identity(target)
    purpose, detail, region, question = _required_analysis_options(
        payload,
        allowed_regions=tuple(VisionRegion),
    )
    publication_binding = _visual_publication_binding(
        purpose=purpose,
        target=target,
        source=source,
        relative=relative,
        resolve_visual_file=resolve_visual_file,
        visual_publisher=visual_publisher,
    )
    result = read_image(target)
    observation = _analyze_visual_source(
        boundary,
        target=target,
        source=source,
        page=1,
        pixel_size=_image_pixel_size(target),
        purpose=purpose,
        detail=detail,
        region=region,
        question=question,
        observation_service=observation_service,
        publication_binding=publication_binding,
        visual_publisher=visual_publisher,
    )
    observations = (observation,)
    semantic_diagnostics = _semantic_gap_diagnostics(observations)
    resolved = observation["status"] == "completed"
    _assert_unchanged(target, source)
    return _document_view(
        result,
        format_name="image",
        relative_path=relative,
        source=source,
        payload=payload,
        selected_pages=(1,),
        physical_pages=1,
        extra_diagnostics=semantic_diagnostics,
        rendering={
            "requested": False,
            "status": "not_requested",
            "reason_code": None,
        },
        analysis=_analysis_summary(
            requested=True,
            purpose=purpose,
            detail=detail,
            region=region,
            observations=observations,
        ),
        observations=observations,
        visual_semantics_resolved=resolved,
    )


def _analyze_pdf_page(
    boundary: FrozenWorkspaceToolBoundary,
    payload: Mapping[str, Any],
    *,
    observation_service: VisualObservationService,
    resolve_visual_file: Callable[[str], AuthorizedFileSource] | None = None,
    visual_publisher: VisualObservationPublisher | None = None,
) -> dict[str, Any]:
    target, relative = _resolve_file(boundary, payload, PDF_SUFFIXES)
    source = _source_identity(target)
    pages = _required_page_numbers(payload.get("pages"))
    purpose, detail, region, question = _required_analysis_options(
        payload,
        allowed_regions=(VisionRegion.PAGE,),
    )
    physical_pages, count_diagnostic = _pdf_page_count(target)
    if physical_pages is None:
        assert count_diagnostic is not None
        return _empty_view(
            status="gap",
            format_name="pdf",
            relative_path=relative,
            source=source,
            source_status="rejected",
            source_complete=False,
            diagnostics=(count_diagnostic,),
            rendering={
                "requested": False,
                "status": "not_requested",
                "reason_code": None,
            },
            analysis=_analysis_summary(
                requested=True,
                purpose=purpose,
                detail=detail,
                region=region,
                observations=(),
                failure_code=count_diagnostic["code"],
            ),
        )
    invalid_pages = [page for page in pages if page > physical_pages]
    if invalid_pages:
        raise ToolBusinessFailure(
            "invalid_page_range",
            "Every page in pages must be within this document; no page was analyzed.",
            {"invalid_pages": invalid_pages, "physical_pages": physical_pages},
        )
    publication_binding = _visual_publication_binding(
        purpose=purpose,
        target=target,
        source=source,
        relative=relative,
        resolve_visual_file=resolve_visual_file,
        visual_publisher=visual_publisher,
    )

    # 保留原 PDF 及精确页码作为权威来源。共享视觉服务会在外发前
    # 每次只渲染当前页，并把实际发送的 PNG 像素与渲染 recipe 写入回执。
    # 这里的 1x1 仅是渲染前的未知占位；准备完成后请求会被真实
    # payload 尺寸规范化，不会传给 provider 或持久化为观察几何。
    observations = []
    for page in pages:
        control = current_tool_execution()
        if control is not None:
            control.checkpoint()
        boundary.require_current_root()
        _assert_unchanged(target, source)
        try:
            observation = _analyze_visual_source(
                boundary,
                target=target,
                source=source,
                page=page,
                pixel_size=PixelSize(1, 1),
                purpose=purpose,
                detail=detail,
                region=region,
                question=question,
                observation_service=observation_service,
                publication_binding=publication_binding,
                visual_publisher=visual_publisher,
            )
        except ToolBusinessFailure as exc:
            known_transport_failure = (
                isinstance(exc.__cause__, MountedVisualCallWaitingExternal)
                and exc.error.details.get("reason_code") in {
                    "vision_request_timeout", "vision_connection_failed",
                }
                and isinstance(exc.error.details.get("call_key"), str)
            )
            if not isinstance(exc.__cause__, VisualPublicationIndexFailure) and not known_transport_failure:
                raise
            # 只收纳已分类的页级失败。取消、来源/项目权限变化、未知持久化
            # 错误仍向外传播；此前已发布页面无需再次调用视觉模型。
            observation = {
                "unit_id": f"workspace_{str(source['sha256'])[:24]}_p{page}",
                "page": page, "purpose": purpose.value, "detail": detail.value,
                "region": region.value, "status": "failed", "at": f"p{page}",
                "observation": None, "observation_id": None, "uncertainty": None,
                "failure_code": exc.error.details.get("reason_code", exc.error.code),
                "resampled": False,
                **({"question": question} if question is not None else {}),
                **({"failure_diagnostics": dict(exc.error.details["failure_diagnostics"])}
                   if "failure_diagnostics" in exc.error.details else {}),
            }
        observations.append(observation)
    bounded_observations, observation_metadata_truncated = _bounded_observations(
        observations
    )
    diagnostics, diagnostic_metadata_truncated = _bounded_diagnostics(
        _semantic_gap_diagnostics(observations),
        metadata_already_truncated=observation_metadata_truncated,
    )
    truncated = observation_metadata_truncated or diagnostic_metadata_truncated
    resolved = all(item["status"] == "completed" for item in observations)
    _assert_unchanged(target, source)
    view = {
        "status": "completed" if resolved and not truncated else "partial",
        "format": "pdf",
        "path": relative,
        "source": source,
        "processor": "pypdfium2-page-render",
        "coverage": {
            "source_status": "complete" if resolved else "partial",
            "source_complete": resolved,
            "selection_complete": resolved and not truncated,
            "needs_vision": not resolved,
            "physical_pages": physical_pages,
            "selected_pages": list(pages),
            "total_text_elements": 0,
            "returned_text_elements": 0,
        },
        "elements": [],
        "images": [],
        "observations": bounded_observations,
        "diagnostics": diagnostics,
        "truncated": truncated,
        "next_cursor": None,
        "next_page": None,
        "rendering": {
            "requested": False,
            "status": "not_requested",
            "reason_code": None,
        },
        "analysis": _analysis_summary(
            requested=True,
            purpose=purpose,
            detail=detail,
            region=region,
            observations=observations,
        ),
    }
    return _enforce_result_output_budget(view)


def _document_view(
    result: ProcessingResult,
    *,
    format_name: str,
    relative_path: str,
    source: dict[str, Any],
    payload: Mapping[str, Any],
    selected_pages: Sequence[int],
    physical_pages: int | None,
    images: Sequence[dict[str, Any]] = (),
    extra_diagnostics: Sequence[dict[str, Any]] = (),
    rendering: dict[str, Any] | None = None,
    analysis: dict[str, Any] | None = None,
    observations: Sequence[dict[str, Any]] = (),
    visual_semantics_resolved: bool = False,
) -> dict[str, Any]:
    selected_page_set = set(selected_pages)
    elements = tuple(
        element
        for element in result.text_elements()
        if not selected_page_set or element.locator.page in selected_page_set
    )
    max_chars = _bounded_int(
        payload.get("max_chars"), DEFAULT_MAX_CHARS, 1, MAX_CHARS, "max_chars"
    )
    max_elements = _bounded_int(
        payload.get("max_elements"),
        DEFAULT_MAX_ELEMENTS,
        1,
        MAX_ELEMENTS,
        "max_elements",
    )
    element_offset = _bounded_int(
        payload.get("element_offset"),
        0,
        0,
        MAX_CURSOR_OFFSET,
        "element_offset",
    )
    character_offset = _bounded_int(
        payload.get("character_offset"),
        0,
        0,
        MAX_CURSOR_OFFSET,
        "character_offset",
    )
    projected, next_cursor, locator_metadata_truncated = _page_elements(
        elements,
        element_offset=element_offset,
        character_offset=character_offset,
        max_chars=max_chars,
        max_elements=max_elements,
    )
    selected_pages_view = list(selected_pages[:MAX_PROJECTED_SELECTED_PAGES])
    selected_pages_truncated = len(selected_pages_view) < len(selected_pages)
    truncated = (
        next_cursor is not None
        or locator_metadata_truncated
        or selected_pages_truncated
    )
    diagnostics = [
        _diagnostic_view(item)
        for item in result.diagnostics
        if not selected_page_set
        or item.locator.page is None
        or item.locator.page in selected_page_set
    ]
    if visual_semantics_resolved:
        diagnostics = [
            item for item in diagnostics if item["code"] != "page_needs_vision"
        ]
    diagnostics.extend(extra_diagnostics)
    bounded_observations, observation_metadata_truncated = _bounded_observations(
        observations
    )
    diagnostics, diagnostic_metadata_truncated = _bounded_diagnostics(
        diagnostics,
        metadata_already_truncated=(
            locator_metadata_truncated
            or observation_metadata_truncated
            or selected_pages_truncated
        ),
    )
    metadata_truncated = (
        locator_metadata_truncated
        or observation_metadata_truncated
        or selected_pages_truncated
        or diagnostic_metadata_truncated
    )
    truncated = truncated or metadata_truncated
    selected_visual_elements = (
        element
        for element in result.elements
        if not selected_page_set or element.locator.page in selected_page_set
    )
    needs_vision = (not visual_semantics_resolved) and (
        any(element.needs_vision for element in selected_visual_elements)
        or any(
            item["code"] in {"page_needs_vision", "slide_render_backend_unavailable"}
            for item in diagnostics
        )
    )
    fatal_codes = {
        DiagnosticCode.PASSWORD_REQUIRED.value,
        DiagnosticCode.PERMISSION_DENIED.value,
        DiagnosticCode.CORRUPT_SOURCE.value,
        DiagnosticCode.EMPTY_SOURCE.value,
        DiagnosticCode.LIMIT_REACHED.value,
    }
    unresolved_source_diagnostics = tuple(
        item
        for item in result.diagnostics
        if not (
            visual_semantics_resolved and item.code is DiagnosticCode.PAGE_NEEDS_VISION
        )
    )
    source_complete = (
        not unresolved_source_diagnostics
        and not extra_diagnostics
        and (visual_semantics_resolved or not result.needs_vision)
    )
    source_status = (
        "rejected"
        if any(item.code.value in fatal_codes for item in unresolved_source_diagnostics)
        and not images
        else "complete"
        if source_complete
        else "partial"
    )
    selection_complete = not truncated and not diagnostics and not needs_vision
    status = (
        "gap"
        if not projected and not images and not bounded_observations
        else "partial"
        if not source_complete or truncated
        else "completed"
    )
    view = {
        "status": status,
        "format": format_name,
        "path": relative_path,
        "source": source,
        "processor": str(result.processor),
        "coverage": {
            "source_status": source_status,
            "source_complete": source_complete,
            "selection_complete": selection_complete,
            "needs_vision": needs_vision,
            "physical_pages": physical_pages,
            "selected_pages": selected_pages_view,
            "total_text_elements": len(elements),
            "returned_text_elements": len(projected),
        },
        "elements": projected,
        "images": list(images),
        "observations": bounded_observations,
        "diagnostics": diagnostics,
        "truncated": truncated,
        "next_cursor": next_cursor,
        "next_page": None,
        "rendering": rendering,
        "analysis": analysis,
    }
    return _enforce_result_output_budget(view)


def _page_elements(
    elements: Sequence[DocumentElement],
    *,
    element_offset: int,
    character_offset: int,
    max_chars: int,
    max_elements: int,
) -> tuple[list[dict[str, Any]], dict[str, int] | None, bool]:
    if element_offset > len(elements):
        raise ToolBusinessFailure(
            "invalid_cursor",
            "element_offset is beyond the selected evidence.",
            {"total_elements": len(elements)},
        )
    if element_offset == len(elements):
        if character_offset:
            raise ToolBusinessFailure(
                "invalid_cursor", "character_offset requires an existing element."
            )
        return [], None, False

    returned: list[dict[str, Any]] = []
    characters = 0
    next_cursor: dict[str, int] | None = None
    metadata_truncated = False
    for index in range(element_offset, len(elements)):
        element = elements[index]
        text = element.text or ""
        offset = character_offset if index == element_offset else 0
        if offset > len(text):
            raise ToolBusinessFailure(
                "invalid_cursor",
                "character_offset is beyond the selected element.",
                {"element_offset": index, "element_characters": len(text)},
            )
        if offset == len(text):
            continue
        remaining = max_chars - characters
        if remaining <= 0 or len(returned) >= max_elements:
            next_cursor = {"element_offset": index, "character_offset": offset}
            break
        end = min(len(text), offset + remaining)
        element_view, element_metadata_truncated = _element_view(
            element,
            index=index,
            start=offset,
            end=end,
        )
        returned.append(element_view)
        metadata_truncated = metadata_truncated or element_metadata_truncated
        characters += end - offset
        if end < len(text):
            next_cursor = {"element_offset": index, "character_offset": end}
            break
        if len(returned) >= max_elements and index + 1 < len(elements):
            next_cursor = {"element_offset": index + 1, "character_offset": 0}
            break
    return returned, next_cursor, metadata_truncated


def _element_view(
    element: DocumentElement, *, index: int, start: int, end: int
) -> tuple[dict[str, Any], bool]:
    locator = element.locator
    section_path, section_truncated = _bounded_section_path(locator.section_path)
    display, display_truncated = _bounded_locator_display(
        locator,
        section_path=section_path,
    )
    source_pages = list(element.source_pages[:32])
    source_pages_truncated = len(source_pages) < len(element.source_pages)
    return {
        "index": index,
        "element_id": element.element_id,
        "kind": element.kind.value,
        "text": (element.text or "")[start:end],
        "text_range": [start, end],
        "locator": {
            "page": locator.page,
            "ordinal": locator.ordinal,
            "section_path": section_path,
            "bbox": list(locator.bbox) if locator.bbox is not None else None,
            "char_range": (
                list(locator.char_range) if locator.char_range is not None else None
            ),
            "display": display,
        },
        "source_pages": source_pages,
        "needs_vision": element.needs_vision,
        "text_evidence": (
            element.text_evidence.to_prompt_dict()
            if element.text_evidence is not None
            else None
        ),
    }, section_truncated or display_truncated or source_pages_truncated


def _bounded_section_path(section_path: Sequence[str]) -> tuple[list[str], bool]:
    """将不可信标题层级投影到有界定位器字段。"""

    bounded: list[str] = []
    remaining = MAX_LOCATOR_SECTION_BYTES
    truncated = len(section_path) > MAX_LOCATOR_SECTION_SEGMENTS
    for segment in section_path[:MAX_LOCATOR_SECTION_SEGMENTS]:
        if remaining <= 0:
            truncated = True
            break
        value, shortened = _truncate_utf8(str(segment), remaining)
        bounded.append(value)
        remaining -= len(value.encode("utf-8"))
        truncated = truncated or shortened
    return bounded, truncated


def _bounded_locator_display(
    locator: DocumentLocator,
    *,
    section_path: Sequence[str],
) -> tuple[str, bool]:
    """只根据已有边界的结构化定位器渲染显示文本。"""

    parts: list[str] = []
    if locator.page is not None:
        parts.append(f"p{locator.page}")
    if section_path:
        parts.append(" › ".join(section_path))
    if locator.char_range is not None:
        parts.append(f"c{locator.char_range[0]}-{locator.char_range[1]}")
    if not parts and locator.ordinal is not None:
        parts.append(f"#{locator.ordinal}")
    elif parts and locator.ordinal is not None and locator.page is not None:
        parts[0] = f"{parts[0]}#{locator.ordinal}"
    raw = " ".join(parts) if parts else "?"
    bounded, display_truncated = _truncate_utf8(raw, MAX_LOCATOR_DISPLAY_BYTES)
    # 结构化章节被缩短，也意味着显示内容只是一段预览，即使预览本身未超出其字段限制。
    return bounded, display_truncated or tuple(section_path) != locator.section_path


def _bounded_observations(
    observations: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    bounded: list[dict[str, Any]] = []
    truncated = False
    for item in observations:
        projected = dict(item)
        for field, limit in (
            ("unit_id", MAX_DIAGNOSTIC_FIELD_BYTES),
            ("at", MAX_DIAGNOSTIC_FIELD_BYTES),
            ("observation", MAX_OBSERVATION_TEXT_BYTES),
            ("observation_id", MAX_DIAGNOSTIC_FIELD_BYTES),
            ("failure_code", MAX_DIAGNOSTIC_CODE_BYTES),
        ):
            raw = projected.get(field)
            if raw is None:
                continue
            projected[field], shortened = _truncate_utf8(str(raw), limit)
            truncated = truncated or shortened
        bounded.append(projected)
    return bounded, truncated


def _bounded_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]],
    *,
    metadata_already_truncated: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    """约束每条模型可见诊断，并预留一个显式缺口条目。"""

    projected: list[dict[str, Any]] = []
    fields_truncated = False
    reserve_gap = metadata_already_truncated
    ordinary_limit = MAX_DIAGNOSTIC_ITEMS - (1 if reserve_gap else 0)
    count_truncated = len(diagnostics) > ordinary_limit
    if count_truncated and not reserve_gap:
        reserve_gap = True
        ordinary_limit = MAX_DIAGNOSTIC_ITEMS - 1
    for item in diagnostics[:ordinary_limit]:
        bounded_item: dict[str, Any] = {}
        for field, limit in (
            ("code", MAX_DIAGNOSTIC_CODE_BYTES),
            ("at", MAX_DIAGNOSTIC_FIELD_BYTES),
            ("detail", MAX_DIAGNOSTIC_FIELD_BYTES),
            ("retry_hint", MAX_DIAGNOSTIC_FIELD_BYTES),
        ):
            raw = item.get(field)
            if raw is None:
                bounded_item[field] = None
                continue
            bounded_item[field], shortened = _truncate_utf8(str(raw), limit)
            fields_truncated = fields_truncated or shortened
        projected.append(bounded_item)

    if fields_truncated and not reserve_gap:
        reserve_gap = True
        if len(projected) == MAX_DIAGNOSTIC_ITEMS:
            projected.pop()
            count_truncated = True
    if reserve_gap or fields_truncated:
        projected.append(
            _result_metadata_gap_diagnostic(
                len(diagnostics),
                count_truncated=count_truncated,
            )
        )
    return projected, fields_truncated or count_truncated


def _result_metadata_gap_diagnostic(
    total_diagnostics: int,
    *,
    count_truncated: bool,
) -> dict[str, Any]:
    detail = (
        "Model-facing locator, coverage, observation, or diagnostic metadata was "
        "shortened to remain inside the tool-result budget."
    )
    if count_truncated:
        detail += (
            f" Only a bounded diagnostic prefix is shown from "
            f"{total_diagnostics} source diagnostics."
        )
    return {
        "code": "result_metadata_truncated",
        "at": "?",
        "detail": detail,
        "retry_hint": (
            "Source text remains losslessly resumable when next_cursor is present. "
            "Narrow page/slide ranges when more located metadata is needed."
        ),
    }


def _truncate_utf8(value: str, maximum_bytes: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return value, False
    marker = "…"
    available = max(0, maximum_bytes - len(marker.encode("utf-8")))
    prefix = encoded[:available].decode("utf-8", errors="ignore")
    return f"{prefix}{marker}" if maximum_bytes >= 3 else prefix, True


def _enforce_result_output_budget(view: dict[str, Any]) -> dict[str, Any]:
    """约束组合输出，同时将省略文本重新转换为游标。"""

    if _encoded_result_size(view) <= MAX_PROJECTED_OUTPUT_BYTES:
        return view

    _mark_result_output_gap(view)
    elements = view.get("elements")
    if isinstance(elements, list):
        while elements and _encoded_result_size(view) > MAX_PROJECTED_OUTPUT_BYTES:
            removed = elements.pop()
            text_range = removed.get("text_range", [0, 0])
            view["next_cursor"] = {
                "element_offset": int(removed.get("index", 0)),
                "character_offset": int(text_range[0]),
            }
            view["coverage"]["returned_text_elements"] = len(elements)

    diagnostics = view.get("diagnostics")
    if isinstance(diagnostics, list):
        while (
            len(diagnostics) > 1
            and _encoded_result_size(view) > MAX_PROJECTED_OUTPUT_BYTES
        ):
            removable = next(
                (
                    index
                    for index in range(len(diagnostics) - 1, -1, -1)
                    if diagnostics[index].get("code")
                    not in {"result_output_truncated", "result_metadata_truncated"}
                ),
                None,
            )
            if removable is None:
                break
            diagnostics.pop(removable)

    observations = view.get("observations")
    if isinstance(observations, list):
        observation_count = len(observations)
        while observations and _encoded_result_size(view) > MAX_PROJECTED_OUTPUT_BYTES:
            observations.pop()
        if len(observations) < observation_count and view.get("analysis") is not None:
            resolved = sum(item.get("status") == "completed" for item in observations)
            view["analysis"]["resolved_units"] = resolved
            view["analysis"]["status"] = "partial" if observations else "failed"
            view["analysis"]["failure_code"] = "result_output_truncated"
        if len(observations) < observation_count:
            view["coverage"]["needs_vision"] = True

    images = view.get("images")
    if isinstance(images, list):
        while images and _encoded_result_size(view) > MAX_PROJECTED_OUTPUT_BYTES:
            images.pop()
        rendering = view.get("rendering")
        if isinstance(rendering, dict) and rendering.get("requested") and not images:
            rendering["status"] = "unavailable"
            rendering["reason_code"] = "result_output_truncated"

    if _encoded_result_size(view) > MAX_PROJECTED_OUTPUT_BYTES:
        # 此处移除的所有证据均已由 next_cursor、next_page 或显式结果输出诊断表示。
        view["elements"] = []
        view["images"] = []
        view["observations"] = []
        view["diagnostics"] = [_result_output_gap_diagnostic()]
        view["coverage"]["returned_text_elements"] = 0
        view["coverage"]["selection_complete"] = False
        view["coverage"]["needs_vision"] = True
        if view.get("rendering") is not None:
            view["rendering"] = {
                "requested": bool(view["rendering"].get("requested")),
                "status": "unavailable",
                "reason_code": "result_output_truncated",
            }
        if view.get("analysis") is not None:
            view["analysis"]["resolved_units"] = 0
            view["analysis"]["status"] = "failed"
            view["analysis"]["failure_code"] = "result_output_truncated"
    view["status"] = (
        "partial"
        if view.get("elements") or view.get("images") or view.get("observations")
        else "gap"
    )
    return view


def _mark_result_output_gap(view: dict[str, Any]) -> None:
    view["truncated"] = True
    view["coverage"]["selection_complete"] = False
    view["status"] = (
        "partial"
        if view.get("elements") or view.get("images") or view.get("observations")
        else "gap"
    )
    diagnostics = view.setdefault("diagnostics", [])
    if not any(item.get("code") == "result_output_truncated" for item in diagnostics):
        if len(diagnostics) >= MAX_DIAGNOSTIC_ITEMS:
            diagnostics.pop()
        diagnostics.append(_result_output_gap_diagnostic())


def _result_output_gap_diagnostic() -> dict[str, Any]:
    return {
        "code": "result_output_truncated",
        "at": "?",
        "detail": (
            "The combined model-facing result reached its aggregate byte budget; "
            "trailing evidence or metadata is not present in this result."
        ),
        "retry_hint": (
            "Continue from next_cursor/next_page when present, or narrow the "
            "page/slide range and max_elements."
        ),
    }


def _encoded_result_size(view: Mapping[str, Any]) -> int:
    return len(
        json.dumps(
            view,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _diagnostic_view(item: ProcessingDiagnostic) -> dict[str, Any]:
    return {
        **item.to_dict(),
        "retry_hint": _retry_hint(item.code),
    }


def _retry_hint(code: DiagnosticCode) -> str | None:
    if code is DiagnosticCode.PAGE_NEEDS_VISION:
        return "Use analyze_image or analyze_pdf_page through the authorized visual bridge."
    if code in {DiagnosticCode.OCR_ENGINE_FAILED, DiagnosticCode.OCR_TEXT_UNCERTAIN}:
        return (
            "Use the authorized visual bridge; OCR alone did not resolve the evidence."
        )
    if code is DiagnosticCode.LIMIT_REACHED:
        return "Narrow the page range or continue with next_cursor."
    if code is DiagnosticCode.PASSWORD_REQUIRED:
        return "Provide an unlocked copy of the PDF."
    return None


def _payload_diagnostic(refusal: PayloadRefusal, *, at: str) -> dict[str, Any]:
    return {
        "code": refusal.failure.value,
        "at": at,
        "detail": refusal.detail,
        "retry_hint": "Narrow the page or lower detail, then retry."
        if "large" in refusal.failure.value
        else None,
    }


def _image_view(payload, *, page: int, detail: VisionDetail) -> dict[str, Any]:
    """投影像素标识，但不把图像字节放入模型上下文。"""

    return {
        "page": page,
        "mime_type": payload.mime_type,
        "width": payload.pixel_size.width,
        "height": payload.pixel_size.height,
        "sent_sha256": payload.sent_sha256,
        "source_sha256": payload.source_sha256,
        "resampled": payload.resampled,
        "detail": detail.value,
    }


def _vision_request(
    target: Path,
    source: Mapping[str, Any],
    *,
    page: int,
    detail: VisionDetail,
    mime_type: str,
) -> VisionRequest:
    sha256 = str(source["sha256"])
    return VisionRequest(
        source_unit_id=f"workspace_{sha256[:24]}_p{page}",
        source_sha256=sha256,
        # 载荷构建器会单独认证实际发送的像素。渲染前，唯一稳定的本地标识是来源哈希。
        image_sha256=sha256,
        locator=DocumentLocator(page=page),
        mime_type=mime_type,
        pixel_size=PixelSize(1, 1),
        byte_count=int(source["byte_count"]),
        purpose=VisionPurpose.GENERAL,
        prompt_contract_version=FORMAT_OBSERVATION_TOOL_CONTRACT_VERSION,
        image_path=str(target),
        detail=detail,
        region=VisionRegion.PAGE,
    )


def _required_analysis_options(
    payload: Mapping[str, Any],
    *,
    allowed_regions: tuple[VisionRegion, ...],
) -> tuple[VisionPurpose, VisionDetail, VisionRegion, str | None]:
    missing = [
        name for name in ("purpose", "detail", "region") if payload.get(name) is None
    ]
    if missing:
        raise ToolBusinessFailure(
            "invalid_request",
            "visual analysis requires explicit purpose, detail, and region.",
            {"missing": missing},
        )
    detail = _vision_detail(payload.get("detail"))
    try:
        purpose = VisionPurpose(str(payload["purpose"]).strip())
    except ValueError as exc:
        raise ToolBusinessFailure("invalid_request", "unknown visual purpose.") from exc
    try:
        region = VisionRegion(str(payload["region"]).strip())
    except ValueError as exc:
        raise ToolBusinessFailure("invalid_request", "unknown visual region.") from exc
    if region not in allowed_regions:
        raise ToolBusinessFailure(
            "region_not_available",
            "This source does not have geometry for the requested region.",
            {"allowed_regions": [item.value for item in allowed_regions]},
        )
    question = parse_visual_question(purpose, payload.get("question"))
    visual_call_identity(purpose)
    return purpose, detail, region, question


def _analyze_visual_source(
    boundary: FrozenWorkspaceToolBoundary,
    *,
    target: Path,
    source: Mapping[str, Any],
    page: int,
    pixel_size: PixelSize,
    purpose: VisionPurpose,
    detail: VisionDetail,
    region: VisionRegion,
    observation_service: VisualObservationService,
    question: str | None = None,
    publication_binding: AuthorizedFileSource | None = None,
    visual_publisher: VisualObservationPublisher | None = None,
) -> dict[str, Any]:
    sha256 = str(source["sha256"])
    unit_id = f"workspace_{sha256[:24]}_p{page}"
    unit = VisualUnitRef(
        unit_id=unit_id,
        kind=(
            DocumentNonTextKind.FORMULA
            if purpose is VisionPurpose.FORMULA
            else DocumentNonTextKind.FIGURE
        ),
        image_path=str(target),
        source_sha256=sha256,
        image_sha256=sha256,
        locator=DocumentLocator(page=page),
        mime_type=(
            "image/png" if target.suffix.lower() in {".png", ".pdf"} else "image/jpeg"
        ),
        pixel_size=pixel_size,
        byte_count=int(source["byte_count"]),
        disclosure_source_path=str(target),
    )
    visual_boundary = FrozenVisualToolBoundary(
        session_id=boundary.session_id, units=(unit,)
    )
    request = VisualObservationRequest(
        unit_id=unit_id,
        purpose=purpose,
        detail=detail,
        region=region,
        question=question,
    )
    publication = None
    if visual_publisher is not None or publication_binding is not None:
        if visual_publisher is None or publication_binding is None:
            raise ToolBusinessFailure(
                "visual_project_publication_unavailable",
                "视觉观察缺少完整的项目发布绑定。",
            )
        try:
            projection, publication = visual_publisher.observe(
                service=observation_service,
                boundary=visual_boundary,
                request=request,
                binding=publication_binding,
            )
            result = projection.to_dict()
        except MountedVisualCallWaitingExternal as exc:
            raise ToolBusinessFailure(
                "visual_completion_unconfirmed",
                "视觉请求未取得可用回执，完成状态不明；当前没有后台等待或自动重试任务。"
                "请依据具体失败原因决定后续操作；重新调用可能再次发送并计费。",
                exc.diagnostic_details(),
            ) from exc
        except VisualPublicationIndexFailure as exc:
            raise ToolBusinessFailure(
                "visual_project_publication_unavailable",
                "视觉回执已保存，但检索索引未完成发布；具体失败码不表示后台必然可恢复。",
                {"publication_state": "ready", "reason_code": exc.reason_code},
            ) from exc
        except MountedVisualCallLedgerError as exc:
            raise ToolBusinessFailure(
                "visual_project_publication_unavailable",
                "视觉观察尚未完成项目发布；保留回执供恢复。",
            ) from exc
    elif purpose is VisionPurpose.QUESTION:
        raise ToolBusinessFailure(
            "visual_project_publication_unavailable",
            "视觉问答缺少项目发布绑定。",
        )
    else:
        result = (
            observation_service.observe(visual_boundary, (request,))
            .results[0]
            .to_dict()
        )
    publication_identity = (
        {}
        if publication is None
        else {
            "picture_id": publication.picture_id,
            "picture_unit_id": publication.picture_unit_id,
            "observation_id": publication.observation_commits[
                0
            ].observation.observation_id,
        }
    )
    return {
        "unit_id": unit_id,
        "page": page,
        "purpose": purpose.value,
        "detail": detail.value,
        "region": region.value,
        "status": result["status"],
        "at": result["at"],
        "observation": result.get("observation"),
        "observation_id": result.get("observation_id"),
        "uncertainty": result.get("uncertainty"),
        "failure_code": result.get("failure_code"),
        **({"failure_diagnostics": result["failure_diagnostics"]}
           if result.get("failure_diagnostics") is not None else {}),
        "resampled": bool(result.get("resampled", False)),
        **({"question": question} if question is not None else {}),
        **publication_identity,
    }


def _visual_publication_binding(
    *,
    purpose: VisionPurpose,
    target: Path,
    source: Mapping[str, Any],
    relative: str,
    resolve_visual_file: Callable[[str], AuthorizedFileSource] | None,
    visual_publisher: VisualObservationPublisher | None,
) -> AuthorizedFileSource | None:
    if resolve_visual_file is None and visual_publisher is None:
        if purpose is VisionPurpose.QUESTION:
            raise ToolBusinessFailure(
                "visual_project_publication_unavailable",
                "视觉问答缺少项目发布绑定。",
            )
        # 低层单元测试和本地 adapter 可以只观测而不组装 Project；
        # 生产 Session 会始终同时传入 resolver 和 publisher。
        return None
    if resolve_visual_file is None or visual_publisher is None:
        raise ToolBusinessFailure(
            "visual_project_publication_unavailable",
            "视觉观察缺少完整的项目发布绑定。",
        )
    visual_publisher.recover_ready()
    try:
        binding = resolve_visual_file(relative)
    except FileAccessError as exc:
        raise ToolBusinessFailure(
            exc.reason_code, "视觉来源的当前文件版本不可用。"
        ) from exc
    if (
        not isinstance(binding, AuthorizedFileSource)
        or not binding.file_id
        or not binding.file_version_id
        or Path(binding.canonical_path) != target
        or binding.fingerprint.sha256 != source["sha256"]
        or binding.fingerprint.size_bytes != source["byte_count"]
    ):
        raise ToolBusinessFailure(
            "visual_file_binding_mismatch", "视觉来源与项目文件版本不一致。"
        )
    return binding


def _analysis_summary(
    *,
    requested: bool,
    purpose: VisionPurpose | None,
    detail: VisionDetail | None,
    region: VisionRegion | None,
    observations: Sequence[Mapping[str, Any]],
    failure_code: str | None = None,
) -> dict[str, Any]:
    resolved = sum(item.get("status") == "completed" for item in observations)
    if not requested:
        status = "not_requested"
    elif observations and resolved == len(observations):
        status = "completed"
    elif resolved:
        status = "partial"
    elif observations and all(
        item.get("status") == "unavailable" for item in observations
    ):
        status = "unavailable"
    elif observations:
        status = "failed"
    else:
        status = "unavailable"
    first_failure = next(
        (
            str(item["failure_code"])
            for item in observations
            if item.get("failure_code")
        ),
        failure_code,
    )
    return {
        "requested": requested,
        "status": status,
        "purpose": purpose.value if purpose is not None else None,
        "detail": detail.value if detail is not None else None,
        "region": region.value if region is not None else None,
        "requested_units": len(observations) if requested else 0,
        "resolved_units": resolved,
        "failure_code": first_failure,
    }


def _semantic_gap_diagnostics(
    observations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    diagnostics: list[dict[str, Any]] = []
    for item in observations:
        if item.get("status") == "completed":
            continue
        failure = str(item.get("failure_code") or "vision_analysis_partial")
        if failure.startswith("disclosure_"):
            retry_hint = "Obtain a disclosure grant for this source, endpoint, model, and purpose."
        elif failure == "external_vision_requires_protected_operation":
            retry_hint = (
                "Use the protected visual-analysis capability after authorization; "
                "changing reader parameters or provider configuration will not help."
            )
        elif failure == "vision_provider_unavailable":
            retry_hint = (
                "Configure an available vision provider, then retry explicitly."
            )
        elif failure.startswith("picture_retrieval_"):
            retry_hint = (
                "The visual receipt is saved but its retrieval index was not published. "
                "Do not request another visual model call solely to repair indexing; "
                "use available pages and state any remaining information gap."
            )
        elif failure in {"vision_request_timeout", "vision_connection_failed"}:
            retry_hint = (
                "Request completion is uncertain; no background wait or automatic retry is active. "
                "A new call may send and charge again. Use available pages and report the gap."
            )
        else:
            retry_hint = "Retry with higher detail or a wider region if the provider is available."
        diagnostics.append(
            {
                "code": failure,
                "at": f"p{item['page']}",
                "detail": "Visual semantics remain unresolved.",
                "retry_hint": retry_hint,
            }
        )
    return tuple(diagnostics)


def _image_pixel_size(target: Path) -> PixelSize:
    try:
        from PIL import Image

        with Image.open(target) as opened:
            return PixelSize(int(opened.width), int(opened.height))
    except Exception as exc:
        raise ToolBusinessFailure(
            "image_geometry_unavailable", "The image dimensions cannot be read."
        ) from exc


def _physical_pages(result: ProcessingResult) -> int | None:
    if result.page_manifest is not None and result.page_manifest.physical_page_count:
        return result.page_manifest.physical_page_count
    pages = {
        page for element in result.elements for page in element.source_pages if page > 0
    }
    return max(pages) if pages else None


def _page_selection(
    payload: Mapping[str, Any],
    *,
    physical_pages: int | None,
    start_key: str,
    end_key: str,
) -> tuple[int, ...]:
    start = _bounded_int(payload.get(start_key), 1, 1, MAX_CURSOR_OFFSET, start_key)
    default_end = physical_pages if physical_pages is not None else start
    end = _bounded_int(payload.get(end_key), default_end, 1, MAX_CURSOR_OFFSET, end_key)
    if end < start:
        raise ToolBusinessFailure(
            "invalid_page_range", f"{end_key} must be at least {start_key}."
        )
    if physical_pages is not None and start > physical_pages:
        raise ToolBusinessFailure(
            "invalid_page_range",
            f"{start_key} is outside this document.",
            {"physical_pages": physical_pages},
        )
    if physical_pages is not None:
        end = min(end, physical_pages)
    return tuple(range(start, end + 1))


def _pdf_page_count(
    target: Path,
) -> tuple[int | None, dict[str, Any] | None]:
    try:
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(target))
        try:
            count = len(document)
        finally:
            document.close()
    except Exception as exc:
        return None, {
            "code": "pdf_render_unavailable",
            "at": "?",
            "detail": type(exc).__name__,
            "retry_hint": "Use read_pdf_text or provide an unlocked, valid PDF.",
        }
    if count < 1:
        return None, {
            "code": "empty_source",
            "at": "?",
            "detail": "PDF contains no physical pages.",
            "retry_hint": None,
        }
    return count, None


def _legacy_office_gap(
    *, format_name: str, relative_path: str, source: dict[str, Any]
) -> dict[str, Any]:
    return _empty_view(
        status="gap",
        format_name=format_name,
        relative_path=relative_path,
        source=source,
        source_status="rejected",
        source_complete=False,
        diagnostics=(
            {
                "code": "legacy_office_bridge_unavailable",
                "at": "?",
                "detail": "Legacy binary Office input needs a sandboxed conversion bridge.",
                "retry_hint": "Provide a DOCX/PPTX copy or configure the conversion bridge.",
            },
        ),
        rendering=None,
    )


def _empty_view(
    *,
    status: str,
    format_name: str,
    relative_path: str,
    source: dict[str, Any],
    source_status: str,
    source_complete: bool,
    diagnostics: Sequence[dict[str, Any]],
    rendering: dict[str, Any] | None,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bounded_diagnostics, metadata_truncated = _bounded_diagnostics(diagnostics)
    view = {
        "status": status,
        "format": format_name,
        "path": relative_path,
        "source": source,
        "processor": None,
        "coverage": {
            "source_status": source_status,
            "source_complete": source_complete,
            "selection_complete": False,
            "needs_vision": False,
            "physical_pages": None,
            "selected_pages": [],
            "total_text_elements": 0,
            "returned_text_elements": 0,
        },
        "elements": [],
        "images": [],
        "observations": [],
        "diagnostics": bounded_diagnostics,
        "truncated": metadata_truncated,
        "next_cursor": None,
        "next_page": None,
        "rendering": rendering,
        "analysis": analysis,
    }
    return _enforce_result_output_budget(view)


def _vision_detail(value: Any) -> VisionDetail:
    raw = str(value or VisionDetail.STANDARD.value).strip()
    try:
        return VisionDetail(raw)
    except ValueError as exc:
        raise ToolBusinessFailure(
            "invalid_request", "unknown visual detail level."
        ) from exc


def _bounded_int(value: Any, default: int, low: int, high: int, name: str) -> int:
    if value is None:
        return default
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not low <= value <= high
    ):
        raise ToolBusinessFailure(
            "invalid_request", f"{name} must be an integer within {low}..{high}."
        )
    return value


def _required_page_numbers(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_UNITS_PER_CALL:
        raise ToolBusinessFailure(
            "invalid_request",
            f"pages must contain 1..{MAX_UNITS_PER_CALL} unique one-based page numbers.",
        )
    if any(
        isinstance(page, bool) or not isinstance(page, int)
        or not 1 <= page <= MAX_CURSOR_OFFSET
        for page in value
    ) or len(set(value)) != len(value):
        raise ToolBusinessFailure(
            "invalid_request",
            f"pages must contain unique integers within 1..{MAX_CURSOR_OFFSET}; no page was analyzed.",
        )
    return tuple(value)


def _boolean(value: Any, default: bool, name: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ToolBusinessFailure("invalid_request", f"{name} must be boolean.")
    return value


def _text_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["path"],
        "properties": {"path": _path_schema(), **_cursor_properties()},
    }


def _paged_text_input_schema(unit: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["path"],
        "properties": {
            "path": _path_schema(),
            f"start_{unit}": {"type": "integer", "minimum": 1},
            f"end_{unit}": {"type": "integer", "minimum": 1},
            **_cursor_properties(),
        },
    }


def _slides_input_schema() -> dict[str, Any]:
    schema = _paged_text_input_schema("slide")
    schema["properties"]["render_visuals"] = {"type": "boolean", "default": False}
    return schema


def _image_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["path"],
        "properties": {
            "path": _path_schema(),
            "detail": _detail_schema(),
            **_cursor_properties(),
        },
    }


def _analyze_image_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["path", "purpose", "detail", "region"],
        "allOf": [visual_question_constraint()],
        "properties": {
            "path": _path_schema(),
            **_required_analysis_input_properties(allowed_regions=tuple(VisionRegion)),
            **_cursor_properties(),
        },
    }


def _analyze_pdf_page_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["path", "pages", "purpose", "detail", "region"],
        "allOf": [visual_question_constraint()],
        "properties": {
            "path": _path_schema(),
            "pages": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_UNITS_PER_CALL,
                "uniqueItems": True,
                "items": {"type": "integer", "minimum": 1, "maximum": MAX_CURSOR_OFFSET},
                "description": "不重复的一基页码，按请求顺序逐页分析；将本 Attempt 的多个页面合并在此列表。",
            },
            **_required_analysis_input_properties(allowed_regions=(VisionRegion.PAGE,)),
        },
    }


def _path_schema() -> dict[str, Any]:
    return {
        "type": "string",
        "minLength": 1,
        "maxLength": MAX_PATH_CHARS,
        "description": (
            "工作区相对路径；优先使用 file_catalog.relative_path 或目录工具返回的 "
            "path。name 只是显示文件名，不等于 path；只有文件确在工作区根目录时，"
            "文件名本身才可作为完整相对路径。绝对路径和越界路径会被拒绝。"
        ),
    }


def _detail_schema() -> dict[str, Any]:
    return {
        "enum": [value.value for value in VisionDetail],
        "default": VisionDetail.STANDARD.value,
    }


def _required_analysis_input_properties(
    *, allowed_regions: tuple[VisionRegion, ...]
) -> dict[str, Any]:
    return {
        "purpose": {"enum": [item.value for item in VisionPurpose]},
        "question": visual_question_schema(),
        "detail": _detail_schema(),
        "region": {"enum": [item.value for item in allowed_regions]},
    }


def _cursor_properties() -> dict[str, Any]:
    return {
        "max_chars": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_CHARS,
            "default": DEFAULT_MAX_CHARS,
        },
        "max_elements": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_ELEMENTS,
            "default": DEFAULT_MAX_ELEMENTS,
        },
        "element_offset": {"type": "integer", "minimum": 0, "default": 0},
        "character_offset": {"type": "integer", "minimum": 0, "default": 0},
    }


def _output_schema() -> dict[str, Any]:
    nullable_integer = {"type": ["integer", "null"], "minimum": 0}
    nullable_string = {"type": ["string", "null"]}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "status",
            "format",
            "path",
            "source",
            "processor",
            "coverage",
            "elements",
            "images",
            "observations",
            "diagnostics",
            "truncated",
            "next_cursor",
            "next_page",
            "rendering",
            "analysis",
        ],
        "properties": {
            "status": {"enum": ["completed", "partial", "gap"]},
            "format": {"enum": ["text", "pdf", "word", "slides", "image"]},
            "path": {"type": "string"},
            "source": {
                "type": "object",
                "additionalProperties": False,
                "required": ["suffix", "byte_count", "sha256"],
                "properties": {
                    "suffix": {"type": "string"},
                    "byte_count": {"type": "integer", "minimum": 1},
                    "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                },
            },
            "processor": nullable_string,
            "coverage": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "source_status",
                    "source_complete",
                    "selection_complete",
                    "needs_vision",
                    "physical_pages",
                    "selected_pages",
                    "total_text_elements",
                    "returned_text_elements",
                ],
                "properties": {
                    "source_status": {"enum": ["complete", "partial", "rejected"]},
                    "source_complete": {"type": "boolean"},
                    "selection_complete": {"type": "boolean"},
                    "needs_vision": {"type": "boolean"},
                    "physical_pages": nullable_integer,
                    "selected_pages": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1},
                        "uniqueItems": True,
                    },
                    "total_text_elements": {"type": "integer", "minimum": 0},
                    "returned_text_elements": {"type": "integer", "minimum": 0},
                },
            },
            "elements": {"type": "array", "items": _element_schema()},
            "images": {"type": "array", "items": _image_schema()},
            "observations": {"type": "array", "items": _observation_schema()},
            "diagnostics": {"type": "array", "items": _diagnostic_schema()},
            "truncated": {"type": "boolean"},
            "next_cursor": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "required": ["element_offset", "character_offset"],
                "properties": {
                    "element_offset": {"type": "integer", "minimum": 0},
                    "character_offset": {"type": "integer", "minimum": 0},
                },
            },
            "next_page": nullable_integer,
            "rendering": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "required": ["requested", "status", "reason_code"],
                "properties": {
                    "requested": {"type": "boolean"},
                    "status": {"enum": ["available", "unavailable", "not_requested"]},
                    "reason_code": nullable_string,
                },
            },
            "analysis": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "required": [
                    "requested",
                    "status",
                    "purpose",
                    "detail",
                    "region",
                    "requested_units",
                    "resolved_units",
                    "failure_code",
                ],
                "properties": {
                    "requested": {"type": "boolean"},
                    "status": {
                        "enum": [
                            "not_requested",
                            "completed",
                            "partial",
                            "unavailable",
                            "failed",
                        ]
                    },
                    "purpose": {
                        "type": ["string", "null"],
                        "enum": [None, *[item.value for item in VisionPurpose]],
                    },
                    "detail": {
                        "type": ["string", "null"],
                        "enum": [None, *[item.value for item in VisionDetail]],
                    },
                    "region": {
                        "type": ["string", "null"],
                        "enum": [None, *[item.value for item in VisionRegion]],
                    },
                    "requested_units": {"type": "integer", "minimum": 0},
                    "resolved_units": {"type": "integer", "minimum": 0},
                    "failure_code": nullable_string,
                },
            },
        },
    }


def _element_schema() -> dict[str, Any]:
    nullable_integer = {"type": ["integer", "null"], "minimum": 0}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "index",
            "element_id",
            "kind",
            "text",
            "text_range",
            "locator",
            "source_pages",
            "needs_vision",
            "text_evidence",
        ],
        "properties": {
            "index": {"type": "integer", "minimum": 0},
            "element_id": {"type": "string"},
            "kind": {"type": "string"},
            "text": {"type": "string"},
            "text_range": {
                "type": "array",
                "prefixItems": [
                    {"type": "integer", "minimum": 0},
                    {"type": "integer", "minimum": 0},
                ],
                "minItems": 2,
                "maxItems": 2,
            },
            "locator": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "page",
                    "ordinal",
                    "section_path",
                    "bbox",
                    "char_range",
                    "display",
                ],
                "properties": {
                    "page": nullable_integer,
                    "ordinal": nullable_integer,
                    "section_path": {"type": "array", "items": {"type": "string"}},
                    "bbox": {
                        "type": ["array", "null"],
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "char_range": {
                        "type": ["array", "null"],
                        "items": {"type": "integer", "minimum": 0},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "display": {"type": "string"},
                },
            },
            "source_pages": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1},
                "uniqueItems": True,
            },
            "needs_vision": {"type": "boolean"},
            "text_evidence": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "required": ["origin", "confidence", "uncertainty", "source_unit_id"],
                "properties": {
                    "origin": {"type": "string"},
                    "confidence": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 1,
                    },
                    "uncertainty": {"type": "number", "minimum": 0, "maximum": 1},
                    "source_unit_id": {"type": "string"},
                },
            },
        },
    }


def _image_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "page",
            "mime_type",
            "width",
            "height",
            "sent_sha256",
            "source_sha256",
            "resampled",
            "detail",
        ],
        "properties": {
            "page": {"type": "integer", "minimum": 1},
            "mime_type": {"enum": ["image/png"]},
            "width": {"type": "integer", "minimum": 1},
            "height": {"type": "integer", "minimum": 1},
            "sent_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "source_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "resampled": {"type": "boolean"},
            "detail": {"enum": [value.value for value in VisionDetail]},
        },
    }


def _observation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "unit_id",
            "page",
            "purpose",
            "detail",
            "region",
            "status",
            "at",
            "observation",
            "observation_id",
            "uncertainty",
            "failure_code",
            "resampled",
        ],
        "properties": {
            "unit_id": {"type": "string"},
            "picture_id": {"type": "string"},
            "picture_unit_id": {"type": "string"},
            "page": {"type": "integer", "minimum": 1},
            "purpose": {"enum": [item.value for item in VisionPurpose]},
            "question": visual_question_schema(),
            "detail": {"enum": [item.value for item in VisionDetail]},
            "region": {"enum": [item.value for item in VisionRegion]},
            "status": {"enum": ["completed", "partial", "unavailable", "failed"]},
            "at": {"type": "string"},
            "observation": {"type": ["string", "null"]},
            "observation_id": {"type": ["string", "null"]},
            "uncertainty": {
                "type": ["number", "null"],
                "minimum": 0,
                "maximum": 1,
            },
            "failure_code": {"type": ["string", "null"]},
            "failure_diagnostics": failure_diagnostics_schema(),
            "resampled": {"type": "boolean"},
        },
    }


def _diagnostic_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["code", "at", "detail", "retry_hint"],
        "properties": {
            "code": {"type": "string"},
            "at": {"type": "string"},
            "detail": {"type": ["string", "null"]},
            "retry_hint": {"type": ["string", "null"]},
        },
    }


__all__ = [
    "EXTERNAL_VISUAL_ANALYSIS_TOOL_IDS",
    "FORMAT_OBSERVATION_LOCAL_TOOL_IDS",
    "FORMAT_OBSERVATION_TOOL_CONTRACT_VERSION",
    "FORMAT_OBSERVATION_TOOL_IMPLEMENTATION_VERSION",
    "VISUAL_OBSERVATION_TOOL_CONTRACT_VERSION",
    "VISUAL_OBSERVATION_TOOL_IMPLEMENTATION_VERSION",
    "build_external_visual_analysis_execution_profile",
    "build_external_visual_analysis_tool_registrations",
    "build_external_visual_analysis_tool_specs",
    "build_format_observation_execution_profile",
    "build_format_observation_tool_registrations",
    "build_format_observation_tool_specs",
]
