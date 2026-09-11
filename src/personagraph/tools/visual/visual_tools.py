"""用于读取一个未解析视觉单元的模型可见注册。

文档读取器已经将每个插图、图示和公式保留为 ``requires_visual_read=True`` 的可寻址单元。
此工具负责解析其中一个：主模型无需亲自查看图像，只判断某个插图“很重要”；真正查看图像的是
该工具背后的视觉适配器。

Host 会冻结存在哪些单元及其像素位置，因此提案可以指定单元，却绝不能指定路径。模型需要选择
``purpose``，因为只有它能根据标题和周边文本判断插图是数据图表还是示意图；Host 只有检测器
给出的结构化 ``kind``。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from ...input_processing.vision.contracts import (
    VisionDetail,
    VisionRegion,
    VisionPurpose,
    VisionStatus,
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
from ..registration import ToolExecutionProfile, ToolRegistration
from .failure_reporting import failure_diagnostics_schema
from .question_contract import (
    parse_visual_question,
    visual_question_constraint,
    visual_question_schema,
)
from .visual_observation_service import (
    MAX_UNITS_PER_CALL,
    VisualObservationRequest,
    VisualObservationService,
    default_vision_adapter,
)
from .visual_tool_boundary import FrozenVisualToolBoundary, VisualUnitRef


VISUAL_TOOL_CONTRACT_VERSION = "visual-read-v1"
VISUAL_TOOL_IMPLEMENTATION_VERSION = "1"


def build_visual_tool_registrations(
    boundary: FrozenVisualToolBoundary,
    *,
    adapter=None,
) -> tuple[ToolRegistration]:
    """为一组冻结单元暴露视觉读取；远程分析默认同意。"""

    service = VisualObservationService(adapter=adapter)
    capabilities = service.capabilities
    transmits = service.transmits_externally
    return (
        ToolRegistration(
            spec=ToolSpec(
                tool_id="read_visual_unit",
                contract_version=VISUAL_TOOL_CONTRACT_VERSION,
                name="Read a figure, diagram or formula",
                description=(
                    "按 unit_id 读取最多三个视觉区域。purpose 可选 chart（图表数据）、"
                    "formula（公式转写）、caption（简短说明）、general（整体描述），"
                    "或 question（结合图像回答 question 中的具体自然语言问题）。"
                    "question 用途必须提供非空问题；其它用途省略 question 或设为 null。"
                    "标准清晰度无法辨认细节时可选择 high。"
                ),
                input_schema=_input_schema(),
                output_schema=_output_schema(),
                catalog_tags=("document", "read"),
            ),
            implementation_version=visual_observation_implementation_version(
                capabilities.processor_fingerprint
            ),
            source=ToolSourceDescriptor(
                kind=ToolSourceKind.LOCAL,
                source_id="personagraph.input_processing.vision",
            ),
            handler=_handler(boundary, service),
            effect_profile=visual_observation_effect_profile(
                boundary.session_id,
                transmits,
            ),
            execution_profile=visual_observation_execution_profile(),
        ),
    )


def visual_observation_implementation_version(
    processor_fingerprint: str,
) -> str:
    return f"{VISUAL_TOOL_IMPLEMENTATION_VERSION}+{processor_fingerprint}"


def visual_observation_execution_profile() -> ToolExecutionProfile:
    return ToolExecutionProfile(
        default_timeout_s=120.0,
        hard_timeout_s=240.0,
        max_output_bytes=256_000,
    )


def visual_observation_effect_profile(
    session_id: str,
    sends_externally: bool,
) -> ToolEffectProfile:
    return ToolEffectProfile(
        (
            EffectDescriptor(
                resource=(
                    EffectResource.NETWORK
                    if sends_externally
                    else EffectResource.FILESYSTEM
                ),
                action=EffectAction.TRANSMIT if sends_externally else EffectAction.READ,
                scope_kind=EffectScopeKind.SESSION,
                default_scope=session_id,
                # 远程提供商会接收图像本身，这是此工具后果最显著的单一事实。
                data_egress=(
                    DataEgress.CONTENT if sends_externally else DataEgress.NONE
                ),
                idempotency=(
                    Idempotency.NOT_IDEMPOTENT
                    if sends_externally
                    else Idempotency.IDEMPOTENT
                ),
                reversibility=(
                    Reversibility.IRREVERSIBLE
                    if sends_externally
                    else Reversibility.REVERSIBLE
                ),
            ),
        )
    )


def _handler(
    boundary: FrozenVisualToolBoundary,
    service: VisualObservationService,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def run(payload: dict[str, Any]) -> dict[str, Any]:
        raw_requests = _parse_units(payload)
        detail = _parse_detail(payload)
        region = _parse_region(payload)
        return service.observe(
            boundary,
            tuple(
                VisualObservationRequest(
                    unit_id=unit_id,
                    purpose=purpose,
                    detail=detail,
                    region=region,
                    question=question,
                )
                for unit_id, purpose, question in raw_requests
            )
        ).to_dict()

    return run


def _parse_units(
    payload: Mapping[str, Any],
) -> list[tuple[str, VisionPurpose, str | None]]:
    raw = payload.get("units")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ToolBusinessFailure("invalid_request", "units must be a non-empty list")
    if len(raw) > MAX_UNITS_PER_CALL:
        raise ToolBusinessFailure(
            "invalid_request", f"at most {MAX_UNITS_PER_CALL} units may be read at once"
        )

    parsed: list[tuple[str, VisionPurpose, str | None]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise ToolBusinessFailure("invalid_request", "each unit must be an object")
        unit_id = str(item.get("unit_id") or "").strip()
        if unit_id in seen:
            raise ToolBusinessFailure("invalid_request", "unit_id values must be unique")
        seen.add(unit_id)
        try:
            purpose = VisionPurpose(str(item.get("purpose") or "").strip())
        except ValueError as exc:
            raise ToolBusinessFailure("invalid_request", "unknown purpose") from exc
        parsed.append((unit_id, purpose, parse_visual_question(purpose, item.get("question"))))
    return parsed


def _parse_detail(payload: Mapping[str, Any]) -> VisionDetail:
    raw = str(payload.get("detail") or VisionDetail.STANDARD.value).strip()
    try:
        return VisionDetail(raw)
    except ValueError as exc:
        raise ToolBusinessFailure("invalid_request", "unknown detail level") from exc


def _parse_region(payload: Mapping[str, Any]) -> VisionRegion:
    raw = str(payload.get("region") or VisionRegion.DETECTED.value).strip()
    try:
        return VisionRegion(raw)
    except ValueError as exc:
        raise ToolBusinessFailure("invalid_request", "unknown region") from exc


def _input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["units"],
        "properties": {
            "units": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_UNITS_PER_CALL,
    # 特意不枚举冻结单元。在这里逐一命名虽能从结构上固定配对，却也会让模式随语料库增长——
    # Session 中每次调用都要为每幅图携带一个分支——而边界安全从来不依赖这一点：
    # 处理器无论如何都会依据边界解析每个 unit_id，并拒绝找不到的项。文件工具也不会列出文件。
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["unit_id", "purpose"],
                    "allOf": [visual_question_constraint()],
                    "properties": {
                        "unit_id": {"type": "string"},
                        "purpose": {"enum": [p.value for p in VisionPurpose]},
                        "question": visual_question_schema(),
                    },
                },
            },
            "detail": {
                "enum": [level.value for level in VisionDetail],
                "default": VisionDetail.STANDARD.value,
                "description": (
                    "How much pixel detail to submit. Use 'high' only after a "
                    "standard read failed to resolve something."
                ),
            },
            "region": {
                "enum": [item.value for item in VisionRegion],
                "default": VisionRegion.DETECTED.value,
                "description": (
                    "How much of the page to include. 'detected' is the figure "
                    "itself. If a reading looks cut off at an edge — an axis "
                    "label or a bar's value missing — retry with 'expanded', "
                    "then 'page'."
                ),
            },
        },
    }


def _output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["results", "requested", "resolved"],
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["unit_id", "purpose", "region", "status", "at"],
                    "properties": {
                        "unit_id": {"type": "string"},
                        "purpose": {"enum": [p.value for p in VisionPurpose]},
                        "question": visual_question_schema(),
                        "region": {"enum": [item.value for item in VisionRegion]},
                        "status": {"enum": [s.value for s in VisionStatus]},
                        "at": {"type": "string"},
                        "observation": {"type": "string"},
                        "observation_id": {"type": "string"},
                        "uncertainty": {"type": "number", "minimum": 0, "maximum": 1},
                        "failure_code": {"type": "string"},
                        "failure_diagnostics": failure_diagnostics_schema(),
                        "resampled": {"type": "boolean"},
                    },
                },
            },
            "requested": {"type": "integer", "minimum": 1},
            "resolved": {"type": "integer", "minimum": 0},
        },
    }


__all__ = [
    "MAX_UNITS_PER_CALL",
    "VISUAL_TOOL_CONTRACT_VERSION",
    'FrozenVisualToolBoundary',
    'VisualUnitRef',
    "build_visual_tool_registrations",
    "default_vision_adapter",
    "visual_observation_effect_profile",
    "visual_observation_execution_profile",
    "visual_observation_implementation_version",
]
