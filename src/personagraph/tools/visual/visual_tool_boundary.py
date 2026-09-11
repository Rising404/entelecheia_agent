"""模型可寻址视觉单元的冻结 Host 权威信息。

本模块只持有封闭单元集合和结构化用途允许列表。适配器解析、披露与载荷 I/O 位于
``visual_observation_service``，工具注册则位于 ``visual_tools``。
"""

from __future__ import annotations

from dataclasses import dataclass

from ...input_processing.documents.contracts import DocumentLocator, DocumentNonTextKind
from ...input_processing.vision.contracts import PixelSize, VisionPurpose


# 哪些用途适用于哪些结构类型。公式绝不是图表，而插图既可能是绘制的数据系列，
# 也可能是照片，只有模型能区分二者。
_PURPOSES_BY_KIND: dict[DocumentNonTextKind, tuple[VisionPurpose, ...]] = {
# 只有上游文档清单仍将表格像素标记为需要视觉读取时，表格才进入此边界。
# 结构化表格（包括普通电子表格数据）绝不需要变成 VisualUnitRef。
    DocumentNonTextKind.TABLE: (VisionPurpose.GENERAL, VisionPurpose.QUESTION),
    DocumentNonTextKind.FORMULA: (VisionPurpose.FORMULA, VisionPurpose.QUESTION),
    DocumentNonTextKind.FIGURE: (
        VisionPurpose.CHART,
        VisionPurpose.CAPTION,
        VisionPurpose.GENERAL,
        VisionPurpose.QUESTION,
    ),
    DocumentNonTextKind.VECTOR_GRAPHICS: (
        VisionPurpose.CHART,
        VisionPurpose.CAPTION,
        VisionPurpose.GENERAL,
        VisionPurpose.QUESTION,
    ),
# 只有调用方明确选择一个没有更小清单单元的精确 PDF 页面后，才会创建页面回退项。
# 在读取前其语义未知；允许中性描述或由模型提出具体问题，不预判内容类型。
    DocumentNonTextKind.PAGE_VISUAL: (VisionPurpose.GENERAL, VisionPurpose.QUESTION),
}


@dataclass(frozen=True)
class VisualUnitRef:
    """Host 对一个像素可获取单元掌握的全部信息。"""

    unit_id: str
    kind: DocumentNonTextKind
    image_path: str
    source_sha256: str
    image_sha256: str
    locator: DocumentLocator
    mime_type: str
    pixel_size: PixelSize
    byte_count: int
    # 如果提供商像素位于临时渲染中，隐式附件权威信息仍须根据原始托管来源分类。
    disclosure_source_path: str | None = None

    def __post_init__(self) -> None:
        if not self.unit_id.strip():
            raise ValueError("unit_id must not be empty")
        if not self.image_path.strip():
            raise ValueError("image_path must not be empty")
        if (
            self.disclosure_source_path is not None
            and not self.disclosure_source_path.strip()
        ):
            raise ValueError("disclosure_source_path must not be empty")
        if self.kind not in _PURPOSES_BY_KIND:
            raise ValueError(f"{self.kind.value} units are not read visually")

    @property
    def allowed_purposes(self) -> tuple[VisionPurpose, ...]:
        return _PURPOSES_BY_KIND[self.kind]

    @property
    def authorization_path(self) -> str:
        return self.disclosure_source_path or self.image_path


@dataclass(frozen=True)
class FrozenVisualToolBoundary:
    """此注册能够解析的封闭单元集合。"""

    session_id: str
    units: tuple[VisualUnitRef, ...]

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        # 空边界是合法状态，并非调用方错误：即使 Session 文档中没有未解析视觉内容，
        # 仍会获得此工具，且所有读取都会按 unit_id 被拒绝。要求至少一个单元，
        # 正是过去迫使此工具逐 Turn 构建、而工作区工具可逐 Session 构建的原因。
        ids = [unit.unit_id for unit in self.units]
        if len(ids) != len(set(ids)):
            raise ValueError("unit_id values must be unique within a boundary")

    def unit(self, unit_id: str) -> VisualUnitRef | None:
        return next((item for item in self.units if item.unit_id == unit_id), None)


__all__ = ['FrozenVisualToolBoundary', 'VisualUnitRef']
