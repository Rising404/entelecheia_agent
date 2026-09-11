"""供模型读取视觉单元的接口。

这些测试覆盖的是边界而非视觉模型：哪些单元可被引用、各单元接受哪些用途、
图片离开本机时声明的效果如何描述，以及未解析单元是否保持显式未解析状态，
而不是直接消失。
"""

from __future__ import annotations

import json

import hashlib
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from PIL import Image

from personagraph.input_processing.documents.contracts import (
    DocumentLocator,
    DocumentNonTextKind,
)
from personagraph.workspace.storage.context import current
from personagraph.workspace.files import (
    FileSource,
    WorkspaceFileAuthority,
)
from personagraph.input_processing.vision.providers import UnavailableVisionModelAdapter
from personagraph.input_processing.vision.contracts import (
    PixelSize,
    VisionCapabilitySnapshot,
    VisionObservation,
    VisionPurpose,
    VisionResult,
    VisionStatus,
)
from personagraph.tools.execution import ToolBusinessFailure
from personagraph.tools.visual.egress_policy import auto_visual_egress_receipt
from personagraph.tools.execution_context import ToolExecutionContext, tool_execution_scope
from personagraph.tools.visual.visual_observation_service import (
    VisualObservationBatch,
    VisualObservationRequest,
    VisualObservationService,
)
from personagraph.tools.visual.visual_tools import (
    FrozenVisualToolBoundary,
    VisualUnitRef,
    build_visual_tool_registrations,
)
from personagraph.session import store as session_store


def _plain(value):
    if hasattr(value, "items"):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class _StubAdapter:
    """无需查看任何内容即可作答的远程视觉提供方。"""

    transmits_externally = True

    def __init__(self, *, available: bool = True) -> None:
        self._available = available
        self.seen: list = []

    def capabilities(self) -> VisionCapabilitySnapshot:
        if not self._available:
            return UnavailableVisionModelAdapter().capabilities()
        return VisionCapabilitySnapshot(
            available=True,
            provider="stub",
            model="stub-vl",
            endpoint_identity="stub:local",
            processor_fingerprint="stub@1",
            supported_purposes=tuple(VisionPurpose),
        )

    def analyze(self, request) -> VisionResult:
        self.seen.append(request)
        return VisionResult(
            status=VisionStatus.COMPLETED,
            provider="stub",
            model="stub-vl",
            endpoint_identity="stub:local",
            processor_fingerprint="stub@1",
            input_sha256="c" * 64,
            observations=(
                VisionObservation(
                    observation_id="vo_stub",
                    kind=request.purpose.value,
                    text="Four bars, tallest on the right.",
                    uncertainty=0.3,
                ),
            ),
        )


@pytest.fixture
def picture(tmp_path: Path) -> Path:
    target = tmp_path / "figure.png"
    Image.new("RGB", (900, 600), "white").save(target)
    return target


def _ref(picture: Path, unit_id: str, kind: DocumentNonTextKind) -> VisualUnitRef:
    raw = picture.read_bytes()
    return VisualUnitRef(
        unit_id=unit_id,
        kind=kind,
        image_path=str(picture),
        source_sha256="a" * 64,
        image_sha256=hashlib.sha256(raw).hexdigest(),
        locator=DocumentLocator(page=4),
        mime_type="image/png",
        pixel_size=PixelSize(900, 600),
        byte_count=len(raw),
    )


@pytest.fixture
def boundary(picture: Path) -> FrozenVisualToolBoundary:
    return FrozenVisualToolBoundary(
        session_id="s1",
        units=(
            _ref(picture, "fig-1", DocumentNonTextKind.FIGURE),
            _ref(picture, "eq-1", DocumentNonTextKind.FORMULA),
        ),
    )




def test_default_egress_receipt_binds_source_provider_and_purpose():
    arguments = dict(
        session_id="s1", source_sha256="a" * 64,
        endpoint_identity="configured:endpoint", model="vision",
        purpose=VisionPurpose.GENERAL,
    )
    receipt = auto_visual_egress_receipt(**arguments)
    assert receipt == auto_visual_egress_receipt(**arguments)
    for name, value in (
        ("session_id", "s2"), ("source_sha256", "b" * 64),
        ("endpoint_identity", "other:endpoint"), ("model", "other-model"),
        ("purpose", VisionPurpose.CHART),
    ):
        assert auto_visual_egress_receipt(**{**arguments, name: value}) != receipt


def _tool(boundary, adapter):
    (registration,) = build_visual_tool_registrations(
        boundary, adapter=adapter
    )
    return registration


def test_an_unresolved_visual_table_may_enter_the_boundary(picture: Path):
    """结构化表格保留为文本；只有未解析的像素表格进入边界。"""

    table = _ref(picture, "tbl-1", DocumentNonTextKind.TABLE)
    assert table.allowed_purposes == (VisionPurpose.GENERAL, VisionPurpose.QUESTION)


def test_duplicate_unit_ids_are_rejected(picture: Path):
    with pytest.raises(ValueError):
        FrozenVisualToolBoundary(
            session_id="s1",
            units=(
                _ref(picture, "fig-1", DocumentNonTextKind.FIGURE),
                _ref(picture, "fig-1", DocumentNonTextKind.FIGURE),
            ),
        )


def test_the_schema_does_not_grow_with_the_corpus(picture: Path):
    """旧实现会列出每个单元，因此图像很多的会话每次调用都要付出额外代价。

    它所限定的配对仍由处理器执行，因为处理器本就需要根据边界解析 unit_id。
    在模式中再次列出单元，只是以随文档数量增长的代价换来第二次检查。
    """

    def schema_for(count: int) -> str:
        units = tuple(
            _ref(picture, f"fig-{index}", DocumentNonTextKind.FIGURE)
            for index in range(count)
        )
        tool = _tool(FrozenVisualToolBoundary(session_id="s1", units=units), _StubAdapter())
        return json.dumps(_plain(tool.spec.input_schema), sort_keys=True)

    assert schema_for(1) == schema_for(64)


def test_a_session_with_no_unresolved_visuals_still_gets_the_tool(picture: Path):
    """强制要求单元正是该工具过去每轮都必须重建的原因。

    现在可以构建空边界，因此该工具与工作区工具具有相同生命周期。对它的每次
    读取都会因 unit_id 被拒绝，与任何未知 ID 得到的拒绝相同。
    """

    tool = _tool(FrozenVisualToolBoundary(session_id="s1", units=()), _StubAdapter())
    with pytest.raises(ToolBusinessFailure) as raised:
        tool.handler({"units": [{"unit_id": "fig-1", "purpose": "chart"}]})
    assert raised.value.error.code == "unknown_unit"


def test_the_declared_schemas_are_valid(boundary):
    tool = _tool(boundary, _StubAdapter())
    Draft202012Validator.check_schema(_plain(tool.spec.input_schema))
    Draft202012Validator.check_schema(_plain(tool.spec.output_schema))


@pytest.mark.parametrize(
    "payload, accepted",
    [
        ({"units": [{"unit_id": "fig-1", "purpose": "chart"}]}, True),
        ({"units": [{"unit_id": "eq-1", "purpose": "formula"}]}, True),
        # 单元是否存在、指定用途是否适合其类型，已不再由模式判断；两者取决于
        # 冻结边界。在这里回答就必须列出每个单元，并在每次调用时承担成本。
        # 这两个问题都由处理器回答，参见下方两项测试。
        ({"units": [{"unit_id": "eq-1", "purpose": "chart"}]}, True),
        ({"units": [{"unit_id": "absent", "purpose": "chart"}]}, True),
        ({"units": [{"unit_id": "fig-1", "purpose": "chart"}] * 4}, False),
        ({"units": [{"unit_id": "fig-1", "purpose": "chart", "extra": 1}]}, False),
        ({"units": [{"unit_id": "fig-1", "purpose": "nonsense"}]}, False),
        ({"units": []}, False),
    ],
    ids=[
        "figure-as-chart", "formula-as-formula",
        "pairing-left-to-handler", "unit-left-to-handler",
        "too-many-units", "extra-field", "unknown-purpose", "empty",
    ],
)
def test_the_schema_guards_shape_and_size_only(boundary, payload, accepted):
    """模式仍负责判断无需边界即可确定的事项。"""

    tool = _tool(boundary, _StubAdapter())
    errors = list(Draft202012Validator(_plain(tool.spec.input_schema)).iter_errors(payload))
    assert (not errors) is accepted


def test_a_wrong_pairing_is_refused_by_the_handler(boundary):
    """模式不再列出单元后，由该防线负责拒绝错误配对。"""

    tool = _tool(boundary, _StubAdapter())
    with pytest.raises(ToolBusinessFailure) as raised:
        tool.handler({"units": [{"unit_id": "eq-1", "purpose": "chart"}]})
    assert raised.value.error.code == "purpose_not_allowed"


def test_an_unknown_unit_is_refused_without_touching_the_provider(boundary):
    adapter = _StubAdapter()
    tool = _tool(boundary, adapter)
    with pytest.raises(ToolBusinessFailure) as raised:
        tool.handler({"units": [{"unit_id": "absent", "purpose": "chart"}]})
    assert raised.value.error.code == "unknown_unit"
    assert adapter.seen == []


def test_the_model_never_supplies_a_path(boundary):
    """身份来自边界；提案只引用单元名称。"""

    adapter = _StubAdapter()
    tool = _tool(boundary, adapter)
    tool.handler({"units": [{"unit_id": "fig-1", "purpose": "chart"}]})

    request = adapter.seen[0]
    assert request.image_path == boundary.units[0].image_path
    assert request.source_unit_id == "fig-1"
    assert "image_path" not in _plain(tool.spec.input_schema)["properties"]


def test_the_chosen_detail_level_reaches_the_request(boundary):
    adapter = _StubAdapter()
    tool = _tool(boundary, adapter)
    tool.handler({"units": [{"unit_id": "fig-1", "purpose": "chart"}], "detail": "high"})
    assert adapter.seen[0].detail.value == "high"


def test_a_successful_read_is_projected_with_its_uncertainty(boundary):
    tool = _tool(boundary, _StubAdapter())
    output = tool.handler({"units": [{"unit_id": "fig-1", "purpose": "chart"}]})
    Draft202012Validator(_plain(tool.spec.output_schema)).validate(output)

    assert output["resolved"] == 1
    result = output["results"][0]
    assert result["status"] == "completed"
    assert result["observation"].startswith("Four bars")
    assert result["uncertainty"] == 0.3


def test_typed_service_is_the_canonical_visual_execution_path(boundary):
    adapter = _StubAdapter()
    service = VisualObservationService(
        adapter=adapter,

    )

    outcome = service.observe(
        boundary,
        (
            VisualObservationRequest(
                unit_id="fig-1",
                purpose=VisionPurpose.CHART,
            ),
        ),
    )

    assert isinstance(outcome, VisualObservationBatch)
    assert outcome.requested == 1
    assert outcome.resolved == 1
    assert outcome.results[0].observation == "Four bars, tallest on the right."
    assert outcome.to_dict()["results"][0]["status"] == "completed"
    assert len(adapter.seen) == 1


def test_adapter_without_explicit_egress_classification_is_rejected(
    boundary,
) -> None:
    class IncompleteAdapter:
        def capabilities(self):
            return UnavailableVisionModelAdapter().capabilities()

        def analyze(self, request):
            raise AssertionError("an incomplete adapter must never execute")

    with pytest.raises(TypeError, match="transmits_externally"):
        VisualObservationService(adapter=IncompleteAdapter())


def test_an_unavailable_provider_leaves_the_unit_visibly_unresolved(boundary):
    tool = _tool(boundary, UnavailableVisionModelAdapter())
    output = tool.handler({"units": [{"unit_id": "fig-1", "purpose": "chart"}]})
    Draft202012Validator(_plain(tool.spec.output_schema)).validate(output)

    assert output["resolved"] == 0
    result = output["results"][0]
    assert result["status"] == "unavailable"
    assert result["failure_code"] == "vision_provider_unavailable"
    assert "observation" not in result


def test_the_declared_effect_says_whether_a_picture_leaves_the_machine(boundary):
    """该工具最关键的事实是图片会被送往何处。"""

    remote = _tool(boundary, _StubAdapter(available=True)).effect_profile.effects[0]
    assert remote.action.value == "transmit"
    assert remote.data_egress.value == "content"

    offline = _tool(boundary, UnavailableVisionModelAdapter()).effect_profile.effects[0]
    assert offline.action.value == "read"
    assert offline.data_egress.value == "none"


def test_the_provider_identity_is_part_of_the_implementation_version(boundary):
    """指向不同模型的两个安装实例不能互换。"""

    tool = _tool(boundary, _StubAdapter())
    assert "stub@1" in tool.implementation_version


# --- 披露门禁 -------------------------------------------------------------
#
# 该门禁之所以存在，是因为配置的远程提供方会收到图片本身。这些测试覆盖关键
# 性质：未经授权的图片不只是不会发送，甚至不会从磁盘读取。




def test_source_changed_is_refused_before_provider_dispatch(
    picture: Path,
) -> None:
    boundary = FrozenVisualToolBoundary(
        session_id="s1",
        units=(_ref(picture, "fig-1", DocumentNonTextKind.FIGURE),),
    )
    adapter = _StubAdapter()
    Image.new("RGB", (900, 600), "black").save(picture)
    output = _tool(boundary, adapter).handler(
        {"units": [{"unit_id": "fig-1", "purpose": "chart"}]}
    )

    assert output["resolved"] == 0
    assert output["results"][0]["failure_code"] == "image_source_changed"
    assert adapter.seen == []


def test_external_adapter_receives_the_verified_immutable_payload(
    picture: Path,
) -> None:
    boundary = FrozenVisualToolBoundary(
        session_id="s1",
        units=(_ref(picture, "fig-1", DocumentNonTextKind.FIGURE),),
    )
    adapter = _StubAdapter()
    _tool(boundary, adapter).handler(
        {"units": [{"unit_id": "fig-1", "purpose": "chart"}]}
    )

    request = adapter.seen[0]
    prepared = request.prepared_payload
    assert prepared is not None
    assert prepared.source_sha256 == boundary.units[0].image_sha256
    assert hashlib.sha256(prepared.data).hexdigest() == prepared.sent_sha256
    assert request.image_path == boundary.units[0].image_path


@pytest.mark.parametrize("area", ["input", "output"])
@pytest.mark.parametrize("purpose", ["chart", "question"])
def test_same_session_uploads_and_agent_outputs_need_no_second_visual_click(
    area: str,
    purpose: str,
    tmp_path: Path,
    partitioned_project_state,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    session_id = session_store.create_session(
        "Entelecheia",
        working_dir=str(project_root),
    )
    directory = project_root / area
    directory.mkdir(parents=True, exist_ok=True)
    picture = directory / "figure.png"
    Image.new("RGB", (900, 600), "white").save(picture)
    with session_store.session_database_scope(session_id):
        database = current()
        assert database is not None
        WorkspaceFileAuthority(database).register_path(
            picture.relative_to(project_root).as_posix(),
            source=(
                FileSource.USER_UPLOAD
                if area == "input"
                else FileSource.AGENT_OUTPUT
            ),
        )
        boundary = FrozenVisualToolBoundary(
            session_id=session_id,
            units=(_ref(picture, "fig-1", DocumentNonTextKind.FIGURE),),
        )
        adapter = _StubAdapter()
        tool = _tool(boundary, adapter)
        request = {"unit_id": "fig-1", "purpose": purpose}
        if purpose == "question":
            request["question"] = "图中各标记表示什么？"
        with tool_execution_scope(ToolExecutionContext(
            deadline_monotonic=None, logical_tool_call_id="host-upload-observation",
        )):
            output = tool.handler({"units": [request]})

    assert output["resolved"] == 1
    assert len(adapter.seen) == 1
    assert adapter.seen[0].disclosure_receipt_id.startswith(
        "auto_visual_egress_"
    )












def test_a_local_adapter_needs_no_consent_at_all(boundary):
    """同意机制管控的是传输，而不是分析。"""

    class _LocalAdapter(_StubAdapter):
        transmits_externally = False

    adapter = _LocalAdapter()
    tool = _tool(boundary, adapter)
    output = tool.handler({"units": [{"unit_id": "fig-1", "purpose": "chart"}]})

    assert output["resolved"] == 1
    assert len(adapter.seen) == 1


# --- 区域升级 -------------------------------------------------------------
#
# 模型读取的是裁剪区域。当某些内容看起来在边缘被截断时，模型需要能自行调整，
# 而不是等人发现后手动重新裁剪。


def test_the_region_the_model_asked_for_reaches_the_request(boundary):
    adapter = _StubAdapter()
    tool = _tool(boundary, adapter)
    tool.handler({"units": [{"unit_id": "fig-1", "purpose": "chart"}], "region": "expanded"})
    assert adapter.seen[0].region.value == "expanded"


def test_a_reading_says_which_region_it_came_from(boundary):
    """否则模型无法知道自己看到的是裁剪区域。"""

    tool = _tool(boundary, _StubAdapter())
    output = tool.handler({"units": [{"unit_id": "fig-1", "purpose": "chart"}]})
    assert output["results"][0]["region"] == "detected"

    wider = tool.handler(
        {"units": [{"unit_id": "fig-1", "purpose": "chart"}], "region": "page"}
    )
    assert wider["results"][0]["region"] == "page"


def test_the_schema_offers_the_region_ladder_and_nothing_else(boundary):
    tool = _tool(boundary, _StubAdapter())
    schema = _plain(tool.spec.input_schema)
    assert schema["properties"]["region"]["enum"] == ["detected", "expanded", "page"]
    # 只看过渲染裁剪图的模型没有依据理解 PDF 点坐标，因此有意不提供原始坐标。
    assert "bbox" not in schema["properties"]


def test_an_unknown_region_is_refused(boundary):
    tool = _tool(boundary, _StubAdapter())
    with pytest.raises(ToolBusinessFailure) as raised:
        tool.handler({"units": [{"unit_id": "fig-1", "purpose": "chart"}], "region": "whole"})
    assert raised.value.error.code == "invalid_request"
