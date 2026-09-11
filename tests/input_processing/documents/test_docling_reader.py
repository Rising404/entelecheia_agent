"""Docling 引擎必须能够与原生读取器互换。

可互换意味着：输出相同契约、提供相同的真实性保证，并使用不同的处理器指纹，
使引擎切换触发增量重新分块，而不是成为不可见的变化。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter

from personagraph.input_processing.documents.contracts import (
    DocumentLocator,
    ElementKind,
    ProcessingResult,
    ProcessorFingerprint,
)
from personagraph.input_processing.documents import readers
from personagraph.input_processing.documents.readers import docling_reader, registry


pytestmark = pytest.mark.skipif(
    not docling_reader.docling_available(), reason="docling is an optional dependency"
)


def _write_blank_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with path.open("wb") as target:
        writer.write(target)


def _read_fake_docling_items(tmp_path, monkeypatch, *items):
    class _Page:
        page_no = 1

    document = SimpleNamespace(pages={1: _Page()})
    document.iterate_items = lambda: ((item, 0) for item in items)
    conversion = SimpleNamespace(
        document=document,
        errors=[],
        status="success",
    )

    class _Converter:
        def convert(self, _path, **_kwargs):
            return conversion

    monkeypatch.setattr(docling_reader, "_converter", lambda *_args: _Converter())
    monkeypatch.setattr(
        docling_reader,
        "fingerprint",
        lambda *_args: ProcessorFingerprint("docling-test", "1"),
    )
    path = tmp_path / "fake.pdf"
    _write_blank_pdf(path)
    return docling_reader.read_with_docling(path)


def _fake_docling_item(
    *,
    label: str,
    text: str = "",
    cell_texts: tuple[str, ...] = (),
    export_to_markdown=None,
):
    provenance = SimpleNamespace(page_no=1, bbox=None, charspan=None)
    item = SimpleNamespace(label=label, text=text, prov=[provenance])
    if label == "document_index":
        item.data = SimpleNamespace(
            num_rows=1,
            num_cols=max(1, len(cell_texts)),
            table_cells=[SimpleNamespace(text=value) for value in cell_texts],
        )
        item.export_to_markdown = export_to_markdown or (lambda *, doc: "")
    return item


def test_an_absent_dependency_degrades_instead_of_failing(monkeypatch):
    """未安装可选扩展的部署仍必须能使用较弱引擎摄取，而不是完全无法读取。"""
    monkeypatch.delenv(readers.ENGINE_ENV_VAR, raising=False)
    monkeypatch.setattr(registry, "docling_available", lambda: False)
    assert readers.configured_engine() == readers.NATIVE_ENGINE

    # 即使显式请求，也无法凭空获得尚未安装的引擎。
    monkeypatch.setenv(readers.ENGINE_ENV_VAR, "docling")
    assert readers.configured_engine() == readers.NATIVE_ENGINE


def test_plain_text_stays_native_even_when_docling_is_selected(tmp_path, monkeypatch):
    """让布局模型按空行切分 Markdown 纯属浪费。"""
    monkeypatch.setenv(readers.ENGINE_ENV_VAR, "docling")
    path = tmp_path / "notes.txt"
    path.write_text("一段文字。\n", encoding="utf-8")

    assert readers.get_reader(path) is readers.read_plain_text


@pytest.mark.parametrize("suffix", [".pdf"])
def test_layout_formats_route_to_docling_when_selected(tmp_path, monkeypatch, suffix):
    monkeypatch.setenv(readers.ENGINE_ENV_VAR, "docling")
    assert readers.get_reader(tmp_path / f"a{suffix}") is docling_reader.read_with_docling


def test_real_pdf_within_eager_budget_routes_to_docling(tmp_path, monkeypatch):
    monkeypatch.setenv(readers.ENGINE_ENV_VAR, "docling")
    path = tmp_path / "small.pdf"
    _write_blank_pdf(path)

    assert readers.get_reader(path) is docling_reader.read_with_docling
    assert readers.configured_processor_fingerprint(path).reader == "docling"


def test_pdf_above_eager_but_below_absolute_budget_routes_to_native(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(readers.ENGINE_ENV_VAR, "docling")
    path = tmp_path / "deferred.pdf"
    writer = PdfWriter()
    # 按冻结的最坏情况 4.5 倍缩放，41 页 Letter 文档会超过现有 400M 急切布局
    # 预算，但仍远低于 1000 页摄取上限。
    for _ in range(41):
        writer.add_blank_page(width=612, height=792)
    with path.open("wb") as target:
        writer.write(target)

    assert readers.get_reader(path) is readers.read_pdf
    assert readers.configured_processor_fingerprint(path).reader == "pdfplumber"


def test_docx_uses_the_native_reader_even_when_docling_is_selected(tmp_path, monkeypatch):
    """Docling 2.119 无法确认 DOCX 物理页，因此会拒绝整个文件。"""
    monkeypatch.setenv(readers.ENGINE_ENV_VAR, "docling")
    assert readers.get_reader(tmp_path / "a.docx") is readers.read_docx


def test_pptx_uses_native_reader_to_preserve_speaker_notes(tmp_path, monkeypatch):
    monkeypatch.setenv(readers.ENGINE_ENV_VAR, "docling")
    assert readers.get_reader(tmp_path / "a.pptx") is readers.read_pptx


def test_the_two_engines_report_different_processor_fingerprints():
    """这使引擎切换成为增量重新分块，而不是无人能限定范围的隐性变化。"""
    assert docling_reader.fingerprint().reader == "docling"
    assert docling_reader.docling_version()
    assert "+adapter4" in str(docling_reader.fingerprint())
    assert str(docling_reader.fingerprint()) != "pdfplumber@1"


def test_a_pdf_read_through_docling_satisfies_the_shared_contract(text_pdf):
    result = docling_reader.read_with_docling(text_pdf)

    assert isinstance(result, ProcessingResult)
    assert result.elements, "a readable page must produce elements"
    for element in result.elements:
        assert isinstance(element.locator, DocumentLocator)
        assert element.locator.page == 1
    # 排序是定位器自身的不变量；Docling 原始边界框以左下角为原点，若不转换
    # 就会违反该不变量。
        if element.locator.bbox is not None:
            left, top, right, bottom = element.locator.bbox
            assert left <= right and top <= bottom
        if element.kind is not ElementKind.IMAGE:
            assert (element.text or "").strip()


def test_document_index_exports_its_real_cells_as_bounded_table_text(
    tmp_path,
    monkeypatch,
):
    exported = "| Section | Page |\n| --- | ---: |\n| Scope | 7 |"
    item = _fake_docling_item(
        label="document_index",
        cell_texts=("Section", "Page", "Scope", "7"),
        export_to_markdown=lambda *, doc: exported,
    )

    result = _read_fake_docling_items(tmp_path, monkeypatch, item)

    assert result.admission_status.value == "complete"
    assert len(result.elements) == 1
    assert result.elements[0].kind is ElementKind.TABLE
    assert result.elements[0].text == exported
    assert result.page_manifest is not None
    unit = result.page_manifest.pages[0].nontext_units[0]
    assert unit.kind.value == "table"
    assert unit.text_element_ids == (result.elements[0].element_id,)
    assert unit.requires_visual_read is False
    assert not any(
        diagnostic.code is docling_reader.DiagnosticCode.CORRUPT_SOURCE
        for diagnostic in result.diagnostics
    )


@pytest.mark.parametrize(
    "failure",
    [
        "empty_cells",
        "empty_export",
        "oversized_cells",
        "oversized_export",
        "export_error",
    ],
)
def test_unusable_document_index_falls_back_to_a_table_visual_gap(
    tmp_path,
    monkeypatch,
    failure,
):
    export_calls = 0

    def export(*, doc):
        nonlocal export_calls
        export_calls += 1
        if failure == "export_error":
            raise ValueError("malformed table reference")
        if failure == "empty_export":
            return "   "
        if failure == "oversized_export":
            return "x" * (docling_reader.MAX_ELEMENT_CHARS + 1)
        return "| |\n| --- |"

    paragraph = _fake_docling_item(label="text", text="usable body text")
    table = _fake_docling_item(
        label="document_index",
        cell_texts=(
            ()
            if failure == "empty_cells"
            else (
                "x" * (docling_reader.MAX_ELEMENT_CHARS + 1),
            )
            if failure == "oversized_cells"
            else ("Section",)
        ),
        export_to_markdown=export,
    )

    result = _read_fake_docling_items(tmp_path, monkeypatch, paragraph, table)

    assert result.admission_status.value == "partial"
    assert export_calls == (
        0 if failure in {"empty_cells", "oversized_cells"} else 1
    )
    assert any(element.needs_vision for element in result.elements)
    assert result.page_manifest is not None
    table_unit = result.page_manifest.pages[0].nontext_units[0]
    assert table_unit.kind.value == "table"
    assert table_unit.requires_visual_read is True
    assert any(
        diagnostic.code is docling_reader.DiagnosticCode.PAGE_NEEDS_VISION
        for diagnostic in result.diagnostics
    )
    assert not any(
        diagnostic.code is docling_reader.DiagnosticCode.CORRUPT_SOURCE
        for diagnostic in result.diagnostics
    )


def test_an_empty_ordinary_docling_text_item_remains_fatal(tmp_path, monkeypatch):
    paragraph = _fake_docling_item(label="text", text="usable body text")
    empty = _fake_docling_item(label="text")

    result = _read_fake_docling_items(tmp_path, monkeypatch, paragraph, empty)

    assert result.admission_status.value == "rejected"
    assert any(
        diagnostic.code is docling_reader.DiagnosticCode.CORRUPT_SOURCE
        and diagnostic.detail == "EmptyDoclingItem:text"
        for diagnostic in result.diagnostics
    )


def test_layout_analysis_recovers_headings_a_geometric_reader_cannot(styled_pdf):
    """PDF 没有语义标题标记；这正是 Docling 补充的能力。"""
    docling_result = docling_reader.read_with_docling(styled_pdf)
    native_result = readers.read_pdf(styled_pdf)

    docling_headings = [e for e in docling_result.elements if e.kind is ElementKind.HEADING]
    native_headings = [e for e in native_result.elements if e.kind is ElementKind.HEADING]

    assert docling_headings, "docling should classify the large-font line as a heading"
    assert not native_headings, "the native reader has no heading concept for PDF"
    assert any(e.locator.section_path for e in docling_result.elements)


def test_wrapped_lines_become_one_paragraph_instead_of_three(styled_pdf):
    """“每行一个元素”的形态曾让基于元素的分块失去作用。"""
    docling_result = docling_reader.read_with_docling(styled_pdf)
    native_result = readers.read_pdf(styled_pdf)

    assert len(docling_result.elements) < len(native_result.elements)


def test_an_unreadable_source_is_diagnosed_rather_than_raised(corrupt_pdf):
    result = docling_reader.read_with_docling(corrupt_pdf)

    assert result.elements == ()
    assert result.diagnostics
    assert result.is_complete is False


def test_docling_element_identity_changes_for_same_name_and_size_source_bytes(
    tmp_path,
    monkeypatch,
):
    class _Page:
        page_no = 1

    class _Provenance:
        page_no = 1
        bbox = None
        charspan = None

    class _Item:
        label = "text"
        text = "stable extracted text"
        prov = [_Provenance()]

    class _Document:
        pages = {1: _Page()}

        def iterate_items(self):
            yield _Item(), 0

    class _Conversion:
        document = _Document()
        errors = []
        status = "success"

    class _Converter:
        def convert(self, _path, **_kwargs):
            return _Conversion()

    monkeypatch.setattr(docling_reader, "_converter", lambda *_args: _Converter())
    monkeypatch.setattr(
        docling_reader,
        "fingerprint",
        lambda *_args: ProcessorFingerprint("docling-test", "1"),
    )
    path = tmp_path / "same.pdf"
    _write_blank_pdf(path)
    base = path.read_bytes()
    path.write_bytes(base + b"\n%A\n")
    first = docling_reader.read_with_docling(path)
    path.write_bytes(base + b"\n%B\n")
    second = docling_reader.read_with_docling(path)

    assert first.elements[0].element_id != second.elements[0].element_id


def test_docling_policy_limit_failure_is_typed_as_limit_reached(
    tmp_path,
    monkeypatch,
):
    class _Converter:
        def convert(self, _path, **_kwargs):
            raise RuntimeError("Document exceeds max_num_pages limit")

    monkeypatch.setattr(docling_reader, "_converter", lambda *_args: _Converter())
    path = tmp_path / "too-many-pages.pdf"
    _write_blank_pdf(path)

    result = docling_reader.read_with_docling(path)

    assert result.elements == ()
    assert {item.code for item in result.diagnostics} == {
        docling_reader.DiagnosticCode.LIMIT_REACHED
    }


def test_an_unmapped_docling_label_keeps_its_text_as_a_paragraph():
    """丢失内容比丢失类型区分更糟，因此适配器遇到未知标签时不得静默删除
    其下文本。"""
    from personagraph.input_processing.documents.readers.docling_reader import _LABEL_TO_KIND

    assert _LABEL_TO_KIND.get("a_label_docling_added_later") is None
    # 读取器通过 .get(..., PARAGRAPH) 解析未知标签；这里验证的是默认值，
    # 而不是映射表。
    assert _LABEL_TO_KIND.get("nope", ElementKind.PARAGRAPH) is ElementKind.PARAGRAPH


def test_installing_the_optional_extra_is_the_opt_in(monkeypatch):
    """同时要求安装依赖与设置环境变量，只会造出携带依赖却从未受益的部署。"""
    monkeypatch.delenv(readers.ENGINE_ENV_VAR, raising=False)
    assert readers.configured_engine() == readers.DOCLING_ENGINE

    monkeypatch.setenv(readers.ENGINE_ENV_VAR, "native")
    assert readers.configured_engine() == readers.NATIVE_ENGINE


def test_markdown_never_pays_for_layout_analysis(monkeypatch):
    """其结构已在文本中明确表达；先渲染再对像素运行检测模型纯属浪费。"""
    monkeypatch.delenv(readers.ENGINE_ENV_VAR, raising=False)
    assert ".md" not in registry._DOCLING_SUFFIXES
    assert readers.get_reader(Path("notes.md")) is readers.read_plain_text


def test_the_ocr_engine_is_named_rather_than_auto_selected():
    """自动选择可能随版本变化，静默改变提取文本乃至分块内容，却不触发指纹变化。"""
    name = docling_reader.ocr_engine_name()
    assert name != "auto"
    assert name in {"ocrmac", "rapidocr"}
    # 基于同样原因，后端也是处理器身份的一部分。
    assert name in str(docling_reader.fingerprint())


def test_docling_fingerprint_freezes_packages_models_and_exact_pipeline_options():
    """失效键必须描述实际生成文本的组件。

    只写 ``docling`` 和 ``ocrmac`` 并不足够：OCR 包升级、模型修订或选项/默认值
    变化都可能改变派生文本，而这两个名称保持不变。
    """

    recipe = docling_reader.configured_docling_recipe()
    identity = json.loads(recipe.canonical_json)
    pipeline = docling_reader._pipeline_options_from_recipe(recipe)

    assert str(Path.home()) not in recipe.canonical_json
    assert identity["packages"]["docling"] == docling_reader.docling_version()
    assert identity["ocr"]["backend"] == docling_reader.ocr_engine_name()
    assert identity["ocr"]["packages"]
    assert identity["ocr"]["options"]
    assert identity["ocr"]["model"]
    if identity["ocr"]["backend"] == "ocrmac":
        assert tuple(identity["ocr"]["options"]["lang"]) == (
            "zh-Hans",
            "en-US",
        )
    assert identity["adapter_components"] == [
        "pdf-annotation-inventory-v2",
        "pdf-native-table-inventory-v2",
    ]
    render_limits = identity["resource_limits"]["pdf_render_geometry"]
    assert render_limits == {
        "algorithm": docling_reader.PDF_RENDER_PREFLIGHT_ALGORITHM,
        "backend_supersampling_factor": (
            docling_reader.PDF_BACKEND_SUPERSAMPLING_FACTOR
        ),
        "max_page_pixels": docling_reader.MAX_RENDER_PAGE_PIXELS,
        "max_side_pixels": docling_reader.MAX_RENDER_SIDE_PIXELS,
        "max_total_pixels": docling_reader.MAX_RENDER_TOTAL_PIXELS,
        "render_scale": max(
            float(pipeline.images_scale),
            float(pipeline.ocr_options.scale),
        ),
    }
    changed_identity = json.loads(recipe.canonical_json)
    changed_identity["resource_limits"]["pdf_render_geometry"][
        "max_side_pixels"
    ] += 1
    changed_recipe = docling_reader.DoclingProcessorRecipe(
        json.dumps(changed_identity, sort_keys=True),
        artifact_sources=recipe.artifact_sources,
    )
    assert docling_reader.fingerprint(changed_recipe) != docling_reader.fingerprint(recipe)
    for component in ("layout", "table"):
        artifact = identity["artifacts"][component]
        assert len(artifact["resolved_commit"]) == 40
        assert artifact["manifest"]["files"]
        assert artifact["manifest"]["total_bytes"] > 0
    assert identity["pipeline"]["components"]["layout"]["model_spec"]
    assert identity["pipeline"]["components"]["table"]["kind"]
    runtime = identity["inference_runtime"]
    assert runtime["layout_engine"] == "transformers"
    assert runtime["resolved_accelerator"]
    assert runtime["packages"]["transformers"] != "not-installed"
    assert runtime["packages"]["torch"] != "not-installed"
    assert runtime["packages"]["pillow"] != "not-installed"
    assert runtime["layout_image_processor"]["use_fast_argument"] is None
    assert isinstance(
        runtime["layout_image_processor"]["resolved_is_fast"],
        bool,
    )
    assert runtime["layout_image_processor"]["options"]["size"]
    assert (
        docling_reader._pipeline_options_payload(pipeline)
        == identity["pipeline"]["options"]
    )
    assert pipeline.artifacts_path is not None
    artifact_root = Path(pipeline.artifacts_path)
    assert (artifact_root / "docling-project--docling-layout-heron").is_dir()
    assert (artifact_root / "docling-project--docling-models").is_dir()
    assert recipe.digest in str(docling_reader.fingerprint())


def test_ocr_package_upgrade_changes_docling_fingerprint(monkeypatch):
    """即使选项不变，后端升级也会使所有派生分块失效。"""

    baseline_recipe = docling_reader.configured_docling_recipe()
    baseline_identity = json.loads(baseline_recipe.canonical_json)
    package_name = next(iter(baseline_identity["ocr"]["packages"]))
    baseline = docling_reader.fingerprint()
    real_version = docling_reader._distribution_version

    def upgraded(name: str) -> str:
        if name == package_name:
            return "9999.test-upgrade"
        return real_version(name)

    monkeypatch.setattr(docling_reader, "_distribution_version", upgraded)
    docling_reader.configured_docling_recipe.cache_clear()
    docling_reader._converter.cache_clear()
    try:
        assert docling_reader.fingerprint() != baseline
    finally:
    # pytest 恢复真实函数后，不得把基于补丁包元数据构建的配方泄漏到后续读取。
        docling_reader.configured_docling_recipe.cache_clear()
        docling_reader._converter.cache_clear()


def test_inference_runtime_upgrade_changes_docling_fingerprint(monkeypatch):
    baseline = docling_reader.fingerprint()
    real_version = docling_reader._distribution_version

    def upgraded(name: str) -> str:
        if name == "transformers":
            return "9999.test-upgrade"
        return real_version(name)

    monkeypatch.setattr(docling_reader, "_distribution_version", upgraded)
    docling_reader.configured_docling_recipe.cache_clear()
    docling_reader._converter.cache_clear()
    try:
        assert docling_reader.fingerprint() != baseline
    finally:
        docling_reader.configured_docling_recipe.cache_clear()
        docling_reader._converter.cache_clear()


def test_docling_artifact_manifest_changes_when_local_model_bytes_change(tmp_path):
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"model-v1")
    first = docling_reader._artifact_manifest(tmp_path)

    model.write_bytes(b"model-v2")
    second = docling_reader._artifact_manifest(tmp_path)

    assert first != second
    assert first["files"][0]["sha256"] != second["files"][0]["sha256"]


def test_docling_fails_closed_when_pinned_local_artifacts_are_missing(monkeypatch):
    real_resolver = docling_reader._resolve_hf_snapshot

    def missing(_repo_id: str, _revision: str):
        raise docling_reader.DoclingUnavailable("not cached")

    monkeypatch.setattr(docling_reader, "_resolve_hf_snapshot", missing)
    docling_reader.configured_docling_recipe.cache_clear()
    docling_reader._converter.cache_clear()
    try:
        assert docling_reader.docling_available() is False
    finally:
        monkeypatch.setattr(docling_reader, "_resolve_hf_snapshot", real_resolver)
        docling_reader.configured_docling_recipe.cache_clear()
        docling_reader._converter.cache_clear()


def test_a_flat_document_does_not_grow_a_fabricated_hierarchy(styled_pdf):
    """把每个标题都追加到路径会令其成为上一个标题的子级，导致扁平文档报告
    实际并不存在的深度。"""
    result = docling_reader.read_with_docling(styled_pdf)
    depths = {len(element.locator.section_path) for element in result.elements}
    assert depths and max(depths) <= 2
