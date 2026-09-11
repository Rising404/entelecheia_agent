"""模型目录只做信息投影；同一 ToolSpec 仍负责真实参数验证。"""

from copy import deepcopy
import json

import pytest
from jsonschema import Draft202012Validator

from personagraph.tools.composition.default_catalog import build_production_default_catalog_seeds
from personagraph.tools.model_interface.catalog import project_model_tool_catalog


def _catalog():
    return [seed.definition.spec.to_dict() | {"effects": [{"action": "read"}]}
            for seed in build_production_default_catalog_seeds()]


def test_catalog_keeps_current_available_tools_and_canonical_input_schema():
    original = _catalog()
    before = deepcopy(original)
    available = {"retrieve_files", "read_file_chunks", "create_output_file"}
    projected = project_model_tool_catalog(original, available_tool_ids=available)
    assert {row["tool_id"] for row in projected} == available
    by_id = {row["tool_id"]: row for row in original}
    for row in projected:
        assert set(row) == {"tool_id", "description", "input_schema"}
        assert row["input_schema"] == by_id[row["tool_id"]]["input_schema"]
        Draft202012Validator.check_schema(row["input_schema"])
    assert original == before
    assert len(json.dumps(projected)) < len(json.dumps(original))


def test_native_chunk_id_and_file_inputs_do_not_require_aliases():
    schemas = {row["tool_id"]: row["input_schema"]
               for row in project_model_tool_catalog(_catalog())}
    for tool_id, arguments in [
        ("read_file_chunks", {"targets": [{"chunk_ids": ["chunk-a"]}]}),
        ("read_file_chunks", {"targets": [{"file_id": "f", "document_version_id": "d", "chunk_sequences": [0]}]}),
        ("prepare_files", {"files": [{"path": "paper.pdf"}, {"file_id": "f"}]}),
        ("retrieve_files", {"queries": ["term"], "file_ids": ["f"]}),
    ]:
        assert not list(Draft202012Validator(schemas[tool_id]).iter_errors(arguments))
    assert list(Draft202012Validator(schemas["read_file_chunks"]).iter_errors(
        {"targets": [{"document_ref": "D1", "chunk_ids": ["chunk-a"]}]},
    ))


@pytest.mark.parametrize("tool_id", [
    "retrieve_files", "check_files_state", "prepare_files", "read_file_chunks",
    "list_tool_results", "read_tool_result",
])
def test_model_catalog_preserves_each_registered_tool_description(tool_id):
    catalog = _catalog()
    canonical = next(entry for entry in catalog if entry["tool_id"] == tool_id)
    projected, = project_model_tool_catalog(catalog, available_tool_ids={tool_id})
    assert projected["description"] == canonical["description"]


def test_model_prepare_description_explains_completed_preparation_and_reading():
    projected, = project_model_tool_catalog(
        _catalog(), available_tool_ids={"prepare_files"},
    )
    description = projected["description"]
    assert "后台处理中会挂起本次调用" in description
    assert "ready_indices 表示准备完成、已可检索" in description
    assert "不代表已经读取正文" in description
    assert "无需重复 prepare_files" in description
    assert "terminal" in description
    assert "原样重试不会恢复" in description
