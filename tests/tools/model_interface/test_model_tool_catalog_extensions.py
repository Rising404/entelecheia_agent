"""附加文档检查和视觉问答能力接入后，其原业务规则仍须通过原生身份界面。"""

from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator

from personagraph.tools.composition.default_catalog import (
    build_production_default_catalog_seeds,
)
from personagraph.tools.model_interface.catalog import project_model_tool_catalog


@pytest.fixture
def extension_schemas():
    catalog = [
        seed.definition.spec.to_dict() | {"effects": [{"action": "read"}]}
        for seed in build_production_default_catalog_seeds()
    ]
    return {
        row["tool_id"]: row["input_schema"]
        for row in project_model_tool_catalog(catalog)
    }


@pytest.mark.parametrize(
    "tool,args",
    [
        ("inspect_file_chunks", {"targets": [{"file_id": "f", "document_version_id": "dv"}]}),
        ("search_file_text", {"targets": [{"file_id": "f", "document_version_id": "dv"}], "text": "term"}),
        (
            "read_file_visuals",
            {
                "requests": [
                    {
                        "file_id": "f",
                        "file_version_id": "fv",
                        "visual_unit_id": "pdf_page:1",
                        "purpose": "question",
                        "question": "What?",
                        "detail": "high",
                        "region": "page",
                    }
                ]
            },
        ),
    ],
)
def test_extended_tool_schemas_preserve_native_identity_shape(extension_schemas, tool, args):
    schema = extension_schemas[tool]
    Draft202012Validator.check_schema(schema)
    assert list(Draft202012Validator(schema).iter_errors(args)) == []


def test_visual_question_still_requires_question_text(extension_schemas):
    visual = {
        "file_id": "f",
                        "file_version_id": "fv",
        "visual_unit_id": "pdf_page:1",
        "purpose": "question",
        "detail": "high",
        "region": "page",
    }
    assert list(
        Draft202012Validator(extension_schemas["read_file_visuals"]).iter_errors(
            {"requests": [visual]}
        )
    )
