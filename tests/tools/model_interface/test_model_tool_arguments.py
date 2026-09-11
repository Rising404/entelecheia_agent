"""公开参数只有原生一种形态；投影不修改用户内容。"""

from personagraph.tools.model_interface import project_tool_arguments


def test_native_arguments_are_copied_without_identity_or_body_rewrites():
    original = {"files": [{"file_id": "f", "file_version_id": "v"}],
                "content": {"document_ref": "user text", "content_sha256": "literal"}}
    projected = project_tool_arguments("prepare_files", original)
    assert projected == original
    assert projected is not original
    projected["files"][0]["file_id"] = "another"
    assert original["files"][0]["file_id"] == "f"
