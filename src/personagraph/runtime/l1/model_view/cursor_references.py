"""两种不透明分页游标的短引用；文本读取的结构化游标不经过此模块。"""

from .reference_bindings import ReferenceBindings, append_binding, file_scope, rename_schema_field

CURSOR_TOOLS = {"list_file_visuals", "list_workspace_directory"}


def cursor_scope(tool_id, row):
    return (tool_id, *file_scope(row)) if tool_id == "list_file_visuals" else (tool_id,)


def collect_cursor_references(rows, tool_id, result):
    value = result.get("next_cursor")
    if tool_id in CURSOR_TOOLS and isinstance(value, str) and value:
        append_binding(rows, prefix="p", value=value, scope=cursor_scope(tool_id, result))


class CursorReferences(ReferenceBindings):
    def __init__(self, rows=()):
        super().__init__("p", rows)

    def project_arguments(self, tool_id, arguments):
        if tool_id in CURSOR_TOOLS:
            self.project_field(arguments, native="cursor", field="cursor_ref", scope=cursor_scope(tool_id, arguments))

    def materialize_arguments(self, tool_id, arguments, path):
        if tool_id in CURSOR_TOOLS:
            self.materialize(arguments, native="cursor", field="cursor_ref", scope=cursor_scope(tool_id, arguments), path=path)

    def project_result(self, tool_id, result, *, source=None):
        if tool_id in CURSOR_TOOLS:
            scope = cursor_scope(tool_id, source if source is not None else result)
            for native, field in (("next_cursor", "next_cursor_ref"), ("cursor", "cursor_ref")):
                self.project_field(result, native=native, field=field, scope=scope)

    def project_schema(self, entry):
        if entry["tool_id"] in CURSOR_TOOLS:
            rename_schema_field(entry["input_schema"], "cursor", "cursor_ref", self.schema)
            entry["description"] = entry["description"].replace("cursor", "cursor_ref")
            entry["description"] += " 分页时将 next_cursor_ref（如 p1）传入 cursor_ref；Host 恢复完整游标，原范围与快照校验继续生效。"
