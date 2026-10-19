"""视觉单元引用以精确 File/FileVersion 为作用域，保留原工具的来源校验。"""

from .reference_bindings import ReferenceBindings, append_binding, file_scope, object_items, rename_schema_field


def visual_records(tool_id, result):
    if tool_id == "list_file_visuals":
        for unit in result.get("visuals", []):
            yield unit, file_scope(result)
    elif tool_id == "read_file_visuals":
        for unit in result.get("results", []):
            yield unit, file_scope(unit)


def collect_visual_references(rows, tool_id, result):
    for unit, scope in visual_records(tool_id, result):
        value = unit.get("visual_unit_id")
        if value and all(scope):
            append_binding(rows, prefix="v", value=value, scope=scope)


class VisualReferences(ReferenceBindings):
    def __init__(self, rows=()):
        super().__init__("v", rows)

    def project_arguments(self, tool_id, arguments):
        if tool_id == "read_file_visuals":
            for _, item in object_items(arguments, "requests"):
                self.project_field(item, native="visual_unit_id", field="visual_ref", scope=file_scope(item))

    def materialize_arguments(self, tool_id, arguments, path):
        if tool_id == "read_file_visuals":
            for i, item in object_items(arguments, "requests"):
                self.materialize(item, native="visual_unit_id", field="visual_ref", scope=file_scope(item),
                                 path=f"{path}/requests/{i}")

    def project_result(self, tool_id, result):
        for unit, scope in visual_records(tool_id, result):
            self.project_field(unit, native="visual_unit_id", field="visual_ref", scope=scope)

    def project_schema(self, entry):
        if entry["tool_id"] == "read_file_visuals":
            target = entry["input_schema"]["properties"]["requests"]["items"]
            rename_schema_field(target, "visual_unit_id", "visual_ref", self.schema)
            entry["description"] += " 使用 list_file_visuals 返回的 visual_ref（如 v1）及对应 file_ref；不要填写 visual_unit_id。"
