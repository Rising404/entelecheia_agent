"""台账条目引用：首次持久 sequence 决定编号，修订、撤回和 FIFO 不重编号。"""

from ....persistent_turn_content.findings import EXECUTION_FINDINGS_TOOL_IDS
from .reference_bindings import ReferenceBindings, append_binding, object_items, rename_schema_field


def build_entry_reference_rows(entry_revisions):
    bindings = {}
    for entry in sorted(entry_revisions, key=lambda item: item.sequence):
        append_binding(bindings, prefix="e", value=entry.entry_id, ordinal=entry.sequence)
    return list(bindings.values())


class EntryReferences(ReferenceBindings):
    def __init__(self, rows=()):
        super().__init__("e", rows)

    def project_arguments(self, tool_id, arguments):
        if tool_id == "revise_execution_finding":
            for _, item in object_items(arguments, "items"):
                self.project_field(item, native="entry_id", field="entry_ref")

    def materialize_arguments(self, tool_id, arguments, path):
        if tool_id == "revise_execution_finding":
            for i, item in object_items(arguments, "items"):
                self.materialize(item, native="entry_id", field="entry_ref", path=f"{path}/items/{i}")

    def project_result(self, tool_id, result):
        if tool_id in EXECUTION_FINDINGS_TOOL_IDS and "affected_entry_ids" in result:
            result["affected_entry_refs"] = [self.project(value) for value in result.pop("affected_entry_ids")]

    def project_findings(self, findings):
        if isinstance(findings, dict):
            for note in findings.get("notes", []):
                self.project_field(note, native="entry_id", field="entry_ref")

    def project_schema(self, entry):
        if entry["tool_id"] == "revise_execution_finding":
            for name in ("supersedeItem", "retractItem"):
                rename_schema_field(entry["input_schema"]["$defs"][name], "entry_id", "entry_ref", self.schema)
            entry["description"] += " 目标使用当前笔记或回执中的 entry_ref（如 e1）；原生 entry_id 由 Host 绑定。"
