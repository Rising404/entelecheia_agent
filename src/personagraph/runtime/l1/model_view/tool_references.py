"""组合各工具领域的短引用；冻结快照由持久台账和已校验回执重建。"""

from copy import deepcopy
import hashlib
import json

from ....persistent_turn_content.evidence import l1_calls_by_ref, parse_l1_call_ref
from .cursor_references import CURSOR_TOOLS, CursorReferences, collect_cursor_references
from .entry_references import EntryReferences, build_entry_reference_rows
from .visual_references import VisualReferences, collect_visual_references


def build_tool_reference_catalog(execution, *, entry_revisions=()):
    visuals, cursors = {}, {}
    calls = l1_calls_by_ref(execution)
    for ref in sorted(calls, key=parse_l1_call_ref):
        call = calls[ref]
        tool_id = call["tool_id"]
        if call.get("status") != "succeeded" or tool_id not in CURSOR_TOOLS | {"read_file_visuals"}:
            continue
        raw = call.get("outcome_json")
        if not isinstance(raw, str) or hashlib.sha256(raw.encode()).hexdigest() != call.get("outcome_hash"):
            raise ValueError("tool reference source outcome hash mismatch")
        outcome = json.loads(raw)
        if outcome.get("status") != call["status"] or not isinstance(outcome.get("result"), dict):
            raise ValueError("tool reference source outcome is invalid")
        collect_visual_references(visuals, tool_id, outcome["result"])
        collect_cursor_references(cursors, tool_id, outcome["result"])
    return {"entries": build_entry_reference_rows(entry_revisions),
            "visuals": list(visuals.values()), "cursors": list(cursors.values())}


class ToolReferences:
    def __init__(self, catalog=None):
        catalog = catalog or {}
        self.entries = EntryReferences(catalog.get("entries", ()))
        self.visuals = VisualReferences(catalog.get("visuals", ()))
        self.cursors = CursorReferences(catalog.get("cursors", ()))
        self.domains = (self.entries, self.visuals, self.cursors)

    def project_arguments(self, tool_id, arguments):
        result = deepcopy(arguments)
        for domain in self.domains:
            domain.project_arguments(tool_id, result)
        return result

    def materialize_arguments(self, tool_id, arguments, *, path):
        result = deepcopy(arguments)
        for domain in self.domains:
            domain.materialize_arguments(tool_id, result, path)
        return result

    def project_result(self, tool_id, result):
        result = deepcopy(result)
        if isinstance(result, dict):
            for domain in self.domains:
                domain.project_result(tool_id, result)
        return result

    def project_schema(self, entry):
        entry = deepcopy(entry)
        for domain in self.domains:
            domain.project_schema(entry)
        return entry
