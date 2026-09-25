"""L1 模型短引用与 Host 调用身份之间的唯一转换边界。"""

from collections.abc import Mapping
from copy import deepcopy

from ....output_protocol.actions import CallToolsAction, ToolCallProposal
from ....output_protocol.l1 import (
    L1AttemptDecision, L1AttemptDecisionProposal, L1CallReferenceProposal,
    L1ResultReference,
)
from ....persistent_turn_content.evidence import l1_calls_by_ref
from ....persistent_turn_content.findings import EXECUTION_FINDINGS_TOOL_IDS
from ....tools.tool_history.definitions import TOOL_HISTORY_TOOL_IDS


class L1ReferenceAdmissionError(ValueError):
    def __init__(self, path: str, available: tuple[str, ...]):
        self.path = path
        self.available = available
        super().__init__("L1 reference is not a successful ordinary call in this run")


def materialize_decision_references(
    proposal: L1AttemptDecisionProposal, *, execution: Mapping[str, object],
) -> L1AttemptDecision:
    calls = {
        ref: call for ref, call in l1_calls_by_ref(execution).items()
        if call.get("status") == "succeeded"
        and call.get("tool_id") not in (*EXECUTION_FINDINGS_TOOL_IDS, *TOOL_HISTORY_TOOL_IDS)
    }

    def resolve(raw: object, path: str) -> L1ResultReference:
        try:
            reference = L1CallReferenceProposal.model_validate(raw)
        except ValueError as exc:
            raise L1ReferenceAdmissionError(path, tuple(calls)[-8:]) from exc
        call = calls.get(reference.call_ref)
        if call is None:
            raise L1ReferenceAdmissionError(path + "/call_ref", tuple(calls)[-8:])
        return L1ResultReference(
            tool_call_id=call["tool_call_id"], result_sha256=call["outcome_hash"],
            chunk_id=reference.chunk_id,
        )

    references = tuple(resolve(ref, f"/references/{index}") for index, ref in enumerate(proposal.references))
    action = proposal.action
    if isinstance(action, CallToolsAction):
        bound_calls = []
        for call_index, call in enumerate(action.calls):
            arguments = call.model_dump(mode="json")["arguments"]
            if call.tool_id in EXECUTION_FINDINGS_TOOL_IDS and isinstance(arguments.get("items"), list):
                for item_index, item in enumerate(arguments["items"]):
                    if not isinstance(item, dict) or not isinstance(item.get("source_refs"), list):
                        continue
                    sources = []
                    for source_index, source in enumerate(item["source_refs"]):
                        path = f"/action/calls/{call_index}/arguments/items/{item_index}/source_refs/{source_index}"
                        ref = resolve(source, path)
                        sources.append({
                            "tool_result_id": ref.tool_call_id,
                            "result_sha256": ref.result_sha256,
                            "chunk_id": ref.chunk_id,
                        })
                    item["source_refs"] = sources
            bound_calls.append(ToolCallProposal(tool_id=call.tool_id, arguments=arguments))
        action = CallToolsAction(calls=tuple(bound_calls))
    return L1AttemptDecision(
        note=proposal.note, plan=proposal.plan, action=action, references=references,
    )


def project_findings_tool_schema(entry: dict) -> dict:
    """从现行工具合同派生 L1 的身份字段视图，其余校验仍由原合同拥有。"""
    if entry["tool_id"] not in EXECUTION_FINDINGS_TOOL_IDS:
        return entry
    projected = deepcopy(entry)
    definitions = projected["input_schema"].get("$defs", {})
    if "sourceRef" not in definitions:
        raise ValueError("findings tool omitted its source reference contract")
    source = definitions["sourceRef"]
    source["properties"].pop("tool_result_id")
    source["properties"].pop("result_sha256", None)
    source["properties"]["call_ref"] = L1CallReferenceProposal.model_json_schema()["properties"]["call_ref"]
    source["required"] = ["call_ref" if key == "tool_result_id" else key for key in source["required"]]
    projected["description"] += " L1 的 source_refs 使用 call_ref（如 c1.1），完整来源身份与结果摘要由 Host 绑定。"
    return projected
