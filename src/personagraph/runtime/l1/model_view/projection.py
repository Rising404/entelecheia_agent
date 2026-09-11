"""L1 模型输入的业务投影；使用原生身份，不创建短引用表。"""

from __future__ import annotations
from copy import deepcopy
from ....tools.model_interface import (
    project_file_record,
    project_model_tool_catalog,
    project_tool_arguments,
    project_tool_result,
    project_tool_result_metadata,
)


def project_attempt_view(payload: dict) -> dict:
    view = {
        key: deepcopy(payload[key])
        for key in (
            "current_user_text",
            "history_pairs",
            "session_summary",
            "execution_limits",
            "attempt_ordinal",
            "attachments",
        )
        if key in payload
    }
    view["file_catalog"] = [
        project_file_record(item) for item in payload.get("file_catalog", [])
    ]
    view["plan"] = project_plan(payload.get("plan"))
    view["prior_tool_results"] = [
        project_result_record(item) for item in payload.get("prior_tool_results", [])
    ]
    view["execution_findings"] = project_findings(payload.get("execution_findings"))
    view["tool_catalog"] = project_model_tool_catalog(payload.get("tool_catalog", []))
    report = payload.get("tool_result_projection") or {}
    view["tool_result_projection"] = {
        key: report[key]
        for key in (
            "projected_tool_result_count",
            "omitted_tool_result_count",
            "durable_tool_result_count",
        )
        if key in report
    }
    feedback = payload.get("verification_feedback")
    if isinstance(feedback, dict):
        view["verification_feedback"] = {
            key: deepcopy(feedback[key])
            for key in (
                "source",
                "feedback",
                "candidate_final_reply",
            )
            if key in feedback
        }
        details = feedback.get("details") or {}
        issues = (details.get("reviewer_result") or {}).get(
            "issues", details.get("issues")
        )
        if issues is not None:
            view["verification_feedback"]["issues"] = [
                {
                    key: deepcopy(value)
                    for key, value in issue.items()
                    if value is not None
                }
                for issue in issues
            ]
    return view


def project_plan(plan: object) -> object:
    if not isinstance(plan, dict):
        return deepcopy(plan)
    return {
        "objective": plan.get("objective"),
        "acceptances": [
            {"acceptance_id": item["acceptance_id"], "criterion": item["criterion"]}
            for item in plan.get("acceptances", [])
        ],
    }


def project_result_record(result: dict) -> dict:
    projected = {
        key: deepcopy(result[key])
        for key in (
            "tool_result_id",
            "tool_id",
            "status",
            "attempt_ordinal",
            "call_ordinal",
            "error",
            "result_partially_compacted",
        )
        if key in result
    }
    tool_id = result.get("tool_id")
    if "arguments" in result:
        projected["arguments"] = project_tool_arguments(tool_id, result["arguments"])
    argument_scope = result.get("arguments_projection")
    if isinstance(argument_scope, dict):
        projected["arguments_projection"] = {
            key: deepcopy(argument_scope[key])
            for key in ("complete", "redacted_value_count", "truncated_value_count")
            if key in argument_scope
        }
    if "result" in result:
        projected["result"] = project_tool_result(tool_id, result["result"])
    metadata = project_tool_result_metadata(result.get("metadata"))
    if metadata:
        projected["metadata"] = metadata
    return projected


def project_findings(findings: object) -> object:
    if not isinstance(findings, dict):
        return deepcopy(findings)
    return {
        key: deepcopy(findings[key])
        for key in (
            "notes",
            "active_entry_count",
            "omitted_active_count",
            "display_note_count",
        )
        if key in findings
    }


__all__ = [
    "project_attempt_view",
    "project_plan",
    "project_result_record",
    "project_findings",
]
