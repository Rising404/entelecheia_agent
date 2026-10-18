"""阅读准入拒绝不完整游标和冻结来源循环，保留正常续读与变化来源。"""

import pytest

from personagraph.output_protocol.actions import CallToolsAction
from personagraph.runtime.l1.identity import canonical_json, sha256_json
from personagraph.runtime.l1.read_admission import check_read_admission


_CATALOG = tuple(
    {
        "tool_id": tool,
        "input_schema": {
            "properties": {
                "element_offset": {"default": 0},
                "character_offset": {"default": 0},
                "max_chars": {"default": 1000},
                "max_elements": {"default": 3},
            }
        },
    }
    for tool in ("read_pdf_text", "read_text")
)


def _action(tool="read_pdf_text", **arguments):
    return CallToolsAction(calls=({"tool_id": tool, "arguments": arguments},))


def _history(action, *, changed=False, status="succeeded"):
    call = action.calls[0]
    calls = []
    for i in (1, 2):
        outcome = {
            "result": {
                "elements": [
                    {"text": f"fact {i if changed else 0}", "locator": {"page": 1}}
                ],
                "coverage": {"physical_pages": 2, "selected_pages": [1, 2]},
                "next_cursor": {"element_offset": 3, "character_offset": 0},
                "truncated": True,
            },
            "error": None,
        }
        calls.append(
            {
                "tool_call_id": f"call-{i}",
                "attempt_id": f"step-{i}",
                "call_ordinal": 1,
                "tool_id": call.tool_id,
                "arguments_json": canonical_json(call.arguments),
                "arguments_hash": sha256_json(call.arguments),
                "status": status,
                "outcome_json": canonical_json(outcome),
                "outcome_hash": sha256_json(outcome),
            }
        )
    return {
        "attempts": [{"attempt_id": f"step-{i}", "ordinal": i} for i in (1, 2)],
        "tool_calls": calls,
    }


def test_cursor_requires_both_coordinates_and_accepts_explicit_zero_element():
    issue = check_read_admission(
        _action("read_text", path="file.txt", character_offset=600),
        execution={"tool_calls": []},
        tool_catalog=_CATALOG,
        attachment_catalog=(),
    )
    assert issue.code == "host_guard.l1_incomplete_read_cursor"
    assert "整个文件偏移" in issue.safe_explanation
    assert (
        check_read_admission(
            _action(
                "read_text", path="file.txt", element_offset=0, character_offset=600
            ),
            execution={"tool_calls": []},
            tool_catalog=_CATALOG,
            attachment_catalog=(),
        )
        is None
    )


def test_third_identical_frozen_attachment_read_is_repaired_before_dispatch():
    action = _action(path="attachments/paper.pdf", start_page=1, end_page=2)
    issue = check_read_admission(
        action,
        execution=_history(action),
        tool_catalog=_CATALOG,
        attachment_catalog=({"relative_path": "attachments/paper.pdf"},),
    )
    assert issue.code == "host_guard.l1_repeated_read_without_progress"
    assert "c2.1" in issue.safe_explanation
    assert "next_cursor" in issue.safe_explanation
    assert issue.paths == ("/action/calls/0/arguments",)


@pytest.mark.parametrize("change", ["workspace", "content", "offset", "failure"])
def test_workspace_change_and_progress_are_not_treated_as_immutable_duplicates(change):
    action = _action(path="paper.pdf")
    history = _history(
        action,
        changed=change == "content",
        status="failed" if change == "failure" else "succeeded",
    )
    if change == "offset":
        action = _action(path="paper.pdf", element_offset=64)
    assert (
        check_read_admission(
            action,
            execution=history,
            tool_catalog=_CATALOG,
            attachment_catalog=(
                () if change == "workspace" else ({"relative_path": "paper.pdf"},)
            ),
        )
        is None
    )


def test_repeated_history_prefix_requires_new_path_or_offset():
    action = _action("read_tool_result", call_ref="c1.1", path="/result/elements")
    issue = check_read_admission(
        action, execution=_history(action), tool_catalog=_CATALOG, attachment_catalog=()
    )
    assert issue.code == "host_guard.l1_repeated_read_without_progress"
    assert "next_offset" in issue.safe_explanation


def test_one_review_read_is_allowed():
    action = _action("read_tool_result", call_ref="c1.1", path="/result/elements")
    history = _history(action)
    history["tool_calls"] = history["tool_calls"][:1]
    assert (
        check_read_admission(
            action, execution=history, tool_catalog=_CATALOG, attachment_catalog=()
        )
        is None
    )


def test_missing_history_path_is_repaired_instead_of_retried_unchanged():
    action = _action("read_tool_result", call_ref="c1.1", path="/result/text")
    history = _history(action, status="failed")
    for call in history["tool_calls"]:
        call["outcome_json"] = canonical_json(
            {"error": {"code": "history_path_unavailable"}}
        )
    issue = check_read_admission(
        action, execution=history, tool_catalog=_CATALOG, attachment_catalog=()
    )
    assert issue.code == "host_guard.l1_invalid_history_path_repeated"
    assert issue.paths == ("/action/calls/0/arguments/path",)
    assert "/result" in issue.safe_explanation
    for call in history["tool_calls"]:
        call["outcome_json"] = canonical_json(
            {"error": {"code": "tool_result_unavailable"}}
        )
    assert (
        check_read_admission(
            action, execution=history, tool_catalog=_CATALOG, attachment_catalog=()
        )
        is None
    )


@pytest.mark.parametrize("batch_size", [1, 8])
def test_catalog_defaults_identify_equivalent_reads_and_give_a_real_cursor(batch_size):
    original = _action(path="paper.pdf", start_page=1, end_page=2)
    explicit = _action(
        path="paper.pdf",
        start_page=1,
        end_page=2,
        element_offset=0,
        character_offset=0,
        max_chars=1000,
        max_elements=3,
    )
    explicit = CallToolsAction(calls=explicit.calls * batch_size)
    issue = check_read_admission(
        explicit,
        execution=_history(original),
        tool_catalog=_CATALOG,
        attachment_catalog=({"relative_path": "paper.pdf"},),
    )
    assert issue.code == "host_guard.l1_repeated_read_without_progress"
    assert "element_offset=3" in issue.safe_explanation
    assert "character_offset=0" in issue.safe_explanation
    assert len(issue.safe_explanation.encode("utf-8")) <= 500
    enlarged = _action(path="paper.pdf", start_page=1, end_page=2, max_chars=1001)
    assert (
        check_read_admission(
            enlarged,
            execution=_history(original),
            tool_catalog=_CATALOG,
            attachment_catalog=({"relative_path": "paper.pdf"},),
        )
        is None
    )


def test_repeat_feedback_stays_bounded_for_eight_distinct_sources():
    calls, proposed, catalog = [], [], []
    for index in range(8):
        action = _action(path=f"paper-{index}.pdf")
        history = _history(action)
        for call in history["tool_calls"]:
            call["call_ordinal"] = index + 1
            call["tool_call_id"] += f"-{index}"
            calls.append(call)
        proposed.extend(action.calls)
        catalog.append({"relative_path": f"paper-{index}.pdf"})
    issue = check_read_admission(
        CallToolsAction(calls=tuple(proposed)),
        execution={"attempts": history["attempts"], "tool_calls": calls},
        tool_catalog=_CATALOG, attachment_catalog=catalog,
    )
    assert len(issue.paths) == 8
    assert "c2.8" in issue.safe_explanation
    assert len(issue.safe_explanation.encode("utf-8")) <= 500
