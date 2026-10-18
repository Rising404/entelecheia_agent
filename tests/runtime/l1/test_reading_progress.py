"""旧正文移出上下文后，真实范围、游标和短引用仍可查，不用笔记猜测。"""

from copy import deepcopy
import json

import pytest

from personagraph.runtime.l1.identity import canonical_json, sha256_json
from personagraph.runtime.l1.tool_context.reading_progress import (
    project_reading_progress,
)


def _call(ordinal, *, result=None, arguments=None, tool_id="read_pdf_text"):
    arguments = arguments or {
        "path": "papers/source.pdf",
        "start_page": 2,
        "end_page": 6,
    }
    outcome = {
        "result": result
        or {
            "coverage": {
                "selected_pages": [2, 3, 4, 5, 6],
                "selection_complete": False,
            },
            "elements": [{"text": "private body", "locator": {"page": 2}}],
            "truncated": True,
            "next_cursor": {"element_offset": 12, "character_offset": 0},
        },
        "error": None,
    }
    return {
        "tool_call_id": f"call-{ordinal}",
        "attempt_id": f"attempt-{ordinal}",
        "call_ordinal": 1,
        "tool_id": tool_id,
        "status": "succeeded",
        "arguments_json": canonical_json(arguments),
        "arguments_hash": sha256_json(arguments),
        "outcome_json": canonical_json(outcome),
        "outcome_hash": sha256_json(outcome),
    }


def _execution(*calls):
    return {
        "tool_calls": list(calls),
        "attempts": [
            {
                "attempt_id": call["attempt_id"],
                "ordinal": int(call["attempt_id"].split("-")[1]),
            }
            for call in calls
        ],
    }


def test_index_deduplicates_without_claiming_requested_pages_were_read():
    execution = _execution(_call(1), _call(2))
    before = deepcopy(execution)
    index = project_reading_progress(execution)
    assert index["omitted_count"] == 0
    assert len(index["results"]) == 1
    item = index["results"][0]
    assert item["call_ref"] == "c2.1"
    assert item["identical_request_count"] == 2
    assert item["returned_pages"] == [2]
    assert item["continuation"]["arguments"] == {
        "path": "papers/source.pdf",
        "start_page": 2,
        "end_page": 6,
        "element_offset": 12,
        "character_offset": 0,
    }
    assert item["read_result"]["arguments"] == {"call_ref": "c2.1", "path": "/result"}
    assert "private body" not in canonical_json(index)
    assert execution == before


def test_index_keeps_different_offsets_and_does_not_invent_eof_continuation():
    first = _call(1)
    end = _call(
        2,
        arguments={"path": "papers/source.pdf", "element_offset": 56},
        result={"elements": [], "truncated": False, "next_cursor": None},
    )
    index = project_reading_progress(_execution(first, end))
    assert len(index["results"]) == 2
    assert "continuation" not in index["results"][-1]
    assert index["results"][-1]["returned_element_count"] == 0


def test_index_bounds_size_and_marks_omission_without_renumbering():
    calls = [_call(n, arguments={"path": f"papers/{n}.pdf"}) for n in range(1, 70)]
    index = project_reading_progress(_execution(*calls))
    assert index["omitted_count"] > 0
    assert index["results"][-1]["call_ref"] == "c69.1"
    assert len(canonical_json(index).encode()) < 66_000


def test_corrupt_outcome_is_not_used_as_reading_progress():
    call = _call(1)
    call["outcome_json"] = "{}"
    with pytest.raises(ValueError, match="hash"):
        project_reading_progress(_execution(call))


def _read_page(ordinal, page, cursor, **arguments):
    return _call(
        ordinal,
        arguments={"path": "paper.pdf", "start_page": 2, "end_page": 6, **arguments},
        result={
            "elements": [{"text": "actual text", "locator": {"page": page}}],
            "coverage": {"physical_pages": 6},
            "next_cursor": cursor,
            "truncated": cursor is not None,
        },
    )


def test_furthest_cursor_does_not_regress_when_a_prefix_is_read_again():
    calls = (
        _read_page(1, 2, {"element_offset": 64, "character_offset": 0}),
        _read_page(
            2, 4, {"element_offset": 128, "character_offset": 0}, element_offset=64
        ),
        _read_page(3, 2, {"element_offset": 64, "character_offset": 0}),
    )
    reading = project_reading_progress(
        _execution(*calls),
        attachment_catalog=({"relative_path": "paper.pdf"},),
    )["attachment_reading"]
    assert reading["pending_reads"] == [
        {
            "source_call_ref": "c2.1",
            "tool_id": "read_pdf_text",
            "arguments": {
                "path": "paper.pdf",
                "start_page": 2,
                "end_page": 6,
                "element_offset": 128,
                "character_offset": 0,
            },
        }
    ]
    assert reading["documents"][0]["pages_with_returned_text"] == [2, 4]
    assert reading["documents"][0]["pages_without_returned_text"] == [1, 3, 5, 6]


@pytest.mark.parametrize("same_selection", [True, False])
def test_eof_closes_only_its_own_selection_without_claiming_full_document_coverage(
    same_selection,
):
    first = _read_page(1, 2, {"element_offset": 64, "character_offset": 0})
    last = _read_page(
        2,
        6,
        None,
        **(
            {"element_offset": 128}
            if same_selection
            else {"start_page": 6, "end_page": 6}
        ),
    )
    reading = project_reading_progress(
        _execution(first, last),
        attachment_catalog=({"relative_path": "paper.pdf"},),
    )["attachment_reading"]
    assert bool(reading["pending_reads"]) is not same_selection
    assert reading["documents"][0]["pages_without_returned_text"] == [1, 3, 4, 5]


def test_attachment_scope_is_bounded_and_does_not_merge_mutable_workspace_reads():
    calls = [
        _call(n, arguments={"path": f"files/{n}-{'x' * 200}.pdf"}) for n in range(1, 80)
    ]
    catalog = [
        {"relative_path": json.loads(call["arguments_json"])["path"]}
        for call in calls[1:]
    ]
    progress = project_reading_progress(_execution(*calls), attachment_catalog=catalog)
    assert len(canonical_json(progress).encode()) < 66_000
    reading = progress["attachment_reading"]
    assert reading["omitted_count"] > 0
    assert all(not item["path"].startswith("files/1-") for item in reading["documents"])
