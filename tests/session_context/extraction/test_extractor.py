import ast
import json
from pathlib import Path

from personagraph.session.context import extractor
from personagraph.session.context.models import EvidenceKind, EvidenceRecord, SessionDomain, SourceKind


NOW = "2026-07-12T10:00:00+00:00"


def _evidence(
    evidence_id="turn:session-a:0",
    *,
    session_id="session-a",
    kind=EvidenceKind.USER_TURN,
    content="这次讲详细一点",
):
    return EvidenceRecord(
        id=evidence_id,
        session_id=session_id,
        kind=kind,
        source_ref=f"source:{evidence_id}",
        content_excerpt=content,
        content_hash="hash",
        created_at=NOW,
    )


def _payload(items):
    return json.dumps({"items": items}, ensure_ascii=False)


def _item(**overrides):
    item = {
        "domain": "user",
        "state_type": "temporary_preference",
        "key": "response_depth",
        "value": "detailed",
        "operation": "set",
        "source_kind": "explicit",
        "evidence_refs": ["turn:session-a:0"],
        "confidence_hint": 1.0,
    }
    item.update(overrides)
    return item


def test_prompt_is_generated_from_catalog_and_contains_safety_boundaries():
    assert "user.temporary_preference" in extractor.EXTRACTION_SYSTEM
    assert "task.progress_delta" in extractor.EXTRACTION_SYSTEM
    assert "interaction.assistant_commitment" in extractor.EXTRACTION_SYSTEM
    assert "不抽助手身份、系统实现、外部作品设定" in extractor.EXTRACTION_SYSTEM
    assert "不直接生成长期用户记忆" in extractor.EXTRACTION_SYSTEM


def test_valid_json_produces_stable_system_derived_candidate():
    record = _evidence()
    raw = _payload([_item()])
    first = extractor.parse_candidates(raw, "session-a", [record])
    second = extractor.parse_candidates(raw, "session-a", [record])
    assert first == second
    assert first.error is None
    assert first.rejected == ()
    candidate = first.candidates[0]
    assert candidate.candidate_id.startswith("candidate:")
    assert candidate.domain == SessionDomain.USER
    assert candidate.source_kind == SourceKind.EXPLICIT
    assert candidate.derived_from == (record.id,)
    assert candidate.valid_from == NOW


def test_one_sentence_can_produce_user_and_interaction_facets():
    user = _evidence(content="我没跟上，你讲得太快了")
    raw = _payload([
        _item(state_type="affect_signal", key="confusion", value="confusion"),
        _item(
            state_type="interaction_feedback",
            key="pacing-feedback-1",
            value="too_fast",
            operation="append",
        ),
    ])
    result = extractor.parse_candidates(raw, "session-a", [user])
    assert [candidate.state_type for candidate in result.candidates] == [
        "affect_signal", "interaction_feedback"
    ]


def test_assistant_commitment_requires_assistant_evidence():
    assistant = _evidence(
        "turn:session-a:1",
        kind=EvidenceKind.ASSISTANT_TURN,
        content="我会补 reducer 测试",
    )
    item = _item(
        domain="interaction",
        state_type="assistant_commitment",
        key="reducer-tests",
        value="补 reducer 测试",
        operation="append",
        source_kind="assistant",
        evidence_refs=[assistant.id],
    )
    accepted = extractor.parse_candidates(_payload([item]), "session-a", [assistant])
    rejected = extractor.parse_candidates(_payload([item]), "session-a", [_evidence()])
    assert accepted.candidates[0].source_kind == SourceKind.ASSISTANT
    assert rejected.candidates == ()
    assert rejected.rejected[0].reason_code == "unknown_evidence_ref"


def test_tool_candidate_requires_tool_or_artifact_evidence():
    tool = _evidence(
        "event:tool-1",
        kind=EvidenceKind.TOOL_RESULT,
        content="pytest: 22 passed",
    )
    item = _item(
        domain="task",
        state_type="tool_result_ref",
        key="pytest-1",
        value={"passed": 22},
        operation="append",
        source_kind="tool",
        evidence_refs=[tool.id],
    )
    result = extractor.parse_candidates(_payload([item]), "session-a", [tool])
    assert len(result.candidates) == 1


def test_invalid_items_are_dropped_individually_with_stable_reasons():
    local = _evidence()
    foreign = _evidence("turn:session-b:0", session_id="session-b")
    items = [
        _item(state_type="unknown"),
        _item(key=" "),
        _item(evidence_refs=["turn:missing:0"]),
        _item(evidence_refs=[foreign.id]),
        _item(source_kind="tool"),
    ]
    result = extractor.parse_candidates(_payload(items), "session-a", [local, foreign])
    assert result.candidates == ()
    assert [item.reason_code for item in result.rejected] == [
        "invalid_type",
        "invalid_key",
        "unknown_evidence_ref",
        "evidence_scope_mismatch",
        "source_not_allowed",
    ]


def test_duplicate_candidates_are_deduplicated_before_reducer():
    item = _item()
    result = extractor.parse_candidates(_payload([item, item]), "session-a", [_evidence()])
    assert len(result.candidates) == 1
    assert result.rejected[0].reason_code == "duplicate_candidate"


def test_schema_or_json_failures_return_empty_candidates():
    malformed = extractor.parse_candidates("not json", "session-a", [_evidence()])
    extra = extractor.parse_candidates(
        _payload([{**_item(), "unexpected": True}]), "session-a", [_evidence()]
    )
    assert malformed.error == "parse_failed"
    assert extra.error == "schema_validation_failed"


def test_existing_mock_gateway_noop_shape_is_accepted():
    result = extractor.parse_candidates(
        '{"should_store": false, "items": []}', "session-a", [_evidence()]
    )
    assert result.candidates == ()
    assert result.error is None


def test_extract_candidates_uses_injected_model_and_fails_soft():
    seen = {}

    def complete(system, user):
        seen["system"] = system
        seen["user"] = user
        return _payload([_item()])

    success = extractor.extract_candidates(
        "session-a", "这次详细点", "好", [_evidence()], complete=complete
    )
    failure = extractor.extract_candidates(
        "session-a", "x", "y", [_evidence()],
        complete=lambda *_: (_ for _ in ()).throw(RuntimeError("down")),
    )
    assert len(success.candidates) == 1
    assert "turn:session-a:0" in seen["user"]
    assert failure.candidates == ()
    assert failure.error == "model_error:RuntimeError"


def test_extractor_dependency_boundary_has_no_store_graph_or_sql_imports():
    path = Path(extractor.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any(
        forbidden in module
        for module in imported
        for forbidden in ("sqlite3", "context_store", "memory.store", "graph")
    )
