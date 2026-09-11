"""L1 ingress must remain independent of durable Task context."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from personagraph.model_io.gateway import ModelResult
from personagraph.runtime.entry.context.contracts import EntryContext
from personagraph.runtime.entry.context.task_catalog import EntryTaskCatalog
from personagraph.runtime.entry.ingress import model as ingress_model
from personagraph.runtime.entry.ingress.contracts import (
    AuthoritativeRuntimeSnapshot,
    CapabilityCeiling,
    TrustedTurnEnvelope,
)
from personagraph.runtime.entry.ingress.model_contracts import EntryClassification
from personagraph.runtime.entry.routing.policy import (
    TurnRoutingPolicy,
    freeze_turn_routing_policy,
)
from personagraph.session.entry_task_contracts import EntryTaskCatalogItem


def _existing_root_match() -> dict[str, object]:
    return {
        "match_type": "existing_root",
        "insession_task_id": "task-existing",
        "source_excerpt": "继续之前的任务",
        "execute_current": False,
    }


def _new_root_match() -> dict[str, object]:
    return {
        "match_type": "new_root",
        "local_key": "new_task",
        "title": "New task",
        "objective": "Create a durable Task",
        "source_excerpt": "创建一个长期任务",
    }


def test_l1_classification_accepts_only_an_empty_task_match_batch() -> None:
    classification = EntryClassification(
        processing_level="L1",
        task_matches=(),
    )

    assert classification.task_matches == ()
    with pytest.raises(ValidationError) as raised:
        EntryClassification(
            processing_level="L1",
            task_matches=(_existing_root_match(),),
        )

    assert any(error["loc"] == ("task_matches",) for error in raised.value.errors())


def test_l0_keeps_read_only_existing_task_references() -> None:
    classification = EntryClassification(
        processing_level="L0",
        task_matches=(_existing_root_match(),),
    )

    assert len(classification.task_matches) == 1
    match = classification.task_matches[0]
    assert match.match_type == "existing_root"
    assert match.insession_task_id == "task-existing"
    assert match.execute_current is False


def test_l2_keeps_durable_task_match_proposals() -> None:
    classification = EntryClassification(
        processing_level="L2",
        task_matches=(_new_root_match(),),
    )

    assert len(classification.task_matches) == 1
    match = classification.task_matches[0]
    assert match.match_type == "new_root"
    assert match.local_key == "new_task"


@pytest.mark.parametrize("l1_enabled", (True, False))
def test_classifier_task_context_disclosure_follows_frozen_mode(
    monkeypatch: pytest.MonkeyPatch,
    l1_enabled: bool,
) -> None:
    captured_requests: list[tuple[str, dict[str, object]]] = []
    processing_level = "L1" if l1_enabled else "L2"

    class _Prepared:
        def dispatch(self, *, model_call_id: str) -> ModelResult:
            return ModelResult(
                reply=json.dumps(
                    {"processing_level": processing_level, "task_matches": []},
                    ensure_ascii=False,
                ),
                provider="test",
                model="test",
                latency_ms=1,
                model_call_id=model_call_id,
                purpose="runtime_entry_classify",
            )

    def provider(*_args: object, **_kwargs: object) -> ModelResult:
        raise AssertionError("prepared test provider must not use a direct call")

    def prepare(
        system_prompt: str,
        user_content: str,
        **_kwargs: object,
    ) -> _Prepared:
        captured_requests.append((system_prompt, json.loads(user_content)))
        return _Prepared()

    provider.prepare = prepare  # type: ignore[attr-defined]
    monkeypatch.setattr(ingress_model, "complete_structured", provider)
    context = EntryContext(
        envelope=TrustedTurnEnvelope(
            turn_id="turn-l1",
            session_id="session-l1",
            received_at=datetime.now(timezone.utc),
            input_kind="user_text",
            user_text="检查当前文件，但不要建立长期任务",
        ),
        snapshot=AuthoritativeRuntimeSnapshot(
            pending_decision_id="decision-secret",
            active_run_id="work-run-secret",
            active_run_status="running",
        ),
        ceiling=CapabilityCeiling(),
        estimated_input_tokens=1,
        history_pairs=(),
        session_summary=None,
        task_catalog=EntryTaskCatalog(
            items=(
                EntryTaskCatalogItem(
                    insession_task_id="task-secret-existing",
                    goal_summary="不应暴露给 L1 classifier 的长期任务",
                    status="active",
                    current_graph_revision=9,
                ),
            ),
            truncated=True,
        ),
        routing_policy=freeze_turn_routing_policy(
            TurnRoutingPolicy(l1_enabled=l1_enabled, l2_enabled=not l1_enabled),
            source="request_override",
        ),
    )

    result = ingress_model.classify_turn(context, lambda _event: None)

    assert result.processing_level == processing_level
    assert result.task_matches == ()
    assert len(captured_requests) == 1
    system_prompt, user_payload = captured_requests[0]
    if l1_enabled:
        assert "task_catalog" not in user_payload
        assert "task_catalog_truncated" not in user_payload
        assert "task-secret-existing" not in json.dumps(user_payload)
        assert "existing_root" not in system_prompt
        assert "new_root" not in system_prompt
        assert '"task_matches":[]' in system_prompt
        assert "has_pending_decision" not in system_prompt
        assert "has_active_work_run" not in system_prompt
        assert user_payload["host_facts"] == {
            "attachment_count": 0,
            "session_summary_status": "empty",
        }
    else:
        assert user_payload["task_catalog"][0]["insession_task_id"] == "task-secret-existing"
        assert user_payload["task_catalog_truncated"] is True
        assert user_payload["host_facts"]["has_pending_decision"] is True
        assert user_payload["host_facts"]["has_active_work_run"] is True
        assert "existing_root" in system_prompt
        assert "new_root" in system_prompt
