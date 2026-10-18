"""Exercise the complete Entry/L1 file-tool workflow."""

from __future__ import annotations

import json
from pathlib import Path


def test_real_file_tools_entry_l1_path_completes_the_turn(
    monkeypatch,
    tmp_path: Path,
    bound_partitioned_session,
) -> None:
    """File registrations and one complete L1 execution use the same catalog."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "brief.md").write_text(
        "# Brief\n\nCandidate V2 Entry/L1 boundary probe.",
        encoding="utf-8",
    )
    session_id = bound_partitioned_session(working_dir=workspace)

    from personagraph.session import store as session_store

    from personagraph.model_io.gateway import ModelResult
    from personagraph.runtime import entry
    from personagraph.runtime.entry.ingress import model as ingress_model
    from personagraph.runtime.l1 import model as l1_model
    from personagraph.runtime.l1.tool_runtime import build_l1_tool_runtime
    from personagraph.runtime.entry.routing.policy import (
        TurnRoutingPolicy,
        freeze_turn_routing_policy,
    )
    from tests.helpers.evidence_submission import (
        add_empty_support_justifications,
    )
    from tests.helpers.prepared_model_provider import (
        as_prepared_test_provider,
    )
    candidate_features = {
        "file_retrieval_write_enabled": True,
        "file_retrieval_read_enabled": True,
        "history_retrieval_write_enabled": False,
        "history_retrieval_read_enabled": False,
        "l1_retrieval_tools_enabled": True,
    }
    candidate_tool_ids = {
        "find_files",
        "check_files_state",
        "prepare_files",
        "retrieve_files",
        "read_file_chunks",
    }
    runtime = build_l1_tool_runtime(
        session_id,
        turn_id="turn-candidate-boundary-probe",
        execution_features=candidate_features,
    )
    exposed_tool_ids = {item["tool_id"] for item in runtime.model_catalog()}
    assert candidate_tool_ids <= exposed_tool_ids
    assert candidate_tool_ids <= set(runtime.registrations_by_tool_id)
    assert runtime.attachment_file_catalog == ()
    def model_result(
        payload: dict[str, object], model_call_id: object
    ) -> ModelResult:
        return ModelResult(
            reply=json.dumps(
                add_empty_support_justifications(payload),
                ensure_ascii=False,
            ),
            provider="mock",
            model="mock-structured",
            latency_ms=1,
            model_call_id=str(model_call_id),
        )
    monkeypatch.setattr(
        ingress_model,
        "complete_structured",
        as_prepared_test_provider(
            lambda *_args, **kwargs: model_result(
                {"processing_level": "L1"},
                kwargs["model_call_id"],
            )
        ),
    )
    model_catalog_tool_ids: set[str] = set()
    def decide(_system: str, user_content: str, **kwargs: object) -> ModelResult:
        payload = json.loads(user_content)
        model_catalog_tool_ids.update(
            item["tool_id"] for item in payload["tool_catalog"]
        )
        return model_result(
            {
                "plan": {
                    "objective": "确认 L1 路径可执行",
                    "acceptances": [
                        {
                            "criterion": "返回 L1 成功确认",
                        }
                    ],
                },
                "action": {
                    "kind": "submit_final_reply",
                    "reply": "L1 execution is available.",
                },
            },
            kwargs["model_call_id"],
        )
    monkeypatch.setattr(
        l1_model,
        "complete_structured",
        as_prepared_test_provider(decide, add_l1_notes=True),
    )
    result = entry.run_entry_turn(
        user_input="请确认 L1 可用",
        features={
            "context_guard_limit": 24_000,
            "l1_semantic_verification_mode": "off",
            **candidate_features,
        },
        session_id=session_id,
        client_request_id="entry-l1-tool-flow",
        routing_policy=freeze_turn_routing_policy(
            TurnRoutingPolicy(l1_enabled=True),
            source="request_override",
        ),
        store=session_store,
    )
    execution = session_store.get_l1_turn_execution(
        session_id=session_id,
        turn_id=result.turn_id,
    )
    assert result.status == "completed"
    assert result.processing_level == "L1"
    assert result.reply == "L1 execution is available."
    assert execution is not None
    assert execution["run"]["status"] == "completed"
    assert [item["action_kind"] for item in execution["attempts"]] == [
        "submit_final_reply",
    ]
    assert candidate_tool_ids <= model_catalog_tool_ids
